import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys

import matplotlib
import numpy as np
import yaml


# This runner saves diagnostics without opening a macOS GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils import (
    average_subsampled_presentations, centered_norms_and_directions,
    compute_rdm_timeseries, cross_temporal_similarity, load_binned_raster,
    load_raster_presentation_names, load_reliable_channels,
    radial_direction_static_regress_out,
)


@dataclass
class Cfg:
    dynamic_exp_name: str = "paul_20260831to0927"
    static_exp_name: str = "paul_20260901to0925"
    reliable_channels_key: str = "paul_20260901to0925"
    good_channels: tuple[int, int] = (1, 64)
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float = 1000
    dynamic_crop_ms: float = 3500
    repetitions_per_stimulus: int = 10
    n_draws: int = 10
    random_seed: int = 0
    regression_type: str = "ridge"
    cv_type: str = "kf"
    n_splits: int = 5
    alphas: tuple[float, ...] = (
        1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5,
    )
    output_dir: Path = PROJECT_ROOT / "results" / "paul_radial_direction_check"
    figure_dpi: int = 180
# EOC


"""
parse_args
Parse the trial-matching, regression and output settings.

OUTPUT:
    - cfg: Cfg -> validated analysis configuration.
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trial-matched Paul dRSA after separate log-norm and population-"
            "direction regressions of the 2000-ms and 2250-ms frames."
        )
    )
    parser.add_argument(
        "--repetitions_per_stimulus", type=int,
        default=Cfg.repetitions_per_stimulus,
    )
    parser.add_argument("--n_draws", type=int, default=Cfg.n_draws)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument("--n_splits", type=int, default=Cfg.n_splits)
    parser.add_argument(
        "--output_dir", type=Path, default=Cfg.output_dir,
    )
    parser.add_argument("--figure_dpi", type=int, default=Cfg.figure_dpi)
    args = parser.parse_args()

    positive_arguments = {
        "repetitions_per_stimulus": args.repetitions_per_stimulus,
        "n_draws": args.n_draws,
        "n_splits": args.n_splits,
        "figure_dpi": args.figure_dpi,
    }
    for argument_name, value in positive_arguments.items():
        if value < 1:
            parser.error(f"--{argument_name} must be positive.")
        # end if invalid positive argument
    # end for argument_name, value
    return Cfg(**vars(args))
# EOF


"""
stimulus_identity
Extract the shared stimulus identity for one movie or static-frame condition.

INPUT:
    - stimulus_name: str -> presentation filename stored in the raster.
    - condition: str -> dynamic, last_frame, 2000ms or 2250ms.

OUTPUT:
    - identity: str | None -> modality-independent identity, or None.
"""
def stimulus_identity(stimulus_name, condition):
    stem = Path(stimulus_name).stem
    prefixes = {
        "dynamic": "vid_",
        "last_frame": "img_",
        "2000ms": "img_2000ms_",
        "2250ms": "img_2250ms_",
    }
    if condition not in prefixes:
        raise ValueError(f"Unsupported condition: {condition!r}.")
    # end if unknown condition

    if condition == "last_frame" and stem.startswith(
            (prefixes["2000ms"], prefixes["2250ms"])):
        return None
    # end if timed image is not the final-frame condition
    prefix = prefixes[condition]
    return stem[len(prefix):] if stem.startswith(prefix) else None
# EOF


"""
condition_identities
Create one identity entry per presentation, retaining None for other conditions.

INPUT:
    - presentation_names: list[str] -> raster presentation filenames.
    - condition: str -> requested movie or static-frame condition.

OUTPUT:
    - identities: list[str | None] -> aligned stimulus identities.
"""
def condition_identities(presentation_names, condition):
    return [stimulus_identity(name, condition) for name in presentation_names]
# EOF


"""
eligible_stimuli
Find stimuli shared by all four conditions with enough presentations for the
requested matched-trial draw.

INPUT:
    - identities_by_condition: dict[str, list[str | None]] -> presentation IDs.
    - repetitions_per_stimulus: int -> minimum count required in every condition.

OUTPUT:
    - stimuli: list[str] -> sorted identities retained in every analysis.
"""
def eligible_stimuli(identities_by_condition, repetitions_per_stimulus):
    counts_by_condition = {
        condition: Counter(identity for identity in identities if identity is not None)
        for condition, identities in identities_by_condition.items()
    }
    shared = set.intersection(*[
        set(counts) for counts in counts_by_condition.values()
    ])
    stimuli = [
        identity for identity in sorted(shared)
        if all(
            counts[identity] >= repetitions_per_stimulus
            for counts in counts_by_condition.values()
        )
    ]
    if len(stimuli) < 3:
        raise ValueError(
            "Fewer than three stimuli have enough repetitions in all conditions."
        )
    # end if too few stimuli
    return stimuli
# EOF


"""
selected_presentation_indices
Return the sorted raster indices belonging to eligible stimuli in any requested
condition.

INPUT:
    - identities_by_condition: dict[str, list[str | None]] -> presentation IDs.
    - conditions: tuple[str, ...] -> conditions stored in the same raster.
    - stimulus_set: set[str] -> eligible identities.

OUTPUT:
    - indices: np.ndarray -> sorted presentation indices.
"""
def selected_presentation_indices(
        identities_by_condition, conditions, stimulus_set,
        ):
    indices = []
    for presentation_index in range(len(identities_by_condition[conditions[0]])):
        if any(
                identities_by_condition[condition][presentation_index] in stimulus_set
                for condition in conditions
                ):
            indices.append(presentation_index)
        # end if presentation belongs to an eligible condition
    # end for presentation_index
    return np.asarray(indices, dtype=int)
# EOF


"""
select_condition_presentations
Restrict a loaded presentation raster to one named condition.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations.
    - presentation_names: list[str] -> loaded presentation filenames.
    - condition: str -> requested condition.

OUTPUT:
    - condition_rasters: np.ndarray -> selected presentation rasters.
    - identities: list[str] -> identity of every selected presentation.
"""
def select_condition_presentations(rasters, presentation_names, condition):
    identities = condition_identities(presentation_names, condition)
    keep = np.asarray([identity is not None for identity in identities])
    selected_identities = [
        identity for identity in identities if identity is not None
    ]
    return rasters[:, :, keep], selected_identities
# EOF


"""
window_mean
Average one dRSA matrix inside requested movie and static time windows.

INPUT:
    - matrix: np.ndarray -> movie time x static time similarities.
    - dynamic_times_ms: np.ndarray -> movie bin times.
    - static_times_ms: np.ndarray -> static bin times.
    - dynamic_window_ms: tuple[float, float] -> movie interval, end exclusive.
    - static_window_ms: tuple[float, float] -> static interval, end exclusive.

OUTPUT:
    - value: float -> mean similarity in the rectangular window.
"""
def window_mean(
        matrix, dynamic_times_ms, static_times_ms,
        dynamic_window_ms, static_window_ms,
        ):
    dynamic_mask = (
        (dynamic_times_ms >= dynamic_window_ms[0])
        & (dynamic_times_ms < dynamic_window_ms[1])
    )
    static_mask = (
        (static_times_ms >= static_window_ms[0])
        & (static_times_ms < static_window_ms[1])
    )
    if not dynamic_mask.any() or not static_mask.any():
        raise ValueError("A requested summary window contains no samples.")
    # end if empty window
    return float(np.nanmean(matrix[np.ix_(dynamic_mask, static_mask)]))
# EOF


"""
summarize_matrices
Compute the sustained, final-frame and movie-baseline summaries used in the
Paul Euclidean diagnostic.

INPUT:
    - matrices: dict[str, np.ndarray] -> mean dRSA matrices.
    - dynamic_times_ms: np.ndarray -> movie bin times.
    - static_times_ms: np.ndarray -> image bin times.

OUTPUT:
    - summaries: dict[str, dict[str, float]] -> window values per analysis.
"""
def summarize_matrices(matrices, dynamic_times_ms, static_times_ms):
    windows = {
        "sustained": ((1500, 3000), (250, 600)),
        "final_frame": ((2500, 2900), (100, 500)),
        "baseline": ((0, 100), (250, 600)),
    }
    return {
        analysis_name: {
            window_name: window_mean(
                matrix, dynamic_times_ms, static_times_ms,
                dynamic_window, static_window,
            )
            for window_name, (dynamic_window, static_window) in windows.items()
        }
        for analysis_name, matrix in matrices.items()
    }
# EOF


"""
plot_matrices
Plot the mean dRSA matrices on a shared colour scale.

INPUT:
    - matrices: dict[str, np.ndarray] -> mean movie x static matrices.
    - summaries: dict[str, dict[str, float]] -> window values per analysis.
    - dynamic_times_ms: np.ndarray -> movie bin times.
    - static_times_ms: np.ndarray -> image bin times.
    - output_path: Path -> PNG destination.
    - figure_dpi: int -> saved resolution.

OUTPUT:
    - None
"""
def plot_matrices(
        matrices, summaries, dynamic_times_ms, static_times_ms,
        output_path, figure_dpi,
        ):
    figure, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()
    finite_values = np.concatenate([
        matrix[np.isfinite(matrix)] for matrix in matrices.values()
    ])
    color_limit = max(abs(np.percentile(finite_values, 1)), np.percentile(finite_values, 99))

    image = None
    for axis, (analysis_name, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(
            matrix.T, origin="lower", aspect="auto", cmap="coolwarm",
            vmin=-color_limit, vmax=color_limit,
            extent=(
                dynamic_times_ms[0], dynamic_times_ms[-1],
                static_times_ms[0], static_times_ms[-1],
            ),
        )
        axis.axvline(2500, color="black", linestyle="--", linewidth=1)
        axis.set_title(
            f"{analysis_name}\n"
            f"sustained={summaries[analysis_name]['sustained']:.3f}, "
            f"final={summaries[analysis_name]['final_frame']:.3f}"
        )
    # end for axis, analysis matrix
    for axis in axes[len(matrices):]:
        axis.axis("off")
    # end for unused axis
    figure.colorbar(
        image, ax=axes[:len(matrices)].tolist(),
        label="Spearman RDM correlation",
    )
    figure.supxlabel("movie response (ms)")
    figure.supylabel("image response (ms)")
    figure.suptitle("Paul: matched-trial radial–direction regress-out")
    figure.savefig(output_path, dpi=figure_dpi)
    plt.close(figure)
# EOF


def main():
    cfg = parse_args()
    data_dir = Path(paths["data_path"]) / "data"
    static_path = data_dir / f"{cfg.static_exp_name}_raster_img.mat"
    dynamic_path = data_dir / f"{cfg.dynamic_exp_name}_raster_vid.mat"

    # Read names first so under-sampled stimuli can be removed before loading data.
    static_names_all = load_raster_presentation_names(static_path)
    dynamic_names_all = load_raster_presentation_names(dynamic_path)
    identities_by_condition = {
        "last_frame": condition_identities(static_names_all, "last_frame"),
        "2000ms": condition_identities(static_names_all, "2000ms"),
        "2250ms": condition_identities(static_names_all, "2250ms"),
        "dynamic": condition_identities(dynamic_names_all, "dynamic"),
    }
    stimuli = eligible_stimuli(
        identities_by_condition, cfg.repetitions_per_stimulus,
    )
    stimulus_set = set(stimuli)
    print(
        f"Retaining {len(stimuli)} stimuli with at least "
        f"{cfg.repetitions_per_stimulus} repetitions in every condition."
    )

    static_indices = selected_presentation_indices(
        identities_by_condition, ("last_frame", "2000ms", "2250ms"),
        stimulus_set,
    )
    dynamic_indices = selected_presentation_indices(
        identities_by_condition, ("dynamic",), stimulus_set,
    )
    first_channel, last_channel = cfg.good_channels
    channel_slice = slice(first_channel - 1, last_channel)
    static_rasters, static_names = load_binned_raster(
        static_path, cfg.source_fs, cfg.new_fs, channel_slice=channel_slice,
        end_sample=int(round(cfg.static_crop_ms * cfg.source_fs / 1000)),
        presentation_indices=static_indices,
    )
    dynamic_rasters, dynamic_names = load_binned_raster(
        dynamic_path, cfg.source_fs, cfg.new_fs, channel_slice=channel_slice,
        end_sample=int(round(cfg.dynamic_crop_ms * cfg.source_fs / 1000)),
        presentation_indices=dynamic_indices,
    )

    # Apply the same reliable-channel list used by the main Paul analysis.
    loaded_channel_numbers = np.arange(first_channel, last_channel + 1)
    reliable_channels = set(load_reliable_channels(
        PROJECT_ROOT / "reliable_channels.yaml", cfg.reliable_channels_key,
    ))
    channel_mask = np.asarray([
        channel in reliable_channels for channel in loaded_channel_numbers
    ])
    static_rasters = static_rasters[channel_mask]
    dynamic_rasters = dynamic_rasters[channel_mask]
    channel_numbers = loaded_channel_numbers[channel_mask]
    print(f"Using {len(channel_numbers)} reliable channels: {channel_numbers.tolist()}")

    static_conditions = {}
    for condition in ("last_frame", "2000ms", "2250ms"):
        static_conditions[condition] = select_condition_presentations(
            static_rasters, static_names, condition,
        )
    # end for static condition
    dynamic_condition = select_condition_presentations(
        dynamic_rasters, dynamic_names, "dynamic",
    )

    analysis_names = (
        "raw_euclidean", "cosine_cnt", "direction_only_euclidean",
        "direction_residual_euclidean", "strength_residual",
    )
    matrices_by_analysis = {name: [] for name in analysis_names}
    direction_variance_removed = []
    norm_variance_removed = []
    direction_alphas = []
    norm_alphas = []
    rng = np.random.default_rng(cfg.random_seed)

    for draw_index in range(cfg.n_draws):
        print(f"Matched-trial draw {draw_index + 1}/{cfg.n_draws}")
        static_means = {
            condition: average_subsampled_presentations(
                rasters, identities, stimuli, cfg.repetitions_per_stimulus, rng,
            )
            for condition, (rasters, identities) in static_conditions.items()
        }
        dynamic_means = average_subsampled_presentations(
            dynamic_condition[0], dynamic_condition[1], stimuli,
            cfg.repetitions_per_stimulus, rng,
        )

        # Existing raw Euclidean and cosine-centred reference analyses.
        raw_dynamic_rdms = compute_rdm_timeseries(dynamic_means, "euclidean")
        raw_static_rdms = compute_rdm_timeseries(
            static_means["last_frame"], "euclidean",
        )
        matrices_by_analysis["raw_euclidean"].append(
            cross_temporal_similarity(
                raw_dynamic_rdms, raw_static_rdms, metric="spearman",
            )
        )
        cosine_dynamic_rdms = compute_rdm_timeseries(dynamic_means, "cosine_cnt")
        cosine_static_rdms = compute_rdm_timeseries(
            static_means["last_frame"], "cosine_cnt",
        )
        matrices_by_analysis["cosine_cnt"].append(
            cross_temporal_similarity(
                cosine_dynamic_rdms, cosine_static_rdms, metric="spearman",
            )
        )

        # Unit-normalized Euclidean is the direction-only reference and should
        # have the same rank geometry as cosine_cnt.
        dynamic_norms, dynamic_directions = centered_norms_and_directions(
            dynamic_means,
        )
        _, static_directions = centered_norms_and_directions(
            static_means["last_frame"],
        )
        dynamic_direction_rdms = compute_rdm_timeseries(
            dynamic_directions, "euclidean",
        )
        static_direction_rdms = compute_rdm_timeseries(
            static_directions, "euclidean",
        )
        matrices_by_analysis["direction_only_euclidean"].append(
            cross_temporal_similarity(
                dynamic_direction_rdms, static_direction_rdms,
                metric="spearman",
            )
        )

        # Fit strength and direction separately using both earlier static frames.
        regression_result = radial_direction_static_regress_out(
            np.stack([
                static_means["2000ms"], static_means["2250ms"],
            ]),
            static_means["last_frame"],
            regression_type=cfg.regression_type, alphas=cfg.alphas,
            cv_type=cfg.cv_type, n_splits=cfg.n_splits,
        )
        residual_direction_rdms = compute_rdm_timeseries(
            regression_result["direction_residuals"], "euclidean",
        )
        matrices_by_analysis["direction_residual_euclidean"].append(
            cross_temporal_similarity(
                dynamic_direction_rdms, residual_direction_rdms,
                metric="spearman",
            )
        )

        # Scalar Euclidean distance is the absolute difference in log strength.
        dynamic_strength_rdms = compute_rdm_timeseries(
            np.log(dynamic_norms)[np.newaxis, :, :], "euclidean",
        )
        residual_strength_rdms = compute_rdm_timeseries(
            regression_result["log_norm_residuals"][np.newaxis, :, :],
            "euclidean",
        )
        matrices_by_analysis["strength_residual"].append(
            cross_temporal_similarity(
                dynamic_strength_rdms, residual_strength_rdms,
                metric="spearman",
            )
        )
        direction_variance_removed.append(
            regression_result["direction_variance_removed"]
        )
        norm_variance_removed.append(
            regression_result["log_norm_variance_removed"]
        )
        direction_alphas.append(regression_result["direction_alphas"])
        norm_alphas.append(regression_result["log_norm_alphas"])
    # end for draw_index

    mean_matrices = {
        name: np.nanmean(np.stack(matrices), axis=0)
        for name, matrices in matrices_by_analysis.items()
    }
    static_times_ms = np.arange(static_rasters.shape[1]) * 1000 / cfg.new_fs
    dynamic_times_ms = np.arange(dynamic_rasters.shape[1]) * 1000 / cfg.new_fs
    summaries = summarize_matrices(
        mean_matrices, dynamic_times_ms, static_times_ms,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.output_dir / (
        f"paul_radial_direction_{cfg.repetitions_per_stimulus}trials_"
        f"{cfg.n_draws}draws_seed{cfg.random_seed}.npz"
    )
    np.savez_compressed(
        output_path,
        **mean_matrices,
        direction_variance_removed=np.stack(direction_variance_removed),
        log_norm_variance_removed=np.stack(norm_variance_removed),
        direction_alphas=np.stack(direction_alphas),
        log_norm_alphas=np.stack(norm_alphas),
        dynamic_times_ms=dynamic_times_ms,
        static_times_ms=static_times_ms,
        shared_stimuli=np.asarray(stimuli),
        channel_numbers=channel_numbers,
        config=np.asarray(str(asdict(cfg))),
    )
    figure_path = output_path.with_suffix(".png")
    plot_matrices(
        mean_matrices, summaries, dynamic_times_ms, static_times_ms,
        figure_path, cfg.figure_dpi,
    )

    print("\nAnalysis                         sustained    final    baseline")
    for analysis_name, analysis_summary in summaries.items():
        print(
            f"{analysis_name:32s} "
            f"{analysis_summary['sustained']:9.3f} "
            f"{analysis_summary['final_frame']:8.3f} "
            f"{analysis_summary['baseline']:11.3f}"
        )
    # end for analysis_name
    print(f"\nSaved {output_path}")
    print(f"Saved {figure_path}")
# EOF


if __name__ == "__main__":
    main()
# end if __name__
