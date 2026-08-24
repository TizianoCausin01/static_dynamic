import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

import matplotlib

# Saving is the default workflow, so avoid opening the macOS GUI backend unless
# the user explicitly requests interactive figures.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
# end if not show

import matplotlib.pyplot as plt
import numpy as np
import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils import (
    compute_cross_temporal_manifold_dynamics,
    compute_manifold_dynamics,
    load_natraster,
    match_static_dynamic_rasters,
    population_response_scores,
    select_manifold_subsets,
)
from useful_stuff.general_utils import TimeSeries


@dataclass
class Cfg:
    static_exp_name: str = "red_20260726to27"
    dynamic_exp_name: str = "red_20260720to24"
    static_path: str | None = None
    dynamic_path: str | None = None
    output_dir: str | None = None

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = None
    source_fs: float = 1000
    analysis_fs: float = 100
    static_crop_ms: float | None = 1000
    dynamic_crop_ms: float | None = None

    # The dynamic default is 60--200 ms after the final-frame onset at 2500 ms.
    static_selection_window_ms: tuple[float, float] = (60, 200)
    dynamic_selection_window_ms: tuple[float, float] = (2560, 2700)
    subset_size: int = 50
    selection_reference: str = "condition"
    n_random_sets: int = 1
    random_seed: int = 0

    rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "correlation"
    n_pc_components: int = 2
    cross_temporal: bool = True
    figure_dpi: int = 180
    show: bool = False
# EOF


"""
channel_selection_name
Build the output-folder label for the selected MATLAB channel range.

INPUT:
    - cfg: Cfg -> analysis configuration containing good_channels

OUTPUT:
    - name: str -> stable channel-selection folder name
"""
def channel_selection_name(cfg: Cfg) -> str:
    if cfg.good_channels is None:
        return "all_channels"
    # end if cfg.good_channels is None
    first_channel, last_channel = cfg.good_channels
    return f"channels_{first_channel}to{last_channel}"
# EOF


"""
parse_args
Parse and validate static/dynamic manifold-dynamics parameters.

OUTPUT:
    - cfg: Cfg -> validated analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Marvi-style RDM autocorrelation and PCA-subspace "
            "rotation for matched static and dynamic stimuli."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument("--static_path")
    parser.add_argument("--dynamic_path")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--good_channels", nargs=2, type=int, metavar=("FIRST", "LAST"),
    )
    parser.add_argument(
        "--all_channels", action="store_true",
        help="Analyze all channels; this is the default.",
    )
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--analysis_fs", type=float, default=Cfg.analysis_fs)
    parser.add_argument("--static_crop_ms", type=float, default=Cfg.static_crop_ms)
    parser.add_argument("--dynamic_crop_ms", type=float)
    parser.add_argument(
        "--static_selection_window_ms", nargs=2, type=float,
        default=list(Cfg.static_selection_window_ms), metavar=("START", "STOP"),
    )
    parser.add_argument(
        "--dynamic_selection_window_ms", nargs=2, type=float,
        default=list(Cfg.dynamic_selection_window_ms), metavar=("START", "STOP"),
    )
    parser.add_argument("--subset_size", type=int, default=Cfg.subset_size)
    parser.add_argument(
        "--selection_reference",
        choices=("condition", "static", "dynamic", "pooled"),
        default=Cfg.selection_reference,
        help=(
            "Data used to rank top/bottom stimuli. 'condition' ranks each "
            "condition separately; the other choices share rankings."
        ),
    )
    parser.add_argument("--n_random_sets", type=int, default=Cfg.n_random_sets)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument("--rdm_metric", default=Cfg.rdm_metric)
    parser.add_argument(
        "--rsa_metric", choices=("correlation", "spearman"),
        default=Cfg.rsa_metric,
    )
    parser.add_argument(
        "--n_pc_components", type=int, default=Cfg.n_pc_components,
    )
    parser.add_argument(
        "--cross_temporal",
        action=argparse.BooleanOptionalAction,
        default=Cfg.cross_temporal,
        help=(
            "Compute dynamic-time by static-time dRSA similarity and "
            "PCA-subspace rotation matrices."
        ),
    )
    parser.add_argument("--figure_dpi", type=int, default=Cfg.figure_dpi)
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=Cfg.show,
    )
    args = parser.parse_args()

    if args.source_fs <= 0 or args.analysis_fs <= 0:
        parser.error("--source_fs and --analysis_fs must be positive.")
    # end if invalid sampling frequency
    for argument_name in ("static_crop_ms", "dynamic_crop_ms"):
        value = getattr(args, argument_name)
        if value is not None and value <= 0:
            parser.error(f"--{argument_name} must be positive.")
        # end if invalid crop
    # end for argument_name
    for argument_name in (
            "static_selection_window_ms", "dynamic_selection_window_ms",
            ):
        start_ms, stop_ms = getattr(args, argument_name)
        if start_ms < 0 or stop_ms <= start_ms:
            parser.error(f"--{argument_name} must be an increasing interval.")
        # end if invalid interval
        setattr(args, argument_name, (start_ms, stop_ms))
    # end for argument_name
    if args.subset_size < 2:
        parser.error("--subset_size must be at least 2.")
    # end if subset_size
    if args.n_random_sets < 1:
        parser.error("--n_random_sets must be positive.")
    # end if n_random_sets
    if args.n_pc_components < 1:
        parser.error("--n_pc_components must be positive.")
    # end if n_pc_components
    if args.figure_dpi < 1:
        parser.error("--figure_dpi must be positive.")
    # end if figure_dpi
    if args.all_channels and args.good_channels is not None:
        parser.error("Set --good_channels or --all_channels, not both.")
    # end if conflicting channel arguments
    if args.good_channels is not None:
        first_channel, last_channel = args.good_channels
        if first_channel < 1 or last_channel < first_channel:
            parser.error("--good_channels must be an increasing positive range.")
        # end if invalid channel range
        args.good_channels = tuple(args.good_channels)
    # end if good_channels

    del args.all_channels
    return Cfg(**vars(args))
# EOF


"""
zscore_stimulus_scores
Standardize scores across stimuli for scale-balanced pooled selection.

INPUT:
    - scores: np.ndarray -> one response score per stimulus

OUTPUT:
    - standardized_scores: np.ndarray -> zero-mean unit-variance scores
"""
def zscore_stimulus_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    score_std = scores.std()
    if score_std == 0:
        raise ValueError("Cannot pool constant population-response scores.")
    # end if score_std
    return (scores - scores.mean()) / score_std
# EOF


"""
build_condition_subsets
Build condition-specific or shared top, bottom, and random stimulus subsets.

INPUT:
    - static_scores: np.ndarray -> static population-response scores
    - dynamic_scores: np.ndarray -> dynamic population-response scores
    - cfg: Cfg -> subset size, selection-reference, and random parameters

OUTPUT:
    - condition_subsets: dict[str, dict[str, np.ndarray]] -> indices by condition
"""
def build_condition_subsets(
        static_scores: np.ndarray,
        dynamic_scores: np.ndarray,
        cfg: Cfg,
        ) -> dict[str, dict[str, np.ndarray]]:
    if len(static_scores) != len(dynamic_scores):
        raise ValueError("Static and dynamic scores must describe matched stimuli.")
    # end if mismatched scores

    rng = np.random.default_rng(cfg.random_seed)
    if cfg.selection_reference == "condition":
        # Draw random controls once, then reuse them so their identities match.
        static_subsets = select_manifold_subsets(
            static_scores, cfg.subset_size, cfg.n_random_sets, rng,
        )
        shared_random = {
            name: indices.copy()
            for name, indices in static_subsets.items()
            if name.startswith("random_")
        }
        dynamic_subsets = select_manifold_subsets(
            dynamic_scores,
            cfg.subset_size,
            cfg.n_random_sets,
            np.random.default_rng(cfg.random_seed + 1),
        )
        dynamic_subsets.update(shared_random)
        return {"static": static_subsets, "dynamic": dynamic_subsets}
    # end if condition-specific selection

    if cfg.selection_reference == "static":
        reference_scores = static_scores
    elif cfg.selection_reference == "dynamic":
        reference_scores = dynamic_scores
    else:
        reference_scores = (
            zscore_stimulus_scores(static_scores)
            + zscore_stimulus_scores(dynamic_scores)
        ) / 2
    # end if selection_reference

    shared_subsets = select_manifold_subsets(
        reference_scores, cfg.subset_size, cfg.n_random_sets, rng,
    )
    return {
        "static": {name: indices.copy() for name, indices in shared_subsets.items()},
        "dynamic": {name: indices.copy() for name, indices in shared_subsets.items()},
    }
# EOF


"""
build_cross_temporal_subsets
Build stimulus subsets whose identities match across static and dynamic data.

For condition-specific ranking, top and bottom selections are analyzed twice:
once using static-ranked identities and once using dynamic-ranked identities.
Random controls are already shared across conditions and are retained once.

INPUT:
    - condition_subsets: dict[str, dict[str, np.ndarray]] -> within-condition sets
    - cfg: Cfg -> configuration containing the selection-reference mode

OUTPUT:
    - cross_subsets: dict[str, np.ndarray] -> shared indices for cross analysis
"""
def build_cross_temporal_subsets(
        condition_subsets: dict[str, dict[str, np.ndarray]],
        cfg: Cfg,
        ) -> dict[str, np.ndarray]:
    static_subsets = condition_subsets["static"]
    dynamic_subsets = condition_subsets["dynamic"]
    if static_subsets.keys() != dynamic_subsets.keys():
        raise ValueError("Static and dynamic subset names must match.")
    # end if mismatched subset names

    cross_subsets = {}
    for subset_name in static_subsets:
        static_indices = static_subsets[subset_name]
        dynamic_indices = dynamic_subsets[subset_name]
        indices_are_shared = np.array_equal(static_indices, dynamic_indices)

        if cfg.selection_reference == "condition" and subset_name in {
                "top", "bottom",
                }:
            # Apply each ranked set unchanged to both recordings so RDM entries
            # always describe the same stimulus pairs across conditions.
            cross_subsets[f"static_{subset_name}"] = static_indices.copy()
            cross_subsets[f"dynamic_{subset_name}"] = dynamic_indices.copy()
        else:
            if not indices_are_shared:
                raise ValueError(
                    f"Cross-temporal subset {subset_name!r} does not contain "
                    "the same stimulus identities in both conditions."
                )
            # end if indices are not shared
            cross_subsets[subset_name] = static_indices.copy()
        # end if condition-specific ranked subset
    # end for subset_name
    return cross_subsets
# EOF


"""
plot_result_matrices
Plot all-condition, preferred, non-preferred, and random analysis matrices.

INPUT:
    - results: dict -> condition and subset analysis matrices
    - times_ms: dict[str, np.ndarray] -> time coordinates by condition
    - matrix_key: str -> result matrix to plot
    - output_path: Path -> destination PNG path
    - dpi: int -> output resolution
    - show: bool -> keep the figure open for interactive display

OUTPUT:
    - None
"""
def plot_result_matrices(
        results: dict,
        times_ms: dict[str, np.ndarray],
        matrix_key: str,
        output_path: Path,
        dpi: int,
        show: bool = False,
        ) -> None:
    subset_names = list(results["static"])
    n_columns = len(subset_names)
    figure, axes = plt.subplots(
        2, n_columns,
        figsize=(4.2 * n_columns, 7.5),
        squeeze=False,
        constrained_layout=True,
    )

    if matrix_key == "drsa_autocorrelation":
        cmap, vmin, vmax = "coolwarm", -1, 1
        colorbar_label = "RDM similarity (r)"
        figure_title = "dRSA autocorrelation"
    else:
        cmap, vmin, vmax = "viridis", 0, 90
        colorbar_label = "Mean principal angle (degrees)"
        figure_title = "Average rotation of dominant PCs"
    # end if matrix_key

    image = None
    for row_index, condition in enumerate(("static", "dynamic")):
        condition_times = times_ms[condition]
        extent = (
            condition_times[0], condition_times[-1],
            condition_times[0], condition_times[-1],
        )
        for column_index, subset_name in enumerate(subset_names):
            matrix = results[condition][subset_name][matrix_key]
            image = axes[row_index, column_index].imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row_index, column_index].set(
                xlabel="Time (ms)",
                ylabel="Time (ms)",
                title=f"{condition.title()} - {subset_name}",
            )
        # end for column_index, subset_name
    # end for row_index, condition

    figure.suptitle(figure_title, fontsize=15, y=1.035)
    figure.colorbar(
        image, ax=axes, label=colorbar_label, shrink=0.86, pad=0.02,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if not show:
        plt.close(figure)
    # end if not show
# EOF


"""
plot_cross_temporal_result_matrices
Plot dynamic-time by static-time matrices for every shared stimulus subset.

INPUT:
    - results: dict[str, dict[str, np.ndarray]] -> cross-temporal results
    - times_ms: dict[str, np.ndarray] -> time coordinates by condition
    - matrix_key: str -> result matrix to plot
    - output_path: Path -> destination PNG path
    - dpi: int -> output resolution
    - show: bool -> keep the figure open for interactive display

OUTPUT:
    - None
"""
def plot_cross_temporal_result_matrices(
        results: dict[str, dict[str, np.ndarray]],
        times_ms: dict[str, np.ndarray],
        matrix_key: str,
        output_path: Path,
        dpi: int,
        show: bool = False,
        ) -> None:
    subset_names = list(results)
    n_columns = min(3, len(subset_names))
    n_rows = int(np.ceil(len(subset_names) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.5 * n_columns, 4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    if matrix_key == "drsa_similarity":
        cmap, vmin, vmax = "coolwarm", -1, 1
        colorbar_label = "RDM similarity (r)"
        figure_title = "Cross-temporal static-dynamic dRSA"
    else:
        cmap, vmin, vmax = "viridis", 0, 90
        colorbar_label = "Mean principal angle (degrees)"
        figure_title = "Cross-temporal static-dynamic PC rotation"
    # end if matrix_key

    static_times = times_ms["static"]
    dynamic_times = times_ms["dynamic"]
    extent = (
        static_times[0], static_times[-1],
        dynamic_times[0], dynamic_times[-1],
    )
    image = None
    for subset_index, subset_name in enumerate(subset_names):
        row_index, column_index = divmod(subset_index, n_columns)
        axis = axes[row_index, column_index]
        image = axis.imshow(
            results[subset_name][matrix_key],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set(
            xlabel="Static time (ms)",
            ylabel="Dynamic time (ms)",
            title=subset_name,
        )
    # end for subset_index, subset_name
    for empty_index in range(len(subset_names), n_rows * n_columns):
        row_index, column_index = divmod(empty_index, n_columns)
        axes[row_index, column_index].set_visible(False)
    # end for empty_index

    figure.suptitle(figure_title, fontsize=15, y=1.035)
    figure.colorbar(
        image, ax=axes, label=colorbar_label, shrink=0.86, pad=0.02,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if not show:
        plt.close(figure)
    # end if not show
# EOF


"""
matrix_summary
Summarize adjacent-time and all-off-diagonal values from a square matrix.

INPUT:
    - matrix: np.ndarray -> square time-by-time analysis matrix

OUTPUT:
    - adjacent_mean: float -> mean of the first upper diagonal
    - off_diagonal_mean: float -> mean across all non-diagonal cells
"""
def matrix_summary(matrix: np.ndarray) -> tuple[float, float]:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    # end if matrix shape
    adjacent_mean = np.nanmean(np.diag(matrix, k=1))
    off_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diagonal_mean = np.nanmean(matrix[off_diagonal_mask])
    return float(adjacent_mean), float(off_diagonal_mean)
# EOF


"""
save_results
Save full matrices, selected identities, configuration, and compact summaries.

INPUT:
    - output_dir: Path -> destination directory
    - cfg: Cfg -> analysis configuration
    - results: dict -> matrices by condition and subset
    - subsets: dict -> stimulus indices by condition and subset
    - scores: dict[str, np.ndarray] -> population-response scores by condition
    - shared_stimuli: list[str] -> matched stimulus identities
    - times_ms: dict[str, np.ndarray] -> time coordinates by condition
    - cross_results: dict -> cross-temporal matrices by shared subset
    - cross_subsets: dict[str, np.ndarray] -> cross-temporal stimulus indices

OUTPUT:
    - None
"""
def save_results(
        output_dir: Path,
        cfg: Cfg,
        results: dict,
        subsets: dict,
        scores: dict[str, np.ndarray],
        shared_stimuli: list[str],
        times_ms: dict[str, np.ndarray],
        cross_results: dict[str, dict[str, np.ndarray]],
        cross_subsets: dict[str, np.ndarray],
        ) -> None:
    arrays = {
        "shared_stimuli": np.asarray(shared_stimuli),
    }
    selection_manifest = {}
    summary_rows = []

    for condition in ("static", "dynamic"):
        arrays[f"{condition}__times_ms"] = times_ms[condition]
        arrays[f"{condition}__population_scores"] = scores[condition]
        selection_manifest[condition] = {}
        for subset_name, indices in subsets[condition].items():
            prefix = f"{condition}__{subset_name}"
            arrays[f"{prefix}__indices"] = indices
            arrays[f"{prefix}__drsa_autocorrelation"] = (
                results[condition][subset_name]["drsa_autocorrelation"]
            )
            arrays[f"{prefix}__pc_rotation_degrees"] = (
                results[condition][subset_name]["pc_rotation_degrees"]
            )
            selection_manifest[condition][subset_name] = [
                shared_stimuli[index] for index in indices
            ]

            drsa_adjacent, drsa_off_diagonal = matrix_summary(
                results[condition][subset_name]["drsa_autocorrelation"]
            )
            rotation_adjacent, rotation_off_diagonal = matrix_summary(
                results[condition][subset_name]["pc_rotation_degrees"]
            )
            summary_rows.append({
                "condition": condition,
                "subset": subset_name,
                "n_stimuli": len(indices),
                "drsa_adjacent": drsa_adjacent,
                "drsa_mean_off_diagonal": drsa_off_diagonal,
                "rotation_adjacent_degrees": rotation_adjacent,
                "rotation_mean_off_diagonal_degrees": rotation_off_diagonal,
            })
        # end for subset_name, indices
    # end for condition

    if cross_results:
        selection_manifest["cross_temporal"] = {}
        for subset_name, indices in cross_subsets.items():
            prefix = f"cross_temporal__{subset_name}"
            arrays[f"{prefix}__indices"] = indices
            arrays[f"{prefix}__drsa_similarity"] = (
                cross_results[subset_name]["drsa_similarity"]
            )
            arrays[f"{prefix}__pc_rotation_degrees"] = (
                cross_results[subset_name]["pc_rotation_degrees"]
            )
            selection_manifest["cross_temporal"][subset_name] = [
                shared_stimuli[index] for index in indices
            ]
        # end for subset_name, indices
    # end if cross_results

    np.savez_compressed(output_dir / "manifold_dynamics_results.npz", **arrays)

    config_dict = asdict(cfg)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    # end with open config
    with open(output_dir / "selected_stimuli.json", "w") as f:
        json.dump(selection_manifest, f, indent=2)
    # end with open selected stimuli
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    # end with open summary
# EOF


"""
main
Load matched recordings, select manifold scales, compute analyses, and save them.
"""
def main() -> None:
    cfg = parse_args()
    data_dir = Path(paths["data_path"]) / "data"
    static_path = Path(
        cfg.static_path
        or data_dir / f"{cfg.static_exp_name}_natraster_img.mat"
    ).expanduser()
    dynamic_path = Path(
        cfg.dynamic_path
        or data_dir / f"{cfg.dynamic_exp_name}_natraster_vid.mat"
    ).expanduser()
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / "static_dynamic_manifold_dynamics"
        / f"{cfg.dynamic_exp_name}_vs_{cfg.static_exp_name}"
        / channel_selection_name(cfg)
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_static, static_names = load_natraster(
        static_path, good_channels=cfg.good_channels,
    )
    loaded_dynamic, dynamic_names = load_natraster(
        dynamic_path, good_channels=cfg.good_channels,
    )
    static_rasters, dynamic_rasters, shared_stimuli = (
        match_static_dynamic_rasters(
            loaded_static, static_names, loaded_dynamic, dynamic_names,
        )
    )
    if cfg.subset_size > len(shared_stimuli):
        raise ValueError(
            f"subset_size={cfg.subset_size} exceeds the "
            f"{len(shared_stimuli)} matched stimuli."
        )
    # end if subset_size

    if cfg.static_crop_ms is not None:
        static_stop = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
        static_rasters = static_rasters[:, :static_stop, :]
    # end if static_crop_ms
    if cfg.dynamic_crop_ms is not None:
        dynamic_stop = int(round(cfg.dynamic_crop_ms * cfg.source_fs / 1000))
        dynamic_rasters = dynamic_rasters[:, :dynamic_stop, :]
    # end if dynamic_crop_ms

    static_ts = TimeSeries(static_rasters, fs=cfg.source_fs)
    dynamic_ts = TimeSeries(dynamic_rasters, fs=cfg.source_fs)
    if cfg.analysis_fs != cfg.source_fs:
        static_ts.resample(cfg.analysis_fs)
        dynamic_ts.resample(cfg.analysis_fs)
    # end if analysis_fs
    static_rasters = static_ts.get_array()
    dynamic_rasters = dynamic_ts.get_array()

    scores = {
        "static": population_response_scores(
            static_rasters, cfg.analysis_fs, cfg.static_selection_window_ms,
        ),
        "dynamic": population_response_scores(
            dynamic_rasters, cfg.analysis_fs, cfg.dynamic_selection_window_ms,
        ),
    }
    subsets = build_condition_subsets(
        scores["static"], scores["dynamic"], cfg,
    )

    results = {}
    for condition, condition_rasters in (
            ("static", static_rasters), ("dynamic", dynamic_rasters),
            ):
        print(
            f"Computing {condition}: {condition_rasters.shape[0]} channels, "
            f"{condition_rasters.shape[1]} timepoints, "
            f"{condition_rasters.shape[2]} stimuli",
            flush=True,
        )
        results[condition] = compute_manifold_dynamics(
            condition_rasters,
            subsets[condition],
            rdm_metric=cfg.rdm_metric,
            rsa_metric=cfg.rsa_metric,
            n_pc_components=cfg.n_pc_components,
        )
    # end for condition, condition_rasters

    cross_subsets = {}
    cross_results = {}
    if cfg.cross_temporal:
        cross_subsets = build_cross_temporal_subsets(subsets, cfg)
        print(
            "Computing cross-temporal static-dynamic analyses: "
            f"{dynamic_rasters.shape[1]} dynamic x "
            f"{static_rasters.shape[1]} static timepoints",
            flush=True,
        )
        cross_results = compute_cross_temporal_manifold_dynamics(
            dynamic_rasters,
            static_rasters,
            cross_subsets,
            rdm_metric=cfg.rdm_metric,
            rsa_metric=cfg.rsa_metric,
            n_pc_components=cfg.n_pc_components,
        )
    # end if cross_temporal

    times_ms = {
        "static": np.arange(static_rasters.shape[1]) * 1000 / cfg.analysis_fs,
        "dynamic": np.arange(dynamic_rasters.shape[1]) * 1000 / cfg.analysis_fs,
    }
    save_results(
        output_dir,
        cfg,
        results,
        subsets,
        scores,
        shared_stimuli,
        times_ms,
        cross_results,
        cross_subsets,
    )
    plot_result_matrices(
        results,
        times_ms,
        "drsa_autocorrelation",
        output_dir / "drsa_autocorrelation.png",
        cfg.figure_dpi,
        show=cfg.show,
    )
    plot_result_matrices(
        results,
        times_ms,
        "pc_rotation_degrees",
        output_dir / "pc_rotation_degrees.png",
        cfg.figure_dpi,
        show=cfg.show,
    )
    if cfg.cross_temporal:
        plot_cross_temporal_result_matrices(
            cross_results,
            times_ms,
            "drsa_similarity",
            output_dir / "cross_temporal_drsa_similarity.png",
            cfg.figure_dpi,
            show=cfg.show,
        )
        plot_cross_temporal_result_matrices(
            cross_results,
            times_ms,
            "pc_rotation_degrees",
            output_dir / "cross_temporal_pc_rotation_degrees.png",
            cfg.figure_dpi,
            show=cfg.show,
        )
    # end if cross_temporal

    print(f"Results saved to {output_dir}", flush=True)
    if cfg.show:
        plt.show()
    # end if show
# EOF


if __name__ == "__main__":
    main()
# EOF
