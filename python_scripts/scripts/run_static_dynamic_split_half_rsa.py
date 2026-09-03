import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

import matplotlib

# File output is the default; avoid macOS GUI-backend failures in headless runs.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
# end if --show not requested
import matplotlib.pyplot as plt
import numpy as np
import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as file:
    config = yaml.safe_load(file)
# end with open

paths = config[ENV]["paths"]
sys.path.extend([paths["src_path"], paths["useful_stuff_path"]])

from project_specific_utils import (
    average_presentations,
    average_repetition_halves,
    compute_rdm_timeseries,
    cross_temporal_similarity,
    load_raster,
    load_reliable_channels,
    rowwise_similarity,
)
from useful_stuff.general_utils import TimeSeries


@dataclass
class Cfg:
    static_exp_name: str = "baby1_260718to27"
    dynamic_exp_name: str = "baby1_260716to24"
    static_path: str | None = None
    dynamic_path: str | None = None
    output_dir: str | None = None

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = (101, 109)
    # Optional reliability list intersected with the good_channels range.
    reliable_channels_config: str | None = None
    reliable_channels_key: str | None = None
    static_crop_ms: float = 1000
    dynamic_crop_ms: float | None = None
    source_fs: float = 1000
    new_fs: float = 100
    rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "correlation"
    n_split_repeats: int = 10
    # Stimuli presented fewer times than this cannot be split in half.
    min_repetitions: int = 2
    random_seed: int = 0
    static_reference_window_ms: tuple[float, float] = (0, 1000)
    figure_dpi: int = 200
    show: bool = False
# EOF


"""
parse_args
Parse paths, channels, timing, reliability, and plotting parameters.

OUTPUT:
    - cfg: Cfg -> validated analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compare static-dynamic RDM similarity with raw-response and "
            "RDM split-half self-consistency within both conditions."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument("--static_path")
    parser.add_argument("--dynamic_path")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--good_channels",
        nargs=2,
        type=int,
        default=list(Cfg.good_channels),
        metavar=("FIRST", "LAST"),
    )
    parser.add_argument("--all_channels", action="store_true")
    parser.add_argument(
        "--reliable_channels_config",
        help=(
            "YAML reliability file. Its non-contiguous channel list is "
            "intersected with --good_channels when both are supplied."
        ),
    )
    parser.add_argument(
        "--reliable_channels_key",
        help=(
            "Dataset key in the reliability YAML; defaults to the static "
            "session name."
        ),
    )
    parser.add_argument("--static_crop_ms", type=float, default=Cfg.static_crop_ms)
    parser.add_argument("--dynamic_crop_ms", type=float)
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--new_fs", type=float, default=Cfg.new_fs)
    parser.add_argument("--rdm_metric", default=Cfg.rdm_metric)
    parser.add_argument(
        "--rsa_metric",
        choices=("correlation", "spearman"),
        default=Cfg.rsa_metric,
    )
    parser.add_argument("--n_split_repeats", type=int, default=Cfg.n_split_repeats)
    parser.add_argument(
        "--min_repetitions", type=int, default=Cfg.min_repetitions,
        help="Drop stimuli with fewer repetitions than this.",
    )
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument(
        "--static_reference_window_ms",
        nargs=2,
        type=float,
        default=list(Cfg.static_reference_window_ms),
        metavar=("START", "END"),
    )
    parser.add_argument("--figure_dpi", type=int, default=Cfg.figure_dpi)
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=Cfg.show,
    )
    args = parser.parse_args()

    if args.all_channels:
        args.good_channels = None
    else:
        first_channel, last_channel = args.good_channels
        if first_channel < 1 or last_channel < first_channel:
            parser.error("--good_channels must be an increasing positive range.")
        # end if channel range
        args.good_channels = tuple(args.good_channels)
    # end if args.all_channels
    del args.all_channels

    positive_values = {
        "--source_fs": args.source_fs,
        "--new_fs": args.new_fs,
        "--static_crop_ms": args.static_crop_ms,
        "--figure_dpi": args.figure_dpi,
    }
    for argument_name, value in positive_values.items():
        if value <= 0:
            parser.error(f"{argument_name} must be positive.")
        # end if value
    # end for argument_name, value
    if args.dynamic_crop_ms is not None and args.dynamic_crop_ms <= 0:
        parser.error("--dynamic_crop_ms must be positive or omitted.")
    # end if args.dynamic_crop_ms
    if args.n_split_repeats < 1:
        parser.error("--n_split_repeats must be positive.")
    # end if args.n_split_repeats
    start_ms, end_ms = args.static_reference_window_ms
    if start_ms < 0 or end_ms <= start_ms:
        parser.error("--static_reference_window_ms must be an increasing non-negative range.")
    # end if reference window
    args.static_reference_window_ms = tuple(args.static_reference_window_ms)
    return Cfg(**vars(args))
# EOF


"""
stimulus_identity
Extract a modality-independent identity while excluding timed-image controls.

INPUT:
    - stimulus_name: str -> presentation filename
    - condition: str -> static or dynamic

OUTPUT:
    - identity: str | None -> matched identity, or None for an excluded stimulus
"""
def stimulus_identity(stimulus_name: str, condition: str) -> str | None:
    stimulus_stem = Path(stimulus_name).stem
    if condition == "static":
        if stimulus_stem.startswith(("img_2000ms_", "img_2250ms_")):
            return None
        # end if timed-image condition
        prefix = "img_"
    elif condition == "dynamic":
        prefix = "vid_"
    else:
        raise ValueError("condition must be 'static' or 'dynamic'.")
    # end if condition
    if not stimulus_stem.startswith(prefix):
        return None
    # end if not stimulus_stem.startswith
    return stimulus_stem[len(prefix):]
# EOF


"""
resample_presentations
Resample presentation rasters along time using the project TimeSeries utility.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations
    - source_fs: float -> source sampling frequency
    - target_fs: float -> requested sampling frequency

OUTPUT:
    - resampled_rasters: np.ndarray -> channels x resampled time x presentations
"""
def resample_presentations(
        rasters: np.ndarray,
        source_fs: float,
        target_fs: float,
        ) -> np.ndarray:
    if source_fs == target_fs:
        return rasters
    # end if source_fs == target_fs
    raster_ts = TimeSeries(rasters, fs=source_fs)
    raster_ts.resample(target_fs)
    return raster_ts.get_array()
# EOF


"""
compute_condition_split_half
Repeat random repetition splits and compute time-resolved raw and RDM
self-consistency for one condition.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations
    - presentation_identities: list[str] -> identity for every presentation
    - stimulus_order: list[str] -> matched stimulus order
    - cfg: Cfg -> analysis configuration
    - rng: np.random.Generator -> random generator

OUTPUT:
    - raw_consistency: np.ndarray -> split repeat x time raw correlations
    - rdm_consistency: np.ndarray -> split repeat x time RDM correlations
"""
def compute_condition_split_half(
        rasters: np.ndarray,
        presentation_identities: list[str],
        stimulus_order: list[str],
        cfg: Cfg,
        rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
    raw_consistency = []
    rdm_consistency = []
    for split_index in range(cfg.n_split_repeats):
        first_half, second_half = average_repetition_halves(
            rasters,
            presentation_identities,
            stimulus_order,
            rng,
        )

        # Each timepoint is compared over the same channel-by-stimulus values.
        first_raw = first_half.transpose(1, 0, 2).reshape(first_half.shape[1], -1)
        second_raw = second_half.transpose(1, 0, 2).reshape(second_half.shape[1], -1)
        raw_consistency.append(rowwise_similarity(first_raw, second_raw))

        first_rdms = compute_rdm_timeseries(first_half, cfg.rdm_metric)
        second_rdms = compute_rdm_timeseries(second_half, cfg.rdm_metric)
        rdm_consistency.append(
            rowwise_similarity(first_rdms, second_rdms, metric=cfg.rsa_metric)
        )
        print(f"  split {split_index + 1}/{cfg.n_split_repeats}")
    # end for split_index
    return np.stack(raw_consistency), np.stack(rdm_consistency)
# EOF


"""
plot_reliability_summary
Plot raw and RDM self-consistency for both conditions plus cross-condition RDM
similarity and the maximum static-explained dynamic variance.

INPUT:
    - results: dict[str, np.ndarray] -> analysis arrays and time coordinates
    - cfg: Cfg -> plotting configuration
    - output_path: Path -> destination PNG

OUTPUT:
    - None
"""
def plot_reliability_summary(results: dict, cfg: Cfg, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    condition_specs = (
        ("static", "tab:blue", results["static_times_ms"]),
        ("dynamic", "tab:orange", results["dynamic_times_ms"]),
    )
    for condition, color, times_ms in condition_specs:
        raw_values = results[f"{condition}_raw_split_half"]
        rdm_values = results[f"{condition}_rdm_split_half"]
        raw_mean = np.nanmean(raw_values, axis=0)
        rdm_mean = np.nanmean(rdm_values, axis=0)
        axes[0, 0].plot(times_ms, raw_mean, color=color, label=condition)
        axes[0, 0].fill_between(
            times_ms,
            np.nanpercentile(raw_values, 2.5, axis=0),
            np.nanpercentile(raw_values, 97.5, axis=0),
            color=color,
            alpha=0.2,
        )
        axes[0, 1].plot(times_ms, rdm_mean, color=color, label=condition)
        axes[0, 1].fill_between(
            times_ms,
            np.nanpercentile(rdm_values, 2.5, axis=0),
            np.nanpercentile(rdm_values, 97.5, axis=0),
            color=color,
            alpha=0.2,
        )
    # end for condition
    axes[0, 0].set(
        title="Raw response split-half consistency",
        xlabel="Time from onset (ms)",
        ylabel="Pearson correlation",
    )
    axes[0, 1].set(
        title="RDM split-half consistency",
        xlabel="Time from onset (ms)",
        ylabel=f"RDM {cfg.rsa_metric}",
    )
    for axis in axes[0]:
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.legend()
    # end for axis

    cross_image = axes[1, 0].imshow(
        results["static_dynamic_rdm_similarity"] ** 2,
        origin="lower",
        aspect="auto",
        extent=(
            results["static_times_ms"][0],
            results["static_times_ms"][-1],
            results["dynamic_times_ms"][0],
            results["dynamic_times_ms"][-1],
        ),
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set(
        title="Static explanation of dynamic RDMs",
        xlabel="Static time (ms)",
        ylabel="Dynamic time (ms)",
    )
    figure.colorbar(cross_image, ax=axes[1, 0], label="Squared RDM correlation")

    dynamic_rdm_reliability = np.nanmean(results["dynamic_rdm_split_half"], axis=0)
    axes[1, 1].plot(
        results["dynamic_times_ms"],
        results["maximum_static_explained_variance"],
        color="black",
        linewidth=2.5,
        label="max static RDM $r^2$",
    )
    axes[1, 1].plot(
        results["dynamic_times_ms"],
        dynamic_rdm_reliability,
        color="tab:orange",
        label="dynamic RDM split-half",
    )
    axes[1, 1].set(
        title="Static explanation against dynamic consistency",
        xlabel="Dynamic time from onset (ms)",
        ylabel="Score",
        ylim=(-0.05, 1.05),
    )
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(figure)
    return None
# EOF


"""
plot_explanation_scatter
Scatter static-explained dynamic RDM variance against raw and RDM split-half
self-consistency, coloring each point by dynamic time.

INPUT:
    - results: dict[str, np.ndarray] -> analysis arrays and time coordinates
    - cfg: Cfg -> plotting configuration
    - output_path: Path -> destination PNG

OUTPUT:
    - None
"""
def plot_explanation_scatter(results: dict, cfg: Cfg, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    consistency_items = (
        ("dynamic_raw_split_half", "Raw split-half correlation"),
        ("dynamic_rdm_split_half", f"RDM split-half {cfg.rsa_metric}"),
    )
    for axis, (result_name, xlabel) in zip(axes, consistency_items):
        consistency = np.nanmean(results[result_name], axis=0)
        scatter = axis.scatter(
            consistency,
            results["maximum_static_explained_variance"],
            c=results["dynamic_times_ms"],
            cmap="viridis",
            s=24,
            alpha=0.8,
        )
        axis.set(
            xlabel=xlabel,
        )
        axis.axhline(0, color="0.7", linewidth=0.8)
        axis.axvline(0, color="0.7", linewidth=0.8)
    # end for axis
    axes[0].set_ylabel(
        "Maximum static-explained dynamic RDM variance ($r^2$)"
    )
    figure.suptitle("Static explanation versus within-condition consistency")
    figure.subplots_adjust(
        left=0.09, right=0.86, bottom=0.12, top=0.88, wspace=0.2,
    )
    colorbar_axis = figure.add_axes((0.89, 0.16, 0.02, 0.66))
    figure.colorbar(
        scatter,
        cax=colorbar_axis,
        label="Dynamic time from onset (ms)",
    )
    figure.savefig(output_path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(figure)
    return None
# EOF


"""
main
Load presentation-level rasters, align static and dynamic identities, compute
split-half consistency and cross-condition RDM similarity, and save results.

OUTPUT:
    - None
"""
def main() -> None:
    cfg = parse_args()
    data_dir = Path(paths["data_path"]) / "data"
    static_path = Path(
        cfg.static_path or data_dir / f"{cfg.static_exp_name}_raster_img.mat"
    ).expanduser()
    dynamic_path = Path(
        cfg.dynamic_path or data_dir / f"{cfg.dynamic_exp_name}_raster_vid.mat"
    ).expanduser()
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / "static_dynamic_split_half_rsa"
        / f"{cfg.dynamic_exp_name}_vs_{cfg.static_exp_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.good_channels is None:
        channel_slice = slice(None)
    else:
        first_channel, last_channel = cfg.good_channels
        channel_slice = slice(first_channel - 1, last_channel)
    # end if cfg.good_channels
    selected_channel_numbers = None
    if cfg.reliable_channels_config is not None:
        # The reliability list is one-based and gets intersected with the
        # optional contiguous range, matching load_natraster's behaviour.
        reliable_channels = load_reliable_channels(
            cfg.reliable_channels_config,
            cfg.reliable_channels_key or cfg.static_exp_name,
        )
        if cfg.good_channels is None:
            selected_channel_numbers = np.asarray(reliable_channels)
        else:
            selected_channel_numbers = np.asarray([
                channel for channel in reliable_channels
                if first_channel <= channel <= last_channel
            ])
        # end if cfg.good_channels is None
        if selected_channel_numbers.size == 0:
            raise ValueError(
                "No reliable channels fall inside the selected range."
            )
        # end if empty selection
        channel_slice = selected_channel_numbers - 1
        print(
            f"Reliable channels retained: {selected_channel_numbers.tolist()}"
        )
    # end if cfg.reliable_channels_config is not None
    static_end_sample = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
    dynamic_end_sample = None
    if cfg.dynamic_crop_ms is not None:
        dynamic_end_sample = int(round(cfg.dynamic_crop_ms * cfg.source_fs / 1000))
    # end if cfg.dynamic_crop_ms

    print("Loading presentation-level static rasters...")
    static_rasters, static_names = load_raster(
        static_path, channel_slice=channel_slice, end_sample=static_end_sample,
    )
    print("Loading presentation-level dynamic rasters...")
    dynamic_rasters, dynamic_names = load_raster(
        dynamic_path, channel_slice=channel_slice, end_sample=dynamic_end_sample,
    )

    static_identities = [stimulus_identity(name, "static") for name in static_names]
    dynamic_identities = [stimulus_identity(name, "dynamic") for name in dynamic_names]
    static_identity_set = {identity for identity in static_identities if identity is not None}
    dynamic_identity_set = {identity for identity in dynamic_identities if identity is not None}
    shared_stimuli = sorted(static_identity_set & dynamic_identity_set)
    # Splitting the repetitions in half needs at least two of them, so stimuli
    # presented once in either condition cannot enter the reliability estimate.
    static_counts = Counter(static_identities)
    dynamic_counts = Counter(dynamic_identities)
    under_sampled = [
        identity for identity in shared_stimuli
        if min(static_counts[identity], dynamic_counts[identity])
        < cfg.min_repetitions
    ]
    if under_sampled:
        print(
            f"Dropping {len(under_sampled)} of {len(shared_stimuli)} stimuli "
            f"with fewer than {cfg.min_repetitions} repetitions in one "
            f"condition: {under_sampled[:5]}"
        )
        shared_stimuli = [
            identity for identity in shared_stimuli
            if identity not in set(under_sampled)
        ]
    # end if under_sampled
    if len(shared_stimuli) < 3:
        raise ValueError("Need at least three stimuli shared by both conditions.")
    # end if len(shared_stimuli)

    shared_stimulus_set = set(shared_stimuli)
    static_keep = np.asarray([identity in shared_stimulus_set for identity in static_identities])
    dynamic_keep = np.asarray([identity in shared_stimulus_set for identity in dynamic_identities])
    static_rasters = static_rasters[:, :, static_keep]
    dynamic_rasters = dynamic_rasters[:, :, dynamic_keep]
    static_identities = [identity for identity, keep in zip(static_identities, static_keep) if keep]
    dynamic_identities = [identity for identity, keep in zip(dynamic_identities, dynamic_keep) if keep]

    print("Resampling presentation rasters...")
    static_rasters = resample_presentations(static_rasters, cfg.source_fs, cfg.new_fs)
    dynamic_rasters = resample_presentations(dynamic_rasters, cfg.source_fs, cfg.new_fs)
    static_times_ms = np.arange(static_rasters.shape[1]) * 1000 / cfg.new_fs
    dynamic_times_ms = np.arange(dynamic_rasters.shape[1]) * 1000 / cfg.new_fs

    rng = np.random.default_rng(cfg.random_seed)
    print("Static split-half consistency:")
    static_raw_split_half, static_rdm_split_half = compute_condition_split_half(
        static_rasters, static_identities, shared_stimuli, cfg, rng,
    )
    print("Dynamic split-half consistency:")
    dynamic_raw_split_half, dynamic_rdm_split_half = compute_condition_split_half(
        dynamic_rasters, dynamic_identities, shared_stimuli, cfg, rng,
    )

    print("Computing full-condition RDM timecourses...")
    static_means = average_presentations(static_rasters, static_identities, shared_stimuli)
    dynamic_means = average_presentations(dynamic_rasters, dynamic_identities, shared_stimuli)
    static_rdms = compute_rdm_timeseries(static_means, cfg.rdm_metric)
    dynamic_rdms = compute_rdm_timeseries(dynamic_means, cfg.rdm_metric)
    static_dynamic_rdm_similarity = cross_temporal_similarity(
        dynamic_rdms, static_rdms, metric=cfg.rsa_metric,
    )

    reference_start_ms, reference_end_ms = cfg.static_reference_window_ms
    reference_indices = (
        (static_times_ms >= reference_start_ms)
        & (static_times_ms <= reference_end_ms)
    )
    if not reference_indices.any():
        raise ValueError("The static reference window contains no samples.")
    # end if not reference_indices.any()
    maximum_static_explained_variance = np.nanmax(
        static_dynamic_rdm_similarity[:, reference_indices] ** 2,
        axis=1,
    )

    results = {
        "static_raw_split_half": static_raw_split_half,
        "dynamic_raw_split_half": dynamic_raw_split_half,
        "static_rdm_split_half": static_rdm_split_half,
        "dynamic_rdm_split_half": dynamic_rdm_split_half,
        "static_dynamic_rdm_similarity": static_dynamic_rdm_similarity,
        "maximum_static_explained_variance": maximum_static_explained_variance,
        "static_times_ms": static_times_ms,
        "dynamic_times_ms": dynamic_times_ms,
        "shared_stimuli": np.asarray(shared_stimuli),
    }
    if selected_channel_numbers is not None:
        results["selected_channel_numbers"] = selected_channel_numbers
    # end if an explicit channel list was used
    summary_path = output_dir / "split_half_and_static_explanation.png"
    scatter_path = output_dir / "static_explanation_vs_consistency.png"
    data_path = output_dir / "split_half_static_dynamic_rsa.npz"
    config_path = output_dir / "config.json"
    plot_reliability_summary(results, cfg, summary_path)
    plot_explanation_scatter(results, cfg, scatter_path)
    np.savez_compressed(data_path, **results)
    with open(config_path, "w") as file:
        json.dump(asdict(cfg), file, indent=2, default=str)
    # end with open

    print(f"Environment: {ENV}")
    print(f"Shared stimuli: {len(shared_stimuli)}")
    print(f"Static presentations retained: {static_rasters.shape[2]}")
    print(f"Dynamic presentations retained: {dynamic_rasters.shape[2]}")
    print(f"Saved results: {output_dir}")

    if cfg.show:
        for figure_path in (summary_path, scatter_path):
            image = plt.imread(figure_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis("off")
        # end for figure_path
        plt.show()
    # end if cfg.show
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
