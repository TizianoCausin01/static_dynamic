import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

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
    channelwise_lag_curves,
    channelwise_static_dynamic_correlation,
    load_natraster,
    match_static_dynamic_rasters,
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
    static_crop_ms: float | None = 1000
    new_fs: float = 100
    max_lag_ms: float = 700
    static_slice_ms: tuple[float, ...] = (0, 250, 500, 750)
    figure_dpi: int = 200
    show: bool = False
# EOF


"""
parse_args
Parse data, channel, timing, slice, and output parameters.

OUTPUT:
    - cfg: Cfg -> validated channel-wise correlation configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate static and dynamic neural responses across matched "
            "trials separately for every channel."
        )
    )
    parser.add_argument(
        "--static_exp_name", default=Cfg.static_exp_name
    )
    parser.add_argument(
        "--dynamic_exp_name", default=Cfg.dynamic_exp_name
    )
    parser.add_argument("--static_path")
    parser.add_argument("--dynamic_path")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--good_channels",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        default=list(Cfg.good_channels),
        help=(
            "Inclusive one-based MATLAB channel range. "
            "Use --all_channels to override."
        ),
    )
    parser.add_argument(
        "--all_channels",
        action="store_true",
        help="Analyze all channels instead of --good_channels.",
    )
    parser.add_argument(
        "--static_crop_ms", type=float, default=Cfg.static_crop_ms
    )
    parser.add_argument("--new_fs", type=float, default=Cfg.new_fs)
    parser.add_argument(
        "--max_lag_ms", type=float, default=Cfg.max_lag_ms
    )
    parser.add_argument(
        "--static_slice_ms",
        nargs="+",
        type=float,
        default=list(Cfg.static_slice_ms),
        help=(
            "Static reference times whose dynamic correlation timecourses "
            "are plotted for every channel."
        ),
    )
    parser.add_argument(
        "--figure_dpi", type=int, default=Cfg.figure_dpi
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=Cfg.show,
    )
    args = parser.parse_args()

    if args.new_fs <= 0:
        parser.error("--new_fs must be positive.")
    # end if args.new_fs
    if args.static_crop_ms is not None and args.static_crop_ms <= 0:
        parser.error("--static_crop_ms must be positive.")
    # end if args.static_crop_ms
    if args.max_lag_ms < 0:
        parser.error("--max_lag_ms must be non-negative.")
    # end if args.max_lag_ms
    if args.figure_dpi < 1:
        parser.error("--figure_dpi must be a positive integer.")
    # end if args.figure_dpi
    if args.all_channels:
        args.good_channels = None
    else:
        first_channel, last_channel = args.good_channels
        if first_channel < 1 or last_channel < first_channel:
            parser.error(
                "--good_channels must be an increasing positive range."
            )
        # end if channel range
        args.good_channels = tuple(args.good_channels)
    # end if args.all_channels

    del args.all_channels
    args.static_slice_ms = tuple(args.static_slice_ms)
    return Cfg(**vars(args))
# EOF


"""
plot_average_correlation_matrix
Plot the channel-averaged dynamic-time by static-time correlation matrix.

INPUT:
    - average_corr_matrix: np.ndarray -> dynamic time x static time
    - static_times_ms: np.ndarray -> static time coordinate in milliseconds
    - dynamic_times_ms: np.ndarray -> dynamic time coordinate in milliseconds
    - output_path: Path -> destination PNG path
    - dpi: int -> output resolution

OUTPUT:
    - None
"""
def plot_average_correlation_matrix(
        average_corr_matrix: np.ndarray,
        static_times_ms: np.ndarray,
        dynamic_times_ms: np.ndarray,
        output_path: Path,
        dpi: int,
        ) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        average_corr_matrix,
        origin="lower",
        aspect="auto",
        extent=(
            static_times_ms[0],
            static_times_ms[-1],
            dynamic_times_ms[0],
            dynamic_times_ms[-1],
        ),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    shared_end_ms = min(static_times_ms[-1], dynamic_times_ms[-1])
    axis.plot(
        [0, shared_end_ms],
        [0, shared_end_ms],
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    axis.set(
        title="Channel-averaged static/dynamic trial correlation",
        xlabel="Static time from onset (ms)",
        ylabel="Dynamic time from onset (ms)",
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Pearson correlation across matched trials")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return None
# EOF


"""
plot_static_time_slices
For selected static reference times, plot the dynamic correlation timecourse of
every channel and their channel average.

INPUT:
    - channel_corr_matrices: np.ndarray -> channels x dynamic time x static time
    - channel_numbers: np.ndarray -> one-based MATLAB channel identifiers
    - static_times_ms: np.ndarray -> static time coordinate in milliseconds
    - dynamic_times_ms: np.ndarray -> dynamic time coordinate in milliseconds
    - requested_slice_ms: tuple[float, ...] -> requested static reference times
    - output_path: Path -> destination PNG path
    - dpi: int -> output resolution

OUTPUT:
    - selected_slice_indices: np.ndarray -> nearest static sample indices
"""
def plot_static_time_slices(
        channel_corr_matrices: np.ndarray,
        channel_numbers: np.ndarray,
        static_times_ms: np.ndarray,
        dynamic_times_ms: np.ndarray,
        requested_slice_ms: tuple[float, ...],
        output_path: Path,
        dpi: int,
        ) -> np.ndarray:
    selected_slice_indices = np.array([
        np.argmin(np.abs(static_times_ms - slice_ms))
        for slice_ms in requested_slice_ms
    ])
    # Avoid plotting the same resampled static time more than once.
    selected_slice_indices = np.unique(selected_slice_indices)
    n_slices = len(selected_slice_indices)
    n_columns = min(2, n_slices)
    n_rows = int(np.ceil(n_slices / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7 * n_columns, 4 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(channel_numbers))
    )

    for axis, static_index in zip(
            axes.flat, selected_slice_indices, strict=False,
            ):
        slice_timecourses = channel_corr_matrices[:, :, static_index]
        for channel_index, channel_number in enumerate(channel_numbers):
            axis.plot(
                dynamic_times_ms,
                slice_timecourses[channel_index],
                color=colors[channel_index],
                alpha=0.7,
                linewidth=1,
                label=f"Ch {channel_number}",
            )
        # end for channel_index, channel_number
        axis.plot(
            dynamic_times_ms,
            np.nanmean(slice_timecourses, axis=0),
            color="black",
            linewidth=3,
            label="Channel average",
        )
        axis.axhline(0, color="0.6", linestyle="--", linewidth=1)
        axis.axvline(
            static_times_ms[static_index],
            color="0.35",
            linestyle=":",
            linewidth=1,
        )
        axis.set_title(
            f"Static reference: {static_times_ms[static_index]:g} ms"
        )
        axis.set_xlabel("Dynamic time from onset (ms)")
        axis.set_ylabel("Correlation across matched trials")
    # end for axis, static_index

    for axis in axes.flat[n_slices:]:
        axis.set_visible(False)
    # end for unused axis
    axes.flat[0].legend(
        loc="upper right",
        fontsize="small",
        ncols=2,
    )
    figure.suptitle(
        "Channel-wise dynamic timecourses from static-time slices"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return selected_slice_indices
# EOF


"""
plot_channel_lag_curves
Plot each channel's diagonal-averaged correlation as a function of lag.

INPUT:
    - lag_curves: np.ndarray -> channels x lag
    - channel_numbers: np.ndarray -> one-based MATLAB channel identifiers
    - lag_times_ms: np.ndarray -> lag coordinate in milliseconds
    - output_path: Path -> destination PNG path
    - dpi: int -> output resolution

OUTPUT:
    - None
"""
def plot_channel_lag_curves(
        lag_curves: np.ndarray,
        channel_numbers: np.ndarray,
        lag_times_ms: np.ndarray,
        output_path: Path,
        dpi: int,
        ) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(channel_numbers))
    )
    for channel_index, channel_number in enumerate(channel_numbers):
        axis.plot(
            lag_times_ms,
            lag_curves[channel_index],
            color=colors[channel_index],
            alpha=0.75,
            linewidth=1.25,
            label=f"Ch {channel_number}",
        )
    # end for channel_index, channel_number
    axis.plot(
        lag_times_ms,
        np.nanmean(lag_curves, axis=0),
        color="black",
        linewidth=3,
        label="Channel average",
    )
    axis.axhline(0, color="0.6", linestyle="--", linewidth=1)
    axis.axvline(0, color="0.35", linestyle=":", linewidth=1)
    axis.set(
        title="Channel-wise static/dynamic lag curves",
        xlabel="Lag: dynamic time - static time (ms)",
        ylabel="Mean correlation along matrix diagonal",
    )
    axis.legend(fontsize="small", ncols=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return None
# EOF


"""
main
Load and align static/dynamic rasters, compute trial correlations separately
for every channel, and save the matrix, time-slice, and lag visualizations.

OUTPUT:
    - None
"""
def main() -> None:
    cfg = parse_args()
    source_fs = 1000
    static_path = Path(
        cfg.static_path
        or Path(paths["data_path"])
        / "data"
        / f"{cfg.static_exp_name}_natraster_img.mat"
    ).expanduser()
    dynamic_path = Path(
        cfg.dynamic_path
        or Path(paths["data_path"])
        / "data"
        / f"{cfg.dynamic_exp_name}_natraster_vid.mat"
    ).expanduser()
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / "channelwise_static_dynamic_correlation"
        / f"{cfg.dynamic_exp_name}_vs_{cfg.static_exp_name}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_static, static_names = load_natraster(static_path)
    loaded_dynamic, dynamic_names = load_natraster(dynamic_path)
    static_rasters, dynamic_rasters, shared_stimuli = (
        match_static_dynamic_rasters(
            loaded_static,
            static_names,
            loaded_dynamic,
            dynamic_names,
        )
    )

    if cfg.good_channels is None:
        channel_indices = np.arange(static_rasters.shape[0])
    else:
        first_channel, last_channel = cfg.good_channels
        if last_channel > static_rasters.shape[0]:
            raise ValueError(
                f"Requested channel {last_channel}, but the data contain "
                f"{static_rasters.shape[0]} channels."
            )
        # end if last_channel
        channel_indices = np.arange(first_channel - 1, last_channel)
    # end if cfg.good_channels
    channel_numbers = channel_indices + 1

    if cfg.static_crop_ms is not None:
        static_end_index = int(
            round(cfg.static_crop_ms * source_fs / 1000)
        )
        static_rasters = static_rasters[:, :static_end_index, :]
    # end if cfg.static_crop_ms

    static_ts = TimeSeries(
        static_rasters[channel_indices], fs=source_fs
    )
    dynamic_ts = TimeSeries(
        dynamic_rasters[channel_indices], fs=source_fs
    )
    if cfg.new_fs != source_fs:
        static_ts.resample(cfg.new_fs)
        dynamic_ts.resample(cfg.new_fs)
    # end if cfg.new_fs

    channel_corr_matrices, average_corr_matrix = (
        channelwise_static_dynamic_correlation(
            dynamic_ts.get_array(),
            static_ts.get_array(),
        )
    )
    max_lag_samples = int(round(cfg.max_lag_ms * cfg.new_fs / 1000))
    lag_curves = channelwise_lag_curves(
        channel_corr_matrices,
        max_lag=max_lag_samples,
    )

    static_times_ms = (
        np.arange(static_ts.shape()[1]) * 1000 / cfg.new_fs
    )
    dynamic_times_ms = (
        np.arange(dynamic_ts.shape()[1]) * 1000 / cfg.new_fs
    )
    lag_times_ms = (
        np.arange(-max_lag_samples, max_lag_samples + 1)
        * 1000
        / cfg.new_fs
    )

    matrix_path = output_dir / "average_correlation_matrix.png"
    slices_path = output_dir / "channel_timecourse_slices.png"
    lag_path = output_dir / "channel_lag_curves.png"
    data_path = output_dir / "channelwise_correlations.npz"
    plot_average_correlation_matrix(
        average_corr_matrix,
        static_times_ms,
        dynamic_times_ms,
        matrix_path,
        cfg.figure_dpi,
    )
    selected_slice_indices = plot_static_time_slices(
        channel_corr_matrices,
        channel_numbers,
        static_times_ms,
        dynamic_times_ms,
        cfg.static_slice_ms,
        slices_path,
        cfg.figure_dpi,
    )
    plot_channel_lag_curves(
        lag_curves,
        channel_numbers,
        lag_times_ms,
        lag_path,
        cfg.figure_dpi,
    )
    np.savez_compressed(
        data_path,
        channel_corr_matrices=channel_corr_matrices,
        average_corr_matrix=average_corr_matrix,
        lag_curves=lag_curves,
        channel_numbers=channel_numbers,
        static_times_ms=static_times_ms,
        dynamic_times_ms=dynamic_times_ms,
        lag_times_ms=lag_times_ms,
        selected_static_slice_indices=selected_slice_indices,
        shared_stimuli=np.asarray(shared_stimuli),
    )

    undefined_fraction = np.mean(~np.isfinite(channel_corr_matrices))
    print(f"Environment: {ENV}")
    print(f"Static data: {static_path}")
    print(f"Dynamic data: {dynamic_path}")
    print(f"Matched trials: {len(shared_stimuli)}")
    print(f"Channels: {channel_numbers.tolist()}")
    print(
        "Correlation matrices: "
        f"{channel_corr_matrices.shape} "
        "(channels, dynamic time, static time)"
    )
    print(f"Undefined correlations: {undefined_fraction:.2%}")
    print(f"Saved results: {output_dir}")

    if cfg.show:
        for result_path in (matrix_path, slices_path, lag_path):
            image = plt.imread(result_path)
            plt.figure(figsize=(10, 7))
            plt.imshow(image)
            plt.axis("off")
        # end for result_path
        plt.show()
    # end if cfg.show
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
