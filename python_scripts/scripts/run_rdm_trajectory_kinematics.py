import argparse
import csv
from dataclasses import dataclass
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
    compute_rdm_timeseries, load_last_frame_movie_rasters, plot_static_dynamic_matrix,
)
from project_specific_utils.manifold_dynamics import bin_average_rasters
from project_specific_utils.rdm_trajectory_kinematics import (
    normalize_rdm_timeseries, rdm_step_lengths, rdm_velocity,
    velocity_cosine_matrix,
)


# Same sessions as the configuration cell of all_monkeys_presentation.ipynb:
# (image session, movie session, channel range, reliable-channel key).
MONKEY_SESSIONS = {
    "baby1": (
        "baby1_260718to0727", "baby1_260716to0928", (84, 186),
        "baby1_260718to0727",
    ),
    "paul": (
        "paul_20260901to1003", "paul_20260831to0927", (1, 64),
        "paul_20260901to0925",
    ),
    "red": (
        "red_20260726to0925", "red_20260720to0928", (1, 64),
        "red_20260726to0925",
    ),
}


@dataclass
class Cfg:
    monkeys: tuple[str, ...] = ("baby1", "paul", "red")
    rdm_metrics: tuple[str, ...] = ("euclidean", "cosine_cnt")
    # 'zscore' removes each RDM's mean and scale, so overall distance growth
    # (e.g. from firing-rate changes) does not count as movement.
    rdm_normalizations: tuple[str, ...] = ("none", "zscore")
    output_dir: str = str(PROJECT_ROOT / "results" / "rdm_trajectory_kinematics")

    source_fs: float = 1000
    bin_ms: float = 50
    static_crop_ms: float = 1000
    last_frame_onset_ms: float = 2500
    # Path-length windows, in ms from the onset of each condition.
    static_window_ms: tuple[float, float] = (0, 1000)
    movie_onset_window_ms: tuple[float, float] = (0, 1000)
    movie_last_frame_window_ms: tuple[float, float] = (2500, 3500)

    # Velocity used for the main cosine figure, and all methods compared.
    main_velocity_method: str = "central"
    velocity_methods: tuple[str, ...] = ("forward", "central", "savgol")
    savgol_window_bins: int = 5
    savgol_order: int = 2
    # Metric and normalization of the velocity-method comparison figure.
    method_comparison: tuple[str, str] = ("cosine_cnt", "none")
    # Movie time drawn in the cosine matrices: the last 1000 ms of the movie.
    movie_display_window_ms: tuple[float, float] = (2500, 3500)
    figure_dpi: int = 160
    show: bool = False
# EOF


"""
parse_args
Parse monkeys, RDM options, binning, windows, and velocity parameters.

OUTPUT:
    - cfg: Cfg -> analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Binned path length of the static and movie neural RDM trajectories "
            "and the cross-temporal cosine between their velocity vectors."
        )
    )
    parser.add_argument("--monkeys", nargs="+", default=list(Cfg.monkeys))
    parser.add_argument("--rdm_metrics", nargs="+", default=list(Cfg.rdm_metrics))
    parser.add_argument(
        "--rdm_normalizations", nargs="+", choices=("none", "zscore"),
        default=list(Cfg.rdm_normalizations),
    )
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--bin_ms", type=float, default=Cfg.bin_ms)
    parser.add_argument("--static_crop_ms", type=float, default=Cfg.static_crop_ms)
    parser.add_argument("--last_frame_onset_ms", type=float, default=Cfg.last_frame_onset_ms)
    for window_name in (
            "static_window_ms", "movie_onset_window_ms", "movie_last_frame_window_ms",
            "movie_display_window_ms",
            ):
        parser.add_argument(
            f"--{window_name}", nargs=2, type=float,
            default=list(getattr(Cfg, window_name)), metavar=("START", "END"),
        )
    # end for window_name
    parser.add_argument(
        "--main_velocity_method", choices=("forward", "central", "savgol"),
        default=Cfg.main_velocity_method,
    )
    parser.add_argument("--velocity_methods", nargs="+", default=list(Cfg.velocity_methods))
    parser.add_argument("--savgol_window_bins", type=int, default=Cfg.savgol_window_bins)
    parser.add_argument("--savgol_order", type=int, default=Cfg.savgol_order)
    parser.add_argument(
        "--method_comparison", nargs=2, default=list(Cfg.method_comparison),
        metavar=("METRIC", "NORMALIZATION"),
    )
    parser.add_argument("--figure_dpi", type=int, default=Cfg.figure_dpi)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=Cfg.show)
    args = parser.parse_args()
    for name in (
            "monkeys", "rdm_metrics", "rdm_normalizations", "static_window_ms",
            "movie_onset_window_ms", "movie_last_frame_window_ms",
            "movie_display_window_ms", "velocity_methods",
            "method_comparison",
            ):
        setattr(args, name, tuple(getattr(args, name)))
    # end for name
    return Cfg(**vars(args))
# EOF


"""
window_path_length
Sum the steps whose two bins both fall inside a time window.

INPUT:
    - step_lengths: np.ndarray -> (bins - 1) step lengths; step k joins bins k and k + 1
    - bin_centres_ms: np.ndarray -> centre time of every bin
    - window_ms: tuple[float, float] -> inclusive-start, exclusive-stop window

OUTPUT:
    - path_length: float -> path length inside the window
"""
def window_path_length(step_lengths, bin_centres_ms, window_ms):
    inside = (bin_centres_ms >= window_ms[0]) & (bin_centres_ms < window_ms[1])
    return float(step_lengths[inside[:-1] & inside[1:]].sum())
# EOF


def main():
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(paths["data_path"]) / "data"
    bin_fs = 1000 / cfg.bin_ms
    conditions = [
        (rdm_metric, normalization)
        for rdm_metric in cfg.rdm_metrics for normalization in cfg.rdm_normalizations
    ]

    results = {}
    summary_rows = []
    for monkey in cfg.monkeys:
        static_name, movie_name, good_channels, reliable_key = MONKEY_SESSIONS[monkey]
        static_rasters, movie_rasters, shared_stimuli = load_last_frame_movie_rasters(
            data_dir / f"{static_name}_natraster_img.mat",
            data_dir / f"{movie_name}_natraster_vid.mat",
            good_channels=good_channels,
            reliable_channels_config=PROJECT_ROOT / "reliable_channels.yaml",
            reliable_channels_key=reliable_key,
            static_crop_ms=cfg.static_crop_ms, source_fs=cfg.source_fs,
        )
        # channels x bins x stimuli, averaged in non-overlapping bins.
        static_binned = bin_average_rasters(static_rasters, cfg.source_fs, bin_fs)
        movie_binned = bin_average_rasters(movie_rasters, cfg.source_fs, bin_fs)
        static_centres_ms = (np.arange(static_binned.shape[1]) + 0.5) * cfg.bin_ms
        movie_centres_ms = (np.arange(movie_binned.shape[1]) + 0.5) * cfg.bin_ms
        print(
            f"{monkey}: {static_binned.shape[0]} channels | {len(shared_stimuli)} stimuli | "
            f"{static_binned.shape[1]} static and {movie_binned.shape[1]} movie bins",
            flush=True,
        )

        for rdm_metric in cfg.rdm_metrics:
            # bins x stimulus-pair distances.
            raw_rdms = {
                "static": compute_rdm_timeseries(static_binned, rdm_metric),
                "movie": compute_rdm_timeseries(movie_binned, rdm_metric),
            }
            for normalization in cfg.rdm_normalizations:
                rdms = {
                    condition: normalize_rdm_timeseries(condition_rdms, normalization)
                    for condition, condition_rdms in raw_rdms.items()
                }
                steps = {condition: rdm_step_lengths(condition_rdms) for condition, condition_rdms in rdms.items()}
                # Velocities of both conditions with every method; movie x static cosines.
                cosines = {}
                for method in cfg.velocity_methods:
                    velocities = {
                        condition: rdm_velocity(
                            condition_rdms, method, cfg.savgol_window_bins, cfg.savgol_order,
                        )
                        for condition, condition_rdms in rdms.items()
                    }
                    cosines[method] = velocity_cosine_matrix(velocities["movie"], velocities["static"])
                # end for method
                results[(monkey, rdm_metric, normalization)] = {
                    "steps": steps, "cosines": cosines,
                    "static_centres_ms": static_centres_ms, "movie_centres_ms": movie_centres_ms,
                }

                # Mean cosine along the aligned diagonal: static t vs movie last-frame onset + t.
                main_cosines = cosines[cfg.main_velocity_method]
                offset_bins = int(round(cfg.last_frame_onset_ms / cfg.bin_ms))
                n_aligned = min(main_cosines.shape[1], main_cosines.shape[0] - offset_bins)
                aligned_cosine = np.diag(main_cosines[offset_bins:offset_bins + n_aligned, :n_aligned])
                summary_rows.append({
                    "monkey": monkey, "rdm_metric": rdm_metric, "normalization": normalization,
                    "static_path_length": window_path_length(
                        steps["static"], static_centres_ms, cfg.static_window_ms),
                    "movie_onset_path_length": window_path_length(
                        steps["movie"], movie_centres_ms, cfg.movie_onset_window_ms),
                    "movie_last_frame_path_length": window_path_length(
                        steps["movie"], movie_centres_ms, cfg.movie_last_frame_window_ms),
                    f"aligned_velocity_cosine_{cfg.main_velocity_method}": float(aligned_cosine.mean()),
                })
            # end for normalization
        # end for rdm_metric
    # end for monkey

    with open(output_dir / "path_length_summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    # end with open
    for row in summary_rows:
        print(" | ".join(f"{value:.3g}" if isinstance(value, float) else str(value) for value in row.values()))
    # end for row

    # Figure 1: step length per bin; rows = metric x normalization, columns = monkeys.
    figure, axes = plt.subplots(
        len(conditions), len(cfg.monkeys),
        figsize=(4.6 * len(cfg.monkeys), 2.8 * len(conditions)), squeeze=False,
    )
    for row, (rdm_metric, normalization) in enumerate(conditions):
        for column, monkey in enumerate(cfg.monkeys):
            result = results[(monkey, rdm_metric, normalization)]
            axis = axes[row, column]
            # Step k joins bins k and k + 1, so it sits on the boundary between them.
            movie_step_ms = result["movie_centres_ms"][:-1] + cfg.bin_ms / 2
            static_step_ms = result["static_centres_ms"][:-1] + cfg.bin_ms / 2
            axis.plot(movie_step_ms, result["steps"]["movie"], color="black", label="movie")
            axis.plot(
                static_step_ms + cfg.last_frame_onset_ms, result["steps"]["static"],
                color="tab:orange", linewidth=2, label="static, at last-frame onset",
            )
            axis.plot(
                static_step_ms, result["steps"]["static"], color="tab:orange",
                linestyle="--", alpha=0.5, label="static, at movie onset",
            )
            axis.axvline(cfg.last_frame_onset_ms, color="red", linestyle=":", linewidth=1)
            axis.set(
                title=f"{monkey} | {rdm_metric} | {normalization}",
                xlabel="Movie time (ms)", ylabel=f"Step length per {cfg.bin_ms:.0f} ms",
            )
            axis.title.set_fontsize(9)
            if row == 0 and column == 0:
                axis.legend(fontsize=7)
            # end if first panel
        # end for monkey
    # end for rdm_metric, normalization
    figure.suptitle(f"Binned path length of the RDM trajectory in the full stimulus-pair space ({cfg.bin_ms:.0f} ms bins)")
    figure.tight_layout()
    figure.savefig(output_dir / "path_length.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    # Figures 2-3: movie x static velocity cosines, for every condition with the
    # main method, and for every method in the comparison condition.
    matrix_style = {
        "cmap": "RdBu_r", "colorbar_label": "cos(v_movie, v_static)",
        # The cropped movie axis starts at the last-frame onset, so no onset line.
        "last_frame_onset_ms": None, "ticks_size": 7, "label_size": 8,
    }
    for file_name, panels, suptitle in (
            (
                "velocity_cosine.png",
                [(rdm_metric, normalization, cfg.main_velocity_method) for rdm_metric, normalization in conditions],
                f"Cosine between movie and static RDM velocities ({cfg.main_velocity_method} difference)",
            ),
            (
                "velocity_cosine_methods.png",
                [(*cfg.method_comparison, method) for method in cfg.velocity_methods],
                "Velocity estimators compared | "
                f"{cfg.method_comparison[0]} | {cfg.method_comparison[1]} "
                f"(savgol: {cfg.savgol_window_bins} bins, order {cfg.savgol_order})",
            ),
            ):
        figure, axes = plt.subplots(
            len(cfg.monkeys), len(panels),
            figsize=(4.4 * len(panels), 3.4 * len(cfg.monkeys)), squeeze=False,
        )
        for row, monkey in enumerate(cfg.monkeys):
            for column, (rdm_metric, normalization, method) in enumerate(panels):
                cosines = results[(monkey, rdm_metric, normalization)]["cosines"][method]
                # Velocity k sits at bin centre k, except forward velocity k, which
                # sits on the boundary between bins k and k + 1 (half a bin later).
                time_offset_ms = cfg.bin_ms / 2 if method == "forward" else 0
                movie_times_ms = np.arange(cosines.shape[0]) * cfg.bin_ms + cfg.bin_ms / 2 + time_offset_ms
                static_times_ms = np.arange(cosines.shape[1]) * cfg.bin_ms + cfg.bin_ms / 2 + time_offset_ms
                shown = (
                    (movie_times_ms >= cfg.movie_display_window_ms[0])
                    & (movie_times_ms < cfg.movie_display_window_ms[1])
                )
                shown_cosines = cosines[shown]
                # Symmetric colour limits per panel, robust to single extreme bins.
                limit = np.percentile(np.abs(shown_cosines), 99)
                axis = axes[row, column]
                image = plot_static_dynamic_matrix(
                    axis, shown_cosines, cfg.bin_ms, vmin=-limit, vmax=limit, **matrix_style,
                )
                # Each cell spans its velocity time +- half a bin on both axes.
                extent = (
                    movie_times_ms[shown][0] - cfg.bin_ms / 2, movie_times_ms[shown][-1] + cfg.bin_ms / 2,
                    static_times_ms[0] - cfg.bin_ms / 2, static_times_ms[-1] + cfg.bin_ms / 2,
                )
                image.set_extent(extent)
                axis.set(xlim=extent[:2], ylim=extent[2:])
                axis.set_title(f"{monkey} | {rdm_metric} | {normalization} | {method}", fontsize=9)
            # end for panel
        # end for monkey
        figure.suptitle(suptitle)
        figure.tight_layout()
        figure.savefig(output_dir / file_name, dpi=cfg.figure_dpi, bbox_inches="tight")
    # end for figure spec

    np.savez(
        output_dir / "rdm_trajectory_kinematics.npz",
        **{
            f"{monkey}_{rdm_metric}_{normalization}_{kind}": array
            for (monkey, rdm_metric, normalization), result in results.items()
            for kind, array in (
                [("static_steps", result["steps"]["static"]), ("movie_steps", result["steps"]["movie"])]
                + [(f"cosine_{method}", matrix) for method, matrix in result["cosines"].items()]
            )
        },
    )
    print(f"Saved to {output_dir}")
    if cfg.show:
        plt.show()
    # end if cfg.show
# EOF


if __name__ == "__main__":
    main()
# EOC
