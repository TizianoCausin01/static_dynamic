import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import matplotlib

# File output is the default; avoid macOS GUI-backend failures in headless runs.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
# end if --show not requested
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
    compute_rdm_timeseries, load_natraster, match_timed_static_movie_rasters,
    plot_static_movie_trajectories_2d,
)
from project_specific_utils.last_frame_decay import resample_array
from project_specific_utils.rdm_trajectory_pca import (
    plot_gradient_trajectory_3d, project_rdm_trajectories,
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
    output_dir: str = str(PROJECT_ROOT / "results" / "rdm_trajectory_pca")

    # Timing and PCA parameters of the presentation notebook (Fig 4).
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float = 1000
    last_frame_onset_ms: float = 2500
    # Movie samples drawn; starting at the last-frame onset aligns them with static.
    movie_window_ms: tuple[float, float] = (2500, 3500)
    # Movie samples the movie basis is fitted on (well past the onset transient).
    movie_fit_window_ms: tuple[float, float] = (2000, 3500)
    smoothing_ms: float = 20
    cmap: str = "viridis"
    figure_dpi: int = 160
    show: bool = False
# EOF


"""
parse_args
Parse monkeys, RDM metrics, and PCA timing parameters.

OUTPUT:
    - cfg: Cfg -> analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Project the static and movie neural RDM trajectories onto the "
            "first two PCs of either condition's RDM timeseries."
        )
    )
    parser.add_argument("--monkeys", nargs="+", default=list(Cfg.monkeys))
    parser.add_argument("--rdm_metrics", nargs="+", default=list(Cfg.rdm_metrics))
    parser.add_argument("--output_dir", default=Cfg.output_dir)
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--new_fs", type=float, default=Cfg.new_fs)
    parser.add_argument("--static_crop_ms", type=float, default=Cfg.static_crop_ms)
    parser.add_argument("--last_frame_onset_ms", type=float, default=Cfg.last_frame_onset_ms)
    parser.add_argument(
        "--movie_window_ms", nargs=2, type=float, default=list(Cfg.movie_window_ms),
        metavar=("START", "END"),
    )
    parser.add_argument(
        "--movie_fit_window_ms", nargs=2, type=float, default=list(Cfg.movie_fit_window_ms),
        metavar=("START", "END"),
    )
    parser.add_argument("--smoothing_ms", type=float, default=Cfg.smoothing_ms)
    parser.add_argument("--cmap", default=Cfg.cmap)
    parser.add_argument("--figure_dpi", type=int, default=Cfg.figure_dpi)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=Cfg.show)
    args = parser.parse_args()
    args.monkeys = tuple(args.monkeys)
    args.rdm_metrics = tuple(args.rdm_metrics)
    args.movie_window_ms = tuple(args.movie_window_ms)
    args.movie_fit_window_ms = tuple(args.movie_fit_window_ms)
    return Cfg(**vars(args))
# EOF


"""
load_monkey_rasters
Load the aligned last-frame static and movie rasters of one monkey on the
reliable channels, crop the static response, and resample both.

INPUT:
    - monkey: str -> key of MONKEY_SESSIONS
    - cfg: Cfg -> analysis configuration

OUTPUT:
    - static_rasters: np.ndarray -> channels x time x stimuli last-frame response
    - movie_rasters: np.ndarray -> channels x time x stimuli movie response
"""
def load_monkey_rasters(monkey, cfg):
    static_name, dynamic_name, good_channels, reliable_key = MONKEY_SESSIONS[monkey]
    data_dir = Path(paths["data_path"]) / "data"
    channel_kwargs = {
        "good_channels": good_channels,
        "reliable_channels_config": PROJECT_ROOT / "reliable_channels.yaml",
        "reliable_channels_key": reliable_key,
        "return_channel_numbers": True,
    }
    static_rasters, static_names, static_channels = load_natraster(
        data_dir / f"{static_name}_natraster_img.mat", **channel_kwargs,
    )
    movie_rasters, movie_names, movie_channels = load_natraster(
        data_dir / f"{dynamic_name}_natraster_vid.mat", **channel_kwargs,
    )
    if not np.array_equal(static_channels, movie_channels):
        raise ValueError(f"{monkey}: sessions retained different channels.")
    # end if channel numbers differ
    last_frame_rasters, movie_rasters, _, _, shared_stimuli, _ = (
        match_timed_static_movie_rasters(
            static_rasters, static_names, movie_rasters, movie_names,
        )
    )
    static_crop_samples = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
    print(
        f"{monkey}: {len(static_channels)} channels | {len(shared_stimuli)} stimuli",
        flush=True,
    )
    return (
        resample_array(last_frame_rasters[:, :static_crop_samples, :], cfg.source_fs, cfg.new_fs),
        resample_array(movie_rasters, cfg.source_fs, cfg.new_fs),
    )
# EOF


def main():
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoothing_bins = cfg.smoothing_ms * cfg.new_fs / 1000
    color_norm = Normalize(*cfg.movie_window_ms)
    colormap = plt.get_cmap(cfg.cmap)
    basis_names = ("static", "movie")

    # Rows are monkeys; columns are metric x basis; one 2D and one 3D figure.
    n_columns = len(cfg.rdm_metrics) * len(basis_names)
    figure, axes = plt.subplots(
        len(cfg.monkeys), n_columns,
        figsize=(4.6 * n_columns, 4 * len(cfg.monkeys)), squeeze=False,
    )
    figure_3d, axes_3d = plt.subplots(
        len(cfg.monkeys), n_columns,
        figsize=(4.6 * n_columns, 4.4 * len(cfg.monkeys)), squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    saved_scores = {}
    for row, monkey in enumerate(cfg.monkeys):
        static_rasters, movie_rasters = load_monkey_rasters(monkey, cfg)
        static_time_ms = np.arange(static_rasters.shape[1]) * 1000 / cfg.new_fs
        movie_time_ms = np.arange(movie_rasters.shape[1]) * 1000 / cfg.new_fs
        movie_window = (
            (movie_time_ms >= cfg.movie_window_ms[0])
            & (movie_time_ms < cfg.movie_window_ms[1])
        )
        movie_fit_mask = (
            (movie_time_ms >= cfg.movie_fit_window_ms[0])
            & (movie_time_ms < cfg.movie_fit_window_ms[1])
        )
        # Static samples coloured by the movie time at which the last frame appears.
        static_color_ms = static_time_ms + cfg.last_frame_onset_ms
        movie_color_ms = movie_time_ms[movie_window]
        landmark_index = np.argmin(np.abs(movie_color_ms - cfg.last_frame_onset_ms))

        for metric_index, rdm_metric in enumerate(cfg.rdm_metrics):
            # time x stimulus-pair distances for every condition.
            static_rdms = compute_rdm_timeseries(static_rasters, rdm_metric)
            movie_rdms = compute_rdm_timeseries(movie_rasters, rdm_metric)
            for basis_index, basis_name in enumerate(basis_names):
                # Three PCs: the first two feed the 2D plot, all three the 3D plot.
                static_scores, movie_scores, explained = project_rdm_trajectories(
                    static_rdms, movie_rdms, basis_name, movie_fit_mask=movie_fit_mask,
                    n_components=3, smoothing_bins=smoothing_bins,
                )
                # Whole movie projected and smoothed before cutting the window.
                movie_scores = movie_scores[movie_window]
                saved_scores[f"{monkey}_{rdm_metric}_{basis_name}_static"] = static_scores
                saved_scores[f"{monkey}_{rdm_metric}_{basis_name}_movie"] = movie_scores
                saved_scores[f"{monkey}_{rdm_metric}_{basis_name}_explained"] = explained

                column = metric_index * len(basis_names) + basis_index
                title = f"{monkey} | {rdm_metric} | {basis_name} basis"
                axis_labels = [f"PC{index + 1} ({ratio:.0%})" for index, ratio in enumerate(explained)]

                axis = axes[row, column]
                plot_static_movie_trajectories_2d(
                    axis, static_scores, movie_scores,
                    static_color_ms, movie_color_ms, colormap, color_norm,
                    linewidth=3, movie_dashed=True, show_legend=(row == 0 and column == 0),
                    static_label="static (last frame)", movie_label="movie",
                    legend_fontsize=7,
                )
                # Red dot: movie at the last-frame onset, where static begins in colour.
                axis.plot(*movie_scores[landmark_index, :2], "o", color="red", markersize=5, zorder=5)
                axis.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title=title)
                axis.title.set_fontsize(9)

                axis_3d = axes_3d[row, column]
                plot_gradient_trajectory_3d(axis_3d, static_scores, static_color_ms, colormap, color_norm, linewidth=3)
                plot_gradient_trajectory_3d(
                    axis_3d, movie_scores, movie_color_ms, colormap, color_norm,
                    linewidth=2, linestyle="dashed",
                )
                axis_3d.scatter(*movie_scores[landmark_index, :3], color="red", s=25, depthshade=False)
                # add_collection3d does not autoscale, so set limits from both trajectories.
                all_scores = np.vstack([static_scores, movie_scores])
                axis_3d.set(
                    xlim=(all_scores[:, 0].min(), all_scores[:, 0].max()),
                    ylim=(all_scores[:, 1].min(), all_scores[:, 1].max()),
                    zlim=(all_scores[:, 2].min(), all_scores[:, 2].max()),
                    xlabel=axis_labels[0], ylabel=axis_labels[1], zlabel=axis_labels[2],
                    title=title,
                )
                axis_3d.title.set_fontsize(9)
                axis_3d.tick_params(labelsize=6)
            # end for basis_name
        # end for rdm_metric
    # end for monkey

    suptitle = (
        "Neural RDM trajectories on the static or movie RDM PCs\n"
        f"solid = static, dashed = movie; movie basis fitted on "
        f"{cfg.movie_fit_window_ms[0]:.0f}-{cfg.movie_fit_window_ms[1]:.0f} ms; "
        f"red dot = movie at {cfg.last_frame_onset_ms:.0f} ms"
    )
    for current_figure, current_axes, file_name in (
            (figure, axes, "rdm_trajectory_pca.png"),
            (figure_3d, axes_3d, "rdm_trajectory_pca_3d.png"),
            ):
        # A standalone mappable serves both figures (an artist belongs to one figure).
        current_figure.colorbar(
            ScalarMappable(norm=color_norm, cmap=colormap), ax=current_axes,
            shrink=0.5, label="Movie time (ms)",
        )
        current_figure.suptitle(suptitle)
        current_figure.savefig(output_dir / file_name, dpi=cfg.figure_dpi, bbox_inches="tight")
    # end for current_figure
    np.savez(output_dir / "rdm_trajectory_pca.npz", **saved_scores)
    print(f"Saved to {output_dir}")
    if cfg.show:
        plt.show()
    # end if cfg.show
# EOF


if __name__ == "__main__":
    main()
# EOC
