import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


# Figures of the model autocorrelation control saved by run_layer_temporal_scores.py
# --similarity_target model: per model, the last-frame (raw and residual) slice
# RDM against the layer's own model RDM timecourse, the centroid ticks, the
# depth-centroid temporal-score scatters, and the centroids of the neural run
# with the same regression next to the model ones. Example from the project root:
# .venv/bin/python python_scripts/scripts/plot_layer_temporal_scores.py \
#     --model_names ijepa_vith14_1k alexnet dino_v3_h convnext_v2_huge \
#     --self_tag modelself_imagenet50k_srp_range2000-2250_window2000 \
#     --neural_runs "ijepa_vith14_1k alexnet dino_v3_h:imagenet50k_srp_range2000-2250_window2000"


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils import (
    depth_latency_score, plot_depth_latency_points,
    plot_layer_curves_with_centroids, summarize_layer_timing,
)
from useful_stuff.general_utils import truncate_colormap


# Same monkey colours and labels as all_monkeys_presentation.ipynb.
MONKEY_COLORS = {"baby1": "#3474BD", "red": "#C4564C", "paul": "#11A579"}
MONKEY_LABELS = {"baby1": "B1-AIT-NPx", "red": "R-AIT", "paul": "P-CIT"}
CONDITION_LABELS = {
    "last_frame": "final frame",
    "last_frame_residual": "final frame, 2000-2250 ms regr out",
}


@dataclass
class Cfg:
    model_names: list[str] = field(default_factory=lambda: ["alexnet"])
    model_pooling: str = "PCimagenet1000"
    # File tags (run_layer_temporal_scores.py --output_tag) of the model-target
    # run and, per model group, of the matching neural-target run.
    self_tag: str = "modelself_imagenet50k_srp_range2000-2250_window2000"
    # "model names:tag" strings; one neural npz holds every model of its run.
    neural_runs: list[str] = field(default_factory=list)
    # Monkey whose stimulus set is used for the model curves (all share them).
    self_monkey: str = "baby1"
    display_range_ms: tuple[float, float] = (1500, 3000)
    cmap_range: tuple[float, float] = (0.1, 0.9)
    results_dir: str | None = None
    dpi: int = 150
# EOF


"""
parse_args
Parse the models and the run tags to plot.

OUTPUT:
    - cfg: Cfg -> configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(description="Plot the model autocorrelation control.")
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument("--model_pooling")
    parser.add_argument("--self_tag")
    parser.add_argument("--neural_runs", nargs="+")
    parser.add_argument("--self_monkey")
    args = parser.parse_args()
    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    cfg.results_dir = str(PROJECT_ROOT / "results" / "layer_temporal_scores")
    return cfg
# EOF


"""
load_run
Curves and config of one run_layer_temporal_scores.py output.

INPUT:
    - results_dir: Path -> directory of the saved runs
    - file_stem: str -> "<models>_<tag>" prefix of the saved files

OUTPUT:
    - curves: dict[str, np.ndarray] -> "monkey|model|pooling|condition" curves
    - run_cfg: dict -> the run configuration
"""
def load_run(results_dir, file_stem):
    with np.load(results_dir / f"{file_stem}_curves.npz") as archive:
        curves = {key: archive[key] for key in archive.files}
    # end with np.load
    with open(results_dir / f"{file_stem}_config.json") as config_file:
        run_cfg = json.load(config_file)["config"]
    # end with open
    return curves, run_cfg
# EOF


"""
layer_centroids
Centroid times (relative to the last-frame onset) of every layer, with the
cutoff, smoothing and window of the run that produced the curves.

INPUT:
    - similarity: np.ndarray -> layers x time curves
    - run_cfg: dict -> configuration of that run

OUTPUT:
    - time_ms: np.ndarray -> (time,) coordinate of the curves
    - centroid_ms: np.ndarray -> (layers,) centroid from the onset, NaN = none
"""
def layer_centroids(similarity, run_cfg):
    time_ms = np.arange(similarity.shape[1]) * 1000 / run_cfg["new_fs"]
    window_ms = tuple(run_cfg["last_frame_latency_window_ms"])
    _, centroid_ms, _, _ = summarize_layer_timing(
        similarity, time_ms, run_cfg["last_frame_onset_ms"], window_ms, window_ms,
        run_cfg["centroid_smoothing"], run_cfg["centroid_similarity_cutoff"],
        run_cfg["peak_similarity_cutoff"], peak_smoothing_sigma=run_cfg["peak_smoothing"],
        relative_cutoffs=run_cfg["relative_similarity_cutoffs"],
    )
    return time_ms, centroid_ms
# EOF


"""
plot_model
One figure per model. Top row: model-self timecourses (raw and residual last
frame). Bottom row: their depth-centroid scatters. Right column: model-self
centroids next to the neural centroids of every monkey for the same condition.

INPUT:
    - model_name: str -> registry model name
    - self_curves: dict -> curves of the model-target run
    - self_cfg: dict -> config of the model-target run
    - neural_curves: dict | None -> curves of the neural run, None = no overlay
    - neural_cfg: dict | None -> config of the neural run
    - cfg: Cfg -> configuration

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_model(model_name, self_curves, self_cfg, neural_curves, neural_cfg, cfg):
    layer_cmap = truncate_colormap(plt.get_cmap("plasma"), *cfg.cmap_range)
    onset_ms = self_cfg["last_frame_onset_ms"]
    window_ms = tuple(self_cfg["last_frame_latency_window_ms"])
    figure, axes = plt.subplots(
        2, 3, figsize=(18, 9), gridspec_kw={"width_ratios": [1.3, 1.3, 1]},
    )
    for column, condition_name in enumerate(CONDITION_LABELS):
        # layers x model time curves; identical for every monkey (same stimuli).
        similarity = self_curves[
            f"{cfg.self_monkey}|{model_name}|{cfg.model_pooling}|{condition_name}"
        ]
        layer_depths = np.linspace(0, 1, similarity.shape[0])
        layer_colors = layer_cmap(layer_depths)
        time_ms, centroid_ms = layer_centroids(similarity, self_cfg)

        timecourse_axis = axes[0, column]
        plot_layer_curves_with_centroids(
            timecourse_axis, time_ms, similarity, centroid_ms + onset_ms,
            layer_colors, onset_ms, window_ms,
        )
        timecourse_axis.set_xlim(cfg.display_range_ms)
        timecourse_axis.set_xlabel("Model time from video onset (ms)")
        timecourse_axis.set_ylabel(r"model-model RSA ($\rho$)")
        timecourse_axis.set_title(f"{model_name} | model self | {CONDITION_LABELS[condition_name]}")

        score = depth_latency_score(layer_depths, centroid_ms)
        scatter_axis = axes[1, column]
        plot_depth_latency_points(
            scatter_axis, centroid_ms, layer_depths, layer_colors, score["rho"], score["pvalue"],
        )
        scatter_axis.set_xlabel("Centroid time from last-frame onset (ms)")
        scatter_axis.set_ylabel("Layer depth")
        scatter_axis.set_title(f"model self | {CONDITION_LABELS[condition_name]}")

        # Right column: model-self vs neural centroid progression across depth.
        comparison_axis = axes[column, 2]
        comparison_axis.plot(
            centroid_ms, layer_depths, "-o", color="black", markersize=4,
            label=f"model self (rho={score['rho']:.2f})",
        )
        if neural_curves is not None:
            for monkey, monkey_color in MONKEY_COLORS.items():
                curve_key = f"{monkey}|{model_name}|{cfg.model_pooling}|{condition_name}"
                if curve_key not in neural_curves:
                    continue
                # end if monkey not in the neural run
                _, neural_centroid_ms = layer_centroids(neural_curves[curve_key], neural_cfg)
                neural_score = depth_latency_score(layer_depths, neural_centroid_ms)
                comparison_axis.plot(
                    neural_centroid_ms, layer_depths, "-o", color=monkey_color, markersize=4,
                    label=f"{MONKEY_LABELS[monkey]} (rho={neural_score['rho']:.2f})",
                )
            # end for monkey
        # end if neural run available
        comparison_axis.axvline(0, color="0.6", linewidth=1, linestyle="--")
        comparison_axis.set_xlim(window_ms[0] - onset_ms - 20, window_ms[1] - onset_ms + 20)
        comparison_axis.set_ylim(-0.05, 1.05)
        comparison_axis.set_xlabel("Centroid time from last-frame onset (ms)")
        comparison_axis.set_ylabel("Layer depth")
        comparison_axis.set_title(f"centroids | {CONDITION_LABELS[condition_name]}")
        comparison_axis.legend(fontsize=8, loc="lower left")
    # end for column, condition_name
    figure.tight_layout()
    figure_dir = Path(cfg.results_dir) / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure_path = figure_dir / f"{model_name}_{cfg.model_pooling}_{cfg.self_tag}.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
main
Load the model-self run and the neural runs, then save one figure per model.
"""
def main() -> None:
    cfg = parse_args()
    results_dir = Path(cfg.results_dir)
    self_curves, self_cfg = load_run(
        results_dir, "_".join(self_cfg_models(results_dir, cfg)) + f"_{cfg.self_tag}",
    )
    # model name -> (curves, config) of the neural run that contains it.
    neural_by_model = {}
    for neural_run in cfg.neural_runs:
        run_models, run_tag = neural_run.split(":")
        run_curves, run_cfg = load_run(results_dir, "_".join(run_models.split()) + f"_{run_tag}")
        for model_name in run_models.split():
            neural_by_model[model_name] = (run_curves, run_cfg)
        # end for model_name
    # end for neural_run
    for model_name in cfg.model_names:
        neural_curves, neural_cfg = neural_by_model.get(model_name, (None, None))
        figure_path = plot_model(
            model_name, self_curves, self_cfg, neural_curves, neural_cfg, cfg,
        )
        print(f"Saved {figure_path}")
    # end for model_name
    return None
# EOF


"""
self_cfg_models
Model names in the file stem of the model-self run: the one config whose tag
matches self_tag.

INPUT:
    - results_dir: Path -> directory of the saved runs
    - cfg: Cfg -> configuration

OUTPUT:
    - model_names: list[str] -> models of that run, in file-stem order
"""
def self_cfg_models(results_dir, cfg):
    for config_path in results_dir.glob(f"*_{cfg.self_tag}_config.json"):
        with open(config_path) as config_file:
            run_cfg = json.load(config_file)["config"]
        # end with open
        if run_cfg["output_tag"] == cfg.self_tag:
            return run_cfg["model_names"]
        # end if matching tag
    # end for config_path
    raise FileNotFoundError(f"No run tagged {cfg.self_tag!r} in {results_dir}.")
# EOF


if __name__ == "__main__":
    main()
# EOF
