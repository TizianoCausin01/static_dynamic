import argparse
from dataclasses import dataclass, field
import csv
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d


# Fig 3 of all_monkeys_presentation.ipynb on the across-monkey average: the
# layer RSA curves saved by run_layer_temporal_scores.py are averaged over the
# monkeys first, then centroids, peaks and depth-latency scores are computed on
# the averaged curves. One figure per Gaussian sigma (samples at new_fs) in
# smoothing_sigmas; that sigma smooths the averaged curves for the centroids,
# the peaks and the plot alike. Example from the project root:
# .venv/bin/python python_scripts/scripts/plot_layer_temporal_scores_monkey_average.py \
#     --run_models ijepa_vith14_1k alexnet dino_v3_h \
#     --run_tag imagenet50k_srp_range2000-2250_window2000 --smoothing_sigmas 0 1 2 3
# Mean-pooled IJEPA:
#     ... --run_models ijepa_vith14_1k --model_pooling mean \
#     --run_tag mean_range2000-2250_window2000 --smoothing_sigmas 0 1 2 3
# Single-monkey figures of the same run (the "average" over one monkey):
#     ... --monkeys red


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

# load_run lives next to this script (plot_layer_temporal_scores.py).
sys.path.append(str(Path(__file__).resolve().parent))
from plot_layer_temporal_scores import load_run


MODEL_LABELS = {
    "ijepa_vith14_1k": "IJEPA", "alexnet": "AlexNet", "dino_v3_h": "DINOv3-H",
}
# Short pooling names in the titles.
POOLING_LABELS = {"PCimagenet1000": "PC", "mean": "mean"}


@dataclass
class Cfg:
    # Models in the file stem of the run, in file order, and the run tag.
    run_models: list[str] = field(
        default_factory=lambda: ["ijepa_vith14_1k", "alexnet", "dino_v3_h"]
    )
    run_tag: str = "imagenet50k_srp_range2000-2250_window2000"
    model_pooling: str = "PCimagenet1000"
    monkeys: list[str] = field(default_factory=lambda: ["baby1", "paul", "red"])
    # Gaussian sigmas (samples at new_fs) applied to the averaged curves for
    # centroids, peaks and plotting; None = run settings (centroid 0, peak 3)
    # with the notebook's plot smoothing of 3.
    smoothing_sigmas: list[float] | None = None
    static_xlim_ms: tuple[float, float] = (0, 1000)
    cmap_range: tuple[float, float] = (0.1, 0.9)
    results_dir: str | None = None
    dpi: int = 150
# EOF


"""
parse_args
Parse the run to average and the monkeys to include.

OUTPUT:
    - cfg: Cfg -> configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(description="Fig 3 on the across-monkey average curves.")
    parser.add_argument("--run_models", nargs="+")
    parser.add_argument("--run_tag")
    parser.add_argument("--model_pooling")
    parser.add_argument("--monkeys", nargs="+")
    parser.add_argument("--smoothing_sigmas", nargs="+", type=float)
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
condition_timing
Onset and latency window of every condition, read from the run config.

INPUT:
    - run_cfg: dict -> configuration of run_layer_temporal_scores.py

OUTPUT:
    - timing: dict[str, tuple] -> condition -> (onset_ms, window_ms, label)
"""
def condition_timing(run_cfg):
    last_window_ms = tuple(run_cfg["last_frame_latency_window_ms"])
    residual_window_ms = tuple(run_cfg["residual_latency_window_ms"] or last_window_ms)
    onset_ms = run_cfg["last_frame_onset_ms"]
    # Regressed-out range relative to the last-frame onset, e.g. "-500 to -250".
    # Regressed-out frames relative to the last-frame onset: a range ("-500 to
    # -250") or the listed frames ("-500 & -250"), plus the method if not RDM OLS.
    # Older runs have no previous_frame_range_ms key.
    if run_cfg.get("previous_frame_range_ms") is not None:
        frames_label = " to ".join(f"{frame_ms - onset_ms:g}" for frame_ms in run_cfg["previous_frame_range_ms"])
    else:
        frames_label = " & ".join(f"{frame_ms - onset_ms:g}" for frame_ms in run_cfg["previous_frames_ms"])
    # end if frame range
    method_label = ""
    if run_cfg["regression_level"] == "features_pca":
        method_label = f", feature PCA {run_cfg['pca_n_components']}"
    # end if feature-space PCA
    return {
        "static": (0, tuple(run_cfg["static_latency_window_ms"]), "static"),
        "first_frame": (
            run_cfg["first_frame_ms"], tuple(run_cfg["first_frame_latency_window_ms"]), "first frame",
        ),
        "last_frame": (onset_ms, last_window_ms, "final frame"),
        "last_frame_residual": (onset_ms, residual_window_ms, f"final frame, {frames_label} ms regr out{method_label}"),
    }
# EOF


"""
monkeys_tag
File-name token of the monkeys averaged: "monkey_average" for several,
the monkey name for one.

INPUT:
    - cfg: Cfg -> configuration

OUTPUT:
    - tag: str -> file-name token
"""
def monkeys_tag(cfg):
    return cfg.monkeys[0] if len(cfg.monkeys) == 1 else "monkey_average"
# EOF


"""
plot_model_average
One figure per model: averaged layer timecourses (top), centroid scatters
(middle) and peak scatters (bottom) for the four Fig 3 conditions.

INPUT:
    - model_name: str -> registry model name
    - curves: dict -> "monkey|model|pooling|condition" layers x time curves
    - run_cfg: dict -> configuration of the run
    - cfg: Cfg -> configuration
    - sigma: float | None -> one Gaussian sigma for centroids, peaks and plot;
        None = run settings and a plot sigma of 3

OUTPUT:
    - figure_path: Path -> saved figure
    - rows: list[dict] -> depth-latency scores of the averaged curves
"""
def plot_model_average(model_name, curves, run_cfg, cfg, sigma=None):
    centroid_sigma = run_cfg["centroid_smoothing"] if sigma is None else sigma
    peak_sigma = run_cfg["peak_smoothing"] if sigma is None else sigma
    plot_sigma = 3 if sigma is None else sigma
    model_label = f"{MODEL_LABELS.get(model_name, model_name)} {POOLING_LABELS.get(cfg.model_pooling, cfg.model_pooling)}"
    layer_cmap = truncate_colormap(plt.get_cmap("plasma"), *cfg.cmap_range)
    timing = condition_timing(run_cfg)
    figure, axes = plt.subplots(
        3, 4, figsize=(22, 13), gridspec_kw={"height_ratios": [1.2, 1, 1]},
    )
    rows = []
    for column, (condition_name, (onset_ms, window_ms, label)) in enumerate(timing.items()):
        # monkeys x layers x time -> mean over monkeys (same stimuli and layers).
        monkey_curves = np.stack([
            curves[f"{monkey}|{model_name}|{cfg.model_pooling}|{condition_name}"]
            for monkey in cfg.monkeys
        ])
        mean_similarity = monkey_curves.mean(axis=0)
        layer_depths = np.linspace(0, 1, mean_similarity.shape[0])
        layer_colors = layer_cmap(layer_depths)
        time_ms = np.arange(mean_similarity.shape[1]) * 1000 / run_cfg["new_fs"]
        # Centroids and peaks of the averaged curves, with the run's settings.
        _, centroid_ms, peak_ms, _ = summarize_layer_timing(
            mean_similarity, time_ms, onset_ms, window_ms, window_ms,
            centroid_sigma, run_cfg["centroid_similarity_cutoff"],
            run_cfg["peak_similarity_cutoff"], peak_smoothing_sigma=peak_sigma,
            relative_cutoffs=run_cfg["relative_similarity_cutoffs"],
        )

        # Smoothed curves for display only.
        plotted = mean_similarity
        if plot_sigma > 0:
            plotted = gaussian_filter1d(mean_similarity, sigma=plot_sigma, axis=1)
        # end if plot smoothing
        timecourse_axis = axes[0, column]
        plot_layer_curves_with_centroids(
            timecourse_axis, time_ms, plotted, centroid_ms + onset_ms,
            layer_colors, onset_ms, window_ms,
            show_negative=run_cfg.get("similarity_method", "rsa") == "encoding",
        )
        if condition_name == "static":
            timecourse_axis.set_xlim(cfg.static_xlim_ms)
        # end if static condition
        timecourse_axis.set_xlabel("Time from image onset (ms)" if condition_name == "static"
                                   else "Time from video onset (ms)")
        # Encoding runs store channel-averaged cross-validated R2 (older
        # runs have no similarity_method key and are RSA).
        if run_cfg.get("similarity_method", "rsa") == "encoding":
            timecourse_axis.set_ylabel(f"encoding {run_cfg['encoding_score_type']}")
        else:
            timecourse_axis.set_ylabel(r"dRSA corr ($\rho$)")
        # end if encoding run
        monkeys_label = cfg.monkeys[0] if len(cfg.monkeys) == 1 else f"mean of {len(cfg.monkeys)} monkeys"
        timecourse_axis.set_title(f"{monkeys_label} | {model_label} | sigma={plot_sigma:g}\n{label}")

        for row_index, (timing_name, latency_ms) in enumerate(
                (("centroid", centroid_ms), ("peak", peak_ms)), start=1):
            score = depth_latency_score(layer_depths, latency_ms)
            scatter_axis = axes[row_index, column]
            plot_depth_latency_points(
                scatter_axis, latency_ms, layer_depths, layer_colors, score["rho"], score["pvalue"],
            )
            axis_label = "Centroid time" if timing_name == "centroid" else "Peak latency"
            scatter_axis.set_xlabel(f"{axis_label} from onset (ms)")
            scatter_axis.set_ylabel("Layer depth")
            scatter_axis.set_title(f"{timing_name} | {label}")
            rows.append({
                "model_name": model_name, "model_pooling": cfg.model_pooling,
                "monkeys": "+".join(cfg.monkeys), "sigma": "run" if sigma is None else sigma, "condition": condition_name,
                "timing": timing_name, **score,
            })
        # end for row_index
    # end for column
    figure.tight_layout()
    figure_dir = Path(cfg.results_dir) / "figures"
    figure_dir.mkdir(exist_ok=True)
    sigma_tag = "" if sigma is None else f"_sigma{sigma:g}"
    figure_path = figure_dir / f"{model_name}_{cfg.model_pooling}_{cfg.run_tag}_{monkeys_tag(cfg)}{sigma_tag}.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path, rows
# EOF


"""
main
Average the run's curves over monkeys, save one figure per model and a CSV
of the depth-latency scores of the averaged curves.
"""
def main() -> None:
    cfg = parse_args()
    results_dir = Path(cfg.results_dir)
    curves, run_cfg = load_run(results_dir, "_".join(cfg.run_models) + f"_{cfg.run_tag}")
    all_rows = []
    for sigma in cfg.smoothing_sigmas or [None]:
        for model_name in cfg.run_models:
            figure_path, rows = plot_model_average(model_name, curves, run_cfg, cfg, sigma)
            for row in rows:
                print(
                    f"  sigma={row['sigma']} {model_name:<16} {row['condition']:<20} "
                    f"{row['timing']:<8} rho={row['rho']:+.2f} p={row['pvalue']:.2g} "
                    f"n={row['n_layers']}",
                    flush=True,
                )
            # end for row
            all_rows.extend(rows)
            print(f"Saved {figure_path}")
        # end for model_name
    # end for sigma
    sigma_tag = "" if cfg.smoothing_sigmas is None else "_sigmas" + "-".join(f"{sigma:g}" for sigma in cfg.smoothing_sigmas)
    csv_path = results_dir / f"{'_'.join(cfg.run_models)}_{cfg.run_tag}_{monkeys_tag(cfg)}{sigma_tag}_temporal_scores.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # end with open
    print(f"Saved {csv_path}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
