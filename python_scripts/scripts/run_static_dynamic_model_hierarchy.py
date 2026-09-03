import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import sys

import matplotlib
import numpy as np
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Example from the project root:
# .venv/bin/python python_scripts/scripts/run_static_dynamic_model_hierarchy.py


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.model_zoo import MODEL_ZOO, get_model_spec
from project_specific_utils import (
    load_model_layer_rsa,
    rdm_noise_ceiling,
    summarize_model_analysis,
)
from useful_stuff.general_utils import truncate_colormap


@dataclass
class Cfg:
    monkey_name: str = "baby1"
    static_experiment_name: str = "baby1_260718to27"
    dynamic_experiment_name: str = "baby1_260716to24"
    good_channels: tuple[int, int] | None = (84, 186)
    reliable_channels_key: str | None = "baby1_260718to27"

    model_names: list[str] = field(default_factory=lambda: list(MODEL_ZOO))
    results_dir: str | None = None
    # Split-half NPZ giving the RDM reliability the models are read against.
    noise_ceiling_path: str | None = None
    output_dir: str | None = None
    figs_dir: str | None = None

    # Must match the saved RSA family produced by the RSA sweep.
    signal_rdm_metric: str = "cosine_cnt"
    model_rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "spearman"
    new_fs: float = 100
    static_crop_ms: float | None = 1000
    model_pooling: str | None = "mean"
    model_dataset_name: str = "static_dynamic"

    # Analysis windows, following the presentation notebook.
    first_frame_onset_ms: float = 0
    last_frame_onset_ms: float = 2500
    image_window_ms: tuple[float, float] | None = None
    first_frame_window_ms: tuple[float, float] = (0, 500)
    last_frame_window_ms: tuple[float, float] = (2000, 3000)
    smoothing_sigma: float = 3
    absolute_cutoff: float = 0.02
    relative_cutoff: float = 0.5
    # Layers peaking below this RSA, or below this fraction of the model's own
    # best layer in the same analysis, carry no usable latency information.
    min_peak_similarity: float = 0.02
    min_peak_fraction: float = 0.15
    # (absolute floor, fraction) pairs re-run to show cutoff robustness.
    sensitivity_cutoffs: tuple[tuple[float, float], ...] = (
        (0.0, 0.0), (0.02, 0.15), (0.03, 0.0), (0.02, 0.33),
    )
    primary_latency: str = "centroid_latency_ms"

    grid_shape: tuple[int, int] = (3, 3)
    layer_cmap: str = "plasma"
    layer_cmap_min: float = 0.1
    layer_cmap_max: float = 0.9
    dpi: int = 150
# EOF


ANALYSES = ("image", "first_frame", "last_frame")

ANALYSIS_LABELS = {
    "image": "Static image response",
    "first_frame": "Movie first-frame slice",
    "last_frame": "Movie last-frame slice",
}

LATENCY_LABELS = {
    "centroid_latency_ms": "centroid latency",
    "relative_centroid_latency_ms": "half-height centroid latency",
    "peak_latency_ms": "peak latency",
}

# One colour per architecture family used by every summary figure.
FAMILY_COLORS = {
    "image CNN": "#E1655B",
    "image ViT": "#3474BD",
    "image ViT (SSL)": "#7596C5",
    "image ViT (VLM)": "#1F5C99",
    "video transformer": "#7F3C8D",
    "video world model": "#B95FB3",
    "video CNN": "#11A579",
    "low-level baseline": "#8A8A8A",
}

# Entries with a single layer describe a reference level, not a hierarchy.
BASELINE_FAMILY = "low-level baseline"


"""
parse_args
Parse the model selection, saved-RSA family, and output locations.

OUTPUT:
    - cfg: Cfg -> validated analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize layer depth versus response latency across models for "
            "the static image, first-frame, and last-frame analyses."
        )
    )
    parser.add_argument("--monkey_name", default=Cfg.monkey_name)
    parser.add_argument(
        "--static_experiment_name", default=Cfg.static_experiment_name,
    )
    parser.add_argument(
        "--dynamic_experiment_name", default=Cfg.dynamic_experiment_name,
    )
    parser.add_argument(
        "--model_names", nargs="+",
        help="Registry keys; defaults to the complete model zoo.",
    )
    parser.add_argument(
        "--good_channels", nargs=2, type=int, metavar=("FIRST", "LAST"),
        help="Inclusive one-based MATLAB range used by the saved RSA family.",
    )
    parser.add_argument(
        "--reliable_channels_key",
        help="Reliability YAML key; defaults to the static session name.",
    )
    parser.add_argument("--results_dir")
    parser.add_argument(
        "--noise_ceiling_path",
        help="split_half_static_dynamic_rsa.npz for the same channels.",
    )
    parser.add_argument("--output_dir")
    parser.add_argument("--figs_dir")
    parser.add_argument("--smoothing_sigma", type=float, default=Cfg.smoothing_sigma)
    parser.add_argument("--absolute_cutoff", type=float, default=Cfg.absolute_cutoff)
    parser.add_argument("--relative_cutoff", type=float, default=Cfg.relative_cutoff)
    parser.add_argument(
        "--min_peak_similarity", type=float, default=Cfg.min_peak_similarity,
        help="Minimum peak RSA for a layer to enter the temporal score.",
    )
    parser.add_argument(
        "--min_peak_fraction", type=float, default=Cfg.min_peak_fraction,
        help="Minimum peak RSA relative to the model's best layer.",
    )
    parser.add_argument(
        "--primary_latency", choices=tuple(LATENCY_LABELS),
        default=Cfg.primary_latency,
    )
    args = parser.parse_args()

    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    if args.good_channels is not None:
        cfg.good_channels = tuple(args.good_channels)
    # end if an explicit channel range was given
    # Every session names its reliability entry after its image session.
    if args.reliable_channels_key is None:
        cfg.reliable_channels_key = cfg.static_experiment_name
    # end if the key was not given explicitly
    unknown_names = [name for name in cfg.model_names if name not in MODEL_ZOO]
    if unknown_names:
        parser.error(f"Unknown model names {unknown_names}.")
    # end if unknown_names
    return cfg
# EOF


"""
resolve_cfg_paths
Fill in the default result, output, and figure directories.

INPUT:
    - cfg: Cfg -> user configuration

OUTPUT:
    - cfg: Cfg -> configuration with explicit directories
"""
def resolve_cfg_paths(cfg: Cfg) -> Cfg:
    experiment_name = (
        f"{cfg.dynamic_experiment_name}_vs_{cfg.static_experiment_name}"
    )
    cfg.results_dir = str(Path(
        cfg.results_dir
        or PROJECT_ROOT / "results" / "static_dynamic_neural_model_rsa"
    ).expanduser())
    # Outputs are keyed by experiment so several monkeys can coexist.
    cfg.output_dir = str(Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "static_dynamic_model_hierarchy"
        / experiment_name
    ).expanduser())
    cfg.figs_dir = str(Path(
        cfg.figs_dir
        or Path(paths["data_path"]) / "results" / "figs_model_hierarchy"
        / experiment_name
    ).expanduser())
    if cfg.noise_ceiling_path is not None:
        cfg.noise_ceiling_path = str(
            Path(cfg.noise_ceiling_path).expanduser()
        )
    # end if a ceiling was requested
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.figs_dir).mkdir(parents=True, exist_ok=True)
    return cfg
# EOF


"""
expected_rsa_metadata
Build the metadata filter that isolates one saved RSA parameter family.

INPUT:
    - cfg: Cfg -> analysis configuration
    - spec: ModelSpec -> registry entry whose archives are selected

OUTPUT:
    - metadata: dict -> metadata entries every retained archive must match
"""
def expected_rsa_metadata(cfg: Cfg, spec) -> dict:
    return {
        "model_name": spec.model_name,
        "signal_rdm_metric": cfg.signal_rdm_metric,
        "model_rdm_metric": cfg.model_rdm_metric,
        "rsa_metric_computed": cfg.rsa_metric,
        "model_dataset_name": cfg.model_dataset_name,
        "model_pooling": spec.pooling,
        "new_fs": cfg.new_fs,
        "static_crop_ms": cfg.static_crop_ms,
        "dynamic_crop_ms": None,
        "model_frame_index": -1,
        "normalization": None,
        "good_channels": (
            None if cfg.good_channels is None else list(cfg.good_channels)
        ),
        "reliable_channels_key": cfg.reliable_channels_key,
    }
# EOF


"""
analysis_settings
Return the frame onset and neural window used by one analysis.

INPUT:
    - cfg: Cfg -> analysis configuration
    - analysis: str -> image, first_frame, or last_frame

OUTPUT:
    - frame_onset_ms: float -> model time of the compared frame
    - window_ms: tuple[float, float] | None -> neural analysis window
"""
def analysis_settings(cfg: Cfg, analysis: str):
    if analysis == "image":
        return 0.0, cfg.image_window_ms
    if analysis == "first_frame":
        return cfg.first_frame_onset_ms, cfg.first_frame_window_ms
    return cfg.last_frame_onset_ms, cfg.last_frame_window_ms
# EOF


"""
collect_summaries
Load every model and compute the three latency analyses for each of them.

INPUT:
    - cfg: Cfg -> resolved analysis configuration

OUTPUT:
    - summaries: dict -> {analysis: {model_name: summary}} for the hierarchies
    - baselines: dict -> {analysis: {model_name: summary}} for the reference levels
    - missing_models: list[str] -> models without a complete saved RSA family
"""
def collect_summaries(cfg: Cfg):
    experiment_name = (
        f"{cfg.dynamic_experiment_name}_vs_{cfg.static_experiment_name}"
    )
    latency_kwargs = {
        "absolute_cutoff": cfg.absolute_cutoff,
        "relative_cutoff": cfg.relative_cutoff,
        "smoothing_sigma": cfg.smoothing_sigma,
    }
    summaries = {analysis: {} for analysis in ANALYSES}
    baselines = {analysis: {} for analysis in ANALYSES}
    missing_models = []

    for model_name in cfg.model_names:
        spec = get_model_spec(model_name)
        try:
            results = load_model_layer_rsa(
                cfg.results_dir,
                experiment_name,
                model_name,
                spec.layers,
                expected_rsa_metadata(cfg, spec),
            )
        except FileNotFoundError as error:
            print(f"skipping {model_name}: {error}", flush=True)
            missing_models.append(model_name)
            continue
        # end try
        for analysis in ANALYSES:
            frame_onset_ms, window_ms = analysis_settings(cfg, analysis)
            summary = summarize_model_analysis(
                results, analysis,
                frame_onset_ms=frame_onset_ms,
                window_ms=window_ms,
                latency_kwargs=latency_kwargs,
                min_peak_similarity=cfg.min_peak_similarity,
                min_peak_fraction=cfg.min_peak_fraction,
            )
            summary["label"] = spec.label
            summary["family"] = spec.family
            target = (
                baselines if spec.family == BASELINE_FAMILY else summaries
            )
            target[analysis][model_name] = summary
        # end for analysis
        loaded = (
            baselines if spec.family == BASELINE_FAMILY else summaries
        )["image"][model_name]
        print(
            f"{model_name}: {len(spec.layers)} layers, "
            f"best image RSA {loaded['best_peak_similarity']:.3f}",
            flush=True,
        )
    # end for model_name
    return summaries, baselines, missing_models
# EOF


"""
grid_axes
Create a figure grid large enough for one panel per model.

INPUT:
    - n_panels: int -> number of models to display
    - cfg: Cfg -> grid shape and figure sizing
    - panel_size: tuple[float, float] -> width and height of one panel

OUTPUT:
    - figure: plt.Figure -> created figure
    - axes: list[plt.Axes] -> flattened axes, unused ones already hidden
"""
def grid_axes(n_panels: int, cfg: Cfg, panel_size=(4.2, 3.2)):
    n_columns = cfg.grid_shape[1]
    n_rows = int(np.ceil(n_panels / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows),
        squeeze=False,
    )
    flat_axes = list(axes.ravel())
    for axis in flat_axes[n_panels:]:
        axis.set_visible(False)
    # end for unused axis
    return figure, flat_axes[:n_panels]
# EOF


"""
plot_layer_curves
Plot every layer's RSA timecourse per model, coloured from shallow to deep.

INPUT:
    - summaries: dict -> {model_name: summary} for one analysis
    - analysis: str -> analysis label used in the title and filename
    - cfg: Cfg -> plotting configuration
    - baselines: dict | None -> single-layer reference levels drawn behind

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_layer_curves(
        summaries: dict, analysis: str, cfg: Cfg, baselines=None,
        ) -> Path:
    layer_cmap = truncate_colormap(
        plt.get_cmap(cfg.layer_cmap), cfg.layer_cmap_min, cfg.layer_cmap_max,
    )
    model_names = list(summaries)
    figure, axes = grid_axes(len(model_names), cfg)

    for axis, model_name in zip(axes, model_names):
        summary = summaries[model_name]
        colors = layer_cmap(summary["layer_depths"])
        for layer_index, timecourse in enumerate(
                summary["smoothed_timecourses"]
                ):
            axis.plot(
                summary["time_ms"], timecourse,
                color=colors[layer_index], linewidth=1.6,
            )
        # end for layer_index
        for baseline_name, baseline in (baselines or {}).items():
            axis.plot(
                baseline["time_ms"], baseline["smoothed_timecourses"][0],
                color="0.35", linewidth=1.2,
                linestyle="--" if "flow" in baseline_name else ":",
                label=baseline["label"], zorder=6,
            )
        # end for baseline_name
        if summary["window_ms"] is not None:
            axis.axvspan(
                *summary["window_ms"], color="0.85", alpha=0.35, zorder=0,
            )
        # end if the analysis uses a restricted window
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.axvline(
            summary["onset_ms"], color="grey", linewidth=0.8, linestyle="--",
        )
        axis.set_title(
            f"{summary['label']}  ({summary['family']})", fontsize=10,
        )
        axis.set_xlabel("time (ms)", fontsize=9)
        axis.set_ylabel(r"RSA ($\rho$)", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    # end for axis, model_name
    if baselines:
        axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    # end if baselines

    figure.suptitle(
        f"{ANALYSIS_LABELS[analysis]}: layer RSA timecourses "
        "(dark = shallow, bright = deep)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = Path(cfg.figs_dir) / f"layer_curves_{analysis}.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_latency_scatters
Plot per-model layer depth against latency with the Spearman temporal score.

INPUT:
    - summaries: dict -> {model_name: summary} for one analysis
    - analysis: str -> analysis label used in the title and filename
    - cfg: Cfg -> plotting configuration
    - latency_name: str -> latency measure plotted on the x axis

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_latency_scatters(
        summaries: dict, analysis: str, cfg: Cfg, latency_name: str,
        ) -> Path:
    layer_cmap = truncate_colormap(
        plt.get_cmap(cfg.layer_cmap), cfg.layer_cmap_min, cfg.layer_cmap_max,
    )
    model_names = list(summaries)
    figure, axes = grid_axes(len(model_names), cfg)

    for axis, model_name in zip(axes, model_names):
        summary = summaries[model_name]
        latencies_ms = summary["latencies"][latency_name]
        score = summary["temporal_scores"][latency_name]
        valid_layers = np.isfinite(latencies_ms)
        axis.scatter(
            latencies_ms[valid_layers], summary["layer_depths"][valid_layers],
            c=layer_cmap(summary["layer_depths"][valid_layers]),
            s=60, edgecolors="black", linewidths=0.6,
        )
        # A Spearman p of exactly zero is an underflow, not a real value.
        pvalue_text = (
            "<1e-16" if score["pvalue"] == 0 else f"{score['pvalue']:.1e}"
        )
        axis.text(
            0.04, 0.96,
            f"rho={score['rho']:.2f}\np={pvalue_text}\n"
            f"p_shift={score['shift_pvalue']:.3f}",
            transform=axis.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
        )
        # The dropped-layer count goes in the title so it cannot cover points.
        n_dropped = int((~summary["informative_layers"]).sum())
        dropped_text = f", {n_dropped} below cutoff" if n_dropped else ""
        axis.set_title(
            f"{summary['label']}  (peak "
            fr"$\rho$={summary['best_peak_similarity']:.2f}"
            f"{dropped_text})", fontsize=10,
        )
        axis.set_xlabel(f"{LATENCY_LABELS[latency_name]} (ms)", fontsize=9)
        axis.set_ylabel("layer depth", fontsize=9)
        axis.set_ylim(-0.05, 1.05)
        axis.tick_params(labelsize=8)
    # end for axis, model_name

    figure.suptitle(
        f"{ANALYSIS_LABELS[analysis]}: layer depth versus "
        f"{LATENCY_LABELS[latency_name]}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = (
        Path(cfg.figs_dir) / f"latency_scatter_{analysis}_{latency_name}.png"
    )
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_peak_profiles
Plot each model's peak RSA as a function of normalized layer depth.

INPUT:
    - summaries: dict -> {analysis: {model_name: summary}}
    - cfg: Cfg -> plotting configuration
    - baselines: dict | None -> single-layer reference levels
    - ceilings: dict | None -> per-analysis RDM reliability drawn as a line

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_peak_profiles(
        summaries: dict, cfg: Cfg, baselines=None, ceilings=None,
        ) -> Path:
    figure, axes = plt.subplots(
        1, len(ANALYSES), figsize=(5.4 * len(ANALYSES), 4.4), squeeze=False,
    )
    # Models of the same family share a colour and are told apart by marker.
    family_counts = {}
    markers = ("o", "s", "^", "D", "v", "P")
    model_markers = {}
    for model_name, summary in summaries["image"].items():
        family = summary["family"]
        model_markers[model_name] = markers[
            family_counts.get(family, 0) % len(markers)
        ]
        family_counts[family] = family_counts.get(family, 0) + 1
    # end for model_name

    for axis, analysis in zip(axes[0], ANALYSES):
        for model_name, summary in summaries[analysis].items():
            axis.plot(
                summary["layer_depths"], summary["peak_similarity"],
                marker=model_markers[model_name], markersize=4.5,
                linewidth=1.6,
                color=FAMILY_COLORS.get(summary["family"], "0.4"),
                label=summary["label"],
            )
        # end for model_name
        for baseline_name, baseline in (baselines or {}).get(
                analysis, {}
                ).items():
            axis.axhline(
                baseline["best_peak_similarity"], color="0.45",
                linestyle="--" if "flow" in baseline_name else ":",
                linewidth=1.2, label=baseline["label"],
            )
        # end for baseline_name
        ceiling = (ceilings or {}).get(analysis)
        if ceiling is not None and ceiling == ceiling:
            axis.axhline(
                ceiling, color="#11A579", linewidth=1.4, linestyle="-.",
                label="noise ceiling",
            )
        # end if a ceiling is available
        axis.set_title(ANALYSIS_LABELS[analysis], fontsize=11)
        axis.set_xlabel("normalized layer depth", fontsize=10)
        axis.set_ylabel(r"peak RSA in window ($\rho$)", fontsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    # end for axis, analysis

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center",
        ncol=min(6, len(labels)), frameon=False, fontsize=9,
    )
    figure.suptitle(
        "Neural correspondence along model depth", fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.95))
    figure_path = Path(cfg.figs_dir) / "peak_profiles.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_best_layer_correspondence
Plot the best-layer peak RSA of every model and baseline per analysis.

INPUT:
    - summaries: dict -> {analysis: {model_name: summary}}
    - cfg: Cfg -> plotting configuration
    - baselines: dict | None -> single-layer reference levels
    - ceilings: dict | None -> per-analysis RDM reliability drawn as a line

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_best_layer_correspondence(
        summaries: dict, cfg: Cfg, baselines=None, ceilings=None,
        ) -> Path:
    entries = {
        **summaries["image"], **(baselines or {}).get("image", {}),
    }
    model_names = sorted(
        entries, key=lambda name: entries[name]["best_peak_similarity"],
        reverse=True,
    )
    bar_positions = np.arange(len(model_names))
    bar_width = 0.26

    figure, axis = plt.subplots(figsize=(1.15 * len(model_names) + 4, 5))
    for analysis_index, analysis in enumerate(ANALYSES):
        analysis_entries = {
            **summaries[analysis], **(baselines or {}).get(analysis, {}),
        }
        values = [
            analysis_entries[name]["best_peak_similarity"]
            for name in model_names
        ]
        bars = axis.bar(
            bar_positions + (analysis_index - 1) * bar_width, values,
            bar_width, label=ANALYSIS_LABELS[analysis],
            edgecolor="black", linewidth=0.6,
        )
        # Each analysis has its own RDM reliability, so it gets its own line.
        ceiling = (ceilings or {}).get(analysis)
        if ceiling is not None and ceiling == ceiling:
            axis.axhline(
                ceiling, color=bars[0].get_facecolor(), linewidth=1.3,
                linestyle="-.",
                label=f"{ANALYSIS_LABELS[analysis]} noise ceiling",
            )
        # end if a ceiling is available
    # end for analysis_index

    axis.set_xticks(bar_positions)
    axis.set_xticklabels(
        [entries[name]["label"] for name in model_names],
        rotation=35, ha="right", fontsize=9,
    )
    axis.set_ylabel(r"best-layer peak RSA ($\rho$)", fontsize=10)
    axis.legend(frameon=False, fontsize=8.5, ncol=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title(
        "Neural correspondence of the best layer against each analysis's "
        "noise ceiling, ordered by the image analysis",
        fontsize=12,
    )
    figure.tight_layout()
    figure_path = Path(cfg.figs_dir) / "best_layer_correspondence.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_correlation_versus_score
Plot neural correspondence against the temporal score for every analysis.

INPUT:
    - summaries: dict -> {analysis: {model_name: summary}}
    - cfg: Cfg -> plotting configuration
    - latency_name: str -> latency measure behind the temporal score
    - baselines: dict | None -> single-layer reference levels drawn as lines

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_correlation_versus_score(
        summaries: dict, cfg: Cfg, latency_name: str, baselines=None,
        ) -> Path:
    figure, axes = plt.subplots(
        1, len(ANALYSES), figsize=(5.0 * len(ANALYSES), 4.6), squeeze=False,
    )
    for axis, analysis in zip(axes[0], ANALYSES):
        # Points are drawn left to right so the label offsets can be staggered
        # in a repeating cycle; the models cluster tightly near rho = 1 and
        # collide badly with a single fixed offset.
        ordered_models = sorted(
            summaries[analysis],
            key=lambda name: summaries[analysis][name]["best_peak_similarity"],
        )
        label_offsets = ((8, 5), (8, -11), (-8, 7), (-8, -13))
        alignments = ("left", "left", "right", "right")
        scores = []
        for label_index, model_name in enumerate(ordered_models):
            summary = summaries[analysis][model_name]
            score = summary["temporal_scores"][latency_name]
            scores.append(score["rho"])
            axis.scatter(
                summary["best_peak_similarity"], score["rho"],
                s=140, color=FAMILY_COLORS.get(summary["family"], "0.4"),
                edgecolors="black", linewidths=0.7, zorder=3,
            )
            axis.annotate(
                summary["label"],
                (summary["best_peak_similarity"], score["rho"]),
                textcoords="offset points",
                xytext=label_offsets[label_index % len(label_offsets)],
                ha=alignments[label_index % len(alignments)],
                fontsize=7.5, color="0.15", zorder=4,
            )
        # end for label_index, model_name
        for baseline_name, baseline in (baselines or {}).get(
                analysis, {}
                ).items():
            axis.axvline(
                baseline["best_peak_similarity"], color="0.45",
                linestyle="--" if "flow" in baseline_name else ":",
                linewidth=1.2,
            )
            # Axis fraction keeps the label inside whatever y limits are used.
            axis.text(
                baseline["best_peak_similarity"], 0.02, baseline["label"],
                transform=axis.get_xaxis_transform(),
                rotation=90, fontsize=7, color="0.35",
                ha="right", va="bottom",
            )
        # end for baseline_name
        axis.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        axis.set_title(ANALYSIS_LABELS[analysis], fontsize=11)
        axis.set_xlabel(r"best-layer peak RSA ($\rho$)", fontsize=10)
        axis.set_ylabel(
            f"temporal score (rho depth vs {LATENCY_LABELS[latency_name]})",
            fontsize=10,
        )
        # Keep zero in view as the reference, but do not waste the range below
        # it when every model scores positive.
        finite_scores = [value for value in scores if value == value]
        lower_limit = min(0.0, min(finite_scores, default=0.0)) - 0.08
        axis.set_ylim(lower_limit, 1.12)
        axis.margins(x=0.16)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    # end for axis, analysis

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=9,
            markerfacecolor=color, markeredgecolor="black", label=family,
        )
        for family, color in FAMILY_COLORS.items()
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=len(FAMILY_COLORS),
        frameon=False, fontsize=9,
    )
    figure.suptitle(
        "Neural correspondence versus hierarchical temporal organization",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    figure_path = (
        Path(cfg.figs_dir) / f"correlation_versus_score_{latency_name}.png"
    )
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_temporal_score_bars
Plot the temporal score of every model side by side for the three analyses.

INPUT:
    - summaries: dict -> {analysis: {model_name: summary}}
    - cfg: Cfg -> plotting configuration
    - latency_name: str -> latency measure behind the temporal score

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_temporal_score_bars(
        summaries: dict, cfg: Cfg, latency_name: str,
        ) -> Path:
    model_names = list(summaries["image"])
    # Rank by neural correspondence so the strongest models are read first.
    model_names.sort(
        key=lambda name: summaries["image"][name]["best_peak_similarity"],
        reverse=True,
    )
    bar_positions = np.arange(len(model_names))
    bar_width = 0.26

    figure, axis = plt.subplots(figsize=(1.15 * len(model_names) + 4, 5))
    for analysis_index, analysis in enumerate(ANALYSES):
        scores = [
            summaries[analysis][name]["temporal_scores"][latency_name]["rho"]
            for name in model_names
        ]
        significant = [
            summaries[analysis][name]["temporal_scores"][latency_name][
                "shift_pvalue"
            ] < 0.05
            for name in model_names
        ]
        offsets = (analysis_index - 1) * bar_width
        bars = axis.bar(
            bar_positions + offsets, scores, bar_width,
            label=ANALYSIS_LABELS[analysis],
            edgecolor="black", linewidth=0.6,
        )
        # A filled marker above the bar flags p < 0.05 for that model.
        for bar, is_significant in zip(bars, significant):
            if is_significant:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.03 if bar.get_height() >= 0 else -0.08),
                    "*", ha="center", fontsize=12,
                )
            # end if is_significant
        # end for bar
    # end for analysis_index

    axis.axhline(0, color="black", linewidth=0.9)
    axis.set_xticks(bar_positions)
    axis.set_xticklabels(
        [summaries["image"][name]["label"] for name in model_names],
        rotation=35, ha="right", fontsize=9,
    )
    axis.set_ylabel(
        f"temporal score (rho depth vs {LATENCY_LABELS[latency_name]})",
        fontsize=10,
    )
    axis.set_ylim(-1.05, 1.05)
    axis.legend(frameon=False, fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title(
        "Temporal score per model, ordered by static-image neural correspondence",
        fontsize=12,
    )
    figure.tight_layout()
    figure_path = (
        Path(cfg.figs_dir) / f"temporal_score_bars_{latency_name}.png"
    )
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
cutoff_sensitivity_records
Recompute the temporal scores at several layer-inclusion cutoffs.

The cutoff decides which layers are informative enough to carry a latency, so
the whole result is re-derived under each setting to show that the reported
scores do not depend on that one choice.

INPUT:
    - cfg: Cfg -> resolved analysis configuration
    - summaries: dict -> {analysis: {model_name: summary}} at the main cutoff

OUTPUT:
    - records: list[dict] -> one entry per cutoff, analysis, and model
"""
def cutoff_sensitivity_records(cfg: Cfg, summaries: dict) -> list:
    from project_specific_utils import layer_depth_temporal_score

    records = []
    for absolute_floor, peak_fraction in cfg.sensitivity_cutoffs:
        for analysis in ANALYSES:
            for model_name, summary in summaries[analysis].items():
                peak_similarity = summary["peak_similarity"]
                # The unfiltered latencies are recovered from the timecourses
                # already stored in the summary, so nothing is recomputed twice.
                informative = (
                    (peak_similarity >= absolute_floor)
                    & (
                        peak_similarity
                        >= peak_fraction * summary["best_peak_similarity"]
                    )
                )
                latencies_ms = np.where(
                    informative,
                    summary["unfiltered_latencies"][cfg.primary_latency],
                    np.nan,
                )
                score = layer_depth_temporal_score(
                    latencies_ms, summary["layer_depths"],
                )
                records.append({
                    "absolute_floor": absolute_floor,
                    "peak_fraction": peak_fraction,
                    "analysis": analysis,
                    "model_name": model_name,
                    "label": summary["label"],
                    "n_informative_layers": int(informative.sum()),
                    "rho": score["rho"],
                    "pvalue": score["pvalue"],
                    "shift_pvalue": score["shift_pvalue"],
                })
            # end for model_name
        # end for analysis
    # end for cutoff
    return records
# EOF


"""
nanmean_or_none
Average the finite entries of an array, returning None when there are none.

INPUT:
    - values: np.ndarray -> values that may be entirely NaN

OUTPUT:
    - mean_value: float | None -> finite mean, or None for an empty selection
"""
def nanmean_or_none(values):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None
    # end if nothing finite
    return float(finite_values.mean())
# EOF


"""
save_summary_tables
Write the machine-readable summary of every model, analysis, and latency.

INPUT:
    - summaries: dict -> {analysis: {model_name: summary}}
    - cfg: Cfg -> resolved analysis configuration
    - missing_models: list[str] -> models without a complete saved RSA family
    - cutoff_sensitivity: list -> temporal scores recomputed at other cutoffs
    - noise_ceilings: dict | None -> per-analysis RDM reliability

OUTPUT:
    - output_path: Path -> saved JSON summary
"""
def save_summary_tables(
        summaries: dict, cfg: Cfg, missing_models: list,
        cutoff_sensitivity: list, noise_ceilings=None,
        ) -> Path:
    noise_ceilings = {} if noise_ceilings is None else dict(noise_ceilings)
    records = []
    for analysis in ANALYSES:
        for model_name, summary in summaries[analysis].items():
            record = {
                "analysis": analysis,
                "model_name": model_name,
                "label": summary["label"],
                "family": summary["family"],
                "n_layers": len(summary["layer_names"]),
                "best_layer_index": summary["best_layer_index"],
                "best_layer_name": (
                    summary["layer_names"][summary["best_layer_index"]]
                ),
                "best_peak_similarity": summary["best_peak_similarity"],
                "mean_peak_similarity": float(
                    np.nanmean(summary["peak_similarity"])
                ),
                "n_informative_layers": int(
                    summary["informative_layers"].sum()
                ),
            }
            for latency_name, score in summary["temporal_scores"].items():
                record[f"{latency_name}_rho"] = score["rho"]
                record[f"{latency_name}_pvalue"] = score["pvalue"]
                record[f"{latency_name}_shift_pvalue"] = score["shift_pvalue"]
                latencies_ms = summary["latencies"][latency_name]
                quarter = max(1, len(latencies_ms) // 4)
                # A quarter can be entirely NaN when its layers were dropped.
                record[f"{latency_name}_first"] = nanmean_or_none(
                    latencies_ms[:quarter]
                )
                record[f"{latency_name}_last"] = nanmean_or_none(
                    latencies_ms[-quarter:]
                )
            # end for latency_name
            records.append(record)
        # end for model_name
    # end for analysis

    output_path = Path(cfg.output_dir) / "model_hierarchy_summary.json"
    with open(output_path, "w") as summary_file:
        json.dump(
            {
                "config": asdict(cfg),
                "missing_models": missing_models,
                "records": records,
                "cutoff_sensitivity": cutoff_sensitivity,
                "noise_ceilings": noise_ceilings,
            },
            summary_file, indent=2, sort_keys=True,
        )
    # end with open

    # The layer-resolved arrays are kept separately for later plotting.
    array_path = Path(cfg.output_dir) / "model_hierarchy_layer_arrays.npz"
    arrays = {}
    for analysis in ANALYSES:
        for model_name, summary in summaries[analysis].items():
            prefix = f"{analysis}__{model_name}"
            arrays[f"{prefix}__layer_depths"] = summary["layer_depths"]
            arrays[f"{prefix}__peak_similarity"] = summary["peak_similarity"]
            arrays[f"{prefix}__time_ms"] = summary["time_ms"]
            arrays[f"{prefix}__timecourses"] = summary["smoothed_timecourses"]
            for latency_name, values in summary["latencies"].items():
                arrays[f"{prefix}__{latency_name}"] = values
            # end for latency_name
        # end for model_name
    # end for analysis
    np.savez_compressed(array_path, **arrays)
    return output_path
# EOF


"""
main
Load every model's saved RSA, summarize the hierarchy, and write the figures.
"""
def main() -> None:
    cfg = resolve_cfg_paths(parse_args())
    summaries, baselines, missing_models = collect_summaries(cfg)
    if not summaries["image"]:
        raise FileNotFoundError(
            "No model had a complete saved RSA family; run the RSA sweep first."
        )
    # end if nothing was loaded

    ceilings = None
    if cfg.noise_ceiling_path is not None:
        ceilings = rdm_noise_ceiling(
            cfg.noise_ceiling_path,
            {
                analysis: analysis_settings(cfg, analysis)[1]
                for analysis in ANALYSES
            },
        )
        print(f"Noise ceilings: {ceilings}", flush=True)
    # end if a ceiling was requested

    figure_paths = []
    for analysis in ANALYSES:
        figure_paths.append(
            plot_layer_curves(
                summaries[analysis], analysis, cfg, baselines[analysis],
            )
        )
        for latency_name in LATENCY_LABELS:
            figure_paths.append(
                plot_latency_scatters(
                    summaries[analysis], analysis, cfg, latency_name,
                )
            )
        # end for latency_name
    # end for analysis
    figure_paths.append(
        plot_peak_profiles(summaries, cfg, baselines, ceilings)
    )
    figure_paths.append(
        plot_best_layer_correspondence(summaries, cfg, baselines, ceilings)
    )
    for latency_name in LATENCY_LABELS:
        figure_paths.append(
            plot_correlation_versus_score(
                summaries, cfg, latency_name, baselines,
            )
        )
        figure_paths.append(
            plot_temporal_score_bars(summaries, cfg, latency_name)
        )
    # end for latency_name

    cutoff_sensitivity = cutoff_sensitivity_records(cfg, summaries)
    # Baselines are tabulated alongside the models but carry no temporal score.
    summary_path = save_summary_tables(
        {
            analysis: {**summaries[analysis], **baselines[analysis]}
            for analysis in ANALYSES
        },
        cfg, missing_models, cutoff_sensitivity, ceilings,
    )
    print(f"\nSaved {len(figure_paths)} figures to {cfg.figs_dir}")
    print(f"Saved summary to {summary_path}")
    if missing_models:
        print(f"Models without saved RSA results: {missing_models}")
    # end if missing_models
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
