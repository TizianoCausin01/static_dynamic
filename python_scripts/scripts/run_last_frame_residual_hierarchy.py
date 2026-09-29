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
# .venv/bin/python python_scripts/scripts/run_last_frame_residual_hierarchy.py


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.model_zoo import MODEL_ZOO
from scipy.stats import spearmanr
from project_specific_utils import (
    compute_rdm_timeseries, depth_latency_score,
    last_frame_residual_similarity, load_natraster,
    match_timed_static_movie_rasters, summarize_layer_timing,
)
from useful_stuff.general_utils import TimeSeries


# (dynamic session, static session, channel range, reliable-channel key).
MONKEY_SESSIONS = {
    "baby1": ("baby1_260716to24", "baby1_260718to27", (84, 186), None),
    "red": (
        "red_20260720to0926", "red_20260726to0923", (1, 64),
        "red_20260726to0923",
    ),
    "paul": (
        "paul_20260831to0926", "paul_20260901to0925", (1, 64),
        "paul_20260901to0923",
    ),
}

# Hierarchical models only; the single-layer baselines carry no depth.
HIERARCHICAL_MODELS = [
    name for name, spec in MODEL_ZOO.items()
    if spec.family != "low-level baseline"
]

MONKEY_MARKERS = {"baby1": "o", "red": "s", "paul": "^"}
MONKEY_COLORS = {"baby1": "#3474BD", "red": "#C4564C", "paul": "#11A579"}


@dataclass
class Cfg:
    monkeys: list[str] = field(default_factory=lambda: list(MONKEY_SESSIONS))
    model_names: list[str] = field(
        default_factory=lambda: list(HIERARCHICAL_MODELS)
    )
    model_pooling: str | None = "mean"
    model_dataset_name: str = "static_dynamic"

    source_neural_fs: float = 1000
    analysis_fs: float = 100
    neural_rdm_metric: str = "cosine_cnt"
    model_rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "spearman"

    previous_frame_ms: float = 2250
    last_frame_ms: float = 2500
    regression_fit_intercept: bool = True
    # End-exclusive neural-time windows for the two latency measures.
    centroid_window_ms: tuple[float, float] = (2300, 2700)
    peak_window_ms: tuple[float, float] = (2300, 2700)
    smoothing_sigma: float = 3
    centroid_similarity_cutoff: float = 0.01
    peak_similarity_cutoff: float = 0.01

    output_dir: str | None = None
    figs_dir: str | None = None
    dpi: int = 150
# EOF


"""
parse_args
Parse the monkey and model selection plus the feature pooling.

OUTPUT:
    - cfg: Cfg -> validated configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Layer depth versus last-frame latency, raw and after removing "
            "the earlier-frame model RDM, across monkeys and models."
        )
    )
    parser.add_argument("--monkeys", nargs="+", choices=tuple(MONKEY_SESSIONS))
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument("--model_pooling")
    parser.add_argument("--output_dir")
    parser.add_argument("--figs_dir")
    args = parser.parse_args()

    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    unknown_names = [name for name in cfg.model_names if name not in MODEL_ZOO]
    if unknown_names:
        parser.error(f"Unknown model names {unknown_names}.")
    # end if unknown model names
    pooling_name = cfg.model_pooling or "none"
    cfg.output_dir = str(Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "last_frame_residual_hierarchy"
        / f"{pooling_name}pool"
    ))
    cfg.figs_dir = str(Path(
        cfg.figs_dir
        or Path(paths["data_path"]) / "results"
        / "figs_last_frame_residual_hierarchy" / f"{pooling_name}pool"
    ))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.figs_dir).mkdir(parents=True, exist_ok=True)
    return cfg
# EOF


"""
load_dynamic_neural_rdms
Load one monkey's movie responses in matched stimulus order and build RDMs.

The static session only fixes the matched stimulus order; channel selection is
applied identically to both sessions.

INPUT:
    - monkey: str -> key of MONKEY_SESSIONS
    - cfg: Cfg -> analysis configuration

OUTPUT:
    - dynamic_neural_rdms: np.ndarray -> neural time x stimulus pairs
    - dynamic_time_ms: np.ndarray -> neural time coordinate
    - feature_names: list[str] -> movie names in neural stimulus order
    - n_channels: int -> retained channels
"""
def load_dynamic_neural_rdms(monkey: str, cfg: Cfg):
    dynamic_name, static_name, good_channels, reliable_key = (
        MONKEY_SESSIONS[monkey]
    )
    reliable_config = (
        PROJECT_ROOT / "reliable_channels.yaml" if reliable_key else None
    )
    data_dir = Path(paths["data_path"]) / "data"
    loaded = {}
    for session_name, suffix in ((dynamic_name, "vid"), (static_name, "img")):
        loaded[suffix] = load_natraster(
            data_dir / f"{session_name}_natraster_{suffix}.mat",
            good_channels=good_channels,
            reliable_channels_config=reliable_config,
            reliable_channels_key=reliable_key,
            return_channel_numbers=True,
        )
    # end for session_name, suffix
    if not np.array_equal(loaded["vid"][2], loaded["img"][2]):
        raise ValueError(f"{monkey}: sessions retained different channels.")
    # end if retained channels differ

    (
        _, dynamic_rasters, _, _, _, aligned_names,
    ) = match_timed_static_movie_rasters(
        loaded["img"][0], loaded["img"][1], loaded["vid"][0], loaded["vid"][1],
    )
    # channels x time x stimuli, downsampled to the model analysis rate.
    dynamic_ts = TimeSeries(dynamic_rasters, fs=cfg.source_neural_fs)
    dynamic_ts.resample(cfg.analysis_fs)
    dynamic_neural_rdms = compute_rdm_timeseries(
        dynamic_ts.get_array(), cfg.neural_rdm_metric,
    )
    dynamic_time_ms = (
        np.arange(dynamic_neural_rdms.shape[0]) * 1000 / dynamic_ts.get_fs()
    )
    feature_names = [Path(name).name for name in aligned_names["movie"]]
    return (
        dynamic_neural_rdms, dynamic_time_ms, feature_names,
        int(dynamic_rasters.shape[0]),
    )
# EOF


"""
analyze_model
Raw and residual last-frame latencies and temporal scores for one model.

INPUT:
    - dynamic_neural_rdms: np.ndarray -> neural time x stimulus pairs
    - dynamic_time_ms: np.ndarray -> neural time coordinate
    - feature_names: list[str] -> movie names in neural stimulus order
    - model_name: str -> registry key
    - cfg: Cfg -> analysis configuration

OUTPUT:
    - arrays: dict -> per-layer curves and latencies for the NPZ archive
    - record: dict -> JSON-friendly per-model summary
"""
def analyze_model(
        dynamic_neural_rdms, dynamic_time_ms, feature_names, model_name, cfg,
        ):
    (
        layer_names, raw_similarity, residual_similarity, regression_summaries,
    ) = last_frame_residual_similarity(
        dynamic_neural_rdms, feature_names,
        Path(paths["data_path"]) / "models", model_name,
        cfg.model_dataset_name, cfg.model_pooling, cfg.analysis_fs,
        cfg.previous_frame_ms, cfg.last_frame_ms, cfg.model_rdm_metric,
        cfg.rsa_metric, cfg.regression_fit_intercept,
    )
    layer_depths = np.linspace(0, 1, len(layer_names))
    arrays = {
        "layer_names": np.asarray(layer_names),
        "layer_depths": layer_depths,
        "time_ms": dynamic_time_ms,
        "variance_removed": np.asarray([
            summary["variance_removed"] for summary in regression_summaries
        ]),
    }
    spec = MODEL_ZOO[model_name]
    record = {
        "model_name": model_name, "label": spec.label, "family": spec.family,
        "n_layers": len(layer_names),
        "mean_variance_removed": float(arrays["variance_removed"].mean()),
    }
    for condition_name, similarity in (
            ("raw", raw_similarity), ("residual", residual_similarity)):
        (
            smoothed, centroid_latency_ms, peak_latency_ms, peak_similarity,
        ) = summarize_layer_timing(
            similarity, dynamic_time_ms, cfg.last_frame_ms,
            cfg.centroid_window_ms, cfg.peak_window_ms, cfg.smoothing_sigma,
            cfg.centroid_similarity_cutoff, cfg.peak_similarity_cutoff,
        )
        arrays[f"{condition_name}_similarity"] = similarity
        arrays[f"{condition_name}_smoothed"] = smoothed
        arrays[f"{condition_name}_centroid_latency_ms"] = centroid_latency_ms
        arrays[f"{condition_name}_peak_latency_ms"] = peak_latency_ms
        record[f"{condition_name}_best_peak_similarity"] = float(
            np.nanmax(peak_similarity)
        )
        # Average model fit: in-window peak RSA averaged over all layers.
        record[f"{condition_name}_mean_peak_similarity"] = float(
            np.nanmean(peak_similarity)
        )
        for timing_name, latency_ms in (
                ("centroid", centroid_latency_ms), ("peak", peak_latency_ms)):
            record[f"{condition_name}_{timing_name}"] = depth_latency_score(
                layer_depths, latency_ms,
            )
        # end for timing_name, latency_ms
    # end for condition_name, similarity
    return arrays, record
# EOF


"""
plot_scores_across_monkeys
Temporal score per model and monkey, raw versus residual, centroid and peak.

INPUT:
    - records: dict -> {monkey: {model_name: record}}
    - cfg: Cfg -> plotting configuration

OUTPUT:
    - figure_path: Path -> saved PNG
"""
def plot_scores_across_monkeys(records: dict, cfg: Cfg) -> Path:
    model_names = [
        name for name in cfg.model_names
        if any(name in by_model for by_model in records.values())
    ]
    positions = np.arange(len(model_names))
    figure, axes = plt.subplots(
        2, 2, figsize=(1.0 * len(model_names) * 2 + 3, 8),
        sharex=True, sharey=True, squeeze=False,
    )
    for row_index, timing_name in enumerate(("centroid", "peak")):
        for column_index, condition_name in enumerate(("raw", "residual")):
            axis = axes[row_index, column_index]
            key = f"{condition_name}_{timing_name}"
            means = []
            for position, model_name in zip(positions, model_names):
                values = []
                for monkey, by_model in records.items():
                    if model_name not in by_model:
                        continue
                    # end if model missing for this monkey
                    rho = by_model[model_name][key]["rho"]
                    if not np.isfinite(rho):
                        continue
                    # end if score undefined
                    values.append(rho)
                    axis.scatter(
                        position, rho, s=60, marker=MONKEY_MARKERS[monkey],
                        color=MONKEY_COLORS[monkey], edgecolors="black",
                        linewidths=0.6, zorder=3,
                        label=monkey if position == 0 else None,
                    )
                # end for monkey, by_model
                means.append(np.mean(values) if values else np.nan)
            # end for position, model_name
            # Short horizontal tick = across-monkey mean.
            axis.scatter(
                positions, means, marker="_", s=420, color="0.25",
                linewidths=1.8, zorder=2,
            )
            axis.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            axis.set_title(f"{condition_name.capitalize()} {timing_name}")
            axis.spines[["top", "right"]].set_visible(False)
        # end for column_index, condition_name
        axes[row_index, 0].set_ylabel("temporal score (rho)")
    # end for row_index, timing_name
    for axis in axes[-1]:
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [MODEL_ZOO[name].label for name in model_names],
            rotation=40, ha="right", fontsize=9,
        )
    # end for axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", frameon=False, ncol=3)
    figure.suptitle(
        "Last-frame depth-latency score, raw vs after removing the "
        f"{cfg.previous_frame_ms:g}-ms model RDM"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure_path = Path(cfg.figs_dir) / "temporal_score_across_monkeys.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_fit_vs_score
Average model fit against temporal score, one point per model and monkey.

Model fit is the layer-averaged in-window peak RSA of the same condition (raw
or residual) whose latencies define the score. Spearman rho is reported pooled
over all points and within each monkey.

INPUT:
    - records: dict -> {monkey: {model_name: record}}
    - cfg: Cfg -> plotting configuration

OUTPUT:
    - figure_path: Path -> saved PNG
    - fit_score_correlations: dict -> {condition_timing: {pooled/monkey: rho, p}}
"""
def plot_fit_vs_score(records: dict, cfg: Cfg):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    fit_score_correlations = {}
    for row_index, timing_name in enumerate(("centroid", "peak")):
        for column_index, condition_name in enumerate(("raw", "residual")):
            axis = axes[row_index, column_index]
            key = f"{condition_name}_{timing_name}"
            pooled_fit, pooled_score, text_lines = [], [], []
            fit_score_correlations[key] = {}
            for monkey, by_model in records.items():
                fit = np.asarray([
                    record[f"{condition_name}_mean_peak_similarity"]
                    for record in by_model.values()
                ])
                score = np.asarray([
                    record[key]["rho"] for record in by_model.values()
                ])
                valid = np.isfinite(fit) & np.isfinite(score)
                axis.scatter(
                    fit[valid], score[valid], s=60,
                    marker=MONKEY_MARKERS[monkey], color=MONKEY_COLORS[monkey],
                    edgecolors="black", linewidths=0.6, label=monkey,
                )
                # Label each point so outlying models can be identified.
                for model_name, x_value, y_value in zip(
                        np.asarray(list(by_model))[valid], fit[valid],
                        score[valid]):
                    axis.annotate(
                        MODEL_ZOO[model_name].label, (x_value, y_value),
                        fontsize=6, alpha=0.7, xytext=(3, 2),
                        textcoords="offset points",
                    )
                # end for model_name, x_value, y_value
                result = spearmanr(fit[valid], score[valid])
                fit_score_correlations[key][monkey] = {
                    "rho": float(result.statistic),
                    "pvalue": float(result.pvalue),
                }
                text_lines.append(
                    f"{monkey}: rho={result.statistic:+.2f} "
                    f"(p={result.pvalue:.2g})"
                )
                pooled_fit.extend(fit[valid])
                pooled_score.extend(score[valid])
            # end for monkey, by_model
            pooled = spearmanr(pooled_fit, pooled_score)
            fit_score_correlations[key]["pooled"] = {
                "rho": float(pooled.statistic), "pvalue": float(pooled.pvalue),
            }
            text_lines.insert(
                0, f"pooled: rho={pooled.statistic:+.2f} "
                f"(p={pooled.pvalue:.2g})",
            )
            axis.text(
                0.02, 0.02, "\n".join(text_lines), transform=axis.transAxes,
                fontsize=8, va="bottom", family="monospace",
                bbox={"facecolor": "white", "edgecolor": "0.7"},
            )
            axis.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            axis.set(
                title=f"{condition_name.capitalize()} {timing_name}",
                xlabel="average model fit (layer-mean peak RSA)",
                # Extra room below -0.75 keeps the stats box off the points.
                ylim=(-1.45, 1.1),
            )
            axis.spines[["top", "right"]].set_visible(False)
        # end for column_index, condition_name
        axes[row_index, 0].set_ylabel("temporal score (rho)")
    # end for row_index, timing_name
    axes[0, 0].legend(loc="upper left", frameon=False)
    figure.suptitle("Does the temporal score track how well the model fits IT?")
    figure.tight_layout()
    figure_path = Path(cfg.figs_dir) / "model_fit_vs_temporal_score.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    with open(
            Path(cfg.output_dir) / "fit_score_correlations.json", "w",
            ) as output_file:
        json.dump(fit_score_correlations, output_file, indent=2)
    # end with open
    return figure_path, fit_score_correlations
# EOF


"""
plot_monkey_scatter_grid
Residual (or raw) latency vs layer depth for every model of one monkey.

INPUT:
    - monkey: str -> monkey name
    - model_arrays: dict -> {model_name: arrays from analyze_model}
    - records: dict -> {model_name: record}
    - condition_name: str -> "raw" or "residual"
    - timing_name: str -> "centroid" or "peak"
    - cfg: Cfg -> plotting configuration

OUTPUT:
    - figure_path: Path -> saved PNG
"""
def plot_monkey_scatter_grid(
        monkey, model_arrays, records, condition_name, timing_name, cfg,
        ):
    model_names = list(model_arrays)
    n_columns = 4
    n_rows = int(np.ceil(len(model_names) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(4 * n_columns, 3.3 * n_rows),
        sharey=True, squeeze=False,
    )
    for axis, model_name in zip(axes.flat, model_names):
        arrays = model_arrays[model_name]
        latency_ms = arrays[f"{condition_name}_{timing_name}_latency_ms"]
        depths = arrays["layer_depths"]
        valid_layers = np.isfinite(latency_ms)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(depths)))
        axis.scatter(
            latency_ms[valid_layers], depths[valid_layers],
            c=colors[valid_layers], s=40, edgecolors="black", linewidths=0.5,
        )
        score = records[model_name][f"{condition_name}_{timing_name}"]
        axis.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        axis.set_title(
            f"{MODEL_ZOO[model_name].label}\n"
            fr"$\rho$={score['rho']:.2f}, p={score['pvalue']:.2g}",
            fontsize=10,
        )
        axis.set_xlim(
            cfg.centroid_window_ms[0] - cfg.last_frame_ms - 10,
            cfg.centroid_window_ms[1] - cfg.last_frame_ms + 10,
        )
        axis.spines[["top", "right"]].set_visible(False)
    # end for axis, model_name
    for axis in list(axes.flat)[len(model_names):]:
        axis.set_visible(False)
    # end for unused axes
    figure.supxlabel(f"{timing_name} latency from final-frame onset (ms)")
    figure.supylabel("normalized layer depth")
    figure.suptitle(f"{monkey}: {condition_name} last-frame {timing_name}")
    figure.tight_layout()
    figure_path = (
        Path(cfg.figs_dir)
        / f"{monkey}_{condition_name}_{timing_name}_scatter_grid.png"
    )
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_timecourses
Raw and residual smoothed final-frame RSA by layer for one model, all monkeys.

INPUT:
    - model_name: str -> registry key
    - arrays_by_monkey: dict -> {monkey: arrays from analyze_model}
    - cfg: Cfg -> plotting configuration

OUTPUT:
    - figure_path: Path -> saved PNG
"""
def plot_timecourses(model_name, arrays_by_monkey, cfg):
    monkeys = list(arrays_by_monkey)
    figure, axes = plt.subplots(
        len(monkeys), 2, figsize=(13, 3.4 * len(monkeys)),
        sharex=True, squeeze=False,
    )
    for row_index, monkey in enumerate(monkeys):
        arrays = arrays_by_monkey[monkey]
        colors = plt.cm.plasma(
            np.linspace(0.1, 0.9, len(arrays["layer_depths"]))
        )
        for column_index, condition_name in enumerate(("raw", "residual")):
            axis = axes[row_index, column_index]
            for layer_index, curve in enumerate(
                    arrays[f"{condition_name}_smoothed"]):
                axis.plot(
                    arrays["time_ms"], curve, color=colors[layer_index],
                    linewidth=1.4,
                )
            # end for layer_index, curve
            axis.axhline(0, color="grey", linewidth=0.8)
            axis.axvline(cfg.last_frame_ms, color="grey", linestyle="--")
            axis.axvspan(*cfg.centroid_window_ms, color="0.85", alpha=0.5)
            axis.set_title(f"{monkey}: {condition_name}", fontsize=10)
            axis.spines[["top", "right"]].set_visible(False)
        # end for column_index, condition_name
        axes[row_index, 0].set_ylabel(f"RSA ({cfg.rsa_metric})")
    # end for row_index, monkey
    for axis in axes[-1]:
        axis.set_xlabel("time from movie onset (ms)")
    # end for axis
    figure.suptitle(
        f"{MODEL_ZOO[model_name].label}: final-frame RSA by layer "
        f"(residual removes {cfg.previous_frame_ms:g} ms)"
    )
    figure.tight_layout()
    figure_path = Path(cfg.figs_dir) / f"{model_name}_timecourses.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
main
Run every monkey x model, save per-model archives, a summary JSON and figures.
"""
def main() -> None:
    cfg = parse_args()
    records = {}
    arrays_by_monkey = {}
    monkey_info = {}
    for monkey in cfg.monkeys:
        (
            dynamic_neural_rdms, dynamic_time_ms, feature_names, n_channels,
        ) = load_dynamic_neural_rdms(monkey, cfg)
        monkey_info[monkey] = {
            "n_stimuli": len(feature_names), "n_channels": n_channels,
        }
        print(
            f"{monkey}: {len(feature_names)} stimuli, {n_channels} channels",
            flush=True,
        )
        records[monkey], arrays_by_monkey[monkey] = {}, {}
        for model_name in cfg.model_names:
            try:
                arrays, record = analyze_model(
                    dynamic_neural_rdms, dynamic_time_ms, feature_names,
                    model_name, cfg,
                )
            except FileNotFoundError as error:
                print(f"  skipping {model_name}: {error}", flush=True)
                continue
            # end try
            np.savez(
                Path(cfg.output_dir) / f"{monkey}_{model_name}.npz", **arrays,
            )
            records[monkey][model_name] = record
            arrays_by_monkey[monkey][model_name] = arrays
            print(
                f"  {model_name:<24} residual centroid rho="
                f"{record['residual_centroid']['rho']:+.2f} "
                f"raw centroid rho={record['raw_centroid']['rho']:+.2f}",
                flush=True,
            )
        # end for model_name
    # end for monkey

    with open(Path(cfg.output_dir) / "summary.json", "w") as summary_file:
        json.dump(
            {
                "config": asdict(cfg), "monkeys": monkey_info,
                "sessions": {
                    monkey: MONKEY_SESSIONS[monkey] for monkey in cfg.monkeys
                },
                "records": records,
            },
            summary_file, indent=2,
        )
    # end with open

    plot_scores_across_monkeys(records, cfg)
    plot_fit_vs_score(records, cfg)
    for monkey in cfg.monkeys:
        for condition_name in ("raw", "residual"):
            for timing_name in ("centroid", "peak"):
                plot_monkey_scatter_grid(
                    monkey, arrays_by_monkey[monkey], records[monkey],
                    condition_name, timing_name, cfg,
                )
            # end for timing_name
        # end for condition_name
    # end for monkey
    for model_name in cfg.model_names:
        model_arrays = {
            monkey: arrays_by_monkey[monkey][model_name]
            for monkey in cfg.monkeys
            if model_name in arrays_by_monkey[monkey]
        }
        if model_arrays:
            plot_timecourses(model_name, model_arrays, cfg)
        # end if the model ran for any monkey
    # end for model_name
    print(f"Saved results to {cfg.output_dir} and figures to {cfg.figs_dir}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
