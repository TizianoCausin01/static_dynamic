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
# .venv/bin/python python_scripts/scripts/run_static_dynamic_model_hierarchy_rdm_control.py


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.model_zoo import MODEL_ZOO, get_model_spec
from image_processing.video_feature_extraction import (
    list_video_feature_files,
    match_feature_stimulus_names,
)
from project_specific_utils import (
    latency_profile_smoothness,
    layer_depth_temporal_score,
    permuted_static_rsa,
    prepare_static_neural_rdms,
    static_model_rdms_by_layer,
    summarize_model_analysis,
)


@dataclass
class Cfg:
    monkey_name: str = "baby1"
    static_experiment_name: str = "baby1_260718to27"
    dynamic_experiment_name: str = "baby1_260716to24"
    good_channels: tuple[int, int] | None = (84, 186)
    reliable_channels_config: str | None = None
    reliable_channels_key: str | None = "baby1_260718to27"

    model_names: list[str] = field(default_factory=lambda: list(MODEL_ZOO))
    static_path: str | None = None
    model_features_dir: str | None = None
    output_dir: str | None = None
    figs_dir: str | None = None

    # Must reproduce the saved RSA family the hierarchy result was read from.
    signal_rdm_metric: str = "cosine_cnt"
    model_rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "spearman"
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float | None = 1000
    normalization: str | None = None
    model_pooling: str | None = "mean"
    model_dataset_name: str = "static_dynamic"
    model_frame_index: int = -1

    # The image analysis reads the whole cropped static response.
    image_window_ms: tuple[float, float] | None = None
    smoothing_sigma: float = 3

    # Latency settings shared by the observed and permuted runs. The published
    # analysis drops layers whose RSA is near zero, which a permuted model RDM
    # never clears, so the matched comparison switches every magnitude cutoff
    # off and lets all layers through. The published cutoffs are still applied
    # to both runs, and the permuted pass rate is reported alongside.
    matched_absolute_cutoff: float = 0.0
    matched_relative_cutoff: float = 0.5
    default_absolute_cutoff: float = 0.02
    default_relative_cutoff: float = 0.5
    default_min_peak_similarity: float = 0.02
    default_min_peak_fraction: float = 0.15

    n_permutations: int = 200
    random_seed: int = 0
    primary_latency: str = "relative_centroid_latency_ms"
    n_example_permutations: int = 20

    grid_shape: tuple[int, int] = (3, 3)
    dpi: int = 150
# EOF


LATENCY_NAMES = (
    "centroid_latency_ms", "relative_centroid_latency_ms", "peak_latency_ms",
)

LATENCY_LABELS = {
    "centroid_latency_ms": "centroid latency",
    "relative_centroid_latency_ms": "half-height centroid latency",
    "peak_latency_ms": "peak latency",
}

# Observed and null are one categorical pair; the null cloud stays neutral.
OBSERVED_COLOR = "#2a78d6"
PERMUTED_COLOR = "#eb6834"
NULL_COLOR = "#8A8A8A"


"""
parse_args
Parse the model selection, permutation count, and output locations.

OUTPUT:
    - cfg: Cfg -> validated control configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Permutation control for the image-analysis layer-depth versus "
            "latency result: the neural RDMs are held fixed while every model "
            "RDM is relabelled with one shared stimulus permutation."
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
    parser.add_argument("--reliable_channels_config")
    parser.add_argument("--reliable_channels_key")
    parser.add_argument("--static_path")
    parser.add_argument("--model_features_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--figs_dir")
    parser.add_argument(
        "--n_permutations", type=int, default=Cfg.n_permutations,
        help="Number of shared stimulus relabellings forming the null.",
    )
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument(
        "--primary_latency", choices=LATENCY_NAMES,
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
    if cfg.n_permutations < 1:
        parser.error("--n_permutations must be positive.")
    # end if invalid permutation count
    unknown_names = [name for name in cfg.model_names if name not in MODEL_ZOO]
    if unknown_names:
        parser.error(f"Unknown model names {unknown_names}.")
    # end if unknown_names
    return cfg
# EOF


"""
resolve_cfg_paths
Fill in the data, feature, output, and figure locations.

INPUT:
    - cfg: Cfg -> user configuration

OUTPUT:
    - cfg: Cfg -> configuration with explicit paths
"""
def resolve_cfg_paths(cfg: Cfg) -> Cfg:
    experiment_name = (
        f"{cfg.dynamic_experiment_name}_vs_{cfg.static_experiment_name}"
    )
    data_dir = Path(paths["data_path"]) / "data"
    cfg.static_path = str(Path(
        cfg.static_path
        or data_dir / f"{cfg.static_experiment_name}_natraster_img.mat"
    ).expanduser())
    cfg.model_features_dir = str(Path(
        cfg.model_features_dir or Path(paths["data_path"]) / "models"
    ).expanduser())
    # The reliability YAML lives at the project root, as in the RSA sweep.
    cfg.reliable_channels_config = str(Path(
        cfg.reliable_channels_config or PROJECT_ROOT / "reliable_channels.yaml"
    ).expanduser())
    cfg.output_dir = str(Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "static_dynamic_model_hierarchy_rdm_control"
        / experiment_name
    ).expanduser())
    cfg.figs_dir = str(Path(
        cfg.figs_dir
        or Path(paths["data_path"]) / "results"
        / "figs_model_hierarchy_rdm_control" / experiment_name
    ).expanduser())
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.figs_dir).mkdir(parents=True, exist_ok=True)
    return cfg
# EOF


"""
image_summary
Run the image-analysis latency summary on one layers x neural-time RSA matrix.

The hierarchy utilities read their input as the dict written by
load_model_layer_rsa, so the recomputed RSA is wrapped in the same shape.

INPUT:
    - static_rsa: np.ndarray -> layers x neural time RSA values
    - time_ms: np.ndarray -> neural time coordinates
    - layer_names: list[str] -> layers in shallow-to-deep order
    - cfg: Cfg -> window and smoothing configuration
    - matched: bool -> use the cutoff-free matched settings, else the published
        cutoffs of the original hierarchy analysis

OUTPUT:
    - summary: dict -> output of summarize_model_analysis
"""
def image_summary(
        static_rsa: np.ndarray,
        time_ms: np.ndarray,
        layer_names: list[str],
        cfg: Cfg,
        matched: bool = True,
        ) -> dict:
    results = {
        "model_name": "",
        "layer_names": layer_names,
        "layer_depths": np.linspace(0, 1, len(layer_names)),
        "static_rsa": static_rsa,
        "static_time_ms": time_ms,
    }
    if matched:
        latency_kwargs = {
            "absolute_cutoff": cfg.matched_absolute_cutoff,
            "relative_cutoff": cfg.matched_relative_cutoff,
            "smoothing_sigma": cfg.smoothing_sigma,
        }
        min_peak_similarity, min_peak_fraction = 0.0, 0.0
    else:
        latency_kwargs = {
            "absolute_cutoff": cfg.default_absolute_cutoff,
            "relative_cutoff": cfg.default_relative_cutoff,
            "smoothing_sigma": cfg.smoothing_sigma,
        }
        min_peak_similarity = cfg.default_min_peak_similarity
        min_peak_fraction = cfg.default_min_peak_fraction
    # end if matched settings
    return summarize_model_analysis(
        results, "image",
        window_ms=cfg.image_window_ms,
        latency_kwargs=latency_kwargs,
        min_peak_similarity=min_peak_similarity,
        min_peak_fraction=min_peak_fraction,
    )
# EOF



"""
layer_set_rhos
Score latency profiles on one fixed set of layers.

The published hierarchy analysis drops layers whose RSA is near zero, and a
permuted model RDM never clears that floor. Freezing the layer set to the one
the observed run selected lets the observed and permuted scores be compared on
the same layers and the same depth axis, so the only thing the null changes is
the stimulus labelling.

INPUT:
    - latencies: dict -> {latency name: per-layer latencies}
    - layer_depths: np.ndarray -> normalized depth of every layer
    - layer_mask: np.ndarray[bool] -> layers the observed run retained

OUTPUT:
    - rhos: dict -> {latency name: Spearman rho on the retained layers}
"""
def layer_set_rhos(latencies: dict, layer_depths, layer_mask) -> dict:
    return {
        name: layer_depth_temporal_score(
            np.where(layer_mask, values, np.nan), layer_depths,
        )["rho"]
        for name, values in latencies.items()
    }
# EOF


"""
run_model_control
Compute the observed and permuted depth-to-latency statistics of one model.

The layer RDMs are built once, then reused for every permutation, so the whole
null costs one pass over the feature files.

INPUT:
    - model_name: str -> registry key
    - state: dict -> output of prepare_static_neural_rdms
    - permutations: np.ndarray -> permutations x stimuli, shared by all models
    - cfg: Cfg -> resolved control configuration

OUTPUT:
    - record: dict -> observed statistics, null distributions, and profiles
"""
def run_model_control(
        model_name: str,
        state: dict,
        permutations: np.ndarray,
        cfg: Cfg,
        ) -> dict:
    spec = get_model_spec(model_name)
    feature_paths = list_video_feature_files(
        cfg.model_features_dir, model_name, cfg.model_dataset_name,
        spec.pooling,
    )
    static_feature_names = match_feature_stimulus_names(
        feature_paths[0], state["stimulus_names"],
    )
    layer_rdms, file_layer_names = static_model_rdms_by_layer(
        feature_paths, static_feature_names, cfg.model_rdm_metric,
        cfg.model_frame_index,
    )
    # The feature files come back in natural filename order; the registry holds
    # the shallow-to-deep order the depth axis is built from.
    missing_layers = [
        layer for layer in spec.layers if layer not in file_layer_names
    ]
    if missing_layers:
        raise FileNotFoundError(
            f"{model_name}: no feature file for {missing_layers[:3]}."
        )
    # end if missing_layers
    depth_order = [file_layer_names.index(layer) for layer in spec.layers]
    layer_rdms = layer_rdms[depth_order]

    observed_rsa = permuted_static_rsa(
        state["neural_rdms"], layer_rdms, None, cfg.rsa_metric,
    )
    observed = image_summary(
        observed_rsa, state["time_ms"], spec.layers, cfg, matched=True,
    )
    observed_default = image_summary(
        observed_rsa, state["time_ms"], spec.layers, cfg, matched=False,
    )

    # The layer set the published cutoffs retain, frozen for the null too.
    published_layer_mask = observed_default["informative_layers"]
    observed_pubset_rhos = layer_set_rhos(
        observed["latencies"], observed["layer_depths"],
        published_layer_mask,
    )

    null_rhos = {name: [] for name in LATENCY_NAMES}
    null_pubset_rhos = {name: [] for name in LATENCY_NAMES}
    null_default_rhos = {name: [] for name in LATENCY_NAMES}
    null_steps, null_ranges = [], []
    # The permuted RSA collapses towards zero, so its peak is recorded to
    # show how much of the null's behaviour is a loss of signal rather than
    # a loss of depth alignment.
    null_peaks = []
    example_latencies = []
    for permutation_index, permutation in enumerate(permutations):
        permuted_rsa = permuted_static_rsa(
            state["neural_rdms"], layer_rdms, permutation, cfg.rsa_metric,
        )
        permuted = image_summary(
            permuted_rsa, state["time_ms"], spec.layers, cfg, matched=True,
        )
        permuted_default = image_summary(
            permuted_rsa, state["time_ms"], spec.layers, cfg, matched=False,
        )
        permuted_pubset_rhos = layer_set_rhos(
            permuted["latencies"], permuted["layer_depths"],
            published_layer_mask,
        )
        for name in LATENCY_NAMES:
            null_rhos[name].append(permuted["temporal_scores"][name]["rho"])
            null_pubset_rhos[name].append(permuted_pubset_rhos[name])
            null_default_rhos[name].append(
                permuted_default["temporal_scores"][name]["rho"]
            )
        # end for name
        smoothness = latency_profile_smoothness(
            permuted["latencies"][cfg.primary_latency]
        )
        null_steps.append(smoothness["adjacent_step_ms"])
        null_ranges.append(smoothness["range_ms"])
        null_peaks.append(float(np.nanmax(permuted["peak_similarity"])))
        # A handful of permuted profiles are kept so the figure can show that
        # they stay as smooth as the observed one.
        if permutation_index < cfg.n_example_permutations:
            example_latencies.append(permuted["latencies"][cfg.primary_latency])
        # end if an example profile is kept
    # end for permutation_index

    observed_smoothness = latency_profile_smoothness(
        observed["latencies"][cfg.primary_latency]
    )
    return {
        "model_name": model_name,
        "label": spec.label,
        "family": spec.family,
        "layer_names": list(spec.layers),
        "layer_depths": observed["layer_depths"],
        "observed_peak_similarity": observed["peak_similarity"],
        "observed_latencies": observed["latencies"],
        "observed_rhos": {
            name: observed["temporal_scores"][name]["rho"]
            for name in LATENCY_NAMES
        },
        "observed_default_rhos": {
            name: observed_default["temporal_scores"][name]["rho"]
            for name in LATENCY_NAMES
        },
        "published_layer_mask": published_layer_mask,
        "observed_pubset_rhos": observed_pubset_rhos,
        "null_pubset_rhos": {
            name: np.asarray(values, dtype=float)
            for name, values in null_pubset_rhos.items()
        },
        "observed_default_n_layers": int(
            observed_default["informative_layers"].sum()
        ),
        "observed_smoothness": observed_smoothness,
        "null_rhos": {
            name: np.asarray(values, dtype=float)
            for name, values in null_rhos.items()
        },
        "null_default_rhos": {
            name: np.asarray(values, dtype=float)
            for name, values in null_default_rhos.items()
        },
        "null_adjacent_step_ms": np.asarray(null_steps, dtype=float),
        "null_best_peak_similarity": np.asarray(null_peaks, dtype=float),
        "null_range_ms": np.asarray(null_ranges, dtype=float),
        "example_latencies": np.asarray(example_latencies, dtype=float),
    }
# EOF


"""
permutation_pvalue
Two-sided permutation p value of an observed rank correlation.

INPUT:
    - observed_rho: float -> depth-to-latency correlation of the real RSA
    - null_rhos: np.ndarray -> the same statistic under every permutation

OUTPUT:
    - pvalue: float -> fraction of the null at least as extreme, NaN if empty
"""
def permutation_pvalue(observed_rho: float, null_rhos: np.ndarray) -> float:
    null_rhos = np.asarray(null_rhos, dtype=float)
    finite_rhos = null_rhos[np.isfinite(null_rhos)]
    if not np.isfinite(observed_rho) or finite_rhos.size == 0:
        return float("nan")
    # end if nothing to compare
    return float(
        (1 + np.sum(np.abs(finite_rhos) >= abs(observed_rho)))
        / (finite_rhos.size + 1)
    )
# EOF


"""
summarize_record
Reduce one model's control record to the JSON row reported for it.

INPUT:
    - record: dict -> output of run_model_control
    - cfg: Cfg -> primary latency selection

OUTPUT:
    - row: dict -> observed statistics, null summary, and permutation p values
"""
def summarize_record(record: dict, cfg: Cfg) -> dict:
    row = {
        "model_name": record["model_name"],
        "label": record["label"],
        "family": record["family"],
        "n_layers": len(record["layer_names"]),
        "best_peak_similarity": float(
            np.nanmax(record["observed_peak_similarity"])
        ),
    }
    for name in LATENCY_NAMES:
        observed_rho = record["observed_rhos"][name]
        null_rhos = record["null_rhos"][name]
        finite_rhos = null_rhos[np.isfinite(null_rhos)]
        row[f"{name}_observed_rho"] = float(observed_rho)
        row[f"{name}_observed_default_rho"] = float(
            record["observed_default_rhos"][name]
        )
        row[f"{name}_null_mean_rho"] = (
            float(finite_rhos.mean()) if finite_rhos.size else float("nan")
        )
        row[f"{name}_null_sd_rho"] = (
            float(finite_rhos.std(ddof=1)) if finite_rhos.size > 1
            else float("nan")
        )
        # The null is two-sided, so its |rho| spread is what the observed value
        # has to clear.
        row[f"{name}_null_abs_rho_p95"] = (
            float(np.percentile(np.abs(finite_rhos), 95))
            if finite_rhos.size else float("nan")
        )
        row[f"{name}_permutation_pvalue"] = permutation_pvalue(
            observed_rho, null_rhos,
        )
        # Permutations whose latencies come out constant across layers
        # leave the rank correlation undefined and drop out of the null.
        row[f"{name}_null_defined_fraction"] = float(
            finite_rhos.size / null_rhos.size
        )
        # Same statistic restricted to the layers the published analysis
        # kept, which is the comparison the two runs actually share.
        pubset_observed = record["observed_pubset_rhos"][name]
        pubset_null = record["null_pubset_rhos"][name]
        finite_pubset = pubset_null[np.isfinite(pubset_null)]
        row[f"{name}_pubset_observed_rho"] = float(pubset_observed)
        row[f"{name}_pubset_null_abs_rho_p95"] = (
            float(np.percentile(np.abs(finite_pubset), 95))
            if finite_pubset.size else float("nan")
        )
        row[f"{name}_pubset_permutation_pvalue"] = permutation_pvalue(
            pubset_observed, pubset_null,
        )
        # How often the published cutoffs let a permuted run score at all.
        default_null = record["null_default_rhos"][name]
        row[f"{name}_default_null_defined_fraction"] = float(
            np.mean(np.isfinite(default_null))
        )
    # end for name

    observed_smoothness = record["observed_smoothness"]
    null_steps = record["null_adjacent_step_ms"]
    null_ranges = record["null_range_ms"]
    row.update({
        "observed_adjacent_step_ms": observed_smoothness["adjacent_step_ms"],
        "observed_range_ms": observed_smoothness["range_ms"],
        "observed_step_over_range": observed_smoothness["step_over_range"],
        "null_adjacent_step_ms_mean": float(np.nanmean(null_steps)),
        "null_range_ms_mean": float(np.nanmean(null_ranges)),
        "null_step_over_range_mean": float(
            np.nanmean(null_steps / np.where(null_ranges > 0, null_ranges, np.nan))
        ),
        "null_best_peak_similarity_mean": float(
            np.nanmean(record["null_best_peak_similarity"])
        ),
        "observed_default_n_informative_layers": (
            record["observed_default_n_layers"]
        ),
        "n_published_layers": int(record["published_layer_mask"].sum()),
        "primary_latency": cfg.primary_latency,
    })
    return row
# EOF


"""
grid_axes
Create a figure grid large enough for one panel per model.

INPUT:
    - n_panels: int -> number of models to display
    - cfg: Cfg -> grid shape
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
plot_null_distributions
Show each model's observed depth-to-latency rho against its permutation null.

INPUT:
    - records: list[dict] -> per-model control records
    - cfg: Cfg -> primary latency and figure settings

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_null_distributions(records: list[dict], cfg: Cfg) -> Path:
    # Single-layer baselines describe a reference level, not a hierarchy.
    records = [record for record in records if len(record["layer_names"]) > 1]
    figure, axes = grid_axes(len(records), cfg)
    for axis, record in zip(axes, records):
        # The frozen-layer-set variant: same layers and same depth axis as
        # the published analysis, so only the relabelling separates them.
        null_rhos = record["null_pubset_rhos"][cfg.primary_latency]
        finite_rhos = null_rhos[np.isfinite(null_rhos)]
        axis.hist(
            finite_rhos, bins=np.linspace(-1, 1, 41),
            color=NULL_COLOR, alpha=0.55, edgecolor="none",
            label=f"permuted null (n={finite_rhos.size})",
        )
        observed_rho = record["observed_pubset_rhos"][cfg.primary_latency]
        axis.axvline(
            observed_rho, color=OBSERVED_COLOR, linewidth=2, label="observed",
        )
        axis.annotate(
            f"rho = {observed_rho:.2f}",
            xy=(observed_rho, axis.get_ylim()[1]),
            xytext=(4, -4), textcoords="offset points",
            ha="left", va="top", fontsize=8, color="#52514e",
        )
        axis.set_title(f"{record['label']}", fontsize=10)
        axis.set_xlim(-1, 1)
        axis.set_xlabel("depth-latency Spearman rho")
        axis.set_ylabel("permutations")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        axis.set_axisbelow(True)
    # end for axis, record
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(
        "Image analysis: observed depth-to-latency correlation against the "
        f"shared-permutation null ({LATENCY_LABELS[cfg.primary_latency]})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = Path(cfg.figs_dir) / "rdm_control_null_distributions.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_latency_profiles
Draw the observed latency-versus-depth profile over permuted example profiles.

This is the panel the control was built for: if permuted profiles stay as
smooth as the observed one while losing its slope, the smoothness is structural
and the slope is not.

INPUT:
    - records: list[dict] -> per-model control records
    - cfg: Cfg -> primary latency and figure settings

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_latency_profiles(records: list[dict], cfg: Cfg) -> Path:
    records = [record for record in records if len(record["layer_names"]) > 1]
    figure, axes = grid_axes(len(records), cfg)
    for axis, record in zip(axes, records):
        depths = record["layer_depths"]
        for example_index, latencies in enumerate(record["example_latencies"]):
            axis.plot(
                depths, latencies, color=PERMUTED_COLOR, linewidth=1,
                alpha=0.35,
                label="permuted examples" if example_index == 0 else None,
            )
        # end for example_index
        axis.plot(
            depths, record["observed_latencies"][cfg.primary_latency],
            color=OBSERVED_COLOR, linewidth=2, marker="o", markersize=4,
            label="observed",
        )
        axis.set_title(record["label"], fontsize=10)
        axis.set_xlabel("normalized layer depth")
        axis.set_ylabel(f"{LATENCY_LABELS[cfg.primary_latency]} (ms)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#e6e5e1", linewidth=0.6)
        axis.set_axisbelow(True)
    # end for axis, record
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        "Image analysis: latency versus layer depth, observed and under "
        "shared stimulus relabelling of every model RDM",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure_path = Path(cfg.figs_dir) / "rdm_control_latency_profiles.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
plot_control_summary
Compare every model's observed rho and profile smoothness with its null.

INPUT:
    - rows: list[dict] -> per-model summary rows
    - cfg: Cfg -> primary latency and figure settings

OUTPUT:
    - figure_path: Path -> saved figure
"""
def plot_control_summary(rows: list[dict], cfg: Cfg) -> Path:
    rows = [row for row in rows if row["n_layers"] > 1]
    order = np.argsort([
        row[f"{cfg.primary_latency}_pubset_observed_rho"] for row in rows
    ])
    sorted_rows = [rows[index] for index in order]
    labels = [row["label"] for row in sorted_rows]
    positions = np.arange(len(sorted_rows))
    figure, axes = plt.subplots(
        1, 2, figsize=(13, 0.32 * len(sorted_rows) + 3.2),
    )

    observed_rhos = np.array([
        row[f"{cfg.primary_latency}_pubset_observed_rho"]
        for row in sorted_rows
    ])
    null_p95 = np.array([
        row[f"{cfg.primary_latency}_pubset_null_abs_rho_p95"]
        for row in sorted_rows
    ])
    # The null band is symmetric because the permuted statistic is two-sided.
    axes[0].barh(
        positions, 2 * null_p95, left=-null_p95, height=0.62,
        color=NULL_COLOR, alpha=0.35, edgecolor="none",
        label="permuted null, central 95%",
    )
    axes[0].scatter(
        observed_rhos, positions, color=OBSERVED_COLOR, s=34, zorder=3,
        label="observed",
    )
    axes[0].axvline(0, color="#c3c2b7", linewidth=1)
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_xlabel("depth-latency Spearman rho")
    axes[0].set_title("Depth-to-latency correlation", fontsize=11)

    observed_steps = np.array([
        row["observed_adjacent_step_ms"] for row in sorted_rows
    ])
    null_steps = np.array([
        row["null_adjacent_step_ms_mean"] for row in sorted_rows
    ])
    axes[1].scatter(
        observed_steps, positions, color=OBSERVED_COLOR, s=34, zorder=3,
        label="observed",
    )
    axes[1].scatter(
        null_steps, positions, color=PERMUTED_COLOR, s=34, marker="D",
        zorder=3, label="permuted mean",
    )
    axes[1].set_xlabel("mean |latency step| between adjacent layers (ms)")
    axes[1].set_title("Latency-profile smoothness", fontsize=11)

    for axis in axes:
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#e6e5e1", linewidth=0.6)
        axis.set_axisbelow(True)
        # Anchored outside the data so it cannot sit on a model row.
        axis.legend(
            frameon=False, fontsize=9, loc="upper left",
            bbox_to_anchor=(0, -0.08), ncol=2, markerscale=0.8,
        )
    # end for axis
    figure.suptitle(
        "Shared-permutation control on the image analysis "
        f"({LATENCY_LABELS[cfg.primary_latency]})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure_path = Path(cfg.figs_dir) / "rdm_control_summary.png"
    figure.savefig(figure_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path
# EOF


"""
save_control_results
Write the JSON summary and the per-model arrays behind it.

INPUT:
    - records: list[dict] -> per-model control records
    - rows: list[dict] -> per-model summary rows
    - permutations: np.ndarray -> the shared relabellings that were used
    - cfg: Cfg -> resolved control configuration

OUTPUT:
    - summary_path: Path -> saved JSON summary
"""
def save_control_results(
        records: list[dict], rows: list[dict], permutations: np.ndarray,
        cfg: Cfg,
        ) -> Path:
    summary_path = Path(cfg.output_dir) / "rdm_control_summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(
            {"config": asdict(cfg), "records": rows},
            summary_file, indent=2, sort_keys=True,
        )
    # end with open

    arrays = {"permutations": permutations}
    for record in records:
        model_name = record["model_name"]
        arrays[f"{model_name}__layer_depths"] = record["layer_depths"]
        arrays[f"{model_name}__published_layer_mask"] = (
            record["published_layer_mask"]
        )
        arrays[f"{model_name}__peak_similarity"] = (
            record["observed_peak_similarity"]
        )
        arrays[f"{model_name}__example_latencies"] = (
            record["example_latencies"]
        )
        arrays[f"{model_name}__null_adjacent_step_ms"] = (
            record["null_adjacent_step_ms"]
        )
        arrays[f"{model_name}__null_best_peak_similarity"] = (
            record["null_best_peak_similarity"]
        )
        for name in LATENCY_NAMES:
            arrays[f"{model_name}__observed_{name}"] = (
                record["observed_latencies"][name]
            )
            arrays[f"{model_name}__null_rho_{name}"] = record["null_rhos"][name]
            arrays[f"{model_name}__null_pubset_rho_{name}"] = (
                record["null_pubset_rhos"][name]
            )
        # end for name
    # end for record
    np.savez_compressed(
        Path(cfg.output_dir) / "rdm_control_layer_arrays.npz", **arrays,
    )
    return summary_path
# EOF


"""
main
Run the shared-permutation control over the image analysis of every model.
"""
def main() -> None:
    cfg = resolve_cfg_paths(parse_args())
    state = prepare_static_neural_rdms(
        cfg.static_path,
        good_channels=cfg.good_channels,
        reliable_channels_config=cfg.reliable_channels_config,
        reliable_channels_key=cfg.reliable_channels_key,
        source_fs=cfg.source_fs,
        new_fs=cfg.new_fs,
        crop_ms=cfg.static_crop_ms,
        normalization=cfg.normalization,
        rdm_metric=cfg.signal_rdm_metric,
    )
    n_stimuli = len(state["stimulus_names"])
    print(
        f"Neural RDMs: {state['neural_rdms'].shape} "
        f"(time, pairs) over {n_stimuli} stimuli and "
        f"{len(state['channel_numbers'])} channels",
        flush=True,
    )

    # One relabelling per permutation, reused by every layer of every model, so
    # the between-layer RDM geometry is never disturbed.
    rng = np.random.default_rng(cfg.random_seed)
    permutations = np.stack([
        rng.permutation(n_stimuli) for _ in range(cfg.n_permutations)
    ])

    records = []
    for model_name in cfg.model_names:
        try:
            record = run_model_control(model_name, state, permutations, cfg)
        except (FileNotFoundError, KeyError) as error:
            print(f"skipping {model_name}: {error}", flush=True)
            continue
        # end try
        records.append(record)
        observed_rho = record["observed_pubset_rhos"][cfg.primary_latency]
        null_rhos = record["null_pubset_rhos"][cfg.primary_latency]
        print(
            f"{model_name}: {len(record['layer_names'])} layers, "
            f"observed rho {observed_rho:+.3f}, null |rho| p95 "
            f"{np.nanpercentile(np.abs(null_rhos), 95):.3f}",
            flush=True,
        )
    # end for model_name
    if not records:
        raise RuntimeError("No model produced a usable control record.")
    # end if nothing to report

    rows = [summarize_record(record, cfg) for record in records]
    summary_path = save_control_results(records, rows, permutations, cfg)
    figure_paths = [
        plot_null_distributions(records, cfg),
        plot_latency_profiles(records, cfg),
        plot_control_summary(rows, cfg),
    ]
    print(f"Summary saved to {summary_path}", flush=True)
    for figure_path in figure_paths:
        print(f"Figure saved to {figure_path}", flush=True)
    # end for figure_path
# EOF


if __name__ == "__main__":
    main()
# EOF
