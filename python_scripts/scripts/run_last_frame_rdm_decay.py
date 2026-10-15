import argparse
import csv
from collections import Counter
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

from project_specific_utils import load_reliable_channels
from project_specific_utils.last_frame_decay import (
    load_last_frame_decay_data,
    resolve_visual_feature_path,
)
from project_specific_utils.last_frame_rdm_decay import (
    DECAY_MODELS,
    compute_last_frame_rdm_timecourses,
    curve_lags,
    fit_decay_models,
    plot_rdm_decay_fits,
)


def _layer_specs(model_name, layer_indices):
    """Build "model:PC1000:layer" specs for several layers of one model."""
    return tuple(f"{model_name}:PC1000:{index}" for index in layer_indices)
# EOF


@dataclass
class Cfg:
    static_exp_name: str = "red_20260726to0925"
    dynamic_exp_name: str = "red_20260720to0928"
    output_dir: str | None = None

    # Each entry is "model_name:pooling:layer_index" (or "pixels" for step pixels).
    visual_features: tuple[str, ...] = (
        ("pixel_values_rgb_step1:PC1000:0",)
        + _layer_specs("alexnet", range(7))
        + _layer_specs("dino_v3_h", (0, 4, 8, 12, 16, 20, 24, 28, 31))
        + _layer_specs("ijepa_vith14_1k", (0, 4, 8, 12, 16, 20, 24, 28, 31))
    )
    # Features whose fits are drawn in the per-curve fit figure.
    example_features: tuple[str, ...] = (
        "pixel_values_rgb_step1:PC1000:0",
        "alexnet:PC1000:2",
        "dino_v3_h:PC1000:16",
        "dino_v3_h:PC1000:31",
        "ijepa_vith14_1k:PC1000:31",
    )
    pixel_step: int = 30

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = None
    # Reliable channels keyed by the static session; None disables the filter.
    reliable_channels_config: str | None = str(PROJECT_ROOT / "reliable_channels.yaml")
    source_fs: float = 1000
    neural_fs: float = 100

    last_frame_time_ms: float = 2500
    neural_reference_window_ms: tuple[float, float] = (2500, 2600)
    # Samples in [start, end) are fitted as lags before last_frame_time_ms.
    fit_window_ms: tuple[float, float] = (300, 2500)
    rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "correlation"
    n_subsamples: int = 100
    subsample_fraction: float = 0.8
    max_coupling_lag_ms: float = 150
    random_seed: int = 0
    figure_dpi: int = 160
    show: bool = False
# EOF


"""
parse_args
Parse sessions, features, channels, timing, and fitting parameters.

OUTPUT:
    - cfg: Cfg -> analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the RDM similarity with the last-frame RDM over time in "
            "neural data and visual models, and rank candidate decay shapes."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument("--output_dir")
    parser.add_argument("--visual_features", nargs="+", default=list(Cfg.visual_features))
    parser.add_argument("--example_features", nargs="+", default=list(Cfg.example_features))
    parser.add_argument("--pixel_step", type=int, default=Cfg.pixel_step)
    parser.add_argument("--good_channels", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--reliable_channels_config", default=Cfg.reliable_channels_config)
    parser.add_argument("--no_reliable_channels", action="store_true")
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--neural_fs", type=float, default=Cfg.neural_fs)
    parser.add_argument("--last_frame_time_ms", type=float, default=Cfg.last_frame_time_ms)
    parser.add_argument(
        "--neural_reference_window_ms", nargs=2, type=float,
        default=list(Cfg.neural_reference_window_ms), metavar=("START", "END"),
    )
    parser.add_argument(
        "--fit_window_ms", nargs=2, type=float,
        default=list(Cfg.fit_window_ms), metavar=("START", "END"),
    )
    parser.add_argument("--rdm_metric", default=Cfg.rdm_metric)
    parser.add_argument("--rsa_metric", choices=("correlation", "spearman"), default=Cfg.rsa_metric)
    parser.add_argument("--n_subsamples", type=int, default=Cfg.n_subsamples)
    parser.add_argument("--subsample_fraction", type=float, default=Cfg.subsample_fraction)
    parser.add_argument("--max_coupling_lag_ms", type=float, default=Cfg.max_coupling_lag_ms)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    parser.add_argument("--figure_dpi", type=int, default=Cfg.figure_dpi)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=Cfg.show)
    args = parser.parse_args()

    if args.no_reliable_channels:
        args.reliable_channels_config = None
    # end if args.no_reliable_channels
    del args.no_reliable_channels
    if args.good_channels is not None:
        args.good_channels = tuple(args.good_channels)
    # end if args.good_channels
    for name in (
            "visual_features", "example_features",
            "neural_reference_window_ms", "fit_window_ms",
            ):
        setattr(args, name, tuple(getattr(args, name)))
    # end for name
    return Cfg(**vars(args))
# EOF


"""
feature_path_and_type
Translate one visual-feature spec into its HDF5 path and loader type.

INPUT:
    - feature_spec: str -> "pixels" or "model_name:pooling:layer_index"
    - models_dir: Path -> directory with extracted features
    - pixel_step: int -> pixel sampling step for "pixels"

OUTPUT:
    - feature_path: Path -> selected HDF5 file
    - feature_type: str -> pixels or model
"""
def feature_path_and_type(feature_spec, models_dir, pixel_step):
    if feature_spec == "pixels":
        return resolve_visual_feature_path(
            models_dir, "pixels", pixel_step=pixel_step,
        ), "pixels"
    # end if pixels
    model_name, pooling, layer_index = feature_spec.split(":")
    feature_path = resolve_visual_feature_path(
        models_dir, "model", model_name=model_name,
        model_pooling=pooling, model_layer_index=int(layer_index),
    )
    return feature_path, "model"
# EOF


"""
subsample_fit_summary
Refit every decay shape on each stimulus-subset curve.

INPUT:
    - lags_ms: np.ndarray -> fitted lags
    - subsample_values: np.ndarray -> subsets x lags similarity

OUTPUT:
    - best_counts: Counter -> how often each model has the lowest AIC
    - half_decay_lags: np.ndarray -> best-model half-decay lag per subset
"""
def subsample_fit_summary(lags_ms, subsample_values):
    best_counts = Counter()
    half_decay_lags = np.full(subsample_values.shape[0], np.nan)
    for subsample_index, values in enumerate(subsample_values):
        fits = fit_decay_models(lags_ms, values)
        best_model = min(fits, key=lambda name: fits[name]["aic"])
        best_counts[best_model] += 1
        half_decay_lags[subsample_index] = fits[best_model]["half_decay_lag_ms"]
    # end for subsample_index
    return best_counts, half_decay_lags
# EOF


"""
lagged_curve_correlation
Pearson-correlate the visual curve with the neural curve delayed by each lag,
using the same visual samples at every lag.

INPUT:
    - timecourses: dict -> output of compute_last_frame_rdm_timecourses
    - fit_window_ms: tuple[float, float] -> visual sample window
    - lags_ms: np.ndarray -> neural delays

OUTPUT:
    - correlations: np.ndarray -> one Pearson r per lag
"""
def lagged_curve_correlation(timecourses, fit_window_ms, lags_ms):
    # Visual sample times whose neural partner stays before the reference window.
    grid_ms = timecourses["neural_times_ms"][
        (timecourses["neural_times_ms"] >= fit_window_ms[0])
        & (timecourses["neural_times_ms"] + lags_ms.max() < fit_window_ms[1])
    ]
    visual = np.interp(grid_ms, timecourses["visual_times_ms"], timecourses["visual"])
    correlations = np.array([
        np.corrcoef(
            visual,
            np.interp(grid_ms + lag_ms, timecourses["neural_times_ms"], timecourses["neural"]),
        )[0, 1]
        for lag_ms in lags_ms
    ])
    return correlations
# EOF


def main():
    cfg = parse_args()
    data_dir = Path(paths["data_path"]) / "data"
    models_dir = Path(paths["data_path"]) / "models"
    static_path = data_dir / f"{cfg.static_exp_name}_natraster_img.mat"
    dynamic_path = data_dir / f"{cfg.dynamic_exp_name}_natraster_vid.mat"

    channel_subset = None
    if cfg.reliable_channels_config is not None:
        channel_subset = load_reliable_channels(
            cfg.reliable_channels_config, cfg.static_exp_name,
        )
    # end if reliable channels
    channel_name = "reliable_channels" if channel_subset is not None else "all_channels"
    if cfg.good_channels is not None:
        channel_name += f"_{cfg.good_channels[0]}to{cfg.good_channels[1]}"
    # end if cfg.good_channels
    reference_start, reference_end = cfg.neural_reference_window_ms
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "last_frame_rdm_decay"
        / f"{cfg.dynamic_exp_name}_{channel_name}"
        / f"ref{reference_start:.0f}to{reference_end:.0f}ms_{cfg.rdm_metric}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    coupling_lags_ms = np.arange(0, cfg.max_coupling_lag_ms + 5, 10)

    # curve name -> times, values, subset values, fits, subset summaries.
    curves = {}
    coupling_rows = []
    for feature_spec in cfg.visual_features:
        feature_path, feature_type = feature_path_and_type(
            feature_spec, models_dir, cfg.pixel_step,
        )
        data = load_last_frame_decay_data(
            static_path, dynamic_path, feature_path,
            visual_feature_type=feature_type,
            source_fs=cfg.source_fs, new_fs=cfg.neural_fs,
            good_channels=cfg.good_channels, channel_subset=channel_subset,
        )
        timecourses = compute_last_frame_rdm_timecourses(
            data,
            last_frame_time_ms=cfg.last_frame_time_ms,
            neural_reference_window_ms=cfg.neural_reference_window_ms,
            rdm_metric=cfg.rdm_metric, rsa_metric=cfg.rsa_metric,
            n_subsamples=cfg.n_subsamples,
            subsample_fraction=cfg.subsample_fraction,
            random_seed=cfg.random_seed,
        )
        # The neural curve does not depend on the visual feature: keep it once.
        sources = [(feature_spec, "visual", timecourses["visual_label"])]
        if "neural" not in curves:
            sources.insert(0, ("neural", "neural", f"Neural ({data['dynamic'].shape[0]} ch)"))
        # end if neural not yet stored
        for curve_name, key, label in sources:
            times_ms = timecourses[f"{key}_times_ms"]
            lags_ms, values = curve_lags(
                times_ms, timecourses[key], cfg.fit_window_ms, cfg.last_frame_time_ms,
            )
            _, subsample_values = curve_lags(
                times_ms, timecourses[f"{key}_subsamples"],
                cfg.fit_window_ms, cfg.last_frame_time_ms,
            )
            best_counts, subsample_half_lags = subsample_fit_summary(lags_ms, subsample_values)
            curves[curve_name] = {
                "label": label,
                "times_ms": times_ms,
                "timecourse": timecourses[key],
                "lags_ms": lags_ms,
                "values": values,
                "subsample_values": subsample_values,
                "fits": fit_decay_models(lags_ms, values),
                "best_counts": best_counts,
                "subsample_half_lags": subsample_half_lags,
            }
        # end for curve_name

        coupling = lagged_curve_correlation(timecourses, cfg.fit_window_ms, coupling_lags_ms)
        coupling_rows.append({
            "feature": feature_spec,
            "label": timecourses["visual_label"],
            "best_lag_ms": coupling_lags_ms[np.nanargmax(coupling)],
            "max_r": np.nanmax(coupling),
            "r_at_0ms": coupling[0],
        })
        print(f"{feature_spec}: done, curve r(neural, visual) max={np.nanmax(coupling):.3f}")
    # end for feature_spec

    # ---- Fit table: one row per curve and decay model.
    fit_rows = []
    for curve_name, curve in curves.items():
        for model_name, fit in curve["fits"].items():
            fit_rows.append({
                "curve": curve_name,
                "label": curve["label"],
                "model": model_name,
                "r_squared": fit["r_squared"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "akaike_weight": fit["akaike_weight"],
                "half_decay_lag_ms": fit["half_decay_lag_ms"],
                "subsample_best_fraction": (
                    curve["best_counts"][model_name] / cfg.n_subsamples
                ),
                **{
                    name: value for name, value in zip(
                        fit["parameter_names"], fit["parameters"],
                    )
                },
            })
        # end for model_name
    # end for curve_name
    # Shared columns first, then the union of model-specific parameter names.
    base_fields = list(fit_rows[0])[:9]
    fieldnames = base_fields + sorted(set().union(*fit_rows) - set(base_fields))
    with open(output_dir / "decay_fits.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fit_rows)
    # end with open
    with open(output_dir / "neural_visual_curve_coupling.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=coupling_rows[0])
        writer.writeheader()
        writer.writerows(coupling_rows)
    # end with open
    np.savez(
        output_dir / "rdm_timecourses.npz",
        **{
            f"{name}__{key}": curve[key]
            for name, curve in curves.items()
            for key in ("times_ms", "timecourse", "lags_ms", "subsample_values")
        },
    )

    # ---- Figure 1: raw and range-normalized curves over time.
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    model_families = sorted({name.split(":")[0] for name in curves if name != "neural"})
    colormaps = dict(zip(model_families, ("Greys", "Oranges", "Greens", "Purples", "Reds")))
    for curve_name, curve in curves.items():
        if curve_name == "neural":
            style = {"color": "tab:blue", "linewidth": 2.5, "zorder": 5}
        else:
            family = curve_name.split(":")[0]
            family_names = [name for name in curves if name.startswith(family + ":")]
            shade = 0.35 + 0.6 * family_names.index(curve_name) / max(len(family_names) - 1, 1)
            style = {"color": plt.get_cmap(colormaps[family])(shade), "linewidth": 1}
        # end if neural
        axes[0].plot(curve["times_ms"], curve["timecourse"], **style)
        # Rescale within the fit window so shapes are comparable across curves.
        values = curve["values"]
        normalized = (values - values.min()) / (values.max() - values.min())
        axes[1].plot(curve["lags_ms"], normalized, label=curve["label"], **style)
    # end for curve_name
    axes[0].axvspan(*cfg.neural_reference_window_ms, color="0.85")
    axes[0].axvline(cfg.fit_window_ms[0], color="0.5", linestyle=":")
    axes[0].set(
        xlabel="Time from movie onset (ms)", ylabel="RDM r with last-frame RDM",
        title="RDM similarity to the last frame",
    )
    axes[1].set(
        xlabel="Lag before last frame (ms)", ylabel="Min–max normalized r",
        title="Shapes within the fit window", xscale="log",
    )
    axes[1].legend(frameon=False, fontsize=5, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    # end for axis
    figure.suptitle(cfg.dynamic_exp_name)
    figure.tight_layout()
    figure.savefig(output_dir / "rdm_timecourses.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    # ---- Figure 2: decay fits for the neural curve and example features.
    example_names = ["neural"] + [name for name in cfg.example_features if name in curves]
    figure, axes = plt.subplots(1, len(example_names), figsize=(4.6 * len(example_names), 4.2))
    for axis, curve_name in zip(np.atleast_1d(axes), example_names):
        curve = curves[curve_name]
        plot_rdm_decay_fits(
            axis, curve["lags_ms"], curve["values"], curve["subsample_values"],
            curve["fits"], curve["label"],
        )
    # end for axis
    figure.tight_layout()
    figure.savefig(output_dir / "rdm_decay_fits.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    # ---- Figure 3: model ranking and half-decay lags for every curve.
    curve_names = list(curves)
    model_names = list(DECAY_MODELS)
    weights = np.array([
        [curves[name]["fits"].get(model, {}).get("akaike_weight", np.nan) for model in model_names]
        for name in curve_names
    ])
    subsample_best = np.array([
        [curves[name]["best_counts"][model] / cfg.n_subsamples for model in model_names]
        for name in curve_names
    ])
    figure, axes = plt.subplots(
        1, 3, figsize=(17, 0.28 * len(curve_names) + 2),
        gridspec_kw={"width_ratios": (1, 1, 1.2)},
    )
    for axis, matrix, title in (
            (axes[0], weights, "Akaike weight (full data)"),
            (axes[1], subsample_best, "Best-AIC fraction over stimulus subsets"),
            ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(model_names)), model_names, rotation=45, ha="right", fontsize=7)
        axis.set_yticks(range(len(curve_names)), [curves[name]["label"] for name in curve_names], fontsize=6)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.04)
    # end for axis
    half_lags = np.array([curves[name]["subsample_half_lags"] for name in curve_names])
    medians = np.nanmedian(half_lags, axis=1)
    lower, upper = np.nanpercentile(half_lags, [2.5, 97.5], axis=1)
    positions = np.arange(len(curve_names))
    axes[2].errorbar(
        medians, positions, xerr=(medians - lower, upper - medians),
        fmt="o", color="k", markersize=3,
    )
    axes[2].axvspan(lower[0], upper[0], color="tab:blue", alpha=0.2, label="Neural 95%")
    axes[2].set_yticks(positions, [curves[name]["label"] for name in curve_names], fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set(
        xlabel="Half-decay lag of best model (ms)", xscale="log",
        title="Half-decay lag (median, 95% over subsets)",
    )
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].grid(alpha=0.2)
    figure.suptitle(cfg.dynamic_exp_name)
    figure.tight_layout()
    figure.savefig(output_dir / "decay_model_comparison.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    for curve_name in curve_names:
        curve = curves[curve_name]
        best_model = min(curve["fits"], key=lambda name: curve["fits"][name]["aic"])
        print(
            f"{curve['label']}: best={best_model} "
            f"(w={curve['fits'][best_model]['akaike_weight']:.2f}, "
            f"subsets={curve['best_counts'].most_common(2)}), "
            f"half-decay={np.nanmedian(curve['subsample_half_lags']):.0f} ms"
        )
    # end for curve_name
    print(f"Saved to {output_dir}")
    if cfg.show:
        plt.show()
    # end if cfg.show
# EOF


if __name__ == "__main__":
    main()
# EOC
