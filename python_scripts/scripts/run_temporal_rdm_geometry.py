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
import h5py
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

from image_processing.video_feature_extraction import (
    list_video_feature_files,
    load_aligned_video_features,
    match_feature_stimulus_names,
)
from project_specific_utils import load_reliable_channels
from project_specific_utils.last_frame_decay import load_last_frame_decay_data
from project_specific_utils.temporal_rdm_geometry import (
    compare_temporal_matrices,
    square_rdm_stack,
    subset_rdm_timeseries,
    temporal_rdm_matrix,
    window_rdm_timeseries,
)


@dataclass
class Cfg:
    static_exp_name: str = "red_20260726to0925"
    dynamic_exp_name: str = "red_20260720to0928"
    output_dir: str | None = None

    # "model_name:pooling" uses every extracted layer of that model.
    models: tuple[str, ...] = (
        "pixel_values_rgb_step1:PC1000",
        "alexnet:PC1000",
        "dino_v3_h:PC1000",
        "ijepa_vith14_1k:PC1000",
    )

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = None
    # Reliable channels keyed by the static session; None disables the filter.
    reliable_channels_config: str | None = str(PROJECT_ROOT / "reliable_channels.yaml")
    source_fs: float = 1000
    # 60 Hz bins coincide with the 60-fps movie frames, so lags are whole frames.
    neural_fs: float = 60

    # Movie time window; the neural window is the same window shifted by the lag.
    window_ms: tuple[float, float] = (2000, 2500)
    # Plausible response latencies; longer lags reach the held-last-frame response.
    max_lag_ms: float = 150
    rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "correlation"
    n_subsamples: int = 100
    subsample_fraction: float = 0.8
    random_seed: int = 0
    figure_dpi: int = 160
    show: bool = False
# EOF


"""
parse_args
Parse sessions, models, channels, window, and lag parameters.

OUTPUT:
    - cfg: Cfg -> analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the time x time geometry of neural and model RDM "
            "timeseries over a movie window, allowing for a neural lag."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument("--output_dir")
    parser.add_argument("--models", nargs="+", default=list(Cfg.models))
    parser.add_argument("--good_channels", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--reliable_channels_config", default=Cfg.reliable_channels_config)
    parser.add_argument("--no_reliable_channels", action="store_true")
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--neural_fs", type=float, default=Cfg.neural_fs)
    parser.add_argument(
        "--window_ms", nargs=2, type=float, default=list(Cfg.window_ms),
        metavar=("START", "END"),
    )
    parser.add_argument("--max_lag_ms", type=float, default=Cfg.max_lag_ms)
    parser.add_argument("--rdm_metric", default=Cfg.rdm_metric)
    parser.add_argument("--rsa_metric", choices=("correlation", "spearman"), default=Cfg.rsa_metric)
    parser.add_argument("--n_subsamples", type=int, default=Cfg.n_subsamples)
    parser.add_argument("--subsample_fraction", type=float, default=Cfg.subsample_fraction)
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
    args.models = tuple(args.models)
    args.window_ms = tuple(args.window_ms)
    return Cfg(**vars(args))
# EOF


"""
load_layer_features
Load one layer's video features aligned to the neural stimulus order.

INPUT:
    - feature_path: Path -> extracted-feature HDF5 file
    - stimuli: np.ndarray -> aligned stimulus identities

OUTPUT:
    - features: np.ndarray -> features x frames x stimuli
    - fps: float -> feature sampling rate
    - label: str -> model and layer name
"""
def load_layer_features(feature_path, stimuli):
    requested_names = [f"vid_{stimulus}.mp4" for stimulus in stimuli]
    feature_names = match_feature_stimulus_names(feature_path, requested_names)
    features, layer_name, source_fps = load_aligned_video_features(
        feature_path, feature_names, frame_index=None,
    )
    with h5py.File(feature_path, "r") as feature_file:
        model_name = str(feature_file.attrs.get("model_name", feature_path.stem))
        frame_stride = int(feature_file.attrs.get("frame_stride", 1))
    # end with h5py.File
    return features, source_fps / frame_stride, f"{model_name} {layer_name}"
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
    output_dir = Path(
        cfg.output_dir
        or PROJECT_ROOT / "results" / "temporal_rdm_geometry"
        / f"{cfg.dynamic_exp_name}_{channel_name}"
        / f"window{cfg.window_ms[0]:.0f}to{cfg.window_ms[1]:.0f}ms_{cfg.rdm_metric}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every layer of every requested model, in natural depth order.
    layer_paths = {}
    for model_spec in cfg.models:
        model_name, pooling = model_spec.split(":")
        layer_paths[model_name] = list_video_feature_files(
            models_dir, model_name, "static_dynamic", pooling,
        )
    # end for model_spec

    # Neural data once; the loader needs some visual file, which is ignored here.
    first_path = next(iter(layer_paths.values()))[0]
    data = load_last_frame_decay_data(
        static_path, dynamic_path, first_path, visual_feature_type="model",
        source_fs=cfg.source_fs, new_fs=cfg.neural_fs,
        good_channels=cfg.good_channels, channel_subset=channel_subset,
    )
    n_stimuli = data["stimuli"].size
    neural_times_ms = np.arange(data["dynamic"].shape[1]) * 1000 / cfg.neural_fs
    sample_ms = 1000 / cfg.neural_fs
    lags_ms = np.arange(0, cfg.max_lag_ms + sample_ms / 2, sample_ms)

    # Stimulus subsets shared by the neural and every model geometry.
    generator = np.random.default_rng(cfg.random_seed)
    n_keep = int(round(cfg.subsample_fraction * n_stimuli))
    subsets = [
        np.sort(generator.choice(n_stimuli, n_keep, replace=False))
        for _ in range(cfg.n_subsamples)
    ]

    # Neural time x time geometry for every lag, on all stimuli and on subsets.
    neural_matrices = []
    neural_subset_matrices = []
    for lag_ms in lags_ms:
        neural_rdms, _ = window_rdm_timeseries(
            data["dynamic"], neural_times_ms,
            (cfg.window_ms[0] + lag_ms, cfg.window_ms[1] + lag_ms), cfg.rdm_metric,
        )
        neural_matrices.append(temporal_rdm_matrix(neural_rdms, cfg.rsa_metric))
        square_rdms = square_rdm_stack(neural_rdms)
        neural_subset_matrices.append([
            temporal_rdm_matrix(subset_rdm_timeseries(square_rdms, kept), cfg.rsa_metric)
            for kept in subsets
        ])
    # end for lag_ms
    n_times = neural_matrices[0].shape[0]

    # Reference geometry: RDM similarity that falls linearly with |t1 - t2|.
    time_indices = np.arange(n_times)
    distance_matrix = -np.abs(time_indices[:, np.newaxis] - time_indices[np.newaxis, :])
    distance_r = np.array([
        compare_temporal_matrices(matrix, distance_matrix)[0] for matrix in neural_matrices
    ])

    rows = []
    model_matrices = {}
    for model_name, feature_paths in layer_paths.items():
        for layer_index, feature_path in enumerate(feature_paths):
            features, fps, label = load_layer_features(feature_path, data["stimuli"])
            visual_times_ms = np.arange(features.shape[1]) * 1000 / fps
            model_rdms, model_times_ms = window_rdm_timeseries(
                features, visual_times_ms, cfg.window_ms, cfg.rdm_metric,
            )
            if model_times_ms.size != n_times:
                raise ValueError(
                    f"{label}: {model_times_ms.size} frames vs {n_times} neural samples; "
                    "set --neural_fs to the movie frame rate."
                )
            # end if sample counts differ
            model_matrix = temporal_rdm_matrix(model_rdms, cfg.rsa_metric)
            model_matrices[label] = model_matrix
            square_rdms = square_rdm_stack(model_rdms)
            model_subset_matrices = [
                temporal_rdm_matrix(subset_rdm_timeseries(square_rdms, kept), cfg.rsa_metric)
                for kept in subsets
            ]

            # lags x (r, partial r) on all stimuli; lags x subsets on subsets.
            full = np.array([
                compare_temporal_matrices(neural_matrix, model_matrix)
                for neural_matrix in neural_matrices
            ])
            subset_r = np.array([
                [
                    compare_temporal_matrices(neural_subset, model_subset)
                    for neural_subset, model_subset in zip(lag_subsets, model_subset_matrices)
                ]
                for lag_subsets in neural_subset_matrices
            ])
            best_lag_index = int(np.nanargmax(full[:, 0]))
            best_partial_index = int(np.nanargmax(full[:, 1]))
            rows.append({
                "model": model_name,
                "layer_index": layer_index,
                "relative_depth": layer_index / max(len(feature_paths) - 1, 1),
                "label": label,
                "best_lag_ms": lags_ms[best_lag_index],
                "r": full[best_lag_index, 0],
                "r_ci_low": np.percentile(subset_r[best_lag_index, :, 0], 2.5),
                "r_ci_high": np.percentile(subset_r[best_lag_index, :, 0], 97.5),
                "best_partial_lag_ms": lags_ms[best_partial_index],
                "partial_r": full[best_partial_index, 1],
                "partial_r_ci_low": np.percentile(subset_r[best_partial_index, :, 1], 2.5),
                "partial_r_ci_high": np.percentile(subset_r[best_partial_index, :, 1], 97.5),
                "r_by_lag": full[:, 0],
                "partial_r_by_lag": full[:, 1],
            })
            print(
                f"{label}: r={rows[-1]['r']:.3f} @ {rows[-1]['best_lag_ms']:.0f} ms, "
                f"partial={rows[-1]['partial_r']:.3f} @ {rows[-1]['best_partial_lag_ms']:.0f} ms"
            )
        # end for layer_index
    # end for model_name

    # ---- Table (lag profiles go to the npz).
    table_fields = [key for key in rows[0] if not key.endswith("_by_lag")]
    with open(output_dir / "layer_fits.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=table_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # end with open
    np.savez(
        output_dir / "temporal_rdm_geometry.npz",
        lags_ms=lags_ms,
        labels=np.array([row["label"] for row in rows]),
        r_by_lag=np.array([row["r_by_lag"] for row in rows]),
        partial_r_by_lag=np.array([row["partial_r_by_lag"] for row in rows]),
        distance_r=distance_r,
        neural_matrices=np.array(neural_matrices),
    )

    # ---- Figure 1: best-lag fit per layer against relative depth.
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for model_name in layer_paths:
        model_rows = [row for row in rows if row["model"] == model_name]
        depth = [row["relative_depth"] for row in model_rows]
        for axis, key in ((axes[0], "r"), (axes[1], "partial_r")):
            values = np.array([row[key] for row in model_rows])
            lower = np.array([row[f"{key}_ci_low"] for row in model_rows])
            upper = np.array([row[f"{key}_ci_high"] for row in model_rows])
            line = axis.plot(depth, values, marker="o", markersize=3, label=model_name)[0]
            axis.fill_between(depth, lower, upper, color=line.get_color(), alpha=0.2)
        # end for axis, key
        axes[2].plot(
            depth, [row["best_lag_ms"] for row in model_rows],
            marker="o", markersize=3, label=model_name,
        )
    # end for model_name
    axes[0].axhline(distance_r.max(), color="k", linestyle="--", label="|Δt| only")
    axes[0].set(xlabel="Relative layer depth", ylabel="Spearman r (best lag)",
                title="Neural vs model time×time RDM geometry")
    axes[1].set(xlabel="Relative layer depth", ylabel="Partial Spearman r (best lag)",
                title="Controlling for temporal distance |Δt|")
    axes[2].set(xlabel="Relative layer depth", ylabel="Best neural lag (ms)",
                title="Lag maximizing r")
    for axis in axes:
        axis.grid(alpha=0.2)
    # end for axis
    axes[0].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"{cfg.dynamic_exp_name} | {data['dynamic'].shape[0]} channels | "
        f"window {cfg.window_ms[0]:.0f}–{cfg.window_ms[1]:.0f} ms | "
        f"95% over {cfg.n_subsamples} stimulus subsets"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "layer_profiles.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    # ---- Figure 2: r by lag for every layer.
    figure, axes = plt.subplots(1, 2, figsize=(14, 0.16 * len(rows) + 2))
    for axis, key, title in (
            (axes[0], "r_by_lag", "Spearman r"),
            (axes[1], "partial_r_by_lag", "Partial r (|Δt| removed)"),
            ):
        matrix = np.array([row[key] for row in rows])
        image = axis.imshow(
            matrix, aspect="auto", cmap="viridis",
            extent=(lags_ms[0] - sample_ms / 2, lags_ms[-1] + sample_ms / 2, len(rows) - 0.5, -0.5),
        )
        axis.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=4)
        axis.set(xlabel="Neural lag (ms)", title=title)
        figure.colorbar(image, ax=axis, fraction=0.04)
    # end for axis
    figure.tight_layout()
    figure.savefig(output_dir / "lag_by_layer.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    # ---- Figure 3: neural geometry next to the best layer of every model.
    best_rows = [
        max((row for row in rows if row["model"] == model_name), key=lambda row: row["r"])
        for model_name in layer_paths
    ]
    overall_best = max(best_rows, key=lambda row: row["r"])
    best_lag_index = int(np.argmin(np.abs(lags_ms - overall_best["best_lag_ms"])))
    panels = [(f"Neural (lag {lags_ms[best_lag_index]:.0f} ms)", neural_matrices[best_lag_index])]
    panels += [
        (f"{row['label']}\nr={row['r']:.2f}, partial={row['partial_r']:.2f}", model_matrices[row["label"]])
        for row in best_rows
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6))
    extent = (cfg.window_ms[0], cfg.window_ms[1], cfg.window_ms[1], cfg.window_ms[0])
    for axis, (title, matrix) in zip(axes, panels):
        # Diagonal is 1 by construction; scale colors on the off-diagonal entries.
        off_diagonal = matrix[~np.eye(n_times, dtype=bool)]
        image = axis.imshow(
            matrix, cmap="viridis", extent=extent,
            vmin=np.percentile(off_diagonal, 1), vmax=np.percentile(off_diagonal, 99),
        )
        axis.set(title=title, xlabel="Movie time (ms)")
        axis.title.set_fontsize(8)
        figure.colorbar(image, ax=axis, fraction=0.046)
    # end for axis
    axes[0].set_ylabel("Movie time (ms)")
    figure.tight_layout()
    figure.savefig(output_dir / "best_layer_matrices.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    print(f"|Δt|-only reference r (best lag) = {distance_r.max():.3f}")
    for row in sorted(rows, key=lambda row: row["r"], reverse=True)[:5]:
        print(f"TOP r: {row['label']} r={row['r']:.3f} [{row['r_ci_low']:.3f}, {row['r_ci_high']:.3f}] lag={row['best_lag_ms']:.0f}")
    # end for row
    for row in sorted(rows, key=lambda row: row["partial_r"], reverse=True)[:5]:
        print(f"TOP partial: {row['label']} partial={row['partial_r']:.3f} [{row['partial_r_ci_low']:.3f}, {row['partial_r_ci_high']:.3f}] lag={row['best_partial_lag_ms']:.0f}")
    # end for row
    print(f"Saved to {output_dir}")
    if cfg.show:
        plt.show()
    # end if cfg.show
# EOF


if __name__ == "__main__":
    main()
# EOC
