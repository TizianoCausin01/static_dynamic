import argparse
from dataclasses import asdict, dataclass, field
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


# Fig 3 temporal scores of all_monkeys_presentation.ipynb (static, first frame,
# last frame, last frame minus the 2250-ms model RDM; centroid and peak) for
# several models and feature poolings at once. Example from the project root:
# .venv/bin/python python_scripts/scripts/run_layer_temporal_scores.py \
#     --model_names alexnet --model_poolings PC1000 PCimagenet1000
# Joint 2000 + 2250 ms regress-out (in-fold PCA + ridge on the model features):
#     ... --previous_frames_ms 2000 2250 --regression_level features_ridge \
#     --residual_latency_window_ms 2000 2700 --output_tag ridge2000+2250
# The same frames regressed out at the RDM level (one OLS over stimulus pairs):
#     ... --previous_frames_ms 2000 2250 --regression_level rdm \
#     --residual_latency_window_ms 2000 2700 --output_tag rdm2000+2250
# Same OLS on rank-transformed RDMs (matches the Spearman RSA): --regression_level rdm_rank
# The top 50 PCs of the pooled 2000 + 2250 ms activations removed from the
# 2500 ms activations (feature-space projection, no RDM regression):
#     ... --previous_frames_ms 2000 2250 --regression_level features_pca \
#     --pca_n_components 50 --residual_latency_window_ms 2000 2700 --output_tag pca50
# Ridge on 150 shared PCs of the 2000 / 2250 / 2500 ms activations:
#     ... --previous_frames_ms 2000 2250 --regression_level features_ridge \
#     --feature_pca_components 150 --ridge_pca_variance none ...
# Out-of-fold OLS on the full features or on shared PCs (min-norm when p > n):
#     ... --regression_level features_ols --ridge_pca_variance none [--feature_pca_components 50]
# Autocorrelation control: the slice RDMs against the layer's own model RDM
# timecourse instead of the neural RDMs (same regression, windows and scores):
#     ... --previous_frame_range_ms 2000 2250 --last_frame_latency_window_ms 2000 2700 \
#     --similarity_target model --output_tag modelself_range2000-2250_window2000
# Linear encoding instead of RSA (5-fold ridge from the model features to the
# neural channel pattern at every time bin, channel-averaged R2), residual in
# feature space:
#     ... --similarity_method encoding --previous_frames_ms 2000 2250 \
#     --regression_level features_pca --pca_n_components 50 \
#     --residual_latency_window_ms 2000 2700 --output_tag encoding_pca50_2000+2250_window2000


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils import (
    compute_rdm_timeseries, depth_latency_score, load_natraster,
    match_timed_static_movie_rasters, model_frame_similarity_by_layer,
    summarize_layer_timing,
)
from project_specific_utils.last_frame_decay import resample_array


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

CONDITIONS = ("static", "first_frame", "last_frame", "last_frame_residual")


@dataclass
class Cfg:
    monkeys: list[str] = field(default_factory=lambda: list(MONKEY_SESSIONS))
    model_names: list[str] = field(default_factory=lambda: ["alexnet"])
    model_poolings: list[str] = field(
        default_factory=lambda: ["PC1000", "PCimagenet1000"]
    )
    model_dataset_name: str = "static_dynamic"

    # Neural and RSA parameters of the presentation notebook (Fig 3).
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float = 1000
    neural_rdm_metric: str = "cosine_cnt"
    model_rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "spearman"
    first_frame_ms: float = 0
    # Model frame(s) regressed out of the last frame: "rdm" and "features" take
    # one frame, "features_ridge" stacks several frames into one predictor.
    previous_frames_ms: list[float] = field(default_factory=lambda: [2250])
    # (start, end) model time; when set, every native model frame in it is
    # regressed out together and previous_frames_ms is ignored.
    previous_frame_range_ms: tuple[float, float] | None = None
    last_frame_onset_ms: float = 2500
    residual_fit_intercept: bool = True
    regression_level: str = "rdm"
    # features_ridge: in-fold PCA keeping this predictor variance, then RidgeCV
    # with out-of-fold ("kf") residuals. Model features need larger penalties
    # than the neural regress-out (the 1e5 cap was always selected), so the
    # grid reaches 1e10.
    ridge_pca_variance: float = 0.95
    ridge_alphas: tuple[float, ...] = tuple(10.0 ** np.arange(-2, 11))
    ridge_cv_type: str = "kf"
    ridge_n_splits: int = 5
    # features_pca: previous-frame PCs (pooled over stimuli and frames)
    # projected out of the last-frame activations.
    pca_n_components: int = 50
    # Feature-level regressions: project last + previous frames on this many
    # shared PCs first (None = original features).
    feature_pca_components: int | None = None
    centroid_smoothing: float = 0
    peak_smoothing: float = 3
    static_latency_window_ms: tuple[float, float] = (0, 250)
    first_frame_latency_window_ms: tuple[float, float] = (0, 250)
    last_frame_latency_window_ms: tuple[float, float] = (2300, 2700)
    # Residual-condition window; None reuses last_frame_latency_window_ms.
    # Widened to (2000, 2700) when the 2000 + 2250 ms frames are regressed out.
    residual_latency_window_ms: tuple[float, float] | None = None
    relative_similarity_cutoffs: bool = False
    centroid_similarity_cutoff: float = 0.01
    peak_similarity_cutoff: float = 0.01

    # "neural" = Fig 3; "model" = slice RDMs vs the same layer's model RDM
    # timecourse (autocorrelation control, no "static" condition).
    similarity_target: str = "neural"
    # "rsa" = model vs neural RDMs; "encoding" = cross-validated ridge from the
    # model features to the neural channel pattern at every time bin.
    similarity_method: str = "rsa"
    encoding_alphas: tuple[float, ...] = tuple(10.0 ** np.arange(-2, 11))
    encoding_n_splits: int = 5
    encoding_score_type: str = "r2"
    # One ridge penalty per channel instead of one shared by all channels.
    encoding_alpha_per_target: bool = False

    output_dir: str | None = None
    # Appended to the output file names, so residual variants do not overwrite.
    output_tag: str = ""
# EOF


"""
parse_args
Parse the monkeys, models and poolings to score.

OUTPUT:
    - cfg: Cfg -> configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description="Fig 3 depth-latency temporal scores per model and pooling."
    )
    parser.add_argument("--monkeys", nargs="+", choices=tuple(MONKEY_SESSIONS))
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument("--model_poolings", nargs="+")
    parser.add_argument("--output_dir")
    parser.add_argument("--previous_frames_ms", nargs="+", type=float)
    parser.add_argument("--previous_frame_range_ms", nargs=2, type=float)
    parser.add_argument(
        "--regression_level",
        choices=("rdm", "rdm_rank", "features", "features_ridge", "features_ols", "features_pca"),
    )
    parser.add_argument("--pca_n_components", type=int)
    # "none" disables the in-fold PCA of the ridge predictor.
    parser.add_argument(
        "--ridge_pca_variance",
        type=lambda value: None if value.lower() == "none" else float(value),
    )
    parser.add_argument("--feature_pca_components", type=int)
    parser.add_argument("--last_frame_latency_window_ms", nargs=2, type=float)
    parser.add_argument("--residual_latency_window_ms", nargs=2, type=float)
    parser.add_argument("--similarity_target", choices=("neural", "model"))
    parser.add_argument("--similarity_method", choices=("rsa", "encoding"))
    parser.add_argument("--encoding_score_type", choices=("r2", "r2_pooled", "corr_pooled", "corr"))
    parser.add_argument("--encoding_alpha_per_target", action="store_true", default=None)
    parser.add_argument("--output_tag")
    args = parser.parse_args()

    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    cfg.output_dir = str(Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "layer_temporal_scores"
    ))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg
# EOF


"""
load_neural_rdms
Static (last-frame) and movie neural RDM timecourses of one monkey, with the
movie names in the matched stimulus order. With cfg.similarity_method
"encoding" the resampled rasters are returned instead of their RDMs.

INPUT:
    - monkey: str -> key of MONKEY_SESSIONS
    - cfg: Cfg -> configuration

OUTPUT:
    - static_neural_rdms: np.ndarray -> static time x stimulus pairs
        (encoding: channels x static time x stimuli)
    - dynamic_neural_rdms: np.ndarray -> movie time x stimulus pairs
        (encoding: channels x movie time x stimuli)
    - feature_names: list[str] -> movie names in neural stimulus order
"""
def load_neural_rdms(monkey: str, cfg: Cfg):
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
    dynamic_rasters, dynamic_names, dynamic_channels = load_natraster(
        data_dir / f"{dynamic_name}_natraster_vid.mat", **channel_kwargs,
    )
    if not np.array_equal(static_channels, dynamic_channels):
        raise ValueError(f"{monkey}: sessions retained different channels.")
    # end if channel numbers differ
    (
        last_frame_rasters, video_rasters, _, _, shared_stimuli, aligned_names,
    ) = match_timed_static_movie_rasters(
        static_rasters, static_names, dynamic_rasters, dynamic_names,
    )
    # channels x time x stimuli; the static response is cropped, then both
    # are resampled to the analysis rate.
    static_crop_samples = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
    last_frame_rasters = resample_array(
        last_frame_rasters[:, :static_crop_samples, :], cfg.source_fs, cfg.new_fs,
    )
    video_rasters = resample_array(video_rasters, cfg.source_fs, cfg.new_fs)
    print(
        f"{monkey}: {len(static_channels)} channels | {len(shared_stimuli)} stimuli",
        flush=True,
    )
    feature_names = [Path(name).name for name in aligned_names["movie"]]
    if cfg.similarity_method == "encoding":
        return last_frame_rasters, video_rasters, feature_names
    # end if encoding keeps the rasters
    return (
        compute_rdm_timeseries(last_frame_rasters, cfg.neural_rdm_metric),
        compute_rdm_timeseries(video_rasters, cfg.neural_rdm_metric),
        feature_names,
    )
# EOF


"""
score_model
Centroid and peak temporal scores of every Fig 3 condition for one model.

INPUT:
    - neural_rdms: tuple -> output of load_neural_rdms
    - model_name: str -> registry key
    - model_pooling: str -> pooling tag of the feature files
    - cfg: Cfg -> configuration

OUTPUT:
    - rows: list[dict] -> one row per condition and timing measure
    - layer_similarity: dict[str, np.ndarray] -> layers x neural time RSA
    - variance_removed: np.ndarray -> per-layer variance removed by the residual
"""
def score_model(neural_rdms, model_name, model_pooling, cfg):
    static_neural_rdms, dynamic_neural_rdms, feature_names = neural_rdms
    layer_names, layer_similarity, variance_removed = model_frame_similarity_by_layer(
        static_neural_rdms, dynamic_neural_rdms, feature_names,
        Path(paths["data_path"]) / "models", model_name,
        cfg.model_dataset_name, model_pooling, cfg.new_fs,
        cfg.first_frame_ms, tuple(cfg.previous_frames_ms), cfg.last_frame_onset_ms,
        cfg.model_rdm_metric, cfg.rsa_metric, cfg.residual_fit_intercept,
        regression_level=cfg.regression_level,
        ridge_pca_variance=cfg.ridge_pca_variance, ridge_alphas=cfg.ridge_alphas,
        ridge_cv_type=cfg.ridge_cv_type, ridge_n_splits=cfg.ridge_n_splits,
        pca_n_components=cfg.pca_n_components,
        feature_pca_components=cfg.feature_pca_components,
        previous_frame_range_ms=cfg.previous_frame_range_ms,
        similarity_target=cfg.similarity_target,
        similarity_method=cfg.similarity_method,
        encoding_alphas=cfg.encoding_alphas,
        encoding_n_splits=cfg.encoding_n_splits,
        encoding_score_type=cfg.encoding_score_type,
        encoding_alpha_per_target=cfg.encoding_alpha_per_target,
    )
    layer_depths = np.linspace(0, 1, len(layer_names))
    # Onset and latency window of each condition, as in the notebook.
    condition_timing = {
        "static": (0, cfg.static_latency_window_ms),
        "first_frame": (cfg.first_frame_ms, cfg.first_frame_latency_window_ms),
        "last_frame": (cfg.last_frame_onset_ms, cfg.last_frame_latency_window_ms),
        "last_frame_residual": (
            cfg.last_frame_onset_ms,
            tuple(cfg.residual_latency_window_ms or cfg.last_frame_latency_window_ms),
        ),
    }
    rows = []
    # The model target has no "static" condition.
    for condition_name in [name for name in CONDITIONS if name in layer_similarity]:
        onset_ms, window_ms = condition_timing[condition_name]
        similarity = layer_similarity[condition_name]
        time_ms = np.arange(similarity.shape[1]) * 1000 / cfg.new_fs
        _, centroid_latency_ms, peak_latency_ms, _ = summarize_layer_timing(
            similarity, time_ms, onset_ms, window_ms, window_ms,
            cfg.centroid_smoothing, cfg.centroid_similarity_cutoff,
            cfg.peak_similarity_cutoff, peak_smoothing_sigma=cfg.peak_smoothing,
            relative_cutoffs=cfg.relative_similarity_cutoffs,
        )
        for timing_name, latency_ms in (
                ("centroid", centroid_latency_ms), ("peak", peak_latency_ms)):
            score = depth_latency_score(layer_depths, latency_ms)
            rows.append({
                "model_name": model_name, "model_pooling": model_pooling,
                "condition": condition_name, "timing": timing_name, **score,
            })
        # end for timing_name, latency_ms
    # end for condition_name
    return rows, layer_similarity, variance_removed
# EOF


"""
main
Score every monkey x model x pooling and save a CSV, a JSON summary and the
layer similarity curves (npz, keys "monkey|model|pooling|condition").
"""
def main() -> None:
    cfg = parse_args()
    all_rows, all_curves = [], {}
    for monkey in cfg.monkeys:
        neural_rdms = load_neural_rdms(monkey, cfg)
        for model_name in cfg.model_names:
            for model_pooling in cfg.model_poolings:
                try:
                    rows, layer_similarity, variance_removed = score_model(
                        neural_rdms, model_name, model_pooling, cfg,
                    )
                except FileNotFoundError as error:
                    print(f"  skipping {model_name} {model_pooling}: {error}", flush=True)
                    continue
                # end try
                for row in rows:
                    row["monkey"] = monkey
                    print(
                        f"  {model_name:<16} {model_pooling:<15} "
                        f"{row['condition']:<20} {row['timing']:<8} "
                        f"rho={row['rho']:+.2f} p={row['pvalue']:.2g} "
                        f"n={row['n_layers']}",
                        flush=True,
                    )
                # end for row
                all_rows.extend(rows)
                curve_key = f"{monkey}|{model_name}|{model_pooling}"
                for condition_name, curves in layer_similarity.items():
                    all_curves[f"{curve_key}|{condition_name}"] = curves
                # end for condition_name
                all_curves[f"{curve_key}|variance_removed"] = variance_removed
                print(
                    f"  variance removed per layer: {np.round(variance_removed, 2)}",
                    flush=True,
                )
            # end for model_pooling
        # end for model_name
    # end for monkey

    file_stem = "_".join(cfg.model_names) + (f"_{cfg.output_tag}" if cfg.output_tag else "")
    csv_path = Path(cfg.output_dir) / f"{file_stem}_temporal_scores.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # end with open
    with open(Path(cfg.output_dir) / f"{file_stem}_config.json", "w") as config_file:
        json.dump(
            {"config": asdict(cfg), "sessions": MONKEY_SESSIONS},
            config_file, indent=2,
        )
    # end with open
    np.savez(Path(cfg.output_dir) / f"{file_stem}_curves.npz", **all_curves)
    print(f"Saved {csv_path}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
