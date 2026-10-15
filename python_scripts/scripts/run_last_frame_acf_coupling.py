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

from project_specific_utils import load_reliable_channels
from project_specific_utils.last_frame_decay import (
    compute_last_frame_acf_coupling,
    load_last_frame_decay_data,
    plot_last_frame_acf_coupling,
    resolve_visual_feature_path,
)


@dataclass
class Cfg:
    static_exp_name: str = "red_20260726to0925"
    dynamic_exp_name: str = "red_20260720to0928"
    output_dir: str | None = None

    # Each entry is "pixels" or "model_name:pooling:layer_index".
    visual_features: tuple[str, ...] = (
        "pixels",
        "dino_v3_h:PC1000:0",
        "dino_v3_h:PC1000:10",
        "dino_v3_h:PC1000:20",
        "dino_v3_h:PC1000:31",
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
    visual_window_ms: tuple[float, float] = (200, 2500)
    max_lag_ms: float = 300
    lag_step_ms: float = 10
    n_permutations: int = 1000
    random_seed: int = 0
    figure_dpi: int = 160
    show: bool = False
# EOF


"""
parse_args
Parse sessions, visual features, channels, and timing parameters.

OUTPUT:
    - cfg: Cfg -> analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate pixel/model autocorrelation to the last movie frame with "
            "the neural autocorrelation to the last-frame response."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument("--output_dir")
    parser.add_argument("--visual_features", nargs="+", default=list(Cfg.visual_features))
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
        "--visual_window_ms", nargs=2, type=float,
        default=list(Cfg.visual_window_ms), metavar=("START", "END"),
    )
    parser.add_argument("--max_lag_ms", type=float, default=Cfg.max_lag_ms)
    parser.add_argument("--lag_step_ms", type=float, default=Cfg.lag_step_ms)
    parser.add_argument("--n_permutations", type=int, default=Cfg.n_permutations)
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
    args.visual_features = tuple(args.visual_features)
    args.neural_reference_window_ms = tuple(args.neural_reference_window_ms)
    args.visual_window_ms = tuple(args.visual_window_ms)
    return Cfg(**vars(args))
# EOF


"""
feature_path_and_type
Translate one visual-feature spec into its HDF5 path and loader type.

INPUT:
    - feature_spec: str -> "pixels" or "model_name:pooling:layer_index"
    - models_dir: Path -> directory with extracted features
    - pixel_step: int -> pixel sampling step

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
        or PROJECT_ROOT / "results" / "last_frame_acf_coupling"
        / f"{cfg.dynamic_exp_name}_{channel_name}"
        / f"ref{reference_start:.0f}to{reference_end:.0f}ms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    neural_lags_ms = np.arange(0, cfg.max_lag_ms + cfg.lag_step_ms / 2, cfg.lag_step_ms)

    summary_rows = []
    couplings = {}
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
        coupling = compute_last_frame_acf_coupling(
            data,
            last_frame_time_ms=cfg.last_frame_time_ms,
            neural_reference_window_ms=cfg.neural_reference_window_ms,
            visual_window_ms=cfg.visual_window_ms,
            neural_lags_ms=neural_lags_ms,
            n_permutations=cfg.n_permutations,
            random_seed=cfg.random_seed,
        )
        couplings[feature_spec] = coupling
        feature_token = feature_path.stem

        figure, _ = plot_last_frame_acf_coupling(
            coupling,
            title=(
                f"{cfg.dynamic_exp_name} | {data['dynamic'].shape[0]} channels | "
                f"{coupling['visual_label']}"
            ),
        )
        figure.savefig(
            output_dir / f"{feature_token}_acf_coupling.png",
            dpi=cfg.figure_dpi, bbox_inches="tight",
        )
        np.savez(
            output_dir / f"{feature_token}_acf_coupling.npz",
            stimuli=data["stimuli"],
            channel_numbers=data["channel_numbers"],
            **{
                key: value for key, value in coupling.items()
                if isinstance(value, np.ndarray)
            },
        )

        # One summary row per feature at the lag with the highest matched r.
        best_lag_index = int(np.nanargmax(coupling["mean_matched_r"]))
        summary_rows.append({
            "feature": feature_spec,
            "visual_label": coupling["visual_label"],
            "n_channels": data["dynamic"].shape[0],
            "best_lag_ms": neural_lags_ms[best_lag_index],
            "mean_matched_r": coupling["mean_matched_r"][best_lag_index],
            "mean_mismatched_r": np.nanmean(coupling["mismatched_r"][best_lag_index]),
            "p_fwe": coupling["p_values_fwe"][best_lag_index],
            "across_stimulus_rho": coupling["across_stimulus_rho"][best_lag_index],
            "max_across_stimulus_rho": np.nanmax(coupling["across_stimulus_rho"]),
        })
        print(", ".join(f"{key}={value}" for key, value in summary_rows[-1].items()))
    # end for feature_spec

    # Overlay matched-minus-mismatched coupling for every feature.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for feature_spec, coupling in couplings.items():
        specificity = np.nanmean(coupling["matched_r"] - coupling["mismatched_r"], axis=1)
        axes[0].plot(neural_lags_ms, specificity, label=coupling["visual_label"])
        axes[1].plot(neural_lags_ms, coupling["across_stimulus_rho"], label=coupling["visual_label"])
    # end for feature_spec
    axes[0].set(
        xlabel="Neural lag (ms)", ylabel="Same − other movie r",
        title="Stimulus-specific timecourse coupling",
    )
    axes[1].set(
        xlabel="Neural lag (ms)", ylabel="Spearman ρ across stimuli",
        title="Mean visual ACF vs mean neural ACF",
    )
    for axis in axes:
        axis.axhline(0, color="0.5", linewidth=0.8)
        axis.grid(alpha=0.2)
    # end for axis
    axes[1].legend(frameon=False, fontsize=7)
    figure.suptitle(cfg.dynamic_exp_name)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_comparison.png", dpi=cfg.figure_dpi, bbox_inches="tight")

    with open(output_dir / "summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0])
        writer.writeheader()
        writer.writerows(summary_rows)
    # end with open
    print(f"Saved to {output_dir}")
    if cfg.show:
        plt.show()
    # end if cfg.show
# EOF


if __name__ == "__main__":
    main()
# EOC
