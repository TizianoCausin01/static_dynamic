import argparse
from dataclasses import asdict, dataclass, field
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


# Fig 3 last-frame temporal score with static-response time bins in place of
# model layers: every 50 ms bin of the static last-frame response is compared
# with the movie RDM timecourse, before and after regressing out the same bin
# of the static responses to the 2000 ms and 2250 ms frame images (RDM-level
# OLS). Bin order plays the role of layer depth. From the project root:
# .venv/bin/python python_scripts/scripts/run_static_bin_temporal_scores.py


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
    match_timed_static_movie_rasters, static_bin_similarity_by_window,
    summarize_layer_timing,
)
from project_specific_utils.last_frame_decay import resample_array
from run_layer_temporal_scores import MONKEY_SESSIONS


@dataclass
class Cfg:
    monkeys: list[str] = field(default_factory=lambda: list(MONKEY_SESSIONS))
    # Static bins (ms from image onset, end-exclusive): 100-150, ..., 500-550.
    bin_start_ms: float = 100
    bin_stop_ms: float = 550
    bin_width_ms: float = 50

    # Neural and RSA parameters of the presentation notebook (Fig 3).
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float = 1000
    neural_rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "spearman"
    rank_space: bool = False
    fit_intercept: bool = True
    # Inclusive bin lags of the earlier-frame RDMs, e.g. (-5, 5) also regresses
    # out the 5 bins before and after each target bin; None = same bin only.
    rdm_delay_embedding_lags: tuple[int, int] | None = None
    last_frame_onset_ms: float = 2500
    latency_window_ms: tuple[float, float] = (2000, 2700)
    centroid_smoothing: float = 0
    peak_smoothing: float = 3
    centroid_similarity_cutoff: float = 0.01
    peak_similarity_cutoff: float = 0.01

    output_dir: str | None = None
    output_tag: str = "static_bins50_100to550_rdm2000+2250"
# EOF


"""
parse_args
Parse the monkeys, bins and regression space.

OUTPUT:
    - cfg: Cfg -> configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description="Fig 3 last-frame temporal scores over static-response bins."
    )
    parser.add_argument("--monkeys", nargs="+", choices=tuple(MONKEY_SESSIONS))
    parser.add_argument("--bin_start_ms", type=float)
    parser.add_argument("--bin_stop_ms", type=float)
    parser.add_argument("--bin_width_ms", type=float)
    parser.add_argument("--rank_space", action="store_true", default=None)
    parser.add_argument("--rdm_delay_embedding_lags", type=int, nargs=2)
    parser.add_argument("--latency_window_ms", nargs=2, type=float)
    parser.add_argument("--output_dir")
    parser.add_argument("--output_tag")
    args = parser.parse_args()

    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    if cfg.rdm_delay_embedding_lags is not None:
        cfg.rdm_delay_embedding_lags = tuple(cfg.rdm_delay_embedding_lags)
    # end if lags given
    cfg.output_dir = str(Path(
        cfg.output_dir or PROJECT_ROOT / "results" / "layer_temporal_scores"
    ))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg
# EOF


"""
load_static_and_movie
Static rasters of the last-frame, 2000 ms and 2250 ms images (source rate,
cropped) and the movie RDM timecourse of one monkey, in matched stimulus order.

INPUT:
    - monkey: str -> key of MONKEY_SESSIONS
    - cfg: Cfg -> configuration

OUTPUT:
    - static_rasters: dict[str, np.ndarray] -> "last", "2000ms", "2250ms" ->
        channels x static time x stimuli at cfg.source_fs
    - dynamic_neural_rdms: np.ndarray -> movie time x stimulus pairs at cfg.new_fs
"""
def load_static_and_movie(monkey: str, cfg: Cfg):
    static_name, dynamic_name, good_channels, reliable_key = MONKEY_SESSIONS[monkey]
    data_dir = Path(paths["data_path"]) / "data"
    channel_kwargs = {
        "good_channels": good_channels,
        "reliable_channels_config": PROJECT_ROOT / "reliable_channels.yaml",
        "reliable_channels_key": reliable_key,
        "return_channel_numbers": True,
    }
    image_rasters, image_names, image_channels = load_natraster(
        data_dir / f"{static_name}_natraster_img.mat", **channel_kwargs,
    )
    movie_rasters, movie_names, movie_channels = load_natraster(
        data_dir / f"{dynamic_name}_natraster_vid.mat", **channel_kwargs,
    )
    if not np.array_equal(image_channels, movie_channels):
        raise ValueError(f"{monkey}: sessions retained different channels.")
    # end if channel numbers differ
    (
        last_rasters, aligned_movie_rasters, rasters_2000ms, rasters_2250ms,
        shared_stimuli, _,
    ) = match_timed_static_movie_rasters(
        image_rasters, image_names, movie_rasters, movie_names,
    )
    # Static responses stay at the source rate so the 50 ms bins are exact.
    crop_samples = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
    static_rasters = {
        name: rasters[:, :crop_samples, :]
        for name, rasters in (
            ("last", last_rasters), ("2000ms", rasters_2000ms), ("2250ms", rasters_2250ms),
        )
    }
    # Movie: resampled to the analysis rate, one RDM per time point.
    movie_resampled = resample_array(aligned_movie_rasters, cfg.source_fs, cfg.new_fs)
    print(
        f"{monkey}: {len(image_channels)} channels | {len(shared_stimuli)} stimuli",
        flush=True,
    )
    return static_rasters, compute_rdm_timeseries(movie_resampled, cfg.neural_rdm_metric)
# EOF


"""
main
Score every monkey and save a CSV, a JSON config and the bin similarity curves
(npz, keys "monkey|static_bins|none|condition", as in run_layer_temporal_scores).
"""
def main() -> None:
    cfg = parse_args()
    bin_starts = np.arange(cfg.bin_start_ms, cfg.bin_stop_ms, cfg.bin_width_ms)
    bins_ms = [(float(start), float(start + cfg.bin_width_ms)) for start in bin_starts]
    # Bin order as normalized "depth", like layer depth in the model analysis.
    bin_depths = np.linspace(0, 1, len(bins_ms))

    all_rows, all_curves = [], {}
    for monkey in cfg.monkeys:
        static_rasters, dynamic_neural_rdms = load_static_and_movie(monkey, cfg)
        similarity, variance_removed = static_bin_similarity_by_window(
            static_rasters["last"],
            [static_rasters["2000ms"], static_rasters["2250ms"]],
            dynamic_neural_rdms, cfg.source_fs, bins_ms,
            cfg.neural_rdm_metric, cfg.rsa_metric,
            fit_intercept=cfg.fit_intercept, rank_space=cfg.rank_space,
            delay_embedding_lags=cfg.rdm_delay_embedding_lags,
        )
        for condition_name, bin_similarity in similarity.items():
            time_ms = np.arange(bin_similarity.shape[1]) * 1000 / cfg.new_fs
            _, centroid_latency_ms, peak_latency_ms, _ = summarize_layer_timing(
                bin_similarity, time_ms, cfg.last_frame_onset_ms,
                cfg.latency_window_ms, cfg.latency_window_ms,
                cfg.centroid_smoothing, cfg.centroid_similarity_cutoff,
                cfg.peak_similarity_cutoff, peak_smoothing_sigma=cfg.peak_smoothing,
            )
            for timing_name, latency_ms in (
                    ("centroid", centroid_latency_ms), ("peak", peak_latency_ms)):
                score = depth_latency_score(bin_depths, latency_ms)
                all_rows.append({
                    "monkey": monkey, "condition": condition_name,
                    "timing": timing_name, **score,
                })
                print(
                    f"  {condition_name:<20} {timing_name:<8} "
                    f"rho={score['rho']:+.2f} p={score['pvalue']:.2g} n={score['n_layers']}",
                    flush=True,
                )
            # end for timing_name, latency_ms
            all_curves[f"{monkey}|static_bins|none|{condition_name}"] = bin_similarity
        # end for condition_name
        all_curves[f"{monkey}|static_bins|none|variance_removed"] = variance_removed
        print(f"  variance removed per bin: {np.round(variance_removed, 2)}", flush=True)
    # end for monkey

    output_dir = Path(cfg.output_dir)
    with open(output_dir / f"{cfg.output_tag}_temporal_scores.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # end with open
    with open(output_dir / f"{cfg.output_tag}_config.json", "w") as config_file:
        json.dump({"config": asdict(cfg), "bins_ms": bins_ms,
                   "sessions": MONKEY_SESSIONS}, config_file, indent=2)
    # end with open
    np.savez(output_dir / f"{cfg.output_tag}_curves.npz", **all_curves)
    print(f"Saved {output_dir / cfg.output_tag}_*")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
