import argparse
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


# Example from the project root:
# .venv/bin/python python_scripts/scripts/run_static_dynamic_drsa_permutation.py --n_permutations 10000


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils import (
    cluster_permutation_test, compute_rdm_timeseries, load_natraster,
    match_timed_static_movie_rasters, permutation_p_values,
    permuted_cross_temporal_similarity, previous_response_rasters,
    regress_out_static_response, regressed_condition_suffix,
)
from project_specific_utils.last_frame_decay import resample_array


# Same sessions as MONKEY_SESSIONS in all_monkeys_presentation.ipynb:
# (dynamic session, static session, channel range, reliable-channel key).
MONKEY_SESSIONS = {
    "baby1": ("baby1_260716to0928", "baby1_260718to0727", (84, 186), "baby1_260718to0727"),
    "paul": ("paul_20260831to0927", "paul_20260901to0925", (1, 64), "paul_20260901to0925"),
    "red": ("red_20260720to0928", "red_20260726to0925", (1, 64), "red_20260726to0925"),
}


@dataclass
class Cfg:
    monkeys: list[str] = field(default_factory=lambda: list(MONKEY_SESSIONS))
    # "raw" and every earlier static response regressed out of the last frame;
    # "2000ms+2250ms" regresses both frames out together (timepoint only).
    conditions: list[str] = field(default_factory=lambda: ["raw", "2000ms", "2250ms"])
    use_reliable_channels: bool = True
    source_fs: float = 1000
    new_fs: float = 100
    static_crop_ms: float = 1000
    rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "spearman"
    n_permutations: int = 10000
    random_seed: int = 0
    n_jobs: int = 1
    alpha: float = 0.05
    cluster_alpha: float = 0.05
    # Static window (ms from image onset) of the best-static-match curves: for
    # every permutation, the max over these static bins at each movie bin is
    # saved, so the curve test can apply the same max (and smoothing) to the null.
    best_static_window_ms: tuple[float, float] = (0, 1000)
    # "channelwise": each channel on its own earlier response (in-sample OLS);
    # "timepoint": each channel on all delay-embedded channels, one model per
    # static timepoint (timepoint_static_regress_out).
    regress_out_method: str = "channelwise"
    timepoint_regression_type: str = "ridge"  # "lr" or "ridge"
    timepoint_cv_type: str = "kf"  # "kf" out-of-fold residuals, "same" in-sample
    timepoint_n_splits: int = 5
    timepoint_delay_embedding_lags: tuple[int, int] | None = (-1, 1)
    timepoint_alphas: tuple[float, ...] = (1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5)
    # In-fold PCA of the predictors keeping this variance fraction; None = off.
    timepoint_pca_variance: float | None = None
# EOC


"""
parse_args
Parse the monkey and condition selection plus the permutation settings.

OUTPUT:
    - cfg: Cfg -> validated configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Stimulus-permutation test of the static-dynamic dRSA matrix, raw and "
            "after regressing earlier static responses out of the last frame."
        )
    )
    parser.add_argument("--monkeys", nargs="+", choices=tuple(MONKEY_SESSIONS))
    parser.add_argument(
        "--conditions", nargs="+", choices=("raw", "2000ms", "2250ms", "2000ms+2250ms"),
    )
    parser.add_argument("--n_permutations", type=int)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--cluster_alpha", type=float)
    parser.add_argument(
        "--best_static_window_ms", type=float, nargs=2,
        help="Static window (ms) maximised over for the best-static-match null curves.",
    )
    parser.add_argument("--regress_out_method", choices=("channelwise", "timepoint"))
    parser.add_argument("--timepoint_regression_type", choices=("lr", "ridge"))
    parser.add_argument("--timepoint_cv_type", choices=("kf", "same"))
    parser.add_argument("--timepoint_n_splits", type=int)
    parser.add_argument("--timepoint_pca_variance", type=float)
    parser.add_argument(
        "--timepoint_delay_embedding_lags", type=int, nargs=2,
        help="First and last sample lag, e.g. -1 1; 0 0 uses only the same timepoint.",
    )
    args = parser.parse_args()

    cfg = Cfg()
    for field_name, value in vars(args).items():
        if value is not None:
            setattr(cfg, field_name, value)
        # end if the user supplied the argument
    # end for field_name
    cfg.best_static_window_ms = tuple(cfg.best_static_window_ms)
    if cfg.timepoint_delay_embedding_lags is not None:
        cfg.timepoint_delay_embedding_lags = tuple(cfg.timepoint_delay_embedding_lags)
    # end if lags given
    return cfg
# EOF


"""
load_condition_rdms
Load one monkey, align the stimuli, and build the dynamic RDMs plus the static
RDMs of every requested condition. Mirrors static_dynamic_drsa_permutation.ipynb:
static responses are cropped, everything is resampled to new_fs, and regressed
conditions use regress_out_static_response with cfg.regress_out_method.

INPUT:
    - monkey: str -> key of MONKEY_SESSIONS
    - cfg: Cfg -> analysis configuration

OUTPUT:
    - dynamic_rdms: np.ndarray -> movie time x stimulus pairs
    - static_rdms: dict[str, np.ndarray] -> static time x stimulus pairs per condition
    - shared_stimuli: list[str] -> matched stimulus identities
    - channel_numbers: np.ndarray -> retained one-based MATLAB channels
"""
def load_condition_rdms(monkey: str, cfg: Cfg):
    dynamic_name, static_name, good_channels, reliable_key = MONKEY_SESSIONS[monkey]
    data_dir = Path(paths["data_path"]) / "data"
    channel_kwargs = {
        "good_channels": good_channels,
        "reliable_channels_config": (
            PROJECT_ROOT / "reliable_channels.yaml" if cfg.use_reliable_channels else None
        ),
        "reliable_channels_key": reliable_key,
    }
    loaded_static, static_names, channel_numbers = load_natraster(
        data_dir / f"{static_name}_natraster_img.mat",
        return_channel_numbers=True, **channel_kwargs,
    )
    loaded_dynamic, dynamic_names = load_natraster(
        data_dir / f"{dynamic_name}_natraster_vid.mat", **channel_kwargs,
    )
    (
        last_frame_rasters, dynamic_rasters, image_2000ms_rasters,
        image_2250ms_rasters, shared_stimuli, _,
    ) = match_timed_static_movie_rasters(
        loaded_static, static_names, loaded_dynamic, dynamic_names,
    )

    # channels x time x stimuli, static responses cropped before resampling.
    crop_samples = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
    static_resampled = {
        name: resample_array(rasters[:, :crop_samples, :], cfg.source_fs, cfg.new_fs)
        for name, rasters in (
            ("raw", last_frame_rasters), ("2000ms", image_2000ms_rasters),
            ("2250ms", image_2250ms_rasters),
        )
    }
    static_rdms = {}
    for condition in cfg.conditions:
        condition_rasters = static_resampled["raw"]
        if condition != "raw":
            # One regression per static bin across stimuli; keep the residual.
            condition_rasters = regress_out_static_response(
                previous_response_rasters(static_resampled, condition),
                static_resampled["raw"], cfg.new_fs,
                method=cfg.regress_out_method,
                timepoint_kwargs={
                    "delay_embedding_lags": cfg.timepoint_delay_embedding_lags,
                    "regression_type": cfg.timepoint_regression_type,
                    "alphas": cfg.timepoint_alphas,
                    "cv_type": cfg.timepoint_cv_type,
                    "n_splits": cfg.timepoint_n_splits,
                    "pca_variance": cfg.timepoint_pca_variance,
                },
            )
        # end if regressed condition
        static_rdms[condition] = compute_rdm_timeseries(condition_rasters, cfg.rdm_metric)
    # end for condition
    dynamic_rdms = compute_rdm_timeseries(
        resample_array(dynamic_rasters, cfg.source_fs, cfg.new_fs), cfg.rdm_metric,
    )
    return dynamic_rdms, static_rdms, shared_stimuli, channel_numbers
# EOF


def main() -> None:
    cfg = parse_args()
    for monkey in cfg.monkeys:
        dynamic_name, static_name, _, _ = MONKEY_SESSIONS[monkey]
        output_dir = (
            PROJECT_ROOT / "results" / "static_dynamic_drsa_permutation"
            / f"{dynamic_name}_vs_{static_name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        dynamic_rdms, static_rdms, shared_stimuli, channel_numbers = load_condition_rdms(
            monkey, cfg,
        )
        static_times_ms = np.arange(next(iter(static_rdms.values())).shape[0]) * 1000 / cfg.new_fs
        dynamic_times_ms = np.arange(dynamic_rdms.shape[0]) * 1000 / cfg.new_fs
        best_static_bins = (
            (static_times_ms >= cfg.best_static_window_ms[0])
            & (static_times_ms < cfg.best_static_window_ms[1])
        )
        print(f"{monkey}: {len(channel_numbers)} channels, {len(shared_stimuli)} stimuli", flush=True)

        for condition, condition_rdms in static_rdms.items():
            start_time = time.perf_counter()
            # The static RDMs are relabelled; the movie RDMs stay fixed.
            observed, null = permuted_cross_temporal_similarity(
                dynamic_rdms, condition_rdms,
                n_permutations=cfg.n_permutations, metric=cfg.rsa_metric,
                random_seed=cfg.random_seed, n_jobs=cfg.n_jobs,
            )
            p_values = permutation_p_values(observed, null)
            clusters = cluster_permutation_test(observed, null, cluster_alpha=cfg.cluster_alpha)
            # permutations x movie time: best static match of every null matrix,
            # the same max the plotted curves take over the observed matrix.
            null_best_static_curves = np.nanmax(null[:, :, best_static_bins], axis=2)
            # The full null (permutations x movie x static) is not saved.
            del null

            condition_suffix = "raw" if condition == "raw" else regressed_condition_suffix(
                condition, cfg.regress_out_method, cfg.timepoint_regression_type,
                cfg.timepoint_cv_type, cfg.timepoint_pca_variance,
            )
            output_path = output_dir / (
                f"{cfg.rdm_metric}_{cfg.rsa_metric}_{condition_suffix}_"
                f"{cfg.n_permutations}perm_seed{cfg.random_seed}.npz"
            )
            # Same keys as static_dynamic_drsa_permutation.ipynb.
            np.savez_compressed(
                output_path,
                observed=observed,
                pointwise_p=p_values["pointwise"],
                max_statistic_p=p_values["max_statistic"],
                null_max=p_values["null_max"],
                cluster_labels=clusters["labels"],
                cluster_masses=clusters["masses"],
                cluster_p=clusters["p_values"],
                null_max_cluster_mass=clusters["null_max_mass"],
                null_best_static_curves=null_best_static_curves.astype(np.float32),
                best_static_window_ms=np.asarray(cfg.best_static_window_ms),
                static_times_ms=static_times_ms,
                dynamic_times_ms=dynamic_times_ms,
                shared_stimuli=np.asarray(shared_stimuli),
                channel_numbers=channel_numbers,
                config=np.asarray(str({
                    **asdict(cfg), "dynamic_exp_name": dynamic_name,
                    "static_exp_name": static_name, "condition": condition,
                })),
            )
            print(
                f"  {condition}: {time.perf_counter() - start_time:.0f} s | "
                f"FWE cells {np.sum(p_values['max_statistic'] < cfg.alpha)} | "
                f"saved {output_path.name}",
                flush=True,
            )
        # end for condition
    # end for monkey
# EOF


if __name__ == "__main__":
    main()
# end if __name__
