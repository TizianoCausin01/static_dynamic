import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as file:
    config = yaml.safe_load(file)
# end with open

paths = config[ENV]["paths"]
sys.path.extend([paths["src_path"], paths["useful_stuff_path"]])

from project_specific_utils import (
    average_presentations,
    average_repetition_halves,
    compute_rdm_timeseries,
    compute_split_half_reliability,
    cross_temporal_similarity,
    load_reliable_channels,
    load_raster,
    raw_cross_temporal_similarity,
    split_half_filename_suffix,
)
from useful_stuff.general_utils import TimeSeries
from useful_stuff.general_utils.utils import mean_centering


STATIC_FRAME_TIMINGS = ("last_frame", "2000ms", "2250ms")


@dataclass
class Cfg:
    static_exp_name: str = "baby1_260718to27"
    dynamic_exp_name: str = "baby1_260716to24"
    static_path: str | None = None
    dynamic_path: str | None = None
    output_dir: str | None = None
    reliable_channels_config: str | None = None
    reliable_channels_key: str | None = None

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = (101, 109)
    # Used only when good_channels is None; ranked once from matched movies.
    top_k: int | None = None
    frame_timings: tuple[str, ...] = STATIC_FRAME_TIMINGS
    static_crop_ms: float = 1000
    dynamic_crop_ms: float | None = None
    source_fs: float = 1000
    new_fs: float = 100
    rdm_metric: str = "cosine_cnt"
    rsa_metric: str = "correlation"
    feature_centering: bool = False
    n_split_repeats: int = 10
    n_half_trial_repeats: int = 10
    spearman_brown_k: float = 2
    random_seed: int | None = 0
# EOF


def parse_random_seed(value: str) -> int | None:
    """Parse an integer seed or the case-insensitive string None."""
    if value.lower() == "none":
        return None
    # end if value.lower()
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "random_seed must be an integer or None."
        ) from error
    # end try
# EOF


def channel_selection_name(cfg: Cfg) -> str:
    """Return the directory label describing the configured channel subset."""
    if cfg.reliable_channels_config is not None:
        selection_name = "reliable_channels"
        if cfg.good_channels is not None:
            first_channel, last_channel = cfg.good_channels
            selection_name += f"_{first_channel}to{last_channel}"
        # end if cfg.good_channels
        if cfg.top_k is not None:
            selection_name += f"_top_{cfg.top_k}"
        # end if cfg.top_k
        return selection_name
    # end if cfg.reliable_channels_config
    if cfg.good_channels is not None:
        first_channel, last_channel = cfg.good_channels
        return f"channels_{first_channel}to{last_channel}"
    # end if cfg.good_channels
    if cfg.top_k is not None:
        return f"top_{cfg.top_k}"
    # end if cfg.top_k
    return "all_channels"
# EOF


"""
parse_args
Parse and validate paths, data selection, reliability, and RSA parameters.

OUTPUT:
    - cfg: Cfg -> validated analysis configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Save raw and RSA self-consistency plus full- and half-movie-trial "
            "static-dynamic cross-temporal correlations for every frame timing."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument("--static_path")
    parser.add_argument("--dynamic_path")
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--reliable_channels_config",
        help=(
            "YAML config whose channels are intersected with the selected "
            "range before optional top-k ranking."
        ),
    )
    parser.add_argument(
        "--reliable_channels_key",
        help="Dataset key in --reliable_channels_config; defaults to static_exp_name.",
    )

    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument(
        "--good_channels",
        nargs=2,
        type=int,
        default=list(Cfg.good_channels),
        metavar=("FIRST", "LAST"),
    )
    channel_group.add_argument("--top_k", type=int)
    channel_group.add_argument("--all_channels", action="store_true")

    parser.add_argument(
        "--frame_timings",
        nargs="+",
        choices=STATIC_FRAME_TIMINGS,
        default=list(Cfg.frame_timings),
    )
    parser.add_argument("--static_crop_ms", type=float, default=Cfg.static_crop_ms)
    parser.add_argument("--dynamic_crop_ms", type=float)
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument("--new_fs", type=float, default=Cfg.new_fs)
    parser.add_argument("--rdm_metric", default=Cfg.rdm_metric)
    parser.add_argument(
        "--rsa_metric",
        choices=("correlation", "spearman"),
        default=Cfg.rsa_metric,
    )
    parser.add_argument(
        "--feature_centering",
        action=argparse.BooleanOptionalAction,
        default=Cfg.feature_centering,
        help=(
            "Center each channel/time feature across stimuli after repetition "
            "averaging for raw correlations."
        ),
    )
    parser.add_argument("--n_split_repeats", type=int, default=Cfg.n_split_repeats)
    parser.add_argument(
        "--n_half_trial_repeats",
        type=int,
        default=Cfg.n_half_trial_repeats,
    )
    parser.add_argument(
        "--spearman_brown_k",
        type=float,
        default=Cfg.spearman_brown_k,
    )
    parser.add_argument(
        "--random_seed",
        type=parse_random_seed,
        default=Cfg.random_seed,
        help="Integer seed, or None to generate and record a random seed.",
    )
    args = parser.parse_args()

    if args.top_k is not None or args.all_channels:
        args.good_channels = None
    else:
        first_channel, last_channel = args.good_channels
        if first_channel < 1 or last_channel < first_channel:
            parser.error("--good_channels must be an increasing positive range.")
        # end if channel range
        args.good_channels = tuple(args.good_channels)
    # end if channel selection
    del args.all_channels

    positive_values = {
        "--static_crop_ms": args.static_crop_ms,
        "--source_fs": args.source_fs,
        "--new_fs": args.new_fs,
        "--n_split_repeats": args.n_split_repeats,
        "--n_half_trial_repeats": args.n_half_trial_repeats,
        "--spearman_brown_k": args.spearman_brown_k,
    }
    for argument_name, value in positive_values.items():
        if value <= 0:
            parser.error(f"{argument_name} must be positive.")
        # end if value
    # end for argument_name, value
    if args.dynamic_crop_ms is not None and args.dynamic_crop_ms <= 0:
        parser.error("--dynamic_crop_ms must be positive or omitted.")
    # end if args.dynamic_crop_ms
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top_k must be positive.")
    # end if args.top_k
    args.frame_timings = tuple(dict.fromkeys(args.frame_timings))
    return Cfg(**vars(args))
# EOF


"""
stimulus_identity
Extract the common stimulus identity for one requested static timing or movie.

INPUT:
    - stimulus_name: str -> presentation filename
    - condition: str -> static or dynamic
    - frame_timing: str | None -> requested static timing

OUTPUT:
    - identity: str | None -> matched identity or None for another condition
"""
def stimulus_identity(
        stimulus_name: str,
        condition: str,
        frame_timing: str | None = None,
        ) -> str | None:
    stem = Path(stimulus_name).stem
    if condition == "dynamic":
        prefix = "vid_"
    elif condition == "static":
        static_prefixes = {
            "last_frame": "img_",
            "2000ms": "img_2000ms_",
            "2250ms": "img_2250ms_",
        }
        if frame_timing not in static_prefixes:
            raise ValueError(f"Unsupported static frame timing: {frame_timing!r}.")
        # end if frame_timing
        if stem.startswith(static_prefixes["2000ms"]):
            observed_timing = "2000ms"
        elif stem.startswith(static_prefixes["2250ms"]):
            observed_timing = "2250ms"
        elif stem.startswith(static_prefixes["last_frame"]):
            observed_timing = "last_frame"
        else:
            return None
        # end if static prefix
        if observed_timing != frame_timing:
            return None
        # end if observed_timing
        prefix = static_prefixes[frame_timing]
    else:
        raise ValueError("condition must be 'static' or 'dynamic'.")
    # end if condition
    return stem[len(prefix):] if stem.startswith(prefix) else None
# EOF


"""
resample_rasters
Resample presentation rasters along time with the established TimeSeries helper.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations
    - source_fs: float -> original sampling frequency
    - target_fs: float -> requested sampling frequency

OUTPUT:
    - rasters: np.ndarray -> resampled presentation rasters
"""
def resample_rasters(
        rasters: np.ndarray,
        source_fs: float,
        target_fs: float,
        ) -> np.ndarray:
    if source_fs == target_fs:
        return rasters
    # end if source_fs == target_fs
    raster_series = TimeSeries(rasters, source_fs)
    raster_series.resample(target_fs)
    return raster_series.get_array()
# EOF


def spearman_brown_correction(correlation: np.ndarray, k: float) -> np.ndarray:
    """Predict reliability after changing test length by a factor of k."""
    return k * correlation / (1 + (k - 1) * correlation)
# EOF


"""
add_self_consistency_results
Store uncorrected split values and both uncorrected and Spearman-Brown summary
timecourses under explicit result names.

INPUT:
    - results: dict -> result arrays populated in place
    - result_name: str -> static_raw, static_rsa, dynamic_raw, or dynamic_rsa
    - split_values: np.ndarray -> split repetition x time correlations
    - spearman_brown_k: float -> target/current test-length ratio
    - stimulus_stds: np.ndarray | None -> raw SD across stimuli per split and time

OUTPUT:
    - None
"""
def add_self_consistency_results(
        results: dict,
        result_name: str,
        split_values: np.ndarray,
        spearman_brown_k: float,
        stimulus_stds: np.ndarray | None = None,
        ) -> None:
    mean_values = np.nanmean(split_values, axis=0)
    std_values = np.nanstd(split_values, axis=0)
    lower_values = np.nanpercentile(split_values, 2.5, axis=0)
    upper_values = np.nanpercentile(split_values, 97.5, axis=0)
    corrected_split_values = spearman_brown_correction(
        split_values, spearman_brown_k,
    )

    prefix = f"{result_name}_self_consistency"
    results[f"{prefix}_splits_uncorrected"] = split_values
    results[f"{prefix}_mean_uncorrected"] = mean_values
    results[f"{prefix}_std_uncorrected"] = std_values
    results[f"{prefix}_lower_uncorrected"] = lower_values
    results[f"{prefix}_upper_uncorrected"] = upper_values
    results[f"{prefix}_mean_spearman_brown"] = spearman_brown_correction(
        mean_values, spearman_brown_k,
    )
    results[f"{prefix}_lower_spearman_brown"] = spearman_brown_correction(
        lower_values, spearman_brown_k,
    )
    results[f"{prefix}_upper_spearman_brown"] = spearman_brown_correction(
        upper_values, spearman_brown_k,
    )
    results[f"{prefix}_std_spearman_brown"] = np.nanstd(
        corrected_split_values, axis=0,
    )
    if stimulus_stds is not None:
        results[f"{prefix}_stimulus_std_splits_uncorrected"] = stimulus_stds
        results[f"{prefix}_stimulus_std_mean_uncorrected"] = np.nanmean(
            stimulus_stds, axis=0,
        )
    # end if stimulus_stds
    return None
# EOF


"""
select_channels
Apply the configured range, reliable-channel intersection, and optional top-k
movie response ranking once so every frame timing uses the same channels.

INPUT:
    - static_rasters: np.ndarray -> loaded static presentation rasters
    - dynamic_rasters: np.ndarray -> loaded dynamic presentation rasters
    - dynamic_identities: list[str] -> identity for every movie presentation
    - shared_by_timing: dict[str, list[str]] -> matched identities per timing
    - cfg: Cfg -> channel configuration
    - loaded_channel_numbers: np.ndarray | None -> channels already selected at load

OUTPUT:
    - static_rasters: np.ndarray -> selected static rasters
    - dynamic_rasters: np.ndarray -> selected dynamic rasters
    - channel_numbers: np.ndarray -> one-based MATLAB channel numbers
"""
def select_channels(
        static_rasters: np.ndarray,
        dynamic_rasters: np.ndarray,
        dynamic_identities: list[str],
        shared_by_timing: dict[str, list[str]],
        cfg: Cfg,
        loaded_channel_numbers: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels_filtered_during_load = loaded_channel_numbers is not None
    if channels_filtered_during_load:
        channel_numbers = np.asarray(loaded_channel_numbers, dtype=int)
        if len(channel_numbers) != dynamic_rasters.shape[0]:
            raise ValueError(
                "loaded_channel_numbers must match the loaded channel axis."
            )
        # end if loaded channel count mismatch
    elif cfg.good_channels is None:
        channel_numbers = np.arange(1, dynamic_rasters.shape[0] + 1)
    else:
        first_channel, last_channel = cfg.good_channels
        channel_numbers = np.arange(first_channel, last_channel + 1)
    # end if cfg.good_channels is None

    if (
            cfg.reliable_channels_config is not None
            and not channels_filtered_during_load
            ):
        reliable_channels_key = (
            cfg.reliable_channels_key or cfg.static_exp_name
        )
        reliable_channels = set(load_reliable_channels(
            cfg.reliable_channels_config, reliable_channels_key,
        ))
        reliable_mask = np.asarray([
            channel in reliable_channels for channel in channel_numbers
        ])
        if not reliable_mask.any():
            raise ValueError(
                "No reliable channels fall inside the loaded channel range."
            )
        # end if no reliable channels
        static_rasters = static_rasters[reliable_mask]
        dynamic_rasters = dynamic_rasters[reliable_mask]
        channel_numbers = channel_numbers[reliable_mask]
    # end if cfg.reliable_channels_config

    if cfg.top_k is None:
        return static_rasters, dynamic_rasters, channel_numbers
    # end if cfg.top_k
    if cfg.top_k > dynamic_rasters.shape[0]:
        raise ValueError(
            f"top_k={cfg.top_k} exceeds {dynamic_rasters.shape[0]} loaded channels."
        )
    # end if cfg.top_k

    shared_union = set().union(*map(set, shared_by_timing.values()))
    dynamic_keep = np.asarray([
        identity in shared_union for identity in dynamic_identities
    ])
    channel_scores = dynamic_rasters[:, :, dynamic_keep].max(axis=(1, 2))
    channel_indices = np.argsort(channel_scores)[-cfg.top_k:][::-1]
    channel_numbers = channel_numbers[channel_indices]
    return (
        static_rasters[channel_indices],
        dynamic_rasters[channel_indices],
        channel_numbers,
    )
# EOF


"""
compute_timing_results
Compute every self-consistency and static-dynamic cross-temporal result for one
static frame timing.

INPUT:
    - static_rasters: np.ndarray -> resampled static presentation rasters
    - dynamic_rasters: np.ndarray -> resampled movie presentation rasters
    - static_identities: list[str | None] -> identities for the requested timing
    - dynamic_identities: list[str | None] -> movie identities
    - frame_timing: str -> static timing label saved with the result
    - channel_numbers: np.ndarray -> selected one-based channel numbers
    - cfg: Cfg -> analysis configuration
    - rng: np.random.Generator -> shared reproducible random generator

OUTPUT:
    - results: dict[str, np.ndarray] -> arrays saved for this frame timing
"""
def compute_timing_results(
        static_rasters: np.ndarray,
        dynamic_rasters: np.ndarray,
        static_identities: list[str | None],
        dynamic_identities: list[str | None],
        frame_timing: str,
        channel_numbers: np.ndarray,
        cfg: Cfg,
        rng: np.random.Generator,
        ) -> dict[str, np.ndarray]:
    shared_stimuli = sorted(
        {identity for identity in static_identities if identity is not None}
        & {identity for identity in dynamic_identities if identity is not None}
    )
    if len(shared_stimuli) < 3:
        raise ValueError(
            f"{frame_timing} has only {len(shared_stimuli)} shared stimuli."
        )
    # end if len(shared_stimuli)

    shared_set = set(shared_stimuli)
    static_keep = np.asarray([identity in shared_set for identity in static_identities])
    dynamic_keep = np.asarray([identity in shared_set for identity in dynamic_identities])
    static_condition = static_rasters[:, :, static_keep]
    dynamic_condition = dynamic_rasters[:, :, dynamic_keep]
    static_ids = [
        identity for identity, keep in zip(static_identities, static_keep) if keep
    ]
    dynamic_ids = [
        identity for identity, keep in zip(dynamic_identities, dynamic_keep) if keep
    ]

    results = {
        "frame_timing": np.asarray(frame_timing),
        "rdm_metric": np.asarray(cfg.rdm_metric),
        "rsa_metric": np.asarray(cfg.rsa_metric),
        "raw_feature_centering": np.asarray(cfg.feature_centering),
        "shared_stimuli": np.asarray(shared_stimuli),
        "channel_numbers": channel_numbers,
        "static_times_ms": np.arange(static_condition.shape[1]) * 1000 / cfg.new_fs,
        "dynamic_times_ms": np.arange(dynamic_condition.shape[1]) * 1000 / cfg.new_fs,
    }
    static_identity_array = np.asarray(static_ids)
    dynamic_identity_array = np.asarray(dynamic_ids)
    results["static_repetitions_per_stimulus"] = np.asarray([
        np.sum(static_identity_array == identity) for identity in shared_stimuli
    ])
    results["dynamic_repetitions_per_stimulus"] = np.asarray([
        np.sum(dynamic_identity_array == identity) for identity in shared_stimuli
    ])
    results["half_dynamic_repetitions_per_stimulus"] = (
        results["dynamic_repetitions_per_stimulus"] // 2
    )

    print("  Computing static raw self-consistency...")
    static_raw, static_raw_stimulus_std = compute_split_half_reliability(
        static_condition, static_ids, shared_stimuli, "raw", rng,
        cfg.n_split_repeats, cfg.rdm_metric, cfg.rsa_metric,
        feature_centering=cfg.feature_centering,
        return_stimulus_std=True,
    )
    print("  Computing static RSA self-consistency...")
    static_rsa = compute_split_half_reliability(
        static_condition, static_ids, shared_stimuli, "rsa", rng,
        cfg.n_split_repeats, cfg.rdm_metric, cfg.rsa_metric,
    )
    print("  Computing dynamic raw self-consistency...")
    dynamic_raw, dynamic_raw_stimulus_std = compute_split_half_reliability(
        dynamic_condition, dynamic_ids, shared_stimuli, "raw", rng,
        cfg.n_split_repeats, cfg.rdm_metric, cfg.rsa_metric,
        feature_centering=cfg.feature_centering,
        return_stimulus_std=True,
    )
    print("  Computing dynamic RSA self-consistency...")
    dynamic_rsa = compute_split_half_reliability(
        dynamic_condition, dynamic_ids, shared_stimuli, "rsa", rng,
        cfg.n_split_repeats, cfg.rdm_metric, cfg.rsa_metric,
    )
    add_self_consistency_results(
        results, "static_raw", static_raw, cfg.spearman_brown_k,
        static_raw_stimulus_std,
    )
    add_self_consistency_results(results, "static_rsa", static_rsa, cfg.spearman_brown_k)
    add_self_consistency_results(
        results, "dynamic_raw", dynamic_raw, cfg.spearman_brown_k,
        dynamic_raw_stimulus_std,
    )
    add_self_consistency_results(results, "dynamic_rsa", dynamic_rsa, cfg.spearman_brown_k)

    # Full-condition averages provide the all-trial cross-temporal matrices.
    static_means = average_presentations(static_condition, static_ids, shared_stimuli)
    dynamic_means = average_presentations(dynamic_condition, dynamic_ids, shared_stimuli)
    static_rdms = compute_rdm_timeseries(static_means, cfg.rdm_metric)
    dynamic_rdms = compute_rdm_timeseries(dynamic_means, cfg.rdm_metric)
    static_raw_means = static_means
    dynamic_raw_means = dynamic_means
    if cfg.feature_centering:
        # Center only after all repetitions have been averaged by stimulus.
        static_raw_means = mean_centering(static_means, axis=2)
        dynamic_raw_means = mean_centering(dynamic_means, axis=2)
    # end if cfg.feature_centering
    print("  Computing all-trial raw and RSA cross-temporal matrices...")
    all_raw_mean, all_raw_stimulus_std = raw_cross_temporal_similarity(
        dynamic_raw_means,
        static_raw_means,
        return_stimulus_std=True,
    )
    results["raw_lagged_correlation_all_dynamic_trials"] = all_raw_mean
    results["raw_lagged_correlation_all_dynamic_trials_stimulus_std"] = (
        all_raw_stimulus_std
    )
    results["rsa_lagged_correlation_all_dynamic_trials"] = (
        cross_temporal_similarity(dynamic_rdms, static_rdms, cfg.rsa_metric)
    )

    # Each repeat retains one independently drawn half of every movie's trials.
    half_raw_matrices = []
    half_raw_stimulus_stds = []
    half_rsa_matrices = []
    for repeat_index in range(cfg.n_half_trial_repeats):
        half_dynamic_means, _ = average_repetition_halves(
            dynamic_condition, dynamic_ids, shared_stimuli, rng,
        )
        half_dynamic_rdms = compute_rdm_timeseries(
            half_dynamic_means, cfg.rdm_metric,
        )
        half_dynamic_raw_means = half_dynamic_means
        if cfg.feature_centering:
            # Center after averaging the selected dynamic repetition half.
            half_dynamic_raw_means = mean_centering(
                half_dynamic_means, axis=2,
            )
        # end if cfg.feature_centering
        half_raw_mean, half_raw_stimulus_std = raw_cross_temporal_similarity(
            half_dynamic_raw_means,
            static_raw_means,
            return_stimulus_std=True,
        )
        half_raw_matrices.append(half_raw_mean)
        half_raw_stimulus_stds.append(half_raw_stimulus_std)
        half_rsa_matrices.append(
            cross_temporal_similarity(
                half_dynamic_rdms, static_rdms, cfg.rsa_metric,
            )
        )
        print(
            f"  Half-movie repeat {repeat_index + 1}/"
            f"{cfg.n_half_trial_repeats}"
        )
    # end for repeat_index
    results["raw_lagged_correlation_half_dynamic_trials"] = np.stack(
        half_raw_matrices,
    )
    results["raw_lagged_correlation_half_dynamic_trials_stimulus_std"] = (
        np.stack(half_raw_stimulus_stds)
    )
    results["rsa_lagged_correlation_half_dynamic_trials"] = np.stack(
        half_rsa_matrices,
    )
    return results
# EOF


"""
main
Load both raster files once, reuse them across frame timings, compute all
requested measures, and save one self-contained NPZ file per timing.

OUTPUT:
    - None
"""
def main() -> None:
    cfg = parse_args()
    data_dir = Path(paths["data_path"]) / "data"
    static_path = Path(
        cfg.static_path or data_dir / f"{cfg.static_exp_name}_raster_img.mat"
    ).expanduser()
    dynamic_path = Path(
        cfg.dynamic_path or data_dir / f"{cfg.dynamic_exp_name}_raster_vid.mat"
    ).expanduser()
    output_root = Path(
        cfg.output_dir
        or PROJECT_ROOT
        / "results"
        / "static_dynamic_all_frame_timings"
        / f"{cfg.dynamic_exp_name}_vs_{cfg.static_exp_name}"
    ).expanduser()
    output_dir = output_root / channel_selection_name(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_channel_numbers = None
    if cfg.reliable_channels_config is not None:
        reliable_channels_key = (
            cfg.reliable_channels_key or cfg.static_exp_name
        )
        loaded_channel_numbers = np.asarray(load_reliable_channels(
            cfg.reliable_channels_config, reliable_channels_key,
        ))
        if cfg.good_channels is not None:
            first_channel, last_channel = cfg.good_channels
            loaded_channel_numbers = loaded_channel_numbers[
                (loaded_channel_numbers >= first_channel)
                & (loaded_channel_numbers <= last_channel)
            ]
        # end if cfg.good_channels
        if len(loaded_channel_numbers) == 0:
            raise ValueError(
                "No reliable channels fall inside the requested channel range."
            )
        # end if no reliable channels
        channel_slice = loaded_channel_numbers - 1
    elif cfg.good_channels is not None:
        first_channel, last_channel = cfg.good_channels
        channel_slice = slice(first_channel - 1, last_channel)
    else:
        channel_slice = slice(None)
    # end if channel loading selection
    static_end_sample = int(round(cfg.static_crop_ms * cfg.source_fs / 1000))
    dynamic_end_sample = None
    if cfg.dynamic_crop_ms is not None:
        dynamic_end_sample = int(round(cfg.dynamic_crop_ms * cfg.source_fs / 1000))
    # end if cfg.dynamic_crop_ms

    print("Loading static presentations once...")
    static_rasters, static_names = load_raster(
        static_path, channel_slice=channel_slice, end_sample=static_end_sample,
    )
    print("Loading dynamic presentations once...")
    dynamic_rasters, dynamic_names = load_raster(
        dynamic_path, channel_slice=channel_slice, end_sample=dynamic_end_sample,
    )

    dynamic_identities = [
        stimulus_identity(name, "dynamic") for name in dynamic_names
    ]
    static_identities_by_timing = {
        frame_timing: [
            stimulus_identity(name, "static", frame_timing)
            for name in static_names
        ]
        for frame_timing in cfg.frame_timings
    }
    dynamic_identity_set = {
        identity for identity in dynamic_identities if identity is not None
    }
    shared_by_timing = {
        frame_timing: sorted(
            {
                identity for identity in static_identities
                if identity is not None
            }
            & dynamic_identity_set
        )
        for frame_timing, static_identities
        in static_identities_by_timing.items()
    }

    static_rasters, dynamic_rasters, channel_numbers = select_channels(
        static_rasters, dynamic_rasters, dynamic_identities,
        shared_by_timing, cfg, loaded_channel_numbers,
    )
    print(f"Selected MATLAB channels: {channel_numbers.tolist()}")
    print("Resampling loaded presentations once...")
    static_rasters = resample_rasters(
        static_rasters, cfg.source_fs, cfg.new_fs,
    )
    dynamic_rasters = resample_rasters(
        dynamic_rasters, cfg.source_fs, cfg.new_fs,
    )

    if cfg.random_seed is None:
        cfg.random_seed = int(np.random.SeedSequence().generate_state(1)[0])
        print(f"Generated random seed: {cfg.random_seed}")
    # end if cfg.random_seed
    rng = np.random.default_rng(cfg.random_seed)
    filename_suffix = split_half_filename_suffix(
        cfg.rdm_metric, cfg.feature_centering,
    )
    config_path = output_dir / f"config{filename_suffix}.json"
    manifest_path = output_dir / f"manifest{filename_suffix}.json"
    manifest = {
        "environment": ENV,
        "channel_selection": channel_selection_name(cfg),
        "rdm_metric": cfg.rdm_metric,
        "rsa_metric": cfg.rsa_metric,
        "config_path": str(config_path),
        "timings": {},
    }
    for frame_timing in cfg.frame_timings:
        print(f"\nFrame timing: {frame_timing}")
        results = compute_timing_results(
            static_rasters,
            dynamic_rasters,
            static_identities_by_timing[frame_timing],
            dynamic_identities,
            frame_timing,
            channel_numbers,
            cfg,
            rng,
        )
        result_path = output_dir / f"{frame_timing}{filename_suffix}_results.npz"
        np.savez_compressed(result_path, **results)
        manifest["timings"][frame_timing] = {
            "path": str(result_path),
            "shared_stimuli": len(results["shared_stimuli"]),
            "arrays": {
                result_name: list(np.asarray(result_value).shape)
                for result_name, result_value in results.items()
            },
        }
        print(f"  Saved {result_path}")
    # end for frame_timing

    with open(config_path, "w") as file:
        json.dump(asdict(cfg), file, indent=2)
    # end with open
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)
    # end with open
    print(f"\nSaved all results to {output_dir}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
