import argparse
from dataclasses import asdict, dataclass
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
    last_frame_presentation_indices, load_natraster, load_raster,
    load_raster_presentation_names, match_timed_static_movie_rasters,
    rowwise_similarity, split_half_tuning_reliability, tuning_noise_ceiling,
    window_mean_responses,
)
from project_specific_utils.timed_frame_window_rsa import spearman_brown


@dataclass
class Cfg:
    static_exp_name: str = "paul_20260901to0925"
    dynamic_exp_name: str = "paul_20260831to0927"
    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = (1, 64)
    # None uses the static session's own list in reliable_channels.yaml.
    reliable_channels_key: str | None = None
    source_fs: float = 1000
    # Half-open windows: static from image onset, dynamic from movie onset
    # (same as tuning_static/dynamic_window_ms in all_monkeys_presentation).
    static_window_ms: tuple[float, float] = (100, 500)
    dynamic_window_ms: tuple[float, float] = (2500, 2900)
    metric: str = "spearman"
    n_split_repeats: int = 100
    random_seed: int = 0
# EOC


"""
parse_args
Parse the image/movie datasets, channels, tuning windows and split settings.

OUTPUT:
    - cfg: Cfg -> noise-ceiling configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Split-half noise ceiling of the static-dynamic tuning agreement "
            "for every reliable channel and for their population average."
        )
    )
    parser.add_argument("--static_exp_name", default=Cfg.static_exp_name)
    parser.add_argument("--dynamic_exp_name", default=Cfg.dynamic_exp_name)
    parser.add_argument(
        "--good_channels", nargs=2, type=int, default=list(Cfg.good_channels),
        metavar=("FIRST", "LAST"), help="Inclusive one-based MATLAB channel range.",
    )
    parser.add_argument("--reliable_channels_key", default=Cfg.reliable_channels_key)
    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument(
        "--static_window_ms", nargs=2, type=float,
        default=list(Cfg.static_window_ms), metavar=("START", "END"),
    )
    parser.add_argument(
        "--dynamic_window_ms", nargs=2, type=float,
        default=list(Cfg.dynamic_window_ms), metavar=("START", "END"),
    )
    parser.add_argument("--metric", choices=("spearman", "correlation"), default=Cfg.metric)
    parser.add_argument("--n_split_repeats", type=int, default=Cfg.n_split_repeats)
    parser.add_argument("--random_seed", type=int, default=Cfg.random_seed)
    args = parser.parse_args()
    args.good_channels = tuple(args.good_channels)
    args.static_window_ms = tuple(args.static_window_ms)
    args.dynamic_window_ms = tuple(args.dynamic_window_ms)
    if args.n_split_repeats < 1:
        parser.error("--n_split_repeats must be positive.")
    # end if n_split_repeats
    return Cfg(**vars(args))
# EOF


"""
tuning_noise_ceiling_path
Result file shared by this script (writer) and all_monkeys_presentation (reader).

INPUT:
    - cfg: Cfg -> noise-ceiling configuration

OUTPUT:
    - output_path: Path -> .npz file of the result
"""
def tuning_noise_ceiling_path(cfg: Cfg) -> Path:
    static_start, static_end = cfg.static_window_ms
    dynamic_start, dynamic_end = cfg.dynamic_window_ms
    return (
        PROJECT_ROOT / "results" / "tuning_noise_ceiling"
        / f"{cfg.dynamic_exp_name}_vs_{cfg.static_exp_name}"
        / (
            f"{cfg.metric}_static{static_start:g}-{static_end:g}ms_"
            f"dynamic{dynamic_start:g}-{dynamic_end:g}ms_"
            f"{cfg.n_split_repeats}splits_seed{cfg.random_seed}.npz"
        )
    )
# EOF


"""
load_window_presentations
Load the window-mean response of selected channels for selected presentations,
reading only the window samples of the contiguous channel range that spans them.

INPUT:
    - raster_path: Path -> presentation-level raster file
    - presentation_indices: np.ndarray -> sorted presentation indices
    - channel_numbers: np.ndarray -> sorted one-based MATLAB channels
    - window_ms: tuple[float, float] -> half-open window from stimulus onset
    - fs: float -> raster sampling frequency in Hz

OUTPUT:
    - window_responses: np.ndarray -> channels x presentations window means
"""
def load_window_presentations(raster_path, presentation_indices, channel_numbers, window_ms, fs):
    # Same timestamp selection as window_mean_responses (start <= t < end).
    start_sample = int(np.ceil(window_ms[0] * fs / 1000))
    end_sample = int(np.ceil(window_ms[1] * fs / 1000))
    # HDF5 cannot combine explicit channel and presentation indices in one read,
    # so the contiguous range covering the channels is read and then subset.
    first_channel, last_channel = channel_numbers[0], channel_numbers[-1]
    rasters, _ = load_raster(
        raster_path,
        channel_slice=slice(first_channel - 1, last_channel),
        start_sample=start_sample,
        end_sample=end_sample,
        presentation_indices=presentation_indices,
    )
    rasters = rasters[channel_numbers - first_channel]
    return rasters.mean(axis=1)
# EOF


"""
main
Estimate the split-half reliability of the static and dynamic tuning of every
reliable channel and of the population average, and the resulting noise ceiling
of their agreement; save everything next to the other static-dynamic results.

INPUT:
    - cfg: Cfg -> noise-ceiling configuration

OUTPUT:
    - None
"""
def main(cfg: Cfg) -> None:
    data_dir = Path(paths["data_path"]) / "data"
    reliable_channels_key = cfg.reliable_channels_key or cfg.static_exp_name

    # Same channels and stimuli as the averaged data used by the notebook.
    channel_kwargs = {
        "good_channels": cfg.good_channels,
        "reliable_channels_config": PROJECT_ROOT / "reliable_channels.yaml",
        "reliable_channels_key": reliable_channels_key,
    }
    static_rasters, static_names, channel_numbers = load_natraster(
        data_dir / f"{cfg.static_exp_name}_natraster_img.mat",
        return_channel_numbers=True, **channel_kwargs,
    )
    dynamic_rasters, dynamic_names = load_natraster(
        data_dir / f"{cfg.dynamic_exp_name}_natraster_vid.mat", **channel_kwargs,
    )
    channel_numbers = np.asarray(channel_numbers, dtype=int)
    last_frame_rasters, movie_rasters, _, _, shared_stimuli, _ = (
        match_timed_static_movie_rasters(
            static_rasters, static_names, dynamic_rasters, dynamic_names,
        )
    )
    # channels x stimuli tuning of the averaged data, as in the notebook.
    static_tuning, _ = window_mean_responses(last_frame_rasters, cfg.static_window_ms, cfg.source_fs)
    dynamic_tuning, _ = window_mean_responses(movie_rasters, cfg.dynamic_window_ms, cfg.source_fs)
    # Rows: every reliable channel, then the population average (last row).
    observed = np.append(
        rowwise_similarity(static_tuning, dynamic_tuning, metric=cfg.metric),
        rowwise_similarity(
            static_tuning.mean(axis=0, keepdims=True),
            dynamic_tuning.mean(axis=0, keepdims=True), metric=cfg.metric,
        ),
    )

    # Presentation-level last-frame images of the shared stimuli.
    static_raster_path = data_dir / f"{cfg.static_exp_name}_raster_img.mat"
    image_indices, image_identities = last_frame_presentation_indices(
        load_raster_presentation_names(static_raster_path),
    )
    shared_set = set(shared_stimuli)
    keep_images = np.asarray([identity in shared_set for identity in image_identities])
    image_indices = image_indices[keep_images]
    image_identities = [identity for identity, kept in zip(image_identities, keep_images) if kept]

    # Presentation-level movies of the shared stimuli (names are vid_<identity>).
    dynamic_raster_path = data_dir / f"{cfg.dynamic_exp_name}_raster_vid.mat"
    movie_identities_all = [
        Path(name).stem.removeprefix("vid_")
        for name in load_raster_presentation_names(dynamic_raster_path)
    ]
    movie_indices = np.asarray([
        index for index, identity in enumerate(movie_identities_all) if identity in shared_set
    ], dtype=int)
    movie_identities = [movie_identities_all[index] for index in movie_indices]

    print(
        f"{cfg.static_exp_name} + {cfg.dynamic_exp_name}: {len(channel_numbers)} channels, "
        f"{len(shared_stimuli)} stimuli, {len(image_indices)} image and "
        f"{len(movie_indices)} movie presentations"
    )
    static_presentations = load_window_presentations(
        static_raster_path, image_indices, channel_numbers, cfg.static_window_ms, cfg.source_fs,
    )
    dynamic_presentations = load_window_presentations(
        dynamic_raster_path, movie_indices, channel_numbers, cfg.dynamic_window_ms, cfg.source_fs,
    )
    # Channel average of every presentation = population response.
    static_presentations = np.vstack([static_presentations, static_presentations.mean(axis=0)])
    dynamic_presentations = np.vstack([dynamic_presentations, dynamic_presentations.mean(axis=0)])

    # Sanity check: averaging all presentations should reproduce the natraster tuning.
    presentation_observed = np.array([
        rowwise_similarity(
            np.stack([
                static_presentations[row, np.asarray(image_identities) == stimulus].mean()
                for stimulus in shared_stimuli
            ])[np.newaxis],
            np.stack([
                dynamic_presentations[row, np.asarray(movie_identities) == stimulus].mean()
                for stimulus in shared_stimuli
            ])[np.newaxis],
            metric=cfg.metric,
        )[0]
        for row in range(static_presentations.shape[0])
    ])

    rng = np.random.default_rng(cfg.random_seed)
    # splits x rows uncorrected half-vs-half correlations.
    static_split_reliability = split_half_tuning_reliability(
        static_presentations, image_identities, shared_stimuli, rng,
        cfg.n_split_repeats, metric=cfg.metric,
    )
    dynamic_split_reliability = split_half_tuning_reliability(
        dynamic_presentations, movie_identities, shared_stimuli, rng,
        cfg.n_split_repeats, metric=cfg.metric,
    )
    # Spearman-Brown: reliability of the full data from the mean half correlation.
    static_reliability = spearman_brown(np.nanmean(static_split_reliability, axis=0))
    dynamic_reliability = spearman_brown(np.nanmean(dynamic_split_reliability, axis=0))
    noise_ceiling = tuning_noise_ceiling(static_reliability, dynamic_reliability)
    normalized_observed = observed / noise_ceiling

    output_path = tuning_noise_ceiling_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        channel_numbers=channel_numbers,
        shared_stimuli=np.asarray(shared_stimuli, dtype=str),
        # Every array below has one entry per channel plus the population (last).
        observed=observed,
        presentation_observed=presentation_observed,
        static_split_reliability=static_split_reliability,
        dynamic_split_reliability=dynamic_split_reliability,
        static_reliability=static_reliability,
        dynamic_reliability=dynamic_reliability,
        noise_ceiling=noise_ceiling,
        normalized_observed=normalized_observed,
        n_image_presentations=len(image_indices),
        n_movie_presentations=len(movie_indices),
        config=np.asarray(str(asdict(cfg))),
    )

    print(
        f"max |natraster - presentation-average| {cfg.metric} = "
        f"{np.nanmax(np.abs(observed - presentation_observed)):.2g}"
    )
    print(
        f"population: observed = {observed[-1]:.3f} | reliability static = "
        f"{static_reliability[-1]:.3f}, dynamic = {dynamic_reliability[-1]:.3f} | "
        f"noise ceiling = {noise_ceiling[-1]:.3f} | normalized = {normalized_observed[-1]:.3f}"
    )
    print(
        f"channels: median observed = {np.nanmedian(observed[:-1]):.3f} | median ceiling = "
        f"{np.nanmedian(noise_ceiling[:-1]):.3f} | median normalized = "
        f"{np.nanmedian(normalized_observed[:-1]):.3f} | "
        f"{np.isnan(noise_ceiling[:-1]).sum()} channels without a ceiling"
    )
    print(f"Saved {output_path}")
# EOF


if __name__ == "__main__":
    main(parse_args())
# end if __name__
