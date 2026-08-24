import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import h5py
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
    compute_channel_selectivity_reliability,
    last_frame_presentation_indices,
    load_raster,
    load_raster_presentation_names,
    save_reliable_channels,
    summarize_channel_reliability,
)


@dataclass
class Cfg:
    exp_name: str = "baby1_260718to27"
    raster_path: str | None = None
    reliable_channels_config: str | None = None
    config_key: str | None = None

    # MATLAB channel numbers are one-based and both endpoints are inclusive.
    good_channels: tuple[int, int] | None = (84, 186)
    source_fs: float = 1000
    time_window_ms: tuple[float, float] = (60, 200)
    reliability_threshold: float = 0.4
    n_split_repeats: int = 100
    selection_split_count: int = 50
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


"""
parse_args
Parse image dataset, channel range, response window, and reliability settings.

OUTPUT:
    - cfg: Cfg -> validated reliable-channel selection configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Select reliable channels by correlating random-half mean "
            "last-frame-image selectivity vectors, following Xiao et al. "
            "(2025), and save the result in YAML."
        )
    )
    parser.add_argument("--exp_name", default=Cfg.exp_name)
    parser.add_argument("--raster_path")
    parser.add_argument("--reliable_channels_config")
    parser.add_argument("--config_key")

    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument(
        "--good_channels",
        nargs=2,
        type=int,
        default=list(Cfg.good_channels),
        metavar=("FIRST", "LAST"),
        help="Inclusive one-based MATLAB channel range.",
    )
    channel_group.add_argument(
        "--all_channels",
        action="store_true",
        help="Evaluate every channel instead of --good_channels.",
    )

    parser.add_argument("--source_fs", type=float, default=Cfg.source_fs)
    parser.add_argument(
        "--time_window_ms",
        nargs=2,
        type=float,
        default=list(Cfg.time_window_ms),
        metavar=("START", "END"),
        help="Half-open post-onset response window in milliseconds.",
    )
    parser.add_argument(
        "--reliability_threshold",
        type=float,
        default=Cfg.reliability_threshold,
        help="Minimum mean selection-split Pearson correlation.",
    )
    parser.add_argument(
        "--n_split_repeats", type=int, default=Cfg.n_split_repeats,
    )
    parser.add_argument(
        "--selection_split_count", type=int, default=Cfg.selection_split_count,
    )
    parser.add_argument(
        "--random_seed",
        type=parse_random_seed,
        default=Cfg.random_seed,
        help="Integer seed, or None for a non-deterministic run.",
    )
    args = parser.parse_args()

    if args.all_channels:
        args.good_channels = None
    else:
        first_channel, last_channel = args.good_channels
        if first_channel < 1 or last_channel < first_channel:
            parser.error(
                "--good_channels must be an increasing one-based range."
            )
        # end if invalid channel range
        args.good_channels = tuple(args.good_channels)
    # end if args.all_channels
    del args.all_channels

    if args.source_fs <= 0:
        parser.error("--source_fs must be positive.")
    # end if invalid source_fs
    window_start_ms, window_end_ms = args.time_window_ms
    if window_start_ms < 0 or window_end_ms <= window_start_ms:
        parser.error(
            "--time_window_ms must be an increasing non-negative interval."
        )
    # end if invalid time window
    args.time_window_ms = tuple(args.time_window_ms)
    if not -1 <= args.reliability_threshold <= 1:
        parser.error("--reliability_threshold must be between -1 and 1.")
    # end if invalid reliability threshold
    if args.n_split_repeats < 2:
        parser.error("--n_split_repeats must be at least two.")
    # end if too few split repeats
    if not 1 <= args.selection_split_count < args.n_split_repeats:
        parser.error(
            "--selection_split_count must leave at least one held-out split."
        )
    # end if invalid selection split count

    return Cfg(**vars(args))
# EOF


"""
main
Load last-frame image repetitions, compute channel reliability, and update the
reliable-channel YAML config without loading unused presentations or times.

INPUT:
    - cfg: Cfg -> reliable-channel selection configuration

OUTPUT:
    - None
"""
def main(cfg: Cfg) -> None:
    data_dir = Path(paths["data_path"]) / "data"
    raster_path = Path(
        cfg.raster_path
        or data_dir / f"{cfg.exp_name}_raster_img.mat"
    )
    config_path = Path(
        cfg.reliable_channels_config
        or PROJECT_ROOT / "reliable_channels.yaml"
    )
    config_key = cfg.config_key or cfg.exp_name

    # Identify only standard last-frame image repetitions before raster loading.
    all_presentation_names = load_raster_presentation_names(raster_path)
    presentation_indices, presentation_identities = (
        last_frame_presentation_indices(all_presentation_names)
    )

    with h5py.File(raster_path, "r") as file:
        channel_count = file["raster"].shape[2]
        sample_count = file["raster"].shape[1]
    # end with h5py.File

    if cfg.good_channels is None:
        first_channel, last_channel = 1, channel_count
    else:
        first_channel, last_channel = cfg.good_channels
        if last_channel > channel_count:
            raise IndexError(
                f"Requested channel {last_channel}, but {raster_path.name} "
                f"contains only {channel_count} channels."
            )
        # end if channel range exceeds raster
    # end if cfg.good_channels is None
    channel_numbers = np.arange(first_channel, last_channel + 1)
    channel_slice = slice(first_channel - 1, last_channel)

    window_start_sample = int(round(
        cfg.time_window_ms[0] * cfg.source_fs / 1000,
    ))
    window_end_sample = int(round(
        cfg.time_window_ms[1] * cfg.source_fs / 1000,
    ))
    if window_end_sample > sample_count:
        raise IndexError(
            f"The requested time window ends at sample {window_end_sample}, "
            f"but {raster_path.name} has {sample_count} samples."
        )
    # end if time window exceeds raster

    print(
        f"Loading {len(presentation_indices)} last-frame presentations, "
        f"channels {first_channel}-{last_channel}, and "
        f"{cfg.time_window_ms[0]:g}-{cfg.time_window_ms[1]:g} ms..."
    )
    window_rasters, _ = load_raster(
        raster_path,
        channel_slice=channel_slice,
        start_sample=window_start_sample,
        end_sample=window_end_sample,
        presentation_indices=presentation_indices,
    )
    # Average time first, yielding one response per channel and presentation.
    window_responses = np.nanmean(window_rasters, axis=1)

    rng = np.random.default_rng(cfg.random_seed)
    split_reliabilities, stimulus_order = (
        compute_channel_selectivity_reliability(
            window_responses,
            presentation_identities,
            rng,
            n_split_repeats=cfg.n_split_repeats,
        )
    )
    selection_reliability, heldout_reliability, reliable_mask = (
        summarize_channel_reliability(
            split_reliabilities,
            reliability_threshold=cfg.reliability_threshold,
            selection_split_count=cfg.selection_split_count,
        )
    )
    reliable_channels = channel_numbers[reliable_mask]

    # Store independent scores for retained channels while keeping config brief.
    reliable_selection_scores = {
        int(channel): float(selection_reliability[channel_index])
        for channel_index, channel in enumerate(channel_numbers)
        if reliable_mask[channel_index]
    }
    reliable_heldout_scores = {
        int(channel): float(heldout_reliability[channel_index])
        for channel_index, channel in enumerate(channel_numbers)
        if reliable_mask[channel_index]
    }
    result = {
        "method": "xiao_2025_individual_image_split_half_selectivity",
        "source_raster": raster_path.name,
        "stimulus_condition": "last_frame_images",
        "stimulus_count": len(stimulus_order),
        "presentation_count": len(presentation_indices),
        "source_fs": float(cfg.source_fs),
        "time_window_ms": list(cfg.time_window_ms),
        "channel_range": [int(first_channel), int(last_channel)],
        "reliability_threshold": float(cfg.reliability_threshold),
        "n_split_repeats": int(cfg.n_split_repeats),
        "selection_split_count": int(cfg.selection_split_count),
        "heldout_split_count": int(
            cfg.n_split_repeats - cfg.selection_split_count
        ),
        "random_seed": cfg.random_seed,
        "reliable_channels": reliable_channels.tolist(),
        "reliable_channel_selection_scores": reliable_selection_scores,
        "reliable_channel_heldout_scores": reliable_heldout_scores,
    }
    save_reliable_channels(config_path, config_key, result)

    print(
        f"Selected {len(reliable_channels)} of {len(channel_numbers)} channels "
        f"at reliability >= {cfg.reliability_threshold:g}."
    )
    print(f"Reliable MATLAB channels: {reliable_channels.tolist()}")
    print(f"Saved config entry {config_key!r} to {config_path.resolve()}")
    return None
# EOF


if __name__ == "__main__":
    main(parse_args())
# EOC
