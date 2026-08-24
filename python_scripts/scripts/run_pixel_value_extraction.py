import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.optical_flow import list_movie_paths
from image_processing.pixel_values import extract_pixel_value_dataset


@dataclass
class Cfg:
    stimuli_dir: str | None = None
    output_path: str | None = None
    video_patterns: tuple[str, ...] = (
        "vid_*.mp4",
        "vid_*.mov",
        "vid_*.m4v",
    )
    max_videos: int | None = None
    max_frames: int | None = None
    frame_stride: int = 1

    # Resize every movie to one shared geometry before spatial subsampling.
    image_width: int = 500
    image_height: int = 500
    pixel_step: int = 50
    write_batch_size: int = 64
    compression: str | None = "lzf"
    overwrite: bool = False
# EOF


"""
parse_args
Parses movie, RGB-grid, pixel-subsampling, and output parameters.

OUTPUT:
    - cfg: Cfg -> validated pixel-value extraction configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract every Nth spatial RGB pixel from each vid_* movie frame."
        )
    )
    parser.add_argument("--stimuli_dir")
    parser.add_argument("--output_path")
    parser.add_argument(
        "--video_patterns",
        nargs="+",
        default=list(Cfg.video_patterns),
    )
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--frame_stride", type=int, default=Cfg.frame_stride)
    parser.add_argument("--image_width", type=int, default=Cfg.image_width)
    parser.add_argument("--image_height", type=int, default=Cfg.image_height)
    parser.add_argument("--pixel_step", type=int, default=Cfg.pixel_step)
    parser.add_argument(
        "--write_batch_size",
        type=int,
        default=Cfg.write_batch_size,
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default=Cfg.compression,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    positive_integer_names = (
        "frame_stride",
        "image_width",
        "image_height",
        "pixel_step",
        "write_batch_size",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be a positive integer.")
        # end if parameter
    # end for name

    for name in ("max_videos", "max_frames"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be a positive integer.")
        # end if value
    # end for name

    if args.compression == "none":
        args.compression = None
    # end if compression
    args.video_patterns = tuple(args.video_patterns)
    return Cfg(**vars(args))
# EOF


"""
main
Extracts subsampled RGB pixel vectors for the configured movie stimulus set.

OUTPUT:
    - None
"""
def main():
    cfg = parse_args()
    stimuli_dir = Path(
        cfg.stimuli_dir or Path(paths["data_path"]) / "stimuli"
    ).expanduser()
    output_path = Path(
        cfg.output_path
        or Path(paths["data_path"])
        / "models"
        / f"pixel_values_rgb_step{cfg.pixel_step}_static_dynamic.h5"
    ).expanduser()

    movie_paths = list_movie_paths(
        stimuli_dir,
        video_patterns=cfg.video_patterns,
        max_videos=cfg.max_videos,
    )
    n_spatial_pixels = (
        cfg.image_width * cfg.image_height + cfg.pixel_step - 1
    ) // cfg.pixel_step
    n_features = 3 * n_spatial_pixels

    print(f"Environment: {ENV}")
    print(f"Movies: {len(movie_paths)} from {stimuli_dir}")
    print(f"RGB grid: {cfg.image_height} x {cfg.image_width}")
    print(
        f"Spatial sampling: every {cfg.pixel_step}th pixel -> "
        f"{n_spatial_pixels} locations x 3 RGB channels "
        f"= {n_features} features"
    )
    print(f"Frame stride: {cfg.frame_stride}")
    print(f"Output: {output_path}")

    extract_pixel_value_dataset(
        movie_paths,
        output_path,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
        pixel_step=cfg.pixel_step,
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        write_batch_size=cfg.write_batch_size,
        compression=cfg.compression,
        overwrite=cfg.overwrite,
    )
    print(f"Completed pixel-value extraction: {output_path}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
