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

from image_processing.optical_flow import (
    extract_optical_flow_dataset,
    list_movie_paths,
)


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

    # The dense u/v field is computed directly on this fixed spatial grid.
    flow_width: int = 64
    flow_height: int = 64
    write_batch_size: int = 64
    compression: str | None = "lzf"
    overwrite: bool = False

    # Repair repeated-frame plateaus after raw flow extraction.
    interpolate_zero_flow: bool = True
    zero_flow_threshold: float = 0.05

    # Interpretable OpenCV Farneback parameters.
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
# EOF


"""
parse_args
Parses optical-flow input, output, geometry, and Farneback parameters.

OUTPUT:
    - cfg: Cfg -> validated extraction configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract dense Farneback u/v optical-flow vectors from vid_* movies."
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
    parser.add_argument("--flow_width", type=int, default=Cfg.flow_width)
    parser.add_argument("--flow_height", type=int, default=Cfg.flow_height)
    parser.add_argument(
        "--write_batch_size", type=int, default=Cfg.write_batch_size
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default=Cfg.compression,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--interpolate-zero-flow",
        action=argparse.BooleanOptionalAction,
        default=Cfg.interpolate_zero_flow,
        help=(
            "Linearly interpolate internal near-zero flow runs bounded by "
            "healthy flow fields."
        ),
    )
    parser.add_argument(
        "--zero_flow_threshold",
        type=float,
        default=Cfg.zero_flow_threshold,
        help="Maximum per-frame RMS displacement classified as zero flow.",
    )

    parser.add_argument("--pyr_scale", type=float, default=Cfg.pyr_scale)
    parser.add_argument("--levels", type=int, default=Cfg.levels)
    parser.add_argument("--winsize", type=int, default=Cfg.winsize)
    parser.add_argument("--iterations", type=int, default=Cfg.iterations)
    parser.add_argument("--poly_n", type=int, choices=(5, 7), default=Cfg.poly_n)
    parser.add_argument("--poly_sigma", type=float, default=Cfg.poly_sigma)
    parser.add_argument("--flags", type=int, default=Cfg.flags)
    args = parser.parse_args()

    positive_integer_names = (
        "frame_stride",
        "flow_width",
        "flow_height",
        "write_batch_size",
        "levels",
        "winsize",
        "iterations",
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

    if not 0 < args.pyr_scale < 1:
        parser.error("--pyr_scale must be between 0 and 1.")
    # end if pyr_scale
    if args.poly_sigma <= 0:
        parser.error("--poly_sigma must be positive.")
    # end if poly_sigma
    if args.zero_flow_threshold < 0:
        parser.error("--zero_flow_threshold must be non-negative.")
    # end if zero_flow_threshold
    if args.compression == "none":
        args.compression = None
    # end if compression

    args.video_patterns = tuple(args.video_patterns)
    return Cfg(**vars(args))
# EOF


"""
main
Extracts dense optical-flow vectors for the configured movie stimulus set.

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
        / "optical_flow_farneback_static_dynamic.h5"
    ).expanduser()

    movie_paths = list_movie_paths(
        stimuli_dir,
        video_patterns=cfg.video_patterns,
        max_videos=cfg.max_videos,
    )
    farneback_kwargs = {
        "pyr_scale": cfg.pyr_scale,
        "levels": cfg.levels,
        "winsize": cfg.winsize,
        "iterations": cfg.iterations,
        "poly_n": cfg.poly_n,
        "poly_sigma": cfg.poly_sigma,
        "flags": cfg.flags,
    }

    print(f"Environment: {ENV}")
    print(f"Movies: {len(movie_paths)} from {stimuli_dir}")
    print(
        f"Flow grid: {cfg.flow_height} x {cfg.flow_width} x 2 "
        f"= {2 * cfg.flow_height * cfg.flow_width} features"
    )
    print(f"Frame stride: {cfg.frame_stride}")
    print(
        "Internal zero-flow interpolation: "
        f"{cfg.interpolate_zero_flow} "
        f"(RMS threshold={cfg.zero_flow_threshold:g})"
    )
    print(f"Output: {output_path}")

    extract_optical_flow_dataset(
        movie_paths,
        output_path,
        flow_width=cfg.flow_width,
        flow_height=cfg.flow_height,
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        write_batch_size=cfg.write_batch_size,
        compression=cfg.compression,
        overwrite=cfg.overwrite,
        farneback_kwargs=farneback_kwargs,
        interpolate_zero_flow=cfg.interpolate_zero_flow,
        zero_flow_threshold=cfg.zero_flow_threshold,
    )
    print(f"Completed optical-flow extraction: {output_path}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
