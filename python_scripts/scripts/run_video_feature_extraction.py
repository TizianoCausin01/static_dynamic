import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.video_feature_extraction import (
    build_frame_preprocessor,
    extract_video_dataset_features,
    get_model_source,
    list_video_paths,
)
from useful_stuff.general_utils import get_device
from useful_stuff.image_processing.computational_models import imgANN


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Cfg:
    model_name: str
    pkg: str = "hf"
    dataset_name: str = "static_dynamic"
    pooling: str | None = "mean"
    img_size: int = 224
    batch_size: int = 8
    frame_stride: int = 1
    max_frames: int | None = None
    max_videos: int | None = None
    video_pattern: str = "vid_*.mp4"
    stimuli_dir: str | None = None
    output_dir: str | None = None
    weights_type: str = "DEFAULT"
    repo_url: str | None = None
    revision: str | None = None
    attn_implementation: str | None = "sdpa"
    dtype: str = "float32"
    device: str | None = None
    layers: list[str] | None = None
    compression: str | None = None
    trust_remote_code: bool = False
    overwrite: bool = False
# EOF


"""
parse_args
Parses imgANN, video-decoding, and HDF5 output parameters.

OUTPUT:
    - cfg: Cfg -> validated extraction configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sequential imgANN features from every frame of vid_*.mp4 stimuli."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pkg", default="hf")
    parser.add_argument("--dataset_name", default="static_dynamic")
    parser.add_argument("--pooling", default="mean", help="imgANN pooling; use 'none' for no pooling.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--video_pattern", default="vid_*.mp4")
    parser.add_argument("--stimuli_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--weights-type", default="DEFAULT")
    parser.add_argument("--repo_url")
    parser.add_argument("--revision")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device")
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--compression", choices=("lzf", "gzip"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.pooling.lower() == "none":
        args.pooling = None
    if args.attn_implementation.lower() == "none":
        args.attn_implementation = None
    if args.batch_size < 1 or args.frame_stride < 1:
        parser.error("--batch_size and --frame_stride must be positive integers.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max_frames must be a positive integer.")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max_videos must be a positive integer.")

    return Cfg(**vars(args))
# EOF


"""
main
Initializes imgANN and extracts layer features from the configured stimulus videos.

OUTPUT:
    - None
"""
def main():
    cfg = parse_args()
    stimuli_dir = Path(cfg.stimuli_dir or Path(paths["data_path"]) / "stimuli")
    output_dir = Path(cfg.output_dir or Path(paths["data_path"]) / "models")
    model_source = get_model_source(cfg.model_name, cfg.repo_url)
    device = cfg.device or get_device()
    dtype = DTYPES[cfg.dtype]

    video_paths = list_video_paths(
        stimuli_dir, video_pattern=cfg.video_pattern, max_videos=cfg.max_videos
    )
    ann = imgANN(
        model_name=cfg.model_name,
        pkg=cfg.pkg,
        img_size=cfg.img_size,
        relevant_layers=cfg.layers,
        pooling=cfg.pooling,
        weights_type=cfg.weights_type,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        repo_url=model_source,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
        device=device,
    )
    preprocessor = build_frame_preprocessor(
        cfg.pkg, model_source, cfg.img_size
    )

    print(ann)
    print(f"Videos: {len(video_paths)} from {stimuli_dir}")
    print(f"Layers: {len(ann.get_relevant_layers())}")
    print(f"Output directory: {output_dir}")

    output_paths = extract_video_dataset_features(
        ann,
        preprocessor,
        video_paths,
        output_dir,
        cfg.dataset_name,
        model_source,
        cfg.batch_size,
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        dtype=dtype,
        compression=cfg.compression,
        overwrite=cfg.overwrite,
    )
    print(f"Completed {len(output_paths)} layer files for {cfg.model_name}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
