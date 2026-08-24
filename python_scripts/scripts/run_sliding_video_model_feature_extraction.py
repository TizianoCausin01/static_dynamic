import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
import yaml


# Example from the project root; output defaults to config[data_path]/models:
# .venv/bin/python python_scripts/scripts/run_sliding_video_model_feature_extraction.py \
#     --model_name vjepa2_vitl_fpc64_256 \
#     --stimuli_dir /Users/tizianocausin/sd_local/stimuli/static_dynamic_videos \
#     --window_size_frames 16 \
#     --batch_size 2


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.video_feature_extraction import (
    build_video_preprocessor,
    extract_sliding_video_dataset_features,
    get_model_source,
    list_video_paths,
)
from useful_stuff.general_utils import get_device
from useful_stuff.image_processing import vidANN


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Cfg:
    model_name: str
    pkg: str = "hf"
    architecture: str = "transformer"
    dataset_name: str = "static_dynamic"
    pooling: str | None = "mean"
    last_frame: bool = True

    img_size: int = 256
    window_size_frames: int = 16
    tubelet_size: int | None = None
    batch_size: int = 1
    preprocess_chunk_size: int = 64

    frame_stride: int = 1
    max_frames: int | None = None
    max_videos: int | None = None
    video_pattern: str = "vid_*.mp4"
    stimuli_dir: str | None = None
    output_dir: str | None = None

    hook_module: str = "mlp.fc2"
    layers: list[str] | None = None
    model_input_name: str = "pixel_values_videos"
    skip_predictor: bool = True

    weights_type: str = "DEFAULT"
    repo_url: str | None = None
    revision: str | None = None
    attn_implementation: str | None = "sdpa"
    dtype: str = "float32"
    device: str | None = None
    compression: str | None = None
    trust_remote_code: bool = False
    overwrite: bool = False
# EOF


"""
parse_args
Parses vidANN, sliding-window, preprocessing, and HDF5 output parameters.

OUTPUT:
    - cfg: Cfg -> validated extraction configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract one vidANN feature per video frame using fixed, "
            "left-padded, stride-one sliding windows."
        )
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pkg", default="hf")
    parser.add_argument(
        "--architecture",
        choices=("transformer", "cnn"),
        default="transformer",
    )
    parser.add_argument("--dataset_name", default="static_dynamic")
    parser.add_argument(
        "--pooling",
        default="mean",
        help="vidANN pooling; use 'none' for no pooling.",
    )
    parser.add_argument(
        "--last_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pool only the final temporal step. Use --no-last_frame "
            "to pool the complete window."
        ),
    )

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--window_size_frames", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of overlapping windows processed per model call.",
    )
    parser.add_argument("--preprocess_chunk_size", type=int, default=64)

    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--video_pattern", default="vid_*.mp4")
    parser.add_argument("--stimuli_dir")
    parser.add_argument("--output_dir")

    parser.add_argument(
        "--hook_module",
        default="mlp.fc2",
        help=(
            "Per-block transformer module used when --layers is omitted."
        ),
    )
    parser.add_argument("--layers", nargs="+")
    parser.add_argument(
        "--model_input_name",
        default="pixel_values_videos",
    )
    parser.add_argument(
        "--skip_predictor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pass skip_predictor=True to V-JEPA-style models. Disable "
            "for video models without this argument."
        ),
    )

    parser.add_argument("--weights-type", default="DEFAULT")
    parser.add_argument("--repo_url")
    parser.add_argument("--revision")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device")
    parser.add_argument("--compression", choices=("lzf", "gzip"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.pooling.lower() == "none":
        args.pooling = None
    # end if pooling is none
    if args.attn_implementation.lower() == "none":
        args.attn_implementation = None
    # end if attention implementation is none

    positive_arguments = {
        "--img_size": args.img_size,
        "--window_size_frames": args.window_size_frames,
        "--batch_size": args.batch_size,
        "--preprocess_chunk_size": args.preprocess_chunk_size,
        "--frame_stride": args.frame_stride,
    }
    for argument_name, value in positive_arguments.items():
        if value < 1:
            parser.error(f"{argument_name} must be a positive integer.")
        # end if invalid positive argument
    # end for argument_name

    if args.tubelet_size is not None and args.tubelet_size < 1:
        parser.error("--tubelet_size must be a positive integer.")
    # end if invalid tubelet_size
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max_frames must be a positive integer.")
    # end if invalid max_frames
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max_videos must be a positive integer.")
    # end if invalid max_videos

    return Cfg(**vars(args))
# EOF


"""
resolve_target_layers
Returns explicit layers or one repeated hook-module path per transformer block.

INPUT:
    - cfg: Cfg -> extraction configuration
    - ann: vidANN -> initialized video model wrapper

OUTPUT:
    - target_layers: list[str] -> ordered model module paths
"""
def resolve_target_layers(cfg, ann):
    if cfg.layers is not None:
        return cfg.layers
    # end if explicit layers
    if cfg.architecture != "transformer":
        raise ValueError(
            "--layers is required for CNN video models because block paths "
            "cannot be inferred generically."
        )
    # end if CNN

    num_hidden_layers = getattr(
        getattr(ann.model, "config", None),
        "num_hidden_layers",
        None,
    )
    if num_hidden_layers is None:
        raise ValueError(
            "The model config has no num_hidden_layers; pass --layers explicitly."
        )
    # end if missing num_hidden_layers
    return [
        f"encoder.layer.{block_index}.{cfg.hook_module}"
        for block_index in range(num_hidden_layers)
    ]
# EOF


"""
main
Initializes vidANN and extracts frame-aligned sliding-window video features.

OUTPUT:
    - None
"""
def main():
    cfg = parse_args()
    stimuli_dir = Path(
        cfg.stimuli_dir or Path(paths["data_path"]) / "stimuli"
    )
    output_dir = Path(
        cfg.output_dir or Path(paths["data_path"]) / "models"
    )
    model_source = get_model_source(cfg.model_name, cfg.repo_url)
    device = cfg.device or get_device()
    dtype = DTYPES[cfg.dtype]

    video_paths = list_video_paths(
        stimuli_dir,
        video_pattern=cfg.video_pattern,
        max_videos=cfg.max_videos,
    )
    ann = vidANN(
        model_name=cfg.model_name,
        pkg=cfg.pkg,
        img_size=cfg.img_size,
        num_frames=cfg.window_size_frames,
        architecture=cfg.architecture,
        relevant_layers=[],
        pooling=cfg.pooling,
        last_frame=cfg.last_frame,
        tubelet_size=cfg.tubelet_size,
        weights_type=cfg.weights_type,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        repo_url=model_source,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
        device=device,
    )
    target_layers = resolve_target_layers(cfg, ann)
    ann.set_relevant_layers(target_layers)

    if (
        ann.architecture == "transformer"
        and ann.last_frame
        and cfg.window_size_frames % ann.tubelet_size
    ):
        raise ValueError(
            "window_size_frames must be divisible by the model tubelet_size "
            "when last_frame=True."
        )
    # end if incomplete transformer tubelet

    preprocessor = build_video_preprocessor(
        cfg.pkg, model_source, cfg.img_size,
    )
    model_forward_kwargs = {}
    if cfg.skip_predictor:
        model_forward_kwargs["skip_predictor"] = True
    # end if skip_predictor

    print(ann)
    print(f"Videos: {len(video_paths)} from {stimuli_dir}")
    print(f"Layers: {len(target_layers)}")
    print(
        f"Sliding windows: {cfg.window_size_frames} retained frames, "
        f"stride 1, batch size {cfg.batch_size}"
    )
    print(
        f"Temporal pooling: "
        f"{'last step' if cfg.last_frame else 'complete window'}"
    )
    print(f"Output directory: {output_dir}")

    output_paths = extract_sliding_video_dataset_features(
        ann,
        preprocessor,
        video_paths,
        output_dir,
        cfg.dataset_name,
        model_source,
        window_size=cfg.window_size_frames,
        batch_size=cfg.batch_size,
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        preprocess_chunk_size=cfg.preprocess_chunk_size,
        dtype=dtype,
        model_input_name=cfg.model_input_name,
        model_forward_kwargs=model_forward_kwargs,
        compression=cfg.compression,
        overwrite=cfg.overwrite,
    )
    print(f"Completed {len(output_paths)} layer files for {cfg.model_name}")
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
