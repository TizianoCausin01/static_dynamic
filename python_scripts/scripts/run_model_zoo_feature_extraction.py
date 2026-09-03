import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import traceback

import torch
import yaml


# Example from the project root; output defaults to config[data_path]/models:
# .venv/bin/python python_scripts/scripts/run_model_zoo_feature_extraction.py \
#     --model_names alexnet convnext_base swin_base hiera_base


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.model_zoo import MODEL_ZOO, get_model_spec
from image_processing.video_feature_extraction import (
    build_frame_preprocessor,
    build_video_preprocessor,
    extract_sliding_video_dataset_features,
    extract_video_dataset_features,
    list_video_paths,
)
from useful_stuff.general_utils import get_device
from useful_stuff.image_processing import vidANN
from useful_stuff.image_processing.computational_models import imgANN


@dataclass
class Cfg:
    model_names: list[str] | None = None
    dataset_name: str = "static_dynamic"
    stimuli_dir: str | None = None
    output_dir: str | None = None
    max_videos: int | None = None
    video_pattern: str = "vid_*.mp4"
    dtype: str = "float32"
    device: str | None = None
    compression: str | None = None
    overwrite: bool = False
    # Keep the sweep alive when one model fails to download or hook.
    skip_failures: bool = True
# EOF


DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


"""
parse_args
Parse the model selection and shared extraction parameters for the sweep.

OUTPUT:
    - cfg: Cfg -> validated sweep configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frame-aligned features for several registered models "
            "using each model's own input geometry and hooked layers."
        )
    )
    parser.add_argument(
        "--model_names", nargs="+",
        help="Registry keys; defaults to the complete model zoo.",
    )
    parser.add_argument("--dataset_name", default=Cfg.dataset_name)
    parser.add_argument("--stimuli_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--video_pattern", default=Cfg.video_pattern)
    parser.add_argument("--dtype", choices=DTYPES, default=Cfg.dtype)
    parser.add_argument("--device")
    parser.add_argument("--compression", choices=("lzf", "gzip"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip_failures", action=argparse.BooleanOptionalAction,
        default=Cfg.skip_failures,
    )
    args = parser.parse_args()

    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max_videos must be a positive integer.")
    # end if invalid max_videos
    model_names = args.model_names or list(MODEL_ZOO)
    unknown_names = [name for name in model_names if name not in MODEL_ZOO]
    if unknown_names:
        parser.error(
            f"Unknown model names {unknown_names}; available: {sorted(MODEL_ZOO)}"
        )
    # end if unknown_names
    args.model_names = model_names
    return Cfg(**vars(args))
# EOF


"""
extract_image_model
Extract one per-frame feature vector for every video frame with an image model.

INPUT:
    - spec: ModelSpec -> registry entry describing the model and its layers
    - cfg: Cfg -> shared sweep configuration
    - video_paths: list[Path] -> stimulus videos in sorted order
    - output_dir: Path -> directory receiving the layer HDF5 files
    - device: str -> inference device
    - dtype: torch.dtype -> inference dtype

OUTPUT:
    - output_paths: dict[str, Path] -> layer files keyed by layer name
"""
def extract_image_model(spec, cfg, video_paths, output_dir, device, dtype):
    ann = imgANN(
        model_name=spec.model_name,
        pkg=spec.pkg,
        img_size=spec.img_size,
        relevant_layers=spec.layers,
        pooling=spec.pooling,
        dtype=dtype,
        attn_implementation=spec.attn_implementation,
        repo_url=spec.repo_url,
        device=device,
    )
    if spec.submodule is not None:
        # Hooking the tower directly also keeps the forward call to
        # pixel_values only, which the multi-tower wrapper would reject.
        ann.set_model(getattr(ann.get_model(), spec.submodule))
        ann.set_relevant_layers(spec.layers)
    # end if only one tower is hooked
    preprocessor = build_frame_preprocessor(
        spec.pkg, spec.repo_url, spec.img_size,
    )
    print(ann, flush=True)
    return extract_video_dataset_features(
        ann,
        preprocessor,
        video_paths,
        output_dir,
        cfg.dataset_name,
        spec.repo_url,
        spec.batch_size,
        dtype=dtype,
        compression=cfg.compression,
        overwrite=cfg.overwrite,
    )
# EOF


"""
extract_video_model
Extract one feature per frame with left-padded sliding windows of a video model.

INPUT:
    - spec: ModelSpec -> registry entry describing the model and its layers
    - cfg: Cfg -> shared sweep configuration
    - video_paths: list[Path] -> stimulus videos in sorted order
    - output_dir: Path -> directory receiving the layer HDF5 files
    - device: str -> inference device
    - dtype: torch.dtype -> inference dtype

OUTPUT:
    - output_paths: dict[str, Path] -> layer files keyed by layer name
"""
def extract_video_model(spec, cfg, video_paths, output_dir, device, dtype):
    ann = vidANN(
        model_name=spec.model_name,
        pkg=spec.pkg,
        img_size=spec.img_size,
        num_frames=spec.window_size_frames,
        architecture=spec.architecture,
        relevant_layers=spec.layers,
        pooling=spec.pooling,
        last_frame=True,
        dtype=dtype,
        attn_implementation=spec.attn_implementation,
        repo_url=spec.repo_url,
        device=device,
    )
    if spec.window_size_frames % ann.tubelet_size:
        raise ValueError(
            f"{spec.model_name}: window_size_frames must be divisible by "
            f"tubelet_size={ann.tubelet_size} when pooling the last step."
        )
    # end if incomplete tubelet
    preprocessor = build_video_preprocessor(
        spec.pkg, spec.repo_url, spec.img_size,
        mean=spec.normalization_mean, std=spec.normalization_std,
    )
    # Only V-JEPA-style encoders accept the predictor-skipping keyword.
    model_forward_kwargs = (
        {"skip_predictor": True} if spec.skip_predictor else {}
    )
    print(ann, flush=True)
    return extract_sliding_video_dataset_features(
        ann,
        preprocessor,
        video_paths,
        output_dir,
        cfg.dataset_name,
        spec.repo_url,
        window_size=spec.window_size_frames,
        batch_size=spec.batch_size,
        preprocess_chunk_size=spec.preprocess_chunk_size,
        dtype=dtype,
        model_input_name=spec.model_input_name,
        model_forward_kwargs=model_forward_kwargs,
        compression=cfg.compression,
        overwrite=cfg.overwrite,
    )
# EOF


"""
main
Run the registered models one after another over the same stimulus videos.
"""
def main() -> None:
    cfg = parse_args()
    stimuli_dir = Path(
        cfg.stimuli_dir
        or Path(paths["data_path"]) / "stimuli" / "static_dynamic_videos"
    )
    output_dir = Path(cfg.output_dir or Path(paths["data_path"]) / "models")
    device = cfg.device or get_device()
    dtype = DTYPES[cfg.dtype]
    video_paths = list_video_paths(
        stimuli_dir,
        video_pattern=cfg.video_pattern,
        max_videos=cfg.max_videos,
    )
    print(
        f"{len(video_paths)} videos from {stimuli_dir}\n"
        f"Output directory: {output_dir}\n"
        f"Models: {cfg.model_names}",
        flush=True,
    )

    failed_models = []
    for model_index, model_name in enumerate(cfg.model_names, start=1):
        spec = get_model_spec(model_name)
        if spec.modality == "baseline":
            # Pixel and optical-flow features come from their own scripts.
            print(f"skipping baseline entry {model_name}", flush=True)
            continue
        # end if baseline entry
        print(
            f"\n===== [{model_index}/{len(cfg.model_names)}] {model_name} "
            f"({spec.modality}, {len(spec.layers)} layers) =====",
            flush=True,
        )
        try:
            if spec.modality == "image":
                extract_image_model(
                    spec, cfg, video_paths, output_dir, device, dtype,
                )
            elif spec.modality == "video":
                extract_video_model(
                    spec, cfg, video_paths, output_dir, device, dtype,
                )
            else:
                raise ValueError(
                    f"{model_name}: modality must be 'image' or 'video'."
                )
            # end if spec.modality
            print(f"Completed {model_name}", flush=True)
        except Exception:
            if not cfg.skip_failures:
                raise
            # end if not cfg.skip_failures
            failed_models.append(model_name)
            traceback.print_exc()
            print(f"FAILED {model_name}; continuing.", flush=True)
        # end try
    # end for model_name

    if failed_models:
        print(f"\nModels that failed: {failed_models}", flush=True)
    # end if failed_models
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
