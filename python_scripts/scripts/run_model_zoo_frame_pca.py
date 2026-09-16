import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import traceback

import torch
import yaml


# Fit the frame PCA that replaces token/spatial mean pooling. Example from the
# project root; components land in config[data_path]/models/frame_pca_components:
# .venv/bin/python python_scripts/scripts/run_model_zoo_frame_pca.py \
#     --model_names ijepa_vith14_1k dino_v3_h alexnet --n_components 100


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.frame_pca import (
    build_frame_buffer,
    fit_frame_pca_model,
    selected_frames_per_video,
)
from image_processing.model_zoo import MODEL_ZOO, get_model_spec
from image_processing.video_feature_extraction import (
    build_frame_preprocessor,
    list_video_paths,
)
from useful_stuff.general_utils import get_device
from useful_stuff.image_processing.computational_models import imgANN


@dataclass
class Cfg:
    model_names: list[str] | None = None
    dataset_name: str = "static_dynamic"
    stimuli_dir: str | None = None
    n_components: int = 100
    # Every third frame of the first 2.5 s at 60 fps: indices 0, 3, ..., 150.
    frame_stride: int = 3
    last_frame_index: int = 150
    max_videos: int | None = None
    video_pattern: str = "vid_*.mp4"
    # Rows per IncrementalPCA update; raised to n_components when smaller.
    pca_batch_size: int = 256
    # Memory allowed to solve one layer in RAM, which gives the exact PCA.
    # Layers above it are fitted incrementally from their staged file instead.
    ram_budget_gb: float = 8.0
    # Temporary disk allowed for one staged group of layers. Larger means more
    # layers share a forward pass, which is what makes the fit fast.
    disk_budget_gb: float = 16.0
    stage_dir: str | None = None
    seed: int = 0
    dtype: str = "float32"
    device: str | None = None
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
Parse the model selection and the frame-PCA fitting parameters.

OUTPUT:
    - cfg: Cfg -> validated fitting configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one PCA per hooked layer on the unpooled activations of a "
            "strided subsample of the stimulus video frames."
        )
    )
    parser.add_argument(
        "--model_names", nargs="+",
        help="Registry keys; defaults to the complete model zoo.",
    )
    parser.add_argument("--dataset_name", default=Cfg.dataset_name)
    parser.add_argument("--stimuli_dir")
    parser.add_argument("--n_components", type=int, default=Cfg.n_components)
    parser.add_argument("--frame_stride", type=int, default=Cfg.frame_stride)
    parser.add_argument(
        "--last_frame_index", type=int, default=Cfg.last_frame_index,
        help="Last source frame index kept per video, inclusive.",
    )
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--video_pattern", default=Cfg.video_pattern)
    parser.add_argument(
        "--pca_batch_size", type=int, default=Cfg.pca_batch_size,
    )
    parser.add_argument(
        "--ram_budget_gb", type=float, default=Cfg.ram_budget_gb,
    )
    parser.add_argument(
        "--disk_budget_gb", type=float, default=Cfg.disk_budget_gb,
        help="Temporary disk used to stage one group of layers.",
    )
    parser.add_argument(
        "--stage_dir", help="Directory for the staged activations.",
    )
    parser.add_argument("--seed", type=int, default=Cfg.seed)
    parser.add_argument("--dtype", choices=DTYPES, default=Cfg.dtype)
    parser.add_argument("--device")
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
fit_image_model
Fit and save the frame PCA of every hooked layer of one image model.

INPUT:
    - spec: ModelSpec -> registry entry describing the model and its layers
    - cfg: Cfg -> shared fitting configuration
    - video_paths: list[Path] -> stimulus videos in sorted order
    - device: str -> inference device
    - dtype: torch.dtype -> inference dtype

OUTPUT:
    - save_paths: dict[str, Path] -> component files keyed by layer name
"""
def fit_image_model(spec, cfg, video_paths, device, dtype):
    # pooling="all" flattens tokens and channels instead of averaging them,
    # which is exactly the representation the PCA has to be fitted on.
    ann = imgANN(
        model_name=spec.model_name,
        pkg=spec.pkg,
        img_size=spec.img_size,
        relevant_layers=spec.layers,
        pooling="all",
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
    print(ann, flush=True)

    preprocessor = build_frame_preprocessor(
        spec.pkg, spec.repo_url, spec.img_size,
    )
    max_frames = selected_frames_per_video(
        cfg.frame_stride, cfg.last_frame_index,
    )
    pixel_values, _ = build_frame_buffer(
        video_paths, preprocessor, spec.pkg, cfg.frame_stride, max_frames,
    )
    print(
        f"frame buffer {tuple(pixel_values.shape)} "
        f"({pixel_values.numel() * pixel_values.element_size() / 1e9:.2f} GB)",
        flush=True,
    )
    return fit_frame_pca_model(
        paths,
        ann,
        spec.layers,
        pixel_values,
        cfg.dataset_name,
        cfg.n_components,
        spec.batch_size,
        cfg.pca_batch_size,
        cfg.ram_budget_gb * 1e9,
        cfg.disk_budget_gb * 1e9,
        stage_dir=cfg.stage_dir,
        seed=cfg.seed,
        overwrite=cfg.overwrite,
    )
# EOF


"""
main
Fit the frame PCA of every requested model over the same stimulus videos.
"""
def main() -> None:
    cfg = parse_args()
    stimuli_dir = Path(
        cfg.stimuli_dir
        or Path(paths["data_path"]) / "stimuli" / "static_dynamic_videos"
    )
    device = cfg.device or get_device()
    dtype = DTYPES[cfg.dtype]
    video_paths = list_video_paths(
        stimuli_dir,
        video_pattern=cfg.video_pattern,
        max_videos=cfg.max_videos,
    )
    max_frames = selected_frames_per_video(
        cfg.frame_stride, cfg.last_frame_index,
    )
    print(
        f"{len(video_paths)} videos from {stimuli_dir}\n"
        f"{max_frames} frames per video (stride {cfg.frame_stride}, up to "
        f"source frame {cfg.last_frame_index}) -> "
        f"{max_frames * len(video_paths)} frames in total\n"
        f"Models: {cfg.model_names}",
        flush=True,
    )

    failed_models = []
    for model_index, model_name in enumerate(cfg.model_names, start=1):
        spec = get_model_spec(model_name)
        if spec.modality != "image":
            # The sliding-window video models pool over a temporal window as
            # well, so they need their own fitting path.
            print(
                f"skipping {model_name}: only image models are supported "
                f"(modality={spec.modality})",
                flush=True,
            )
            continue
        # end if not an image model
        print(
            f"\n===== [{model_index}/{len(cfg.model_names)}] {model_name} "
            f"({len(spec.layers)} layers) =====",
            flush=True,
        )
        try:
            fit_image_model(spec, cfg, video_paths, device, dtype)
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
