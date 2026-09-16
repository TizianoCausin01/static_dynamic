import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import yaml


# Extract every video frame and project it on the components fitted by
# run_model_zoo_frame_pca.py. Example from the project root; output defaults to
# config[data_path]/models and carries the "PCpool" tag:
# .venv/bin/python python_scripts/scripts/run_model_zoo_pc_feature_extraction.py \
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
    group_layers_by_bytes,
    load_frame_pcas,
    pca_projection,
)
from image_processing.model_zoo import MODEL_ZOO, get_model_spec
from image_processing.video_feature_extraction import (
    build_frame_preprocessor,
    extract_video_dataset_features,
    list_video_paths,
)
from useful_stuff.general_utils import get_device
from useful_stuff.image_processing.computational_models import imgANN


# Pooling tag of the stored features, so the analysis notebooks reach them with
# the same list_video_feature_files / load_aligned_video_features calls. The
# component count is part of the tag: two runs that differ only in n_components
# would otherwise write to the same files, and the second would find every
# video already present and skip it.
POOLING_PREFIX = "PC"
# Stored components are float32 (see frame_pca.cast_pca_float32).
COMPONENT_ITEMSIZE = 4


@dataclass
class Cfg:
    model_names: list[str] | None = None
    dataset_name: str = "static_dynamic"
    # Dataset the components were fitted on; defaults to dataset_name.
    PCs_dataset: str | None = None
    n_components: int = 100
    stimuli_dir: str | None = None
    output_dir: str | None = None
    max_videos: int | None = None
    video_pattern: str = "vid_*.mp4"
    dtype: str = "float32"
    device: str | None = None
    compression: str | None = None
    # Components are held in RAM for every layer projected in one pass, so at
    # a high component count the layers have to be split over several passes.
    ram_budget_gb: float = 4.0
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
Parse the model selection and shared projection parameters for the sweep.

OUTPUT:
    - cfg: Cfg -> validated sweep configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frame-aligned features for several registered models and "
            "project them on their stored frame PCA components."
        )
    )
    parser.add_argument(
        "--model_names", nargs="+",
        help="Registry keys; defaults to the complete model zoo.",
    )
    parser.add_argument("--dataset_name", default=Cfg.dataset_name)
    parser.add_argument(
        "--PCs_dataset",
        help="Dataset used to locate the stored components; defaults to --dataset_name.",
    )
    parser.add_argument("--n_components", type=int, default=Cfg.n_components)
    parser.add_argument("--stimuli_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_videos", type=int)
    parser.add_argument("--video_pattern", default=Cfg.video_pattern)
    parser.add_argument("--dtype", choices=DTYPES, default=Cfg.dtype)
    parser.add_argument("--device")
    parser.add_argument("--compression", choices=("lzf", "gzip"))
    parser.add_argument(
        "--ram_budget_gb", type=float, default=Cfg.ram_budget_gb,
        help="Memory allowed for the components held in one projection pass.",
    )
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
    args.PCs_dataset = args.PCs_dataset or args.dataset_name
    return Cfg(**vars(args))
# EOF


"""
project_image_model
Extract every video frame with an image model and store its PC scores.

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
def project_image_model(spec, cfg, video_paths, output_dir, device, dtype):
    # The components were fitted on flattened activations, so the hook has to
    # deliver the same unpooled representation here.
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
    preprocessor = build_frame_preprocessor(
        spec.pkg, spec.repo_url, spec.img_size,
    )
    # One PCA holds n_components x n_features floats, so a deep model at a high
    # component count cannot keep every layer resident at once. The layers are
    # split into groups that fit, each group costing one pass over the videos.
    feature_sizes = {
        layer: int(np.prod(ann.get_layer_output_shape(layer)))
        for layer in spec.layers
    }
    layer_groups = group_layers_by_bytes(
        spec.layers, feature_sizes, cfg.n_components,
        cfg.ram_budget_gb * 1e9, COMPONENT_ITEMSIZE,
    )
    print(ann, flush=True)
    print(
        f"{len(spec.layers)} layers over {len(layer_groups)} projection "
        f"pass(es) at a {cfg.ram_budget_gb:.1f} GB component budget",
        flush=True,
    )

    output_paths = {}
    for group_index, layer_group in enumerate(layer_groups, start=1):
        print(
            f"\n--- projection pass {group_index}/{len(layer_groups)}: "
            f"{len(layer_group)} layers ---",
            flush=True,
        )
        # extract_video_dataset_features hooks whatever the ANN reports, so the
        # active layer list is narrowed to this group.
        ann.set_relevant_layers(layer_group)
        pcas = load_frame_pcas(
            paths, spec.model_name, layer_group, cfg.PCs_dataset,
            cfg.n_components,
        )
        feature_transforms = {
            layer: pca_projection(pcas[layer]) for layer in layer_group
        }
        output_paths.update(extract_video_dataset_features(
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
            feature_transforms=feature_transforms,
            pooling_name=f"{POOLING_PREFIX}{cfg.n_components}",
            extra_metadata={"n_components": cfg.n_components},
        ))
        del pcas, feature_transforms
    # end for layer_group
    return output_paths
# EOF


"""
main
Project the registered models one after another over the same stimulus videos.
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
        f"Components: {cfg.n_components} fitted on {cfg.PCs_dataset}\n"
        f"Models: {cfg.model_names}",
        flush=True,
    )

    failed_models = []
    for model_index, model_name in enumerate(cfg.model_names, start=1):
        spec = get_model_spec(model_name)
        if spec.modality != "image":
            # Only the per-frame image models have frame PCA components.
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
            project_image_model(
                spec, cfg, video_paths, output_dir, device, dtype,
            )
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
