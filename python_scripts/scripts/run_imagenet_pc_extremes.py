import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
import yaml


# Score every ImageNet validation image on the leading PCs of each layer and
# keep, per PC, the indices of the most extreme images, for the image grids of
# PCs_visualization.ipynb. Components come from run_model_zoo_frame_pca.py.
# Example from the project root:
# .venv/bin/python python_scripts/scripts/run_imagenet_pc_extremes.py \
#     --model_names alexnet ijepa_vith14_1k dino_v3_h convnext_v2_huge


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.frame_pca import (
    load_frame_pcas,
    pc_extremes_path,
    save_pc_extremes,
    score_images_on_leading_pcs,
    select_imagenet_val_images,
)
from image_processing.model_zoo import MODEL_ZOO, get_model_spec
from image_processing.video_feature_extraction import build_frame_preprocessor
from useful_stuff.general_utils import get_device
from useful_stuff.image_processing.computational_models import imgANN


@dataclass
class Cfg:
    model_names: list[str]
    # Dataset label and size of the stored components to score images on.
    components_dataset: str = "imagenet_val"
    n_components: int = 1000
    # Leading PCs scored per layer, and images kept at each end of each PC.
    n_pcs: int = 20
    n_extreme: int = 100
    # 50 per class is the whole validation set, the images the PCs were fit on.
    n_imagenet_per_class: int = 50
    seed: int = 0
    # Full-width float32 activations held at once, which sets the chunk size.
    chunk_gb: float = 2.0
    device: str | None = None
    overwrite: bool = False
# EOF


"""
parse_args
Parse the models and the scoring parameters.

OUTPUT:
    - cfg: Cfg -> validated configuration
"""
def parse_args() -> Cfg:
    parser = argparse.ArgumentParser(
        description=(
            "Score ImageNet validation images on the leading PCs of every "
            "layer and store the most extreme images of each PC."
        )
    )
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--components_dataset", default=Cfg.components_dataset)
    parser.add_argument("--n_components", type=int, default=Cfg.n_components)
    parser.add_argument("--n_pcs", type=int, default=Cfg.n_pcs)
    parser.add_argument("--n_extreme", type=int, default=Cfg.n_extreme)
    parser.add_argument(
        "--n_imagenet_per_class", type=int, default=Cfg.n_imagenet_per_class,
    )
    parser.add_argument("--seed", type=int, default=Cfg.seed)
    parser.add_argument("--chunk_gb", type=float, default=Cfg.chunk_gb)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    unknown_names = [name for name in args.model_names if name not in MODEL_ZOO]
    if unknown_names:
        parser.error(
            f"Unknown model names {unknown_names}; available: {sorted(MODEL_ZOO)}"
        )
    # end if unknown_names
    return Cfg(**vars(args))
# EOF


"""
build_ann
Load one registry model with the unpooled hooks its components were fit on.

INPUT:
    - spec: ModelSpec -> registry entry
    - device: str -> inference device

OUTPUT:
    - ann: imgANN -> model wrapper with pooling="all"
"""
def build_ann(spec, device):
    ann = imgANN(
        model_name=spec.model_name,
        pkg=spec.pkg,
        img_size=spec.img_size,
        relevant_layers=spec.layers,
        pooling="all",
        dtype=torch.float32,
        attn_implementation=spec.attn_implementation,
        repo_url=spec.repo_url,
        device=device,
    )
    if spec.submodule is not None:
        ann.set_model(getattr(ann.get_model(), spec.submodule))
        ann.set_relevant_layers(spec.layers)
    # end if only one tower is hooked
    # Read by create_layer_hooks to keep only the patch tokens.
    ann.n_prefix_tokens = spec.n_prefix_tokens
    return ann
# EOF


"""
main
Score the ImageNet validation images for every requested model.
"""
def main() -> None:
    cfg = parse_args()
    device = cfg.device or get_device()
    image_paths = select_imagenet_val_images(
        paths["imagenet_path"], cfg.n_imagenet_per_class, seed=cfg.seed,
    )
    print(f"{len(image_paths)} ImageNet validation images", flush=True)

    for model_name in cfg.model_names:
        spec = get_model_spec(model_name)
        save_path = pc_extremes_path(
            paths, model_name, cfg.components_dataset, cfg.n_components,
            "imagenet_val",
        )
        if save_path.exists() and not cfg.overwrite:
            print(f"skipping {model_name}: {save_path} exists", flush=True)
            continue
        # end if already scored
        print(f"\n===== {model_name} ({len(spec.layers)} layers) =====", flush=True)
        projections = load_frame_pcas(
            paths, model_name, spec.layers, cfg.components_dataset,
            cfg.n_components,
        )
        ann = build_ann(spec, device)
        preprocessor = build_frame_preprocessor(
            spec.pkg, spec.repo_url, spec.img_size,
        )
        scores = score_images_on_leading_pcs(
            ann, spec.layers, projections, image_paths, preprocessor, spec.pkg,
            cfg.n_pcs, spec.batch_size, chunk_bytes=cfg.chunk_gb * 1e9,
        )
        save_pc_extremes(
            save_path, scores, image_paths, cfg.n_extreme,
            {
                "model_name": model_name,
                "components_dataset": cfg.components_dataset,
                "n_components": cfg.n_components,
                "n_pcs": cfg.n_pcs,
                "imagenet_path": str(paths["imagenet_path"]),
                "n_imagenet_per_class": cfg.n_imagenet_per_class,
                "seed": cfg.seed,
            },
        )
        print(f"saved {save_path}", flush=True)
        del ann, projections, scores
        if device == "mps":
            torch.mps.empty_cache()
        # end if mps
    # end for model_name
    return None
# EOF


if __name__ == "__main__":
    main()
# EOF
