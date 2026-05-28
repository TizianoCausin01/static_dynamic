#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "python_scripts" / "src"
sys.path.append(str(SRC_PATH))

from image_processing.feature_extraction import ipca_imagenet_wrapper, paths
from project_specific_utils.dataloader import imagenet_val_dataloader
from useful_stuff.image_processing.computational_models import imgANN
from useful_stuff.general_utils import print_wise


# example to run it
# run_imagenet_ipca.py --model_name='alexnet' --pkg=torchvision --pooling=all --n_components=10 --batch_size=100 --img_size=224

parser = argparse.ArgumentParser(
    description="Fit and save ImageNet validation IncrementalPCA components for an imgANN model."
)
parser.add_argument("--model_name", required=True,)
parser.add_argument("--pkg", required=True)
parser.add_argument("--n_components", required=True, type=int)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--pooling")
parser.add_argument("--img_size", type=int)
parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
parser.add_argument("--weights-type", default="DEFAULT", help="Weights type passed to imgANN.")
parser.add_argument("--repo_url", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
parser.add_argument("--revision", default=None, help="Optional model revision.")
parser.add_argument("--attn-implementation", default='sdpa', help="Optional attention implementation passed to imgANN.",)


cfg = parser.parse_args()
print_wise(cfg)

ann = imgANN(
    model_name=cfg.model_name,
    pkg=cfg.pkg,
    img_size=cfg.img_size,
    pooling=cfg.pooling,
    weights_type=cfg.weights_type,
    dtype=torch.float32,
    attn_implementation=cfg.attn_implementation,
    repo_url=cfg.repo_url,
    revision=cfg.revision,
)
target_layers = ann.get_relevant_layers()
print_wise(f"Running ImageNet iPCA for {cfg.model_name}: {len(target_layers)} layers")

_, loader = imagenet_val_dataloader(
    paths,
    cfg.img_size,
    cfg.batch_size,
    num_workers=cfg.num_workers,
    shuffle=True,
)

ipca_imagenet_wrapper(
    paths=paths,
    rank=0,
    target_layers=target_layers,
    ann=ann,
    loader=loader,
    n_components=cfg.n_components,
    batch_size=cfg.batch_size,
)


