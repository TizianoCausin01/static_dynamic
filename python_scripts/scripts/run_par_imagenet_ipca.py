import os, yaml, sys
from pathlib import Path
import argparse
import joblib
import torch
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from image_processing.feature_extraction import ipca_imagenet_wrapper, paths
from project_specific_utils.dataloader import imagenet_val_dataloader
from useful_stuff.image_processing.computational_models import imgANN, get_relevant_output_layers
from useful_stuff.general_utils import print_wise
from useful_stuff.parallel.parallel_funcs import parallel_setup, master_workers_queue

# e.g. to call it:
# mpiexec -np 4 run_imagenet_ipca.py --model_name='alexnet' --pkg=torchvision --pooling=all --n_components=10 --batch_size=100 --img_size=224

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
task_list = get_relevant_output_layers(cfg.model_name)
task_list = [[l,] for l in task_list] # because the function accepts only lists as layers 
_, rank, _ = parallel_setup()
if rank != 0:
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
    print_wise(ann, rank=rank)
    _, loader = imagenet_val_dataloader(
        paths,
        cfg.img_size,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    # end for l in ANN.relevant_layers:
else:
    ann = None
    loader = None
# end if rank != 0:

master_workers_queue(task_list, paths, ipca_imagenet_wrapper, *(ann, loader, cfg.n_components, cfg.batch_size)) 
