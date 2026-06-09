import os, yaml, sys
from pathlib import Path
import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.feature_extraction import srp_pca_project_dataset_wrapper
from useful_stuff.image_processing.computational_models import imgANN, get_relevant_output_layers
from useful_stuff.general_utils import print_wise, get_device
from useful_stuff.parallel.parallel_funcs import parallel_setup, master_workers_queue

# e.g. to call it:
# mpiexec -np 4 run_par_stimuli_srp_pca_extraction_projection.py --model_name='alexnet' --pkg=torchvision --folder_name=my_imagefolder --PCs_dataset=imagenet_val --pooling=all --n_srp_components=1000 --n_pca_components=10 --batch_size=100 --img_size=224

parser = argparse.ArgumentParser(
    description="Project activations from an ImageFolder under livingstone_lab/Stimuli into a dataset-specific SRP-to-PCA space."
)
parser.add_argument("--model_name", required=True)
parser.add_argument("--pkg", required=True)
parser.add_argument("--folder_name", required=True, help="ImageFolder name under paths['livingstone_lab']/Stimuli.")
parser.add_argument("--dataset_name", default=None, help="Name used in the saved projected feature filename. Defaults to folder_name.")
parser.add_argument("--PCs_dataset", default="imagenet_val", help="Dataset name used to locate the stored SRP/PCA objects.")
parser.add_argument("--n_srp_components", required=True, type=int)
parser.add_argument("--n_pca_components", required=True, type=int)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--pooling")
parser.add_argument("--img_size", required=True, type=int)
parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
parser.add_argument("--weights-type", default="DEFAULT", help="Weights type passed to imgANN.")
parser.add_argument("--repo_url", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
parser.add_argument("--revision", default=None, help="Optional model revision.")
parser.add_argument("--attn-implementation", default="sdpa", help="Optional attention implementation passed to imgANN.")


cfg = parser.parse_args()
folder_path = Path(paths["livingstone_lab"]) / "Stimuli" / cfg.folder_name
cfg.dataset_name = cfg.dataset_name or Path(cfg.folder_name).name

task_list = get_relevant_output_layers(cfg.model_name, cfg.pkg)
task_list = [[l,] for l in task_list] # because the function accepts only lists as layers 
_, rank, _ = parallel_setup()
if ENV == "dipsen_hpc" or ENV == "o2_cluster": # only if I'm on a cluster
    n_gpus = torch.cuda.device_count()
    if rank == 0: # rank 0 (the master) will have the cpu
        device = "cpu"
    else:
        if n_gpus == 0: # if there is no gpu available
            device = get_device()
        else: # otherwise evenly distribute the gpus among processes
            gpu_id = (rank - 1) % n_gpus # the remainder of the division yields the gpu
            device = torch.device(f"cuda:{gpu_id}") 
else: # if I'm on the local, I want mps to be detected instead
    device = get_device()

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
        device=device,
    )
    print_wise(ann, rank=rank)
    preprocess = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(
        root=folder_path,
        transform=preprocess,
        is_valid_file=lambda x: not x.endswith("Thumbs.db"),
        allow_empty=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print_wise(f"Loaded {len(dataset)} images from {folder_path}", rank=rank)
else:
    ann = None
    loader = None
# end if rank != 0:

master_workers_queue(
    task_list,
    paths,
    srp_pca_project_dataset_wrapper,
    *(ann, loader, cfg.dataset_name, cfg.PCs_dataset, cfg.n_srp_components, cfg.n_pca_components),
    **{"device": device},
) 
