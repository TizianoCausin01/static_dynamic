import os, sys, yaml
import gc
import time
from pathlib import Path
import numpy as np 
import torch
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.random_projection import SparseRandomProjection
import joblib

ENV = os.getenv("MY_ENV", "tiziano_local")
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append(paths["src_path"])

from useful_stuff.image_processing.computational_models import imgANN, pool_features
from useful_stuff.image_processing.dim_redu import compute_img_ipca, compute_img_srp
from useful_stuff.general_utils import get_device, print_wise, convert_dtype_by_name


"""
save_imagenet_val_ipca
Builds the save path for ImageNet validation iPCA components from a model and layer name.
INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> ANN layer name used to compute activations
    - n_components: int -> number of iPCA components saved in the file

OUTPUT:
    - save_path: Path -> path where the ImageNet validation iPCA object is saved
"""
def save_imagenet_val_ipca(paths, model_name, layer_name, n_components, pooling):
    save_dir = Path(paths["data_path"]) / "models" / "imagenet_components"
    file_name = f"{model_name}_{layer_name}_imagenet_val_{n_components}components_{pooling}pool.pkl"
    save_path = save_dir / file_name
    return save_path
# EOF

"""
ipca_imagenet_wrapper
Fits and saves ImageNet validation iPCA objects for target layers that have not been computed yet.
INPUT:
    - paths: dict[str: str]
    - rank: int
    - ann: imgANN -> model wrapper used to extract target-layer activations
    - loader: DataLoader -> ImageNet validation image batches
    - target_layers: list[str] -> ANN layer names where activations are extracted
    - n_components: int -> number of iPCA components to fit for each layer
    - batch_size: int -> iPCA batch size used for each IncrementalPCA object

OUTPUT:
    - ipcas: dict[str, IncrementalPCA] -> newly fitted iPCA objects keyed by layer name
"""
def ipca_imagenet_wrapper(paths, rank, target_layers, ann, loader, n_components, batch_size, sub_batch_size=None, device=get_device()):
    # Build all expected output paths and keep only layers whose files do not exist yet.
    save_paths = {
        layer: save_imagenet_val_ipca(paths, ann.model_name, layer, n_components, ann.pooling)
        for layer in target_layers
    }
    missing_layers = [layer for layer in target_layers if not save_paths[layer].exists()]
    existing_layers = [layer for layer in target_layers if save_paths[layer].exists()]

    if existing_layers:
        print_wise(f"Skipping existing {ann.model_name} layers: {', '.join(existing_layers)}", rank=rank)
    if not missing_layers:
        print_wise(f"All requested {ann.model_name} iPCA files already exist; skipping iPCA.", rank=rank)
        return {}

    # Register hooks only for layers that still need to be computed.
    ann.features = {}
    ann.create_forward_hook(layer_names=missing_layers)

    # Create one iPCA object per missing layer, capped by that layer's feature dimensionality.
    ipcas = {
        layer: IncrementalPCA(
            n_components=min(n_components, np.prod(ann.get_layer_output_shape(layer))),
            batch_size=batch_size,
        )
        for layer in missing_layers
    }

    # Fit iPCA objects by streaming ImageNet validation activations batch by batch.
    ipcas = compute_img_ipca(ann, loader, ipcas, device, sub_batch_size=sub_batch_size, rank=rank)

    # Save only the newly fitted layer-specific iPCA objects.
    for layer in missing_layers:
        save_paths[layer].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ipcas[layer], save_paths[layer])
        print_wise(f"PCs saved at {save_paths[layer]}", rank=rank)
    return ipcas
# EOF


"""
save_dataset_pca_srp
Builds the save path for dataset-specific SRP-to-PCA components from a model and layer name.
INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> ANN layer name used to compute activations
    - PCs_dataset: str -> dataset used to fit the SRP and PCA objects
    - n_srp_components: int -> number of SRP components saved in the first reduction step
    - n_pca_components: int -> number of PCA components saved after SRP
    - pooling: str -> pooling strategy used to compute activations

OUTPUT:
    - save_path: Path -> path where the dataset-specific SRP-to-PCA object is saved
"""
def save_dataset_pca_srp(paths, model_name, layer_name, PCs_dataset, n_srp_components, n_pca_components, pooling):
    save_dir = Path(paths["data_path"]) / "models" / "imagenet_components"
    file_name = f"{model_name}_{layer_name}_{PCs_dataset}_{n_srp_components}_to_{n_pca_components}_srp_to_pca_components_{pooling}pool.pkl"
    save_path = save_dir / file_name
    return save_path
# EOF

"""
save_dataset_srp
Builds the save path for dataset-specific SRP components from a model and layer name.
INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> ANN layer name used to compute activations
    - PCs_dataset: str -> dataset used to fit the SRP object
    - n_srp_components: int -> number of SRP components saved in the file
    - pooling: str -> pooling strategy used to compute activations

OUTPUT:
    - save_path: Path -> path where the dataset-specific SRP object is saved
"""
def save_dataset_srp(paths, model_name, layer_name, PCs_dataset, n_srp_components, pooling):
    save_dir = Path(paths["data_path"]) / "models" / "imagenet_components"
    file_name = f"{model_name}_{layer_name}_{PCs_dataset}_{n_srp_components}_srp_components_{pooling}pool.pkl"
    save_path = save_dir / file_name
    return save_path
# EOF

"""
save_imagenet_val_pca_srp
Builds the legacy ImageNet validation SRP-to-PCA component save path.
INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> ANN layer name used to compute activations
    - n_srp_components: int -> number of SRP components saved in the first reduction step
    - n_pca_components: int -> number of PCA components saved after SRP
    - pooling: str -> pooling strategy used to compute activations

OUTPUT:
    - save_path: Path -> path where the ImageNet validation SRP-to-PCA object is saved
"""
def save_imagenet_val_pca_srp(paths, model_name, layer_name, n_srp_components, n_pca_components, pooling):
    return save_dataset_pca_srp(paths, model_name, layer_name, "imagenet_val", n_srp_components, n_pca_components, pooling)
# EOF

"""
save_imagenet_val_srp
Builds the legacy ImageNet validation SRP component save path.
INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> ANN layer name used to compute activations
    - n_srp_components: int -> number of SRP components saved in the file
    - pooling: str -> pooling strategy used to compute activations

OUTPUT:
    - save_path: Path -> path where the ImageNet validation SRP object is saved
"""
def save_imagenet_val_srp(paths, model_name, layer_name, n_srp_components, pooling):
    return save_dataset_srp(paths, model_name, layer_name, "imagenet_val", n_srp_components, pooling)
# EOF


"""
srp_pca_dataset_wrapper
Fits and saves dataset-specific SRP and PCA objects for target layers that have not been computed yet.
INPUT:
    - paths: dict[str: str]
    - rank: int
    - target_layers: list[str] -> ANN layer names where activations are extracted
    - ann: imgANN -> model wrapper used to extract target-layer activations
    - loader: DataLoader -> image batches from the dataset used to fit PCs
    - PCs_dataset: str -> dataset name used in the saved SRP/PCA filenames
    - n_srp_components: int -> number of SRP components to fit for each layer
    - n_pca_components: int -> number of PCA components to fit after SRP for each layer
    - device: torch.device -> device used to stream activations through the ANN

OUTPUT:
    - None
"""
def srp_pca_dataset_wrapper(paths, rank, target_layers, ann, loader, PCs_dataset, n_srp_components, n_pca_components, device=get_device()):
    if isinstance(target_layers, str):
        target_layers = [target_layers]

    # Build all expected output paths and keep only layers whose files do not exist yet.
    save_paths_pca = {
        layer: save_dataset_pca_srp(paths, ann.model_name, layer, PCs_dataset, n_srp_components, n_pca_components, ann.pooling)
        for layer in target_layers
    }
    save_paths_srp = {
        layer: save_dataset_srp(paths, ann.model_name, layer, PCs_dataset, n_srp_components, ann.pooling)
        for layer in target_layers
    }
    missing_layers = [layer for layer in target_layers if not save_paths_pca[layer].exists()]
    existing_layers = [layer for layer in target_layers if save_paths_pca[layer].exists()]

    if existing_layers:
        print_wise(f"Skipping existing {ann.model_name} {PCs_dataset} layers: {', '.join(existing_layers)}", rank=rank)
    if not missing_layers:
        print_wise(f"All requested {ann.model_name} {PCs_dataset} srp-PCA files already exist; skipping srp-PCA.", rank=rank)
        return {}

    # Register hooks only for layers that still need to be computed.
    ann.features = {}
    ann.create_forward_hook(layer_names=missing_layers)
    n_features_per_layer = {layer: int(np.prod(ann.get_layer_output_shape(layer))) for layer in missing_layers}
    # Create one SRP and one PCA object per missing layer, capped by valid feature dimensions.
    srps = {
        layer: SparseRandomProjection(
            n_components=min(n_srp_components, n_features_per_layer[layer]),
        ).fit(np.zeros((1, n_features_per_layer[layer]), dtype=convert_dtype_by_name(ann.dtype, "numpy")))
        for layer in missing_layers
    }
    pcas = {
        layer: PCA(
            n_components=min(n_pca_components, srps[layer].n_components),
        )
        for layer in missing_layers
    }

    # Fit PCA objects on SRP-transformed activations from the requested PC dataset.
    f_redu = compute_img_srp(ann, loader, srps, device, rank=rank)
    for layer, pca in pcas.items():
        pca.fit(f_redu[layer])
    # Save only the newly fitted layer-specific SRP and PCA objects.
    for layer in missing_layers:
        save_paths_pca[layer].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(srps[layer], save_paths_srp[layer])
        joblib.dump(pcas[layer], save_paths_pca[layer])
        print_wise(f"SRPs saved at {save_paths_srp[layer]}", rank=rank)
        print_wise(f"PCs saved at {save_paths_pca[layer]}", rank=rank)
    return None
# EOF

"""
srp_pca_imagenet_wrapper
Fits and saves ImageNet validation SRP and PCA objects for target layers that have not been computed yet.
INPUT:
    - paths: dict[str: str]
    - rank: int
    - target_layers: list[str] -> ANN layer names where activations are extracted
    - ann: imgANN -> model wrapper used to extract target-layer activations
    - loader: DataLoader -> ImageNet validation image batches
    - n_srp_components: int -> number of SRP components to fit for each layer
    - n_pca_components: int -> number of PCA components to fit after SRP for each layer
    - device: torch.device -> device used to stream activations through the ANN

OUTPUT:
    - None
"""
def srp_pca_imagenet_wrapper(paths, rank, target_layers, ann, loader, n_srp_components, n_pca_components, device=get_device()):
    return srp_pca_dataset_wrapper(paths, rank, target_layers, ann, loader, "imagenet_val", n_srp_components, n_pca_components, device=device)
# EOF

"""
save_srp_pca_projected_features
Builds the save path for low-dimensional features projected onto a dataset-specific SRP/PCA space.
INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> ANN layer name used to compute activations
    - dataset_name: str -> dataset being projected
    - PCs_dataset: str -> dataset used to fit the SRP and PCA objects
    - n_srp_components: int -> number of SRP components used before PCA
    - n_pca_components: int -> number of PCA components used for the final projection
    - pooling: str -> pooling strategy used to compute activations

OUTPUT:
    - save_path: Path -> path where the projected low-dimensional features are saved
"""
def save_srp_pca_projected_features(paths, model_name, layer_name, dataset_name, PCs_dataset, n_srp_components, n_pca_components, pooling):
    save_dir = Path(paths["data_path"]) / "models"
    file_name = f"{model_name}_{layer_name}_{dataset_name}_{n_srp_components}_to_{n_pca_components}_srp_to_pca_{PCs_dataset}_components_{pooling}pool.npz"
    save_path = save_dir / file_name
    return save_path
# EOF

"""
get_images_from_batch
Extracts an image tensor from common DataLoader batch formats.
INPUT:
    - batch: Tensor | tuple | list | dict -> DataLoader batch

OUTPUT:
    - images: Tensor -> image batch tensor
"""
def get_images_from_batch(batch):
    if isinstance(batch, dict):
        for key in ("images", "image", "pixel_values"):
            if key in batch:
                return batch[key]
        raise KeyError("Could not find images in dataloader batch dict.")
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch
# EOF

"""
srp_pca_project_dataset_wrapper
Projects dataset features into an SRP/PCA space fitted on another dataset and saves the low-dimensional features.
INPUT:
    - paths: dict[str: str]
    - rank: int
    - target_layers: list[str] -> ANN layer names where activations are extracted
    - ann: imgANN -> model wrapper used to extract target-layer activations
    - loader: DataLoader -> image batches from the dataset being projected
    - dataset_name: str -> dataset being projected
    - PCs_dataset: str -> dataset used to fit the SRP and PCA objects
    - n_srp_components: int -> number of SRP components used before PCA
    - n_pca_components: int -> number of PCA components used for the final projection
    - device: torch.device -> device used to stream activations through the ANN

OUTPUT:
    - f_low_dim: dict[str, np.ndarray] -> saved low-dimensional features keyed by layer name
"""
def srp_pca_project_dataset_wrapper(paths, rank, target_layers, ann, loader, dataset_name, PCs_dataset, n_srp_components, n_pca_components, device=get_device()):
    if isinstance(target_layers, str):
        target_layers = [target_layers]

    save_paths = {
        layer: save_srp_pca_projected_features(
            paths, ann.model_name, layer, dataset_name, PCs_dataset, n_srp_components, n_pca_components, ann.pooling
        )
        for layer in target_layers
    }
    save_paths_pca = {
        layer: save_dataset_pca_srp(paths, ann.model_name, layer, PCs_dataset, n_srp_components, n_pca_components, ann.pooling)
        for layer in target_layers
    }
    save_paths_srp = {
        layer: save_dataset_srp(paths, ann.model_name, layer, PCs_dataset, n_srp_components, ann.pooling)
        for layer in target_layers
    }
    missing_layers = [layer for layer in target_layers if not save_paths[layer].exists()]
    existing_layers = [layer for layer in target_layers if save_paths[layer].exists()]

    if existing_layers:
        print_wise(f"Skipping existing {ann.model_name} {dataset_name} layers: {', '.join(existing_layers)}", rank=rank)
    if not missing_layers:
        print_wise(f"All requested {ann.model_name} {dataset_name} SRP-PCA features already exist; skipping projection.", rank=rank)
        return {}

    missing_components = [
        layer
        for layer in missing_layers
        if not save_paths_srp[layer].exists() or not save_paths_pca[layer].exists()
    ]
    if missing_components:
        missing_msg = ", ".join(missing_components)
        raise FileNotFoundError(f"Missing {PCs_dataset} SRP/PCA component files for layers: {missing_msg}")

    ann.features = {}
    ann.create_forward_hook(layer_names=missing_layers)
    ann.model.eval()

    srps = {layer: joblib.load(save_paths_srp[layer]) for layer in missing_layers}
    pcas = {layer: joblib.load(save_paths_pca[layer]) for layer in missing_layers}
    f_low_dim = {layer: [] for layer in missing_layers}

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            st_forw = time.time()
            images = get_images_from_batch(batch).to(device)
            if getattr(ann, "pkg", None) == "hf":
                ann.model(pixel_values=images)
            else:
                ann.model(images)
            end_forw = time.time()
            print_wise(f"forward took {end_forw - st_forw}", rank=rank)

            for layer in missing_layers:
                features = ann.features.get(layer)
                if features is None:
                    continue
                st_proj = time.time()
                features = features.detach().cpu().numpy()
                f_srp = srps[layer].transform(features)
                f_low_dim[layer].append(pcas[layer].transform(f_srp))
                ann.features[layer] = None
                end_proj = time.time()
                print_wise(f"srp-PCA projection {layer} took {end_proj - st_proj}\n shape {f_low_dim[layer][-1].shape}", rank=rank)

            del images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print_wise(f"Projected batch {idx}/{len(loader)-1} of {ann.model_name}: {missing_layers}", rank=rank)

    for layer in missing_layers:
        f_low_dim[layer] = np.concatenate(f_low_dim[layer], axis=0)
        save_paths[layer].parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_paths[layer], features=f_low_dim[layer])
        print_wise(f"SRP-PCA features saved at {save_paths[layer]}", rank=rank)
    return f_low_dim
# EOF

"""
srp_pca_backproject_imagenet_wrapper
Backward-compatible alias for srp_pca_project_dataset_wrapper.
"""
def srp_pca_backproject_imagenet_wrapper(paths, rank, target_layers, ann, loader, dataset_name, PCs_dataset, n_srp_components, n_pca_components, device=get_device()):
    return srp_pca_project_dataset_wrapper(
        paths, rank, target_layers, ann, loader, dataset_name, PCs_dataset,
        n_srp_components, n_pca_components, device=device
    )
# EOF
