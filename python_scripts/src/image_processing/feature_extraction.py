import os, sys, yaml
from pathlib import Path
import numpy as np 
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
save_imagenet_val_pca_srp
Builds the save path for ImageNet validation SRP-to-PCA components from a model and layer name.
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
    save_dir = Path(paths["data_path"]) / "models" / "imagenet_components"
    file_name = f"{model_name}_{layer_name}_imagenet_val_{n_srp_components}_to_{n_pca_components}_srp_to_pca_components_{pooling}pool.pkl"
    save_path = save_dir / file_name
    return save_path
# EOF

"""
save_imagenet_val_srp
Builds the save path for ImageNet validation SRP components from a model and layer name.
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
    save_dir = Path(paths["data_path"]) / "models" / "imagenet_components"
    file_name = f"{model_name}_{layer_name}_imagenet_val_{n_srp_components}_srp_components_{pooling}pool.pkl"
    save_path = save_dir / file_name
    return save_path
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
    # Build all expected output paths and keep only layers whose files do not exist yet.
    save_paths_pca = {
        layer: save_imagenet_val_pca_srp(paths, ann.model_name, layer, n_srp_components, n_pca_components, ann.pooling)
        for layer in target_layers
    }
    save_paths_srp = {
        layer: save_imagenet_val_srp(paths, ann.model_name, layer, n_srp_components, ann.pooling)
        for layer in target_layers
    }
    missing_layers = [layer for layer in target_layers if not save_paths_pca[layer].exists()]
    existing_layers = [layer for layer in target_layers if save_paths_pca[layer].exists()]

    if existing_layers:
        print_wise(f"Skipping existing {ann.model_name} layers: {', '.join(existing_layers)}", rank=rank)
    if not missing_layers:
        print_wise(f"All requested {ann.model_name} srp-PCA files already exist; skipping srp-PCA.", rank=rank)
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

    # Fit PCA objects on SRP-transformed ImageNet validation activations.
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
