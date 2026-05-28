import os, sys, yaml
from pathlib import Path
import numpy as np 
from sklearn.decomposition import IncrementalPCA
import joblib

ENV = os.getenv("MY_ENV", "tiziano_local")
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["useful_stuff_path"])
sys.path.append(paths["src_path"])

from useful_stuff.image_processing.computational_models import imgANN, pool_features
from useful_stuff.image_processing.dim_redu import compute_img_ipca
from useful_stuff.general_utils import get_device, print_wise


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
def ipca_imagenet_wrapper(paths, rank, target_layers, ann, loader, n_components, batch_size, device=get_device()):
    # Build all expected output paths and keep only layers whose files do not exist yet.
    save_paths = {
        layer: save_imagenet_val_ipca(paths, ann.model_name, layer, n_components, ann.pooling)
        for layer in target_layers
    }
    missing_layers = [layer for layer in target_layers if not save_paths[layer].exists()]
    existing_layers = [layer for layer in target_layers if save_paths[layer].exists()]

    if existing_layers:
        print_wise(f"Skipping existing {ann.model_name} layers: {', '.join(existing_layers)}")
    if not missing_layers:
        print_wise(f"All requested {ann.model_name} iPCA files already exist; skipping iPCA.")
        return {}

    # Register hooks only for layers that still need to be computed.
    ann.features = {}
    ann.create_forward_hook(missing_layers)

    # Create one iPCA object per missing layer, capped by that layer's feature dimensionality.
    ipcas = {
        layer: IncrementalPCA(
            n_components=min(n_components, np.prod(ann.get_layer_output_shape(layer))),
            batch_size=batch_size,
        )
        for layer in missing_layers
    }

    # Fit iPCA objects by streaming ImageNet validation activations batch by batch.
    ipcas = compute_img_ipca(ann, loader, ipcas, device)

    # Save only the newly fitted layer-specific iPCA objects.
    for layer in missing_layers:
        save_paths[layer].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ipcas[layer], save_paths[layer])
    return ipcas
# EOF
