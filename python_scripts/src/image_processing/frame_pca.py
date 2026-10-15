"""
PCA over the complete (unpooled) activations of the stimulus video frames.

The default feature extraction averages ViT token / CNN spatial positions away
before storing anything, which throws out every spatially resolved component of
the representation. Here the flattened activation of every hooked layer is kept
instead, a PCA is fitted on a subsample of the stimulus frames, and the stored
components later replace mean pooling as the dimensionality reduction step.
Files produced from those components carry the "PCpool" tag, so the analysis
notebooks load them through the same helpers used for "meanpool" features.

Fitting streams the frames through the model once per layer group and either
holds the whole activation matrix in RAM (exact PCA) or updates an
IncrementalPCA chunk by chunk, depending on a caller-supplied memory budget.
Frames are always shuffled before fitting, so the incremental path never sees a
chunk made of a single video and its running estimate stays unbiased.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

from .video_feature_extraction import (
    clear_model_cache,
    create_layer_hooks,
    decode_selected_video_frames,
    forward_imgann,
    preprocess_frames,
)


# Activations are collected as float32, the dtype every PCA object is fitted on.
ACTIVATION_ITEMSIZE = 4
# sklearn solves in float64 regardless of the input dtype, so a layer being
# fitted also carries n_components x n_features of float64 state next to its
# activation chunk. For a ViT layer that is comparable to the chunk itself and
# has to be part of the memory budget.
PCA_STATE_ITEMSIZE = 8
# Staged activations are written as float16: they are only the input of a PCA
# fit, and halving the file halves both the temporary disk and the read cost.
STAGE_ITEMSIZE = 2
# PCA centers a working copy of the design matrix, so the exact path needs about
# twice the raw activation size; the incremental path only ever holds one chunk.
EXACT_PCA_MEMORY_FACTOR = 2


"""
save_frame_pca
Builds the save path of the frame PCA object fitted on one model layer.

INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layer_name: str -> hooked ANN layer name
    - dataset_name: str -> dataset the frames were taken from
    - n_components: int -> number of requested principal components

OUTPUT:
    - save_path: Path -> path where the layer-specific PCA object is saved
"""
def save_frame_pca(paths, model_name, layer_name, dataset_name, n_components):
    save_dir = Path(paths["data_path"]) / "models" / "frame_pca_components"
    file_name = (
        f"{model_name}_{layer_name}_{dataset_name}_"
        f"{n_components}components_PCpool.pkl"
    )
    return save_dir / file_name
# EOF


"""
load_frame_pcas
Loads the stored frame PCA object of every requested layer.

INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layers: list[str] -> hooked ANN layer names
    - dataset_name: str -> dataset the components were fitted on
    - n_components: int -> number of components used in the stored filenames

OUTPUT:
    - pcas: dict[str, PCA | IncrementalPCA] -> fitted objects keyed by layer
"""
def load_frame_pcas(paths, model_name, layers, dataset_name, n_components):
    pcas = {}
    missing_paths = []
    for layer in layers:
        pca_path = save_frame_pca(
            paths, model_name, layer, dataset_name, n_components,
        )
        if not pca_path.exists():
            missing_paths.append(pca_path)
            continue
        # end if the layer has no stored components
        pcas[layer] = joblib.load(pca_path)
    # end for layer
    if missing_paths:
        missing_names = ", ".join(path.name for path in missing_paths)
        raise FileNotFoundError(f"Missing frame PCA files: {missing_names}")
    # end if missing_paths
    return pcas
# EOF


"""
load_imagenet_pcas
Loads the ImageNet-validation IncrementalPCA object of every requested layer,
so the video frames can be projected on ImageNet axes instead of frame axes.

INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - layers: list[str] -> hooked ANN layer names
    - n_components: int -> number of components used in the stored filenames
    - pooling: str -> pooling tag of the fit; "all" = unpooled activations

OUTPUT:
    - pcas: dict[str, IncrementalPCA] -> float32 fitted objects keyed by layer
"""
def load_imagenet_pcas(paths, model_name, layers, n_components, pooling="all"):
    # Imported here: feature_extraction reads config.yaml when it is imported.
    from .feature_extraction import save_imagenet_val_ipca

    pca_paths = {
        layer: save_imagenet_val_ipca(
            paths, model_name, layer, n_components, pooling,
        )
        for layer in layers
    }
    missing_names = [path.name for path in pca_paths.values() if not path.exists()]
    if missing_names:
        raise FileNotFoundError(
            f"Missing ImageNet PCA files: {', '.join(missing_names)}"
        )
    # end if missing_names
    # The ImageNet fits were saved in float64; float32 halves the resident cost.
    return {
        layer: cast_pca_float32(joblib.load(pca_path))
        for layer, pca_path in pca_paths.items()
    }
# EOF


"""
truncate_pc_feature_file
Writes a copy of a PC-projected feature file keeping only its leading
components. PCs are ordered by explained variance, so the first k columns of
a 1000-component projection are the top-k subspace of the same fit.

INPUT:
    - source_path: Path -> "<...>_PC<n>pool.h5" feature file
    - target_path: Path -> output file
    - n_keep: int -> leading components kept
    - pooling_name: str -> pooling tag written in the output attributes

OUTPUT:
    - target_path: Path -> written file
"""
def truncate_pc_feature_file(source_path, target_path, n_keep, pooling_name):
    import h5py

    with h5py.File(source_path, "r") as source, h5py.File(target_path, "w") as target:
        target.attrs.update(source.attrs)
        target.attrs["pooling"] = pooling_name
        target.attrs["n_components"] = n_keep
        # One dataset per video: frames x components.
        for video_name, features in source.items():
            target.create_dataset(video_name, data=features[:, :n_keep])
            target[video_name].attrs.update(features.attrs)
        # end for video_name, features
    # end with h5py.File
    return target_path
# EOF


"""
pca_projection
Wraps one fitted PCA into the per-batch transform used during extraction.

INPUT:
    - pca: PCA | IncrementalPCA -> fitted components of a single layer

OUTPUT:
    - project: callable -> maps a (frames, features) batch to its PC scores
"""
def pca_projection(pca):
    def project(features):
        # float32 keeps the stored features in the same dtype as the meanpool
        # files, whatever dtype the fitted PCA object happens to carry.
        return pca.transform(features).astype(np.float32)
    # EOF
    return project
# EOF


"""
cast_pca_float32
Shrinks the fitted arrays of a PCA object to float32 in place.

sklearn solves in float64, so the components of a single ViT layer take a few
hundred MB and the projection step would have to hold that for every layer at
once. float32 halves both the file and that resident cost, and stays far below
the precision of the float32 activations the components are applied to.

INPUT:
    - pca: PCA | IncrementalPCA -> fitted object

OUTPUT:
    - pca: PCA | IncrementalPCA -> the same object with float32 arrays
"""
def cast_pca_float32(pca):
    # var_ only exists on IncrementalPCA, the rest is shared by both solvers.
    for attribute in (
        "components_", "mean_", "var_", "explained_variance_",
        "explained_variance_ratio_", "singular_values_",
    ):
        values = getattr(pca, attribute, None)
        if values is not None:
            setattr(pca, attribute, values.astype(np.float32))
        # end if the attribute exists
    # end for attribute
    return pca
# EOF


"""
selected_frames_per_video
Number of frames kept per video by a stride that stops at a last frame index.

INPUT:
    - frame_stride: int -> keep one frame every frame_stride source frames
    - last_frame_index: int -> last source frame index kept, inclusive

OUTPUT:
    - n_frames: int -> retained frames per video
"""
def selected_frames_per_video(frame_stride, last_frame_index):
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    # end if invalid frame_stride
    if last_frame_index < 0:
        raise ValueError("last_frame_index must be non-negative.")
    # end if invalid last_frame_index
    # The stride starts at frame 0, so the kept indices are 0, s, 2s, ...
    return last_frame_index // frame_stride + 1
# EOF


"""
build_frame_buffer
Decodes and preprocesses the PCA fitting frames of every video once.

Preprocessing is done here rather than per chunk so that shuffling the frames
costs nothing, and the buffer is stored in float16 to halve the resident memory
of the ~5k selected frames.

INPUT:
    - video_paths: list[Path] -> stimulus videos in sorted order
    - preprocessor: AutoImageProcessor | transforms.Compose -> frame preprocessor
    - pkg: str -> model package used by imgANN
    - frame_stride: int -> keep one frame every frame_stride source frames
    - max_frames: int -> retained frames per video
    - dtype: torch.dtype -> storage dtype of the buffer

OUTPUT:
    - pixel_values: torch.Tensor -> CPU tensor (videos * max_frames) x C x H x W
    - frame_sources: list[tuple[str, int]] -> video name and source frame index
"""
def build_frame_buffer(
        video_paths, preprocessor, pkg, frame_stride, max_frames,
        dtype=torch.float16,
        ):
    video_buffers = []
    frame_sources = []
    for video_index, video_path in enumerate(video_paths, start=1):
        frames, _, decoded_indices = decode_selected_video_frames(
            video_path, frame_stride=frame_stride, max_frames=max_frames,
        )
        if len(frames) < max_frames:
            raise RuntimeError(
                f"{video_path.name} yielded {len(frames)} frames with stride "
                f"{frame_stride}, expected {max_frames}."
            )
        # end if the video is too short for the requested frame selection
        video_buffers.append(
            preprocess_frames(frames, preprocessor, pkg, "cpu", dtype)
        )
        frame_sources.extend(
            (video_path.name, int(index)) for index in decoded_indices
        )
        print(
            f"[{video_index}/{len(video_paths)}] buffered {len(frames)} frames "
            f"from {video_path.name}",
            flush=True,
        )
    # end for video_path
    return torch.cat(video_buffers, dim=0), frame_sources
# EOF


"""
select_imagenet_val_images
Picks a class-balanced, seeded subset of the ImageNet validation images.

INPUT:
    - imagenet_path: str | Path -> ImageNet root readable by torchvision
    - n_per_class: int -> images kept for each of the 1000 classes
    - seed: int -> seed of the within-class selection

OUTPUT:
    - image_paths: list[Path] -> selected image files, class by class
"""
def select_imagenet_val_images(imagenet_path, n_per_class, seed=0):
    from torchvision.datasets import ImageNet

    # samples is a list of (path, class index) over the 50,000 val images.
    samples = ImageNet(root=imagenet_path, split="val").samples
    paths_by_class = {}
    for image_path, class_index in samples:
        paths_by_class.setdefault(class_index, []).append(image_path)
    # end for image_path, class_index
    rng = np.random.default_rng(seed)
    image_paths = []
    for class_index in sorted(paths_by_class):
        class_paths = sorted(paths_by_class[class_index])
        chosen = rng.choice(len(class_paths), size=n_per_class, replace=False)
        image_paths.extend(Path(class_paths[index]) for index in sorted(chosen))
    # end for class_index
    return image_paths
# EOF


"""
build_image_buffer
Loads, center-crops to a square, and preprocesses still images once.

The stimulus frames are square, while ImageNet images are not and some
processors (e.g. ViTImageProcessor) resize without cropping; the square crop
keeps the aspect ratio the same as the frames the components are applied to.

INPUT:
    - image_paths: list[Path] -> image files
    - preprocessor: AutoImageProcessor | transforms.Compose -> frame preprocessor
    - pkg: str -> model package used by imgANN
    - chunk_size: int -> images preprocessed per call
    - dtype: torch.dtype -> storage dtype of the buffer
    - verbose: bool -> print one progress line per chunk

OUTPUT:
    - pixel_values: torch.Tensor -> CPU tensor N x C x H x W
"""
def build_image_buffer(
        image_paths, preprocessor, pkg, chunk_size=256, dtype=torch.float16,
        verbose=True,
        ):
    from PIL import Image

    buffers = []
    for start in range(0, len(image_paths), chunk_size):
        images = []
        for image_path in image_paths[start:start + chunk_size]:
            image = np.asarray(Image.open(image_path).convert("RGB"))
            # Central square of side equal to the shortest edge.
            height, width = image.shape[:2]
            side = min(height, width)
            top, left = (height - side) // 2, (width - side) // 2
            images.append(image[top:top + side, left:left + side])
        # end for image_path
        buffers.append(preprocess_frames(images, preprocessor, pkg, "cpu", dtype))
        if not verbose:
            continue
        # end if progress lines are off
        print(
            f"buffered images {min(start + chunk_size, len(image_paths))}"
            f"/{len(image_paths)}",
            flush=True,
        )
    # end for start
    return torch.cat(buffers, dim=0)
# EOF


"""
layer_feature_sizes
Reads the flattened activation width of every hooked layer with one forward call.

INPUT:
    - ann: imgANN -> model wrapper created with pooling="all"
    - layers: list[str] -> layers to measure

OUTPUT:
    - feature_sizes: dict[str, int] -> flattened feature count keyed by layer
"""
def layer_feature_sizes(ann, layers):
    ann.features = {}
    create_layer_hooks(ann, layers)
    try:
        with torch.inference_mode():
            proxy = torch.zeros(
                1, 3, ann.img_size, ann.img_size,
                device=ann.device, dtype=ann.dtype,
            )
            forward_imgann(ann, proxy)
            feature_sizes = {}
            for layer in layers:
                features = ann.features.get(layer)
                if features is None:
                    raise RuntimeError(
                        f"Hook did not capture features for {layer}"
                    )
                # end if the hook stayed empty
                # pooling="all" flattens everything but the batch axis.
                feature_sizes[layer] = int(features.shape[1])
                ann.features[layer] = None
            # end for layer
        # end with torch.inference_mode()
    finally:
        ann.clear_hooks()
        clear_model_cache(ann.device)
    # end try
    return feature_sizes
# EOF


"""
group_layers_by_bytes
Splits the layers into groups whose staged activations fit one budget.

INPUT:
    - layers: list[str] -> layers to distribute, kept in depth order
    - feature_sizes: dict[str, int] -> flattened feature count per layer
    - n_rows: int -> activation rows stored for each layer
    - max_bytes: float -> budget shared by one group
    - itemsize: int -> bytes per stored activation value

OUTPUT:
    - layer_groups: list[list[str]] -> layers staged together in one pass
"""
def group_layers_by_bytes(layers, feature_sizes, n_rows, max_bytes, itemsize):
    layer_groups = []
    current_group = []
    current_bytes = 0
    for layer in layers:
        layer_bytes = n_rows * feature_sizes[layer] * itemsize
        # A layer that alone exceeds the budget still gets its own pass; there
        # is nothing smaller to fall back to.
        if current_group and current_bytes + layer_bytes > max_bytes:
            layer_groups.append(current_group)
            current_group = []
            current_bytes = 0
        # end if the group is full
        current_group.append(layer)
        current_bytes += layer_bytes
    # end for layer
    if current_group:
        layer_groups.append(current_group)
    # end if current_group
    return layer_groups
# EOF


"""
balanced_chunk_size
Rounds a requested chunk size up so that no trailing chunk is left too short.

IncrementalPCA refuses a batch with fewer rows than components, so a remainder
would otherwise be dropped from the fit. Spreading the frames evenly over the
same number of chunks keeps every frame in the estimate.

INPUT:
    - n_rows: int -> rows to split
    - requested: int -> chunk size asked for

OUTPUT:
    - chunk_size: int -> chunk size that divides n_rows without a short tail
"""
def balanced_chunk_size(n_rows, requested):
    n_chunks = max(1, n_rows // requested)
    return -(-n_rows // n_chunks)
# EOF


"""
forward_frame_chunk
Runs one chunk of preprocessed frames through the model and returns activations.

INPUT:
    - ann: imgANN -> model wrapper with hooks already registered
    - chunk_pixels: torch.Tensor -> CPU frames of this chunk, N x C x H x W
    - layers: list[str] -> hooked layers to collect
    - forward_batch_size: int -> frames per forward call

OUTPUT:
    - chunk_features: dict[str, np.ndarray] -> (N, features) float32 per layer
"""
def forward_frame_chunk(ann, chunk_pixels, layers, forward_batch_size):
    batch_features = {layer: [] for layer in layers}
    with torch.inference_mode():
        for start in range(0, chunk_pixels.shape[0], forward_batch_size):
            batch_pixels = chunk_pixels[start:start + forward_batch_size].to(
                device=ann.device, dtype=ann.dtype,
            )
            forward_imgann(ann, batch_pixels)
            for layer in layers:
                features = ann.features.get(layer)
                if features is None:
                    raise RuntimeError(
                        f"Hook did not capture features for {layer}"
                    )
                # end if the hook stayed empty
                batch_features[layer].append(
                    features.detach().float().cpu().numpy()
                )
                ann.features[layer] = None
            # end for layer
            del batch_pixels
            clear_model_cache(ann.device)
        # end for start
    # end with torch.inference_mode()
    return {
        layer: np.concatenate(batch_features[layer], axis=0)
        for layer in layers
    }
# EOF


"""
stage_group_activations
Writes the activations of a layer group to one memmap per layer.

This is what keeps the model from being run once per layer: a single pass over
the frames fills every layer of the group, and the PCA of each layer is then
fitted from its file. The cost of a pass moves from RAM to temporary disk, so
many more layers fit in one pass.

INPUT:
    - ann: imgANN -> model wrapper with hooks already registered
    - pixel_values: torch.Tensor -> frame buffer, N x C x H x W
    - layers: list[str] -> layers of this pass
    - feature_sizes: dict[str, int] -> flattened feature count per layer
    - forward_batch_size: int -> frames per forward call
    - stage_dir: Path -> directory receiving the temporary memmaps
    - stage_dtype: np.dtype -> stored activation dtype

OUTPUT:
    - stage_paths: dict[str, Path] -> staged activation files keyed by layer
"""
def stage_group_activations(
        ann, pixel_values, layers, feature_sizes, forward_batch_size,
        stage_dir, stage_dtype=np.float16,
        ):
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    n_frames = pixel_values.shape[0]
    # Layer names carry dots, so only the path separator has to be replaced.
    stage_paths = {
        layer: stage_dir / f"{layer.replace('/', '_')}.npy" for layer in layers
    }
    staged = {
        layer: np.lib.format.open_memmap(
            stage_paths[layer], mode="w+", dtype=stage_dtype,
            shape=(n_frames, feature_sizes[layer]),
        )
        for layer in layers
    }
    try:
        for start in range(0, n_frames, forward_batch_size):
            stop = min(start + forward_batch_size, n_frames)
            chunk_features = forward_frame_chunk(
                ann, pixel_values[start:stop], layers, forward_batch_size,
            )
            for layer in layers:
                staged[layer][start:stop] = chunk_features[layer]
            # end for layer
            del chunk_features
            if stop % (forward_batch_size * 20) == 0 or stop == n_frames:
                print(f"staged frames {stop}/{n_frames}", flush=True)
            # end if a progress line is due
        # end for start
    finally:
        for layer in layers:
            staged[layer].flush()
            del staged[layer]
        # end for layer
    # end try
    return stage_paths
# EOF


"""
fit_layer_pca
Fits the PCA of one layer from its staged activations.

The complete matrix is used when it fits the memory budget, which gives the
exact solution; otherwise the fit streams the file chunk by chunk. The frames
were shuffled before staging, so the streamed chunks mix videos and the
incremental estimate stays unbiased.

INPUT:
    - stage_path: Path -> staged activations of this layer
    - n_components: int -> principal components to keep
    - pca_batch_size: int -> rows per incremental update
    - ram_budget_bytes: float -> memory allowed for an in-RAM solve

OUTPUT:
    - pca: PCA | IncrementalPCA -> fitted object
"""
def fit_layer_pca(stage_path, n_components, pca_batch_size, ram_budget_bytes):
    activations = np.load(stage_path, mmap_mode="r")
    n_rows, n_features = activations.shape
    n_components = min(n_components, n_rows, n_features)
    # The solver works in float32 and centers a copy of whatever it is given.
    exact_bytes = (
        n_rows * n_features * ACTIVATION_ITEMSIZE * EXACT_PCA_MEMORY_FACTOR
    )
    if exact_bytes <= ram_budget_bytes:
        pca = PCA(n_components=n_components).fit(
            np.asarray(activations, dtype=np.float32)
        )
        solver = "exact"
    else:
        chunk_size = balanced_chunk_size(
            n_rows, max(pca_batch_size, n_components),
        )
        pca = IncrementalPCA(
            n_components=n_components, batch_size=chunk_size,
        )
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            if stop - start < n_components:
                continue
            # end if the chunk cannot support n_components
            pca.partial_fit(
                np.asarray(activations[start:stop], dtype=np.float32)
            )
        # end for start
        solver = f"incremental ({-(-n_rows // chunk_size)} x {chunk_size})"
    # end if the layer fits in RAM
    del activations
    print(
        f"  {solver}: {pca.explained_variance_ratio_.sum():.3f} variance kept",
        flush=True,
    )
    return pca
# EOF


"""
fit_frame_pca_model
Fits and saves the frame PCA of every requested layer of one model.

Layers whose component file already exists are skipped unless overwrite is set.
The frames are forwarded once per layer group, each group's activations are
staged on disk, and every layer is then fitted from its own file.

INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - ann: imgANN -> model wrapper created with pooling="all"
    - layers: list[str] -> hooked ANN layer names in depth order
    - pixel_values: torch.Tensor -> preprocessed frame buffer, N x C x H x W
    - dataset_name: str -> dataset label used in the component filenames
    - n_components: int -> number of principal components to keep
    - forward_batch_size: int -> frames per forward call
    - pca_batch_size: int -> rows per incremental update
    - ram_budget_bytes: float -> memory allowed for an in-RAM solve
    - disk_budget_bytes: float -> temporary disk allowed for one staged group
    - stage_dir: Path | None -> where the staged activations are written
    - seed: int -> seed of the frame shuffling
    - overwrite: bool -> refit layers whose component file already exists

OUTPUT:
    - save_paths: dict[str, Path] -> saved component files keyed by layer
"""
def fit_frame_pca_model(
        paths, ann, layers, pixel_values, dataset_name, n_components,
        forward_batch_size, pca_batch_size, ram_budget_bytes,
        disk_budget_bytes, stage_dir=None, seed=0, overwrite=False,
        ):
    save_paths = {
        layer: save_frame_pca(
            paths, ann.model_name, layer, dataset_name, n_components,
        )
        for layer in layers
    }
    missing_layers = [
        layer for layer in layers
        if overwrite or not save_paths[layer].exists()
    ]
    existing_layers = [layer for layer in layers if layer not in missing_layers]
    if existing_layers:
        print(
            f"skipping {len(existing_layers)} layers with stored components",
            flush=True,
        )
    # end if existing_layers
    if not missing_layers:
        return save_paths
    # end if nothing left to fit

    n_frames = pixel_values.shape[0]
    feature_sizes = layer_feature_sizes(ann, missing_layers)
    stage_dir = Path(
        stage_dir or Path(paths["data_path"]) / "models" / "frame_pca_stage"
    )
    layer_groups = group_layers_by_bytes(
        missing_layers, feature_sizes, n_frames, disk_budget_bytes,
        STAGE_ITEMSIZE,
    )
    staged_gb = (
        max(len(group) for group in layer_groups)
        * n_frames * max(feature_sizes.values()) * STAGE_ITEMSIZE / 1e9
    )
    print(
        f"{n_frames} frames, {len(missing_layers)} layers to fit, "
        f"{n_components} components\n"
        f"staging up to {staged_gb:.0f} GB at a time in {stage_dir} -> "
        f"{len(layer_groups)} forward pass(es)",
        flush=True,
    )

    # One shuffling for the whole model: it is what makes the streamed chunks
    # mix videos instead of following the stimulus order, and it leaves the
    # exact solution unchanged.
    shuffled_order = np.random.default_rng(seed).permutation(n_frames)
    pixel_values = pixel_values[torch.from_numpy(shuffled_order)]

    for group_index, layer_group in enumerate(layer_groups, start=1):
        print(
            f"\n--- pass {group_index}/{len(layer_groups)}: "
            f"{len(layer_group)} layers ---",
            flush=True,
        )
        ann.features = {}
        create_layer_hooks(ann, layer_group)
        try:
            stage_paths = stage_group_activations(
                ann, pixel_values, layer_group, feature_sizes,
                forward_batch_size, stage_dir,
            )
        finally:
            ann.clear_hooks()
            clear_model_cache(ann.device)
        # end try

        try:
            for layer in layer_group:
                print(f"fitting {layer}", flush=True)
                pca = fit_layer_pca(
                    stage_paths[layer], n_components, pca_batch_size,
                    ram_budget_bytes,
                )
                save_paths[layer].parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(cast_pca_float32(pca), save_paths[layer])
                print(f"  saved {save_paths[layer].name}", flush=True)
                del pca
                # The staged file is only needed by its own layer.
                stage_paths[layer].unlink(missing_ok=True)
            # end for layer
        finally:
            # Never leave tens of GB behind when a fit raises.
            for path in stage_paths.values():
                path.unlink(missing_ok=True)
            # end for path
        # end try
    # end for layer_group
    return save_paths
# EOF


"""
fit_layer_srp
Builds the sparse random projection that first shrinks one wide layer.

The projection is data independent, so it is "fitted" on a single zero row
only to fix its random matrix. Layers no wider than the target are left alone.

INPUT:
    - n_features: int -> flattened width of the layer
    - n_components: int -> width after the projection
    - seed: int -> seed of the random matrix

OUTPUT:
    - srp: SparseRandomProjection | None -> float32 projection, or None when
        the layer is already narrow enough
"""
def fit_layer_srp(n_features, n_components, seed=0):
    if n_features <= n_components:
        return None
    # end if the layer is already narrow enough
    # Fitting on float32 makes the stored sparse matrix float32 as well, so
    # the transform keeps float32 activations in float32.
    return SparseRandomProjection(
        n_components=n_components, random_state=seed,
    ).fit(np.zeros((1, n_features), dtype=np.float32))
# EOF


"""
stage_image_srp_activations
Streams images through the model and writes every layer's (optionally
SRP-reduced) activations to one memmap per layer.

Unlike stage_group_activations the images are decoded chunk by chunk, so the
whole ImageNet validation set never has to sit in RAM as pixels; the random
projections of the layers run in parallel threads (scipy releases the GIL).

INPUT:
    - ann: imgANN -> model wrapper with hooks already registered
    - image_paths: list[Path] -> images to stream
    - preprocessor: object -> frame preprocessor of the model
    - pkg: str -> model package, selects the preprocessing call
    - layers: list[str] -> layers of this pass
    - srps: dict[str, SparseRandomProjection | None] -> projection per layer
    - feature_sizes: dict[str, int] -> full flattened width per layer
    - stage_widths: dict[str, int] -> stored width per layer
    - forward_batch_size: int -> images per forward call
    - stage_dir: Path -> directory receiving the temporary memmaps
    - chunk_bytes: float -> RAM allowed for the full-width float32
        activations of one chunk, which sets how many images are decoded at once
    - stage_dtype: np.dtype -> stored activation dtype

OUTPUT:
    - stage_paths: dict[str, Path] -> staged activation files keyed by layer
"""
def stage_image_srp_activations(
        ann, image_paths, preprocessor, pkg, layers, srps, feature_sizes,
        stage_widths, forward_batch_size, stage_dir, chunk_bytes=2e9,
        stage_dtype=np.float16,
        ):
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    n_images = len(image_paths)
    # A chunk holds every layer at full width before the projection (about
    # 42 MB per image for a ViT-H with 32 hooked layers), so its size follows
    # from the memory allowed rather than from a fixed image count.
    bytes_per_image = ACTIVATION_ITEMSIZE * sum(feature_sizes[layer] for layer in layers)
    chunk_size = max(forward_batch_size, int(chunk_bytes // bytes_per_image))
    stage_paths = {
        layer: stage_dir / f"{layer.replace('/', '_')}.npy" for layer in layers
    }
    staged = {
        layer: np.lib.format.open_memmap(
            stage_paths[layer], mode="w+", dtype=stage_dtype,
            shape=(n_images, stage_widths[layer]),
        )
        for layer in layers
    }

    # (rows, n_features) -> (rows, stage width) for one layer.
    def reduce_layer(layer, features):
        if srps[layer] is None:
            return features
        # end if the layer is kept at full width
        return srps[layer].transform(features)
    # EOF

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            for start in range(0, n_images, chunk_size):
                stop = min(start + chunk_size, n_images)
                # N x C x H x W pixels of this chunk only.
                chunk_pixels = build_image_buffer(
                    image_paths[start:stop], preprocessor, pkg,
                    chunk_size=chunk_size, verbose=False,
                )
                chunk_features = forward_frame_chunk(
                    ann, chunk_pixels, layers, forward_batch_size,
                )
                reduced = executor.map(
                    lambda layer: reduce_layer(layer, chunk_features[layer]),
                    layers,
                )
                for layer, layer_features in zip(layers, reduced):
                    staged[layer][start:stop] = layer_features
                # end for layer, layer_features
                del chunk_pixels, chunk_features
                if stop % (chunk_size * 20) < chunk_size or stop == n_images:
                    print(f"staged images {stop}/{n_images}", flush=True)
                # end if a progress line is due
            # end for start
        # end with ThreadPoolExecutor
    finally:
        for layer in layers:
            staged[layer].flush()
            del staged[layer]
        # end for layer
    # end try
    return stage_paths
# EOF


"""
fit_image_srp_pca_model
Fits and saves, for every requested layer, a sparse random projection followed
by an exact PCA on a set of images (e.g. the whole ImageNet validation set).

Wide layers are first reduced to n_srp_components random features, which makes
the activation matrix of tens of thousands of images small enough for an
exact in-RAM PCA instead of an IncrementalPCA. Each layer is saved as one
sklearn Pipeline (SRP -> PCA), or as the bare PCA when the layer is no wider
than n_srp_components, under the same filename as save_frame_pca, so
load_frame_pcas and pca_projection use it unchanged.

INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - ann: imgANN -> model wrapper created with pooling="all"
    - layers: list[str] -> hooked ANN layer names in depth order
    - image_paths: list[Path] -> images the components are fitted on
    - preprocessor: object -> frame preprocessor of the model
    - pkg: str -> model package, selects the preprocessing call
    - dataset_name: str -> dataset label used in the component filenames
    - n_srp_components: int -> width after the random projection
    - n_components: int -> number of principal components to keep
    - forward_batch_size: int -> images per forward call
    - ram_budget_bytes: float -> memory allowed for an in-RAM solve
    - disk_budget_bytes: float -> temporary disk allowed for one staged group
    - stage_dir: Path | None -> where the staged activations are written
    - seed: int -> seed of the random projections
    - overwrite: bool -> refit layers whose component file already exists

OUTPUT:
    - save_paths: dict[str, Path] -> saved component files keyed by layer
"""
def fit_image_srp_pca_model(
        paths, ann, layers, image_paths, preprocessor, pkg, dataset_name,
        n_srp_components, n_components, forward_batch_size, ram_budget_bytes,
        disk_budget_bytes, stage_dir=None, seed=0, overwrite=False,
        ):
    save_paths = {
        layer: save_frame_pca(
            paths, ann.model_name, layer, dataset_name, n_components,
        )
        for layer in layers
    }
    missing_layers = [
        layer for layer in layers
        if overwrite or not save_paths[layer].exists()
    ]
    if len(missing_layers) < len(layers):
        print(
            f"skipping {len(layers) - len(missing_layers)} layers with stored "
            f"components",
            flush=True,
        )
    # end if some layers are already fitted
    if not missing_layers:
        return save_paths
    # end if nothing left to fit

    feature_sizes = layer_feature_sizes(ann, missing_layers)
    # One random matrix per layer; a different seed per layer keeps the
    # projections independent across layers.
    srps = {
        layer: fit_layer_srp(feature_sizes[layer], n_srp_components, seed + index)
        for index, layer in enumerate(missing_layers)
    }
    stage_widths = {
        layer: min(feature_sizes[layer], n_srp_components)
        for layer in missing_layers
    }
    n_images = len(image_paths)
    stage_dir = Path(
        stage_dir or Path(paths["data_path"]) / "models" / "frame_pca_stage"
    )
    layer_groups = group_layers_by_bytes(
        missing_layers, stage_widths, n_images, disk_budget_bytes,
        STAGE_ITEMSIZE,
    )
    print(
        f"{n_images} images, {len(missing_layers)} layers, SRP to "
        f"{n_srp_components} then {n_components} PCs\n"
        f"staging in {stage_dir} -> {len(layer_groups)} forward pass(es)",
        flush=True,
    )

    for group_index, layer_group in enumerate(layer_groups, start=1):
        print(
            f"\n--- pass {group_index}/{len(layer_groups)}: "
            f"{len(layer_group)} layers ---",
            flush=True,
        )
        ann.features = {}
        create_layer_hooks(ann, layer_group)
        try:
            stage_paths = stage_image_srp_activations(
                ann, image_paths, preprocessor, pkg, layer_group, srps,
                feature_sizes, stage_widths, forward_batch_size, stage_dir,
            )
        finally:
            ann.clear_hooks()
            clear_model_cache(ann.device)
        # end try

        try:
            for layer in layer_group:
                print(f"fitting {layer}", flush=True)
                # The staged matrix is already narrow, so this is the exact
                # PCA whenever the RAM budget allows it.
                pca = cast_pca_float32(fit_layer_pca(
                    stage_paths[layer], n_components, n_components,
                    ram_budget_bytes,
                ))
                projection = pca if srps[layer] is None else make_pipeline(
                    srps[layer], pca,
                )
                save_paths[layer].parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(projection, save_paths[layer])
                print(f"  saved {save_paths[layer].name}", flush=True)
                del pca, projection
                stage_paths[layer].unlink(missing_ok=True)
            # end for layer
        finally:
            # Never leave tens of GB behind when a fit raises.
            for path in stage_paths.values():
                path.unlink(missing_ok=True)
            # end for path
        # end try
    # end for layer_group
    return save_paths
# EOF


"""
pc_extremes_path
Path of the file holding, for one model, the image scores on the leading PCs
of every layer and the indices of each PC's most extreme images.

INPUT:
    - paths: dict -> config paths dictionary containing the data root path
    - model_name: str -> ANN model name used to compute activations
    - components_dataset: str -> dataset the components were fitted on
    - n_components: int -> number of components used in the stored filenames
    - image_dataset: str -> dataset whose images are scored

OUTPUT:
    - save_path: Path -> HDF5 file of the scores and extreme indices
"""
def pc_extremes_path(
        paths, model_name, components_dataset, n_components, image_dataset,
        ):
    save_dir = Path(paths["data_path"]) / "models" / "pc_extremes"
    file_name = (
        f"{model_name}_{components_dataset}_{n_components}components_"
        f"{image_dataset}_extremes.h5"
    )
    return save_dir / file_name
# EOF


"""
leading_pc_projection
Splits a stored layer projection into the parts needed to score images on its
first PCs only, so the 1000-component product is never formed.

INPUT:
    - projection: Pipeline | PCA | IncrementalPCA -> stored layer projection,
        either SRP -> PCA or a bare PCA
    - n_pcs: int -> leading components kept

OUTPUT:
    - srp: SparseRandomProjection | None -> random projection, None if absent
    - mean: np.ndarray -> PCA centre, shape (n_features,)
    - components: np.ndarray -> leading components, shape (n_pcs, n_features)
"""
def leading_pc_projection(projection, n_pcs):
    srp, pca = None, projection
    if hasattr(projection, "steps"):
        srp, pca = projection.steps[0][1], projection.steps[-1][1]
    # end if the projection is an SRP -> PCA pipeline
    if getattr(pca, "whiten", False):
        raise ValueError("Whitened PCA objects are not supported.")
    # end if whiten
    mean = pca.mean_.astype(np.float32)
    components = np.ascontiguousarray(pca.components_[:n_pcs], dtype=np.float32)
    return srp, mean, components
# EOF


"""
score_images_on_leading_pcs
Runs a set of images through the model once and scores them, for every layer,
on the leading components of that layer's stored projection.

INPUT:
    - ann: imgANN -> model wrapper created with pooling="all"
    - layers: list[str] -> hooked ANN layer names in depth order
    - projections: dict[str, Pipeline | PCA] -> stored projection per layer
    - image_paths: list[Path] -> images to score
    - preprocessor: object -> frame preprocessor of the model
    - pkg: str -> model package, selects the preprocessing call
    - n_pcs: int -> leading components scored per layer
    - forward_batch_size: int -> images per forward call
    - chunk_bytes: float -> RAM allowed for the full-width float32
        activations of one chunk, which sets how many images are decoded at once

OUTPUT:
    - scores: dict[str, np.ndarray] -> (n_images, n_pcs) scores keyed by layer
"""
def score_images_on_leading_pcs(
        ann, layers, projections, image_paths, preprocessor, pkg, n_pcs,
        forward_batch_size, chunk_bytes=2e9,
        ):
    n_images = len(image_paths)
    leading = {
        layer: leading_pc_projection(projections[layer], n_pcs)
        for layer in layers
    }
    feature_sizes = layer_feature_sizes(ann, layers)
    bytes_per_image = ACTIVATION_ITEMSIZE * sum(feature_sizes.values())
    chunk_size = max(forward_batch_size, int(chunk_bytes // bytes_per_image))
    scores = {
        layer: np.empty((n_images, n_pcs), dtype=np.float32) for layer in layers
    }

    # (rows, n_features) -> (rows, n_pcs), the first columns of pca.transform.
    def project_layer(layer, features):
        srp, mean, components = leading[layer]
        if srp is not None:
            features = srp.transform(features)
        # end if the layer was randomly projected before its PCA
        return (np.asarray(features, dtype=np.float32) - mean) @ components.T
    # EOF

    create_layer_hooks(ann, layers)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for start in range(0, n_images, chunk_size):
            stop = min(start + chunk_size, n_images)
            chunk_pixels = build_image_buffer(
                image_paths[start:stop], preprocessor, pkg,
                chunk_size=chunk_size, verbose=False,
            )
            chunk_features = forward_frame_chunk(
                ann, chunk_pixels, layers, forward_batch_size,
            )
            projected = executor.map(
                lambda layer: project_layer(layer, chunk_features[layer]),
                layers,
            )
            for layer, layer_scores in zip(layers, projected):
                scores[layer][start:stop] = layer_scores
            # end for layer, layer_scores
            del chunk_pixels, chunk_features
            if stop % (chunk_size * 20) < chunk_size or stop == n_images:
                print(f"scored images {stop}/{n_images}", flush=True)
            # end if a progress line is due
        # end for start
    # end with ThreadPoolExecutor
    return scores
# EOF


"""
extreme_indices
Indices of the highest and lowest scoring rows of every column.

INPUT:
    - scores: np.ndarray -> (n_images, n_pcs) scores
    - n_extreme: int -> rows kept at each end

OUTPUT:
    - top: np.ndarray -> (n_pcs, n_extreme) indices, highest score first
    - bottom: np.ndarray -> (n_pcs, n_extreme) indices, lowest score first
"""
def extreme_indices(scores, n_extreme):
    order = np.argsort(scores, axis=0, kind="stable").T
    return order[:, ::-1][:, :n_extreme].copy(), order[:, :n_extreme].copy()
# EOF


"""
save_pc_extremes
Writes the per-layer scores and extreme indices of one model to HDF5.

Layout: one group per layer with "scores" (n_images x n_pcs), "top_indices"
and "bottom_indices" (n_pcs x n_extreme, most extreme first), plus the scored
image paths in "image_paths", whose order the indices refer to.

INPUT:
    - save_path: Path -> destination file
    - scores: dict[str, np.ndarray] -> (n_images, n_pcs) scores keyed by layer
    - image_paths: list[Path] -> scored images, in row order
    - n_extreme: int -> images kept at each end of every PC
    - metadata: dict -> file attributes

OUTPUT:
    - save_path: Path -> the written file
"""
def save_pc_extremes(save_path, scores, image_paths, n_extreme, metadata):
    import h5py

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = save_path.with_suffix(".tmp.h5")
    with h5py.File(temporary_path, "w") as h5_file:
        h5_file.attrs.update({**metadata, "n_extreme": n_extreme})
        h5_file.create_dataset(
            "image_paths", data=np.array([str(path) for path in image_paths], dtype="S"),
        )
        for layer, layer_scores in scores.items():
            top, bottom = extreme_indices(layer_scores, n_extreme)
            group = h5_file.create_group(layer)
            group.create_dataset("scores", data=layer_scores)
            group.create_dataset("top_indices", data=top)
            group.create_dataset("bottom_indices", data=bottom)
        # end for layer, layer_scores
    # end with h5py.File
    # Only a complete file ever carries the final name.
    temporary_path.replace(save_path)
    return save_path
# EOF
