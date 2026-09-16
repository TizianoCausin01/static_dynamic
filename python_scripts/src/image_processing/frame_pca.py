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

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA, IncrementalPCA

from .video_feature_extraction import (
    clear_model_cache,
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
    ann.create_forward_hook(layer_names=layers)
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
        ann.create_forward_hook(layer_names=layer_group)
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
