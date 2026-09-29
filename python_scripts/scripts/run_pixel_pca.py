import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile

import h5py
import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as file:
    config = yaml.safe_load(file)
# end with open

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from image_processing.frame_pca import (
    cast_pca_float32,
    fit_layer_pca,
    save_frame_pca,
)


@dataclass
class Cfg:
    source_path: str | None = None
    output_dir: str | None = None
    dataset_name: str = "static_dynamic"
    pixel_step: int = 1
    n_components: int = 1000
    pca_batch_size: int = 256
    projection_batch_size: int = 64
    stage_frame_block_size: int = 4
    ram_budget_gb: float = 8.0
    randomized_oversamples: int = 16
    randomized_power_iterations: int = 1
    randomized_seed: int = 0
    streaming_batch_size: int = 180
    feature_block_size: int = 32768
    stage_dir: str | None = None
    compression: str | None = "lzf"
    overwrite: bool = False
# EOF


"""
parse_args
Parses the RGB-pixel PCA configuration.

OUTPUT:
    - cfg: Cfg -> validated PCA fitting and projection configuration
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit PCA across flattened RGB movie frames and save frame-wise "
            "principal-component scores in the model-feature HDF5 format."
        )
    )
    parser.add_argument("--source_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--dataset_name", default=Cfg.dataset_name)
    parser.add_argument("--pixel_step", type=int, default=Cfg.pixel_step)
    parser.add_argument("--n_components", type=int, default=Cfg.n_components)
    parser.add_argument("--pca_batch_size", type=int, default=Cfg.pca_batch_size)
    parser.add_argument(
        "--projection_batch_size",
        type=int,
        default=Cfg.projection_batch_size,
    )
    parser.add_argument(
        "--stage_frame_block_size",
        type=int,
        default=Cfg.stage_frame_block_size,
    )
    parser.add_argument("--ram_budget_gb", type=float, default=Cfg.ram_budget_gb)
    parser.add_argument(
        "--randomized_oversamples",
        type=int,
        default=Cfg.randomized_oversamples,
    )
    parser.add_argument(
        "--randomized_power_iterations",
        type=int,
        default=Cfg.randomized_power_iterations,
    )
    parser.add_argument(
        "--randomized_seed", type=int, default=Cfg.randomized_seed,
    )
    parser.add_argument(
        "--streaming_batch_size",
        type=int,
        default=Cfg.streaming_batch_size,
    )
    parser.add_argument(
        "--feature_block_size", type=int, default=Cfg.feature_block_size,
    )
    parser.add_argument("--stage_dir")
    parser.add_argument(
        "--compression", choices=("lzf", "gzip"), default=Cfg.compression,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    positive_integer_names = (
        "pixel_step",
        "n_components",
        "pca_batch_size",
        "projection_batch_size",
        "stage_frame_block_size",
        "randomized_oversamples",
        "streaming_batch_size",
        "feature_block_size",
    )
    for argument_name in positive_integer_names:
        if getattr(args, argument_name) < 1:
            parser.error(f"--{argument_name} must be positive.")
        # end if invalid positive integer
    # end for argument_name
    if args.randomized_power_iterations < 0:
        parser.error("--randomized_power_iterations cannot be negative.")
    # end if invalid power iteration count
    if args.ram_budget_gb <= 0:
        parser.error("--ram_budget_gb must be positive.")
    # end if invalid RAM budget
    return Cfg(**vars(args))
# EOF


"""
inspect_pixel_file
Validates an RGB-pixel HDF5 file and reports its shared array geometry.

INPUT:
    - source_path: str | Path -> raw RGB-pixel feature file

OUTPUT:
    - dataset_names: list[str] -> movie datasets in stable lexical order
    - n_frames: int -> frames stored for every movie
    - n_features: int -> flattened RGB values stored for every frame
"""
def inspect_pixel_file(source_path):
    with h5py.File(source_path, "r") as source_file:
        if source_file.attrs.get("channel_order") != "RGB":
            raise ValueError("Pixel PCA requires an RGB source feature file.")
        # end if source is not RGB
        dataset_names = sorted(
            name for name, value in source_file.items()
            if isinstance(value, h5py.Dataset)
        )
        if not dataset_names:
            raise ValueError(f"No movie datasets found in {source_path}.")
        # end if no datasets

        frame_counts = {source_file[name].shape[0] for name in dataset_names}
        feature_sizes = {source_file[name].shape[1] for name in dataset_names}
        if len(frame_counts) != 1 or len(feature_sizes) != 1:
            raise ValueError(
                "All movies must share one frame count and RGB feature size."
            )
        # end if incompatible datasets
        n_frames = frame_counts.pop()
        n_features = feature_sizes.pop()
    # end with source_file
    return dataset_names, n_frames, n_features
# EOF


"""
stage_pixel_frames
Stages RGB frames in time-major order so incremental batches mix movies.

The source vectors remain `[R0, G0, B0, R1, G1, B1, ...]`. PCA therefore uses
all three color channels and may learn components that combine color and space.

INPUT:
    - source_path: str | Path -> raw RGB-pixel HDF5 file
    - dataset_names: list[str] -> ordered movie dataset names
    - n_frames: int -> frames stored per movie
    - n_features: int -> flattened RGB features per frame
    - stage_path: str | Path -> temporary NumPy memmap path
    - frame_block_size: int -> adjacent times read per movie in one block

OUTPUT:
    - stage_path: Path -> float16 time-major frame matrix
"""
def stage_pixel_frames(
        source_path,
        dataset_names,
        n_frames,
        n_features,
        stage_path,
        frame_block_size,
        ):
    stage_path = Path(stage_path)
    n_rows = len(dataset_names) * n_frames
    staged_frames = np.lib.format.open_memmap(
        stage_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_rows, n_features),
    )

    cursor = 0
    with h5py.File(source_path, "r") as source_file:
        for frame_start in range(0, n_frames, frame_block_size):
            frame_stop = min(frame_start + frame_block_size, n_frames)
            frame_block = np.concatenate([
                source_file[name][frame_start:frame_stop]
                for name in dataset_names
            ], axis=0)
            next_cursor = cursor + frame_block.shape[0]
            staged_frames[cursor:next_cursor] = frame_block
            cursor = next_cursor
            print(
                f"staged source frames {frame_stop}/{n_frames} "
                f"across {len(dataset_names)} movies",
                flush=True,
            )
        # end for frame_start
    # end with source_file
    staged_frames.flush()
    del staged_frames

    if cursor != n_rows:
        raise RuntimeError(f"Staged {cursor} rows; expected {n_rows}.")
    # end if incomplete staging
    return stage_path
# EOF


"""
iter_pixel_batches
Streams movie frames in stable movie-major, time-major order.

INPUT:
    - source_path: str | Path -> RGB-pixel HDF5 file
    - dataset_names: list[str] -> ordered movie dataset names
    - batch_size: int -> frames returned per batch

OUTPUT:
    - batches: iterator -> tuples of global row start and float32 RGB frames
"""
def iter_pixel_batches(source_path, dataset_names, batch_size):
    global_start = 0
    with h5py.File(source_path, "r") as source_file:
        for dataset_name in dataset_names:
            dataset = source_file[dataset_name]
            for start in range(0, dataset.shape[0], batch_size):
                stop = min(start + batch_size, dataset.shape[0])
                pixel_batch = np.asarray(dataset[start:stop], dtype=np.float32)
                yield global_start, pixel_batch
                global_start += pixel_batch.shape[0]
            # end for start
        # end for dataset_name
    # end with source_file
# EOF


"""
form_reduced_rgb_basis
Forms Q.T @ (X - mean) while streaming full-resolution RGB frames.

INPUT:
    - source_path: str | Path -> full RGB-pixel HDF5 file
    - dataset_names: list[str] -> ordered movie dataset names
    - frame_basis: np.ndarray -> orthonormal basis in frame space
    - feature_mean: np.ndarray -> mean of every RGB pixel feature
    - batch_size: int -> frames read per streaming multiplication
    - feature_block_size: int -> feature columns accumulated together
    - progress_label: str -> label printed with streaming progress

OUTPUT:
    - reduced_rgb_basis: np.ndarray -> Q.T @ centered RGB frame matrix
"""
def form_reduced_rgb_basis(
        source_path,
        dataset_names,
        frame_basis,
        feature_mean,
        batch_size,
        feature_block_size,
        progress_label,
        ):
    n_samples, range_size = frame_basis.shape
    n_features = feature_mean.size
    reduced_rgb_basis = np.zeros(
        (range_size, n_features), dtype=np.float32,
    )

    # Accumulate the uncentered product without constructing centered frames.
    processed_rows = 0
    for global_start, pixel_batch in iter_pixel_batches(
            source_path, dataset_names, batch_size,
            ):
        global_stop = global_start + pixel_batch.shape[0]
        frame_weights = frame_basis[global_start:global_stop].T
        for feature_start in range(0, n_features, feature_block_size):
            feature_stop = min(feature_start + feature_block_size, n_features)
            reduced_rgb_basis[:, feature_start:feature_stop] += (
                frame_weights @ pixel_batch[:, feature_start:feature_stop]
            )
        # end for feature_start
        processed_rows = global_stop
        print(
            f"{progress_label}: {processed_rows}/{n_samples} frames",
            flush=True,
        )
    # end for pixel_batch

    # Apply centering algebraically: Q.T @ X - sum(Q).T @ mean(X).
    frame_weight_sum = frame_basis.sum(axis=0, dtype=np.float64).astype(
        np.float32,
    )
    for feature_start in range(0, n_features, feature_block_size):
        feature_stop = min(feature_start + feature_block_size, n_features)
        reduced_rgb_basis[:, feature_start:feature_stop] -= np.outer(
            frame_weight_sum,
            feature_mean[feature_start:feature_stop],
        )
    # end for feature_start
    return reduced_rgb_basis
# EOF


"""
project_reduced_rgb_basis
Computes (X - mean) @ B.T for one randomized PCA power iteration.

INPUT:
    - source_path: str | Path -> full RGB-pixel HDF5 file
    - dataset_names: list[str] -> ordered movie dataset names
    - n_samples: int -> total number of frames
    - reduced_rgb_basis: np.ndarray -> reduced feature-space basis B
    - feature_mean: np.ndarray -> mean of every RGB pixel feature
    - batch_size: int -> frames read per streaming multiplication
    - progress_label: str -> label printed with streaming progress

OUTPUT:
    - projected_frames: np.ndarray -> powered range in frame space
"""
def project_reduced_rgb_basis(
        source_path,
        dataset_names,
        n_samples,
        reduced_rgb_basis,
        feature_mean,
        batch_size,
        progress_label,
        ):
    range_size = reduced_rgb_basis.shape[0]
    projected_frames = np.empty(
        (n_samples, range_size), dtype=np.float32,
    )
    mean_projection = feature_mean @ reduced_rgb_basis.T

    processed_rows = 0
    for global_start, pixel_batch in iter_pixel_batches(
            source_path, dataset_names, batch_size,
            ):
        global_stop = global_start + pixel_batch.shape[0]
        projected_frames[global_start:global_stop] = (
            pixel_batch @ reduced_rgb_basis.T - mean_projection
        )
        processed_rows = global_stop
        print(
            f"{progress_label}: {processed_rows}/{n_samples} frames",
            flush=True,
        )
    # end for pixel_batch
    return projected_frames
# EOF


"""
fit_streaming_randomized_pixel_pca
Fits a randomized PCA without materializing the full frame-by-pixel matrix.

This is the full-pixel solver. It keeps all RGB features, constructs a random
range of `n_components + oversamples` dimensions, refines it with streaming
power iterations, and forms the small PCA problem in a final pass. No spatial
subsampling is applied: every component has one weight for every R, G, and B
input value.

INPUT:
    - source_path: str | Path -> full RGB-pixel HDF5 file
    - dataset_names: list[str] -> ordered movie dataset names
    - n_frames: int -> frames stored per movie
    - n_features: int -> complete flattened RGB feature count
    - n_components: int -> principal components to retain
    - oversamples: int -> extra randomized range dimensions
    - power_iterations: int -> streamed subspace iterations for accuracy
    - seed: int -> random seed used for the range finder
    - batch_size: int -> frames read per streaming matrix multiplication
    - feature_block_size: int -> feature columns updated together in the basis

OUTPUT:
    - pca: PCA -> fitted PCA-compatible object with full RGB components
"""
def fit_streaming_randomized_pixel_pca(
        source_path,
        dataset_names,
        n_frames,
        n_features,
        n_components,
        oversamples,
        power_iterations,
        seed,
        batch_size,
        feature_block_size,
        ):
    n_samples = len(dataset_names) * n_frames
    if n_components >= min(n_samples, n_features):
        raise ValueError(
            "n_components must be smaller than both the frame and feature counts."
        )
    # end if too many components
    range_size = min(n_components + oversamples, n_samples, n_features)
    random_generator = np.random.default_rng(seed)

    print(
        f"Streaming randomized PCA: {n_samples} x {n_features}, "
        f"range size {range_size}",
        flush=True,
    )
    print(
        f"Allocating {n_features * range_size * 4 / 1e9:.2f} GB "
        "RGB random-range matrix",
        flush=True,
    )
    random_range = random_generator.standard_normal(
        size=(n_features, range_size), dtype=np.float32,
    )
    projected_frames = np.empty((n_samples, range_size), dtype=np.float32)
    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_square_sum = np.zeros(n_features, dtype=np.float64)

    # First pass: estimate the RGB mean/variance and the randomized range.
    processed_rows = 0
    for global_start, pixel_batch in iter_pixel_batches(
            source_path, dataset_names, batch_size,
            ):
        global_stop = global_start + pixel_batch.shape[0]
        feature_sum += pixel_batch.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.square(
            pixel_batch, dtype=np.float32,
        ).sum(axis=0, dtype=np.float64)
        projected_frames[global_start:global_stop] = pixel_batch @ random_range
        processed_rows = global_stop
        print(
            f"randomized range: {processed_rows}/{n_samples} frames",
            flush=True,
        )
    # end for pixel_batch

    feature_mean = feature_sum / n_samples
    feature_variance = (
        feature_square_sum - n_samples * np.square(feature_mean)
    ) / (n_samples - 1)
    total_variance = float(np.maximum(feature_variance, 0).sum())
    mean_projection = feature_mean.astype(np.float32) @ random_range
    projected_frames -= mean_projection
    del feature_sum, feature_square_sum, feature_variance, mean_projection
    del random_range

    print("Orthogonalizing the randomized frame range", flush=True)
    frame_basis = np.linalg.qr(projected_frames, mode="reduced")[0].astype(
        np.float32,
        copy=False,
    )
    del projected_frames

    feature_mean = feature_mean.astype(np.float32)
    for iteration in range(power_iterations):
        reduced_rgb_basis = form_reduced_rgb_basis(
            source_path,
            dataset_names,
            frame_basis,
            feature_mean,
            batch_size,
            feature_block_size,
            f"power {iteration + 1} reduced basis",
        )
        projected_frames = project_reduced_rgb_basis(
            source_path,
            dataset_names,
            n_samples,
            reduced_rgb_basis,
            feature_mean,
            batch_size,
            f"power {iteration + 1} frame range",
        )
        frame_basis = np.linalg.qr(projected_frames, mode="reduced")[0].astype(
            np.float32,
            copy=False,
        )
        del reduced_rgb_basis, projected_frames
    # end for iteration

    # Final pass: form the reduced matrix used to recover full RGB components.
    reduced_rgb_basis = form_reduced_rgb_basis(
        source_path,
        dataset_names,
        frame_basis,
        feature_mean,
        batch_size,
        feature_block_size,
        "final reduced RGB basis",
    )

    # The reduced matrix has only ~1,000 rows, so its left eigensystem is small.
    print("Solving the reduced RGB eigensystem", flush=True)
    reduced_covariance = reduced_rgb_basis @ reduced_rgb_basis.T
    eigenvalues, left_vectors = np.linalg.eigh(reduced_covariance)
    descending_order = np.argsort(eigenvalues)[::-1][:n_components]
    singular_values = np.sqrt(
        np.maximum(eigenvalues[descending_order], 0)
    ).astype(np.float32)
    left_vectors = left_vectors[:, descending_order].astype(
        np.float32,
        copy=False,
    )
    del reduced_covariance, eigenvalues

    print(
        f"Recovering {n_components} full RGB component vectors", flush=True,
    )
    components = left_vectors.T @ reduced_rgb_basis
    nonzero_singular_values = singular_values > 0
    components[nonzero_singular_values] /= singular_values[
        nonzero_singular_values, None
    ]
    components[~nonzero_singular_values] = 0
    del frame_basis, left_vectors, reduced_rgb_basis

    explained_variance = np.square(singular_values) / (n_samples - 1)
    explained_variance_ratio = explained_variance / total_variance
    residual_dimensions = min(n_samples, n_features) - n_components
    residual_variance = max(
        total_variance - float(explained_variance.sum()), 0,
    )

    # Populate sklearn's fitted attributes so the saved basis supports transform.
    pca = PCA(n_components=n_components)
    pca.components_ = components.astype(np.float32, copy=False)
    pca.mean_ = feature_mean
    pca.explained_variance_ = explained_variance.astype(np.float32, copy=False)
    pca.explained_variance_ratio_ = explained_variance_ratio.astype(
        np.float32, copy=False,
    )
    pca.singular_values_ = singular_values
    pca.n_components_ = n_components
    pca.n_samples_ = n_samples
    pca.n_features_in_ = n_features
    pca.noise_variance_ = (
        residual_variance / residual_dimensions
        if residual_dimensions > 0 else 0.0
    )
    pca._fit_svd_solver = "streaming_randomized"
    print(
        f"Streaming PCA retained "
        f"{pca.explained_variance_ratio_.sum():.3f} variance",
        flush=True,
    )
    return pca
# EOF


"""
fit_pixel_pca
Fits or loads the PCA basis of all RGB frames.

INPUT:
    - cfg: Cfg -> PCA settings
    - source_path: Path -> raw RGB-pixel HDF5 file
    - dataset_names: list[str] -> ordered movie dataset names
    - n_frames: int -> frames per movie
    - n_features: int -> flattened RGB features per frame
    - model_name: str -> model identifier used in saved filenames

OUTPUT:
    - pca: PCA | IncrementalPCA -> fitted RGB-pixel PCA object
    - pca_path: Path -> saved PCA object path
"""
def fit_pixel_pca(
        cfg,
        source_path,
        dataset_names,
        n_frames,
        n_features,
        model_name,
        ):
    pca_path = save_frame_pca(
        paths,
        model_name,
        "pixels",
        cfg.dataset_name,
        cfg.n_components,
    )
    if pca_path.exists() and not cfg.overwrite:
        print(f"Loading existing PCA: {pca_path}", flush=True)
        return joblib.load(pca_path), pca_path
    # end if stored PCA exists

    n_rows = len(dataset_names) * n_frames
    exact_fit_bytes = n_rows * n_features * 4 * 2
    if exact_fit_bytes > cfg.ram_budget_gb * 1e9:
        pca = fit_streaming_randomized_pixel_pca(
            source_path,
            dataset_names,
            n_frames,
            n_features,
            cfg.n_components,
            cfg.randomized_oversamples,
            cfg.randomized_power_iterations,
            cfg.randomized_seed,
            cfg.streaming_batch_size,
            cfg.feature_block_size,
        )
    else:
        stage_parent = Path(cfg.stage_dir or "/private/tmp").expanduser()
        stage_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="pixel_pca_", dir=stage_parent,
                ) as temporary_dir:
            stage_path = Path(temporary_dir) / "rgb_frames.npy"
            stage_pixel_frames(
                source_path,
                dataset_names,
                n_frames,
                n_features,
                stage_path,
                cfg.stage_frame_block_size,
            )
            pca = fit_layer_pca(
                stage_path,
                cfg.n_components,
                cfg.pca_batch_size,
                cfg.ram_budget_gb * 1e9,
            )
        # end with temporary directory
    # end if the full matrix fits in RAM

    pca_path.parent.mkdir(parents=True, exist_ok=True)
    pca = cast_pca_float32(pca)
    joblib.dump(pca, pca_path)
    print(f"Saved PCA: {pca_path}", flush=True)
    return pca, pca_path
# EOF


"""
project_pixel_pca
Projects every movie frame and saves PC scores in model-feature HDF5 format.

INPUT:
    - cfg: Cfg -> projection settings
    - source_path: Path -> raw RGB-pixel HDF5 file
    - output_path: Path -> destination HDF5 feature file
    - dataset_names: list[str] -> ordered movie dataset names
    - pca: PCA | IncrementalPCA -> fitted RGB-pixel PCA object
    - pca_path: Path -> PCA component file recorded in metadata
    - model_name: str -> model identifier stored in metadata

OUTPUT:
    - output_path: Path -> HDF5 file containing frame-wise PC scores
"""
def project_pixel_pca(
        cfg,
        source_path,
        output_path,
        dataset_names,
        pca,
        pca_path,
        model_name,
        ):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_metadata = {
        "model_name": model_name,
        "layer_name": "pixels",
        "dataset_name": cfg.dataset_name,
        "pooling": f"PC{cfg.n_components}",
        "feature_type": "pca_rgb_pixels",
        "channel_order": "RGB",
        "input_flatten_order": "pixel_major_RGB",
        "n_components": cfg.n_components,
        "feature_size": cfg.n_components,
        "source_feature_size": int(pca.n_features_in_),
        "source_feature_path": str(source_path),
        "pca_path": str(pca_path),
        "explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
        "value_dtype": "float32",
        "pca_solver": str(getattr(pca, "_fit_svd_solver", "unknown")),
        "randomized_oversamples": cfg.randomized_oversamples,
        "randomized_power_iterations": cfg.randomized_power_iterations,
        "randomized_seed": cfg.randomized_seed,
    }

    with h5py.File(source_path, "r") as source_file, h5py.File(
            output_path, "a",
            ) as output_file:
        for attr_name, expected_value in root_metadata.items():
            if (
                attr_name in output_file.attrs
                and output_file.attrs[attr_name] != expected_value
            ):
                raise ValueError(
                    f"{output_path.name} has {attr_name}="
                    f"{output_file.attrs[attr_name]!r}; "
                    f"expected {expected_value!r}."
                )
            # end if incompatible stored metadata
            output_file.attrs[attr_name] = expected_value
        # end for attr_name

        for movie_index, dataset_name in enumerate(dataset_names, start=1):
            source_dataset = source_file[dataset_name]
            if dataset_name in output_file and not cfg.overwrite:
                print(
                    f"[{movie_index}/{len(dataset_names)}] "
                    f"Skipping existing {dataset_name}",
                    flush=True,
                )
                continue
            # end if existing movie
            if dataset_name in output_file:
                del output_file[dataset_name]
            # end if overwrite movie

            temporary_name = f"__incomplete__{dataset_name}"
            if temporary_name in output_file:
                del output_file[temporary_name]
            # end if stale temporary dataset
            output_dataset = output_file.create_dataset(
                temporary_name,
                shape=(source_dataset.shape[0], cfg.n_components),
                dtype=np.float32,
                chunks=True,
                compression=cfg.compression,
            )
            # Recompute scores from the final PCA components so the stored
            # values exactly equal `pca.transform` for every movie frame.
            for start in range(
                    0, source_dataset.shape[0], cfg.projection_batch_size,
                    ):
                stop = min(
                    start + cfg.projection_batch_size,
                    source_dataset.shape[0],
                )
                pixel_batch = np.asarray(
                    source_dataset[start:stop], dtype=np.float32,
                )
                output_dataset[start:stop] = pca.transform(
                    pixel_batch,
                ).astype(
                    np.float32,
                    copy=False,
                )
            # end for start
            for attr_name, attr_value in source_dataset.attrs.items():
                output_dataset.attrs[attr_name] = attr_value
            # end for dataset metadata
            output_file.move(temporary_name, dataset_name)
            output_file.flush()
            print(
                f"[{movie_index}/{len(dataset_names)}] projected {dataset_name}",
                flush=True,
            )
        # end for dataset_name
    # end with HDF5 files
    return output_path
# EOF


"""
main
Fits PCA to RGB pixel vectors and saves every movie's temporal PC scores.
"""
def main():
    cfg = parse_args()
    model_dir = Path(cfg.output_dir or Path(paths["data_path"]) / "models")
    source_path = Path(
        cfg.source_path
        or model_dir
        / f"pixel_values_rgb_step{cfg.pixel_step}_{cfg.dataset_name}.h5"
    ).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Pixel feature file does not exist: {source_path}")
    # end if source file is missing

    dataset_names, n_frames, n_features = inspect_pixel_file(source_path)
    model_name = f"pixel_values_rgb_step{cfg.pixel_step}"
    output_path = (
        model_dir
        / f"{model_name}_pixels_{cfg.dataset_name}_PC{cfg.n_components}pool.h5"
    )
    print(cfg, flush=True)
    print(
        f"RGB PCA input: {len(dataset_names)} movies x {n_frames} frames, "
        f"{n_features} RGB features per frame",
        flush=True,
    )
    pca, pca_path = fit_pixel_pca(
        cfg,
        source_path,
        dataset_names,
        n_frames,
        n_features,
        model_name,
    )
    project_pixel_pca(
        cfg,
        source_path,
        output_path,
        dataset_names,
        pca,
        pca_path,
        model_name,
    )
    print(
        f"Saved {cfg.n_components} RGB-pixel PCs to {output_path}\n"
        f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )
# EOF


if __name__ == "__main__":
    main()
# EOF
