import json
from pathlib import Path
import re

import numpy as np

from image_processing.video_feature_extraction import load_aligned_video_features
from useful_stuff.general_utils import TimeSeries, create_RDM, get_lagplot_subset

from .split_half_rsa import compute_rdm_timeseries, cross_temporal_similarity


"""
normalize_rsa_metric
Map the user-facing Pearson alias to the metric name used by the RSA utilities.

INPUT:
    - rsa_metric: str -> pearson, correlation, or spearman

OUTPUT:
    - normalized_metric: str -> correlation or spearman
"""
def normalize_rsa_metric(rsa_metric: str) -> str:
    normalized_metric = rsa_metric.lower().strip()
    if normalized_metric == "pearson":
        normalized_metric = "correlation"
    # end if normalized_metric == "pearson"
    if normalized_metric not in {"correlation", "spearman"}:
        raise ValueError(
            "rsa_metric must be 'pearson', 'correlation', or 'spearman'."
        )
    # end if invalid normalized_metric
    return normalized_metric
# EOF


"""
filename_token
Convert one analysis label into a filesystem-safe filename component.

INPUT:
    - value: object -> value represented in the filename

OUTPUT:
    - token: str -> compact filename-safe value
"""
def filename_token(value) -> str:
    if value is None:
        return "none"
    # end if value is None
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    token = token.strip("-_")
    if not token:
        raise ValueError(f"Cannot construct a filename token from {value!r}.")
    # end if not token
    return token
# EOF


"""
build_neural_model_rsa_filename
Build a collision-resistant result name from all result-defining parameters.

INPUT:
    - experiment_name: str -> label shared by the static and dynamic analyses
    - model_name: str -> computational model name
    - layer_name: str -> network layer stored in the feature file
    - signal_rdm_metric: str -> neural-signal RDM dissimilarity
    - model_rdm_metric: str -> model-feature RDM dissimilarity
    - rsa_metric: str -> RDM correlation measure
    - new_fs: float -> neural and dynamic-model analysis frequency
    - model_dataset_name: str -> extracted-feature dataset label
    - model_pooling: str | None -> feature pooling label
    - channel_name: str -> neural channel selection label
    - normalization: str | None -> neural normalization label
    - static_crop_ms: float | None -> static response duration
    - dynamic_crop_ms: float | None -> dynamic response duration
    - model_frame_index: int -> model frame used for static RSA

OUTPUT:
    - filename: str -> parameter-complete NPZ filename
"""
def build_neural_model_rsa_filename(
        experiment_name: str,
        model_name: str,
        layer_name: str,
        signal_rdm_metric: str,
        model_rdm_metric: str,
        rsa_metric: str,
        new_fs: float,
        model_dataset_name: str,
        model_pooling: str | None,
        channel_name: str,
        normalization: str | None,
        static_crop_ms: float | None,
        dynamic_crop_ms: float | None,
        model_frame_index: int,
        ) -> str:
    fs_name = f"{new_fs:g}"
    pooling_name = "none" if model_pooling is None else model_pooling
    normalization_name = "raw" if normalization is None else normalization
    static_crop_name = "all" if static_crop_ms is None else f"{static_crop_ms:g}ms"
    dynamic_crop_name = "all" if dynamic_crop_ms is None else f"{dynamic_crop_ms:g}ms"
    rsa_name = "pearson" if rsa_metric == "correlation" else rsa_metric
    frame_name = "last" if model_frame_index == -1 else str(model_frame_index)

    filename_parts = [
        experiment_name,
        model_name,
        layer_name,
        f"{signal_rdm_metric}-{model_rdm_metric}",
        f"RSA_{rsa_name}",
        f"{fs_name}Hz",
        model_dataset_name,
        f"{pooling_name}pool",
        channel_name,
        normalization_name,
        f"static-{static_crop_name}",
        f"dynamic-{dynamic_crop_name}",
        f"frame-{frame_name}",
    ]
    return "_".join(filename_token(part) for part in filename_parts) + ".npz"
# EOF


"""
compute_neural_rdm_timeseries
Compute the neural RDM once per timepoint for reuse across every model layer.

INPUT:
    - neural_ts: TimeSeries -> channels x time x stimuli neural responses
    - rdm_metric: str -> neural RDM dissimilarity measure

OUTPUT:
    - neural_rdms: np.ndarray -> neural time x stimulus-pair distances
"""
def compute_neural_rdm_timeseries(
        neural_ts: TimeSeries,
        rdm_metric: str,
        ) -> np.ndarray:
    return compute_rdm_timeseries(neural_ts.get_array(), rdm_metric)
# EOF


"""
compute_layer_neural_model_rsa
Compute static final-frame RSA and dynamic cross-temporal RSA for one model layer.

INPUT:
    - feature_path: str | Path -> layer-specific sequential feature HDF5 file
    - static_feature_names: list[str] -> static stimuli mapped to HDF5 datasets
    - dynamic_feature_names: list[str] -> dynamic stimuli mapped to HDF5 datasets
    - static_neural_rdms: np.ndarray -> static neural time x RDM entries
    - dynamic_neural_rdms: np.ndarray -> dynamic neural time x RDM entries
    - analysis_fs: float -> target sampling frequency for dynamic model features
    - model_rdm_metric: str -> model-feature RDM dissimilarity measure
    - rsa_metric: str -> pearson/correlation or spearman RDM similarity
    - model_frame_index: int -> feature frame used as the static model RDM

OUTPUT:
    - results: dict[str, object] -> RSA arrays and layer/model time metadata
"""
def compute_layer_neural_model_rsa(
        feature_path: str | Path,
        static_feature_names: list[str],
        dynamic_feature_names: list[str],
        static_neural_rdms: np.ndarray,
        dynamic_neural_rdms: np.ndarray,
        analysis_fs: float,
        model_rdm_metric: str,
        rsa_metric: str,
        model_frame_index: int = -1,
        ) -> dict[str, object]:
    rsa_metric = normalize_rsa_metric(rsa_metric)

    # A single final-frame RDM is compared with every static neural timepoint.
    static_features, static_layer_name, static_source_fs = (
        load_aligned_video_features(
            feature_path,
            static_feature_names,
            frame_index=model_frame_index,
        )
    )
    static_model_rdm = create_RDM(static_features, metric=model_rdm_metric)
    static_rsa = cross_temporal_similarity(
        static_neural_rdms,
        static_model_rdm[np.newaxis, :],
        metric=rsa_metric,
    )[:, 0]

    # Sequential model RDMs are resampled to the neural grid before dRSA.
    dynamic_features, dynamic_layer_name, dynamic_source_fs = (
        load_aligned_video_features(
            feature_path,
            dynamic_feature_names,
            frame_index=None,
        )
    )
    if static_layer_name != dynamic_layer_name:
        raise ValueError(
            "The static and dynamic feature reads returned different layer names."
        )
    # end if inconsistent layer name
    if not np.isclose(static_source_fs, dynamic_source_fs):
        raise ValueError(
            "The static and dynamic feature reads returned different source rates."
        )
    # end if inconsistent source rate

    dynamic_model_ts = TimeSeries(dynamic_features, fs=dynamic_source_fs)
    dynamic_model_ts.resample(analysis_fs)
    dynamic_model_rdms = compute_rdm_timeseries(
        dynamic_model_ts.get_array(), model_rdm_metric,
    )
    dynamic_rsa = cross_temporal_similarity(
        dynamic_neural_rdms,
        dynamic_model_rdms,
        metric=rsa_metric,
    )
    return {
        "static_rsa": static_rsa,
        "dynamic_rsa": dynamic_rsa,
        "layer_name": dynamic_layer_name,
        "source_model_fs": dynamic_source_fs,
        "model_fs": dynamic_model_ts.get_fs(),
        "model_timepoints": len(dynamic_model_ts),
    }
# EOF


"""
save_layer_neural_model_rsa
Save one layer's static/dynamic RSA arrays with coordinates and full metadata.

INPUT:
    - output_path: str | Path -> destination NPZ path
    - results: dict[str, object] -> output of compute_layer_neural_model_rsa
    - metadata: dict -> JSON-serializable analysis parameters and provenance
    - static_time_ms: np.ndarray -> static neural time coordinates
    - dynamic_neural_time_ms: np.ndarray -> dynamic neural time coordinates
    - selected_channel_numbers: np.ndarray -> retained one-based MATLAB channels
    - static_stimulus_names: list[str] -> static neural stimulus ordering
    - dynamic_stimulus_names: list[str] -> dynamic neural stimulus ordering

OUTPUT:
    - output_path: Path -> saved NPZ path
"""
def save_layer_neural_model_rsa(
        output_path: str | Path,
        results: dict[str, object],
        metadata: dict,
        static_time_ms: np.ndarray,
        dynamic_neural_time_ms: np.ndarray,
        selected_channel_numbers: np.ndarray,
        static_stimulus_names: list[str],
        dynamic_stimulus_names: list[str],
        ) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_time_ms = (
        np.arange(results["model_timepoints"]) * 1000 / results["model_fs"]
    )
    np.savez_compressed(
        output_path,
        static_rsa=results["static_rsa"],
        dynamic_rsa=results["dynamic_rsa"],
        static_time_ms=static_time_ms,
        dynamic_neural_time_ms=dynamic_neural_time_ms,
        dynamic_model_time_ms=model_time_ms,
        selected_channel_numbers=np.asarray(selected_channel_numbers, dtype=int),
        layer_name=np.asarray(results["layer_name"]),
        source_model_fs=np.asarray(results["source_model_fs"]),
        model_fs=np.asarray(results["model_fs"]),
        static_stimulus_names=np.asarray(static_stimulus_names),
        dynamic_stimulus_names=np.asarray(dynamic_stimulus_names),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path
# EOF


"""
compute_model_rdm_timeseries
Compute one model RDM per source frame and upsample the RDM timecourse to the
neural analysis grid. Upsampling in TimeSeries repeats frame indices, so this
equals upsampling the features first while never materializing the large
upsampled feature array (e.g. dense optical flow).

INPUT:
    - model_features: np.ndarray -> features x source frames x stimuli
    - source_fs: float -> model frame rate
    - analysis_fs: float -> neural analysis sampling rate (>= source_fs)
    - rdm_metric: str -> model-feature RDM dissimilarity measure

OUTPUT:
    - model_rdms: np.ndarray -> analysis time x stimulus-pair distances
"""
def compute_model_rdm_timeseries(
        model_features: np.ndarray,
        source_fs: float,
        analysis_fs: float,
        rdm_metric: str,
        ) -> np.ndarray:
    if analysis_fs < source_fs:
        raise ValueError(
            "analysis_fs must be >= source_fs; downsampling RDMs is not "
            "equivalent to downsampling features."
        )
    # end if analysis_fs < source_fs
    source_rdms = compute_rdm_timeseries(model_features, rdm_metric)
    # TimeSeries expects features x time, so RDM entries become the features.
    rdm_ts = TimeSeries(np.ascontiguousarray(source_rdms.T), fs=source_fs)
    rdm_ts.resample(analysis_fs)
    return np.ascontiguousarray(rdm_ts.get_array().T)
# EOF


"""
drsa_lag_profile
Average a neural x model dRSA matrix along its diagonals within a time window
and express every diagonal as a physical lag (neural time - model time).

INPUT:
    - drsa_matrix: np.ndarray -> neural time x model time dRSA values
    - analysis_fs: float -> shared sampling rate of both time axes
    - analysis_end_ms: float -> only times before this bound enter the profile
    - max_lag_ms: float -> largest absolute lag returned
    - model_time_offset_ms: float -> physical time of model sample 0

OUTPUT:
    - lag_ms: np.ndarray -> physical lags; positive means neural lags model
    - lag_profile: np.ndarray -> mean dRSA at every lag
"""
def drsa_lag_profile(
        drsa_matrix: np.ndarray,
        analysis_fs: float,
        analysis_end_ms: float,
        max_lag_ms: float,
        model_time_offset_ms: float = 0.0,
        ) -> tuple[np.ndarray, np.ndarray]:
    neural_time_ms = np.arange(drsa_matrix.shape[0]) * 1000 / analysis_fs
    model_time_ms = (
        model_time_offset_ms
        + np.arange(drsa_matrix.shape[1]) * 1000 / analysis_fs
    )
    neural_indices = np.flatnonzero(neural_time_ms < analysis_end_ms)
    model_indices = np.flatnonzero(model_time_ms < analysis_end_ms)
    max_lag_samples = min(
        int(round(max_lag_ms * analysis_fs / 1000)),
        len(neural_indices) - 1,
        len(model_indices) - 1,
    )
    if max_lag_samples < 1:
        raise ValueError("The lag-analysis window is too short.")
    # end if max_lag_samples < 1

    lag_profile = get_lagplot_subset(
        drsa_matrix,
        neural_idx=neural_indices,
        model_idx=model_indices,
        max_lag=max_lag_samples,
    )
    # get_lagplot_subset orders entries by neural index - model index.
    sample_lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    lag_ms = sample_lags * 1000 / analysis_fs - model_time_offset_ms
    return lag_ms, lag_profile
# EOF


"""
fit_ols_svd_subspace
Fit an ordinary least-squares model-to-neural map and return its ordered model
input directions. The directions are the left singular vectors of the
features-by-neurons coefficient matrix; no singular-value gain is retained.

INPUT:
    - model_samples: np.ndarray -> samples x model features
    - neural_samples: np.ndarray -> matched samples x neural features
    - relative_tolerance: float or None -> numerical-rank cutoff relative to
      the largest coefficient singular value

OUTPUT:
    - result: dict -> orthonormal model basis, singular values, rank, and OLS fit
"""
def fit_ols_svd_subspace(
        model_samples: np.ndarray,
        neural_samples: np.ndarray,
        relative_tolerance: float | None = None,
        ) -> dict[str, np.ndarray | int | float]:
    model_samples = np.asarray(model_samples, dtype=float)
    neural_samples = np.asarray(neural_samples, dtype=float)
    if model_samples.ndim != 2 or neural_samples.ndim != 2:
        raise ValueError("model_samples and neural_samples must be two-dimensional.")
    # end if inputs are not matrices
    if model_samples.shape[0] != neural_samples.shape[0]:
        raise ValueError("Model and neural spaces must contain the same samples.")
    # end if sample counts differ
    if model_samples.shape[0] < 2:
        raise ValueError("At least two matched samples are required.")
    # end if too few samples
    if not np.all(np.isfinite(model_samples)) or not np.all(
            np.isfinite(neural_samples)):
        raise ValueError("Model and neural samples must contain only finite values.")
    # end if non-finite values

    model_mean = model_samples.mean(axis=0, keepdims=True)
    neural_mean = neural_samples.mean(axis=0, keepdims=True)
    centered_model = model_samples - model_mean
    centered_neural = neural_samples - neural_mean

    # This is the coefficient matrix of OLS with an intercept. Its model-side
    # singular vectors define the nested predictive subspaces.
    coefficient, _, _, _ = np.linalg.lstsq(
        centered_model, centered_neural, rcond=None,
    )
    model_basis, singular_values, neural_basis_t = np.linalg.svd(
        coefficient, full_matrices=False,
    )

    if relative_tolerance is None:
        relative_tolerance = (
            max(coefficient.shape) * np.finfo(float).eps
        )
    # end if relative_tolerance is None
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative.")
    # end if relative_tolerance is negative

    largest_singular_value = singular_values.max(initial=0.0)
    active = singular_values > relative_tolerance * largest_singular_value
    return {
        "basis": model_basis[:, active],
        "singular_values": singular_values,
        "active": active,
        "rank": int(np.sum(active)),
        "coefficient": coefficient,
        "model_mean": model_mean,
        "neural_mean": neural_mean,
        "neural_basis_t": neural_basis_t,
        "relative_tolerance": float(relative_tolerance),
    }
# EOF


"""
project_onto_ols_subspace
Express model samples in the first fitted OLS/SVD input directions without
applying regression coefficients, singular values, or output-space stretches.

INPUT:
    - model_samples: np.ndarray -> array whose final axis contains model features
    - subspace_fit: dict -> output of fit_ols_svd_subspace
    - rank: int -> number of ordered model directions to retain

OUTPUT:
    - projected_samples: np.ndarray -> input array with final axis replaced by rank
"""
def project_onto_ols_subspace(
        model_samples: np.ndarray,
        subspace_fit: dict,
        rank: int,
        ) -> np.ndarray:
    model_samples = np.asarray(model_samples)
    basis = np.asarray(subspace_fit["basis"])
    if model_samples.ndim < 2:
        raise ValueError("model_samples must have a sample and feature axis.")
    # end if model_samples has too few dimensions
    if model_samples.shape[-1] != basis.shape[0]:
        raise ValueError(
            "The final model_samples axis does not match the fitted feature count."
        )
    # end if feature counts differ
    if not 1 <= rank <= basis.shape[1]:
        raise ValueError(f"rank must be between 1 and {basis.shape[1]}.")
    # end if rank is invalid

    # Deliberately omit both centering and Sigma. Constant translations do not
    # affect Euclidean/cosine_cnt RDMs, and this matches the metrics_II transform.
    return model_samples @ basis[:, :rank]
# EOF
