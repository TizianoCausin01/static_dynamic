"""
Last-frame layer-depth versus latency after removing the earlier model RDM.

For every model layer, the final-frame model RDM (2500 ms) is regressed on the
same layer's RDM a fixed delay earlier (2250 ms) across stimulus pairs. Both the
raw and the residual final-frame RDM are compared with every dynamic-neural RDM
time point; the neural data are never residualized. Centroid and peak latencies
inside a window around the final frame are then rank-correlated with normalized
layer depth to give the temporal score.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

from image_processing.video_feature_extraction import (
    list_video_feature_files, load_aligned_video_features,
    match_feature_stimulus_names,
)
from useful_stuff.general_utils import TimeSeries, create_RDM

from .split_half_rsa import cross_temporal_similarity


"""
regress_out_rdm
Remove one predictor RDM from a target RDM across matched stimulus pairs.

INPUT:
    - target_rdm: np.ndarray -> final-frame vectorized RDM.
    - predictor_rdm: np.ndarray -> earlier-frame vectorized RDM.
    - fit_intercept: bool -> whether to include a constant regression term.

OUTPUT:
    - residual_rdm: np.ndarray -> target variance unexplained by the predictor.
    - regression_summary: dict -> coefficients and variance diagnostics.
"""
def regress_out_rdm(target_rdm, predictor_rdm, fit_intercept=True):
    target_rdm = np.asarray(target_rdm, dtype=float)
    predictor_rdm = np.asarray(predictor_rdm, dtype=float)
    if target_rdm.ndim != 1 or target_rdm.shape != predictor_rdm.shape:
        raise ValueError(
            "target_rdm and predictor_rdm must be matching vectors."
        )
    # end if RDM shapes do not match
    if not np.all(np.isfinite(target_rdm)) or not np.all(
            np.isfinite(predictor_rdm)):
        raise ValueError("RDM regression requires finite distances.")
    # end if an RDM contains non-finite values

    # One observation per unique stimulus pair.
    design_columns = [predictor_rdm]
    if fit_intercept:
        design_columns.append(np.ones_like(predictor_rdm))
    # end if fit_intercept
    design_matrix = np.column_stack(design_columns)
    coefficients = np.linalg.lstsq(
        design_matrix, target_rdm, rcond=None,
    )[0]
    residual_rdm = target_rdm - design_matrix @ coefficients

    target_variance = np.var(target_rdm)
    regression_summary = {
        "beta": float(coefficients[0]),
        "variance_removed": float(
            1 - np.var(residual_rdm) / target_variance
            if target_variance > 0 else np.nan
        ),
        "predictor_target_spearman": float(spearmanr(
            predictor_rdm, target_rdm,
        ).statistic),
        "predictor_residual_spearman": float(spearmanr(
            predictor_rdm, residual_rdm,
        ).statistic),
    }
    return residual_rdm, regression_summary
# EOF


"""
summarize_layer_timing
Compute smoothed curves plus centroid and peak latency for every model layer.

Curves are smoothed along neural time, then values at or below the cutoff
(including all negative values) receive zero weight inside each window.

INPUT:
    - similarity: np.ndarray -> layers x dynamic-neural-time RSA values.
    - time_ms: np.ndarray -> dynamic neural time coordinate.
    - frame_onset_ms: float -> latencies are expressed relative to this time.
    - centroid_window_ms: tuple[float, float] -> end-exclusive centroid window.
    - peak_window_ms: tuple[float, float] -> end-exclusive peak window.
    - smoothing_sigma: float -> Gaussian sigma in samples.
    - centroid_cutoff: float -> minimum RSA carrying centroid weight.
    - peak_cutoff: float -> minimum RSA eligible as a peak.

OUTPUT:
    - smoothed_similarity: np.ndarray -> curves used for timing summaries.
    - centroid_latency_ms: np.ndarray -> positive-mass centroid minus onset.
    - peak_latency_ms: np.ndarray -> positive peak time minus onset.
    - peak_similarity: np.ndarray -> smoothed peak RSA inside the peak window.
"""
def summarize_layer_timing(
        similarity, time_ms, frame_onset_ms, centroid_window_ms,
        peak_window_ms, smoothing_sigma, centroid_cutoff, peak_cutoff,
        ):
    smoothed_similarity = gaussian_filter1d(
        similarity, sigma=smoothing_sigma, axis=1,
    )
    centroid_mask = (
        (time_ms >= centroid_window_ms[0]) & (time_ms < centroid_window_ms[1])
    )
    peak_mask = (time_ms >= peak_window_ms[0]) & (time_ms < peak_window_ms[1])
    if not centroid_mask.any() or not peak_mask.any():
        raise ValueError("Timing windows contain no neural samples.")
    # end if a timing window is empty

    n_layers = smoothed_similarity.shape[0]
    centroid_latency_ms = np.full(n_layers, np.nan)
    peak_latency_ms = np.full(n_layers, np.nan)
    for layer_index, layer_curve in enumerate(smoothed_similarity):
        centroid_curve = layer_curve[centroid_mask]
        centroid_weights = np.where(
            np.isfinite(centroid_curve) & (centroid_curve > centroid_cutoff),
            centroid_curve, 0,
        )
        if centroid_weights.sum() > 0:
            centroid_latency_ms[layer_index] = np.average(
                time_ms[centroid_mask], weights=centroid_weights,
            ) - frame_onset_ms
        # end if centroid mass exists

        peak_curve = layer_curve[peak_mask]
        peak_values = np.where(
            np.isfinite(peak_curve) & (peak_curve > peak_cutoff),
            peak_curve, 0,
        )
        if peak_values.sum() > 0:
            peak_latency_ms[layer_index] = (
                time_ms[peak_mask][np.argmax(peak_values)] - frame_onset_ms
            )
        # end if positive peak exists
    # end for layer_index, layer_curve
    peak_similarity = np.nanmax(smoothed_similarity[:, peak_mask], axis=1)
    return (
        smoothed_similarity, centroid_latency_ms, peak_latency_ms,
        peak_similarity,
    )
# EOF


"""
depth_latency_score
Spearman rho between normalized layer depth and latency over valid layers.

INPUT:
    - layer_depths: np.ndarray -> normalized depth in [0, 1] per layer.
    - latency_ms: np.ndarray -> latency per layer, NaN when undefined.

OUTPUT:
    - score: dict -> rho, p value, and number of layers entering the score.
"""
def depth_latency_score(layer_depths, latency_ms):
    valid_layers = np.isfinite(latency_ms)
    rho, pvalue = np.nan, np.nan
    # A rank correlation needs three layers and some variation in latency.
    if valid_layers.sum() >= 3 and np.ptp(latency_ms[valid_layers]) > 0:
        result = spearmanr(layer_depths[valid_layers], latency_ms[valid_layers])
        rho, pvalue = float(result.statistic), float(result.pvalue)
    # end if enough informative layers
    return {"rho": rho, "pvalue": pvalue, "n_layers": int(valid_layers.sum())}
# EOF


"""
last_frame_residual_similarity
Raw and residualized final-frame model-neural RSA timecourses for one model.

INPUT:
    - dynamic_neural_rdms: np.ndarray -> time x stimulus pairs neural RDMs.
    - feature_names: list[str] -> movie names in the neural stimulus order.
    - model_features_dir: Path -> directory of HDF5 layer feature files.
    - model_name: str -> registry model name.
    - model_dataset_name: str -> dataset token of the feature files.
    - model_pooling: str | None -> pooling token of the feature files.
    - analysis_fs: float -> sampling rate the model features are resampled to.
    - previous_frame_ms: float -> predictor RDM time.
    - last_frame_ms: float -> target RDM time.
    - model_rdm_metric: str -> distance used for model RDMs.
    - rsa_metric: str -> model-neural RDM comparison metric.
    - fit_intercept: bool -> include a constant in the RDM regression.

OUTPUT:
    - layer_names: list[str] -> layers in forward-pass order.
    - raw_similarity: np.ndarray -> layers x neural time, raw final-frame RSA.
    - residual_similarity: np.ndarray -> layers x neural time, residual RSA.
    - regression_summaries: list[dict] -> per-layer regression diagnostics.
"""
def last_frame_residual_similarity(
        dynamic_neural_rdms, feature_names, model_features_dir, model_name,
        model_dataset_name, model_pooling, analysis_fs, previous_frame_ms,
        last_frame_ms, model_rdm_metric, rsa_metric, fit_intercept=True,
        ):
    feature_paths = list_video_feature_files(
        model_features_dir, model_name, model_dataset_name, model_pooling,
    )
    # Resolve the exact HDF5 dataset names once; every layer shares them.
    model_stimulus_names = match_feature_stimulus_names(
        feature_paths[0], feature_names,
    )

    layer_names, raw_similarity, residual_similarity = [], [], []
    regression_summaries = []
    for feature_path in feature_paths:
        # features x frames x stimuli, resampled to the neural analysis rate.
        model_features, layer_name, source_fs = load_aligned_video_features(
            feature_path, model_stimulus_names, frame_index=None,
        )
        model_ts = TimeSeries(model_features, fs=source_fs)
        model_ts.resample(analysis_fs)
        model_time_ms = np.arange(len(model_ts)) * 1000 / model_ts.get_fs()

        frame_rdms = []
        for frame_ms in (previous_frame_ms, last_frame_ms):
            frame_index = int(np.argmin(np.abs(model_time_ms - frame_ms)))
            if not np.isclose(model_time_ms[frame_index], frame_ms):
                raise ValueError(
                    f"{layer_name}: resampled model grid misses {frame_ms} ms."
                )
            # end if frame does not lie on the model grid
            frame_rdms.append(create_RDM(
                np.ascontiguousarray(model_ts.get_array()[:, frame_index, :]),
                metric=model_rdm_metric,
            ))
        # end for frame_ms
        previous_frame_rdm, last_frame_rdm = frame_rdms
        residual_rdm, regression_summary = regress_out_rdm(
            last_frame_rdm, previous_frame_rdm, fit_intercept=fit_intercept,
        )

        # One model RDM against every neural time point -> (time,) vector.
        raw_similarity.append(cross_temporal_similarity(
            dynamic_neural_rdms, last_frame_rdm[np.newaxis, :],
            metric=rsa_metric,
        )[:, 0])
        residual_similarity.append(cross_temporal_similarity(
            dynamic_neural_rdms, residual_rdm[np.newaxis, :],
            metric=rsa_metric,
        )[:, 0])
        layer_names.append(layer_name)
        regression_summaries.append(regression_summary)
    # end for feature_path
    return (
        layer_names, np.stack(raw_similarity), np.stack(residual_similarity),
        regression_summaries,
    )
# EOF


"""
model_frame_similarity_by_layer
Layer-wise model-neural RSA timecourses for the static response and for three
slices of the movie response: the first frame, the last frame, and the last
frame after regressing out the earlier-frame model RDM (as in
last_frame_residual_similarity). Only the model RDMs change across the four
conditions; the neural RDMs are never residualized.

INPUT:
    - static_neural_rdms: np.ndarray -> static time x stimulus pairs neural RDMs.
    - dynamic_neural_rdms: np.ndarray -> movie time x stimulus pairs neural RDMs.
    - feature_names: list[str] -> movie names in the neural stimulus order.
    - model_features_dir: Path -> directory of HDF5 layer feature files.
    - model_name: str -> registry model name.
    - model_dataset_name: str -> dataset token of the feature files.
    - model_pooling: str | None -> pooling token of the feature files.
    - analysis_fs: float -> sampling rate the model features are resampled to.
    - first_frame_ms: float -> model time of the first-frame RDM.
    - previous_frame_ms: float -> model time of the regressed-out RDM.
    - last_frame_ms: float -> model time of the last-frame (and static) RDM.
    - model_rdm_metric: str -> distance used for model RDMs.
    - rsa_metric: str -> model-neural RDM comparison metric.
    - fit_intercept: bool -> include a constant in the RDM regression.

OUTPUT:
    - layer_names: list[str] -> layers in forward-pass order.
    - similarity: dict[str, np.ndarray] -> layers x neural time RSA for
        "static", "first_frame", "last_frame" and "last_frame_residual".
    - variance_removed: np.ndarray -> per-layer fraction of last-frame RDM
        variance explained by the previous-frame RDM.
"""
def model_frame_similarity_by_layer(
        static_neural_rdms, dynamic_neural_rdms, feature_names,
        model_features_dir, model_name, model_dataset_name, model_pooling,
        analysis_fs, first_frame_ms, previous_frame_ms, last_frame_ms,
        model_rdm_metric, rsa_metric, fit_intercept=True,
        ):
    feature_paths = list_video_feature_files(
        model_features_dir, model_name, model_dataset_name, model_pooling,
    )
    # Resolve the exact HDF5 dataset names once; every layer shares them.
    model_stimulus_names = match_feature_stimulus_names(
        feature_paths[0], feature_names,
    )

    layer_names, variance_removed = [], []
    similarity = {
        "static": [], "first_frame": [], "last_frame": [],
        "last_frame_residual": [],
    }
    for feature_path in feature_paths:
        # features x frames x stimuli, resampled to the neural analysis rate.
        model_features, layer_name, source_fs = load_aligned_video_features(
            feature_path, model_stimulus_names, frame_index=None,
        )
        model_ts = TimeSeries(model_features, fs=source_fs)
        model_ts.resample(analysis_fs)
        model_time_ms = np.arange(len(model_ts)) * 1000 / model_ts.get_fs()

        frame_rdms = {}
        for frame_name, frame_ms in (
                ("first", first_frame_ms), ("previous", previous_frame_ms),
                ("last", last_frame_ms)):
            frame_index = int(np.argmin(np.abs(model_time_ms - frame_ms)))
            if not np.isclose(model_time_ms[frame_index], frame_ms):
                raise ValueError(
                    f"{layer_name}: resampled model grid misses {frame_ms} ms."
                )
            # end if frame does not lie on the model grid
            frame_rdms[frame_name] = create_RDM(
                np.ascontiguousarray(model_ts.get_array()[:, frame_index, :]),
                metric=model_rdm_metric,
            )
        # end for frame_name, frame_ms
        residual_rdm, regression_summary = regress_out_rdm(
            frame_rdms["last"], frame_rdms["previous"],
            fit_intercept=fit_intercept,
        )

        # One model RDM against every neural time point -> (time,) vector.
        condition_inputs = (
            ("static", static_neural_rdms, frame_rdms["last"]),
            ("first_frame", dynamic_neural_rdms, frame_rdms["first"]),
            ("last_frame", dynamic_neural_rdms, frame_rdms["last"]),
            ("last_frame_residual", dynamic_neural_rdms, residual_rdm),
        )
        for condition_name, neural_rdms, model_rdm in condition_inputs:
            similarity[condition_name].append(cross_temporal_similarity(
                neural_rdms, model_rdm[np.newaxis, :], metric=rsa_metric,
            )[:, 0])
        # end for condition_name
        layer_names.append(layer_name)
        variance_removed.append(regression_summary["variance_removed"])
    # end for feature_path
    similarity = {
        condition_name: np.stack(curves)
        for condition_name, curves in similarity.items()
    }
    return layer_names, similarity, np.asarray(variance_removed)
# EOF
