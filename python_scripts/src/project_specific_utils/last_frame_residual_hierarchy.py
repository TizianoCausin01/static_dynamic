"""
Last-frame layer-depth versus latency after removing the earlier model RDM.

For every model layer, the final-frame model RDM (2500 ms) is regressed on the
same layer's RDM a fixed delay earlier (2250 ms) across stimulus pairs, or,
at the feature level, each final-frame unit is regressed on the same unit at
the earlier frame across stimuli before the residual RDM is built. Both the
raw and the residual final-frame RDM are compared with every dynamic-neural RDM
time point; the neural data are never residualized. Centroid and peak latencies
inside a window around the final frame are then rank-correlated with normalized
layer depth to give the temporal score.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata, spearmanr

from image_processing.video_feature_extraction import (
    list_video_feature_files, load_aligned_video_features,
    match_feature_stimulus_names,
)
from useful_stuff.general_utils import TimeSeries, create_RDM

from .split_half_rsa import compute_rdm_timeseries, cross_temporal_similarity
from .static_response_regress_out import _slicewise_regress_out, window_patterns
from .time_series_regression import encoding_timecourse


"""
regress_out_rdm
Remove one or several predictor RDMs from a target RDM across matched stimulus
pairs with one OLS fit (stimulus pairs are the samples).

INPUT:
    - target_rdm: np.ndarray -> final-frame vectorized RDM.
    - predictor_rdm: np.ndarray -> earlier-frame vectorized RDM, or
        predictors x pairs to regress several earlier frames out together.
    - fit_intercept: bool -> whether to include a constant regression term.
    - rank_space: bool -> rank-transform the target and every predictor RDM
        across pairs before the fit, so the residual is free of the monotonic
        (Spearman) relation the RSA measures, not only the linear one.

OUTPUT:
    - residual_rdm: np.ndarray -> target variance unexplained by the predictors
        (in rank units when rank_space is True).
    - regression_summary: dict -> coefficients and variance diagnostics; the
        "beta" and Spearman entries refer to the first predictor, "betas"
        lists one slope per predictor.
"""
def regress_out_rdm(target_rdm, predictor_rdm, fit_intercept=True, rank_space=False):
    target_rdm = np.asarray(target_rdm, dtype=float)
    # predictors x pairs; a single RDM becomes one row.
    predictor_rdms = np.atleast_2d(np.asarray(predictor_rdm, dtype=float))
    if target_rdm.ndim != 1 or predictor_rdms.shape[1:] != target_rdm.shape:
        raise ValueError(
            "target_rdm and predictor_rdm must be matching vectors."
        )
    # end if RDM shapes do not match
    if not np.all(np.isfinite(target_rdm)) or not np.all(
            np.isfinite(predictor_rdms)):
        raise ValueError("RDM regression requires finite distances.")
    # end if an RDM contains non-finite values
    if rank_space:
        # Average ranks over pairs; ties share a rank as in spearmanr.
        target_rdm = rankdata(target_rdm)
        predictor_rdms = np.stack([rankdata(rdm) for rdm in predictor_rdms])
    # end if rank_space

    # One observation per unique stimulus pair, one column per predictor RDM.
    design_columns = list(predictor_rdms)
    if fit_intercept:
        design_columns.append(np.ones_like(target_rdm))
    # end if fit_intercept
    design_matrix = np.column_stack(design_columns)
    coefficients = np.linalg.lstsq(
        design_matrix, target_rdm, rcond=None,
    )[0]
    residual_rdm = target_rdm - design_matrix @ coefficients

    target_variance = np.var(target_rdm)
    regression_summary = {
        "beta": float(coefficients[0]),
        "betas": coefficients[:len(predictor_rdms)].tolist(),
        "variance_removed": float(
            1 - np.var(residual_rdm) / target_variance
            if target_variance > 0 else np.nan
        ),
        "predictor_target_spearman": float(spearmanr(
            predictor_rdms[0], target_rdm,
        ).statistic),
        "predictor_residual_spearman": float(spearmanr(
            predictor_rdms[0], residual_rdm,
        ).statistic),
    }
    return residual_rdm, regression_summary
# EOF


"""
regress_out_features
Remove each earlier-frame model unit from the same unit of the final frame.

Every feature dimension gets its own regression across stimuli, so the
residual keeps only the part of each unit's final-frame tuning that its
earlier-frame tuning cannot predict linearly.

INPUT:
    - target_features: np.ndarray -> features x stimuli, final frame.
    - predictor_features: np.ndarray -> features x stimuli, earlier frame.
    - fit_intercept: bool -> whether to include a per-unit constant term.

OUTPUT:
    - residual_features: np.ndarray -> features x stimuli residual.
    - regression_summary: dict -> mean slope and pooled variance removed.
"""
def regress_out_features(target_features, predictor_features, fit_intercept=True):
    target_features = np.asarray(target_features, dtype=float)
    predictor_features = np.asarray(predictor_features, dtype=float)
    if target_features.ndim != 2 or target_features.shape != predictor_features.shape:
        raise ValueError("Feature matrices must both be features x stimuli.")
    # end if feature shapes do not match

    # Centering both sides across stimuli is the closed form of the intercept.
    if fit_intercept:
        target_centered = target_features - target_features.mean(axis=1, keepdims=True)
        predictor_centered = predictor_features - predictor_features.mean(axis=1, keepdims=True)
    else:
        target_centered, predictor_centered = target_features, predictor_features
    # end if fit_intercept
    # One slope per unit: cov(prev, last) / var(prev); constant units get 0.
    predictor_power = (predictor_centered ** 2).sum(axis=1)
    slopes = np.divide(
        (predictor_centered * target_centered).sum(axis=1), predictor_power,
        out=np.zeros_like(predictor_power), where=predictor_power > 0,
    )
    residual_features = target_centered - slopes[:, np.newaxis] * predictor_centered

    target_power = (target_centered ** 2).sum()
    regression_summary = {
        "beta": float(np.mean(slopes)),
        "variance_removed": float(
            1 - (residual_features ** 2).sum() / target_power
            if target_power > 0 else np.nan
        ),
    }
    return residual_features, regression_summary
# EOF


"""
regress_out_features_ridge
Remove several earlier frames together from the final-frame features with one
multi-output ridge model across stimuli. The earlier frames are stacked into
one predictor, reduced by a PCA fitted on the training stimuli of each fold,
and every final-frame unit is predicted from those components. With
cv_type="kf" each stimulus is predicted by a model that never saw it, so the
residual is not shrunk by overfitting the ~100-stimulus sample.

INPUT:
    - target_features: np.ndarray -> features x stimuli, final frame.
    - predictor_features: list[np.ndarray] -> one features x stimuli matrix
        per earlier frame, stacked along features.
    - pca_variance: float | None -> fraction of predictor variance kept by the
        in-fold PCA; None regresses on the raw stacked features.
    - alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - cv_type: str -> "kf" for out-of-fold residuals, "same" for in-sample.
    - n_splits: int -> number of stimulus folds when cv_type is "kf".
    - fit_intercept: bool -> whether the ridge model has an intercept.
    - regression_type: str -> "ridge" (RidgeCV over alphas) or "lr" (OLS;
        with more predictors than training stimuli sklearn returns the
        minimum-norm least-squares solution).

OUTPUT:
    - residual_features: np.ndarray -> features x stimuli residual.
    - regression_summary: dict -> ridge penalty, PCA size, variance removed.
"""
def regress_out_features_ridge(
        target_features, predictor_features, pca_variance=0.95,
        alphas=(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5), cv_type="kf",
        n_splits=5, fit_intercept=True, regression_type="ridge",
        ):
    target_features = np.asarray(target_features, dtype=float)
    # (frames * features) x stimuli: all earlier frames form one predictor.
    stacked_predictor = np.concatenate(
        [np.asarray(features, dtype=float) for features in predictor_features],
    )
    # The slicewise helper expects features x slices x stimuli; one slice here.
    predicted_features, chosen_alphas, n_components = _slicewise_regress_out(
        stacked_predictor[:, np.newaxis, :], target_features[:, np.newaxis, :],
        regression_type, alphas, cv_type, n_splits, False, fit_intercept,
        pca_variance=pca_variance,
    )
    residual_features = target_features - predicted_features[:, 0, :]

    # Fraction of across-stimulus variance (summed over units) removed.
    target_variance = np.var(target_features, axis=1).sum()
    residual_variance = np.var(residual_features, axis=1).sum()
    regression_summary = {
        "alpha": float(chosen_alphas[0]),
        "n_components": float(n_components[0]),
        "variance_removed": float(
            1 - residual_variance / target_variance
            if target_variance > 0 else np.nan
        ),
    }
    return residual_features, regression_summary
# EOF


"""
project_frames_on_pcs
Project several frames of one layer onto a shared PCA basis in feature space.
The frames are pooled as samples (stimuli x frames), so with 100 stimuli and
three frames the basis spans at most 299 directions; keeping fewer components
removes dimensions the regression cannot constrain anyway.

INPUT:
    - frame_features: dict[str, np.ndarray] -> frame name -> features x stimuli.
    - n_components: int -> number of shared PCs kept.

OUTPUT:
    - projected_features: dict[str, np.ndarray] -> frame name -> PCs x stimuli.
    - variance_kept: float -> fraction of pooled variance in the kept PCs.
"""
def project_frames_on_pcs(frame_features, n_components):
    # features x (frames * stimuli): each frame-stimulus pattern is one sample.
    pooled = np.concatenate(
        [np.asarray(features, dtype=float) for features in frame_features.values()],
        axis=1,
    )
    if n_components >= pooled.shape[1]:
        raise ValueError(
            f"n_components={n_components} needs more than {pooled.shape[1]} samples."
        )
    # end if more components than centred samples
    pooled_mean = pooled.mean(axis=1, keepdims=True)
    # Left singular vectors of the centred pool are the PC axes in feature space.
    component_axes, singular_values, _ = np.linalg.svd(
        pooled - pooled_mean, full_matrices=False,
    )
    component_axes = component_axes[:, :n_components]  # features x components
    projected_features = {
        frame_name: component_axes.T @ (np.asarray(features, dtype=float) - pooled_mean)
        for frame_name, features in frame_features.items()
    }
    variance_kept = float(
        (singular_values[:n_components] ** 2).sum() / (singular_values ** 2).sum()
    )
    return projected_features, variance_kept
# EOF


"""
regress_out_features_pca
Remove the dominant feature-space directions of the earlier frames from the
final-frame activations. The earlier frames are pooled as samples (stimuli x
frames), a PCA in the model feature space keeps their top components, and
every final-frame stimulus pattern is regressed on those component axes. As
the axes are orthonormal, the residual is the pattern projected onto their
orthogonal complement.

INPUT:
    - target_features: np.ndarray -> features x stimuli, final frame.
    - predictor_features: list[np.ndarray] -> one features x stimuli matrix
        per earlier frame, pooled along stimuli as PCA samples.
    - n_components: int -> number of earlier-frame PCs removed.
    - fit_intercept: bool -> center each final-frame unit across stimuli.

OUTPUT:
    - residual_features: np.ndarray -> features x stimuli residual.
    - regression_summary: dict -> PCA size, predictor variance kept, and
        final-frame variance removed.
"""
def regress_out_features_pca(
        target_features, predictor_features, n_components=50, fit_intercept=True,
        ):
    target_features = np.asarray(target_features, dtype=float)
    # features x (frames * stimuli): each earlier-frame stimulus is one sample.
    pooled_predictor = np.concatenate(
        [np.asarray(features, dtype=float) for features in predictor_features],
        axis=1,
    )
    if n_components >= pooled_predictor.shape[1]:
        raise ValueError(
            f"n_components={n_components} needs more than "
            f"{pooled_predictor.shape[1]} earlier-frame samples."
        )
    # end if the PCA would span every centred sample
    # PCA through the SVD of the centred features x samples matrix; the left
    # singular vectors are the component axes in feature space.
    pooled_centered = pooled_predictor - pooled_predictor.mean(axis=1, keepdims=True)
    component_axes, singular_values, _ = np.linalg.svd(
        pooled_centered, full_matrices=False,
    )
    component_axes = component_axes[:, :n_components]  # features x components

    if fit_intercept:
        target_centered = target_features - target_features.mean(axis=1, keepdims=True)
    else:
        target_centered = target_features
    # end if fit_intercept
    # Per-stimulus OLS on the orthonormal axes: scores = axes^T x, fit = axes @ scores.
    component_scores = component_axes.T @ target_centered  # components x stimuli
    residual_features = target_centered - component_axes @ component_scores

    target_power = (target_centered ** 2).sum()
    regression_summary = {
        "n_components": float(n_components),
        "predictor_variance_kept": float(
            (singular_values[:n_components] ** 2).sum() / (singular_values ** 2).sum()
        ),
        "variance_removed": float(
            1 - (residual_features ** 2).sum() / target_power
            if target_power > 0 else np.nan
        ),
    }
    return residual_features, regression_summary
# EOF


"""
summarize_layer_timing
Compute smoothed curves plus centroid and peak latency for every model layer.

Curves are smoothed along neural time, then values at or below the cutoff
(including all negative values) receive zero weight inside each window.
Centroids and peaks can use different smoothing; a sigma of 0 leaves the
curves unsmoothed.

INPUT:
    - similarity: np.ndarray -> layers x dynamic-neural-time RSA values.
    - time_ms: np.ndarray -> dynamic neural time coordinate.
    - frame_onset_ms: float -> latencies are expressed relative to this time.
    - centroid_window_ms: tuple[float, float] -> end-exclusive centroid window.
    - peak_window_ms: tuple[float, float] -> end-exclusive peak window.
    - smoothing_sigma: float -> Gaussian sigma in samples for the returned
        curves and the centroid latency (0 = no smoothing).
    - centroid_cutoff: float -> minimum RSA carrying centroid weight.
    - peak_cutoff: float -> minimum RSA eligible as a peak.
    - peak_smoothing_sigma: float | None -> Gaussian sigma in samples for the
        peak latency (0 = no smoothing); None reuses smoothing_sigma.
    - relative_cutoffs: bool -> if True, centroid_cutoff and peak_cutoff are
        fractions of the maximum smoothed RSA over all layers inside the
        centroid and peak window; a layer never exceeding the cutoff gets NaN.

OUTPUT:
    - smoothed_similarity: np.ndarray -> curves smoothed with smoothing_sigma.
    - centroid_latency_ms: np.ndarray -> positive-mass centroid minus onset.
    - peak_latency_ms: np.ndarray -> positive peak time minus onset.
    - peak_similarity: np.ndarray -> peak RSA inside the peak window, smoothed
        with peak_smoothing_sigma.
"""
def summarize_layer_timing(
        similarity, time_ms, frame_onset_ms, centroid_window_ms,
        peak_window_ms, smoothing_sigma, centroid_cutoff, peak_cutoff,
        peak_smoothing_sigma=None, relative_cutoffs=False,
        ):
    if peak_smoothing_sigma is None:
        peak_smoothing_sigma = smoothing_sigma
    # end if peak smoothing not given
    if smoothing_sigma < 0 or peak_smoothing_sigma < 0:
        raise ValueError("Smoothing sigmas must be non-negative.")
    # end if negative sigma
    # layers x time curves for the centroid (and plotting) and for the peak.
    smoothed_similarity = np.asarray(similarity, dtype=float)
    if smoothing_sigma > 0:
        smoothed_similarity = gaussian_filter1d(similarity, sigma=smoothing_sigma, axis=1)
    # end if centroid smoothing
    peak_smoothed_similarity = np.asarray(similarity, dtype=float)
    if peak_smoothing_sigma > 0:
        peak_smoothed_similarity = gaussian_filter1d(
            similarity, sigma=peak_smoothing_sigma, axis=1,
        )
    # end if peak smoothing
    centroid_mask = (
        (time_ms >= centroid_window_ms[0]) & (time_ms < centroid_window_ms[1])
    )
    peak_mask = (time_ms >= peak_window_ms[0]) & (time_ms < peak_window_ms[1])
    if not centroid_mask.any() or not peak_mask.any():
        raise ValueError("Timing windows contain no neural samples.")
    # end if a timing window is empty

    if relative_cutoffs:
        # Cutoffs become fractions of the largest value over all layers inside
        # each window, so layers are kept relative to the best-fitting layer.
        # Clamped at 0 so negative RSA never carries weight when no layer fits.
        centroid_cutoff = max(
            0.0, centroid_cutoff * np.nanmax(smoothed_similarity[:, centroid_mask]),
        )
        peak_cutoff = max(
            0.0, peak_cutoff * np.nanmax(peak_smoothed_similarity[:, peak_mask]),
        )
    # end if relative_cutoffs

    n_layers = smoothed_similarity.shape[0]
    centroid_latency_ms = np.full(n_layers, np.nan)
    peak_latency_ms = np.full(n_layers, np.nan)
    for layer_index in range(n_layers):
        centroid_curve = smoothed_similarity[layer_index, centroid_mask]
        centroid_weights = np.where(
            np.isfinite(centroid_curve) & (centroid_curve > centroid_cutoff),
            centroid_curve, 0,
        )
        if centroid_weights.sum() > 0:
            centroid_latency_ms[layer_index] = np.average(
                time_ms[centroid_mask], weights=centroid_weights,
            ) - frame_onset_ms
        # end if centroid mass exists

        peak_curve = peak_smoothed_similarity[layer_index, peak_mask]
        peak_values = np.where(
            np.isfinite(peak_curve) & (peak_curve > peak_cutoff),
            peak_curve, 0,
        )
        if peak_values.sum() > 0:
            peak_latency_ms[layer_index] = (
                time_ms[peak_mask][np.argmax(peak_values)] - frame_onset_ms
            )
        # end if positive peak exists
    # end for layer_index
    peak_similarity = np.nanmax(peak_smoothed_similarity[:, peak_mask], axis=1)
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
    - regression_level: str -> "rdm" regresses the earlier-frame RDM out of
        the final-frame RDM; "features" regresses each earlier-frame unit out
        of the same final-frame unit before building the residual RDM.

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
        regression_level="rdm",
        ):
    if regression_level not in ("rdm", "features"):
        raise ValueError("regression_level must be 'rdm' or 'features'.")
    # end if unknown regression level
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

        # features x stimuli snapshot at each of the two frame times.
        frame_features = []
        for frame_ms in (previous_frame_ms, last_frame_ms):
            frame_index = int(np.argmin(np.abs(model_time_ms - frame_ms)))
            if not np.isclose(model_time_ms[frame_index], frame_ms):
                raise ValueError(
                    f"{layer_name}: resampled model grid misses {frame_ms} ms."
                )
            # end if frame does not lie on the model grid
            frame_features.append(
                np.ascontiguousarray(model_ts.get_array()[:, frame_index, :])
            )
        # end for frame_ms
        previous_features, last_features = frame_features
        last_frame_rdm = create_RDM(last_features, metric=model_rdm_metric)
        if regression_level == "rdm":
            residual_rdm, regression_summary = regress_out_rdm(
                last_frame_rdm,
                create_RDM(previous_features, metric=model_rdm_metric),
                fit_intercept=fit_intercept,
            )
        else:
            residual_features, regression_summary = regress_out_features(
                last_features, previous_features, fit_intercept=fit_intercept,
            )
            residual_rdm = create_RDM(
                np.ascontiguousarray(residual_features), metric=model_rdm_metric,
            )
        # end if regression_level

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
    - previous_frame_ms: float | tuple[float, ...] | None -> model time(s) of
        the regressed-out frame(s); "features" takes a single frame, "rdm"
        regresses all earlier RDMs out together with one OLS fit. Ignored
        (may be None) when previous_frame_range_ms is set.
    - last_frame_ms: float -> model time of the last-frame (and static) RDM.
    - model_rdm_metric: str -> distance used for model RDMs.
    - rsa_metric: str -> model-neural RDM comparison metric.
    - fit_intercept: bool -> include a constant in the regression.
    - regression_level: str -> "rdm" regresses the previous-frame RDM out of
        the last-frame RDM; "features" regresses each previous-frame unit out
        of the same last-frame unit before building the residual RDM;
        "features_ridge" regresses all previous frames out together with an
        in-fold PCA + ridge model (regress_out_features_ridge);
        "features_pca" removes the top PCs of the pooled previous-frame
        activations from the last frame (regress_out_features_pca);
        "rdm_rank" is "rdm" fitted on rank-transformed RDMs; "features_ols" is
        "features_ridge" with plain OLS (out-of-fold with ridge_cv_type="kf").
    - ridge_pca_variance: float | None -> predictor variance kept by the PCA.
    - ridge_alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - ridge_cv_type: str -> "kf" out-of-fold residuals, "same" in-sample.
    - ridge_n_splits: int -> stimulus folds when ridge_cv_type is "kf".
    - pca_n_components: int -> previous-frame PCs removed by "features_pca".
    - feature_pca_components: int | None -> for the feature-level regressions,
        first project the last and previous frames onto this many shared PCs
        of their pooled activations (project_frames_on_pcs); None keeps the
        original features.
    - previous_frame_range_ms: tuple[float, float] | None -> (start, end) model
        time (inclusive); every native model frame in it is regressed out
        together. Native frames are used (not the analysis_fs grid) because
        upsampling repeats frames, which would add duplicate predictors.
    - similarity_target: str -> "neural" compares the model RDMs with the
        neural RDMs; "model" compares them with the same layer's own model RDM
        timecourse on the analysis_fs grid (autocorrelation control: the neural
        RDMs are ignored and "static" is left out, the model has no static
        response).
    - similarity_method: str -> "rsa" correlates model and neural RDMs;
        "encoding" predicts the neural channel pattern from the model features
        with cross-validated ridge at every neural timepoint
        (encoding_timecourse). With "encoding", static_neural_rdms and
        dynamic_neural_rdms are channels x time x stimuli rasters at
        analysis_fs, the residual must be built in feature space (no "rdm" /
        "rdm_rank"), and only the neural target is supported.
    - encoding_alphas: tuple[float, ...] -> ridge penalties of the encoding.
    - encoding_n_splits: int -> stimulus folds of the encoding.
    - encoding_score_type: str -> "r2" (channel-averaged), "r2_pooled",
        "corr_pooled" or "corr" (see encoding_timecourse).
    - encoding_alpha_per_target: bool -> one ridge penalty per channel.

OUTPUT:
    - layer_names: list[str] -> layers in forward-pass order.
    - similarity: dict[str, np.ndarray] -> layers x neural (or model) time RSA
        for "static" (neural target only), "first_frame", "last_frame" and
        "last_frame_residual".
    - variance_removed: np.ndarray -> per-layer fraction of last-frame RDM
        (or feature) variance explained by the previous frame(s).
"""
def model_frame_similarity_by_layer(
        static_neural_rdms, dynamic_neural_rdms, feature_names,
        model_features_dir, model_name, model_dataset_name, model_pooling,
        analysis_fs, first_frame_ms, previous_frame_ms, last_frame_ms,
        model_rdm_metric, rsa_metric, fit_intercept=True, regression_level="rdm",
        ridge_pca_variance=0.95,
        ridge_alphas=(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5),
        ridge_cv_type="kf", ridge_n_splits=5, pca_n_components=50,
        feature_pca_components=None, previous_frame_range_ms=None,
        similarity_target="neural", similarity_method="rsa",
        encoding_alphas=tuple(10.0 ** np.arange(-2, 11)), encoding_n_splits=5,
        encoding_score_type="r2", encoding_alpha_per_target=False,
        ):
    if similarity_method not in ("rsa", "encoding"):
        raise ValueError("similarity_method must be 'rsa' or 'encoding'.")
    # end if unknown similarity method
    if similarity_method == "encoding" and (
            regression_level in ("rdm", "rdm_rank") or similarity_target != "neural"):
        raise ValueError(
            "Encoding needs a feature-level regression_level and the neural target."
        )
    # end if encoding with an RDM-level residual or the model target
    if similarity_target not in ("neural", "model"):
        raise ValueError("similarity_target must be 'neural' or 'model'.")
    # end if unknown similarity target
    if regression_level not in (
            "rdm", "rdm_rank", "features", "features_ridge", "features_ols",
            "features_pca"):
        raise ValueError(
            "regression_level must be 'rdm', 'rdm_rank', 'features', "
            "'features_ridge', 'features_ols' or 'features_pca'."
        )
    # end if unknown regression level
    if previous_frame_range_ms is None:
        previous_frames_ms = tuple(np.atleast_1d(previous_frame_ms).tolist())
        if regression_level == "features" and len(previous_frames_ms) != 1:
            raise ValueError("regression_level='features' takes one previous frame.")
        # end if several frames with the unit-wise regression
    elif regression_level == "features":
        raise ValueError("regression_level='features' takes one previous frame, not a range.")
    # end if fixed previous frames
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
        if previous_frame_range_ms is not None:
            # Every native frame whose time lies in the range (inclusive).
            native_time_ms = np.arange(model_features.shape[1]) * 1000 / source_fs
            range_start_ms, range_end_ms = previous_frame_range_ms
            range_indices = np.flatnonzero(
                (native_time_ms >= range_start_ms - 1e-6)
                & (native_time_ms <= range_end_ms + 1e-6)
            )
            if len(range_indices) == 0:
                raise ValueError(f"{layer_name}: no model frame in {previous_frame_range_ms} ms.")
            # end if empty range
            previous_frames_ms = tuple(native_time_ms[range_indices].tolist())
            # features x stimuli snapshot of every native frame in the range.
            native_previous_features = {
                f"previous_{native_time_ms[index]:g}": np.ascontiguousarray(
                    model_features[:, index, :]
                )
                for index in range_indices
            }
        # end if previous frame range
        model_ts = TimeSeries(model_features, fs=source_fs)
        model_ts.resample(analysis_fs)
        model_time_ms = np.arange(len(model_ts)) * 1000 / model_ts.get_fs()

        frame_rdms, frame_features = {}, {}
        # Previous frames are keyed by their time, e.g. "previous_2250".
        frame_times = [("first", first_frame_ms), ("last", last_frame_ms)]
        if previous_frame_range_ms is None:
            frame_times += [
                (f"previous_{frame_ms:g}", frame_ms) for frame_ms in previous_frames_ms
            ]
        else:
            # Native range frames do not need the analysis_fs grid.
            for frame_name, features in native_previous_features.items():
                frame_features[frame_name] = features
                frame_rdms[frame_name] = create_RDM(features, metric=model_rdm_metric)
            # end for frame_name, features
        # end if fixed previous frames
        for frame_name, frame_ms in frame_times:
            frame_index = int(np.argmin(np.abs(model_time_ms - frame_ms)))
            if not np.isclose(model_time_ms[frame_index], frame_ms):
                raise ValueError(
                    f"{layer_name}: resampled model grid misses {frame_ms} ms."
                )
            # end if frame does not lie on the model grid
            # features x stimuli snapshot at this frame time.
            frame_features[frame_name] = np.ascontiguousarray(
                model_ts.get_array()[:, frame_index, :]
            )
            frame_rdms[frame_name] = create_RDM(
                frame_features[frame_name], metric=model_rdm_metric,
            )
        # end for frame_name, frame_ms
        previous_names = [f"previous_{frame_ms:g}" for frame_ms in previous_frames_ms]
        if regression_level in ("rdm", "rdm_rank"):
            # previous frames x pairs: all earlier RDMs enter one OLS fit.
            residual_rdm, regression_summary = regress_out_rdm(
                frame_rdms["last"],
                np.stack([frame_rdms[name] for name in previous_names]),
                fit_intercept=fit_intercept,
                rank_space=regression_level == "rdm_rank",
            )
        else:
            # Frames entering the feature-level regression: last + previous.
            regression_features = {
                name: frame_features[name] for name in ["last"] + previous_names
            }
            if feature_pca_components is not None:
                regression_features, _ = project_frames_on_pcs(
                    regression_features, feature_pca_components,
                )
            # end if shared PCA projection
            if regression_level == "features":
                residual_features, regression_summary = regress_out_features(
                    regression_features["last"], regression_features[previous_names[0]],
                    fit_intercept=fit_intercept,
                )
            elif regression_level == "features_pca":
                residual_features, regression_summary = regress_out_features_pca(
                    regression_features["last"],
                    [regression_features[name] for name in previous_names],
                    n_components=pca_n_components, fit_intercept=fit_intercept,
                )
            else:
                residual_features, regression_summary = regress_out_features_ridge(
                    regression_features["last"],
                    [regression_features[name] for name in previous_names],
                    pca_variance=ridge_pca_variance, alphas=ridge_alphas,
                    cv_type=ridge_cv_type, n_splits=ridge_n_splits,
                    fit_intercept=fit_intercept,
                    regression_type="lr" if regression_level == "features_ols" else "ridge",
                )
            # end if regression_level == "features" / "features_pca" / ridge / ols
            residual_rdm = create_RDM(
                np.ascontiguousarray(residual_features), metric=model_rdm_metric,
            )
        # end if regression_level

        if similarity_target == "neural":
            target_rdms = dynamic_neural_rdms
        else:
            # time x pairs RDMs of this layer over the whole movie, on the
            # same analysis_fs grid as the neural RDMs.
            target_rdms = compute_rdm_timeseries(model_ts.get_array(), model_rdm_metric)
        # end if similarity_target
        if similarity_method == "encoding":
            # Model features x stimuli predicting every neural timepoint; the
            # "RDM" inputs are channels x time x stimuli rasters here.
            condition_inputs = [
                ("static", static_neural_rdms, frame_features["last"]),
                ("first_frame", dynamic_neural_rdms, frame_features["first"]),
                ("last_frame", dynamic_neural_rdms, frame_features["last"]),
                ("last_frame_residual", dynamic_neural_rdms, residual_features),
            ]
            for condition_name, neural_rasters, features in condition_inputs:
                similarity[condition_name].append(encoding_timecourse(
                    features, neural_rasters, analysis_fs,
                    alphas=encoding_alphas, n_splits=encoding_n_splits,
                    score_type=encoding_score_type,
                    alpha_per_target=encoding_alpha_per_target,
                ))
            # end for condition_name
            layer_names.append(layer_name)
            variance_removed.append(regression_summary["variance_removed"])
            print(f"    encoded {layer_name}", flush=True)
            continue
        # end if encoding
        # One model RDM against every target time point -> (time,) vector.
        condition_inputs = [
            ("first_frame", target_rdms, frame_rdms["first"]),
            ("last_frame", target_rdms, frame_rdms["last"]),
            ("last_frame_residual", target_rdms, residual_rdm),
        ]
        if similarity_target == "neural":
            condition_inputs.insert(0, ("static", static_neural_rdms, frame_rdms["last"]))
        # end if neural target
        for condition_name, target_condition_rdms, model_rdm in condition_inputs:
            similarity[condition_name].append(cross_temporal_similarity(
                target_condition_rdms, model_rdm[np.newaxis, :], metric=rsa_metric,
            )[:, 0])
        # end for condition_name
        layer_names.append(layer_name)
        variance_removed.append(regression_summary["variance_removed"])
    # end for feature_path
    # Conditions without curves ("static" for the model target) are dropped.
    similarity = {
        condition_name: np.stack(curves)
        for condition_name, curves in similarity.items() if curves
    }
    return layer_names, similarity, np.asarray(variance_removed)
# EOF


"""
static_bin_similarity_by_window
Neural counterpart of model_frame_similarity_by_layer: every time bin of the
static last-frame response plays the role of a model layer. Each bin RDM is
compared with every movie RDM time point, before and after regressing out the
RDMs of the static responses to the earlier frames (e.g. the 2000 ms and
2250 ms frame images) in the same bin, and optionally in its neighbouring
bins, with one OLS over stimulus pairs per bin (regress_out_rdm). Only the static side is regressed; the movie RDMs are
never residualized.

INPUT:
    - static_rasters: np.ndarray -> channels x static time x stimuli response
        to the last-frame image.
    - previous_rasters: list[np.ndarray] -> one channels x static time x
        stimuli response per earlier-frame image, same channels and stimuli.
    - dynamic_neural_rdms: np.ndarray -> movie time x stimulus pairs RDMs.
    - fs: float -> sampling rate of the static rasters (Hz).
    - bins_ms: list[tuple[float, float]] -> end-exclusive static bins (ms
        from image onset); their order is the "depth" of the hierarchy.
    - rdm_metric: str -> distance used for the bin RDMs.
    - rsa_metric: str -> bin-movie RDM comparison metric.
    - fit_intercept: bool -> include a constant in the RDM regression.
    - rank_space: bool -> fit the regression on rank-transformed RDMs.
    - delay_embedding_lags: tuple[int, int] | None -> inclusive bin lags, e.g.
        (-5, 5) regresses each bin on the earlier-frame RDMs of the bins
        shifted by -5 ... +5 bin widths (all frames and lags in one OLS);
        the shifted bins must lie inside the static response. None uses
        only the same bin.

OUTPUT:
    - similarity: dict[str, np.ndarray] -> bins x movie time RSA for
        "last_frame" (raw bin RDMs) and "last_frame_residual".
    - variance_removed: np.ndarray -> per-bin fraction of last-frame RDM
        variance explained by the earlier-frame RDMs.
"""
def static_bin_similarity_by_window(
        static_rasters, previous_rasters, dynamic_neural_rdms, fs, bins_ms,
        rdm_metric, rsa_metric, fit_intercept=True, rank_space=False,
        delay_embedding_lags=None,
        ):
    lags = (
        [0] if delay_embedding_lags is None
        else list(range(delay_embedding_lags[0], delay_embedding_lags[1] + 1))
    )
    # channels x bins x stimuli: the response averaged inside every bin.
    static_patterns = window_patterns(TimeSeries(static_rasters, fs=fs), bins_ms)
    # bins x stimulus pairs, one RDM per bin.
    static_rdms = compute_rdm_timeseries(static_patterns, rdm_metric)

    response_ms = static_rasters.shape[1] * 1000 / fs
    residual_rdms, variance_removed = [], []
    for bin_index, (start_ms, end_ms) in enumerate(bins_ms):
        # The same bin shifted by every lag (in bin widths), e.g. 10 ms bins
        # with lags -5..5 span 50 ms before to 50 ms after the target bin.
        width_ms = end_ms - start_ms
        lagged_bins_ms = [
            (start_ms + lag * width_ms, end_ms + lag * width_ms) for lag in lags
        ]
        if lagged_bins_ms[0][0] < 0 or lagged_bins_ms[-1][1] > response_ms:
            raise ValueError(
                f"Lagged bins {lagged_bins_ms[0]}-{lagged_bins_ms[-1]} ms exceed "
                f"the {response_ms:g} ms static response."
            )
        # end if a lagged bin falls outside the response
        # (frames * lags) x pairs: every earlier frame at every lagged bin.
        predictor_rdms = np.concatenate([
            compute_rdm_timeseries(
                window_patterns(TimeSeries(rasters, fs=fs), lagged_bins_ms), rdm_metric,
            )
            for rasters in previous_rasters
        ])
        residual_rdm, regression_summary = regress_out_rdm(
            static_rdms[bin_index], predictor_rdms,
            fit_intercept=fit_intercept, rank_space=rank_space,
        )
        residual_rdms.append(residual_rdm)
        variance_removed.append(regression_summary["variance_removed"])
    # end for bin_index, (start_ms, end_ms)

    # movie time x bins -> transposed to bins x movie time, as for layers.
    similarity = {
        condition_name: cross_temporal_similarity(
            dynamic_neural_rdms, bin_rdms, metric=rsa_metric,
        ).T
        for condition_name, bin_rdms in (
            ("last_frame", static_rdms),
            ("last_frame_residual", np.stack(residual_rdms)),
        )
    }
    return similarity, np.asarray(variance_removed)
# EOF
