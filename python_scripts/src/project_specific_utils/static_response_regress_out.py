"""
Alternative ways of regressing an earlier static response out of the
last-frame static response, plus static response windows used as "layers".

The earlier static response (e.g. the 2000 ms or 2250 ms movie frame shown as
an image) is the predictor, and the last-frame static response is the target.
Both are channels x time x stimuli and share the same stimulus order.
"""

import numpy as np
from sklearn.decomposition import PCA

from useful_stuff.general_utils import TimeSeries, create_RDM, dyn_linear_encoding

from .channelwise_correlation import channelwise_regress_out
from .split_half_rsa import (
    compute_rdm_timeseries, cross_temporal_similarity, whiten_rasters,
)


"""
_embedded_predictor_array
Delay-embed the predictor response so each timepoint also sees its neighbours.

INPUT:
    - predictor_ts: TimeSeries -> channels x time x stimuli predictor response.
    - delay_embedding_lags: tuple[int, int] | None -> inclusive sample lags;
        None keeps the raw channels.

OUTPUT:
    - predictor_array: np.ndarray -> (channels * lags) x time x stimuli.
"""
def _embedded_predictor_array(predictor_ts, delay_embedding_lags):
    if delay_embedding_lags is None:
        return np.asarray(predictor_ts.get_array(), dtype=np.float64)
    # end if delay_embedding_lags is None
    # Edge padding keeps the time axis length unchanged and stimuli independent.
    return np.asarray(predictor_ts.delay_embeddings(
        lags=delay_embedding_lags, pad_mode='edge',
    ).get_array(), dtype=np.float64)
# EOF


"""
_check_static_pair
Validate that predictor and target responses can be regressed on each other.

INPUT:
    - predictor_ts: TimeSeries -> channels x time x stimuli predictor response.
    - target_ts: TimeSeries -> channels x time x stimuli target response.

OUTPUT:
    - None
"""
def _check_static_pair(predictor_ts, target_ts):
    predictor_shape = predictor_ts.get_array().shape
    target_shape = target_ts.get_array().shape
    if len(predictor_shape) != 3 or len(target_shape) != 3:
        raise ValueError('Responses must have shape channels x time x stimuli.')
    # end if input dimensions
    if predictor_shape[1:] != target_shape[1:]:
        raise ValueError('Predictor and target must share time and stimuli.')
    # end if time or stimuli differ
    if predictor_ts.get_fs() != target_ts.get_fs():
        raise ValueError('Static responses have different sampling rates.')
    # end if sampling rates differ
# EOF


"""
_variance_explained
Fraction of across-stimulus variance (summed over channels) removed per time.

INPUT:
    - target_array: np.ndarray -> channels x time x stimuli original response.
    - residual_array: np.ndarray -> channels x time x stimuli residual.

OUTPUT:
    - variance_explained: np.ndarray -> one value per timepoint.
"""
def _variance_explained(target_array, residual_array):
    target_variance = np.var(target_array, axis=2).sum(axis=0)
    residual_variance = np.var(residual_array, axis=2).sum(axis=0)
    variance_explained = np.full(target_variance.shape, np.nan)
    np.divide(
        target_variance - residual_variance, target_variance,
        out=variance_explained, where=target_variance > 0,
    )
    return variance_explained
# EOF


"""
_slicewise_regress_out
Fit one independent regression per slice (timepoint or window) with stimuli as
samples, and return the prediction for every stimulus. With cv_type="kf" each
stimulus is predicted by a model that never saw it (out-of-fold); with
cv_type="same" the model is fitted and predicted in-sample.

INPUT:
    - predictor_array: np.ndarray -> features x slices x stimuli predictor.
    - target_array: np.ndarray -> channels x slices x stimuli target.
    - regression_type: str -> useful_stuff regression type ("ridge" = RidgeCV).
    - alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - cv_type: str -> "kf" for out-of-fold predictions, "same" for in-sample.
    - n_splits: int -> number of stimulus folds when cv_type is "kf".
    - shuffle: bool -> whether the stimulus folds are shuffled.
    - fit_intercept: bool -> whether every slice model has an intercept.
    - pca_variance: float | None -> if given, the predictor features of every
        slice are projected onto the principal components that keep this
        fraction of their variance before the regression. The PCA is fitted on
        the training stimuli of each fold only.

OUTPUT:
    - predicted_array: np.ndarray -> channels x slices x stimuli prediction.
    - chosen_alphas: np.ndarray -> ridge penalty of the last fit per slice.
    - n_components: np.ndarray -> PCA components kept in the last fold per
        slice (NaN without PCA).
"""
def _slicewise_regress_out(
        predictor_array, target_array, regression_type, alphas, cv_type,
        n_splits, shuffle, fit_intercept, pca_variance=None,
        ):
    if cv_type not in ('kf', 'same'):
        raise ValueError("cv_type must be 'kf' or 'same'.")
    # end if cv_type
    n_slices = target_array.shape[1]
    predicted_array = np.zeros_like(target_array)
    chosen_alphas = np.full(n_slices, np.nan)
    n_components = np.full(n_slices, np.nan)
    for slice_index in range(n_slices):
        # features x stimuli within this single slice.
        predictor_s = predictor_array[:, slice_index, :]
        target_s = target_array[:, slice_index, :]
        # A fresh model per slice, so RidgeCV picks its own penalty.
        regression_model = dyn_linear_encoding(
            regression_type=regression_type, cv_type=cv_type, max_lag=0,
            alphas=np.asarray(alphas), n_splits=n_splits, shuffle=shuffle,
            fit_intercept=fit_intercept,
        )
        # The splitter yields (all, all) for "same" and stimulus folds for "kf".
        stimulus_indices = np.arange(target_s.shape[1])
        for train_idx, test_idx in regression_model.get_cv_obj().split(
                stimulus_indices):
            train_predictor = predictor_s[:, train_idx]
            test_predictor = predictor_s[:, test_idx]
            if pca_variance is not None:
                # sklearn wants stimuli x features; the model wants features x stimuli.
                pca = PCA(n_components=pca_variance, svd_solver='full')
                train_predictor = pca.fit_transform(train_predictor.T).T
                test_predictor = pca.transform(test_predictor.T).T
                n_components[slice_index] = pca.n_components_
            # end if PCA requested
            regression_model.fit(train_predictor, target_s[:, train_idx])
            predicted_array[:, slice_index, test_idx] = regression_model.predict(
                test_predictor,
            )
        # end for train_idx, test_idx
        if hasattr(regression_model.get_regression_obj(), 'alpha_'):
            chosen_alphas[slice_index] = regression_model.get_regression_obj().alpha_
        # end if the regression object tunes a penalty
    # end for slice_index
    return predicted_array, chosen_alphas, n_components
# EOF


"""
pooled_static_regress_out
Reference approach: one linear map shared by every timepoint and stimulus.
Every (time, stimulus) pair is one regression sample, fitted and removed
in-sample, exactly as in the summary_figs_report notebook.

INPUT:
    - predictor_ts: TimeSeries -> channels x time x stimuli earlier response.
    - target_ts: TimeSeries -> channels x time x stimuli last-frame response.
    - delay_embedding_lags: tuple[int, int] | None -> predictor sample lags.
    - regression_type: str -> useful_stuff regression type ("lr" is OLS).

OUTPUT:
    - residual_ts: TimeSeries -> channels x time x stimuli residual response.
    - variance_explained: np.ndarray -> fraction of variance removed per time.
"""
def pooled_static_regress_out(
        predictor_ts, target_ts, delay_embedding_lags=(-1, 1),
        regression_type='lr',
        ):
    _check_static_pair(predictor_ts, target_ts)
    predictor_array = _embedded_predictor_array(
        predictor_ts, delay_embedding_lags,
    )
    target_array = np.asarray(target_ts.get_array(), dtype=np.float64)

    # Flatten time and stimuli into one sample axis: features x (time * stimuli).
    predictor_flat_ts = TimeSeries(
        predictor_array.reshape(predictor_array.shape[0], -1),
        fs=target_ts.get_fs(),
    )
    target_flat_ts = TimeSeries(
        target_array.reshape(target_array.shape[0], -1), fs=target_ts.get_fs(),
    )
    regression_model = dyn_linear_encoding(
        regression_type=regression_type, cv_type='same', max_lag=0,
    )
    residual_flat_ts = regression_model.pointwise_regress_out(
        predictor_flat_ts, target_flat_ts,
    )
    residual_array = residual_flat_ts.get_array().reshape(target_array.shape)

    residual_ts = TimeSeries(residual_array, fs=target_ts.get_fs())
    return residual_ts, _variance_explained(target_array, residual_array)
# EOF


"""
timepoint_static_regress_out
Fit one independent ridge model per static timepoint. At time t the stimuli
are the samples, the (delay-embedded) earlier response at t is the predictor,
and the last-frame response at t is the target. No weights or ridge
penalties are shared across timepoints.

With cv_type="kf" the residual of each stimulus comes from a model that never
saw that stimulus (out-of-fold prediction), so the per-timepoint fit cannot
remove variance just by overfitting the ~stimuli-sized sample. With
cv_type="same" the model is fitted and removed in-sample like the reference.

INPUT:
    - predictor_ts: TimeSeries -> channels x time x stimuli earlier response.
    - target_ts: TimeSeries -> channels x time x stimuli last-frame response.
    - delay_embedding_lags: tuple[int, int] | None -> predictor sample lags.
    - regression_type: str -> useful_stuff regression type ("ridge" = RidgeCV).
    - alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - cv_type: str -> "kf" for out-of-fold residuals, "same" for in-sample.
    - n_splits: int -> number of stimulus folds when cv_type is "kf".
    - shuffle: bool -> whether the stimulus folds are shuffled.
    - fit_intercept: bool -> whether every timepoint model has an intercept.
    - pca_variance: float | None -> fraction of predictor variance kept by an
        in-fold PCA before every timepoint regression; None disables the PCA.

OUTPUT:
    - residual_ts: TimeSeries -> channels x time x stimuli residual response.
    - variance_explained: np.ndarray -> fraction of variance removed per time.
    - chosen_alphas: np.ndarray -> ridge penalty of the last fit per time.
"""
def timepoint_static_regress_out(
        predictor_ts, target_ts, delay_embedding_lags=(-1, 1),
        regression_type='ridge',
        alphas=(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5),
        cv_type='kf', n_splits=5, shuffle=False, fit_intercept=True,
        pca_variance=None,
        ):
    _check_static_pair(predictor_ts, target_ts)
    predictor_array = _embedded_predictor_array(
        predictor_ts, delay_embedding_lags,
    )
    target_array = np.asarray(target_ts.get_array(), dtype=np.float64)
    # Every static timepoint is one slice with its own model.
    predicted_array, chosen_alphas, _ = _slicewise_regress_out(
        predictor_array, target_array, regression_type, alphas, cv_type,
        n_splits, shuffle, fit_intercept, pca_variance=pca_variance,
    )

    residual_array = target_array - predicted_array
    residual_ts = TimeSeries(residual_array, fs=target_ts.get_fs())
    return (
        residual_ts, _variance_explained(target_array, residual_array),
        chosen_alphas,
    )
# EOF


"""
centered_norms_and_directions
Separate every population pattern into its distance from the across-stimulus
centroid and its unit-length direction from that centroid.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli response patterns.
    - epsilon: float -> smallest norm accepted as a non-zero pattern.

OUTPUT:
    - norms: np.ndarray -> time x stimuli positive pattern norms.
    - directions: np.ndarray -> channels x time x stimuli unit directions.
"""
def centered_norms_and_directions(rasters, epsilon=1e-12):
    rasters = np.asarray(rasters, dtype=np.float64)
    if rasters.ndim != 3:
        raise ValueError("rasters must have shape channels x time x stimuli.")
    # end if raster dimensions

    # The centroid is one population pattern per timepoint, averaged over stimuli.
    centered = rasters - rasters.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms <= epsilon):
        raise ValueError(
            "At least one centered stimulus pattern has zero or near-zero norm."
        )
    # end if a direction is undefined
    directions = centered / norms[np.newaxis, :, :]
    return norms, directions
# EOF


"""
radial_direction_static_regress_out
Regress earlier-frame response strength and pattern direction out of a target
static response as two separate models. Strength is represented by log norm,
while pattern direction is represented by a centred unit population vector.

The direction residuals are centred over stimuli and normalized again before
they are returned. Euclidean distances between the returned vectors therefore
measure residual direction without reintroducing residual-vector magnitude.
Strength residuals remain separate because there is no unique scale on which
to combine scalar radial residuals with directional residual vectors.

INPUT:
    - predictor_rasters: np.ndarray -> frames x channels x time x stimuli, or
        channels x time x stimuli for one earlier frame.
    - target_rasters: np.ndarray -> channels x time x stimuli target response.
    - regression_type: str -> useful_stuff regression type ("ridge" or "lr").
    - alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - cv_type: str -> "kf" for out-of-fold predictions or "same" in-sample.
    - n_splits: int -> number of stimulus folds when cv_type is "kf".
    - shuffle: bool -> whether stimulus folds are shuffled.
    - fit_intercept: bool -> whether both regressions include an intercept.
    - epsilon: float -> numerical floor for vector norms and logarithms.

OUTPUT:
    - result: dict -> normalized direction residuals, log-norm residuals,
        variance removed by both models, and their selected ridge penalties.
"""
def radial_direction_static_regress_out(
        predictor_rasters, target_rasters, regression_type="ridge",
        alphas=(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5),
        cv_type="kf", n_splits=5, shuffle=False, fit_intercept=True,
        epsilon=1e-12,
        ):
    predictor_rasters = np.asarray(predictor_rasters, dtype=np.float64)
    target_rasters = np.asarray(target_rasters, dtype=np.float64)
    if predictor_rasters.ndim == 3:
        predictor_rasters = predictor_rasters[np.newaxis, :, :, :]
    # end if one predictor frame
    if predictor_rasters.ndim != 4 or target_rasters.ndim != 3:
        raise ValueError(
            "predictor_rasters must be frames x channels x time x stimuli "
            "and target_rasters must be channels x time x stimuli."
        )
    # end if raster dimensions
    if predictor_rasters.shape[1:] != target_rasters.shape:
        raise ValueError(
            "Every predictor frame must match the target channels, time and stimuli."
        )
    # end if predictor and target shapes differ
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    # end if invalid epsilon

    # Decompose the target once and every earlier frame independently.
    target_norms, target_directions = centered_norms_and_directions(
        target_rasters, epsilon,
    )
    predictor_norms = []
    predictor_directions = []
    for frame_rasters in predictor_rasters:
        frame_norms, frame_directions = centered_norms_and_directions(
            frame_rasters, epsilon,
        )
        predictor_norms.append(frame_norms)
        predictor_directions.append(frame_directions)
    # end for frame_rasters
    predictor_norms = np.stack(predictor_norms)
    predictor_directions = np.concatenate(predictor_directions, axis=0)

    # Log norm turns multiplicative population gain into an additive quantity.
    target_log_norms = np.log(np.maximum(target_norms, epsilon))[np.newaxis, :, :]
    predictor_log_norms = np.log(np.maximum(predictor_norms, epsilon))
    predicted_log_norms, norm_alphas, _ = _slicewise_regress_out(
        predictor_log_norms, target_log_norms, regression_type, alphas,
        cv_type, n_splits, shuffle, fit_intercept,
    )
    log_norm_residuals = target_log_norms - predicted_log_norms

    # Regress directions only on directions, so response strength cannot enter
    # the multivariate pattern model through either its inputs or its targets.
    predicted_directions, direction_alphas, _ = _slicewise_regress_out(
        predictor_directions, target_directions, regression_type, alphas,
        cv_type, n_splits, shuffle, fit_intercept,
    )
    direction_residuals = target_directions - predicted_directions

    # Out-of-fold residuals from different fits need not share an exact centroid.
    # Re-centre and normalize so the final Euclidean RDM is direction-only.
    direction_residuals -= direction_residuals.mean(axis=2, keepdims=True)
    residual_norms = np.linalg.norm(direction_residuals, axis=0)
    if np.any(residual_norms <= epsilon):
        raise ValueError(
            "At least one directional regression residual has zero or near-zero norm."
        )
    # end if residual direction is undefined
    normalized_direction_residuals = (
        direction_residuals / residual_norms[np.newaxis, :, :]
    )

    return {
        "direction_residuals": normalized_direction_residuals,
        "log_norm_residuals": log_norm_residuals[0],
        "direction_variance_removed": _variance_explained(
            target_directions, direction_residuals,
        ),
        "log_norm_variance_removed": _variance_explained(
            target_log_norms, log_norm_residuals,
        ),
        "direction_alphas": direction_alphas,
        "log_norm_alphas": norm_alphas,
    }
# EOF


"""
window_patterns
Average the response inside each time window (start inclusive, end
exclusive), turning a static timecourse into one pattern per window.

INPUT:
    - static_ts: TimeSeries -> channels x time x stimuli static response.
    - windows_ms: list[tuple[float, float]] -> windows relative to image onset.

OUTPUT:
    - patterns: np.ndarray -> channels x windows x stimuli window averages.
"""
def window_patterns(static_ts, windows_ms):
    static_array = np.asarray(static_ts.get_array(), dtype=np.float64)
    static_time_ms = np.arange(static_array.shape[1]) * 1000 / static_ts.get_fs()

    patterns = []
    for start_ms, end_ms in windows_ms:
        window_mask = (static_time_ms >= start_ms) & (static_time_ms < end_ms)
        if not window_mask.any():
            raise ValueError(f'Window {start_ms}-{end_ms} ms has no samples.')
        # end if the window is empty
        # channels x stimuli after averaging the window samples.
        patterns.append(static_array[:, window_mask, :].mean(axis=1))
    # end for start_ms, end_ms
    return np.stack(patterns, axis=1)
# EOF


"""
window_static_regress_out
Window-to-window regression: average both responses inside each window, then
fit one independent ridge model per window that predicts the last-frame
window pattern from the earlier response's pattern in the same window.
Stimuli are the samples; no weights are shared across windows.

INPUT:
    - predictor_ts: TimeSeries -> channels x time x stimuli earlier response.
    - target_ts: TimeSeries -> channels x time x stimuli last-frame response.
    - windows_ms: list[tuple[float, float]] -> windows relative to image onset.
    - regression_type: str -> useful_stuff regression type ("ridge" = RidgeCV).
    - alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - cv_type: str -> "kf" for out-of-fold residuals, "same" for in-sample.
    - n_splits: int -> number of stimulus folds when cv_type is "kf".
    - shuffle: bool -> whether the stimulus folds are shuffled.
    - fit_intercept: bool -> whether every window model has an intercept.

OUTPUT:
    - residual_patterns: np.ndarray -> channels x windows x stimuli residuals.
    - variance_explained: np.ndarray -> fraction of variance removed per window.
    - chosen_alphas: np.ndarray -> ridge penalty of the last fit per window.
"""
def window_static_regress_out(
        predictor_ts, target_ts, windows_ms, regression_type='ridge',
        alphas=(1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5),
        cv_type='kf', n_splits=5, shuffle=False, fit_intercept=True,
        ):
    _check_static_pair(predictor_ts, target_ts)
    # channels x windows x stimuli for both responses.
    predictor_patterns = window_patterns(predictor_ts, windows_ms)
    target_patterns = window_patterns(target_ts, windows_ms)
    # Every window is one slice with its own model.
    predicted_patterns, chosen_alphas, _ = _slicewise_regress_out(
        predictor_patterns, target_patterns, regression_type, alphas, cv_type,
        n_splits, shuffle, fit_intercept,
    )
    residual_patterns = target_patterns - predicted_patterns
    return (
        residual_patterns,
        _variance_explained(target_patterns, residual_patterns),
        chosen_alphas,
    )
# EOF


"""
window_model_timecourses
Correlate every window RDM with the dynamic RDM at each movie time, giving
one "layer" timecourse per window, as if every window were a DNN layer.

INPUT:
    - dynamic_rdms: np.ndarray -> movie time x stimulus-pair distances.
    - patterns: np.ndarray -> channels x windows x stimuli window patterns.
    - rdm_metric: str -> distance metric for the window RDMs.
    - rsa_metric: str -> "correlation" or "spearman".

OUTPUT:
    - similarity: np.ndarray -> windows x movie time RDM similarity.
"""
def window_model_timecourses(dynamic_rdms, patterns, rdm_metric, rsa_metric):
    # windows x stimulus pairs, one model RDM per window.
    window_rdms = compute_rdm_timeseries(patterns, rdm_metric)
    # movie time x windows -> transpose to windows x movie time.
    return cross_temporal_similarity(
        dynamic_rdms, window_rdms, metric=rsa_metric,
    ).T
# EOF


"""
previous_response_rasters
Earlier static response(s) used as predictors for one regressed condition.
A name joining several frames with "+" (e.g. "2000ms+2250ms") stacks their
responses along the channel axis, so they are regressed out together.

INPUT:
    - rasters_by_frame: dict[str, np.ndarray] -> channels x static time x
        stimuli response of every earlier frame (e.g. "2000ms", "2250ms").
    - condition_name: str -> one frame name or several joined by "+".

OUTPUT:
    - predictor_rasters: np.ndarray -> (channels * frames) x static time x stimuli.
"""
def previous_response_rasters(rasters_by_frame, condition_name):
    return np.concatenate(
        [rasters_by_frame[frame] for frame in condition_name.split("+")], axis=0,
    )
# EOF


"""
regress_out_static_response
Remove an earlier static response from the last-frame static response with one
of two per-timepoint regressions (stimuli are the samples in both):
    - "channelwise": channelwise_regress_out, every channel regressed on the
      same channel's earlier response (one predictor, in-sample OLS);
    - "timepoint": timepoint_static_regress_out, every channel regressed on all
      (delay-embedded) channels of the earlier response, "lr" or "ridge",
      with out-of-fold ("kf") or in-sample ("same") residuals, optionally
      after an in-fold PCA of the predictors ("pca_variance").
The earlier response may hold more channels than the last frame (several
frames stacked by previous_response_rasters); only "timepoint" allows this.

INPUT:
    - previous_rasters: np.ndarray -> channels x static time x stimuli earlier response.
    - static_rasters: np.ndarray -> channels x static time x stimuli last frame.
    - fs: float -> sampling rate of both responses in Hz.
    - method: str -> "channelwise" or "timepoint".
    - timepoint_kwargs: dict | None -> keyword arguments of
        timepoint_static_regress_out (delay_embedding_lags, regression_type,
        alphas, cv_type, n_splits, pca_variance); ignored for "channelwise".

OUTPUT:
    - residual_rasters: np.ndarray -> channels x static time x stimuli residual.
"""
def regress_out_static_response(
        previous_rasters, static_rasters, fs, method="channelwise",
        timepoint_kwargs=None,
        ):
    if method == "channelwise":
        if np.shape(previous_rasters) != np.shape(static_rasters):
            raise ValueError(
                "channelwise regression needs one predictor channel per target "
                "channel; use method='timepoint' for stacked frames."
            )
        # end if predictor and target channels differ
        residual_rasters, _, _, _ = channelwise_regress_out(
            previous_rasters, static_rasters,
        )
        return residual_rasters
    elif method == "timepoint":
        residual_ts, _, _ = timepoint_static_regress_out(
            TimeSeries(np.asarray(previous_rasters, dtype=np.float64), fs=fs),
            TimeSeries(np.asarray(static_rasters, dtype=np.float64), fs=fs),
            **(timepoint_kwargs or {}),
        )
        return residual_ts.get_array()
    # end if method
    raise ValueError("method must be 'channelwise' or 'timepoint'.")
# EOF


"""
regress_out_static_rdms
Remove earlier-frame RDMs from the last-frame RDM in RDM space: at every
static timepoint t, the last-frame RDM is regressed (OLS with intercept) on the
RDMs of every earlier frame at t (or at t + lag for every delay-embedding lag),
with stimulus pairs as samples, and the residual RDM is kept.

INPUT:
    - previous_rdms: list[np.ndarray] -> one static time x stimulus-pair RDM
        timeseries per earlier frame (several frames are regressed out together).
    - static_rdms: np.ndarray -> static time x stimulus-pair last-frame RDMs.
    - delay_embedding_lags: tuple[int, int] | None -> inclusive sample lags,
        e.g. (-2, 2) uses the earlier RDMs at t-2 ... t+2; times beyond the
        response edges repeat the first / last RDM (edge padding, as in
        _embedded_predictor_array). None uses only time t.

OUTPUT:
    - residual_rdms: np.ndarray -> static time x stimulus-pair residual RDMs.
    - variance_removed: np.ndarray -> static time fraction of RDM variance
        explained by the earlier-frame RDMs.
"""
def regress_out_static_rdms(previous_rdms, static_rdms, delay_embedding_lags=None):
    static_rdms = np.asarray(static_rdms, dtype=np.float64)
    n_times = static_rdms.shape[0]
    lags = (
        [0] if delay_embedding_lags is None
        else range(delay_embedding_lags[0], delay_embedding_lags[1] + 1)
    )
    residual_rdms = np.empty_like(static_rdms)
    variance_removed = np.empty(n_times)
    for time_index, target_rdm in enumerate(static_rdms):
        # Predictor times t + lag, clipped to the response (edge padding).
        predictor_times = [min(max(time_index + lag, 0), n_times - 1) for lag in lags]
        # stimulus pairs x (frames * lags + 1) design: earlier RDMs plus intercept.
        design_matrix = np.column_stack(
            [rdms[predictor_time] for rdms in previous_rdms for predictor_time in predictor_times]
            + [np.ones_like(target_rdm)]
        )
        coefficients = np.linalg.lstsq(design_matrix, target_rdm, rcond=None)[0]
        residual_rdms[time_index] = target_rdm - design_matrix @ coefficients
        variance_removed[time_index] = 1 - residual_rdms[time_index].var() / target_rdm.var()
    # end for time_index
    return residual_rdms, variance_removed
# EOF


"""
window_rdm_timeseries
One RDM of an earlier-frame response averaged over a static window, repeated
at every static timepoint so it can enter regress_out_static_rdms as a fixed
predictor: every last-frame bin is then regressed on the same earlier-frame
geometry. Averaging the window before building the RDM lowers its noise.

INPUT:
    - rasters: np.ndarray -> channels x static time x stimuli earlier response.
    - fs: float -> sampling rate of rasters (Hz).
    - window_ms: tuple[float, float] -> end-exclusive static window.
    - rdm_metric: str -> create_RDM distance.

OUTPUT:
    - window_rdms: np.ndarray -> static time x stimulus pairs, every row the
        same window-averaged RDM.
"""
def window_rdm_timeseries(rasters, fs, window_ms, rdm_metric):
    time_ms = np.arange(rasters.shape[1]) * 1000 / fs
    window = (time_ms >= window_ms[0]) & (time_ms < window_ms[1])
    if not window.any():
        raise ValueError(f"Static window {window_ms} contains no samples.")
    # end if empty window
    # channels x stimuli: mean response in the window, then one RDM.
    window_rdm = create_RDM(
        np.ascontiguousarray(rasters[:, window, :].mean(axis=1)), metric=rdm_metric,
    )
    return np.tile(window_rdm, (rasters.shape[1], 1))
# EOF


"""
regressed_condition_suffix
File-name token of one regressed static condition, shared by the permutation
runner and the notebooks that load its results. The channelwise method keeps
the historical "<frame>-signal-regressed" name; the RDM-space method is
"<frame>-rdm-regressed", or "<frame>-rdm-lags<first>to<last>-regressed" with
delay-embedding lags.

INPUT:
    - previous_name: str -> regressed-out response, e.g. "2250ms".
    - method: str -> "channelwise", "timepoint" or "rdm".
    - regression_type: str -> "lr" or "ridge" (timepoint only).
    - cv_type: str -> "kf" or "same" (timepoint only).
    - pca_variance: float | None -> PCA variance fraction (timepoint only).
    - rdm_delay_embedding_lags: tuple[int, int] | None -> lags of the RDM-space
        regression (rdm only); None = same timepoint only.

OUTPUT:
    - suffix: str -> e.g. "2250ms-signal-regressed",
        "2250ms-timepoint-ridge-kf-regressed",
        "2000ms+2250ms-timepoint-ridge-kf-pca95-regressed" or
        "2000ms+2250ms-rdm-lags-2to2-regressed" or
        "2250ms-rdm-win100to300-regressed".
    - rdm_window_ms: tuple[float, float] | None -> static window of the
        window-averaged earlier RDM (rdm only); overrides the lags token.
"""
def regressed_condition_suffix(previous_name, method="channelwise",
                               regression_type="ridge", cv_type="kf",
                               pca_variance=None, rdm_delay_embedding_lags=None,
                               rdm_window_ms=None):
    if method == "channelwise":
        return f"{previous_name}-signal-regressed"
    elif method == "rdm":
        if rdm_window_ms is not None:
            return f"{previous_name}-rdm-win{rdm_window_ms[0]:g}to{rdm_window_ms[1]:g}-regressed"
        # end if window-averaged predictor
        if rdm_delay_embedding_lags is None:
            return f"{previous_name}-rdm-regressed"
        # end if no lags
        first_lag, last_lag = rdm_delay_embedding_lags
        return f"{previous_name}-rdm-lags{first_lag:d}to{last_lag:d}-regressed"
    # end if channelwise
    pca_token = "" if pca_variance is None else f"-pca{round(100 * pca_variance):d}"
    return f"{previous_name}-{method}-{regression_type}-{cv_type}{pca_token}-regressed"
# EOF


"""
regressed_static_dynamic_drsa
Static-dynamic dRSA (movie time x static time RDM similarity) for the raw
last-frame response and after regressing out each earlier static response
with regress_out_static_response; the static RDMs are built from the
residual responses (or, with "rdm", regressed directly in RDM space).

INPUT:
    - static_rasters: np.ndarray -> channels x static time x stimuli last frame.
    - dynamic_rasters: np.ndarray -> channels x movie time x stimuli.
    - previous_rasters_by_name: dict[str, np.ndarray] -> earlier static
        responses per condition (e.g. {"2000ms": ..., "2250ms": ...,
        "2000ms+2250ms": stacked}), sharing static time and stimuli with
        static_rasters.
    - rdm_metric: str -> distance metric for the neural RDMs.
    - rsa_metric: str -> "correlation" or "spearman".
    - fs: float -> sampling rate of the static responses in Hz.
    - regress_out_method: str -> "channelwise", "timepoint" or "rdm" (the
        static RDMs are regressed on the earlier-frame RDMs with
        regress_out_static_rdms instead of regressing the responses).
    - timepoint_kwargs: dict | None -> options of timepoint_static_regress_out.
    - rdm_delay_embedding_lags: tuple[int, int] | None -> lags of the earlier
        RDMs in the RDM-space regression (regress_out_static_rdms); "rdm" only.
    - rdm_window_ms: tuple[float, float] | None -> "rdm" only: average every
        earlier response over this static window and regress the one resulting
        RDM per frame out of every last-frame bin (window_rdm_timeseries);
        overrides rdm_delay_embedding_lags.
    - static_whiteners: np.ndarray | None -> static time x channels x channels
        noise whiteners of the last frame, applied after the regression (to the
        raw and residual responses); with rdm_metric "sqeuclidean" the RDMs are
        squared Mahalanobis distances
    - dynamic_whiteners: np.ndarray | None -> movie time x channels x channels
        noise whiteners of the movie

OUTPUT:
    - drsa_by_condition: dict[str, np.ndarray] -> movie time x static time
        similarity for "raw" and for every regressed-out response name.
"""
def regressed_static_dynamic_drsa(
        static_rasters, dynamic_rasters, previous_rasters_by_name,
        rdm_metric="cosine_cnt", rsa_metric="spearman", fs=100,
        regress_out_method="channelwise", timepoint_kwargs=None,
        rdm_delay_embedding_lags=None, rdm_window_ms=None,
        static_whiteners=None, dynamic_whiteners=None,
        ):
    if dynamic_whiteners is not None:
        dynamic_rasters = whiten_rasters(dynamic_rasters, dynamic_whiteners)
    # end if movie whitening
    dynamic_rdms = compute_rdm_timeseries(dynamic_rasters, rdm_metric)
    if regress_out_method == "rdm":
        if static_whiteners is not None:
            raise NotImplementedError(
                "RDM-space regression of Mahalanobis RDMs needs noise models of the earlier frames."
            )
        # end if whitening
        # Regression on the RDMs themselves: frames stacked along the channel
        # axis (previous_response_rasters) are split back into one RDM each.
        static_rdms = compute_rdm_timeseries(static_rasters, rdm_metric)
        static_rdms_by_condition = {"raw": static_rdms}
        for previous_name, previous_rasters in previous_rasters_by_name.items():
            frame_rasters = np.split(
                np.asarray(previous_rasters), len(previous_rasters) // len(static_rasters), axis=0,
            )
            if rdm_window_ms is not None:
                # One window-averaged RDM per frame, the same at every static bin.
                previous_rdms = [
                    window_rdm_timeseries(rasters, fs, rdm_window_ms, rdm_metric)
                    for rasters in frame_rasters
                ]
            else:
                previous_rdms = [
                    compute_rdm_timeseries(rasters, rdm_metric) for rasters in frame_rasters
                ]
            # end if window-averaged predictor
            static_rdms_by_condition[previous_name], _ = regress_out_static_rdms(
                previous_rdms, static_rdms,
                # A fixed predictor has no time axis to embed.
                delay_embedding_lags=None if rdm_window_ms is not None else rdm_delay_embedding_lags,
            )
        # end for previous_name
        return {
            condition_name: cross_temporal_similarity(
                dynamic_rdms, condition_rdms, metric=rsa_metric,
            )
            for condition_name, condition_rdms in static_rdms_by_condition.items()
        }
    # end if rdm regression
    # The raw last-frame response is the reference condition.
    static_rasters_by_condition = {"raw": static_rasters}
    for previous_name, previous_rasters in previous_rasters_by_name.items():
        static_rasters_by_condition[previous_name] = regress_out_static_response(
            previous_rasters, static_rasters, fs,
            method=regress_out_method, timepoint_kwargs=timepoint_kwargs,
        )
    # end for previous_name
    if static_whiteners is not None:
        # Residuals are expressed in units of the last frame's trial noise.
        static_rasters_by_condition = {
            condition_name: whiten_rasters(condition_rasters, static_whiteners)
            for condition_name, condition_rasters in static_rasters_by_condition.items()
        }
    # end if static whitening
    return {
        condition_name: cross_temporal_similarity(
            dynamic_rdms,
            compute_rdm_timeseries(condition_rasters, rdm_metric),
            metric=rsa_metric,
        )
        for condition_name, condition_rasters in static_rasters_by_condition.items()
    }
# EOF
