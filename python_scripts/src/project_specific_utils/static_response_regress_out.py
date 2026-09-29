"""
Alternative ways of regressing an earlier static response out of the
last-frame static response, plus static response windows used as "layers".

The earlier static response (e.g. the 2000 ms or 2250 ms movie frame shown as
an image) is the predictor, and the last-frame static response is the target.
Both are channels x time x stimuli and share the same stimulus order.
"""

import numpy as np

from useful_stuff.general_utils import TimeSeries, dyn_linear_encoding

from .channelwise_correlation import channelwise_regress_out
from .split_half_rsa import compute_rdm_timeseries, cross_temporal_similarity


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

OUTPUT:
    - predicted_array: np.ndarray -> channels x slices x stimuli prediction.
    - chosen_alphas: np.ndarray -> ridge penalty of the last fit per slice.
"""
def _slicewise_regress_out(
        predictor_array, target_array, regression_type, alphas, cv_type,
        n_splits, shuffle, fit_intercept,
        ):
    if cv_type not in ('kf', 'same'):
        raise ValueError("cv_type must be 'kf' or 'same'.")
    # end if cv_type
    n_slices = target_array.shape[1]
    predicted_array = np.zeros_like(target_array)
    chosen_alphas = np.full(n_slices, np.nan)
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
            regression_model.fit(
                predictor_s[:, train_idx], target_s[:, train_idx],
            )
            predicted_array[:, slice_index, test_idx] = regression_model.predict(
                predictor_s[:, test_idx],
            )
        # end for train_idx, test_idx
        if hasattr(regression_model.get_regression_obj(), 'alpha_'):
            chosen_alphas[slice_index] = regression_model.get_regression_obj().alpha_
        # end if the regression object tunes a penalty
    # end for slice_index
    return predicted_array, chosen_alphas
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
        ):
    _check_static_pair(predictor_ts, target_ts)
    predictor_array = _embedded_predictor_array(
        predictor_ts, delay_embedding_lags,
    )
    target_array = np.asarray(target_ts.get_array(), dtype=np.float64)
    # Every static timepoint is one slice with its own model.
    predicted_array, chosen_alphas = _slicewise_regress_out(
        predictor_array, target_array, regression_type, alphas, cv_type,
        n_splits, shuffle, fit_intercept,
    )

    residual_array = target_array - predicted_array
    residual_ts = TimeSeries(residual_array, fs=target_ts.get_fs())
    return (
        residual_ts, _variance_explained(target_array, residual_array),
        chosen_alphas,
    )
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
    predicted_patterns, chosen_alphas = _slicewise_regress_out(
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
regressed_static_dynamic_drsa
Static-dynamic dRSA (movie time x static time RDM similarity) for the raw
last-frame response and after regressing out each earlier static response.
The regression is channelwise_regress_out: for every channel and static
timepoint, the last-frame response across stimuli is regressed on the same
channel's earlier-frame response, and the static RDMs are built from the
residuals (the "signal" option of static_dynamic_drsa_permutation.ipynb).

INPUT:
    - static_rasters: np.ndarray -> channels x static time x stimuli last frame.
    - dynamic_rasters: np.ndarray -> channels x movie time x stimuli.
    - previous_rasters_by_name: dict[str, np.ndarray] -> earlier static
        responses (e.g. {"2000ms": ..., "2250ms": ...}), shaped like
        static_rasters.
    - rdm_metric: str -> distance metric for the neural RDMs.
    - rsa_metric: str -> "correlation" or "spearman".

OUTPUT:
    - drsa_by_condition: dict[str, np.ndarray] -> movie time x static time
        similarity for "raw" and for every regressed-out response name.
"""
def regressed_static_dynamic_drsa(
        static_rasters, dynamic_rasters, previous_rasters_by_name,
        rdm_metric="cosine_cnt", rsa_metric="spearman",
        ):
    dynamic_rdms = compute_rdm_timeseries(dynamic_rasters, rdm_metric)
    # The raw last-frame response is the reference condition.
    static_rasters_by_condition = {"raw": static_rasters}
    for previous_name, previous_rasters in previous_rasters_by_name.items():
        residual_rasters, _, _, _ = channelwise_regress_out(
            previous_rasters, static_rasters,
        )
        static_rasters_by_condition[previous_name] = residual_rasters
    # end for previous_name
    return {
        condition_name: cross_temporal_similarity(
            dynamic_rdms,
            compute_rdm_timeseries(condition_rasters, rdm_metric),
            metric=rsa_metric,
        )
        for condition_name, condition_rasters in static_rasters_by_condition.items()
    }
# EOF
