import numpy as np
from sklearn.model_selection import KFold

from useful_stuff.general_utils import TimeSeries, dyn_linear_encoding


"""
cross_temporal_static_dynamic_regression
Use each predictor timepoint to predict every target timepoint across stimuli.
The dyn_linear_encoding class fits and cross-validates every predictor-time by
target-time pair independently.

INPUT:
    - predictor_ts: TimeSeries -> predictor features x predictor time x stimuli.
    - target_ts: TimeSeries -> target features x target time x stimuli.
    - regression_type: str -> useful_stuff regression model type.
    - cv_type: str -> useful_stuff cross-validation type.
    - score_type: str -> "corr" for pattern correlation or "r2".
    - n_splits: int -> number of folds when cv_type is "kf".
    - shuffle: bool -> whether the useful_stuff CV splitter shuffles stimuli.
    - alphas: tuple[float, ...] -> regularization values for ridge-like models.
    - fit_intercept: bool -> whether the linear model includes an intercept.

OUTPUT:
    - score_matrix: np.ndarray -> target time x predictor time regression scores.
"""
def cross_temporal_static_dynamic_regression(
        predictor_ts: TimeSeries,
        target_ts: TimeSeries,
        regression_type: str = "ridge",
        cv_type: str = "kf",
        score_type: str = "corr",
        n_splits: int = 5,
        shuffle: bool = False,
        alphas: tuple[float, ...] = (
            1e-6, 1e-4, 1e-2, 1, 1e2, 1e4,
        ),
        fit_intercept: bool = True,
        ) -> np.ndarray:
    predictor_array = np.asarray(predictor_ts.get_array())
    target_array = np.asarray(target_ts.get_array())
    if predictor_array.ndim != 3 or target_array.ndim != 3:
        raise ValueError(
            "predictor_ts and target_ts must have shape "
            "features x time x stimuli."
        )
    # end if predictor_array.ndim != 3 or target_array.ndim != 3
    if predictor_array.shape[2] != target_array.shape[2]:
        raise ValueError(
            "Predictor and target responses must contain the same stimuli."
        )
    # end if predictor_array.shape[2] != target_array.shape[2]
    if predictor_array.shape[2] < 2:
        raise ValueError("Need at least two stimuli for regression.")
    # end if predictor_array.shape[2] < 2
    if score_type not in ("corr", "r2"):
        raise ValueError("score_type must be either 'corr' or 'r2'.")
    # end if score_type not in ("corr", "r2")

    score_matrix = np.empty(
        (target_array.shape[1], predictor_array.shape[1]),
        dtype=float,
    )

    for predictor_time in range(predictor_array.shape[1]):
        predictor = predictor_array[:, predictor_time, :]
        regression_model = dyn_linear_encoding(
            regression_type=regression_type,
            cv_type=cv_type,
            max_lag=0,
            score_type=score_type,
            n_splits=n_splits,
            shuffle=shuffle,
            alphas=np.asarray(alphas),
            fit_intercept=fit_intercept,
        )

        # crossvalidate_static_dyn calls crossvalidate independently for every
        # target timepoint. Ridge parameters and weights are therefore never
        # shared between different movie times or predictor times.
        time_scores = regression_model.crossvalidate_static_dyn(
            predictor,
            target_ts,
            transpose=True,
        ).get_array()
        if score_type == "corr":
            # The class returns one pattern-correlation score per target time.
            score_matrix[:, predictor_time] = np.squeeze(time_scores)
        else:
            # The class returns channel x target-time R2 values.
            score_matrix[:, predictor_time] = np.nanmean(
                time_scores,
                axis=0,
            )
        # end if score_type == "corr"
    # end for predictor_time

    return score_matrix
# EOF


"""
autoregressive_regress_out
Fit a shared linear autoregressive model across timepoints and stimuli, then
return the innovation that is not predicted by the response's own past.

INPUT:
    - response_ts: TimeSeries -> channels x time x stimuli neural response.
    - first_delay: int -> closest causal lag in samples; n gives x(t-n).
    - n_delays: int -> number of consecutive predictors starting at first_delay.
    - regression_type: str -> useful_stuff linear model type.
    - fit_intercept: bool -> whether to fit an affine intercept in addition to A.

OUTPUT:
    - residual_ts: TimeSeries -> channels x valid time x stimuli innovations.
    - regression_model: dyn_linear_encoding -> fitted shared autoregressive model.
    - variance_explained: np.ndarray -> fraction of variance removed per channel.
"""
def autoregressive_regress_out(
        response_ts: TimeSeries,
        first_delay: int = 1,
        n_delays: int = 1,
        regression_type: str = "lr",
        fit_intercept: bool = False,
        ) -> tuple[TimeSeries, dyn_linear_encoding, np.ndarray]:
    response_array = np.asarray(response_ts.get_array())
    if response_array.ndim != 3:
        raise ValueError(
            "response_ts must have shape channels x time x stimuli."
        )
    # end if response_array.ndim != 3
    if not isinstance(first_delay, int) or first_delay < 1:
        raise ValueError("first_delay must be a positive integer.")
    # end if not isinstance(first_delay, int) or first_delay < 1
    if not isinstance(n_delays, int) or n_delays < 1:
        raise ValueError("n_delays must be a positive integer.")
    # end if not isinstance(n_delays, int) or n_delays < 1

    # Example: first_delay=5 and n_delays=3 uses lags [5, 6, 7].
    delays = range(first_delay, first_delay + n_delays)
    maximum_delay = first_delay + n_delays - 1
    if maximum_delay >= response_array.shape[1]:
        raise ValueError(
            "The largest predictor delay must be smaller than the number of "
            "response timepoints."
        )
    # end if maximum_delay >= response_array.shape[1]

    # All targets start where every requested causal predictor is available.
    target_array = response_array[:, maximum_delay:, :]

    # Stack causal channel patterns in lag order:
    # [x(t-first_delay), ..., x(t-maximum_delay)].
    delayed_predictors = []
    for delay in delays:
        delayed_predictors.append(
            response_array[
                :,
                maximum_delay - delay:response_array.shape[1] - delay,
                :,
            ]
        )
    # end for delay
    predictor_array = np.concatenate(delayed_predictors, axis=0)

    # Treat every valid time-stimulus pair as one sample and fit one shared A.
    target_shape = target_array.shape
    predictor_flat = predictor_array.reshape(predictor_array.shape[0], -1)
    target_flat = target_array.reshape(target_shape[0], -1)
    regression_model = dyn_linear_encoding(
        regression_type=regression_type,
        cv_type="same",
        max_lag=0,
        fit_intercept=fit_intercept,
    )
    regression_model.fit(predictor_flat, target_flat)
    predicted_flat = regression_model.predict(predictor_flat)
    residual_array = (target_flat - predicted_flat).reshape(target_shape)

    # Report the channel-wise fraction removed without assuming an intercept.
    target_variance = np.var(target_array, axis=(1, 2))
    residual_variance = np.var(residual_array, axis=(1, 2))
    variance_explained = np.zeros_like(target_variance, dtype=float)
    nonconstant_channels = target_variance > 0
    variance_explained[nonconstant_channels] = (
        1
        - residual_variance[nonconstant_channels]
        / target_variance[nonconstant_channels]
    )

    residual_ts = TimeSeries(residual_array, fs=response_ts.get_fs())
    return residual_ts, regression_model, variance_explained
# EOF


"""
static_pattern_regress_out
Fit, independently at every dynamic timepoint, a linear map from one static
response pattern to the dynamic response across stimuli, then remove the
predicted component. Stimuli are the regression samples, static channels are
the predictors, and dynamic channels are the outputs.

INPUT:
    - static_pattern: np.ndarray -> static channels x stimuli predictor pattern.
    - dynamic_ts: TimeSeries -> dynamic channels x time x stimuli response.
    - regression_type: str -> useful_stuff linear model type ("lr" is OLS).
    - fit_intercept: bool -> whether every timepoint fit includes an intercept.

OUTPUT:
    - residual_ts: TimeSeries -> dynamic channels x time x stimuli residuals.
    - variance_explained: np.ndarray -> time fraction of across-stimulus
        dynamic variance (summed over channels) removed by the static pattern.
"""
def static_pattern_regress_out(
        static_pattern: np.ndarray,
        dynamic_ts: TimeSeries,
        regression_type: str = "lr",
        fit_intercept: bool = True,
        ) -> tuple[TimeSeries, np.ndarray]:
    static_pattern = np.asarray(static_pattern, dtype=np.float64)
    dynamic_array = np.asarray(dynamic_ts.get_array(), dtype=np.float64)
    if static_pattern.ndim != 2 or dynamic_array.ndim != 3:
        raise ValueError(
            "static_pattern must be channels x stimuli and dynamic_ts "
            "channels x time x stimuli."
        )
    # end if input dimensions
    if static_pattern.shape[1] != dynamic_array.shape[2]:
        raise ValueError("Static and dynamic responses must share stimuli.")
    # end if static_pattern.shape[1] != dynamic_array.shape[2]
    if static_pattern.shape[0] + int(fit_intercept) >= static_pattern.shape[1]:
        raise ValueError(
            "The OLS fit needs more stimuli than predictors; otherwise the "
            "residual is trivially zero."
        )
    # end if too many predictors

    # fit_static_dyn stores one (dynamic x static channels) weight matrix per
    # dynamic timepoint, so no weights are shared across movie time.
    regression_model = dyn_linear_encoding(
        regression_type=regression_type,
        cv_type="same",
        max_lag=0,
        fit_intercept=fit_intercept,
    )
    fit_ts = TimeSeries(dynamic_array, fs=dynamic_ts.get_fs())
    regression_model.fit_static_dyn(static_pattern, fit_ts)
    predicted_ts = regression_model.predict_static_dyn(static_pattern)
    # The prediction is a list of channels x stimuli arrays; stack along time.
    predicted_ts.to_numpy()
    residual_array = dynamic_array - predicted_ts.get_array()

    # Across-stimulus variance summed over channels, one value per timepoint.
    target_variance = np.var(dynamic_array, axis=2).sum(axis=0)
    residual_variance = np.var(residual_array, axis=2).sum(axis=0)
    variance_explained = np.zeros_like(target_variance)
    np.divide(
        target_variance - residual_variance,
        target_variance,
        out=variance_explained,
        where=target_variance > 0,
    )

    residual_ts = TimeSeries(residual_array, fs=dynamic_ts.get_fs())
    return residual_ts, variance_explained
# EOF


"""
encoding_timecourse
Cross-validated linear encoding of a neural response timecourse from one
static model feature pattern. At every neural timepoint an independent ridge
model maps the model features to the channel pattern across stimuli
(dyn_linear_encoding.crossvalidate_static_dyn), and the held-out score is
averaged over channels. The stimulus folds are fixed (seeded), so every
timepoint, layer and condition is scored on the same train/test splits.

INPUT:
    - model_features: np.ndarray -> features x stimuli predictor pattern.
    - neural_rasters: np.ndarray -> channels x time x stimuli response.
    - fs: float -> sampling rate of neural_rasters (Hz).
    - regression_type: str -> useful_stuff regression model type.
    - alphas: tuple[float, ...] -> ridge penalties searched by RidgeCV.
    - n_splits: int -> number of stimulus folds.
    - score_type: str -> "r2" (per channel and fold, then averaged), "corr"
        (useful_stuff pattern correlation), or "r2_pooled": the held-out
        predictions of all folds are collected first and R2 is computed once
        per channel over every stimulus, then averaged over channels. Pooling
        avoids scoring each small test fold against its own mean, which
        pushes the fold-wise R2 of an uninformative model to about -n_test /
        (n_test - 1) * (1 / n_test + 1 / n_train) instead of about 0.
        "corr_pooled" uses the same pooled held-out predictions but scores
        each channel by its Pearson correlation with the response across
        stimuli (insensitive to offset and scale errors of the prediction),
        then averages over channels.
    - random_state: int -> seed of the shuffled stimulus folds.
    - alpha_per_target: bool -> RidgeCV picks a separate penalty for every
        channel instead of one shared penalty for all channels.

OUTPUT:
    - scores: np.ndarray -> (time,) cross-validated encoding score.
"""
def encoding_timecourse(
        model_features: np.ndarray,
        neural_rasters: np.ndarray,
        fs: float,
        regression_type: str = "ridge",
        alphas: tuple[float, ...] = tuple(10.0 ** np.arange(-2, 11)),
        n_splits: int = 5,
        score_type: str = "r2",
        random_state: int = 0,
        alpha_per_target: bool = False,
        ) -> np.ndarray:
    model_features = np.asarray(model_features, dtype=float)
    neural_rasters = np.asarray(neural_rasters, dtype=float)
    if model_features.shape[1] != neural_rasters.shape[2]:
        raise ValueError("Model features and neural responses must share stimuli.")
    # end if stimulus counts differ
    regression_model = dyn_linear_encoding(
        regression_type=regression_type,
        cv_type="kf",
        max_lag=0,
        score_type="r2" if score_type in ("r2_pooled", "corr_pooled") else score_type,
        n_splits=n_splits,
        alphas=np.asarray(alphas),
        # Forwarded to RidgeCV; one penalty per output channel when True.
        alpha_per_target=alpha_per_target,
    )
    # choose_CV_type has no seed, so the shuffled KFold is replaced by a
    # seeded one: otherwise every timepoint would get different folds.
    regression_model.cv_obj = KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )
    if score_type in ("r2_pooled", "corr_pooled"):
        # channels x time x stimuli held-out predictions, one fold at a time;
        # fit_static_dyn fits an independent model at every timepoint.
        predicted = np.empty_like(neural_rasters)
        for train_index, test_index in regression_model.cv_obj.split(model_features.T):
            regression_model.fit_static_dyn(
                model_features[:, train_index],
                TimeSeries(neural_rasters[:, :, train_index], fs=fs),
            )
            fold_prediction = regression_model.predict_static_dyn(
                model_features[:, test_index],
            )
            # list of channels x test stimuli arrays -> channels x time x stimuli
            fold_prediction.to_numpy()
            predicted[:, :, test_index] = fold_prediction.get_array()
        # end for train_index, test_index
        if score_type == "corr_pooled":
            # Pearson r across stimuli per channel and timepoint.
            response_centered = neural_rasters - neural_rasters.mean(axis=2, keepdims=True)
            predicted_centered = predicted - predicted.mean(axis=2, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                channel_corr = (response_centered * predicted_centered).sum(axis=2) / np.sqrt(
                    (response_centered ** 2).sum(axis=2) * (predicted_centered ** 2).sum(axis=2)
                )
            # end with np.errstate
            return np.nanmean(channel_corr, axis=0)
        # end if pooled correlation
        # R2 over all stimuli per channel and timepoint; flat channels -> NaN.
        residual_power = ((neural_rasters - predicted) ** 2).sum(axis=2)
        total_power = (
            (neural_rasters - neural_rasters.mean(axis=2, keepdims=True)) ** 2
        ).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            channel_r2 = 1 - residual_power / total_power
        # end with np.errstate
        return np.nanmean(channel_r2, axis=0)
    # end if pooled R2
    # r2: channels x time; corr: 1 x time (pattern correlation).
    time_scores = regression_model.crossvalidate_static_dyn(
        model_features, TimeSeries(neural_rasters, fs=fs), transpose=True,
    ).get_array()
    return np.nanmean(time_scores, axis=0)
# EOF
