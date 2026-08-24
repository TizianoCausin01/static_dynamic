import numpy as np


"""
channelwise_regress_out
For every neural channel and response time independently, remove from a target
trial vector the linear component explained by the matched predictor trial
vector. Trials are the regression observations, so information is never mixed
across channels or timepoints.

INPUT:
    - predictor_rasters: np.ndarray -> channels x time x matched trials
    - target_rasters: np.ndarray -> channels x time x matched trials
    - fit_intercept: bool -> whether each channel regression includes an intercept

OUTPUT:
    - residual_rasters: np.ndarray -> target residuals with the original shape
    - slopes: np.ndarray -> channels x time fitted predictor coefficients
    - intercepts: np.ndarray -> channels x time fitted intercepts
    - variance_explained: np.ndarray -> channels x time fraction of target
        across-trial variance removed
"""
def channelwise_regress_out(
        predictor_rasters: np.ndarray,
        target_rasters: np.ndarray,
        fit_intercept: bool = True,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictor_rasters = np.asarray(predictor_rasters, dtype=np.float64)
    target_rasters = np.asarray(target_rasters, dtype=np.float64)
    if predictor_rasters.ndim != 3 or target_rasters.ndim != 3:
        raise ValueError(
            "Both rasters must have shape channels x time x matched trials."
        )
    # end if raster dimensions
    if predictor_rasters.shape != target_rasters.shape:
        raise ValueError(
            "Predictor and target rasters must have identical shapes."
        )
    # end if raster shapes
    if not (
            np.all(np.isfinite(predictor_rasters))
            and np.all(np.isfinite(target_rasters))
            ):
        raise ValueError("Input rasters contain non-finite values.")
    # end if finite inputs

    if target_rasters.shape[2] < 2:
        raise ValueError("At least two matched trials are required.")
    # end if trial count

    if fit_intercept:
        # Center each channel-time trial vector independently. The fitted slope
        # therefore describes only across-trial covariation at that timepoint.
        predictor_mean = predictor_rasters.mean(axis=2)
        target_mean = target_rasters.mean(axis=2)
        predictor_centered = (
            predictor_rasters - predictor_mean[:, :, np.newaxis]
        )
        target_centered = (
            target_rasters - target_mean[:, :, np.newaxis]
        )
        slope_numerators = np.sum(
            predictor_centered * target_centered,
            axis=2,
        )
        slope_denominators = np.sum(predictor_centered ** 2, axis=2)
        slopes = np.zeros(target_rasters.shape[:2], dtype=np.float64)
        np.divide(
            slope_numerators,
            slope_denominators,
            out=slopes,
            where=slope_denominators > 0,
        )
        intercepts = target_mean - slopes * predictor_mean
    else:
        slope_numerators = np.sum(
            predictor_rasters * target_rasters,
            axis=2,
        )
        slope_denominators = np.sum(predictor_rasters ** 2, axis=2)
        slopes = np.zeros(target_rasters.shape[:2], dtype=np.float64)
        np.divide(
            slope_numerators,
            slope_denominators,
            out=slopes,
            where=slope_denominators > 0,
        )
        intercepts = np.zeros(target_rasters.shape[:2], dtype=np.float64)
    # end if fit_intercept

    predicted_rasters = (
        slopes[:, :, np.newaxis] * predictor_rasters
        + intercepts[:, :, np.newaxis]
    )
    residual_rasters = target_rasters - predicted_rasters

    target_variance = np.var(target_rasters, axis=2)
    residual_variance = np.var(residual_rasters, axis=2)
    variance_explained = np.zeros(
        target_rasters.shape[:2],
        dtype=np.float64,
    )
    np.divide(
        target_variance - residual_variance,
        target_variance,
        out=variance_explained,
        where=target_variance > 0,
    )

    return residual_rasters, slopes, intercepts, variance_explained
# EOF


"""
channelwise_static_dynamic_correlation
For every neural channel, correlate matched trials between every dynamic and
static timepoint. Unlike population-vector correlation, trials are the
observations and channels are kept separate.

INPUT:
    - dynamic_rasters: np.ndarray -> channels x dynamic time x matched trials
    - static_rasters: np.ndarray -> channels x static time x matched trials

OUTPUT:
    - channel_corr_matrices: np.ndarray -> channels x dynamic time x static time
    - average_corr_matrix: np.ndarray -> dynamic time x static time, averaged
        over channels while ignoring undefined correlations
"""
def channelwise_static_dynamic_correlation(
        dynamic_rasters: np.ndarray,
        static_rasters: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    dynamic_rasters = np.asarray(dynamic_rasters, dtype=np.float64)
    static_rasters = np.asarray(static_rasters, dtype=np.float64)
    if dynamic_rasters.ndim != 3 or static_rasters.ndim != 3:
        raise ValueError(
            "Both rasters must have shape channels x time x matched trials."
        )
    # end if raster dimensions
    if dynamic_rasters.shape[0] != static_rasters.shape[0]:
        raise ValueError(
            "Static and dynamic rasters must contain the same channels."
        )
    # end if channel counts
    if dynamic_rasters.shape[2] != static_rasters.shape[2]:
        raise ValueError(
            "Static and dynamic rasters must contain the same matched trials."
        )
    # end if trial counts
    if dynamic_rasters.shape[2] < 2:
        raise ValueError("At least two matched trials are required.")
    # end if trial count
    if not (
            np.all(np.isfinite(dynamic_rasters))
            and np.all(np.isfinite(static_rasters))
            ):
        raise ValueError("Input rasters contain non-finite values.")
    # end if finite inputs

    # Center every channel/time trial vector independently before computing
    # every dynamic-static dot product within each channel.
    dynamic_centered = (
        dynamic_rasters - dynamic_rasters.mean(axis=2, keepdims=True)
    )
    static_centered = (
        static_rasters - static_rasters.mean(axis=2, keepdims=True)
    )
    covariance_numerators = np.einsum(
        "cdk,csk->cds",
        dynamic_centered,
        static_centered,
        optimize=True,
    )
    dynamic_norms = np.linalg.norm(dynamic_centered, axis=2)
    static_norms = np.linalg.norm(static_centered, axis=2)
    correlation_denominators = (
        dynamic_norms[:, :, np.newaxis]
        * static_norms[:, np.newaxis, :]
    )

    # A timepoint with no across-trial variance has undefined correlation.
    channel_corr_matrices = np.full(
        covariance_numerators.shape,
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        covariance_numerators,
        correlation_denominators,
        out=channel_corr_matrices,
        where=correlation_denominators > 0,
    )
    channel_corr_matrices = np.clip(
        channel_corr_matrices, -1.0, 1.0
    )

    valid_counts = np.sum(np.isfinite(channel_corr_matrices), axis=0)
    correlation_sums = np.nansum(channel_corr_matrices, axis=0)
    average_corr_matrix = np.full(
        channel_corr_matrices.shape[1:],
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        correlation_sums,
        valid_counts,
        out=average_corr_matrix,
        where=valid_counts > 0,
    )
    return channel_corr_matrices, average_corr_matrix
# EOF


"""
channelwise_lag_curves
Average each channel's dynamic-static correlation matrix along diagonals. Lag
is defined as dynamic time minus static time, so positive values mean that the
dynamic response occurs later.

INPUT:
    - channel_corr_matrices: np.ndarray -> channels x dynamic time x static time
    - max_lag: int -> maximum positive and negative lag in samples

OUTPUT:
    - lag_curves: np.ndarray -> channels x (2 * max_lag + 1)
"""
def channelwise_lag_curves(
        channel_corr_matrices: np.ndarray,
        max_lag: int,
        ) -> np.ndarray:
    channel_corr_matrices = np.asarray(
        channel_corr_matrices, dtype=np.float64
    )
    if channel_corr_matrices.ndim != 3:
        raise ValueError(
            "channel_corr_matrices must have shape "
            "channels x dynamic time x static time."
        )
    # end if channel_corr_matrices.ndim
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")
    # end if max_lag
    if max_lag >= min(channel_corr_matrices.shape[1:]):
        raise ValueError(
            "max_lag must be smaller than both time dimensions."
        )
    # end if max_lag

    lags = np.arange(-max_lag, max_lag + 1)
    lag_curves = np.full(
        (channel_corr_matrices.shape[0], len(lags)),
        np.nan,
        dtype=np.float64,
    )
    for lag_index, lag in enumerate(lags):
        # np.diag offset is column minus row, hence the negative lag.
        diagonal_values = np.diagonal(
            channel_corr_matrices,
            offset=-lag,
            axis1=1,
            axis2=2,
        )
        valid_counts = np.sum(np.isfinite(diagonal_values), axis=1)
        diagonal_sums = np.nansum(diagonal_values, axis=1)
        np.divide(
            diagonal_sums,
            valid_counts,
            out=lag_curves[:, lag_index],
            where=valid_counts > 0,
        )
    # end for lag_index, lag
    return lag_curves
# EOF
