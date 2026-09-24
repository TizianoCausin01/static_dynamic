import numpy as np
from scipy.ndimage import gaussian_filter1d


"""
window_mean_responses
Average a channels x time x stimuli raster inside a half-open time window.
The returned sample indices select timestamps satisfying start_ms <= t < end_ms.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - window_ms: tuple[float, float] -> start and exclusive end in milliseconds
    - fs: float -> raster sampling frequency in Hz

OUTPUT:
    - mean_responses: np.ndarray -> channels x stimuli window-averaged responses
    - sample_window: tuple[int, int] -> start and exclusive end sample indices
"""
def window_mean_responses(
        rasters: np.ndarray,
        window_ms: tuple[float, float],
        fs: float,
        ) -> tuple[np.ndarray, tuple[int, int]]:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3:
        raise ValueError(
            "rasters must have shape channels x time x stimuli."
        )
    # end if rasters.ndim
    if len(window_ms) != 2:
        raise ValueError("window_ms must contain (start_ms, end_ms).")
    # end if len(window_ms)
    if fs <= 0:
        raise ValueError("fs must be positive.")
    # end if fs
    if not np.all(np.isfinite(rasters)):
        raise ValueError("rasters contain non-finite values.")
    # end if finite rasters

    start_ms, end_ms = window_ms
    if not (0 <= start_ms < end_ms):
        raise ValueError(
            "window_ms must satisfy 0 <= start_ms < end_ms."
        )
    # end if invalid window

    # Ceil implements the documented timestamp selection for non-integer bins.
    start_index = int(np.ceil(start_ms * fs / 1000))
    end_index = int(np.ceil(end_ms * fs / 1000))
    if start_index >= rasters.shape[1] or end_index > rasters.shape[1]:
        duration_ms = rasters.shape[1] * 1000 / fs
        raise ValueError(
            f"Window {window_ms} ms lies outside the {duration_ms:g} ms raster."
        )
    # end if window outside raster
    if end_index <= start_index:
        raise ValueError("window_ms selects no raster samples.")
    # end if empty window

    mean_responses = rasters[:, start_index:end_index, :].mean(axis=1)
    return mean_responses, (start_index, end_index)
# EOF


"""
window_smoothed_response_latencies
Gaussian-smooth a channels x time x stimuli raster, then summarize each
channel-stimulus timecourse by its peak and positive-mass centroid latency.
Smoothing is applied before windowing so the requested window does not create
artificial filter edges. Centroids are undefined when a window has no positive
response mass.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - window_ms: tuple[float, float] -> start and exclusive end in milliseconds
    - fs: float -> raster sampling frequency in Hz
    - smoothing_sigma_ms: float -> Gaussian standard deviation in milliseconds

OUTPUT:
    - peak_responses: np.ndarray -> channels x stimuli smoothed peak responses
    - peak_latencies_ms: np.ndarray -> channels x stimuli absolute peak times
    - centroid_latencies_ms: np.ndarray -> channels x stimuli absolute centroids
    - sample_window: tuple[int, int] -> start and exclusive end sample indices
"""
def window_smoothed_response_latencies(
        rasters: np.ndarray,
        window_ms: tuple[float, float],
        fs: float,
        smoothing_sigma_ms: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3:
        raise ValueError(
            "rasters must have shape channels x time x stimuli."
        )
    # end if rasters.ndim
    if len(window_ms) != 2:
        raise ValueError("window_ms must contain (start_ms, end_ms).")
    # end if len(window_ms)
    if fs <= 0:
        raise ValueError("fs must be positive.")
    # end if fs
    if smoothing_sigma_ms <= 0:
        raise ValueError("smoothing_sigma_ms must be positive.")
    # end if smoothing sigma
    if not np.all(np.isfinite(rasters)):
        raise ValueError("rasters contain non-finite values.")
    # end if finite rasters

    start_ms, end_ms = window_ms
    if not (0 <= start_ms < end_ms):
        raise ValueError(
            "window_ms must satisfy 0 <= start_ms < end_ms."
        )
    # end if invalid window

    # Ceil selects exactly the samples whose timestamps satisfy the window.
    start_index = int(np.ceil(start_ms * fs / 1000))
    end_index = int(np.ceil(end_ms * fs / 1000))
    if start_index >= rasters.shape[1] or end_index > rasters.shape[1]:
        duration_ms = rasters.shape[1] * 1000 / fs
        raise ValueError(
            f"Window {window_ms} ms lies outside the {duration_ms:g} ms raster."
        )
    # end if window outside raster
    if end_index <= start_index:
        raise ValueError("window_ms selects no raster samples.")
    # end if empty window

    sigma_samples = smoothing_sigma_ms * fs / 1000
    smoothed_rasters = gaussian_filter1d(
        rasters, sigma=sigma_samples, axis=1, mode="nearest",
    )
    windowed_rasters = smoothed_rasters[:, start_index:end_index, :]
    peak_offsets = np.argmax(windowed_rasters, axis=1)
    peak_responses = np.take_along_axis(
        windowed_rasters, peak_offsets[:, np.newaxis, :], axis=1,
    )[:, 0, :]
    peak_latencies_ms = (start_index + peak_offsets) * 1000 / fs

    # Only positive response mass supplies centroid weight, matching the
    # positive-mass latency convention used by the project's dRSA analyses.
    centroid_weights = np.clip(windowed_rasters, 0, None)
    window_time_ms = np.arange(start_index, end_index) * 1000 / fs
    centroid_weight_sums = centroid_weights.sum(axis=1)
    centroid_weighted_times = np.einsum(
        "cts,t->cs", centroid_weights, window_time_ms,
    )
    centroid_latencies_ms = np.full(
        centroid_weight_sums.shape, np.nan, dtype=float,
    )
    np.divide(
        centroid_weighted_times,
        centroid_weight_sums,
        out=centroid_latencies_ms,
        where=centroid_weight_sums > 0,
    )

    return (
        peak_responses,
        peak_latencies_ms,
        centroid_latencies_ms,
        (start_index, end_index),
    )
# EOF


"""
window_peak_responses
Gaussian-smooth a channels x time x stimuli raster, then return each
channel-stimulus peak and its absolute latency inside a half-open time window.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - window_ms: tuple[float, float] -> start and exclusive end in milliseconds
    - fs: float -> raster sampling frequency in Hz
    - smoothing_sigma_ms: float -> Gaussian standard deviation in milliseconds

OUTPUT:
    - peak_responses: np.ndarray -> channels x stimuli smoothed peak responses
    - peak_latencies_ms: np.ndarray -> channels x stimuli absolute peak times
    - sample_window: tuple[int, int] -> start and exclusive end sample indices
"""
def window_peak_responses(
        rasters: np.ndarray,
        window_ms: tuple[float, float],
        fs: float,
        smoothing_sigma_ms: float,
        ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    (
        peak_responses,
        peak_latencies_ms,
        _,
        sample_window,
    ) = window_smoothed_response_latencies(
        rasters, window_ms, fs, smoothing_sigma_ms,
    )
    return peak_responses, peak_latencies_ms, sample_window
# EOF


"""
rowwise_orthogonal_regression
Fit one total-least-squares line to every row of paired observations.
Unlike ordinary least squares, orthogonal regression treats deviations along
both axes symmetrically. Predictor and target must use comparable units.

INPUT:
    - predictor: np.ndarray -> rows x observations values on the x-axis
    - target: np.ndarray -> rows x observations values on the y-axis

OUTPUT:
    - intercepts: np.ndarray -> row-wise target-axis intercepts
    - slopes: np.ndarray -> row-wise target change per predictor unit
"""
def rowwise_orthogonal_regression(
        predictor: np.ndarray,
        target: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    predictor = np.asarray(predictor, dtype=float)
    target = np.asarray(target, dtype=float)
    if predictor.ndim != 2 or target.ndim != 2:
        raise ValueError("predictor and target must be two-dimensional.")
    # end if invalid dimensions
    if predictor.shape != target.shape:
        raise ValueError("predictor and target must have identical shapes.")
    # end if shapes differ
    if predictor.shape[1] < 2:
        raise ValueError("At least two paired observations are required.")
    # end if too few observations
    if not np.all(np.isfinite(predictor)) or not np.all(np.isfinite(target)):
        raise ValueError("predictor and target must contain only finite values.")
    # end if non-finite inputs

    row_count = predictor.shape[0]
    intercepts = np.full(row_count, np.nan)
    slopes = np.full(row_count, np.nan)

    # The leading covariance eigenvector gives the orthogonal-fit direction.
    for row_index in range(row_count):
        predictor_mean = predictor[row_index].mean()
        target_mean = target[row_index].mean()
        centered_pairs = np.column_stack((
            predictor[row_index] - predictor_mean,
            target[row_index] - target_mean,
        ))
        pair_covariance = centered_pairs.T @ centered_pairs
        eigenvalues, eigenvectors = np.linalg.eigh(pair_covariance)
        line_direction = eigenvectors[:, np.argmax(eigenvalues)]

        # A vertical fitted line has no finite target-on-predictor slope.
        if np.isclose(line_direction[0], 0):
            continue
        # end if vertical line
        if np.isclose(eigenvalues.max(), 0):
            continue
        # end if constant paired observations

        slopes[row_index] = line_direction[1] / line_direction[0]
        intercepts[row_index] = (
            target_mean - slopes[row_index] * predictor_mean
        )
    # end for row_index
    return intercepts, slopes
# EOF


"""
bootstrap_rowwise_orthogonal_slopes
Estimate percentile confidence intervals for row-wise orthogonal slopes by
resampling the paired observation axis with replacement.

INPUT:
    - predictor: np.ndarray -> rows x observations values on the x-axis
    - target: np.ndarray -> rows x observations values on the y-axis
    - bootstrap_repeats: int -> number of paired bootstrap resamples
    - confidence_level: float -> central interval probability in (0, 1)
    - random_seed: int -> random-number seed for reproducible resampling

OUTPUT:
    - slope_confidence_intervals: np.ndarray -> rows x (lower, upper) bounds
"""
def bootstrap_rowwise_orthogonal_slopes(
        predictor: np.ndarray,
        target: np.ndarray,
        bootstrap_repeats: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 0,
        ) -> np.ndarray:
    predictor = np.asarray(predictor, dtype=float)
    target = np.asarray(target, dtype=float)

    # Validate paired inputs once using the estimator used by every resample.
    rowwise_orthogonal_regression(predictor, target)
    if bootstrap_repeats < 2:
        raise ValueError("bootstrap_repeats must be at least 2.")
    # end if too few bootstrap repeats
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    # end if invalid confidence level

    random_generator = np.random.default_rng(random_seed)
    bootstrap_slopes = np.full((predictor.shape[0], bootstrap_repeats), np.nan)
    for bootstrap_index in range(bootstrap_repeats):
        observation_indices = random_generator.integers(
            0, predictor.shape[1], size=predictor.shape[1],
        )
        _, bootstrap_slopes[:, bootstrap_index] = (
            rowwise_orthogonal_regression(
                predictor[:, observation_indices],
                target[:, observation_indices],
            )
        )
    # end for bootstrap_index

    tail_probability = (1 - confidence_level) / 2
    interval_quantiles = (tail_probability, 1 - tail_probability)
    slope_confidence_intervals = np.nanquantile(
        bootstrap_slopes, interval_quantiles, axis=1,
    ).T
    return slope_confidence_intervals
# EOF
