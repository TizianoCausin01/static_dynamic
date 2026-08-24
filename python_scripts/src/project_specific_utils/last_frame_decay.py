from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from image_processing.pixel_values import load_aligned_pixel_value_features
from image_processing.video_feature_extraction import (
    list_video_feature_files,
    load_aligned_video_features,
    match_feature_stimulus_names,
)
from useful_stuff.general_utils import TimeSeries
from useful_stuff.general_utils.utils import mean_centering

from .dataloader import load_natraster, match_timed_static_movie_rasters


"""
resample_array
Resample a channels x time x stimuli array with the established TimeSeries helper.

INPUT:
    - data: np.ndarray -> channels x time x stimuli values
    - source_fs: float -> current sampling frequency in Hz
    - target_fs: float -> requested sampling frequency in Hz

OUTPUT:
    - resampled_data: np.ndarray -> data sampled at target_fs
"""
def resample_array(data, source_fs, target_fs):
    if source_fs == target_fs:
        return data
    # end if source_fs == target_fs
    time_series = TimeSeries(data, source_fs)
    time_series.resample(target_fs)
    return time_series.get_array()
# EOF


"""
resolve_visual_feature_path
Resolve either the pixel file or one layer from an extracted video model.

INPUT:
    - models_dir: str | Path -> directory containing visual-feature files
    - visual_feature_type: str -> pixels or model
    - visual_feature_path: str | Path | None -> optional explicit HDF5 file
    - pixel_step: int -> pixel sampling step used in the default pixel filename
    - model_name: str -> model prefix used by list_video_feature_files
    - model_dataset_name: str -> extracted model dataset label
    - model_pooling: str | None -> extracted feature pooling
    - model_layer_index: int -> naturally ordered model layer index

OUTPUT:
    - feature_path: Path -> selected visual-feature HDF5 file
"""
def resolve_visual_feature_path(
        models_dir,
        visual_feature_type="pixels",
        visual_feature_path=None,
        pixel_step=30,
        model_name="dino_v3_l",
        model_dataset_name="static_dynamic",
        model_pooling="mean",
        model_layer_index=-1,
        ):
    if visual_feature_path is not None:
        return Path(visual_feature_path).expanduser()
    # end if visual_feature_path
    models_dir = Path(models_dir).expanduser()
    if visual_feature_type == "pixels":
        return models_dir / (
            f"pixel_values_rgb_step{pixel_step}_static_dynamic.h5"
        )
    # end if visual_feature_type == pixels
    if visual_feature_type != "model":
        raise ValueError("visual_feature_type must be 'pixels' or 'model'.")
    # end if invalid visual_feature_type
    model_paths = list_video_feature_files(
        models_dir, model_name, model_dataset_name, model_pooling,
    )
    try:
        return model_paths[model_layer_index]
    except IndexError as error:
        raise IndexError(
            f"model_layer_index={model_layer_index} is invalid for "
            f"{len(model_paths)} {model_name} layers."
        ) from error
    # end try
# EOF


"""
load_last_frame_decay_data
Load and align averaged neural responses and pixel or model video features.

INPUT:
    - static_path: str | Path -> averaged static-condition natraster file
    - dynamic_path: str | Path -> averaged movie-condition natraster file
    - visual_feature_path: str | Path -> pixel or extracted-model HDF5 file
    - visual_feature_type: str -> pixels or model
    - source_fs: float -> neural source sampling frequency in Hz
    - new_fs: float -> neural analysis sampling frequency in Hz
    - good_channels: tuple[int, int] | None -> inclusive one-based channel range
    - center_neural_trials: bool -> center every neural feature over axis 2

OUTPUT:
    - data: dict -> aligned arrays, stimulus names, sampling rates, and metadata
"""
def load_last_frame_decay_data(
        static_path,
        dynamic_path,
        visual_feature_path,
        visual_feature_type="pixels",
        source_fs=1000,
        new_fs=100,
        good_channels=None,
        center_neural_trials=False,
        ):
    static_rasters, static_names = load_natraster(static_path)
    dynamic_rasters, dynamic_names = load_natraster(dynamic_path)
    (
        last_frame_rasters,
        dynamic_rasters,
        _,
        frame_2250_rasters,
        shared_stimuli,
        aligned_names,
    ) = match_timed_static_movie_rasters(
        static_rasters,
        static_names,
        dynamic_rasters,
        dynamic_names,
    )

    if good_channels is not None:
        first_channel, last_channel = good_channels
        if first_channel < 1 or last_channel < first_channel:
            raise ValueError("good_channels must be an increasing one-based range.")
        # end if invalid channel range
        channel_slice = slice(first_channel - 1, last_channel)
        last_frame_rasters = last_frame_rasters[channel_slice]
        frame_2250_rasters = frame_2250_rasters[channel_slice]
        dynamic_rasters = dynamic_rasters[channel_slice]
        channel_numbers = np.arange(first_channel, last_channel + 1)
    else:
        channel_numbers = np.arange(1, dynamic_rasters.shape[0] + 1)
    # end if good_channels

    # Downsample after alignment so all neural conditions share the same time grid.
    last_frame_rasters = resample_array(
        last_frame_rasters, source_fs, new_fs,
    )
    frame_2250_rasters = resample_array(
        frame_2250_rasters, source_fs, new_fs,
    )
    dynamic_rasters = resample_array(dynamic_rasters, source_fs, new_fs)

    if center_neural_trials:
        # Axis 2 is the aligned stimulus/trial axis in all three neural arrays.
        last_frame_rasters = mean_centering(last_frame_rasters, axis=2)
        frame_2250_rasters = mean_centering(frame_2250_rasters, axis=2)
        dynamic_rasters = mean_centering(dynamic_rasters, axis=2)
    # end if center_neural_trials

    requested_visual_names = [
        f"vid_{stimulus_identity}.mp4" for stimulus_identity in shared_stimuli
    ]
    visual_names = match_feature_stimulus_names(
        visual_feature_path, requested_visual_names,
    )
    if visual_feature_type == "pixels":
        visual_features, visual_fs, visual_metadata = (
            load_aligned_pixel_value_features(
                visual_feature_path, visual_names,
            )
        )
        visual_label = f"Pixels (step {visual_metadata['pixel_step']})"
    elif visual_feature_type == "model":
        visual_features, layer_name, source_fps = load_aligned_video_features(
            visual_feature_path, visual_names, frame_index=None,
        )
        with h5py.File(visual_feature_path, "r") as feature_file:
            model_name = str(feature_file.attrs.get("model_name", "Model"))
            model_pooling = str(feature_file.attrs.get("pooling", "unknown"))
            frame_stride = int(feature_file.attrs.get("frame_stride", 1))
        # end with h5py.File
        visual_fs = source_fps / frame_stride
        visual_metadata = {
            "feature_type": "model",
            "model_name": model_name,
            "layer_name": layer_name,
            "pooling": model_pooling,
            "frame_stride": frame_stride,
            "feature_path": str(visual_feature_path),
        }
        visual_label = f"{model_name} {layer_name}"
    else:
        raise ValueError("visual_feature_type must be 'pixels' or 'model'.")
    # end if visual_feature_type

    return {
        "stimuli": np.asarray(shared_stimuli),
        "dynamic": dynamic_rasters,
        "static_last_frame": last_frame_rasters,
        "static_2250ms": frame_2250_rasters,
        "visual_features": visual_features,
        "neural_fs": float(new_fs),
        "visual_fs": float(visual_fs),
        "channel_numbers": channel_numbers,
        "visual_feature_type": visual_feature_type,
        "visual_label": visual_label,
        "visual_metadata": visual_metadata,
        "aligned_names": aligned_names,
        "center_neural_trials": bool(center_neural_trials),
    }
# EOF


"""
window_mean
Average a time window while retaining the feature x stimulus axes.

INPUT:
    - data: np.ndarray -> features x time x stimuli values
    - times_ms: np.ndarray -> time of every sample in milliseconds
    - window_ms: tuple[float, float] -> inclusive averaging window

OUTPUT:
    - mean_vectors: np.ndarray -> features x stimuli window means
"""
def window_mean(data, times_ms, window_ms):
    start_ms, end_ms = window_ms
    window_indices = (times_ms >= start_ms) & (times_ms <= end_ms)
    if not window_indices.any():
        raise ValueError(f"Window {window_ms} contains no samples.")
    # end if no window samples
    return np.mean(data[:, window_indices, :], axis=1)
# EOF


"""
stimuluswise_reference_correlation
Correlate each stimulus timecourse with its matching reference feature vector.

INPUT:
    - timecourses: np.ndarray -> features x time x stimuli values
    - reference_vectors: np.ndarray -> features x stimuli reference values

OUTPUT:
    - correlations: np.ndarray -> time x stimuli Pearson correlations
"""
def stimuluswise_reference_correlation(timecourses, reference_vectors):
    timecourses = np.asarray(timecourses)
    reference_vectors = np.asarray(reference_vectors)
    expected_reference_shape = (timecourses.shape[0], timecourses.shape[2])
    if timecourses.ndim != 3 or reference_vectors.shape != expected_reference_shape:
        raise ValueError(
            "Expected timecourses [features,time,stimuli] and matching "
            "reference_vectors [features,stimuli]."
        )
    # end if input shapes

    correlations = np.full(
        (timecourses.shape[1], timecourses.shape[2]),
        np.nan,
        dtype=np.float64,
    )
    # Process one stimulus at a time so large pixel arrays are never cast to
    # float64 in their entirety.
    for stimulus_index in range(timecourses.shape[2]):
        stimulus_values = np.asarray(
            timecourses[:, :, stimulus_index], dtype=np.float64,
        )
        reference = np.asarray(
            reference_vectors[:, stimulus_index], dtype=np.float64,
        )
        stimulus_values -= stimulus_values.mean(axis=0, keepdims=True)
        reference -= reference.mean()
        numerator = reference @ stimulus_values
        denominator = np.linalg.norm(reference) * np.linalg.norm(
            stimulus_values, axis=0,
        )
        np.divide(
            numerator,
            denominator,
            out=correlations[:, stimulus_index],
            where=denominator > 0,
        )
    # end for stimulus_index
    return np.clip(correlations, -1, 1)
# EOF


def exponential_decay(lag_ms, amplitude, tau_ms, baseline):
    """Evaluate baseline + amplitude * exp(-lag / tau)."""
    return baseline + amplitude * np.exp(-lag_ms / tau_ms)
# EOF


def peak_anchored_exponential_decay(lag_ms, tau_ms, baseline, peak_value):
    """Evaluate an exponential constrained to equal peak_value at zero lag."""
    amplitude = peak_value - baseline
    return exponential_decay(lag_ms, amplitude, tau_ms, baseline)
# EOF


"""
smooth_correlations_through_peaks
Gaussian-smooth each curve only through its selected peak, without using later data.

INPUT:
    - correlations: np.ndarray -> time x stimuli correlation timecourses
    - peak_indices: np.ndarray -> one included peak index per stimulus
    - sigma_samples: float | None -> gaussian_filter1d sigma in samples

OUTPUT:
    - smoothed_correlations: np.ndarray -> curves smoothed only before each peak
"""
def smooth_correlations_through_peaks(
        correlations,
        peak_indices,
        sigma_samples,
        ):
    smoothed_correlations = np.asarray(
        correlations, dtype=np.float64,
    ).copy()
    if sigma_samples is None or sigma_samples <= 0:
        return smoothed_correlations
    # end if smoothing disabled

    for stimulus_index, peak_index in enumerate(peak_indices):
        segment = smoothed_correlations[:peak_index + 1, stimulus_index]
        finite = np.isfinite(segment)
        if finite.sum() < 2:
            continue
        # end if too few finite samples
        if not finite.all():
            sample_indices = np.arange(segment.size)
            segment = np.interp(
                sample_indices, sample_indices[finite], segment[finite],
            )
        # end if missing values
        filtered_segment = gaussian_filter1d(
            segment, sigma=sigma_samples, mode="nearest",
        )
        # Preserve the selected peak exactly while smoothing only its history.
        filtered_segment[-1] = correlations[peak_index, stimulus_index]
        smoothed_correlations[:peak_index + 1, stimulus_index] = (
            filtered_segment
        )
    # end for stimulus_index
    return smoothed_correlations
# EOF


"""
fit_peak_exponential_decays
Fit one backward-from-peak exponential decay independently for every stimulus.

INPUT:
    - correlations: np.ndarray -> time x stimuli correlation timecourses
    - times_ms: np.ndarray -> time of every row in milliseconds
    - peak_indices: np.ndarray -> one included peak index per stimulus
    - lookback_ms: float -> duration preceding each peak included in its fit
    - smoothing_sigma_ms: float | None -> Gaussian sigma, or None for raw curves

OUTPUT:
    - fits: dict -> tau, rate, R-squared, parameters, curves, and fit masks
"""
def fit_peak_exponential_decays(
        correlations,
        times_ms,
        peak_indices,
        lookback_ms,
        smoothing_sigma_ms=None,
        ):
    correlations = np.asarray(correlations, dtype=np.float64)
    times_ms = np.asarray(times_ms, dtype=np.float64)
    peak_indices = np.asarray(peak_indices, dtype=int)
    if correlations.shape != (times_ms.size, peak_indices.size):
        raise ValueError("correlations, times_ms, and peak_indices do not align.")
    # end if input shapes
    if lookback_ms <= 0:
        raise ValueError("lookback_ms must be positive.")
    # end if lookback_ms

    n_stimuli = correlations.shape[1]
    parameters = np.full((n_stimuli, 3), np.nan)
    r_squared = np.full(n_stimuli, np.nan)
    tau_at_bound = np.zeros(n_stimuli, dtype=bool)
    fitted_curves = np.full(correlations.shape, np.nan)
    fit_masks = np.zeros(correlations.shape, dtype=bool)
    sample_interval_ms = np.median(np.diff(times_ms))
    smoothing_sigma_samples = None
    if smoothing_sigma_ms is not None:
        smoothing_sigma_samples = smoothing_sigma_ms / sample_interval_ms
    # end if smoothing_sigma_ms
    smoothed_curves = smooth_correlations_through_peaks(
        correlations, peak_indices, smoothing_sigma_samples,
    )

    for stimulus_index, peak_index in enumerate(peak_indices):
        peak_time_ms = times_ms[peak_index]
        fit_mask = (
            (times_ms >= peak_time_ms - lookback_ms)
            & (times_ms <= peak_time_ms)
            & np.isfinite(smoothed_curves[:, stimulus_index])
        )
        fit_masks[:, stimulus_index] = fit_mask
        if fit_mask.sum() < 4:
            continue
        # end if too few samples

        fit_times_ms = times_ms[fit_mask]
        fit_values = smoothed_curves[fit_mask, stimulus_index]
        lags_ms = peak_time_ms - fit_times_ms
        early_count = max(1, fit_values.size // 5)
        initial_baseline = float(np.median(fit_values[:early_count]))
        initial_tau_ms = max(sample_interval_ms, lookback_ms / 3)
        peak_value = float(smoothed_curves[peak_index, stimulus_index])
        lower_tau_ms = sample_interval_ms / 2
        upper_tau_ms = lookback_ms * 10

        try:
            fitted_parameters, _ = curve_fit(
                lambda lag_ms, tau_ms, baseline: peak_anchored_exponential_decay(
                    lag_ms, tau_ms, baseline, peak_value,
                ),
                lags_ms,
                fit_values,
                p0=(initial_tau_ms, initial_baseline),
                bounds=(
                    (lower_tau_ms, -1),
                    (upper_tau_ms, 1),
                ),
                maxfev=20000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        # end try

        tau_ms, baseline = fitted_parameters
        amplitude = peak_value - baseline
        fitted_values = exponential_decay(
            lags_ms, amplitude, tau_ms, baseline,
        )
        residual_sum_squares = np.sum((fit_values - fitted_values) ** 2)
        total_sum_squares = np.sum((fit_values - fit_values.mean()) ** 2)
        parameters[stimulus_index] = (amplitude, tau_ms, baseline)
        fitted_curves[fit_mask, stimulus_index] = fitted_values
        tau_at_bound[stimulus_index] = (
            np.isclose(tau_ms, lower_tau_ms, rtol=1e-3)
            or np.isclose(tau_ms, upper_tau_ms, rtol=1e-3)
        )
        if total_sum_squares > 0:
            r_squared[stimulus_index] = 1 - residual_sum_squares / total_sum_squares
        # end if total_sum_squares
    # end for stimulus_index

    tau_ms = parameters[:, 1]
    return {
        "parameters": parameters,
        "tau_ms": tau_ms,
        "rate_per_s": 1000 / tau_ms,
        "r_squared": r_squared,
        "tau_at_bound": tau_at_bound,
        "fitted_curves": fitted_curves,
        "smoothed_curves": smoothed_curves,
        "smoothing_sigma_ms": smoothing_sigma_ms,
        "fit_masks": fit_masks,
        "peak_indices": peak_indices,
    }
# EOF


"""
compute_last_frame_decay_measures
Compute the three stimulus-level timecourses, exponential fits, and static-frame
population-vector correlations used by the last-frame association analysis.

INPUT:
    - data: dict -> output of load_last_frame_decay_data
    - last_frame_time_ms: float -> designated final movie frame
    - static_last_window_ms: tuple[float, float] -> last-frame response average
    - static_2250_window_ms: tuple[float, float] -> 2250-ms response average
    - dynamic_peak_window_ms: tuple[float, float] -> allowed similarity peak times
    - decay_lookback_ms: float -> duration preceding each peak fitted exponentially
    - smoothing_sigma_ms: float | None -> pre-peak Gaussian smoothing sigma

OUTPUT:
    - analysis: dict -> timecourses, reference vectors, correlations, and fits
"""
def compute_last_frame_decay_measures(
        data,
        last_frame_time_ms=2500,
        static_last_window_ms=(60, 200),
        static_2250_window_ms=(60, 200),
        dynamic_peak_window_ms=(2500, 3000),
        decay_lookback_ms=1000,
        smoothing_sigma_ms=None,
        ):
    neural_times_ms = (
        np.arange(data["dynamic"].shape[1]) * 1000 / data["neural_fs"]
    )
    static_times_ms = (
        np.arange(data["static_last_frame"].shape[1])
        * 1000 / data["neural_fs"]
    )
    visual_times_ms = (
        np.arange(data["visual_features"].shape[1])
        * 1000 / data["visual_fs"]
    )
    neural_last_index = int(
        np.argmin(np.abs(neural_times_ms - last_frame_time_ms))
    )
    visual_last_index = int(
        np.argmin(np.abs(visual_times_ms - last_frame_time_ms))
    )

    # Both autocorrelations use the matching stimulus's designated final frame.
    visual_acf = stimuluswise_reference_correlation(
        data["visual_features"][:, :visual_last_index + 1, :],
        data["visual_features"][:, visual_last_index, :],
    )
    neural_acf = stimuluswise_reference_correlation(
        data["dynamic"][:, :neural_last_index + 1, :],
        data["dynamic"][:, neural_last_index, :],
    )

    static_last_vectors = window_mean(
        data["static_last_frame"], static_times_ms, static_last_window_ms,
    )
    static_2250_vectors = window_mean(
        data["static_2250ms"], static_times_ms, static_2250_window_ms,
    )
    static_dynamic_similarity = stimuluswise_reference_correlation(
        data["dynamic"], static_last_vectors,
    )

    # End each static-dynamic curve at its own peak inside the chosen window.
    peak_start_ms, peak_end_ms = dynamic_peak_window_ms
    peak_candidates = np.flatnonzero(
        (neural_times_ms >= peak_start_ms)
        & (neural_times_ms <= peak_end_ms)
    )
    if peak_candidates.size == 0:
        raise ValueError("dynamic_peak_window_ms contains no neural samples.")
    # end if no peak candidates
    dynamic_peak_indices = peak_candidates[
        np.nanargmax(static_dynamic_similarity[peak_candidates], axis=0)
    ]

    fixed_visual_peaks = np.full(data["stimuli"].size, visual_last_index)
    fixed_neural_peaks = np.full(data["stimuli"].size, neural_last_index)
    fits = {
        "visual": fit_peak_exponential_decays(
            visual_acf,
            visual_times_ms[:visual_last_index + 1],
            fixed_visual_peaks,
            decay_lookback_ms,
            smoothing_sigma_ms,
        ),
        "neural_acf": fit_peak_exponential_decays(
            neural_acf,
            neural_times_ms[:neural_last_index + 1],
            fixed_neural_peaks,
            decay_lookback_ms,
            smoothing_sigma_ms,
        ),
        "static_dynamic": fit_peak_exponential_decays(
            static_dynamic_similarity,
            neural_times_ms,
            dynamic_peak_indices,
            decay_lookback_ms,
            smoothing_sigma_ms,
        ),
    }

    static_frame_correlation = stimuluswise_reference_correlation(
        static_last_vectors[:, np.newaxis, :], static_2250_vectors,
    )[0]
    return {
        "times_ms": {
            "visual": visual_times_ms[:visual_last_index + 1],
            "neural_acf": neural_times_ms[:neural_last_index + 1],
            "static_dynamic": neural_times_ms,
        },
        "timecourses": {
            "visual": visual_acf,
            "neural_acf": neural_acf,
            "static_dynamic": static_dynamic_similarity,
        },
        "fits": fits,
        "static_last_vectors": static_last_vectors,
        "static_2250_vectors": static_2250_vectors,
        "static_frame_correlation": static_frame_correlation,
        "dynamic_peak_times_ms": neural_times_ms[dynamic_peak_indices],
        "visual_label": data["visual_label"],
        "last_frame_times_ms": {
            "visual": visual_times_ms[visual_last_index],
            "neural": neural_times_ms[neural_last_index],
        },
    }
# EOF


def _scatter_with_spearman(axis, x, y, xlabel, ylabel, stimulus_names=None):
    """Draw one finite-value scatterplot and annotate its Spearman association."""
    finite = np.isfinite(x) & np.isfinite(y)
    axis.scatter(x[finite], y[finite], s=28, alpha=0.75, edgecolor="none")
    if finite.sum() >= 3:
        rho, p_value = spearmanr(x[finite], y[finite])
        annotation = f"Spearman $\\rho$={rho:.2f}, p={p_value:.3g}, n={finite.sum()}"
    else:
        annotation = f"n={finite.sum()}"
    # end if finite.sum()
    axis.text(0.04, 0.96, annotation, transform=axis.transAxes, va="top")
    if stimulus_names is not None:
        for stimulus_name, x_value, y_value, keep in zip(
                stimulus_names, x, y, finite,
                ):
            if keep:
                axis.annotate(
                    stimulus_name,
                    (x_value, y_value),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    alpha=0.7,
                )
            # end if keep
        # end for stimulus_name
    # end if stimulus_names
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.2)
    return finite
# EOF


"""
plot_decay_scatterplots
Plot all three pairwise decay associations plus the static-frame comparison.

INPUT:
    - analysis: dict -> output of compute_last_frame_decay_measures
    - stimulus_names: np.ndarray -> aligned stimulus labels
    - annotate_stimuli: bool -> draw every retained stimulus label
    - min_fit_r_squared: float | None -> optional fit-quality inclusion threshold

OUTPUT:
    - figure: matplotlib.figure.Figure -> four-panel scatterplot figure
    - axes: np.ndarray -> corresponding two-by-two axes
"""
def plot_decay_scatterplots(
        analysis,
        stimulus_names,
        annotate_stimuli=False,
        min_fit_r_squared=None,
        ):
    taus = {
        name: fit["tau_ms"].copy() for name, fit in analysis["fits"].items()
    }
    if min_fit_r_squared is not None:
        for name, fit in analysis["fits"].items():
            taus[name][fit["r_squared"] < min_fit_r_squared] = np.nan
        # end for name
    # end if min_fit_r_squared
    # A bound-hit tau reports the fitting limit, not an identified decay time.
    for name, fit in analysis["fits"].items():
        taus[name][fit["tau_at_bound"]] = np.nan
    # end for name
    labels = stimulus_names if annotate_stimuli else None
    visual_tau_label = f"{analysis['visual_label']} ACF $\\tau$ (ms)"

    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    comparisons = (
        (
            taus["visual"], taus["neural_acf"],
            visual_tau_label, "Neural ACF $\\tau$ (ms)",
        ),
        (
            taus["visual"], taus["static_dynamic"],
            visual_tau_label, "Static–dynamic $\\tau$ (ms)",
        ),
        (
            taus["neural_acf"], taus["static_dynamic"],
            "Neural ACF $\\tau$ (ms)", "Static–dynamic $\\tau$ (ms)",
        ),
        (
            analysis["static_frame_correlation"], taus["static_dynamic"],
            "Static last-frame vs 2250-ms correlation",
            "Static–dynamic $\\tau$ (ms)",
        ),
    )
    for axis, comparison in zip(axes.flat, comparisons):
        _scatter_with_spearman(axis, *comparison, stimulus_names=labels)
    # end for axis
    figure.suptitle("Stimulus-level associations between fitted decays")
    figure.tight_layout()
    return figure, axes
# EOF


"""
plot_decay_fit_examples
Show the observed and fitted decay for one stimulus in each analysis.

INPUT:
    - analysis: dict -> output of compute_last_frame_decay_measures
    - stimulus_names: np.ndarray -> aligned stimulus labels
    - stimulus_index: int -> aligned stimulus chosen for display

OUTPUT:
    - figure: matplotlib.figure.Figure -> three-panel fit diagnostic
    - axes: np.ndarray -> corresponding axes
"""
def plot_decay_fit_examples(analysis, stimulus_names, stimulus_index=0):
    plot_names = {
        "visual": f"{analysis['visual_label']} ACF to last frame",
        "neural_acf": "Neural ACF to last frame",
        "static_dynamic": "Static–dynamic similarity",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, analysis_name in zip(axes, plot_names):
        times_ms = analysis["times_ms"][analysis_name]
        observed = analysis["timecourses"][analysis_name][:, stimulus_index]
        fit_results = analysis["fits"][analysis_name]
        smoothed = fit_results["smoothed_curves"][:, stimulus_index]
        fitted = fit_results["fitted_curves"][:, stimulus_index]
        peak_index = fit_results["peak_indices"][stimulus_index]
        fit_mask = fit_results["fit_masks"][:, stimulus_index]
        axis.plot(times_ms, observed, color="0.7", linewidth=1.4, label="Observed")
        if fit_results["smoothing_sigma_ms"] is not None:
            axis.plot(
                times_ms[fit_mask], smoothed[fit_mask],
                color="tab:blue", linewidth=1.8, label="Pre-peak smoothed",
            )
        # end if smoothing enabled
        axis.plot(
            times_ms[fit_mask], fitted[fit_mask],
            color="tab:red", linewidth=2.2, label="Exponential fit",
        )
        axis.axvline(times_ms[peak_index], color="0.3", linestyle=":")
        tau_ms = analysis["fits"][analysis_name]["tau_ms"][stimulus_index]
        r_squared = analysis["fits"][analysis_name]["r_squared"][stimulus_index]
        axis.set(
            title=f"{plot_names[analysis_name]}\n$\\tau$={tau_ms:.0f} ms, $R^2$={r_squared:.2f}",
            xlabel="Time from movie onset (ms)",
            ylabel="Pearson correlation",
        )
        axis.grid(alpha=0.2)
    # end for axis
    axes[0].legend(frameon=False)
    figure.suptitle(f"Fit check: {stimulus_names[stimulus_index]}")
    figure.tight_layout()
    return figure, axes
# EOF
