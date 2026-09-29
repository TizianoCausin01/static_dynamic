import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, wilcoxon
from useful_stuff.general_utils.utils import mean_centering

from .dataloader import load_natraster, match_timed_static_movie_rasters
from .split_half_rsa import (
    compute_rdm_timeseries,
    cross_temporal_similarity,
    rowwise_similarity,
)
from .last_frame_decay import (
    _scatter_with_spearman,
    resample_array,
    stimuluswise_reference_correlation,
)


"""
half_open_window_mean
Average channels x time x stimuli rasters over the half-open window [start, end) ms.
Bin k averages [k, k + 1) * 1000 / fs ms, so [80, 250) ms at 100 Hz is bins 8..24.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli values
    - fs: float -> sampling frequency of the time axis in Hz
    - window_ms: tuple[float, float] -> half-open averaging window in ms
    - center: bool -> subtract every channel's mean across stimuli after averaging

OUTPUT:
    - window_means: np.ndarray -> channels x stimuli window averages
"""
def half_open_window_mean(rasters, fs, window_ms, center=False):
    # Round to avoid float truncation, e.g. 0.1 * 70 = 7.000000000000001.
    start_index = round(window_ms[0] * fs / 1000)
    end_index = round(window_ms[1] * fs / 1000)
    if end_index <= start_index:
        raise ValueError(f"Window {window_ms} ms contains no samples at {fs} Hz.")
    # end if empty window
    window_means = rasters[:, start_index:end_index, :].mean(axis=1)
    if center:
        # Axis 1 of the channels x stimuli averages is the stimulus axis.
        window_means = mean_centering(window_means, axis=1)
    # end if center
    return window_means
# EOF


"""
load_frame_similarity_rasters
Load and align the static last-frame, 2000-ms, 2250-ms, and movie rasters.

INPUT:
    - static_path: str | Path -> averaged static-condition natraster file
    - dynamic_path: str | Path -> averaged movie-condition natraster file
    - source_fs: float -> neural source sampling frequency in Hz
    - new_fs: float -> neural analysis sampling frequency in Hz
    - good_channels: tuple[int, int] | None -> inclusive one-based channel range
    - reliable_channels_config: str | Path | None -> optional reliability YAML
    - reliable_channels_key: str | None -> dataset key inside the reliability YAML
    - center_neural_trials: bool -> center every channel x time over stimuli

OUTPUT:
    - data: dict -> aligned channels x time x stimuli arrays and metadata
"""
def load_frame_similarity_rasters(
        static_path,
        dynamic_path,
        source_fs=1000,
        new_fs=100,
        good_channels=None,
        reliable_channels_config=None,
        reliable_channels_key=None,
        center_neural_trials=False,
        ):
    # The same channel selection is applied to both sessions so the population
    # vectors of the static and movie conditions share the same feature axis.
    channel_kwargs = {
        "good_channels": good_channels,
        "reliable_channels_config": reliable_channels_config,
        "reliable_channels_key": reliable_channels_key,
    }
    static_rasters, static_names, channel_numbers = load_natraster(
        static_path, return_channel_numbers=True, **channel_kwargs,
    )
    dynamic_rasters, dynamic_names = load_natraster(
        dynamic_path, **channel_kwargs,
    )
    (
        last_frame_rasters,
        dynamic_rasters,
        frame_2000_rasters,
        frame_2250_rasters,
        shared_stimuli,
        aligned_names,
    ) = match_timed_static_movie_rasters(
        static_rasters, static_names, dynamic_rasters, dynamic_names,
    )

    rasters = {
        "dynamic": dynamic_rasters,
        "static_last_frame": last_frame_rasters,
        "static_2000ms": frame_2000_rasters,
        "static_2250ms": frame_2250_rasters,
    }
    for condition, condition_rasters in rasters.items():
        # Downsample after alignment so all conditions share one time grid.
        condition_rasters = resample_array(condition_rasters, source_fs, new_fs)
        if center_neural_trials:
            # Axis 2 is the aligned stimulus axis in every array.
            condition_rasters = mean_centering(condition_rasters, axis=2)
        # end if center_neural_trials
        rasters[condition] = condition_rasters
    # end for condition

    return {
        **rasters,
        "stimuli": np.asarray(shared_stimuli),
        "neural_fs": float(new_fs),
        "channel_numbers": channel_numbers,
        "aligned_names": aligned_names,
        "center_neural_trials": bool(center_neural_trials),
    }
# EOF


"""
similarity_curve_centroids
Estimate the peak time and the centroid of the supra-threshold similarity mass
inside one dynamic time window, independently for every stimulus. The weights
are max(r - threshold_fraction * peak_r, 0), so threshold_fraction=0 gives the
centroid of all positive correlations and larger values focus on the maximum.

INPUT:
    - curves: np.ndarray -> time x stimuli similarity timecourses
    - times_ms: np.ndarray -> time of every row in milliseconds
    - window_ms: tuple[float, float] -> inclusive dynamic search window
    - threshold_fraction: float -> fraction of each peak used as weight floor

OUTPUT:
    - centroid_ms: np.ndarray -> weighted-mean time per stimulus (NaN if peak <= 0)
    - peak_ms: np.ndarray -> time of the within-window maximum per stimulus
    - peak_values: np.ndarray -> within-window maximum correlation per stimulus
"""
def similarity_curve_centroids(
        curves,
        times_ms,
        window_ms,
        threshold_fraction=0.5,
        ):
    window_mask = (times_ms >= window_ms[0]) & (times_ms <= window_ms[1])
    if not window_mask.any():
        raise ValueError(f"Window {window_ms} contains no samples.")
    # end if no window samples
    window_times_ms = times_ms[window_mask]
    window_curves = curves[window_mask]  # window time x stimuli

    peak_indices = np.nanargmax(window_curves, axis=0)
    peak_values = window_curves[peak_indices, np.arange(curves.shape[1])]
    peak_ms = window_times_ms[peak_indices]

    # Weight each sample by how far it rises above a fraction of its own peak.
    thresholds = threshold_fraction * np.clip(peak_values, 0, None)
    weights = np.clip(np.nan_to_num(window_curves - thresholds, nan=0), 0, None)
    weight_sums = weights.sum(axis=0)
    centroid_ms = np.full(curves.shape[1], np.nan)
    np.divide(
        window_times_ms @ weights, weight_sums,
        out=centroid_ms, where=weight_sums > 0,
    )
    return centroid_ms, peak_ms, peak_values
# EOF


"""
compute_frame_similarity_latency
Relate static last-frame vs earlier-frame similarity to the latency of the raw
static-dynamic similarity curve. For each stimulus the static last-frame
response is averaged in static_window_ms and correlated (Pearson, across
channels) with (1) the same window of the static earlier-frame response and
(2) the movie response at every time sample.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - earlier_frame: str -> "2000ms" or "2250ms" static frame condition
    - static_window_ms: tuple[float, float] -> static response averaging window
    - dynamic_window_ms: tuple[float, float] -> centroid search window in the movie
    - threshold_fraction: float -> centroid weight floor as a fraction of the peak
    - smoothing_sigma_ms: float | None -> Gaussian sigma applied before centroids
    - center_averages: bool -> center every window-averaged pattern across stimuli;
      stored in the output so downstream analyses average the same way

OUTPUT:
    - analysis: dict -> similarity curves, frame similarity, centroids, and peaks
"""
def compute_frame_similarity_latency(
        data,
        earlier_frame="2250ms",
        static_window_ms=(80, 250),
        dynamic_window_ms=(2000, 2500),
        threshold_fraction=0.5,
        smoothing_sigma_ms=None,
        center_averages=False,
        ):
    if earlier_frame not in ("2000ms", "2250ms"):
        raise ValueError("earlier_frame must be '2000ms' or '2250ms'.")
    # end if invalid earlier_frame
    dynamic_times_ms = (
        np.arange(data["dynamic"].shape[1]) * 1000 / data["neural_fs"]
    )

    # Population vectors are channels x stimuli half-open window averages.
    last_frame_vectors = half_open_window_mean(
        data["static_last_frame"], data["neural_fs"], static_window_ms,
        center=center_averages,
    )
    earlier_frame_vectors = half_open_window_mean(
        data[f"static_{earlier_frame}"], data["neural_fs"], static_window_ms,
        center=center_averages,
    )
    # A singleton time axis turns the helper into one correlation per stimulus.
    frame_similarity = stimuluswise_reference_correlation(
        earlier_frame_vectors[:, np.newaxis, :], last_frame_vectors,
    )[0]

    # Raw lagged correlation: the fixed last-frame pattern of each stimulus is
    # correlated with that same stimulus's movie response at every time sample.
    similarity_curves = stimuluswise_reference_correlation(
        data["dynamic"], last_frame_vectors,
    )  # dynamic time x stimuli

    centroid_curves = similarity_curves
    if smoothing_sigma_ms is not None and smoothing_sigma_ms > 0:
        sigma_samples = smoothing_sigma_ms * data["neural_fs"] / 1000
        centroid_curves = gaussian_filter1d(
            similarity_curves, sigma=sigma_samples, axis=0, mode="nearest",
        )
    # end if smoothing
    centroid_ms, peak_ms, peak_values = similarity_curve_centroids(
        centroid_curves, dynamic_times_ms, dynamic_window_ms, threshold_fraction,
    )
    return {
        "dynamic_times_ms": dynamic_times_ms,
        "similarity_curves": similarity_curves,
        "centroid_curves": centroid_curves,
        "frame_similarity": frame_similarity,
        "last_frame_vectors": last_frame_vectors,
        "centroid_ms": centroid_ms,
        "peak_ms": peak_ms,
        "peak_values": peak_values,
        "earlier_frame": earlier_frame,
        "static_window_ms": static_window_ms,
        "dynamic_window_ms": dynamic_window_ms,
        "threshold_fraction": threshold_fraction,
        "center_averages": center_averages,
    }
# EOF



"""
compute_frame_similarity_rsa_latency
RSA version of the stimulus-level latency analysis, in the spirit of static
dRSA but with one window-averaged static RDM instead of a static time axis.
The static last-frame RDM (window average) is compared with the movie RDM at
every movie time sample only. Stimulus-wise curves correlate the stimulus's own
RDM row (its dissimilarities to all other stimuli, diagonal excluded) between
the static window and each movie time; the population curve correlates the
full vectorized RDMs.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - analysis: dict -> output of compute_frame_similarity_latency (supplies
      the last-frame window vectors, frame similarity, and window settings)
    - rdm_metric: str -> distance metric accepted by create_RDM
    - smoothing_sigma_ms: float | None -> Gaussian sigma applied before centroids

OUTPUT:
    - rsa_analysis: dict -> stimulus-wise and population RSA curves, centroids,
      and peaks, with the keys used by plot_frame_similarity_latency
"""
def compute_frame_similarity_rsa_latency(
        data,
        analysis,
        rdm_metric="cosine_cnt",
        smoothing_sigma_ms=None,
        ):
    n_stimuli = data["stimuli"].size
    # Static RDM from the window-averaged last-frame patterns: stimuli x stimuli.
    static_rdm_vector = compute_rdm_timeseries(
        analysis["last_frame_vectors"][:, np.newaxis, :], rdm_metric,
    )  # 1 x stimulus pairs
    static_rdm = squareform(static_rdm_vector[0])
    # Movie RDMs at every time sample: time x stimulus pairs.
    dynamic_rdm_vectors = compute_rdm_timeseries(data["dynamic"], rdm_metric)

    # Each stimulus row without its own (zero) self-distance: stimuli x stimuli-1.
    off_diagonal = ~np.eye(n_stimuli, dtype=bool)
    static_rows = static_rdm[off_diagonal].reshape(n_stimuli, n_stimuli - 1)
    stimulus_curves = np.full((dynamic_rdm_vectors.shape[0], n_stimuli), np.nan)
    for time_index, dynamic_rdm_vector in enumerate(dynamic_rdm_vectors):
        dynamic_rows = squareform(dynamic_rdm_vector)[off_diagonal].reshape(
            n_stimuli, n_stimuli - 1,
        )
        stimulus_curves[time_index] = rowwise_similarity(static_rows, dynamic_rows)
    # end for time_index

    # Population dRSA curve: full static RDM vs movie RDM at every time sample.
    population_curve = cross_temporal_similarity(
        dynamic_rdm_vectors, static_rdm_vector,
    )[:, 0]

    centroid_curves = stimulus_curves
    centroid_population = population_curve
    if smoothing_sigma_ms is not None and smoothing_sigma_ms > 0:
        sigma_samples = smoothing_sigma_ms * data["neural_fs"] / 1000
        centroid_curves = gaussian_filter1d(
            stimulus_curves, sigma=sigma_samples, axis=0, mode="nearest",
        )
        centroid_population = gaussian_filter1d(
            population_curve, sigma=sigma_samples, mode="nearest",
        )
    # end if smoothing
    times_ms = analysis["dynamic_times_ms"]
    centroid_ms, peak_ms, peak_values = similarity_curve_centroids(
        centroid_curves, times_ms, analysis["dynamic_window_ms"],
        analysis["threshold_fraction"],
    )
    population_centroid, population_peak, population_peak_value = (
        similarity_curve_centroids(
            centroid_population[:, np.newaxis], times_ms,
            analysis["dynamic_window_ms"], analysis["threshold_fraction"],
        )
    )
    return {
        "dynamic_times_ms": times_ms,
        "similarity_curves": stimulus_curves,
        "centroid_curves": centroid_curves,
        "frame_similarity": analysis["frame_similarity"],
        "centroid_ms": centroid_ms,
        "peak_ms": peak_ms,
        "peak_values": peak_values,
        "population_curve": population_curve,
        "population_centroid_curve": centroid_population,
        "population_centroid_ms": population_centroid[0],
        "population_peak_ms": population_peak[0],
        "population_peak_value": population_peak_value[0],
        "rdm_metric": rdm_metric,
        "earlier_frame": analysis["earlier_frame"],
        "static_window_ms": analysis["static_window_ms"],
        "dynamic_window_ms": analysis["dynamic_window_ms"],
        "threshold_fraction": analysis["threshold_fraction"],
    }
# EOF


"""
plot_population_rsa_curve
Plot the population static-window vs movie RSA curve with its centroid and peak.

INPUT:
    - rsa_analysis: dict -> output of compute_frame_similarity_rsa_latency
    - last_frame_time_ms: float -> movie time at which the last frame appears

OUTPUT:
    - figure: matplotlib.figure.Figure -> one-panel figure
    - axis: matplotlib.axes.Axes -> corresponding axis
"""
def plot_population_rsa_curve(rsa_analysis, last_frame_time_ms=2500):
    times_ms = rsa_analysis["dynamic_times_ms"]
    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(
        times_ms, rsa_analysis["population_curve"],
        color="0.6", linewidth=1.2, label="Raw",
    )
    if rsa_analysis["population_centroid_curve"] is not rsa_analysis["population_curve"]:
        axis.plot(
            times_ms, rsa_analysis["population_centroid_curve"],
            color="tab:blue", linewidth=1.8, label="Smoothed",
        )
    # end if smoothed curve exists
    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.axvspan(*rsa_analysis["dynamic_window_ms"], color="tab:orange", alpha=0.12)
    axis.axvline(last_frame_time_ms, color="0.2", linestyle="--", label="Last frame")
    axis.axvline(
        rsa_analysis["population_centroid_ms"], color="tab:green", linewidth=2,
        label=f"Centroid {rsa_analysis['population_centroid_ms']:.0f} ms",
    )
    axis.axvline(
        rsa_analysis["population_peak_ms"], color="tab:green", linestyle=":",
        label=f"Peak {rsa_analysis['population_peak_ms']:.0f} ms",
    )
    axis.set(
        xlabel="Time from movie onset (ms)",
        ylabel="RDM correlation",
        title=(
            f"Static last-frame RDM {rsa_analysis['static_window_ms']} ms vs "
            f"movie RDM(t) ({rsa_analysis['rdm_metric']})"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure, axis
# EOF


"""
plot_centroid_example
Show one stimulus's static-dynamic similarity curve with its centroid and peak.

INPUT:
    - analysis: dict -> output of compute_frame_similarity_latency
    - stimulus_names: np.ndarray -> aligned stimulus labels
    - stimulus_index: int -> aligned stimulus chosen for display
    - last_frame_time_ms: float -> movie time at which the last frame appears

OUTPUT:
    - figure: matplotlib.figure.Figure -> one-panel diagnostic
    - axis: matplotlib.axes.Axes -> corresponding axis
"""
def plot_centroid_example(
        analysis, stimulus_names, stimulus_index=0, last_frame_time_ms=2500,
        ):
    times_ms = analysis["dynamic_times_ms"]
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(
        times_ms, analysis["similarity_curves"][:, stimulus_index],
        color="0.6", linewidth=1.2, label="Raw",
    )
    if analysis["centroid_curves"] is not analysis["similarity_curves"]:
        axis.plot(
            times_ms, analysis["centroid_curves"][:, stimulus_index],
            color="tab:blue", linewidth=1.8, label="Smoothed",
        )
    # end if smoothed curves exist
    axis.axvspan(*analysis["dynamic_window_ms"], color="tab:orange", alpha=0.12)
    axis.axvline(
        analysis["centroid_ms"][stimulus_index], color="tab:red",
        linewidth=2, label="Centroid",
    )
    axis.axvline(
        analysis["peak_ms"][stimulus_index], color="tab:red",
        linestyle=":", label="Peak",
    )
    axis.axvline(last_frame_time_ms, color="0.2", linestyle="--", label="Last frame")
    axis.set(
        title=(
            f"{stimulus_names[stimulus_index]} | last vs "
            f"{analysis['earlier_frame']} r="
            f"{analysis['frame_similarity'][stimulus_index]:.2f}"
        ),
        xlabel="Time from movie onset (ms)",
        ylabel="r (static last frame, movie)",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure, axis
# EOF


"""
plot_frame_similarity_latency
Plot curves sorted by frame similarity and the stimulus-level associations of
frame similarity with the centroid and peak latencies.

INPUT:
    - analysis: dict -> output of compute_frame_similarity_latency
    - stimulus_names: np.ndarray -> aligned stimulus labels
    - annotate_stimuli: bool -> draw every retained stimulus label
    - display_window_ms: tuple[float, float] -> time range shown in the heatmap
    - heatmap_title: str -> title of the sorted-curve heatmap

OUTPUT:
    - figure: matplotlib.figure.Figure -> three-panel figure
    - axes: np.ndarray -> corresponding axes
"""
def plot_frame_similarity_latency(
        analysis,
        stimulus_names,
        annotate_stimuli=False,
        display_window_ms=(1500, 3000),
        heatmap_title="Static last-frame vs movie similarity",
        ):
    times_ms = analysis["dynamic_times_ms"]
    frame_similarity = analysis["frame_similarity"]
    frame_label = f"Static last vs {analysis['earlier_frame']} frame r"
    labels = stimulus_names if annotate_stimuli else None

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Rows sorted from the least to the most similar pair of static frames.
    order = np.argsort(frame_similarity)
    shown = (times_ms >= display_window_ms[0]) & (times_ms <= display_window_ms[1])
    image = axes[0].imshow(
        analysis["centroid_curves"][shown][:, order].T,
        aspect="auto", origin="lower", cmap="viridis",
        extent=(times_ms[shown][0], times_ms[shown][-1], 0, order.size),
    )
    axes[0].scatter(
        analysis["centroid_ms"][order], np.arange(order.size) + 0.5,
        s=6, color="tab:red", label="Centroid",
    )
    for edge_ms in analysis["dynamic_window_ms"]:
        axes[0].axvline(edge_ms, color="white", linestyle="--", linewidth=1)
    # end for edge_ms
    axes[0].set(
        xlabel="Time from movie onset (ms)",
        ylabel=f"Stimuli (sorted by {frame_label})",
        title=heatmap_title,
    )
    figure.colorbar(image, ax=axes[0], label="Pearson r")

    _scatter_with_spearman(
        axes[1], frame_similarity, analysis["centroid_ms"],
        frame_label, "Similarity centroid (ms)", stimulus_names=labels,
    )
    _scatter_with_spearman(
        axes[2], frame_similarity, analysis["peak_ms"],
        frame_label, "Similarity peak (ms)", stimulus_names=labels,
    )
    figure.suptitle(
        f"Static window {analysis['static_window_ms']} ms | dynamic window "
        f"{analysis['dynamic_window_ms']} ms"
    )
    figure.tight_layout()
    return figure, axes
# EOF


"""
frame_similarity_timecourse_association
Correlate, across stimuli, the static frame similarity with the static-dynamic
similarity at every movie time sample, avoiding any centroid or peak estimate.

INPUT:
    - frame_similarity: np.ndarray -> one static last vs earlier frame r per stimulus
    - similarity_curves: np.ndarray -> time x stimuli static-dynamic similarity

OUTPUT:
    - rho: np.ndarray -> Spearman correlation over stimuli at every time sample
    - p_values: np.ndarray -> matching uncorrected p-values
"""
def frame_similarity_timecourse_association(frame_similarity, similarity_curves):
    rho = np.full(similarity_curves.shape[0], np.nan)
    p_values = np.full(similarity_curves.shape[0], np.nan)
    for time_index, curve_values in enumerate(similarity_curves):
        # Only stimuli with a finite value at both measures enter the test.
        finite = np.isfinite(frame_similarity) & np.isfinite(curve_values)
        if finite.sum() >= 3:
            rho[time_index], p_values[time_index] = spearmanr(
                frame_similarity[finite], curve_values[finite],
            )
        # end if enough stimuli
    # end for time_index
    return rho, p_values
# EOF


"""
plot_timecourse_association
Plot the across-stimulus association between frame similarity and the
static-dynamic similarity at every movie time sample.

INPUT:
    - times_ms: np.ndarray -> movie time of every sample in milliseconds
    - rho: np.ndarray -> Spearman correlation at every sample
    - p_values: np.ndarray -> matching uncorrected p-values
    - earlier_frame: str -> label of the static frame compared with the last frame
    - dynamic_window_ms: tuple[float, float] -> window shaded for reference
    - last_frame_time_ms: float -> movie time at which the last frame appears
    - alpha: float -> uncorrected significance level marked on the curve
    - centroid_ms: float | None -> optional centroid of rho to mark
    - peak_ms: float | None -> optional peak time of rho to mark

OUTPUT:
    - figure: matplotlib.figure.Figure -> one-panel figure
    - axis: matplotlib.axes.Axes -> corresponding axis
"""
def plot_timecourse_association(
        times_ms,
        rho,
        p_values,
        earlier_frame="2250ms",
        dynamic_window_ms=(2000, 2500),
        last_frame_time_ms=2500,
        alpha=0.05,
        centroid_ms=None,
        peak_ms=None,
        ):
    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(times_ms, rho, color="tab:blue", linewidth=1.5)
    significant = p_values < alpha
    axis.scatter(
        times_ms[significant], rho[significant], s=10, color="tab:red",
        zorder=3, label=f"p < {alpha} (uncorrected)",
    )
    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.axvspan(*dynamic_window_ms, color="tab:orange", alpha=0.12)
    axis.axvline(last_frame_time_ms, color="0.2", linestyle="--", label="Last frame")
    if centroid_ms is not None:
        axis.axvline(
            centroid_ms, color="tab:green", linewidth=2,
            label=f"Centroid {centroid_ms:.0f} ms",
        )
    # end if centroid_ms
    if peak_ms is not None:
        axis.axvline(
            peak_ms, color="tab:green", linestyle=":",
            label=f"Peak {peak_ms:.0f} ms",
        )
    # end if peak_ms
    axis.set(
        xlabel="Time from movie onset (ms)",
        ylabel="Spearman $\\rho$ over stimuli",
        title=(
            f"Static last vs {earlier_frame} frame r  ~  "
            "static last-frame vs movie r(t)"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure, axis
# EOF


"""
compute_lagged_max_latency
Full lagged comparison per stimulus: correlate (across channels) the static
last-frame pattern at every static time with the same stimulus's movie pattern
at every movie time, then keep the maximum over static time. This yields one
lagged similarity curve per stimulus over movie time, used for centroids and
peaks exactly as in compute_frame_similarity_latency.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - analysis: dict -> output of compute_frame_similarity_latency (supplies
      frame similarity and the window settings)
    - static_search_ms: tuple[float, float] -> half-open static times entering the max
    - smoothing_sigma_ms: float | None -> Gaussian sigma applied before centroids

OUTPUT:
    - lagged_analysis: dict -> max curves, best static lags, mean cross-temporal
      matrix, centroids, and peaks, with the keys used by plot_frame_similarity_latency
"""
def compute_lagged_max_latency(
        data,
        analysis,
        static_search_ms=(50, 400),
        smoothing_sigma_ms=None,
        ):
    fs = data["neural_fs"]
    start_index = round(static_search_ms[0] * fs / 1000)
    end_index = round(static_search_ms[1] * fs / 1000)
    static_rasters = data["static_last_frame"][:, start_index:end_index, :]
    static_times_ms = np.arange(start_index, end_index) * 1000 / fs
    n_static_times = static_rasters.shape[1]
    n_dynamic_times, n_stimuli = data["dynamic"].shape[1:]
    if n_static_times == 0:
        raise ValueError(f"static_search_ms {static_search_ms} contains no samples.")
    # end if empty static search

    max_curves = np.full((n_dynamic_times, n_stimuli), np.nan)  # movie time x stimuli
    best_static_ms = np.full((n_dynamic_times, n_stimuli), np.nan)
    matrix_sum = np.zeros((n_static_times, n_dynamic_times))
    for stimulus_index in range(n_stimuli):
        # Center and unit-normalize every pattern over channels, so the matrix
        # product below is the Pearson correlation of every time pair.
        static_patterns = np.asarray(static_rasters[:, :, stimulus_index], dtype=np.float64)
        movie_patterns = np.asarray(data["dynamic"][:, :, stimulus_index], dtype=np.float64)
        static_patterns = static_patterns - static_patterns.mean(axis=0)
        movie_patterns = movie_patterns - movie_patterns.mean(axis=0)
        static_patterns /= np.linalg.norm(static_patterns, axis=0)
        movie_patterns /= np.linalg.norm(movie_patterns, axis=0)
        cross_temporal = static_patterns.T @ movie_patterns  # static time x movie time

        matrix_sum += cross_temporal
        best_indices = np.argmax(cross_temporal, axis=0)
        max_curves[:, stimulus_index] = cross_temporal[best_indices, np.arange(n_dynamic_times)]
        best_static_ms[:, stimulus_index] = static_times_ms[best_indices]
    # end for stimulus_index

    centroid_curves = max_curves
    if smoothing_sigma_ms is not None and smoothing_sigma_ms > 0:
        centroid_curves = gaussian_filter1d(
            max_curves, sigma=smoothing_sigma_ms * fs / 1000, axis=0, mode="nearest",
        )
    # end if smoothing
    centroid_ms, peak_ms, peak_values = similarity_curve_centroids(
        centroid_curves, analysis["dynamic_times_ms"],
        analysis["dynamic_window_ms"], analysis["threshold_fraction"],
    )
    return {
        "dynamic_times_ms": analysis["dynamic_times_ms"],
        "static_times_ms": static_times_ms,
        "mean_cross_temporal": matrix_sum / n_stimuli,
        "similarity_curves": max_curves,
        "centroid_curves": centroid_curves,
        "best_static_ms": best_static_ms,
        "frame_similarity": analysis["frame_similarity"],
        "centroid_ms": centroid_ms,
        "peak_ms": peak_ms,
        "peak_values": peak_values,
        "earlier_frame": analysis["earlier_frame"],
        "static_window_ms": static_search_ms,
        "dynamic_window_ms": analysis["dynamic_window_ms"],
        "threshold_fraction": analysis["threshold_fraction"],
    }
# EOF


"""
plot_mean_cross_temporal
Show the stimulus-averaged static-time x movie-time correlation matrix.

INPUT:
    - lagged_analysis: dict -> output of compute_lagged_max_latency
    - last_frame_time_ms: float -> movie time at which the last frame appears

OUTPUT:
    - figure: matplotlib.figure.Figure -> one-panel figure
    - axis: matplotlib.axes.Axes -> corresponding axis
"""
def plot_mean_cross_temporal(lagged_analysis, last_frame_time_ms=2500):
    static_times_ms = lagged_analysis["static_times_ms"]
    dynamic_times_ms = lagged_analysis["dynamic_times_ms"]
    figure, axis = plt.subplots(figsize=(10, 3.8))
    image = axis.imshow(
        lagged_analysis["mean_cross_temporal"], aspect="auto", origin="lower",
        cmap="viridis",
        extent=(
            dynamic_times_ms[0], dynamic_times_ms[-1],
            static_times_ms[0], static_times_ms[-1],
        ),
    )
    axis.axvline(last_frame_time_ms, color="white", linestyle="--", linewidth=1)
    for edge_ms in lagged_analysis["dynamic_window_ms"]:
        axis.axvline(edge_ms, color="tab:orange", linestyle=":", linewidth=1)
    # end for edge_ms
    axis.set(
        xlabel="Time from movie onset (ms)",
        ylabel="Static last-frame time (ms)",
        title="Mean over stimuli of the per-stimulus static x movie correlation",
    )
    figure.colorbar(image, ax=axis, label="Pearson r")
    figure.tight_layout()
    return figure, axis
# EOF


"""
compute_anticipation_partial
Test whether the movie response resembles the static last-frame response beyond
what the static earlier-frame response explains. For every stimulus and movie
time, the movie pattern is correlated (across channels) with the last-frame
pattern, with the earlier-frame pattern, and with the last-frame pattern after
partialling out the earlier-frame pattern:
    r_partial = (r_ML - r_MP * r_LP) / sqrt((1 - r_MP^2) * (1 - r_LP^2)).
The window-mean partial correlation is compared, paired over stimuli, with the
same quantity in a baseline window.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - analysis: dict -> output of compute_frame_similarity_latency
    - test_window_ms: tuple[float, float] -> half-open movie window tested
    - baseline_window_ms: tuple[float, float] -> half-open movie reference window

OUTPUT:
    - anticipation: dict -> time x stimuli curves, per-stimulus window means,
      and the paired Wilcoxon test
"""
def compute_anticipation_partial(
        data,
        analysis,
        test_window_ms=(2300, 2550),
        baseline_window_ms=(1000, 1500),
        ):
    earlier_frame_vectors = half_open_window_mean(
        data[f"static_{analysis['earlier_frame']}"], data["neural_fs"],
        analysis["static_window_ms"], center=analysis["center_averages"],
    )
    last_frame_vectors = analysis["last_frame_vectors"]
    # Every r_* curve is movie time x stimuli.
    r_movie_last = stimuluswise_reference_correlation(data["dynamic"], last_frame_vectors)
    r_movie_earlier = stimuluswise_reference_correlation(data["dynamic"], earlier_frame_vectors)
    r_last_earlier = analysis["frame_similarity"][np.newaxis, :]  # 1 x stimuli

    denominators = np.sqrt((1 - r_movie_earlier ** 2) * (1 - r_last_earlier ** 2))
    partial_curves = np.full(r_movie_last.shape, np.nan)
    np.divide(
        r_movie_last - r_movie_earlier * r_last_earlier, denominators,
        out=partial_curves, where=denominators > 0,
    )

    # Per-stimulus window means use the same half-open bins as the static window;
    # a leading singleton axis makes the time x stimuli curves fit the helper.
    test_partial = half_open_window_mean(
        partial_curves[np.newaxis], data["neural_fs"], test_window_ms,
    )[0]
    baseline_partial = half_open_window_mean(
        partial_curves[np.newaxis], data["neural_fs"], baseline_window_ms,
    )[0]
    statistic, p_value = wilcoxon(test_partial, baseline_partial, nan_policy="omit")
    return {
        "dynamic_times_ms": analysis["dynamic_times_ms"],
        "r_movie_last": r_movie_last,
        "r_movie_earlier": r_movie_earlier,
        "partial_curves": partial_curves,
        "test_partial": test_partial,
        "baseline_partial": baseline_partial,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
        "earlier_frame": analysis["earlier_frame"],
        "test_window_ms": test_window_ms,
        "baseline_window_ms": baseline_window_ms,
    }
# EOF


"""
plot_anticipation_partial
Plot mean (+/- SEM over stimuli) movie similarity to the last frame, to the
earlier frame, and to the last frame after partialling out the earlier frame,
plus the paired per-stimulus comparison of the test and baseline windows.

INPUT:
    - anticipation: dict -> output of compute_anticipation_partial
    - last_frame_time_ms: float -> movie time at which the last frame appears

OUTPUT:
    - figure: matplotlib.figure.Figure -> two-panel figure
    - axes: np.ndarray -> corresponding axes
"""
def plot_anticipation_partial(anticipation, last_frame_time_ms=2500):
    times_ms = anticipation["dynamic_times_ms"]
    earlier_frame = anticipation["earlier_frame"]
    figure, axes = plt.subplots(
        1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": (3, 1)},
    )
    curve_specs = (
        ("r_movie_last", "Movie vs last frame", "tab:blue"),
        ("r_movie_earlier", f"Movie vs {earlier_frame} frame", "tab:purple"),
        ("partial_curves", f"Movie vs last | {earlier_frame} (partial)", "tab:red"),
    )
    for key, label, color in curve_specs:
        curves = anticipation[key]
        mean_curve = np.nanmean(curves, axis=1)
        sem_curve = np.nanstd(curves, axis=1) / np.sqrt(np.sum(np.isfinite(curves), axis=1))
        axes[0].plot(times_ms, mean_curve, color=color, label=label)
        axes[0].fill_between(
            times_ms, mean_curve - sem_curve, mean_curve + sem_curve,
            color=color, alpha=0.2, linewidth=0,
        )
    # end for key
    axes[0].axvspan(*anticipation["test_window_ms"], color="tab:orange", alpha=0.15)
    axes[0].axvspan(*anticipation["baseline_window_ms"], color="0.5", alpha=0.12)
    axes[0].axvline(last_frame_time_ms, color="0.2", linestyle="--")
    axes[0].axhline(0, color="0.3", linewidth=0.8)
    axes[0].set(
        xlabel="Time from movie onset (ms)",
        ylabel="Pearson r across channels",
        title="Mean over stimuli (+/- SEM); orange = test, grey = baseline",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    # Paired per-stimulus window means: one line per stimulus.
    paired = np.vstack((anticipation["baseline_partial"], anticipation["test_partial"]))
    axes[1].plot([0, 1], paired, color="0.6", alpha=0.3, linewidth=0.8)
    axes[1].plot([0, 1], np.nanmean(paired, axis=1), color="tab:red", linewidth=2.5)
    axes[1].axhline(0, color="0.3", linewidth=0.8)
    axes[1].set(
        xticks=(0, 1), xticklabels=("Baseline", "Test"), xlim=(-0.3, 1.3),
        ylabel="Window-mean partial r",
        title=f"Paired Wilcoxon p={anticipation['wilcoxon_p']:.3g}",
    )
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return figure, axes
# EOF


"""
compute_dynamic_autocorrelation_latency
Movie-only autocorrelation latency per stimulus. The movie response averaged in
reference_window_ms is the reference pattern; it is correlated (across
channels) with the same stimulus's movie pattern at every time sample, and the
centroid and peak are taken in centroid_window_ms, which must end before the
reference window starts so the curve's trivial self-similarity is excluded.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - analysis: dict -> output of compute_frame_similarity_latency (supplies
      frame similarity and the centroid threshold)
    - reference_window_ms: tuple[float, float] -> half-open movie reference window
    - centroid_window_ms: tuple[float, float] -> inclusive earlier movie window
    - smoothing_sigma_ms: float | None -> Gaussian sigma applied before centroids

OUTPUT:
    - autocorrelation_analysis: dict -> time x stimuli curves, centroids, and
      peaks, with the keys used by plot_frame_similarity_latency
"""
def compute_dynamic_autocorrelation_latency(
        data,
        analysis,
        reference_window_ms=(2500, 2600),
        centroid_window_ms=(1500, 2500),
        smoothing_sigma_ms=None,
        ):
    if centroid_window_ms[1] > reference_window_ms[0]:
        raise ValueError("centroid_window_ms must end before reference_window_ms starts.")
    # end if windows overlap
    # Reference movie patterns: channels x stimuli.
    reference_vectors = half_open_window_mean(
        data["dynamic"], data["neural_fs"], reference_window_ms,
        center=analysis["center_averages"],
    )
    # Movie time x stimuli correlation with each stimulus's own reference.
    autocorrelation_curves = stimuluswise_reference_correlation(
        data["dynamic"], reference_vectors,
    )

    centroid_curves = autocorrelation_curves
    if smoothing_sigma_ms is not None and smoothing_sigma_ms > 0:
        centroid_curves = gaussian_filter1d(
            autocorrelation_curves, sigma=smoothing_sigma_ms * data["neural_fs"] / 1000,
            axis=0, mode="nearest",
        )
    # end if smoothing
    centroid_ms, peak_ms, peak_values = similarity_curve_centroids(
        centroid_curves, analysis["dynamic_times_ms"],
        centroid_window_ms, analysis["threshold_fraction"],
    )
    return {
        "dynamic_times_ms": analysis["dynamic_times_ms"],
        "similarity_curves": autocorrelation_curves,
        "centroid_curves": centroid_curves,
        "frame_similarity": analysis["frame_similarity"],
        "centroid_ms": centroid_ms,
        "peak_ms": peak_ms,
        "peak_values": peak_values,
        "earlier_frame": analysis["earlier_frame"],
        "static_window_ms": analysis["static_window_ms"],
        "dynamic_window_ms": centroid_window_ms,
        "reference_window_ms": reference_window_ms,
        "threshold_fraction": analysis["threshold_fraction"],
    }
# EOF


"""
leave_one_out_ridge_predictions
Predict every stimulus's target pattern from its feature pattern with a ridge
regression (full cross-channel weight matrix plus intercept) fitted on all
other stimuli, so no stimulus contributes to its own prediction.

INPUT:
    - features: np.ndarray -> stimuli x input channels predictor patterns
    - targets: np.ndarray -> stimuli x output channels target patterns
    - ridge_alpha: float -> penalty relative to the mean feature variance

OUTPUT:
    - predictions: np.ndarray -> stimuli x output channels held-out predictions
"""
def leave_one_out_ridge_predictions(features, targets, ridge_alpha=1.0):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n_stimuli, n_features = features.shape
    predictions = np.full(targets.shape, np.nan)
    for held_out in range(n_stimuli):
        training = np.arange(n_stimuli) != held_out
        # Center on the training stimuli only, so the intercept is not leaked.
        feature_means = features[training].mean(axis=0)
        target_means = targets[training].mean(axis=0)
        training_features = features[training] - feature_means
        training_targets = targets[training] - target_means
        gram = training_features.T @ training_features  # features x features
        # Scale the penalty with the data so ridge_alpha is unit-free.
        penalty = ridge_alpha * np.trace(gram) / n_features
        weights = np.linalg.solve(
            gram + penalty * np.eye(n_features),
            training_features.T @ training_targets,
        )  # input channels x output channels: cross-channel mapping
        predictions[held_out] = (
            (features[held_out] - feature_means) @ weights + target_means
        )
    # end for held_out
    return predictions
# EOF


"""
compute_regression_fit_latency
Cross-channel regression version of the latency analysis. The static
last-frame pattern of each stimulus is predicted (leave-one-stimulus-out ridge)
from (1) its static earlier-frame pattern and (2) its movie pattern at every
movie time, with one regression per time. Each prediction is scored by its
Pearson correlation (across channels) with the stimulus's actual last-frame
pattern; the static score replaces frame similarity on the x-axis.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - analysis: dict -> output of compute_frame_similarity_latency
    - ridge_alpha: float -> relative ridge penalty
    - centroid_window_ms: tuple[float, float] -> inclusive movie centroid window
    - smoothing_sigma_ms: float | None -> Gaussian sigma applied before centroids

OUTPUT:
    - regression_analysis: dict -> static fit, time x stimuli fit curves,
      centroids, and peaks, with the keys used by plot_frame_similarity_latency
"""
def compute_regression_fit_latency(
        data,
        analysis,
        ridge_alpha=1.0,
        centroid_window_ms=(2000, 2500),
        smoothing_sigma_ms=None,
        ):
    # Patterns are transposed to stimuli x channels for the regression.
    last_frame_patterns = analysis["last_frame_vectors"].T
    earlier_frame_patterns = half_open_window_mean(
        data[f"static_{analysis['earlier_frame']}"], data["neural_fs"],
        analysis["static_window_ms"], center=analysis["center_averages"],
    ).T

    static_predictions = leave_one_out_ridge_predictions(
        earlier_frame_patterns, last_frame_patterns, ridge_alpha,
    )
    static_fit = rowwise_similarity(static_predictions, last_frame_patterns)

    n_times = data["dynamic"].shape[1]
    fit_curves = np.full((n_times, last_frame_patterns.shape[0]), np.nan)  # time x stimuli
    for time_index in range(n_times):
        movie_patterns = data["dynamic"][:, time_index, :].T  # stimuli x channels
        movie_predictions = leave_one_out_ridge_predictions(
            movie_patterns, last_frame_patterns, ridge_alpha,
        )
        fit_curves[time_index] = rowwise_similarity(movie_predictions, last_frame_patterns)
    # end for time_index

    centroid_curves = fit_curves
    if smoothing_sigma_ms is not None and smoothing_sigma_ms > 0:
        centroid_curves = gaussian_filter1d(
            fit_curves, sigma=smoothing_sigma_ms * data["neural_fs"] / 1000,
            axis=0, mode="nearest",
        )
    # end if smoothing
    centroid_ms, peak_ms, peak_values = similarity_curve_centroids(
        centroid_curves, analysis["dynamic_times_ms"],
        centroid_window_ms, analysis["threshold_fraction"],
    )
    return {
        "dynamic_times_ms": analysis["dynamic_times_ms"],
        "similarity_curves": fit_curves,
        "centroid_curves": centroid_curves,
        # The static regression fit takes the place of the raw frame correlation.
        "frame_similarity": static_fit,
        "raw_frame_similarity": analysis["frame_similarity"],
        "centroid_ms": centroid_ms,
        "peak_ms": peak_ms,
        "peak_values": peak_values,
        "earlier_frame": analysis["earlier_frame"],
        "static_window_ms": analysis["static_window_ms"],
        "dynamic_window_ms": centroid_window_ms,
        "threshold_fraction": analysis["threshold_fraction"],
        "ridge_alpha": ridge_alpha,
    }
# EOF


"""
dynamic_window_similarity
Correlate, per stimulus and across channels, the movie patterns averaged in two
half-open movie windows.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - first_window_ms: tuple[float, float] -> earlier half-open movie window
    - second_window_ms: tuple[float, float] -> later half-open movie window
    - center_averages: bool -> center both window averages across stimuli

OUTPUT:
    - window_similarity: np.ndarray -> one Pearson r per stimulus
"""
def dynamic_window_similarity(data, first_window_ms, second_window_ms, center_averages=False):
    # Both are channels x stimuli window averages of the movie response.
    first_vectors = half_open_window_mean(
        data["dynamic"], data["neural_fs"], first_window_ms, center=center_averages,
    )
    second_vectors = half_open_window_mean(
        data["dynamic"], data["neural_fs"], second_window_ms, center=center_averages,
    )
    # A singleton time axis turns the helper into one correlation per stimulus.
    return stimuluswise_reference_correlation(
        first_vectors[:, np.newaxis, :], second_vectors,
    )[0]
# EOF


"""
plot_static_dynamic_window_similarity
Scatter the static frame correlation against the movie window correlation.

INPUT:
    - static_similarity: np.ndarray -> static last vs earlier frame r per stimulus
    - movie_similarity: np.ndarray -> movie window correlation per stimulus
    - earlier_frame: str -> label of the static earlier frame
    - first_window_ms: tuple[float, float] -> earlier movie window
    - second_window_ms: tuple[float, float] -> later movie window
    - stimulus_names: np.ndarray | None -> labels drawn next to points, or None

OUTPUT:
    - figure: matplotlib.figure.Figure -> one-panel scatterplot
    - axis: matplotlib.axes.Axes -> corresponding axis
"""
def plot_static_dynamic_window_similarity(
        static_similarity,
        movie_similarity,
        earlier_frame,
        first_window_ms,
        second_window_ms,
        stimulus_names=None,
        ):
    figure, axis = plt.subplots(figsize=(5.5, 5))
    _scatter_with_spearman(
        axis, static_similarity, movie_similarity,
        f"Static last vs {earlier_frame} frame r",
        f"Movie {first_window_ms} vs {second_window_ms} ms r",
        stimulus_names=stimulus_names,
    )
    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.axvline(0, color="0.3", linewidth=0.8)
    figure.tight_layout()
    return figure, axis
# EOF


"""
dynamic_window_regression_fit
Leave-one-stimulus-out cross-channel ridge fit between two movie windows: each
stimulus's second-window pattern is predicted from its first-window pattern by
a mapping fitted on all other stimuli, and scored by the Pearson correlation
(across channels) between predicted and actual second-window patterns.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - first_window_ms: tuple[float, float] -> earlier half-open movie window (predictor)
    - second_window_ms: tuple[float, float] -> later half-open movie window (target)
    - ridge_alpha: float -> relative ridge penalty
    - center_averages: bool -> center both window averages across stimuli

OUTPUT:
    - window_fit: np.ndarray -> one held-out prediction r per stimulus
"""
def dynamic_window_regression_fit(
        data, first_window_ms, second_window_ms, ridge_alpha=1.0, center_averages=False,
        ):
    # Transposed to stimuli x channels for the regression.
    first_patterns = half_open_window_mean(
        data["dynamic"], data["neural_fs"], first_window_ms, center=center_averages,
    ).T
    second_patterns = half_open_window_mean(
        data["dynamic"], data["neural_fs"], second_window_ms, center=center_averages,
    ).T
    predictions = leave_one_out_ridge_predictions(first_patterns, second_patterns, ridge_alpha)
    return rowwise_similarity(predictions, second_patterns)
# EOF
