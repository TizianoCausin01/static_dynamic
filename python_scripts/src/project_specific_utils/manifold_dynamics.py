from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

from .channelwise_correlation import channelwise_regress_out
from .split_half_rsa import (
    compute_rdm_timeseries,
    cross_temporal_similarity,
)


"""
population_response_scores
Average the population response for every stimulus inside a time window.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - fs: float -> sampling frequency in Hz
    - window_ms: tuple[float, float] -> inclusive-start, exclusive-stop window

OUTPUT:
    - scores: np.ndarray -> mean population response for every stimulus
"""
def population_response_scores(
        rasters: np.ndarray,
        fs: float,
        window_ms: tuple[float, float],
        ) -> np.ndarray:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3:
        raise ValueError("rasters must have shape channels x time x stimuli.")
    # end if rasters.ndim
    if fs <= 0:
        raise ValueError("fs must be positive.")
    # end if fs

    window_start_ms, window_stop_ms = window_ms
    if window_start_ms < 0 or window_stop_ms <= window_start_ms:
        raise ValueError("window_ms must be an increasing non-negative interval.")
    # end if invalid window

    start_index = int(np.ceil(window_start_ms * fs / 1000))
    stop_index = int(np.ceil(window_stop_ms * fs / 1000))
    if start_index >= rasters.shape[1] or stop_index > rasters.shape[1]:
        duration_ms = rasters.shape[1] * 1000 / fs
        raise ValueError(
            f"window_ms={window_ms} exceeds the {duration_ms:g} ms recording."
        )
    # end if window exceeds recording

    return rasters[:, start_index:stop_index, :].mean(axis=(0, 1))
# EOF


"""
select_manifold_subsets
Select the highest, lowest, and seeded random stimulus subsets.

INPUT:
    - scores: np.ndarray -> one population-response score per stimulus
    - subset_size: int -> number of stimuli retained in every subset
    - n_random_sets: int -> number of size-matched random controls
    - rng: np.random.Generator -> random generator controlling random subsets

OUTPUT:
    - subsets: dict[str, np.ndarray] -> stimulus indices for every subset
"""
def select_manifold_subsets(
        scores: np.ndarray,
        subset_size: int,
        n_random_sets: int,
        rng: np.random.Generator,
        ) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite one-dimensional array.")
    # end if invalid scores
    if not isinstance(subset_size, int) or not 2 <= subset_size <= len(scores):
        raise ValueError("subset_size must be between 2 and the stimulus count.")
    # end if invalid subset_size
    if not isinstance(n_random_sets, int) or n_random_sets < 1:
        raise ValueError("n_random_sets must be a positive integer.")
    # end if invalid n_random_sets

    # A stable sort makes ties reproducible in the original stimulus order.
    sorted_indices = np.argsort(scores, kind="stable")
    subsets = {
        "all": np.arange(len(scores)),
        "top": sorted_indices[-subset_size:][::-1],
        "bottom": sorted_indices[:subset_size],
    }
    for random_index in range(n_random_sets):
        subset_name = f"random_{random_index + 1:03d}"
        subsets[subset_name] = np.sort(
            rng.choice(len(scores), size=subset_size, replace=False)
        )
    # end for random_index
    return subsets
# EOF


"""
compute_drsa_autocorrelation
Compute Marvi-style time-by-time similarity between neural RDMs.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - rdm_metric: str -> dissimilarity used to construct each timepoint RDM
    - rsa_metric: str -> Pearson correlation or Spearman RDM similarity

OUTPUT:
    - autocorrelation: np.ndarray -> time x time RDM-similarity matrix
"""
def compute_drsa_autocorrelation(
        rasters: np.ndarray,
        rdm_metric: str = "cosine_cnt",
        rsa_metric: str = "correlation",
        ) -> np.ndarray:
    rdm_timeseries = compute_rdm_timeseries(rasters, metric=rdm_metric)
    return cross_temporal_similarity(
        rdm_timeseries, rdm_timeseries, metric=rsa_metric,
    )
# EOF


"""
compute_cross_temporal_drsa
Compare dynamic and static RDM time series across every pair of timepoints.

INPUT:
    - dynamic_rasters: np.ndarray -> channels x dynamic time x matched stimuli
    - static_rasters: np.ndarray -> channels x static time x matched stimuli
    - rdm_metric: str -> dissimilarity used to construct each timepoint RDM
    - rsa_metric: str -> Pearson correlation or Spearman RDM similarity

OUTPUT:
    - similarity: np.ndarray -> dynamic time x static time RDM similarity
"""
def compute_cross_temporal_drsa(
        dynamic_rasters: np.ndarray,
        static_rasters: np.ndarray,
        rdm_metric: str = "cosine_cnt",
        rsa_metric: str = "correlation",
        ) -> np.ndarray:
    dynamic_rasters = np.asarray(dynamic_rasters)
    static_rasters = np.asarray(static_rasters)
    if dynamic_rasters.ndim != 3 or static_rasters.ndim != 3:
        raise ValueError(
            "dynamic_rasters and static_rasters must have shape "
            "channels x time x stimuli."
        )
    # end if invalid dimensions
    if dynamic_rasters.shape[0] != static_rasters.shape[0]:
        raise ValueError("Dynamic and static rasters must use matching channels.")
    # end if mismatched channels
    if dynamic_rasters.shape[2] != static_rasters.shape[2]:
        raise ValueError("Dynamic and static rasters must use matching stimuli.")
    # end if mismatched stimuli

    dynamic_rdms = compute_rdm_timeseries(dynamic_rasters, metric=rdm_metric)
    static_rdms = compute_rdm_timeseries(static_rasters, metric=rdm_metric)
    return cross_temporal_similarity(
        dynamic_rdms, static_rdms, metric=rsa_metric,
    )
# EOF


"""
bin_average_rasters
Average consecutive samples into non-overlapping time bins.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - source_fs: float -> sampling frequency of rasters in Hz
    - bin_fs: float -> output sampling frequency; source_fs / bin_fs must be
      an integer

OUTPUT:
    - binned_rasters: np.ndarray -> channels x bins x stimuli; an incomplete
      final bin is dropped
"""
def bin_average_rasters(
        rasters: np.ndarray,
        source_fs: float,
        bin_fs: float,
        ) -> np.ndarray:
    bin_size = source_fs / bin_fs
    if bin_size < 1 or not np.isclose(bin_size, round(bin_size)):
        raise ValueError("source_fs / bin_fs must be a positive integer.")
    # end if invalid bin size
    bin_size = int(round(bin_size))
    n_channels, n_samples, n_stimuli = rasters.shape
    n_bins = n_samples // bin_size
    # channels x bins x samples-per-bin x stimuli, then average within bins.
    return rasters[:, :n_bins * bin_size, :].reshape(
        n_channels, n_bins, bin_size, n_stimuli,
    ).mean(axis=2)
# EOF



"""
regress_out_rdm_timeseries
Remove a predictor RDM from a target RDM independently at every timepoint.
At each time, the target RDM vector is regressed on the predictor RDM vector
(OLS with intercept) across stimulus pairs, and the residual is kept.

INPUT:
    - target_rdms: np.ndarray -> time x stimulus-pair distances
    - predictor_rdms: np.ndarray -> time x stimulus-pair distances, same shape

OUTPUT:
    - residual_rdms: np.ndarray -> time x stimulus-pair residual distances
    - variance_removed: np.ndarray -> fraction of target variance explained
      by the predictor at every timepoint
"""
def regress_out_rdm_timeseries(
        target_rdms: np.ndarray,
        predictor_rdms: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
    target_rdms = np.asarray(target_rdms, dtype=np.float64)
    predictor_rdms = np.asarray(predictor_rdms, dtype=np.float64)
    if target_rdms.shape != predictor_rdms.shape:
        raise ValueError("target_rdms and predictor_rdms must match in shape.")
    # end if shapes differ

    # Centering each row implements the intercept of the per-time regression.
    target_centered = target_rdms - target_rdms.mean(axis=1, keepdims=True)
    predictor_centered = (
        predictor_rdms - predictor_rdms.mean(axis=1, keepdims=True)
    )
    predictor_variance = (predictor_centered ** 2).sum(axis=1)
    betas = np.zeros(len(target_rdms))
    np.divide(
        (target_centered * predictor_centered).sum(axis=1),
        predictor_variance, out=betas, where=predictor_variance > 0,
    )
    residual_rdms = target_centered - betas[:, None] * predictor_centered

    target_variance = (target_centered ** 2).sum(axis=1)
    variance_removed = np.full(len(target_rdms), np.nan)
    np.divide(
        target_variance - (residual_rdms ** 2).sum(axis=1),
        target_variance, out=variance_removed, where=target_variance > 0,
    )
    return residual_rdms, variance_removed
# EOF


"""
static_dynamic_drsa_peak
Find the static and dynamic timepoints whose RDMs match best in the
cross-temporal static-dynamic dRSA. Rasters are bin-averaged first, and the
peak is searched only among bins whose full +/- half_window_ms window lies
inside the requested search ranges. If previous_static_rasters is given, the
previous-frame response is regressed out of the static response at every
static time, so only last-frame-specific structure remains. With
previous_regression="rdm" the static RDM is residualized on the previous-frame
RDM across stimulus pairs; with "signal" each channel's static response is
residualized on the same channel's previous-frame response across stimuli, and
the static RDM is then built from those residuals.

INPUT:
    - static_rasters: np.ndarray -> channels x static time x matched stimuli
    - dynamic_rasters: np.ndarray -> channels x dynamic time x matched stimuli
    - source_fs: float -> sampling frequency of both rasters in Hz
    - drsa_fs: float -> sampling frequency used for the dRSA bins
    - static_search_ms: tuple[float, float] -> allowed static window span
    - dynamic_search_ms: tuple[float, float] -> allowed dynamic window span
    - half_window_ms: float -> half-width of the windows around the peak
    - rdm_metric: str -> dissimilarity used to construct each timepoint RDM
    - rsa_metric: str -> correlation or spearman RDM similarity
    - smoothing_sigma_bins: float -> Gaussian sigma (in dRSA bins) applied
      to the matrix before the peak search; 0 disables smoothing
    - previous_static_rasters: np.ndarray | None -> channels x static time x
      matched stimuli responses to the previous movie frame shown as an image
    - previous_regression: str -> "rdm" (regress RDMs) or "signal" (regress
      responses channel by channel, then build RDMs from the residuals)

OUTPUT:
    - drsa_peak: dict -> full dRSA matrix (dynamic time x static time), bin
      centre times, search mask, peak bin indices, peak times, peak value,
      and per-static-bin variance removed by the previous frame (or None);
      for "signal" this is the across-stimulus response variance, pooled
      over channels
"""
def static_dynamic_drsa_peak(
        static_rasters: np.ndarray,
        dynamic_rasters: np.ndarray,
        source_fs: float,
        drsa_fs: float,
        static_search_ms: tuple[float, float],
        dynamic_search_ms: tuple[float, float],
        half_window_ms: float,
        rdm_metric: str = "cosine_cnt",
        rsa_metric: str = "correlation",
        smoothing_sigma_bins: float = 0,
        previous_static_rasters: np.ndarray | None = None,
        previous_regression: str = "rdm",
        ) -> dict:
    if previous_regression not in ("rdm", "signal"):
        raise ValueError("previous_regression must be 'rdm' or 'signal'.")
    # end if invalid previous_regression
    static_binned = bin_average_rasters(static_rasters, source_fs, drsa_fs)
    dynamic_binned = bin_average_rasters(dynamic_rasters, source_fs, drsa_fs)

    previous_binned = None
    if previous_static_rasters is not None:
        previous_binned = bin_average_rasters(
            previous_static_rasters, source_fs, drsa_fs,
        )
        if previous_binned.shape != static_binned.shape:
            raise ValueError(
                "previous_static_rasters must match static_rasters in shape."
            )
        # end if previous-frame shape differs
    # end if previous frame given

    variance_removed = None
    if previous_binned is not None and previous_regression == "signal":
        # Per channel and static bin: OLS of the last-frame response on the
        # previous-frame response across stimuli; keep the residual response.
        target_binned = static_binned
        static_binned, _, _, _ = channelwise_regress_out(
            previous_binned, target_binned,
        )
        # Fraction of across-stimulus variance removed, pooled over channels.
        target_variance = target_binned.var(axis=2).sum(axis=0)
        residual_variance = static_binned.var(axis=2).sum(axis=0)
        variance_removed = np.full(len(target_variance), np.nan)
        np.divide(
            target_variance - residual_variance, target_variance,
            out=variance_removed, where=target_variance > 0,
        )
    # end if signal regression

    # RDM time series are time x stimulus pairs.
    static_rdms = compute_rdm_timeseries(static_binned, metric=rdm_metric)
    dynamic_rdms = compute_rdm_timeseries(dynamic_binned, metric=rdm_metric)

    if previous_binned is not None and previous_regression == "rdm":
        previous_rdms = compute_rdm_timeseries(
            previous_binned, metric=rdm_metric,
        )
        static_rdms, variance_removed = regress_out_rdm_timeseries(
            static_rdms, previous_rdms,
        )
    # end if RDM regression

    similarity = cross_temporal_similarity(
        dynamic_rdms, static_rdms, metric=rsa_metric,
    )

    # Bin i averages samples [i, i + 1) / drsa_fs, so its centre is i + 0.5.
    bin_ms = 1000 / drsa_fs
    static_time_ms = (np.arange(static_binned.shape[1]) + 0.5) * bin_ms
    dynamic_time_ms = (np.arange(dynamic_binned.shape[1]) + 0.5) * bin_ms

    # Keep only centres whose full window fits inside the search range.
    static_allowed = (
        (static_time_ms - half_window_ms >= static_search_ms[0])
        & (static_time_ms + half_window_ms <= static_search_ms[1])
    )
    dynamic_allowed = (
        (dynamic_time_ms - half_window_ms >= dynamic_search_ms[0])
        & (dynamic_time_ms + half_window_ms <= dynamic_search_ms[1])
    )
    search_mask = dynamic_allowed[:, None] & static_allowed[None, :]
    if not search_mask.any():
        raise ValueError("No dRSA bin fits inside both search ranges.")
    # end if empty search range

    # Smooth only the copy used to find the peak; single bins are noisy.
    peak_similarity = similarity
    if smoothing_sigma_bins > 0:
        peak_similarity = gaussian_filter(
            np.nan_to_num(similarity, nan=np.nanmin(similarity)),
            sigma=smoothing_sigma_bins,
        )
    # end if smoothing
    searched_similarity = np.where(search_mask, peak_similarity, np.nan)
    dynamic_index, static_index = np.unravel_index(
        np.nanargmax(searched_similarity), similarity.shape,
    )
    return {
        "similarity": similarity,
        "static_time_ms": static_time_ms,
        "dynamic_time_ms": dynamic_time_ms,
        "search_mask": search_mask,
        "peak_static_index": int(static_index),
        "peak_dynamic_index": int(dynamic_index),
        "peak_static_ms": float(static_time_ms[static_index]),
        "peak_dynamic_ms": float(dynamic_time_ms[dynamic_index]),
        "peak_value": float(similarity[dynamic_index, static_index]),
        "previous_frame_variance_removed": variance_removed,
        "peak_smoothed_value": float(
            peak_similarity[dynamic_index, static_index]
        ),
    }
# EOF


"""
compute_pc_subspaces
Fit a centered PCA independently at every timepoint and retain its loadings.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - n_components: int -> number of dominant response axes to retain

OUTPUT:
    - subspaces: np.ndarray -> time x channels x components orthonormal bases
"""
def compute_pc_subspaces(
        rasters: np.ndarray,
        n_components: int = 2,
        ) -> np.ndarray:
    rasters = np.asarray(rasters, dtype=np.float64)
    if rasters.ndim != 3:
        raise ValueError("rasters must have shape channels x time x stimuli.")
    # end if rasters.ndim
    maximum_components = min(rasters.shape[0], rasters.shape[2] - 1)
    if not isinstance(n_components, int) or not 1 <= n_components <= maximum_components:
        raise ValueError(
            f"n_components must be between 1 and {maximum_components}."
        )
    # end if invalid n_components

    subspaces = []
    for time_index in range(rasters.shape[1]):
        # PCA samples are stimuli and PCA features are neural channels.
        response_matrix = rasters[:, time_index, :].T
        response_matrix -= response_matrix.mean(axis=0, keepdims=True)
        _, _, right_singular_vectors = np.linalg.svd(
            response_matrix, full_matrices=False,
        )
        subspaces.append(right_singular_vectors[:n_components].T)
    # end for time_index
    return np.stack(subspaces)
# EOF


"""
compute_average_pc_rotation
Compute the mean principal angle between dominant PCA subspaces over time.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - n_components: int -> number of PCA axes defining each subspace

OUTPUT:
    - rotation_degrees: np.ndarray -> time x time mean principal angle in degrees
"""
def compute_average_pc_rotation(
        rasters: np.ndarray,
        n_components: int = 2,
        ) -> np.ndarray:
    subspaces = compute_pc_subspaces(rasters, n_components=n_components)
    n_timepoints = subspaces.shape[0]
    rotation_degrees = np.zeros((n_timepoints, n_timepoints), dtype=np.float64)

    for first_time in range(n_timepoints):
        first_basis = subspaces[first_time]
        for second_time in range(first_time + 1, n_timepoints):
            second_basis = subspaces[second_time]

            # Singular values are cosines of the principal angles. Clipping
            # prevents floating-point error from producing invalid arccos input.
            cosines = np.linalg.svd(
                first_basis.T @ second_basis, compute_uv=False,
            )
            principal_angles = np.arccos(np.clip(cosines, -1, 1))
            average_angle = np.degrees(principal_angles).mean()
            rotation_degrees[first_time, second_time] = average_angle
            rotation_degrees[second_time, first_time] = average_angle
        # end for second_time
    # end for first_time
    return rotation_degrees
# EOF


"""
compute_cross_temporal_pc_rotation
Compute principal-angle rotation between dynamic and static PCA subspaces.

INPUT:
    - dynamic_rasters: np.ndarray -> channels x dynamic time x matched stimuli
    - static_rasters: np.ndarray -> channels x static time x matched stimuli
    - n_components: int -> number of PCA axes defining each subspace

OUTPUT:
    - rotation_degrees: np.ndarray -> dynamic time x static time mean angle
"""
def compute_cross_temporal_pc_rotation(
        dynamic_rasters: np.ndarray,
        static_rasters: np.ndarray,
        n_components: int = 2,
        ) -> np.ndarray:
    dynamic_rasters = np.asarray(dynamic_rasters)
    static_rasters = np.asarray(static_rasters)
    if dynamic_rasters.ndim != 3 or static_rasters.ndim != 3:
        raise ValueError(
            "dynamic_rasters and static_rasters must have shape "
            "channels x time x stimuli."
        )
    # end if invalid dimensions
    if dynamic_rasters.shape[0] != static_rasters.shape[0]:
        raise ValueError("Dynamic and static rasters must use matching channels.")
    # end if mismatched channels
    if dynamic_rasters.shape[2] != static_rasters.shape[2]:
        raise ValueError("Dynamic and static rasters must use matching stimuli.")
    # end if mismatched stimuli

    dynamic_subspaces = compute_pc_subspaces(
        dynamic_rasters, n_components=n_components,
    )
    static_subspaces = compute_pc_subspaces(
        static_rasters, n_components=n_components,
    )
    rotation_degrees = np.zeros(
        (dynamic_subspaces.shape[0], static_subspaces.shape[0]),
        dtype=np.float64,
    )

    for dynamic_time, dynamic_basis in enumerate(dynamic_subspaces):
        for static_time, static_basis in enumerate(static_subspaces):
            # Singular values are cosines of the principal angles between the
            # two condition-specific population subspaces.
            cosines = np.linalg.svd(
                dynamic_basis.T @ static_basis, compute_uv=False,
            )
            principal_angles = np.arccos(np.clip(cosines, -1, 1))
            rotation_degrees[dynamic_time, static_time] = np.degrees(
                principal_angles,
            ).mean()
        # end for static_time, static_basis
    # end for dynamic_time, dynamic_basis
    return rotation_degrees
# EOF


"""
compute_manifold_dynamics
Compute RDM autocorrelation and PCA-subspace rotation for every subset.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli neural responses
    - subsets: dict[str, np.ndarray] -> stimulus indices for every analysis
    - rdm_metric: str -> dissimilarity used to construct timepoint RDMs
    - rsa_metric: str -> Pearson correlation or Spearman RDM similarity
    - n_pc_components: int -> number of PCA axes defining each subspace

OUTPUT:
    - results: dict[str, dict[str, np.ndarray]] -> matrices for every subset
"""
def compute_manifold_dynamics(
        rasters: np.ndarray,
        subsets: dict[str, np.ndarray],
        rdm_metric: str = "cosine_cnt",
        rsa_metric: str = "correlation",
        n_pc_components: int = 2,
        ) -> dict[str, dict[str, np.ndarray]]:
    rasters = np.asarray(rasters)
    results = {}
    for subset_name, stimulus_indices in subsets.items():
        subset_rasters = rasters[:, :, stimulus_indices]
        results[subset_name] = {
            "drsa_autocorrelation": compute_drsa_autocorrelation(
                subset_rasters,
                rdm_metric=rdm_metric,
                rsa_metric=rsa_metric,
            ),
            "pc_rotation_degrees": compute_average_pc_rotation(
                subset_rasters,
                n_components=n_pc_components,
            ),
        }
    # end for subset_name, stimulus_indices
    return results
# EOF


"""
compute_cross_temporal_manifold_dynamics
Compute static-dynamic RDM similarity and PCA rotation for shared subsets.

INPUT:
    - dynamic_rasters: np.ndarray -> channels x dynamic time x matched stimuli
    - static_rasters: np.ndarray -> channels x static time x matched stimuli
    - subsets: dict[str, np.ndarray] -> shared stimulus indices per analysis
    - rdm_metric: str -> dissimilarity used to construct timepoint RDMs
    - rsa_metric: str -> Pearson correlation or Spearman RDM similarity
    - n_pc_components: int -> number of PCA axes defining each subspace

OUTPUT:
    - results: dict[str, dict[str, np.ndarray]] -> cross-temporal matrices
"""
def compute_cross_temporal_manifold_dynamics(
        dynamic_rasters: np.ndarray,
        static_rasters: np.ndarray,
        subsets: dict[str, np.ndarray],
        rdm_metric: str = "cosine_cnt",
        rsa_metric: str = "correlation",
        n_pc_components: int = 2,
        ) -> dict[str, dict[str, np.ndarray]]:
    dynamic_rasters = np.asarray(dynamic_rasters)
    static_rasters = np.asarray(static_rasters)
    if dynamic_rasters.ndim != 3 or static_rasters.ndim != 3:
        raise ValueError(
            "dynamic_rasters and static_rasters must have shape "
            "channels x time x stimuli."
        )
    # end if invalid dimensions
    if dynamic_rasters.shape[0] != static_rasters.shape[0]:
        raise ValueError("Dynamic and static rasters must use matching channels.")
    # end if mismatched channels
    if dynamic_rasters.shape[2] != static_rasters.shape[2]:
        raise ValueError("Dynamic and static rasters must use matching stimuli.")
    # end if mismatched stimuli

    results = {}
    for subset_name, stimulus_indices in subsets.items():
        dynamic_subset = dynamic_rasters[:, :, stimulus_indices]
        static_subset = static_rasters[:, :, stimulus_indices]
        results[subset_name] = {
            "drsa_similarity": compute_cross_temporal_drsa(
                dynamic_subset,
                static_subset,
                rdm_metric=rdm_metric,
                rsa_metric=rsa_metric,
            ),
            "pc_rotation_degrees": compute_cross_temporal_pc_rotation(
                dynamic_subset,
                static_subset,
                n_components=n_pc_components,
            ),
        }
    # end for subset_name, stimulus_indices
    return results
# EOF


"""
plot_static_dynamic_matrix
Draw a dynamic time x static time matrix in the OC_presentation style: movie
time on x, static-image time on y, both in ms from onset, and a red dashed
line at the last-frame onset.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - matrix: np.ndarray -> dynamic time x static time values
    - bin_ms: float -> duration of one time bin in ms
    - cmap: str -> matplotlib colormap name
    - vmin: float | None -> lower color limit
    - vmax: float | None -> upper color limit
    - colorbar_label: str -> colorbar label
    - last_frame_onset_ms: float | None -> movie time of the last-frame onset
    - last_frame_color: str -> color of the last-frame line
    - ticks_size: float -> tick-label font size
    - label_size: float -> axis-label font size

OUTPUT:
    - image: matplotlib.image.AxesImage -> drawn matrix image
"""
def plot_static_dynamic_matrix(
        axis,
        matrix: np.ndarray,
        bin_ms: float,
        cmap: str = "viridis",
        vmin: float | None = 0,
        vmax: float | None = 0.65,
        colorbar_label: str = r"RSA corr ($\rho$)",
        last_frame_onset_ms: float | None = 2500,
        last_frame_color: str = "red",
        ticks_size: float = 20,
        label_size: float = 25,
        ):
    # Bin i spans [i, i + 1) * bin_ms, so the axes run from 0 to n * bin_ms.
    dynamic_time_end_ms = matrix.shape[0] * bin_ms
    static_time_end_ms = matrix.shape[1] * bin_ms
    image = axis.imshow(
        matrix.T, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", origin="lower",
        extent=(0, dynamic_time_end_ms, 0, static_time_end_ms),
    )
    if last_frame_onset_ms is not None:
        axis.axvline(
            last_frame_onset_ms, linestyle="--", color=last_frame_color,
        )
    # end if last-frame line
    axis.set(xlabel="vid response (ms)", ylabel="img response (ms)")
    axis.tick_params(axis="both", labelsize=ticks_size)
    axis.xaxis.label.set_size(label_size)
    axis.yaxis.label.set_size(label_size)
    colorbar = axis.figure.colorbar(image, ax=axis)
    colorbar.ax.tick_params(labelsize=ticks_size)
    colorbar.set_label(colorbar_label, fontsize=label_size)
    return image
# EOF


"""
plot_static_dynamic_drsa_peak
Draw the static-dynamic dRSA in the OC_presentation style, with the peak
search box (dashed white) and the selected peak (white cross).

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - drsa_peak: dict -> output of static_dynamic_drsa_peak
    - static_search_ms: tuple[float, float] -> static search range
    - dynamic_search_ms: tuple[float, float] -> dynamic search range
    - **matrix_kwargs -> style options passed to plot_static_dynamic_matrix

OUTPUT:
    - image: matplotlib.image.AxesImage -> drawn dRSA image
"""
def plot_static_dynamic_drsa_peak(
        axis,
        drsa_peak: dict,
        static_search_ms: tuple[float, float],
        dynamic_search_ms: tuple[float, float],
        **matrix_kwargs,
        ):
    static_time_ms = drsa_peak["static_time_ms"]
    bin_ms = static_time_ms[1] - static_time_ms[0]
    image = plot_static_dynamic_matrix(
        axis, drsa_peak["similarity"], bin_ms, **matrix_kwargs,
    )
    axis.add_patch(plt.Rectangle(
        (dynamic_search_ms[0], static_search_ms[0]),
        dynamic_search_ms[1] - dynamic_search_ms[0],
        static_search_ms[1] - static_search_ms[0],
        fill=False, edgecolor="white", linestyle="--", linewidth=1,
    ))
    axis.plot(
        drsa_peak["peak_dynamic_ms"], drsa_peak["peak_static_ms"],
        marker="x", color="white", markersize=10, markeredgewidth=2,
    )
    return image
# EOF


"""
plot_significance_masked_matrix
Draw a static-dynamic matrix with plot_static_dynamic_matrix, keeping the
significant cells sharp and outlined while the rest is blurred and dimmed.
Dimming blends the non-significant cells' colours toward dim_color, so their
values stay readable.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - matrix: np.ndarray -> dynamic time x static time values
    - significant: np.ndarray -> dynamic time x static time boolean mask
    - bin_ms: float -> duration of one matrix bin in ms
    - nonsignificant_blur_sigma: float -> Gaussian sigma (bins) applied to the
        non-significant cells; 0 disables the blur
    - nonsignificant_dim_alpha: float -> opacity of the dim_color veil drawn
        over the non-significant cells; 0 disables the dimming
    - nonsignificant_dim_color: str -> colour the non-significant cells are
        blended toward ("black" darkens them)
    - contour_color: str -> colour of the outline around significant cells
    - contour_linewidth: float -> width of that outline
    - **matrix_kwargs -> style options passed to plot_static_dynamic_matrix

OUTPUT:
    - image: matplotlib.image.AxesImage -> drawn matrix image
"""
def plot_significance_masked_matrix(
        axis,
        matrix: np.ndarray,
        significant: np.ndarray,
        bin_ms: float,
        nonsignificant_blur_sigma: float = 2,
        nonsignificant_dim_alpha: float = 0.4,
        nonsignificant_dim_color: str = "black",
        contour_color: str = "white",
        contour_linewidth: float = 2,
        **matrix_kwargs,
        ):
    matrix = np.asarray(matrix, dtype=np.float64)
    significant = np.asarray(significant, dtype=bool)
    if significant.shape != matrix.shape:
        raise ValueError("significant must match matrix in shape.")
    # end if shapes differ

    # Blur only what is shown outside the significant region.
    displayed = matrix.copy()
    if nonsignificant_blur_sigma > 0:
        blurred = gaussian_filter(
            np.nan_to_num(matrix, nan=np.nanmean(matrix)),
            sigma=nonsignificant_blur_sigma,
        )
        displayed = np.where(significant, matrix, blurred)
    # end if blur requested
    image = plot_static_dynamic_matrix(
        axis, displayed, bin_ms, **matrix_kwargs,
    )

    # Same extent as plot_static_dynamic_matrix: bin i spans [i, i + 1) * bin_ms.
    extent = (0, matrix.shape[0] * bin_ms, 0, matrix.shape[1] * bin_ms)
    if nonsignificant_dim_alpha > 0:
        # RGBA veil of dim_color, transparent over the significant cells.
        veil = np.zeros(matrix.T.shape + (4,))
        veil[..., :3] = to_rgb(nonsignificant_dim_color)
        veil[..., 3] = np.where(significant.T, 0, nonsignificant_dim_alpha)
        axis.imshow(veil, aspect="auto", origin="lower", extent=extent)
    # end if dimming requested
    if significant.any() and not significant.all():
        # Contour through bin centres outlines the significant region.
        dynamic_centres_ms = (np.arange(matrix.shape[0]) + 0.5) * bin_ms
        static_centres_ms = (np.arange(matrix.shape[1]) + 0.5) * bin_ms
        axis.contour(
            dynamic_centres_ms, static_centres_ms, significant.T.astype(float),
            levels=[0.5], colors=contour_color, linewidths=contour_linewidth,
        )
    # end if the mask has a boundary
    return image
# EOF


"""
plot_gradient_trajectory_2d
Draw a 2D trajectory as a line whose colour follows time. The trajectory is
cubic-spline interpolated in time so both the curve and the colour gradient
look continuous. A dashed line keeps alternating stretches of equal arc length,
measured in axis-fraction units; set the axis limits before calling so the
dashes have a uniform size. Optional arrowheads, evenly spaced along the arc,
point in the direction of increasing time.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - scores: np.ndarray -> (time, 2) trajectory coordinates
    - times_s: np.ndarray -> (time,) time of each sample, mapped to colour
    - cmap: str | matplotlib.colors.Colormap -> colormap (name or object)
    - norm: matplotlib.colors.Normalize -> time-to-colour normalization
    - upsample: int -> interpolated points per original sample interval
    - linewidth: float -> line width
    - alpha: float -> line opacity
    - dashed: bool -> draw a dashed instead of a solid line
    - dash_length: float -> length of each dash as a fraction of the axis
    - gap_length: float -> length of each gap as a fraction of the axis
    - outline_color: str | None -> colour of an outline drawn under the line;
        None draws no outline
    - outline_width: float -> outline thickness added on each side of the line
    - n_arrows: int -> arrowheads along the trajectory; 0 draws none
    - arrow_size: float -> arrowhead size in points
    - zorder: float -> drawing order

OUTPUT:
    - collection: matplotlib.collections.LineCollection -> drawn line, usable
        as the colorbar mappable
"""
def plot_gradient_trajectory_2d(
        axis,
        scores: np.ndarray,
        times_s: np.ndarray,
        cmap,
        norm,
        upsample: int = 10,
        linewidth: float = 2,
        alpha: float = 1,
        dashed: bool = False,
        dash_length: float = 0.025,
        gap_length: float = 0.015,
        outline_color: str | None = None,
        outline_width: float = 1,
        n_arrows: int = 0,
        arrow_size: float = 30,
        zorder: float = 1,
        ):
    # Cubic spline through the samples gives a smooth (fine_time, 2) curve.
    fine_times_s = np.linspace(
        times_s[0], times_s[-1], (len(times_s) - 1) * upsample + 1,
    )
    fine_scores = CubicSpline(times_s, scores, axis=0)(fine_times_s)

    # Segment k joins points k and k + 1 and takes its midpoint time's colour.
    segments = np.stack([fine_scores[:-1], fine_scores[1:]], axis=1)
    segment_times_s = (fine_times_s[:-1] + fine_times_s[1:]) / 2

    # Arc length in axis fractions, so dashes and arrow spacing look the same
    # on x and y whatever the data ranges.
    axis_span = np.array([np.ptp(axis.get_xlim()), np.ptp(axis.get_ylim())])
    segment_lengths = np.linalg.norm(np.diff(fine_scores, axis=0) / axis_span, axis=1)
    midpoint_arc = np.cumsum(segment_lengths) - segment_lengths / 2

    if dashed:
        # Repeat dash, gap, dash, ... along the arc; keep only the dash stretches.
        keep = midpoint_arc % (dash_length + gap_length) < dash_length
        segments = segments[keep]
        segment_times_s = segment_times_s[keep]
    # end if dashed
    # Flat ends keep dashes crisp; round ends join a solid line smoothly.
    capstyle = "butt" if dashed else "round"

    if outline_color is not None:
        # One wider single-colour line underneath, so the outline never covers
        # neighbouring segments (per-segment path effects would).
        axis.add_collection(LineCollection(
            segments, colors=outline_color, alpha=alpha, capstyle=capstyle,
            linewidth=linewidth + 2 * outline_width, zorder=zorder - 0.1,
        ))
    # end if outline_color

    collection = LineCollection(
        segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha,
        capstyle=capstyle, zorder=zorder,
    )
    collection.set_array(segment_times_s)
    axis.add_collection(collection)

    if n_arrows > 0:
        colormap = plt.get_cmap(cmap)
        # Arrows at the centres of n_arrows equal arc-length stretches.
        arrow_arcs = (np.arange(n_arrows) + 0.5) / n_arrows * midpoint_arc[-1]
        for arrow_arc in arrow_arcs:
            segment_index = min(np.searchsorted(midpoint_arc, arrow_arc), len(segment_lengths) - 1)
            # Direction from the local tangent: segment start -> segment end.
            start_point, end_point = fine_scores[segment_index], fine_scores[segment_index + 1]
            # Colour of the arrow = time at that point of the (undashed) curve.
            arrow_time = (fine_times_s[segment_index] + fine_times_s[segment_index + 1]) / 2
            axis.annotate(
                "", xy=end_point, xytext=start_point,
                arrowprops=dict(
                    arrowstyle="-|>", mutation_scale=arrow_size, shrinkA=0, shrinkB=0,
                    facecolor=colormap(norm(arrow_time)),
                    edgecolor="black", linewidth=0.8,
                ),
                zorder=zorder + 0.5,
            )
        # end for arrow_arc
    # end if n_arrows
    return collection
# EOF


"""
plot_static_movie_trajectories_2d
Draw the PC1-PC2 stimulus-average trajectories of the static image (solid,
outlined) and the movie (dashed without outline by default, or solid and
outlined), both coloured by movie-aligned time, optionally with arrowheads in
the direction of time; fit the axis limits to them and optionally add a legend.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - static_scores: np.ndarray -> (static time, components) PCA scores
    - movie_scores: np.ndarray -> (movie time, components) PCA scores
    - static_color_times: np.ndarray -> (static time,) movie-aligned time of
        every static sample, mapped to colour
    - movie_color_times: np.ndarray -> (movie time,) time of every movie sample
    - cmap: str | matplotlib.colors.Colormap -> colormap (name or object)
    - norm: matplotlib.colors.Normalize -> time-to-colour normalization
    - linewidth: float -> trajectory line width
    - movie_linewidth: float | None -> dashed movie line width; None = linewidth
    - movie_alpha: float -> opacity of the dashed movie line
    - upsample: int -> cubic-spline points per sample interval
    - dash_length: float -> movie dash length as a fraction of the axis
    - gap_length: float -> movie gap length as a fraction of the axis
    - movie_dashed: bool -> dashed movie line (True) or a solid one (False)
    - movie_outline: bool -> draw the black outline under the movie line too
    - show_legend: bool -> add the static/movie legend
    - n_arrows: int -> arrowheads per trajectory; 0 draws none
    - arrow_size: float -> arrowhead size in points
    - static_label: str -> legend label of the static trajectory
    - movie_label: str -> legend label of the movie trajectory
    - legend_fontsize: float | None -> legend font size; None = default
    - legend_kwargs: dict | None -> extra axis.legend options (e.g. loc,
        bbox_to_anchor, ncol); None = matplotlib defaults

OUTPUT:
    - collection: matplotlib.collections.LineCollection -> opaque static line,
        usable as the colorbar mappable (the movie line may be transparent)
"""
def plot_static_movie_trajectories_2d(
        axis,
        static_scores: np.ndarray,
        movie_scores: np.ndarray,
        static_color_times: np.ndarray,
        movie_color_times: np.ndarray,
        cmap,
        norm,
        linewidth: float = 3.5,
        movie_linewidth: float | None = None,
        movie_alpha: float = 1,
        upsample: int = 10,
        dash_length: float = 0.025,
        gap_length: float = 0.015,
        movie_dashed: bool = True,
        movie_outline: bool = False,
        show_legend: bool = True,
        n_arrows: int = 0,
        arrow_size: float = 30,
        static_label: str = "static",
        movie_label: str = "movie",
        legend_fontsize: float | None = None,
        legend_kwargs: dict | None = None,
        ):
    # Set limits first: dashes and arrow spacing are measured in axis fractions.
    all_xy = np.vstack([static_scores[:, :2], movie_scores[:, :2]])
    lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
    margin = 0.05 * (upper - lower)
    axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])

    if movie_linewidth is None:
        movie_linewidth = linewidth
    # end if movie_linewidth not given
    # Separate drawing layers (static below, movie above, each outline just under
    # its own line), so where the trajectories cross the top line keeps its outline.
    for is_movie, scores, color_times, dashed, outlined, line_width, line_alpha, layer in (
            (False, static_scores, static_color_times, False, True, linewidth, 1, 1),
            (True, movie_scores, movie_color_times, movie_dashed, movie_outline,
             movie_linewidth, movie_alpha, 2),
            ):
        # The black outline keeps the light end of the colormap visible on white.
        line = plot_gradient_trajectory_2d(
            axis, scores[:, :2], color_times, cmap, norm,
            upsample=upsample, linewidth=line_width, alpha=line_alpha, dashed=dashed,
            dash_length=dash_length, gap_length=gap_length,
            outline_color="black" if outlined else None, outline_width=1,
            n_arrows=n_arrows, arrow_size=arrow_size, zorder=layer,
        )
        if not is_movie:
            static_line = line
        # end if static line
    # end for is_movie, scores, color_times, dashed, outlined, line_width, line_alpha, layer

    if show_legend:
        # Line collections have no legend entry, so use grey proxy lines.
        axis.legend(handles=[
            Line2D([], [], color="grey", linewidth=2.5, label=static_label),
            Line2D(
                [], [], color="grey", linewidth=2.5, linestyle="--", alpha=movie_alpha,
                label=movie_label,
            ),
        ], fontsize=legend_fontsize, **(legend_kwargs or {}))
    # end if show_legend
    return static_line
# EOF


"""
plot_significance_bars
Draw one horizontal bar per curve below zero (or below the lowest data,
whichever is lower), covering the timebins where that curve is significant.
Call after the curves are drawn, so the y limits reflect the data.

INPUT:
    - axis: matplotlib.axes.Axes -> axis holding the curves
    - time: np.ndarray -> (time,) x coordinates shared by the masks
    - masks: list[np.ndarray] -> (time,) boolean significance per curve
    - colors: list -> bar colour per curve
    - spacing_fraction: float -> vertical step between bars, as a fraction of
        the y range of the data
    - linewidth: float -> bar thickness

OUTPUT:
    - None
"""
def plot_significance_bars(
        axis,
        time: np.ndarray,
        masks: list,
        colors: list,
        spacing_fraction: float = 0.04,
        linewidth: float = 4,
        ):
    lower, upper = axis.get_ylim()
    spacing = spacing_fraction * (upper - lower)
    # Start just under zero, or under the curves if they dip below it.
    base = min(0.0, lower + spacing)
    for bar_index, (mask, color) in enumerate(zip(masks, colors)):
        bar_y = base - (bar_index + 1) * spacing
        # NaN outside the significant bins breaks the bar into segments.
        bar = np.where(np.asarray(mask, dtype=bool), bar_y, np.nan)
        axis.plot(
            time, bar, color=color, linewidth=linewidth,
            solid_capstyle="butt", label="_nolegend_",
        )
    # end for bar_index
    axis.set_ylim(base - (len(masks) + 1) * spacing, upper)
# EOF


"""
plot_trajectory_band_2d
Draw an error band (e.g. SEM across stimuli) around a 2D trajectory. The
trajectory and its errors are cubic-spline / linearly upsampled; at every point
the band extends, on both sides along the curve normal, by the extent of the
error ellipse (semi-axes = PC1 and PC2 errors) in that direction. Normals and
extents are computed in axis-scaled units, so the band looks right on axes with
very different ranges. The band is opaque (use a light colour) so overlapping
pieces never darken.

INPUT:
    - axis: matplotlib.axes.Axes -> axis to draw on
    - scores: np.ndarray -> (time, 2+) trajectory coordinates (first two used)
    - errors: np.ndarray -> (time, 2+) error along each coordinate
    - color: str | tuple -> band colour
    - axis_span: np.ndarray | None -> (2,) x and y ranges used to scale the
        normals; None uses the trajectory's own range
    - upsample: int -> interpolated points per sample interval
    - zorder: float -> drawing order (keep it below the lines)

OUTPUT:
    - collection: matplotlib.collections.PolyCollection -> drawn band
"""
def plot_trajectory_band_2d(
        axis,
        scores: np.ndarray,
        errors: np.ndarray,
        color,
        axis_span: np.ndarray | None = None,
        upsample: int = 10,
        zorder: float = 1,
        ):
    sample_index = np.arange(len(scores))
    fine_index = np.linspace(0, sample_index[-1], sample_index[-1] * upsample + 1)
    # (fine time, 2) smooth curve and its interpolated errors.
    fine_xy = CubicSpline(sample_index, scores[:, :2])(fine_index)
    fine_errors = np.column_stack([
        np.interp(fine_index, sample_index, errors[:, dimension]) for dimension in range(2)
    ])
    if axis_span is None:
        axis_span = np.ptp(fine_xy, axis=0)
    # end if axis_span not given
    axis_span = np.asarray(axis_span, dtype=float)

    # Unit tangent and normal in axis-scaled units (x and y in comparable units).
    tangent = np.gradient(fine_xy, axis=0) / axis_span
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.where(tangent_norm > 0, tangent_norm, 1)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    # Extent of the error ellipse along the normal, still in scaled units.
    scaled_errors = fine_errors / axis_span
    half_width = np.sqrt(
        (scaled_errors[:, 0] * normal[:, 0]) ** 2 + (scaled_errors[:, 1] * normal[:, 1]) ** 2
    )
    # Back to data units: offset = normal * half width, rescaled per axis.
    offset = normal * half_width[:, None] * axis_span
    upper, lower = fine_xy + offset, fine_xy - offset
    # One quadrilateral per step; edges in the same colour hide the seams.
    quads = np.stack([upper[:-1], upper[1:], lower[1:], lower[:-1]], axis=1)
    collection = PolyCollection(
        quads, facecolors=color, edgecolors=color, linewidths=0.5, zorder=zorder,
    )
    axis.add_collection(collection)
    axis.autoscale_view()
    return collection
# EOF
