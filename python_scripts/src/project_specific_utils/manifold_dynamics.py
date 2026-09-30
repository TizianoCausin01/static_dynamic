import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
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
