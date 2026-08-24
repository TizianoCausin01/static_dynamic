import numpy as np

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
