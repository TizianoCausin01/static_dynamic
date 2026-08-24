import numpy as np
from scipy.stats import rankdata

from useful_stuff.general_utils import create_RDM
from useful_stuff.general_utils.utils import mean_centering


"""
split_half_filename_suffix
Build the metric-aware suffix shared by split-half result writers and loaders.

INPUT:
    - rdm_metric: str -> dissimilarity metric used to construct the RDMs
    - feature_centering: bool -> whether raw features were mean-centered

OUTPUT:
    - suffix: str -> filename suffix identifying the saved computation
"""
def split_half_filename_suffix(
        rdm_metric: str,
        feature_centering: bool = False,
        ) -> str:
    metric_name = str(rdm_metric).strip()
    if not metric_name:
        raise ValueError("rdm_metric must be a non-empty string.")
    # end if not metric_name
    if not all(character.isalnum() or character in {"-", "_"}
               for character in metric_name):
        raise ValueError(
            "rdm_metric may contain only letters, numbers, hyphens, and underscores."
        )
    # end if invalid filename character
    centering_suffix = "_feat_cnt" if feature_centering else ""
    return f"_rdm_{metric_name}{centering_suffix}"
# EOF


"""
average_presentations
Average every stimulus's repetitions while preserving a requested order.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations
    - presentation_identities: list[str] -> identity for every presentation
    - stimulus_order: list[str] -> identities retained in the output

OUTPUT:
    - stimulus_means: np.ndarray -> channels x time x stimuli
"""
def average_presentations(
        rasters: np.ndarray,
        presentation_identities: list[str],
        stimulus_order: list[str],
        ) -> np.ndarray:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3 or rasters.shape[2] != len(presentation_identities):
        raise ValueError(
            "rasters and presentation_identities must describe matching presentations."
        )
    # end if input shapes
    identity_array = np.asarray(presentation_identities)
    stimulus_means = []
    for identity in stimulus_order:
        stimulus_indices = np.flatnonzero(identity_array == identity)
        if len(stimulus_indices) == 0:
            raise ValueError(f"Stimulus {identity!r} has no presentations.")
        # end if len(stimulus_indices)
        stimulus_means.append(rasters[:, :, stimulus_indices].mean(axis=2))
    # end for identity
    return np.stack(stimulus_means, axis=2)
# EOF


"""
average_repetition_halves
Randomly divide each stimulus's repetitions and average the two independent
halves while preserving a requested stimulus order.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations
    - presentation_identities: list[str] -> identity for every presentation
    - stimulus_order: list[str] -> identities retained in the output
    - rng: np.random.Generator -> random generator controlling the split

OUTPUT:
    - first_half: np.ndarray -> channels x time x stimuli first-half averages
    - second_half: np.ndarray -> channels x time x stimuli second-half averages
"""
def average_repetition_halves(
        rasters: np.ndarray,
        presentation_identities: list[str],
        stimulus_order: list[str],
        rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3:
        raise ValueError("rasters must have shape channels x time x presentations.")
    # end if rasters.ndim
    if rasters.shape[2] != len(presentation_identities):
        raise ValueError("presentation_identities must match the presentation axis.")
    # end if rasters.shape[2]
    if len(stimulus_order) < 3:
        raise ValueError("At least three stimuli are required for RDM reliability.")
    # end if len(stimulus_order)

    indices_by_identity = {}
    for presentation_index, identity in enumerate(presentation_identities):
        indices_by_identity.setdefault(identity, []).append(presentation_index)
    # end for presentation_index, identity

    first_half_means = []
    second_half_means = []
    for identity in stimulus_order:
        stimulus_indices = np.asarray(indices_by_identity.get(identity, []), dtype=int)
        if len(stimulus_indices) < 2:
            raise ValueError(
                f"Stimulus {identity!r} has {len(stimulus_indices)} repetitions; "
                "split-half analysis needs at least two."
            )
        # end if len(stimulus_indices)
        shuffled_indices = rng.permutation(stimulus_indices)
        split_index = len(shuffled_indices) // 2
        first_half_means.append(rasters[:, :, shuffled_indices[:split_index]].mean(axis=2))
        second_half_means.append(rasters[:, :, shuffled_indices[split_index:]].mean(axis=2))
    # end for identity

    first_half = np.stack(first_half_means, axis=2)
    second_half = np.stack(second_half_means, axis=2)
    return first_half, second_half
# EOF


"""
compute_rdm_timeseries
Compute one vectorized stimulus RDM at every response timepoint.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli
    - metric: str -> distance metric accepted by create_RDM

OUTPUT:
    - rdm_timeseries: np.ndarray -> time x stimulus-pair distances
"""
def compute_rdm_timeseries(rasters: np.ndarray, metric: str) -> np.ndarray:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3:
        raise ValueError("rasters must have shape channels x time x stimuli.")
    # end if rasters.ndim
    return np.stack([
        create_RDM(np.ascontiguousarray(rasters[:, time_index, :]), metric=metric)
        for time_index in range(rasters.shape[1])
    ])
# EOF


"""
rowwise_similarity
Correlate matching rows of two matrices using Pearson or Spearman similarity.

INPUT:
    - first: np.ndarray -> observations x features
    - second: np.ndarray -> observations x features
    - metric: str -> correlation or spearman

OUTPUT:
    - similarities: np.ndarray -> one similarity per observation
"""
def rowwise_similarity(
        first: np.ndarray,
        second: np.ndarray,
        metric: str = "correlation",
        ) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("first and second must be matching observations x features matrices.")
    # end if first.shape
    if metric == "spearman":
        first = rankdata(first, axis=1)
        second = rankdata(second, axis=1)
    elif metric != "correlation":
        raise ValueError("metric must be 'correlation' or 'spearman'.")
    # end if metric

    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    numerators = np.sum(first * second, axis=1)
    denominators = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    similarities = np.full(first.shape[0], np.nan, dtype=np.float64)
    np.divide(numerators, denominators, out=similarities, where=denominators > 0)
    return np.clip(similarities, -1, 1)
# EOF


"""
cross_temporal_similarity
Correlate every row of one feature matrix with every row of another.

INPUT:
    - first: np.ndarray -> first time x features
    - second: np.ndarray -> second time x features
    - metric: str -> correlation or spearman

OUTPUT:
    - similarity_matrix: np.ndarray -> first time x second time
"""
def cross_temporal_similarity(
        first: np.ndarray,
        second: np.ndarray,
        metric: str = "correlation",
        ) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("Inputs must be time x matching-feature matrices.")
    # end if input dimensions
    if metric == "spearman":
        first = rankdata(first, axis=1)
        second = rankdata(second, axis=1)
    elif metric != "correlation":
        raise ValueError("metric must be 'correlation' or 'spearman'.")
    # end if metric

    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    first_norms = np.linalg.norm(first, axis=1)
    second_norms = np.linalg.norm(second, axis=1)
    denominators = first_norms[:, np.newaxis] * second_norms[np.newaxis, :]
    similarity_matrix = np.full((first.shape[0], second.shape[0]), np.nan)
    np.divide(
        first @ second.T,
        denominators,
        out=similarity_matrix,
        where=denominators > 0,
    )
    return np.clip(similarity_matrix, -1, 1)
# EOF


"""
raw_cross_temporal_similarity
Average stimulus-wise raw response correlations across every pair of timepoints.

INPUT:
    - first: np.ndarray -> channels x first time x stimuli
    - second: np.ndarray -> channels x second time x stimuli
    - return_stimulus_std: bool -> also return the SD across stimulus correlations

OUTPUT:
    - similarity_matrix: np.ndarray -> first time x second time
    - stimulus_std: np.ndarray -> optional SD across stimuli at each time pair
"""
def raw_cross_temporal_similarity(
        first: np.ndarray,
        second: np.ndarray,
        return_stimulus_std: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.ndim != 3 or second.ndim != 3:
        raise ValueError("Inputs must have shape channels x time x stimuli.")
    # end if input dimensions
    if first.shape[0] != second.shape[0] or first.shape[2] != second.shape[2]:
        raise ValueError("Inputs must have matching channel and stimulus axes.")
    # end if matching axes

    # Retain the stimulus-level correlations until both requested summaries exist.
    stimulus_similarities = []
    for stimulus_index in range(first.shape[2]):
        stimulus_similarities.append(
            cross_temporal_similarity(
                first[:, :, stimulus_index].T,
                second[:, :, stimulus_index].T,
            )
        )
    # end for stimulus_index
    stimulus_similarities = np.stack(stimulus_similarities)
    similarity_mean = np.nanmean(stimulus_similarities, axis=0)
    if return_stimulus_std:
        stimulus_std = np.nanstd(stimulus_similarities, axis=0)
        return similarity_mean, stimulus_std
    # end if return_stimulus_std
    return similarity_mean
# EOF


"""
compute_split_half_reliability
Repeat within-stimulus presentation splits and compute time-resolved raw or
RDM self-consistency using the same definitions as split_half_tiziano.ipynb.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations
    - presentation_identities: list[str] -> identity for every presentation
    - stimulus_order: list[str] -> matched identities retained in the analysis
    - analysis_type: str -> raw or rsa
    - rng: np.random.Generator -> random generator controlling each split
    - n_split_repeats: int -> number of random repetition splits
    - rdm_metric: str -> distance metric used to construct RDMs
    - rsa_metric: str -> correlation or spearman RDM similarity
    - feature_centering: bool -> center each feature across stimuli after averaging
    - return_stimulus_std: bool -> return raw-correlation SD across stimuli

OUTPUT:
    - reliability: np.ndarray -> split repetition x time correlations
    - stimulus_std: np.ndarray -> optional split repetition x time raw SD
"""
def compute_split_half_reliability(
        rasters: np.ndarray,
        presentation_identities: list[str],
        stimulus_order: list[str],
        analysis_type: str,
        rng: np.random.Generator,
        n_split_repeats: int,
        rdm_metric: str = "cosine_cnt",
        rsa_metric: str = "correlation",
        feature_centering: bool = False,
        return_stimulus_std: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if analysis_type not in {"raw", "rsa"}:
        raise ValueError("analysis_type must be 'raw' or 'rsa'.")
    # end if analysis_type
    if n_split_repeats < 1:
        raise ValueError("n_split_repeats must be positive.")
    # end if n_split_repeats
    if return_stimulus_std and analysis_type != "raw":
        raise ValueError(
            "Stimulus-level correlation SD is available only for raw analysis."
        )
    # end if return_stimulus_std

    reliability = []
    stimulus_stds = []
    for split_index in range(n_split_repeats):
        first_half, second_half = average_repetition_halves(
            rasters, presentation_identities, stimulus_order, rng,
        )
        if analysis_type == "raw":
            # Center features only after each repetition half has been averaged.
            if feature_centering:
                first_half = mean_centering(first_half, axis=2)
                second_half = mean_centering(second_half, axis=2)
            # end if feature_centering
            stimulus_reliability = []
            for stimulus_index in range(len(stimulus_order)):
                stimulus_reliability.append(
                    rowwise_similarity(
                        first_half[:, :, stimulus_index].T,
                        second_half[:, :, stimulus_index].T,
                    )
                )
            # end for stimulus_index
            stimulus_reliability = np.stack(stimulus_reliability)
            split_reliability = np.nanmean(stimulus_reliability, axis=0)
            if return_stimulus_std:
                stimulus_stds.append(np.nanstd(stimulus_reliability, axis=0))
            # end if return_stimulus_std
        else:
            first_rdms = compute_rdm_timeseries(first_half, rdm_metric)
            second_rdms = compute_rdm_timeseries(second_half, rdm_metric)
            split_reliability = rowwise_similarity(
                first_rdms, second_rdms, metric=rsa_metric,
            )
        # end if analysis_type
        reliability.append(split_reliability)
    # end for split_index
    reliability = np.stack(reliability)
    if return_stimulus_std:
        return reliability, np.stack(stimulus_stds)
    # end if return_stimulus_std
    return reliability
# EOF
