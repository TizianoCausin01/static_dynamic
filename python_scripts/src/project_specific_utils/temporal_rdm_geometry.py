import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from .split_half_rsa import compute_rdm_timeseries, cross_temporal_similarity


"""
window_rdm_timeseries
Compute the stimulus RDM at every sample of one time window.

INPUT:
    - rasters: np.ndarray -> features x time x stimuli
    - times_ms: np.ndarray -> time of every sample
    - window_ms: tuple[float, float] -> inclusive start and end
    - rdm_metric: str -> create_RDM distance

OUTPUT:
    - rdm_timeseries: np.ndarray -> window time x stimulus-pair distances
    - window_times_ms: np.ndarray -> times of the kept samples
"""
def window_rdm_timeseries(rasters, times_ms, window_ms, rdm_metric="cosine_cnt"):
    # Half-sample tolerance so float sample times on the edges are kept.
    tolerance = np.median(np.diff(times_ms)) / 2
    kept = (times_ms >= window_ms[0] - tolerance) & (times_ms <= window_ms[1] + tolerance)
    rdm_timeseries = compute_rdm_timeseries(
        np.asarray(rasters[:, kept, :], dtype=np.float64), rdm_metric,
    )
    return rdm_timeseries, times_ms[kept]
# EOF


"""
temporal_rdm_matrix
Second-order geometry: similarity between the RDMs of every pair of timepoints.

INPUT:
    - rdm_timeseries: np.ndarray -> time x stimulus-pair distances
    - rsa_metric: str -> correlation or spearman between RDMs

OUTPUT:
    - matrix: np.ndarray -> time x time RDM similarity
"""
def temporal_rdm_matrix(rdm_timeseries, rsa_metric="correlation"):
    return cross_temporal_similarity(rdm_timeseries, rdm_timeseries, metric=rsa_metric)
# EOF


"""
compare_temporal_matrices
Spearman-correlate the off-diagonal upper triangles of two time x time
matrices, raw and after partialling out the temporal distance |t1 - t2|.

The partial correlation asks whether two geometries agree beyond the generic
fact that nearby timepoints have similar RDMs.

INPUT:
    - first: np.ndarray -> time x time matrix
    - second: np.ndarray -> time x time matrix of the same size

OUTPUT:
    - r: float -> Spearman correlation
    - partial_r: float -> Spearman partial correlation controlling for |dt|
"""
def compare_temporal_matrices(first, second):
    rows, columns = np.triu_indices(first.shape[0], k=1)
    ranks = [
        rankdata(values) for values in
        (first[rows, columns], second[rows, columns], np.abs(rows - columns))
    ]
    # Rank-transform, then residualize both matrices on the time-distance ranks.
    first_ranks, second_ranks, distance_ranks = [
        values - values.mean() for values in ranks
    ]
    r = np.corrcoef(first_ranks, second_ranks)[0, 1]
    residuals = [
        values - distance_ranks * (values @ distance_ranks) / (distance_ranks @ distance_ranks)
        for values in (first_ranks, second_ranks)
    ]
    partial_r = np.corrcoef(*residuals)[0, 1]
    return r, partial_r
# EOF


"""
square_rdm_stack
Convert a time x pairs RDM timeseries into time x stimuli x stimuli.

INPUT:
    - rdm_timeseries: np.ndarray -> time x stimulus-pair distances

OUTPUT:
    - square_rdms: np.ndarray -> time x stimuli x stimuli distances
"""
def square_rdm_stack(rdm_timeseries):
    return np.stack([squareform(rdm) for rdm in rdm_timeseries])
# EOF


"""
subset_rdm_timeseries
Keep only the stimulus pairs among one stimulus subset.

INPUT:
    - square_rdms: np.ndarray -> time x stimuli x stimuli distances
    - kept_stimuli: np.ndarray -> sorted stimulus indices to keep

OUTPUT:
    - rdm_timeseries: np.ndarray -> time x kept-pair distances
"""
def subset_rdm_timeseries(square_rdms, kept_stimuli):
    rows, columns = np.triu_indices(kept_stimuli.size, k=1)
    return square_rdms[:, kept_stimuli[rows], kept_stimuli[columns]]
# EOF
