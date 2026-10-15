"""
Kinematics of RDM trajectories in the full stimulus-pair space: step (path)
lengths and velocity vectors, and the cross-temporal alignment of the
velocities of two conditions.
"""

import numpy as np
from scipy.signal import savgol_filter


"""
normalize_rdm_timeseries
Optionally z-score every timepoint's RDM across stimulus pairs, so that path
lengths and velocities ignore changes in the overall scale of the distances.

INPUT:
    - rdms: np.ndarray -> time x stimulus-pair distances
    - normalization: str -> 'none' keeps the raw distances; 'zscore' removes each
        RDM's mean and divides by its standard deviation

OUTPUT:
    - normalized_rdms: np.ndarray -> time x stimulus-pair distances
"""
def normalize_rdm_timeseries(rdms, normalization="none"):
    if normalization == "none":
        return np.asarray(rdms, dtype=np.float64)
    elif normalization == "zscore":
        rdms = np.asarray(rdms, dtype=np.float64)
        return (rdms - rdms.mean(axis=1, keepdims=True)) / rdms.std(axis=1, keepdims=True)
    else:
        raise ValueError("normalization must be 'none' or 'zscore'.")
    # end if normalization
# EOF


"""
rdm_step_lengths
Euclidean length of every step between consecutive RDMs; their sum is the
path length of the RDM trajectory.

INPUT:
    - rdms: np.ndarray -> time x stimulus-pair distances

OUTPUT:
    - step_lengths: np.ndarray -> (time - 1) step lengths; step k joins bins k and k + 1
"""
def rdm_step_lengths(rdms):
    return np.linalg.norm(np.diff(rdms, axis=0), axis=1)
# EOF


"""
rdm_velocity
Velocity vector of the RDM trajectory at every bin, in distance units per bin.

'forward' is the plain difference R(t + 1) - R(t): it sits between two bins
(half a bin late on the bin grid) and carries the full noise of both bins.
'central' is (R(t + 1) - R(t - 1)) / 2: it is centred on bin t, is exact for
quadratic trends (second-order accurate), and averages out part of the noise;
the two edge bins fall back to one-sided differences.
'savgol' is the slope of a local polynomial (Savitzky-Golay) fitted over
window_bins bins: centred, and the most noise-robust at the cost of temporal
smoothing.

INPUT:
    - rdms: np.ndarray -> time x stimulus-pair distances
    - method: str -> 'forward', 'central' or 'savgol'
    - savgol_window_bins: int -> odd window length of the 'savgol' fit
    - savgol_order: int -> polynomial order of the 'savgol' fit

OUTPUT:
    - velocity: np.ndarray -> time x stimulus pairs ('forward': time - 1 rows)
"""
def rdm_velocity(rdms, method="central", savgol_window_bins=5, savgol_order=2):
    if method == "forward":
        return np.diff(rdms, axis=0)
    elif method == "central":
        return np.gradient(rdms, axis=0)
    elif method == "savgol":
        # 'interp' fits the polynomial to the edge window instead of padding.
        return savgol_filter(
            rdms, savgol_window_bins, savgol_order, deriv=1, axis=0, mode="interp",
        )
    else:
        raise ValueError("method must be 'forward', 'central' or 'savgol'.")
    # end if method
# EOF


"""
velocity_cosine_matrix
Cosine between every velocity of one condition and every velocity of another.

INPUT:
    - first_velocity: np.ndarray -> first-condition time x stimulus pairs
    - second_velocity: np.ndarray -> second-condition time x stimulus pairs

OUTPUT:
    - cosines: np.ndarray -> first time x second time cosine similarity
"""
def velocity_cosine_matrix(first_velocity, second_velocity):
    first_unit = first_velocity / np.linalg.norm(first_velocity, axis=1, keepdims=True)
    second_unit = second_velocity / np.linalg.norm(second_velocity, axis=1, keepdims=True)
    return first_unit @ second_unit.T
# EOF


"""
stimulus_rdm_step_lengths
Step length of every stimulus's RDM row between consecutive RDMs: the
Euclidean norm of the change of its distances to all other stimuli. Squared,
these sum (over stimuli) to twice the squared step of the full RDM, so they
split the trajectory speed into one paired observation per stimulus.

INPUT:
    - rdms: np.ndarray -> time x stimulus-pair distances, pairs in condensed
        (row-major upper-triangle) order

OUTPUT:
    - step_lengths: np.ndarray -> (time - 1) x stimuli step lengths; step k
        joins bins k and k + 1
"""
def stimulus_rdm_step_lengths(rdms):
    squared_steps = np.diff(rdms, axis=0) ** 2
    # Number of stimuli from the number of pairs: n_pairs = n (n - 1) / 2.
    n_stimuli = int(round((1 + np.sqrt(1 + 8 * squared_steps.shape[1])) / 2))
    first, second = np.triu_indices(n_stimuli, k=1)
    # pairs x stimuli incidence: pair p involves stimuli first[p] and second[p].
    incidence = np.zeros((first.size, n_stimuli))
    incidence[np.arange(first.size), first] = 1
    incidence[np.arange(first.size), second] = 1
    return np.sqrt(squared_steps @ incidence)
# EOF
