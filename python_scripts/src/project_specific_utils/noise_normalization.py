"""
Noise normalization (whitening) for Mahalanobis RDMs, following
static_dynamic_mahalanobis_rsa.ipynb.

At every timepoint, repetition noise is each presentation's deviation from its
stimulus mean; pooled over stimuli it gives a channel x channel noise
covariance Sigma_t, shrunk toward isotropic noise. Whitening the stimulus means
with Sigma_t^(-1/2) turns squared Euclidean distances into squared Mahalanobis
distances. Static and movie responses get separate noise models.
"""

from pathlib import Path

import numpy as np

from .split_half_rsa import average_presentations
from .timed_frame_window_rsa import load_dynamic_presentations, load_static_presentations


"""
mahalanobis_metric_token
File-name token of a Mahalanobis RDM with a given covariance shrinkage, used
in place of the RDM metric name in saved results.

INPUT:
    - shrinkage: float -> covariance weight given to isotropic noise

OUTPUT:
    - token: str -> e.g. "mahalanobis-shrink0.1"
"""
def mahalanobis_metric_token(shrinkage: float) -> str:
    return f"mahalanobis-shrink{shrinkage:g}"
# EOF


"""
noise_whitener_cache_path
Cache file of the noise whiteners of one session and condition, shared by the
permutation runner and the notebooks.

INPUT:
    - results_dir: str | Path -> project results directory
    - session_name: str -> raster session the noise is estimated from
    - condition: str -> "movie" or "static"
    - shrinkage: float -> covariance shrinkage
    - neural_fs: float -> analysis sampling frequency in Hz

OUTPUT:
    - cache_path: Path -> .npz file of the whiteners
"""
def noise_whitener_cache_path(results_dir, session_name, condition, shrinkage, neural_fs):
    return (
        Path(results_dir) / "noise_whiteners" / session_name
        / f"{condition}_{mahalanobis_metric_token(shrinkage)}_{neural_fs:g}Hz.npz"
    )
# EOF


"""
noise_whiteners
Time-resolved within-stimulus noise covariance and its inverse square root.

INPUT:
    - presentations: np.ndarray -> channels x time x presentations
    - identities: list[str] -> stimulus identity of every presentation
    - stimulus_order: list[str] -> stimuli whose repetitions define the noise
    - shrinkage: float -> covariance weight given to isotropic noise (0-1)
    - eigenvalue_floor_ratio: float -> minimum eigenvalue relative to the mean

OUTPUT:
    - whiteners: np.ndarray -> time x channels x channels symmetric Sigma_t^(-1/2)
    - mean_noise_variances: np.ndarray -> mean covariance eigenvalue per time
    - condition_numbers: np.ndarray -> regularized covariance conditioning per time
"""
def noise_whiteners(
        presentations, identities, stimulus_order, shrinkage=0.1,
        eigenvalue_floor_ratio=1e-8,
        ):
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1].")
    # end if invalid shrinkage
    presentations = np.asarray(presentations, dtype=np.float64)
    identity_to_index = {identity: index for index, identity in enumerate(stimulus_order)}
    kept = np.asarray([identity in identity_to_index for identity in identities])
    presentations = presentations[:, :, kept]
    stimulus_indices = np.asarray([
        identity_to_index[identity] for identity, keep in zip(identities, kept) if keep
    ])
    repetition_counts = np.bincount(stimulus_indices, minlength=len(stimulus_order))
    if np.any(repetition_counts < 2):
        raise ValueError("Every stimulus needs at least two repetitions.")
    # end if too few repetitions
    degrees_of_freedom = presentations.shape[2] - len(stimulus_order)

    # channels x time x presentations deviations from each stimulus mean.
    stimulus_means = average_presentations(
        presentations, [stimulus_order[index] for index in stimulus_indices],
        stimulus_order,
    )
    residuals = presentations - stimulus_means[:, :, stimulus_indices]
    # time x channels x channels pooled within-stimulus covariance.
    covariances = np.einsum("itp,jtp->tij", residuals, residuals) / degrees_of_freedom

    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    eigenvalues = np.maximum(eigenvalues, 0)
    mean_noise_variances = eigenvalues.mean(axis=1)
    if np.any(~np.isfinite(mean_noise_variances) | (mean_noise_variances <= 0)):
        raise ValueError("The noise covariance is degenerate at some timepoint.")
    # end if degenerate covariance
    # Shrink toward the mean eigenvalue: keeps the noise scale, stabilizes the inverse.
    regularized = (1 - shrinkage) * eigenvalues + shrinkage * mean_noise_variances[:, None]
    regularized = np.maximum(regularized, eigenvalue_floor_ratio * mean_noise_variances[:, None])
    whiteners = np.einsum(
        "tik,tk,tjk->tij", eigenvectors, 1 / np.sqrt(regularized), eigenvectors,
    )
    condition_numbers = regularized[:, -1] / regularized[:, 0]
    return whiteners, mean_noise_variances, condition_numbers
# EOF


"""
load_or_compute_noise_whiteners
Noise whiteners of the movie or of the last-frame image response, cached on
disk. The presentation-level raster is read only when no cached result matches
the requested channels, stimuli and settings.

INPUT:
    - cache_path: str | Path -> .npz file holding (or receiving) the whiteners
    - raster_path: str | Path -> presentation-level movie or image raster file
    - channel_numbers: np.ndarray -> one-based MATLAB channels to keep
    - stimuli: list[str] -> stimuli defining the noise (>= 2 repetitions each)
    - neural_fs: float -> analysis sampling frequency in Hz
    - end_ms: float -> last movie time loaded, or static crop length
    - source_fs: float -> neural source sampling frequency in Hz
    - shrinkage: float -> covariance weight given to isotropic noise
    - condition: str -> "movie" (vid_ presentations) or "static" (last frame)

OUTPUT:
    - whiteners: np.ndarray -> time x channels x channels
    - diagnostics: dict -> mean noise variance and condition number per time
"""
def load_or_compute_noise_whiteners(
        cache_path, raster_path, channel_numbers, stimuli, neural_fs, end_ms,
        source_fs, shrinkage, condition="movie",
        ):
    if condition not in ("movie", "static"):
        raise ValueError("condition must be 'movie' or 'static'.")
    # end if invalid condition
    cache_path = Path(cache_path)
    settings = {
        "channel_numbers": np.asarray(channel_numbers, dtype=int),
        "stimuli": np.asarray(stimuli, dtype=str),
        "numeric_settings": np.asarray(
            [neural_fs, end_ms, source_fs, shrinkage], dtype=float,
        ),
        "condition": np.asarray(condition),
    }
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            # Reuse the cache only if every setting that shapes the result matches.
            matches = all(
                name in archive.files and np.array_equal(archive[name], value)
                for name, value in settings.items()
            )
            if matches:
                return archive["whiteners"], {
                    "mean_noise_variances": archive["mean_noise_variances"],
                    "condition_numbers": archive["condition_numbers"],
                }
            # end if matches
        # end with np.load
        print(f"{cache_path.name}: cached settings differ, recomputing.")
    # end if cache_path.is_file()

    # channels x time x presentations at neural_fs, restricted to the stimuli.
    if condition == "movie":
        presentations, identities = load_dynamic_presentations(
            raster_path,
            {"channel_numbers": channel_numbers, "stimuli": stimuli, "neural_fs": neural_fs},
            end_ms, source_fs=source_fs, margin_ms=0,
        )
    else:
        presentations, identities = load_static_presentations(
            raster_path, channel_numbers, stimuli, neural_fs, end_ms, source_fs=source_fs,
        )
    # end if condition
    whiteners, mean_noise_variances, condition_numbers = noise_whiteners(
        presentations, identities, list(stimuli), shrinkage=shrinkage,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, whiteners=whiteners, mean_noise_variances=mean_noise_variances,
        condition_numbers=condition_numbers, **settings,
    )
    return whiteners, {
        "mean_noise_variances": mean_noise_variances,
        "condition_numbers": condition_numbers,
    }
# EOF
