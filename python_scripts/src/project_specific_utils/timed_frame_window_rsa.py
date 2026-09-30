from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from pathlib import Path

from useful_stuff.general_utils import create_RDM

from .dataloader import load_raster
from .frame_similarity_latency import half_open_window_mean
from .last_frame_decay import resample_array
from .split_half_rsa import average_repetition_halves, compute_split_half_reliability


"""
window_rdm
Average a half-open time window, center every channel across stimuli, then
build the stimulus RDM. The order (average -> center -> RDM) is deliberate:
centering is applied to the window average, not to single time samples.

INPUT:
    - rasters: np.ndarray -> channels x time x stimuli values
    - fs: float -> sampling frequency of the time axis in Hz
    - window_ms: tuple[float, float] -> half-open averaging window in ms
    - rdm_metric: str -> distance metric accepted by create_RDM

OUTPUT:
    - rdm_vector: np.ndarray -> condensed stimulus-pair distances
"""
def window_rdm(rasters, fs, window_ms, rdm_metric="cosine"):
    # channels x stimuli, each channel centered across stimuli after averaging.
    window_patterns = half_open_window_mean(rasters, fs, window_ms, center=True)
    return create_RDM(np.ascontiguousarray(window_patterns), metric=rdm_metric)
# EOF


"""
rdm_similarity
Pearson or Spearman correlation between two condensed RDMs.

INPUT:
    - first_rdm: np.ndarray -> condensed stimulus-pair distances
    - second_rdm: np.ndarray -> condensed stimulus-pair distances
    - similarity_metric: str -> pearson or spearman

OUTPUT:
    - r: float -> RDM correlation
"""
def rdm_similarity(first_rdm, second_rdm, similarity_metric="spearman"):
    if similarity_metric == "spearman":
        first_rdm, second_rdm = rankdata(first_rdm), rankdata(second_rdm)
    elif similarity_metric != "pearson":
        raise ValueError("similarity_metric must be 'pearson' or 'spearman'.")
    # end if similarity_metric
    return np.corrcoef(first_rdm, second_rdm)[0, 1]
# EOF


"""
bootstrap_rdm_similarity
Resample stimuli with replacement and recompute the correlation between one
target RDM and several other RDMs on the same resampled stimulus set. Pairs of
a stimulus with its own copy are dropped because their distance is zero.

INPUT:
    - target_rdm: np.ndarray -> condensed RDM every other RDM is compared with
    - rdms: list[np.ndarray] -> condensed RDMs to compare with target_rdm
    - n_bootstraps: int -> number of stimulus resamples
    - similarity_metric: str -> pearson or spearman
    - seed: int -> random seed

OUTPUT:
    - bootstrap_r: np.ndarray -> n_bootstraps x len(rdms) correlations
"""
def bootstrap_rdm_similarity(
        target_rdm, rdms, n_bootstraps=1000, similarity_metric="spearman", seed=0,
        ):
    rng = np.random.default_rng(seed)
    target_square = squareform(target_rdm)
    square_rdms = [squareform(rdm) for rdm in rdms]
    n_stimuli = target_square.shape[0]
    upper_triangle = np.triu(np.ones((n_stimuli, n_stimuli), dtype=bool), k=1)
    bootstrap_r = np.full((n_bootstraps, len(rdms)), np.nan)
    for bootstrap_index in range(n_bootstraps):
        stimulus_indices = rng.integers(0, n_stimuli, n_stimuli)
        # Keep upper-triangle pairs made of two distinct original stimuli.
        pair_mask = upper_triangle & (
            stimulus_indices[:, np.newaxis] != stimulus_indices[np.newaxis, :]
        )
        resampled_target = target_square[np.ix_(stimulus_indices, stimulus_indices)][pair_mask]
        for rdm_index, square_rdm in enumerate(square_rdms):
            resampled_rdm = square_rdm[np.ix_(stimulus_indices, stimulus_indices)][pair_mask]
            bootstrap_r[bootstrap_index, rdm_index] = rdm_similarity(
                resampled_target, resampled_rdm, similarity_metric,
            )
        # end for rdm_index
    # end for bootstrap_index
    return bootstrap_r
# EOF


"""
compute_timed_frame_window_rsa
RSA between window-averaged static frame responses (2000 ms, 2250 ms, last
frame) and one window-averaged movie response. The movie is also compared with
itself: its RDM in the same static window shifted to every frame onset
(frame time + static window) is correlated with the movie target-window RDM.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - static_window_ms: tuple[float, float] -> half-open window after static onset
    - movie_window_ms: tuple[float, float] -> half-open movie target window
    - frame_times_ms: dict[str, float] -> movie time of every static frame
    - rdm_metric: str -> distance metric accepted by create_RDM
    - similarity_metric: str -> pearson or spearman RDM correlation
    - n_bootstraps: int -> stimulus bootstrap resamples (0 skips the bootstrap)
    - seed: int -> bootstrap random seed

OUTPUT:
    - rsa: dict -> RDMs, static/movie correlations, bootstraps, and the full
        correlation matrix among all window RDMs
"""
def compute_timed_frame_window_rsa(
        data,
        static_window_ms=(80, 250),
        movie_window_ms=(2580, 2750),
        frame_times_ms=None,
        rdm_metric="cosine",
        similarity_metric="spearman",
        n_bootstraps=1000,
        seed=0,
        ):
    if frame_times_ms is None:
        frame_times_ms = {"2000ms": 2000, "2250ms": 2250, "last_frame": 2500}
    # end if frame_times_ms is None
    fs = data["neural_fs"]
    frame_names = list(frame_times_ms)

    # One RDM per static frame, from the static_window_ms average after onset.
    static_rdms = {
        frame: window_rdm(data[f"static_{frame}"], fs, static_window_ms, rdm_metric)
        for frame in frame_names
    }
    # Movie target RDM and the movie RDMs in the frame-aligned windows.
    movie_rdm = window_rdm(data["dynamic"], fs, movie_window_ms, rdm_metric)
    aligned_movie_windows = {
        frame: tuple(frame_times_ms[frame] + time_ms for time_ms in static_window_ms)
        for frame in frame_names
    }
    aligned_movie_rdms = {
        frame: window_rdm(data["dynamic"], fs, aligned_movie_windows[frame], rdm_metric)
        for frame in frame_names
    }

    static_movie_r = np.array([
        rdm_similarity(static_rdms[frame], movie_rdm, similarity_metric)
        for frame in frame_names
    ])
    movie_movie_r = np.array([
        rdm_similarity(aligned_movie_rdms[frame], movie_rdm, similarity_metric)
        for frame in frame_names
    ])

    # Full matrix among all window RDMs (static frames, then aligned movie windows).
    matrix_labels = (
        [f"static {frame}" for frame in frame_names]
        + [f"movie {start:g}-{end:g}" for start, end in aligned_movie_windows.values()]
    )
    matrix_rdms = list(static_rdms.values()) + list(aligned_movie_rdms.values())
    similarity_matrix = np.array([
        [rdm_similarity(first, second, similarity_metric) for second in matrix_rdms]
        for first in matrix_rdms
    ])

    static_bootstrap_r = movie_bootstrap_r = None
    if n_bootstraps > 0:
        # Static and movie comparisons share the same resamples via the same seed.
        static_bootstrap_r = bootstrap_rdm_similarity(
            movie_rdm, list(static_rdms.values()), n_bootstraps, similarity_metric, seed,
        )
        movie_bootstrap_r = bootstrap_rdm_similarity(
            movie_rdm, list(aligned_movie_rdms.values()), n_bootstraps, similarity_metric, seed,
        )
    # end if n_bootstraps

    return {
        "frame_names": frame_names,
        "frame_times_ms": frame_times_ms,
        "static_rdms": static_rdms,
        "movie_rdm": movie_rdm,
        "aligned_movie_windows": aligned_movie_windows,
        "aligned_movie_rdms": aligned_movie_rdms,
        "static_movie_r": static_movie_r,
        "movie_movie_r": movie_movie_r,
        "static_bootstrap_r": static_bootstrap_r,
        "movie_bootstrap_r": movie_bootstrap_r,
        "matrix_labels": matrix_labels,
        "similarity_matrix": similarity_matrix,
        "static_window_ms": tuple(static_window_ms),
        "movie_window_ms": tuple(movie_window_ms),
        "rdm_metric": rdm_metric,
        "similarity_metric": similarity_metric,
    }
# EOF


"""
spearman_brown
Spearman-Brown prophecy for doubling the data: predicts the full-data
correlation from a correlation between two half-data estimates.

INPUT:
    - half_r: np.ndarray | float -> correlations between half-data estimates

OUTPUT:
    - full_r: np.ndarray | float -> 2r / (1 + r)
"""
def spearman_brown(half_r):
    return 2 * half_r / (1 + half_r)
# EOF

"""
load_dynamic_presentations
Load single movie presentations with the same channels and stimuli as the
averaged data, cropped to end_ms and resampled to the analysis rate.

INPUT:
    - dynamic_raster_path: str | Path -> presentation-level movie raster file
    - data: dict -> output of load_frame_similarity_rasters
    - end_ms: float -> last movie time needed; a margin is added before resampling
    - source_fs: float -> neural source sampling frequency in Hz
    - margin_ms: float -> extra time loaded past end_ms to avoid resampling edges

OUTPUT:
    - presentations: np.ndarray -> channels x time x presentations at data["neural_fs"]
    - identities: list[str] -> stimulus identity of every presentation
"""
def load_dynamic_presentations(
        dynamic_raster_path, data, end_ms, source_fs=1000, margin_ms=100,
        ):
    end_sample = int(round((end_ms + margin_ms) * source_fs / 1000))
    # Zero-based indices of the channels already retained in the averaged data.
    presentations, names = load_raster(
        dynamic_raster_path,
        channel_slice=np.asarray(data["channel_numbers"]) - 1,
        end_sample=end_sample,
    )
    # Movie names are vid_<identity>.<ext>; keep only the analysed stimuli.
    stimulus_set = set(data["stimuli"])
    identities = [Path(name).stem.removeprefix("vid_") for name in names]
    keep = np.asarray([identity in stimulus_set for identity in identities])
    presentations = resample_array(presentations[:, :, keep], source_fs, data["neural_fs"])
    identities = [identity for identity, kept in zip(identities, keep) if kept]
    return presentations, identities
# EOF


"""
load_or_compute_movie_split_half_reliability
Split-half reliability of the movie RDM at every movie bin, cached on disk.
The presentation-level raster (tens of GB for Neuropixels) is read only when
no cached result matches the requested channels, stimuli and settings. The
cache keeps the uncorrected correlations, so any Spearman-Brown correction is
applied by the caller.

INPUT:
    - cache_path: str | Path -> .npz file holding (or receiving) the result
    - dynamic_raster_path: str | Path -> presentation-level movie raster file
    - channel_numbers: np.ndarray -> one-based MATLAB channels to keep
    - stimuli: list[str] -> candidate stimuli; those with >= 2 repetitions are used
    - neural_fs: float -> analysis sampling frequency in Hz
    - end_ms: float -> last movie time loaded
    - source_fs: float -> neural source sampling frequency in Hz
    - n_split_repeats: int -> number of random repetition splits
    - seed: int -> split random seed
    - rdm_metric: str -> neural RDM metric
    - rsa_metric: str -> half-vs-half RDM similarity metric

OUTPUT:
    - reliability_splits: np.ndarray -> splits x movie time uncorrected RDM correlations
    - reliability_stimuli: list[str] -> stimuli with at least two repetitions
    - n_presentations: int -> presentations of the candidate stimuli
"""
def load_or_compute_movie_split_half_reliability(
        cache_path, dynamic_raster_path, channel_numbers, stimuli, neural_fs,
        end_ms, source_fs, n_split_repeats, seed, rdm_metric, rsa_metric,
        ):
    cache_path = Path(cache_path)
    settings = {
        "channel_numbers": np.asarray(channel_numbers, dtype=int),
        "stimuli": np.asarray(stimuli, dtype=str),
        "numeric_settings": np.asarray(
            [neural_fs, end_ms, source_fs, n_split_repeats, seed], dtype=float,
        ),
        "metrics": np.asarray([rdm_metric, rsa_metric], dtype=str),
    }
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            # Reuse the cache only if every setting that shapes the result matches.
            matches = all(
                name in archive.files and np.array_equal(archive[name], value)
                for name, value in settings.items()
            )
            if matches:
                return (
                    archive["reliability_splits"],
                    archive["reliability_stimuli"].tolist(),
                    int(archive["n_presentations"]),
                )
            # end if matches
        # end with np.load
        print(f"{cache_path.name}: cached settings differ, recomputing.")
    # end if cache_path.is_file()

    # channels x time x presentations at neural_fs, restricted to the stimuli.
    presentations, identities = load_dynamic_presentations(
        dynamic_raster_path,
        {"channel_numbers": channel_numbers, "stimuli": stimuli, "neural_fs": neural_fs},
        end_ms, source_fs=source_fs, margin_ms=0,
    )
    repetition_counts = Counter(identities)
    reliability_stimuli = [
        stimulus for stimulus in stimuli if repetition_counts[stimulus] >= 2
    ]
    reliability_splits = compute_split_half_reliability(
        presentations, identities, reliability_stimuli, "rsa",
        np.random.default_rng(seed), n_split_repeats,
        rdm_metric=rdm_metric, rsa_metric=rsa_metric,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        reliability_splits=reliability_splits,
        reliability_stimuli=np.asarray(reliability_stimuli, dtype=str),
        n_presentations=len(identities),
        **settings,
    )
    return reliability_splits, reliability_stimuli, len(identities)
# EOF


"""
compute_dynamic_split_half_window_rsa
Split the movie repetitions of every stimulus into two random halves. The RDM
of every frame-aligned movie window in one half is correlated with the movie
target-window RDM of the other half, in both directions (A->B, B->A) and then
averaged. Unlike the within-average comparison, the target-window bar is the
split-half reliability of the target RDM instead of 1.

Optional corrections for the half-data attenuation:
    - "spearman_brown": r -> 2r / (1 + r) on every bar (exact only for the
      target bar, approximate when the two windows differ in reliability);
    - "noise_ceiling": r / sqrt(rel_frame * rel_target), where rel is the
      half-vs-half RDM correlation of each window (classic disattenuation;
      both numerator and denominator are half-data, so no Spearman-Brown).
      The target bar becomes 1; values are NaN when a reliability is <= 0.

INPUT:
    - presentations: np.ndarray -> channels x time x presentations
    - identities: list[str] -> stimulus identity of every presentation
    - rsa: dict -> output of compute_timed_frame_window_rsa (windows and metrics)
    - stimuli: list[str] -> stimulus order of the averaged data
    - fs: float -> sampling frequency of the presentations in Hz
    - n_split_repeats: int -> number of random repetition splits
    - seed: int -> split random seed
    - correction: str | None -> None, "spearman_brown", or "noise_ceiling"

OUTPUT:
    - split_rsa: dict -> raw and (optionally) corrected split x frame
        correlations, window reliabilities, and the mean of the plotted values
"""
def compute_dynamic_split_half_window_rsa(
        presentations, identities, rsa, stimuli, fs, n_split_repeats=100, seed=0,
        correction=None,
        ):
    if correction not in (None, "spearman_brown", "noise_ceiling"):
        raise ValueError("correction must be None, 'spearman_brown', or 'noise_ceiling'.")
    # end if correction
    rng = np.random.default_rng(seed)
    frame_names = rsa["frame_names"]
    rdm_metric, similarity_metric = rsa["rdm_metric"], rsa["similarity_metric"]
    split_r = np.full((n_split_repeats, len(frame_names)), np.nan)
    # Half-vs-half RDM correlation of the same window (uncorrected reliability).
    frame_half_r = np.full((n_split_repeats, len(frame_names)), np.nan)
    target_half_r = np.full(n_split_repeats, np.nan)
    for split_index in range(n_split_repeats):
        # Two channels x time x stimuli averages from disjoint repetitions.
        halves = average_repetition_halves(presentations, identities, list(stimuli), rng)
        target_rdms = [window_rdm(half, fs, rsa["movie_window_ms"], rdm_metric) for half in halves]
        target_half_r[split_index] = rdm_similarity(*target_rdms, similarity_metric)
        for frame_index, frame in enumerate(frame_names):
            aligned_rdms = [
                window_rdm(half, fs, rsa["aligned_movie_windows"][frame], rdm_metric)
                for half in halves
            ]
            # Aligned window of one half vs target window of the other half.
            split_r[split_index, frame_index] = np.mean([
                rdm_similarity(aligned_rdms[0], target_rdms[1], similarity_metric),
                rdm_similarity(aligned_rdms[1], target_rdms[0], similarity_metric),
            ])
            frame_half_r[split_index, frame_index] = rdm_similarity(*aligned_rdms, similarity_metric)
        # end for frame_index
    # end for split_index

    plotted_r = split_r
    if correction == "spearman_brown":
        plotted_r = spearman_brown(split_r)
    elif correction == "noise_ceiling":
        # Half-data reliabilities of both windows; <= 0 leaves the ratio undefined.
        ceiling = frame_half_r * target_half_r[:, np.newaxis]
        plotted_r = np.full_like(split_r, np.nan)
        np.divide(split_r, np.sqrt(np.clip(ceiling, 0, None)), out=plotted_r, where=ceiling > 0)
    # end if correction
    return {
        "raw_split_r": split_r,
        "split_r": plotted_r,
        "mean_r": np.nanmean(plotted_r, axis=0),
        "frame_reliability": spearman_brown(frame_half_r),
        "target_reliability": spearman_brown(target_half_r),
        "correction": correction,
    }
# EOF


"""
sliding_movie_window_rsa
Slide a movie window of fixed width over the movie; at every position the
window is averaged, centered, turned into an RDM, and correlated with every
reference RDM.

INPUT:
    - data: dict -> output of load_frame_similarity_rasters
    - reference_rdms: dict[str, np.ndarray] -> condensed RDMs to compare with
    - window_width_ms: float -> width of every sliding movie window
    - step_ms: float -> distance between consecutive window starts
    - rdm_metric: str -> distance metric accepted by create_RDM
    - similarity_metric: str -> pearson or spearman

OUTPUT:
    - window_centers_ms: np.ndarray -> center time of every movie window
    - curves: dict[str, np.ndarray] -> RDM correlation per window and reference
"""
def sliding_movie_window_rsa(
        data, reference_rdms, window_width_ms, step_ms=10,
        rdm_metric="cosine", similarity_metric="spearman",
        ):
    fs = data["neural_fs"]
    movie_duration_ms = data["dynamic"].shape[1] * 1000 / fs
    window_starts_ms = np.arange(0, movie_duration_ms - window_width_ms + 1e-9, step_ms)
    curves = {name: np.full(window_starts_ms.size, np.nan) for name in reference_rdms}
    for window_index, start_ms in enumerate(window_starts_ms):
        movie_rdm = window_rdm(
            data["dynamic"], fs, (start_ms, start_ms + window_width_ms), rdm_metric,
        )
        for name, reference_rdm in reference_rdms.items():
            curves[name][window_index] = rdm_similarity(
                reference_rdm, movie_rdm, similarity_metric,
            )
        # end for name
    # end for window_index
    return window_starts_ms + window_width_ms / 2, curves
# EOF


"""
plot_timed_frame_window_rsa
Bars of the static-frame vs movie-window RSA and of the frame-aligned movie
vs movie-window RSA (95% stimulus-bootstrap intervals), optionally the
split-half version of the movie comparison (95% range across splits), plus
the correlation matrix among all window RDMs.

INPUT:
    - rsa: dict -> output of compute_timed_frame_window_rsa
    - title: str -> figure title
    - split_rsa: dict | None -> output of compute_dynamic_split_half_window_rsa

OUTPUT:
    - figure: matplotlib.figure.Figure -> three- or four-panel figure
    - axes: np.ndarray -> corresponding axes
"""
def plot_timed_frame_window_rsa(rsa, title="", split_rsa=None):
    frame_names = rsa["frame_names"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(frame_names)))
    n_bar_panels = 2 if split_rsa is None else 3
    figure, axes = plt.subplots(
        1, n_bar_panels + 1, figsize=(5 * n_bar_panels + 5, 4.2),
        gridspec_kw={"width_ratios": [1] * n_bar_panels + [1.3]},
    )
    bar_panels = [
        (axes[0], rsa["static_movie_r"], rsa["static_bootstrap_r"],
         [f"static {frame}\n{rsa['static_window_ms']}" for frame in frame_names],
         f"Static frame vs movie {rsa['movie_window_ms']} ms"),
        (axes[1], rsa["movie_movie_r"], rsa["movie_bootstrap_r"],
         [f"movie\n{rsa['aligned_movie_windows'][frame]}" for frame in frame_names],
         f"Movie (frame-aligned) vs movie {rsa['movie_window_ms']} ms"),
    ]
    if split_rsa is not None:
        # Errors span the random-split distribution (same percentiles as the bootstrap).
        bar_panels.append(
            (axes[2], split_rsa["mean_r"], split_rsa["split_r"],
             [f"movie split\n{rsa['aligned_movie_windows'][frame]}" for frame in frame_names],
             f"Split-half movie vs other-half movie {rsa['movie_window_ms']} ms"
             + (f"\n({split_rsa['correction']} corrected)" if split_rsa["correction"] else "")),
        )
    # end if split_rsa
    for axis, r_values, bootstrap_r, labels, panel_title in bar_panels:
        error = None
        if bootstrap_r is not None:
            # Percentile interval converted into asymmetric bar errors.
            low, high = np.nanpercentile(bootstrap_r, [2.5, 97.5], axis=0)
            error = np.vstack([r_values - low, high - r_values])
        # end if bootstrap_r
        axis.bar(np.arange(len(r_values)), r_values, yerr=error, color=colors, capsize=4)
        for bar_index, r_value in enumerate(r_values):
            axis.text(bar_index, r_value / 2, f"{r_value:.2f}", ha="center", color="w", fontsize=9)
        # end for bar_index
        axis.set_xticks(np.arange(len(r_values)), labels, fontsize=8)
        axis.set(ylabel=f"RDM {rsa['similarity_metric']} r", title=panel_title)
        axis.axhline(0, color="0.3", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
    # end for axis
    # Shared y range so the bar panels are directly comparable.
    y_max = max(axes[panel].get_ylim()[1] for panel in range(n_bar_panels))
    for panel in range(n_bar_panels):
        axes[panel].set_ylim(top=y_max)
    # end for panel

    matrix_axis = axes[-1]
    matrix = rsa["similarity_matrix"]
    image = matrix_axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            matrix_axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                             color="w" if matrix[row, column] < 0.6 else "k", fontsize=8)
        # end for column
    # end for row
    ticks = np.arange(len(rsa["matrix_labels"]))
    matrix_axis.set_xticks(ticks, rsa["matrix_labels"], rotation=45, ha="right", fontsize=8)
    matrix_axis.set_yticks(ticks, rsa["matrix_labels"], fontsize=8)
    matrix_axis.set_title("RDM correlation among windows")
    figure.colorbar(image, ax=matrix_axis, fraction=0.046)
    figure.suptitle(f"{title} | {rsa['rdm_metric']} RDMs (window avg -> center -> RDM)")
    figure.tight_layout()
    return figure, axes
# EOF


"""
plot_sliding_movie_window_rsa
Plot the sliding movie-window RSA curves with frame onsets and the target window.

INPUT:
    - window_centers_ms: np.ndarray -> center of every sliding movie window
    - curves: dict[str, np.ndarray] -> RDM correlation per window and reference
    - rsa: dict -> output of compute_timed_frame_window_rsa (for times and windows)
    - title: str -> axis title
    - x_range_ms: tuple[float, float] | None -> displayed movie time range

OUTPUT:
    - figure: matplotlib.figure.Figure -> one-panel figure
    - axis: matplotlib.axes.Axes -> corresponding axis
"""
def plot_sliding_movie_window_rsa(window_centers_ms, curves, rsa, title="", x_range_ms=None):
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rsa["frame_names"])))
    figure, axis = plt.subplots(figsize=(10, 3.8))
    for color, frame in zip(colors, rsa["frame_names"]):
        axis.plot(window_centers_ms, curves[frame], color=color, linewidth=1.8, label=f"static {frame}")
        axis.axvline(rsa["frame_times_ms"][frame], color=color, linestyle="--", linewidth=1)
    # end for frame
    axis.axvspan(*rsa["movie_window_ms"], color="tab:orange", alpha=0.15, label="movie target window")
    axis.axhline(0, color="0.3", linewidth=0.8)
    if x_range_ms is not None:
        axis.set_xlim(x_range_ms)
    # end if x_range_ms
    axis.set(
        xlabel="Movie window center (ms from movie onset)",
        ylabel=f"RDM {rsa['similarity_metric']} r",
        title=title,
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure, axis
# EOF
