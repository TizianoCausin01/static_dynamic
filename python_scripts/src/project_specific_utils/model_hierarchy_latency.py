"""
Layer-depth versus response-latency analysis on top of the saved neural-to-model
RSA archives. One model contributes three timecourse families: the static image
response, the first-frame slice of the dynamic cross-temporal matrix, and the
last-frame slice. Each layer gets a latency, and the temporal score is the rank
correlation between normalized layer depth and that latency.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr


"""
load_model_layer_rsa
Load every saved layer of one model in registry depth order.

Files are matched on the metadata written by run_static_dynamic_neural_model_rsa
so that a directory holding several parameter families returns only the one
requested here.

INPUT:
    - results_dir: str | Path -> directory holding the saved RSA archives
    - experiment_name: str -> dynamic_vs_static result label
    - model_name: str -> model whose layers are collected
    - layer_names: list[str] -> hooked layers in shallow-to-deep order
    - expected_metadata: dict -> metadata entries every archive must match
    - channel_numbers: np.ndarray | None -> required one-based MATLAB channels

OUTPUT:
    - results: dict -> stacked RSA arrays, time axes, and the retained layers
"""
def load_model_layer_rsa(
        results_dir,
        experiment_name: str,
        model_name: str,
        layer_names: list[str],
        expected_metadata: dict,
        channel_numbers=None,
        ) -> dict:
    results_dir = Path(results_dir)
    archives_by_layer = {}
    for result_path in sorted(
            results_dir.glob(f"{experiment_name}_{model_name}_*.npz")
            ):
        with np.load(result_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if any(
                    metadata.get(key) != value
                    for key, value in expected_metadata.items()
                    ):
                continue
            # end if the archive belongs to another parameter family
            if channel_numbers is not None and not np.array_equal(
                    archive["selected_channel_numbers"].astype(int),
                    np.asarray(channel_numbers, dtype=int),
                    ):
                continue
            # end if the archive used different channels

            layer_name = str(archive["layer_name"].item())
            archives_by_layer[layer_name] = {
                "static_rsa": archive["static_rsa"].copy(),
                "dynamic_rsa": archive["dynamic_rsa"].copy(),
                "static_time_ms": archive["static_time_ms"].copy(),
                "dynamic_neural_time_ms": (
                    archive["dynamic_neural_time_ms"].copy()
                ),
                "dynamic_model_time_ms": (
                    archive["dynamic_model_time_ms"].copy()
                ),
            }
        # end with np.load
    # end for result_path

    missing_layers = [
        layer for layer in layer_names if layer not in archives_by_layer
    ]
    if missing_layers:
        raise FileNotFoundError(
            f"{model_name}: {len(missing_layers)} layers have no matching "
            f"archive in {results_dir} (first: {missing_layers[:3]})."
        )
    # end if missing_layers

    first_archive = archives_by_layer[layer_names[0]]
    for layer_name in layer_names[1:]:
        archive = archives_by_layer[layer_name]
        if (
                archive["static_rsa"].shape
                != first_archive["static_rsa"].shape
                or archive["dynamic_rsa"].shape
                != first_archive["dynamic_rsa"].shape
                ):
            raise ValueError(
                f"{model_name}: layer {layer_name} has inconsistent RSA shapes."
            )
        # end if inconsistent shapes
    # end for layer_name

    return {
        "model_name": model_name,
        "layer_names": list(layer_names),
        # Depth is normalized so models of different length share one axis.
        "layer_depths": np.linspace(0, 1, len(layer_names)),
        "static_rsa": np.stack(
            [archives_by_layer[layer]["static_rsa"] for layer in layer_names]
        ),
        "dynamic_rsa": np.stack(
            [archives_by_layer[layer]["dynamic_rsa"] for layer in layer_names]
        ),
        "static_time_ms": first_archive["static_time_ms"],
        "dynamic_neural_time_ms": first_archive["dynamic_neural_time_ms"],
        "dynamic_model_time_ms": first_archive["dynamic_model_time_ms"],
    }
# EOF


"""
timecourse_latency
Summarize when one layer's RSA timecourse peaks inside an analysis window.

Three latencies are returned. The absolute centroid reproduces the existing
project analysis; the relative centroid weights only the samples above a
fraction of that layer's own peak, which removes the dependence on overall RSA
magnitude when models are compared; the peak latency is the argmax.

INPUT:
    - similarity: np.ndarray -> RSA values on the neural time axis
    - time_ms: np.ndarray -> neural time coordinates matching similarity
    - onset_ms: float -> stimulus event the latency is measured from
    - window_ms: tuple[float, float] | None -> analysis window, None keeps all
    - absolute_cutoff: float -> minimum RSA entering the absolute centroid
    - relative_cutoff: float -> fraction of the layer peak for the relative one
    - smoothing_sigma: float | None -> Gaussian smoothing in samples

OUTPUT:
    - summary: dict -> latencies, peak/mean similarity, and the smoothed curve
"""
def timecourse_latency(
        similarity: np.ndarray,
        time_ms: np.ndarray,
        onset_ms: float = 0.0,
        window_ms=None,
        absolute_cutoff: float = 0.02,
        relative_cutoff: float = 0.5,
        smoothing_sigma=3,
        ) -> dict:
    similarity = np.asarray(similarity, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    if similarity.shape != time_ms.shape:
        raise ValueError("similarity and time_ms must have the same length.")
    # end if mismatched lengths

    # Smoothing precedes windowing so the window edges stay uncontaminated.
    if smoothing_sigma:
        similarity = gaussian_filter1d(similarity, sigma=smoothing_sigma)
    # end if smoothing_sigma

    if window_ms is None:
        window_mask = np.ones(time_ms.shape, dtype=bool)
    else:
        window_mask = (time_ms >= window_ms[0]) & (time_ms < window_ms[1])
    # end if window_ms is None
    if not window_mask.any():
        raise ValueError(f"The window {window_ms} contains no samples.")
    # end if empty window

    window_similarity = similarity[window_mask]
    window_time_ms = time_ms[window_mask]
    finite_similarity = np.where(
        np.isfinite(window_similarity), window_similarity, 0.0,
    )
    peak_similarity = float(finite_similarity.max())

    absolute_weights = np.where(
        finite_similarity > absolute_cutoff, finite_similarity, 0.0,
    )
    # The relative threshold is scale free, so weak and strong models are
    # summarized at the same point of their own response profile.
    relative_weights = np.where(
        finite_similarity > relative_cutoff * peak_similarity,
        finite_similarity, 0.0,
    ) if peak_similarity > 0 else np.zeros_like(finite_similarity)

    def weighted_latency(weights):
        if weights.sum() <= 0:
            return np.nan
        # end if no suprathreshold sample
        return float(
            np.average(window_time_ms, weights=weights) - onset_ms
        )
    # EOF

    peak_latency_ms = (
        float(window_time_ms[int(np.argmax(finite_similarity))] - onset_ms)
        if peak_similarity > 0 else np.nan
    )
    return {
        "centroid_latency_ms": weighted_latency(absolute_weights),
        "relative_centroid_latency_ms": weighted_latency(relative_weights),
        "peak_latency_ms": peak_latency_ms,
        "peak_similarity": peak_similarity,
        # Degenerate feature frames (an all-zero optical-flow frame, say) leave
        # an undefined RDM and therefore an all-NaN window.
        "mean_similarity": (
            float(np.nanmean(window_similarity))
            if np.isfinite(window_similarity).any() else np.nan
        ),
        "smoothed_similarity": similarity,
    }
# EOF


"""
rdm_noise_ceiling
Read the split-half RDM consistency and turn it into a per-analysis ceiling.

The saved consistency is the raw correlation between the RDMs of two halves of
the repetitions. Spearman-Brown corrects it to the reliability expected from
the full set of repetitions, which is the level a perfect model could reach
when compared with the all-repetition RDM used in the RSA.

INPUT:
    - split_half_path: str | Path -> NPZ written by the split-half RSA script
    - windows_ms: dict -> {analysis: (start, end)} neural windows, None for all
    - spearman_brown: bool -> apply the 2r/(1+r) correction

OUTPUT:
    - ceilings: dict -> {analysis: peak corrected consistency in its window}
"""
def rdm_noise_ceiling(
        split_half_path,
        windows_ms: dict,
        spearman_brown: bool = True,
        ) -> dict:
    with np.load(split_half_path, allow_pickle=False) as archive:
        consistency = {
            "image": (
                archive["static_rdm_split_half"], archive["static_times_ms"],
            ),
            "first_frame": (
                archive["dynamic_rdm_split_half"], archive["dynamic_times_ms"],
            ),
            "last_frame": (
                archive["dynamic_rdm_split_half"], archive["dynamic_times_ms"],
            ),
        }
    # end with np.load

    ceilings = {}
    for analysis, (values, times_ms) in consistency.items():
        if analysis not in windows_ms:
            continue
        # end if the analysis was not requested
        window_ms = windows_ms[analysis]
        if window_ms is None:
            window_mask = np.ones(times_ms.shape, dtype=bool)
        else:
            window_mask = (
                (times_ms >= window_ms[0]) & (times_ms < window_ms[1])
            )
        # end if window_ms is None
        # The saved arrays are split repeats x time; average the repeats.
        values = np.asarray(values, dtype=float)
        if values.ndim == 2:
            values = np.nanmean(values, axis=0)
        # end if repeats were kept
        window_values = values[window_mask]
        window_values = window_values[np.isfinite(window_values)]
        if window_values.size == 0:
            ceilings[analysis] = np.nan
            continue
        # end if no finite sample
        peak_consistency = float(window_values.max())
        if spearman_brown:
            peak_consistency = (
                2 * peak_consistency / (1 + peak_consistency)
                if peak_consistency > -1 else np.nan
            )
        # end if spearman_brown
        ceilings[analysis] = peak_consistency
    # end for analysis
    return ceilings
# EOF


"""
layer_depth_temporal_score
Rank-correlate normalized layer depth with the per-layer latency.

Two p values are returned. The parametric Spearman p treats layers as
independent observations, which they are not: neighbouring layers share most of
their computation and therefore have strongly autocorrelated latencies. The
circular-shift p keeps the latency profile intact and only rotates it against
the depth axis, so the null preserves that autocorrelation. Its resolution is
bounded by the number of usable layers, which makes it conservative for short
networks.

INPUT:
    - latencies_ms: np.ndarray -> one latency per layer, NaN where undefined
    - layer_depths: np.ndarray -> normalized depth of every layer

OUTPUT:
    - score: dict -> Spearman statistic, both p values, and the layers used
"""
def layer_depth_temporal_score(latencies_ms, layer_depths) -> dict:
    latencies_ms = np.asarray(latencies_ms, dtype=float)
    layer_depths = np.asarray(layer_depths, dtype=float)
    valid_layers = np.isfinite(latencies_ms)
    if valid_layers.sum() < 3:
        return {
            "rho": np.nan, "pvalue": np.nan, "shift_pvalue": np.nan,
            "n_layers": int(valid_layers.sum()),
        }
    # end if too few usable layers

    valid_latencies_ms = latencies_ms[valid_layers]
    valid_depths = layer_depths[valid_layers]
    rho_result = spearmanr(valid_latencies_ms, valid_depths)
    observed_rho = float(rho_result.statistic)

    # Every non-identity rotation of the latency profile forms the null.
    n_valid = valid_latencies_ms.size
    shifted_rhos = np.array([
        spearmanr(np.roll(valid_latencies_ms, shift), valid_depths).statistic
        for shift in range(1, n_valid)
    ])
    shift_pvalue = float(
        (1 + np.sum(np.abs(shifted_rhos) >= abs(observed_rho))) / n_valid
    )
    return {
        "rho": observed_rho,
        "pvalue": float(rho_result.pvalue),
        "shift_pvalue": shift_pvalue,
        "n_layers": int(n_valid),
    }
# EOF


"""
analysis_timecourses
Extract the layer x neural-time RSA matrix of one of the three analyses.

INPUT:
    - results: dict -> output of load_model_layer_rsa
    - analysis: str -> "image", "first_frame", or "last_frame"
    - frame_onset_ms: float -> model time of the frame slice, ignored for image

OUTPUT:
    - timecourses: np.ndarray -> layers x neural time RSA values
    - time_ms: np.ndarray -> neural time coordinates
    - onset_ms: float -> event the latency is measured from
"""
def analysis_timecourses(
        results: dict, analysis: str, frame_onset_ms: float = 0.0,
        ):
    if analysis == "image":
        # The static analysis already compares one final-frame model RDM with
        # every timepoint of the image-session response.
        return results["static_rsa"], results["static_time_ms"], 0.0
    # end if static image analysis
    if analysis not in ("first_frame", "last_frame"):
        raise ValueError(
            "analysis must be 'image', 'first_frame', or 'last_frame'."
        )
    # end if unknown analysis

    model_time_ms = results["dynamic_model_time_ms"]
    frame_index = int(np.argmin(np.abs(model_time_ms - frame_onset_ms)))
    timecourses = results["dynamic_rsa"][:, :, frame_index]
    return timecourses, results["dynamic_neural_time_ms"], frame_onset_ms
# EOF


"""
summarize_model_analysis
Compute per-layer latencies and the temporal score for one model and analysis.

INPUT:
    - results: dict -> output of load_model_layer_rsa
    - analysis: str -> "image", "first_frame", or "last_frame"
    - frame_onset_ms: float -> model time of the frame slice
    - window_ms: tuple[float, float] | None -> neural analysis window
    - latency_kwargs: dict | None -> overrides passed to timecourse_latency
    - min_peak_similarity: float -> absolute RSA floor a layer must clear
    - min_peak_fraction: float -> fraction of the model's best layer in this
        analysis that a layer must also clear. Both criteria drop layers whose
        latency would be read off a flat, near-zero timecourse; the fractional
        one keeps the criterion comparable across analyses whose overall RSA
        magnitudes differ by a factor of several

OUTPUT:
    - summary: dict -> timecourses, per-layer latencies, and temporal scores
"""
def summarize_model_analysis(
        results: dict,
        analysis: str,
        frame_onset_ms: float = 0.0,
        window_ms=None,
        latency_kwargs=None,
        min_peak_similarity: float = 0.0,
        min_peak_fraction: float = 0.0,
        ) -> dict:
    latency_kwargs = {} if latency_kwargs is None else dict(latency_kwargs)
    timecourses, time_ms, onset_ms = analysis_timecourses(
        results, analysis, frame_onset_ms,
    )

    layer_summaries = [
        timecourse_latency(
            layer_timecourse, time_ms, onset_ms=onset_ms,
            window_ms=window_ms, **latency_kwargs,
        )
        for layer_timecourse in timecourses
    ]
    latency_names = (
        "centroid_latency_ms",
        "relative_centroid_latency_ms",
        "peak_latency_ms",
    )
    peak_similarity = np.array(
        [summary["peak_similarity"] for summary in layer_summaries]
    )
    # Layers without a measurable RSA response get no latency at all.
    best_peak_similarity = float(peak_similarity.max())
    informative_layers = (
        (peak_similarity >= min_peak_similarity)
        & (peak_similarity >= min_peak_fraction * best_peak_similarity)
    )
    unfiltered_latencies = {
        name: np.array([summary[name] for summary in layer_summaries])
        for name in latency_names
    }
    latencies = {
        name: np.where(informative_layers, values, np.nan)
        for name, values in unfiltered_latencies.items()
    }
    temporal_scores = {
        name: layer_depth_temporal_score(values, results["layer_depths"])
        for name, values in latencies.items()
    }
    return {
        "model_name": results["model_name"],
        "analysis": analysis,
        "layer_names": results["layer_names"],
        "layer_depths": results["layer_depths"],
        "time_ms": time_ms,
        "onset_ms": onset_ms,
        "window_ms": window_ms,
        "smoothed_timecourses": np.stack(
            [summary["smoothed_similarity"] for summary in layer_summaries]
        ),
        "peak_similarity": peak_similarity,
        "informative_layers": informative_layers,
        "min_peak_similarity": min_peak_similarity,
        "min_peak_fraction": min_peak_fraction,
        "mean_similarity": np.array(
            [summary["mean_similarity"] for summary in layer_summaries]
        ),
        # Model-level neural correspondence: the best layer's peak RSA.
        "best_layer_index": int(np.argmax(peak_similarity)),
        "best_peak_similarity": best_peak_similarity,
        "latencies": latencies,
        # Kept so a different inclusion cutoff can be applied afterwards.
        "unfiltered_latencies": unfiltered_latencies,
        "temporal_scores": temporal_scores,
    }
# EOF
