"""
Permutation control for the layer-depth versus response-latency analysis.

The hierarchy result reads a monotonic depth-to-latency relation off the RSA
between one neural RDM per timepoint and one model RDM per layer. Adjacent
layers hold nearly the same RDM, so their latencies are strongly autocorrelated
and a smooth latency profile could in principle order itself along depth for
purely structural reasons.

The control keeps the neural RDMs untouched and relabels the stimuli of every
model RDM with one permutation shared by all layers. That leaves the geometry
among layers intact - adjacent layers stay adjacent, the latency profile stays
as smooth as it was - while destroying the correspondence with the neural RDM.
Whatever depth-to-latency relation survives is imposed by that structure rather
than carried by the neural data.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import squareform

from image_processing.video_feature_extraction import (
    load_aligned_video_features,
)
from useful_stuff.general_utils import TimeSeries, create_RDM, subsample_RDM

from .dataloader import (
    load_natraster, min_max_normalization, select_stimulus_rasters,
)
from .neural_model_rsa import normalize_rsa_metric
from .split_half_rsa import compute_rdm_timeseries, cross_temporal_similarity


"""
permute_rdm_entries
Relabel the stimuli of one condensed RDM with a permutation.

The condensed vector is expanded, reindexed on both axes with useful_stuff's
subsample_RDM, and condensed again. Reindexing rows and columns together keeps
the matrix a valid RDM: the same set of pairwise dissimilarities is preserved
and only the stimulus identity attached to each entry changes.

INPUT:
    - rdm: np.ndarray (n*(n-1)/2,) -> condensed upper-triangular RDM
    - permutation: np.ndarray (n,) -> new stimulus order

OUTPUT:
    - permuted_rdm: np.ndarray (n*(n-1)/2,) -> condensed relabelled RDM
"""
def permute_rdm_entries(rdm: np.ndarray, permutation) -> np.ndarray:
    square_rdm = squareform(np.asarray(rdm, dtype=float), checks=False)
    permutation = np.asarray(permutation, dtype=int)
    if permutation.shape[0] != square_rdm.shape[0]:
        raise ValueError(
            f"The permutation has {permutation.shape[0]} entries but the RDM "
            f"holds {square_rdm.shape[0]} stimuli."
        )
    # end if mismatched permutation length
    return squareform(subsample_RDM(square_rdm, permutation), checks=False)
# EOF


"""
prepare_static_neural_rdms
Load the image session and turn it into one neural RDM per timepoint.

This mirrors the static half of run_static_dynamic_neural_model_rsa so the
control reproduces the saved RSA family exactly before anything is permuted.

INPUT:
    - static_path: str | Path -> image-session natraster file
    - good_channels: tuple[int, int] | None -> inclusive one-based MATLAB range
    - reliable_channels_config: str | Path | None -> reliability YAML
    - reliable_channels_key: str | None -> dataset key inside that YAML
    - source_fs: float -> raster sampling frequency
    - new_fs: float -> analysis sampling frequency
    - crop_ms: float | None -> duration retained from stimulus onset
    - normalization: str | None -> optional min_max neural normalization
    - rdm_metric: str -> neural RDM dissimilarity measure

OUTPUT:
    - state: dict -> neural RDMs, time axis, stimulus names, and channels
"""
def prepare_static_neural_rdms(
        static_path,
        good_channels=None,
        reliable_channels_config=None,
        reliable_channels_key=None,
        source_fs: float = 1000,
        new_fs: float = 100,
        crop_ms=1000,
        normalization=None,
        rdm_metric: str = "cosine_cnt",
        ) -> dict:
    rasters, all_stimulus_names, channel_numbers = load_natraster(
        static_path,
        good_channels=good_channels,
        reliable_channels_config=reliable_channels_config,
        reliable_channels_key=reliable_channels_key,
        return_channel_numbers=True,
    )
    # The image file also stores the earlier timed static controls.
    rasters, stimulus_names = select_stimulus_rasters(
        rasters, all_stimulus_names, "img_",
        excluded_prefixes=("img_2000ms_", "img_2250ms_"),
    )
    if crop_ms is not None:
        stop_sample = int(round(crop_ms * source_fs / 1000))
        if stop_sample > rasters.shape[1]:
            raise ValueError(
                f"{crop_ms:g} ms crop needs {stop_sample} samples, but "
                f"{Path(static_path).name} contains {rasters.shape[1]}."
            )
        # end if crop exceeds raster
        rasters = rasters[:, :stop_sample, :]
    # end if crop_ms is not None
    if normalization == "min_max":
        rasters = min_max_normalization(rasters)
    # end if min_max normalization

    neural_ts = TimeSeries(rasters, fs=source_fs)
    neural_ts.resample(new_fs)
    return {
        # neural time x stimulus pairs, held fixed across every permutation.
        "neural_rdms": compute_rdm_timeseries(
            neural_ts.get_array(), rdm_metric,
        ),
        "time_ms": np.arange(len(neural_ts)) * 1000 / neural_ts.get_fs(),
        "stimulus_names": stimulus_names,
        "channel_numbers": channel_numbers,
    }
# EOF


"""
static_model_rdms_by_layer
Build the final-frame model RDM of every layer of one model.

The RDMs are computed once and reused for all permutations, which keeps the
HDF5 reads out of the permutation loop.

INPUT:
    - feature_paths: list[str | Path] -> one sequential feature file per layer
    - static_feature_names: list[str] -> stimuli in the neural RDM order
    - model_rdm_metric: str -> model-feature RDM dissimilarity measure
    - model_frame_index: int -> feature frame used as the static model RDM

OUTPUT:
    - model_rdms: np.ndarray -> layers x stimulus pairs condensed RDMs
    - layer_names: list[str] -> layer of each row, in file order
"""
def static_model_rdms_by_layer(
        feature_paths,
        static_feature_names: list[str],
        model_rdm_metric: str = "cosine_cnt",
        model_frame_index: int = -1,
        ):
    model_rdms = []
    layer_names = []
    for feature_path in feature_paths:
        features, layer_name, _ = load_aligned_video_features(
            feature_path, static_feature_names, frame_index=model_frame_index,
        )
        model_rdms.append(create_RDM(features, metric=model_rdm_metric))
        layer_names.append(layer_name)
    # end for feature_path
    return np.stack(model_rdms), layer_names
# EOF


"""
permuted_static_rsa
Correlate the fixed neural RDMs with relabelled model RDMs, layer by layer.

Passing the identity permutation reproduces the saved static RSA, which is how
the control is validated before the null is drawn.

INPUT:
    - neural_rdms: np.ndarray -> neural time x stimulus pairs
    - model_rdms: np.ndarray -> layers x stimulus pairs
    - permutation: np.ndarray | None -> shared stimulus relabelling, None keeps
        the original order
    - rsa_metric: str -> pearson/correlation or spearman RDM similarity

OUTPUT:
    - static_rsa: np.ndarray -> layers x neural time RSA values
"""
def permuted_static_rsa(
        neural_rdms: np.ndarray,
        model_rdms: np.ndarray,
        permutation=None,
        rsa_metric: str = "spearman",
        ) -> np.ndarray:
    rsa_metric = normalize_rsa_metric(rsa_metric)
    if permutation is not None:
        # One permutation for every layer, so the between-layer geometry that
        # makes neighbouring latencies similar is carried over untouched.
        model_rdms = np.stack([
            permute_rdm_entries(layer_rdm, permutation)
            for layer_rdm in model_rdms
        ])
    # end if permutation is not None
    # cross_temporal_similarity returns neural time x layers; the hierarchy
    # utilities expect layers on the first axis.
    return cross_temporal_similarity(
        neural_rdms, model_rdms, metric=rsa_metric,
    ).T
# EOF


"""
latency_profile_smoothness
Measure how tightly neighbouring layers hold the same latency.

The control asks whether a smooth latency profile is enough to produce a
depth-to-latency relation, so the smoothness itself has to be reported next to
the rank correlation. Both statistics are scaled by the profile spread as well
as given in milliseconds, because a permuted profile can be smooth simply by
being flat.

INPUT:
    - latencies_ms: np.ndarray -> one latency per layer, NaN where undefined

OUTPUT:
    - smoothness: dict -> adjacent-layer step, profile range, and their ratio
"""
def latency_profile_smoothness(latencies_ms) -> dict:
    latencies_ms = np.asarray(latencies_ms, dtype=float)
    valid_latencies = latencies_ms[np.isfinite(latencies_ms)]
    if valid_latencies.size < 3:
        return {
            "adjacent_step_ms": np.nan, "range_ms": np.nan,
            "step_over_range": np.nan,
        }
    # end if too few usable layers
    adjacent_step_ms = float(np.mean(np.abs(np.diff(valid_latencies))))
    range_ms = float(valid_latencies.max() - valid_latencies.min())
    return {
        "adjacent_step_ms": adjacent_step_ms,
        "range_ms": range_ms,
        # A profile that walks along depth in small steps relative to its own
        # spread is smooth; one whose steps are as large as its range is not.
        "step_over_range": (
            adjacent_step_ms / range_ms if range_ms > 0 else np.nan
        ),
    }
# EOF
