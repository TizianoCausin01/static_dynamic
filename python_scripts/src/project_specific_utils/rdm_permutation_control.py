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

from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import label, sum_labels
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

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


"""
condensed_permutation_index
Map one stimulus permutation onto the condensed RDM entries it moves.

Indexing a condensed RDM with this array is identical to squareform -> reorder
rows and columns with the permutation -> squareform (permute_rdm_entries), but
it applies to every timepoint of an RDM timeseries in a single gather.

INPUT:
    - n_stimuli: int -> number of stimuli in the RDM
    - permutation: np.ndarray (n_stimuli,) -> new stimulus order

OUTPUT:
    - condensed_index: np.ndarray (n*(n-1)/2,) -> source entry of every entry
"""
def condensed_permutation_index(n_stimuli: int, permutation) -> np.ndarray:
    upper_rows, upper_cols = np.triu_indices(n_stimuli, k=1)
    square_index = np.zeros((n_stimuli, n_stimuli), dtype=int)
    square_index[upper_rows, upper_cols] = np.arange(len(upper_rows))
    square_index += square_index.T
    permutation = np.asarray(permutation, dtype=int)
    return square_index[permutation[upper_rows], permutation[upper_cols]]
# EOF


"""
standardize_rdm_rows
Rank (for Spearman), center and unit-normalize every RDM, so that a matrix
product of two standardized RDM sets gives their correlations. Relabelling the
stimuli only reorders the entries, so this runs once, outside the permutations.

INPUT:
    - rdms: np.ndarray -> time x stimulus-pair distances
    - metric: str -> correlation or spearman

OUTPUT:
    - standardized_rdms: np.ndarray -> time x stimulus pairs, unit-norm rows
"""
def standardize_rdm_rows(rdms: np.ndarray, metric: str) -> np.ndarray:
    rdms = np.asarray(rdms, dtype=np.float64)
    if metric == "spearman":
        rdms = rankdata(rdms, axis=1)
    elif metric != "correlation":
        raise ValueError("metric must be 'correlation' or 'spearman'.")
    # end if metric
    rdms = rdms - rdms.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Constant RDMs (e.g. silent bins) have no norm and become NaN rows.
        return rdms / np.linalg.norm(rdms, axis=1, keepdims=True)
    # end with np.errstate
# EOF


"""
permutation_chunk
Draw a block of stimulus permutations and compute each null similarity matrix.
Runs inside one worker, with BLAS limited to a single thread when parallel.

INPUT:
    - fixed_rdms: np.ndarray -> standardized fixed time x stimulus pairs
    - permuted_rdms: np.ndarray -> standardized relabelled time x stimulus pairs
    - n_permutations: int -> permutations drawn by this worker
    - seed: np.random.SeedSequence -> independent seed of this worker
    - single_thread: bool -> limit BLAS to one thread

OUTPUT:
    - null: np.ndarray (n_permutations, fixed time, permuted time) float32
"""
def permutation_chunk(fixed_rdms, permuted_rdms, n_permutations, seed, single_thread):
    rng = np.random.default_rng(seed)
    n_stimuli = int(round((1 + np.sqrt(1 + 8 * permuted_rdms.shape[1])) / 2))
    null = np.empty(
        (n_permutations, fixed_rdms.shape[0], permuted_rdms.shape[0]),
        dtype=np.float32,
    )
    with threadpool_limits(1 if single_thread else None):
        for permutation_index in range(n_permutations):
            condensed_index = condensed_permutation_index(
                n_stimuli, rng.permutation(n_stimuli),
            )
            null[permutation_index] = fixed_rdms @ permuted_rdms[:, condensed_index].T
        # end for permutation_index
    # end with threadpool_limits
    return null
# EOF


"""
permuted_cross_temporal_similarity
Cross-temporal RDM similarity (as cross_temporal_similarity) plus its null
distribution from relabelling the stimuli of the second RDM timeseries. Every
permutation is shared by all timepoints, so the temporal structure of both
RDM timeseries is preserved and only the stimulus correspondence is broken.

INPUT:
    - fixed_rdms: np.ndarray -> first time x stimulus pairs, kept untouched
    - permuted_rdms: np.ndarray -> second time x stimulus pairs, relabelled
    - n_permutations: int -> number of stimulus permutations
    - metric: str -> correlation or spearman
    - random_seed: int -> seed of the permutation generator
    - n_jobs: int -> parallel workers (1 = serial with multithreaded BLAS)

OUTPUT:
    - observed: np.ndarray -> first time x second time similarity
    - null: np.ndarray (n_permutations, first time, second time) float32
"""
def permuted_cross_temporal_similarity(
        fixed_rdms: np.ndarray,
        permuted_rdms: np.ndarray,
        n_permutations: int = 1000,
        metric: str = "spearman",
        random_seed: int = 0,
        n_jobs: int = 1,
        ) -> tuple[np.ndarray, np.ndarray]:
    if fixed_rdms.ndim != 2 or fixed_rdms.shape[1] != permuted_rdms.shape[1]:
        raise ValueError("Inputs must be time x matching stimulus-pair matrices.")
    # end if input shapes
    fixed_rdms = standardize_rdm_rows(fixed_rdms, metric)
    permuted_rdms = standardize_rdm_rows(permuted_rdms, metric)
    observed = np.clip(fixed_rdms @ permuted_rdms.T, -1, 1)

    worker_sizes = [
        len(chunk) for chunk in np.array_split(np.arange(n_permutations), n_jobs)
    ]
    worker_seeds = np.random.SeedSequence(random_seed).spawn(n_jobs)
    null_chunks = Parallel(n_jobs=n_jobs)(
        delayed(permutation_chunk)(
            fixed_rdms, permuted_rdms, worker_size, worker_seed, n_jobs > 1,
        )
        for worker_size, worker_seed in zip(worker_sizes, worker_seeds)
    )
    return observed, np.clip(np.concatenate(null_chunks), -1, 1)
# EOF


"""
permutation_p_values
One-sided (observed > null) permutation p-values for every cell of a
similarity matrix, uncorrected and family-wise corrected with the maximum
statistic over the whole matrix.

INPUT:
    - observed: np.ndarray -> first time x second time similarity
    - null: np.ndarray -> permutations x first time x second time

OUTPUT:
    - p_values: dict -> pointwise and max_statistic p maps, null maxima
"""
def permutation_p_values(observed: np.ndarray, null: np.ndarray) -> dict:
    n_permutations = null.shape[0]
    null_max = np.nanmax(null.reshape(n_permutations, -1), axis=1)
    pointwise = (1 + (null >= observed).sum(axis=0)) / (1 + n_permutations)
    max_statistic = (
        (1 + (null_max[:, None, None] >= observed).sum(axis=0))
        / (1 + n_permutations)
    )
    # NaN cells compare False against every null value; keep them undefined.
    undefined = ~np.isfinite(observed)
    pointwise[undefined] = np.nan
    max_statistic[undefined] = np.nan
    return {
        "pointwise": pointwise,
        "max_statistic": max_statistic,
        "null_max": null_max,
    }
# EOF


"""
cluster_permutation_test
Cluster-mass permutation test on a similarity matrix. Cells above the
pointwise (1 - cluster_alpha) null percentile form clusters (4-connectivity),
each cluster's mass is the sum of its similarities, and every observed cluster
is compared with the largest cluster mass of every null matrix.

INPUT:
    - observed: np.ndarray -> first time x second time similarity
    - null: np.ndarray -> permutations x first time x second time
    - cluster_alpha: float -> pointwise cluster-forming threshold

OUTPUT:
    - clusters: dict -> labels map, masses, p_values, null_max_mass, threshold
"""
def cluster_permutation_test(
        observed: np.ndarray,
        null: np.ndarray,
        cluster_alpha: float = 0.05,
        ) -> dict:
    threshold = np.nanpercentile(null, 100 * (1 - cluster_alpha), axis=0)

    def cluster_masses(similarity):
        labels, n_clusters = label(np.nan_to_num(similarity > threshold))
        masses = sum_labels(similarity, labels, index=np.arange(1, n_clusters + 1))
        return labels, np.asarray(masses, dtype=float)
    # EOF

    labels, masses = cluster_masses(observed)
    null_max_mass = np.asarray([
        max(cluster_masses(null_matrix)[1], default=0.0) for null_matrix in null
    ])
    p_values = (
        (1 + (null_max_mass[:, None] >= masses[None, :]).sum(axis=0))
        / (1 + len(null_max_mass))
    )
    return {
        "labels": labels,
        "masses": masses,
        "p_values": p_values,
        "null_max_mass": null_max_mass,
        "threshold": threshold,
    }
# EOF


"""
permutation_significance_mask
Boolean map of the significant cells of a saved permutation result, for one
of the three corrections computed by static_dynamic_drsa_permutation.ipynb.

INPUT:
    - permutation_result: dict | NpzFile -> saved archive with pointwise_p,
        max_statistic_p, cluster_labels and cluster_p
    - correction: str -> "pointwise" (uncorrected), "max_statistic" (FWE over
        the whole matrix) or "cluster" (cluster-mass)
    - alpha: float -> significance level

OUTPUT:
    - significant: np.ndarray -> first time x second time boolean mask
"""
def permutation_significance_mask(
        permutation_result,
        correction: str = "cluster",
        alpha: float = 0.05,
        ) -> np.ndarray:
    if correction == "pointwise":
        return permutation_result["pointwise_p"] < alpha
    elif correction == "max_statistic":
        return permutation_result["max_statistic_p"] < alpha
    elif correction == "cluster":
        # Cluster labels are 1-based; label 0 marks cells outside every cluster.
        significant_labels = (
            np.flatnonzero(permutation_result["cluster_p"] < alpha) + 1
        )
        return np.isin(permutation_result["cluster_labels"], significant_labels)
    # end if correction
    raise ValueError(
        "correction must be 'pointwise', 'max_statistic', or 'cluster'."
    )
# EOF


"""
remove_small_significant_regions
Drop connected significant regions smaller than a minimum number of cells.
Regions use the same 4-connectivity as cluster_permutation_test. Cells are
only ever removed, never added, so the result is at most as liberal as the
input mask; holes inside a region are left untouched.

INPUT:
    - significant: np.ndarray -> first time x second time boolean mask
    - min_region_cells: int -> smallest region kept; 0 or 1 keeps everything

OUTPUT:
    - cleaned: np.ndarray -> boolean mask without the small regions
"""
def remove_small_significant_regions(
        significant: np.ndarray,
        min_region_cells: int = 0,
        ) -> np.ndarray:
    significant = np.asarray(significant, dtype=bool)
    if min_region_cells <= 1:
        return significant
    # end if nothing to remove
    labels, n_regions = label(significant)
    # Cell count of every region; index 0 is the non-significant background.
    region_sizes = np.bincount(labels.ravel(), minlength=n_regions + 1)
    kept_regions = np.flatnonzero(region_sizes >= min_region_cells)
    kept_regions = kept_regions[kept_regions > 0]
    return np.isin(labels, kept_regions)
# EOF


"""
curve_cluster_significance
Cluster-mass permutation test on one timecourse, e.g. the best static match
per movie time. Each null curve must come from the same pipeline as the
observed curve (same reduction over the second time axis, same smoothing),
so that biases such as the upward shift of a maximum are present in the null
too. Clusters are runs of consecutive timebins above the pointwise null
percentile; every cluster mass is compared with the largest null cluster mass.

INPUT:
    - observed_curve: np.ndarray -> (time,) observed values
    - null_curves: np.ndarray -> (permutations, time) null values
    - cluster_alpha: float -> pointwise cluster-forming threshold
    - alpha: float -> cluster-level significance level

OUTPUT:
    - significant: np.ndarray -> (time,) boolean, True inside significant clusters
    - clusters: dict -> output of cluster_permutation_test
"""
def curve_cluster_significance(
        observed_curve: np.ndarray,
        null_curves: np.ndarray,
        cluster_alpha: float = 0.05,
        alpha: float = 0.05,
        ) -> tuple[np.ndarray, dict]:
    observed_curve = np.asarray(observed_curve, dtype=float)
    null_curves = np.asarray(null_curves, dtype=float)
    if null_curves.ndim != 2 or null_curves.shape[1] != observed_curve.shape[0]:
        raise ValueError("null_curves must be permutations x the observed time axis.")
    # end if shapes differ
    # 1D labelling: a cluster is a run of consecutive supra-threshold bins.
    clusters = cluster_permutation_test(observed_curve, null_curves, cluster_alpha)
    significant_labels = np.flatnonzero(clusters["p_values"] < alpha) + 1
    significant = np.isin(clusters["labels"], significant_labels)
    return significant, clusters
# EOF

