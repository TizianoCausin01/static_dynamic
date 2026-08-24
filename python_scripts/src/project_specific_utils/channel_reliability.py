from pathlib import Path

import numpy as np
import yaml


"""
last_frame_presentation_indices
Select standard last-frame image presentations and extract image identities.
Timed control images are excluded even though they share the img_ prefix.

INPUT:
    - presentation_names: list[str] -> stimulus name for every presentation

OUTPUT:
    - presentation_indices: np.ndarray -> selected presentation indices
    - presentation_identities: list[str] -> image identity for each selection
"""
def last_frame_presentation_indices(
        presentation_names: list[str],
        ) -> tuple[np.ndarray, list[str]]:
    excluded_prefixes = ("img_2000ms_", "img_2250ms_")
    presentation_indices = []
    presentation_identities = []

    for presentation_index, presentation_name in enumerate(presentation_names):
        stimulus_stem = Path(presentation_name).stem
        if not stimulus_stem.startswith("img_"):
            continue
        # end if not last-frame image
        if stimulus_stem.startswith(excluded_prefixes):
            continue
        # end if timed control image

        presentation_indices.append(presentation_index)
        presentation_identities.append(stimulus_stem[len("img_"):])
    # end for presentation_index, presentation_name

    unique_identities = list(dict.fromkeys(presentation_identities))
    if len(unique_identities) < 2:
        raise ValueError(
            "Need at least two standard last-frame image identities."
        )
    # end if too few last-frame images

    repetition_counts = {
        identity: presentation_identities.count(identity)
        for identity in unique_identities
    }
    insufficient_identities = [
        identity for identity, count in repetition_counts.items() if count < 2
    ]
    if insufficient_identities:
        raise ValueError(
            "Every last-frame image needs at least two presentations; "
            f"insufficient identities: {insufficient_identities}."
        )
    # end if insufficient identities

    return np.asarray(presentation_indices, dtype=int), presentation_identities
# EOF


"""
rowwise_pearson_correlation
Correlate corresponding rows of two channel-by-stimulus matrices.

INPUT:
    - first: np.ndarray -> channels x stimuli first-half selectivity
    - second: np.ndarray -> channels x stimuli second-half selectivity

OUTPUT:
    - correlations: np.ndarray -> Pearson correlation for every channel
"""
def rowwise_pearson_correlation(
        first: np.ndarray,
        second: np.ndarray,
        ) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(
            "first and second must be matching channels x stimuli matrices."
        )
    # end if invalid matrices

    # Use the same finite stimulus pairs for both means and both norms.
    valid_values = np.isfinite(first) & np.isfinite(second)
    valid_counts = valid_values.sum(axis=1)
    safe_counts = np.maximum(valid_counts, 1)
    first_means = np.where(valid_values, first, 0).sum(
        axis=1, keepdims=True,
    ) / safe_counts[:, np.newaxis]
    second_means = np.where(valid_values, second, 0).sum(
        axis=1, keepdims=True,
    ) / safe_counts[:, np.newaxis]
    first = np.where(valid_values, first - first_means, 0)
    second = np.where(valid_values, second - second_means, 0)

    numerators = np.sum(first * second, axis=1)
    denominators = np.sqrt(
        np.sum(first ** 2, axis=1) * np.sum(second ** 2, axis=1)
    )

    correlations = np.full(first.shape[0], np.nan, dtype=float)
    np.divide(
        numerators,
        denominators,
        out=correlations,
        where=(valid_counts >= 2) & (denominators > 0),
    )
    return np.clip(correlations, -1, 1)
# EOF


"""
compute_channel_selectivity_reliability
Compute Xiao et al.-style within-session individual-image self-consistency.
For every random split, repetitions of each image are divided into halves,
averaged within half, and the two image-selectivity vectors are correlated.

INPUT:
    - window_responses: np.ndarray -> channels x presentations window means
    - presentation_identities: list[str] -> image identity per presentation
    - rng: np.random.Generator -> random generator controlling trial splits
    - n_split_repeats: int -> number of random repetition splits

OUTPUT:
    - split_reliabilities: np.ndarray -> splits x channel correlations
    - stimulus_order: list[str] -> image order used for selectivity vectors
"""
def compute_channel_selectivity_reliability(
        window_responses: np.ndarray,
        presentation_identities: list[str],
        rng: np.random.Generator,
        n_split_repeats: int = 100,
        ) -> tuple[np.ndarray, list[str]]:
    window_responses = np.asarray(window_responses)
    if window_responses.ndim != 2:
        raise ValueError(
            "window_responses must have shape channels x presentations."
        )
    # end if window_responses dimensions
    if window_responses.shape[1] != len(presentation_identities):
        raise ValueError(
            "The response presentation axis must match presentation_identities."
        )
    # end if presentation count mismatch
    if n_split_repeats < 2:
        raise ValueError("n_split_repeats must be at least two.")
    # end if too few splits

    stimulus_order = list(dict.fromkeys(presentation_identities))
    presentation_indices_by_stimulus = {
        stimulus: np.flatnonzero(
            np.asarray(presentation_identities) == stimulus
        )
        for stimulus in stimulus_order
    }
    insufficient_stimuli = [
        stimulus for stimulus, indices
        in presentation_indices_by_stimulus.items()
        if len(indices) < 2
    ]
    if insufficient_stimuli:
        raise ValueError(
            "Every stimulus needs at least two presentations; "
            f"insufficient identities: {insufficient_stimuli}."
        )
    # end if insufficient stimuli

    split_reliabilities = np.empty(
        (n_split_repeats, window_responses.shape[0]), dtype=float,
    )
    for split_index in range(n_split_repeats):
        first_selectivity = np.empty(
            (window_responses.shape[0], len(stimulus_order)), dtype=float,
        )
        second_selectivity = np.empty_like(first_selectivity)

        # Split repetitions independently within every image identity.
        for stimulus_index, stimulus in enumerate(stimulus_order):
            presentation_indices = presentation_indices_by_stimulus[stimulus]
            shuffled_indices = rng.permutation(presentation_indices)
            first_half_count = len(shuffled_indices) // 2
            first_indices = shuffled_indices[:first_half_count]
            second_indices = shuffled_indices[first_half_count:]

            first_selectivity[:, stimulus_index] = np.nanmean(
                window_responses[:, first_indices], axis=1,
            )
            second_selectivity[:, stimulus_index] = np.nanmean(
                window_responses[:, second_indices], axis=1,
            )
        # end for stimulus_index, stimulus

        split_reliabilities[split_index] = rowwise_pearson_correlation(
            first_selectivity, second_selectivity,
        )
    # end for split_index

    return split_reliabilities, stimulus_order
# EOF


"""
summarize_channel_reliability
Threshold the mean of the selection splits and independently summarize the
remaining held-out splits, following Xiao et al.'s first-50/second-50 scheme.

INPUT:
    - split_reliabilities: np.ndarray -> splits x channels correlations
    - reliability_threshold: float -> minimum mean selection reliability
    - selection_split_count: int -> leading splits used to select channels

OUTPUT:
    - selection_reliability: np.ndarray -> mean leading-split reliability
    - heldout_reliability: np.ndarray -> mean remaining-split reliability
    - reliable_mask: np.ndarray -> channels meeting the requested threshold
"""
def summarize_channel_reliability(
        split_reliabilities: np.ndarray,
        reliability_threshold: float = 0.4,
        selection_split_count: int = 50,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_reliabilities = np.asarray(split_reliabilities, dtype=float)
    if split_reliabilities.ndim != 2:
        raise ValueError(
            "split_reliabilities must have shape splits x channels."
        )
    # end if split reliability dimensions
    if not -1 <= reliability_threshold <= 1:
        raise ValueError("reliability_threshold must be between -1 and 1.")
    # end if invalid threshold
    if not 1 <= selection_split_count < split_reliabilities.shape[0]:
        raise ValueError(
            "selection_split_count must leave at least one held-out split."
        )
    # end if invalid selection split count

    selection_reliability = np.nanmean(
        split_reliabilities[:selection_split_count], axis=0,
    )
    heldout_reliability = np.nanmean(
        split_reliabilities[selection_split_count:], axis=0,
    )
    reliable_mask = (
        np.isfinite(selection_reliability)
        & (selection_reliability >= reliability_threshold)
    )
    return selection_reliability, heldout_reliability, reliable_mask
# EOF


"""
save_reliable_channels
Save or replace one dataset's reliable-channel result in a YAML config.

INPUT:
    - config_path: str | Path -> reliable-channel YAML destination
    - dataset_key: str -> stable dataset identifier used by load_natraster
    - result: dict -> YAML-serializable reliability metadata and channels

OUTPUT:
    - None
"""
def save_reliable_channels(
        config_path: str | Path,
        dataset_key: str,
        result: dict,
        ) -> None:
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as file:
            reliability_config = yaml.safe_load(file) or {}
        # end with open
    else:
        reliability_config = {}
    # end if config_path.exists

    datasets = reliability_config.setdefault("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(
            f"{config_path} must contain a top-level 'datasets' mapping."
        )
    # end if invalid datasets mapping
    datasets[dataset_key] = result

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as file:
        yaml.safe_dump(
            reliability_config, file, sort_keys=False, default_flow_style=False,
        )
    # end with open
    return None
# EOF


"""
load_reliable_channels
Load one dataset's one-based reliable channel numbers from a YAML config.

INPUT:
    - config_path: str | Path -> reliable-channel YAML source
    - dataset_key: str -> dataset identifier stored under datasets

OUTPUT:
    - reliable_channels: list[int] -> sorted one-based channel numbers
"""
def load_reliable_channels(
        config_path: str | Path,
        dataset_key: str,
        ) -> list[int]:
    config_path = Path(config_path)
    with open(config_path, "r") as file:
        reliability_config = yaml.safe_load(file) or {}
    # end with open

    datasets = reliability_config.get("datasets", {})
    if dataset_key not in datasets:
        raise KeyError(
            f"Dataset {dataset_key!r} is not present in {config_path}."
        )
    # end if dataset missing
    dataset_config = datasets[dataset_key]
    reliable_channels = dataset_config.get("reliable_channels")
    if reliable_channels is None:
        raise KeyError(
            f"Dataset {dataset_key!r} has no 'reliable_channels' entry."
        )
    # end if reliable channels missing

    reliable_channels = [int(channel) for channel in reliable_channels]
    if any(channel < 1 for channel in reliable_channels):
        raise ValueError("Reliable channel numbers must be positive and one-based.")
    # end if invalid channel number
    if len(reliable_channels) != len(set(reliable_channels)):
        raise ValueError("Reliable channel numbers must not contain duplicates.")
    # end if duplicate channel number
    return sorted(reliable_channels)
# EOF
