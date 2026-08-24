import numpy as np

from .split_half_rsa import average_repetition_halves


"""
participation_ratio
Compute the effective or normalized participation ratio along one array axis.

INPUT:
    - values: np.ndarray -> non-negative weights such as responses or eigenvalues
    - axis: int -> axis whose entries participate in the ratio
    - normalized: bool -> divide the effective count by the axis length

OUTPUT:
    - ratio: np.ndarray -> participation ratio with the selected axis removed
"""
def participation_ratio(
        values: np.ndarray,
        axis: int = -1,
        normalized: bool = False,
        ) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not -values.ndim <= axis < values.ndim:
        raise ValueError(
            f"axis {axis} is invalid for an array with {values.ndim} dimensions."
        )
    # end if axis is outside the array
    axis = axis % values.ndim
    if np.any(values < -np.finfo(np.float64).eps * 100):
        raise ValueError("Participation-ratio values must be non-negative.")
    # end if values contain negative entries

    # Remove tiny negative eigensolver round-off without changing real values.
    values = np.maximum(values, 0)
    numerator = np.sum(values, axis=axis) ** 2
    denominator = np.sum(values ** 2, axis=axis)
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    if normalized:
        ratio = ratio / values.shape[axis]
    # end if normalized
    return ratio
# EOF


"""
treves_rolls_sparseness
Compute the Treves-Rolls sparseness used in DNNDYnamics.pdf.

INPUT:
    - responses: np.ndarray -> non-negative response values
    - axis: int -> neurons for population sparsity or images for lifetime sparsity

OUTPUT:
    - sparseness: np.ndarray -> 1 minus the normalized participation ratio
"""
def treves_rolls_sparseness(
        responses: np.ndarray,
        axis: int = -1,
        ) -> np.ndarray:
    return 1 - participation_ratio(responses, axis=axis, normalized=True)
# EOF


"""
robust_normalize_responses
Scale each neuron and timepoint across images using robust percentile bounds.

INPUT:
    - responses: np.ndarray -> neurons x time x images response array
    - percentile_bounds: tuple[float, float] -> lower and upper percentiles

OUTPUT:
    - normalized_responses: np.ndarray -> responses clipped to the [0, 1] range
"""
def robust_normalize_responses(
        responses: np.ndarray,
        percentile_bounds: tuple[float, float] = (2.5, 97.5),
        ) -> np.ndarray:
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 3:
        raise ValueError("responses must have shape neurons x time x images.")
    # end if responses.ndim

    lower_percentile, upper_percentile = percentile_bounds
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError(
            "percentile_bounds must be increasing values between 0 and 100."
        )
    # end if invalid percentile bounds

    lower = np.nanpercentile(
        responses, lower_percentile, axis=2, keepdims=True,
    )
    upper = np.nanpercentile(
        responses, upper_percentile, axis=2, keepdims=True,
    )
    response_range = upper - lower
    normalized_responses = np.zeros_like(responses, dtype=np.float64)
    np.divide(
        responses - lower,
        response_range,
        out=normalized_responses,
        where=response_range > 0,
    )
    return np.clip(normalized_responses, 0, 1)
# EOF


"""
stimulus_min_max_normalization
Scale each neuron independently across stimuli at every available timepoint.

INPUT:
    - responses: np.ndarray -> neurons x stimuli or neurons x time x stimuli

OUTPUT:
    - normalized_responses: np.ndarray -> stimulus responses scaled to [0, 1]
"""
def stimulus_min_max_normalization(responses: np.ndarray) -> np.ndarray:
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim not in (2, 3):
        raise ValueError(
            "responses must have shape neurons x stimuli or "
            "neurons x time x stimuli."
        )
    # end if responses.ndim
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses must contain only finite values.")
    # end if non-finite responses

    # Keeping the stimulus axis allows every neuron/timepoint to use local bounds.
    minimum_response = responses.min(axis=-1, keepdims=True)
    response_range = (
        responses.max(axis=-1, keepdims=True) - minimum_response
    )
    normalized_responses = np.zeros_like(responses, dtype=np.float64)
    np.divide(
        responses - minimum_response,
        response_range,
        out=normalized_responses,
        where=response_range > 0,
    )
    return normalized_responses
# EOF


"""
representation_measure_timecourses
Compute dimensionality, population, and lifetime participation/sparseness at
every timepoint of one condition.

INPUT:
    - responses: np.ndarray -> neurons x time x matched images response array
    - percentile_bounds: tuple[float, float] -> robust activity scaling bounds

OUTPUT:
    - measures: dict[str, np.ndarray] -> ED and two sparseness timecourses
"""
def representation_measure_timecourses(
        responses: np.ndarray,
        percentile_bounds: tuple[float, float] = (2.5, 97.5),
        ) -> dict[str, np.ndarray]:
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 3:
        raise ValueError("responses must have shape neurons x time x images.")
    # end if responses.ndim
    if responses.shape[0] < 2 or responses.shape[2] < 2:
        raise ValueError("At least two neurons and two images are required.")
    # end if too few neurons or images
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses must contain only finite values.")
    # end if non-finite responses

    # Covariance features are neurons and observations are matched images.
    # Center each neuron across images independently at every timepoint.
    centered_responses = responses - responses.mean(axis=2, keepdims=True)
    time_by_neuron_by_image = np.moveaxis(centered_responses, 1, 0)
    singular_values = np.linalg.svd(
        time_by_neuron_by_image, compute_uv=False,
    )
    covariance_eigenvalues = (
        singular_values ** 2 / (responses.shape[2] - 1)
    )
    dimensionality_pr = participation_ratio(
        covariance_eigenvalues, axis=1, normalized=False,
    )

    # The paper's percentile scaling keeps activity non-negative. Mean-centering
    # here would make both activity participation and sparseness invalid.
    normalized_responses = robust_normalize_responses(
        responses, percentile_bounds=percentile_bounds,
    )
    population_sparseness_by_image = treves_rolls_sparseness(
        normalized_responses, axis=0,
    )
    lifetime_sparseness_by_neuron = treves_rolls_sparseness(
        normalized_responses, axis=2,
    )

    measures = {
        "dimensionality_pr": dimensionality_pr,
        "population_sparseness": np.nanmean(
            population_sparseness_by_image, axis=1,
        ),
        "lifetime_sparseness": np.nanmean(
            lifetime_sparseness_by_neuron, axis=0,
        ),
    }
    return measures
# EOF


"""
cvpca_participation_ratio
Estimate the positive cross-validated PCA spectrum and its participation ratio
from independent repetition-half averages at every timepoint.

INPUT:
    - training_half: np.ndarray -> neurons x time x images training averages
    - validation_half: np.ndarray -> matching independent validation averages

OUTPUT:
    - measures: dict[str, np.ndarray] -> cvPCA PR and spectrum diagnostics by time
"""
def cvpca_participation_ratio(
        training_half: np.ndarray,
        validation_half: np.ndarray,
        ) -> dict[str, np.ndarray]:
    training_half = np.asarray(training_half, dtype=np.float64)
    validation_half = np.asarray(validation_half, dtype=np.float64)
    if training_half.shape != validation_half.shape or training_half.ndim != 3:
        raise ValueError(
            "training_half and validation_half must have matching "
            "neurons x time x images shapes."
        )
    # end if input shapes differ
    if training_half.shape[0] < 2 or training_half.shape[2] < 2:
        raise ValueError("At least two neurons and two images are required.")
    # end if too few neurons or images

    # Center each half independently so only stimulus-dependent activity enters.
    training_centered = training_half - training_half.mean(axis=2, keepdims=True)
    validation_centered = (
        validation_half - validation_half.mean(axis=2, keepdims=True)
    )
    training_matrices = np.moveaxis(training_centered, 1, 0)
    validation_matrices = np.moveaxis(validation_centered, 1, 0)

    # Fit the component directions only on the training repetition half.
    training_directions, _, _ = np.linalg.svd(
        training_matrices, full_matrices=False,
    )
    transposed_directions = np.swapaxes(training_directions, 1, 2)
    training_scores = transposed_directions @ training_matrices
    validation_scores = transposed_directions @ validation_matrices
    cv_eigenvalues = np.sum(
        training_scores * validation_scores, axis=2,
    ) / (training_half.shape[2] - 1)

    # Finite data can give negative cross-validated variances. Negative modes
    # are retained as a diagnostic but cannot represent covariance dimensions.
    positive_eigenvalues = np.maximum(cv_eigenvalues, 0)
    absolute_spectrum_mass = np.sum(np.abs(cv_eigenvalues), axis=1)
    negative_spectrum_mass = np.sum(
        np.abs(np.minimum(cv_eigenvalues, 0)), axis=1,
    )
    negative_fraction = np.full(training_half.shape[1], np.nan)
    np.divide(
        negative_spectrum_mass,
        absolute_spectrum_mass,
        out=negative_fraction,
        where=absolute_spectrum_mass > 0,
    )
    return {
        "cvpca_pr": participation_ratio(
            positive_eigenvalues, axis=1, normalized=False,
        ),
        "positive_signal_variance": np.sum(positive_eigenvalues, axis=1),
        "negative_spectrum_fraction": negative_fraction,
    }
# EOF


"""
split_half_cvpca_timecourses
Repeat within-stimulus presentation splits and summarize cross-validated PCA
participation ratio and spectrum diagnostics over random splits.

INPUT:
    - rasters: np.ndarray -> neurons x time x presentations response array
    - presentation_identities: list[str] -> identity for every presentation
    - stimulus_order: list[str] -> matched image identities retained in order
    - n_split_repeats: int -> number of random repetition splits
    - rng: np.random.Generator -> random generator controlling the splits

OUTPUT:
    - summary: dict[str, np.ndarray] -> split values and time-resolved summaries
"""
def split_half_cvpca_timecourses(
        rasters: np.ndarray,
        presentation_identities: list[str],
        stimulus_order: list[str],
        n_split_repeats: int,
        rng: np.random.Generator,
        ) -> dict[str, np.ndarray]:
    if n_split_repeats < 1:
        raise ValueError("n_split_repeats must be positive.")
    # end if invalid n_split_repeats

    split_measures = {
        "cvpca_pr": [],
        "positive_signal_variance": [],
        "negative_spectrum_fraction": [],
    }
    for split_index in range(n_split_repeats):
        first_half, second_half = average_repetition_halves(
            rasters, presentation_identities, stimulus_order, rng,
        )
        # Alternate the PCA-training half so odd repetition counts do not always
        # assign the smaller half to training or the larger half to validation.
        if split_index % 2 == 1:
            first_half, second_half = second_half, first_half
        # end if odd split_index
        measures = cvpca_participation_ratio(first_half, second_half)
        for measure_name, values in measures.items():
            split_measures[measure_name].append(values)
        # end for measure_name
    # end for split_index

    summary = {}
    for measure_name, values in split_measures.items():
        split_values = np.stack(values)
        summary[f"{measure_name}_splits"] = split_values
        summary[f"{measure_name}_mean"] = np.nanmean(split_values, axis=0)
        summary[f"{measure_name}_lower"] = np.nanpercentile(
            split_values, 2.5, axis=0,
        )
        summary[f"{measure_name}_upper"] = np.nanpercentile(
            split_values, 97.5, axis=0,
        )
    # end for measure_name
    return summary
# EOF
