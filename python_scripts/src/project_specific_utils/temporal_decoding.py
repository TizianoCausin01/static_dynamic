from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


CLASSIFIER_NAMES = (
    "ridge", "logistic_regression", "linear_svm", "rbf_svm",
)


"""
stimulus_identity
Extract the modality-independent stimulus identity from one presentation name.

INPUT:
    - presentation_name: str -> image or movie presentation filename
    - condition: str -> "static" or "dynamic"
    - static_response: str -> retained static frame control

OUTPUT:
    - identity: str | None -> shared stimulus identity or None when excluded
"""
def stimulus_identity(presentation_name, condition, static_response="2000ms"):
    stem = Path(presentation_name).stem
    if condition == "static":
        static_prefixes = {
            "last_frame": "img_",
            "2000ms": "img_2000ms_",
            "2250ms": "img_2250ms_",
        }
        if static_response not in static_prefixes:
            raise ValueError(
                f"static_response must be one of {sorted(static_prefixes)}."
            )
        # end if invalid static_response

        # Test the more specific prefixes before the generic img_ prefix.
        if stem.startswith(static_prefixes["2000ms"]):
            response = "2000ms"
        elif stem.startswith(static_prefixes["2250ms"]):
            response = "2250ms"
        elif stem.startswith(static_prefixes["last_frame"]):
            response = "last_frame"
        else:
            return None
        # end if static prefix

        if response != static_response:
            return None
        # end if response != static_response
        prefix = static_prefixes[response]
    elif condition == "dynamic":
        prefix = "vid_"
    else:
        raise ValueError("condition must be 'static' or 'dynamic'.")
    # end if condition

    return stem[len(prefix):] if stem.startswith(prefix) else None
# EOF


"""
select_shared_decoding_presentations
Select repeated static and dynamic presentations of the same stimulus classes.

INPUT:
    - static_names: list[str] -> static-condition presentation names
    - dynamic_names: list[str] -> dynamic-condition presentation names
    - static_response: str -> retained static frame control

OUTPUT:
    - static_indices: np.ndarray -> selected static presentation indices
    - dynamic_indices: np.ndarray -> selected dynamic presentation indices
    - static_labels: np.ndarray -> integer class label per static presentation
    - dynamic_labels: np.ndarray -> integer class label per dynamic presentation
    - shared_stimuli: list[str] -> sorted shared stimulus identities
"""
def select_shared_decoding_presentations(
        static_names, dynamic_names, static_response="2000ms",
        ):
    static_identities = [
        stimulus_identity(name, "static", static_response)
        for name in static_names
    ]
    dynamic_identities = [
        stimulus_identity(name, "dynamic", static_response)
        for name in dynamic_names
    ]

    shared_stimuli = sorted(
        {identity for identity in static_identities if identity is not None}
        & {identity for identity in dynamic_identities if identity is not None}
    )
    if len(shared_stimuli) < 2:
        raise ValueError("Need at least two shared stimulus identities.")
    # end if too few shared stimuli

    class_by_stimulus = {
        stimulus: class_index
        for class_index, stimulus in enumerate(shared_stimuli)
    }
    static_indices = np.asarray([
        index for index, identity in enumerate(static_identities)
        if identity in class_by_stimulus
    ], dtype=int)
    dynamic_indices = np.asarray([
        index for index, identity in enumerate(dynamic_identities)
        if identity in class_by_stimulus
    ], dtype=int)
    static_labels = np.asarray([
        class_by_stimulus[static_identities[index]]
        for index in static_indices
    ], dtype=int)
    dynamic_labels = np.asarray([
        class_by_stimulus[dynamic_identities[index]]
        for index in dynamic_indices
    ], dtype=int)
    return (
        static_indices, dynamic_indices, static_labels, dynamic_labels,
        shared_stimuli,
    )
# EOF


def _validate_decoding_data(rasters, labels, data_name):
    rasters = np.asarray(rasters)
    labels = np.asarray(labels)
    if rasters.ndim != 3:
        raise ValueError(
            f"{data_name} rasters must be channels x time x presentations."
        )
    # end if rasters.ndim
    if labels.ndim != 1 or len(labels) != rasters.shape[2]:
        raise ValueError(
            f"{data_name} labels must match the presentation axis."
        )
    # end if labels shape
    if not np.isfinite(rasters).all():
        raise ValueError(f"{data_name} rasters contain non-finite values.")
    # end if non-finite rasters
    return rasters, labels
# EOF


"""
make_temporal_decoder
Build a standardized, class-balanced decoder from a common regularization
parameterization. For SVM and logistic models, C is defined as 1 / alpha so a
larger alpha always means stronger regularization.

INPUT:
    - classifier_name: str -> ridge, logistic_regression, linear_svm, or rbf_svm
    - alpha: float -> positive regularization strength
    - random_seed: int -> reproducible classifier seed where applicable
    - max_iter: int -> iteration cap for iterative linear classifiers
    - rbf_gamma: str | float -> RBF kernel width passed to sklearn SVC

OUTPUT:
    - decoder: sklearn.pipeline.Pipeline -> scaled classification pipeline
"""
def make_temporal_decoder(
        classifier_name, alpha, random_seed=0, max_iter=5000,
        rbf_gamma="scale",
        ):
    if classifier_name not in CLASSIFIER_NAMES:
        raise ValueError(
            f"classifier_name must be one of {CLASSIFIER_NAMES}."
        )
    # end if invalid classifier_name
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    # end if invalid alpha
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    # end if invalid max_iter

    if classifier_name == "ridge":
        classifier = RidgeClassifier(
            alpha=alpha, class_weight="balanced",
        )
    elif classifier_name == "logistic_regression":
        classifier = LogisticRegression(
            C=1 / alpha, class_weight="balanced", solver="lbfgs",
            max_iter=max_iter, random_state=random_seed,
        )
    elif classifier_name == "linear_svm":
        classifier = LinearSVC(
            C=1 / alpha, class_weight="balanced", dual="auto",
            max_iter=max_iter, random_state=random_seed,
        )
    else:
        # The RBF model is nonlinear and substantially more computationally costly.
        classifier = SVC(
            C=1 / alpha, gamma=rbf_gamma, kernel="rbf",
            class_weight="balanced",
        )
    # end if classifier_name
    return make_pipeline(StandardScaler(), classifier)
# EOF


def _validate_alpha_values(alpha, alpha_values):
    if alpha_values is None:
        alpha_values = (alpha,)
    # end if alpha_values is None
    alpha_values = np.asarray(alpha_values, dtype=float)
    if (
            alpha_values.ndim != 1 or len(alpha_values) == 0
            or not np.isfinite(alpha_values).all()
            or np.any(alpha_values <= 0)
            ):
        raise ValueError("alpha_values must contain finite positive values.")
    # end if invalid alpha_values
    return np.unique(alpha_values)
# EOF


"""
cross_validate_decoder_alpha
Choose alpha with stratified cross-validation using only the supplied training
samples and balanced accuracy.

INPUT:
    - train_samples: np.ndarray -> presentations x neural channels
    - train_labels: np.ndarray -> integer stimulus label per presentation
    - classifier_name: str -> configured classifier family
    - alpha_values: np.ndarray -> candidate regularization strengths
    - n_splits: int -> number of inner stratified folds
    - random_seed: int -> reproducible inner-fold and classifier seed
    - max_iter: int -> iterative-classifier iteration cap
    - rbf_gamma: str | float -> RBF kernel width

OUTPUT:
    - selected_alpha: float -> best mean inner-fold regularization strength
    - mean_scores: np.ndarray -> balanced accuracy for every candidate alpha
"""
def cross_validate_decoder_alpha(
        train_samples, train_labels, classifier_name, alpha_values,
        n_splits=3, random_seed=0, max_iter=5000, rbf_gamma="scale",
        ):
    if n_splits < 2:
        raise ValueError("alpha_cv_splits must be at least two.")
    # end if n_splits
    _, class_counts = np.unique(train_labels, return_counts=True)
    if class_counts.min() < n_splits:
        raise ValueError(
            "Every training class needs at least alpha_cv_splits repetitions; "
            f"minimum count is {class_counts.min()} and requested folds are "
            f"{n_splits}."
        )
    # end if insufficient inner-fold repetitions

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed,
    )
    splits = list(splitter.split(train_samples, train_labels))
    mean_scores = np.empty(len(alpha_values), dtype=float)
    for alpha_index, candidate_alpha in enumerate(alpha_values):
        candidate_scores = []
        for fit_indices, validation_indices in splits:
            decoder = make_temporal_decoder(
                classifier_name, candidate_alpha, random_seed=random_seed,
                max_iter=max_iter, rbf_gamma=rbf_gamma,
            )
            decoder.fit(train_samples[fit_indices], train_labels[fit_indices])
            predictions = decoder.predict(train_samples[validation_indices])
            candidate_scores.append(balanced_accuracy_score(
                train_labels[validation_indices], predictions,
            ))
        # end for fit_indices, validation_indices
        mean_scores[alpha_index] = np.mean(candidate_scores)
    # end for alpha_index, candidate_alpha

    # Prefer stronger regularization when numerical ties have equal CV accuracy.
    best_score = np.max(mean_scores)
    tied_indices = np.flatnonzero(np.isclose(mean_scores, best_score))
    selected_alpha = float(np.max(alpha_values[tied_indices]))
    return selected_alpha, mean_scores
# EOF


"""
temporal_generalization
Train a configured class-balanced decoder at every source time and evaluate it
at every target time. Scaling and optional alpha selection use training data only.

INPUT:
    - train_rasters: np.ndarray -> channels x train times x presentations
    - train_labels: np.ndarray -> integer class label per training presentation
    - test_rasters: np.ndarray -> channels x test times x presentations
    - test_labels: np.ndarray -> integer class label per testing presentation
    - classifier_name: str -> classifier family selected from CLASSIFIER_NAMES
    - alpha: float -> fixed regularization when alpha_values is None
    - alpha_values: tuple[float, ...] | None -> inner-CV candidates
    - alpha_cv_splits: int -> inner stratified folds used to select alpha
    - random_seed: int -> reproducible inner folds and classifier seed
    - max_iter: int -> iterative-classifier iteration cap
    - rbf_gamma: str | float -> RBF kernel width
    - return_selected_alphas: bool -> also return alpha chosen at each train time
    - verbose: bool -> print compact progress updates

OUTPUT:
    - scores: np.ndarray -> train times x test times balanced accuracy
    - selected_alphas: np.ndarray -> optional alpha per training time
"""
def temporal_generalization(
        train_rasters, train_labels, test_rasters, test_labels,
        classifier_name="ridge", alpha=1.0, alpha_values=None,
        alpha_cv_splits=3, random_seed=0, max_iter=5000,
        rbf_gamma="scale", return_selected_alphas=False, verbose=False,
        ):
    train_rasters, train_labels = _validate_decoding_data(
        train_rasters, train_labels, "Training",
    )
    test_rasters, test_labels = _validate_decoding_data(
        test_rasters, test_labels, "Testing",
    )
    if train_rasters.shape[0] != test_rasters.shape[0]:
        raise ValueError("Training and testing data must use the same channels.")
    # end if channel counts differ
    if set(np.unique(train_labels)) != set(np.unique(test_labels)):
        raise ValueError("Training and testing data must contain the same classes.")
    # end if class sets differ
    alpha_values = _validate_alpha_values(alpha, alpha_values)

    # Stack all target-time samples once; reshape predictions back by time.
    test_by_time = test_rasters.transpose(1, 2, 0)
    flat_test = test_by_time.reshape(-1, test_rasters.shape[0])
    scores = np.empty(
        (train_rasters.shape[1], test_rasters.shape[1]), dtype=float,
    )
    selected_alphas = np.empty(train_rasters.shape[1], dtype=float)

    for train_time_index in range(train_rasters.shape[1]):
        train_samples = train_rasters[:, train_time_index, :].T
        if len(alpha_values) == 1:
            selected_alpha = float(alpha_values[0])
        else:
            selected_alpha, _ = cross_validate_decoder_alpha(
                train_samples, train_labels, classifier_name, alpha_values,
                n_splits=alpha_cv_splits, random_seed=random_seed,
                max_iter=max_iter, rbf_gamma=rbf_gamma,
            )
        # end if one alpha candidate
        selected_alphas[train_time_index] = selected_alpha

        decoder = make_temporal_decoder(
            classifier_name, selected_alpha, random_seed=random_seed,
            max_iter=max_iter, rbf_gamma=rbf_gamma,
        )
        decoder.fit(train_samples, train_labels)
        predictions = decoder.predict(flat_test).reshape(
            test_rasters.shape[1], test_rasters.shape[2],
        )
        scores[train_time_index] = [
            balanced_accuracy_score(test_labels, time_predictions)
            for time_predictions in predictions
        ]

        if verbose and (
                train_time_index == 0
                or (train_time_index + 1) % 25 == 0
                or train_time_index + 1 == train_rasters.shape[1]
                ):
            print(
                f"trained {train_time_index + 1}/"
                f"{train_rasters.shape[1]} timepoints "
                f"(alpha={selected_alpha:g})"
            )
        # end if progress update
    # end for train_time_index
    if return_selected_alphas:
        return scores, selected_alphas
    # end if return_selected_alphas
    return scores
# EOF


"""
cross_validated_temporal_generalization
Estimate within-condition temporal generalization with stratified folds.

INPUT:
    - rasters: np.ndarray -> channels x time x repeated presentations
    - labels: np.ndarray -> integer stimulus label per presentation
    - n_splits: int -> number of stratified cross-validation folds
    - classifier_name: str -> classifier family selected from CLASSIFIER_NAMES
    - alpha: float -> fixed regularization when alpha_values is None
    - alpha_values: tuple[float, ...] | None -> nested-CV candidates
    - alpha_cv_splits: int -> inner folds used to select alpha
    - random_seed: int -> reproducible fold-shuffling seed
    - max_iter: int -> iterative-classifier iteration cap
    - rbf_gamma: str | float -> RBF kernel width
    - return_selected_alphas: bool -> also return fold x train-time alphas
    - verbose: bool -> print one progress line per completed fold

OUTPUT:
    - mean_scores: np.ndarray -> mean train-time x test-time balanced accuracy
    - fold_scores: np.ndarray -> folds x train times x test times accuracy
    - fold_alphas: np.ndarray -> optional folds x train times selected alpha
"""
def cross_validated_temporal_generalization(
        rasters, labels, n_splits=5, classifier_name="ridge", alpha=1.0,
        alpha_values=None, alpha_cv_splits=3, random_seed=0,
        max_iter=5000, rbf_gamma="scale", return_selected_alphas=False,
        verbose=False,
        ):
    rasters, labels = _validate_decoding_data(rasters, labels, "Within-condition")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    # end if n_splits
    _, class_counts = np.unique(labels, return_counts=True)
    if class_counts.min() < n_splits:
        raise ValueError(
            "Every stimulus needs at least n_splits repetitions; "
            f"minimum count is {class_counts.min()} and n_splits is {n_splits}."
        )
    # end if too few repetitions

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed,
    )
    fold_scores = []
    fold_alphas = []
    for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels), start=1,
            ):
        scores, selected_alphas = temporal_generalization(
            rasters[:, :, train_indices], labels[train_indices],
            rasters[:, :, test_indices], labels[test_indices],
            classifier_name=classifier_name, alpha=alpha,
            alpha_values=alpha_values, alpha_cv_splits=alpha_cv_splits,
            random_seed=random_seed + fold_index, max_iter=max_iter,
            rbf_gamma=rbf_gamma, return_selected_alphas=True,
            verbose=False,
        )
        fold_scores.append(scores)
        fold_alphas.append(selected_alphas)
        if verbose:
            print(f"completed fold {fold_index}/{n_splits}")
        # end if verbose
    # end for fold_index

    fold_scores = np.stack(fold_scores)
    fold_alphas = np.stack(fold_alphas)
    if return_selected_alphas:
        return fold_scores.mean(axis=0), fold_scores, fold_alphas
    # end if return_selected_alphas
    return fold_scores.mean(axis=0), fold_scores
# EOF
