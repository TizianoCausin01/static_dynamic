from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LocalDMDResult:
    """Container for one local HAVOK/DMD transition fit."""

    transition_matrix: np.ndarray
    current_coordinates: np.ndarray
    next_coordinates: np.ndarray
    predicted_coordinates: np.ndarray
    singular_values: np.ndarray
    cumulative_explained_variance: np.ndarray
    prediction_r2: float
    prediction_mse: float
    time_index: int
    target_time_index: int
    n_delays: int
    delay_interval: int


@dataclass
class PAVFResult:
    """Container for Procrustes Analysis over Vector Fields outputs."""

    transformation: np.ndarray
    aligned_matrix: np.ndarray
    angular_distance: float
    cosine_similarity: float
    normalized_similarity: float
    normalized_frobenius_distance: float
    loss_curve: np.ndarray


@dataclass
class SplitHalfDSAResult:
    """Container for repeated within-stimulus dynamic split-half comparisons."""

    angular_distances: np.ndarray
    cosine_similarities: np.ndarray
    normalized_similarities: np.ndarray
    first_prediction_r2: np.ndarray
    second_prediction_r2: np.ndarray


@dataclass
class CrossTemporalDSAResult:
    """Container for a dynamic-time by static-time local DSA analysis."""

    angular_distances: np.ndarray
    normalized_similarities: np.ndarray
    static_time_indices: np.ndarray
    dynamic_time_indices: np.ndarray
    static_prediction_r2: np.ndarray
    dynamic_prediction_r2: np.ndarray


"""
delay_embed_trials
Construct a Hankel delay embedding independently within every trial.

INPUT:
    - trials: np.ndarray -> trials x time x channels data
    - n_delays: int -> number of states in each embedding; 1 means no delays
    - delay_interval: int -> samples between adjacent embedded states

OUTPUT:
    - embedding: np.ndarray -> trials x valid time x (channels * n_delays)
"""
def delay_embed_trials(
        trials: np.ndarray,
        n_delays: int = 1,
        delay_interval: int = 1,
        ) -> np.ndarray:
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 3:
        raise ValueError("trials must have shape trials x time x channels.")
    # end if trials.ndim
    if not isinstance(n_delays, int) or n_delays < 1:
        raise ValueError("n_delays must be a positive integer.")
    # end if n_delays
    if not isinstance(delay_interval, int) or delay_interval < 1:
        raise ValueError("delay_interval must be a positive integer.")
    # end if delay_interval

    history_samples = (n_delays - 1) * delay_interval
    valid_timepoints = trials.shape[1] - history_samples
    if valid_timepoints < 1:
        raise ValueError(
            "The requested delay embedding is longer than the available trial."
        )
    # end if valid_timepoints

    # Concatenate the current state followed by progressively older states,
    # matching [x(t), x(t-tau), ...] in Ostrow et al.'s HAVOK construction.
    delayed_states = []
    for delay_index in range(n_delays):
        start = history_samples - delay_index * delay_interval
        stop = start + valid_timepoints
        delayed_states.append(trials[:, start:stop, :])
    # end for delay_index
    return np.concatenate(delayed_states, axis=2)
# EOF


"""
fit_local_dmd
Fit one reduced-rank HAVOK/DMD transition at a selected response timepoint.
The fit pools repetitions as independent trials and never joins trial boundaries.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations neural data
    - time_index: int -> input-state sample index t
    - rank: int -> shared DMD rank used for between-system comparison
    - n_delays: int -> number of states per delay embedding; 1 disables delays
    - delay_interval: int -> samples separating adjacent delayed states
    - steps_ahead: int -> prediction horizon in samples
    - ridge: float -> non-negative ridge penalty for the transition regression

OUTPUT:
    - result: LocalDMDResult -> fitted transition and diagnostic coordinates
"""
def fit_local_dmd(
        rasters: np.ndarray,
        time_index: int,
        rank: int,
        n_delays: int = 1,
        delay_interval: int = 1,
        steps_ahead: int = 1,
        ridge: float = 1e-8,
        ) -> LocalDMDResult:
    rasters = np.asarray(rasters, dtype=np.float64)
    if rasters.ndim != 3:
        raise ValueError("rasters must have shape channels x time x presentations.")
    # end if rasters.ndim
    if not isinstance(time_index, (int, np.integer)):
        raise TypeError("time_index must be an integer.")
    # end if time_index
    time_index = int(time_index)
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer.")
    # end if rank
    if not isinstance(steps_ahead, int) or steps_ahead < 1:
        raise ValueError("steps_ahead must be a positive integer.")
    # end if steps_ahead
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    # end if ridge

    history_samples = (n_delays - 1) * delay_interval
    first_sample = time_index - history_samples
    target_time_index = time_index + steps_ahead
    if first_sample < 0 or target_time_index >= rasters.shape[1]:
        raise ValueError(
            "The selected transition does not have enough history or future data "
            "for the requested delays and prediction horizon."
        )
    # end if local window bounds

    # Keep exactly the samples needed to produce one input-target pair per trial.
    local_trials = rasters[
        :, first_sample:target_time_index + 1, :
    ].transpose(2, 1, 0)
    hankel = delay_embed_trials(local_trials, n_delays, delay_interval)
    expected_hankel_time = steps_ahead + 1
    if hankel.shape[1] != expected_hankel_time:
        raise RuntimeError("Unexpected local Hankel time dimension.")
    # end if hankel.shape

    # H.T = U S V.T; rows of V are the whitened eigen-time-delay coordinates.
    flattened_hankel = hankel.reshape(-1, hankel.shape[2])
    _, singular_values, right_coordinates_t = np.linalg.svd(
        flattened_hankel.T, full_matrices=False,
    )
    all_coordinates = right_coordinates_t.T
    maximum_rank = all_coordinates.shape[1]
    if rank > maximum_rank:
        raise ValueError(
            f"rank={rank} exceeds the local HAVOK maximum rank {maximum_rank}."
        )
    # end if rank > maximum_rank

    ranked_coordinates = all_coordinates[:, :rank].reshape(
        hankel.shape[0], hankel.shape[1], rank,
    )
    current_coordinates = ranked_coordinates[:, 0, :]
    next_coordinates = ranked_coordinates[:, steps_ahead, :]

    # Solve Z(t + dt) = A Z(t) using the paper's ridge least-squares form.
    gram_matrix = current_coordinates.T @ current_coordinates
    regression_rhs = current_coordinates.T @ next_coordinates
    transition_matrix = np.linalg.solve(
        gram_matrix + ridge * np.eye(rank), regression_rhs,
    ).T
    predicted_coordinates = current_coordinates @ transition_matrix.T

    residual_sum_squares = np.sum(
        (next_coordinates - predicted_coordinates) ** 2
    )
    total_sum_squares = np.sum(
        (next_coordinates - next_coordinates.mean(axis=0, keepdims=True)) ** 2
    )
    prediction_r2 = (
        np.nan if total_sum_squares == 0
        else 1 - residual_sum_squares / total_sum_squares
    )
    prediction_mse = np.mean(
        (next_coordinates - predicted_coordinates) ** 2
    )

    singular_variance = singular_values ** 2
    cumulative_explained_variance = np.cumsum(singular_variance)
    cumulative_explained_variance /= cumulative_explained_variance[-1]
    return LocalDMDResult(
        transition_matrix=transition_matrix,
        current_coordinates=current_coordinates,
        next_coordinates=next_coordinates,
        predicted_coordinates=predicted_coordinates,
        singular_values=singular_values,
        cumulative_explained_variance=cumulative_explained_variance,
        prediction_r2=float(prediction_r2),
        prediction_mse=float(prediction_mse),
        time_index=time_index,
        target_time_index=target_time_index,
        n_delays=n_delays,
        delay_interval=delay_interval,
    )
# EOF


"""
split_presentation_indices
Split repetitions independently within every stimulus without averaging trials.

INPUT:
    - presentation_identities: list[str] -> stimulus identity per presentation
    - stimulus_order: list[str] -> identities retained in the analysis
    - rng: np.random.Generator -> random generator controlling the split

OUTPUT:
    - first_indices: np.ndarray -> presentation indices in the first half
    - second_indices: np.ndarray -> presentation indices in the second half
"""
def split_presentation_indices(
        presentation_identities: list[str],
        stimulus_order: list[str],
        rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
    indices_by_identity = {}
    for presentation_index, identity in enumerate(presentation_identities):
        indices_by_identity.setdefault(identity, []).append(presentation_index)
    # end for presentation_index, identity

    first_indices = []
    second_indices = []
    for identity in stimulus_order:
        stimulus_indices = np.asarray(
            indices_by_identity.get(identity, []), dtype=int,
        )
        if len(stimulus_indices) < 2:
            raise ValueError(
                f"Stimulus {identity!r} has fewer than two repetitions."
            )
        # end if len(stimulus_indices)
        shuffled_indices = rng.permutation(stimulus_indices)
        split_index = len(shuffled_indices) // 2
        first_indices.extend(shuffled_indices[:split_index])
        second_indices.extend(shuffled_indices[split_index:])
    # end for identity
    return np.asarray(first_indices), np.asarray(second_indices)
# EOF


"""
procrustes_vector_fields
Align two square transition matrices with the Ostrow et al. orthogonal
similarity transform, minimizing ||A - C B C^-1|| over both components of O(r).

INPUT:
    - reference_matrix: np.ndarray -> square transition matrix A
    - comparison_matrix: np.ndarray -> square transition matrix B
    - n_iterations: int -> Adam optimization steps per initialization
    - learning_rate: float -> Adam learning rate
    - n_restarts: int -> random starts for each component of O(r)
    - random_seed: int -> reproducible optimizer initialization
    - device: str -> PyTorch device

OUTPUT:
    - result: PAVFResult -> optimal transform and angular similarity metrics
"""
def procrustes_vector_fields(
        reference_matrix: np.ndarray,
        comparison_matrix: np.ndarray,
        n_iterations: int = 300,
        learning_rate: float = 0.01,
        n_restarts: int = 3,
        random_seed: int = 0,
        device: str = "cpu",
        ) -> PAVFResult:
    reference_matrix = np.asarray(reference_matrix, dtype=np.float64)
    comparison_matrix = np.asarray(comparison_matrix, dtype=np.float64)
    if (
            reference_matrix.ndim != 2
            or reference_matrix.shape[0] != reference_matrix.shape[1]
            or reference_matrix.shape != comparison_matrix.shape
            ):
        raise ValueError("Both transition matrices must be square and the same size.")
    # end if matrix shapes
    if n_iterations < 1 or n_restarts < 1 or learning_rate <= 0:
        raise ValueError("Optimizer settings must all be positive.")
    # end if optimizer settings

    torch_device = torch.device(device)
    reference = torch.as_tensor(
        reference_matrix, dtype=torch.float64, device=torch_device,
    )
    comparison = torch.as_tensor(
        comparison_matrix, dtype=torch.float64, device=torch_device,
    )
    reference_norm = torch.linalg.norm(reference)
    comparison_norm = torch.linalg.norm(comparison)
    if reference_norm == 0 or comparison_norm == 0:
        raise ValueError("Transition matrices must have non-zero Frobenius norm.")
    # end if matrix norm
    reference_normalized = reference / reference_norm
    comparison_normalized = comparison / comparison_norm

    rank = reference.shape[0]
    identity = torch.eye(rank, dtype=torch.float64, device=torch_device)
    reflection = identity.clone()
    if rank > 1:
        reflection[[0, 1], :] = reflection[[1, 0], :]
    else:
        reflection[0, 0] = -1
    # end if rank > 1

    best_loss = np.inf
    best_transformation = None
    best_curve = None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    # SVD-derived bases give the optimizer informed starts when the two systems
    # already have approximately matching singular directions.
    reference_left, _, reference_right_t = torch.linalg.svd(reference_normalized)
    comparison_left, _, comparison_right_t = torch.linalg.svd(comparison_normalized)
    left_base = reference_left @ comparison_left.T
    right_base = reference_right_t.T @ comparison_right_t
    combined_base = left_base + right_base
    combined_left, _, combined_right_t = torch.linalg.svd(combined_base)
    combined_base = combined_left @ combined_right_t
    informed_bases = [identity, combined_base, left_base, right_base]

    # The Cayley map covers SO(r). Random orthogonal bases explore each of the
    # two disconnected components instead of starting every fit near identity.
    for determinant_sign in (1, -1):
        for restart_index in range(n_restarts):
            if restart_index < len(informed_bases):
                base_transformation = informed_bases[restart_index].clone()
            else:
                random_matrix = torch.randn(
                    (rank, rank), generator=generator, dtype=torch.float64,
                ).to(torch_device)
                base_transformation, _ = torch.linalg.qr(random_matrix)
            # end if restart_index

            base_determinant = torch.linalg.det(base_transformation)
            if torch.sign(base_determinant) != determinant_sign:
                base_transformation = base_transformation @ reflection
            # end if base determinant

            component_comparison = (
                base_transformation
                @ comparison_normalized
                @ base_transformation.T
            )
            initial_raw = torch.zeros((rank, rank), dtype=torch.float64)
            raw_matrix = torch.nn.Parameter(initial_raw.to(torch_device))
            optimizer = torch.optim.Adam([raw_matrix], lr=learning_rate)
            losses = []

            for iteration_index in range(n_iterations):
                optimizer.zero_grad()
                skew_matrix = raw_matrix - raw_matrix.T
                cayley_matrix = torch.linalg.solve(
                    identity + skew_matrix, identity - skew_matrix,
                )
                aligned = cayley_matrix @ component_comparison @ cayley_matrix.T
                loss = torch.sum((reference_normalized - aligned) ** 2)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            # end for iteration_index

            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_transformation = (
                    cayley_matrix.detach() @ base_transformation
                ).cpu().numpy()
                best_curve = np.asarray(losses)
            # end if losses[-1]
        # end for restart_index
    # end for determinant_sign

    aligned_matrix = (
        best_transformation @ comparison_matrix @ best_transformation.T
    )
    cosine_similarity = np.trace(
        reference_matrix.T @ aligned_matrix
    ) / (
        np.linalg.norm(reference_matrix) * np.linalg.norm(comparison_matrix)
    )
    cosine_similarity = float(np.clip(cosine_similarity, -1, 1))
    angular_distance = float(np.arccos(cosine_similarity))
    normalized_similarity = float(1 - angular_distance / np.pi)
    normalized_frobenius_distance = float(np.sqrt(max(best_loss, 0)))
    return PAVFResult(
        transformation=best_transformation,
        aligned_matrix=aligned_matrix,
        angular_distance=angular_distance,
        cosine_similarity=cosine_similarity,
        normalized_similarity=normalized_similarity,
        normalized_frobenius_distance=normalized_frobenius_distance,
        loss_curve=best_curve,
    )
# EOF


"""
compute_split_half_dsa
Estimate a dynamic-condition DSA reliability ceiling by repeatedly splitting
raw repetitions within each stimulus, fitting one local DMD per half, and
comparing the two transition matrices with the same PAVF metric.

INPUT:
    - rasters: np.ndarray -> channels x time x presentations dynamic data
    - presentation_identities: list[str] -> stimulus identity per presentation
    - stimulus_order: list[str] -> identities retained in both halves
    - time_index: int -> selected local transition timepoint
    - rank: int -> DMD rank shared by both halves
    - n_split_repeats: int -> number of random within-stimulus splits
    - random_seed: int -> random split and optimizer seed
    - n_delays: int -> number of states per delay embedding
    - delay_interval: int -> samples between embedded states
    - steps_ahead: int -> prediction horizon in samples
    - ridge: float -> DMD ridge penalty
    - pavf_iterations: int -> PAVF Adam steps per initialization
    - pavf_learning_rate: float -> PAVF Adam learning rate
    - pavf_restarts: int -> optimizer starts per component of O(r)
    - device: str -> PyTorch device

OUTPUT:
    - result: SplitHalfDSAResult -> split-wise DSA scores and fit diagnostics
"""
def compute_split_half_dsa(
        rasters: np.ndarray,
        presentation_identities: list[str],
        stimulus_order: list[str],
        time_index: int,
        rank: int,
        n_split_repeats: int,
        random_seed: int = 0,
        n_delays: int = 1,
        delay_interval: int = 1,
        steps_ahead: int = 1,
        ridge: float = 1e-8,
        pavf_iterations: int = 300,
        pavf_learning_rate: float = 0.01,
        pavf_restarts: int = 3,
        device: str = "cpu",
        ) -> SplitHalfDSAResult:
    rasters = np.asarray(rasters)
    if rasters.ndim != 3 or rasters.shape[2] != len(presentation_identities):
        raise ValueError(
            "rasters and presentation_identities must describe matching presentations."
        )
    # end if input shapes
    if n_split_repeats < 1:
        raise ValueError("n_split_repeats must be positive.")
    # end if n_split_repeats

    rng = np.random.default_rng(random_seed)
    angular_distances = []
    cosine_similarities = []
    normalized_similarities = []
    first_prediction_r2 = []
    second_prediction_r2 = []

    for split_index in range(n_split_repeats):
        first_indices, second_indices = split_presentation_indices(
            presentation_identities, stimulus_order, rng,
        )
        first_dmd = fit_local_dmd(
            rasters[:, :, first_indices], time_index=time_index, rank=rank,
            n_delays=n_delays, delay_interval=delay_interval,
            steps_ahead=steps_ahead, ridge=ridge,
        )
        second_dmd = fit_local_dmd(
            rasters[:, :, second_indices], time_index=time_index, rank=rank,
            n_delays=n_delays, delay_interval=delay_interval,
            steps_ahead=steps_ahead, ridge=ridge,
        )
        split_comparison = procrustes_vector_fields(
            first_dmd.transition_matrix, second_dmd.transition_matrix,
            n_iterations=pavf_iterations,
            learning_rate=pavf_learning_rate,
            n_restarts=pavf_restarts,
            random_seed=random_seed + split_index + 1,
            device=device,
        )
        angular_distances.append(split_comparison.angular_distance)
        cosine_similarities.append(split_comparison.cosine_similarity)
        normalized_similarities.append(split_comparison.normalized_similarity)
        first_prediction_r2.append(first_dmd.prediction_r2)
        second_prediction_r2.append(second_dmd.prediction_r2)
    # end for split_index

    return SplitHalfDSAResult(
        angular_distances=np.asarray(angular_distances),
        cosine_similarities=np.asarray(cosine_similarities),
        normalized_similarities=np.asarray(normalized_similarities),
        first_prediction_r2=np.asarray(first_prediction_r2),
        second_prediction_r2=np.asarray(second_prediction_r2),
    )
# EOF


"""
batched_pavf_angular_distances
Compute PAVF angular distances for matching batches of transition-matrix pairs.
All pairs and Cayley-map initializations are optimized in parallel.

INPUT:
    - reference_matrices: np.ndarray -> pairs x rank x rank matrices A
    - comparison_matrices: np.ndarray -> pairs x rank x rank matrices B
    - n_iterations: int -> Adam steps for every pair
    - learning_rate: float -> Adam learning rate
    - batch_size: int -> matrix pairs optimized simultaneously
    - device: str -> PyTorch device

OUTPUT:
    - angular_distances: np.ndarray -> one optimized distance per matrix pair
"""
def batched_pavf_angular_distances(
        reference_matrices: np.ndarray,
        comparison_matrices: np.ndarray,
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 512,
        device: str = "cpu",
        ) -> np.ndarray:
    reference_matrices = np.asarray(reference_matrices, dtype=np.float64)
    comparison_matrices = np.asarray(comparison_matrices, dtype=np.float64)
    if (
            reference_matrices.ndim != 3
            or reference_matrices.shape != comparison_matrices.shape
            or reference_matrices.shape[1] != reference_matrices.shape[2]
            ):
        raise ValueError(
            "Transition inputs must be matching pairs x rank x rank arrays."
        )
    # end if matrix shapes
    if n_iterations < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("PAVF scan optimizer settings must be positive.")
    # end if optimizer settings

    torch_device = torch.device(device)
    rank = reference_matrices.shape[1]
    all_distances = []

    for batch_start in range(0, len(reference_matrices), batch_size):
        batch_stop = min(batch_start + batch_size, len(reference_matrices))
        reference = torch.as_tensor(
            reference_matrices[batch_start:batch_stop],
            dtype=torch.float64, device=torch_device,
        )
        comparison = torch.as_tensor(
            comparison_matrices[batch_start:batch_stop],
            dtype=torch.float64, device=torch_device,
        )
        reference /= torch.linalg.norm(
            reference, dim=(1, 2), keepdim=True,
        )
        comparison /= torch.linalg.norm(
            comparison, dim=(1, 2), keepdim=True,
        )

        pair_count = reference.shape[0]
        identity = torch.eye(
            rank, dtype=torch.float64, device=torch_device,
        ).expand(pair_count, -1, -1)
        reflection = torch.eye(
            rank, dtype=torch.float64, device=torch_device,
        )
        if rank > 1:
            reflection[[0, 1], :] = reflection[[1, 0], :]
        else:
            reflection[0, 0] = -1
        # end if rank > 1
        reflection = reflection.expand(pair_count, -1, -1)

        # An SVD-informed basis substantially reduces the iterations needed for
        # a cross-temporal scan while the Cayley optimization remains the metric.
        reference_left, _, reference_right_t = torch.linalg.svd(reference)
        comparison_left, _, comparison_right_t = torch.linalg.svd(comparison)
        left_base = reference_left @ comparison_left.transpose(1, 2)
        right_base = (
            reference_right_t.transpose(1, 2) @ comparison_right_t
        )
        combined_left, _, combined_right_t = torch.linalg.svd(
            left_base + right_base
        )
        combined_base = combined_left @ combined_right_t
        negative_determinant = torch.linalg.det(combined_base) < 0
        combined_base[negative_determinant] = (
            combined_base[negative_determinant]
            @ reflection[negative_determinant]
        )

        # Evaluate identity/SVD starts in both components of O(rank).
        base_transformations = torch.stack([
            identity,
            reflection,
            combined_base,
            combined_base @ reflection,
        ], dim=1)
        comparison_by_start = comparison[:, None, :, :].expand(-1, 4, -1, -1)
        base_comparison = (
            base_transformations
            @ comparison_by_start
            @ base_transformations.transpose(2, 3)
        )
        target = reference[:, None, :, :].expand(-1, 4, -1, -1)
        identity_by_start = torch.eye(
            rank, dtype=torch.float64, device=torch_device,
        ).expand(pair_count, 4, -1, -1)
        raw_matrix = torch.nn.Parameter(torch.zeros_like(identity_by_start))
        optimizer = torch.optim.Adam([raw_matrix], lr=learning_rate)

        for iteration_index in range(n_iterations):
            optimizer.zero_grad()
            skew_matrix = raw_matrix - raw_matrix.transpose(2, 3)
            cayley_matrix = torch.linalg.solve(
                identity_by_start + skew_matrix,
                identity_by_start - skew_matrix,
            )
            aligned = (
                cayley_matrix
                @ base_comparison
                @ cayley_matrix.transpose(2, 3)
            )
            losses = torch.sum((target - aligned) ** 2, dim=(2, 3))
            losses.sum().backward()
            optimizer.step()
        # end for iteration_index

        # For unit-Frobenius matrices, ||A-B||^2 = 2 - 2 cos(theta).
        best_losses = losses.min(dim=1).values.detach()
        cosine_similarities = torch.clamp(1 - best_losses / 2, -1, 1)
        batch_distances = torch.arccos(cosine_similarities)
        all_distances.append(batch_distances.cpu().numpy())
    # end for batch_start
    return np.concatenate(all_distances)
# EOF


"""
compute_cross_temporal_dsa
Fit one local DMD at every requested static and dynamic timepoint, then compare
every static-dynamic transition pair with batched PAVF optimization.

INPUT:
    - static_rasters: np.ndarray -> channels x time x static presentations
    - dynamic_rasters: np.ndarray -> channels x time x dynamic presentations
    - static_time_indices: np.ndarray -> static input-state sample indices
    - dynamic_time_indices: np.ndarray -> dynamic input-state sample indices
    - rank: int -> shared DMD rank
    - n_delays: int -> states per delay embedding
    - delay_interval: int -> samples between embedded states
    - steps_ahead: int -> prediction horizon in samples
    - ridge: float -> DMD ridge penalty
    - pavf_iterations: int -> Adam steps for the cross-temporal scan
    - pavf_learning_rate: float -> scan Adam learning rate
    - pavf_batch_size: int -> matrix pairs optimized simultaneously
    - device: str -> PyTorch device

OUTPUT:
    - result: CrossTemporalDSAResult -> DSA matrices, indices, and fit quality
"""
def compute_cross_temporal_dsa(
        static_rasters: np.ndarray,
        dynamic_rasters: np.ndarray,
        static_time_indices: np.ndarray,
        dynamic_time_indices: np.ndarray,
        rank: int,
        n_delays: int = 1,
        delay_interval: int = 1,
        steps_ahead: int = 1,
        ridge: float = 1e-8,
        pavf_iterations: int = 100,
        pavf_learning_rate: float = 0.01,
        pavf_batch_size: int = 512,
        device: str = "cpu",
        ) -> CrossTemporalDSAResult:
    static_time_indices = np.asarray(static_time_indices, dtype=int)
    dynamic_time_indices = np.asarray(dynamic_time_indices, dtype=int)
    if static_time_indices.ndim != 1 or dynamic_time_indices.ndim != 1:
        raise ValueError("Time indices must be one-dimensional arrays.")
    # end if time indices
    if len(static_time_indices) == 0 or len(dynamic_time_indices) == 0:
        raise ValueError("The cross-temporal DSA scan has no timepoints.")
    # end if empty time indices

    static_fits = [
        fit_local_dmd(
            static_rasters, time_index=time_index, rank=rank,
            n_delays=n_delays, delay_interval=delay_interval,
            steps_ahead=steps_ahead, ridge=ridge,
        )
        for time_index in static_time_indices
    ]
    dynamic_fits = [
        fit_local_dmd(
            dynamic_rasters, time_index=time_index, rank=rank,
            n_delays=n_delays, delay_interval=delay_interval,
            steps_ahead=steps_ahead, ridge=ridge,
        )
        for time_index in dynamic_time_indices
    ]
    static_matrices = np.stack([
        fit.transition_matrix for fit in static_fits
    ])
    dynamic_matrices = np.stack([
        fit.transition_matrix for fit in dynamic_fits
    ])

    # Flatten dynamic rows by static columns to match the displayed matrix.
    reference_pairs = np.tile(
        static_matrices, (len(dynamic_matrices), 1, 1),
    )
    comparison_pairs = np.repeat(
        dynamic_matrices, len(static_matrices), axis=0,
    )
    angular_distances = batched_pavf_angular_distances(
        reference_pairs, comparison_pairs,
        n_iterations=pavf_iterations,
        learning_rate=pavf_learning_rate,
        batch_size=pavf_batch_size,
        device=device,
    ).reshape(len(dynamic_matrices), len(static_matrices))

    return CrossTemporalDSAResult(
        angular_distances=angular_distances,
        normalized_similarities=1 - angular_distances / np.pi,
        static_time_indices=static_time_indices,
        dynamic_time_indices=dynamic_time_indices,
        static_prediction_r2=np.asarray([
            fit.prediction_r2 for fit in static_fits
        ]),
        dynamic_prediction_r2=np.asarray([
            fit.prediction_r2 for fit in dynamic_fits
        ]),
    )
# EOF


"""
vector_field_grid
Evaluate the first two coordinates of a discrete linear transition as a 2D
vector field, using observed state coordinates to set robust plotting limits.

INPUT:
    - transition_matrix: np.ndarray -> square discrete transition matrix A
    - coordinates: np.ndarray -> observed states x DMD coordinates
    - grid_size: int -> points along each plot dimension
    - limit_quantile: float -> symmetric coordinate-limit quantile

OUTPUT:
    - x_grid, y_grid: np.ndarray -> 2D state-coordinate grid
    - u_grid, v_grid: np.ndarray -> displacement (A - I)z on the grid
"""
def vector_field_grid(
        transition_matrix: np.ndarray,
        coordinates: np.ndarray,
        grid_size: int = 15,
        limit_quantile: float = 0.98,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if transition_matrix.ndim != 2 or transition_matrix.shape[0] < 2:
        raise ValueError("transition_matrix must contain at least two coordinates.")
    # end if transition_matrix
    if coordinates.ndim != 2 or coordinates.shape[1] != transition_matrix.shape[0]:
        raise ValueError("coordinates must match the transition-matrix rank.")
    # end if coordinates
    if grid_size < 2 or not 0 < limit_quantile <= 1:
        raise ValueError("Invalid vector-field grid settings.")
    # end if grid settings

    coordinate_limit = np.quantile(np.abs(coordinates[:, :2]), limit_quantile)
    if not np.isfinite(coordinate_limit) or coordinate_limit == 0:
        coordinate_limit = 1.0
    # end if coordinate_limit
    axis_values = np.linspace(-coordinate_limit, coordinate_limit, grid_size)
    x_grid, y_grid = np.meshgrid(axis_values, axis_values)

    # Higher coordinates are fixed at zero so the plot shows the PC1-PC2 plane.
    grid_states = np.zeros((x_grid.size, transition_matrix.shape[0]))
    grid_states[:, 0] = x_grid.ravel()
    grid_states[:, 1] = y_grid.ravel()
    next_states = grid_states @ transition_matrix.T
    displacements = next_states - grid_states
    u_grid = displacements[:, 0].reshape(x_grid.shape)
    v_grid = displacements[:, 1].reshape(y_grid.shape)
    return x_grid, y_grid, u_grid, v_grid
# EOF
