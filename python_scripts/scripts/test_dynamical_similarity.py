import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from project_specific_utils.dynamical_similarity import (
    batched_pavf_angular_distances, delay_embed_trials, fit_local_dmd,
    procrustes_vector_fields,
)


def test_delay_embedding_shape_and_order():
    trials = np.arange(2 * 5 * 2).reshape(2, 5, 2)
    embedded = delay_embed_trials(trials, n_delays=3, delay_interval=1)

    assert embedded.shape == (2, 3, 6)
    np.testing.assert_array_equal(
        embedded[0, 0], np.concatenate([trials[0, 2], trials[0, 1], trials[0, 0]]),
    )
# EOF


def test_local_dmd_predicts_linear_trials():
    rng = np.random.default_rng(1)
    true_transition = np.array([
        [0.90, -0.15, 0.00],
        [0.10, 0.85, 0.05],
        [0.00, 0.00, 0.75],
    ])
    n_trials = 300
    n_timepoints = 8
    trajectories = np.empty((n_trials, n_timepoints, 3))
    trajectories[:, 0, :] = rng.normal(size=(n_trials, 3))
    for time_index in range(1, n_timepoints):
        trajectories[:, time_index, :] = (
            trajectories[:, time_index - 1, :] @ true_transition.T
        )
    # end for time_index

    rasters = trajectories.transpose(2, 1, 0)
    result = fit_local_dmd(rasters, time_index=4, rank=3)
    assert result.prediction_r2 > 0.999999
    assert result.prediction_mse < 1e-12
# EOF


def test_vector_field_procrustes_recovers_conjugacy():
    rng = np.random.default_rng(2)
    reference = np.array([
        [0.90, -0.25, 0.10],
        [0.20, 0.80, 0.05],
        [0.00, -0.10, 0.70],
    ])
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    comparison = orthogonal.T @ reference @ orthogonal

    result = procrustes_vector_fields(
        reference, comparison, n_iterations=500, n_restarts=5,
        learning_rate=0.02, random_seed=3,
    )
    assert result.angular_distance < 1e-4
    assert result.cosine_similarity > 0.999999
# EOF


def test_batched_vector_field_procrustes_recovers_conjugacy():
    rng = np.random.default_rng(4)
    reference = np.array([
        [0.90, -0.25, 0.10],
        [0.20, 0.80, 0.05],
        [0.00, -0.10, 0.70],
    ])
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    comparison = orthogonal.T @ reference @ orthogonal
    distances = batched_pavf_angular_distances(
        np.stack([reference, reference]),
        np.stack([reference, comparison]),
        n_iterations=300, learning_rate=0.02, batch_size=2,
    )

    assert distances.shape == (2,)
    assert np.all(distances < 1e-4)
# EOF
