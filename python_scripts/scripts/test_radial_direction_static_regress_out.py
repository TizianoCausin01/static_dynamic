import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from project_specific_utils.static_response_regress_out import (
    centered_norms_and_directions, radial_direction_static_regress_out,
)


def test_centered_norms_and_directions_reconstruct_centered_patterns():
    rng = np.random.default_rng(0)
    rasters = rng.normal(size=(5, 4, 20))

    norms, directions = centered_norms_and_directions(rasters)
    reconstructed = directions * norms[np.newaxis, :, :]
    expected = rasters - rasters.mean(axis=2, keepdims=True)

    np.testing.assert_allclose(reconstructed, expected)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=0), 1)
# EOF


def test_radial_direction_regression_returns_unit_direction_residuals():
    rng = np.random.default_rng(1)
    predictors = rng.normal(size=(2, 5, 4, 30))
    target = (
        0.4 * predictors[0]
        - 0.2 * predictors[1]
        + rng.normal(size=(5, 4, 30))
    )

    result = radial_direction_static_regress_out(
        predictors, target, regression_type="lr", cv_type="same",
    )

    assert result["direction_residuals"].shape == target.shape
    assert result["log_norm_residuals"].shape == target.shape[1:]
    np.testing.assert_allclose(
        np.linalg.norm(result["direction_residuals"], axis=0), 1,
    )
# EOF
