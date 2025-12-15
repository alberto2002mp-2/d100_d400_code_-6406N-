"""Tests for the QuantileClipper transformer."""

import numpy as np
import pytest

from package.feature_engineering.quantile_clipper import QuantileClipper


@pytest.mark.parametrize(
    "X, lower_q, upper_q",
    [
        # Simple increasing values; expect clipping to 25th and 75th percentiles.
        (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 0.25, 0.75),
        # Mix of negatives, zeros, and positives with wider quantile range.
        (np.array([[-10.0, 0.0], [0.0, 10.0], [10.0, -5.0]]), 0.1, 0.9),
        # Constant columns should remain unchanged after clipping.
        (np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]), 0.2, 0.8),
        # Very small array (single sample) should pass through unchanged.
        (np.array([[42.0, -3.0]]), 0.05, 0.95),
    ],
)
def test_quantile_clipper_transformations(X, lower_q, upper_q):
    clipper = QuantileClipper(lower_quantile=lower_q, upper_quantile=upper_q)

    # Preserve copy to ensure input is not mutated.
    X_original = X.copy()

    transformed = clipper.fit_transform(X)
    lower_bounds = np.quantile(X, lower_q, axis=0)
    upper_bounds = np.quantile(X, upper_q, axis=0)
    expected = np.clip(X, lower_bounds, upper_bounds)

    # Shape is preserved.
    assert transformed.shape == X.shape
    # No mutation of input data.
    np.testing.assert_allclose(X, X_original)
    # Values are clipped as expected.
    np.testing.assert_allclose(transformed, expected)
