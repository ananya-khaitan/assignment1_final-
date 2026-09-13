from __future__ import annotations

import numpy as np

from code.models.adaptive_routing import _convex_weights, development_and_test_slices


def test_adaptive_split_locks_final_twenty_percent():
    subtrain, development, test = development_and_test_slices(500)
    assert (subtrain.start, subtrain.stop) == (0, 320)
    assert (development.start, development.stop) == (320, 400)
    assert (test.start, test.stop) == (400, 500)


def test_stacking_weights_are_nonnegative_and_sum_to_one():
    actual = np.arange(20, dtype=float)
    predictions = np.vstack([actual, actual + 2.0, actual - 3.0])
    weights = _convex_weights(actual, predictions)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights @ predictions, actual, atol=1e-6)


def test_stacking_is_scale_invariant_for_large_price_levels():
    actual = 100_000.0 + np.arange(30, dtype=float) * 500.0
    predictions = np.vstack([actual, actual + 10_000.0, actual - 15_000.0])
    weights = _convex_weights(actual, predictions)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    assert np.sqrt(np.mean((weights @ predictions - actual) ** 2)) < 0.01
