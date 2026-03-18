"""Tests for custom Brier-based LightGBM training objective."""

import numpy as np
import pytest

from src.ml.ensemble.brier_objective import brier_objective, brier_eval


class MockTrainData:
    """Mock LightGBM Dataset for testing."""

    def __init__(self, labels):
        self._labels = np.array(labels, dtype=np.float64)

    def get_label(self):
        return self._labels


class TestBrierObjective:
    def test_gradient_shape(self):
        """Gradient and hessian should match prediction shape."""
        preds = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
        data = MockTrainData([1, 0, 1, 0, 1])
        grad, hess = brier_objective(preds, data)
        assert grad.shape == preds.shape
        assert hess.shape == preds.shape

    def test_hessian_positive(self):
        """Hessian should always be positive (clamped)."""
        rng = np.random.default_rng(42)
        preds = rng.normal(0, 2, size=1000)
        labels = rng.integers(0, 2, size=1000).astype(float)
        data = MockTrainData(labels)
        _, hess = brier_objective(preds, data)
        assert np.all(hess > 0), "Hessian should be positive everywhere"

    def test_gradient_finite_difference(self):
        """Verify gradient by finite difference approximation."""
        rng = np.random.default_rng(123)
        preds = rng.normal(0, 1, size=50)
        labels = rng.integers(0, 2, size=50).astype(float)
        data = MockTrainData(labels)

        grad, _ = brier_objective(preds, data)

        # Finite difference
        eps = 1e-5
        for i in range(min(10, len(preds))):
            preds_plus = preds.copy()
            preds_minus = preds.copy()
            preds_plus[i] += eps
            preds_minus[i] -= eps

            s_plus = 1.0 / (1.0 + np.exp(-preds_plus))
            s_minus = 1.0 / (1.0 + np.exp(-preds_minus))
            brier_plus = np.mean((s_plus - labels) ** 2)
            brier_minus = np.mean((s_minus - labels) ** 2)

            fd_grad = (brier_plus - brier_minus) / (2 * eps)

            # grad[i] is per-sample, fd_grad is averaged
            # Compare the per-sample contribution
            assert abs(grad[i] / len(preds) - fd_grad) < 1e-4, (
                f"Gradient mismatch at index {i}: analytic={grad[i]/len(preds):.6f}, "
                f"finite_diff={fd_grad:.6f}"
            )

    def test_zero_gradient_at_perfect_prediction(self):
        """When sigmoid(f) == y, gradient should be zero."""
        # f=large positive → sigmoid≈1, y=1 → gradient≈0
        preds = np.array([10.0])
        data = MockTrainData([1.0])
        grad, _ = brier_objective(preds, data)
        assert abs(grad[0]) < 1e-4

    def test_gradient_direction(self):
        """Gradient should push predictions toward the true labels."""
        # pred=0 (sigmoid=0.5), label=1 → gradient should be negative
        # (to increase f, increasing sigmoid toward 1)
        preds = np.array([0.0])
        data = MockTrainData([1.0])
        grad, _ = brier_objective(preds, data)
        assert grad[0] < 0, "Gradient should be negative when pred < label"

        # pred=0 (sigmoid=0.5), label=0 → gradient should be positive
        data = MockTrainData([0.0])
        grad, _ = brier_objective(preds, data)
        assert grad[0] > 0, "Gradient should be positive when pred > label"


class TestBrierEval:
    def test_perfect_predictions(self):
        """Perfect predictions should give Brier = 0."""
        preds = np.array([100.0, -100.0])  # sigmoid ≈ 1, ≈ 0
        data = MockTrainData([1.0, 0.0])
        name, value, higher_better = brier_eval(preds, data)
        assert name == "brier"
        assert value < 0.01
        assert higher_better is False

    def test_worst_predictions(self):
        """Worst predictions should give Brier close to 1."""
        preds = np.array([-100.0, 100.0])  # sigmoid ≈ 0, ≈ 1
        data = MockTrainData([1.0, 0.0])
        name, value, _ = brier_eval(preds, data)
        assert value > 0.9

    def test_random_predictions(self):
        """Random predictions should give Brier around 0.25."""
        preds = np.zeros(1000)  # sigmoid = 0.5 for all
        data = MockTrainData(np.random.default_rng(42).integers(0, 2, size=1000).astype(float))
        _, value, _ = brier_eval(preds, data)
        assert 0.20 < value < 0.30
