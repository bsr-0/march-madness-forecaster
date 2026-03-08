"""Tests for pipeline schema contracts (S19-1)."""

import numpy as np
import pytest

from src.data.schemas import (
    validate_feature_matrix,
    validate_loyo_fold,
    validate_predictions,
)


class TestValidateFeatureMatrix:
    """Test feature matrix validation."""

    def test_valid_matrix_passes(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        result = validate_feature_matrix(X)
        assert result.passed
        assert len(result.errors) == 0

    def test_1d_array_fails(self):
        result = validate_feature_matrix(np.array([1, 2, 3]))
        assert not result.passed
        assert any("2D" in e for e in result.errors)

    def test_too_few_samples_fails(self):
        X = np.array([[1, 2], [3, 4]])
        result = validate_feature_matrix(X, min_samples=10)
        assert not result.passed
        assert any("Too few samples" in e for e in result.errors)

    def test_high_nan_warns(self):
        X = np.full((100, 3), np.nan)
        X[:10, :] = 1.0  # Only 10% non-NaN
        result = validate_feature_matrix(X, max_nan_fraction=0.5)
        assert len(result.warnings) > 0

    def test_infinite_values_fail(self):
        X = np.ones((50, 3))
        X[0, 0] = np.inf
        result = validate_feature_matrix(X)
        assert not result.passed
        assert any("infinite" in e for e in result.errors)

    def test_constant_columns_warn(self):
        X = np.ones((50, 3))
        X[:, 0] = np.arange(50)  # Only column 0 varies
        result = validate_feature_matrix(X)
        assert len(result.warnings) > 0
        assert any("constant" in w for w in result.warnings)

    def test_feature_names_in_warnings(self):
        X = np.full((100, 2), np.nan)
        result = validate_feature_matrix(
            X, feature_names=["feat_a", "feat_b"], max_nan_fraction=0.5,
        )
        assert any("feat_a" in str(w) for w in result.warnings)

    def test_serialization(self):
        X = np.ones((50, 3))
        result = validate_feature_matrix(X)
        d = result.to_dict()
        assert "passed" in d
        assert "warnings" in d
        assert "errors" in d


class TestValidatePredictions:
    """Test prediction array validation."""

    def test_valid_predictions_pass(self):
        preds = np.array([0.1, 0.5, 0.9, 0.3])
        result = validate_predictions(preds)
        assert result.passed

    def test_nan_predictions_fail(self):
        preds = np.array([0.5, np.nan, 0.3])
        result = validate_predictions(preds)
        assert not result.passed
        assert any("NaN" in e for e in result.errors)

    def test_out_of_range_fails(self):
        preds = np.array([0.5, 1.5, -0.1])
        result = validate_predictions(preds)
        assert not result.passed
        assert any("out of" in e for e in result.errors)

    def test_zero_variance_warns(self):
        preds = np.full(100, 0.5)
        result = validate_predictions(preds)
        assert result.passed  # Passes but warns
        assert len(result.warnings) > 0
        assert any("variance" in w for w in result.warnings)

    def test_infinite_predictions_fail(self):
        preds = np.array([0.5, np.inf])
        result = validate_predictions(preds)
        assert not result.passed

    def test_2d_array_fails(self):
        preds = np.array([[0.5, 0.3], [0.7, 0.2]])
        result = validate_predictions(preds)
        assert not result.passed


class TestValidateLoyoFold:
    """Test LOYO fold validation."""

    def test_valid_fold_passes(self):
        preds = np.array([0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2, 0.9, 0.1, 0.55])
        actuals = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        result = validate_loyo_fold(2023, 0.18, 10, preds, actuals)
        assert result.passed

    def test_few_games_warns(self):
        preds = np.array([0.5, 0.5, 0.5])
        actuals = np.array([1, 0, 1])
        result = validate_loyo_fold(2023, 0.20, 3, preds, actuals)
        assert any("few" in w for w in result.warnings)

    def test_high_brier_warns(self):
        preds = np.array([0.5] * 20)
        actuals = np.array([1] * 20)
        result = validate_loyo_fold(2023, 0.40, 20, preds, actuals)
        assert any("high" in w.lower() for w in result.warnings)

    def test_length_mismatch_fails(self):
        preds = np.array([0.5, 0.5])
        actuals = np.array([1])
        result = validate_loyo_fold(2023, 0.20, 2, preds, actuals)
        assert not result.passed
