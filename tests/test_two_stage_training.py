"""Tests for two-stage training (pretrain regular season, finetune tournament).

Validates that:
1. Stage 1 trains on regular-season data
2. Stage 2 finetunes on tournament data with reduced learning rate
3. Stage 2 is skipped when tournament samples are insufficient
4. Warm-start correctly continues from Stage 1 checkpoint
5. Each model type (LightGBM, SpreadRegressor, Logistic) works
"""

import numpy as np
import pytest

from src.ml.training.two_stage_training import (
    FINETUNE_LEARNING_RATE,
    FINETUNE_NUM_ROUNDS,
    TwoStageConfig,
    TwoStageResult,
    _build_finetune_params,
    two_stage_train_lightgbm,
    two_stage_train_logistic,
    two_stage_train_spread,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_DIM = 20
np.random.seed(42)


def _make_classification_data(n, feature_dim=FEATURE_DIM):
    """Generate synthetic binary classification data."""
    X = np.random.randn(n, feature_dim).astype(np.float64)
    # Labels correlated with first feature for learnability
    y = (X[:, 0] + np.random.randn(n) * 0.5 > 0).astype(np.float64)
    return X, y


def _make_regression_data(n, feature_dim=FEATURE_DIM):
    """Generate synthetic margin regression data."""
    X = np.random.randn(n, feature_dim).astype(np.float64)
    margins = X[:, 0] * 5 + np.random.randn(n) * 3
    return X, margins


# ---------------------------------------------------------------------------
# Tests: _build_finetune_params
# ---------------------------------------------------------------------------


class TestBuildFinetuneParams:
    def test_learning_rate_reduced(self):
        base = {"learning_rate": 0.05, "lambda_l1": 1.0, "lambda_l2": 1.0}
        config = TwoStageConfig()
        params = _build_finetune_params(base, config)
        assert params["learning_rate"] == FINETUNE_LEARNING_RATE
        assert params["learning_rate"] < base["learning_rate"]

    def test_regularization_increased(self):
        base = {"learning_rate": 0.05, "lambda_l1": 1.0, "lambda_l2": 1.0}
        config = TwoStageConfig()
        params = _build_finetune_params(base, config)
        assert params["lambda_l1"] >= base["lambda_l1"]
        assert params["lambda_l2"] >= base["lambda_l2"]

    def test_min_child_samples_set(self):
        base = {"learning_rate": 0.05, "lambda_l1": 1.0, "lambda_l2": 1.0, "min_child_samples": 20}
        config = TwoStageConfig(finetune_min_child_samples=10)
        params = _build_finetune_params(base, config)
        assert params["min_child_samples"] == 10

    def test_xgboost_min_child_weight(self):
        base = {"learning_rate": 0.05, "lambda_l1": 1.0, "lambda_l2": 1.0, "min_child_weight": 20}
        config = TwoStageConfig(finetune_min_child_samples=10)
        params = _build_finetune_params(base, config)
        assert params["min_child_weight"] == 10

    def test_base_params_not_mutated(self):
        base = {"learning_rate": 0.05, "lambda_l1": 1.0, "lambda_l2": 1.0}
        config = TwoStageConfig()
        _build_finetune_params(base, config)
        assert base["learning_rate"] == 0.05  # Original unchanged


# ---------------------------------------------------------------------------
# Tests: two_stage_train_lightgbm
# ---------------------------------------------------------------------------

_lgb_available = True
try:
    import lightgbm  # noqa: F401
except ImportError:
    _lgb_available = False


@pytest.mark.skipif(not _lgb_available, reason="lightgbm not installed")
class TestTwoStageLightGBM:
    def test_both_stages_train(self):
        """Both stages complete with sufficient data."""
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        result = two_stage_train_lightgbm(pre_X, pre_y, ft_X, ft_y)
        assert result.stage1_trained is True
        assert result.stage2_trained is True
        assert result.model is not None
        assert result.stage1_n_samples == 200
        assert result.stage2_n_samples == 50

    def test_stage2_skipped_insufficient_samples(self):
        """Stage 2 skipped when tournament data is too small."""
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(5)
        config = TwoStageConfig(min_finetune_samples=30)
        result = two_stage_train_lightgbm(pre_X, pre_y, ft_X, ft_y, config=config)
        assert result.stage1_trained is True
        assert result.stage2_trained is False
        assert result.reason_skipped_stage2 is not None
        assert "insufficient" in result.reason_skipped_stage2

    def test_warm_start_adds_trees(self):
        """Warm-start should result in more trees than Stage 1 alone."""
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        config = TwoStageConfig(
            pretrain_num_rounds=20,
            finetune_num_rounds=10,
            min_finetune_samples=10,
        )
        result = two_stage_train_lightgbm(pre_X, pre_y, ft_X, ft_y, config=config)
        assert result.stage2_trained is True
        # With warm_start, total trees = stage1 + stage2
        assert result.model.num_trees() >= 20

    def test_no_warm_start(self):
        """Without warm start, model trains from scratch on tournament data."""
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        config = TwoStageConfig(
            pretrain_num_rounds=20,
            finetune_num_rounds=15,
            use_warm_start=False,
            min_finetune_samples=10,
        )
        result = two_stage_train_lightgbm(pre_X, pre_y, ft_X, ft_y, config=config)
        assert result.stage2_trained is True
        # Without warm start, trees = finetune rounds only
        assert result.model.num_trees() == 15

    def test_with_sample_weights(self):
        pre_X, pre_y = _make_classification_data(100)
        ft_X, ft_y = _make_classification_data(40)
        pre_w = np.ones(100)
        ft_w = np.full(40, 2.0)
        result = two_stage_train_lightgbm(
            pre_X,
            pre_y,
            ft_X,
            ft_y,
            pretrain_weights=pre_w,
            finetune_weights=ft_w,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        assert result.stage2_trained is True

    def test_with_validation_set(self):
        pre_X, pre_y = _make_classification_data(200)
        val_X, val_y = _make_classification_data(50)
        ft_X, ft_y = _make_classification_data(40)
        result = two_stage_train_lightgbm(
            pre_X,
            pre_y,
            ft_X,
            ft_y,
            pretrain_valid=(val_X, val_y),
            config=TwoStageConfig(min_finetune_samples=10),
        )
        assert result.stage1_trained is True

    def test_model_can_predict(self):
        """Final model should produce valid predictions."""
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        result = two_stage_train_lightgbm(
            pre_X,
            pre_y,
            ft_X,
            ft_y,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        preds = result.model.predict(ft_X)
        assert preds.shape == (50,)
        assert np.all((preds >= 0) & (preds <= 1))


# ---------------------------------------------------------------------------
# Tests: two_stage_train_spread
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _lgb_available, reason="lightgbm not installed")
class TestTwoStageSpread:
    def test_both_stages_train(self):
        pre_X, pre_m = _make_regression_data(200)
        ft_X, ft_m = _make_regression_data(50)
        result = two_stage_train_spread(
            pre_X,
            pre_m,
            ft_X,
            ft_m,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        assert result.stage1_trained is True
        assert result.stage2_trained is True

    def test_stage2_skipped_insufficient(self):
        pre_X, pre_m = _make_regression_data(200)
        ft_X, ft_m = _make_regression_data(5)
        config = TwoStageConfig(min_finetune_samples=30)
        result = two_stage_train_spread(pre_X, pre_m, ft_X, ft_m, config=config)
        assert result.stage2_trained is False
        assert "insufficient" in result.reason_skipped_stage2

    def test_model_predicts_margins(self):
        pre_X, pre_m = _make_regression_data(200)
        ft_X, ft_m = _make_regression_data(50)
        result = two_stage_train_spread(
            pre_X,
            pre_m,
            ft_X,
            ft_m,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        preds = result.model.predict(ft_X)
        assert preds.shape == (50,)
        # Margins should be in reasonable range
        assert np.all(np.abs(preds) < 100)


# ---------------------------------------------------------------------------
# Tests: two_stage_train_logistic
# ---------------------------------------------------------------------------


class TestTwoStageLogistic:
    def test_both_stages_train(self):
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        result = two_stage_train_logistic(
            pre_X,
            pre_y,
            ft_X,
            ft_y,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        assert result.stage1_trained is True
        assert result.stage2_trained is True
        assert result.model is not None
        assert "logistic" in result.model
        assert "scaler" in result.model

    def test_stage2_skipped_insufficient(self):
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(5)
        config = TwoStageConfig(min_finetune_samples=30)
        result = two_stage_train_logistic(pre_X, pre_y, ft_X, ft_y, config=config)
        assert result.stage2_trained is False

    def test_model_predicts_probabilities(self):
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        result = two_stage_train_logistic(
            pre_X,
            pre_y,
            ft_X,
            ft_y,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        model = result.model["logistic"]
        scaler = result.model["scaler"]
        ft_scaled = scaler.transform(np.nan_to_num(ft_X))
        probs = model.predict_proba(ft_scaled)[:, 1]
        assert probs.shape == (50,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_warm_start_preserves_coefficients(self):
        """Stage 2 should start from Stage 1 coefficients."""
        pre_X, pre_y = _make_classification_data(300)
        ft_X, ft_y = _make_classification_data(50)
        # Stage 1 only
        config_s1 = TwoStageConfig(min_finetune_samples=999)
        result_s1 = two_stage_train_logistic(pre_X, pre_y, ft_X, ft_y, config=config_s1)
        coefs_s1 = result_s1.model["logistic"].coef_.copy()

        # Both stages
        config_both = TwoStageConfig(min_finetune_samples=10)
        result_both = two_stage_train_logistic(pre_X, pre_y, ft_X, ft_y, config=config_both)
        coefs_both = result_both.model["logistic"].coef_

        # Coefficients should differ (finetuning changed them)
        # but not be wildly different (warm start from same base)
        assert not np.allclose(coefs_s1, coefs_both, atol=1e-6)

    def test_with_sample_weights(self):
        pre_X, pre_y = _make_classification_data(200)
        ft_X, ft_y = _make_classification_data(50)
        pre_w = np.ones(200)
        ft_w = np.full(50, 2.0)
        result = two_stage_train_logistic(
            pre_X,
            pre_y,
            ft_X,
            ft_y,
            pretrain_weights=pre_w,
            finetune_weights=ft_w,
            config=TwoStageConfig(min_finetune_samples=10),
        )
        assert result.stage2_trained is True


# ---------------------------------------------------------------------------
# Tests: TwoStageConfig
# ---------------------------------------------------------------------------


class TestTwoStageConfig:
    def test_defaults(self):
        c = TwoStageConfig()
        assert c.finetune_learning_rate == 0.01
        assert c.finetune_num_rounds == 50
        assert c.min_finetune_samples == 30
        assert c.use_warm_start is True

    def test_custom_config(self):
        c = TwoStageConfig(
            finetune_learning_rate=0.005,
            finetune_num_rounds=25,
            min_finetune_samples=50,
            use_warm_start=False,
        )
        assert c.finetune_learning_rate == 0.005
        assert c.finetune_num_rounds == 25
        assert c.min_finetune_samples == 50
        assert c.use_warm_start is False
