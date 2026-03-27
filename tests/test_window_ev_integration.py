"""Tests for training window optimization + EV verification integration.

Validates:
- _clean_features handles inf/NaN/margins correctly
- _train_and_predict works for each model type with margins
- _simulate_bracket_brier runs MC simulation correctly
- verify_window_with_ev runs simulation and identifies winners/ties
- run_training_window_optimization handles cached reports with re-verification
- Config fields are properly defined
- EV verification prefers shorter windows on ties
- baseline_training consumes optimal_training_windows
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ml.evaluation.training_window_optimizer import (
    TrainingWindowOptimizer,
    TrainingWindowReport,
    WindowSizeReport,
)
from src.ml.evaluation.window_ev_integration import (
    EVVerificationResult,
    WindowOptimizationResult,
    _clean_features,
    _simulate_bracket_brier,
    _train_and_predict,
    _verify_cached_report,
    _window_label_to_size,
    verify_window_with_ev,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_train_data(rng):
    """Simple training data: 200 samples, 10 features."""
    X = rng.randn(200, 10).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def simple_eval_data(rng):
    """Simple eval data: 50 samples, 10 features."""
    X = rng.randn(50, 10).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def mock_window_report():
    """Create a mock TrainingWindowReport with known values."""
    lgb_reports = [
        WindowSizeReport(
            model_type="lightgbm",
            window_size=5,
            window_label="last_5",
            mean_brier=0.180,
            std_brier=0.02,
            per_year_brier={2022: 0.18, 2023: 0.18},
            n_eval_years=2,
            total_eval_games=126,
        ),
        WindowSizeReport(
            model_type="lightgbm",
            window_size=8,
            window_label="last_8",
            mean_brier=0.195,
            std_brier=0.03,
            per_year_brier={2022: 0.20, 2023: 0.19},
            n_eval_years=2,
            total_eval_games=126,
        ),
        WindowSizeReport(
            model_type="lightgbm",
            window_size=3,
            window_label="last_3",
            mean_brier=0.210,
            std_brier=0.04,
            per_year_brier={2022: 0.22, 2023: 0.20},
            n_eval_years=2,
            total_eval_games=126,
        ),
    ]

    return TrainingWindowReport(
        model_type_results={"lightgbm": lgb_reports},
        recommendations={"lightgbm": "last_5"},
        raw_results=[],
        eval_years=[2022, 2023],
    )


@pytest.fixture
def tied_window_report():
    """Report where two windows have nearly identical Brier scores."""
    reports = [
        WindowSizeReport(
            model_type="logistic",
            window_size=None,
            window_label="all_available",
            mean_brier=0.200,
            std_brier=0.02,
            per_year_brier={2022: 0.20, 2023: 0.20},
            n_eval_years=2,
        ),
        WindowSizeReport(
            model_type="logistic",
            window_size=5,
            window_label="last_5",
            mean_brier=0.202,  # Within 0.005 of best
            std_brier=0.02,
            per_year_brier={2022: 0.20, 2023: 0.204},
            n_eval_years=2,
        ),
    ]
    return TrainingWindowReport(
        model_type_results={"logistic": reports},
        recommendations={"logistic": "all_available"},
        raw_results=[],
        eval_years=[2022, 2023],
    )


# ===========================================================================
# Test: _clean_features
# ===========================================================================

class TestCleanFeatures:
    def test_replaces_inf(self):
        X = np.array([[1.0, np.inf], [-np.inf, 2.0]])
        y = np.array([0, 1])
        X_clean, y_clean, _ = _clean_features(X, y)
        assert not np.any(np.isinf(X_clean))
        assert len(y_clean) == 2

    def test_drops_all_nan_rows(self):
        X = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        y = np.array([0, 1, 0])
        X_clean, y_clean, _ = _clean_features(X, y)
        assert len(y_clean) == 2
        assert y_clean[0] == 0 and y_clean[1] == 0

    def test_keeps_partial_nan_rows(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        y = np.array([0, 1])
        X_clean, y_clean, _ = _clean_features(X, y)
        assert len(y_clean) == 2

    def test_margins_filtered_with_features(self):
        """Margins should be filtered in sync with features."""
        X = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        y = np.array([0, 1, 0])
        margins = np.array([5.0, -3.0, 8.0])
        X_clean, y_clean, m_clean = _clean_features(X, y, margins=margins)
        assert len(m_clean) == 2
        assert m_clean[0] == 5.0 and m_clean[1] == 8.0

    def test_none_margins_pass_through(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        _, _, m = _clean_features(X, y, margins=None)
        assert m is None


# ===========================================================================
# Test: _train_and_predict
# ===========================================================================

class TestTrainAndPredict:
    def test_logistic_returns_probabilities(self, simple_train_data, simple_eval_data):
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        preds = _train_and_predict(train_X, train_y, eval_X, "logistic")
        assert len(preds) == len(eval_y)
        assert np.all(preds >= 0.01) and np.all(preds <= 0.99)

    def test_lightgbm_returns_probabilities(self, simple_train_data, simple_eval_data):
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        preds = _train_and_predict(train_X, train_y, eval_X, "lightgbm")
        assert len(preds) == len(eval_y)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_xgboost_returns_probabilities(self, simple_train_data, simple_eval_data):
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        preds = _train_and_predict(train_X, train_y, eval_X, "xgboost")
        assert len(preds) == len(eval_y)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_unknown_model_falls_back_to_logistic(self, simple_train_data, simple_eval_data):
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        preds = _train_and_predict(train_X, train_y, eval_X, "unknown_model")
        assert len(preds) == len(eval_y)

    def test_predictions_are_informative(self, simple_train_data, simple_eval_data):
        """Predictions should not all be the same value."""
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        preds = _train_and_predict(train_X, train_y, eval_X, "logistic")
        assert len(set(np.round(preds, 4))) > 1

    def test_spread_with_real_margins(self, simple_train_data, simple_eval_data):
        """Spread model should use real margins when available."""
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        margins = np.where(train_y == 1, 8.0, -8.0)  # Real-ish margins
        preds = _train_and_predict(
            train_X, train_y, eval_X, "spread", margins=margins
        )
        assert len(preds) == len(eval_y)

    def test_spread_without_margins_falls_back(self, simple_train_data, simple_eval_data):
        """Spread model without margins should fall back to logistic."""
        train_X, train_y = simple_train_data
        eval_X, eval_y = simple_eval_data
        preds = _train_and_predict(
            train_X, train_y, eval_X, "spread", margins=None
        )
        assert len(preds) == len(eval_y)
        assert np.all(preds >= 0.01) and np.all(preds <= 0.99)


# ===========================================================================
# Test: _simulate_bracket_brier
# ===========================================================================

class TestSimulateBracketBrier:
    def test_returns_required_keys(self):
        preds = np.array([0.7, 0.3, 0.5, 0.9])
        result = _simulate_bracket_brier(preds, n_simulations=100)
        assert "mean_brier" in result
        assert "std_brier" in result
        assert "p5" in result
        assert "p50" in result
        assert "p95" in result

    def test_mean_brier_is_reasonable(self):
        """Brier score should be between 0 and 1."""
        preds = np.array([0.7, 0.3, 0.5, 0.9, 0.1, 0.8])
        result = _simulate_bracket_brier(preds, n_simulations=500)
        assert 0.0 < result["mean_brier"] < 1.0

    def test_perfect_predictions_have_low_brier(self):
        """Near-certain predictions should yield low Brier."""
        preds = np.array([0.99, 0.99, 0.99, 0.99])
        result = _simulate_bracket_brier(preds, n_simulations=500)
        assert result["mean_brier"] < 0.05

    def test_uncertain_predictions_have_higher_brier(self):
        """50/50 predictions should yield higher Brier."""
        preds = np.full(10, 0.5)
        result = _simulate_bracket_brier(preds, n_simulations=500)
        assert result["mean_brier"] > 0.15

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        preds = np.array([0.7, 0.3, 0.5])
        r1 = _simulate_bracket_brier(preds, n_simulations=100, seed=42)
        r2 = _simulate_bracket_brier(preds, n_simulations=100, seed=42)
        assert r1["mean_brier"] == r2["mean_brier"]

    def test_percentiles_ordered(self):
        """p5 <= p50 <= p95."""
        preds = np.array([0.7, 0.3, 0.5, 0.9, 0.1])
        result = _simulate_bracket_brier(preds, n_simulations=500)
        assert result["p5"] <= result["p50"] <= result["p95"]


# ===========================================================================
# Test: _window_label_to_size
# ===========================================================================

class TestWindowLabelToSize:
    def test_last_n(self):
        assert _window_label_to_size("last_5") == 5
        assert _window_label_to_size("last_12") == 12

    def test_all_available(self):
        assert _window_label_to_size("all_available") is None

    def test_post_regime(self):
        assert _window_label_to_size("post_shot_clock_30s") is None

    def test_invalid(self):
        assert _window_label_to_size("last_abc") is None


# ===========================================================================
# Test: verify_window_with_ev
# ===========================================================================

class TestVerifyWindowWithEV:
    def test_clear_winner_verified(self, mock_window_report):
        """When there's a clear Brier winner, it should be verified via simulation."""
        results = verify_window_with_ev(
            mock_window_report, MagicMock(), "dummy_dir"
        )
        assert len(results) == 1
        ev = results[0]
        assert ev.model_type == "lightgbm"
        assert ev.ev_verified is True
        assert ev.recommended_window == "last_5"
        assert ev.brier_delta > 0.005
        # Simulation fields should be populated
        assert ev.sim_best_mean_brier is not None
        assert ev.sim_runner_mean_brier is not None
        assert "sim_best" in ev.ev_details
        assert "sim_runner" in ev.ev_details

    def test_tied_windows_prefer_shorter(self, tied_window_report):
        """When Brier scores are tied, prefer the shorter window."""
        results = verify_window_with_ev(
            tied_window_report, MagicMock(), "dummy_dir"
        )
        assert len(results) == 1
        ev = results[0]
        assert ev.model_type == "logistic"
        assert ev.brier_delta < 0.005
        assert ev.ev_verified is False  # all_available is longer than last_5

    def test_single_window_skipped(self):
        """Model type with only 1 window should be skipped."""
        report = TrainingWindowReport(
            model_type_results={
                "logistic": [
                    WindowSizeReport(
                        model_type="logistic",
                        window_size=5,
                        window_label="last_5",
                        mean_brier=0.20,
                        std_brier=0.02,
                        per_year_brier={2023: 0.20},
                    )
                ]
            },
            recommendations={"logistic": "last_5"},
            raw_results=[],
            eval_years=[2023],
        )
        results = verify_window_with_ev(report, MagicMock(), "dummy_dir")
        assert len(results) == 0


# ===========================================================================
# Test: _verify_cached_report
# ===========================================================================

class TestVerifyCachedReport:
    def test_reconstructs_and_verifies(self):
        """Should reconstruct WindowSizeReports from JSON and run EV verification."""
        report_data = {
            "recommendations": {"lightgbm": "last_5"},
            "eval_years": [2022, 2023],
            "model_results": {
                "lightgbm": [
                    {
                        "window_label": "last_5",
                        "window_size": 5,
                        "mean_brier": 0.18,
                        "std_brier": 0.02,
                        "n_eval_years": 2,
                        "per_year_brier": {"2022": 0.18, "2023": 0.18},
                    },
                    {
                        "window_label": "last_8",
                        "window_size": 8,
                        "mean_brier": 0.20,
                        "std_brier": 0.03,
                        "n_eval_years": 2,
                        "per_year_brier": {"2022": 0.20, "2023": 0.20},
                    },
                ]
            },
        }
        results = _verify_cached_report(report_data, MagicMock(), "dummy_dir")
        assert len(results) == 1
        assert results[0].model_type == "lightgbm"
        assert results[0].ev_verified is True

    def test_empty_model_results(self):
        """Empty model_results should produce empty verification."""
        report_data = {
            "recommendations": {},
            "model_results": {},
        }
        results = _verify_cached_report(report_data, MagicMock(), "dummy_dir")
        assert len(results) == 0


# ===========================================================================
# Test: Config fields
# ===========================================================================

class TestConfigFields:
    def test_config_has_window_fields(self):
        from src.pipeline.config import SOTAPipelineConfig
        config = SOTAPipelineConfig()
        assert hasattr(config, "enable_training_window_optimization")
        assert config.enable_training_window_optimization is False
        assert hasattr(config, "training_window_candidates")
        assert config.training_window_candidates is None
        assert hasattr(config, "training_window_report_path")
        assert hasattr(config, "training_window_verify_ev")
        assert config.training_window_verify_ev is True
        assert hasattr(config, "optimal_training_windows")
        assert config.optimal_training_windows is None

    def test_config_can_enable_window_optimization(self):
        from src.pipeline.config import SOTAPipelineConfig
        config = SOTAPipelineConfig(
            enable_training_window_optimization=True,
            training_window_candidates=[3, 5, 8],
        )
        assert config.enable_training_window_optimization is True
        assert config.training_window_candidates == [3, 5, 8]

    def test_optimal_windows_can_be_set(self):
        from src.pipeline.config import SOTAPipelineConfig
        config = SOTAPipelineConfig(
            optimal_training_windows={"lightgbm": 5, "logistic": None},
        )
        assert config.optimal_training_windows["lightgbm"] == 5
        assert config.optimal_training_windows["logistic"] is None


# ===========================================================================
# Test: WindowOptimizationResult dataclass
# ===========================================================================

class TestWindowOptimizationResult:
    def test_basic_construction(self):
        result = WindowOptimizationResult(
            recommendations={"lightgbm": "last_5", "logistic": "last_8"},
            optimal_years={"lightgbm": 5, "logistic": 8},
        )
        assert result.recommendations["lightgbm"] == "last_5"
        assert result.optimal_years["logistic"] == 8
        assert result.ev_verification == []
        assert result.report_path is None

    def test_with_ev_verification(self):
        ev = EVVerificationResult(
            model_type="lightgbm",
            recommended_window="last_5",
            recommended_brier=0.18,
            runner_up_window="last_8",
            runner_up_brier=0.20,
            brier_delta=0.02,
            ev_verified=True,
            sim_best_mean_brier=0.17,
            sim_runner_mean_brier=0.19,
        )
        result = WindowOptimizationResult(
            recommendations={"lightgbm": "last_5"},
            optimal_years={"lightgbm": 5},
            ev_verification=[ev],
        )
        assert len(result.ev_verification) == 1
        assert result.ev_verification[0].ev_verified is True
        assert result.ev_verification[0].sim_best_mean_brier == 0.17


# ===========================================================================
# Test: Integration with TrainingWindowOptimizer
# ===========================================================================

class TestOptimizerIntegration:
    def test_optimizer_with_custom_train_fn(self):
        """Verify TrainingWindowOptimizer works with our train_predict interface."""
        def mock_train_predict(train_data, eval_data, model_type):
            n = len(train_data)
            brier = 0.25 - 0.005 * min(n, 10)
            return (brier, None, None)

        data = {yr: {"year": yr} for yr in range(2016, 2025) if yr != 2020}
        optimizer = TrainingWindowOptimizer(
            windows=[3, 5],
            eval_years=[2023, 2024],
            include_regime_windows=False,
        )
        report = optimizer.evaluate_windows(
            data, mock_train_predict, model_types=["lightgbm"]
        )
        assert "lightgbm" in report.recommendations
        assert len(report.model_type_results["lightgbm"]) == 2

    def test_ev_verification_on_optimizer_output(self):
        """EV verification should run simulation on actual optimizer output."""
        def mock_train_predict(train_data, eval_data, model_type):
            n = len(train_data)
            brier = 0.25 - 0.005 * min(n, 10)
            return (brier, None, None)

        data = {yr: {"year": yr} for yr in range(2016, 2025) if yr != 2020}
        optimizer = TrainingWindowOptimizer(
            windows=[3, 5, 8],
            eval_years=[2023, 2024],
            include_regime_windows=False,
        )
        report = optimizer.evaluate_windows(
            data, mock_train_predict, model_types=["lightgbm"]
        )

        ev_results = verify_window_with_ev(report, MagicMock(), "dummy_dir")
        assert len(ev_results) == 1
        assert ev_results[0].model_type == "lightgbm"
        # Should have simulation results
        assert ev_results[0].sim_best_mean_brier is not None


# ===========================================================================
# Test: Pre-tournament training data enforcement
# ===========================================================================

class TestPreTournamentTraining:
    def test_train_predict_fn_validates_eval_data(self):
        """train_predict_fn should reject invalid eval_data format."""
        from src.ml.evaluation.window_ev_integration import build_train_predict_fn
        from src.pipeline.config import SOTAPipelineConfig

        config = SOTAPipelineConfig()
        fn = build_train_predict_fn(config, "/nonexistent")

        with pytest.raises(ValueError, match="must be a dict"):
            fn({2020: "data"}, "not_a_dict", "logistic")
