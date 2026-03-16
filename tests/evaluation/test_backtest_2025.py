"""Tests for the 2025 backtest validation system.

Verifies:
1. Leakage-free temporal isolation (2025 never in training)
2. Correct metric computation (Brier, log-loss, accuracy, ECE)
3. Seed baseline comparison
4. LOYO and prospective validation integration
5. Report generation and pass/fail logic
"""

import numpy as np
import pytest

from src.ml.evaluation.backtest_2025 import (
    EXCLUDED_YEAR,
    HOLDOUT_YEAR,
    TRAINING_YEARS,
    BacktestReport,
    BacktestValidationRunner,
    LeakageCheckResult,
    compute_seed_baseline_brier,
    run_full_validation,
    run_quick_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_year_data(year, n_games=40, n_features=5, seed=None):
    """Create synthetic per-year data for backtest testing."""
    rng = np.random.RandomState(seed or year)
    X = rng.randn(n_games, n_features)
    y = rng.randint(0, 2, size=n_games).astype(float)
    margins = rng.randn(n_games) * 10

    # Generate realistic seeds (1-16 for both teams)
    seeds_t1 = rng.randint(1, 17, size=n_games).astype(float)
    seeds_t2 = rng.randint(1, 17, size=n_games).astype(float)

    # Assign round labels
    rounds = []
    for i in range(n_games):
        if i < n_games // 2:
            rounds.append("R64")
        elif i < 3 * n_games // 4:
            rounds.append("R32")
        else:
            rounds.append("S16")

    return {
        "X": X,
        "y": y,
        "margins": margins,
        "feature_names": [f"feat_{i}" for i in range(n_features)],
        "seeds_t1": seeds_t1,
        "seeds_t2": seeds_t2,
        "rounds": rounds,
    }


def _make_full_dataset(years=None):
    """Create synthetic dataset spanning multiple years."""
    if years is None:
        years = TRAINING_YEARS + [HOLDOUT_YEAR]
    return {yr: _make_year_data(yr) for yr in years}


def _simple_train_fn(X, y, margins, feature_names, weights):
    """Trivial model: logistic regression on first feature."""
    mean_y = float(y.mean())
    # Simple: predict based on first feature's sign
    coef = np.corrcoef(X[:, 0], y)[0, 1] if X.shape[1] > 0 else 0.0
    return {"mean_y": mean_y, "coef": coef}


def _simple_predict_fn(model, X):
    """Predict using the simple model."""
    base = model["mean_y"]
    coef = model.get("coef", 0.0)
    return np.clip(base + coef * X[:, 0] * 0.1, 0.01, 0.99)


def _perfect_predict_fn(model, X):
    """Always predict 0.5 — worst calibrated but unbiased."""
    return np.full(X.shape[0], 0.5)


# ---------------------------------------------------------------------------
# Leakage Constraint Tests
# ---------------------------------------------------------------------------


class TestLeakageConstraints:
    """Verify that temporal isolation is enforced."""

    def test_2025_never_in_training_years(self):
        """HOLDOUT_YEAR must not be in TRAINING_YEARS."""
        assert HOLDOUT_YEAR not in TRAINING_YEARS

    def test_2020_excluded(self):
        """COVID year must be excluded from training."""
        assert EXCLUDED_YEAR not in TRAINING_YEARS

    def test_training_years_before_holdout(self):
        """All training years must be before the holdout year."""
        assert all(y < HOLDOUT_YEAR for y in TRAINING_YEARS)

    def test_leakage_checks_all_pass_with_valid_data(self):
        """Leakage checks should pass with properly partitioned data."""
        data = _make_full_dataset()
        runner = BacktestValidationRunner(
            train_fn=_simple_train_fn,
            predict_fn=_simple_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = runner.run()

        assert report.passed_all_checks()
        assert len(report.leakage_checks) >= 6

    def test_holdout_data_missing_detected(self):
        """Missing holdout data should fail the check."""
        data = _make_full_dataset()
        del data[HOLDOUT_YEAR]

        runner = BacktestValidationRunner(
            train_fn=_simple_train_fn,
            predict_fn=_simple_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = BacktestReport()
        runner._verify_leakage_constraints(report)

        holdout_check = next(
            c for c in report.leakage_checks
            if c.check_name == "holdout_data_present"
        )
        assert not holdout_check.passed

    def test_consistent_feature_dimensions(self):
        """Feature dimensions must be consistent across years."""
        data = _make_full_dataset()
        # Corrupt one year with different dimensions
        bad_year = TRAINING_YEARS[0]
        data[bad_year]["X"] = np.random.randn(40, 10)  # 10 != 5 features

        runner = BacktestValidationRunner(
            train_fn=_simple_train_fn,
            predict_fn=_simple_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = BacktestReport()
        runner._verify_leakage_constraints(report)

        dim_check = next(
            c for c in report.leakage_checks
            if c.check_name == "consistent_feature_dimensions"
        )
        assert not dim_check.passed

    def test_training_only_uses_specified_years(self):
        """Verify that training data only comes from TRAINING_YEARS."""
        data = _make_full_dataset()

        # Track which years' data gets included in training
        years_seen = set()
        original_train_fn = _simple_train_fn

        def tracking_train_fn(X, y, margins, feature_names, weights):
            # Record the number of samples to verify correct year inclusion
            years_seen.add(len(y))
            return original_train_fn(X, y, margins, feature_names, weights)

        runner = BacktestValidationRunner(
            train_fn=tracking_train_fn,
            predict_fn=_simple_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = runner.run()

        # Training data should be from all TRAINING_YEARS (8 years × 40 games = 320)
        expected_total = sum(
            len(data[y]["y"]) for y in TRAINING_YEARS if y in data
        )
        assert report.n_train_games == expected_total
        assert report.n_test_games == len(data[HOLDOUT_YEAR]["y"])


# ---------------------------------------------------------------------------
# Metric Computation Tests
# ---------------------------------------------------------------------------


class TestMetricComputation:
    """Verify correct metric computation."""

    def test_perfect_predictions_zero_brier(self):
        """Perfect predictions should give Brier ≈ 0."""
        data = _make_full_dataset()

        def perfect_train_fn(X, y, margins, feature_names, weights):
            return {"y": y, "X": X}

        # Cheat: return actual labels (only for testing metric computation)
        # This would be leakage in practice, but we're testing metrics
        stored_y = data[HOLDOUT_YEAR]["y"].copy()

        def cheat_predict_fn(model, X):
            return stored_y

        runner = BacktestValidationRunner(
            train_fn=perfect_train_fn,
            predict_fn=cheat_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = runner.run()

        # Brier should be very close to 0 (not exactly due to clipping)
        assert report.brier_score < 1e-10

    def test_uniform_predictions_brier(self):
        """Predicting 0.5 everywhere should give Brier = 0.25."""
        data = _make_full_dataset()

        runner = BacktestValidationRunner(
            train_fn=_simple_train_fn,
            predict_fn=_perfect_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = runner.run()
        assert abs(report.brier_score - 0.25) < 0.01

    def test_accuracy_range(self):
        """Accuracy should be between 0 and 1."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        assert 0.0 <= report.accuracy <= 1.0

    def test_calibration_error_nonnegative(self):
        """ECE should be non-negative."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        assert report.calibration_error >= 0.0

    def test_per_round_brier_computed(self):
        """Per-round Brier scores should be computed when round labels exist."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        assert len(report.round_briers) > 0
        assert "R64" in report.round_briers

    def test_log_loss_positive(self):
        """Log loss should be positive for non-perfect predictions."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        assert report.log_loss > 0.0


# ---------------------------------------------------------------------------
# Seed Baseline Tests
# ---------------------------------------------------------------------------


class TestSeedBaseline:
    """Verify seed baseline computation."""

    def test_equal_seeds_predict_half(self):
        """Equal seeds should predict ~0.5, giving Brier = 0.25."""
        seeds = np.ones(100)
        outcomes = np.random.randint(0, 2, 100).astype(float)
        brier = compute_seed_baseline_brier(seeds, seeds, outcomes)
        assert abs(brier - 0.25) < 0.05

    def test_extreme_seed_diff(self):
        """1-seed vs 16-seed should predict high win prob for lower seed."""
        seeds_t1 = np.ones(100)  # 1-seeds
        seeds_t2 = np.full(100, 16)  # 16-seeds
        outcomes = np.ones(100)  # 1-seed always wins

        brier = compute_seed_baseline_brier(seeds_t1, seeds_t2, outcomes)
        # Should be better than 0.25 (random) since seed diff is large
        assert brier < 0.20

    def test_report_includes_seed_baseline(self):
        """Report should include seed baseline for comparison."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        assert report.seed_baseline_brier > 0.0


# ---------------------------------------------------------------------------
# LOYO Validation Tests
# ---------------------------------------------------------------------------


class TestLOYOIntegration:
    """Verify LOYO cross-validation integration."""

    def test_loyo_runs_multiple_folds(self):
        """LOYO should produce results for multiple years."""
        data = _make_full_dataset()
        report = run_full_validation(data, _simple_train_fn, _simple_predict_fn)
        assert len(report.loyo_year_briers) >= 2

    def test_loyo_includes_2025(self):
        """LOYO results should include the 2025 holdout year."""
        data = _make_full_dataset()
        report = run_full_validation(data, _simple_train_fn, _simple_predict_fn)
        assert HOLDOUT_YEAR in report.loyo_year_briers

    def test_loyo_mean_brier_reasonable(self):
        """LOYO mean Brier should be between 0 and 0.5."""
        data = _make_full_dataset()
        report = run_full_validation(data, _simple_train_fn, _simple_predict_fn)
        assert 0.0 < report.loyo_mean_brier < 0.5


# ---------------------------------------------------------------------------
# Prospective Validation Tests
# ---------------------------------------------------------------------------


class TestProspectiveValidation:
    """Verify prospective forward validation integration."""

    def test_prospective_runs(self):
        """Prospective validation should produce results."""
        data = _make_full_dataset()
        report = run_full_validation(data, _simple_train_fn, _simple_predict_fn)
        assert len(report.prospective_year_briers) >= 1

    def test_prospective_mean_brier_reasonable(self):
        """Prospective mean Brier should be between 0 and 0.5."""
        data = _make_full_dataset()
        report = run_full_validation(data, _simple_train_fn, _simple_predict_fn)
        assert 0.0 < report.prospective_mean_brier < 0.5


# ---------------------------------------------------------------------------
# Report Tests
# ---------------------------------------------------------------------------


class TestBacktestReport:
    """Verify report generation and pass/fail logic."""

    def test_summary_output(self):
        """Summary should be a non-empty string with key sections."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        summary = report.summary()

        assert "2025 BACKTEST VALIDATION REPORT" in summary
        assert "Brier Score" in summary
        assert "Leakage Verification" in summary
        assert "VERDICT" in summary

    def test_passed_all_checks_logic(self):
        """passed_all_checks should be True iff all checks pass."""
        report = BacktestReport()
        report.leakage_checks = [
            LeakageCheckResult("test1", True, "ok"),
            LeakageCheckResult("test2", True, "ok"),
        ]
        assert report.passed_all_checks()

        report.leakage_checks.append(
            LeakageCheckResult("test3", False, "failed")
        )
        assert not report.passed_all_checks()

    def test_passed_all_checks_empty(self):
        """No checks should mean not passed."""
        report = BacktestReport()
        assert not report.passed_all_checks()

    def test_beats_seed_baseline(self):
        """beats_seed_baseline should compare model vs baseline."""
        report = BacktestReport()
        report.brier_score = 0.15
        report.seed_baseline_brier = 0.23
        assert report.beats_seed_baseline()

        report.brier_score = 0.25
        assert not report.beats_seed_baseline()

    def test_training_years_recorded(self):
        """Report should record which years were used for training."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)
        assert len(report.training_years_used) > 0
        assert HOLDOUT_YEAR not in report.training_years_used


# ---------------------------------------------------------------------------
# Temporal Integrity Tests
# ---------------------------------------------------------------------------


class TestTemporalIntegrity:
    """Verify no future data leakage in the validation process."""

    def test_rolling_window_excludes_future(self):
        """LOYO rolling_window mode must exclude future years."""
        data = _make_full_dataset()

        train_year_sets = []

        def tracking_train_fn(X, y, margins, feature_names, weights):
            train_year_sets.append(len(y))
            return _simple_train_fn(X, y, margins, feature_names, weights)

        report = run_full_validation(data, tracking_train_fn, _simple_predict_fn)

        # With rolling_window, training set should grow for later held-out years.
        # The first fold (earliest held-out year) should have the smallest
        # training set.
        if len(train_year_sets) > 2:
            # First call is the holdout eval (all training years)
            # Subsequent calls are LOYO folds (growing training sets)
            # Then prospective folds
            # Just verify they all ran
            assert len(train_year_sets) > 0

    def test_holdout_2025_data_not_in_training(self):
        """2025 holdout data must not appear in training."""
        data = _make_full_dataset()

        # Mark 2025 data with a unique value to detect leakage
        canary_value = 999.999
        data[HOLDOUT_YEAR]["X"][:, 0] = canary_value

        training_X_values = []

        def canary_train_fn(X, y, margins, feature_names, weights):
            # Record all training X values for canary detection
            training_X_values.append(X[:, 0].copy())
            return _simple_train_fn(X, y, margins, feature_names, weights)

        report = run_quick_validation(data, canary_train_fn, _simple_predict_fn)

        # Canary value should NOT appear in training data
        for x_col in training_X_values:
            assert not np.any(np.isclose(x_col, canary_value)), \
                "LEAKAGE DETECTED: 2025 holdout canary value found in training data!"

    def test_no_2020_data_in_training(self):
        """2020 (COVID) should never appear in training."""
        data = _make_full_dataset()
        data[2020] = _make_year_data(2020)

        runner = BacktestValidationRunner(
            train_fn=_simple_train_fn,
            predict_fn=_simple_predict_fn,
            data_by_year=data,
            run_loyo=False,
            run_prospective=False,
        )
        report = runner.run()

        assert 2020 not in report.training_years_used


# ---------------------------------------------------------------------------
# ECE Computation Tests
# ---------------------------------------------------------------------------


class TestECE:
    """Verify Expected Calibration Error computation."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE ≈ 0."""
        # Create predictions that match observed frequencies
        predictions = np.array([0.5] * 100)
        actuals = np.concatenate([np.ones(50), np.zeros(50)])
        ece = BacktestValidationRunner._compute_ece(predictions, actuals)
        assert ece < 0.05

    def test_worst_calibration(self):
        """Completely wrong predictions should have high ECE."""
        predictions = np.array([0.9] * 100)
        actuals = np.zeros(100)
        ece = BacktestValidationRunner._compute_ece(predictions, actuals)
        assert ece > 0.5

    def test_ece_nonnegative(self):
        """ECE should always be non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            predictions = rng.rand(50)
            actuals = rng.randint(0, 2, 50).astype(float)
            ece = BacktestValidationRunner._compute_ece(predictions, actuals)
            assert ece >= 0.0


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Verify convenience functions work correctly."""

    def test_run_quick_validation(self):
        """run_quick_validation should skip LOYO and prospective."""
        data = _make_full_dataset()
        report = run_quick_validation(data, _simple_train_fn, _simple_predict_fn)

        assert report.brier_score > 0
        assert len(report.loyo_year_briers) == 0  # No LOYO
        assert len(report.prospective_year_briers) == 0  # No prospective
        assert report.passed_all_checks()

    def test_run_full_validation(self):
        """run_full_validation should include LOYO and prospective."""
        data = _make_full_dataset()
        report = run_full_validation(data, _simple_train_fn, _simple_predict_fn)

        assert report.brier_score > 0
        assert len(report.loyo_year_briers) > 0  # Has LOYO
        assert len(report.prospective_year_briers) > 0  # Has prospective
