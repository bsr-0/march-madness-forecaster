"""Tests for calibration metric computation.

Verifies:
- Brier, log loss, ECE compute correctly on known distributions
- Bootstrap CIs have correct coverage properties
- Seed-gap calibration bins are correct
- Baselines produce expected outputs
- Calibration gate pass/fail logic works
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.bootstrap_metrics import (
    BootstrapResult,
    accuracy,
    bootstrap_accuracy,
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_log_loss,
    bootstrap_metric,
    brier_score,
    compute_all_metrics_with_ci,
    expected_calibration_error,
    log_loss,
)
from src.evaluation.calibration_diagnostics import (
    CalibrationDiagnostics,
    CalibrationReport,
    ReliabilityDiagram,
    compute_predicted_vs_actual_bins,
    compute_reliability_diagram,
    compute_upset_distributions,
)
from src.evaluation.seed_gap_calibration import (
    SeedGapCalibrationReport,
    calibration_by_favorite_strength,
    calibration_by_round,
    calibration_by_seed_gap,
    compute_seed_gap_calibration,
)
from src.evaluation.baselines import (
    BaselineComparison,
    compare_to_baseline,
    make_constant_baseline_predictions,
    make_seed_baseline_predictions,
    run_all_baseline_comparisons,
    seed_baseline_probability,
)
from src.evaluation.calibration_gate import (
    CalibrationGate,
    CalibrationGateResult,
    FrozenProbabilities,
)


# ---------------------------------------------------------------------------
# Core metric computation tests
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Verify Brier score computation."""

    def test_perfect_predictions(self):
        """Perfect predictions yield Brier = 0."""
        preds = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(preds, outcomes) == 0.0

    def test_worst_predictions(self):
        """Completely wrong predictions yield Brier = 1."""
        preds = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(preds, outcomes) == 1.0

    def test_constant_half(self):
        """Constant 0.5 predictions yield Brier = 0.25."""
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert abs(brier_score(preds, outcomes) - 0.25) < 1e-10

    def test_known_value(self):
        """Check a specific known Brier calculation."""
        preds = np.array([0.9, 0.1])
        outcomes = np.array([1.0, 0.0])
        expected = ((0.9 - 1.0) ** 2 + (0.1 - 0.0) ** 2) / 2  # = 0.01
        assert abs(brier_score(preds, outcomes) - expected) < 1e-10


class TestLogLoss:
    """Verify log loss computation."""

    def test_perfect_predictions(self):
        """Near-perfect predictions yield very low log loss."""
        preds = np.array([0.999, 0.001, 0.999])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert log_loss(preds, outcomes) < 0.01

    def test_constant_half(self):
        """Constant 0.5 predictions yield log loss = ln(2) ≈ 0.693."""
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        expected = math.log(2)
        assert abs(log_loss(preds, outcomes) - expected) < 1e-6

    def test_clipping_prevents_inf(self):
        """Predictions of exactly 0 or 1 don't produce inf."""
        preds = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        result = log_loss(preds, outcomes)
        assert math.isfinite(result)


class TestAccuracy:
    """Verify accuracy computation."""

    def test_perfect_accuracy(self):
        """All correct predictions yield accuracy = 1.0."""
        preds = np.array([0.9, 0.1, 0.8, 0.2])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert accuracy(preds, outcomes) == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions yield accuracy = 0.0."""
        preds = np.array([0.1, 0.9, 0.2, 0.8])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert accuracy(preds, outcomes) == 0.0

    def test_half_accuracy(self):
        """Mixed predictions yield 0.5 accuracy."""
        preds = np.array([0.9, 0.9, 0.1, 0.1])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert accuracy(preds, outcomes) == 0.5


class TestECE:
    """Verify Expected Calibration Error computation."""

    def test_perfectly_calibrated(self):
        """Perfectly calibrated predictions yield ECE ≈ 0."""
        # 100 predictions at 0.7, 70% of which are true
        rng = np.random.default_rng(42)
        n = 1000
        preds = np.full(n, 0.7)
        outcomes = (rng.random(n) < 0.7).astype(float)
        ece = expected_calibration_error(preds, outcomes, n_bins=5)
        assert ece < 0.05  # Should be very small with large n

    def test_empty_predictions(self):
        """Empty arrays yield ECE = 0."""
        assert expected_calibration_error(np.array([]), np.array([]), 10) == 0.0

    def test_miscalibrated(self):
        """Clearly miscalibrated predictions yield non-zero ECE."""
        # Predict 0.9 but actual rate is 0.5
        preds = np.full(100, 0.9)
        outcomes = np.concatenate([np.ones(50), np.zeros(50)])
        ece = expected_calibration_error(preds, outcomes, n_bins=5)
        assert ece > 0.3


# ---------------------------------------------------------------------------
# Bootstrap confidence interval tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    """Verify bootstrap CI properties."""

    def test_ci_contains_point_estimate(self):
        """CI should contain the point estimate."""
        preds = np.array([0.7, 0.6, 0.8, 0.9, 0.3, 0.5, 0.4, 0.6, 0.7, 0.8])
        outcomes = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        result = bootstrap_brier(preds, outcomes, n_bootstrap=1000)
        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_ci_width_decreases_with_n(self):
        """Larger samples should produce narrower CIs."""
        rng = np.random.default_rng(42)

        small_preds = rng.random(20)
        small_outcomes = (rng.random(20) > 0.5).astype(float)
        small_result = bootstrap_brier(small_preds, small_outcomes, n_bootstrap=500)

        large_preds = rng.random(200)
        large_outcomes = (rng.random(200) > 0.5).astype(float)
        large_result = bootstrap_brier(large_preds, large_outcomes, n_bootstrap=500)

        small_width = small_result.ci_upper - small_result.ci_lower
        large_width = large_result.ci_upper - large_result.ci_lower
        assert large_width < small_width

    def test_bootstrap_reproducible(self):
        """Same random seed produces same result."""
        preds = np.array([0.7, 0.3, 0.6, 0.8, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        r1 = bootstrap_brier(preds, outcomes, random_seed=123)
        r2 = bootstrap_brier(preds, outcomes, random_seed=123)
        assert r1.estimate == r2.estimate
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_small_sample_returns_point_only(self):
        """With < 3 samples, CI equals point estimate."""
        preds = np.array([0.7, 0.3])
        outcomes = np.array([1.0, 0.0])
        result = bootstrap_brier(preds, outcomes)
        assert result.ci_lower == result.estimate
        assert result.ci_upper == result.estimate

    def test_empty_returns_zero(self):
        """Empty arrays produce zero result."""
        result = bootstrap_brier(np.array([]), np.array([]))
        assert result.estimate == 0.0
        assert result.n_samples == 0

    def test_compute_all_metrics_returns_all_keys(self):
        """compute_all_metrics_with_ci returns all expected metrics."""
        preds = np.array([0.7, 0.3, 0.6, 0.8, 0.5, 0.4, 0.9, 0.2, 0.6, 0.7])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        result = compute_all_metrics_with_ci(preds, outcomes, n_bootstrap=100)
        assert "brier_score" in result
        assert "log_loss" in result
        assert "accuracy" in result
        assert "ece" in result


# ---------------------------------------------------------------------------
# Reliability diagram tests
# ---------------------------------------------------------------------------

class TestReliabilityDiagram:
    """Verify reliability diagram computation."""

    def test_bins_cover_full_range(self):
        """Bins should cover [0, 1]."""
        preds = np.array([0.1, 0.5, 0.9])
        outcomes = np.array([0.0, 1.0, 1.0])
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=5)
        assert diagram.bins[0].bin_lower == 0.0
        assert diagram.bins[-1].bin_upper == 1.0

    def test_n_bins_correct(self):
        """Number of bins matches request."""
        preds = np.linspace(0, 1, 100)
        outcomes = (preds > 0.5).astype(float)
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=10)
        assert diagram.n_bins == 10
        assert len(diagram.bins) == 10

    def test_visualization_ready_data(self):
        """x_predicted and y_actual are available for plotting."""
        preds = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=5)
        assert len(diagram.x_predicted) > 0
        assert len(diagram.y_actual) > 0

    def test_serializable(self):
        """Diagram can be serialized to dict."""
        preds = np.array([0.3, 0.7])
        outcomes = np.array([0.0, 1.0])
        diagram = compute_reliability_diagram(preds, outcomes, n_bins=5)
        d = diagram.to_dict()
        assert "bins" in d
        assert "ece" in d


# ---------------------------------------------------------------------------
# Predicted vs actual bins tests
# ---------------------------------------------------------------------------

class TestPredictedVsActualBins:
    """Verify predicted vs actual win rate bins."""

    def test_default_bins(self):
        """Default produces 10 bins."""
        preds = np.linspace(0, 1, 100)
        outcomes = (preds > 0.5).astype(float)
        bins = compute_predicted_vs_actual_bins(preds, outcomes)
        assert len(bins) == 10

    def test_bin_fields(self):
        """Each bin has required fields."""
        preds = np.array([0.3, 0.7])
        outcomes = np.array([0.0, 1.0])
        bins = compute_predicted_vs_actual_bins(preds, outcomes)
        for b in bins:
            assert "bin_lower" in b
            assert "bin_upper" in b
            assert "count" in b
            assert "actual_win_rate" in b
            assert "mean_predicted" in b


# ---------------------------------------------------------------------------
# Upset distribution tests
# ---------------------------------------------------------------------------

class TestUpsetDistributions:
    """Verify upset probability distributions by seed gap."""

    def test_groups_by_seed_gap(self):
        """Results are grouped by seed gap."""
        preds = np.array([0.95, 0.9, 0.6, 0.55])
        outcomes = np.array([1.0, 1.0, 1.0, 0.0])
        fav_seeds = np.array([1, 2, 7, 8])
        dog_seeds = np.array([16, 15, 10, 9])
        result = compute_upset_distributions(preds, outcomes, fav_seeds, dog_seeds)
        gaps = [r.seed_gap for r in result]
        assert 15 in gaps  # 1v16
        assert 13 in gaps  # 2v15
        assert 3 in gaps   # 7v10
        assert 1 in gaps   # 8v9

    def test_upset_rate_correct(self):
        """Historical upset rate is correct for each gap."""
        # Two 1v16 games, one upset
        preds = np.array([0.95, 0.9])
        outcomes = np.array([1.0, 0.0])
        fav_seeds = np.array([1, 1])
        dog_seeds = np.array([16, 16])
        result = compute_upset_distributions(preds, outcomes, fav_seeds, dog_seeds)
        assert len(result) == 1
        assert result[0].seed_gap == 15
        assert result[0].n_upsets == 1
        assert abs(result[0].historical_upset_rate - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# Seed-gap calibration tests
# ---------------------------------------------------------------------------

class TestSeedGapCalibration:
    """Verify seed-gap calibration breakdowns."""

    def test_calibration_by_seed_gap(self):
        """Calibration computed per seed gap."""
        preds = np.array([0.95, 0.9, 0.6, 0.55, 0.7, 0.65])
        outcomes = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        gaps = np.array([15, 15, 3, 1, 3, 1])
        result = calibration_by_seed_gap(preds, outcomes, gaps, n_bootstrap=50)
        assert len(result) >= 2
        gap_names = [b.bucket_name for b in result]
        assert "gap_15" in gap_names

    def test_calibration_by_favorite_strength(self):
        """Calibration computed per favorite strength bin."""
        preds = np.array([0.55, 0.65, 0.75, 0.85, 0.95, 0.6, 0.7, 0.8, 0.9, 0.55])
        outcomes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        result = calibration_by_favorite_strength(preds, outcomes, n_bootstrap=50)
        assert len(result) == 5  # Default 5 bins
        for bucket in result:
            assert "strength_" in bucket.bucket_name

    def test_calibration_by_round(self):
        """Calibration computed per round."""
        preds = np.array([0.7, 0.8, 0.6, 0.7, 0.55, 0.5])
        outcomes = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        rounds = ["R64", "R64", "R32", "R32", "S16", "S16"]
        result = calibration_by_round(preds, outcomes, rounds, n_bootstrap=50)
        round_names = [b.bucket_name for b in result]
        assert "R64" in round_names
        assert "R32" in round_names

    def test_full_seed_gap_calibration_report(self):
        """Full report has all sections."""
        preds = np.array([0.95, 0.8, 0.6, 0.55, 0.7, 0.65, 0.9, 0.85, 0.75, 0.5])
        outcomes = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        gaps = np.array([15, 13, 3, 1, 7, 5, 11, 9, 3, 1])
        rounds = ["R64"] * 10
        report = compute_seed_gap_calibration(
            preds, outcomes, gaps, round_names=rounds, n_bootstrap=50,
        )
        assert report.n_total == 10
        assert len(report.by_seed_gap) > 0
        assert len(report.by_favorite_strength) > 0
        assert len(report.by_round) > 0

    def test_report_serializable(self):
        """Report can be serialized to dict."""
        preds = np.array([0.8, 0.6, 0.7])
        outcomes = np.array([1.0, 0.0, 1.0])
        gaps = np.array([15, 3, 1])
        report = compute_seed_gap_calibration(preds, outcomes, gaps, n_bootstrap=50)
        d = report.to_dict()
        assert "by_seed_gap" in d
        assert "by_favorite_strength" in d
        assert "overall_ece" in d


# ---------------------------------------------------------------------------
# Baseline model tests
# ---------------------------------------------------------------------------

class TestBaselines:
    """Verify baseline prediction functions."""

    def test_seed_baseline_1v16(self):
        """1-seed should beat 16-seed ~98.5% historically."""
        p = seed_baseline_probability(1, 16)
        assert 0.95 < p < 1.0

    def test_seed_baseline_8v9(self):
        """8v9 should be close to 50/50."""
        p = seed_baseline_probability(8, 9)
        assert 0.45 < p < 0.55

    def test_seed_baseline_symmetric(self):
        """P(A beats B) + P(B beats A) = 1."""
        p1 = seed_baseline_probability(3, 14)
        p2 = seed_baseline_probability(14, 3)
        assert abs(p1 + p2 - 1.0) < 1e-10

    def test_seed_baseline_equal_seeds(self):
        """Equal seeds yield 0.5."""
        assert seed_baseline_probability(5, 5) == 0.5

    def test_make_seed_baseline_predictions(self):
        """make_seed_baseline_predictions returns correct structure."""
        matchups = [("teamA", "teamB"), ("teamC", "teamD")]
        seeds = {"teamA": 1, "teamB": 16, "teamC": 8, "teamD": 9}
        preds = make_seed_baseline_predictions(matchups, seeds)
        assert ("teamA", "teamB") in preds
        assert preds[("teamA", "teamB")] > 0.9

    def test_constant_baseline(self):
        """Constant baseline returns the specified constant."""
        matchups = [("a", "b"), ("c", "d")]
        preds = make_constant_baseline_predictions(matchups, 0.5)
        assert all(v == 0.5 for v in preds.values())

    def test_compare_to_baseline(self):
        """compare_to_baseline produces valid comparison."""
        model_preds = np.array([0.8, 0.3, 0.7, 0.9, 0.4, 0.6, 0.2, 0.85, 0.55, 0.75])
        baseline_preds = np.full(10, 0.5)
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        result = compare_to_baseline(
            model_preds, baseline_preds, outcomes,
            n_bootstrap=100,
        )
        assert result.n_games == 10
        assert result.model_brier is not None
        assert result.baseline_brier is not None
        # Model should beat constant 0.5 baseline
        assert result.brier_improvement > 0

    def test_run_all_baseline_comparisons(self):
        """run_all_baseline_comparisons produces at least seed + constant baselines."""
        model_preds = np.array([0.9, 0.2, 0.7, 0.8, 0.4])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        seeds_t1 = np.array([1, 16, 3, 2, 8])
        seeds_t2 = np.array([16, 1, 14, 15, 9])
        results = run_all_baseline_comparisons(
            model_preds, outcomes, seeds_t1, seeds_t2,
            n_bootstrap=100,
        )
        baseline_names = [r.baseline_name for r in results]
        assert "seed_only" in baseline_names
        assert "constant_0.5" in baseline_names


# ---------------------------------------------------------------------------
# Calibration gate tests
# ---------------------------------------------------------------------------

class TestCalibrationGate:
    """Verify calibration gate pass/fail logic."""

    def test_good_model_passes_gate(self):
        """A well-calibrated model passes the gate."""
        # Generate well-calibrated predictions
        rng = np.random.default_rng(42)
        n = 200
        true_probs = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < true_probs).astype(float)

        gate = CalibrationGate(n_bootstrap=100)
        result = gate.evaluate(true_probs, outcomes)
        assert result.stage1_passed

    def test_bad_model_fails_gate(self):
        """A poorly calibrated model fails the gate."""
        # Predictions are all 0.9 but outcomes are 50/50
        preds = np.full(100, 0.9)
        outcomes = np.concatenate([np.ones(50), np.zeros(50)])

        gate = CalibrationGate(
            thresholds={"max_brier": 0.15, "max_ece": 0.05},
            n_bootstrap=100,
        )
        result = gate.evaluate(preds, outcomes)
        assert not result.stage1_passed
        assert len(result.failure_reasons) > 0

    def test_custom_thresholds(self):
        """Custom thresholds are used correctly."""
        gate = CalibrationGate(thresholds={"max_brier": 0.5})
        preds = np.array([0.5] * 20)
        outcomes = np.array([1.0, 0.0] * 10)
        result = gate.evaluate(preds, outcomes, random_seed=42)
        # Brier = 0.25, which is < 0.5 threshold
        assert result.stage1_passed

    def test_gate_result_serializable(self):
        """Gate result can be serialized."""
        gate = CalibrationGate(n_bootstrap=50)
        preds = np.array([0.7, 0.3, 0.6, 0.8, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        result = gate.evaluate(preds, outcomes)
        d = result.to_dict()
        assert "stage1_passed" in d
        assert "checks" in d


# ---------------------------------------------------------------------------
# Frozen probabilities tests
# ---------------------------------------------------------------------------

class TestFrozenProbabilities:
    """Verify FrozenProbabilities is read-only."""

    def test_get_existing_key(self):
        """Can retrieve stored probabilities."""
        fp = FrozenProbabilities({("a", "b"): 0.7})
        assert fp.get("a", "b") == 0.7

    def test_get_reverse_key(self):
        """Reverse key returns 1 - stored probability."""
        fp = FrozenProbabilities({("a", "b"): 0.7})
        assert abs(fp.get("b", "a") - 0.3) < 1e-10

    def test_get_missing_key(self):
        """Missing key returns default."""
        fp = FrozenProbabilities({("a", "b"): 0.7})
        assert fp.get("x", "y") == 0.5

    def test_contains(self):
        """__contains__ works for both orientations."""
        fp = FrozenProbabilities({("a", "b"): 0.7})
        assert ("a", "b") in fp
        assert ("b", "a") in fp
        assert ("x", "y") not in fp

    def test_immutable_copy(self):
        """Modifying the original dict doesn't affect FrozenProbabilities."""
        orig = {("a", "b"): 0.7}
        fp = FrozenProbabilities(orig)
        orig[("a", "b")] = 0.1
        assert fp.get("a", "b") == 0.7


# ---------------------------------------------------------------------------
# CalibrationDiagnostics integration test
# ---------------------------------------------------------------------------

class TestCalibrationDiagnosticsIntegration:
    """Verify the CalibrationDiagnostics class produces a complete report."""

    def test_full_report(self):
        """Full diagnostics run produces all sections."""
        rng = np.random.default_rng(42)
        n = 100
        preds = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < preds).astype(float)
        fav_seeds = rng.integers(1, 9, n)
        dog_seeds = 17 - fav_seeds

        diag = CalibrationDiagnostics()
        report = diag.compute(preds, outcomes, fav_seeds, dog_seeds)

        assert report.reliability is not None
        assert report.ece >= 0
        assert len(report.predicted_vs_actual_bins) > 0
        assert len(report.upset_distributions) > 0

    def test_report_without_seeds(self):
        """Report works without seed data (no upset distributions)."""
        preds = np.array([0.7, 0.3, 0.6])
        outcomes = np.array([1.0, 0.0, 1.0])

        diag = CalibrationDiagnostics()
        report = diag.compute(preds, outcomes)

        assert report.reliability is not None
        assert len(report.upset_distributions) == 0

    def test_report_serializable(self):
        """Full report can be serialized."""
        preds = np.array([0.7, 0.3, 0.6, 0.8, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0, 0.0])

        diag = CalibrationDiagnostics()
        report = diag.compute(preds, outcomes)
        d = report.to_dict()
        assert "reliability" in d
        assert "ece" in d
