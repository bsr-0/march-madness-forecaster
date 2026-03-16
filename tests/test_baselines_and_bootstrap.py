"""Tests for evaluation baselines and bootstrap metrics modules."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bootstrap Metrics
# ---------------------------------------------------------------------------

class TestBootstrapMetrics:
    """Tests for src.evaluation.bootstrap_metrics."""

    def test_brier_score_perfect(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert brier_score(preds, outcomes) == pytest.approx(0.0)

    def test_brier_score_worst(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert brier_score(preds, outcomes) == pytest.approx(1.0)

    def test_brier_score_moderate(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([0.7, 0.3])
        outcomes = np.array([1.0, 0.0])
        expected = ((0.7 - 1.0)**2 + (0.3 - 0.0)**2) / 2
        assert brier_score(preds, outcomes) == pytest.approx(expected)

    def test_log_loss_perfect_ish(self):
        from src.evaluation.bootstrap_metrics import log_loss
        preds = np.array([0.99, 0.01])
        outcomes = np.array([1.0, 0.0])
        assert log_loss(preds, outcomes) < 0.1

    def test_log_loss_bad(self):
        from src.evaluation.bootstrap_metrics import log_loss
        preds = np.array([0.1, 0.9])
        outcomes = np.array([1.0, 0.0])
        assert log_loss(preds, outcomes) > 1.0

    def test_accuracy_perfect(self):
        from src.evaluation.bootstrap_metrics import accuracy
        preds = np.array([0.9, 0.1, 0.8])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert accuracy(preds, outcomes) == pytest.approx(1.0)

    def test_accuracy_half(self):
        from src.evaluation.bootstrap_metrics import accuracy
        preds = np.array([0.9, 0.9])  # predict both team1
        outcomes = np.array([1.0, 0.0])  # one right, one wrong
        assert accuracy(preds, outcomes) == pytest.approx(0.5)

    def test_expected_calibration_error_perfect(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        # If predicted prob matches actual frequency, ECE should be low
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])  # 50% actual
        assert expected_calibration_error(preds, outcomes) < 0.1

    def test_expected_calibration_error_empty(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_upset_recall_no_upsets(self):
        from src.evaluation.bootstrap_metrics import upset_recall
        preds = np.array([0.8, 0.9])
        outcomes = np.array([1.0, 1.0])  # no upsets
        is_fav = np.array([True, True])
        assert upset_recall(preds, outcomes, is_fav) == 0.0

    def test_upset_recall_all_predicted(self):
        from src.evaluation.bootstrap_metrics import upset_recall
        preds = np.array([0.3, 0.2])  # predicted upsets (< 0.5)
        outcomes = np.array([0.0, 0.0])  # actual upsets
        is_fav = np.array([True, True])
        assert upset_recall(preds, outcomes, is_fav) == pytest.approx(1.0)

    def test_bootstrap_result_to_dict(self):
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(
            estimate=0.2, ci_lower=0.15, ci_upper=0.25,
            ci_level=0.95, n_bootstrap=1000, n_samples=100,
        )
        d = br.to_dict()
        assert d["estimate"] == 0.2
        assert d["ci_lower"] == 0.15

    def test_bootstrap_metric_basic(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        preds = np.array([0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1])
        outcomes = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        result = bootstrap_metric(brier_score, preds, outcomes, n_bootstrap=100, random_seed=42)
        assert result.estimate < 0.2
        assert result.ci_lower <= result.estimate <= result.ci_upper
        assert result.n_samples == 8

    def test_bootstrap_metric_empty(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        result = bootstrap_metric(brier_score, np.array([]), np.array([]))
        assert result.estimate == 0.0
        assert result.n_samples == 0

    def test_bootstrap_metric_tiny(self):
        from src.evaluation.bootstrap_metrics import bootstrap_metric, brier_score
        result = bootstrap_metric(brier_score, np.array([0.5]), np.array([1.0]))
        assert result.n_bootstrap == 0  # too few samples

    def test_bootstrap_brier(self):
        from src.evaluation.bootstrap_metrics import bootstrap_brier
        preds = np.random.default_rng(42).uniform(0.1, 0.9, 50)
        outcomes = (np.random.default_rng(42).random(50) > 0.5).astype(float)
        result = bootstrap_brier(preds, outcomes, n_bootstrap=100, random_seed=42)
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_bootstrap_log_loss(self):
        from src.evaluation.bootstrap_metrics import bootstrap_log_loss
        preds = np.random.default_rng(42).uniform(0.1, 0.9, 50)
        outcomes = (np.random.default_rng(42).random(50) > 0.5).astype(float)
        result = bootstrap_log_loss(preds, outcomes, n_bootstrap=100, random_seed=42)
        assert result.estimate > 0

    def test_bootstrap_accuracy(self):
        from src.evaluation.bootstrap_metrics import bootstrap_accuracy
        preds = np.random.default_rng(42).uniform(0.1, 0.9, 50)
        outcomes = (np.random.default_rng(42).random(50) > 0.5).astype(float)
        result = bootstrap_accuracy(preds, outcomes, n_bootstrap=100, random_seed=42)
        assert 0 <= result.estimate <= 1


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    """Tests for src.evaluation.baselines."""

    def test_seed_win_rates_dict(self):
        from src.evaluation.baselines import SEED_WIN_RATES
        assert (1, 16) in SEED_WIN_RATES
        assert SEED_WIN_RATES[(1, 16)] == pytest.approx(0.985)
        assert (8, 9) in SEED_WIN_RATES

    def test_seed_baseline_same_seed(self):
        from src.evaluation.baselines import seed_baseline_probability
        assert seed_baseline_probability(5, 5) == 0.5

    def test_seed_baseline_known_matchup(self):
        from src.evaluation.baselines import seed_baseline_probability
        p = seed_baseline_probability(1, 16)
        assert p == pytest.approx(0.985)

    def test_seed_baseline_reversed(self):
        from src.evaluation.baselines import seed_baseline_probability
        p1 = seed_baseline_probability(1, 16)
        p2 = seed_baseline_probability(16, 1)
        assert p1 + p2 == pytest.approx(1.0)

    def test_seed_baseline_unknown_matchup(self):
        from src.evaluation.baselines import seed_baseline_probability
        p = seed_baseline_probability(1, 8)  # not in SEED_WIN_RATES dict
        assert 0.5 < p < 1.0  # 1-seed should beat 8-seed

    def test_seed_baseline_logistic_approximation(self):
        from src.evaluation.baselines import seed_baseline_probability
        # 1 vs 2 should still favor seed 1 but closer to 50/50
        p = seed_baseline_probability(1, 2)
        assert 0.5 < p < 0.7


# ---------------------------------------------------------------------------
# Calibration diagnostics (imports coverage)
# ---------------------------------------------------------------------------

class TestCalibrationDiagnostics:
    """Tests for src.evaluation.calibration_diagnostics."""

    def test_compute_reliability_diagram(self):
        from src.evaluation.calibration_diagnostics import compute_reliability_diagram
        preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1], dtype=float)
        result = compute_reliability_diagram(preds, outcomes, n_bins=5)
        assert result is not None
        assert hasattr(result, "bins") or hasattr(result, "ece")


# ---------------------------------------------------------------------------
# Evaluation report (import coverage)
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    """Tests for src.evaluation.evaluation_report."""

    def test_module_imports(self):
        from src.evaluation import evaluation_report
        assert evaluation_report is not None


# ---------------------------------------------------------------------------
# Seed gap calibration (import coverage)
# ---------------------------------------------------------------------------

class TestSeedGapCalibration:
    """Tests for src.evaluation.seed_gap_calibration."""

    def test_module_imports(self):
        from src.evaluation import seed_gap_calibration
        assert seed_gap_calibration is not None

    def test_calibration_by_round(self):
        from src.evaluation.seed_gap_calibration import calibration_by_round
        preds = np.random.default_rng(42).uniform(0.3, 0.9, 20)
        outcomes = (np.random.default_rng(42).random(20) > 0.4).astype(float)
        rounds = ["R64"] * 8 + ["R32"] * 4 + ["S16"] * 4 + ["E8"] * 2 + ["F4"] + ["NCG"]
        result = calibration_by_round(preds, outcomes, rounds)
        assert result is not None
        assert isinstance(result, (dict, list))


# ---------------------------------------------------------------------------
# Round analysis (import coverage)
# ---------------------------------------------------------------------------

class TestRoundAnalysis:
    """Tests for src.evaluation.round_analysis."""

    def test_round_name_map_exists(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert isinstance(ROUND_NAME_MAP, dict)
        # Common round names should be in the map
        assert len(ROUND_NAME_MAP) > 0


# ---------------------------------------------------------------------------
# Snapshot integrity (import coverage)
# ---------------------------------------------------------------------------

class TestSnapshotIntegrity:
    """Tests for src.evaluation.snapshot_integrity."""

    def test_module_imports(self):
        from src.evaluation import snapshot_integrity
        assert snapshot_integrity is not None


# ---------------------------------------------------------------------------
# Bracket snapshot (import coverage)
# ---------------------------------------------------------------------------

class TestBracketSnapshot:
    """Tests for src.evaluation.bracket_snapshot."""

    def test_module_imports(self):
        from src.evaluation import bracket_snapshot
        assert bracket_snapshot is not None


# ---------------------------------------------------------------------------
# Calibration benchmark (import coverage)
# ---------------------------------------------------------------------------

class TestCalibrationBenchmark:
    """Tests for src.evaluation.calibration_benchmark."""

    def test_module_imports(self):
        from src.evaluation import calibration_benchmark
        assert calibration_benchmark is not None
