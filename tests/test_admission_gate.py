"""Tests for the Phase 2 hard admission gate.

Verifies:
- Passing candidates are admitted
- Failing candidates are rejected on each condition
- Report artifacts are emitted
- Multi-condition logic works correctly
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.ml.evaluation.admission_gate import (
    AdmissionGate,
    AdmissionResult,
    FoldMetrics,
    LOYOEvaluation,
)


def _make_loyo(name, fold_briers, fold_eces):
    """Helper to build a LOYOEvaluation from lists."""
    years = list(range(2018, 2018 + len(fold_briers)))
    folds = [
        FoldMetrics(year=y, brier_score=b, calibration_error=e, n_games=60)
        for y, b, e in zip(years, fold_briers, fold_eces)
    ]
    return LOYOEvaluation(name=name, fold_metrics=folds)


class TestAdmissionGatePass:
    """Test cases where candidate should pass."""

    def test_clear_improvement_passes(self):
        baseline = _make_loyo("baseline", [0.20, 0.22, 0.21], [0.03, 0.04, 0.03])
        candidate = _make_loyo("candidate", [0.19, 0.20, 0.19], [0.03, 0.03, 0.03])

        gate = AdmissionGate()
        result = gate.evaluate("test_candidate", baseline, candidate)

        assert result.passed is True
        assert result.brier_improvement > 0
        assert result.fold_improvement_rate >= 0.60
        assert "PASSED" in result.reason

    def test_marginal_improvement_passes(self):
        """Candidate that barely improves on most folds."""
        baseline = _make_loyo("baseline", [0.200, 0.220, 0.210, 0.230, 0.200],
                              [0.03, 0.04, 0.03, 0.04, 0.03])
        candidate = _make_loyo("candidate", [0.199, 0.219, 0.209, 0.229, 0.201],
                               [0.03, 0.04, 0.03, 0.04, 0.03])

        gate = AdmissionGate()
        result = gate.evaluate("marginal", baseline, candidate)

        # Wins 4/5 folds, mean improvement positive
        assert result.passed is True
        assert result.fold_improvement_rate == pytest.approx(0.80)


class TestAdmissionGateFailBrier:
    """Test failure on Brier score conditions."""

    def test_negative_improvement_fails(self):
        baseline = _make_loyo("baseline", [0.20, 0.22, 0.21], [0.03, 0.04, 0.03])
        candidate = _make_loyo("candidate", [0.22, 0.24, 0.23], [0.03, 0.04, 0.03])

        gate = AdmissionGate()
        result = gate.evaluate("worse", baseline, candidate)

        assert result.passed is False
        assert result.brier_improvement < 0
        assert "Mean Brier improvement" in result.reason

    def test_zero_improvement_fails(self):
        baseline = _make_loyo("baseline", [0.20, 0.22, 0.21], [0.03, 0.04, 0.03])
        candidate = _make_loyo("same", [0.20, 0.22, 0.21], [0.03, 0.04, 0.03])

        gate = AdmissionGate()
        result = gate.evaluate("same", baseline, candidate)

        assert result.passed is False
        assert "Mean Brier improvement" in result.reason


class TestAdmissionGateFailFoldConsistency:
    """Test failure on fold improvement rate."""

    def test_inconsistent_folds_fail(self):
        """Candidate improves on mean but only wins 2/5 folds."""
        baseline = _make_loyo("baseline", [0.20, 0.22, 0.21, 0.23, 0.24],
                              [0.03, 0.04, 0.03, 0.04, 0.03])
        # Big win on fold 5, but loses 3 folds
        candidate = _make_loyo("inconsistent", [0.21, 0.23, 0.22, 0.24, 0.10],
                               [0.03, 0.04, 0.03, 0.04, 0.03])

        gate = AdmissionGate()
        result = gate.evaluate("inconsistent", baseline, candidate)

        assert result.passed is False
        assert result.fold_improvement_rate < 0.60
        assert "Fold improvement rate" in result.reason


class TestAdmissionGateFailCalibration:
    """Test failure on calibration degradation."""

    def test_calibration_degradation_fails(self):
        baseline = _make_loyo("baseline", [0.20, 0.22, 0.21], [0.03, 0.03, 0.03])
        # Better Brier but much worse calibration
        candidate = _make_loyo("bad_cal", [0.19, 0.20, 0.19], [0.08, 0.09, 0.08])

        gate = AdmissionGate()
        result = gate.evaluate("bad_calibration", baseline, candidate)

        assert result.passed is False
        assert result.calibration_delta > 0.01
        assert "Calibration degradation" in result.reason


class TestAdmissionGateAbsoluteThreshold:
    """Test optional absolute Brier improvement threshold."""

    def test_below_absolute_threshold_fails(self):
        baseline = _make_loyo("baseline", [0.200, 0.220, 0.210], [0.03, 0.04, 0.03])
        candidate = _make_loyo("tiny", [0.1999, 0.2199, 0.2099], [0.03, 0.04, 0.03])

        gate = AdmissionGate(min_absolute_brier_improvement=0.001)
        result = gate.evaluate("tiny_improvement", baseline, candidate)

        assert result.passed is False
        assert "Absolute Brier improvement" in result.reason


class TestAdmissionResultSerialization:
    """Test report serialization and artifact output."""

    def test_to_dict_has_required_fields(self):
        baseline = _make_loyo("baseline", [0.20], [0.03])
        candidate = _make_loyo("candidate", [0.19], [0.03])

        gate = AdmissionGate()
        result = gate.evaluate("test", baseline, candidate)

        d = result.to_dict()
        assert "candidate_name" in d
        assert "baseline_brier" in d
        assert "candidate_brier" in d
        assert "brier_improvement" in d
        assert "calibration_delta" in d
        assert "n_folds" in d
        assert "fold_improvement_rate" in d
        assert "evaluation_protocol" in d
        assert "passed" in d
        assert "reason" in d
        assert "fold_details" in d
        assert "timestamp" in d

    def test_report_artifact_saved(self):
        baseline = _make_loyo("baseline", [0.20], [0.03])
        candidate = _make_loyo("candidate", [0.19], [0.03])

        with tempfile.TemporaryDirectory() as tmpdir:
            gate = AdmissionGate(report_dir=tmpdir)
            result = gate.evaluate("test_artifact", baseline, candidate)

            # Check report file was created
            reports = list(Path(tmpdir).glob("admission_*.json"))
            assert len(reports) == 1

            # Check contents are valid JSON with required fields
            with open(reports[0]) as f:
                data = json.load(f)
            assert data["candidate_name"] == "test_artifact"
            assert data["passed"] is True


class TestLOYOEvaluation:
    """Test LOYOEvaluation helper properties."""

    def test_mean_brier(self):
        loyo = _make_loyo("test", [0.20, 0.22, 0.24], [0.03, 0.04, 0.05])
        assert loyo.mean_brier == pytest.approx(0.22)

    def test_mean_calibration_error(self):
        loyo = _make_loyo("test", [0.20, 0.22, 0.24], [0.03, 0.04, 0.05])
        assert loyo.mean_calibration_error == pytest.approx(0.04)

    def test_n_folds(self):
        loyo = _make_loyo("test", [0.20, 0.22, 0.24], [0.03, 0.04, 0.05])
        assert loyo.n_folds == 3

    def test_empty_evaluation(self):
        loyo = LOYOEvaluation(name="empty", fold_metrics=[])
        assert loyo.mean_brier == float("inf")
        assert loyo.n_folds == 0
