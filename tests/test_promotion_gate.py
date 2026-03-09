"""Tests for the model promotion gate.

Implements Agent Directive V7 S14/S15: verifies that the promotion gate
correctly blocks underperforming candidates and promotes superior ones.
"""

import pytest

from src.ml.evaluation.promotion_gate import PromotionGate, PromotionResult
from src.ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry


class FakeRegistry:
    """Minimal registry stub for testing."""

    def __init__(self, best_record=None):
        self._best = best_record

    def best(self, metric="loyo_mean_brier"):
        return self._best


class TestPromotionGate:
    def test_no_incumbent_auto_promotes(self):
        gate = PromotionGate()
        result = gate.check(0.20, FakeRegistry(None))
        assert result.approved is True
        assert result.incumbent_brier is None
        assert "auto-promoted" in result.reason

    def test_better_candidate_promoted(self):
        incumbent = ExperimentRecord(
            experiment_id="inc-1",
            loyo_mean_brier=0.200,
        )
        gate = PromotionGate(min_improvement=0.001)
        result = gate.check(0.195, FakeRegistry(incumbent))
        assert result.approved is True
        assert result.delta < 0
        assert "beats" in result.reason

    def test_worse_candidate_blocked(self):
        incumbent = ExperimentRecord(
            experiment_id="inc-1",
            loyo_mean_brier=0.200,
        )
        gate = PromotionGate(min_improvement=0.001)
        result = gate.check(0.201, FakeRegistry(incumbent))
        assert result.approved is False
        assert result.delta > 0
        assert "does not beat" in result.reason

    def test_marginal_improvement_blocked(self):
        """Improvement less than threshold is blocked."""
        incumbent = ExperimentRecord(
            experiment_id="inc-1",
            loyo_mean_brier=0.200,
        )
        gate = PromotionGate(min_improvement=0.005)
        # 0.002 improvement is below 0.005 threshold
        result = gate.check(0.198, FakeRegistry(incumbent))
        assert result.approved is False

    def test_exact_threshold_approved(self):
        """Improvement exactly at threshold is approved (>= min_improvement)."""
        incumbent = ExperimentRecord(
            experiment_id="inc-1",
            loyo_mean_brier=0.200,
        )
        gate = PromotionGate(min_improvement=0.005)
        result = gate.check(0.195, FakeRegistry(incumbent))
        # delta = -0.005, threshold requires delta < -0.005
        assert result.approved is True

    def test_custom_threshold(self):
        incumbent = ExperimentRecord(
            experiment_id="inc-1",
            loyo_mean_brier=0.200,
        )
        gate = PromotionGate(min_improvement=0.01)
        # 0.015 improvement exceeds 0.01 threshold
        result = gate.check(0.185, FakeRegistry(incumbent))
        assert result.approved is True

    def test_result_fields_populated(self):
        incumbent = ExperimentRecord(
            experiment_id="inc-1",
            loyo_mean_brier=0.200,
        )
        gate = PromotionGate()
        result = gate.check(0.190, FakeRegistry(incumbent))
        assert isinstance(result, PromotionResult)
        assert result.candidate_brier == 0.190
        assert result.incumbent_brier == 0.200
        assert abs(result.delta - (-0.010)) < 1e-10
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_with_real_registry(self, tmp_path):
        """Test with actual ExperimentRegistry file."""
        ledger = tmp_path / "ledger.jsonl"
        registry = ExperimentRegistry(ledger_path=str(ledger))

        # No experiments yet
        gate = PromotionGate()
        result = gate.check(0.200, registry)
        assert result.approved is True

        # Log an experiment
        record = ExperimentRecord(loyo_mean_brier=0.200)
        registry.log(record)

        # Better candidate should be promoted
        result = gate.check(0.190, registry)
        assert result.approved is True

        # Worse candidate should be blocked
        result = gate.check(0.210, registry)
        assert result.approved is False
