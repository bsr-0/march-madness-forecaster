"""Tests for compute budget manager (S20 compute budget management)."""

import time

import pytest

from src.monitoring.budget_manager import (
    BudgetManager,
    StageBudget,
)


class TestBudgetManager:
    def test_default_allocations(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)
        # Check that default stages got their allocations
        assert "data_loading" in mgr._stages
        assert "model_training" in mgr._stages
        # model_training gets 45%
        assert mgr._stages["model_training"].max_seconds == pytest.approx(450.0)

    def test_stage_lifecycle(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)

        with mgr.track_stage("data_loading") as budget:
            assert budget.started is True
            time.sleep(0.01)

        assert budget.completed is True
        assert budget.actual_seconds > 0

    def test_over_budget_detection(self):
        mgr = BudgetManager(total_budget_seconds=100.0)

        # data_loading gets 10% = 10s; simulate exceeding it
        budget = mgr._stages["data_loading"]
        budget.started = True
        budget.actual_seconds = 50.0
        budget.completed = True

        # Manually trigger alert check via utilization report
        report = mgr.utilization_report()
        assert report["stages"]["data_loading"]["utilization_pct"] == pytest.approx(500.0)

    def test_remaining_budget(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)

        with mgr.track_stage("data_loading"):
            time.sleep(0.01)

        remaining = mgr.remaining_budget()
        assert remaining < 1000.0
        assert remaining > 0

    def test_stage_remaining(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)
        # data_loading gets 10% = 100s
        remaining = mgr.stage_remaining("data_loading")
        assert remaining == pytest.approx(100.0)

        # Unknown stage returns 0
        assert mgr.stage_remaining("nonexistent") == 0.0

    def test_summary_output(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)

        with mgr.track_stage("data_loading"):
            time.sleep(0.01)

        summary = mgr.summary()
        assert "Compute Budget Report" in summary
        assert "data_loading" in summary

    def test_utilization_report(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)

        with mgr.track_stage("data_loading"):
            time.sleep(0.01)

        report = mgr.utilization_report()
        assert report["total_budget_seconds"] == 1000.0
        assert report["total_used_seconds"] >= 0
        assert "data_loading" in report["stages"]
        assert report["stages"]["data_loading"]["completed"] is True

    def test_custom_allocations(self):
        allocs = {
            "stage_a": {"fraction": 0.50, "priority": 1},
            "stage_b": {"fraction": 0.50, "priority": 2},
        }
        mgr = BudgetManager(total_budget_seconds=200.0, allocations=allocs)
        assert mgr._stages["stage_a"].max_seconds == pytest.approx(100.0)
        assert mgr._stages["stage_b"].max_seconds == pytest.approx(100.0)

    def test_allocate_new_stage(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)
        budget = mgr.allocate("custom_stage", fraction=0.20, priority=5)
        assert budget.stage_name == "custom_stage"
        assert budget.max_seconds == pytest.approx(200.0)
        assert budget.priority == 5

    def test_auto_allocate_unknown_stage_in_track(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)
        with mgr.track_stage("unknown_stage") as budget:
            assert budget.stage_name == "unknown_stage"
            # Auto-allocated with fraction=0, priority=0
            assert budget.max_seconds == 0.0

    def test_prioritized_stages(self):
        allocs = {
            "low": {"fraction": 0.30, "priority": 1},
            "high": {"fraction": 0.30, "priority": 3},
            "mid": {"fraction": 0.40, "priority": 2},
        }
        mgr = BudgetManager(total_budget_seconds=100.0, allocations=allocs)
        ordered = mgr.prioritized_stages()
        assert ordered[0] == "high"
        assert ordered[-1] == "low"

    def test_should_shed(self):
        mgr = BudgetManager(total_budget_seconds=100.0)
        # model_training has priority 1; simulate >80% budget consumed
        mgr._stages["data_loading"].actual_seconds = 85.0
        mgr._stages["data_loading"].completed = True

        # model_training (priority=1) should be shed when budget >80% used
        assert mgr.should_shed("model_training") is True
        # data_loading (priority=3) should NOT be shed
        assert mgr.should_shed("data_loading") is False

    def test_alerts_on_exceeded(self):
        mgr = BudgetManager(total_budget_seconds=100.0)
        # data_loading gets 10% = 10s
        budget = mgr._stages["data_loading"]

        # Simulate tracking that exceeds budget via the context manager
        with mgr.track_stage("data_loading"):
            # Manually set actual_seconds to simulate long run
            pass

        # Force an over-budget scenario by directly manipulating
        budget.actual_seconds = 20.0  # 200% of 10s budget
        mgr._add_alert(budget, "exceeded", 2.0)

        assert len(mgr.alerts) >= 1
        assert mgr.alerts[-1].alert_type == "exceeded"

    def test_get_budget(self):
        mgr = BudgetManager(total_budget_seconds=1000.0)
        budget = mgr.get_budget("model_training")
        assert budget is not None
        assert budget.stage_name == "model_training"
        assert mgr.get_budget("nonexistent") is None
