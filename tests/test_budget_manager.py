"""Tests for compute budget manager (S20 compute budget management)."""

import pytest

from src.monitoring.budget_manager import (
    AgentBudget,
    ComputeBudgetManager,
)


class TestComputeBudgetManager:
    def test_default_allocations(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        # Check that agents got their default allocations
        assert "data_agent" in mgr._agent_budgets
        assert "model_agent" in mgr._agent_budgets
        # model_agent gets 45%
        assert mgr._agent_budgets["model_agent"].max_wall_seconds == pytest.approx(450.0)

    def test_agent_lifecycle(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        mgr.start()

        budget = mgr.agent_started("data_agent")
        assert budget.status == "running"

        budget = mgr.agent_completed("data_agent", wall_seconds=50.0)
        assert budget.status == "completed"
        assert budget.actual_wall_seconds == 50.0

    def test_over_budget_detection(self):
        mgr = ComputeBudgetManager(total_budget_seconds=100.0)
        mgr.start()
        mgr.agent_started("data_agent")

        # data_agent gets 15% = 15s, but takes 50s
        budget = mgr.agent_completed("data_agent", wall_seconds=50.0)
        assert budget.status == "over_budget"

    def test_rebalancing(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        mgr.start()

        # data_agent gets 150s but finishes in 50s — 100s saved
        model_budget_before = mgr._agent_budgets["model_agent"].max_wall_seconds
        mgr.agent_started("data_agent")
        mgr.agent_completed("data_agent", wall_seconds=50.0)

        # model_agent should have received a share of the 100s savings
        model_budget_after = mgr._agent_budgets["model_agent"].max_wall_seconds
        assert model_budget_after > model_budget_before

    def test_estimated_cost(self):
        mgr = ComputeBudgetManager(total_budget_seconds=3600.0, cost_per_hour=1.00)
        mgr.start()
        mgr.agent_started("data_agent")
        mgr.agent_completed("data_agent", wall_seconds=3600.0)
        cost = mgr.estimated_cost()
        assert cost == pytest.approx(1.00, abs=0.01)

    def test_auto_allocate_unknown_agent(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        mgr.start()
        budget = mgr.agent_started("unknown_agent")
        assert budget.agent_name == "unknown_agent"
        assert budget.max_wall_seconds > 0

    def test_summary_output(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        mgr.start()
        mgr.agent_started("data_agent")
        mgr.agent_completed("data_agent", wall_seconds=50.0)
        summary = mgr.summary()
        assert "Compute Budget Summary" in summary
        assert "data_agent" in summary

    def test_to_dict(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        mgr.start()
        mgr.agent_started("data_agent")
        mgr.agent_completed("data_agent", wall_seconds=100.0)
        d = mgr.to_dict()
        assert d["total_budget_seconds"] == 1000.0
        assert d["total_consumed_seconds"] == 100.0
        assert "data_agent" in d["agents"]

    def test_check_budget(self):
        mgr = ComputeBudgetManager(total_budget_seconds=1000.0)
        mgr.start()
        mgr.agent_started("data_agent")
        mgr.agent_completed("data_agent", wall_seconds=100.0)
        status = mgr.check_budget("data_agent")
        assert status["status"] == "completed"
        assert status["actual_seconds"] == 100.0

    def test_custom_allocations(self):
        allocs = {"agent_a": 0.50, "agent_b": 0.50}
        mgr = ComputeBudgetManager(total_budget_seconds=200.0, allocations=allocs)
        assert mgr._agent_budgets["agent_a"].max_wall_seconds == pytest.approx(100.0)
        assert mgr._agent_budgets["agent_b"].max_wall_seconds == pytest.approx(100.0)
