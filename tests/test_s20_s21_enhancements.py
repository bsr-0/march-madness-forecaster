"""Tests for S20 (compute budget) and S21 (governance) enhancements.

Covers:
  - BudgetManager: per-stage allocation, tracking, alerts, shedding
  - ComplianceGate: stage-boundary compliance checks
  - EscalationProtocol: auto-escalation, resolve, query
  - Integration: BudgetManager + ComplianceGate working together
"""

import time

import pytest


# ---------------------------------------------------------------------------
# BudgetManager tests (S20)
# ---------------------------------------------------------------------------


class TestBudgetManager:
    """S20: Per-stage compute budget allocation and tracking."""

    def test_default_allocations(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(total_budget_seconds=1000)
        budget = manager.get_budget("model_training")
        assert budget is not None
        assert budget.max_seconds == 450.0  # 45% of 1000

    def test_custom_allocation(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(
            total_budget_seconds=600,
            allocations={"my_stage": {"fraction": 0.5, "priority": 2}},
        )
        budget = manager.get_budget("my_stage")
        assert budget is not None
        assert budget.max_seconds == 300.0

    def test_track_stage(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(total_budget_seconds=3600)
        with manager.track_stage("data_loading") as budget:
            time.sleep(0.01)

        assert budget.completed
        assert budget.actual_seconds > 0

    def test_remaining_budget(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(total_budget_seconds=100)
        with manager.track_stage("data_loading"):
            time.sleep(0.01)

        remaining = manager.remaining_budget()
        assert remaining < 100
        assert remaining > 0

    def test_alert_on_budget_exceeded(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(
            total_budget_seconds=1.0,
            allocations={"slow_stage": {"fraction": 0.01, "priority": 1}},
        )
        # Budget for slow_stage = 0.01s; sleeping 0.05s should exceed
        with manager.track_stage("slow_stage"):
            time.sleep(0.05)

        alerts = manager.alerts
        assert len(alerts) >= 1
        assert alerts[0].alert_type in ("exceeded", "critical")

    def test_no_alert_under_budget(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(
            total_budget_seconds=3600,
            allocations={"fast_stage": {"fraction": 0.5, "priority": 1}},
        )
        with manager.track_stage("fast_stage"):
            pass  # Instant

        assert len(manager.alerts) == 0

    def test_utilization_report(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(total_budget_seconds=1000)
        with manager.track_stage("data_loading"):
            pass

        report = manager.utilization_report()
        assert "total_budget_seconds" in report
        assert "stages" in report
        assert "data_loading" in report["stages"]
        assert report["stages"]["data_loading"]["completed"]

    def test_summary(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(total_budget_seconds=3600)
        with manager.track_stage("data_loading"):
            pass
        summary = manager.summary()
        assert "Compute Budget Report" in summary
        assert "data_loading" in summary

    def test_prioritized_stages(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(total_budget_seconds=100)
        stages = manager.prioritized_stages()
        assert len(stages) > 0
        # data_loading has priority 3, model_training has priority 1
        assert stages.index("data_loading") < stages.index("model_training")

    def test_should_shed_under_pressure(self):
        from src.monitoring.budget_manager import BudgetManager

        manager = BudgetManager(
            total_budget_seconds=1.0,
            allocations={
                "important": {"fraction": 0.9, "priority": 3},
                "optional": {"fraction": 0.1, "priority": 1},
            },
        )
        # Consume 90% of budget
        with manager.track_stage("important"):
            time.sleep(0.9)

        # optional should be shed (low priority, budget >80% used)
        assert manager.should_shed("optional")


# ---------------------------------------------------------------------------
# ComplianceGate tests (S21)
# ---------------------------------------------------------------------------


class TestComplianceGate:
    """S21: Compliance checkpoints at stage boundaries."""

    def test_data_loading_pass(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("data_loading", {
            "n_teams": 68,
            "has_stale_data": False,
        })
        assert result.passed
        assert result.n_errors == 0

    def test_data_loading_fail_too_few_teams(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("data_loading", {
            "n_teams": 10,
            "has_stale_data": False,
        })
        assert not result.passed
        assert result.n_errors >= 1

    def test_model_training_leakage_critical(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("model_training", {
            "brier_score": 0.21,
            "has_leakage": True,
        })
        assert not result.passed

    def test_model_training_pass(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("model_training", {
            "brier_score": 0.21,
            "has_leakage": False,
        })
        assert result.passed

    def test_stale_data_warning(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("data_loading", {
            "n_teams": 68,
            "has_stale_data": True,
        })
        # Stale data is a warning, not an error — gate still passes
        assert result.passed
        assert result.n_warnings >= 1

    def test_custom_rule(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        gate.add_rule(
            "model_training",
            name="ensemble_size",
            check=lambda ctx: ctx.get("n_models", 0) >= 3,
            severity="error",
            message="Need at least 3 models in ensemble",
        )
        result = gate.check("model_training", {
            "brier_score": 0.21,
            "has_leakage": False,
            "n_models": 2,
        })
        assert not result.passed

    def test_unknown_stage_passes(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("unknown_stage", {})
        assert result.passed  # No rules = no failures

    def test_checkpoint_to_dict(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        result = gate.check("data_loading", {"n_teams": 68})
        d = result.to_dict()
        assert "checkpoint_id" in d
        assert "passed" in d
        assert "checks" in d

    def test_history_tracked(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        gate.check("data_loading", {"n_teams": 68})
        gate.check("model_training", {"brier_score": 0.21, "has_leakage": False})
        assert len(gate.history) == 2

    def test_summary(self):
        from src.governance.compliance import ComplianceGate

        gate = ComplianceGate()
        gate.check("data_loading", {"n_teams": 68})
        summary = gate.summary()
        assert "Compliance Checkpoint" in summary
        assert "data_loading" in summary


# ---------------------------------------------------------------------------
# EscalationProtocol tests (S21)
# ---------------------------------------------------------------------------


class TestEscalationProtocol:
    """S21: Escalation of governance issues."""

    def test_escalate_failed_checkpoint(self, tmp_path):
        from src.governance.compliance import (
            ComplianceGate,
            EscalationProtocol,
        )

        gate = ComplianceGate()
        result = gate.check("data_loading", {"n_teams": 10})
        assert not result.passed

        protocol = EscalationProtocol(
            log_path=str(tmp_path / "escalations.jsonl")
        )
        escalation = protocol.escalate(result)
        assert escalation.level == "ml_lead"  # ERROR → ml_lead
        assert "data_loading" in escalation.stage_name

    def test_critical_escalates_to_operator(self, tmp_path):
        from src.governance.compliance import (
            ComplianceGate,
            EscalationProtocol,
        )

        gate = ComplianceGate()
        result = gate.check("model_training", {
            "brier_score": 0.21,
            "has_leakage": True,  # CRITICAL
        })

        protocol = EscalationProtocol(
            log_path=str(tmp_path / "escalations.jsonl")
        )
        escalation = protocol.escalate(result)
        assert escalation.level == "operator"

    def test_resolve_escalation(self, tmp_path):
        from src.governance.compliance import (
            ComplianceGate,
            EscalationProtocol,
        )

        gate = ComplianceGate()
        result = gate.check("data_loading", {"n_teams": 10})

        protocol = EscalationProtocol(
            log_path=str(tmp_path / "escalations.jsonl")
        )
        escalation = protocol.escalate(result)

        resolved = protocol.resolve(
            escalation.escalation_id,
            resolved_by="alice",
            resolution="Loaded additional teams from backup source",
        )
        assert resolved.resolved
        assert resolved.resolved_by == "alice"

    def test_pending_escalations(self, tmp_path):
        from src.governance.compliance import (
            ComplianceGate,
            EscalationProtocol,
        )

        gate = ComplianceGate()
        protocol = EscalationProtocol(
            log_path=str(tmp_path / "escalations.jsonl")
        )

        result1 = gate.check("data_loading", {"n_teams": 10})
        protocol.escalate(result1)
        result2 = gate.check("model_training", {"brier_score": 0.9, "has_leakage": False})
        protocol.escalate(result2)

        pending = protocol.pending_escalations()
        assert len(pending) == 2

    def test_explicit_level_override(self, tmp_path):
        from src.governance.compliance import (
            ComplianceGate,
            EscalationProtocol,
        )

        gate = ComplianceGate()
        result = gate.check("data_loading", {"n_teams": 10})

        protocol = EscalationProtocol(
            log_path=str(tmp_path / "escalations.jsonl")
        )
        escalation = protocol.escalate(result, level="emergency")
        assert escalation.level == "emergency"

    def test_query_by_level(self, tmp_path):
        from src.governance.compliance import (
            CheckpointResult,
            ComplianceCheck,
            CheckpointSeverity,
            EscalationProtocol,
        )

        protocol = EscalationProtocol(
            log_path=str(tmp_path / "escalations.jsonl")
        )

        # Create two escalations at different levels
        cp1 = CheckpointResult(stage_name="stage1")
        cp1.checks = [ComplianceCheck(
            check_name="check1", passed=False,
            severity=CheckpointSeverity.ERROR,
        )]
        protocol.escalate(cp1)  # → ml_lead

        cp2 = CheckpointResult(stage_name="stage2")
        cp2.checks = [ComplianceCheck(
            check_name="check2", passed=False,
            severity=CheckpointSeverity.CRITICAL,
        )]
        protocol.escalate(cp2)  # → operator

        ml_lead_issues = protocol.query(level="ml_lead")
        assert len(ml_lead_issues) >= 1

        operator_issues = protocol.query(level="operator")
        assert len(operator_issues) >= 1
