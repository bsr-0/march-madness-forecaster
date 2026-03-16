"""Comprehensive tests for all agent files targeting maximum coverage.

Tests AuditAgent, DataAgent, FeatureAgent, ModelAgent, ResearchOrchestrator,
AgentRegistry, and MessageBus.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.base import AgentMessage, AgentStatus
from src.agents.audit_agent import AuditAgent
from src.agents.data_agent import DataAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.model_agent import ModelAgent
from src.agents.orchestrator import ResearchOrchestrator
from src.agents.registry import AgentRegistry, MessageBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msg(
    sender="test",
    recipient="target",
    msg_type="request",
    payload=None,
    correlation_id="abc123",
):
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        msg_type=msg_type,
        payload=payload or {},
        correlation_id=correlation_id,
    )


# ===========================================================================
# AuditAgent tests
# ===========================================================================


class TestAuditAgent:
    """Tests for src.agents.audit_agent.AuditAgent."""

    def test_init_default_threshold(self):
        agent = AuditAgent()
        assert agent.brier_threshold == 0.25

    def test_init_custom_threshold(self):
        agent = AuditAgent(brier_threshold=0.10)
        assert agent.brier_threshold == 0.10

    def test_name_property(self):
        assert AuditAgent().name == "audit_agent"

    def test_capabilities(self):
        caps = AuditAgent().capabilities()
        assert "run_ablation" in caps
        assert "rdof_audit" in caps
        assert "check_drift" in caps
        assert "validate_predictions" in caps

    def test_initial_status_idle(self):
        assert AuditAgent().status() == AgentStatus.IDLE

    # -- handle: run_ablation --

    def test_handle_run_ablation(self):
        agent = AuditAgent()
        msg = _make_msg(payload={"action": "run_ablation"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "ablation_complete"
        assert agent.status() == AgentStatus.COMPLETED

    # -- handle: rdof_audit --

    def test_handle_rdof_audit(self):
        agent = AuditAgent()
        msg = _make_msg(payload={"action": "rdof_audit"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "audit_passed"
        assert agent.status() == AgentStatus.COMPLETED

    # -- handle: check_drift --

    def test_handle_check_drift_no_ctx(self):
        agent = AuditAgent()
        msg = _make_msg(payload={"action": "check_drift"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "no_drift_detected"
        assert resp.payload["delegated"] is False

    def test_handle_check_drift_with_ctx_features(self):
        agent = AuditAgent()
        ctx = SimpleNamespace(team_features={"t1": [1.0, 2.0], "t2": [3.0, 4.0]})
        msg = _make_msg(payload={"action": "check_drift"})
        # DriftAlertManager import will likely fail, so we hit the except branch
        resp = agent.handle(msg, ctx=ctx)
        assert resp.msg_type == "result"
        # Either drift_checked (if import succeeds) or no_drift_detected (if not)
        assert resp.payload["action"] == "check_drift"

    def test_do_check_drift_with_mock_drift_manager(self):
        """Force the DriftAlertManager import to succeed via mock."""
        agent = AuditAgent()
        ctx = SimpleNamespace(team_features={"t1": [1.0, 2.0], "t2": [3.0, 4.0]})
        mock_manager_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {"src.deployment.drift_alerts": MagicMock(DriftAlertManager=mock_manager_cls)},
        ):
            result = agent._do_check_drift(ctx, {"action": "check_drift"})
        assert result["status"] == "drift_checked"
        assert result["delegated"] is True
        assert result["feature_count"] == 2

    # -- handle: validate_predictions --

    def test_handle_validate_predictions_default_action(self):
        """When no action is specified, defaults to validate_predictions."""
        agent = AuditAgent()
        msg = _make_msg(payload={})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload.get("checks_passed") is True
        assert resp.payload["status"] == "predictions_valid"

    def test_validate_predictions_force_veto(self):
        agent = AuditAgent()
        msg = _make_msg(
            payload={
                "action": "validate_predictions",
                "upstream_results": {"force_veto": True},
            }
        )
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert resp.payload["veto"] is True
        assert "Forced veto" in resp.payload["reason"]

    def test_validate_predictions_brier_exceeds_threshold(self):
        agent = AuditAgent(brier_threshold=0.20)
        msg = _make_msg(
            payload={
                "action": "validate_predictions",
                "upstream_results": {
                    "model_agent": {
                        "baseline_stats": {"loyo_brier": 0.30},
                    }
                },
            }
        )
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert resp.payload["veto"] is True
        assert "Brier score" in resp.payload["reason"]

    def test_validate_predictions_brier_below_threshold(self):
        agent = AuditAgent(brier_threshold=0.50)
        msg = _make_msg(
            payload={
                "action": "validate_predictions",
                "upstream_results": {
                    "model_agent": {
                        "baseline_stats": {"loyo_brier": 0.20},
                    }
                },
            }
        )
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["checks_passed"] is True

    def test_validate_predictions_brier_score_fallback_key(self):
        """Uses brier_score when loyo_brier is absent."""
        agent = AuditAgent(brier_threshold=0.10)
        msg = _make_msg(
            payload={
                "action": "validate_predictions",
                "upstream_results": {
                    "model_agent": {
                        "baseline_stats": {"brier_score": 0.30},
                    }
                },
            }
        )
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["veto"] is True

    def test_validate_predictions_nan_features(self):
        agent = AuditAgent()
        ctx = SimpleNamespace(
            team_features={"t1": [1.0, float("nan")], "t2": [3.0, 4.0]},
            team_struct={},
        )
        msg = _make_msg(payload={"action": "validate_predictions", "upstream_results": {}})
        resp = agent.handle(msg, ctx=ctx)
        assert resp.msg_type == "error"
        assert resp.payload["veto"] is True
        assert "NaN" in resp.payload["reason"]

    def test_validate_predictions_missing_features(self):
        """Fewer feature vectors than teams triggers a veto."""
        agent = AuditAgent()
        ctx = SimpleNamespace(
            team_features={"t1": [1.0, 2.0]},
            team_struct={"t1": "Team1", "t2": "Team2", "t3": "Team3"},
        )
        msg = _make_msg(payload={"action": "validate_predictions", "upstream_results": {}})
        resp = agent.handle(msg, ctx=ctx)
        assert resp.payload["veto"] is True
        assert "feature vectors" in resp.payload["reason"]

    def test_validate_predictions_all_valid_with_ctx(self):
        agent = AuditAgent()
        ctx = SimpleNamespace(
            team_features={"t1": [1.0, 2.0], "t2": [3.0, 4.0]},
            team_struct={"t1": "Team1", "t2": "Team2"},
        )
        msg = _make_msg(payload={"action": "validate_predictions", "upstream_results": {}})
        resp = agent.handle(msg, ctx=ctx)
        assert resp.msg_type == "result"
        assert resp.payload["checks_passed"] is True

    # -- handle: unknown action --

    def test_handle_unknown_action(self):
        agent = AuditAgent()
        msg = _make_msg(payload={"action": "does_not_exist"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert "Unknown action" in resp.payload["error"]
        assert agent.status() == AgentStatus.FAILED

    # -- handle: exception path --

    def test_handle_exception_sets_failed(self):
        agent = AuditAgent()
        msg = _make_msg(payload={"action": "check_drift"})
        # Make _do_check_drift raise
        with patch.object(agent, "_do_check_drift", side_effect=RuntimeError("boom")):
            resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert "boom" in resp.payload["error"]
        assert agent.status() == AgentStatus.FAILED

    def test_response_correlation_id_matches(self):
        agent = AuditAgent()
        msg = _make_msg(payload={"action": "run_ablation"}, correlation_id="xyz789")
        resp = agent.handle(msg, ctx=None)
        assert resp.correlation_id == "xyz789"

    def test_response_sender_recipient(self):
        agent = AuditAgent()
        msg = _make_msg(sender="orchestrator", payload={"action": "rdof_audit"})
        resp = agent.handle(msg, ctx=None)
        assert resp.sender == "audit_agent"
        assert resp.recipient == "orchestrator"


# ===========================================================================
# DataAgent tests
# ===========================================================================


class TestDataAgent:
    """Tests for src.agents.data_agent.DataAgent."""

    def test_init(self):
        agent = DataAgent()
        assert agent.status() == AgentStatus.IDLE

    def test_name(self):
        assert DataAgent().name == "data_agent"

    def test_capabilities(self):
        caps = DataAgent().capabilities()
        assert "load_data" in caps
        assert "validate_freshness" in caps
        assert "snapshot_data" in caps

    def test_handle_load_data_no_ctx(self):
        agent = DataAgent()
        msg = _make_msg(payload={"action": "load_data"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "loaded"
        assert resp.payload["delegated"] is False
        assert agent.status() == AgentStatus.COMPLETED

    def test_handle_load_data_default_action(self):
        agent = DataAgent()
        msg = _make_msg(payload={})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["action"] == "load_data"

    def test_handle_validate_freshness_no_ctx(self):
        agent = DataAgent()
        msg = _make_msg(payload={"action": "validate_freshness"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "validated"
        assert resp.payload["delegated"] is False

    def test_handle_validate_freshness_with_ctx(self):
        agent = DataAgent()
        config = SimpleNamespace(max_feed_age_hours=48)
        ctx = SimpleNamespace(config=config)
        msg = _make_msg(payload={"action": "validate_freshness"})
        resp = agent.handle(msg, ctx=ctx)
        assert resp.payload["delegated"] is True
        assert resp.payload["max_feed_age_hours"] == 48

    def test_handle_validate_freshness_ctx_no_max_feed_age(self):
        agent = DataAgent()
        config = SimpleNamespace()  # no max_feed_age_hours attr
        ctx = SimpleNamespace(config=config)
        msg = _make_msg(payload={"action": "validate_freshness"})
        resp = agent.handle(msg, ctx=ctx)
        assert resp.payload["max_feed_age_hours"] == 168  # default

    def test_handle_snapshot_data(self):
        agent = DataAgent()
        msg = _make_msg(payload={"action": "snapshot_data"})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["status"] == "snapshot_created"

    def test_handle_unknown_action(self):
        agent = DataAgent()
        msg = _make_msg(payload={"action": "nope"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert "Unknown action" in resp.payload["error"]
        assert agent.status() == AgentStatus.FAILED

    def test_handle_exception(self):
        agent = DataAgent()
        msg = _make_msg(payload={"action": "load_data"})
        with patch.object(agent, "_do_load_data", side_effect=ValueError("bad data")):
            resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert "bad data" in resp.payload["error"]
        assert agent.status() == AgentStatus.FAILED

    def test_status_returns_current(self):
        agent = DataAgent()
        assert agent.status() == AgentStatus.IDLE
        msg = _make_msg(payload={"action": "snapshot_data"})
        agent.handle(msg, ctx=None)
        assert agent.status() == AgentStatus.COMPLETED


# ===========================================================================
# FeatureAgent tests
# ===========================================================================


class TestFeatureAgent:
    """Tests for src.agents.feature_agent.FeatureAgent."""

    def test_init(self):
        agent = FeatureAgent()
        assert agent.status() == AgentStatus.IDLE

    def test_name(self):
        assert FeatureAgent().name == "feature_agent"

    def test_capabilities(self):
        caps = FeatureAgent().capabilities()
        assert "engineer_features" in caps
        assert "select_features" in caps
        assert "compute_drift" in caps

    def test_handle_engineer_features_no_ctx(self):
        agent = FeatureAgent()
        msg = _make_msg(payload={"action": "engineer_features"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "engineered"
        assert resp.payload["delegated"] is False

    def test_handle_engineer_features_default_action(self):
        agent = FeatureAgent()
        msg = _make_msg(payload={})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["action"] == "engineer_features"

    def test_handle_engineer_features_empty_teams(self):
        """When upstream provides empty teams list, returns not delegated."""
        agent = FeatureAgent()
        ctx = SimpleNamespace(feature_engineer=MagicMock())
        msg = _make_msg(
            payload={
                "action": "engineer_features",
                "upstream_results": {"data_agent": {"teams": []}},
            }
        )
        resp = agent.handle(msg, ctx=ctx)
        assert resp.payload["delegated"] is False
        assert resp.payload.get("reason") == "no_upstream_teams"

    def test_handle_select_features(self):
        agent = FeatureAgent()
        msg = _make_msg(payload={"action": "select_features"})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["status"] == "selected"

    def test_handle_compute_drift(self):
        agent = FeatureAgent()
        msg = _make_msg(payload={"action": "compute_drift"})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["status"] == "drift_computed"

    def test_handle_unknown_action(self):
        agent = FeatureAgent()
        msg = _make_msg(payload={"action": "bad_action"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert "Unknown action" in resp.payload["error"]
        assert agent.status() == AgentStatus.FAILED

    def test_handle_exception(self):
        agent = FeatureAgent()
        msg = _make_msg(payload={"action": "engineer_features"})
        with patch.object(agent, "_do_engineer_features", side_effect=TypeError("oops")):
            resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert agent.status() == AgentStatus.FAILED

    def test_status(self):
        agent = FeatureAgent()
        assert agent.status() == AgentStatus.IDLE
        msg = _make_msg(payload={"action": "select_features"})
        agent.handle(msg, ctx=None)
        assert agent.status() == AgentStatus.COMPLETED


# ===========================================================================
# ModelAgent tests
# ===========================================================================


class TestModelAgent:
    """Tests for src.agents.model_agent.ModelAgent."""

    def test_init(self):
        agent = ModelAgent()
        assert agent.status() == AgentStatus.IDLE

    def test_name(self):
        assert ModelAgent().name == "model_agent"

    def test_capabilities(self):
        caps = ModelAgent().capabilities()
        assert "train_models" in caps
        assert "calibrate" in caps
        assert "predict" in caps

    def test_handle_train_models_no_ctx(self):
        agent = ModelAgent()
        msg = _make_msg(payload={"action": "train_models"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "trained"
        assert resp.payload["delegated"] is False

    def test_handle_train_models_default_action(self):
        agent = ModelAgent()
        msg = _make_msg(payload={})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["action"] == "train_models"

    def test_handle_train_models_no_game_flows(self):
        """ctx exists but upstream has no game_flows."""
        agent = ModelAgent()
        ctx = SimpleNamespace(_train_baseline_model=MagicMock())
        msg = _make_msg(
            payload={
                "action": "train_models",
                "upstream_results": {"data_agent": {"game_flows": {}}},
            }
        )
        resp = agent.handle(msg, ctx=ctx)
        assert resp.payload["delegated"] is False
        assert resp.payload.get("reason") == "no_game_flows"

    def test_handle_calibrate_no_ctx(self):
        agent = ModelAgent()
        msg = _make_msg(payload={"action": "calibrate"})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["status"] == "calibrated"
        assert resp.payload["delegated"] is False

    def test_handle_predict(self):
        agent = ModelAgent()
        msg = _make_msg(payload={"action": "predict"})
        resp = agent.handle(msg, ctx=None)
        assert resp.payload["status"] == "predicted"

    def test_handle_unknown_action(self):
        agent = ModelAgent()
        msg = _make_msg(payload={"action": "explode"})
        resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert "Unknown action" in resp.payload["error"]
        assert agent.status() == AgentStatus.FAILED

    def test_handle_exception(self):
        agent = ModelAgent()
        msg = _make_msg(payload={"action": "train_models"})
        with patch.object(agent, "_do_train_models", side_effect=RuntimeError("gpu error")):
            resp = agent.handle(msg, ctx=None)
        assert resp.msg_type == "error"
        assert agent.status() == AgentStatus.FAILED

    def test_status(self):
        agent = ModelAgent()
        assert agent.status() == AgentStatus.IDLE
        agent.handle(_make_msg(payload={"action": "predict"}), ctx=None)
        assert agent.status() == AgentStatus.COMPLETED


# ===========================================================================
# AgentRegistry and MessageBus tests
# ===========================================================================


class TestAgentRegistry:
    """Tests for src.agents.registry.AgentRegistry."""

    def test_register_and_get(self):
        reg = AgentRegistry()
        agent = DataAgent()
        reg.register(agent)
        assert reg.get("data_agent") is agent

    def test_get_missing_raises(self):
        reg = AgentRegistry()
        with pytest.raises(KeyError, match="Agent not found"):
            reg.get("nonexistent")

    def test_list_agents_sorted(self):
        reg = AgentRegistry()
        reg.register(ModelAgent())
        reg.register(AuditAgent())
        reg.register(DataAgent())
        names = reg.list_agents()
        assert names == sorted(names)

    def test_find_by_capability(self):
        reg = AgentRegistry()
        reg.register(AuditAgent())
        reg.register(DataAgent())
        found = reg.find_by_capability("load_data")
        assert len(found) == 1
        assert found[0].name == "data_agent"

    def test_find_by_capability_no_match(self):
        reg = AgentRegistry()
        reg.register(DataAgent())
        assert reg.find_by_capability("nonexistent_cap") == []

    def test_health_check(self):
        reg = AgentRegistry()
        reg.register(DataAgent())
        reg.register(AuditAgent())
        health = reg.health_check()
        assert health["data_agent"] == "idle"
        assert health["audit_agent"] == "idle"

    def test_register_overwrites(self):
        reg = AgentRegistry()
        a1 = DataAgent()
        a2 = DataAgent()
        reg.register(a1)
        reg.register(a2)
        assert reg.get("data_agent") is a2


class TestMessageBus:
    """Tests for src.agents.registry.MessageBus."""

    def test_send(self):
        reg = AgentRegistry()
        reg.register(DataAgent())
        bus = MessageBus(reg)
        msg = _make_msg(recipient="data_agent", payload={"action": "load_data"})
        resp = bus.send(msg)
        assert resp.msg_type == "result"
        assert resp.payload["status"] == "loaded"

    def test_send_records_history(self):
        reg = AgentRegistry()
        reg.register(DataAgent())
        bus = MessageBus(reg)
        msg = _make_msg(recipient="data_agent", payload={"action": "snapshot_data"})
        bus.send(msg)
        hist = bus.history()
        assert len(hist) == 2  # request + response

    def test_broadcast(self):
        reg = AgentRegistry()
        reg.register(DataAgent())
        reg.register(AuditAgent())
        bus = MessageBus(reg)
        msg = _make_msg(
            recipient="*",
            payload={"action": "run_ablation"},  # audit handles this; data defaults to load_data
        )
        responses = bus.broadcast(msg)
        assert len(responses) == 2

    def test_send_with_ctx(self):
        reg = AgentRegistry()
        reg.register(DataAgent())
        bus = MessageBus(reg)
        config = SimpleNamespace(max_feed_age_hours=24)
        ctx = SimpleNamespace(config=config)
        msg = _make_msg(recipient="data_agent", payload={"action": "validate_freshness"})
        resp = bus.send(msg, ctx=ctx)
        assert resp.payload["delegated"] is True


# ===========================================================================
# ResearchOrchestrator tests
# ===========================================================================


class TestResearchOrchestrator:
    """Tests for src.agents.orchestrator.ResearchOrchestrator."""

    @staticmethod
    def _build_orchestrator(ctx=None, max_retries=0, retry_delay=0.0):
        reg = AgentRegistry()
        reg.register(DataAgent())
        reg.register(FeatureAgent())
        reg.register(ModelAgent())
        reg.register(AuditAgent())
        bus = MessageBus(reg)
        return ResearchOrchestrator(
            registry=reg,
            bus=bus,
            ctx=ctx,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay,
        )

    def test_run_pipeline_success(self):
        orch = self._build_orchestrator()
        result = orch.run_pipeline()
        assert result["status"] == "completed"
        assert "results" in result
        assert "data_agent" in result["results"]
        assert "feature_agent" in result["results"]
        assert "model_agent" in result["results"]
        assert "audit_agent" in result["results"]

    def test_run_pipeline_veto(self):
        """When audit agent vetoes, pipeline returns vetoed status."""
        reg = AgentRegistry()
        reg.register(DataAgent())
        reg.register(FeatureAgent())
        reg.register(ModelAgent())
        # Use audit agent with very low threshold so brier score causes veto
        audit = AuditAgent(brier_threshold=0.001)
        reg.register(audit)
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(registry=reg, bus=bus, max_retries=0, retry_delay_seconds=0)

        # The orchestrator uses capabilities()[0] as the action. For AuditAgent that
        # is "run_ablation", not "validate_predictions". We also need the model_agent
        # to return a brier score. Patch both: audit capabilities and model handle.
        original_model_handle = ModelAgent.handle

        def patched_model_handle(self, message, ctx):
            resp = original_model_handle(self, message, ctx)
            resp.payload["baseline_stats"] = {"loyo_brier": 0.50}
            return resp

        with patch.object(ModelAgent, "handle", patched_model_handle), \
             patch.object(AuditAgent, "capabilities",
                          return_value=["validate_predictions", "run_ablation",
                                        "rdof_audit", "check_drift"]):
            result = orch.run_pipeline()
        assert result["status"] == "vetoed"
        assert result["stage"] == "audit_agent"

    def test_run_pipeline_error_no_retry(self):
        """Non-veto error halts pipeline."""
        reg = AgentRegistry()
        # Only register a data_agent that will fail
        data_agent = DataAgent()
        reg.register(data_agent)
        reg.register(FeatureAgent())
        reg.register(ModelAgent())
        reg.register(AuditAgent())
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(
            registry=reg, bus=bus, max_retries=0, retry_delay_seconds=0
        )

        with patch.object(
            DataAgent,
            "handle",
            return_value=AgentMessage(
                sender="data_agent",
                recipient="orchestrator",
                msg_type="error",
                payload={"error": "disk full"},
            ),
        ):
            result = orch.run_pipeline()
        assert result["status"] == "failed"
        assert result["stage"] == "data_agent"
        assert "disk full" in result["error"]

    def test_run_pipeline_missing_agent_skipped(self):
        """If an agent is not registered, its stage is skipped."""
        reg = AgentRegistry()
        reg.register(DataAgent())
        # Skip feature_agent, model_agent, audit_agent
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(registry=reg, bus=bus, max_retries=0, retry_delay_seconds=0)
        result = orch.run_pipeline()
        assert result["status"] == "completed"
        assert "data_agent" in result["results"]
        assert "feature_agent" not in result["results"]

    def test_run_with_retry_succeeds_on_retry(self):
        """Error on first attempt, success on second."""
        reg = AgentRegistry()
        reg.register(DataAgent())
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(
            registry=reg, bus=bus, max_retries=1, retry_delay_seconds=0.0
        )

        call_count = 0
        original_send = bus.send

        def flaky_send(msg, ctx=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AgentMessage(
                    sender="data_agent",
                    recipient="orchestrator",
                    msg_type="error",
                    payload={"error": "transient"},
                )
            return original_send(msg, ctx)

        bus.send = flaky_send
        msg = _make_msg(recipient="data_agent", payload={"action": "load_data"})
        resp = orch._run_with_retry(msg, "data_agent")
        assert resp.msg_type == "result"
        assert call_count == 2

    def test_run_with_retry_veto_not_retried(self):
        """Veto errors are returned immediately, no retry."""
        reg = AgentRegistry()
        reg.register(AuditAgent())
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(
            registry=reg, bus=bus, max_retries=3, retry_delay_seconds=0.0
        )

        call_count = 0
        original_send = bus.send

        def veto_send(msg, ctx=None):
            nonlocal call_count
            call_count += 1
            return AgentMessage(
                sender="audit_agent",
                recipient="orchestrator",
                msg_type="error",
                payload={"veto": True, "reason": "bad predictions"},
            )

        bus.send = veto_send
        msg = _make_msg(recipient="audit_agent", payload={"action": "validate_predictions"})
        resp = orch._run_with_retry(msg, "audit_agent")
        assert resp.payload["veto"] is True
        assert call_count == 1  # no retries for veto

    def test_run_with_retry_exhausted(self):
        """All retries fail, returns last error."""
        reg = AgentRegistry()
        reg.register(DataAgent())
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(
            registry=reg, bus=bus, max_retries=2, retry_delay_seconds=0.0
        )

        call_count = 0

        def always_fail(msg, ctx=None):
            nonlocal call_count
            call_count += 1
            return AgentMessage(
                sender="data_agent",
                recipient="orchestrator",
                msg_type="error",
                payload={"error": f"fail #{call_count}"},
            )

        bus.send = always_fail
        msg = _make_msg(recipient="data_agent", payload={"action": "load_data"})
        resp = orch._run_with_retry(msg, "data_agent")
        assert resp.msg_type == "error"
        assert call_count == 3  # 1 initial + 2 retries

    def test_log_health_status(self):
        """_log_health_status runs without error and checks all agents."""
        orch = self._build_orchestrator()
        # Should not raise
        orch._log_health_status("data_agent")

    def test_log_health_status_with_missing_agent(self):
        """Registry with an agent whose status() raises still produces output."""
        reg = AgentRegistry()
        broken = MagicMock()
        broken.name = "broken_agent"
        broken.capabilities.return_value = ["nothing"]
        broken.status.side_effect = AttributeError("no status")
        reg.register(broken)
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(registry=reg, bus=bus)
        # Should handle the AttributeError gracefully
        orch._log_health_status("some_stage")

    def test_run_experiment_success(self):
        orch = self._build_orchestrator()
        result = orch.run_experiment(hypothesis="seed matters")
        assert result["status"] == "completed"
        assert result["hypothesis"] == "seed matters"
        assert "results" in result

    def test_run_experiment_with_overrides(self):
        orch = self._build_orchestrator()
        result = orch.run_experiment(
            hypothesis="test hypo",
            config_overrides={"learning_rate": 0.01},
        )
        assert result["status"] == "completed"
        assert result["hypothesis"] == "test hypo"

    def test_run_experiment_veto(self):
        """Experiment pipeline can also be vetoed."""
        reg = AgentRegistry()
        reg.register(DataAgent())
        reg.register(FeatureAgent())
        reg.register(ModelAgent())
        audit = AuditAgent(brier_threshold=0.001)
        reg.register(audit)
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(
            registry=reg, bus=bus, max_retries=0, retry_delay_seconds=0
        )

        original_handle = ModelAgent.handle

        def patched_handle(self, message, ctx):
            resp = original_handle(self, message, ctx)
            resp.payload["baseline_stats"] = {"loyo_brier": 0.50}
            return resp

        with patch.object(ModelAgent, "handle", patched_handle), \
             patch.object(AuditAgent, "capabilities",
                          return_value=["validate_predictions", "run_ablation",
                                        "rdof_audit", "check_drift"]):
            result = orch.run_experiment(hypothesis="veto test")
        assert result["status"] == "vetoed"
        assert result["hypothesis"] == "veto test"

    def test_run_experiment_missing_agents(self):
        """Experiment still completes if some agents are missing."""
        reg = AgentRegistry()
        reg.register(DataAgent())
        bus = MessageBus(reg)
        orch = ResearchOrchestrator(registry=reg, bus=bus, max_retries=0, retry_delay_seconds=0)
        result = orch.run_experiment(hypothesis="partial")
        assert result["status"] == "completed"


# ===========================================================================
# AgentMessage tests
# ===========================================================================


class TestAgentMessage:
    """Tests for the AgentMessage dataclass."""

    def test_defaults(self):
        msg = AgentMessage(sender="a", recipient="b", msg_type="request")
        assert msg.payload == {}
        assert msg.timestamp  # non-empty
        assert msg.correlation_id  # non-empty

    def test_custom_fields(self):
        msg = AgentMessage(
            sender="s",
            recipient="r",
            msg_type="result",
            payload={"key": "val"},
            correlation_id="custom",
        )
        assert msg.payload == {"key": "val"}
        assert msg.correlation_id == "custom"


# ===========================================================================
# AgentStatus tests
# ===========================================================================


class TestAgentStatus:
    """Tests for the AgentStatus enum."""

    def test_values(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"

    def test_members(self):
        assert len(AgentStatus) == 4
