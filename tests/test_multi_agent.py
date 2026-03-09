"""Tests for multi-agent architecture (S2/S22 compliance).

Covers:
  - Agent protocol and base class
  - MessageBus pub/sub communication
  - ConflictResolver detection and resolution hierarchy
  - DissentRegistry append-only persistence
  - AuditAgent veto power (S22.3)
  - OrchestratorAgent coordination
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# MessageBus tests
# ---------------------------------------------------------------------------


class TestMessageBus:
    """S2: Inter-agent communication via message bus."""

    def test_publish_and_retrieve(self):
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType

        bus = MessageBus()
        msg = AgentMessage(
            sender=AgentRole.DATA_SCOUT,
            msg_type=MessageType.DATA_READY,
            payload={"n_teams": 68},
        )
        bus.publish(msg)
        assert len(bus) == 1
        assert bus.all_messages()[0].payload["n_teams"] == 68

    def test_filter_by_topic(self):
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType

        bus = MessageBus()
        bus.publish(AgentMessage(
            sender=AgentRole.DATA_SCOUT,
            msg_type=MessageType.DATA_READY,
        ))
        bus.publish(AgentMessage(
            sender=AgentRole.MODELER,
            msg_type=MessageType.MODEL_READY,
        ))
        bus.publish(AgentMessage(
            sender=AgentRole.AUDITOR,
            msg_type=MessageType.AUDIT_RESULT,
        ))

        data_msgs = bus.get_messages(topic=MessageType.DATA_READY)
        assert len(data_msgs) == 1
        assert data_msgs[0].sender == AgentRole.DATA_SCOUT

    def test_filter_by_sender(self):
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType

        bus = MessageBus()
        bus.publish(AgentMessage(
            sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS,
        ))
        bus.publish(AgentMessage(
            sender=AgentRole.MODELER, msg_type=MessageType.STATUS,
        ))

        msgs = bus.get_messages(sender=AgentRole.DATA_SCOUT)
        assert len(msgs) == 1

    def test_has_veto_false_when_no_veto(self):
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType

        bus = MessageBus()
        bus.publish(AgentMessage(
            sender=AgentRole.DATA_SCOUT,
            msg_type=MessageType.DATA_READY,
        ))
        assert not bus.has_veto()

    def test_has_veto_true_when_veto_published(self):
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType

        bus = MessageBus()
        bus.publish(AgentMessage(
            sender=AgentRole.AUDITOR,
            msg_type=MessageType.VETO,
            payload={"reason": "leakage detected"},
        ))
        assert bus.has_veto()

    def test_clear(self):
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType

        bus = MessageBus()
        bus.publish(AgentMessage(
            sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS,
        ))
        assert len(bus) == 1
        bus.clear()
        assert len(bus) == 0


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


class TestBaseAgent:
    """S2: Agent protocol and base class."""

    def test_subclass_instantiation(self):
        from src.agents import AgentResult, AgentRole, BaseAgent, MessageBus

        class TestAgent(BaseAgent):
            @property
            def role(self):
                return AgentRole.DATA_SCOUT

            @property
            def name(self):
                return "TestAgent"

            def run(self, ctx, bus, **kwargs):
                return AgentResult(agent_role=self.role, success=True)

        agent = TestAgent()
        assert agent.role == AgentRole.DATA_SCOUT
        assert agent.name == "TestAgent"

    def test_publish_helper(self):
        from src.agents import AgentRole, BaseAgent, AgentResult, MessageBus, MessageType

        class TestAgent(BaseAgent):
            @property
            def role(self):
                return AgentRole.MODELER

            @property
            def name(self):
                return "Test"

            def run(self, ctx, bus, **kwargs):
                self.publish(bus, MessageType.STATUS, {"msg": "hello"})
                return AgentResult(agent_role=self.role)

        agent = TestAgent()
        bus = MessageBus()
        agent.run(None, bus)
        assert len(bus) == 1
        assert bus.all_messages()[0].sender == AgentRole.MODELER
        assert bus.all_messages()[0].payload["msg"] == "hello"

    def test_log_finding(self):
        from src.agents import AgentRole, BaseAgent, AgentResult, MessageBus, MessageType

        class TestAgent(BaseAgent):
            @property
            def role(self):
                return AgentRole.AUDITOR

            @property
            def name(self):
                return "Test"

            def run(self, ctx, bus, **kwargs):
                return AgentResult(agent_role=self.role)

        agent = TestAgent()
        bus = MessageBus()
        agent.log_finding(bus, "Something notable happened")
        msgs = bus.get_messages(topic=MessageType.STATUS)
        assert len(msgs) == 1
        assert msgs[0].payload["finding"] == "Something notable happened"

    def test_agent_result_fields(self):
        from src.agents import AgentResult, AgentRole

        result = AgentResult(
            agent_role=AgentRole.DATA_SCOUT,
            success=True,
            output={"data": [1, 2, 3]},
            findings=["loaded 3 items"],
            messages_sent=2,
            wall_seconds=1.5,
        )
        assert result.success
        assert result.messages_sent == 2
        assert len(result.findings) == 1


# ---------------------------------------------------------------------------
# ConflictResolver tests
# ---------------------------------------------------------------------------


class TestConflictResolver:
    """S22: Conflict detection and resolution."""

    def test_detect_safety_conflict_from_veto(self):
        from src.agents import AgentMessage, AgentRole, MessageType
        from src.agents.conflict import ConflictCategory, ConflictResolver

        resolver = ConflictResolver()
        messages = [
            AgentMessage(
                sender=AgentRole.MODELER,
                msg_type=MessageType.MODEL_READY,
                payload={"brier_score": 0.21},
            ),
            AgentMessage(
                sender=AgentRole.AUDITOR,
                msg_type=MessageType.VETO,
                payload={"reason": "leakage detected"},
            ),
        ]
        conflicts = resolver.detect_conflicts(messages)
        assert len(conflicts) == 1
        assert conflicts[0].category == ConflictCategory.SAFETY

    def test_safety_conflict_resolved_by_audit_veto(self):
        from src.agents import AgentMessage, AgentRole, MessageType
        from src.agents.conflict import Conflict, ConflictCategory, ConflictResolver

        resolver = ConflictResolver()
        conflict = Conflict(
            category=ConflictCategory.SAFETY,
            agents_involved=["auditor", "modeler"],
            description="Audit veto vs modeler success",
        )
        resolved = resolver.resolve(conflict, [])
        assert resolved.resolved_by == AgentRole.AUDITOR.value
        assert "veto upheld" in resolved.resolution

    def test_method_conflict_resolved_by_evidence(self):
        from src.agents import AgentMessage, AgentRole, MessageType
        from src.agents.conflict import Conflict, ConflictCategory, ConflictResolver

        resolver = ConflictResolver()
        conflict = Conflict(
            category=ConflictCategory.METHOD,
            agents_involved=["modeler", "feature_engineer"],
            description="Disagreement on feature set",
        )
        messages = [
            AgentMessage(
                sender=AgentRole.MODELER,
                msg_type=MessageType.MODEL_READY,
                payload={"brier_score": 0.20},
            ),
            AgentMessage(
                sender=AgentRole.FEATURE_ENGINEER,
                msg_type=MessageType.FEATURES_READY,
                payload={"brier_score": 0.22},
            ),
        ]
        resolved = resolver.resolve(conflict, messages)
        assert resolved.resolved_by == "empirical_evidence"
        assert "modeler" in resolved.resolution  # Lower Brier wins

    def test_check_audit_veto_returns_veto_message(self):
        from src.agents import AgentMessage, AgentRole, MessageType
        from src.agents.conflict import ConflictResolver

        resolver = ConflictResolver()
        messages = [
            AgentMessage(
                sender=AgentRole.DATA_SCOUT,
                msg_type=MessageType.DATA_READY,
            ),
            AgentMessage(
                sender=AgentRole.AUDITOR,
                msg_type=MessageType.VETO,
                payload={"reason": "contamination"},
            ),
        ]
        veto = resolver.check_audit_veto(messages)
        assert veto is not None
        assert veto.sender == AgentRole.AUDITOR
        assert veto.payload["reason"] == "contamination"

    def test_no_veto_returns_none(self):
        from src.agents import AgentMessage, AgentRole, MessageType
        from src.agents.conflict import ConflictResolver

        resolver = ConflictResolver()
        messages = [
            AgentMessage(
                sender=AgentRole.DATA_SCOUT,
                msg_type=MessageType.DATA_READY,
            ),
        ]
        assert resolver.check_audit_veto(messages) is None

    def test_no_conflicts_in_clean_messages(self):
        from src.agents import AgentMessage, AgentRole, MessageType
        from src.agents.conflict import ConflictResolver

        resolver = ConflictResolver()
        messages = [
            AgentMessage(
                sender=AgentRole.DATA_SCOUT,
                msg_type=MessageType.DATA_READY,
            ),
            AgentMessage(
                sender=AgentRole.AUDITOR,
                msg_type=MessageType.AUDIT_RESULT,
                payload={"passed": True},
            ),
        ]
        conflicts = resolver.detect_conflicts(messages)
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# DissentRegistry tests
# ---------------------------------------------------------------------------


class TestDissentRegistry:
    """S22.4: Agent dissent registry."""

    def test_file_and_query_dissent(self, tmp_path):
        from src.agents import AgentRole
        from src.agents.conflict import DissentRecord, DissentRegistry

        registry = DissentRegistry(path=str(tmp_path / "dissent.jsonl"))
        record = DissentRecord(
            agent_role=AgentRole.MODELER.value,
            reasoning="Audit veto threshold is too aggressive",
            evidence={"psi_threshold": 0.1},
        )
        dissent_id = registry.file_dissent(record)
        assert dissent_id

        records = registry.query()
        assert len(records) == 1
        assert records[0].reasoning == "Audit veto threshold is too aggressive"

    def test_query_by_agent(self, tmp_path):
        from src.agents import AgentRole
        from src.agents.conflict import DissentRecord, DissentRegistry

        registry = DissentRegistry(path=str(tmp_path / "dissent.jsonl"))
        registry.file_dissent(DissentRecord(
            agent_role=AgentRole.MODELER.value,
            reasoning="Reason A",
        ))
        registry.file_dissent(DissentRecord(
            agent_role=AgentRole.FEATURE_ENGINEER.value,
            reasoning="Reason B",
        ))

        modeler_records = registry.query(agent=AgentRole.MODELER.value)
        assert len(modeler_records) == 1
        assert modeler_records[0].reasoning == "Reason A"

    def test_open_dissents_filter(self, tmp_path):
        from src.agents import AgentRole
        from src.agents.conflict import DissentRecord, DissentRegistry

        registry = DissentRegistry(path=str(tmp_path / "dissent.jsonl"))
        registry.file_dissent(DissentRecord(
            agent_role=AgentRole.MODELER.value,
            reasoning="Open issue",
        ))
        open_records = registry.open_dissents()
        assert len(open_records) == 1
        assert open_records[0].status == "open"

    def test_review_marks_as_reviewed(self, tmp_path):
        from src.agents import AgentRole
        from src.agents.conflict import DissentRecord, DissentRegistry

        registry = DissentRegistry(path=str(tmp_path / "dissent.jsonl"))
        record = DissentRecord(
            agent_role=AgentRole.AUDITOR.value,
            reasoning="Threshold concern",
        )
        dissent_id = registry.file_dissent(record)

        reviewed = registry.review(dissent_id, reviewer="alice", notes="Valid concern")
        assert reviewed.status == "reviewed"
        assert reviewed.reviewer == "alice"

        # Re-query to confirm persistence
        open_records = registry.open_dissents()
        assert len(open_records) == 0


# ---------------------------------------------------------------------------
# DataScoutAgent tests
# ---------------------------------------------------------------------------


class TestDataScoutAgent:
    """S2: DataScout agent wrapping data loading."""

    def test_run_publishes_data_ready(self):
        from src.agents import MessageBus, MessageType
        from src.agents.concrete import DataScoutAgent

        agent = DataScoutAgent()
        bus = MessageBus()

        # Mock pipeline and context
        mock_ctx = MagicMock()
        mock_ctx.resource_tracker = None
        mock_ctx.phase_timer = None

        mock_pipeline = MagicMock()

        # Mock DataLoadingStage.run to avoid real data loading
        from src.pipeline.stages import LoadedData

        mock_loaded = LoadedData(
            teams=[MagicMock(team_name="Duke")],
            rosters={"Duke": {}},
        )

        with patch(
            "src.pipeline.stages.data_loader.DataLoadingStage"
        ) as MockLoader:
            MockLoader.return_value.run.return_value = mock_loaded
            result = agent.run(mock_ctx, bus, pipeline=mock_pipeline)

        assert result.success
        assert result.output is mock_loaded
        data_msgs = bus.get_messages(topic=MessageType.DATA_READY)
        assert len(data_msgs) == 1
        assert data_msgs[0].payload["n_teams"] == 1


# ---------------------------------------------------------------------------
# AuditAgent tests
# ---------------------------------------------------------------------------


class TestAuditAgent:
    """S22.3: AuditAgent with veto power."""

    def test_passes_clean_data(self):
        from src.agents import MessageBus, MessageType
        from src.agents.concrete import AuditAgent
        from src.pipeline.stages import EngineeredFeatures, TrainedModels

        agent = AuditAgent()
        bus = MessageBus()
        mock_ctx = MagicMock()

        features = EngineeredFeatures(
            feature_names=["elo", "momentum", "sos"],
        )
        models = TrainedModels()

        # No pipeline training data → robustness skipped, no leakage
        result = agent.run(
            mock_ctx, bus,
            pipeline=MagicMock(_X_train=None, _y_train=None),
            models=models,
            features=features,
        )

        assert result.success
        assert not bus.has_veto()

    def test_veto_on_leakage_detection(self):
        from src.agents import MessageBus, MessageType
        from src.agents.concrete import AuditAgent
        from src.pipeline.stages import EngineeredFeatures, TrainedModels

        agent = AuditAgent()
        bus = MessageBus()
        mock_ctx = MagicMock()

        # Features with leakage keywords
        features = EngineeredFeatures(
            feature_names=["elo", "future_wins", "tournament_outcome"],
        )
        models = TrainedModels()

        result = agent.run(
            mock_ctx, bus,
            pipeline=MagicMock(_X_train=None, _y_train=None),
            models=models,
            features=features,
        )

        assert not result.success
        assert bus.has_veto()
        veto_msgs = bus.get_messages(topic=MessageType.VETO)
        assert len(veto_msgs) >= 1
        assert "leakage" in veto_msgs[0].payload["reason"].lower()

    def test_veto_is_absolute(self):
        """S22.3: Audit veto cannot be overridden by other agents."""
        from src.agents import AgentMessage, AgentRole, MessageBus, MessageType
        from src.agents.conflict import ConflictResolver

        bus = MessageBus()

        # Modeler says "all good"
        bus.publish(AgentMessage(
            sender=AgentRole.MODELER,
            msg_type=MessageType.MODEL_READY,
            payload={"brier_score": 0.19},
        ))

        # Auditor vetoes
        bus.publish(AgentMessage(
            sender=AgentRole.AUDITOR,
            msg_type=MessageType.VETO,
            payload={"reason": "temporal leakage"},
        ))

        resolver = ConflictResolver()
        veto = resolver.check_audit_veto(bus.all_messages())
        assert veto is not None

        # Resolve conflicts — safety wins
        conflicts = resolver.detect_conflicts(bus.all_messages())
        assert len(conflicts) == 1
        resolved = resolver.resolve(conflicts[0], bus.all_messages())
        assert resolved.resolved_by == AgentRole.AUDITOR.value

    def test_audit_findings_in_result(self):
        from src.agents import MessageBus
        from src.agents.concrete import AuditAgent
        from src.pipeline.stages import EngineeredFeatures, TrainedModels

        agent = AuditAgent()
        bus = MessageBus()
        mock_ctx = MagicMock()

        features = EngineeredFeatures(feature_names=["elo", "sos"])
        result = agent.run(
            mock_ctx, bus,
            pipeline=MagicMock(_X_train=None),
            models=TrainedModels(),
            features=features,
        )
        assert len(result.findings) > 0


# ---------------------------------------------------------------------------
# OrchestratorAgent tests
# ---------------------------------------------------------------------------


class TestOrchestratorAgent:
    """S2: Orchestrator coordinates all agents."""

    def test_full_pipeline_execution(self):
        from src.agents import MessageBus, MessageType
        from src.agents.concrete import OrchestratorAgent
        from src.pipeline.stages import (
            LoadedData, EngineeredFeatures, TrainedModels,
            CalibratedPipeline, SimulationResults,
        )

        orchestrator = OrchestratorAgent()
        bus = MessageBus()
        mock_ctx = MagicMock()

        mock_pipeline = MagicMock()
        mock_pipeline._X_train = None
        mock_pipeline._y_train = None

        mock_loaded = LoadedData(teams=[MagicMock(team_name="Duke")])
        mock_features = EngineeredFeatures(
            team_features={"Duke": [1, 2, 3]},
            feature_names=["elo", "sos", "rpi"],
        )
        mock_models = TrainedModels()
        mock_calibration = CalibratedPipeline(calibration_method="isotonic")
        mock_simulation = SimulationResults(num_simulations=10000)

        with patch("src.agents.concrete.DataScoutAgent.run") as mock_ds, \
             patch("src.agents.concrete.FeatureEngineerAgent.run") as mock_fe, \
             patch("src.agents.concrete.ModelingAgent.run") as mock_mod, \
             patch("src.agents.concrete.AuditAgent.run") as mock_aud:

            from src.agents import AgentResult, AgentRole

            mock_ds.return_value = AgentResult(
                agent_role=AgentRole.DATA_SCOUT,
                success=True,
                output=mock_loaded,
            )
            mock_fe.return_value = AgentResult(
                agent_role=AgentRole.FEATURE_ENGINEER,
                success=True,
                output=mock_features,
            )
            mock_mod.return_value = AgentResult(
                agent_role=AgentRole.MODELER,
                success=True,
                output={
                    "models": mock_models,
                    "calibration": mock_calibration,
                    "simulation": mock_simulation,
                },
            )
            mock_aud.return_value = AgentResult(
                agent_role=AgentRole.AUDITOR,
                success=True,
                output=None,
            )

            result = orchestrator.run(mock_ctx, bus, pipeline=mock_pipeline)

        assert result.success
        assert result.output["multi_agent"] is True
        assert result.output["pipeline_passed"] is True

    def test_pipeline_aborts_on_data_scout_failure(self):
        from src.agents import AgentResult, AgentRole, MessageBus
        from src.agents.concrete import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        bus = MessageBus()
        mock_ctx = MagicMock()
        mock_pipeline = MagicMock()

        with patch("src.agents.concrete.DataScoutAgent.run") as mock_ds:
            mock_ds.return_value = AgentResult(
                agent_role=AgentRole.DATA_SCOUT,
                success=False,
                output=None,
                findings=["Data loading failed"],
            )

            result = orchestrator.run(mock_ctx, bus, pipeline=mock_pipeline)

        assert not result.success
        assert any("ABORT" in f for f in result.findings)

    def test_pipeline_aborts_on_audit_veto(self):
        from src.agents import AgentMessage, AgentResult, AgentRole, MessageBus, MessageType
        from src.agents.concrete import OrchestratorAgent
        from src.pipeline.stages import LoadedData, EngineeredFeatures, TrainedModels

        orchestrator = OrchestratorAgent()
        bus = MessageBus()
        mock_ctx = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline._X_train = None

        def audit_run(ctx, bus, **kwargs):
            # Simulate audit veto
            bus.publish(AgentMessage(
                sender=AgentRole.AUDITOR,
                msg_type=MessageType.VETO,
                payload={"reason": "leakage detected"},
            ))
            return AgentResult(
                agent_role=AgentRole.AUDITOR,
                success=False,
                findings=["VETO: leakage"],
            )

        with patch("src.agents.concrete.DataScoutAgent.run") as mock_ds, \
             patch("src.agents.concrete.FeatureEngineerAgent.run") as mock_fe, \
             patch("src.agents.concrete.ModelingAgent.run") as mock_mod, \
             patch("src.agents.concrete.AuditAgent.run", side_effect=audit_run):

            mock_ds.return_value = AgentResult(
                agent_role=AgentRole.DATA_SCOUT,
                success=True,
                output=LoadedData(teams=[]),
            )
            mock_fe.return_value = AgentResult(
                agent_role=AgentRole.FEATURE_ENGINEER,
                success=True,
                output=EngineeredFeatures(feature_names=["f1"]),
            )
            mock_mod.return_value = AgentResult(
                agent_role=AgentRole.MODELER,
                success=True,
                output={"models": TrainedModels()},
            )

            result = orchestrator.run(mock_ctx, bus, pipeline=mock_pipeline)

        assert not result.success
        assert any("BLOCKED" in f or "veto" in f.lower() for f in result.findings)

    def test_all_agent_roles_present(self):
        """Verify all 5 agent roles are defined."""
        from src.agents import AgentRole

        roles = list(AgentRole)
        assert len(roles) == 5
        assert AgentRole.DATA_SCOUT in roles
        assert AgentRole.FEATURE_ENGINEER in roles
        assert AgentRole.MODELER in roles
        assert AgentRole.AUDITOR in roles
        assert AgentRole.ORCHESTRATOR in roles


# ---------------------------------------------------------------------------
# AgentMessage tests
# ---------------------------------------------------------------------------


class TestAgentMessage:
    """S2: Message serialization and fields."""

    def test_message_auto_id_and_timestamp(self):
        from src.agents import AgentMessage, AgentRole, MessageType

        msg = AgentMessage(
            sender=AgentRole.DATA_SCOUT,
            msg_type=MessageType.DATA_READY,
        )
        assert msg.message_id  # Auto-generated
        assert msg.timestamp  # Auto-generated

    def test_message_to_dict(self):
        from src.agents import AgentMessage, AgentRole, MessageType

        msg = AgentMessage(
            sender=AgentRole.AUDITOR,
            msg_type=MessageType.VETO,
            payload={"reason": "test"},
            recipient=AgentRole.ORCHESTRATOR,
        )
        d = msg.to_dict()
        assert d["sender"] == "auditor"
        assert d["msg_type"] == "veto"
        assert d["recipient"] == "orchestrator"
        assert d["payload"]["reason"] == "test"
