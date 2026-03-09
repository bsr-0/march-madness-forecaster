"""Tests for resource-aware agent scheduler (S2 multi-agent architecture)."""

import pytest

from src.agents.base import AgentMessage, AgentStatus
from src.agents.registry import AgentRegistry, MessageBus
from src.agents.scheduler import (
    AgentDependency,
    AgentDependencyGraph,
    DEFAULT_PIPELINE_GRAPH,
    ResourceAwareScheduler,
    SchedulePriority,
    ScheduleResult,
)


class _StubAgent:
    """Minimal agent for testing."""

    def __init__(self, agent_name: str, caps: list = None, fail: bool = False,
                 veto: bool = False):
        self._name = agent_name
        self._caps = caps or ["action"]
        self._fail = fail
        self._veto = veto
        self._status = AgentStatus.IDLE

    @property
    def name(self) -> str:
        return self._name

    def capabilities(self):
        return self._caps

    def handle(self, message: AgentMessage, ctx=None) -> AgentMessage:
        self._status = AgentStatus.RUNNING
        if self._veto:
            self._status = AgentStatus.COMPLETED
            return AgentMessage(
                sender=self._name, recipient=message.sender,
                msg_type="error",
                payload={"veto": True, "reason": "test veto"},
                correlation_id=message.correlation_id,
            )
        if self._fail:
            self._status = AgentStatus.FAILED
            return AgentMessage(
                sender=self._name, recipient=message.sender,
                msg_type="error",
                payload={"error": "test failure"},
                correlation_id=message.correlation_id,
            )
        self._status = AgentStatus.COMPLETED
        return AgentMessage(
            sender=self._name, recipient=message.sender,
            msg_type="result",
            payload={"status": "ok", "agent": self._name},
            correlation_id=message.correlation_id,
        )

    def status(self):
        return self._status


def _make_pipeline(agents=None, graph=None):
    """Create a scheduler with registered agents."""
    registry = AgentRegistry()
    if agents is None:
        agents = [
            _StubAgent("data_agent", ["load_data"]),
            _StubAgent("feature_agent", ["engineer_features"]),
            _StubAgent("model_agent", ["train_models"]),
            _StubAgent("audit_agent", ["run_ablation"]),
        ]
    for agent in agents:
        registry.register(agent)
    bus = MessageBus(registry)
    return ResourceAwareScheduler(registry, bus, graph=graph)


class TestAgentDependencyGraph:
    def test_topological_sort_linear(self):
        g = AgentDependencyGraph()
        g.add(AgentDependency("a", []))
        g.add(AgentDependency("b", ["a"]))
        g.add(AgentDependency("c", ["b"]))
        levels = g.topological_sort()
        assert levels == [["a"], ["b"], ["c"]]

    def test_topological_sort_parallel(self):
        g = AgentDependencyGraph()
        g.add(AgentDependency("a", []))
        g.add(AgentDependency("b", []))
        g.add(AgentDependency("c", ["a", "b"]))
        levels = g.topological_sort()
        assert len(levels[0]) == 2
        assert levels[1] == ["c"]

    def test_cycle_detection(self):
        g = AgentDependencyGraph()
        g.add(AgentDependency("a", ["b"]))
        g.add(AgentDependency("b", ["a"]))
        with pytest.raises(ValueError, match="Cycle detected"):
            g.topological_sort()

    def test_default_pipeline_graph_sorts(self):
        levels = DEFAULT_PIPELINE_GRAPH.topological_sort()
        assert len(levels) >= 1
        # data_agent should be in level 0
        assert "data_agent" in levels[0]

    def test_priority_ordering_within_level(self):
        g = AgentDependencyGraph()
        g.add(AgentDependency("low", [], priority=SchedulePriority.LOW))
        g.add(AgentDependency("critical", [], priority=SchedulePriority.CRITICAL))
        g.add(AgentDependency("normal", [], priority=SchedulePriority.NORMAL))
        levels = g.topological_sort()
        # All at level 0, sorted by priority
        assert levels[0][0] == "critical"


class TestResourceAwareScheduler:
    def test_full_pipeline_execution(self):
        scheduler = _make_pipeline()
        result = scheduler.execute()
        assert result.status == "completed"
        assert len(result.execution_order) == 4
        assert "data_agent" in result.results
        assert all(t > 0 or t == 0 for t in result.timing.values())

    def test_pipeline_with_veto(self):
        agents = [
            _StubAgent("data_agent", ["load_data"]),
            _StubAgent("feature_agent", ["engineer_features"]),
            _StubAgent("model_agent", ["train_models"]),
            _StubAgent("audit_agent", ["run_ablation"], veto=True),
        ]
        scheduler = _make_pipeline(agents)
        result = scheduler.execute()
        assert result.status == "vetoed"

    def test_pipeline_with_failure(self):
        agents = [
            _StubAgent("data_agent", ["load_data"], fail=True),
            _StubAgent("feature_agent", ["engineer_features"]),
            _StubAgent("model_agent", ["train_models"]),
            _StubAgent("audit_agent", ["run_ablation"]),
        ]
        scheduler = _make_pipeline(agents)
        result = scheduler.execute()
        assert "data_agent" in result.failed_agents
        # feature_agent depends on data_agent so should also fail
        assert "feature_agent" in result.failed_agents

    def test_negotiate_capability(self):
        agents = [
            _StubAgent("a", ["load_data"]),
            _StubAgent("b", ["load_data", "validate"]),
        ]
        scheduler = _make_pipeline(agents, graph=AgentDependencyGraph())
        # Add agents to graph with different priorities
        scheduler._graph.add(AgentDependency("a", [], priority=SchedulePriority.NORMAL))
        scheduler._graph.add(AgentDependency("b", [], priority=SchedulePriority.CRITICAL))
        best = scheduler.negotiate_capability("load_data")
        assert best == "b"

    def test_execution_plan(self):
        scheduler = _make_pipeline()
        plan = scheduler.execution_plan()
        assert "levels" in plan
        assert plan["total_agents"] == 4

    def test_missing_agent_skipped_if_not_required(self):
        g = AgentDependencyGraph()
        g.add(AgentDependency("data_agent", [], required=True))
        g.add(AgentDependency("optional_agent", ["data_agent"], required=False))
        agents = [_StubAgent("data_agent", ["load_data"])]
        scheduler = _make_pipeline(agents, graph=g)
        result = scheduler.execute()
        # Should complete since optional_agent is not required
        assert result.status == "completed" or result.status == "partial"
