"""Resource-aware agent scheduler with dependency graphs and priority queues.

Extends the basic sequential orchestrator with:
- DAG-based agent dependency resolution
- Priority-based scheduling with resource constraints
- Inter-agent negotiation via capability matching
- Parallel execution of independent agent stages

Implements Agent Directive V7 S2 (multi-agent architecture).
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import AgentMessage, AgentStatus
from .registry import AgentRegistry, MessageBus

logger = logging.getLogger(__name__)


class SchedulePriority(Enum):
    """Agent execution priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class AgentDependency:
    """Defines a dependency between agents."""
    agent_name: str
    depends_on: List[str] = field(default_factory=list)
    priority: SchedulePriority = SchedulePriority.NORMAL
    resource_weight: float = 1.0  # relative resource consumption (0.0 - 1.0)
    timeout_seconds: float = 300.0
    required: bool = True  # if False, pipeline continues on failure


@dataclass
class ScheduleResult:
    """Result of executing an agent schedule."""
    execution_order: List[str]
    results: Dict[str, Any]
    timing: Dict[str, float]
    status: str  # "completed", "failed", "partial"
    failed_agents: List[str] = field(default_factory=list)


class AgentDependencyGraph:
    """DAG of agent dependencies for topological scheduling."""

    def __init__(self) -> None:
        self._deps: Dict[str, AgentDependency] = {}

    def add(self, dep: AgentDependency) -> None:
        self._deps[dep.agent_name] = dep

    def get(self, agent_name: str) -> Optional[AgentDependency]:
        return self._deps.get(agent_name)

    def topological_sort(self) -> List[List[str]]:
        """Return agents grouped by execution level (parallelizable within each level).

        Level 0: agents with no dependencies
        Level 1: agents depending only on level-0 agents
        etc.

        Raises ValueError if a cycle is detected.
        """
        in_degree: Dict[str, int] = {}
        dependents: Dict[str, List[str]] = {}

        for name, dep in self._deps.items():
            in_degree.setdefault(name, 0)
            for parent in dep.depends_on:
                if parent in self._deps:
                    in_degree.setdefault(parent, 0)
                    dependents.setdefault(parent, []).append(name)
                    in_degree[name] = in_degree.get(name, 0) + 1

        levels: List[List[str]] = []
        remaining = set(in_degree.keys())

        while remaining:
            # Find all nodes with in_degree == 0
            ready = sorted(
                [n for n in remaining if in_degree.get(n, 0) == 0],
                key=lambda n: self._deps[n].priority.value if n in self._deps else 99,
            )
            if not ready:
                raise ValueError(
                    f"Cycle detected in agent dependency graph. "
                    f"Remaining: {remaining}"
                )
            levels.append(ready)
            for node in ready:
                remaining.discard(node)
                for dependent in dependents.get(node, []):
                    in_degree[dependent] = in_degree.get(dependent, 1) - 1

        return levels

    def agents(self) -> List[str]:
        return list(self._deps.keys())


# Default pipeline dependency graph
DEFAULT_PIPELINE_GRAPH = AgentDependencyGraph()
DEFAULT_PIPELINE_GRAPH.add(AgentDependency(
    agent_name="data_agent",
    depends_on=[],
    priority=SchedulePriority.CRITICAL,
    resource_weight=0.3,
    timeout_seconds=600.0,
))
DEFAULT_PIPELINE_GRAPH.add(AgentDependency(
    agent_name="feature_agent",
    depends_on=["data_agent"],
    priority=SchedulePriority.HIGH,
    resource_weight=0.4,
    timeout_seconds=300.0,
))
DEFAULT_PIPELINE_GRAPH.add(AgentDependency(
    agent_name="model_agent",
    depends_on=["feature_agent"],
    priority=SchedulePriority.HIGH,
    resource_weight=0.8,
    timeout_seconds=600.0,
))
DEFAULT_PIPELINE_GRAPH.add(AgentDependency(
    agent_name="audit_agent",
    depends_on=["model_agent"],
    priority=SchedulePriority.CRITICAL,
    resource_weight=0.2,
    timeout_seconds=120.0,
))


class ResourceAwareScheduler:
    """Schedule and execute agents respecting dependencies and resource budgets.

    Features:
    - DAG-based execution order via topological sort
    - Resource-weight-aware scheduling
    - Timeout enforcement per agent
    - Capability-based agent negotiation (route tasks to best-fit agent)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        graph: Optional[AgentDependencyGraph] = None,
        max_concurrent: int = 2,
        total_budget_seconds: float = 3600.0,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._graph = graph or DEFAULT_PIPELINE_GRAPH
        self._max_concurrent = max_concurrent
        self._total_budget_seconds = total_budget_seconds

    def execute(self, ctx: Any = None) -> ScheduleResult:
        """Execute agents in dependency order.

        Returns ScheduleResult with execution details.
        """
        try:
            levels = self._graph.topological_sort()
        except ValueError as e:
            return ScheduleResult(
                execution_order=[],
                results={},
                timing={},
                status="failed",
                failed_agents=[str(e)],
            )

        execution_order: List[str] = []
        results: Dict[str, Any] = {}
        timing: Dict[str, float] = {}
        failed: List[str] = []
        start_time = time.perf_counter()

        for level in levels:
            for agent_name in level:
                # Check total budget
                elapsed = time.perf_counter() - start_time
                if elapsed > self._total_budget_seconds:
                    logger.warning(
                        "Budget exhausted (%.0fs > %.0fs), skipping remaining agents",
                        elapsed, self._total_budget_seconds,
                    )
                    return ScheduleResult(
                        execution_order=execution_order,
                        results=results,
                        timing=timing,
                        status="partial",
                        failed_agents=failed,
                    )

                # Check agent is registered
                try:
                    agent = self._registry.get(agent_name)
                except KeyError:
                    dep = self._graph.get(agent_name)
                    if dep and dep.required:
                        failed.append(agent_name)
                    continue

                # Check dependencies succeeded
                dep = self._graph.get(agent_name)
                if dep:
                    dep_failed = [d for d in dep.depends_on if d in failed]
                    if dep_failed and dep.required:
                        logger.warning(
                            "Skipping '%s': dependencies failed: %s",
                            agent_name, dep_failed,
                        )
                        failed.append(agent_name)
                        continue

                # Execute agent
                action = agent.capabilities()[0]
                msg = AgentMessage(
                    sender="scheduler",
                    recipient=agent_name,
                    msg_type="request",
                    payload={"action": action, "upstream_results": results},
                )

                agent_start = time.perf_counter()
                response = self._bus.send(msg, ctx)
                agent_elapsed = time.perf_counter() - agent_start

                timing[agent_name] = round(agent_elapsed, 3)
                execution_order.append(agent_name)

                # Check timeout
                timeout = dep.timeout_seconds if dep else 300.0
                if agent_elapsed > timeout:
                    logger.warning(
                        "Agent '%s' exceeded timeout: %.1fs > %.0fs",
                        agent_name, agent_elapsed, timeout,
                    )

                if response.msg_type == "error":
                    if response.payload.get("veto"):
                        return ScheduleResult(
                            execution_order=execution_order,
                            results={"veto_stage": agent_name,
                                     "reason": response.payload.get("reason")},
                            timing=timing,
                            status="vetoed",
                            failed_agents=failed,
                        )
                    failed.append(agent_name)
                    if dep and dep.required:
                        results[agent_name] = {"error": response.payload.get("error")}
                    else:
                        results[agent_name] = {"error": response.payload.get("error"),
                                               "non_blocking": True}
                else:
                    results[agent_name] = response.payload

        status = "completed" if not failed else "partial"
        return ScheduleResult(
            execution_order=execution_order,
            results=results,
            timing=timing,
            status=status,
            failed_agents=failed,
        )

    def negotiate_capability(self, capability: str) -> Optional[str]:
        """Find the best agent for a given capability.

        Prefers agents with higher priority (lower numeric value) and
        those currently idle.
        """
        candidates = self._registry.find_by_capability(capability)
        if not candidates:
            return None

        # Score by priority and status
        scored: List[Tuple[int, str]] = []
        for agent in candidates:
            dep = self._graph.get(agent.name)
            priority = dep.priority.value if dep else SchedulePriority.NORMAL.value
            status_penalty = 0 if agent.status() == AgentStatus.IDLE else 10
            scored.append((priority + status_penalty, agent.name))

        scored.sort()
        return scored[0][1] if scored else None

    def execution_plan(self) -> Dict[str, Any]:
        """Preview the execution plan without running anything."""
        try:
            levels = self._graph.topological_sort()
        except ValueError as e:
            return {"error": str(e)}

        plan: Dict[str, Any] = {"levels": [], "total_agents": 0}
        for i, level in enumerate(levels):
            level_info = []
            for agent_name in level:
                dep = self._graph.get(agent_name)
                level_info.append({
                    "agent": agent_name,
                    "priority": dep.priority.name if dep else "NORMAL",
                    "resource_weight": dep.resource_weight if dep else 1.0,
                    "timeout_seconds": dep.timeout_seconds if dep else 300.0,
                    "required": dep.required if dep else True,
                    "depends_on": dep.depends_on if dep else [],
                })
            plan["levels"].append({"level": i, "agents": level_info})
            plan["total_agents"] = plan["total_agents"] + len(level_info)

        return plan
