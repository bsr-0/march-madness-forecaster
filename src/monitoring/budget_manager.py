"""Compute budget manager with per-agent allocation and real-time rebalancing.

Extends ResourceTracker and CostTracker with:
- Per-agent budget allocation and tracking
- Real-time budget rebalancing when agents finish early/late
- Cost-aware scheduling recommendations
- Budget forecasting based on historical data

Implements Agent Directive V7 S20 (compute budget management).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentBudget:
    """Budget allocation for a single agent."""
    agent_name: str
    max_wall_seconds: float
    max_memory_mb: float = 4096.0
    priority_weight: float = 1.0  # higher = more important for rebalancing
    actual_wall_seconds: float = 0.0
    actual_memory_mb: float = 0.0
    status: str = "pending"  # "pending", "running", "completed", "over_budget"


@dataclass
class BudgetSnapshot:
    """Point-in-time budget utilization snapshot."""
    timestamp: float
    total_budget_seconds: float
    total_consumed_seconds: float
    total_remaining_seconds: float
    utilization_pct: float
    agent_utilizations: Dict[str, float]
    projected_overrun: bool
    projected_total_seconds: float


class ComputeBudgetManager:
    """Manage per-agent compute budgets with real-time rebalancing.

    Features:
    - Pre-allocate budgets to agents based on historical profiles
    - Track consumption in real-time
    - Rebalance remaining budget when agents finish early
    - Project whether the pipeline will exceed total budget
    - Provide scheduling recommendations
    """

    # Default agent budget allocations as fraction of total
    DEFAULT_ALLOCATIONS: Dict[str, float] = {
        "data_agent": 0.15,
        "feature_agent": 0.20,
        "model_agent": 0.45,
        "audit_agent": 0.10,
        "reserve": 0.10,  # headroom for retries and overhead
    }

    def __init__(
        self,
        total_budget_seconds: float = 3600.0,
        total_memory_mb: float = 8192.0,
        allocations: Optional[Dict[str, float]] = None,
        cost_per_hour: float = 0.10,
    ) -> None:
        self._total_budget = total_budget_seconds
        self._total_memory = total_memory_mb
        self._cost_per_hour = cost_per_hour
        self._allocations = allocations or dict(self.DEFAULT_ALLOCATIONS)
        self._agent_budgets: Dict[str, AgentBudget] = {}
        self._snapshots: List[BudgetSnapshot] = []
        self._start_time: Optional[float] = None

        self._allocate_budgets()

    def _allocate_budgets(self) -> None:
        """Distribute total budget across agents based on allocation fractions."""
        for agent_name, fraction in self._allocations.items():
            if agent_name == "reserve":
                continue
            self._agent_budgets[agent_name] = AgentBudget(
                agent_name=agent_name,
                max_wall_seconds=self._total_budget * fraction,
                max_memory_mb=self._total_memory * fraction,
                priority_weight=1.0 / max(fraction, 0.01),
            )

    def start(self) -> None:
        """Mark the start of budget tracking."""
        self._start_time = time.perf_counter()

    def agent_started(self, agent_name: str) -> AgentBudget:
        """Mark an agent as running and return its budget."""
        budget = self._agent_budgets.get(agent_name)
        if budget is None:
            # Auto-allocate from reserve
            reserve_frac = self._allocations.get("reserve", 0.10)
            budget = AgentBudget(
                agent_name=agent_name,
                max_wall_seconds=self._total_budget * reserve_frac * 0.5,
                max_memory_mb=self._total_memory * 0.1,
            )
            self._agent_budgets[agent_name] = budget

        budget.status = "running"
        logger.info(
            "Agent '%s' started with budget: %.0fs wall, %.0f MB",
            agent_name, budget.max_wall_seconds, budget.max_memory_mb,
        )
        return budget

    def agent_completed(
        self,
        agent_name: str,
        wall_seconds: float,
        memory_mb: float = 0.0,
    ) -> AgentBudget:
        """Record agent completion and trigger rebalancing if under budget."""
        budget = self._agent_budgets.get(agent_name)
        if budget is None:
            budget = AgentBudget(agent_name=agent_name, max_wall_seconds=0)
            self._agent_budgets[agent_name] = budget

        budget.actual_wall_seconds = wall_seconds
        budget.actual_memory_mb = memory_mb

        if wall_seconds > budget.max_wall_seconds:
            budget.status = "over_budget"
            logger.warning(
                "Agent '%s' over budget: %.1fs > %.0fs (%.0f%% over)",
                agent_name, wall_seconds, budget.max_wall_seconds,
                (wall_seconds / budget.max_wall_seconds - 1) * 100,
            )
        else:
            budget.status = "completed"
            saved = budget.max_wall_seconds - wall_seconds
            if saved > 1.0:
                logger.info(
                    "Agent '%s' under budget by %.1fs — rebalancing",
                    agent_name, saved,
                )
                self._rebalance(agent_name, saved)

        self._take_snapshot()
        return budget

    def _rebalance(self, completed_agent: str, saved_seconds: float) -> None:
        """Redistribute saved budget to remaining pending/running agents."""
        remaining_agents = [
            b for b in self._agent_budgets.values()
            if b.status in ("pending", "running") and b.agent_name != completed_agent
        ]
        if not remaining_agents:
            return

        total_weight = sum(b.priority_weight for b in remaining_agents)
        for budget in remaining_agents:
            share = (budget.priority_weight / total_weight) * saved_seconds
            budget.max_wall_seconds += share
            logger.debug(
                "Rebalanced +%.1fs to '%s' (new budget: %.0fs)",
                share, budget.agent_name, budget.max_wall_seconds,
            )

    def check_budget(self, agent_name: str) -> Dict[str, Any]:
        """Check if an agent is within its budget."""
        budget = self._agent_budgets.get(agent_name)
        if budget is None:
            return {"agent": agent_name, "status": "unknown"}

        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        return {
            "agent": agent_name,
            "status": budget.status,
            "budget_seconds": budget.max_wall_seconds,
            "actual_seconds": budget.actual_wall_seconds,
            "remaining_seconds": max(0, budget.max_wall_seconds - budget.actual_wall_seconds),
            "utilization_pct": round(
                (budget.actual_wall_seconds / budget.max_wall_seconds * 100)
                if budget.max_wall_seconds > 0 else 0, 1
            ),
        }

    def _take_snapshot(self) -> None:
        """Capture current budget utilization state."""
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        consumed = sum(b.actual_wall_seconds for b in self._agent_budgets.values())
        remaining = max(0, self._total_budget - consumed)

        agent_utils: Dict[str, float] = {}
        for name, budget in self._agent_budgets.items():
            if budget.max_wall_seconds > 0:
                agent_utils[name] = round(
                    budget.actual_wall_seconds / budget.max_wall_seconds * 100, 1
                )
            else:
                agent_utils[name] = 0.0

        # Project total based on current rate
        completed_agents = [b for b in self._agent_budgets.values()
                           if b.status in ("completed", "over_budget")]
        pending_agents = [b for b in self._agent_budgets.values()
                         if b.status in ("pending", "running")]

        projected = consumed + sum(b.max_wall_seconds for b in pending_agents)

        snapshot = BudgetSnapshot(
            timestamp=elapsed,
            total_budget_seconds=self._total_budget,
            total_consumed_seconds=consumed,
            total_remaining_seconds=remaining,
            utilization_pct=round(consumed / self._total_budget * 100, 1)
            if self._total_budget > 0 else 0,
            agent_utilizations=agent_utils,
            projected_overrun=projected > self._total_budget,
            projected_total_seconds=projected,
        )
        self._snapshots.append(snapshot)

    def estimated_cost(self) -> float:
        """Estimate dollar cost based on consumed time."""
        consumed = sum(b.actual_wall_seconds for b in self._agent_budgets.values())
        return round((consumed / 3600) * self._cost_per_hour, 4)

    def summary(self) -> str:
        """Human-readable budget summary."""
        consumed = sum(b.actual_wall_seconds for b in self._agent_budgets.values())
        lines = [
            "Compute Budget Summary",
            "=" * 70,
            f"Total budget: {self._total_budget:.0f}s  |  "
            f"Consumed: {consumed:.1f}s  |  "
            f"Remaining: {max(0, self._total_budget - consumed):.1f}s  |  "
            f"Est. cost: ${self.estimated_cost():.4f}",
            "",
            f"  {'Agent':<25s} {'Budget':>10s} {'Actual':>10s} {'Util%':>8s} {'Status':<12s}",
            "-" * 70,
        ]

        for name, budget in sorted(self._agent_budgets.items()):
            util = (budget.actual_wall_seconds / budget.max_wall_seconds * 100
                    if budget.max_wall_seconds > 0 else 0)
            lines.append(
                f"  {name:<25s} {budget.max_wall_seconds:>9.0f}s "
                f"{budget.actual_wall_seconds:>9.1f}s {util:>7.1f}% {budget.status:<12s}"
            )

        lines.append("-" * 70)

        # Projection
        if self._snapshots:
            latest = self._snapshots[-1]
            if latest.projected_overrun:
                lines.append(
                    f"WARNING: Projected total {latest.projected_total_seconds:.0f}s "
                    f"exceeds budget {self._total_budget:.0f}s"
                )
            else:
                lines.append("Budget: OK (within limits)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Structured output for integration."""
        return {
            "total_budget_seconds": self._total_budget,
            "total_consumed_seconds": sum(
                b.actual_wall_seconds for b in self._agent_budgets.values()
            ),
            "estimated_cost_usd": self.estimated_cost(),
            "agents": {
                name: {
                    "budget_seconds": b.max_wall_seconds,
                    "actual_seconds": b.actual_wall_seconds,
                    "status": b.status,
                }
                for name, b in self._agent_budgets.items()
            },
            "snapshots_count": len(self._snapshots),
        }
