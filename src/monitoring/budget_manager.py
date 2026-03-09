"""Per-stage compute budget allocation and cost tracking.

Extends the ResourceTracker with explicit per-stage budget allocation,
cost logging to the ExperimentRegistry, and resource-prioritized
experiment scheduling.

Implements Agent Directive V7 S20 (compute budget management):
  - Pre-cycle budget allocation across pipeline stages
  - Per-stage cost tracking and enforcement
  - Cost-per-improvement integration with experiment scheduling
  - Budget utilization reporting

Usage:
    from src.monitoring.budget_manager import BudgetManager, StageBudget

    manager = BudgetManager(total_budget_seconds=3600)
    manager.allocate("data_loading", fraction=0.10)
    manager.allocate("model_training", fraction=0.50)

    with manager.track_stage("data_loading"):
        load_data()

    print(manager.utilization_report())
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------


@dataclass
class StageBudget:
    """Budget allocation for a single pipeline stage."""

    stage_name: str
    fraction: float  # Fraction of total budget (0.0 - 1.0)
    max_seconds: float = 0.0  # Computed from total budget * fraction
    priority: int = 1  # Higher = more important; low priority shed first
    actual_seconds: float = 0.0
    started: bool = False
    completed: bool = False


@dataclass
class BudgetAlert:
    """Alert raised when a stage approaches or exceeds its budget."""

    stage_name: str
    alert_type: str  # "warning" (>80%), "exceeded" (>100%), "critical" (>150%)
    allocated_seconds: float
    actual_seconds: float
    utilization_pct: float
    message: str


# Default stage budget allocations (fractions summing to 1.0)
DEFAULT_STAGE_ALLOCATIONS: Dict[str, Dict[str, Any]] = {
    "data_loading": {"fraction": 0.10, "priority": 3},
    "feature_engineering": {"fraction": 0.10, "priority": 2},
    "model_training": {"fraction": 0.45, "priority": 1},
    "calibration": {"fraction": 0.10, "priority": 2},
    "simulation": {"fraction": 0.10, "priority": 2},
    "robustness_audit": {"fraction": 0.10, "priority": 3},
    "reporting": {"fraction": 0.05, "priority": 3},
}


class BudgetManager:
    """Manages per-stage compute budget allocation and tracking.

    Provides pre-cycle budget planning, real-time tracking during
    execution, alerts when stages approach limits, and post-cycle
    utilization reporting.
    """

    def __init__(
        self,
        total_budget_seconds: float = 3600.0,
        allocations: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._total_budget = total_budget_seconds
        self._stages: Dict[str, StageBudget] = {}
        self._alerts: List[BudgetAlert] = []
        self._start_time = time.perf_counter()

        # Apply allocations
        allocs = allocations or DEFAULT_STAGE_ALLOCATIONS
        for stage_name, params in allocs.items():
            self.allocate(
                stage_name,
                fraction=params["fraction"],
                priority=params.get("priority", 1),
            )

    @property
    def total_budget(self) -> float:
        return self._total_budget

    def allocate(
        self, stage_name: str, fraction: float, priority: int = 1
    ) -> StageBudget:
        """Allocate a fraction of the total budget to a stage."""
        budget = StageBudget(
            stage_name=stage_name,
            fraction=fraction,
            max_seconds=self._total_budget * fraction,
            priority=priority,
        )
        self._stages[stage_name] = budget
        return budget

    @contextmanager
    def track_stage(self, stage_name: str) -> Generator[StageBudget, None, None]:
        """Context manager to track a stage's compute usage against budget.

        Yields the StageBudget for the stage. On exit, checks utilization
        and logs alerts if approaching or exceeding budget.
        """
        budget = self._stages.get(stage_name)
        if budget is None:
            # Auto-allocate with minimal budget
            budget = self.allocate(stage_name, fraction=0.0, priority=0)
            logger.warning(
                "Stage '%s' has no budget allocation; tracking only",
                stage_name,
            )

        budget.started = True
        t0 = time.perf_counter()

        try:
            yield budget
        finally:
            elapsed = time.perf_counter() - t0
            budget.actual_seconds = elapsed
            budget.completed = True

            # Check utilization and generate alerts
            if budget.max_seconds > 0:
                utilization = elapsed / budget.max_seconds
                if utilization > 1.5:
                    self._add_alert(budget, "critical", utilization)
                elif utilization > 1.0:
                    self._add_alert(budget, "exceeded", utilization)
                elif utilization > 0.8:
                    self._add_alert(budget, "warning", utilization)

                logger.info(
                    "Stage '%s': %.1fs / %.1fs (%.0f%% of budget)",
                    stage_name,
                    elapsed,
                    budget.max_seconds,
                    utilization * 100,
                )

    def get_budget(self, stage_name: str) -> Optional[StageBudget]:
        """Get the budget allocation for a stage."""
        return self._stages.get(stage_name)

    def remaining_budget(self) -> float:
        """Compute remaining wall-clock budget across all stages."""
        used = sum(s.actual_seconds for s in self._stages.values() if s.completed)
        return max(0, self._total_budget - used)

    def stage_remaining(self, stage_name: str) -> float:
        """Seconds remaining in a specific stage's budget."""
        budget = self._stages.get(stage_name)
        if budget is None:
            return 0.0
        return max(0, budget.max_seconds - budget.actual_seconds)

    @property
    def alerts(self) -> List[BudgetAlert]:
        """All budget alerts generated during tracking."""
        return list(self._alerts)

    def utilization_report(self) -> Dict[str, Any]:
        """Generate a structured utilization report.

        Returns a dict suitable for logging to ExperimentRegistry.
        """
        total_used = sum(
            s.actual_seconds for s in self._stages.values() if s.completed
        )
        total_elapsed = time.perf_counter() - self._start_time

        stages_report = {}
        for name, budget in self._stages.items():
            utilization = (
                (budget.actual_seconds / budget.max_seconds * 100)
                if budget.max_seconds > 0
                else 0
            )
            stages_report[name] = {
                "allocated_seconds": round(budget.max_seconds, 1),
                "actual_seconds": round(budget.actual_seconds, 1),
                "utilization_pct": round(utilization, 1),
                "priority": budget.priority,
                "completed": budget.completed,
            }

        return {
            "total_budget_seconds": self._total_budget,
            "total_used_seconds": round(total_used, 1),
            "total_elapsed_seconds": round(total_elapsed, 1),
            "budget_utilization_pct": round(
                total_used / self._total_budget * 100
                if self._total_budget > 0
                else 0,
                1,
            ),
            "remaining_seconds": round(self.remaining_budget(), 1),
            "n_stages_completed": sum(
                1 for s in self._stages.values() if s.completed
            ),
            "n_alerts": len(self._alerts),
            "stages": stages_report,
            "alerts": [
                {
                    "stage": a.stage_name,
                    "type": a.alert_type,
                    "utilization_pct": round(a.utilization_pct, 1),
                    "message": a.message,
                }
                for a in self._alerts
            ],
        }

    def summary(self) -> str:
        """Human-readable budget utilization summary."""
        total_used = sum(
            s.actual_seconds for s in self._stages.values() if s.completed
        )
        lines = [
            "Compute Budget Report (S20)",
            "=" * 70,
            f"Total budget: {self._total_budget:.0f}s | "
            f"Used: {total_used:.0f}s | "
            f"Remaining: {self.remaining_budget():.0f}s",
            "",
            f"  {'Stage':<25s} {'Budget':>8s}  {'Actual':>8s}  {'Util%':>7s}  {'Status':>10s}",
            "-" * 70,
        ]

        for name, budget in self._stages.items():
            util = (
                f"{budget.actual_seconds / budget.max_seconds * 100:.0f}%"
                if budget.max_seconds > 0
                else "N/A"
            )
            status = (
                "DONE" if budget.completed
                else "RUNNING" if budget.started
                else "PENDING"
            )
            lines.append(
                f"  {name:<25s} {budget.max_seconds:>7.0f}s  "
                f"{budget.actual_seconds:>7.1f}s  {util:>7s}  {status:>10s}"
            )

        if self._alerts:
            lines.append("")
            lines.append(f"Alerts ({len(self._alerts)}):")
            for alert in self._alerts:
                lines.append(f"  [{alert.alert_type.upper()}] {alert.message}")

        return "\n".join(lines)

    def prioritized_stages(self) -> List[str]:
        """Return stages sorted by priority (highest first).

        Useful for experiment scheduling: when budget is tight,
        shed low-priority stages first.
        """
        return [
            s.stage_name
            for s in sorted(
                self._stages.values(),
                key=lambda s: s.priority,
                reverse=True,
            )
        ]

    def should_shed(self, stage_name: str) -> bool:
        """Check if a stage should be shed due to budget pressure.

        Returns True if total budget is >80% consumed and this
        stage is low priority (priority <= 1).
        """
        total_used = sum(
            s.actual_seconds for s in self._stages.values() if s.completed
        )
        utilization = total_used / self._total_budget if self._total_budget > 0 else 0
        budget = self._stages.get(stage_name)
        if budget is None:
            return False
        return utilization > 0.8 and budget.priority <= 1

    def _add_alert(
        self, budget: StageBudget, alert_type: str, utilization: float
    ) -> None:
        alert = BudgetAlert(
            stage_name=budget.stage_name,
            alert_type=alert_type,
            allocated_seconds=budget.max_seconds,
            actual_seconds=budget.actual_seconds,
            utilization_pct=utilization * 100,
            message=(
                f"Stage '{budget.stage_name}' at {utilization * 100:.0f}% "
                f"of budget ({budget.actual_seconds:.1f}s / "
                f"{budget.max_seconds:.1f}s)"
            ),
        )
        self._alerts.append(alert)
        log_fn = logger.warning if alert_type != "warning" else logger.info
        log_fn("BUDGET %s: %s", alert_type.upper(), alert.message)
