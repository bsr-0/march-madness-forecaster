"""Cost-performance tracking and Pareto frontier analysis.

Extends resource tracking with dollar-cost attribution, historical
baseline comparison, and Pareto frontier computation across runs.

Implements Agent Directive V7 S20 (compute budget — cost attribution).

Usage:
    from src.monitoring.cost_tracker import CostTracker, CostModel

    model = CostModel(cost_per_wall_hour=0.10)
    tracker = CostTracker(model=model)
    tracker.add_run(brier=0.189, wall_seconds=310, peak_memory_mb=2048)
    frontier = tracker.pareto_frontier()
    print(tracker.summary())
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_COST_HISTORY_PATH = "data/runs/cost_performance.jsonl"


@dataclass
class CostModel:
    """Cost model for pipeline execution.

    Configurable per-unit costs for compute resources. Defaults assume
    modest cloud pricing (e.g., GCP n1-standard-4).
    """

    cost_per_wall_hour: float = 0.10  # $/hour wall-clock
    cost_per_cpu_hour: float = 0.05  # $/hour CPU time
    cost_per_gb_hour: float = 0.01  # $/hour per GB memory


@dataclass
class CostRecord:
    """Cost and performance data for a single pipeline run."""

    record_id: str = ""
    timestamp: str = ""
    brier_score: float = 0.0
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    total_cost: float = 0.0
    phase_costs: Dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    is_pareto_optimal: bool = False

    def __post_init__(self) -> None:
        if not self.record_id:
            import uuid

            self.record_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostTracker:
    """Tracks cost-performance across pipeline runs.

    Computes per-phase cost attribution, maintains a Pareto frontier
    of optimal runs (best Brier for a given cost), and compares
    against historical baselines.
    """

    def __init__(
        self,
        model: Optional[CostModel] = None,
        history_path: str = DEFAULT_COST_HISTORY_PATH,
        baseline: Optional[Dict[str, float]] = None,
    ) -> None:
        self._model = model or CostModel()
        self._history_path = Path(history_path)
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline = baseline  # Optional baseline phase costs

    def compute_cost(
        self,
        wall_seconds: float,
        cpu_seconds: float = 0.0,
        peak_memory_mb: float = 0.0,
    ) -> float:
        """Compute total cost for given resource usage."""
        wall_cost = (wall_seconds / 3600) * self._model.cost_per_wall_hour
        cpu_cost = (cpu_seconds / 3600) * self._model.cost_per_cpu_hour
        mem_cost = (peak_memory_mb / 1024) * (wall_seconds / 3600) * self._model.cost_per_gb_hour
        return round(wall_cost + cpu_cost + mem_cost, 4)

    def compute_phase_costs(
        self,
        phase_records: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute cost per pipeline phase.

        Args:
            phase_records: phase_name → {wall_seconds, cpu_seconds, peak_memory_mb}

        Returns:
            phase_name → cost in dollars
        """
        costs: Dict[str, float] = {}
        for phase, data in phase_records.items():
            costs[phase] = self.compute_cost(
                wall_seconds=data.get("wall_seconds", 0),
                cpu_seconds=data.get("cpu_seconds", 0),
                peak_memory_mb=data.get("peak_memory_mb", 0),
            )
        return costs

    def add_run(
        self,
        brier_score: float,
        wall_seconds: float,
        cpu_seconds: float = 0.0,
        peak_memory_mb: float = 0.0,
        phase_costs: Optional[Dict[str, float]] = None,
        config_hash: str = "",
    ) -> CostRecord:
        """Record a pipeline run's cost and performance."""
        total_cost = self.compute_cost(wall_seconds, cpu_seconds, peak_memory_mb)

        record = CostRecord(
            brier_score=brier_score,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_memory_mb=peak_memory_mb,
            total_cost=total_cost,
            phase_costs=phase_costs or {},
            config_hash=config_hash,
        )

        with open(self._history_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(
            "Recorded run: Brier=%.4f, cost=$%.4f, wall=%.0fs",
            brier_score,
            total_cost,
            wall_seconds,
        )
        return record

    def pareto_frontier(self) -> List[CostRecord]:
        """Compute the Pareto frontier of cost vs performance.

        Returns runs that are not dominated by any other run (i.e., no other
        run has both lower Brier AND lower cost).
        """
        records = self._load_history()
        if not records:
            return []

        dominated = set()
        for i, r1 in enumerate(records):
            for j, r2 in enumerate(records):
                if i != j:
                    if (
                        r2.brier_score <= r1.brier_score
                        and r2.total_cost <= r1.total_cost
                        and (
                            r2.brier_score < r1.brier_score
                            or r2.total_cost < r1.total_cost
                        )
                    ):
                        dominated.add(i)
                        break

        frontier = []
        for i, r in enumerate(records):
            r.is_pareto_optimal = i not in dominated
            if r.is_pareto_optimal:
                frontier.append(r)

        return sorted(frontier, key=lambda r: r.total_cost)

    def compare_to_baseline(
        self,
        phase_costs: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compare current phase costs against the baseline.

        Returns:
            phase → {current, baseline, delta, delta_pct}
        """
        if not self._baseline:
            return {}

        comparison: Dict[str, Dict[str, float]] = {}
        for phase, current in phase_costs.items():
            baseline = self._baseline.get(phase, 0)
            delta = current - baseline
            delta_pct = (delta / baseline * 100) if baseline > 0 else 0
            comparison[phase] = {
                "current": current,
                "baseline": baseline,
                "delta": round(delta, 4),
                "delta_pct": round(delta_pct, 1),
            }
        return comparison

    def summary(self) -> str:
        """Human-readable cost-performance summary."""
        records = self._load_history()
        if not records:
            return "No cost-performance data recorded."

        frontier = self.pareto_frontier()
        latest = records[-1]

        lines = [
            "Cost-Performance Summary",
            "=" * 60,
            f"Total runs tracked: {len(records)}",
            f"Latest: Brier={latest.brier_score:.4f}, cost=${latest.total_cost:.4f}",
            "",
            f"Pareto frontier: {len(frontier)} optimal runs",
        ]

        for r in frontier:
            lines.append(
                f"  Brier={r.brier_score:.4f}  cost=${r.total_cost:.4f}  "
                f"wall={r.wall_seconds:.0f}s"
            )

        # Phase cost breakdown for latest run
        if latest.phase_costs:
            lines.append("")
            lines.append("Latest run — phase cost breakdown:")
            total = sum(latest.phase_costs.values())
            for phase, cost in sorted(
                latest.phase_costs.items(), key=lambda x: -x[1]
            ):
                pct = (cost / total * 100) if total > 0 else 0
                lines.append(f"  {phase:<30s} ${cost:.4f}  ({pct:5.1f}%)")

        return "\n".join(lines)

    def _load_history(self) -> List[CostRecord]:
        """Load cost-performance history from disk."""
        if not self._history_path.exists():
            return []
        records: List[CostRecord] = []
        with open(self._history_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(CostRecord(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records
