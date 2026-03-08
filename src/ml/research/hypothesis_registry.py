"""Hypothesis registry and experiment scheduler for continuous research loop.

Implements Agent Directive V7 S14 (continuous research loop):
  - Maintain a registry of research hypotheses with status tracking
  - Schedule experiments based on priority and expected impact
  - Track hypothesis → experiment → outcome linkage
  - Support automated hypothesis generation from model diagnostics

Usage:
    from src.ml.research.hypothesis_registry import HypothesisRegistry, Hypothesis

    registry = HypothesisRegistry()
    h = registry.add(
        title="Adding momentum features improves late-season prediction",
        rationale="Teams on winning streaks tend to outperform Elo predictions",
        expected_impact=0.005,  # Expected Brier improvement
        priority="high",
    )
    registry.schedule(h.hypothesis_id, experiment_config={...})
    registry.resolve(h.hypothesis_id, outcome="confirmed", brier_delta=-0.003)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "data/hypothesis_registry.jsonl"


class HypothesisStatus(str, Enum):
    """Lifecycle states for a research hypothesis."""

    PROPOSED = "proposed"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class HypothesisPriority(str, Enum):
    """Priority levels for research hypotheses."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Hypothesis:
    """A single research hypothesis in the registry."""

    hypothesis_id: str = ""
    title: str = ""
    rationale: str = ""
    expected_impact: float = 0.0  # Expected Brier improvement (negative = better)
    priority: str = "medium"
    status: str = "proposed"

    # Linkage
    experiment_ids: List[str] = field(default_factory=list)
    experiment_config: Dict[str, Any] = field(default_factory=dict)

    # Outcome
    actual_impact: float = 0.0
    outcome_notes: str = ""

    # Metadata
    created_at: str = ""
    resolved_at: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""  # "manual", "auto_diagnostic", "ablation_study"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Hypothesis:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class ExperimentSchedule:
    """A scheduled experiment linked to a hypothesis."""

    schedule_id: str
    hypothesis_id: str
    config: Dict[str, Any]
    priority: int  # Lower = higher priority
    created_at: str
    status: str = "pending"  # pending, running, completed, failed


class HypothesisRegistry:
    """Append-only registry of research hypotheses with experiment scheduling.

    Persists hypotheses in a JSONL file for cross-session tracking.
    Supports automated hypothesis generation from model diagnostics.
    """

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self._schedule: List[ExperimentSchedule] = []

    def add(
        self,
        title: str,
        rationale: str = "",
        expected_impact: float = 0.0,
        priority: str = "medium",
        tags: Optional[List[str]] = None,
        source: str = "manual",
    ) -> Hypothesis:
        """Register a new hypothesis."""
        h = Hypothesis(
            hypothesis_id=str(uuid.uuid4())[:8],
            title=title,
            rationale=rationale,
            expected_impact=expected_impact,
            priority=priority,
            status=HypothesisStatus.PROPOSED.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or [],
            source=source,
        )
        self._append(h)
        logger.info("Registered hypothesis %s: %s", h.hypothesis_id, title)
        return h

    def schedule(
        self,
        hypothesis_id: str,
        experiment_config: Dict[str, Any],
    ) -> ExperimentSchedule:
        """Schedule an experiment for a hypothesis."""
        h = self.get(hypothesis_id)
        if h is None:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        priority_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        schedule = ExperimentSchedule(
            schedule_id=str(uuid.uuid4())[:8],
            hypothesis_id=hypothesis_id,
            config=experiment_config,
            priority=priority_map.get(h.priority, 2),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._schedule.append(schedule)

        # Update hypothesis status
        h.status = HypothesisStatus.SCHEDULED.value
        h.experiment_config = experiment_config
        self._update(h)

        logger.info(
            "Scheduled experiment %s for hypothesis %s",
            schedule.schedule_id,
            hypothesis_id,
        )
        return schedule

    def resolve(
        self,
        hypothesis_id: str,
        outcome: str,
        actual_impact: float = 0.0,
        notes: str = "",
        experiment_id: str = "",
    ) -> None:
        """Record the outcome of a hypothesis test."""
        h = self.get(hypothesis_id)
        if h is None:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        h.status = outcome  # "confirmed" or "rejected"
        h.actual_impact = actual_impact
        h.outcome_notes = notes
        h.resolved_at = datetime.now(timezone.utc).isoformat()
        if experiment_id:
            h.experiment_ids.append(experiment_id)
        self._update(h)

        logger.info(
            "Resolved hypothesis %s: %s (impact=%.6f)",
            hypothesis_id,
            outcome,
            actual_impact,
        )

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Retrieve a hypothesis by ID prefix."""
        for h in self._load_all():
            if h.hypothesis_id.startswith(hypothesis_id):
                return h
        return None

    def list(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        n: int = 50,
    ) -> List[Hypothesis]:
        """List hypotheses, optionally filtered by status/priority."""
        all_h = self._load_all()
        if status:
            all_h = [h for h in all_h if h.status == status]
        if priority:
            all_h = [h for h in all_h if h.priority == priority]
        return all_h[-n:]

    def next_experiments(self, n: int = 5) -> List[ExperimentSchedule]:
        """Return the next N scheduled experiments by priority."""
        pending = [s for s in self._schedule if s.status == "pending"]
        return sorted(pending, key=lambda s: s.priority)[:n]

    def generate_from_diagnostics(
        self,
        dropout_report: Optional[Dict[str, Any]] = None,
        shift_report: Optional[Dict[str, Any]] = None,
        loyo_year_briers: Optional[Dict[int, float]] = None,
    ) -> List[Hypothesis]:
        """Auto-generate hypotheses from model diagnostics.

        Examines robustness reports and validation results to propose
        actionable research directions.
        """
        generated: List[Hypothesis] = []

        # From feature dropout: fragile features suggest alternatives
        if dropout_report and dropout_report.get("fragile_features"):
            fragile = dropout_report["fragile_features"]
            h = self.add(
                title=f"Reduce dependence on fragile features: {', '.join(fragile[:3])}",
                rationale=(
                    f"Feature dropout test found {len(fragile)} fragile features "
                    f"with degradation > threshold. Consider feature engineering "
                    f"alternatives or regularization."
                ),
                expected_impact=0.003,
                priority="high",
                tags=["auto_generated", "robustness"],
                source="auto_diagnostic",
            )
            generated.append(h)

        # From distribution shift: significant shifts suggest adaptation
        if shift_report and shift_report.get("significant_features"):
            sig = shift_report["significant_features"]
            h = self.add(
                title=f"Address distribution shift in {len(sig)} features",
                rationale=(
                    f"PSI > 0.2 detected for: {', '.join(sig[:3])}. "
                    f"Consider domain adaptation, feature normalization, "
                    f"or temporal windowing."
                ),
                expected_impact=0.005,
                priority="high",
                tags=["auto_generated", "distribution_shift"],
                source="auto_diagnostic",
            )
            generated.append(h)

        # From LOYO: worst year suggests era-specific issues
        if loyo_year_briers and len(loyo_year_briers) >= 3:
            worst_year = max(loyo_year_briers, key=loyo_year_briers.get)
            mean_brier = sum(loyo_year_briers.values()) / len(loyo_year_briers)
            worst_brier = loyo_year_briers[worst_year]
            if worst_brier > mean_brier * 1.2:
                h = self.add(
                    title=f"Investigate poor performance in {worst_year}",
                    rationale=(
                        f"LOYO Brier for {worst_year} ({worst_brier:.4f}) is "
                        f">20% worse than mean ({mean_brier:.4f}). "
                        f"Check for data quality issues or regime change."
                    ),
                    expected_impact=0.002,
                    priority="medium",
                    tags=["auto_generated", "loyo_outlier"],
                    source="auto_diagnostic",
                )
                generated.append(h)

        return generated

    def summary(self) -> str:
        """Human-readable summary of the hypothesis registry."""
        all_h = self._load_all()
        if not all_h:
            return "No hypotheses registered yet."

        by_status: Dict[str, int] = {}
        for h in all_h:
            by_status[h.status] = by_status.get(h.status, 0) + 1

        lines = [
            f"Hypothesis Registry: {len(all_h)} hypotheses",
            f"  Status breakdown: {by_status}",
        ]

        confirmed = [h for h in all_h if h.status == "confirmed"]
        if confirmed:
            total_impact = sum(h.actual_impact for h in confirmed)
            lines.append(
                f"  Confirmed: {len(confirmed)} hypotheses, "
                f"total impact: {total_impact:.6f} Brier"
            )

        pending = [h for h in all_h if h.status in ("proposed", "scheduled")]
        if pending:
            lines.append(f"  Pending: {len(pending)} hypotheses to investigate")

        return "\n".join(lines)

    # --- Persistence ---

    def _append(self, h: Hypothesis) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "a") as f:
            f.write(json.dumps(h.to_dict()) + "\n")

    def _update(self, h: Hypothesis) -> None:
        """Update a hypothesis in place (rewrite file)."""
        all_h = self._load_all()
        updated = False
        for i, existing in enumerate(all_h):
            if existing.hypothesis_id == h.hypothesis_id:
                all_h[i] = h
                updated = True
                break

        if not updated:
            all_h.append(h)

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            for rec in all_h:
                f.write(json.dumps(rec.to_dict()) + "\n")

    def _load_all(self) -> List[Hypothesis]:
        if not self.registry_path.exists():
            return []
        records = []
        with open(self.registry_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(Hypothesis.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed hypothesis line: %s", e)
        return records
