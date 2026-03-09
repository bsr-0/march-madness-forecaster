"""Multi-agent conflict resolution and dissent registry.

Implements Agent Directive V7 S22 (conflict resolution protocol):
  - Conflict detection from contradictory agent messages
  - Resolution hierarchy: safety > empirical evidence > orchestrator
  - Audit Agent absolute veto on safety issues (S22.3)
  - Dissent registry for agents to file objections (S22.4)

Usage:
    from src.agents.conflict import ConflictResolver, DissentRegistry

    resolver = ConflictResolver()
    conflicts = resolver.detect_conflicts(bus.all_messages())
    for c in conflicts:
        resolver.resolve(c, bus.all_messages())

    registry = DissentRegistry()
    registry.file_dissent(DissentRecord(
        agent_role=AgentRole.MODELER,
        reasoning="Audit veto is based on marginal PSI threshold",
    ))
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

from . import AgentMessage, AgentRole, MessageType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conflict types
# ---------------------------------------------------------------------------


class ConflictCategory(str, Enum):
    """Categories of inter-agent conflict (Directive V7 S22.1)."""

    PRIORITY = "priority"  # Disagreement about what to work on next
    METHOD = "method"  # Disagreement about approach/algorithm
    RESOURCE = "resource"  # Contention over shared resources
    SAFETY = "safety"  # Safety/integrity concern (triggers audit veto)


@dataclass
class Conflict:
    """A detected conflict between agents."""

    conflict_id: str = ""
    category: ConflictCategory = ConflictCategory.METHOD
    agents_involved: List[str] = field(default_factory=list)
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.conflict_id:
            self.conflict_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


# ---------------------------------------------------------------------------
# Conflict Resolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Detect and resolve inter-agent conflicts per S22 hierarchy.

    Resolution hierarchy (S22.2):
        1. Safety conflicts → Audit Agent veto (absolute, S22.3)
        2. Empirical conflicts → Higher-evidence position wins
        3. Priority/resource → Orchestrator decides
    """

    def detect_conflicts(
        self, messages: List[AgentMessage]
    ) -> List[Conflict]:
        """Scan messages for contradictory agent positions.

        Detects:
          - VETO messages (safety conflict)
          - Contradictory audit pass/fail signals
          - Conflicting recommendations from different agents
        """
        conflicts: List[Conflict] = []

        # Check for audit veto vs. other agents claiming success
        veto_msgs = [m for m in messages if m.msg_type == MessageType.VETO]
        success_msgs = [
            m for m in messages
            if m.msg_type in (
                MessageType.MODEL_READY,
                MessageType.FEATURES_READY,
                MessageType.DATA_READY,
            )
        ]

        if veto_msgs and success_msgs:
            involved = list(
                {m.sender.value for m in veto_msgs}
                | {m.sender.value for m in success_msgs}
            )
            conflicts.append(
                Conflict(
                    category=ConflictCategory.SAFETY,
                    agents_involved=sorted(involved),
                    description=(
                        f"Audit agent vetoed pipeline but "
                        f"{len(success_msgs)} agents reported success"
                    ),
                    evidence={
                        "veto_reasons": [
                            m.payload.get("reason", "unknown")
                            for m in veto_msgs
                        ],
                        "success_agents": [
                            m.sender.value for m in success_msgs
                        ],
                    },
                )
            )

        # Check for conflicting method recommendations
        conflict_msgs = [
            m for m in messages if m.msg_type == MessageType.CONFLICT
        ]
        for msg in conflict_msgs:
            conflicts.append(
                Conflict(
                    category=ConflictCategory.METHOD,
                    agents_involved=[msg.sender.value],
                    description=msg.payload.get("description", "Method conflict"),
                    evidence=msg.payload,
                )
            )

        return conflicts

    def resolve(
        self, conflict: Conflict, messages: List[AgentMessage]
    ) -> Conflict:
        """Resolve a conflict using the S22 hierarchy.

        Returns the conflict with resolution and resolved_by filled in.
        """
        if conflict.category == ConflictCategory.SAFETY:
            # S22.3: Audit Agent veto is absolute
            conflict.resolution = (
                "Audit Agent veto upheld — safety concern must be "
                "addressed before pipeline can proceed"
            )
            conflict.resolved_by = AgentRole.AUDITOR.value
            logger.warning(
                "Safety conflict resolved by Audit Agent veto: %s",
                conflict.description,
            )

        elif conflict.category == ConflictCategory.METHOD:
            # S22.2: Empirical evidence wins
            conflict.resolution = self._resolve_by_evidence(
                conflict, messages
            )
            conflict.resolved_by = "empirical_evidence"

        elif conflict.category in (
            ConflictCategory.PRIORITY,
            ConflictCategory.RESOURCE,
        ):
            # Orchestrator decides
            conflict.resolution = (
                "Orchestrator resolved — "
                + conflict.evidence.get("orchestrator_decision", "deferred")
            )
            conflict.resolved_by = AgentRole.ORCHESTRATOR.value

        return conflict

    def check_audit_veto(
        self, messages: List[AgentMessage]
    ) -> Optional[AgentMessage]:
        """Check if the Audit Agent has issued a veto.

        Returns the veto message if found. The veto is absolute per S22.3 —
        no other agent can override it.
        """
        for msg in messages:
            if (
                msg.msg_type == MessageType.VETO
                and msg.sender == AgentRole.AUDITOR
            ):
                return msg
        return None

    def _resolve_by_evidence(
        self,
        conflict: Conflict,
        messages: List[AgentMessage],
    ) -> str:
        """Resolve a method conflict by comparing empirical evidence."""
        evidence = conflict.evidence

        # Compare Brier scores if available
        brier_scores = {}
        for msg in messages:
            brier = msg.payload.get("brier_score")
            if brier is not None:
                brier_scores[msg.sender.value] = brier

        if brier_scores:
            best_agent = min(brier_scores, key=brier_scores.get)  # type: ignore[arg-type]
            return (
                f"Resolved by empirical evidence — {best_agent} has best "
                f"Brier score ({brier_scores[best_agent]:.4f})"
            )

        return (
            "Resolved by evidence review — insufficient data for "
            "quantitative comparison, deferring to orchestrator"
        )


# ---------------------------------------------------------------------------
# Dissent Registry (S22.4)
# ---------------------------------------------------------------------------


DEFAULT_DISSENT_PATH = "data/governance/dissent_registry.jsonl"


@dataclass
class DissentRecord:
    """A dissent filed by an agent disagreeing with a decision.

    Per S22.4, agents can register formal disagreement for future review.
    """

    dissent_id: str = ""
    agent_role: str = ""
    conflict_id: str = ""
    reasoning: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    status: str = "open"  # open, reviewed
    reviewer: str = ""
    review_notes: str = ""

    def __post_init__(self) -> None:
        if not self.dissent_id:
            self.dissent_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DissentRegistry:
    """Append-only registry for agent dissent records (S22.4).

    Follows the same JSONL pattern as ExperimentRegistry and
    GovernanceAuditTrail for consistency.
    """

    def __init__(self, path: str = DEFAULT_DISSENT_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def file_dissent(self, record: DissentRecord) -> str:
        """Append a dissent record. Returns the dissent_id."""
        with open(self._path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.info(
            "Dissent filed by %s (id=%s): %s",
            record.agent_role,
            record.dissent_id,
            record.reasoning[:80],
        )
        return record.dissent_id

    def query(
        self,
        agent: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DissentRecord]:
        """Query dissent records with optional filters."""
        records = self._load_all()
        if agent:
            records = [r for r in records if r.agent_role == agent]
        if status:
            records = [r for r in records if r.status == status]
        return records

    def open_dissents(self) -> List[DissentRecord]:
        """Return all unreviewed dissent records."""
        return self.query(status="open")

    def review(
        self, dissent_id: str, reviewer: str, notes: str = ""
    ) -> DissentRecord:
        """Mark a dissent record as reviewed.

        Rewrites the JSONL file with the updated record. In production
        this would use a proper database; for our JSONL approach we
        reload, update, and rewrite.
        """
        records = self._load_all()
        target = None
        for rec in records:
            if rec.dissent_id == dissent_id:
                rec.status = "reviewed"
                rec.reviewer = reviewer
                rec.review_notes = notes
                target = rec
                break

        if target is None:
            raise ValueError(f"Dissent {dissent_id} not found")

        # Rewrite file
        with open(self._path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")

        logger.info("Dissent %s reviewed by %s", dissent_id, reviewer)
        return target

    def _load_all(self) -> List[DissentRecord]:
        """Load all records from JSONL."""
        if not self._path.exists():
            return []
        records: List[DissentRecord] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(DissentRecord(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records
