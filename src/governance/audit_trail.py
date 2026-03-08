"""Immutable governance audit trail for pipeline decisions.

Records every governance action (approval requests, decisions, overrides,
escalations) to an append-only JSONL log with timestamps, actor identities,
justifications, and outcomes.

Implements Agent Directive V7 S21 (governance audit trail).

Usage:
    from src.governance.audit_trail import GovernanceAuditTrail, GovernanceRecord

    trail = GovernanceAuditTrail()
    trail.log(GovernanceRecord(
        action="submit_kaggle",
        actor="alice",
        decision="approved",
        justification="Final pre-tournament submission",
    ))
    recent = trail.query(action="submit_kaggle", limit=5)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_PATH = "data/governance/audit_trail.jsonl"


@dataclass
class GovernanceRecord:
    """A single entry in the governance audit trail."""

    record_id: str = ""
    timestamp: str = ""
    action: str = ""  # ActionType value
    event_type: str = ""  # request, approval, denial, override, escalation, rollback
    actor: str = ""  # Who initiated the action
    decision: str = ""  # approved, denied, escalated, auto_approved
    approver: str = ""  # Who approved/denied (may differ from actor)
    justification: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id:
            import uuid

            self.record_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GovernanceAuditTrail:
    """Append-only JSONL audit trail for governance events.

    All entries are immutable once written. Supports querying by action,
    actor, event type, and time range.
    """

    def __init__(self, audit_path: str = DEFAULT_AUDIT_PATH) -> None:
        self._path = Path(audit_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: GovernanceRecord) -> str:
        """Append a governance record to the audit trail.

        Returns the record_id.
        """
        with open(self._path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(
            "Governance: %s %s by %s → %s",
            record.event_type or record.action,
            record.action,
            record.actor,
            record.decision,
        )
        return record.record_id

    def query(
        self,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[GovernanceRecord]:
        """Query audit trail with optional filters.

        Returns records in reverse chronological order (newest first).
        """
        records = self._load_all()

        if action:
            records = [r for r in records if r.action == action]
        if actor:
            records = [r for r in records if r.actor == actor]
        if event_type:
            records = [r for r in records if r.event_type == event_type]

        return list(reversed(records[-limit:]))

    def count_by_action(self) -> Dict[str, int]:
        """Count governance events grouped by action type."""
        counts: Dict[str, int] = {}
        for rec in self._load_all():
            counts[rec.action] = counts.get(rec.action, 0) + 1
        return counts

    def summary(self) -> str:
        """Human-readable summary of the audit trail."""
        records = self._load_all()
        if not records:
            return "Governance audit trail is empty."

        lines = [
            "Governance Audit Trail",
            "=" * 70,
            f"Total records: {len(records)}",
            "",
            "Events by action:",
        ]
        for action, count in sorted(self.count_by_action().items()):
            lines.append(f"  {action:<35s} {count}")

        lines.append("")
        lines.append("Last 5 events:")
        for rec in records[-5:]:
            ts = rec.timestamp[:19] if rec.timestamp else "?"
            lines.append(
                f"  [{rec.record_id}] {ts} {rec.event_type or rec.action}: "
                f"{rec.action} by {rec.actor} → {rec.decision}"
            )

        return "\n".join(lines)

    def _load_all(self) -> List[GovernanceRecord]:
        """Load all records from the JSONL file."""
        if not self._path.exists():
            return []
        records: List[GovernanceRecord] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(GovernanceRecord(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.debug("Skipping malformed audit record: %s", exc)
        return records
