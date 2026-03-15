"""Append-only governance audit log."""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GovernanceRecord:
    """A single governance event."""

    action: str
    actor: str
    decision: str
    event_type: str = ""
    approver: str = ""
    justification: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GovernanceAuditTrail:
    """Append-only, queryable governance audit trail."""

    def __init__(self, audit_path: str = "data/governance_audit.jsonl"):
        self.path = Path(audit_path)

    def log(self, record: GovernanceRecord) -> None:
        """Append a governance record to the audit trail."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "action": record.action,
            "actor": record.actor,
            "decision": record.decision,
            "event_type": record.event_type,
            "approver": record.approver,
            "justification": record.justification,
            "timestamp": record.timestamp,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def query(
        self,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[GovernanceRecord]:
        """Query the audit trail with optional filters."""
        records = self._load_all()
        if action:
            records = [r for r in records if r.action == action]
        if actor:
            records = [r for r in records if r.actor == actor]
        if since:
            records = [r for r in records if r.timestamp >= since]
        return records

    def count_by_action(self) -> Dict[str, int]:
        """Return counts of records grouped by action."""
        return dict(Counter(r.action for r in self._load_all()))

    def summary(self) -> str:
        """Return a human-readable summary of the audit trail."""
        records = self._load_all()
        lines = [
            "Governance Audit Trail",
            f"Total records: {len(records)}",
        ]
        if records:
            counts = Counter(r.action for r in records)
            for action, count in counts.most_common():
                lines.append(f"  {action}: {count}")
        return "\n".join(lines)

    def _load_all(self) -> List[GovernanceRecord]:
        if not self.path.exists():
            return []
        records: List[GovernanceRecord] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(GovernanceRecord(
                        action=data.get("action", ""),
                        actor=data.get("actor", ""),
                        decision=data.get("decision", ""),
                        event_type=data.get("event_type", ""),
                        approver=data.get("approver", ""),
                        justification=data.get("justification", ""),
                        timestamp=data.get("timestamp", ""),
                    ))
                except json.JSONDecodeError:
                    continue
        return records


class GovernanceAuditLog:
    """Legacy audit log interface (dict-based)."""

    def __init__(self, path: str = "data/governance_audit.jsonl"):
        self.path = Path(path)

    def log_action(self, action: str, actor: str, details: Dict[str, Any] = None) -> None:
        """Log an arbitrary governance action."""
        entry = {
            "type": "action",
            "action": action,
            "actor": actor,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append(entry)

    def log_approval(self, request_id: str, action: str, status: str,
                     reviewer: str, notes: str = "") -> None:
        """Log an approval/rejection decision."""
        entry = {
            "type": "approval",
            "request_id": request_id,
            "action": action,
            "status": status,
            "reviewer": reviewer,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append(entry)

    def log_compliance_check(self, checkpoint: str, status: str,
                             details: str = "", stage: str = "") -> None:
        """Log a compliance check result."""
        entry = {
            "type": "compliance",
            "checkpoint": checkpoint,
            "status": status,
            "details": details,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append(entry)

    def query(self, action_type: Optional[str] = None,
              since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query the audit log with optional filters."""
        entries = self._load_all()
        if action_type:
            entries = [e for e in entries if e.get("type") == action_type]
        if since:
            entries = [e for e in entries if e.get("timestamp", "") >= since]
        return entries

    def _append(self, entry: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries
