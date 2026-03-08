"""Append-only governance audit log."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GovernanceAuditLog:
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
