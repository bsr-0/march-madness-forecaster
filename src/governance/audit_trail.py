"""Minimal governance audit log.

The production pipeline only needs lightweight in-memory logging hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GovernanceAuditLog:
    """In-memory audit log used by pipeline governance hooks."""

    entries: List[Dict[str, Any]] = field(default_factory=list)

    def log_action(self, action: str, actor: str, details: Dict[str, Any]) -> None:
        self.entries.append(
            {
                "timestamp": _utc_now(),
                "kind": "action",
                "action": action,
                "actor": actor,
                "details": dict(details or {}),
            }
        )

    def log_compliance_check(self, checkpoint: str, status: str, details: str) -> None:
        self.entries.append(
            {
                "timestamp": _utc_now(),
                "kind": "compliance_check",
                "checkpoint": checkpoint,
                "status": status,
                "details": details,
            }
        )
