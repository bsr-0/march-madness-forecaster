"""Escalation policies and role-based access control for governance.

Extends the governance framework with:
- Time-bound approval expiry
- Automatic escalation when approvals are not actioned
- Role-based access control (RBAC) for pipeline actions
- Escalation chain: operator -> lead -> admin

Implements Agent Directive V7 S21 (governance framework).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Role(Enum):
    """Pipeline roles with increasing privilege."""
    VIEWER = "viewer"
    OPERATOR = "operator"
    LEAD = "lead"
    ADMIN = "admin"


# Role hierarchy: higher roles inherit lower permissions
ROLE_HIERARCHY: Dict[Role, int] = {
    Role.VIEWER: 0,
    Role.OPERATOR: 1,
    Role.LEAD: 2,
    Role.ADMIN: 3,
}


@dataclass
class RBACPolicy:
    """Role-based access control policy for a pipeline action."""
    action_pattern: str
    minimum_role: Role
    description: str = ""


DEFAULT_RBAC_POLICIES: List[RBACPolicy] = [
    RBACPolicy("view_*", Role.VIEWER, "View pipeline state and results"),
    RBACPolicy("modify_config.*", Role.OPERATOR, "Modify pipeline configuration"),
    RBACPolicy("retrain_model", Role.OPERATOR, "Retrain models"),
    RBACPolicy("promote_model", Role.LEAD, "Promote model to production"),
    RBACPolicy("submit_predictions", Role.LEAD, "Submit predictions to Kaggle"),
    RBACPolicy("override_threshold.*", Role.LEAD, "Override quality thresholds"),
    RBACPolicy("deploy_production", Role.ADMIN, "Deploy to production"),
    RBACPolicy("modify_governance", Role.ADMIN, "Modify governance rules"),
]


@dataclass
class EscalationLevel:
    """A level in the escalation chain."""
    role: Role
    timeout_hours: float  # hours before escalating to next level
    notification_method: str = "log"  # "log", "email", "slack"


DEFAULT_ESCALATION_CHAIN: List[EscalationLevel] = [
    EscalationLevel(Role.OPERATOR, timeout_hours=4.0),
    EscalationLevel(Role.LEAD, timeout_hours=12.0),
    EscalationLevel(Role.ADMIN, timeout_hours=24.0),
]


@dataclass
class EscalationRecord:
    """Tracks escalation state for a pending approval."""
    request_id: str
    action: str
    created_at: str
    current_level: int = 0  # index into escalation chain
    escalation_history: List[Dict[str, str]] = field(default_factory=list)
    resolved: bool = False


class RBACManager:
    """Role-based access control for pipeline actions."""

    def __init__(
        self,
        policies: Optional[List[RBACPolicy]] = None,
        user_roles: Optional[Dict[str, Role]] = None,
    ) -> None:
        self._policies = policies or list(DEFAULT_RBAC_POLICIES)
        self._user_roles = user_roles or {}

    def assign_role(self, user: str, role: Role) -> None:
        """Assign a role to a user."""
        self._user_roles[user] = role
        logger.info("Assigned role %s to user '%s'", role.value, user)

    def get_role(self, user: str) -> Role:
        """Get a user's role. Defaults to VIEWER."""
        return self._user_roles.get(user, Role.VIEWER)

    def check_permission(self, user: str, action: str) -> bool:
        """Check if a user has permission for an action."""
        import fnmatch

        user_role = self.get_role(user)
        user_level = ROLE_HIERARCHY[user_role]

        for policy in self._policies:
            if fnmatch.fnmatch(action, policy.action_pattern):
                required_level = ROLE_HIERARCHY[policy.minimum_role]
                return user_level >= required_level

        # Default: operators and above can perform unlisted actions
        return user_level >= ROLE_HIERARCHY[Role.OPERATOR]

    def audit_permissions(self, user: str) -> Dict[str, bool]:
        """List all action patterns and whether the user can perform them."""
        return {
            policy.action_pattern: self.check_permission(user, policy.action_pattern)
            for policy in self._policies
        }


class EscalationManager:
    """Manage approval escalation chains.

    When approvals are not actioned within the timeout, automatically
    escalates to the next role in the chain.
    """

    def __init__(
        self,
        chain: Optional[List[EscalationLevel]] = None,
    ) -> None:
        self._chain = chain or list(DEFAULT_ESCALATION_CHAIN)
        self._active: Dict[str, EscalationRecord] = {}

    def track(self, request_id: str, action: str) -> EscalationRecord:
        """Start tracking a new approval request for escalation."""
        record = EscalationRecord(
            request_id=request_id,
            action=action,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._active[request_id] = record
        logger.info(
            "Tracking escalation for request '%s' (action='%s')",
            request_id, action,
        )
        return record

    def check_escalations(self) -> List[EscalationRecord]:
        """Check all active requests and escalate any that have timed out.

        Returns list of requests that were escalated.
        """
        now = datetime.now(timezone.utc)
        escalated: List[EscalationRecord] = []

        for request_id, record in list(self._active.items()):
            if record.resolved:
                continue

            created = datetime.fromisoformat(record.created_at)
            hours_elapsed = (now - created).total_seconds() / 3600

            # Check if current level has timed out
            if record.current_level < len(self._chain):
                level = self._chain[record.current_level]
                if hours_elapsed > level.timeout_hours:
                    # Escalate to next level
                    record.escalation_history.append({
                        "from_level": record.current_level,
                        "to_level": record.current_level + 1,
                        "timestamp": now.isoformat(),
                        "reason": f"Timeout after {level.timeout_hours}h",
                    })
                    record.current_level += 1
                    escalated.append(record)

                    if record.current_level < len(self._chain):
                        next_level = self._chain[record.current_level]
                        logger.warning(
                            "Escalated request '%s' to %s (level %d)",
                            request_id, next_level.role.value,
                            record.current_level,
                        )
                    else:
                        logger.error(
                            "Request '%s' has exhausted all escalation levels",
                            request_id,
                        )

        return escalated

    def resolve(self, request_id: str) -> bool:
        """Mark a request as resolved (approved or denied)."""
        record = self._active.get(request_id)
        if record:
            record.resolved = True
            return True
        return False

    def pending_count(self) -> int:
        """Number of unresolved escalation-tracked requests."""
        return sum(1 for r in self._active.values() if not r.resolved)

    def summary(self) -> str:
        """Human-readable escalation status."""
        active = [r for r in self._active.values() if not r.resolved]
        lines = [
            "Escalation Status",
            "=" * 50,
            f"Active requests: {len(active)}",
            f"Resolved: {sum(1 for r in self._active.values() if r.resolved)}",
        ]

        for record in active:
            level_name = (
                self._chain[record.current_level].role.value
                if record.current_level < len(self._chain)
                else "EXHAUSTED"
            )
            lines.append(
                f"  [{record.request_id}] {record.action} "
                f"level={record.current_level} ({level_name}) "
                f"escalations={len(record.escalation_history)}"
            )

        return "\n".join(lines)
