"""Decision authority matrix and approval gate for pipeline actions.

Defines which pipeline actions require human approval, maps actions to
required roles, and enforces approval gates before high-stakes operations.

Implements Agent Directive V7 S21 (decision authority matrix, approval gates).

Usage:
    from src.governance.decision_authority import DecisionAuthority, ActionType

    authority = DecisionAuthority()
    authority.request_approval(ActionType.SUBMIT_KAGGLE, actor="alice",
                               justification="Final tournament submission")
    # Creates pending approval; blocks until approved or raises
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Pipeline actions that may require governance approval."""

    SUBMIT_KAGGLE = "submit_kaggle"
    CHANGE_ENSEMBLE_WEIGHTS = "change_ensemble_weights"
    CHANGE_CALIBRATION_METHOD = "change_calibration_method"
    CHANGE_FEATURE_SET = "change_feature_set"
    OVERRIDE_CIRCUIT_BREAKER = "override_circuit_breaker"
    EMERGENCY_ROLLBACK = "emergency_rollback"
    DEPLOY_PRODUCTION = "deploy_production"
    MODIFY_HYPERPARAMETERS = "modify_hyperparameters"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    AUTO_APPROVED = "auto_approved"
    EXPIRED = "expired"


@dataclass
class ApprovalPolicy:
    """Policy for a specific action type."""

    action: ActionType
    requires_approval: bool = True
    required_role: str = "operator"
    auto_approve_if: Optional[str] = None  # condition description
    description: str = ""


@dataclass
class ApprovalRequest:
    """A request for human approval of a pipeline action."""

    request_id: str = ""
    action: str = ""
    actor: str = ""
    justification: str = ""
    timestamp: str = ""
    status: str = ApprovalStatus.PENDING.value
    approver: Optional[str] = None
    approval_timestamp: Optional[str] = None
    denial_reason: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# Default authority matrix mapping actions to policies
DEFAULT_AUTHORITY_MATRIX: Dict[ActionType, ApprovalPolicy] = {
    ActionType.SUBMIT_KAGGLE: ApprovalPolicy(
        action=ActionType.SUBMIT_KAGGLE,
        requires_approval=True,
        required_role="operator",
        description="Submit predictions to Kaggle competition",
    ),
    ActionType.CHANGE_ENSEMBLE_WEIGHTS: ApprovalPolicy(
        action=ActionType.CHANGE_ENSEMBLE_WEIGHTS,
        requires_approval=True,
        required_role="ml_lead",
        auto_approve_if="delta < 0.05 for all weights",
        description="Modify ensemble model weight distribution",
    ),
    ActionType.CHANGE_CALIBRATION_METHOD: ApprovalPolicy(
        action=ActionType.CHANGE_CALIBRATION_METHOD,
        requires_approval=True,
        required_role="ml_lead",
        description="Switch calibration method (temperature/Platt/isotonic)",
    ),
    ActionType.CHANGE_FEATURE_SET: ApprovalPolicy(
        action=ActionType.CHANGE_FEATURE_SET,
        requires_approval=False,
        required_role="researcher",
        auto_approve_if="Brier improvement >= 0.001 on LOYO",
        description="Add or remove features from the active set",
    ),
    ActionType.OVERRIDE_CIRCUIT_BREAKER: ApprovalPolicy(
        action=ActionType.OVERRIDE_CIRCUIT_BREAKER,
        requires_approval=True,
        required_role="operator",
        description="Manually override a tripped circuit breaker",
    ),
    ActionType.EMERGENCY_ROLLBACK: ApprovalPolicy(
        action=ActionType.EMERGENCY_ROLLBACK,
        requires_approval=True,
        required_role="operator",
        description="Rollback to a previous model/config version",
    ),
    ActionType.DEPLOY_PRODUCTION: ApprovalPolicy(
        action=ActionType.DEPLOY_PRODUCTION,
        requires_approval=True,
        required_role="operator",
        description="Deploy model to production/live prediction",
    ),
    ActionType.MODIFY_HYPERPARAMETERS: ApprovalPolicy(
        action=ActionType.MODIFY_HYPERPARAMETERS,
        requires_approval=False,
        required_role="researcher",
        auto_approve_if="Within Optuna search bounds",
        description="Change model hyperparameters",
    ),
}


class DecisionAuthority:
    """Manages the decision authority matrix and approval workflow.

    Tracks which actions require approval, creates approval requests,
    and enforces gates before high-stakes operations.
    """

    def __init__(
        self,
        authority_matrix: Optional[Dict[ActionType, ApprovalPolicy]] = None,
        approvals_dir: str = "data/governance/approvals",
    ) -> None:
        self._matrix = authority_matrix or dict(DEFAULT_AUTHORITY_MATRIX)
        self._approvals_dir = Path(approvals_dir)
        self._approvals_dir.mkdir(parents=True, exist_ok=True)

    @property
    def authority_matrix(self) -> Dict[ActionType, ApprovalPolicy]:
        """Return the current authority matrix."""
        return dict(self._matrix)

    def requires_approval(self, action: ActionType) -> bool:
        """Check if an action requires human approval."""
        policy = self._matrix.get(action)
        if policy is None:
            return True  # Default to requiring approval for unknown actions
        return policy.requires_approval

    def get_policy(self, action: ActionType) -> Optional[ApprovalPolicy]:
        """Get the approval policy for an action."""
        return self._matrix.get(action)

    def request_approval(
        self,
        action: ActionType,
        actor: str,
        justification: str,
        metadata: Optional[Dict] = None,
    ) -> ApprovalRequest:
        """Create an approval request for a pipeline action.

        If the action doesn't require approval, auto-approves immediately.

        Returns:
            ApprovalRequest with status set appropriately.
        """
        import uuid

        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            request_id=str(uuid.uuid4())[:8],
            action=action.value,
            actor=actor,
            justification=justification,
            timestamp=now.isoformat(),
            metadata=metadata or {},
        )

        policy = self._matrix.get(action)
        if policy is None or not policy.requires_approval:
            request.status = ApprovalStatus.AUTO_APPROVED.value
            request.approver = "system"
            request.approval_timestamp = now.isoformat()
            logger.info(
                "Auto-approved %s for %s: %s",
                action.value,
                actor,
                justification,
            )
        else:
            request.status = ApprovalStatus.PENDING.value
            logger.info(
                "Approval requested for %s by %s (role=%s required): %s",
                action.value,
                actor,
                policy.required_role,
                justification,
            )

        self._save_request(request)
        return request

    def approve(
        self,
        request_id: str,
        approver: str,
    ) -> ApprovalRequest:
        """Approve a pending request."""
        request = self._load_request(request_id)
        if request is None:
            raise ValueError(f"Approval request {request_id} not found")
        if request.status != ApprovalStatus.PENDING.value:
            raise ValueError(
                f"Request {request_id} is {request.status}, not pending"
            )

        request.status = ApprovalStatus.APPROVED.value
        request.approver = approver
        request.approval_timestamp = datetime.now(timezone.utc).isoformat()
        self._save_request(request)

        logger.info(
            "Approved %s (id=%s) by %s",
            request.action,
            request_id,
            approver,
        )
        return request

    def deny(
        self,
        request_id: str,
        approver: str,
        reason: str = "",
    ) -> ApprovalRequest:
        """Deny a pending request."""
        request = self._load_request(request_id)
        if request is None:
            raise ValueError(f"Approval request {request_id} not found")
        if request.status != ApprovalStatus.PENDING.value:
            raise ValueError(
                f"Request {request_id} is {request.status}, not pending"
            )

        request.status = ApprovalStatus.DENIED.value
        request.approver = approver
        request.approval_timestamp = datetime.now(timezone.utc).isoformat()
        request.denial_reason = reason
        self._save_request(request)

        logger.info(
            "Denied %s (id=%s) by %s: %s",
            request.action,
            request_id,
            approver,
            reason,
        )
        return request

    def check_gate(self, action: ActionType, actor: str) -> bool:
        """Check if an action is approved for the given actor.

        Returns True if the action doesn't require approval or has an
        approved request. Returns False if pending/denied.
        """
        if not self.requires_approval(action):
            return True

        # Check for approved requests for this action
        requests = self._list_requests(action=action.value)
        for req in reversed(requests):  # Most recent first
            if req.status == ApprovalStatus.APPROVED.value:
                return True
            if req.status == ApprovalStatus.AUTO_APPROVED.value:
                return True
        return False

    def pending_requests(self) -> List[ApprovalRequest]:
        """List all pending approval requests."""
        return [
            r
            for r in self._list_requests()
            if r.status == ApprovalStatus.PENDING.value
        ]

    def summary(self) -> str:
        """Human-readable summary of authority matrix and pending requests."""
        lines = [
            "Decision Authority Matrix",
            "=" * 60,
            f"  {'Action':<35s} {'Approval?':<12s} {'Role':<15s}",
            "-" * 60,
        ]
        for action, policy in self._matrix.items():
            approval = "Required" if policy.requires_approval else "Auto"
            lines.append(
                f"  {action.value:<35s} {approval:<12s} {policy.required_role:<15s}"
            )

        pending = self.pending_requests()
        if pending:
            lines.append("")
            lines.append(f"Pending Approvals: {len(pending)}")
            for req in pending:
                lines.append(
                    f"  [{req.request_id}] {req.action} by {req.actor}: {req.justification}"
                )
        else:
            lines.append("\nNo pending approvals.")

        return "\n".join(lines)

    def _save_request(self, request: ApprovalRequest) -> None:
        """Save an approval request to disk."""
        path = self._approvals_dir / f"{request.request_id}.json"
        with open(path, "w") as f:
            json.dump(request.to_dict(), f, indent=2)

    def _load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Load an approval request by ID."""
        path = self._approvals_dir / f"{request_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ApprovalRequest(**data)

    def _list_requests(
        self, action: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """List all approval requests, optionally filtered by action."""
        requests: List[ApprovalRequest] = []
        if not self._approvals_dir.exists():
            return requests
        for path in sorted(self._approvals_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                req = ApprovalRequest(**data)
                if action is None or req.action == action:
                    requests.append(req)
            except (json.JSONDecodeError, TypeError):
                continue
        return requests
