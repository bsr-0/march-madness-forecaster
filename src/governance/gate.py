"""Unified governance gate composing authority matrix, approval gate, and audit trail.

Consolidates three overlapping governance systems into a single entry point
for pipeline governance checks. Actions are evaluated against the authority
matrix; auto-approvable actions proceed immediately, while others create
pending approval requests.

Usage:
    from src.governance.gate import GovernanceGate

    gate = GovernanceGate()
    gate.check_or_request("retrain_model", evidence={"loyo_regression": False})
    # Auto-approved if evidence satisfies conditions

    gate.check_or_request("submit_predictions", evidence={})
    # Raises GovernanceApprovalRequired if human approval needed
"""

import logging
from typing import Any, Dict, Optional

from .approval_gate import ApprovalGate, ApprovalRequest
from .audit_trail import GovernanceAuditLog
from .authority_matrix import AuthorityMatrix

logger = logging.getLogger(__name__)


class GovernanceGate:
    """Unified governance gate for pipeline action approval.

    Composes:
    - AuthorityMatrix: glob-pattern rule matching with auto-approve conditions
    - ApprovalGate: JSONL append-only ledger for request persistence
    - GovernanceAuditLog: immutable audit trail for all decisions
    """

    def __init__(
        self,
        authority_matrix: Optional[AuthorityMatrix] = None,
        approval_gate: Optional[ApprovalGate] = None,
        audit_log: Optional[GovernanceAuditLog] = None,
        approval_timeout_hours: float = 24.0,
    ) -> None:
        self._matrix = authority_matrix or AuthorityMatrix()
        self._gate = approval_gate or ApprovalGate()
        self._audit = audit_log or GovernanceAuditLog()
        self._timeout_hours = approval_timeout_hours

    def check_or_request(
        self,
        action: str,
        evidence: Dict[str, Any],
        actor: str = "pipeline",
    ) -> bool:
        """Check if an action can proceed or create a pending approval request.

        Returns True if the action is auto-approved. Raises
        GovernanceApprovalRequired if human approval is needed.
        """
        from ..exceptions import GovernanceApprovalRequired

        rule = self._matrix.get_rule(action)

        # No matching rule -- default allow for unknown low-risk actions
        if rule is None:
            self._audit.log_action(
                action=action,
                actor=actor,
                details={"decision": "allowed", "reason": "no_matching_rule"},
            )
            logger.info("Governance: action '%s' allowed (no matching rule)", action)
            return True

        # Check auto-approve conditions
        if self._matrix.can_auto_approve(action, evidence):
            self._audit.log_action(
                action=action,
                actor=actor,
                details={
                    "decision": "auto_approved",
                    "rule": rule.action_pattern,
                    "risk_level": rule.risk_level,
                    "evidence": evidence,
                },
            )
            logger.info(
                "Governance: action '%s' auto-approved (rule=%s, risk=%s)",
                action, rule.action_pattern, rule.risk_level,
            )
            return True

        # Requires human approval -- create request
        request = ApprovalRequest(
            action=action,
            requester=actor,
            description=f"Pipeline requests approval for '{action}'",
            risk_level=rule.risk_level,
            evidence=evidence,
        )
        request_id = self._gate.request_approval(request)

        self._audit.log_action(
            action=action,
            actor=actor,
            details={
                "decision": "pending_approval",
                "request_id": request_id,
                "risk_level": rule.risk_level,
                "required_approver": rule.required_approver,
                "timeout_hours": self._timeout_hours,
            },
        )

        logger.warning(
            "Governance: action '%s' requires %s approval (request_id=%s, timeout=%.0fh)",
            action, rule.required_approver, request_id, self._timeout_hours,
        )

        raise GovernanceApprovalRequired(
            f"Action '{action}' requires {rule.required_approver} approval. "
            f"Request ID: {request_id}. Timeout: {self._timeout_hours}h. "
            f"Use 'governance approve {request_id}' to approve.",
            request_id=request_id,
        )

    def approve(self, request_id: str, reviewer: str, notes: str = "") -> bool:
        """Approve a pending request."""
        success = self._gate.approve(request_id, reviewer, notes)
        if success:
            self._audit.log_approval(
                request_id=request_id,
                action="",
                status="approved",
                reviewer=reviewer,
                notes=notes,
            )
        return success

    def deny(self, request_id: str, reviewer: str, notes: str = "") -> bool:
        """Deny a pending request."""
        success = self._gate.reject(request_id, reviewer, notes)
        if success:
            self._audit.log_approval(
                request_id=request_id,
                action="",
                status="rejected",
                reviewer=reviewer,
                notes=notes,
            )
        return success

    def pending_requests(self):
        """Return all pending approval requests."""
        return self._gate.pending_requests()

    def status(self) -> str:
        """Summary of governance state."""
        pending = self._gate.pending_requests()
        rules = self._matrix.rules
        lines = [
            "Governance Status",
            "=" * 50,
            f"Authority rules: {len(rules)}",
        ]
        for rule in rules:
            lines.append(
                f"  {rule.action_pattern:<30s} risk={rule.risk_level:<10s} "
                f"approver={rule.required_approver}"
            )
        lines.append("")
        lines.append(f"Pending requests: {len(pending)}")
        for req in pending:
            lines.append(
                f"  [{req.request_id}] {req.action} "
                f"(risk={req.risk_level}, by={req.requester}, at={req.timestamp[:19]})"
            )
        return "\n".join(lines)
