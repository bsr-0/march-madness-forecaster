"""Tests for governance framework (S21 compliance).

Covers:
  - Decision authority matrix and approval policies
  - Approval request/approve/deny workflow
  - Governance audit trail (append-only, queryable)
  - Gate enforcement for high-stakes actions
"""

import json

import pytest


class TestDecisionAuthority:
    """S21: Decision authority matrix and approval gates."""

    def test_default_matrix_has_all_actions(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        matrix = authority.authority_matrix
        for action in ActionType:
            assert action in matrix

    def test_submit_kaggle_requires_approval(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        assert authority.requires_approval(ActionType.SUBMIT_KAGGLE)

    def test_change_feature_set_auto_approves(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        assert not authority.requires_approval(ActionType.CHANGE_FEATURE_SET)

    def test_request_approval_creates_pending(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            ApprovalStatus,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        request = authority.request_approval(
            ActionType.SUBMIT_KAGGLE,
            actor="alice",
            justification="Tournament submission",
        )
        assert request.status == ApprovalStatus.PENDING.value
        assert request.actor == "alice"
        assert request.request_id

    def test_auto_approve_for_non_required_actions(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            ApprovalStatus,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        request = authority.request_approval(
            ActionType.CHANGE_FEATURE_SET,
            actor="bob",
            justification="Added momentum feature",
        )
        assert request.status == ApprovalStatus.AUTO_APPROVED.value

    def test_approve_pending_request(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            ApprovalStatus,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        request = authority.request_approval(
            ActionType.SUBMIT_KAGGLE,
            actor="alice",
            justification="Final submission",
        )
        approved = authority.approve(request.request_id, approver="carol")
        assert approved.status == ApprovalStatus.APPROVED.value
        assert approved.approver == "carol"

    def test_deny_pending_request(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            ApprovalStatus,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        request = authority.request_approval(
            ActionType.SUBMIT_KAGGLE,
            actor="alice",
            justification="Premature submission",
        )
        denied = authority.deny(
            request.request_id, approver="carol", reason="Not ready yet"
        )
        assert denied.status == ApprovalStatus.DENIED.value
        assert denied.denial_reason == "Not ready yet"

    def test_check_gate_blocks_unapproved(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        assert not authority.check_gate(ActionType.SUBMIT_KAGGLE, actor="alice")

    def test_check_gate_passes_after_approval(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        request = authority.request_approval(
            ActionType.SUBMIT_KAGGLE, actor="alice", justification="Ready"
        )
        authority.approve(request.request_id, approver="carol")
        assert authority.check_gate(ActionType.SUBMIT_KAGGLE, actor="alice")

    def test_pending_requests_listed(self, tmp_path):
        from src.governance.decision_authority import (
            ActionType,
            DecisionAuthority,
        )

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        authority.request_approval(
            ActionType.SUBMIT_KAGGLE, actor="alice", justification="Sub 1"
        )
        authority.request_approval(
            ActionType.DEPLOY_PRODUCTION, actor="bob", justification="Deploy"
        )
        pending = authority.pending_requests()
        assert len(pending) == 2

    def test_summary_includes_matrix(self, tmp_path):
        from src.governance.decision_authority import DecisionAuthority

        authority = DecisionAuthority(approvals_dir=str(tmp_path / "approvals"))
        summary = authority.summary()
        assert "Decision Authority Matrix" in summary
        assert "submit_kaggle" in summary


class TestGovernanceAuditTrail:
    """S21: Immutable governance audit trail."""

    def test_log_and_query(self, tmp_path):
        from src.governance.audit_trail import GovernanceAuditTrail, GovernanceRecord

        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(
            GovernanceRecord(
                action="submit_kaggle",
                event_type="approval",
                actor="alice",
                decision="approved",
                approver="carol",
                justification="Tournament ready",
            )
        )
        records = trail.query(action="submit_kaggle")
        assert len(records) == 1
        assert records[0].actor == "alice"
        assert records[0].decision == "approved"

    def test_audit_trail_is_append_only(self, tmp_path):
        from src.governance.audit_trail import GovernanceAuditTrail, GovernanceRecord

        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="a", actor="x", decision="approved"))
        trail.log(GovernanceRecord(action="b", actor="y", decision="denied"))

        # Read raw file — should be 2 lines
        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_query_by_actor(self, tmp_path):
        from src.governance.audit_trail import GovernanceAuditTrail, GovernanceRecord

        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="a", actor="alice", decision="approved"))
        trail.log(GovernanceRecord(action="b", actor="bob", decision="denied"))
        trail.log(GovernanceRecord(action="c", actor="alice", decision="approved"))

        alice_records = trail.query(actor="alice")
        assert len(alice_records) == 2

    def test_count_by_action(self, tmp_path):
        from src.governance.audit_trail import GovernanceAuditTrail, GovernanceRecord

        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="submit", actor="a", decision="ok"))
        trail.log(GovernanceRecord(action="submit", actor="b", decision="ok"))
        trail.log(GovernanceRecord(action="rollback", actor="c", decision="ok"))

        counts = trail.count_by_action()
        assert counts["submit"] == 2
        assert counts["rollback"] == 1

    def test_summary(self, tmp_path):
        from src.governance.audit_trail import GovernanceAuditTrail, GovernanceRecord

        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="test", actor="user", decision="ok"))
        summary = trail.summary()
        assert "Governance Audit Trail" in summary
        assert "Total records: 1" in summary
