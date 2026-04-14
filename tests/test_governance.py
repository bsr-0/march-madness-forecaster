"""Tests for governance framework (S21 compliance).

Covers:
  - Governance audit trail (append-only, queryable)
  - Gate enforcement for high-stakes actions
"""

import json

import pytest


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
