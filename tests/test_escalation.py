"""Tests for governance escalation and RBAC (S21 governance framework)."""

import pytest
from datetime import datetime, timezone, timedelta

from src.governance.escalation import (
    DEFAULT_ESCALATION_CHAIN,
    DEFAULT_RBAC_POLICIES,
    EscalationLevel,
    EscalationManager,
    EscalationRecord,
    RBACManager,
    RBACPolicy,
    Role,
    ROLE_HIERARCHY,
)


class TestRBACManager:
    def test_default_role_is_viewer(self):
        rbac = RBACManager()
        assert rbac.get_role("unknown_user") == Role.VIEWER

    def test_assign_and_check_role(self):
        rbac = RBACManager()
        rbac.assign_role("alice", Role.OPERATOR)
        assert rbac.get_role("alice") == Role.OPERATOR

    def test_viewer_cannot_modify_config(self):
        rbac = RBACManager()
        rbac.assign_role("viewer_user", Role.VIEWER)
        assert not rbac.check_permission("viewer_user", "modify_config.threshold")

    def test_operator_can_modify_config(self):
        rbac = RBACManager()
        rbac.assign_role("op_user", Role.OPERATOR)
        assert rbac.check_permission("op_user", "modify_config.threshold")

    def test_operator_cannot_submit_predictions(self):
        rbac = RBACManager()
        rbac.assign_role("op_user", Role.OPERATOR)
        assert not rbac.check_permission("op_user", "submit_predictions")

    def test_lead_can_submit_predictions(self):
        rbac = RBACManager()
        rbac.assign_role("lead_user", Role.LEAD)
        assert rbac.check_permission("lead_user", "submit_predictions")

    def test_admin_can_do_everything(self):
        rbac = RBACManager()
        rbac.assign_role("admin_user", Role.ADMIN)
        assert rbac.check_permission("admin_user", "submit_predictions")
        assert rbac.check_permission("admin_user", "deploy_production")
        assert rbac.check_permission("admin_user", "modify_governance")

    def test_lead_cannot_deploy_production(self):
        rbac = RBACManager()
        rbac.assign_role("lead_user", Role.LEAD)
        assert not rbac.check_permission("lead_user", "deploy_production")

    def test_audit_permissions(self):
        rbac = RBACManager()
        rbac.assign_role("op_user", Role.OPERATOR)
        perms = rbac.audit_permissions("op_user")
        assert perms["view_*"] is True
        assert perms["modify_config.*"] is True
        assert perms["submit_predictions"] is False
        assert perms["deploy_production"] is False

    def test_role_hierarchy_ordering(self):
        assert ROLE_HIERARCHY[Role.VIEWER] < ROLE_HIERARCHY[Role.OPERATOR]
        assert ROLE_HIERARCHY[Role.OPERATOR] < ROLE_HIERARCHY[Role.LEAD]
        assert ROLE_HIERARCHY[Role.LEAD] < ROLE_HIERARCHY[Role.ADMIN]

    def test_viewer_can_view(self):
        rbac = RBACManager()
        rbac.assign_role("v", Role.VIEWER)
        assert rbac.check_permission("v", "view_results")


class TestEscalationManager:
    def test_track_new_request(self):
        mgr = EscalationManager()
        record = mgr.track("req1", "submit_predictions")
        assert record.request_id == "req1"
        assert record.current_level == 0
        assert not record.resolved

    def test_resolve_request(self):
        mgr = EscalationManager()
        mgr.track("req1", "test_action")
        assert mgr.resolve("req1")
        assert mgr.pending_count() == 0

    def test_resolve_unknown_returns_false(self):
        mgr = EscalationManager()
        assert not mgr.resolve("nonexistent")

    def test_pending_count(self):
        mgr = EscalationManager()
        mgr.track("req1", "a")
        mgr.track("req2", "b")
        assert mgr.pending_count() == 2
        mgr.resolve("req1")
        assert mgr.pending_count() == 1

    def test_escalation_on_timeout(self):
        chain = [
            EscalationLevel(Role.OPERATOR, timeout_hours=0.0),  # immediate timeout
            EscalationLevel(Role.LEAD, timeout_hours=12.0),
        ]
        mgr = EscalationManager(chain=chain)
        mgr.track("req1", "submit_predictions")
        # Should escalate immediately since timeout is 0
        escalated = mgr.check_escalations()
        assert len(escalated) == 1
        assert escalated[0].current_level == 1

    def test_no_escalation_before_timeout(self):
        chain = [
            EscalationLevel(Role.OPERATOR, timeout_hours=100.0),
        ]
        mgr = EscalationManager(chain=chain)
        mgr.track("req1", "test_action")
        escalated = mgr.check_escalations()
        assert len(escalated) == 0

    def test_resolved_requests_not_escalated(self):
        chain = [
            EscalationLevel(Role.OPERATOR, timeout_hours=0.0),
        ]
        mgr = EscalationManager(chain=chain)
        mgr.track("req1", "test")
        mgr.resolve("req1")
        escalated = mgr.check_escalations()
        assert len(escalated) == 0

    def test_summary(self):
        mgr = EscalationManager()
        mgr.track("req1", "action_a")
        mgr.track("req2", "action_b")
        mgr.resolve("req2")
        summary = mgr.summary()
        assert "Active requests: 1" in summary
        assert "Resolved: 1" in summary

    def test_default_chain_exists(self):
        assert len(DEFAULT_ESCALATION_CHAIN) >= 3
        # Chain should go operator -> lead -> admin
        assert DEFAULT_ESCALATION_CHAIN[0].role == Role.OPERATOR
        assert DEFAULT_ESCALATION_CHAIN[1].role == Role.LEAD
        assert DEFAULT_ESCALATION_CHAIN[2].role == Role.ADMIN
