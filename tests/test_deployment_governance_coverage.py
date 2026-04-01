"""Tests for low-coverage deployment and governance modules.

Covers:
  - src/deployment/model_store.py
  - src/deployment/orchestrator.py
  - src/deployment/shadow_mode.py
  - src/governance/approval_gate.py
  - src/governance/gate.py
  - src/governance/authority_matrix.py
  - src/governance/audit_trail.py
  - src/governance/production_validator.py
  - src/data/manifest_generator.py
  - src/data/models/game_flow.py
  - src/data/loader.py
"""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. ModelStore / ModelVersion
# ---------------------------------------------------------------------------
from src.deployment.model_store import ModelStore, ModelVersion


class TestModelVersion:
    def test_defaults(self):
        v = ModelVersion()
        assert v.version_id == ""
        assert v.model_name == ""
        assert v.brier_score is None
        assert v.is_production is False
        assert v.metadata == {}

    def test_fields(self):
        v = ModelVersion(
            version_id="v1",
            model_name="ensemble",
            brier_score=0.19,
            is_production=True,
            metadata={"key": "val"},
        )
        assert v.version_id == "v1"
        assert v.brier_score == 0.19
        assert v.is_production is True


class TestModelStore:
    def test_init_creates_directory(self, tmp_path):
        store_dir = tmp_path / "models"
        store = ModelStore(store_dir=str(store_dir))
        assert store_dir.exists()

    def test_hash_config_none(self):
        assert ModelStore._hash_config(None) == "noconfig"

    def test_hash_config_dict(self):
        h = ModelStore._hash_config({"a": 1, "b": 2})
        assert isinstance(h, str) and len(h) == 64

    def test_hash_config_dataclass(self):
        @dataclass
        class Cfg:
            lr: float = 0.01
        h = ModelStore._hash_config(Cfg())
        assert len(h) == 64

    def test_hash_config_other(self):
        h = ModelStore._hash_config("some_string")
        assert len(h) == 64

    def test_save_creates_sidecar(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        version = store.save("test_model", {"weights": [1, 2]}, config={"lr": 0.1}, metrics={"brier_score": 0.2})
        assert version.model_name == "test_model"
        assert version.brier_score == 0.2
        assert version.config_hash != ""
        sidecar = Path(store._store_dir) / version.version_id / "metadata.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["model_name"] == "test_model"

    def test_save_without_joblib(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == "joblib":
                raise ImportError("no joblib")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=fake_import):
            version = store.save("no_joblib_model", object())
        assert version.artifact_path == ""

    def test_list_versions_empty(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "empty_store"))
        assert store.list_versions() == []

    def test_list_versions_returns_saved(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        store.save("m1", object(), metrics={"brier_score": 0.1})
        store.save("m2", object(), metrics={"brier_score": 0.2})
        versions = store.list_versions()
        assert len(versions) == 2

    def test_list_versions_filter_by_name(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        store.save("alpha", object())
        store.save("beta", object())
        assert len(store.list_versions("alpha")) == 1

    def test_list_versions_skips_bad_json(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        bad_dir = store._store_dir / "bad_version"
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.json").write_text("NOT JSON")
        assert store.list_versions() == []

    def test_get_production_none(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        store.save("m", object())
        assert store.get_production() is None

    def test_promote_and_get_production(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        v = store.save("m", object())
        promoted = store.promote(v.version_id)
        assert promoted.is_production is True
        prod = store.get_production()
        assert prod is not None
        assert prod.version_id == v.version_id

    def test_promote_demotes_previous(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        v1 = store.save("m", object())
        v2 = store.save("m", object())
        store.promote(v1.version_id)
        store.promote(v2.version_id)
        prod = store.get_production()
        assert prod.version_id == v2.version_id

    def test_promote_not_found(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        with pytest.raises(FileNotFoundError):
            store.promote("nonexistent")

    def test_load_not_found(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        with pytest.raises((FileNotFoundError, RuntimeError)):
            store.load("nonexistent")

    def test_load_round_trip(self, tmp_path):
        store = ModelStore(store_dir=str(tmp_path / "store"))
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")
        v = store.save("rt", {"data": 42}, config={"x": 1})
        artifact, version = store.load(v.version_id)
        assert artifact == {"data": 42}
        assert version.model_name == "rt"


# ---------------------------------------------------------------------------
# 2. ShadowRunner / ShadowResult
# ---------------------------------------------------------------------------
from src.deployment.shadow_mode import ShadowResult, ShadowRunner


class TestShadowResult:
    def test_creation(self):
        sr = ShadowResult(
            primary_brier=0.2,
            shadow_brier=0.18,
            delta=-0.02,
            per_game_diffs=[0.01, -0.02],
            max_divergence=0.05,
            correlation=0.95,
            recommendation="promote_shadow",
        )
        assert sr.delta == -0.02
        assert sr.recommendation == "promote_shadow"


class TestShadowRunner:
    def test_run_no_common_keys(self):
        runner = ShadowRunner()
        result = runner.run({"a": 0.5}, {"b": 0.6}, {"c": 1.0})
        assert math.isnan(result.primary_brier)
        assert result.recommendation == "needs_review"

    def test_run_promote_shadow(self):
        runner = ShadowRunner()
        # Shadow is better (lower Brier) but divergence < 0.15
        primary = {"g1": 0.5, "g2": 0.5, "g3": 0.5, "g4": 0.5}
        shadow = {"g1": 0.6, "g2": 0.4, "g3": 0.6, "g4": 0.4}
        outcomes = {"g1": 1.0, "g2": 0.0, "g3": 1.0, "g4": 0.0}
        result = runner.run(primary, shadow, outcomes)
        assert result.delta < -runner.PROMOTE_THRESHOLD
        assert result.max_divergence < runner.MAX_DIVERGENCE_THRESHOLD
        assert result.recommendation == "promote_shadow"

    def test_run_keep_primary(self):
        runner = ShadowRunner()
        # Both give same predictions
        preds = {"g1": 0.5, "g2": 0.5}
        outcomes = {"g1": 1.0, "g2": 0.0}
        result = runner.run(preds, preds, outcomes)
        assert result.delta == 0.0
        assert result.recommendation == "keep_primary"

    def test_run_needs_review_high_divergence(self):
        runner = ShadowRunner()
        # Shadow is better but with high divergence
        primary = {"g1": 0.5, "g2": 0.5}
        shadow = {"g1": 0.99, "g2": 0.01}  # max divergence = 0.49 > 0.15
        outcomes = {"g1": 1.0, "g2": 0.0}
        result = runner.run(primary, shadow, outcomes)
        if result.delta < -runner.PROMOTE_THRESHOLD and result.max_divergence >= runner.MAX_DIVERGENCE_THRESHOLD:
            assert result.recommendation == "needs_review"

    def test_run_single_game(self):
        runner = ShadowRunner()
        result = runner.run({"g1": 0.5}, {"g1": 0.5}, {"g1": 1.0})
        assert result.correlation == 1.0  # single game fallback

    def test_compare_configs_dicts(self):
        runner = ShadowRunner()
        diffs = runner.compare_configs({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert "b" in diffs
        assert "a" not in diffs

    def test_compare_configs_dataclass(self):
        @dataclass
        class Cfg:
            lr: float = 0.01
            depth: int = 5

        runner = ShadowRunner()
        diffs = runner.compare_configs(Cfg(lr=0.01), Cfg(lr=0.02))
        assert "lr" in diffs

    def test_compare_configs_other_types(self):
        runner = ShadowRunner()
        diffs = runner.compare_configs("string1", "string2")
        assert diffs == {"primary": "string1", "shadow": "string2"}

    def test_compare_configs_mixed_types(self):
        runner = ShadowRunner()
        diffs = runner.compare_configs({"a": 1}, "not_a_dict")
        assert "primary" in diffs and "shadow" in diffs


# ---------------------------------------------------------------------------
# 3. DeploymentOrchestrator
# ---------------------------------------------------------------------------
from src.deployment.orchestrator import DeploymentOrchestrator


class TestDeploymentOrchestrator:
    def _make_orchestrator(self):
        store = MagicMock()
        shadow = MagicMock()
        drift = MagicMock()
        return DeploymentOrchestrator(model_store=store, shadow_runner=shadow, drift_manager=drift), store, shadow, drift

    def test_deploy_shadow_not_found(self):
        orch, store, _, _ = self._make_orchestrator()
        store.list_versions.return_value = []
        result = orch.deploy_shadow("missing_id")
        assert "error" in result

    def test_deploy_shadow_no_outcomes(self):
        orch, store, _, _ = self._make_orchestrator()
        v = ModelVersion(version_id="candidate_1", model_name="m")
        store.list_versions.return_value = [v]
        store.get_production.return_value = None
        result = orch.deploy_shadow("candidate_1")
        assert result["status"] == "shadow_setup_complete"

    def test_deploy_shadow_with_outcomes(self):
        orch, store, _, _ = self._make_orchestrator()
        v = ModelVersion(version_id="cand", model_name="m")
        store.list_versions.return_value = [v]
        store.get_production.return_value = None
        result = orch.deploy_shadow("cand", outcomes={"g1": 1.0})
        assert result["status"] == "shadow_comparison_pending"

    def test_promote_or_rollback_promote(self):
        orch, _, _, _ = self._make_orchestrator()
        sr = ShadowResult(0.2, 0.18, -0.02, [], 0.05, 0.95, "promote_shadow")
        assert orch.promote_or_rollback(sr) == "promote"

    def test_promote_or_rollback_keep(self):
        orch, _, _, _ = self._make_orchestrator()
        sr = ShadowResult(0.2, 0.2, 0.0, [], 0.0, 1.0, "keep_primary")
        assert orch.promote_or_rollback(sr) == "keep"

    def test_promote_or_rollback_review(self):
        orch, _, _, _ = self._make_orchestrator()
        sr = ShadowResult(0.2, 0.19, -0.01, [], 0.2, 0.9, "needs_review")
        assert orch.promote_or_rollback(sr) == "review"

    def test_check_drift_no_alerts(self):
        orch, _, _, drift = self._make_orchestrator()
        drift.check_and_alert.return_value = []
        alerts = orch.check_drift({}, {})
        assert alerts == []

    def test_check_drift_with_alerts(self):
        orch, _, _, drift = self._make_orchestrator()
        mock_alert = MagicMock()
        drift.check_and_alert.return_value = [mock_alert]
        drift.generate_report.return_value = {"total_alerts": 1, "by_severity": {"critical": 1}}
        alerts = orch.check_drift({"f": {"mean": 1.0}}, {"f": {"mean": 0.5}})
        assert len(alerts) == 1

    def test_get_deployment_status(self):
        orch, store, _, _ = self._make_orchestrator()
        store.list_versions.return_value = []
        store.get_production.return_value = None
        status = orch.get_deployment_status()
        assert status["total_versions"] == 0
        assert status["production"] is None


# ---------------------------------------------------------------------------
# 4. ApprovalGate / ApprovalRequest
# ---------------------------------------------------------------------------
from src.governance.approval_gate import ApprovalGate, ApprovalRequest


class TestApprovalRequest:
    def test_defaults(self):
        r = ApprovalRequest()
        assert r.status == "pending"
        assert r.risk_level == "medium"
        assert r.reviewer is None

    def test_fields(self):
        r = ApprovalRequest(action="promote_model", requester="user1", risk_level="high")
        assert r.action == "promote_model"


class TestApprovalGate:
    def test_request_approval_generates_id(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        req = ApprovalRequest(action="test", requester="user1")
        rid = gate.request_approval(req)
        assert len(rid) == 8
        assert req.status == "pending"

    def test_request_approval_preserves_id(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        req = ApprovalRequest(request_id="custom01", action="test", requester="u")
        rid = gate.request_approval(req)
        assert rid == "custom01"

    def test_check_approval(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        req = ApprovalRequest(action="promote", requester="u")
        rid = gate.request_approval(req)
        found = gate.check_approval(rid)
        assert found is not None
        assert found.action == "promote"

    def test_check_approval_not_found(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        assert gate.check_approval("nonexistent") is None

    def test_approve(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        req = ApprovalRequest(action="deploy", requester="user1")
        rid = gate.request_approval(req)
        assert gate.approve(rid, "reviewer1", "looks good") is True
        updated = gate.check_approval(rid)
        assert updated.status == "approved"
        assert updated.reviewer == "reviewer1"

    def test_reject(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        req = ApprovalRequest(action="deploy", requester="user1")
        rid = gate.request_approval(req)
        assert gate.reject(rid, "reviewer1", "not ready") is True
        updated = gate.check_approval(rid)
        assert updated.status == "rejected"

    def test_pending_requests(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        gate.request_approval(ApprovalRequest(action="a1", requester="u"))
        gate.request_approval(ApprovalRequest(action="a2", requester="u"))
        pending = gate.pending_requests()
        assert len(pending) == 2

    def test_approve_nonexistent_returns_false(self, tmp_path):
        gate = ApprovalGate(ledger_path=str(tmp_path / "ledger.jsonl"))
        assert gate.approve("nope", "reviewer") is False

    def test_load_all_empty_file(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text("")
        gate = ApprovalGate(ledger_path=str(ledger))
        assert gate.pending_requests() == []

    def test_load_all_skips_bad_lines(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text("NOT JSON\n")
        gate = ApprovalGate(ledger_path=str(ledger))
        assert gate.pending_requests() == []


# ---------------------------------------------------------------------------
# 5. AuthorityMatrix / AuthorityRule
# ---------------------------------------------------------------------------
from src.governance.authority_matrix import AuthorityMatrix, AuthorityRule, DEFAULT_AUTHORITY_MATRIX


class TestAuthorityRule:
    def test_creation(self):
        rule = AuthorityRule("submit_*", "critical", "operator")
        assert rule.action_pattern == "submit_*"
        assert rule.auto_approve_conditions is None


class TestAuthorityMatrix:
    def test_default_rules(self):
        m = AuthorityMatrix()
        assert len(m.rules) == len(DEFAULT_AUTHORITY_MATRIX)

    def test_get_rule_exact_match(self):
        m = AuthorityMatrix()
        rule = m.get_rule("submit_predictions")
        assert rule is not None
        assert rule.risk_level == "critical"

    def test_get_rule_glob_match(self):
        m = AuthorityMatrix()
        rule = m.get_rule("modify_config.lr")
        assert rule is not None
        assert rule.action_pattern == "modify_config.*"

    def test_get_rule_no_match(self):
        m = AuthorityMatrix()
        assert m.get_rule("unknown_action_xyz") is None

    def test_requires_approval_high(self):
        m = AuthorityMatrix()
        assert m.requires_approval("submit_predictions", "critical") is True

    def test_requires_approval_auto(self):
        m = AuthorityMatrix()
        assert m.requires_approval("retrain_model", "medium") is False

    def test_requires_approval_unknown_high(self):
        m = AuthorityMatrix()
        assert m.requires_approval("totally_new_action", "high") is True

    def test_requires_approval_unknown_low(self):
        m = AuthorityMatrix()
        assert m.requires_approval("totally_new_action", "low") is False

    def test_can_auto_approve_with_evidence(self):
        m = AuthorityMatrix()
        evidence = {"loyo_regression": False}
        assert m.can_auto_approve("retrain_model", evidence) is True

    def test_can_auto_approve_bad_evidence(self):
        m = AuthorityMatrix()
        evidence = {"loyo_regression": True}
        assert m.can_auto_approve("retrain_model", evidence) is False

    def test_can_auto_approve_no_rule(self):
        m = AuthorityMatrix()
        assert m.can_auto_approve("unknown_action", {}) is False

    def test_can_auto_approve_no_conditions_auto(self):
        rule = AuthorityRule("simple_action", "low", "auto", auto_approve_conditions=None)
        m = AuthorityMatrix(rules=[rule])
        assert m.can_auto_approve("simple_action", {}) is True

    def test_can_auto_approve_non_auto_approver(self):
        m = AuthorityMatrix()
        assert m.can_auto_approve("submit_predictions", {}) is False

    def test_custom_rules(self):
        rules = [AuthorityRule("custom.*", "low", "auto")]
        m = AuthorityMatrix(rules=rules)
        assert m.get_rule("custom.something") is not None
        assert m.get_rule("submit_predictions") is None


# ---------------------------------------------------------------------------
# 6. GovernanceAuditTrail / GovernanceAuditLog
# ---------------------------------------------------------------------------
from src.governance.audit_trail import GovernanceAuditLog, GovernanceAuditTrail, GovernanceRecord


class TestGovernanceRecord:
    def test_defaults(self):
        r = GovernanceRecord(action="test", actor="pipeline", decision="approved")
        assert r.event_type == ""
        assert r.timestamp != ""


class TestGovernanceAuditTrail:
    def test_log_and_query(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="deploy", actor="system", decision="approved"))
        trail.log(GovernanceRecord(action="retrain", actor="user1", decision="auto"))
        records = trail.query()
        assert len(records) == 2

    def test_query_by_action(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="deploy", actor="s", decision="ok"))
        trail.log(GovernanceRecord(action="retrain", actor="s", decision="ok"))
        assert len(trail.query(action="deploy")) == 1

    def test_query_by_actor(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="a", actor="alice", decision="ok"))
        trail.log(GovernanceRecord(action="a", actor="bob", decision="ok"))
        assert len(trail.query(actor="alice")) == 1

    def test_query_since(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="a", actor="s", decision="ok", timestamp="2025-01-01T00:00:00"))
        trail.log(GovernanceRecord(action="a", actor="s", decision="ok", timestamp="2026-06-01T00:00:00"))
        assert len(trail.query(since="2026-01-01")) == 1

    def test_count_by_action(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="deploy", actor="s", decision="ok"))
        trail.log(GovernanceRecord(action="deploy", actor="s", decision="ok"))
        trail.log(GovernanceRecord(action="retrain", actor="s", decision="ok"))
        counts = trail.count_by_action()
        assert counts["deploy"] == 2
        assert counts["retrain"] == 1

    def test_summary(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "audit.jsonl"))
        trail.log(GovernanceRecord(action="deploy", actor="s", decision="ok"))
        s = trail.summary()
        assert "Governance Audit Trail" in s
        assert "deploy" in s

    def test_summary_empty(self, tmp_path):
        trail = GovernanceAuditTrail(audit_path=str(tmp_path / "no_audit.jsonl"))
        s = trail.summary()
        assert "Total records: 0" in s

    def test_load_all_skips_bad_json(self, tmp_path):
        audit = tmp_path / "audit.jsonl"
        audit.write_text("BAD LINE\n")
        trail = GovernanceAuditTrail(audit_path=str(audit))
        assert trail.query() == []


class TestGovernanceAuditLog:
    def test_log_action(self, tmp_path):
        log = GovernanceAuditLog(path=str(tmp_path / "log.jsonl"))
        log.log_action("deploy", "pipeline", {"version": "v1"})
        entries = log._load_all()
        assert len(entries) == 1
        assert entries[0]["type"] == "action"

    def test_log_approval(self, tmp_path):
        log = GovernanceAuditLog(path=str(tmp_path / "log.jsonl"))
        log.log_approval("req123", "deploy", "approved", "admin", "ok")
        entries = log._load_all()
        assert entries[0]["type"] == "approval"
        assert entries[0]["request_id"] == "req123"

    def test_log_compliance_check(self, tmp_path):
        log = GovernanceAuditLog(path=str(tmp_path / "log.jsonl"))
        log.log_compliance_check("data_quality", "passed", "all clear", "training")
        entries = log._load_all()
        assert entries[0]["type"] == "compliance"
        assert entries[0]["checkpoint"] == "data_quality"

    def test_query_by_type(self, tmp_path):
        log = GovernanceAuditLog(path=str(tmp_path / "log.jsonl"))
        log.log_action("a1", "actor1")
        log.log_approval("r1", "a2", "approved", "rev")
        results = log.query(action_type="approval")
        assert len(results) == 1

    def test_query_by_since(self, tmp_path):
        log = GovernanceAuditLog(path=str(tmp_path / "log.jsonl"))
        log.log_action("a1", "actor1")
        # All entries are "now" so querying from far past returns all
        results = log.query(since="2020-01-01")
        assert len(results) == 1

    def test_load_all_empty(self, tmp_path):
        log = GovernanceAuditLog(path=str(tmp_path / "nonexistent.jsonl"))
        assert log._load_all() == []

    def test_load_all_skips_bad_json(self, tmp_path):
        f = tmp_path / "log.jsonl"
        f.write_text("NOT JSON\n")
        log = GovernanceAuditLog(path=str(f))
        assert log._load_all() == []


# ---------------------------------------------------------------------------
# 7. GovernanceGate
# ---------------------------------------------------------------------------
from src.governance.gate import GovernanceGate
from src.exceptions import GovernanceApprovalRequired


class TestGovernanceGate:
    def test_init_defaults(self):
        gate = GovernanceGate()
        assert gate._timeout_hours == 24.0

    def test_check_or_request_no_rule_allows(self):
        matrix = MagicMock()
        matrix.get_rule.return_value = None
        audit = MagicMock()
        gate = GovernanceGate(authority_matrix=matrix, audit_log=audit)
        assert gate.check_or_request("unknown_action", {}) is True
        audit.log_action.assert_called_once()

    def test_check_or_request_auto_approved(self):
        matrix = MagicMock()
        rule = MagicMock()
        rule.action_pattern = "retrain_model"
        rule.risk_level = "medium"
        matrix.get_rule.return_value = rule
        matrix.can_auto_approve.return_value = True
        audit = MagicMock()
        gate = GovernanceGate(authority_matrix=matrix, audit_log=audit)
        assert gate.check_or_request("retrain_model", {"loyo_regression": False}) is True

    def test_check_or_request_requires_approval(self):
        matrix = MagicMock()
        rule = MagicMock()
        rule.action_pattern = "submit_predictions"
        rule.risk_level = "critical"
        rule.required_approver = "operator"
        matrix.get_rule.return_value = rule
        matrix.can_auto_approve.return_value = False
        approval_gate = MagicMock()
        approval_gate.request_approval.return_value = "req123"
        audit = MagicMock()
        gate = GovernanceGate(authority_matrix=matrix, approval_gate=approval_gate, audit_log=audit)
        with pytest.raises(GovernanceApprovalRequired) as exc_info:
            gate.check_or_request("submit_predictions", {})
        assert "req123" in str(exc_info.value)

    def test_approve_delegates(self):
        approval_gate = MagicMock()
        approval_gate.approve.return_value = True
        audit = MagicMock()
        gate = GovernanceGate(approval_gate=approval_gate, audit_log=audit)
        assert gate.approve("req1", "reviewer1", "lgtm") is True
        audit.log_approval.assert_called_once()

    def test_approve_failure(self):
        approval_gate = MagicMock()
        approval_gate.approve.return_value = False
        audit = MagicMock()
        gate = GovernanceGate(approval_gate=approval_gate, audit_log=audit)
        assert gate.approve("nonexistent", "reviewer1") is False
        audit.log_approval.assert_not_called()

    def test_deny(self):
        approval_gate = MagicMock()
        approval_gate.reject.return_value = True
        audit = MagicMock()
        gate = GovernanceGate(approval_gate=approval_gate, audit_log=audit)
        assert gate.deny("req1", "reviewer1", "denied") is True
        audit.log_approval.assert_called_once()

    def test_deny_failure(self):
        approval_gate = MagicMock()
        approval_gate.reject.return_value = False
        audit = MagicMock()
        gate = GovernanceGate(approval_gate=approval_gate, audit_log=audit)
        assert gate.deny("nope", "reviewer1") is False
        audit.log_approval.assert_not_called()

    def test_pending_requests(self):
        approval_gate = MagicMock()
        approval_gate.pending_requests.return_value = ["r1", "r2"]
        gate = GovernanceGate(approval_gate=approval_gate)
        assert gate.pending_requests() == ["r1", "r2"]

    def test_status(self):
        matrix = MagicMock()
        matrix.rules = [
            AuthorityRule("submit_predictions", "critical", "operator"),
        ]
        approval_gate = MagicMock()
        req = ApprovalRequest(
            request_id="abc", action="submit", requester="user",
            risk_level="critical", timestamp="2026-03-16T12:00:00+00:00",
        )
        approval_gate.pending_requests.return_value = [req]
        gate = GovernanceGate(authority_matrix=matrix, approval_gate=approval_gate)
        s = gate.status()
        assert "Governance Status" in s
        assert "submit_predictions" in s
        assert "abc" in s


# ---------------------------------------------------------------------------
# 8. ProductionValidator
# ---------------------------------------------------------------------------
from src.governance.production_validator import (
    ProductionValidationError,
    validate_2026_production_config,
    validate_production_2026,
    REQUIRED_CONFIG_VALUES,
    EXPECTED_TRAINING_YEARS,
    EXPECTED_DEV_YEARS,
    EXPECTED_HOLDOUT_YEARS,
    ALL_PATH_FIELDS,
)
from src.pipeline.config import SOTAPipelineConfig


class TestProductionValidator:
    def _make_valid_config(self, tmp_path):
        """Create a config that passes validation."""
        # Create required dirs and files
        for d in ["multi_year_games", "kaggle", "external_ratings"]:
            (tmp_path / d).mkdir(exist_ok=True)
        for f in ["teams.json", "torvik.json", "historical.json", "roster.json",
                   "picks.json", "scoring.json", "mc_cal.json", "freeze.json"]:
            (tmp_path / f).write_text("{}")

        cfg = SOTAPipelineConfig(
            year=2026,
            probability_profile="production",
            mode="calibration",
            model_complexity="standard",
            use_agent_orchestration=False,
            enable_gnn=False,
            enable_transformer=False,
            enable_embedding_projections=False,
            enable_stacking=True,
            enable_feature_selection=True,
            enable_brier_sharpening=False,
            enable_seed_overrides=False,
            enable_goto_conversion=True,
            enable_round_weighted_calibration=True,
            enable_bayesian_bt=True,
            enable_multi_year_training=True,
            enable_multi_year_calibration=True,
            enable_loyo_cv=True,
            strict_leakage_mode=True,
            require_freeze_file=True,
            calibration_method="temperature",
            enable_tournament_adaptation=True,
            scoring_metric="brier",
            random_seed=2026,
            num_simulations=10000,
            tournament_shrinkage=0.06,
            massey_blend_weight=0.25,
            massey_sigma=4.5,
            optimize_ensemble_weights=True,
            pre_calibration_clip_lo=0.001,
            pre_calibration_clip_hi=0.999,
            mc_noise_std=0.16,
            enable_spread_model=True,
            enable_recency_weighting=True,
            enable_symmetric_augmentation=True,
            scrape_live=False,
            enable_market_blend=True,
            training_years=list(EXPECTED_TRAINING_YEARS),
            dev_years=list(EXPECTED_DEV_YEARS),
            holdout_years=list(EXPECTED_HOLDOUT_YEARS),
            seed_prior_weight=0.10,
            consistency_bonus_max=0.02,
            multi_year_games_dir=str(tmp_path / "multi_year_games"),
            kaggle_dir=str(tmp_path / "kaggle"),
            external_ratings_dir=str(tmp_path / "external_ratings"),
            teams_json=str(tmp_path / "teams.json"),
            torvik_json=str(tmp_path / "torvik.json"),
            historical_games_json=str(tmp_path / "historical.json"),
            roster_json=str(tmp_path / "roster.json"),
            public_picks_json=str(tmp_path / "picks.json"),
            scoring_rules_json=str(tmp_path / "scoring.json"),
            mc_calibration_json=str(tmp_path / "mc_cal.json"),
            freeze_file=str(tmp_path / "freeze.json"),
        )
        return cfg

    def test_valid_config_passes(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        # Should not raise
        validate_2026_production_config(cfg)

    def test_wrong_year_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.year = 2025
        with pytest.raises(ProductionValidationError, match="year"):
            validate_2026_production_config(cfg)

    def test_wrong_training_years_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.training_years = [2020, 2021]
        with pytest.raises(ProductionValidationError, match="training_years"):
            validate_2026_production_config(cfg)

    def test_missing_training_years_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.training_years = None
        with pytest.raises(ProductionValidationError, match="training_years"):
            validate_2026_production_config(cfg)

    def test_missing_dev_years_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.dev_years = None
        with pytest.raises(ProductionValidationError, match="dev_years"):
            validate_2026_production_config(cfg)

    def test_missing_holdout_years_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.holdout_years = None
        with pytest.raises(ProductionValidationError, match="holdout_years"):
            validate_2026_production_config(cfg)

    def test_2025_in_dev_years_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.dev_years = list(EXPECTED_DEV_YEARS) + [2025]
        with pytest.raises(ProductionValidationError, match="2025.*dev_years"):
            validate_2026_production_config(cfg)

    def test_2020_in_training_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.training_years = list(EXPECTED_TRAINING_YEARS) + [2020]
        with pytest.raises(ProductionValidationError, match="2020"):
            validate_2026_production_config(cfg)

    def test_seed_prior_weight_excessive_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.seed_prior_weight = 0.6
        with pytest.raises(ProductionValidationError, match="seed_prior_weight"):
            validate_2026_production_config(cfg)

    def test_consistency_bonus_excessive_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.consistency_bonus_max = 0.15
        with pytest.raises(ProductionValidationError, match="consistency_bonus_max"):
            validate_2026_production_config(cfg)

    def test_missing_path_fields_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.teams_json = None
        with pytest.raises(ProductionValidationError, match="teams_json"):
            validate_2026_production_config(cfg)

    def test_auto_path_value_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.multi_year_games_dir = "auto"
        with pytest.raises(ProductionValidationError, match="auto"):
            validate_2026_production_config(cfg)

    def test_validate_production_2026_skip_disk_check(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        # Should pass with disk check skipped
        validate_production_2026(cfg, check_paths_on_disk=False)

    def test_validate_production_2026_disk_check_missing_dir(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.multi_year_games_dir = str(tmp_path / "nonexistent_dir")
        with pytest.raises(ProductionValidationError, match="directory not found"):
            validate_production_2026(cfg, check_paths_on_disk=True)

    def test_validate_production_2026_disk_check_missing_file(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.teams_json = str(tmp_path / "no_such_file.json")
        with pytest.raises(ProductionValidationError, match="file not found"):
            validate_production_2026(cfg, check_paths_on_disk=True)

    def test_calibration_years_wrong_value_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.calibration_years = [2024]
        with pytest.raises(ProductionValidationError, match="calibration_years"):
            validate_2026_production_config(cfg)

    def test_wrong_holdout_years_fails(self, tmp_path):
        cfg = self._make_valid_config(tmp_path)
        cfg.holdout_years = [2024]
        with pytest.raises(ProductionValidationError, match="holdout_years"):
            validate_2026_production_config(cfg)


# ---------------------------------------------------------------------------
# 9. ManifestGenerator
# ---------------------------------------------------------------------------
from src.data.manifest_generator import ManifestGenerator, _sha256_file, _sha256_bytes, _git_commit_sha


class TestManifestHelpers:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = _sha256_file(str(f))
        assert len(h) == 64
        assert h == hashlib.sha256(b"hello").hexdigest()

    def test_sha256_file_not_found(self):
        assert _sha256_file("/nonexistent/path") == "FILE_NOT_FOUND"

    def test_sha256_bytes(self):
        h = _sha256_bytes(b"test data")
        assert len(h) == 64

    def test_git_commit_sha_returns_string(self):
        result = _git_commit_sha()
        assert isinstance(result, str)


class TestManifestGenerator:
    def test_init_default(self):
        gen = ManifestGenerator()
        assert gen.config_path == ""

    def test_init_custom(self):
        gen = ManifestGenerator(config_path="configs/prod.json")
        assert gen.config_path == "configs/prod.json"

    def test_generate(self):
        gen = ManifestGenerator(config_path="configs/test.json")
        manifest = gen.generate(
            run_id="test_run",
            config_hash="abc123",
            raw_inputs=[{"provider": "kenpom", "path_or_uri": "/data", "snapshot_timestamp": "2026-01-01", "checksum": "xxx"}],
            training_data_hash="def456",
            model_artifact_hash="ghi789",
            random_seed=42,
            model_artifact_path="/models/m.pkl",
            training_data_path="/data/train.csv",
        )
        assert manifest["run_id"] == "test_run"
        assert manifest["random_seed"] == 42
        assert manifest["config"]["path"] == "configs/test.json"
        assert manifest["config"]["hash"] == "abc123"
        assert "commit_sha" in manifest["code"]
        assert "python_version" in manifest["environment"]

    def test_save(self, tmp_path):
        gen = ManifestGenerator()
        manifest = {"run_id": "save_test", "data": 42}
        out_path = str(tmp_path / "output" / "test.manifest.json")
        saved = gen.save(manifest, out_path)
        assert Path(saved).exists()
        loaded = json.loads(Path(saved).read_text())
        assert loaded["run_id"] == "save_test"

    def test_validate_level_a_pass(self):
        manifest = {
            "run_id": "r1",
            "created_at": "2026-01-01",
            "code": {"commit_sha": "abc", "diff_hash": "clean"},
            "config": {"path": "p.json", "hash": "h123"},
            "raw_inputs": [{"provider": "x", "path_or_uri": "y", "snapshot_timestamp": "t", "checksum": "c"}],
            "feature_contracts": {"path": "fc.yaml", "hash": "fch"},
            "training_data": {"path_or_uri": "td.csv", "hash": "tdh"},
            "model": {"artifact_path_or_uri": "m.pkl", "artifact_hash": "mh"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "lh"},
        }
        violations = ManifestGenerator.validate_level_a(manifest)
        assert violations == []

    def test_validate_level_a_missing_fields(self):
        violations = ManifestGenerator.validate_level_a({})
        assert len(violations) > 0
        assert any("run_id" in v for v in violations)

    def test_validate_level_a_empty_fields(self):
        manifest = {
            "run_id": "",
            "created_at": "t",
            "code": {"commit_sha": "", "diff_hash": "x"},
            "config": {"path": "p", "hash": ""},
            "raw_inputs": [],
            "feature_contracts": {"path": "", "hash": "x"},
            "training_data": {"path_or_uri": "x", "hash": ""},
            "model": {"artifact_path_or_uri": "", "artifact_hash": "x"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "", "lockfile_hash": "x"},
        }
        violations = ManifestGenerator.validate_level_a(manifest)
        assert len(violations) > 0

    def test_validate_level_b_pass(self):
        manifest = {
            "run_id": "r1",
            "created_at": "2026-01-01",
            "code": {"commit_sha": "abc", "diff_hash": "clean"},
            "config": {"path": "p.json", "hash": "h123"},
            "raw_inputs": [{"provider": "x", "path_or_uri": "y", "snapshot_timestamp": "t", "checksum": "c"}],
            "feature_contracts": {"path": "fc.yaml", "hash": "fch"},
            "training_data": {"path_or_uri": "td.csv", "hash": "correct_hash"},
            "model": {"artifact_path_or_uri": "m.pkl", "artifact_hash": "mh"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "lh"},
        }
        violations = ManifestGenerator.validate_level_b(manifest, "correct_hash")
        assert violations == []

    def test_validate_level_b_mismatch(self):
        manifest = {
            "run_id": "r1",
            "created_at": "2026-01-01",
            "code": {"commit_sha": "abc", "diff_hash": "clean"},
            "config": {"path": "p.json", "hash": "h123"},
            "raw_inputs": [{"provider": "x", "path_or_uri": "y", "snapshot_timestamp": "t", "checksum": "c"}],
            "feature_contracts": {"path": "fc.yaml", "hash": "fch"},
            "training_data": {"path_or_uri": "td.csv", "hash": "expected_hash"},
            "model": {"artifact_path_or_uri": "m.pkl", "artifact_hash": "mh"},
            "random_seed": 42,
            "environment": {"python_version": "3.11", "os": "Linux", "lockfile_hash": "lh"},
        }
        violations = ManifestGenerator.validate_level_b(manifest, "different_hash")
        assert any("Level B FAILED" in v for v in violations)


# ---------------------------------------------------------------------------
# 10. GameFlow / Possession / FourFactors
# ---------------------------------------------------------------------------
from src.data.models.game_flow import (
    FourFactors,
    GameFlow,
    Possession,
    PossessionOutcome,
    ShotType,
)


class TestShotType:
    def test_values(self):
        assert ShotType.RIM.value == "rim"
        assert ShotType.CORNER_THREE.value == "corner_three"


class TestPossessionOutcome:
    def test_values(self):
        assert PossessionOutcome.MADE_SHOT.value == "made"
        assert PossessionOutcome.TURNOVER.value == "turnover"


class TestPossession:
    def test_creation(self):
        p = Possession(possession_id="p1", game_id="g1", team_id="t1", period=1, game_clock=1200.0)
        assert p.xp == 0.0
        assert p.actual_points == 0

    def test_xp_differential(self):
        p = Possession(possession_id="p1", game_id="g1", team_id="t1", period=1, game_clock=1200.0, xp=1.0, actual_points=2)
        assert p.xp_differential == 1.0

    def test_calculate_xp_rim(self):
        xp = Possession.calculate_xp(ShotType.RIM)
        assert xp == 1.30

    def test_calculate_xp_contested(self):
        xp_open = Possession.calculate_xp(ShotType.CORNER_THREE, is_contested=False)
        xp_contested = Possession.calculate_xp(ShotType.CORNER_THREE, is_contested=True)
        assert xp_contested < xp_open
        assert xp_contested == pytest.approx(1.14 * 0.85)

    def test_calculate_xp_all_types(self):
        for st in ShotType:
            xp = Possession.calculate_xp(st)
            assert xp > 0


class TestGameFlow:
    def test_creation(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2")
        assert gf.lead_history == []
        assert gf.possessions == []

    def test_lead_changes_empty(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2")
        assert gf.lead_changes == 0

    def test_lead_changes(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[1, 2, -1, -3, 2, -1])
        changes = gf.lead_changes
        assert changes >= 2  # At least team1->team2 and team2->team1

    def test_lead_changes_cached(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[5, -5, 5])
        _ = gf.lead_changes
        # Second call uses cached value
        assert gf.lead_changes == gf._lead_changes

    def test_lead_volatility_empty(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2")
        assert gf.lead_volatility == 0.0

    def test_lead_volatility(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[10, -10, 10, -10])
        assert gf.lead_volatility > 0.0

    def test_entropy_empty(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2")
        assert gf.entropy == 0.0

    def test_entropy_one_sided(self):
        # All blowout ahead — single bucket, entropy = 0
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[20, 22, 25, 30])
        assert gf.entropy == 0.0

    def test_entropy_diverse(self):
        # Spread across multiple buckets
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[-20, -10, -3, 0, 3, 10, 20])
        assert gf.entropy > 0.0

    def test_comeback_factor_short(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[5])
        assert gf.comeback_factor == 0.0

    def test_comeback_factor(self):
        # Team goes behind by 6, then comes back to lead
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       lead_history=[-6, -7, -3, 1, 5, -6, 2])
        assert gf.comeback_factor > 0.0

    def test_get_xp_margin(self):
        p1 = Possession(possession_id="p1", game_id="g1", team_id="t1", period=1, game_clock=1200.0, xp=1.3)
        p2 = Possession(possession_id="p2", game_id="g1", team_id="t2", period=1, game_clock=1150.0, xp=0.7)
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2", possessions=[p1, p2])
        assert gf.get_xp_margin() == pytest.approx(0.6)

    def test_get_luck_factor(self):
        p1 = Possession(possession_id="p1", game_id="g1", team_id="t1", period=1, game_clock=1200.0, xp=1.0)
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2",
                       possessions=[p1], lead_history=[5])
        luck = gf.get_luck_factor()
        assert luck == pytest.approx(5 - 1.0)

    def test_get_luck_factor_no_history(self):
        gf = GameFlow(game_id="g1", team1_id="t1", team2_id="t2")
        assert gf.get_luck_factor() == 0.0


class TestFourFactors:
    def test_defaults(self):
        ff = FourFactors()
        assert ff.effective_fg_pct == 0.0

    def test_offensive_rating(self):
        ff = FourFactors(effective_fg_pct=0.5, turnover_rate=0.15, offensive_reb_rate=0.3, free_throw_rate=0.25)
        rating = ff.offensive_rating
        assert rating > 0

    def test_defensive_rating(self):
        ff = FourFactors(opp_effective_fg_pct=0.45, opp_turnover_rate=0.2, defensive_reb_rate=0.7, opp_free_throw_rate=0.3)
        rating = ff.defensive_rating
        assert rating > 0

    def test_net_rating(self):
        ff = FourFactors(
            effective_fg_pct=0.5, turnover_rate=0.15, offensive_reb_rate=0.3, free_throw_rate=0.25,
            opp_effective_fg_pct=0.45, opp_turnover_rate=0.2, defensive_reb_rate=0.7, opp_free_throw_rate=0.3,
        )
        # Just verify it's a float
        assert isinstance(ff.net_rating, float)

    def test_to_vector(self):
        ff = FourFactors(effective_fg_pct=0.5, turnover_rate=0.15, offensive_reb_rate=0.3, free_throw_rate=0.25,
                         opp_effective_fg_pct=0.45, opp_turnover_rate=0.2, defensive_reb_rate=0.7, opp_free_throw_rate=0.3)
        vec = ff.to_vector()
        assert len(vec) == 8
        assert vec[0] == 0.5


# ---------------------------------------------------------------------------
# 11. DataLoader
# ---------------------------------------------------------------------------
from src.data.loader import DataLoader


class TestDataLoader:
    def test_hash_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("reproducibility test")
        h = DataLoader.hash_file(str(f))
        assert len(h) == 64
        # Same content should give same hash
        assert h == DataLoader.hash_file(str(f))

    def test_hash_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DataLoader.hash_file("/nonexistent/file.txt")

    def test_hash_dataset(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("col1,col2\n1,2")
        f2.write_text("col1,col2\n3,4")
        result = DataLoader.hash_dataset([str(f1), str(f2)], labels=["train", "valid"])
        assert "train" in result
        assert "valid" in result
        assert "combined" in result
        assert len(result["combined"]) == 64

    def test_hash_dataset_missing_file(self, tmp_path):
        f1 = tmp_path / "exists.csv"
        f1.write_text("data")
        result = DataLoader.hash_dataset([str(f1), "/nonexistent.csv"])
        assert result["nonexistent.csv"] == "MISSING"
        assert "combined" in result

    def test_hash_dataset_no_labels(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("x")
        result = DataLoader.hash_dataset([str(f)])
        assert "data.csv" in result

    def test_load_teams_from_json(self, tmp_path):
        teams_file = tmp_path / "teams.json"
        teams_file.write_text(json.dumps({"teams": []}))
        teams = DataLoader.load_teams_from_json(str(teams_file))
        assert teams == []

    def test_load_matchups(self, tmp_path):
        matchups_file = tmp_path / "matchups.json"
        matchups_file.write_text(json.dumps({
            "matchups": [
                {"team1": "Duke", "team2": "UNC", "round": 1, "game_id": 1},
                {"team1": "Kansas", "team2": "Kentucky", "round": 2, "game_id": 2},
            ]
        }))
        matchups = DataLoader.load_matchups(str(matchups_file))
        assert len(matchups) == 2
        assert matchups[0] == ("Duke", "UNC", 1, 1)

    def test_save_and_load_bracket(self, tmp_path):
        # Test save_bracket_to_json with a mock bracket
        bracket = MagicMock()
        bracket.to_dict.return_value = {"bracket": "test_data", "games": []}
        out_path = str(tmp_path / "bracket.json")
        DataLoader.save_bracket_to_json(bracket, out_path)
        data = json.loads(Path(out_path).read_text())
        assert data["bracket"] == "test_data"
