"""Phase 4: Point-in-time feature contract validation tests.

Validates contracts/feature_contracts.yaml for:
- JSON schema compliance
- Forward completeness (every active feature has a contract)
- Reverse completeness (every contract maps to active/deprecated feature)
- Matchup lineage (diff_*/abs_* trace to team-level contracts)
- available_at_logic is executable (not free text)
- transformation_logic_ref resolves to real code
- snapshot_policy set for high/critical features
- risk_tier coverage
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_PATH = REPO_ROOT / "contracts" / "feature_contracts.yaml"
SCHEMA_PATH = REPO_ROOT / "schemas" / "feature_contract.schema.json"


@pytest.fixture(scope="module")
def contracts():
    with open(CONTRACTS_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def feature_list(contracts):
    return contracts.get("features", [])


@pytest.fixture(scope="module")
def contract_names(feature_list):
    return {f["feature_name"] for f in feature_list}


# ── PIT-SCHEMA: All contracts pass JSON schema ──────────────────────────


class TestSchemaValid:
    def test_contracts_file_exists(self):
        assert CONTRACTS_PATH.exists(), f"Missing {CONTRACTS_PATH}"

    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), f"Missing {SCHEMA_PATH}"

    def test_schema_validates(self, contracts, schema):
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(instance=contracts, schema=schema)

    def test_version_present(self, contracts):
        assert "version" in contracts
        assert "features" in contracts

    def test_minimum_feature_count(self, feature_list):
        assert len(feature_list) >= 79, (
            f"Expected >= 79 features in contracts, got {len(feature_list)}"
        )


# ── PIT-FORWARD: Every active feature has a contract ────────────────────


class TestForwardCompleteness:
    def test_all_active_features_have_contracts(self, contract_names):
        from src.data.features.feature_engineering import TeamFeatures
        active_names = set(TeamFeatures.get_feature_names(include_embeddings=False))
        missing = active_names - contract_names
        assert not missing, (
            f"Active features missing contracts: {sorted(missing)}"
        )


# ── PIT-REVERSE: Every contract maps to active or deprecated feature ────


class TestReverseCompleteness:
    def test_no_orphan_contracts(self, feature_list, contract_names):
        from src.data.features.feature_engineering import TeamFeatures
        active_names = set(TeamFeatures.get_feature_names(include_embeddings=False))
        for feat in feature_list:
            name = feat["feature_name"]
            is_deprecated = feat.get("deprecated", False)
            if not is_deprecated:
                assert name in active_names, (
                    f"Contract '{name}' is not in active features and not "
                    f"marked deprecated — possible dead contract"
                )

    def test_deprecated_contracts_have_sunset_date(self, feature_list):
        for feat in feature_list:
            if feat.get("deprecated", False):
                assert feat.get("sunset_date"), (
                    f"Deprecated feature '{feat['feature_name']}' must have sunset_date"
                )


# ── PIT-MATCHUP: Derived features trace to team-level contracts ─────────


class TestMatchupLineage:
    def test_diff_features_have_base_contracts(self, contract_names):
        from src.pipeline.config import FIXED_FEATURE_SET
        missing = []
        for feat_name in FIXED_FEATURE_SET:
            if feat_name.startswith("diff_"):
                base = feat_name[5:]
                if base not in contract_names:
                    missing.append((feat_name, base))
        assert not missing, (
            f"Derived diff features reference bases without contracts: {missing}"
        )

    def test_abs_features_have_base_contracts(self, contract_names):
        from src.pipeline.config import FIXED_FEATURE_SET
        missing = []
        for feat_name in FIXED_FEATURE_SET:
            if feat_name.startswith("abs_"):
                base = feat_name[4:]
                if base not in contract_names:
                    missing.append((feat_name, base))
        assert not missing, (
            f"Derived abs features reference bases without contracts: {missing}"
        )


# ── PIT-AVAILABLE_AT: Structured and evaluable, not prose ───────────────


class TestAvailableAtExecutable:
    VALID_TYPES = {"offset_from_event", "function_ref", "static"}

    def test_all_have_valid_type(self, feature_list):
        for feat in feature_list:
            logic = feat.get("available_at_logic", {})
            assert logic.get("type") in self.VALID_TYPES, (
                f"Feature '{feat['feature_name']}' has invalid "
                f"available_at_logic type: {logic.get('type')}"
            )

    def test_offset_has_required_fields(self, feature_list):
        for feat in feature_list:
            logic = feat.get("available_at_logic", {})
            if logic.get("type") == "offset_from_event":
                assert "event_timestamp_field" in logic, (
                    f"Feature '{feat['feature_name']}': offset_from_event "
                    f"missing event_timestamp_field"
                )
                assert "offset_hours" in logic, (
                    f"Feature '{feat['feature_name']}': offset_from_event "
                    f"missing offset_hours"
                )
                assert isinstance(logic["offset_hours"], (int, float)), (
                    f"Feature '{feat['feature_name']}': offset_hours must be numeric"
                )

    def test_function_ref_has_valid_ref(self, feature_list):
        for feat in feature_list:
            logic = feat.get("available_at_logic", {})
            if logic.get("type") == "function_ref":
                ref = logic.get("ref", "")
                assert "::" in ref, (
                    f"Feature '{feat['feature_name']}': function_ref '{ref}' "
                    f"must be 'module::function' format"
                )


# ── PIT-TRANSFORM_REF: Referenced code exists ───────────────────────────


class TestTransformationRefExists:
    def test_all_refs_point_to_existing_files(self, feature_list):
        missing = []
        for feat in feature_list:
            ref = feat.get("transformation_logic_ref", "")
            if "::" not in ref:
                missing.append((feat["feature_name"], f"Invalid format: {ref}"))
                continue
            file_part = ref.rsplit("::", 1)[0]
            file_path = REPO_ROOT / file_part
            if not file_path.exists():
                missing.append((feat["feature_name"], f"File not found: {file_part}"))
        assert not missing, (
            f"Transformation refs with missing files: {missing}"
        )


# ── PIT-SNAPSHOT: High/critical features have snapshot_policy ────────────


class TestSnapshotPolicyForHighRisk:
    def test_high_critical_have_snapshot_policy(self, feature_list):
        missing = []
        for feat in feature_list:
            tier = feat.get("risk_tier", "low")
            if tier in ("high", "critical") and not feat.get("snapshot_policy"):
                missing.append(feat["feature_name"])
        assert not missing, (
            f"High/critical features missing snapshot_policy: {missing}"
        )

    def test_high_critical_have_revision_behavior(self, feature_list):
        missing = []
        for feat in feature_list:
            tier = feat.get("risk_tier", "low")
            if tier in ("high", "critical") and not feat.get("revision_behavior"):
                missing.append(feat["feature_name"])
        assert not missing, (
            f"High/critical features missing revision_behavior: {missing}"
        )


# ── PIT-RISK_TIER: Every feature has a valid tier ───────────────────────


class TestRiskTierCoverage:
    VALID_TIERS = {"low", "medium", "high", "critical"}

    def test_all_features_have_valid_tier(self, feature_list):
        invalid = []
        for feat in feature_list:
            tier = feat.get("risk_tier")
            if tier not in self.VALID_TIERS:
                invalid.append((feat["feature_name"], tier))
        assert not invalid, f"Features with invalid risk_tier: {invalid}"

    def test_critical_features_exist(self, feature_list):
        critical = [f["feature_name"] for f in feature_list if f.get("risk_tier") == "critical"]
        assert len(critical) >= 3, (
            f"Expected at least 3 critical features, got {len(critical)}: {critical}"
        )

    def test_critical_features_have_historical_test_ids(self, feature_list):
        """Critical features must have at least one HIST-* test_id."""
        missing = []
        for feat in feature_list:
            if feat.get("risk_tier") == "critical":
                test_ids = feat.get("test_ids", [])
                has_hist = any(tid.startswith("HIST-") for tid in test_ids)
                if not has_hist:
                    missing.append(feat["feature_name"])
        assert not missing, (
            f"Critical features missing historical test evidence: {missing}"
        )

    def test_high_features_have_historical_test_ids(self, feature_list):
        """High-risk features must have at least one HIST-* test_id."""
        missing = []
        for feat in feature_list:
            if feat.get("risk_tier") == "high":
                test_ids = feat.get("test_ids", [])
                has_hist = any(tid.startswith("HIST-") for tid in test_ids)
                if not has_hist:
                    missing.append(feat["feature_name"])
        assert not missing, (
            f"High-risk features missing historical test evidence: {missing}"
        )


# ── PIT-LEAKAGE_CHECKS: All required boolean fields present ─────────────


class TestLeakageCheckFields:
    REQUIRED_CHECKS = [
        "same_game_outcome_leakage",
        "future_opponent_outcome_leakage",
        "post_game_aggregate_leakage",
        "tournament_info_leakage_pre_tournament",
    ]

    def test_all_leakage_checks_present(self, feature_list):
        missing = []
        for feat in feature_list:
            checks = feat.get("leakage_checks", {})
            for check in self.REQUIRED_CHECKS:
                if check not in checks:
                    missing.append((feat["feature_name"], check))
        assert not missing, (
            f"Features missing leakage check fields: {missing}"
        )

    def test_leakage_flags_are_boolean(self, feature_list):
        bad = []
        for feat in feature_list:
            checks = feat.get("leakage_checks", {})
            for check in self.REQUIRED_CHECKS:
                val = checks.get(check)
                if val is not None and not isinstance(val, bool):
                    bad.append((feat["feature_name"], check, type(val).__name__))
        assert not bad, f"Non-boolean leakage check values: {bad}"

    def test_tournament_leakage_flagged_for_critical(self, feature_list):
        """Critical features that depend on tournament data must flag it."""
        tournament_sources = {"bracket", "conference_results"}
        for feat in feature_list:
            if feat.get("risk_tier") == "critical":
                source_dataset = feat.get("source", {}).get("dataset", "")
                if source_dataset in tournament_sources:
                    checks = feat.get("leakage_checks", {})
                    assert checks.get("tournament_info_leakage_pre_tournament") is True, (
                        f"Critical feature '{feat['feature_name']}' with tournament "
                        f"source must flag tournament_info_leakage_pre_tournament=true"
                    )
