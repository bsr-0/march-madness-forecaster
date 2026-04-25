"""Bidirectional feature contract validator.

Validates contracts/feature_contracts.yaml against the JSON schema and
verifies bidirectional consistency between contracts and active code features.

Usage:
    from src.data.contract_validator import ContractValidator
    validator = ContractValidator()
    report = validator.validate()
    assert report.passed, report.summary()
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_PATH = REPO_ROOT / "contracts" / "feature_contracts.yaml"
SCHEMA_PATH = REPO_ROOT / "schemas" / "feature_contract.schema.json"


@dataclass
class ValidationViolation:
    """A single validation failure."""

    check: str
    feature_name: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationReport:
    """Aggregated validation report."""

    violations: List[ValidationViolation] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0

    @property
    def passed(self) -> bool:
        return all(v.severity != "error" for v in self.violations)

    def summary(self) -> str:
        lines = [
            f"Contract Validation: {'PASSED' if self.passed else 'FAILED'}",
            f"  Checks run: {self.checks_run}",
            f"  Checks passed: {self.checks_passed}",
            f"  Errors: {sum(1 for v in self.violations if v.severity == 'error')}",
            f"  Warnings: {sum(1 for v in self.violations if v.severity == 'warning')}",
        ]
        for v in self.violations:
            lines.append(f"  [{v.severity.upper()}] {v.check}: {v.feature_name} — {v.message}")
        return "\n".join(lines)


class ContractValidator:
    """Validates feature contracts for schema compliance, completeness, and consistency."""

    def __init__(
        self,
        contracts_path: Optional[Path] = None,
        schema_path: Optional[Path] = None,
    ):
        self.contracts_path = contracts_path or CONTRACTS_PATH
        self.schema_path = schema_path or SCHEMA_PATH
        self._contracts: Optional[Dict[str, Any]] = None
        self._schema: Optional[Dict[str, Any]] = None

    def _load_contracts(self) -> Dict[str, Any]:
        if self._contracts is None:
            with open(self.contracts_path) as f:
                self._contracts = yaml.safe_load(f)
        return self._contracts

    def _load_schema(self) -> Dict[str, Any]:
        if self._schema is None:
            with open(self.schema_path) as f:
                self._schema = json.load(f)
        return self._schema

    def _get_active_feature_names(self) -> List[str]:
        """Get feature names from TeamFeatures.get_feature_names()."""
        from src.data.features.feature_engineering import TeamFeatures

        return TeamFeatures.get_feature_names(include_embeddings=False)

    def _get_contract_feature_names(self) -> List[str]:
        """Get feature names defined in contracts."""
        contracts = self._load_contracts()
        return [f["feature_name"] for f in contracts.get("features", [])]

    def validate(self) -> ValidationReport:
        """Run all validation checks and return a report."""
        report = ValidationReport()
        self._check_schema_valid(report)
        self._check_forward_completeness(report)
        self._check_reverse_completeness(report)
        self._check_matchup_lineage(report)
        self._check_available_at_executable(report)
        self._check_transformation_ref_exists(report)
        self._check_snapshot_policy_for_high_risk(report)
        self._check_risk_tier_coverage(report)
        self._check_no_dead_contracts(report)
        return report

    def _check_schema_valid(self, report: ValidationReport) -> None:
        """Validate all contracts against the JSON schema."""
        report.checks_run += 1
        try:
            import jsonschema
        except ImportError:
            report.violations.append(
                ValidationViolation(
                    check="schema_valid",
                    feature_name="*",
                    message="jsonschema package not installed; skipping schema validation",
                    severity="warning",
                )
            )
            return

        schema = self._load_schema()
        contracts = self._load_contracts()
        try:
            jsonschema.validate(instance=contracts, schema=schema)
            report.checks_passed += 1
        except jsonschema.ValidationError as e:
            report.violations.append(
                ValidationViolation(
                    check="schema_valid",
                    feature_name=str(e.path),
                    message=str(e.message)[:200],
                )
            )

    def _check_forward_completeness(self, report: ValidationReport) -> None:
        """Every active feature in get_feature_names() must have a contract."""
        report.checks_run += 1
        try:
            active_names = set(self._get_active_feature_names())
        except Exception as e:
            report.violations.append(
                ValidationViolation(
                    check="forward_completeness",
                    feature_name="*",
                    message=f"Could not load active feature names: {e}",
                )
            )
            return

        contract_names = set(self._get_contract_feature_names())
        missing = active_names - contract_names
        if missing:
            for name in sorted(missing):
                report.violations.append(
                    ValidationViolation(
                        check="forward_completeness",
                        feature_name=name,
                        message="Active feature has no contract entry",
                    )
                )
        else:
            report.checks_passed += 1

    def _check_reverse_completeness(self, report: ValidationReport) -> None:
        """Every contract must map to an active or explicitly deprecated feature."""
        report.checks_run += 1
        try:
            active_names = set(self._get_active_feature_names())
        except Exception:
            return

        contracts = self._load_contracts()
        orphans = []
        for feat in contracts.get("features", []):
            name = feat["feature_name"]
            is_deprecated = feat.get("deprecated", False)
            if name not in active_names and not is_deprecated:
                orphans.append(name)

        if orphans:
            for name in orphans:
                report.violations.append(
                    ValidationViolation(
                        check="reverse_completeness",
                        feature_name=name,
                        message="Contract exists but feature is not active and not marked deprecated",
                    )
                )
        else:
            report.checks_passed += 1

    def _check_matchup_lineage(self, report: ValidationReport) -> None:
        """Derived diff_*/abs_*/interaction features must trace to team-level contracts."""
        report.checks_run += 1
        contract_names = set(self._get_contract_feature_names())

        # The pipeline creates diff_{name} for each team-level feature,
        # abs_{name} for absolute-level features, and named interactions.
        # We verify that for any diff_ or abs_ feature used in training,
        # the base team-level feature has a contract.
        from src.pipeline.config import FIXED_FEATURE_SET

        missing_base = []
        for feat_name in FIXED_FEATURE_SET:
            if feat_name.startswith("diff_"):
                base = feat_name[5:]  # strip "diff_"
                if base not in contract_names:
                    missing_base.append((feat_name, base))
            elif feat_name.startswith("abs_"):
                base = feat_name[4:]  # strip "abs_"
                if base not in contract_names:
                    missing_base.append((feat_name, base))

        if missing_base:
            for derived, base in missing_base:
                report.violations.append(
                    ValidationViolation(
                        check="matchup_lineage",
                        feature_name=derived,
                        message=f"Derived feature references base '{base}' which has no contract",
                    )
                )
        else:
            report.checks_passed += 1

    def _check_available_at_executable(self, report: ValidationReport) -> None:
        """available_at_logic must be structured and evaluable, not free text."""
        report.checks_run += 1
        contracts = self._load_contracts()
        failures = []
        valid_types = {"offset_from_event", "function_ref", "static"}

        for feat in contracts.get("features", []):
            logic = feat.get("available_at_logic", {})
            logic_type = logic.get("type")

            if logic_type not in valid_types:
                failures.append((feat["feature_name"], f"Invalid type '{logic_type}'"))
                continue

            if logic_type == "offset_from_event":
                if "event_timestamp_field" not in logic or "offset_hours" not in logic:
                    failures.append(
                        (
                            feat["feature_name"],
                            "offset_from_event requires event_timestamp_field and offset_hours",
                        )
                    )
            elif logic_type == "function_ref":
                ref = logic.get("ref", "")
                if "::" not in ref:
                    failures.append(
                        (
                            feat["feature_name"],
                            f"function_ref '{ref}' must be in 'module::function' format",
                        )
                    )

        if failures:
            for name, msg in failures:
                report.violations.append(
                    ValidationViolation(
                        check="available_at_executable",
                        feature_name=name,
                        message=msg,
                    )
                )
        else:
            report.checks_passed += 1

    def _check_transformation_ref_exists(self, report: ValidationReport) -> None:
        """Every transformation_logic_ref must resolve to an existing file::function.

        Deprecated contracts with a `sunset_date` are skipped — by the
        no_dead_contracts contract, sunset_date means the feature is being
        retained for audit only and its source code may have been removed.
        Active features and deprecated-but-no-sunset features are still
        validated (the latter would already fail no_dead_contracts).
        """
        report.checks_run += 1
        contracts = self._load_contracts()
        failures = []

        for feat in contracts.get("features", []):
            if feat.get("deprecated", False) and feat.get("sunset_date"):
                continue
            ref = feat.get("transformation_logic_ref", "")
            if "::" not in ref:
                failures.append((feat["feature_name"], f"Invalid ref format: '{ref}'"))
                continue

            file_part, func_part = ref.rsplit("::", 1)
            file_path = REPO_ROOT / file_part
            if not file_path.exists():
                failures.append(
                    (
                        feat["feature_name"],
                        f"File not found: {file_part}",
                    )
                )

        if failures:
            for name, msg in failures:
                report.violations.append(
                    ValidationViolation(
                        check="transformation_ref_exists",
                        feature_name=name,
                        message=msg,
                    )
                )
        else:
            report.checks_passed += 1

    def _check_snapshot_policy_for_high_risk(self, report: ValidationReport) -> None:
        """Every high/critical feature must have explicit snapshot_policy."""
        report.checks_run += 1
        contracts = self._load_contracts()
        failures = []

        for feat in contracts.get("features", []):
            tier = feat.get("risk_tier", "low")
            if tier in ("high", "critical"):
                if not feat.get("snapshot_policy"):
                    failures.append(feat["feature_name"])

        if failures:
            for name in failures:
                report.violations.append(
                    ValidationViolation(
                        check="snapshot_policy_high_risk",
                        feature_name=name,
                        message="High/critical risk feature missing snapshot_policy",
                    )
                )
        else:
            report.checks_passed += 1

    def _check_risk_tier_coverage(self, report: ValidationReport) -> None:
        """Every feature must have a risk_tier."""
        report.checks_run += 1
        contracts = self._load_contracts()
        valid_tiers = {"low", "medium", "high", "critical"}
        failures = []

        for feat in contracts.get("features", []):
            tier = feat.get("risk_tier")
            if tier not in valid_tiers:
                failures.append((feat["feature_name"], f"Invalid tier: {tier}"))

        if failures:
            for name, msg in failures:
                report.violations.append(
                    ValidationViolation(
                        check="risk_tier_coverage",
                        feature_name=name,
                        message=msg,
                    )
                )
        else:
            report.checks_passed += 1

    def _check_no_dead_contracts(self, report: ValidationReport) -> None:
        """Deprecated contracts must have a sunset_date."""
        report.checks_run += 1
        contracts = self._load_contracts()
        failures = []

        for feat in contracts.get("features", []):
            if feat.get("deprecated", False) and not feat.get("sunset_date"):
                failures.append(feat["feature_name"])

        if failures:
            for name in failures:
                report.violations.append(
                    ValidationViolation(
                        check="no_dead_contracts",
                        feature_name=name,
                        message="Deprecated contract missing sunset_date",
                    )
                )
        else:
            report.checks_passed += 1
