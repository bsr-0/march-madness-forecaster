"""Compliance checkpoints and escalation protocol for pipeline governance.

Extends the governance framework with:
  - Compliance checkpoints at pipeline stage boundaries
  - Escalation protocol for unresolved governance issues
  - Role-based access control enforcement
  - Governance gate integration with pipeline stages

Implements Agent Directive V7 S21 (Human-in-the-Loop Governance).

Usage:
    from src.governance.compliance import (
        ComplianceCheckpoint, ComplianceGate, EscalationProtocol
    )

    gate = ComplianceGate()
    result = gate.check("model_training", {
        "brier_score": 0.21,
        "n_features": 45,
        "has_leakage": False,
    })
    if not result.passed:
        escalation = EscalationProtocol()
        escalation.escalate(result, level="ml_lead")
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compliance checkpoints
# ---------------------------------------------------------------------------


class CheckpointSeverity(str, Enum):
    """Severity of a compliance check result."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ComplianceCheck:
    """Result of a single compliance check."""

    check_name: str
    passed: bool
    severity: CheckpointSeverity = CheckpointSeverity.INFO
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointResult:
    """Aggregate result of all compliance checks at a checkpoint."""

    checkpoint_id: str = ""
    stage_name: str = ""
    timestamp: str = ""
    checks: List[ComplianceCheck] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            self.checkpoint_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def passed(self) -> bool:
        """True if no ERROR or CRITICAL checks failed."""
        return all(
            c.passed
            for c in self.checks
            if c.severity in (CheckpointSeverity.ERROR, CheckpointSeverity.CRITICAL)
        )

    @property
    def n_warnings(self) -> int:
        return sum(
            1 for c in self.checks
            if not c.passed and c.severity == CheckpointSeverity.WARNING
        )

    @property
    def n_errors(self) -> int:
        return sum(
            1 for c in self.checks
            if not c.passed
            and c.severity in (CheckpointSeverity.ERROR, CheckpointSeverity.CRITICAL)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_warnings": self.n_warnings,
            "n_errors": self.n_errors,
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Compliance Gate (stage-boundary enforcement)
# ---------------------------------------------------------------------------


# Default compliance rules per stage
DEFAULT_COMPLIANCE_RULES: Dict[str, List[Dict[str, Any]]] = {
    "data_loading": [
        {
            "name": "min_teams_loaded",
            "check": lambda ctx: ctx.get("n_teams", 0) >= 64,
            "severity": "error",
            "message": "At least 64 teams required for tournament prediction",
        },
        {
            "name": "data_freshness",
            "check": lambda ctx: not ctx.get("has_stale_data", False),
            "severity": "warning",
            "message": "Some data sources are stale",
        },
    ],
    "feature_engineering": [
        {
            "name": "feature_dim_valid",
            "check": lambda ctx: ctx.get("feature_dim", 0) > 0,
            "severity": "error",
            "message": "Feature dimension must be positive",
        },
        {
            "name": "no_nan_features",
            "check": lambda ctx: ctx.get("nan_count", 0) == 0,
            "severity": "warning",
            "message": "Features contain NaN values",
        },
    ],
    "model_training": [
        {
            "name": "brier_reasonable",
            "check": lambda ctx: (
                ctx.get("brier_score") is None
                or 0.0 < ctx.get("brier_score", 1.0) < 0.5
            ),
            "severity": "error",
            "message": "Brier score outside reasonable range [0, 0.5]",
        },
        {
            "name": "no_leakage",
            "check": lambda ctx: not ctx.get("has_leakage", False),
            "severity": "critical",
            "message": "Temporal leakage detected in training data",
        },
    ],
    "calibration": [
        {
            "name": "calibration_fitted",
            "check": lambda ctx: ctx.get("calibration_method", "none") != "none",
            "severity": "warning",
            "message": "No calibration method applied",
        },
    ],
    "simulation": [
        {
            "name": "sufficient_simulations",
            "check": lambda ctx: ctx.get("num_simulations", 0) >= 1000,
            "severity": "warning",
            "message": "Fewer than 1000 Monte Carlo simulations",
        },
    ],
    "robustness_audit": [
        {
            "name": "deployment_gate",
            "check": lambda ctx: ctx.get("deployment_gate_passed", True),
            "severity": "error",
            "message": "Robustness deployment gate failed",
        },
    ],
}


class ComplianceGate:
    """Enforces compliance checks at pipeline stage boundaries.

    Each stage has a set of compliance rules. Before proceeding to
    the next stage, the gate checks all rules and returns a
    CheckpointResult indicating pass/fail.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self._rules = rules or DEFAULT_COMPLIANCE_RULES
        self._history: List[CheckpointResult] = []

    def check(
        self, stage_name: str, context: Dict[str, Any]
    ) -> CheckpointResult:
        """Run all compliance checks for a stage.

        Args:
            stage_name: Pipeline stage name.
            context: Dict with stage output metrics for compliance checking.

        Returns:
            CheckpointResult with all check outcomes.
        """
        result = CheckpointResult(stage_name=stage_name)

        stage_rules = self._rules.get(stage_name, [])
        for rule in stage_rules:
            try:
                check_fn = rule["check"]
                passed = bool(check_fn(context))
            except Exception as exc:
                passed = False
                logger.debug(
                    "Compliance check '%s' raised: %s", rule["name"], exc
                )

            check = ComplianceCheck(
                check_name=rule["name"],
                passed=passed,
                severity=CheckpointSeverity(rule.get("severity", "info")),
                message="" if passed else rule.get("message", "Check failed"),
                evidence={k: v for k, v in context.items() if not callable(v)},
            )
            result.checks.append(check)

        self._history.append(result)

        if result.passed:
            logger.info(
                "Compliance gate '%s' PASSED (%d checks, %d warnings)",
                stage_name,
                len(result.checks),
                result.n_warnings,
            )
        else:
            logger.warning(
                "Compliance gate '%s' FAILED (%d errors)",
                stage_name,
                result.n_errors,
            )

        return result

    def add_rule(
        self,
        stage_name: str,
        name: str,
        check: Callable[[Dict[str, Any]], bool],
        severity: str = "warning",
        message: str = "",
    ) -> None:
        """Add a custom compliance rule for a stage."""
        if stage_name not in self._rules:
            self._rules[stage_name] = []
        self._rules[stage_name].append({
            "name": name,
            "check": check,
            "severity": severity,
            "message": message,
        })

    @property
    def history(self) -> List[CheckpointResult]:
        return list(self._history)

    def summary(self) -> str:
        """Human-readable compliance summary."""
        if not self._history:
            return "No compliance checkpoints evaluated."

        lines = [
            "Compliance Checkpoint Summary (S21)",
            "=" * 60,
        ]
        for result in self._history:
            status = "PASS" if result.passed else "FAIL"
            lines.append(
                f"  [{status}] {result.stage_name}: "
                f"{len(result.checks)} checks, "
                f"{result.n_warnings} warnings, "
                f"{result.n_errors} errors"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Escalation Protocol
# ---------------------------------------------------------------------------


class EscalationLevel(str, Enum):
    """Escalation levels for governance issues."""

    AUTOMATED = "automated"  # Auto-resolved by system rules
    RESEARCHER = "researcher"  # Requires researcher review
    ML_LEAD = "ml_lead"  # Requires ML lead decision
    OPERATOR = "operator"  # Requires operator/admin action
    EMERGENCY = "emergency"  # Emergency halt — all hands


@dataclass
class EscalationRecord:
    """Record of a governance escalation."""

    escalation_id: str = ""
    source_checkpoint_id: str = ""
    stage_name: str = ""
    level: str = "researcher"
    reason: str = ""
    checks_failed: List[str] = field(default_factory=list)
    timestamp: str = ""
    resolved: bool = False
    resolved_by: str = ""
    resolution: str = ""

    def __post_init__(self) -> None:
        if not self.escalation_id:
            self.escalation_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ESCALATION_LOG = "data/governance/escalations.jsonl"


class EscalationProtocol:
    """Manages escalation of governance issues.

    When compliance checks fail, issues are escalated to the
    appropriate level based on severity:
      - WARNING checks → researcher level
      - ERROR checks → ml_lead level
      - CRITICAL checks → operator level (pipeline halt)

    Escalations are logged to an append-only JSONL file.
    """

    def __init__(self, log_path: str = DEFAULT_ESCALATION_LOG) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[EscalationRecord] = []

    def escalate(
        self,
        checkpoint: CheckpointResult,
        level: Optional[str] = None,
    ) -> EscalationRecord:
        """Escalate failed compliance checks.

        If level is not specified, auto-determines based on severity:
          - CRITICAL → operator
          - ERROR → ml_lead
          - WARNING → researcher
        """
        if level is None:
            level = self._determine_level(checkpoint)

        failed_checks = [
            c.check_name for c in checkpoint.checks if not c.passed
        ]

        record = EscalationRecord(
            source_checkpoint_id=checkpoint.checkpoint_id,
            stage_name=checkpoint.stage_name,
            level=level,
            reason=f"Compliance gate failed at {checkpoint.stage_name}",
            checks_failed=failed_checks,
        )

        self._records.append(record)
        self._persist(record)

        logger.warning(
            "ESCALATION [%s] at %s: %s (checks: %s)",
            level.upper(),
            checkpoint.stage_name,
            record.reason,
            ", ".join(failed_checks),
        )

        return record

    def resolve(
        self, escalation_id: str, resolved_by: str, resolution: str
    ) -> EscalationRecord:
        """Mark an escalation as resolved."""
        for record in self._records:
            if record.escalation_id == escalation_id:
                record.resolved = True
                record.resolved_by = resolved_by
                record.resolution = resolution
                return record
        raise ValueError(f"Escalation {escalation_id} not found")

    def pending_escalations(self) -> List[EscalationRecord]:
        """Return all unresolved escalations."""
        # Reload from disk to include any persisted records
        all_records = self._load_all()
        return [r for r in all_records if not r.resolved]

    def query(
        self,
        level: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[EscalationRecord]:
        """Query escalation history."""
        records = self._load_all()
        if level:
            records = [r for r in records if r.level == level]
        if stage:
            records = [r for r in records if r.stage_name == stage]
        return records

    def _determine_level(self, checkpoint: CheckpointResult) -> str:
        """Auto-determine escalation level from check severities."""
        severities = [
            c.severity for c in checkpoint.checks if not c.passed
        ]
        if CheckpointSeverity.CRITICAL in severities:
            return EscalationLevel.OPERATOR.value
        if CheckpointSeverity.ERROR in severities:
            return EscalationLevel.ML_LEAD.value
        return EscalationLevel.RESEARCHER.value

    def _persist(self, record: EscalationRecord) -> None:
        """Append record to JSONL log."""
        with open(self._path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _load_all(self) -> List[EscalationRecord]:
        """Load all records from JSONL."""
        if not self._path.exists():
            return list(self._records)
        records: List[EscalationRecord] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(EscalationRecord(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records
