"""Minimal compliance gate primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ComplianceCheckResult:
    passed: bool = True
    checks: List[Dict[str, Any]] = field(default_factory=list)
    n_errors: int = 0
    n_warnings: int = 0


class ComplianceGate:
    """Conservative default gate: report success unless a caller extends it."""

    def check(self, stage_name: str, stage_context: Dict[str, Any]) -> ComplianceCheckResult:
        return ComplianceCheckResult(
            passed=True,
            checks=[{"stage": stage_name, "status": "pass", "context": dict(stage_context or {})}],
            n_errors=0,
            n_warnings=0,
        )
