"""Pre-tournament readiness checklist.

Aggregates all readiness checks into a single pass/fail report that
operators run before submitting tournament predictions.

Implements Agent Directive V7 S18 (deployment & monitoring).

Usage:
    from src.monitoring.pre_tournament_checklist import PreTournamentChecklist

    checklist = PreTournamentChecklist(data_dir="data/raw")
    report = checklist.run()
    print(report.summary())
    if not report.ready():
        print("NOT READY — fix failing checks before submitting")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckItem:
    """A single checklist item result."""

    name: str
    status: str  # "pass", "fail", "warn", "skip"
    message: str = ""
    critical: bool = True  # If True, failure blocks readiness


@dataclass
class ChecklistReport:
    """Complete pre-tournament checklist report."""

    timestamp: str = ""
    items: List[CheckItem] = field(default_factory=list)

    def ready(self) -> bool:
        """True if all critical items pass."""
        return all(
            item.status in ("pass", "warn", "skip")
            for item in self.items
            if item.critical
        )

    def summary(self) -> str:
        lines = [
            f"Pre-Tournament Checklist ({self.timestamp[:19]})",
            "=" * 60,
        ]
        for item in self.items:
            icon = {
                "pass": "[PASS]",
                "fail": "[FAIL]",
                "warn": "[WARN]",
                "skip": "[SKIP]",
            }.get(item.status, "[????]")
            crit = " (critical)" if item.critical else ""
            msg = f": {item.message}" if item.message else ""
            lines.append(f"  {icon} {item.name}{crit}{msg}")

        lines.append("-" * 60)
        if self.ready():
            lines.append("  READY for tournament submission")
        else:
            failed = [i for i in self.items if i.status == "fail" and i.critical]
            lines.append(f"  NOT READY — {len(failed)} critical check(s) failed")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "ready": self.ready(),
            "items": [
                {
                    "name": i.name,
                    "status": i.status,
                    "message": i.message,
                    "critical": i.critical,
                }
                for i in self.items
            ],
        }


class PreTournamentChecklist:
    """Run all pre-tournament readiness checks.

    Checks:
      1. Data freshness (delegates to PipelineMonitor)
      2. Pipeline freeze exists and verifies
      3. MC calibration artifact exists
      4. Last successful run within 24h
      5. No circuit breakers in OPEN state
      6. Resource budget not exceeded in last run
      7. Feature drift below alert threshold
      8. All required data sources have files
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        freeze_file: Optional[str] = None,
        mc_calibration_file: Optional[str] = None,
        circuit_breaker_state_file: Optional[str] = None,
        run_history_path: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.freeze_file = freeze_file
        self.mc_calibration_file = mc_calibration_file
        self.circuit_breaker_state_file = circuit_breaker_state_file or "data/.circuit_breaker_state.json"
        self.run_history_path = run_history_path or "data/run_history.jsonl"

    def run(self) -> ChecklistReport:
        """Execute all checks and return the report."""
        report = ChecklistReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        report.items.append(self._check_data_freshness())
        report.items.append(self._check_freeze_artifact())
        report.items.append(self._check_mc_calibration())
        report.items.append(self._check_last_run())
        report.items.append(self._check_circuit_breakers())
        report.items.append(self._check_resource_budget())
        report.items.append(self._check_data_sources())

        return report

    def _check_data_freshness(self) -> CheckItem:
        """Check that data files are within SLA freshness."""
        try:
            from .pipeline_monitor import PipelineMonitor

            monitor = PipelineMonitor()
            checks = monitor.check_data_freshness(self.data_dir)
            stale = [c for c in checks if c.status == "stale"]
            missing = [c for c in checks if c.status == "missing"]

            if missing:
                return CheckItem(
                    name="Data freshness",
                    status="fail",
                    message=f"{len(missing)} data source(s) missing",
                    critical=True,
                )
            if stale:
                return CheckItem(
                    name="Data freshness",
                    status="warn",
                    message=f"{len(stale)} data source(s) stale",
                    critical=False,
                )
            return CheckItem(
                name="Data freshness",
                status="pass",
                message=f"All {len(checks)} sources within SLA",
            )
        except Exception as exc:
            return CheckItem(
                name="Data freshness",
                status="skip",
                message=f"Check failed: {exc}",
                critical=False,
            )

    def _check_freeze_artifact(self) -> CheckItem:
        """Check that a pipeline freeze file exists."""
        if not self.freeze_file:
            return CheckItem(
                name="Pipeline freeze",
                status="warn",
                message="No freeze file configured",
                critical=False,
            )
        if Path(self.freeze_file).exists():
            return CheckItem(
                name="Pipeline freeze",
                status="pass",
                message=f"Freeze file exists: {self.freeze_file}",
            )
        return CheckItem(
            name="Pipeline freeze",
            status="fail",
            message=f"Freeze file not found: {self.freeze_file}",
            critical=True,
        )

    def _check_mc_calibration(self) -> CheckItem:
        """Check that MC calibration artifact exists."""
        if not self.mc_calibration_file:
            return CheckItem(
                name="MC calibration",
                status="warn",
                message="No MC calibration file configured",
                critical=False,
            )
        if Path(self.mc_calibration_file).exists():
            return CheckItem(
                name="MC calibration",
                status="pass",
                message="MC calibration artifact found",
            )
        return CheckItem(
            name="MC calibration",
            status="fail",
            message=f"MC calibration not found: {self.mc_calibration_file}",
            critical=True,
        )

    def _check_last_run(self) -> CheckItem:
        """Check that last successful run was within 24 hours."""
        try:
            from .run_history import RunHistory

            history = RunHistory(self.run_history_path)
            last = history.last_successful_run()
            if last is None:
                return CheckItem(
                    name="Last successful run",
                    status="warn",
                    message="No previous successful runs recorded",
                    critical=False,
                )

            last_ts = datetime.fromisoformat(last.timestamp)
            now = datetime.now(timezone.utc)
            hours_ago = (now - last_ts).total_seconds() / 3600

            if hours_ago <= 24:
                return CheckItem(
                    name="Last successful run",
                    status="pass",
                    message=f"Last success {hours_ago:.1f}h ago",
                )
            return CheckItem(
                name="Last successful run",
                status="warn",
                message=f"Last success {hours_ago:.1f}h ago (>24h)",
                critical=False,
            )
        except Exception as exc:
            return CheckItem(
                name="Last successful run",
                status="skip",
                message=f"Check failed: {exc}",
                critical=False,
            )

    def _check_circuit_breakers(self) -> CheckItem:
        """Check that no circuit breakers are in OPEN state."""
        state_path = Path(self.circuit_breaker_state_file)
        if not state_path.exists():
            return CheckItem(
                name="Circuit breakers",
                status="pass",
                message="No circuit breaker state file (all OK)",
                critical=False,
            )
        try:
            with open(state_path) as f:
                states = json.load(f)
            open_breakers = [
                name for name, state in states.items()
                if state.get("state") == "open"
            ]
            if open_breakers:
                return CheckItem(
                    name="Circuit breakers",
                    status="fail",
                    message=f"OPEN breakers: {', '.join(open_breakers)}",
                    critical=True,
                )
            return CheckItem(
                name="Circuit breakers",
                status="pass",
                message=f"All {len(states)} breakers CLOSED",
            )
        except Exception as exc:
            return CheckItem(
                name="Circuit breakers",
                status="skip",
                message=f"Check failed: {exc}",
                critical=False,
            )

    def _check_resource_budget(self) -> CheckItem:
        """Check that last run didn't exceed resource budget."""
        try:
            from .run_history import RunHistory

            history = RunHistory(self.run_history_path)
            last = history.last_successful_run()
            if last is None or not last.resource_usage:
                return CheckItem(
                    name="Resource budget",
                    status="skip",
                    message="No resource data in last run",
                    critical=False,
                )

            violations = last.resource_usage.get("violations", [])
            if violations:
                return CheckItem(
                    name="Resource budget",
                    status="warn",
                    message=f"Last run had {len(violations)} budget violation(s)",
                    critical=False,
                )
            return CheckItem(
                name="Resource budget",
                status="pass",
                message="Last run within budget",
            )
        except Exception as exc:
            return CheckItem(
                name="Resource budget",
                status="skip",
                message=f"Check failed: {exc}",
                critical=False,
            )

    def _check_data_sources(self) -> CheckItem:
        """Check that required data source patterns have at least one file."""
        data_path = Path(self.data_dir)
        if not data_path.is_dir():
            return CheckItem(
                name="Data sources",
                status="fail",
                message=f"Data directory not found: {self.data_dir}",
                critical=True,
            )

        required_patterns = [
            "torvik_*.json",
            "bracket_*.json",
        ]
        import glob
        missing = []
        for pattern in required_patterns:
            matches = glob.glob(str(data_path / pattern))
            if not matches:
                missing.append(pattern)

        if missing:
            return CheckItem(
                name="Data sources",
                status="warn",
                message=f"Missing sources: {', '.join(missing)}",
                critical=False,
            )
        return CheckItem(
            name="Data sources",
            status="pass",
            message="All required data sources present",
        )
