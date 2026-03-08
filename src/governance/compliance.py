"""Compliance checkpoints integrated into pipeline stages."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckItem:
    """Result of a compliance check. Mirrors pre_tournament_checklist.CheckItem."""
    name: str
    status: str  # "pass", "fail", "warning", "skipped"
    details: str = ""
    timestamp: str = ""


@dataclass
class ComplianceCheckpoint:
    name: str
    stage: str  # pipeline stage where this runs: "post_data_load", "post_training", "pre_submission"
    check_fn: Callable[..., CheckItem]  # (ctx, data) -> CheckItem
    blocking: bool = True  # if True, pipeline halts on failure


class ComplianceRunner:
    def __init__(self, checkpoints: Optional[List[ComplianceCheckpoint]] = None):
        self._checkpoints = checkpoints or []
        self._results: Dict[str, List[CheckItem]] = {}

    def add_checkpoint(self, checkpoint: ComplianceCheckpoint) -> None:
        self._checkpoints.append(checkpoint)

    def run_stage_checks(self, stage: str, ctx: Any = None, data: Any = None) -> List[CheckItem]:
        """Run all checkpoints for a given stage. Returns list of CheckItems."""
        results = []
        for cp in self._checkpoints:
            if cp.stage != stage:
                continue
            try:
                item = cp.check_fn(ctx, data)
                if not item.timestamp:
                    item.timestamp = datetime.now(timezone.utc).isoformat()
                results.append(item)
                if cp.blocking and item.status == "fail":
                    logger.error("Blocking compliance check failed: %s - %s", cp.name, item.details)
            except Exception as e:
                results.append(CheckItem(
                    name=cp.name, status="fail",
                    details=f"Check raised exception: {e}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
        self._results[stage] = results
        return results

    def all_passed(self, stage: str) -> bool:
        """True if all checks for the stage passed (no failures)."""
        results = self._results.get(stage, [])
        return all(r.status != "fail" for r in results)

    def has_blocking_failure(self, stage: str) -> bool:
        """True if any blocking checkpoint failed."""
        results = self._results.get(stage, [])
        blocking_names = {cp.name for cp in self._checkpoints if cp.stage == stage and cp.blocking}
        return any(r.status == "fail" and r.name in blocking_names for r in results)

    def audit_trail(self) -> List[Dict[str, Any]]:
        """Full log of all checks run across all stages."""
        trail = []
        for stage, results in self._results.items():
            for item in results:
                trail.append({
                    "stage": stage,
                    "name": item.name,
                    "status": item.status,
                    "details": item.details,
                    "timestamp": item.timestamp,
                })
        return trail
