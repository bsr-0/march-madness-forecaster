"""Append-only pipeline run history log.

Records every pipeline execution with its outcome, duration, resource usage,
and Brier score.  Enables regression detection across runs.

Implements Agent Directive V7 S18 (deployment & monitoring — run history).

Usage:
    from src.monitoring.run_history import RunHistory, RunRecord

    history = RunHistory()
    record = RunRecord(mode="calibration", year=2026, status="success", brier_score=0.189)
    history.log_run(record)

    last = history.last_successful_run()
    runs = history.list_runs(limit=10)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = "data/run_history.jsonl"


@dataclass
class RunRecord:
    """A single pipeline run entry."""

    run_id: str = ""
    timestamp: str = ""
    config_hash: str = ""
    mode: str = "calibration"
    year: int = 2026
    status: str = "success"  # "success", "failure", "error"
    duration_seconds: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    brier_score: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class RunHistory:
    """Append-only JSONL log of pipeline runs.

    Provides querying, regression detection, and summary statistics.
    """

    def __init__(self, history_path: str = DEFAULT_HISTORY_PATH) -> None:
        self._path = Path(history_path)

    def log_run(self, record: RunRecord) -> None:
        """Append a run record to the history log."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        logger.info(
            "Logged run %s: status=%s, brier=%s, duration=%.1fs",
            record.run_id,
            record.status,
            record.brier_score,
            record.duration_seconds,
        )

    def list_runs(
        self,
        limit: int = 50,
        mode: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunRecord]:
        """List recent runs, newest first.

        Args:
            limit: Maximum number of runs to return.
            mode: Filter by pipeline mode.
            status: Filter by run status.
        """
        records = self._load_all()
        if mode:
            records = [r for r in records if r.mode == mode]
        if status:
            records = [r for r in records if r.status == status]
        return records[-limit:][::-1]

    def last_successful_run(self, mode: Optional[str] = None) -> Optional[RunRecord]:
        """Return the most recent successful run."""
        records = self._load_all()
        for record in reversed(records):
            if record.status == "success":
                if mode is None or record.mode == mode:
                    return record
        return None

    def check_regression(
        self,
        current_brier: float,
        threshold: float = 0.01,
        mode: str = "calibration",
    ) -> Optional[str]:
        """Check if current Brier score regressed from the last successful run.

        Args:
            current_brier: Current run's Brier score.
            threshold: Maximum allowed regression (higher Brier = worse).
            mode: Pipeline mode to compare against.

        Returns:
            Warning message if regression detected, None otherwise.
        """
        last = self.last_successful_run(mode=mode)
        if last is None or last.brier_score is None:
            return None

        delta = current_brier - last.brier_score
        if delta > threshold:
            return (
                f"Brier score regression detected: {current_brier:.4f} vs "
                f"previous {last.brier_score:.4f} (delta={delta:+.4f}, "
                f"threshold={threshold:.4f})"
            )
        return None

    def summary(self) -> str:
        """Human-readable summary of recent run history."""
        records = self.list_runs(limit=10)
        if not records:
            return "No pipeline runs recorded."

        lines = [
            "Pipeline Run History (last 10)",
            "=" * 70,
            f"  {'Run ID':<10s} {'Timestamp':<22s} {'Mode':<13s} {'Status':<8s} {'Brier':>8s} {'Duration':>10s}",
            "-" * 70,
        ]
        for r in records:
            brier_str = f"{r.brier_score:.4f}" if r.brier_score is not None else "n/a"
            dur_str = f"{r.duration_seconds:.1f}s"
            ts = r.timestamp[:19] if r.timestamp else "?"
            lines.append(
                f"  {r.run_id:<10s} {ts:<22s} {r.mode:<13s} {r.status:<8s} "
                f"{brier_str:>8s} {dur_str:>10s}"
            )

        # Trend
        successes = [r for r in records if r.status == "success" and r.brier_score is not None]
        if len(successes) >= 2:
            latest = successes[0].brier_score
            prev = successes[1].brier_score
            if latest is not None and prev is not None:
                delta = latest - prev
                trend = "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged"
                lines.append(f"\n  Trend: {trend} ({delta:+.4f})")

        return "\n".join(lines)

    def _load_all(self) -> List[RunRecord]:
        """Load all records from the JSONL file."""
        if not self._path.exists():
            return []
        records: List[RunRecord] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(RunRecord(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.debug("Skipping malformed run record: %s", exc)
        return records
