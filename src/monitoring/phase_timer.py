"""Wall-clock timing for pipeline phases.

Records elapsed time per pipeline phase and produces a timing report.
Integrates with ExperimentRegistry for cross-run comparison.

Usage:
    timer = PhaseTimer()
    with timer.phase("data_loading"):
        load_data()
    with timer.phase("feature_engineering"):
        build_features()
    print(timer.summary())
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List

logger = logging.getLogger(__name__)


@dataclass
class PhaseTimingRecord:
    """Timing record for a single pipeline phase."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_seconds: float = 0.0


class PhaseTimer:
    """Wall-clock timer for pipeline phases.

    Tracks elapsed time per named phase and produces summary reports.
    Thread-safe for sequential (non-overlapping) phase usage.
    """

    def __init__(self) -> None:
        self._records: List[PhaseTimingRecord] = []
        self._active_phase: str | None = None
        self._start_time: float = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Context manager to time a named pipeline phase.

        Args:
            name: Human-readable phase name (e.g. "data_loading").
        """
        if self._active_phase is not None:
            logger.warning(
                "Starting phase '%s' while '%s' is still active",
                name,
                self._active_phase,
            )
        self._active_phase = name
        record = PhaseTimingRecord(name=name, start_time=time.perf_counter())
        try:
            yield
        finally:
            record.end_time = time.perf_counter()
            record.elapsed_seconds = record.end_time - record.start_time
            self._records.append(record)
            self._active_phase = None
            logger.info(
                "Phase '%s' completed in %.1fs",
                name,
                record.elapsed_seconds,
            )

    def get_timings(self) -> Dict[str, float]:
        """Return phase name → elapsed seconds mapping.

        Includes a ``total`` key with total wall-clock time since
        the timer was created.
        """
        timings = {r.name: round(r.elapsed_seconds, 2) for r in self._records}
        timings["total"] = round(time.perf_counter() - self._start_time, 2)
        return timings

    def total_seconds(self) -> float:
        """Total wall-clock seconds since timer creation."""
        return time.perf_counter() - self._start_time

    def summary(self) -> str:
        """Human-readable timing summary."""
        timings = self.get_timings()
        total = timings.pop("total", 0.0)
        lines = ["Pipeline Phase Timings", "=" * 40]
        for name, elapsed in timings.items():
            pct = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"  {name:30s} {elapsed:8.1f}s  ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':30s} {total:8.1f}s")
        return "\n".join(lines)
