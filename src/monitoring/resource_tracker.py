"""Resource tracking for pipeline phases: wall-clock time, memory, and CPU.

Extends the :class:`PhaseTimer` pattern with memory profiling via
``tracemalloc`` and CPU time via ``time.process_time()``.  Supports optional
compute-budget enforcement that warns or raises when limits are exceeded.

Implements Agent Directive V7 S20 (compute budget management).

Usage:
    from src.monitoring.resource_tracker import ResourceTracker, ResourceBudget

    budget = ResourceBudget(max_wall_seconds=3600, max_memory_mb=8192)
    tracker = ResourceTracker(budget=budget)
    with tracker.phase("data_loading"):
        load_data()
    with tracker.phase("model_training"):
        train_model()
    print(tracker.summary())
    tracker.check_budget()  # Warns or raises if over budget
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceBudget:
    """Compute budget limits for pipeline execution.

    Set any field to ``None`` to skip that check.
    """

    max_wall_seconds: Optional[float] = 3600.0  # 1 hour default
    max_memory_mb: Optional[float] = 8192.0  # 8 GB default
    max_total_cpu_seconds: Optional[float] = None  # No CPU limit by default
    strict: bool = False  # If True, raise on budget exceeded; else warn


@dataclass
class PhaseResourceRecord:
    """Resource usage for a single pipeline phase."""

    name: str
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0


class ResourceTracker:
    """Track wall-clock time, CPU time, and memory per pipeline phase.

    Drop-in replacement for :class:`PhaseTimer` with additional resource
    tracking.  The ``phase()`` context manager interface is identical.
    """

    def __init__(self, budget: Optional[ResourceBudget] = None) -> None:
        self._records: List[PhaseResourceRecord] = []
        self._active_phase: Optional[str] = None
        self._start_time: float = time.perf_counter()
        self._budget = budget or ResourceBudget()
        self._tracemalloc_started: bool = False

        # Start tracemalloc if not already running
        if not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
                self._tracemalloc_started = True
            except Exception:
                logger.debug("tracemalloc.start() failed; memory tracking disabled")

    @contextmanager
    def phase(self, name: str) -> Generator[None, None, None]:
        """Context manager to time and measure a named pipeline phase.

        Tracks wall-clock time, CPU time, and peak memory allocation.
        """
        if self._active_phase is not None:
            logger.warning(
                "Starting phase '%s' while '%s' is still active",
                name,
                self._active_phase,
            )
        self._active_phase = name

        # Snapshot memory before phase
        mem_before = self._current_memory_mb()
        cpu_before = time.process_time()
        wall_before = time.perf_counter()

        # Reset tracemalloc peak for per-phase measurement
        if tracemalloc.is_tracing():
            try:
                tracemalloc.reset_peak()
            except Exception:
                pass

        try:
            yield
        finally:
            wall_after = time.perf_counter()
            cpu_after = time.process_time()

            # Peak memory during this phase
            peak_mb = self._peak_memory_mb()
            mem_after = self._current_memory_mb()

            record = PhaseResourceRecord(
                name=name,
                wall_seconds=round(wall_after - wall_before, 2),
                cpu_seconds=round(cpu_after - cpu_before, 2),
                peak_memory_mb=round(peak_mb, 1),
                memory_allocated_mb=round(max(mem_after - mem_before, 0), 1),
            )
            self._records.append(record)
            self._active_phase = None

            logger.info(
                "Phase '%s' completed: %.1fs wall, %.1fs CPU, %.1f MB peak",
                name,
                record.wall_seconds,
                record.cpu_seconds,
                record.peak_memory_mb,
            )

    def get_timings(self) -> Dict[str, float]:
        """Return phase name -> elapsed wall-clock seconds (PhaseTimer compat)."""
        timings = {r.name: r.wall_seconds for r in self._records}
        timings["total"] = round(time.perf_counter() - self._start_time, 2)
        return timings

    def total_seconds(self) -> float:
        """Total wall-clock seconds since tracker creation."""
        return time.perf_counter() - self._start_time

    def check_budget(self) -> List[str]:
        """Check resource usage against budget limits.

        Returns list of violation messages.  If ``budget.strict`` is True,
        raises :class:`ComputeBudgetExceeded` on the first violation.
        """
        from ..exceptions import ComputeBudgetExceeded

        violations: List[str] = []
        budget = self._budget

        # Wall-clock check
        if budget.max_wall_seconds is not None:
            total_wall = self.total_seconds()
            if total_wall > budget.max_wall_seconds:
                violations.append(
                    f"Wall-clock budget exceeded: {total_wall:.0f}s > {budget.max_wall_seconds:.0f}s"
                )

        # Memory check
        if budget.max_memory_mb is not None:
            peak = max((r.peak_memory_mb for r in self._records), default=0)
            if peak > budget.max_memory_mb:
                violations.append(
                    f"Memory budget exceeded: {peak:.0f} MB > {budget.max_memory_mb:.0f} MB"
                )

        # CPU time check
        if budget.max_total_cpu_seconds is not None:
            total_cpu = sum(r.cpu_seconds for r in self._records)
            if total_cpu > budget.max_total_cpu_seconds:
                violations.append(
                    f"CPU budget exceeded: {total_cpu:.0f}s > {budget.max_total_cpu_seconds:.0f}s"
                )

        for v in violations:
            if budget.strict:
                raise ComputeBudgetExceeded(v)
            logger.warning("BUDGET: %s", v)

        return violations

    def summary(self) -> str:
        """Human-readable resource usage summary."""
        total_wall = self.total_seconds()
        lines = [
            "Pipeline Resource Usage",
            "=" * 70,
            f"  {'Phase':<30s} {'Wall':>8s}  {'CPU':>8s}  {'Peak MB':>10s}  {'Alloc MB':>10s}",
            "-" * 70,
        ]
        for r in self._records:
            pct = (r.wall_seconds / total_wall * 100) if total_wall > 0 else 0
            lines.append(
                f"  {r.name:<30s} {r.wall_seconds:>7.1f}s  {r.cpu_seconds:>7.1f}s"
                f"  {r.peak_memory_mb:>9.1f}  {r.memory_allocated_mb:>9.1f}"
                f"  ({pct:5.1f}%)"
            )
        lines.append("-" * 70)
        total_cpu = sum(r.cpu_seconds for r in self._records)
        peak_overall = max((r.peak_memory_mb for r in self._records), default=0)
        lines.append(
            f"  {'TOTAL':<30s} {total_wall:>7.1f}s  {total_cpu:>7.1f}s"
            f"  {peak_overall:>9.1f}"
        )

        # Budget status
        violations = self._check_budget_silent()
        if violations:
            lines.append("")
            lines.append("BUDGET VIOLATIONS:")
            for v in violations:
                lines.append(f"  ! {v}")
        else:
            lines.append("Budget: OK")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        """Structured output for experiment registry integration."""
        return {
            "total_wall_seconds": round(self.total_seconds(), 2),
            "total_cpu_seconds": round(sum(r.cpu_seconds for r in self._records), 2),
            "peak_memory_mb": round(
                max((r.peak_memory_mb for r in self._records), default=0), 1
            ),
            "phases": {
                r.name: {
                    "wall_seconds": r.wall_seconds,
                    "cpu_seconds": r.cpu_seconds,
                    "peak_memory_mb": r.peak_memory_mb,
                    "memory_allocated_mb": r.memory_allocated_mb,
                }
                for r in self._records
            },
            "budget": {
                "max_wall_seconds": self._budget.max_wall_seconds,
                "max_memory_mb": self._budget.max_memory_mb,
                "max_total_cpu_seconds": self._budget.max_total_cpu_seconds,
            },
            "violations": self._check_budget_silent(),
        }

    def _check_budget_silent(self) -> List[str]:
        """Check budget without logging or raising."""
        violations: List[str] = []
        budget = self._budget
        if budget.max_wall_seconds is not None:
            total_wall = self.total_seconds()
            if total_wall > budget.max_wall_seconds:
                violations.append(
                    f"Wall-clock: {total_wall:.0f}s > {budget.max_wall_seconds:.0f}s"
                )
        if budget.max_memory_mb is not None:
            peak = max((r.peak_memory_mb for r in self._records), default=0)
            if peak > budget.max_memory_mb:
                violations.append(
                    f"Memory: {peak:.0f} MB > {budget.max_memory_mb:.0f} MB"
                )
        if budget.max_total_cpu_seconds is not None:
            total_cpu = sum(r.cpu_seconds for r in self._records)
            if total_cpu > budget.max_total_cpu_seconds:
                violations.append(
                    f"CPU: {total_cpu:.0f}s > {budget.max_total_cpu_seconds:.0f}s"
                )
        return violations

    @staticmethod
    def _current_memory_mb() -> float:
        """Current memory allocated (MB)."""
        if not tracemalloc.is_tracing():
            return 0.0
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)

    @staticmethod
    def _peak_memory_mb() -> float:
        """Peak memory since last reset (MB)."""
        if not tracemalloc.is_tracing():
            return 0.0
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)
