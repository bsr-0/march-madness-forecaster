"""Lightweight resource tracking stubs for pipeline monitoring."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ResourceBudget:
    max_wall_seconds: float = 3600
    max_memory_mb: int = 8192
    phase_budgets: Dict[str, float] = field(default_factory=dict)


class ResourceTracker:
    def __init__(self, budget: Optional[ResourceBudget] = None):
        self.budget = budget or ResourceBudget()
        self._start = time.monotonic()
        self._phases: Dict[str, float] = {}
        self._current_phase: Optional[str] = None

    @contextmanager
    def phase(self, name: str):
        t0 = time.monotonic()
        self._current_phase = name
        try:
            yield self
        finally:
            self._phases[name] = time.monotonic() - t0
            self._current_phase = None

    def summary(self) -> str:
        elapsed = time.monotonic() - self._start
        parts = [f"Total wall: {elapsed:.1f}s"]
        for name, dur in self._phases.items():
            parts.append(f"  {name}: {dur:.1f}s")
        return "\n".join(parts)

    def check_budget(self) -> None:
        pass

    def budget_utilization(self) -> float:
        elapsed = time.monotonic() - self._start
        if self.budget.max_wall_seconds > 0:
            return elapsed / self.budget.max_wall_seconds
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        elapsed = time.monotonic() - self._start
        return {
            "total_wall_seconds": round(elapsed, 1),
            "total_cpu_seconds": round(elapsed, 1),
            "peak_memory_mb": 0,
            "phases": {k: round(v, 1) for k, v in self._phases.items()},
        }
