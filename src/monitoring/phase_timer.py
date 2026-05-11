"""Lightweight phase timer stub for pipeline monitoring."""

from __future__ import annotations

import time
from typing import Any, Dict


class PhaseTimer:
    def __init__(self):
        self._start = time.monotonic()
        self._timings: Dict[str, float] = {}

    def get_timings(self) -> Dict[str, float]:
        return dict(self._timings)

    def total_seconds(self) -> float:
        return time.monotonic() - self._start

    def summary(self) -> str:
        lines = []
        for name, dur in self._timings.items():
            lines.append(f"  {name}: {dur:.1f}s")
        lines.append(f"  Total: {self.total_seconds():.1f}s")
        return "\n".join(lines)
