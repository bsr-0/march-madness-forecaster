"""Lightweight cost tracking stubs for pipeline monitoring."""

from __future__ import annotations

from typing import Any, Dict


class CostTracker:
    def __init__(self):
        self._runs = []

    def compute_phase_costs(self, phases: Dict[str, Any]) -> Dict[str, float]:
        return {k: 0.0 for k in phases}

    def add_run(self, **kwargs) -> None:
        self._runs.append(kwargs)

    def summary(self) -> str:
        return f"CostTracker: {len(self._runs)} run(s) logged"
