"""Lightweight experiment registry stub."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ExperimentRegistry:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> str:
        self._records.append(record)
        return f"exp_{len(self._records)}"

    def best(self, metric: str = "loyo_mean_brier") -> Optional[Dict[str, Any]]:
        if not self._records:
            return None
        valid = [r for r in self._records if metric in r]
        if not valid:
            return None
        return min(valid, key=lambda r: r[metric])
