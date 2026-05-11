"""Lightweight run history stubs for pipeline monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunRecord:
    mode: str = ""
    year: int = 0
    status: str = ""
    duration_seconds: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    brier_score: Optional[float] = None
    error_message: Optional[str] = None


class RunHistory:
    def __init__(self):
        self._runs: List[RunRecord] = []

    def log_run(self, record: RunRecord) -> None:
        self._runs.append(record)

    def check_regression(self, brier: float, mode: str = "") -> Optional[str]:
        return None
