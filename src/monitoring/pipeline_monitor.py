"""Lightweight pipeline monitor stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FreshnessCheck:
    source: str = ""
    status: str = "ok"
    staleness_hours: float = 0.0
    sla_hours: float = 24.0


class PipelineMonitor:
    def check_data_freshness(self, cache_dir: str = "") -> List[FreshnessCheck]:
        return []
