"""Risk report for LOYO backtest results.

Stub implementation — provides the RiskReport interface consumed by
backtest_harness.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class RiskReport:
    """Risk assessment from LOYO fold-level Brier scores."""

    mean_brier: float = 0.0
    std_brier: float = 0.0
    worst_year: int = 0
    worst_brier: float = 0.0
    n_years: int = 0

    @classmethod
    def from_loyo_results(cls, year_briers: Dict[int, float]) -> "RiskReport":
        if not year_briers:
            return cls()
        briers = list(year_briers.values())
        worst_year = max(year_briers, key=year_briers.get)
        return cls(
            mean_brier=float(np.mean(briers)),
            std_brier=float(np.std(briers, ddof=1)) if len(briers) > 1 else 0.0,
            worst_year=worst_year,
            worst_brier=year_briers[worst_year],
            n_years=len(briers),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_brier": self.mean_brier,
            "std_brier": self.std_brier,
            "worst_year": self.worst_year,
            "worst_brier": self.worst_brier,
            "n_years": self.n_years,
        }
