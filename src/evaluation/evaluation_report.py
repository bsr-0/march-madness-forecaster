"""Evaluation report data structures for LOYO backtesting.

Provides EvaluationReport, AggregateEvaluationReport, and ReproducibilityInfo
consumed by backtest_harness.py.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .bootstrap_metrics import BootstrapResult
from .calibration_diagnostics import CalibrationReport


@dataclass
class ReproducibilityInfo:
    """Captures environment info for reproducibility tracking."""

    python_version: str = ""
    platform_info: str = ""
    random_seed: int = 0
    feature_hash: str = ""

    @classmethod
    def capture(cls, random_seed: int = 42) -> "ReproducibilityInfo":
        return cls(
            python_version=sys.version,
            platform_info=platform.platform(),
            random_seed=random_seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform_info,
            "random_seed": self.random_seed,
            "feature_hash": self.feature_hash,
        }


@dataclass
class EvaluationReport:
    """Structured evaluation report for a single LOYO fold."""

    year: int
    model_name: str = "pipeline"
    global_metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    round_analysis: Any = None  # RoundReport from round_analysis.py
    calibration_report: Optional[CalibrationReport] = None
    training_years: List[int] = field(default_factory=list)
    reproducibility: Optional[ReproducibilityInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "year": self.year,
            "model_name": self.model_name,
            "training_years": self.training_years,
            "feature_hash": self.reproducibility.feature_hash if self.reproducibility else "",
            "global_metrics": {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.global_metrics.items()},
        }
        if self.round_analysis is not None:
            result["round_analysis"] = (
                self.round_analysis.to_dict() if hasattr(self.round_analysis, "to_dict") else self.round_analysis
            )
        if self.calibration_report is not None:
            result["calibration_diagnostics"] = self.calibration_report.to_dict()
        if self.reproducibility is not None:
            result["reproducibility"] = self.reproducibility.to_dict()
        return result


@dataclass
class AggregateEvaluationReport:
    """Aggregate of per-year evaluation reports."""

    year_reports: List[EvaluationReport] = field(default_factory=list)
    aggregate_metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    reproducibility: Optional[ReproducibilityInfo] = None

    @property
    def summary(self) -> str:
        n = len(self.year_reports)
        brier = self.aggregate_metrics.get("brier_score")
        b_str = f"Brier={brier.estimate:.4f}" if brier else "Brier=N/A"
        years = [r.year for r in self.year_reports]
        return f"Aggregate ({n} years, {min(years)}-{max(years)}): {b_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_years": len(self.year_reports),
            "years": [r.year for r in self.year_reports],
            "aggregate_metrics": {
                k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.aggregate_metrics.items()
            },
            "year_reports": [r.to_dict() for r in self.year_reports],
        }
