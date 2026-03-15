"""Unified evaluation report for Selection Sunday backtests.

Instead of scattered outputs, produces a single artifact per backtest year
containing all metrics, diagnostics, baseline comparisons, and metadata.

Includes reproducibility controls: random_seed, simulation_runs,
library_versions.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .baselines import BaselineComparison
from .bootstrap_metrics import BootstrapResult
from .calibration_diagnostics import CalibrationReport
from .calibration_gate import CalibrationGateResult
from .round_analysis import RoundAnalysisReport
from .seed_gap_calibration import SeedGapCalibrationReport
from .snapshot_integrity import SnapshotMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility info
# ---------------------------------------------------------------------------

def collect_library_versions() -> Dict[str, str]:
    """Collect versions of key libraries for reproducibility."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    for lib_name in ("sklearn", "lightgbm", "xgboost", "scipy"):
        try:
            mod = __import__(lib_name)
            versions[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return versions


@dataclass
class ReproducibilityInfo:
    """Reproducibility controls for a backtest run."""

    random_seed: Optional[int] = None
    simulation_runs: int = 0
    library_versions: Dict[str, str] = field(default_factory=dict)
    python_version: str = ""
    platform_info: str = ""
    run_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "simulation_runs": self.simulation_runs,
            "library_versions": self.library_versions,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "run_timestamp": self.run_timestamp,
        }

    @classmethod
    def capture(
        cls,
        random_seed: Optional[int] = None,
        simulation_runs: int = 0,
    ) -> ReproducibilityInfo:
        """Capture current environment info."""
        return cls(
            random_seed=random_seed,
            simulation_runs=simulation_runs,
            library_versions=collect_library_versions(),
            python_version=platform.python_version(),
            platform_info=platform.platform(),
            run_timestamp=datetime.utcnow().isoformat() + "Z",
        )


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Unified evaluation artifact for a single backtest year.

    Contains everything needed to assess model quality:
    - Global metrics with CIs
    - Per-segment metrics with CIs
    - Calibration diagnostics
    - Seed-gap calibration breakdown
    - Baseline comparisons
    - Snapshot metadata
    - Reproducibility controls
    """

    year: int = 0
    model_name: str = ""

    # Metrics
    global_metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    round_analysis: Optional[RoundAnalysisReport] = None
    calibration_report: Optional[CalibrationReport] = None
    seed_gap_calibration: Optional[SeedGapCalibrationReport] = None
    calibration_gate: Optional[CalibrationGateResult] = None

    # Baselines
    baseline_comparisons: List[BaselineComparison] = field(default_factory=list)

    # Snapshot info
    snapshot_metadata: Optional[SnapshotMetadata] = None
    training_years: List[int] = field(default_factory=list)
    feature_hash: str = ""

    # Reproducibility
    reproducibility: Optional[ReproducibilityInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "year": self.year,
            "model_name": self.model_name,
            "training_years": self.training_years,
            "feature_hash": self.feature_hash,
        }

        if self.global_metrics:
            result["global_metrics"] = {
                k: v.to_dict() for k, v in self.global_metrics.items()
            }

        if self.round_analysis:
            result["round_analysis"] = self.round_analysis.to_dict()

        if self.calibration_report:
            result["calibration_diagnostics"] = self.calibration_report.to_dict()

        if self.seed_gap_calibration:
            result["seed_gap_calibration"] = self.seed_gap_calibration.to_dict()

        if self.calibration_gate:
            result["calibration_gate"] = self.calibration_gate.to_dict()

        if self.baseline_comparisons:
            result["baseline_comparisons"] = [
                bc.to_dict() for bc in self.baseline_comparisons
            ]

        if self.snapshot_metadata:
            result["snapshot_metadata"] = self.snapshot_metadata.to_dict()

        if self.reproducibility:
            result["reproducibility"] = self.reproducibility.to_dict()

        return result

    def to_json(self, path: str, indent: int = 2) -> None:
        """Serialise report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)
        logger.info("Evaluation report written to %s", path)

    @property
    def summary(self) -> str:
        """One-line summary of the report."""
        parts = [f"Year {self.year}"]
        if "brier_score" in self.global_metrics:
            b = self.global_metrics["brier_score"]
            parts.append(f"Brier={b.estimate:.4f} [{b.ci_lower:.4f}, {b.ci_upper:.4f}]")
        if self.calibration_gate:
            gate = "PASS" if self.calibration_gate.stage1_passed else "FAIL"
            parts.append(f"Gate={gate}")
        if self.baseline_comparisons:
            for bc in self.baseline_comparisons:
                if bc.baseline_name == "seed_only":
                    parts.append(f"vs seed: {bc.brier_improvement:+.4f}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Multi-year aggregate report
# ---------------------------------------------------------------------------

@dataclass
class AggregateEvaluationReport:
    """Aggregate evaluation across multiple backtest years."""

    year_reports: List[EvaluationReport] = field(default_factory=list)
    aggregate_metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    reproducibility: Optional[ReproducibilityInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_years": len(self.year_reports),
            "years": [r.year for r in self.year_reports],
            "aggregate_metrics": {
                k: v.to_dict() for k, v in self.aggregate_metrics.items()
            },
            "year_reports": [r.to_dict() for r in self.year_reports],
            "reproducibility": self.reproducibility.to_dict() if self.reproducibility else None,
        }

    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)

    @property
    def summary(self) -> str:
        lines = [f"Aggregate report: {len(self.year_reports)} years"]
        for yr in self.year_reports:
            lines.append(f"  {yr.summary}")
        if "brier_score" in self.aggregate_metrics:
            b = self.aggregate_metrics["brier_score"]
            lines.append(
                f"  Aggregate Brier={b.estimate:.4f} "
                f"[{b.ci_lower:.4f}, {b.ci_upper:.4f}]"
            )
        return "\n".join(lines)
