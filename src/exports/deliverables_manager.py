"""Structured deliverables manager for pipeline output artifacts.

Organizes predictions, reports, audit trails, and metadata into a versioned
directory structure with confidence intervals and decision records.

Implements Agent Directive V7 S16 (final deliverables).

Usage:
    from src.exports.deliverables_manager import DeliverablesManager

    mgr = DeliverablesManager(year=2026, mode="calibration")
    mgr.export_predictions(predictions_dict)
    mgr.export_risk_report(risk_report)
    mgr.export_decision_record(record)
    mgr.finalize()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_BASE = "outputs"


@dataclass
class DecisionRecord:
    """Documents the rationale for key pipeline decisions.

    Captures why specific choices were made for ensemble weights,
    calibration method, feature set, and hyperparameters.
    """

    timestamp: str = ""
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    ensemble_rationale: str = ""
    calibration_method: str = ""
    calibration_rationale: str = ""
    feature_set_size: int = 0
    feature_selection_rationale: str = ""
    hyperparameter_summary: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_rationale: str = ""
    key_tradeoffs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeliverableManifest:
    """Manifest tracking all artifacts in a deliverable package."""

    timestamp: str = ""
    year: int = 2026
    mode: str = "calibration"
    artifacts: Dict[str, str] = field(default_factory=dict)  # name → relative path
    config_hash: str = ""
    code_version: str = ""
    data_hashes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeliverablesManager:
    """Manages the structured output of pipeline deliverables.

    Creates a versioned directory structure:
        outputs/{year}/{mode}_{timestamp}/
            predictions/    — prediction CSV and confidence intervals
            reports/        — risk report, regime analysis, scenario projections
            audit/          — decision records, governance audit trail
            metadata/       — manifest, config snapshot, data hashes
    """

    def __init__(
        self,
        year: int = 2026,
        mode: str = "calibration",
        output_base: str = DEFAULT_OUTPUT_BASE,
        timestamp: Optional[str] = None,
    ) -> None:
        self._year = year
        self._mode = mode
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._output_dir = Path(output_base) / str(year) / f"{mode}_{ts}"
        self._manifest = DeliverableManifest(year=year, mode=mode)
        self._finalized = False

        # Create directory structure
        for subdir in ["predictions", "reports", "audit", "metadata"]:
            (self._output_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def export_predictions(
        self,
        predictions: Dict[str, float],
        confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Export predictions with optional confidence intervals.

        Args:
            predictions: team_matchup → probability mapping.
            confidence_intervals: Optional team_matchup → {lower, upper, mean}
                from bootstrap LOYO folds.

        Returns:
            Path to the predictions file.
        """
        pred_path = self._output_dir / "predictions" / "predictions.json"
        with open(pred_path, "w") as f:
            json.dump(
                {
                    "predictions": predictions,
                    "count": len(predictions),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        self._manifest.artifacts["predictions"] = "predictions/predictions.json"

        if confidence_intervals:
            ci_path = self._output_dir / "predictions" / "confidence_intervals.json"
            with open(ci_path, "w") as f:
                json.dump(confidence_intervals, f, indent=2)
            self._manifest.artifacts["confidence_intervals"] = (
                "predictions/confidence_intervals.json"
            )

        logger.info("Exported %d predictions", len(predictions))
        return str(pred_path)

    def export_risk_report(self, risk_data: Dict[str, Any]) -> str:
        """Export risk assessment report.

        Args:
            risk_data: Risk metrics dict (drawdown, tail-loss, trend, etc.)

        Returns:
            Path to the risk report file.
        """
        path = self._output_dir / "reports" / "risk_report.json"
        with open(path, "w") as f:
            json.dump(risk_data, f, indent=2)
        self._manifest.artifacts["risk_report"] = "reports/risk_report.json"

        # Also generate human-readable summary
        summary_path = self._output_dir / "reports" / "risk_summary.txt"
        summary_lines = ["Risk Assessment Summary", "=" * 50]
        for key, val in risk_data.items():
            if isinstance(val, (int, float)):
                summary_lines.append(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
            elif isinstance(val, dict):
                summary_lines.append(f"  {key}:")
                for k2, v2 in val.items():
                    summary_lines.append(f"    {k2}: {v2}")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines))
        self._manifest.artifacts["risk_summary"] = "reports/risk_summary.txt"

        return str(path)

    def export_regime_analysis(self, regime_data: Dict[str, Any]) -> str:
        """Export regime-conditional analysis."""
        path = self._output_dir / "reports" / "regime_analysis.json"
        with open(path, "w") as f:
            json.dump(regime_data, f, indent=2)
        self._manifest.artifacts["regime_analysis"] = "reports/regime_analysis.json"
        return str(path)

    def export_scenario_analysis(self, scenario_data: Dict[str, Any]) -> str:
        """Export named scenario projections."""
        path = self._output_dir / "reports" / "scenario_analysis.json"
        with open(path, "w") as f:
            json.dump(scenario_data, f, indent=2)
        self._manifest.artifacts["scenario_analysis"] = "reports/scenario_analysis.json"
        return str(path)

    def export_decision_record(self, record: DecisionRecord) -> str:
        """Export the decision record documenting key choices."""
        path = self._output_dir / "audit" / "decision_record.json"
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        self._manifest.artifacts["decision_record"] = "audit/decision_record.json"
        return str(path)

    def export_config_snapshot(self, config: Dict[str, Any]) -> str:
        """Export the pipeline configuration snapshot."""
        path = self._output_dir / "metadata" / "config_snapshot.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        self._manifest.artifacts["config_snapshot"] = "metadata/config_snapshot.json"
        return str(path)

    def export_evaluation_matrix(self, metrics: Dict[str, Any]) -> str:
        """Export the full evaluation matrix (S13 compliance)."""
        path = self._output_dir / "reports" / "evaluation_matrix.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        self._manifest.artifacts["evaluation_matrix"] = "reports/evaluation_matrix.json"
        return str(path)

    def set_provenance(
        self,
        config_hash: str = "",
        code_version: str = "",
        data_hashes: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set provenance metadata for the deliverable package."""
        self._manifest.config_hash = config_hash
        self._manifest.code_version = code_version
        if data_hashes:
            self._manifest.data_hashes = data_hashes

    def finalize(self) -> str:
        """Write the deliverable manifest and mark the package as complete.

        Returns:
            Path to the manifest file.
        """
        manifest_path = self._output_dir / "metadata" / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self._manifest.to_dict(), f, indent=2)
        self._finalized = True

        logger.info(
            "Deliverable package finalized: %s (%d artifacts)",
            self._output_dir,
            len(self._manifest.artifacts),
        )
        return str(manifest_path)

    def summary(self) -> str:
        """Human-readable summary of the deliverable package."""
        lines = [
            f"Deliverable Package: {self._year} {self._mode}",
            "=" * 50,
            f"Output directory: {self._output_dir}",
            f"Artifacts: {len(self._manifest.artifacts)}",
        ]
        for name, path in sorted(self._manifest.artifacts.items()):
            lines.append(f"  {name}: {path}")
        if self._finalized:
            lines.append("\nStatus: FINALIZED")
        else:
            lines.append("\nStatus: In progress")
        return "\n".join(lines)
