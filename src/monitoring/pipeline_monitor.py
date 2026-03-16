"""Lightweight pipeline monitoring for data freshness, feature drift, and prediction quality.

Designed for an annual-run tournament prediction system.  Generates JSON
reports rather than real-time dashboards.

Usage:
    python -m src.main monitor --data-dir data/raw
    python -m src.main save-baseline --features data/processed/team_game_features.parquet
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data freshness SLA definitions
# ---------------------------------------------------------------------------

# source_pattern → max acceptable staleness in hours
DEFAULT_FRESHNESS_SLA: Dict[str, float] = {
    "torvik_*.json": 168,        # Weekly refresh
    "advanced_metrics_*.json": 168,
    "bracket_*.json": 24,        # Daily during tournament
    "rosters_*.json": 336,       # Bi-weekly
    "sports_reference_*.json": 168,
    "manifest_*.json": 168,
    "historical_games_*.json": 720,  # Monthly
}


@dataclass
class DataFreshnessCheck:
    """Result of a single data source freshness check."""

    source: str
    file_path: str
    last_modified: str  # ISO 8601
    staleness_hours: float
    sla_hours: float
    status: str  # "ok", "stale", "missing"


@dataclass
class FeatureDriftCheck:
    """Result of a single feature drift check."""

    feature: str
    psi: float  # Population Stability Index
    status: str  # "ok", "warning", "alert"
    baseline_mean: float = 0.0
    current_mean: float = 0.0
    baseline_std: float = 0.0
    current_std: float = 0.0


@dataclass
class MonitoringReport:
    """Complete monitoring report."""

    timestamp: str
    data_freshness: List[DataFreshnessCheck] = field(default_factory=list)
    feature_drift: List[FeatureDriftCheck] = field(default_factory=list)
    prediction_quality: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "data_freshness": [asdict(c) for c in self.data_freshness],
            "feature_drift": [asdict(c) for c in self.feature_drift],
            "prediction_quality": self.prediction_quality,
            "alerts": self.alerts,
        }

    def summary(self) -> str:
        lines = [
            f"Monitoring Report ({self.timestamp[:19]})",
            f"  Data freshness: {sum(1 for c in self.data_freshness if c.status == 'ok')}/{len(self.data_freshness)} sources OK",
            f"  Feature drift: {sum(1 for c in self.feature_drift if c.status == 'ok')}/{len(self.feature_drift)} features OK",
        ]
        if self.prediction_quality:
            lines.append(f"  Prediction quality: Brier={self.prediction_quality.get('brier', 'n/a')}")
        if self.alerts:
            lines.append(f"  ALERTS ({len(self.alerts)}):")
            for alert in self.alerts:
                lines.append(f"    - {alert}")
        else:
            lines.append("  No alerts.")
        return "\n".join(lines)


class PipelineMonitor:
    """Lightweight monitoring for annual tournament prediction pipeline."""

    # PSI thresholds (standard industry values)
    PSI_WARNING = 0.1
    PSI_ALERT = 0.2

    def __init__(
        self,
        freshness_sla: Optional[Dict[str, float]] = None,
    ):
        self.freshness_sla = freshness_sla or DEFAULT_FRESHNESS_SLA

    def check_data_freshness(
        self, data_dir: str = "data/raw"
    ) -> List[DataFreshnessCheck]:
        """Check file modification times against SLA thresholds."""
        checks = []
        now = time.time()
        data_path = Path(data_dir)

        for pattern, sla_hours in self.freshness_sla.items():
            try:
                matches = sorted(glob.glob(str(data_path / pattern)))
            except (PermissionError, OSError):
                matches = []
            if not matches:
                checks.append(DataFreshnessCheck(
                    source=pattern,
                    file_path="",
                    last_modified="",
                    staleness_hours=float("inf"),
                    sla_hours=sla_hours,
                    status="missing",
                ))
                continue

            # Use the most recently modified file
            latest = max(matches, key=os.path.getmtime)
            mtime = os.path.getmtime(latest)
            staleness_hours = (now - mtime) / 3600
            last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

            status = "ok" if staleness_hours <= sla_hours else "stale"
            checks.append(DataFreshnessCheck(
                source=pattern,
                file_path=latest,
                last_modified=last_modified,
                staleness_hours=round(staleness_hours, 1),
                sla_hours=sla_hours,
                status=status,
            ))

        return checks

    def check_feature_drift(
        self,
        current_stats: Dict[str, Dict[str, float]],
        baseline_stats: Dict[str, Dict[str, float]],
    ) -> List[FeatureDriftCheck]:
        """Compare current feature distributions against baseline using PSI.

        Args:
            current_stats: feature_name → {"mean": ..., "std": ..., "histogram": [...]}
            baseline_stats: feature_name → {"mean": ..., "std": ..., "histogram": [...]}
        """
        checks = []

        for feat_name in sorted(set(current_stats) & set(baseline_stats)):
            curr = current_stats[feat_name]
            base = baseline_stats[feat_name]

            # Compute PSI from histograms if available
            if "histogram" in curr and "histogram" in base:
                psi = self._compute_psi(
                    np.array(base["histogram"]),
                    np.array(curr["histogram"]),
                )
            else:
                # Fallback: approximate PSI from mean/std shift
                psi = self._approximate_psi(
                    base.get("mean", 0), base.get("std", 1),
                    curr.get("mean", 0), curr.get("std", 1),
                )

            if psi >= self.PSI_ALERT:
                status = "alert"
            elif psi >= self.PSI_WARNING:
                status = "warning"
            else:
                status = "ok"

            checks.append(FeatureDriftCheck(
                feature=feat_name,
                psi=round(psi, 4),
                status=status,
                baseline_mean=base.get("mean", 0),
                current_mean=curr.get("mean", 0),
                baseline_std=base.get("std", 0),
                current_std=curr.get("std", 0),
            ))

        return checks

    def generate_report(
        self,
        data_dir: str = "data/raw",
        current_stats: Optional[Dict[str, Dict[str, float]]] = None,
        baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
        predictions: Optional[np.ndarray] = None,
        outcomes: Optional[np.ndarray] = None,
    ) -> MonitoringReport:
        """Generate a complete monitoring report."""
        report = MonitoringReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Data freshness
        report.data_freshness = self.check_data_freshness(data_dir)
        for check in report.data_freshness:
            if check.status == "stale":
                report.alerts.append(
                    f"STALE: {check.source} is {check.staleness_hours:.0f}h old "
                    f"(SLA: {check.sla_hours:.0f}h)"
                )
            elif check.status == "missing":
                report.alerts.append(f"MISSING: No files matching {check.source}")

        # Feature drift
        if current_stats and baseline_stats:
            report.feature_drift = self.check_feature_drift(current_stats, baseline_stats)
            for check in report.feature_drift:
                if check.status == "alert":
                    report.alerts.append(
                        f"DRIFT: Feature '{check.feature}' PSI={check.psi:.3f} "
                        f"(threshold: {self.PSI_ALERT})"
                    )

        # Prediction quality
        if predictions is not None and outcomes is not None:
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
            brier = float(np.mean((predictions - outcomes) ** 2))
            logloss = float(-np.mean(
                outcomes * np.log(predictions) + (1 - outcomes) * np.log(1 - predictions)
            ))
            accuracy = float(np.mean((predictions > 0.5) == outcomes))
            report.prediction_quality = {
                "brier": round(brier, 6),
                "log_loss": round(logloss, 6),
                "accuracy": round(accuracy, 4),
                "n_predictions": len(predictions),
            }

        return report

    @staticmethod
    def compute_feature_stats(
        features: np.ndarray,
        feature_names: List[str],
        n_bins: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-feature statistics for drift detection baseline.

        Returns dict of feature_name → {"mean", "std", "histogram"}.
        """
        stats = {}
        for i, name in enumerate(feature_names):
            if i >= features.shape[1]:
                break
            col = features[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                stats[name] = {"mean": 0.0, "std": 0.0, "histogram": [1.0 / n_bins] * n_bins}
                continue

            hist, _ = np.histogram(valid, bins=n_bins, density=True)
            hist = hist / (hist.sum() + 1e-10)  # Normalize to proportions

            stats[name] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "n_valid": int(len(valid)),
                "n_nan": int(np.isnan(col).sum()),
                "histogram": [float(h) for h in hist],
            }
        return stats

    @staticmethod
    def save_baseline(stats: Dict[str, Dict[str, float]], output_path: str) -> None:
        """Save feature statistics as baseline for future drift detection."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved feature baseline to %s (%d features)", output_path, len(stats))

    @staticmethod
    def load_baseline(baseline_path: str) -> Dict[str, Dict[str, float]]:
        """Load a previously saved feature baseline."""
        with open(baseline_path) as f:
            return json.load(f)

    @staticmethod
    def _compute_psi(baseline: np.ndarray, current: np.ndarray) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-6
        baseline = np.clip(baseline, eps, None)
        current = np.clip(current, eps, None)
        # Normalize
        baseline = baseline / baseline.sum()
        current = current / current.sum()
        return float(np.sum((current - baseline) * np.log(current / baseline)))

    @staticmethod
    def _approximate_psi(
        base_mean: float, base_std: float,
        curr_mean: float, curr_std: float,
    ) -> float:
        """Approximate PSI from mean/std shift (when histograms unavailable)."""
        if base_std < 1e-10:
            return 0.0
        # Use standardized mean shift as a proxy
        z_shift = abs(curr_mean - base_mean) / base_std
        std_ratio = curr_std / (base_std + 1e-10)
        # Rough PSI approximation
        return float(0.5 * z_shift**2 + 0.5 * (std_ratio - 1)**2)
