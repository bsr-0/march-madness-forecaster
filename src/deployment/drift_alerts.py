"""Drift alerting: monitor feature distribution shifts and generate alerts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DriftAlert:
    """Alert for detected feature drift."""

    feature: str
    psi: float
    severity: str  # "info", "warning", "critical"
    suggested_action: str
    timestamp: str


class DriftAlertManager:
    """Monitor feature distributions and generate drift alerts."""

    PSI_INFO = 0.05
    PSI_WARNING = 0.1
    PSI_CRITICAL = 0.25

    def __init__(self) -> None:
        self._alert_history: List[DriftAlert] = []

    def check_and_alert(
        self,
        current_stats: Dict[str, Dict[str, float]],
        baseline_stats: Dict[str, Dict[str, float]],
    ) -> List[DriftAlert]:
        """Check feature drift and generate alerts.

        Each stats dict maps feature_name -> {"mean": ..., "std": ...}.
        PSI is approximated from the shift in mean/std.

        Args:
            current_stats: Current feature statistics.
            baseline_stats: Baseline (reference) feature statistics.

        Returns:
            List of DriftAlert objects for features exceeding PSI_INFO.
        """
        alerts: List[DriftAlert] = []
        now = datetime.now(timezone.utc).isoformat()

        common_features = sorted(set(current_stats) & set(baseline_stats))

        for feature in common_features:
            cur = current_stats[feature]
            base = baseline_stats[feature]

            psi = self._approximate_psi(
                cur.get("mean", 0.0),
                cur.get("std", 1.0),
                base.get("mean", 0.0),
                base.get("std", 1.0),
            )

            if psi < self.PSI_INFO:
                continue

            severity = self._classify_severity(psi)
            suggested_action = self._suggest_action(severity, feature)

            alert = DriftAlert(
                feature=feature,
                psi=round(psi, 4),
                severity=severity,
                suggested_action=suggested_action,
                timestamp=now,
            )
            alerts.append(alert)

        self._alert_history.extend(alerts)
        return alerts

    def generate_report(self, alerts: List[DriftAlert]) -> Dict[str, Any]:
        """Summarize alerts into a report.

        Returns:
            Dict with total_alerts, by_severity counts, features_affected, and alerts list.
        """
        by_severity: Dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
        features_affected: List[str] = []

        for alert in alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            features_affected.append(alert.feature)

        return {
            "total_alerts": len(alerts),
            "by_severity": by_severity,
            "features_affected": sorted(set(features_affected)),
            "alerts": alerts,
        }

    def _approximate_psi(
        self,
        mean_current: float,
        std_current: float,
        mean_baseline: float,
        std_baseline: float,
    ) -> float:
        """Approximate PSI from mean/std shift between distributions.

        Uses a simplified PSI approximation based on the KL-divergence between
        two normal distributions:
            KL(P||Q) = log(std_q/std_p) + (std_p^2 + (mean_p - mean_q)^2)/(2*std_q^2) - 0.5
            PSI ~ KL(P||Q) + KL(Q||P)
        """
        # Guard against zero or negative std
        std_current = max(std_current, 1e-10)
        std_baseline = max(std_baseline, 1e-10)

        # KL(current || baseline)
        kl_forward = (
            np.log(std_baseline / std_current)
            + (std_current**2 + (mean_current - mean_baseline) ** 2)
            / (2.0 * std_baseline**2)
            - 0.5
        )

        # KL(baseline || current)
        kl_reverse = (
            np.log(std_current / std_baseline)
            + (std_baseline**2 + (mean_baseline - mean_current) ** 2)
            / (2.0 * std_current**2)
            - 0.5
        )

        psi = float(kl_forward + kl_reverse)
        return max(psi, 0.0)

    def _classify_severity(self, psi: float) -> str:
        """Classify drift severity based on PSI thresholds."""
        if psi >= self.PSI_CRITICAL:
            return "critical"
        elif psi >= self.PSI_WARNING:
            return "warning"
        else:
            return "info"

    def _suggest_action(self, severity: str, feature: str) -> str:
        """Generate a suggested action based on severity."""
        if severity == "critical":
            return f"Retrain model immediately; feature '{feature}' has severe distribution shift."
        elif severity == "warning":
            return f"Investigate feature '{feature}'; consider retraining if drift persists."
        else:
            return f"Monitor feature '{feature}'; drift is minor."
