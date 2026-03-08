"""Robustness testing suite for model evaluation.

Implements Directive V7 requirements:
  S11-1: Feature dropout robustness test — measure degradation when each
         top feature is removed.
  S11-3: Distribution shift detection — compare feature distributions
         between training and prediction year using PSI.
  M5:    Feature importance stability across LOYO folds — Kendall tau
         correlation of importance rankings.

Usage:
    from src.ml.evaluation.robustness import (
        FeatureDropoutTest,
        DistributionShiftDetector,
        FeatureStabilityReport,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Dropout Robustness Test (S11-1)
# ---------------------------------------------------------------------------

@dataclass
class FeatureDropoutResult:
    """Result of dropping a single feature from the model."""

    feature_index: int
    feature_name: str
    baseline_brier: float
    dropout_brier: float
    degradation: float  # dropout_brier - baseline_brier (positive = worse)
    fragile: bool  # degradation > threshold


@dataclass
class FeatureDropoutReport:
    """Full report from feature dropout robustness test."""

    baseline_brier: float
    results: List[FeatureDropoutResult] = field(default_factory=list)
    fragile_features: List[str] = field(default_factory=list)
    max_degradation: float = 0.0
    mean_degradation: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_brier": round(self.baseline_brier, 6),
            "max_degradation": round(self.max_degradation, 6),
            "mean_degradation": round(self.mean_degradation, 6),
            "fragile_features": self.fragile_features,
            "n_features_tested": len(self.results),
            "per_feature": [
                {
                    "name": r.feature_name,
                    "degradation": round(r.degradation, 6),
                    "fragile": r.fragile,
                }
                for r in sorted(self.results, key=lambda r: -r.degradation)
            ],
        }


class FeatureDropoutTest:
    """Test model robustness by zeroing out features one at a time.

    For each of the top-K features (by importance), sets that feature
    to zero across all samples, re-evaluates the model, and records
    Brier degradation.  Features causing >threshold degradation are
    flagged as fragile dependencies.
    """

    def __init__(
        self,
        fragility_threshold: float = 0.01,
        top_k: int = 10,
    ):
        self.fragility_threshold = fragility_threshold
        self.top_k = top_k

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None,
    ) -> FeatureDropoutReport:
        """Run feature dropout test.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary outcomes (n_samples,).
            predict_fn: Function that takes X and returns probabilities.
            feature_names: Optional names for each feature column.
            feature_importances: Optional importance scores to select top-K.
                If None, tests all features up to top_k.

        Returns:
            FeatureDropoutReport with per-feature degradation analysis.
        """
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Determine which features to test
        if feature_importances is not None:
            top_indices = np.argsort(feature_importances)[::-1][: self.top_k]
        else:
            top_indices = list(range(min(self.top_k, n_features)))

        # Baseline Brier
        baseline_preds = predict_fn(X)
        baseline_brier = float(np.mean((baseline_preds - y) ** 2))

        results: List[FeatureDropoutResult] = []
        for idx in top_indices:
            X_dropped = X.copy()
            X_dropped[:, idx] = 0.0

            dropped_preds = predict_fn(X_dropped)
            dropped_brier = float(np.mean((dropped_preds - y) ** 2))
            degradation = dropped_brier - baseline_brier
            fragile = degradation > self.fragility_threshold

            results.append(FeatureDropoutResult(
                feature_index=int(idx),
                feature_name=feature_names[idx],
                baseline_brier=baseline_brier,
                dropout_brier=dropped_brier,
                degradation=degradation,
                fragile=fragile,
            ))

        fragile_features = [r.feature_name for r in results if r.fragile]
        degradations = [r.degradation for r in results]

        report = FeatureDropoutReport(
            baseline_brier=baseline_brier,
            results=results,
            fragile_features=fragile_features,
            max_degradation=max(degradations) if degradations else 0.0,
            mean_degradation=float(np.mean(degradations)) if degradations else 0.0,
        )

        logger.info(
            "Feature dropout test: %d features tested, %d fragile (threshold=%.3f), "
            "max degradation=%.6f",
            len(results),
            len(fragile_features),
            self.fragility_threshold,
            report.max_degradation,
        )
        return report


# ---------------------------------------------------------------------------
# Distribution Shift Detection (S11-3)
# ---------------------------------------------------------------------------

@dataclass
class ShiftResult:
    """PSI result for a single feature."""

    feature_name: str
    psi: float
    severity: str  # "none", "moderate", "significant"


@dataclass
class DistributionShiftReport:
    """Full distribution shift report."""

    results: List[ShiftResult] = field(default_factory=list)
    aggregate_psi: float = 0.0
    significant_features: List[str] = field(default_factory=list)
    moderate_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_psi": round(self.aggregate_psi, 4),
            "significant_features": self.significant_features,
            "moderate_features": self.moderate_features,
            "n_features": len(self.results),
            "per_feature": [
                {
                    "name": r.feature_name,
                    "psi": round(r.psi, 4),
                    "severity": r.severity,
                }
                for r in sorted(self.results, key=lambda r: -r.psi)
                if r.psi > 0.05  # Only report non-trivial shifts
            ],
        }


class DistributionShiftDetector:
    """Detect covariate shift between training and prediction feature distributions.

    Uses Population Stability Index (PSI) per feature:
      PSI < 0.1:  No significant shift
      PSI 0.1-0.2: Moderate shift (investigate)
      PSI > 0.2:  Significant shift (action required)
    """

    def __init__(
        self,
        moderate_threshold: float = 0.1,
        significant_threshold: float = 0.2,
        n_bins: int = 10,
    ):
        self.moderate_threshold = moderate_threshold
        self.significant_threshold = significant_threshold
        self.n_bins = n_bins

    def detect(
        self,
        X_train: np.ndarray,
        X_predict: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DistributionShiftReport:
        """Compare feature distributions between training and prediction data.

        Args:
            X_train: Training feature matrix.
            X_predict: Prediction-year feature matrix.
            feature_names: Optional names for features.

        Returns:
            DistributionShiftReport with per-feature PSI analysis.
        """
        n_features = X_train.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results: List[ShiftResult] = []
        for i in range(n_features):
            train_col = X_train[:, i]
            predict_col = X_predict[:, i]

            # Skip constant columns
            if np.std(train_col) < 1e-10:
                results.append(ShiftResult(
                    feature_name=feature_names[i],
                    psi=0.0,
                    severity="none",
                ))
                continue

            psi = self._compute_psi(train_col, predict_col)

            if psi >= self.significant_threshold:
                severity = "significant"
            elif psi >= self.moderate_threshold:
                severity = "moderate"
            else:
                severity = "none"

            results.append(ShiftResult(
                feature_name=feature_names[i],
                psi=psi,
                severity=severity,
            ))

        significant = [r.feature_name for r in results if r.severity == "significant"]
        moderate = [r.feature_name for r in results if r.severity == "moderate"]
        aggregate_psi = float(np.mean([r.psi for r in results])) if results else 0.0

        report = DistributionShiftReport(
            results=results,
            aggregate_psi=aggregate_psi,
            significant_features=significant,
            moderate_features=moderate,
        )

        if significant:
            logger.warning(
                "Distribution shift detected: %d features with significant shift (PSI>%.1f): %s",
                len(significant),
                self.significant_threshold,
                significant,
            )

        return report

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-6

        # Use expected distribution's quantiles to define bins
        breakpoints = np.percentile(
            expected,
            np.linspace(0, 100, self.n_bins + 1),
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        # Remove duplicate breakpoints
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        expected_pct = expected_counts / len(expected) + eps
        actual_pct = actual_counts / len(actual) + eps

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return max(psi, 0.0)


# ---------------------------------------------------------------------------
# Feature Importance Stability (M5)
# ---------------------------------------------------------------------------

@dataclass
class FeatureStabilityReport:
    """Feature importance stability across LOYO folds."""

    mean_kendall_tau: float = 0.0
    per_pair_tau: List[Dict[str, Any]] = field(default_factory=list)
    rank_variance: Dict[str, float] = field(default_factory=dict)  # feature → rank variance
    unstable_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_kendall_tau": round(self.mean_kendall_tau, 4),
            "n_pairs": len(self.per_pair_tau),
            "unstable_features": self.unstable_features,
            "rank_variance": {
                k: round(v, 4) for k, v in sorted(
                    self.rank_variance.items(), key=lambda x: -x[1]
                )[:10]  # Top 10 most variable
            },
        }

    @classmethod
    def from_importance_rankings(
        cls,
        fold_importances: Dict[int, Dict[str, float]],
        instability_threshold: float = 5.0,
    ) -> FeatureStabilityReport:
        """Compute stability from per-fold feature importance rankings.

        Args:
            fold_importances: {fold_year: {feature_name: importance_score}}
            instability_threshold: Features with rank variance above this
                are flagged as unstable.

        Returns:
            FeatureStabilityReport with Kendall tau and rank variance.
        """
        if len(fold_importances) < 2:
            return cls()

        years = sorted(fold_importances.keys())
        all_features = set()
        for imp in fold_importances.values():
            all_features.update(imp.keys())
        all_features_sorted = sorted(all_features)

        # Compute per-fold rankings
        fold_ranks: Dict[int, Dict[str, int]] = {}
        for year, importances in fold_importances.items():
            sorted_features = sorted(
                all_features_sorted,
                key=lambda f: importances.get(f, 0),
                reverse=True,
            )
            fold_ranks[year] = {f: rank for rank, f in enumerate(sorted_features)}

        # Pairwise Kendall tau
        taus: List[Dict[str, Any]] = []
        tau_values: List[float] = []
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                y1, y2 = years[i], years[j]
                ranks1 = [fold_ranks[y1].get(f, len(all_features)) for f in all_features_sorted]
                ranks2 = [fold_ranks[y2].get(f, len(all_features)) for f in all_features_sorted]
                tau = _kendall_tau(ranks1, ranks2)
                tau_values.append(tau)
                taus.append({
                    "year_a": y1,
                    "year_b": y2,
                    "kendall_tau": round(tau, 4),
                })

        # Per-feature rank variance
        rank_variance: Dict[str, float] = {}
        for feature in all_features_sorted:
            ranks = [fold_ranks[y].get(feature, len(all_features)) for y in years]
            rank_variance[feature] = float(np.var(ranks))

        unstable = [
            f for f, v in rank_variance.items()
            if v > instability_threshold
        ]

        return cls(
            mean_kendall_tau=float(np.mean(tau_values)) if tau_values else 0.0,
            per_pair_tau=taus,
            rank_variance=rank_variance,
            unstable_features=unstable,
        )


def _kendall_tau(ranks_a: List[int], ranks_b: List[int]) -> float:
    """Compute Kendall tau-b rank correlation coefficient."""
    n = len(ranks_a)
    if n < 2:
        return 0.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_a = ranks_a[i] - ranks_a[j]
            sign_b = ranks_b[i] - ranks_b[j]
            product = sign_a * sign_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs
