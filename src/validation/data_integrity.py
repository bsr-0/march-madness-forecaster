"""Phase 4 — Data Integrity & Feature Validity.

Unified validation orchestrator that ensures all features are clean, valid,
and pre-tournament only before the dataset enters model training.

Checks performed:
    1. Leaky / future feature removal
    2. Missing value detection and enforcement
    3. Feature drift vs historical baseline (PSI + z-score)
    4. Pre-tournament-only enforcement (PIT tiers)
    5. Feature dimension consistency (TEAM_FEATURE_DIM = 86)

If any hard check fails, the pipeline is stopped via LeakageError,
IntegrityError, or DataRequirementError.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.data.features.feature_engineering import TEAM_FEATURE_DIM
from src.exceptions import (
    DataRequirementError,
    IntegrityError,
    LeakageError,
)
from src.monitoring.pipeline_monitor import PipelineMonitor
from src.pipeline.stages.pit_validation import (
    PITValidator,
    SELECTION_SUNDAY_DATES,
)

logger = logging.getLogger(__name__)

# Features that must never appear in a training dataset — they encode
# tournament outcomes or post-season information.
KNOWN_LEAKY_FEATURES: Set[str] = frozenset({
    "tournament_wins",
    "tournament_round_reached",
    "final_four_flag",
    "champion_flag",
    "tournament_seed_result",
    "postseason_result",
    "postseason_wins",
    "march_madness_result",
    "tourney_score",
    "opponent_tournament_seed",
    "game_result",
    "win_flag",
    "loss_flag",
    "point_spread_result",
})

# Suffix patterns that strongly suggest future-leak features.
LEAKY_SUFFIXES: Tuple[str, ...] = (
    "_postseason",
    "_tournament_result",
    "_final",
    "_future",
    "_outcome",
)

# Maximum allowable fraction of NaN per feature before hard-stop.
MAX_MISSING_FRACTION = 0.20

# PSI threshold for hard-stop (very large shift).
PSI_HARD_STOP = 0.25


@dataclass
class FeatureRemovalRecord:
    """Record of a single removed feature with reason."""
    feature: str
    reason: str


@dataclass
class ValidationReport:
    """Complete output of the data integrity validation pass."""

    passed: bool = True
    n_features_input: int = 0
    n_features_output: int = 0
    removed_features: List[FeatureRemovalRecord] = field(default_factory=list)
    missing_value_stats: Dict[str, float] = field(default_factory=dict)
    drift_alerts: List[Dict[str, Any]] = field(default_factory=list)
    pit_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Data Integrity Report: {'PASSED' if self.passed else 'FAILED'}",
            f"  Features: {self.n_features_input} input → {self.n_features_output} output",
            f"  Removed: {len(self.removed_features)}",
            f"  Drift alerts: {len(self.drift_alerts)}",
            f"  PIT violations: {len(self.pit_violations)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Errors: {len(self.errors)}",
        ]
        if self.removed_features:
            lines.append("  Removed features:")
            for rec in self.removed_features:
                lines.append(f"    - {rec.feature}: {rec.reason}")
        return "\n".join(lines)


class DataIntegrityValidator:
    """Orchestrates all data integrity checks for Phase 4.

    Usage::

        validator = DataIntegrityValidator(year=2026, strict=True)
        clean_features, clean_names, report = validator.validate(
            feature_matrix=X,
            feature_names=names,
            baseline_stats=saved_baseline,
        )
        # clean_features is the validated np.ndarray ready for Phase 5.
    """

    def __init__(
        self,
        year: int = 2026,
        strict: bool = True,
        max_missing_fraction: float = MAX_MISSING_FRACTION,
        psi_hard_stop: float = PSI_HARD_STOP,
        extra_leaky_features: Optional[Set[str]] = None,
        manifest_path: Optional[str] = None,
    ):
        self.year = year
        self.strict = strict
        self.max_missing_fraction = max_missing_fraction
        self.psi_hard_stop = psi_hard_stop

        self.leaky_features = set(KNOWN_LEAKY_FEATURES)
        if extra_leaky_features:
            self.leaky_features |= extra_leaky_features

        self.pit_validator = PITValidator(manifest_path=manifest_path)
        self.monitor = PipelineMonitor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
        feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, List[str], ValidationReport]:
        """Run all integrity checks and return a clean feature matrix.

        Args:
            feature_matrix: (n_teams, n_features) array.
            feature_names: Ordered feature names matching columns.
            baseline_stats: Historical baseline stats for drift detection
                (feature_name → {"mean", "std", "histogram"}).
            feature_metadata: Per-feature metadata for PIT checks
                (feature_name → {"latest_game_date", "snapshot_date", ...}).

        Returns:
            Tuple of (clean_matrix, clean_names, report).

        Raises:
            LeakageError: If leaky features remain after removal (strict).
            IntegrityError: If missing-value or drift thresholds are breached.
            DataRequirementError: If feature dimension is wrong.
        """
        report = ValidationReport(n_features_input=len(feature_names))

        if feature_matrix.ndim != 2:
            raise DataRequirementError(
                f"feature_matrix must be 2-D, got shape {feature_matrix.shape}"
            )
        if feature_matrix.shape[1] != len(feature_names):
            raise DataRequirementError(
                f"feature_matrix has {feature_matrix.shape[1]} columns "
                f"but {len(feature_names)} names provided"
            )

        # --- Step 1: Remove leaky / future features -----------------------
        feature_matrix, feature_names = self._remove_leaky_features(
            feature_matrix, feature_names, report
        )

        # --- Step 2: PIT enforcement (pre-tournament only) ----------------
        feature_matrix, feature_names = self._enforce_pit(
            feature_matrix, feature_names, feature_metadata, report
        )

        # --- Step 3: Missing value check ----------------------------------
        feature_matrix, feature_names = self._check_missingness(
            feature_matrix, feature_names, report
        )

        # --- Step 4: Feature drift vs baseline ----------------------------
        self._check_feature_drift(
            feature_matrix, feature_names, baseline_stats, report
        )

        # --- Step 5: Dimension consistency --------------------------------
        self._check_dimension(feature_names, report)

        report.n_features_output = len(feature_names)

        if report.errors:
            report.passed = False

        # Log and return
        logger.info(report.summary())

        if not report.passed and self.strict:
            first_error = report.errors[0] if report.errors else "unknown"
            raise IntegrityError(
                f"Data integrity validation failed: {first_error}"
            )

        return feature_matrix, feature_names, report

    # ------------------------------------------------------------------
    # Step 1 — Leaky feature removal
    # ------------------------------------------------------------------

    def _remove_leaky_features(
        self,
        matrix: np.ndarray,
        names: List[str],
        report: ValidationReport,
    ) -> Tuple[np.ndarray, List[str]]:
        """Remove features that encode tournament outcomes or future data."""
        keep_mask = []
        for name in names:
            is_leaky = False
            reason = ""

            if name in self.leaky_features:
                is_leaky = True
                reason = "known leaky feature (encodes tournament outcome)"
            elif any(name.endswith(suffix) for suffix in LEAKY_SUFFIXES):
                is_leaky = True
                reason = f"leaky suffix pattern ({name.split('_')[-1]})"

            if is_leaky:
                report.removed_features.append(
                    FeatureRemovalRecord(feature=name, reason=reason)
                )
                logger.warning("REMOVED leaky feature: %s — %s", name, reason)
                keep_mask.append(False)
            else:
                keep_mask.append(True)

        keep_mask = np.array(keep_mask)
        clean_names = [n for n, k in zip(names, keep_mask) if k]
        clean_matrix = matrix[:, keep_mask]
        return clean_matrix, clean_names

    # ------------------------------------------------------------------
    # Step 2 — PIT enforcement
    # ------------------------------------------------------------------

    def _enforce_pit(
        self,
        matrix: np.ndarray,
        names: List[str],
        feature_metadata: Optional[Dict[str, Dict[str, Any]]],
        report: ValidationReport,
    ) -> Tuple[np.ndarray, List[str]]:
        """Enforce pre-tournament-only constraint via PIT tier checks."""
        selection_sunday = SELECTION_SUNDAY_DATES.get(self.year)
        if selection_sunday is None:
            report.warnings.append(
                f"No Selection Sunday date for year {self.year}; "
                "PIT temporal checks skipped."
            )
            return matrix, names

        if feature_metadata is None:
            report.warnings.append(
                "No feature metadata provided; PIT temporal checks limited."
            )
            # Still run structural check via PITValidator
            pit_result = self.pit_validator.validate_fold(
                year=self.year,
                feature_names=names,
                feature_metadata=None,
                strict=False,
            )
            report.warnings.extend(pit_result.warnings)
            return matrix, names

        # Full PIT validation with metadata
        pit_result = self.pit_validator.validate_fold(
            year=self.year,
            feature_names=names,
            feature_metadata=feature_metadata,
            strict=False,  # We handle strictness ourselves
        )
        report.warnings.extend(pit_result.warnings)

        if pit_result.violations:
            report.pit_violations.extend(pit_result.violations)
            # Remove violating features
            violating_names = set()
            for violation_msg in pit_result.violations:
                # Extract feature name from violation message
                for name in names:
                    if f"'{name}'" in violation_msg:
                        violating_names.add(name)

            if violating_names:
                keep_mask = np.array([n not in violating_names for n in names])
                for vn in violating_names:
                    report.removed_features.append(
                        FeatureRemovalRecord(
                            feature=vn,
                            reason="PIT violation (post-tournament data)",
                        )
                    )
                    logger.warning(
                        "REMOVED PIT-violating feature: %s", vn
                    )
                names = [n for n, k in zip(names, keep_mask) if k]
                matrix = matrix[:, keep_mask]

            if self.strict and not violating_names:
                # Violations detected but couldn't parse feature names —
                # treat as hard error.
                report.errors.append(
                    f"PIT violations detected: {pit_result.violations}"
                )

        return matrix, names

    # ------------------------------------------------------------------
    # Step 3 — Missing value check
    # ------------------------------------------------------------------

    def _check_missingness(
        self,
        matrix: np.ndarray,
        names: List[str],
        report: ValidationReport,
    ) -> Tuple[np.ndarray, List[str]]:
        """Check for excessive missing values; remove features that exceed threshold."""
        n_rows = matrix.shape[0]
        if n_rows == 0:
            report.warnings.append("Empty feature matrix (0 rows).")
            return matrix, names

        keep_mask = []
        for idx, name in enumerate(names):
            col = matrix[:, idx]
            n_missing = int(np.isnan(col).sum()) if np.issubdtype(col.dtype, np.floating) else 0
            frac = n_missing / n_rows

            report.missing_value_stats[name] = round(frac, 4)

            if frac > self.max_missing_fraction:
                report.removed_features.append(
                    FeatureRemovalRecord(
                        feature=name,
                        reason=f"missing fraction {frac:.1%} > {self.max_missing_fraction:.0%} threshold",
                    )
                )
                logger.warning(
                    "REMOVED feature %s: %.1f%% missing (threshold %.0f%%)",
                    name, frac * 100, self.max_missing_fraction * 100,
                )
                keep_mask.append(False)
            else:
                keep_mask.append(True)

        keep_mask = np.array(keep_mask)
        clean_names = [n for n, k in zip(names, keep_mask) if k]
        clean_matrix = matrix[:, keep_mask]
        return clean_matrix, clean_names

    # ------------------------------------------------------------------
    # Step 4 — Feature drift
    # ------------------------------------------------------------------

    def _check_feature_drift(
        self,
        matrix: np.ndarray,
        names: List[str],
        baseline_stats: Optional[Dict[str, Dict[str, float]]],
        report: ValidationReport,
    ) -> None:
        """Detect feature drift vs historical baseline using PSI."""
        if baseline_stats is None:
            report.warnings.append(
                "No baseline stats provided; drift detection skipped."
            )
            return

        if matrix.shape[0] == 0:
            return

        current_stats = PipelineMonitor.compute_feature_stats(
            matrix, names, n_bins=10
        )

        drift_checks = self.monitor.check_feature_drift(
            current_stats=current_stats,
            baseline_stats=baseline_stats,
        )

        for check in drift_checks:
            if check.status == "alert":
                alert_info = {
                    "feature": check.feature,
                    "psi": check.psi,
                    "baseline_mean": check.baseline_mean,
                    "current_mean": check.current_mean,
                    "baseline_std": check.baseline_std,
                    "current_std": check.current_std,
                }
                report.drift_alerts.append(alert_info)
                logger.warning(
                    "DRIFT ALERT: %s PSI=%.4f (baseline μ=%.3f → current μ=%.3f)",
                    check.feature, check.psi,
                    check.baseline_mean, check.current_mean,
                )

                if check.psi >= self.psi_hard_stop:
                    report.errors.append(
                        f"Feature '{check.feature}' PSI={check.psi:.4f} "
                        f"exceeds hard-stop threshold {self.psi_hard_stop}"
                    )
            elif check.status == "warning":
                report.warnings.append(
                    f"Drift warning: {check.feature} PSI={check.psi:.4f}"
                )

    # ------------------------------------------------------------------
    # Step 5 — Dimension consistency
    # ------------------------------------------------------------------

    def _check_dimension(
        self,
        names: List[str],
        report: ValidationReport,
    ) -> None:
        """Verify feature count matches TEAM_FEATURE_DIM after cleaning.

        A mismatch is a warning (features may have been legitimately removed).
        If features *exceed* the expected dim, that's an error — something
        unexpected was injected.
        """
        n = len(names)
        if n > TEAM_FEATURE_DIM:
            report.errors.append(
                f"Feature count {n} exceeds TEAM_FEATURE_DIM={TEAM_FEATURE_DIM}. "
                "Unexpected features present."
            )
        elif n < TEAM_FEATURE_DIM:
            report.warnings.append(
                f"Feature count {n} < TEAM_FEATURE_DIM={TEAM_FEATURE_DIM} "
                f"(removed {TEAM_FEATURE_DIM - n} features during validation)."
            )
