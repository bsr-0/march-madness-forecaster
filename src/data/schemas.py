"""Schema contracts for inter-stage pipeline data.

Lightweight dataclass-based validation for data flowing between pipeline
stages.  Catches silent schema violations (wrong shapes, NaN explosions,
out-of-range predictions) that would otherwise produce garbage output.

Implements Directive V7 S19-1 without adding a Pydantic dependency.

Usage:
    from src.data.schemas import validate_feature_matrix, validate_predictions
    validate_feature_matrix(X, feature_names)
    validate_predictions(probs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a schema validation check."""

    passed: bool
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def validate_feature_matrix(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_nan_fraction: float = 0.5,
    min_samples: int = 10,
    min_features: int = 1,
) -> ValidationResult:
    """Validate a feature matrix at pipeline stage boundaries.

    Checks:
    - Shape is 2-dimensional
    - Minimum samples and features present
    - NaN fraction below threshold per feature
    - No constant columns (zero variance)
    - No infinite values

    Args:
        X: Feature matrix (n_samples, n_features).
        feature_names: Optional names for reporting.
        max_nan_fraction: Maximum allowed NaN fraction per feature.
        min_samples: Minimum required samples.
        min_features: Minimum required features.

    Returns:
        ValidationResult with pass/fail and diagnostic messages.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if X.ndim != 2:
        errors.append(f"Feature matrix must be 2D, got {X.ndim}D with shape {X.shape}")
        return ValidationResult(passed=False, warnings=warnings, errors=errors)

    n_samples, n_features = X.shape

    if n_samples < min_samples:
        errors.append(f"Too few samples: {n_samples} < {min_samples}")

    if n_features < min_features:
        errors.append(f"Too few features: {n_features} < {min_features}")

    # NaN check per feature
    nan_counts = np.isnan(X).sum(axis=0)
    nan_fractions = nan_counts / max(n_samples, 1)
    high_nan_features = np.where(nan_fractions > max_nan_fraction)[0]
    if len(high_nan_features) > 0:
        names = (
            [feature_names[i] for i in high_nan_features]
            if feature_names
            else [f"feature_{i}" for i in high_nan_features]
        )
        warnings.append(
            f"{len(high_nan_features)} features with >{max_nan_fraction*100:.0f}% NaN: {names[:5]}"
        )

    # Infinite value check
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        errors.append(f"Feature matrix contains {inf_count} infinite values")

    # Constant columns warning
    stds = np.nanstd(X, axis=0)
    constant_cols = np.where(stds < 1e-10)[0]
    if len(constant_cols) > 0:
        names = (
            [feature_names[i] for i in constant_cols]
            if feature_names
            else [f"feature_{i}" for i in constant_cols]
        )
        warnings.append(f"{len(constant_cols)} constant features (zero variance): {names[:5]}")

    passed = len(errors) == 0
    if not passed:
        logger.error("Feature matrix validation FAILED: %s", errors)
    elif warnings:
        logger.warning("Feature matrix validation warnings: %s", warnings)

    return ValidationResult(passed=passed, warnings=warnings, errors=errors)


def validate_predictions(
    predictions: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    min_variance: float = 1e-6,
) -> ValidationResult:
    """Validate prediction array at pipeline output boundary.

    Checks:
    - All values in [min_val, max_val]
    - No NaN values
    - Predictions are not all identical (variance check)
    - No infinite values

    Args:
        predictions: Array of predicted probabilities.
        min_val: Minimum allowed prediction value.
        max_val: Maximum allowed prediction value.
        min_variance: Minimum variance (catches all-0.5 degenerate predictions).

    Returns:
        ValidationResult with pass/fail and diagnostic messages.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if predictions.ndim != 1:
        errors.append(f"Predictions must be 1D, got {predictions.ndim}D")
        return ValidationResult(passed=False, warnings=warnings, errors=errors)

    # NaN check
    nan_count = np.isnan(predictions).sum()
    if nan_count > 0:
        errors.append(f"Predictions contain {nan_count} NaN values")

    # Range check
    valid_mask = ~np.isnan(predictions)
    if valid_mask.any():
        out_of_range = ((predictions[valid_mask] < min_val) | (predictions[valid_mask] > max_val)).sum()
        if out_of_range > 0:
            errors.append(
                f"{out_of_range} predictions out of [{min_val}, {max_val}] range. "
                f"Min={predictions[valid_mask].min():.4f}, Max={predictions[valid_mask].max():.4f}"
            )

        # Variance check
        pred_var = float(np.var(predictions[valid_mask]))
        if pred_var < min_variance:
            warnings.append(
                f"Predictions have near-zero variance ({pred_var:.2e}). "
                f"Model may be producing degenerate output."
            )

    # Infinite check
    inf_count = np.isinf(predictions).sum()
    if inf_count > 0:
        errors.append(f"Predictions contain {inf_count} infinite values")

    passed = len(errors) == 0
    if not passed:
        logger.error("Prediction validation FAILED: %s", errors)

    return ValidationResult(passed=passed, warnings=warnings, errors=errors)


def validate_loyo_fold(
    year: int,
    brier_score: float,
    n_games: int,
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> ValidationResult:
    """Validate a single LOYO fold result.

    Args:
        year: Held-out year.
        brier_score: Computed Brier score for this fold.
        n_games: Number of games in this fold.
        predictions: Predicted probabilities.
        actuals: Binary outcomes.

    Returns:
        ValidationResult.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if n_games < 10:
        warnings.append(f"Year {year}: only {n_games} games (suspiciously few)")

    if brier_score < 0 or brier_score > 1:
        errors.append(f"Year {year}: Brier score {brier_score:.4f} out of [0, 1] range")

    if brier_score > 0.35:
        warnings.append(
            f"Year {year}: Brier score {brier_score:.4f} is very high "
            f"(worse than coin-flip baseline of 0.25)"
        )

    if len(predictions) != len(actuals):
        errors.append(
            f"Year {year}: predictions ({len(predictions)}) and actuals ({len(actuals)}) "
            f"have different lengths"
        )

    pred_validation = validate_predictions(predictions)
    if not pred_validation.passed:
        errors.extend([f"Year {year}: {e}" for e in pred_validation.errors])

    passed = len(errors) == 0
    return ValidationResult(passed=passed, warnings=warnings, errors=errors)
