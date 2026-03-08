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


# ---------------------------------------------------------------------------
# Ensemble weight contracts (S19-1)
# ---------------------------------------------------------------------------


def validate_ensemble_weights(
    weights: Dict[str, float],
    expected_components: Optional[List[str]] = None,
    tolerance: float = 0.05,
) -> ValidationResult:
    """Validate ensemble component weights at the fusion boundary.

    Checks:
    - All weights are non-negative
    - Weights sum to approximately 1.0
    - No unexpected components
    - No missing expected components

    Args:
        weights: Component name → weight mapping.
        expected_components: If provided, validates exact component set.
        tolerance: Maximum deviation of weight sum from 1.0.

    Returns:
        ValidationResult with pass/fail and diagnostic messages.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if not weights:
        errors.append("Ensemble weights are empty")
        return ValidationResult(passed=False, warnings=warnings, errors=errors)

    # Non-negative check
    negative = {k: v for k, v in weights.items() if v < 0}
    if negative:
        errors.append(f"Negative weights: {negative}")

    # Sum-to-one check
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > tolerance:
        errors.append(
            f"Weights sum to {weight_sum:.4f}, expected ~1.0 (tolerance={tolerance})"
        )

    # Component set check
    if expected_components is not None:
        expected_set = set(expected_components)
        actual_set = set(weights.keys())
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if missing:
            warnings.append(f"Missing expected components: {missing}")
        if extra:
            warnings.append(f"Unexpected components: {extra}")

    passed = len(errors) == 0
    if not passed:
        logger.error("Ensemble weight validation FAILED: %s", errors)
    return ValidationResult(passed=passed, warnings=warnings, errors=errors)


def validate_calibration_data(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    min_samples: int = 30,
) -> ValidationResult:
    """Validate data entering the calibration pipeline.

    Checks:
    - Matching array lengths
    - Outcomes are binary (0 or 1)
    - Predictions in [0, 1]
    - Minimum sample size for reliable calibration
    - Both classes present in outcomes

    Args:
        predictions: Raw model probabilities.
        outcomes: Binary ground truth labels.
        min_samples: Minimum samples for reliable calibration.

    Returns:
        ValidationResult.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if len(predictions) != len(outcomes):
        errors.append(
            f"Length mismatch: predictions={len(predictions)}, outcomes={len(outcomes)}"
        )
        return ValidationResult(passed=False, warnings=warnings, errors=errors)

    if len(predictions) < min_samples:
        warnings.append(
            f"Only {len(predictions)} samples for calibration "
            f"(recommended minimum: {min_samples})"
        )

    # Outcomes binary check
    unique_outcomes = set(np.unique(outcomes))
    if not unique_outcomes.issubset({0.0, 1.0}):
        errors.append(f"Outcomes must be binary (0 or 1), found: {unique_outcomes}")

    if len(unique_outcomes) < 2:
        errors.append(
            f"Only one class in outcomes ({unique_outcomes}). "
            "Calibration requires both classes."
        )

    # Predictions range check
    pred_result = validate_predictions(predictions)
    if not pred_result.passed:
        errors.extend(pred_result.errors)
    warnings.extend(pred_result.warnings)

    passed = len(errors) == 0
    return ValidationResult(passed=passed, warnings=warnings, errors=errors)


def validate_matchup_vector(
    vector: np.ndarray,
    expected_dim: Optional[int] = None,
) -> ValidationResult:
    """Validate a single matchup feature vector.

    Checks:
    - Vector is 1-dimensional
    - Expected dimensionality (if specified)
    - No infinite values
    - NaN fraction below threshold

    Args:
        vector: Matchup feature vector (team1 - team2 differentials).
        expected_dim: Expected feature count (e.g., 77 for full matchup).

    Returns:
        ValidationResult.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if vector.ndim != 1:
        errors.append(f"Matchup vector must be 1D, got {vector.ndim}D")
        return ValidationResult(passed=False, warnings=warnings, errors=errors)

    if expected_dim is not None and len(vector) != expected_dim:
        errors.append(
            f"Matchup vector dimension {len(vector)} != expected {expected_dim}"
        )

    inf_count = int(np.isinf(vector).sum())
    if inf_count > 0:
        errors.append(f"Matchup vector contains {inf_count} infinite values")

    nan_count = int(np.isnan(vector).sum())
    nan_frac = nan_count / max(len(vector), 1)
    if nan_frac > 0.5:
        errors.append(
            f"Matchup vector has {nan_frac:.0%} NaN ({nan_count}/{len(vector)})"
        )
    elif nan_frac > 0.2:
        warnings.append(
            f"Matchup vector has {nan_frac:.0%} NaN ({nan_count}/{len(vector)})"
        )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, warnings=warnings, errors=errors)
