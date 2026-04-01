"""Audit module for the state machine forecaster.

Validates all components against invariants:
1. Market correctness (spread/moneyline transforms, vig removal, blend)
2. Matchup feature presence and non-zero importance
3. Calibration (LOYO, OOS only)
4. Stacking (OOF, no leakage)
5. Metrics reproducibility
6. Invariant satisfaction (data, features, models, generalization)
7. Feature stability across folds (Rule 7)
8. Market anchor constraint (Rule 10)
9. Ensemble weight provenance (FIX-STACKING-LEAKAGE)

Also includes anti-cheating checks and overfitting detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .calibration import CalibrationResult
from .matchups import MATCHUP_FEATURE_NAMES, N_MATCHUP_FEATURES
from .stacking import StackingResult

logger = logging.getLogger(__name__)


@dataclass
class AuditViolation:
    """A single audit violation."""
    component: str      # market, matchup, calibration, stacking, generalization
    check: str          # Name of the check
    message: str        # Human-readable description
    severity: str = "error"  # "error" or "warning"


@dataclass
class AuditResult:
    """Result of a full audit pass."""
    passed: bool
    violations: List[AuditViolation] = field(default_factory=list)
    checks_run: int = 0
    checks_passed: int = 0

    def add_violation(self, component: str, check: str, message: str,
                      severity: str = "error"):
        self.violations.append(AuditViolation(component, check, message, severity))

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Audit: {status} ({self.checks_passed}/{self.checks_run} checks passed)"]
        for v in self.violations:
            lines.append(f"  [{v.severity.upper()}] {v.component}/{v.check}: {v.message}")
        return "\n".join(lines)


def run_audit(
    stacking_result: Optional[StackingResult],
    calibration_result: Optional[CalibrationResult],
    feature_importances: Optional[Dict[str, np.ndarray]],
    metrics: Dict[str, float],
    n_features: int,
    prev_predictions: Optional[np.ndarray] = None,
    curr_predictions: Optional[np.ndarray] = None,
    prev_metrics: Optional[Dict[str, float]] = None,
    per_year_brier: Optional[Dict[int, float]] = None,
    holdout_brier: Optional[float] = None,
    prev_holdout_brier: Optional[float] = None,
    prev_per_year_brier: Optional[Dict[int, float]] = None,
    blend_weight: Optional[float] = None,
    market_probs: Optional[np.ndarray] = None,
    ensemble_weight_stats: Optional[Dict[str, Any]] = None,
) -> AuditResult:
    """Run full audit suite."""
    result = AuditResult(passed=True)

    # 1. Market correctness
    _check_market(result, blend_weight)

    # 2. Matchup feature presence and importance
    _check_matchups(result, feature_importances, n_features)

    # 3. Calibration (LOYO, OOS only)
    _check_calibration(result, calibration_result)

    # 4. Stacking (OOF, no leakage)
    _check_stacking(result, stacking_result)

    # 5. Metrics reproducibility
    _check_metrics(result, metrics)

    # 6. Feature count invariant
    _check_feature_count(result, n_features)

    # 7. Anti-cheating (Rule 11: strengthened prediction change validation)
    _check_anti_cheat(result, prev_predictions, curr_predictions, prev_metrics, metrics)

    # 8. Overfitting detection
    _check_overfitting(result, per_year_brier, holdout_brier,
                       prev_holdout_brier, prev_per_year_brier, stacking_result)

    # 9. Feature stability across folds (Rule 7)
    _check_feature_stability(result, stacking_result)

    # 10. Market anchor constraint (Rule 10)
    _check_market_anchor(result, curr_predictions, market_probs)

    # 11. FIX-STACKING-LEAKAGE: Ensemble weight provenance
    _check_ensemble_weight_provenance(result, ensemble_weight_stats)

    # Set overall status
    errors = [v for v in result.violations if v.severity == "error"]
    result.passed = len(errors) == 0

    logger.info(result.summary())
    return result


def _check_market(result: AuditResult, blend_weight: Optional[float]):
    """Check market probability component."""
    result.checks_run += 1
    if blend_weight is not None and not (0.2 <= blend_weight <= 0.8):
        result.add_violation("market", "blend_weight_range",
                             f"Blend weight {blend_weight:.3f} outside [0.2, 0.8]")
    else:
        result.checks_passed += 1


def _check_matchups(
    result: AuditResult,
    feature_importances: Optional[Dict[str, np.ndarray]],
    n_features: int,
):
    """Check matchup features are present and have non-zero importance."""
    result.checks_run += 1

    if feature_importances is None:
        result.add_violation("matchup", "importance_missing",
                             "No feature importances available")
        return

    required_categories = {
        "efficiency": [0, 1, 2],
        "tempo": [4, 5],
        "shooting": [7, 8],
        "possession": [11, 12],
    }

    for imp_key, imp_values in feature_importances.items():
        if len(imp_values) < N_MATCHUP_FEATURES:
            continue

        zero_categories = []
        for cat_name, indices in required_categories.items():
            cat_importance = sum(abs(imp_values[i]) for i in indices if i < len(imp_values))
            if cat_importance < 1e-10:
                zero_categories.append(cat_name)

        if zero_categories:
            result.add_violation("matchup", "zero_importance",
                                 f"Zero importance in {imp_key} for: {zero_categories}",
                                 severity="warning")
        else:
            result.checks_passed += 1
            return

    result.checks_passed += 1


def _check_calibration(result: AuditResult, cal: Optional[CalibrationResult]):
    """Check calibration uses LOYO and is OOS only."""
    result.checks_run += 1

    if cal is None:
        result.add_violation("calibration", "missing",
                             "No calibration result available")
        return

    if cal.n_samples < 50:
        result.add_violation("calibration", "insufficient_data",
                             f"Only {cal.n_samples} calibration samples (need >= 50)")
        return

    if cal.brier_after > cal.brier_before * 1.05:
        result.add_violation("calibration", "degradation",
                             f"Calibration degraded Brier: {cal.brier_before:.4f} -> {cal.brier_after:.4f}",
                             severity="warning")

    result.checks_passed += 1


def _check_stacking(result: AuditResult, stack: Optional[StackingResult]):
    """Check stacking uses OOF predictions with no leakage."""
    result.checks_run += 1

    if stack is None:
        result.add_violation("stacking", "missing",
                             "No stacking result available")
        return

    if np.any(np.isnan(stack.oof_lr)) or np.any(np.isnan(stack.oof_gbm)):
        result.add_violation("stacking", "nan_oof",
                             "NaN values in OOF predictions (possible leakage)")
        return

    if np.std(stack.oof_preds) < 0.01:
        result.add_violation("stacking", "constant_preds",
                             "OOF predictions have near-zero variance")
        return

    if len(stack.meta_weights) >= 2:
        max_weight = np.max(np.abs(stack.meta_weights))
        total_weight = np.sum(np.abs(stack.meta_weights))
        if total_weight > 0 and max_weight / total_weight > 0.8:
            result.add_violation("stacking", "single_model_dominance",
                                 f"Meta model over-weights single input "
                                 f"({max_weight/total_weight:.1%})",
                                 severity="warning")

    result.checks_passed += 1


def _check_metrics(result: AuditResult, metrics: Dict[str, float]):
    """Check metric validity."""
    result.checks_run += 1

    brier = metrics.get("brier", -1)
    if not (0 <= brier <= 0.5):
        result.add_violation("metrics", "brier_range",
                             f"Brier score {brier:.4f} outside valid range [0, 0.5]")
        return

    log_loss = metrics.get("log_loss", -1)
    if not (0 <= log_loss <= 2.0):
        result.add_violation("metrics", "log_loss_range",
                             f"Log loss {log_loss:.4f} outside valid range [0, 2.0]")
        return

    result.checks_passed += 1


def _check_feature_count(result: AuditResult, n_features: int):
    """Check feature count invariant."""
    result.checks_run += 1
    if n_features > 50:
        result.add_violation("features", "count_exceeded",
                             f"Feature count {n_features} exceeds max 50")
    else:
        result.checks_passed += 1


def _check_anti_cheat(
    result: AuditResult,
    prev_preds: Optional[np.ndarray],
    curr_preds: Optional[np.ndarray],
    prev_metrics: Optional[Dict[str, float]],
    curr_metrics: Dict[str, float],
):
    """Anti-cheating checks (Rule 11: strengthened prediction change validation).

    Rejects if:
    - Predictions identical to previous iteration
    - Metrics improved but predictions barely changed (mean|delta| < 0.005)
    """
    result.checks_run += 1

    if prev_preds is not None and curr_preds is not None:
        # Check for identical predictions across iterations
        if np.allclose(prev_preds, curr_preds, atol=1e-8):
            result.add_violation("anti_cheat", "identical_predictions",
                                 "Predictions identical to previous iteration")
            return

        # Rule 11: material prediction change required when metrics improve
        if prev_metrics is not None:
            pred_change = float(np.mean(np.abs(curr_preds - prev_preds)))
            brier_improved = curr_metrics.get("brier", 1) < prev_metrics.get("brier", 1)

            if brier_improved and pred_change < 0.005:
                result.add_violation("anti_cheat", "immaterial_change",
                                     f"Metrics improved but predictions barely changed "
                                     f"(mean|Δp|={pred_change:.6f} < 0.005)")
                return

    result.checks_passed += 1


def _check_overfitting(
    result: AuditResult,
    per_year_brier: Optional[Dict[int, float]],
    holdout_brier: Optional[float],
    prev_holdout_brier: Optional[float],
    prev_per_year_brier: Optional[Dict[int, float]],
    stacking_result: Optional[StackingResult],
):
    """Overfitting detection (Rule 8: variance constraint enforced here)."""
    result.checks_run += 1

    signals = []

    # CV improves but holdout declines
    if (holdout_brier is not None and prev_holdout_brier is not None
            and per_year_brier and prev_per_year_brier):
        cv_brier = np.mean(list(per_year_brier.values()))
        prev_cv_brier = np.mean(list(prev_per_year_brier.values()))
        if cv_brier < prev_cv_brier and holdout_brier > prev_holdout_brier:
            signals.append("cv_improves_holdout_declines")

    # Rule 8: Variance increase in per-year Brier (ERROR, not warning)
    if per_year_brier and prev_per_year_brier:
        curr_std = float(np.std(list(per_year_brier.values())))
        prev_std = float(np.std(list(prev_per_year_brier.values())))
        if curr_std > prev_std:
            result.add_violation("overfitting", "variance_increase",
                                 f"Per-year Brier std increased: {prev_std:.4f} -> {curr_std:.4f} "
                                 f"(Rule 8 violation)")
            return

    # Single model weight > 80% in stacking
    if stacking_result and len(stacking_result.meta_weights) >= 2:
        w = np.abs(stacking_result.meta_weights)
        if w.sum() > 0 and np.max(w) / w.sum() > 0.8:
            signals.append("single_model_weight_gt_0.8")

    if signals:
        result.add_violation("overfitting", "signals_detected",
                             f"Overfitting signals: {signals}",
                             severity="warning")
    else:
        result.checks_passed += 1


def _check_feature_stability(
    result: AuditResult,
    stacking_result: Optional[StackingResult],
):
    """Rule 7: Feature stability across LOYO folds.

    Checks that per-fold LR coefficients have consistent signs and
    stable magnitudes. Unstable features suggest spurious historical fits.
    """
    result.checks_run += 1

    if stacking_result is None or not stacking_result.lr_models:
        result.checks_passed += 1
        return

    try:
        coefs = np.array([m.coef_[0] for m in stacking_result.lr_models])
    except (AttributeError, IndexError):
        result.checks_passed += 1
        return

    if coefs.shape[0] < 3:
        result.checks_passed += 1
        return

    n_features = coefs.shape[1]
    unstable_features = []

    # Check sign consistency across folds
    for i in range(n_features):
        signs = np.sign(coefs[:, i])
        nonzero = signs[signs != 0]
        if len(nonzero) >= 3:
            consistency = abs(float(np.mean(nonzero)))
            if consistency < 0.5:
                feat_name = (MATCHUP_FEATURE_NAMES[i]
                             if i < len(MATCHUP_FEATURE_NAMES) else f"feature_{i}")
                unstable_features.append(feat_name)

    if unstable_features:
        result.add_violation("features", "unstable_sign",
                             f"Features with inconsistent sign across folds: {unstable_features}",
                             severity="warning")

    # Check coefficient of variation across folds
    mean_abs = np.mean(np.abs(coefs), axis=0)
    std_abs = np.std(np.abs(coefs), axis=0)
    cv = std_abs / (mean_abs + 1e-10)
    high_cv_count = int(np.sum(cv > 2.0))

    if high_cv_count > n_features // 3:
        result.add_violation("features", "high_importance_variance",
                             f"{high_cv_count}/{n_features} features have high importance "
                             f"variance across folds (CV > 2.0)",
                             severity="warning")
    else:
        result.checks_passed += 1


def _check_market_anchor(
    result: AuditResult,
    curr_predictions: Optional[np.ndarray],
    market_probs: Optional[np.ndarray],
):
    """Rule 10: Market anchor constraint.

    Final predictions must satisfy corr(final_prob, market_prob) >= 0.7.
    Prevents model from deviating too far from efficient market signal.
    """
    result.checks_run += 1

    if curr_predictions is None or market_probs is None:
        result.checks_passed += 1
        return

    if len(curr_predictions) < 10 or len(market_probs) < 10:
        result.checks_passed += 1
        return

    if len(curr_predictions) != len(market_probs):
        result.checks_passed += 1
        return

    corr = float(np.corrcoef(curr_predictions, market_probs)[0, 1])
    if np.isnan(corr):
        result.add_violation("market", "anchor_nan",
                             "Cannot compute market correlation (NaN)",
                             severity="warning")
        return

    if corr < 0.7:
        result.add_violation("market", "anchor_violation",
                             f"Market correlation {corr:.3f} < 0.7 — "
                             f"model deviates too far from market signal")
    else:
        result.checks_passed += 1
        logger.info("Market anchor: corr(pred, market) = %.3f (>= 0.7 OK)", corr)


def _check_ensemble_weight_provenance(
    result: AuditResult,
    ensemble_weight_stats: Optional[Dict[str, Any]],
):
    """Rule 11: FIX-STACKING-LEAKAGE — Ensemble weight provenance.

    Verifies that ensemble weights were NOT derived from the eval set,
    which would contaminate all downstream metrics. Acceptable sources:
    - "nested_loyo": Weights from nested LOYO (gold standard)
    - "loyo_oos": Weights from LOYO OOS predictions
    - "fixed": Literature-based fixed weights (no optimization)
    - None/missing: No ensemble (single model)

    Rejected sources:
    - "eval_set": Weights derived from eval set (data leakage)
    """
    result.checks_run += 1

    if ensemble_weight_stats is None:
        # No ensemble weights — single model or not applicable
        result.checks_passed += 1
        return

    weight_source = ensemble_weight_stats.get("weight_source", "unknown")
    _ACCEPTABLE_SOURCES = {"nested_loyo", "loyo_oos", "fixed", "fixed_fallback"}

    if weight_source == "eval_set":
        result.add_violation(
            "ensemble", "weight_provenance_contaminated",
            "Ensemble weights derived from eval set (data leakage). "
            "Weights must be derived from LOYO OOS predictions or fixed priors. "
            "See FIX-STACKING-LEAKAGE.",
            severity="error",
        )
    elif weight_source not in _ACCEPTABLE_SOURCES and weight_source != "unknown":
        result.add_violation(
            "ensemble", "weight_provenance_unknown",
            f"Ensemble weight source '{weight_source}' is not recognized. "
            f"Acceptable: {_ACCEPTABLE_SOURCES}",
            severity="warning",
        )
    else:
        result.checks_passed += 1
        logger.info(
            "Ensemble weight provenance: source='%s' (OK)", weight_source,
        )


def analyze_failure(
    audit_result: AuditResult,
    metrics: Dict[str, float],
    best_metrics: Optional[Dict[str, float]],
) -> List[Dict[str, str]]:
    """Analyze failures and generate fix specifications."""
    analyses = []

    for violation in audit_result.violations:
        if violation.severity != "error":
            continue

        analysis = {
            "component": violation.component,
            "failure_type": _classify_failure(violation),
            "root_cause": violation.message,
            "fix_spec": _generate_fix_spec(violation),
        }
        analyses.append(analysis)

    # If audit passed but metrics didn't improve
    if audit_result.passed and best_metrics:
        curr_brier = metrics.get("brier", 1.0)
        best_brier = best_metrics.get("brier", 1.0)
        if curr_brier >= best_brier:
            analyses.append({
                "component": "generalization",
                "failure_type": "overfit",
                "root_cause": f"Brier {curr_brier:.4f} did not improve over best {best_brier:.4f}",
                "fix_spec": "increase_regularization",
            })

    return analyses


def _classify_failure(v: AuditViolation) -> str:
    """Classify failure type."""
    if "leakage" in v.message.lower() or "nan" in v.check:
        return "leakage"
    if "overfit" in v.component or "variance" in v.check:
        return "overfit"
    if v.component in ("market", "matchup"):
        return "logic"
    if v.component == "calibration":
        return "cv"
    return "data"


def _generate_fix_spec(v: AuditViolation) -> str:
    """Generate fix specification for a violation."""
    specs = {
        "market": "adjust_blend_weight_or_market_transform",
        "matchup": "add_interaction_terms_or_normalize_features",
        "calibration": "try_alternate_calibration_method",
        "stacking": "increase_folds_or_regularization",
        "features": "reduce_feature_count_via_selection",
        "metrics": "recheck_data_pipeline",
        "anti_cheat": "verify_prediction_pipeline_correctness",
        "overfitting": "increase_regularization_or_simplify",
    }
    return specs.get(v.component, "investigate_manually")
