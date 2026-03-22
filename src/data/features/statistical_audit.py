"""Statistical audit tools for feature engineering evaluation.

Addresses three concerns from senior statistician review:

1. **Systematic Redundancy Audit**: Exhaustive pairwise correlation check
   (replacing the manual REMOVED_REDUNDANCIES list) with algebraic
   dependency detection via eigenvalue analysis.

2. **Effective DoF Budget**: Harrell's rule-of-thumb enforcement —
   computes the maximum number of features the sample size can support
   and enforces it as a ceiling on feature selection.

3. **Absolute-Level Feature Validation**: Bootstrap CI on marginal Brier
   improvement from absolute-level features to ensure they earn their
   dimensionality cost.

References:
    Harrell, F.E. (2015). "Regression Modeling Strategies", §4.3.
    Meinshausen & Buhlmann (2010). "Stability Selection", JRSS-B.
    Steyerberg et al. (2001). "Internal validation of predictive models",
    Journal of Clinical Epidemiology, 54(8), 774-781.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Systematic Redundancy Audit
# ---------------------------------------------------------------------------


@dataclass
class RedundancyPair:
    """A pair of features flagged as redundant."""

    feature_a: str
    feature_b: str
    correlation: float
    redundancy_type: str  # "pairwise_correlation", "algebraic_dependency", "eigenvalue_collapse"


@dataclass
class RedundancyAuditResult:
    """Result of exhaustive redundancy audit."""

    flagged_pairs: List[RedundancyPair]
    eigenvalue_spectrum: Optional[np.ndarray]
    effective_rank: int  # Number of eigenvalues > threshold
    n_features: int
    correlation_threshold: float
    n_above_threshold: int  # Count of pairs above threshold
    audit_complete: bool


class SystematicRedundancyAuditor:
    """Exhaustive pairwise correlation audit with eigenvalue analysis.

    Replaces the manual REMOVED_REDUNDANCIES list with a systematic check
    of ALL pairwise correlations above a threshold.  Also performs
    eigenvalue analysis to detect multi-way linear dependencies that
    pairwise correlation misses.

    This audit is diagnostic — it reports what it finds but does not
    modify the feature matrix.  The FeatureSelector pipeline handles
    the actual pruning.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        eigenvalue_threshold: float = 1e-4,
    ):
        self.correlation_threshold = correlation_threshold
        self.eigenvalue_threshold = eigenvalue_threshold

    def audit(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> RedundancyAuditResult:
        """Run exhaustive redundancy audit on the feature matrix.

        Args:
            X: Feature matrix [N, D]
            feature_names: Feature names matching columns of X

        Returns:
            RedundancyAuditResult with all flagged pairs and eigenvalue analysis
        """
        n_samples, n_features = X.shape
        flagged_pairs: List[RedundancyPair] = []

        # --- Pairwise correlation audit (exhaustive) ---
        stds = np.std(X, axis=0)
        X_safe = X.copy()
        # Handle constant columns to avoid NaN in corrcoef
        constant_mask = stds < 1e-10
        if np.any(constant_mask):
            rng = np.random.default_rng(42)
            X_safe[:, constant_mask] = rng.standard_normal(
                (n_samples, int(np.sum(constant_mask)))
            ) * 1e-6

        corr_matrix = np.corrcoef(X_safe.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        n_above = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                r = abs(corr_matrix[i, j])
                if r > self.correlation_threshold:
                    n_above += 1
                    flagged_pairs.append(RedundancyPair(
                        feature_a=feature_names[i],
                        feature_b=feature_names[j],
                        correlation=float(r),
                        redundancy_type="pairwise_correlation",
                    ))

        # --- Eigenvalue analysis for multi-way dependencies ---
        # Standardize before PCA-style analysis
        X_centered = X_safe - np.mean(X_safe, axis=0)
        X_std = np.std(X_centered, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_normed = X_centered / X_std

        try:
            # Compute eigenvalues of the correlation/covariance matrix
            cov = np.dot(X_normed.T, X_normed) / max(n_samples - 1, 1)
            # Guard against NaN/inf from degenerate inputs
            if not np.all(np.isfinite(cov)):
                cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # descending

            # Effective rank: number of eigenvalues above threshold
            effective_rank = int(np.sum(eigenvalues > self.eigenvalue_threshold))

            # Flag collapsed eigenvalue directions
            n_collapsed = n_features - effective_rank
            if n_collapsed > 0:
                logger.info(
                    "Eigenvalue analysis: %d/%d directions collapsed "
                    "(eigenvalue < %.1e). Effective rank = %d.",
                    n_collapsed, n_features,
                    self.eigenvalue_threshold, effective_rank,
                )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            eigenvalues = None
            effective_rank = n_features
            logger.warning("Eigenvalue computation failed; skipping eigenvalue analysis.")

        if n_above > 0:
            logger.info(
                "Redundancy audit: %d feature pairs exceed |r| > %.2f "
                "out of %d total pairs.",
                n_above, self.correlation_threshold,
                n_features * (n_features - 1) // 2,
            )

        return RedundancyAuditResult(
            flagged_pairs=flagged_pairs,
            eigenvalue_spectrum=eigenvalues,
            effective_rank=effective_rank,
            n_features=n_features,
            correlation_threshold=self.correlation_threshold,
            n_above_threshold=n_above,
            audit_complete=True,
        )


# ---------------------------------------------------------------------------
# 2. Effective DoF Budget (Harrell's Rule)
# ---------------------------------------------------------------------------


@dataclass
class DoFBudgetResult:
    """Result of effective degrees-of-freedom budget calculation."""

    n_events: int  # min(n_positive, n_negative) for binary classification
    n_samples: int
    events_per_predictor_target: float  # Harrell's guideline (10-20)
    max_features_allowed: int  # n_events / events_per_predictor
    current_features: int
    budget_exceeded: bool
    recommended_max_features: int
    message: str


def compute_dof_budget(
    n_samples: int,
    n_positive: int,
    n_features: int,
    events_per_predictor: float = 15.0,
    n_tuned_constants: int = 58,
) -> DoFBudgetResult:
    """Compute the effective degrees-of-freedom budget per Harrell's rule.

    Harrell's rule of thumb: the number of candidate predictors should not
    exceed min(n_positive, n_negative) / EPP, where EPP (events per
    predictor) is typically 10-20.  We use 15 as a balanced default.

    The budget also accounts for tuned pipeline constants (hyperparameters,
    scaling factors, etc.) that consume degrees of freedom even though
    they are not "features" in the traditional sense.

    Args:
        n_samples: Total number of samples
        n_positive: Number of positive outcomes (wins)
        n_features: Current number of features
        events_per_predictor: Harrell's EPP target (default 15)
        n_tuned_constants: Number of tuned pipeline constants

    Returns:
        DoFBudgetResult with budget analysis
    """
    n_negative = n_samples - n_positive
    n_events = min(n_positive, n_negative)

    # Total DoF budget
    total_budget = int(n_events / events_per_predictor)

    # Features get the budget minus tuned constants
    # (constants consume DoF even if they're not ML features)
    feature_budget = max(total_budget - n_tuned_constants, 10)

    # But also cap by the simpler rule: features alone shouldn't
    # exceed n_events / EPP
    feature_only_budget = int(n_events / events_per_predictor)

    # Use the more conservative estimate
    recommended = min(feature_budget, feature_only_budget)
    # Floor at 10 to avoid degenerate cases
    recommended = max(recommended, 10)

    budget_exceeded = n_features > recommended

    if budget_exceeded:
        message = (
            f"DoF budget EXCEEDED: {n_features} features > {recommended} max "
            f"(Harrell's rule: {n_events} events / {events_per_predictor:.0f} EPP = "
            f"{feature_only_budget}). Recommend reducing to <= {recommended} features."
        )
        logger.warning(message)
    else:
        message = (
            f"DoF budget OK: {n_features} features <= {recommended} max "
            f"({n_events} events / {events_per_predictor:.0f} EPP)."
        )
        logger.info(message)

    return DoFBudgetResult(
        n_events=n_events,
        n_samples=n_samples,
        events_per_predictor_target=events_per_predictor,
        max_features_allowed=feature_only_budget,
        current_features=n_features,
        budget_exceeded=budget_exceeded,
        recommended_max_features=recommended,
        message=message,
    )


# ---------------------------------------------------------------------------
# 3. Bootstrap CI for Absolute-Level Feature Validation
# ---------------------------------------------------------------------------


@dataclass
class AbsoluteFeatureValidationResult:
    """Result of bootstrap validation for absolute-level features."""

    brier_with_absolute: float
    brier_without_absolute: float
    brier_improvement: float  # positive = absolute features help
    bootstrap_ci_lower: float  # 95% CI lower bound
    bootstrap_ci_upper: float  # 95% CI upper bound
    n_bootstrap: int
    significant: bool  # True if CI excludes zero (improvement is real)
    absolute_feature_names: List[str]
    message: str


def validate_absolute_features(
    X_diff: np.ndarray,
    X_absolute: np.ndarray,
    X_interactions: np.ndarray,
    y: np.ndarray,
    absolute_feature_names: List[str],
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> AbsoluteFeatureValidationResult:
    """Validate that absolute-level features provide significant Brier improvement.

    Computes the marginal Brier improvement from including absolute-level
    features (avg of both teams' top metrics) and constructs a bootstrap
    confidence interval around the improvement.

    Uses a simple logistic regression model (not the full ensemble) to
    isolate the marginal contribution of absolute features without
    confounding from model complexity.

    Args:
        X_diff: Differential features [N, D_diff]
        X_absolute: Absolute-level features [N, D_abs]
        X_interactions: Interaction features [N, D_int]
        y: Binary outcome labels [N]
        absolute_feature_names: Names of absolute-level features
        n_bootstrap: Number of bootstrap iterations
        confidence_level: CI confidence level (default 0.95)
        random_seed: Random seed for reproducibility

    Returns:
        AbsoluteFeatureValidationResult with Brier scores and bootstrap CI
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return AbsoluteFeatureValidationResult(
            brier_with_absolute=0.0,
            brier_without_absolute=0.0,
            brier_improvement=0.0,
            bootstrap_ci_lower=0.0,
            bootstrap_ci_upper=0.0,
            n_bootstrap=0,
            significant=False,
            absolute_feature_names=absolute_feature_names,
            message="sklearn not available; skipping absolute feature validation.",
        )

    n_samples = len(y)
    rng = np.random.default_rng(random_seed)

    # Build the two feature matrices
    X_without = np.column_stack([X_diff, X_interactions])
    X_with = np.column_stack([X_diff, X_absolute, X_interactions])

    def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(np.mean((y_true - y_prob) ** 2))

    def _evaluate_model(X_train, y_train, X_val, y_val):
        """Fit logistic regression and return Brier score on validation set."""
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_seed, solver="lbfgs"
        )
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_val_s)[:, 1]
        return _brier_score(y_val, probs)

    # Point estimate: 3-fold cross-validation Brier
    def _cv_brier(X_full):
        """Simple 3-fold CV Brier score."""
        fold_size = n_samples // 3
        briers = []
        indices = np.arange(n_samples)
        rng_cv = np.random.default_rng(random_seed)
        rng_cv.shuffle(indices)
        for fold in range(3):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < 2 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            if len(train_idx) < 20 or len(val_idx) < 5:
                continue
            brier = _evaluate_model(
                X_full[train_idx], y[train_idx],
                X_full[val_idx], y[val_idx],
            )
            briers.append(brier)
        return float(np.mean(briers)) if briers else 0.25

    brier_without = _cv_brier(X_without)
    brier_with = _cv_brier(X_with)
    point_improvement = brier_without - brier_with  # positive = abs features help

    # Bootstrap CI on the improvement
    bootstrap_improvements = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_with_boot = X_with[boot_idx]
        X_without_boot = X_without[boot_idx]
        y_boot = y[boot_idx]

        try:
            b_without = _cv_brier(X_without_boot)
            b_with = _cv_brier(X_with_boot)
            bootstrap_improvements.append(b_without - b_with)
        except Exception:
            continue

    if len(bootstrap_improvements) < 10:
        return AbsoluteFeatureValidationResult(
            brier_with_absolute=brier_with,
            brier_without_absolute=brier_without,
            brier_improvement=point_improvement,
            bootstrap_ci_lower=0.0,
            bootstrap_ci_upper=0.0,
            n_bootstrap=len(bootstrap_improvements),
            significant=False,
            absolute_feature_names=absolute_feature_names,
            message="Too few successful bootstrap iterations for CI estimation.",
        )

    improvements = np.array(bootstrap_improvements)
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(improvements, 100 * alpha / 2))
    ci_upper = float(np.percentile(improvements, 100 * (1 - alpha / 2)))

    # Significant if the entire CI is above zero (improvement is real)
    significant = ci_lower > 0

    if significant:
        message = (
            f"Absolute-level features VALIDATED: Brier improvement = "
            f"{point_improvement:.5f} [{ci_lower:.5f}, {ci_upper:.5f}] 95% CI. "
            f"CI excludes zero — the {len(absolute_feature_names)} features "
            f"({', '.join(absolute_feature_names)}) earn their dimensionality cost."
        )
        logger.info(message)
    else:
        message = (
            f"Absolute-level features NOT significant: Brier improvement = "
            f"{point_improvement:.5f} [{ci_lower:.5f}, {ci_upper:.5f}] 95% CI. "
            f"CI includes zero — consider removing the {len(absolute_feature_names)} "
            f"absolute features to reduce DoF pressure."
        )
        logger.warning(message)

    return AbsoluteFeatureValidationResult(
        brier_with_absolute=brier_with,
        brier_without_absolute=brier_without,
        brier_improvement=point_improvement,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        n_bootstrap=len(bootstrap_improvements),
        significant=significant,
        absolute_feature_names=absolute_feature_names,
        message=message,
    )


# ---------------------------------------------------------------------------
# 4. Interaction Feature Validation (Solution 8)
# ---------------------------------------------------------------------------


@dataclass
class InteractionValidationResult:
    """Result of interaction feature marginal Brier validation."""

    brier_with_interactions: float
    brier_without_interactions: float
    marginal_improvement: float  # positive = interactions help
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    n_bootstrap: int
    significant: bool  # True if CI excludes zero
    per_feature_mi: Dict[str, float]  # Mutual information per interaction feature
    interaction_names: List[str]
    message: str


def validate_interaction_features(
    X_base: np.ndarray,
    X_interactions: np.ndarray,
    y: np.ndarray,
    interaction_names: List[str],
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> InteractionValidationResult:
    """Validate that interaction features provide significant Brier improvement.

    Solution 8 + FIX C5: Interaction features split into "always" (seed_diff,
    seed_em_residual — domain knowledge) and "optional" (5 product terms trees
    can learn natively).  This validates the optional interactions earn their
    dimensionality cost via bootstrap CI on marginal Brier improvement.

    Args:
        X_base: Base features (diff + absolute) [N, D_base]
        X_interactions: Interaction features [N, D_int]
        y: Binary outcome labels [N]
        interaction_names: Names of interaction features
        n_bootstrap: Number of bootstrap iterations
        confidence_level: CI confidence level
        random_seed: Random seed

    Returns:
        InteractionValidationResult with Brier improvement and per-feature MI
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return InteractionValidationResult(
            brier_with_interactions=0.0, brier_without_interactions=0.0,
            marginal_improvement=0.0, bootstrap_ci_lower=0.0,
            bootstrap_ci_upper=0.0, n_bootstrap=0, significant=False,
            per_feature_mi={}, interaction_names=interaction_names,
            message="sklearn not available; skipping interaction validation.",
        )

    n_samples = len(y)
    rng = np.random.default_rng(random_seed)

    X_without = X_base
    X_with = np.column_stack([X_base, X_interactions])

    def _brier(y_true, y_prob):
        return float(np.mean((y_true - y_prob) ** 2))

    def _cv_brier(X_full):
        """3-fold CV Brier score."""
        fold_size = n_samples // 3
        briers = []
        indices = np.arange(n_samples)
        rng_cv = np.random.default_rng(random_seed)
        rng_cv.shuffle(indices)
        for fold in range(3):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < 2 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            if len(train_idx) < 20 or len(val_idx) < 5:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_full[train_idx])
            X_va = scaler.transform(X_full[val_idx])
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed, solver="lbfgs")
            model.fit(X_tr, y[train_idx])
            probs = model.predict_proba(X_va)[:, 1]
            briers.append(_brier(y[val_idx], probs))
        return float(np.mean(briers)) if briers else 0.25

    brier_without = _cv_brier(X_without)
    brier_with = _cv_brier(X_with)
    point_improvement = brier_without - brier_with

    # Bootstrap CI
    bootstrap_improvements = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        try:
            b_without = _cv_brier(X_without[boot_idx])
            b_with = _cv_brier(X_with[boot_idx])
            bootstrap_improvements.append(b_without - b_with)
        except Exception:
            continue

    # Per-feature MI
    per_feature_mi = {}
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(
            X_interactions, y, n_neighbors=5, random_state=random_seed
        )
        for i, name in enumerate(interaction_names):
            if i < len(mi_scores):
                per_feature_mi[name] = float(mi_scores[i])
    except Exception:
        pass

    if len(bootstrap_improvements) < 10:
        return InteractionValidationResult(
            brier_with_interactions=brier_with,
            brier_without_interactions=brier_without,
            marginal_improvement=point_improvement,
            bootstrap_ci_lower=0.0, bootstrap_ci_upper=0.0,
            n_bootstrap=len(bootstrap_improvements), significant=False,
            per_feature_mi=per_feature_mi, interaction_names=interaction_names,
            message="Too few bootstrap iterations for CI estimation.",
        )

    improvements = np.array(bootstrap_improvements)
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(improvements, 100 * alpha / 2))
    ci_upper = float(np.percentile(improvements, 100 * (1 - alpha / 2)))
    significant = ci_lower > 0

    if significant:
        message = (
            f"Interaction features VALIDATED: Brier improvement = "
            f"{point_improvement:.5f} [{ci_lower:.5f}, {ci_upper:.5f}] 95% CI. "
            f"CI excludes zero — the {len(interaction_names)} interactions "
            f"earn their dimensionality cost."
        )
        logger.info(message)
    else:
        message = (
            f"Interaction features NOT significant: Brier improvement = "
            f"{point_improvement:.5f} [{ci_lower:.5f}, {ci_upper:.5f}] 95% CI. "
            f"CI includes zero — consider removing interaction features."
        )
        logger.warning(message)

    return InteractionValidationResult(
        brier_with_interactions=brier_with,
        brier_without_interactions=brier_without,
        marginal_improvement=point_improvement,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        n_bootstrap=len(bootstrap_improvements),
        significant=significant,
        per_feature_mi=per_feature_mi,
        interaction_names=interaction_names,
        message=message,
    )
