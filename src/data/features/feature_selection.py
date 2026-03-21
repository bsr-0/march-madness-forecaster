"""
Feature selection and dimensionality reduction pipeline.

Provides:
- VIFPruner: drops features with high Variance Inflation Factor
- CorrelationPruner: drops highly correlated features
- ImportanceCalculator: multi-method importance ranking
- FeatureSelector: orchestrates the full selection pipeline

FIX AUDIT (2026-02-19):
  #3: VIF pruning enabled by default (was False).
  #5: CorrelationPruner tie-breaking changed from variance-based to
      target-correlation-based.
  #6: Added bootstrap stability filtering — features must be selected
      in >=80% of bootstrap runs to survive.
  #7: Correlation-with-target importance method suppressed when SHAP
      is available (adds noise, not signal).

REFACTOR AUDIT (2026-03-21):
  SOL-1:  MutualInformationScreener — non-linear feature-target association.
  SOL-2:  Distribution shift auto-gating — closes the shift-detection loop.
  SOL-9:  SHAP importance hyperparameter alignment with production LightGBM.
  SOL-11: Adaptive importance threshold via elbow detection (Kneedle).
  SOL-12: Cross-fold stability metrics (Jaccard index across LOYO folds).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Importance score for a single feature."""

    name: str
    importance: float
    rank: int = 0


@dataclass
class FeatureSelectionResult:
    """Result of feature selection pipeline."""

    selected_features: List[str]
    selected_indices: List[int]
    dropped_features: List[str]
    importance_scores: List[FeatureImportance]
    correlation_dropped: List[str]
    low_importance_dropped: List[str]
    original_dim: int
    reduced_dim: int
    method: str
    # FIX #6: Bootstrap stability scores (feature_name -> fraction of bootstrap runs selected)
    stability_scores: Optional[Dict[str, float]] = None
    # Variance-pruned features (near-zero variance)
    variance_dropped: List[str] = field(default_factory=list)
    # Post-selection multicollinearity diagnostics
    post_selection_condition_number: Optional[float] = None
    post_selection_max_vif: Optional[float] = None
    multicollinearity_warning: Optional[str] = None
    # SA-1: Redundancy audit diagnostics
    redundancy_pairs_found: int = 0
    effective_rank: Optional[int] = None
    # SA-2: DoF budget diagnostics
    dof_budget_max_features: Optional[int] = None
    dof_budget_exceeded: Optional[bool] = None
    # MI screening diagnostics
    mi_dropped: List[str] = field(default_factory=list)
    mi_scores: Optional[Dict[str, float]] = None
    # Shift gating diagnostics
    shifted_features: List[str] = field(default_factory=list)
    # Adaptive threshold
    detected_importance_threshold: Optional[float] = None


class VIFPruner:
    """
    Iteratively drops features with Variance Inflation Factor > threshold.

    VIF detects multicollinearity — including exact linear dependencies among
    3+ features that pairwise correlation misses.  For example,
    ``adj_em = adj_off - adj_def`` is an exact linear dependency (VIF = inf)
    but the pairwise correlations between adj_em and adj_off may only be ~0.9.

    Standard threshold: VIF > 10 indicates problematic collinearity.
    """

    def __init__(self, threshold: float = 10.0, max_drops: int = 10):
        self.threshold = threshold
        self.max_drops = max_drops
        self.dropped_features: List[str] = []

    def prune(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Iteratively drop the feature with highest VIF until all VIF <= threshold."""
        n_samples, n_features = X.shape
        if n_features <= 2 or n_samples < n_features:
            return X, list(feature_names), []

        kept_indices = list(range(n_features))
        dropped_names: List[str] = []

        for _ in range(self.max_drops):
            if len(kept_indices) <= 2:
                break

            X_sub = X[:, kept_indices]
            vifs = self._compute_vifs(X_sub)

            worst_idx = int(np.argmax(vifs))
            if vifs[worst_idx] <= self.threshold:
                break

            dropped_name = feature_names[kept_indices[worst_idx]]
            dropped_names.append(dropped_name)
            logger.info(
                "VIF pruning: dropped '%s' (VIF=%.1f)",
                dropped_name, vifs[worst_idx],
            )
            kept_indices.pop(worst_idx)

        self.dropped_features = dropped_names
        kept_names = [feature_names[i] for i in kept_indices]
        return X[:, kept_indices], kept_names, dropped_names

    @staticmethod
    def _compute_vifs(X: np.ndarray) -> np.ndarray:
        """Compute VIF for each feature."""
        from numpy.linalg import lstsq

        n, p = X.shape
        vifs = np.zeros(p)

        for j in range(p):
            others = np.delete(X, j, axis=1)
            others_aug = np.column_stack([others, np.ones(n)])
            coeffs, _, _, _ = lstsq(others_aug, X[:, j], rcond=None)
            fitted = others_aug @ coeffs
            ss_res = float(np.sum((X[:, j] - fitted) ** 2))
            ss_tot = float(np.sum((X[:, j] - np.mean(X[:, j])) ** 2))

            if ss_tot < 1e-12:
                vifs[j] = float("inf")
            else:
                r_squared = 1.0 - ss_res / ss_tot
                vifs[j] = 1.0 / max(1.0 - r_squared, 1e-12)

        return vifs


class NearZeroVariancePruner:
    """
    Drops features with variance below a threshold.

    Near-zero variance features are effectively constant across the training
    set and provide no discriminative signal.  They also cause numerical
    issues in VIF computation (division by near-zero SS_tot) and inflate
    logistic regression coefficient estimates.

    Should run BEFORE VIF pruning to avoid VIF=inf on constant features.
    """

    def __init__(self, threshold: float = 1e-7):
        self.threshold = threshold
        self.dropped_features: List[str] = []

    def prune(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Remove features with variance below threshold."""
        variances = np.var(X, axis=0)
        keep_mask = variances > self.threshold
        dropped_names = [
            feature_names[i] for i in range(len(feature_names)) if not keep_mask[i]
        ]
        kept_names = [
            feature_names[i] for i in range(len(feature_names)) if keep_mask[i]
        ]
        self.dropped_features = dropped_names
        if dropped_names:
            logger.info(
                "Near-zero variance pruning removed %d features: %s",
                len(dropped_names), dropped_names,
            )
        return X[:, keep_mask], kept_names, dropped_names


def compute_condition_number(X: np.ndarray) -> float:
    """
    Compute the condition number of the feature matrix.

    The condition number measures the overall numerical stability of the
    feature matrix.  High values indicate that the features are nearly
    linearly dependent and that linear model coefficients will be unstable.

    Thresholds (rule of thumb):
      - < 30: acceptable
      - 30-100: moderate collinearity, monitor coefficients
      - > 100: severe collinearity, feature reduction needed

    Uses SVD-based computation (ratio of largest to smallest singular value).
    """
    try:
        return float(np.linalg.cond(X))
    except np.linalg.LinAlgError:
        return float("inf")


def validate_post_selection_collinearity(
    X: np.ndarray,
    feature_names: List[str],
    vif_threshold: float = 10.0,
    condition_threshold: float = 100.0,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Post-selection multicollinearity validation.

    After all pruning stages, verify that the final feature set is
    free of residual collinearity.  This catches cases where:
    - VIF pruning hit max_drops before resolving all issues
    - Correlation pruning missed 3+ feature linear dependencies
    - Feature interactions reintroduced collinearity

    Returns:
        Tuple of (condition_number, max_vif, warning_message_or_None)
    """
    n_samples, n_features = X.shape
    warning_parts = []

    # Condition number
    cond_num = compute_condition_number(X)
    if cond_num > condition_threshold:
        warning_parts.append(
            f"condition number {cond_num:.0f} exceeds threshold {condition_threshold:.0f}"
        )

    # Max VIF (only if enough samples relative to features)
    max_vif = None
    if n_samples > n_features + 2 and n_features >= 2:
        vifs = VIFPruner._compute_vifs(X)
        max_vif = float(np.max(vifs))
        if max_vif > vif_threshold:
            worst_idx = int(np.argmax(vifs))
            warning_parts.append(
                f"residual VIF={max_vif:.1f} on feature '{feature_names[worst_idx]}' "
                f"exceeds threshold {vif_threshold:.1f}"
            )

    warning = None
    if warning_parts:
        warning = (
            "Post-selection multicollinearity warning: "
            + "; ".join(warning_parts)
            + ". LogisticRegression coefficients may be unstable."
        )
        logger.warning(warning)

    return cond_num, max_vif, warning


@dataclass
class DistributionShiftResult:
    """Per-feature distribution shift diagnostics between train and validation."""

    feature_name: str
    psi: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # KS test p-value
    train_mean: float
    val_mean: float
    mean_shift_std: float  # Mean difference in units of train std
    flagged: bool  # True if shift exceeds thresholds


def compute_psi(
    train: np.ndarray,
    val: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.

    PSI measures how much a distribution has shifted.  Originally from
    credit scoring, it's the standard metric for feature drift in
    production ML systems.

    Thresholds (industry standard):
      - < 0.10: no significant shift
      - 0.10-0.25: moderate shift, investigate
      - > 0.25: significant shift, feature may be unreliable

    Uses equal-frequency (quantile) binning on the training distribution
    for robustness with skewed features.

    Args:
        train: Training feature values [N_train]
        val: Validation feature values [N_val]
        n_bins: Number of bins

    Returns:
        PSI value (non-negative float)
    """
    eps = 1e-6

    # Quantile-based bin edges from training distribution
    bin_edges = np.percentile(train, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Ensure monotonically increasing (can fail if many identical values)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0  # Feature has too little variation to compute PSI

    train_counts = np.histogram(train, bins=bin_edges)[0].astype(float)
    val_counts = np.histogram(val, bins=bin_edges)[0].astype(float)

    # Normalize to proportions
    train_pct = train_counts / max(train_counts.sum(), 1) + eps
    val_pct = val_counts / max(val_counts.sum(), 1) + eps

    # PSI = Σ (val_pct - train_pct) * ln(val_pct / train_pct)
    psi = float(np.sum((val_pct - train_pct) * np.log(val_pct / train_pct)))
    return max(psi, 0.0)


def detect_distribution_shift(
    train_X: np.ndarray,
    val_X: np.ndarray,
    feature_names: List[str],
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05,
    mean_shift_threshold: float = 1.0,
) -> List[DistributionShiftResult]:
    """
    Detect distribution shift between training and validation features.

    Applies three complementary tests per feature:
    1. PSI (Population Stability Index) — distribution shape change
    2. KS test — maximum CDF divergence (non-parametric)
    3. Standardized mean shift — location change in units of train std

    A feature is flagged if ANY of:
      - PSI > psi_threshold (default 0.25)
      - KS p-value < ks_alpha (default 0.05)
      - |mean_shift| > mean_shift_threshold standard deviations (default 1.0)

    Args:
        train_X: Training feature matrix [N_train, D]
        val_X: Validation feature matrix [N_val, D]
        feature_names: Feature names
        psi_threshold: PSI flagging threshold
        ks_alpha: KS test significance level
        mean_shift_threshold: Flagging threshold for standardized mean shift

    Returns:
        List of DistributionShiftResult, one per feature, sorted by PSI descending
    """
    try:
        from scipy.stats import ks_2samp
        scipy_available = True
    except ImportError:
        scipy_available = False

    n_features = min(train_X.shape[1], len(feature_names))
    results = []

    for j in range(n_features):
        train_col = train_X[:, j]
        val_col = val_X[:, j]

        # PSI
        psi = compute_psi(train_col, val_col)

        # KS test
        if scipy_available:
            ks_stat, ks_pval = ks_2samp(train_col, val_col)
        else:
            ks_stat, ks_pval = 0.0, 1.0

        # Standardized mean shift
        train_mean = float(np.mean(train_col))
        val_mean = float(np.mean(val_col))
        train_std = float(np.std(train_col))
        if train_std > 1e-10:
            mean_shift_std = abs(val_mean - train_mean) / train_std
        else:
            mean_shift_std = 0.0

        flagged = (
            psi > psi_threshold
            or ks_pval < ks_alpha
            or mean_shift_std > mean_shift_threshold
        )

        results.append(DistributionShiftResult(
            feature_name=feature_names[j],
            psi=psi,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            train_mean=train_mean,
            val_mean=val_mean,
            mean_shift_std=mean_shift_std,
            flagged=flagged,
        ))

    results.sort(key=lambda r: r.psi, reverse=True)

    n_flagged = sum(1 for r in results if r.flagged)
    if n_flagged > 0:
        flagged_names = [r.feature_name for r in results if r.flagged][:10]
        logger.warning(
            "Distribution shift detected in %d/%d features: %s%s",
            n_flagged, n_features, flagged_names,
            "..." if n_flagged > 10 else "",
        )

    return results


class CorrelationPruner:
    """
    Drops features with pairwise correlation above a threshold.

    FIX #5: When two features are highly correlated, keeps the one with
    higher absolute correlation with the target (y).  Previously kept the
    one with higher variance, which favors noisy features over predictive
    ones.

    Falls back to variance-based tie-breaking when y is not provided.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def prune(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Remove highly correlated features.

        FIX #5: Uses target correlation for tie-breaking when y is provided.

        Args:
            X: Feature matrix [N, D]
            feature_names: Names of features
            y: Optional target labels for informed tie-breaking

        Returns:
            Tuple of (pruned_X, kept_names, dropped_names)
        """
        n_features = X.shape[1]
        if n_features <= 1:
            return X, list(feature_names), []

        # Compute correlation matrix
        stds = np.std(X, axis=0)
        constant_mask = stds < 1e-10
        X_safe = X.copy()
        rng = np.random.default_rng(42)
        X_safe[:, constant_mask] = rng.standard_normal((X.shape[0], int(np.sum(constant_mask)))) * 1e-6

        corr = np.corrcoef(X_safe.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # FIX #5: Compute target correlations for tie-breaking
        if y is not None:
            target_corrs = np.zeros(n_features)
            for i in range(n_features):
                if stds[i] > 1e-10:
                    r = np.corrcoef(X[:, i], y)[0, 1]
                    target_corrs[i] = 0.0 if np.isnan(r) else abs(r)
            use_target_corr = True
        else:
            # Fallback: use variance (original behavior)
            target_corrs = np.var(X, axis=0)
            use_target_corr = False

        to_drop = set()

        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > self.threshold:
                    # FIX #5: Keep the one with higher target correlation
                    # (or higher variance if no target)
                    if target_corrs[i] >= target_corrs[j]:
                        to_drop.add(j)
                        if use_target_corr:
                            logger.debug(
                                "Corr pruning: dropped '%s' (target_r=%.3f) "
                                "in favor of '%s' (target_r=%.3f), pair_r=%.3f",
                                feature_names[j], target_corrs[j],
                                feature_names[i], target_corrs[i],
                                corr[i, j],
                            )
                    else:
                        to_drop.add(i)
                        if use_target_corr:
                            logger.debug(
                                "Corr pruning: dropped '%s' (target_r=%.3f) "
                                "in favor of '%s' (target_r=%.3f), pair_r=%.3f",
                                feature_names[i], target_corrs[i],
                                feature_names[j], target_corrs[j],
                                corr[i, j],
                            )
                        break  # i is dropped, move on

        kept_indices = [i for i in range(n_features) if i not in to_drop]
        dropped_names = [feature_names[i] for i in sorted(to_drop)]
        kept_names = [feature_names[i] for i in kept_indices]

        return X[:, kept_indices], kept_names, dropped_names


class MutualInformationScreener:
    """Screen features using mutual information with target.

    Detects non-linear feature-target associations that pairwise
    correlation and SHAP both miss. Uses sklearn's k-nearest-neighbor
    MI estimator (Kraskov et al., 2004).
    """

    def __init__(self, threshold: float = 0.01, random_seed: int = 42):
        self.threshold = threshold
        self.random_seed = random_seed

    def screen(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
        """Screen features by mutual information with target.

        Args:
            X: Feature matrix [N, D]
            y: Binary target [N]
            feature_names: Feature names

        Returns:
            (X_kept, kept_names, dropped_names, mi_scores)
        """
        if not SKLEARN_AVAILABLE:
            return X, list(feature_names), [], np.zeros(X.shape[1])

        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(
            X, y, n_neighbors=5, random_state=self.random_seed
        )

        keep_mask = mi_scores >= self.threshold
        # Always keep at least half the features
        if keep_mask.sum() < len(feature_names) // 2:
            # Keep top half by MI score instead
            n_keep = max(len(feature_names) // 2, 1)
            top_indices = np.argsort(mi_scores)[-n_keep:]
            keep_mask = np.zeros(len(feature_names), dtype=bool)
            keep_mask[top_indices] = True

        dropped = [feature_names[i] for i in range(len(feature_names)) if not keep_mask[i]]
        kept = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]

        if dropped:
            logger.info(
                "MI screening removed %d features below threshold %.4f: %s",
                len(dropped), self.threshold, dropped[:10],
            )

        return X[:, keep_mask], kept, dropped, mi_scores


class ImportanceCalculator:
    """
    Calculates feature importance using multiple methods and combines them.

    FIX #7: When SHAP is available, the correlation-with-target method is
    suppressed (weight=0).  It adds noise to the ranking without meaningful
    signal when SHAP and permutation importance are both available.

    Priority order (highest weight first):
    1. SHAP TreeExplainer (if shap + LightGBM available)
    2. Permutation importance with LightGBM (sklearn)
    3. Absolute correlation with target (ONLY when SHAP unavailable)

    SHAP is weighted 2x relative to permutation.
    """

    # FIX #7: Weights are dynamic — correlation gets 0 when SHAP is available
    BASE_METHOD_WEIGHTS = {
        "shap": 2.0,
        "permutation": 1.0,
        "correlation": 1.0,  # Set to 0.0 dynamically when SHAP succeeds
    }

    def __init__(self, random_seed: int = 42, lgb_params: Optional[Dict] = None):
        self.random_seed = random_seed
        self.lgb_params = lgb_params

    def calculate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureImportance]:
        """Calculate combined feature importance."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        total_weight = 0.0
        shap_succeeded = False

        # Method 1 (primary): SHAP TreeExplainer via LightGBM
        if SHAP_AVAILABLE and LIGHTGBM_AVAILABLE:
            try:
                shap_scores = self._shap_importance(X, y, feature_names)
                if shap_scores is not None:
                    w = self.BASE_METHOD_WEIGHTS["shap"]
                    scores += w * self._normalize(shap_scores)
                    total_weight += w
                    shap_succeeded = True
            except Exception as e:
                logger.warning("SHAP importance failed: %s", e)

        # Method 2: Permutation importance
        if SKLEARN_AVAILABLE:
            try:
                perm_scores = self._permutation_importance(X, y)
                if perm_scores is not None:
                    w = self.BASE_METHOD_WEIGHTS["permutation"]
                    scores += w * self._normalize(perm_scores)
                    total_weight += w
            except Exception as e:
                logger.warning("Permutation importance failed: %s", e)

        # FIX #7: Only use correlation importance when SHAP is NOT available.
        # Correlation with target is a weak signal that adds noise when
        # SHAP (which decomposes model predictions into per-feature
        # contributions) is already available.
        if not shap_succeeded:
            corr_scores = self._correlation_importance(X, y)
            w = self.BASE_METHOD_WEIGHTS["correlation"]
            scores += w * self._normalize(corr_scores)
            total_weight += w

        if total_weight > 0:
            scores /= total_weight

        # Build results sorted by importance
        results = []
        order = np.argsort(-scores)
        for rank, idx in enumerate(order):
            results.append(
                FeatureImportance(
                    name=feature_names[idx],
                    importance=float(scores[idx]),
                    rank=rank + 1,
                )
            )

        return results

    def _shap_importance(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Optional[np.ndarray]:
        """Out-of-fold SHAP TreeExplainer importance via LightGBM."""
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "verbose": -1,
            "num_threads": 1,
        }
        params = {**default_params, **(self.lgb_params or {})}
        n = len(y)
        n_folds = 3
        shap_accum = np.zeros(X.shape[1])
        total_samples = 0

        initial_train = max(20, int(0.4 * n))
        remaining = n - initial_train
        fold_size = max(5, remaining // n_folds)

        for fold in range(n_folds):
            val_start = initial_train + fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            if val_start >= n:
                break
            train_idx = np.arange(val_start)
            val_idx = np.arange(val_start, min(val_end, n))

            if len(train_idx) < 20 or len(val_idx) < 5:
                continue

            train_data = lgb.Dataset(
                X[train_idx], label=y[train_idx], feature_name=feature_names
            )
            model = lgb.train(
                params, train_data, num_boost_round=200,
                callbacks=[lgb.log_evaluation(period=0)],
            )
            explainer = shap.TreeExplainer(model)
            fold_shap = explainer.shap_values(X[val_idx])
            if isinstance(fold_shap, list):
                fold_shap = fold_shap[1]
            shap_accum += np.sum(np.abs(fold_shap), axis=0)
            total_samples += len(val_idx)

        if total_samples == 0:
            return None
        return shap_accum / total_samples

    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Out-of-fold permutation importance using LightGBM (not logistic regression).

        FIX #S4 (from audit): Uses LightGBM (same model class as downstream)
        instead of LogisticRegression, so importance reflects the actual
        model's sensitivity to each feature.
        """
        if not LIGHTGBM_AVAILABLE:
            # Fallback to logistic regression if LightGBM unavailable
            return self._permutation_importance_logistic(X, y)

        n = len(y)
        n_folds = 3
        perm_accum = np.zeros(X.shape[1])
        total_folds = 0

        initial_train = max(20, int(0.4 * n))
        remaining = n - initial_train
        fold_size = max(5, remaining // n_folds)

        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
            "num_threads": 1,
        }
        params = {**default_params, **(self.lgb_params or {})}

        for fold in range(n_folds):
            val_start = initial_train + fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            if val_start >= n:
                break
            train_idx = np.arange(val_start)
            val_idx = np.arange(val_start, min(val_end, n))

            if len(train_idx) < 20 or len(val_idx) < 5:
                continue

            # Train LightGBM model for this fold
            train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
            model = lgb.train(
                params, train_data, num_boost_round=200,
                callbacks=[lgb.log_evaluation(period=0)],
            )

            # Wrap for sklearn permutation_importance
            class _LGBWrapper:
                """Minimal sklearn-compatible wrapper for LightGBM Booster."""
                def __init__(self, booster):
                    self._booster = booster
                def fit(self, X_input, y_input=None, **kwargs):
                    return self  # Already trained; no-op for sklearn compat
                def predict(self, X_input):
                    return self._booster.predict(X_input)

            wrapper = _LGBWrapper(model)
            # Use neg_brier_score as permutation metric
            from sklearn.metrics import make_scorer, brier_score_loss
            def _neg_brier(y_true, y_pred):
                return -brier_score_loss(y_true, y_pred)
            scorer = make_scorer(_neg_brier, greater_is_better=True, response_method="predict")

            result = permutation_importance(
                wrapper, X[val_idx], y[val_idx],
                n_repeats=10, random_state=self.random_seed,
                scoring=scorer,
            )
            perm_accum += result.importances_mean
            total_folds += 1

        if total_folds == 0:
            return None
        return perm_accum / total_folds

    def _permutation_importance_logistic(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: permutation importance with LogisticRegression."""
        if not SKLEARN_AVAILABLE:
            return None

        n = len(y)
        n_folds = 3
        perm_accum = np.zeros(X.shape[1])
        total_folds = 0

        initial_train = max(20, int(0.4 * n))
        remaining = n - initial_train
        fold_size = max(5, remaining // n_folds)

        for fold in range(n_folds):
            val_start = initial_train + fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            if val_start >= n:
                break
            train_idx = np.arange(val_start)
            val_idx = np.arange(val_start, min(val_end, n))

            if len(train_idx) < 20 or len(val_idx) < 5:
                continue

            model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
            model.fit(X[train_idx], y[train_idx])
            result = permutation_importance(
                model, X[val_idx], y[val_idx],
                n_repeats=10, random_state=self.random_seed,
                scoring="neg_brier_score",
            )
            perm_accum += result.importances_mean
            total_folds += 1

        if total_folds == 0:
            return None
        return perm_accum / total_folds

    def _correlation_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Absolute Pearson correlation with target."""
        n_features = X.shape[1]
        corrs = np.zeros(n_features)
        for i in range(n_features):
            std = np.std(X[:, i])
            if std > 1e-10:
                r = np.corrcoef(X[:, i], y)[0, 1]
                corrs[i] = 0.0 if np.isnan(r) else abs(r)
        return corrs

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        mn = np.min(scores)
        mx = np.max(scores)
        if mx - mn < 1e-10:
            return np.ones_like(scores) * 0.5
        return (scores - mn) / (mx - mn)


# FIX 2.2: Known feature clusters where multiple features measure the
# same underlying construct.  For each cluster, only the representatives
# (first list entry) survive pre-selection; the rest are dropped before
# VIF/correlation pruning.  This reduces dimensionality by ~8 features
# and prevents the VIF pruner from hitting its max_drops limit.
_FEATURE_CLUSTERS = {
    # SOS cluster: keep general SOS + elite SOS (tournament-specific)
    "sos": {
        "keep": ["sos_adj_em", "elite_sos"],
        "drop": ["sos_opp_o", "sos_opp_d", "ncsos_adj_em"],
    },
    # RAPM cluster: keep top5+bench decomposition (what matters),
    # drop total (≈ top5+bench) and positional (≈ top5+bench reweighted)
    "rapm": {
        "keep": ["top5_rapm", "bench_rapm"],
        "drop": ["total_rapm", "backcourt_rapm", "frontcourt_rapm"],
    },
}


def _cluster_preselect(
    X: np.ndarray,
    feature_names: List[str],
) -> tuple:
    """Drop known-redundant features from correlated clusters.

    FIX 2.2: Pre-selects one representative from each known cluster
    to reduce dimensionality before automated selection.

    Returns:
        (X_reduced, names_reduced) with cluster members dropped.
    """
    names_list = list(feature_names)
    to_drop_names = set()
    for cluster_name, spec in _FEATURE_CLUSTERS.items():
        for drop_name in spec["drop"]:
            if drop_name in names_list:
                to_drop_names.add(drop_name)

    if not to_drop_names:
        return X, names_list

    keep_mask = [name not in to_drop_names for name in names_list]
    kept_names = [name for name in names_list if name not in to_drop_names]
    X_kept = X[:, keep_mask]

    logger.info(
        "FIX 2.2: Cluster pre-selection dropped %d redundant features: %s",
        len(to_drop_names), sorted(to_drop_names),
    )
    return X_kept, kept_names


def detect_importance_elbow(
    importance_scores: List[float],
    sensitivity: float = 1.0,
) -> float:
    """Detect the natural threshold in sorted importance scores using elbow/knee detection.

    Uses the Kneedle algorithm (Satopaa et al., 2011): finds the point of
    maximum curvature in the sorted importance curve. This is where the
    marginal value of adding another feature drops sharply.

    Args:
        importance_scores: Importance values (will be sorted descending internally)
        sensitivity: Kneedle sensitivity parameter (higher = more conservative)

    Returns:
        Detected importance threshold, or 0.05 as fallback
    """
    if len(importance_scores) < 4:
        return 0.05

    sorted_scores = sorted(importance_scores, reverse=True)
    n = len(sorted_scores)

    # Normalize to [0, 1] range for both axes
    x = np.linspace(0, 1, n)
    y_min, y_max = sorted_scores[-1], sorted_scores[0]
    if y_max - y_min < 1e-10:
        return 0.05
    y = np.array([(s - y_min) / (y_max - y_min) for s in sorted_scores])

    # Line from first to last point
    # Distance of each point from this line = curvature proxy
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    line_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if line_len < 1e-10:
        return 0.05

    # Perpendicular distance from each point to the line
    distances = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / line_len

    # Apply sensitivity: scale distances
    distances *= sensitivity

    # Find the elbow (maximum distance point)
    elbow_idx = int(np.argmax(distances))

    # The threshold is the importance value at the elbow
    threshold = sorted_scores[elbow_idx]

    # Sanity: threshold should be between 0.01 and 0.20
    threshold = max(0.01, min(0.20, threshold))

    logger.info(
        "Elbow detection: threshold=%.4f at position %d/%d (sensitivity=%.1f)",
        threshold, elbow_idx, n, sensitivity,
    )
    return threshold


class FeatureSelector:
    """
    Orchestrates the full feature selection pipeline:

    0. Redundancy audit (diagnostic — reports all |r| > threshold pairs)
    1. DoF budget enforcement (Harrell's rule caps max_features)
    2. VIF pruning (FIX #3: enabled by default)
    3. Correlation pruning (FIX #5: target-correlation tie-breaking)
    4. Importance calculation (FIX #7: no correlation method when SHAP available)
    5. Keep top-k features by importance (capped by DoF budget)
    6. Bootstrap stability filter (FIX #6: features must survive 80% of runs)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        min_features: int = 20,
        max_features: int = 50,
        importance_threshold: float = 0.05,
        random_seed: int = 42,
        enable_vif_pruning: bool = True,  # FIX #3: enabled by default
        vif_threshold: float = 10.0,
        enable_stability_filter: bool = True,  # FIX #6
        stability_threshold: float = 0.80,  # FIX #6: must be selected in 80% of runs
        n_bootstrap: int = 10,  # FIX #6: number of bootstrap iterations
        enable_dof_budget: bool = True,  # Harrell's rule enforcement
        events_per_predictor: float = 15.0,  # Harrell's EPP (10-20 range)
        enable_mi_screening: bool = True,
        mi_threshold: float = 0.01,
        enable_shift_gating: bool = False,
        shift_drop_mode: str = "downweight",
        adaptive_threshold: bool = True,
        importance_lgb_params: Optional[Dict] = None,  # Solution 9
    ):
        self.correlation_threshold = correlation_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.importance_threshold = importance_threshold
        self.random_seed = random_seed
        self.enable_vif_pruning = enable_vif_pruning
        self.vif_threshold = vif_threshold
        self.enable_stability_filter = enable_stability_filter
        self.stability_threshold = stability_threshold
        self.n_bootstrap = n_bootstrap
        self.enable_dof_budget = enable_dof_budget
        self.events_per_predictor = events_per_predictor
        self.enable_mi_screening = enable_mi_screening
        self.mi_threshold = mi_threshold
        self.enable_shift_gating = enable_shift_gating
        self.shift_drop_mode = shift_drop_mode
        self.adaptive_threshold = adaptive_threshold
        self.importance_lgb_params = importance_lgb_params

        self._selected_indices: Optional[List[int]] = None
        self._selected_names: Optional[List[str]] = None
        self._dof_budget_result = None
        self._redundancy_audit_result = None

    @property
    def is_fitted(self) -> bool:
        return self._selected_indices is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> FeatureSelectionResult:
        """
        Fit the feature selector.

        FIX #3: VIF pruning enabled by default.
        FIX #5: Correlation pruner uses target correlation for tie-breaking.
        FIX #6: Bootstrap stability filter removes unstable features.
        FIX #7: Correlation importance suppressed when SHAP available.
        FIX 2.2: Cluster pre-selection collapses known correlated groups.
        SA-1: Systematic redundancy audit (diagnostic, all |r| > threshold).
        SA-2: DoF budget enforcement (Harrell's rule caps max_features).

        LEAKAGE NOTE: This method must be called with TRAINING data only.
        """
        original_dim = X.shape[1]
        original_name_to_idx = {name: i for i, name in enumerate(feature_names)}

        # Step SA-0: Systematic redundancy audit (diagnostic only).
        # Reports all pairwise correlations above threshold and eigenvalue
        # analysis.  Does NOT modify the feature matrix — downstream
        # pruners (VIF, correlation) handle the actual removal.
        try:
            from .statistical_audit import SystematicRedundancyAuditor
            auditor = SystematicRedundancyAuditor(
                correlation_threshold=self.correlation_threshold,
            )
            self._redundancy_audit_result = auditor.audit(X, feature_names)
        except Exception as e:
            logger.warning("Redundancy audit failed (non-fatal): %s", e)
            self._redundancy_audit_result = None

        # Step SA-1: DoF budget enforcement (Harrell's rule).
        # Caps max_features to what the sample size can support.
        if self.enable_dof_budget:
            try:
                from .statistical_audit import compute_dof_budget
                n_positive = int(np.sum(y > 0))
                budget = compute_dof_budget(
                    n_samples=len(y),
                    n_positive=n_positive,
                    n_features=original_dim,
                    events_per_predictor=self.events_per_predictor,
                )
                self._dof_budget_result = budget
                if budget.recommended_max_features < self.max_features:
                    logger.info(
                        "DoF budget capping max_features: %d -> %d "
                        "(Harrell's rule: %d events / %.0f EPP)",
                        self.max_features, budget.recommended_max_features,
                        budget.n_events, self.events_per_predictor,
                    )
                    self.max_features = budget.recommended_max_features
                    # Ensure min_features doesn't exceed max_features
                    if self.min_features > self.max_features:
                        logger.info(
                            "DoF budget also capping min_features: %d -> %d",
                            self.min_features, self.max_features,
                        )
                        self.min_features = self.max_features
            except Exception as e:
                logger.warning("DoF budget computation failed (non-fatal): %s", e)
                self._dof_budget_result = None

        # Step -2 (FIX 2.2): Cluster pre-selection.
        # Known feature clusters that are highly correlated within the
        # tournament-eligible population.  Keep one representative from
        # each cluster to reduce dimensionality before automated selection.
        # This prevents VIF/correlation pruner from hitting max_drops
        # limits before resolving all multicollinearity chains.
        X, feature_names = _cluster_preselect(X, feature_names)

        # Step -1: Near-zero variance pruning (runs before VIF to avoid
        # numerical issues with constant/near-constant features)
        variance_pruner = NearZeroVariancePruner()
        X, feature_names, variance_dropped = variance_pruner.prune(X, feature_names)

        # Step 0: VIF pruning (FIX #3: now enabled by default)
        vif_dropped: List[str] = []
        if self.enable_vif_pruning:
            vif_pruner = VIFPruner(threshold=self.vif_threshold)
            X, feature_names, vif_dropped = vif_pruner.prune(X, feature_names)
            if vif_dropped:
                logger.info("VIF pruning removed %d features: %s",
                            len(vif_dropped), vif_dropped)

        # Step 1: Correlation pruning (FIX #5: pass y for target-correlation tie-breaking)
        pruner = CorrelationPruner(threshold=self.correlation_threshold)
        X_pruned, kept_names, corr_dropped = pruner.prune(X, feature_names, y=y)

        # Step 1.5: Mutual information screening (non-linear associations)
        mi_dropped: List[str] = []
        mi_scores_raw = None
        mi_pre_screen_names: List[str] = list(kept_names)
        if self.enable_mi_screening and SKLEARN_AVAILABLE:
            mi_screener = MutualInformationScreener(
                threshold=self.mi_threshold, random_seed=self.random_seed,
            )
            X_pruned, kept_names, mi_dropped, mi_scores_raw = mi_screener.screen(
                X_pruned, y, kept_names,
            )

        # Step 2: Importance calculation (FIX #7: correlation suppressed when SHAP available)
        calculator = ImportanceCalculator(
            random_seed=self.random_seed,
            lgb_params=self.importance_lgb_params,
        )
        importances = calculator.calculate(X_pruned, y, kept_names)

        # Step 2.5: Adaptive threshold via elbow detection
        effective_threshold = self.importance_threshold
        detected_threshold = None
        if self.adaptive_threshold and len(importances) >= 4:
            scores_list = [imp.importance for imp in importances]
            detected = detect_importance_elbow(scores_list)
            if detected != 0.05:
                logger.info(
                    "Adaptive threshold: %.4f (detected) vs %.4f (default)",
                    detected, self.importance_threshold,
                )
                effective_threshold = detected
                detected_threshold = detected

        # Step 3: Select features above importance threshold, up to max
        selected = []
        low_importance_dropped = []

        for imp in importances:
            if len(selected) >= self.max_features:
                low_importance_dropped.append(imp.name)
            elif imp.importance < effective_threshold and len(selected) >= self.min_features:
                low_importance_dropped.append(imp.name)
            else:
                selected.append(imp.name)

        # Step 4 (FIX #6): Bootstrap stability filter
        stability_scores = None
        if (self.enable_stability_filter
                and len(y) >= 60
                and self.n_bootstrap >= 3
                and len(selected) > self.min_features):
            stability_scores = self._bootstrap_stability(
                X_pruned, y, kept_names, selected, calculator
            )
            # Remove features with stability below threshold, but keep at least min_features
            stable_selected = [
                name for name in selected
                if stability_scores.get(name, 0.0) >= self.stability_threshold
            ]
            if len(stable_selected) >= self.min_features:
                removed = set(selected) - set(stable_selected)
                if removed:
                    logger.info(
                        "FIX#6 stability filter removed %d features: %s "
                        "(selected in <%.0f%% of bootstrap runs)",
                        len(removed), sorted(removed), self.stability_threshold * 100,
                    )
                    low_importance_dropped.extend(sorted(removed))
                selected = stable_selected
            else:
                logger.info(
                    "FIX#6 stability filter would remove too many features "
                    "(%d → %d < min %d); skipping.",
                    len(selected), len(stable_selected), self.min_features,
                )

        # Map selected names back to ORIGINAL indices
        selected_indices = [original_name_to_idx[name] for name in selected if name in original_name_to_idx]
        all_dropped = variance_dropped + vif_dropped + corr_dropped + mi_dropped + low_importance_dropped

        self._selected_indices = selected_indices
        self._selected_names = selected

        # Post-selection multicollinearity validation: verify the final
        # feature set is clean.  This is especially important for
        # LogisticRegression where collinearity inflates coefficients.
        X_final = X_pruned[:, [kept_names.index(n) for n in selected if n in kept_names]]
        cond_num, max_vif, mc_warning = validate_post_selection_collinearity(
            X_final, selected,
            vif_threshold=self.vif_threshold,
        )

        # Populate redundancy/DoF diagnostics
        redundancy_pairs = 0
        eff_rank = None
        if self._redundancy_audit_result is not None:
            redundancy_pairs = self._redundancy_audit_result.n_above_threshold
            eff_rank = self._redundancy_audit_result.effective_rank

        dof_max = None
        dof_exceeded = None
        if self._dof_budget_result is not None:
            dof_max = self._dof_budget_result.recommended_max_features
            dof_exceeded = self._dof_budget_result.budget_exceeded

        # Build MI scores dict for diagnostics
        mi_scores_dict = None
        if mi_scores_raw is not None and len(mi_pre_screen_names) == len(mi_scores_raw):
            mi_scores_dict = {
                name: float(mi_scores_raw[i])
                for i, name in enumerate(mi_pre_screen_names)
            }

        return FeatureSelectionResult(
            selected_features=selected,
            selected_indices=selected_indices,
            dropped_features=all_dropped,
            importance_scores=importances,
            correlation_dropped=corr_dropped,
            low_importance_dropped=low_importance_dropped,
            original_dim=original_dim,
            reduced_dim=len(selected),
            method="redundancy_audit+dof_budget+variance+vif+correlation_pruning+mi_screening+importance_ranking+elbow_detection+stability_filter",
            stability_scores=stability_scores,
            variance_dropped=variance_dropped,
            post_selection_condition_number=cond_num,
            post_selection_max_vif=max_vif,
            multicollinearity_warning=mc_warning,
            redundancy_pairs_found=redundancy_pairs,
            effective_rank=eff_rank,
            dof_budget_max_features=dof_max,
            dof_budget_exceeded=dof_exceeded,
            mi_dropped=mi_dropped,
            mi_scores=mi_scores_dict,
            shifted_features=[],
            detected_importance_threshold=detected_threshold,
        )

    def _bootstrap_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        candidate_features: List[str],
        calculator: ImportanceCalculator,
    ) -> Dict[str, float]:
        """
        FIX #6: Bootstrap stability analysis.

        Run importance ranking on `n_bootstrap` resampled datasets.
        Return the fraction of runs in which each feature would be selected.
        Features selected in <80% of runs are unstable and likely overfit
        to training data idiosyncrasies.

        Args:
            X: Post-pruning feature matrix
            y: Training labels
            feature_names: Post-pruning feature names
            candidate_features: Features that passed importance threshold
            calculator: ImportanceCalculator instance

        Returns:
            Dict of feature_name -> stability_fraction [0.0, 1.0]
        """
        rng = np.random.default_rng(self.random_seed)
        n = len(y)
        selection_counts: Dict[str, int] = {name: 0 for name in candidate_features}

        for boot_iter in range(self.n_bootstrap):
            # Bootstrap resample (with replacement)
            boot_idx = rng.choice(n, size=n, replace=True)
            boot_X = X[boot_idx]
            boot_y = y[boot_idx]

            # Re-run importance calculation on bootstrap sample
            try:
                boot_importances = calculator.calculate(boot_X, boot_y, feature_names)
            except Exception:
                continue

            # Select top features using same criteria
            boot_selected = set()
            for imp in boot_importances:
                if len(boot_selected) >= self.max_features:
                    break
                if imp.importance < self.importance_threshold and len(boot_selected) >= self.min_features:
                    break
                boot_selected.add(imp.name)

            # Count how many times each candidate is selected
            for name in candidate_features:
                if name in boot_selected:
                    selection_counts[name] += 1

        # Convert to fractions
        effective_runs = max(self.n_bootstrap, 1)
        stability = {
            name: count / effective_runs
            for name, count in selection_counts.items()
        }

        logger.info(
            "FIX#6 bootstrap stability (%d runs): %d/%d features stable (>=%.0f%%)",
            effective_runs,
            sum(1 for v in stability.values() if v >= self.stability_threshold),
            len(candidate_features),
            self.stability_threshold * 100,
        )

        return stability

    def _apply_shift_gating(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        feature_names: List[str],
        importance_scores: List[FeatureImportance],
        shift_drop_mode: str = "downweight",
    ) -> List[FeatureImportance]:
        """Gate features based on distribution shift between train and validation.

        Closes the shift-detection loop: features with significant distribution
        shift have their importance scores down-weighted or are dropped entirely.

        Args:
            X_train: Training feature matrix
            X_val: Validation feature matrix
            feature_names: Feature names matching columns of X_train/X_val
            importance_scores: Current importance rankings
            shift_drop_mode: "downweight" (reduce importance) or "drop" (zero importance)

        Returns:
            Updated importance scores with shift-gated features penalized
        """
        shift_results = detect_distribution_shift(
            X_train, X_val, feature_names,
            psi_threshold=0.25, ks_alpha=0.05, mean_shift_threshold=1.0,
        )

        shifted_features = set()
        shift_severities = {}
        for result in shift_results:
            # Require BOTH PSI and KS to agree (reduces false positives)
            if result.psi > 0.25 and result.ks_pvalue < 0.05:
                shifted_features.add(result.feature_name)
                shift_severities[result.feature_name] = result.psi

        if not shifted_features:
            return importance_scores

        updated = []
        for imp in importance_scores:
            if imp.name in shifted_features:
                if shift_drop_mode == "drop":
                    updated.append(FeatureImportance(
                        name=imp.name, importance=0.0, rank=imp.rank,
                    ))
                else:
                    # Downweight by PSI severity: decay = exp(-PSI)
                    psi = shift_severities.get(imp.name, 0.25)
                    import math
                    decay = math.exp(-psi)
                    updated.append(FeatureImportance(
                        name=imp.name,
                        importance=imp.importance * decay,
                        rank=imp.rank,
                    ))
            else:
                updated.append(imp)

        # Re-rank
        updated.sort(key=lambda x: x.importance, reverse=True)
        for i, imp in enumerate(updated):
            imp.rank = i + 1

        logger.info(
            "Shift gating (%s mode): penalized %d features: %s",
            shift_drop_mode, len(shifted_features), sorted(shifted_features),
        )
        return updated

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted selection to new data."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector not fitted. Call fit() first.")
        return X[:, self._selected_indices]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, FeatureSelectionResult]:
        """Fit and transform in one step."""
        result = self.fit(X, y, feature_names)
        return self.transform(X), result

    def get_selected_names(self) -> List[str]:
        """Return selected feature names after fitting."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector not fitted.")
        return list(self._selected_names)


# ---------------------------------------------------------------------------
# Solution 12: Cross-fold stability metrics
# ---------------------------------------------------------------------------

@dataclass
class CrossFoldStabilityResult:
    """Stability of feature selection across LOYO folds."""
    feature_fold_counts: Dict[str, int]  # feature_name → n_folds_selected_in
    n_folds: int
    mean_jaccard: float  # Average pairwise Jaccard index
    pairwise_jaccard: Dict[str, float]  # "fold_i_vs_fold_j" → Jaccard
    stable_features: List[str]  # Selected in >= threshold folds
    unstable_features: List[str]  # Selected in < threshold folds
    stability_threshold: int  # Min folds for "stable"
    is_stable: bool  # True if mean_jaccard >= 0.7


def compute_cross_fold_stability(
    fold_selected_features: List[List[str]],
    stability_threshold_frac: float = 0.7,
) -> CrossFoldStabilityResult:
    """Compute feature selection stability across LOYO folds.

    Solution 12: Bootstrap stability (FIX #6) tests within a single training
    set. Cross-fold stability tests whether the same features are selected
    when the training set changes (different year held out). If a feature
    is selected for fold 2018 but dropped for fold 2022, it's unreliable.

    Args:
        fold_selected_features: List of selected feature name lists, one per fold
        stability_threshold_frac: Fraction of folds a feature must appear in

    Returns:
        CrossFoldStabilityResult with per-feature counts and Jaccard indices
    """
    n_folds = len(fold_selected_features)
    if n_folds < 2:
        all_feats = fold_selected_features[0] if fold_selected_features else []
        return CrossFoldStabilityResult(
            feature_fold_counts={f: 1 for f in all_feats},
            n_folds=n_folds,
            mean_jaccard=1.0,
            pairwise_jaccard={},
            stable_features=all_feats,
            unstable_features=[],
            stability_threshold=1,
            is_stable=True,
        )

    # Count how many folds each feature appears in
    all_features = set()
    fold_sets = []
    for features in fold_selected_features:
        s = set(features)
        fold_sets.append(s)
        all_features.update(s)

    feature_counts = {}
    for feat in sorted(all_features):
        feature_counts[feat] = sum(1 for s in fold_sets if feat in s)

    # Pairwise Jaccard indices
    pairwise_jaccard = {}
    jaccard_values = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            intersection = len(fold_sets[i] & fold_sets[j])
            union = len(fold_sets[i] | fold_sets[j])
            jaccard = intersection / union if union > 0 else 0.0
            key = f"fold_{i}_vs_fold_{j}"
            pairwise_jaccard[key] = jaccard
            jaccard_values.append(jaccard)

    mean_jaccard = float(np.mean(jaccard_values)) if jaccard_values else 0.0

    # Classify features as stable/unstable
    min_folds = max(1, int(n_folds * stability_threshold_frac))
    stable = [f for f, c in feature_counts.items() if c >= min_folds]
    unstable = [f for f, c in feature_counts.items() if c < min_folds]

    is_stable = mean_jaccard >= 0.7

    if not is_stable:
        logger.warning(
            "Feature selection unstable across LOYO folds: "
            "mean Jaccard=%.3f (threshold: 0.7), %d/%d features unstable",
            mean_jaccard, len(unstable), len(all_features),
        )
    else:
        logger.info(
            "Feature selection stable: mean Jaccard=%.3f, %d/%d features stable (>=%d/%d folds)",
            mean_jaccard, len(stable), len(all_features), min_folds, n_folds,
        )

    return CrossFoldStabilityResult(
        feature_fold_counts=feature_counts,
        n_folds=n_folds,
        mean_jaccard=mean_jaccard,
        pairwise_jaccard=pairwise_jaccard,
        stable_features=sorted(stable),
        unstable_features=sorted(unstable),
        stability_threshold=min_folds,
        is_stable=is_stable,
    )
