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

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

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
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "verbose": -1,
            "num_threads": 1,
        }
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
                params, train_data, num_boost_round=100,
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

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
            "num_threads": 1,
        }

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
                params, train_data, num_boost_round=100,
                callbacks=[lgb.log_evaluation(period=0)],
            )

            # Wrap for sklearn permutation_importance
            class _LGBWrapper:
                """Minimal sklearn-compatible wrapper for LightGBM Booster."""
                def __init__(self, booster):
                    self._booster = booster
                def predict(self, X_input):
                    return self._booster.predict(X_input)

            wrapper = _LGBWrapper(model)
            # Use neg_brier_score as permutation metric
            from sklearn.metrics import make_scorer, brier_score_loss
            def _neg_brier(y_true, y_pred):
                return -brier_score_loss(y_true, y_pred)
            scorer = make_scorer(_neg_brier, needs_proba=False, greater_is_better=True)

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


class FeatureSelector:
    """
    Orchestrates the full feature selection pipeline:

    1. VIF pruning (FIX #3: enabled by default)
    2. Correlation pruning (FIX #5: target-correlation tie-breaking)
    3. Importance calculation (FIX #7: no correlation method when SHAP available)
    4. Keep top-k features by importance
    5. Bootstrap stability filter (FIX #6: features must survive 80% of runs)
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

        self._selected_indices: Optional[List[int]] = None
        self._selected_names: Optional[List[str]] = None

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

        LEAKAGE NOTE: This method must be called with TRAINING data only.
        """
        original_dim = X.shape[1]
        original_name_to_idx = {name: i for i, name in enumerate(feature_names)}

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

        # Step 2: Importance calculation (FIX #7: correlation suppressed when SHAP available)
        calculator = ImportanceCalculator(random_seed=self.random_seed)
        importances = calculator.calculate(X_pruned, y, kept_names)

        # Step 3: Select features above importance threshold, up to max
        selected = []
        low_importance_dropped = []

        for imp in importances:
            if len(selected) >= self.max_features:
                low_importance_dropped.append(imp.name)
            elif imp.importance < self.importance_threshold and len(selected) >= self.min_features:
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
        all_dropped = vif_dropped + corr_dropped + low_importance_dropped

        self._selected_indices = selected_indices
        self._selected_names = selected

        return FeatureSelectionResult(
            selected_features=selected,
            selected_indices=selected_indices,
            dropped_features=all_dropped,
            importance_scores=importances,
            correlation_dropped=corr_dropped,
            low_importance_dropped=low_importance_dropped,
            original_dim=original_dim,
            reduced_dim=len(selected),
            method="vif+correlation_pruning+importance_ranking+stability_filter",
            stability_scores=stability_scores,
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
