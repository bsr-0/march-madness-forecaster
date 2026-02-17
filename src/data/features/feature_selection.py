"""
Feature selection and dimensionality reduction pipeline.

Provides:
- CorrelationPruner: drops highly correlated features
- PermutationImportanceSelector: sklearn permutation importance
- LightGBMImportanceSelector: native gain-based importance
- FeatureSelector: orchestrates the full selection pipeline
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


class CorrelationPruner:
    """
    Drops features with pairwise correlation above a threshold.

    When two features are highly correlated, keeps the one with higher
    variance (more information content).
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def prune(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Remove highly correlated features.

        Args:
            X: Feature matrix [N, D]
            feature_names: Names of features

        Returns:
            Tuple of (pruned_X, kept_names, dropped_names)
        """
        n_features = X.shape[1]
        if n_features <= 1:
            return X, list(feature_names), []

        # Compute correlation matrix
        # Handle constant columns (zero std)
        stds = np.std(X, axis=0)
        constant_mask = stds < 1e-10
        X_safe = X.copy()
        X_safe[:, constant_mask] = np.random.randn(X.shape[0], int(np.sum(constant_mask))) * 1e-6

        corr = np.corrcoef(X_safe.T)
        # Replace NaN from degenerate columns
        corr = np.nan_to_num(corr, nan=0.0)

        variances = np.var(X, axis=0)
        to_drop = set()

        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > self.threshold:
                    # Drop the one with lower variance
                    if variances[i] >= variances[j]:
                        to_drop.add(j)
                    else:
                        to_drop.add(i)
                        break  # i is dropped, move on

        kept_indices = [i for i in range(n_features) if i not in to_drop]
        dropped_names = [feature_names[i] for i in sorted(to_drop)]
        kept_names = [feature_names[i] for i in kept_indices]

        return X[:, kept_indices], kept_names, dropped_names


class ImportanceCalculator:
    """
    Calculates feature importance using multiple methods and combines them.

    Priority order (highest weight first):
    1. SHAP TreeExplainer (if shap + LightGBM available) — theoretically
       grounded Shapley values, exact for tree models, no bias toward
       high-cardinality features like LightGBM gain importance.
    2. Permutation importance with logistic regression (sklearn)
    3. Absolute correlation with target (always available)

    SHAP is weighted 2x relative to permutation and correlation to reflect
    its stronger theoretical grounding.
    """

    # Relative weights for each method (un-normalized)
    METHOD_WEIGHTS = {
        "shap": 2.0,
        "permutation": 1.0,
        "correlation": 1.0,
    }

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def calculate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureImportance]:
        """
        Calculate combined feature importance.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            feature_names: Feature names

        Returns:
            List of FeatureImportance, sorted by importance descending
        """
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        total_weight = 0.0

        # Method 1 (primary): SHAP TreeExplainer via LightGBM
        if SHAP_AVAILABLE and LIGHTGBM_AVAILABLE:
            try:
                shap_scores = self._shap_importance(X, y, feature_names)
                if shap_scores is not None:
                    w = self.METHOD_WEIGHTS["shap"]
                    scores += w * self._normalize(shap_scores)
                    total_weight += w
            except Exception as e:
                logger.warning("SHAP importance failed: %s", e)

        # Method 2: Permutation importance
        if SKLEARN_AVAILABLE:
            try:
                perm_scores = self._permutation_importance(X, y)
                if perm_scores is not None:
                    w = self.METHOD_WEIGHTS["permutation"]
                    scores += w * self._normalize(perm_scores)
                    total_weight += w
            except Exception as e:
                logger.warning("Permutation importance failed: %s", e)

        # Method 3: Absolute correlation with target
        corr_scores = self._correlation_importance(X, y)
        w = self.METHOD_WEIGHTS["correlation"]
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
        """SHAP TreeExplainer importance via LightGBM."""
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "verbose": -1,
        }
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        model = lgb.train(
            params, train_data, num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classification LightGBM, shap_values may be a list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        return np.mean(np.abs(shap_values), axis=0)

    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """Permutation importance using logistic regression."""
        model = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        model.fit(X, y)
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=self.random_seed, scoring="neg_brier_score"
        )
        return result.importances_mean

    def _correlation_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Absolute Pearson correlation with target."""
        n_features = X.shape[1]
        corrs = np.zeros(n_features)
        for i in range(n_features):
            std = np.std(X[:, i])
            if std > 1e-10:
                corrs[i] = abs(np.corrcoef(X[:, i], y)[0, 1])
        return corrs

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        mn = np.min(scores)
        mx = np.max(scores)
        if mx - mn < 1e-10:
            return np.ones_like(scores) * 0.5
        return (scores - mn) / (mx - mn)


class FeatureSelector:
    """
    Orchestrates the full feature selection pipeline:

    1. Correlation pruning (drop r > threshold)
    2. Importance calculation (LightGBM gain + permutation + correlation)
    3. Keep top-k features by importance

    The result is a reduced feature set that mitigates overfitting.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        min_features: int = 20,
        max_features: int = 50,
        importance_threshold: float = 0.05,
        random_seed: int = 42,
    ):
        self.correlation_threshold = correlation_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.importance_threshold = importance_threshold
        self.random_seed = random_seed

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
        Fit the feature selector: prune correlations, rank importance, select top-k.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            feature_names: Feature names

        Returns:
            FeatureSelectionResult with selected/dropped features
        """
        original_dim = X.shape[1]

        # Step 1: Correlation pruning
        pruner = CorrelationPruner(threshold=self.correlation_threshold)
        X_pruned, kept_names, corr_dropped = pruner.prune(X, feature_names)

        # Step 2: Importance calculation on pruned features
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

        # Map selected names back to original indices
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        selected_indices = [name_to_idx[name] for name in selected if name in name_to_idx]
        all_dropped = corr_dropped + low_importance_dropped

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
            method="correlation_pruning+importance_ranking",
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted selection to new data.

        Args:
            X: Feature matrix [N, D]

        Returns:
            Reduced feature matrix [N, D']
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector not fitted. Call fit() first.")
        return X[:, self._selected_indices]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, FeatureSelectionResult]:
        """
        Fit and transform in one step.

        Returns:
            Tuple of (reduced_X, selection_result)
        """
        result = self.fit(X, y, feature_names)
        return self.transform(X), result

    def get_selected_names(self) -> List[str]:
        """Return selected feature names after fitting."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector not fitted.")
        return list(self._selected_names)
