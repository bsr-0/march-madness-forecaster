"""Feature explainability for matchup predictions.

Provides:
1. MatchupExplainer: Per-prediction SHAP-based feature attribution
2. LOYOFeatureImportance: Cross-validated feature importance stable across folds
3. PerRoundFeatureImportance: Feature importance stratified by tournament round
4. FeatureContribution: Dataclass for per-feature contributions to a prediction

SHAP is soft-imported — all functions degrade gracefully when unavailable,
falling back to permutation importance or LightGBM native importance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """A single feature's contribution to a matchup prediction."""
    name: str
    value: float  # Feature value in this matchup
    contribution: float  # SHAP value or importance-weighted contribution
    rank: int = 0


@dataclass
class MatchupExplanation:
    """Full explanation for a single matchup prediction."""
    team1: str
    team2: str
    predicted_probability: float
    base_value: float  # Expected value (model baseline)
    contributions: List[FeatureContribution]
    method: str  # "shap", "permutation", or "native_importance"
    top_positive: List[FeatureContribution] = field(default_factory=list)
    top_negative: List[FeatureContribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team1": self.team1,
            "team2": self.team2,
            "predicted_probability": self.predicted_probability,
            "base_value": self.base_value,
            "method": self.method,
            "top_positive": [
                {"name": c.name, "value": c.value, "contribution": c.contribution}
                for c in self.top_positive
            ],
            "top_negative": [
                {"name": c.name, "value": c.value, "contribution": c.contribution}
                for c in self.top_negative
            ],
        }


@dataclass
class FeatureImportanceSummary:
    """Aggregated feature importance across LOYO folds."""
    name: str
    mean_importance: float
    std_importance: float
    stability: float  # Fraction of folds where feature was in top-K
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean_importance": self.mean_importance,
            "std_importance": self.std_importance,
            "stability": self.stability,
            "rank": self.rank,
        }


@dataclass
class PerRoundImportance:
    """Feature importance for a specific tournament round."""
    round_name: str
    n_games: int
    importances: List[FeatureImportanceSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_name": self.round_name,
            "n_games": self.n_games,
            "top_10": [imp.to_dict() for imp in self.importances[:10]],
        }


@dataclass
class LOYOFeatureImportanceReport:
    """Full LOYO-aggregated feature importance report."""
    global_importances: List[FeatureImportanceSummary]
    per_fold_importances: Dict[int, List[FeatureImportanceSummary]]
    per_round_importances: Dict[str, PerRoundImportance]
    method: str
    n_folds: int
    n_features: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_folds": self.n_folds,
            "n_features": self.n_features,
            "global_top_20": [imp.to_dict() for imp in self.global_importances[:20]],
            "per_round": {k: v.to_dict() for k, v in self.per_round_importances.items()},
        }


# ---------------------------------------------------------------------------
# Matchup explainer
# ---------------------------------------------------------------------------

class MatchupExplainer:
    """Per-matchup SHAP-based feature attribution.

    Wraps a trained model and provides per-prediction explanations showing
    which features most influence each win probability.

    Uses SHAP TreeExplainer when available (fast, exact for tree models),
    falling back to permutation-based attribution.

    Usage:
        explainer = MatchupExplainer(model, feature_names)
        explanation = explainer.explain(X_matchup, team1="Duke", team2="UNC")
        print(explanation.top_positive[:5])  # Top 5 factors favoring team1
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100,
    ):
        """
        Args:
            model: Trained model (LightGBM Booster, sklearn estimator, etc).
            feature_names: Feature names matching model input columns.
            background_data: Background dataset for SHAP KernelExplainer.
                If None, SHAP TreeExplainer is used (requires tree model).
            n_background: Number of background samples for KernelExplainer.
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.n_background = n_background
        self._explainer = None
        self._method = "native_importance"

        self._init_explainer()

    def _init_explainer(self):
        """Initialize the best available explainer."""
        if SHAP_AVAILABLE:
            try:
                # Try TreeExplainer first (fast, exact for LightGBM/XGBoost)
                self._explainer = shap.TreeExplainer(self.model)
                self._method = "shap_tree"
                logger.info("Using SHAP TreeExplainer for matchup explanations.")
                return
            except Exception:
                pass

            # Fall back to KernelExplainer
            if self.background_data is not None:
                try:
                    bg = self.background_data
                    if len(bg) > self.n_background:
                        idx = np.random.RandomState(42).choice(
                            len(bg), self.n_background, replace=False
                        )
                        bg = bg[idx]
                    self._explainer = shap.KernelExplainer(
                        self._predict_proba, bg
                    )
                    self._method = "shap_kernel"
                    logger.info("Using SHAP KernelExplainer for matchup explanations.")
                    return
                except Exception:
                    pass

        logger.info("SHAP unavailable; using native feature importance fallback.")

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Wrapper to get probabilities from any model type."""
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.model, 'predict'):
            return np.asarray(self.model.predict(X))
        raise ValueError("Model has no predict or predict_proba method.")

    def explain(
        self,
        X: np.ndarray,
        team1: str = "Team 1",
        team2: str = "Team 2",
        top_k: int = 10,
    ) -> MatchupExplanation:
        """Explain a single matchup prediction.

        Args:
            X: Feature vector(s), shape (D,) or (1, D).
            team1: Name of team 1 (higher-seed / first listed).
            team2: Name of team 2.
            top_k: Number of top features to highlight.

        Returns:
            MatchupExplanation with feature contributions.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        prob = float(self._predict_proba(X)[0])

        contributions = self._compute_contributions(X)

        # Sort by absolute contribution
        sorted_contribs = sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)
        for rank, c in enumerate(sorted_contribs):
            c.rank = rank + 1

        # Split into positive (favoring team1) and negative (favoring team2)
        top_pos = sorted(
            [c for c in contributions if c.contribution > 0],
            key=lambda c: c.contribution, reverse=True,
        )[:top_k]
        top_neg = sorted(
            [c for c in contributions if c.contribution < 0],
            key=lambda c: c.contribution,
        )[:top_k]

        base_value = 0.5
        if self._explainer is not None and hasattr(self._explainer, 'expected_value'):
            ev = self._explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_value = float(ev[-1]) if len(ev) > 1 else float(ev[0])
            else:
                base_value = float(ev)

        return MatchupExplanation(
            team1=team1,
            team2=team2,
            predicted_probability=prob,
            base_value=base_value,
            contributions=sorted_contribs,
            method=self._method,
            top_positive=top_pos,
            top_negative=top_neg,
        )

    def explain_batch(
        self,
        X: np.ndarray,
        team_names: Optional[List[Tuple[str, str]]] = None,
        top_k: int = 10,
    ) -> List[MatchupExplanation]:
        """Explain a batch of matchup predictions.

        Args:
            X: Feature matrix, shape (N, D).
            team_names: Optional list of (team1, team2) tuples.
            top_k: Number of top features per explanation.
        """
        if team_names is None:
            team_names = [(f"Team1_{i}", f"Team2_{i}") for i in range(len(X))]

        explanations = []
        for i in range(len(X)):
            t1, t2 = team_names[i]
            explanations.append(self.explain(X[i], team1=t1, team2=t2, top_k=top_k))

        return explanations

    def _compute_contributions(self, X: np.ndarray) -> List[FeatureContribution]:
        """Compute per-feature contributions using the best available method."""
        if self._explainer is not None and self._method.startswith("shap"):
            return self._shap_contributions(X)
        return self._native_importance_contributions(X)

    def _shap_contributions(self, X: np.ndarray) -> List[FeatureContribution]:
        """Compute SHAP-based feature contributions."""
        shap_values = self._explainer.shap_values(X)

        # Handle multi-output (binary classification returns list of 2)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        if shap_values.ndim == 2:
            shap_values = shap_values[0]  # Single sample

        contributions = []
        n_features = min(len(self.feature_names), len(shap_values))
        for i in range(n_features):
            contributions.append(FeatureContribution(
                name=self.feature_names[i],
                value=float(X[0, i]) if X.ndim == 2 else float(X[i]),
                contribution=float(shap_values[i]),
            ))

        return contributions

    def _native_importance_contributions(self, X: np.ndarray) -> List[FeatureContribution]:
        """Approximate contributions using native importance × feature value.

        This is a rough approximation: importance(feature) × sign(feature_value).
        It doesn't capture interactions but provides directional guidance.
        """
        importances = self._get_native_importances()

        contributions = []
        n_features = min(len(self.feature_names), X.shape[1] if X.ndim == 2 else len(X))
        for i in range(n_features):
            val = float(X[0, i]) if X.ndim == 2 else float(X[i])
            imp = importances.get(self.feature_names[i], 0.0)
            # Approximate contribution: importance * normalized value
            contribution = imp * np.sign(val) * min(abs(val), 1.0)
            contributions.append(FeatureContribution(
                name=self.feature_names[i],
                value=val,
                contribution=contribution,
            ))

        return contributions

    def _get_native_importances(self) -> Dict[str, float]:
        """Extract native feature importances from the model."""
        if hasattr(self.model, 'feature_importance'):
            imp = self.model.feature_importance(importance_type='gain')
            names = self.feature_names[:len(imp)]
            total = sum(imp) if sum(imp) > 0 else 1.0
            return {n: float(v / total) for n, v in zip(names, imp)}

        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
            names = self.feature_names[:len(imp)]
            total = sum(imp) if sum(imp) > 0 else 1.0
            return {n: float(v / total) for n, v in zip(names, imp)}

        if hasattr(self.model, 'coef_'):
            coef = np.abs(self.model.coef_.ravel())
            names = self.feature_names[:len(coef)]
            total = sum(coef) if sum(coef) > 0 else 1.0
            return {n: float(v / total) for n, v in zip(names, coef)}

        return {n: 1.0 / len(self.feature_names) for n in self.feature_names}


# ---------------------------------------------------------------------------
# LOYO feature importance
# ---------------------------------------------------------------------------

class LOYOFeatureImportance:
    """Cross-validated feature importance using LOYO folds.

    Computes feature importance for each LOYO fold and aggregates to
    find features that are consistently important across years.
    This is more robust than single-model importance because it's
    not overfit to any particular year's data.

    Supports SHAP TreeExplainer (preferred) or LightGBM native importance.
    """

    def __init__(
        self,
        feature_names: List[str],
        top_k: int = 20,
        method: str = "auto",
    ):
        """
        Args:
            feature_names: List of feature names.
            top_k: Number of top features for stability scoring.
            method: "shap", "native", or "auto" (SHAP if available).
        """
        self.feature_names = feature_names
        self.top_k = top_k

        if method == "auto":
            self.method = "shap" if SHAP_AVAILABLE and LIGHTGBM_AVAILABLE else "native"
        else:
            self.method = method

    def compute(
        self,
        years: List[int],
        data_by_year: Dict[int, Tuple[np.ndarray, np.ndarray]],
        train_fn: Optional[Callable] = None,
        models_by_fold: Optional[Dict[int, Any]] = None,
    ) -> LOYOFeatureImportanceReport:
        """Compute LOYO-aggregated feature importance.

        Either provide train_fn (trains a new model per fold) or
        models_by_fold (pre-trained models indexed by holdout year).

        Args:
            years: LOYO validation years.
            data_by_year: {year: (X, y)} for each year.
            train_fn: Callable(X_train, y_train) -> model.
            models_by_fold: Pre-trained {holdout_year: model}.

        Returns:
            LOYOFeatureImportanceReport with global and per-fold importances.
        """
        per_fold_importances: Dict[int, np.ndarray] = {}

        for holdout_year in years:
            train_years = [yr for yr in years if yr != holdout_year]

            # Get or train model for this fold
            if models_by_fold and holdout_year in models_by_fold:
                model = models_by_fold[holdout_year]
            elif train_fn:
                X_parts, y_parts = [], []
                for yr in train_years:
                    if yr in data_by_year:
                        X_yr, y_yr = data_by_year[yr]
                        X_parts.append(X_yr)
                        y_parts.append(y_yr)
                if not X_parts:
                    continue
                X_train = np.vstack(X_parts)
                y_train = np.concatenate(y_parts)
                model = train_fn(X_train, y_train)
            else:
                raise ValueError("Must provide either train_fn or models_by_fold.")

            # Compute importance for this fold
            fold_importance = self._compute_fold_importance(
                model, data_by_year.get(holdout_year)
            )
            if fold_importance is not None:
                per_fold_importances[holdout_year] = fold_importance

        if not per_fold_importances:
            return LOYOFeatureImportanceReport(
                global_importances=[],
                per_fold_importances={},
                per_round_importances={},
                method=self.method,
                n_folds=0,
                n_features=len(self.feature_names),
            )

        # Aggregate across folds
        global_imp, fold_summaries = self._aggregate_importances(per_fold_importances)

        return LOYOFeatureImportanceReport(
            global_importances=global_imp,
            per_fold_importances=fold_summaries,
            per_round_importances={},  # Populated separately by PerRoundFeatureImportance
            method=self.method,
            n_folds=len(per_fold_importances),
            n_features=len(self.feature_names),
        )

    def _compute_fold_importance(
        self,
        model: Any,
        holdout_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """Compute feature importance for a single fold's model."""
        if self.method == "shap" and SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                if holdout_data is not None:
                    X_test = holdout_data[0]
                    n_explain = min(200, len(X_test))
                    shap_values = explainer.shap_values(X_test[:n_explain])
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    return np.mean(np.abs(shap_values), axis=0)
                else:
                    # No test data — can't compute SHAP
                    pass
            except Exception as e:
                logger.debug("SHAP failed for fold: %s. Falling back to native.", e)

        # Native importance fallback
        return self._native_importance(model)

    def _native_importance(self, model: Any) -> Optional[np.ndarray]:
        """Extract native importance from model."""
        if hasattr(model, 'feature_importance'):
            return np.array(model.feature_importance(importance_type='gain'), dtype=float)
        if hasattr(model, 'feature_importances_'):
            return np.array(model.feature_importances_, dtype=float)
        if hasattr(model, 'coef_'):
            return np.abs(model.coef_.ravel())
        return None

    def _aggregate_importances(
        self,
        per_fold: Dict[int, np.ndarray],
    ) -> Tuple[List[FeatureImportanceSummary], Dict[int, List[FeatureImportanceSummary]]]:
        """Aggregate per-fold importances into global ranking."""
        n_features = len(self.feature_names)
        fold_years = sorted(per_fold.keys())

        # Normalize each fold's importances to sum to 1
        all_imp = np.zeros((len(fold_years), n_features))
        for i, year in enumerate(fold_years):
            imp = per_fold[year]
            if len(imp) > n_features:
                imp = imp[:n_features]
            elif len(imp) < n_features:
                padded = np.zeros(n_features)
                padded[:len(imp)] = imp
                imp = padded
            total = np.sum(imp)
            if total > 0:
                all_imp[i] = imp / total
            else:
                all_imp[i] = imp

        # Global importance: mean across folds
        mean_imp = np.mean(all_imp, axis=0)
        std_imp = np.std(all_imp, axis=0, ddof=1) if len(fold_years) > 1 else np.zeros(n_features)

        # Stability: fraction of folds where feature is in top-K
        stability = np.zeros(n_features)
        for i in range(len(fold_years)):
            top_k_idx = np.argsort(-all_imp[i])[:self.top_k]
            stability[top_k_idx] += 1
        stability /= len(fold_years)

        # Build global summary
        order = np.argsort(-mean_imp)
        global_summary = []
        for rank, idx in enumerate(order):
            global_summary.append(FeatureImportanceSummary(
                name=self.feature_names[idx] if idx < len(self.feature_names) else f"f_{idx}",
                mean_importance=float(mean_imp[idx]),
                std_importance=float(std_imp[idx]),
                stability=float(stability[idx]),
                rank=rank + 1,
            ))

        # Per-fold summaries
        fold_summaries = {}
        for i, year in enumerate(fold_years):
            fold_order = np.argsort(-all_imp[i])
            fold_summary = []
            for rank, idx in enumerate(fold_order):
                fold_summary.append(FeatureImportanceSummary(
                    name=self.feature_names[idx] if idx < len(self.feature_names) else f"f_{idx}",
                    mean_importance=float(all_imp[i, idx]),
                    std_importance=0.0,
                    stability=float(stability[idx]),
                    rank=rank + 1,
                ))
            fold_summaries[year] = fold_summary

        return global_summary, fold_summaries


# ---------------------------------------------------------------------------
# Per-round feature importance
# ---------------------------------------------------------------------------

# Tournament round labels
ROUND_NAMES = {
    1: "R64",
    2: "R32",
    3: "S16",
    4: "E8",
    5: "F4",
    6: "NCG",
}


class PerRoundFeatureImportance:
    """Feature importance stratified by tournament round.

    Answers: "Which features matter most in Round of 64 vs. Final Four?"
    Useful for understanding if the model relies on different signals
    at different stages of the tournament.

    Requires round labels for each sample.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rounds: np.ndarray,
        model: Any = None,
        train_fn: Optional[Callable] = None,
    ) -> Dict[str, PerRoundImportance]:
        """Compute per-round feature importance.

        Args:
            X: Feature matrix, shape (N, D).
            y: Binary labels, shape (N,).
            rounds: Round labels for each sample, shape (N,).
                Values should be 1-6 (R64 through NCG) or string names.
            model: Pre-trained model (optional).
            train_fn: Training function (optional, used if model is None).

        Returns:
            Dict[round_name, PerRoundImportance].
        """
        if model is None and train_fn is not None:
            model = train_fn(X, y)

        unique_rounds = np.unique(rounds)
        results = {}

        for r in unique_rounds:
            mask = rounds == r
            n_games = int(np.sum(mask))
            if n_games < 5:
                continue

            round_name = ROUND_NAMES.get(r, str(r)) if isinstance(r, (int, np.integer)) else str(r)

            # Compute importance for this round's subset
            X_round = X[mask]
            importances = self._compute_round_importance(model, X_round)
            if importances is None:
                continue

            # Build summary
            n_features = min(len(self.feature_names), len(importances))
            total = np.sum(importances[:n_features])
            if total > 0:
                norm_imp = importances[:n_features] / total
            else:
                norm_imp = importances[:n_features]

            order = np.argsort(-norm_imp)
            summaries = []
            for rank, idx in enumerate(order):
                summaries.append(FeatureImportanceSummary(
                    name=self.feature_names[idx],
                    mean_importance=float(norm_imp[idx]),
                    std_importance=0.0,
                    stability=1.0,
                    rank=rank + 1,
                ))

            results[round_name] = PerRoundImportance(
                round_name=round_name,
                n_games=n_games,
                importances=summaries,
            )

        return results

    def _compute_round_importance(
        self,
        model: Any,
        X_round: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute importance for a round's subset of data."""
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)
                n_explain = min(100, len(X_round))
                shap_values = explainer.shap_values(X_round[:n_explain])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                return np.mean(np.abs(shap_values), axis=0)
            except Exception:
                pass

        # Fallback: permutation importance on this round's data
        # Return native importance (not round-specific, but best available)
        if hasattr(model, 'feature_importance'):
            return np.array(model.feature_importance(importance_type='gain'), dtype=float)
        if hasattr(model, 'feature_importances_'):
            return np.array(model.feature_importances_, dtype=float)

        return None

    def compute_with_bootstrap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rounds: np.ndarray,
        model: Any = None,
        train_fn: Optional[Callable] = None,
        n_bootstrap: int = 200,
        random_seed: int = 42,
    ) -> Dict[str, PerRoundImportance]:
        """Compute per-round feature importance with bootstrap confidence intervals.

        Solution 6: Adds uncertainty estimates to per-round importance. With
        only ~16 games per round per year, single-point estimates are noisy.
        Bootstrap CIs reveal which features are reliably important in late
        rounds vs. just noise.

        Args:
            X: Feature matrix, shape (N, D).
            y: Binary labels, shape (N,).
            rounds: Round labels for each sample, shape (N,).
            model: Pre-trained model.
            train_fn: Training function (used if model is None).
            n_bootstrap: Number of bootstrap resamples.
            random_seed: For reproducibility.

        Returns:
            Dict[round_name, PerRoundImportance] with bootstrap CIs populated.
        """
        if model is None and train_fn is not None:
            model = train_fn(X, y)

        rng = np.random.default_rng(random_seed)
        unique_rounds = np.unique(rounds)
        results = {}

        for r in unique_rounds:
            mask = rounds == r
            n_games = int(np.sum(mask))
            if n_games < 5:
                continue

            round_name = ROUND_NAMES.get(r, str(r)) if isinstance(r, (int, np.integer)) else str(r)
            X_round = X[mask]

            # Point estimate
            base_imp = self._compute_round_importance(model, X_round)
            if base_imp is None:
                continue

            n_features = min(len(self.feature_names), len(base_imp))
            total = np.sum(base_imp[:n_features])
            if total > 0:
                base_norm = base_imp[:n_features] / total
            else:
                base_norm = base_imp[:n_features]

            # Bootstrap
            boot_importances = np.zeros((n_bootstrap, n_features))
            successful_boots = 0
            for b in range(n_bootstrap):
                boot_idx = rng.choice(len(X_round), size=len(X_round), replace=True)
                boot_imp = self._compute_round_importance(model, X_round[boot_idx])
                if boot_imp is not None:
                    bt = boot_imp[:n_features]
                    bt_total = np.sum(bt)
                    if bt_total > 0:
                        bt = bt / bt_total
                    boot_importances[successful_boots] = bt
                    successful_boots += 1

            if successful_boots < 10:
                # Not enough bootstrap samples; fall back to point estimate
                order = np.argsort(-base_norm)
                summaries = [
                    FeatureImportanceSummary(
                        name=self.feature_names[idx],
                        mean_importance=float(base_norm[idx]),
                        std_importance=0.0,
                        stability=1.0,
                        rank=rank + 1,
                    )
                    for rank, idx in enumerate(order)
                ]
                results[round_name] = PerRoundImportance(
                    round_name=round_name, n_games=n_games, importances=summaries,
                )
                continue

            boot_importances = boot_importances[:successful_boots]
            boot_means = np.mean(boot_importances, axis=0)
            boot_stds = np.std(boot_importances, axis=0)

            order = np.argsort(-boot_means)
            summaries = []
            for rank, idx in enumerate(order):
                ci_lower = float(np.percentile(boot_importances[:, idx], 2.5))
                ci_upper = float(np.percentile(boot_importances[:, idx], 97.5))
                # Feature is significant if CI excludes zero
                is_significant = ci_lower > 0
                summaries.append(FeatureImportanceSummary(
                    name=self.feature_names[idx],
                    mean_importance=float(boot_means[idx]),
                    std_importance=float(boot_stds[idx]),
                    stability=1.0 if is_significant else 0.0,
                    rank=rank + 1,
                ))

            results[round_name] = PerRoundImportance(
                round_name=round_name, n_games=n_games, importances=summaries,
            )
            n_sig = sum(1 for s in summaries if s.stability > 0)
            logger.info(
                "Round %s: %d/%d features significant (bootstrap CI excludes 0)",
                round_name, n_sig, len(summaries),
            )

        return results


# ---------------------------------------------------------------------------
# Solution 13: SHAP interaction effects
# ---------------------------------------------------------------------------

@dataclass
class InteractionEffect:
    """A detected feature interaction."""
    feature_a: str
    feature_b: str
    interaction_strength: float
    direction: str  # "synergistic" or "antagonistic"


@dataclass
class InteractionEffects:
    """Results of SHAP interaction analysis."""
    interaction_matrix: Dict[str, Dict[str, float]]
    top_interactions: List[InteractionEffect]
    n_features_analyzed: int
    method: str


class InteractionAnalyzer:
    """Compute SHAP interaction values for top-K features.

    Solution 13: SHAP main effects miss interaction effects. This reveals
    whether feature pairs have synergistic effects (e.g., does three_pt_variance
    only matter when seed_diff is large?).

    Uses SHAP TreeExplainer with interaction_values for tree-based models.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def compute_interaction_effects(
        self,
        model: Any,
        X: np.ndarray,
        top_k: int = 10,
        max_samples: int = 200,
    ) -> InteractionEffects:
        """Compute SHAP interaction values for top-K features.

        Args:
            model: Trained tree model (LightGBM or XGBoost).
            X: Feature matrix [N, D].
            top_k: Number of top features to analyze interactions for.
            max_samples: Max samples to compute interactions on (O(n*k^2)).

        Returns:
            InteractionEffects with interaction matrix and top interactions.
        """
        n_features = min(len(self.feature_names), X.shape[1])
        top_k = min(top_k, n_features)

        # First get main SHAP values to identify top-K features
        if not SHAP_AVAILABLE:
            return InteractionEffects(
                interaction_matrix={}, top_interactions=[],
                n_features_analyzed=0, method="unavailable",
            )

        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            return InteractionEffects(
                interaction_matrix={}, top_interactions=[],
                n_features_analyzed=0, method="tree_explainer_failed",
            )

        # Subsample for performance
        n_explain = min(max_samples, len(X))
        if n_explain < len(X):
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X), size=n_explain, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

        # Get main effects to find top-K features
        try:
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(-mean_abs_shap)[:top_k]
        except Exception as e:
            logger.warning("SHAP main effects failed: %s", e)
            return InteractionEffects(
                interaction_matrix={}, top_interactions=[],
                n_features_analyzed=0, method="shap_failed",
            )

        # Try SHAP interaction values
        try:
            interaction_values = explainer.shap_interaction_values(X_sample)
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1] if len(interaction_values) > 1 else interaction_values[0]
            # interaction_values shape: (n_samples, n_features, n_features)
            method = "shap_interaction"
        except Exception:
            # Fallback: estimate interactions from correlation of SHAP values
            interaction_values = None
            method = "shap_correlation_proxy"

        # Build interaction matrix for top-K features
        interaction_matrix = {}
        top_interactions = []

        if interaction_values is not None:
            # Use actual SHAP interaction values
            for i in range(len(top_indices)):
                fi = int(top_indices[i])
                name_a = self.feature_names[fi]
                interaction_matrix[name_a] = {}
                for j in range(len(top_indices)):
                    fj = int(top_indices[j])
                    name_b = self.feature_names[fj]
                    # Mean absolute interaction value
                    strength = float(np.mean(np.abs(interaction_values[:, fi, fj])))
                    interaction_matrix[name_a][name_b] = strength

                    if i < j:  # Avoid duplicates
                        # Direction: correlation of interaction with prediction
                        mean_interaction = float(np.mean(interaction_values[:, fi, fj]))
                        direction = "synergistic" if mean_interaction > 0 else "antagonistic"
                        top_interactions.append(InteractionEffect(
                            feature_a=name_a, feature_b=name_b,
                            interaction_strength=strength, direction=direction,
                        ))
        else:
            # Proxy: correlation of SHAP values
            for i in range(len(top_indices)):
                fi = int(top_indices[i])
                name_a = self.feature_names[fi]
                interaction_matrix[name_a] = {}
                for j in range(len(top_indices)):
                    fj = int(top_indices[j])
                    name_b = self.feature_names[fj]
                    if i == j:
                        interaction_matrix[name_a][name_b] = 1.0
                    else:
                        r = np.corrcoef(shap_values[:, fi], shap_values[:, fj])[0, 1]
                        r = 0.0 if np.isnan(r) else float(r)
                        interaction_matrix[name_a][name_b] = abs(r)

                        if i < j:
                            direction = "synergistic" if r > 0 else "antagonistic"
                            top_interactions.append(InteractionEffect(
                                feature_a=name_a, feature_b=name_b,
                                interaction_strength=abs(r), direction=direction,
                            ))

        # Sort by strength
        top_interactions.sort(key=lambda x: x.interaction_strength, reverse=True)

        logger.info(
            "Interaction analysis (%s): top 3 interactions: %s",
            method,
            [(x.feature_a, x.feature_b, f"{x.interaction_strength:.4f}") for x in top_interactions[:3]],
        )

        return InteractionEffects(
            interaction_matrix=interaction_matrix,
            top_interactions=top_interactions,
            n_features_analyzed=top_k,
            method=method,
        )
