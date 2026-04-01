"""LOYO-derived ensemble stacking weights and diversity metrics.

Solution 3: Replace hardcoded ensemble weights with a meta-learner trained
on LOYO out-of-fold predictions. Learns optimal model weights from cross-
validated performance.

Solution 14: Ensemble diversity metrics to measure whether component models
make diverse errors. Low diversity → ensemble provides little benefit.

FIX-STACKING-LEAKAGE: Added NestedLOYOEnsembleResult and fit_nested_loyo()
to structurally prevent stacking weight contamination.  Ensemble weights
are now derived via nested LOYO: for each outer fold the weights are
optimized on inner folds only, so the held-out fold's Brier is never
influenced by weights that saw its data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solution 3: Stacking weight optimizer
# ---------------------------------------------------------------------------

@dataclass
class StackingWeightResult:
    """Result of stacking weight optimization."""
    weights: Dict[str, float]
    brier_with_stacking: float
    brier_with_fixed: float
    improvement: float
    model_names: List[str]
    effective_model_count: float  # 1/sum(w^2) — higher = more diverse weighting
    method: str = "constrained_logistic"


@dataclass(frozen=True)
class NestedLOYOEnsembleResult:
    """Ensemble metrics from nested LOYO — structurally honest.

    This dataclass can ONLY represent metrics where each fold's ensemble
    weights were derived without seeing that fold's data.  The frozen
    dataclass prevents mutation after construction.

    FIX-STACKING-LEAKAGE: This is the ONLY acceptable source of ensemble
    evaluation metrics.  Any code that reports ensemble Brier must use
    this type, not raw pooled optimization results.
    """

    per_fold_weights: Dict[int, Dict[str, float]]  # year -> {model: weight}
    per_fold_brier: Dict[int, float]                # year -> honest Brier
    mean_brier: float
    std_brier: float
    n_folds: int
    n_total_samples: int
    production_weights: Dict[str, float]            # Final weights for inference
    weight_source: str = "nested_loyo"              # Immutable provenance tag


class StackingWeightOptimizer:
    """Derive ensemble weights from LOYO out-of-fold predictions.

    Instead of hardcoding weights (e.g., spread=0.45, lgb=0.20), fits a
    constrained meta-learner on cross-validated predictions. Constraints:
    - All weights >= 0
    - Sum of weights == 1 (simplex)
    - L2 regularization toward uniform (prevents overfit to single model)

    Usage:
        optimizer = StackingWeightOptimizer()
        result = optimizer.fit(
            fold_predictions={"lgb": lgb_preds, "xgb": xgb_preds, ...},
            outcomes=y,
        )
        print(result.weights)  # {"lgb": 0.35, "xgb": 0.25, ...}
    """

    def __init__(
        self,
        regularization: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Args:
            regularization: L2 penalty toward uniform weights (0 = no penalty)
            random_seed: For reproducibility
        """
        self.regularization = regularization
        self.random_seed = random_seed

    def fit(
        self,
        fold_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
        fixed_weights: Optional[Dict[str, float]] = None,
    ) -> StackingWeightResult:
        """Optimize ensemble weights on out-of-fold predictions.

        Args:
            fold_predictions: Dict mapping model_name → predictions array [N]
            outcomes: Binary outcomes [N]
            fixed_weights: Optional baseline fixed weights for comparison

        Returns:
            StackingWeightResult with optimized weights
        """
        model_names = sorted(fold_predictions.keys())
        n_models = len(model_names)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(outcomes)

        if n_models < 2:
            name = model_names[0] if model_names else "unknown"
            return StackingWeightResult(
                weights={name: 1.0},
                brier_with_stacking=float(np.mean((fold_predictions[name] - outcomes) ** 2)),
                brier_with_fixed=float(np.mean((fold_predictions[name] - outcomes) ** 2)),
                improvement=0.0,
                model_names=model_names,
                effective_model_count=1.0,
            )

        # Stack predictions into matrix [N, K]
        pred_matrix = np.column_stack([
            np.asarray(fold_predictions[name], dtype=float)
            for name in model_names
        ])

        # Optimize weights via scipy if available, else grid search
        try:
            from scipy.optimize import minimize
            weights = self._optimize_scipy(pred_matrix, outcomes, n_models)
        except ImportError:
            weights = self._optimize_grid(pred_matrix, outcomes, n_models)

        # Compute Brier with optimized weights
        stacked_preds = pred_matrix @ weights
        brier_stacking = float(np.mean((stacked_preds - outcomes) ** 2))

        # Compute Brier with fixed weights (for comparison)
        if fixed_weights:
            fixed_w = np.array([fixed_weights.get(name, 1.0 / n_models) for name in model_names])
            fixed_w = fixed_w / fixed_w.sum()
            fixed_preds = pred_matrix @ fixed_w
            brier_fixed = float(np.mean((fixed_preds - outcomes) ** 2))
        else:
            # Use uniform weights as baseline
            uniform_preds = pred_matrix @ np.ones(n_models) / n_models
            brier_fixed = float(np.mean((uniform_preds - outcomes) ** 2))

        weight_dict = {name: float(w) for name, w in zip(model_names, weights)}
        emc = 1.0 / float(np.sum(weights ** 2)) if np.sum(weights ** 2) > 0 else 1.0

        logger.info(
            "Stacking weights: %s (Brier: %.4f → %.4f, EMC: %.1f)",
            {k: f"{v:.3f}" for k, v in weight_dict.items()},
            brier_fixed, brier_stacking, emc,
        )

        return StackingWeightResult(
            weights=weight_dict,
            brier_with_stacking=brier_stacking,
            brier_with_fixed=brier_fixed,
            improvement=brier_fixed - brier_stacking,
            model_names=model_names,
            effective_model_count=emc,
        )

    def fit_nested_loyo(
        self,
        predictions_by_year: Dict[int, Dict[str, np.ndarray]],
        outcomes_by_year: Dict[int, np.ndarray],
        fixed_weights: Optional[Dict[str, float]] = None,
    ) -> NestedLOYOEnsembleResult:
        """Derive ensemble weights via nested LOYO — structurally honest.

        For each outer fold (held-out year Y):
          1. Pool predictions from all OTHER years
          2. Optimize weights on that pool (inner optimization)
          3. Apply those weights to year Y's predictions
          4. Record year Y's Brier (weights never saw year Y)

        The reported mean Brier is guaranteed uncontaminated because each
        fold's weights are derived without access to that fold's data.

        Args:
            predictions_by_year: {year: {model_name: predictions [N_year]}}
            outcomes_by_year: {year: outcomes [N_year]}
            fixed_weights: Optional baseline for comparison

        Returns:
            NestedLOYOEnsembleResult with honest per-fold metrics
        """
        years = sorted(predictions_by_year.keys())
        if len(years) < 3:
            raise ValueError(
                f"Nested LOYO requires >= 3 years, got {len(years)}"
            )

        model_names = sorted(next(iter(predictions_by_year.values())).keys())
        per_fold_weights: Dict[int, Dict[str, float]] = {}
        per_fold_brier: Dict[int, float] = {}
        all_inner_preds: Dict[str, list] = {n: [] for n in model_names}
        all_inner_outcomes: list = []
        n_total = 0

        for hold_year in years:
            # --- Inner optimization: pool all years except hold_year ---
            inner_preds: Dict[str, list] = {n: [] for n in model_names}
            inner_y: list = []

            for yr in years:
                if yr == hold_year:
                    continue
                for name in model_names:
                    inner_preds[name].extend(
                        predictions_by_year[yr][name].tolist()
                    )
                inner_y.extend(outcomes_by_year[yr].tolist())

            inner_pred_arrays = {
                name: np.array(vals) for name, vals in inner_preds.items()
            }
            inner_outcomes = np.array(inner_y)

            # Fit weights on inner data only
            inner_result = self.fit(
                inner_pred_arrays, inner_outcomes, fixed_weights
            )
            fold_weights = inner_result.weights
            per_fold_weights[hold_year] = fold_weights

            # --- Evaluate on held-out year with inner-derived weights ---
            hold_preds = predictions_by_year[hold_year]
            hold_y = outcomes_by_year[hold_year]
            n_hold = len(hold_y)

            weighted_pred = np.zeros(n_hold)
            for name in model_names:
                weighted_pred += fold_weights.get(name, 0.0) * hold_preds[name]

            fold_brier = float(np.mean((weighted_pred - hold_y) ** 2))
            per_fold_brier[hold_year] = fold_brier
            n_total += n_hold

            # Collect for production weights (full-data fit at the end)
            for name in model_names:
                all_inner_preds[name].extend(hold_preds[name].tolist())
            all_inner_outcomes.extend(hold_y.tolist())

            logger.info(
                "Nested LOYO fold %d: Brier=%.5f, weights=%s",
                hold_year, fold_brier,
                {k: f"{v:.3f}" for k, v in fold_weights.items()},
            )

        # Production weights: fit on ALL data (safe — used only for
        # inference on unseen 2026 data, never for metric reporting)
        all_pred_arrays = {
            name: np.array(vals) for name, vals in all_inner_preds.items()
        }
        all_outcomes = np.array(all_inner_outcomes)
        production_result = self.fit(all_pred_arrays, all_outcomes, fixed_weights)

        brier_values = list(per_fold_brier.values())
        mean_brier = float(np.mean(brier_values))
        std_brier = float(np.std(brier_values, ddof=1)) if len(brier_values) > 1 else 0.0

        logger.info(
            "Nested LOYO ensemble: mean_Brier=%.5f (std=%.5f) across %d folds",
            mean_brier, std_brier, len(years),
        )

        return NestedLOYOEnsembleResult(
            per_fold_weights=per_fold_weights,
            per_fold_brier=per_fold_brier,
            mean_brier=mean_brier,
            std_brier=std_brier,
            n_folds=len(years),
            n_total_samples=n_total,
            production_weights=production_result.weights,
            weight_source="nested_loyo",
        )

    def _optimize_scipy(
        self, pred_matrix: np.ndarray, outcomes: np.ndarray, n_models: int,
    ) -> np.ndarray:
        """Optimize weights using scipy with simplex constraints."""
        from scipy.optimize import minimize

        uniform = np.ones(n_models) / n_models

        def objective(w):
            preds = pred_matrix @ w
            brier = float(np.mean((preds - outcomes) ** 2))
            # L2 regularization toward uniform
            reg = self.regularization * float(np.sum((w - uniform) ** 2))
            return brier + reg

        # Constraints: sum(w) = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        # Bounds: w_i >= 0
        bounds = [(0.0, 1.0)] * n_models

        result = minimize(
            objective,
            x0=uniform,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )

        if result.success:
            return result.x
        else:
            logger.warning("Scipy optimization did not converge; using uniform weights")
            return uniform

    def _optimize_grid(
        self, pred_matrix: np.ndarray, outcomes: np.ndarray, n_models: int,
    ) -> np.ndarray:
        """Fallback: grid search over weight simplex."""
        if n_models == 2:
            best_w = None
            best_brier = float("inf")
            for alpha in np.linspace(0, 1, 101):
                w = np.array([alpha, 1 - alpha])
                preds = pred_matrix @ w
                brier = float(np.mean((preds - outcomes) ** 2))
                if brier < best_brier:
                    best_brier = brier
                    best_w = w
            return best_w

        # For 3+ models, use random search on simplex
        rng = np.random.default_rng(self.random_seed)
        best_w = np.ones(n_models) / n_models
        best_brier = float("inf")

        for _ in range(10000):
            raw = rng.exponential(1.0, size=n_models)
            w = raw / raw.sum()
            preds = pred_matrix @ w
            brier = float(np.mean((preds - outcomes) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_w = w

        return best_w


# ---------------------------------------------------------------------------
# Solution 14: Ensemble diversity metrics
# ---------------------------------------------------------------------------

@dataclass
class DiversityResult:
    """Ensemble diversity analysis results."""
    disagreement_rate: float  # Fraction of predictions where models disagree on winner
    error_correlation_matrix: Dict[str, Dict[str, float]]  # Pairwise error correlations
    mean_error_correlation: float
    kuncheva_index: float  # Kuncheva diversity index (higher = more diverse)
    q_statistics: Dict[str, float]  # Pairwise Yule's Q statistics
    mean_q: float
    is_diverse: bool  # True if kuncheva_index > 0.3
    model_names: List[str]
    n_samples: int


class EnsembleDiversity:
    """Compute diversity metrics for an ensemble of models.

    Measures whether component models make diverse errors. If all models
    err on the same games, the ensemble provides no benefit over its
    best component.

    Metrics:
    - Disagreement rate: fraction of games where models disagree on winner
    - Error correlation: pairwise correlation of squared errors
    - Kuncheva diversity index: average pairwise diversity
    - Yule's Q statistic: agreement on correct/incorrect predictions
    """

    def __init__(self, decision_threshold: float = 0.5):
        self.decision_threshold = decision_threshold

    def compute(
        self,
        model_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
    ) -> DiversityResult:
        """Compute all diversity metrics.

        Args:
            model_predictions: Dict mapping model_name → predicted probabilities
            outcomes: Actual binary outcomes

        Returns:
            DiversityResult with all metrics
        """
        model_names = sorted(model_predictions.keys())
        n_models = len(model_names)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(outcomes)

        # Binary predictions for each model
        binary_preds = {}
        squared_errors = {}
        for name in model_names:
            preds = np.asarray(model_predictions[name], dtype=float)
            binary_preds[name] = (preds >= self.decision_threshold).astype(float)
            squared_errors[name] = (preds - outcomes) ** 2

        # 1. Disagreement rate
        if n_models >= 2:
            disagreements = 0
            n_pairs = 0
            for name_a, name_b in combinations(model_names, 2):
                disagreements += int(np.sum(binary_preds[name_a] != binary_preds[name_b]))
                n_pairs += 1
            disagreement_rate = disagreements / (n_pairs * n) if n_pairs * n > 0 else 0.0
        else:
            disagreement_rate = 0.0

        # 2. Error correlation matrix
        error_corr_matrix = {}
        corr_values = []
        for name_a in model_names:
            error_corr_matrix[name_a] = {}
            for name_b in model_names:
                if name_a == name_b:
                    error_corr_matrix[name_a][name_b] = 1.0
                else:
                    r = np.corrcoef(squared_errors[name_a], squared_errors[name_b])[0, 1]
                    r = 0.0 if np.isnan(r) else float(r)
                    error_corr_matrix[name_a][name_b] = r
                    if name_a < name_b:
                        corr_values.append(r)

        mean_error_corr = float(np.mean(corr_values)) if corr_values else 0.0

        # 3. Kuncheva diversity index (based on disagreement)
        # κ = 1 - (average pairwise agreement)
        # Agreement = fraction of samples where both models make same binary prediction
        kuncheva_values = []
        q_statistics = {}

        for name_a, name_b in combinations(model_names, 2):
            pa = binary_preds[name_a]
            pb = binary_preds[name_b]
            correct_a = (pa == outcomes).astype(int)
            correct_b = (pb == outcomes).astype(int)

            # Contingency table
            n11 = int(np.sum(correct_a * correct_b))  # Both correct
            n00 = int(np.sum((1 - correct_a) * (1 - correct_b)))  # Both wrong
            n10 = int(np.sum(correct_a * (1 - correct_b)))  # A correct, B wrong
            n01 = int(np.sum((1 - correct_a) * correct_b))  # A wrong, B correct

            # Yule's Q
            denom = n11 * n00 + n10 * n01
            if denom > 0:
                q = (n11 * n00 - n10 * n01) / denom
            else:
                q = 0.0
            q_statistics[f"{name_a}_vs_{name_b}"] = float(q)

            # Kuncheva: proportion of disagreements
            total = n11 + n00 + n10 + n01
            if total > 0:
                kappa = (n10 + n01) / total  # disagreement proportion
                kuncheva_values.append(kappa)

        kuncheva_index = float(np.mean(kuncheva_values)) if kuncheva_values else 0.0
        mean_q = float(np.mean(list(q_statistics.values()))) if q_statistics else 0.0
        is_diverse = kuncheva_index > 0.3

        logger.info(
            "Ensemble diversity: disagreement=%.3f, mean_error_corr=%.3f, "
            "kuncheva=%.3f, mean_Q=%.3f, diverse=%s",
            disagreement_rate, mean_error_corr, kuncheva_index, mean_q, is_diverse,
        )

        return DiversityResult(
            disagreement_rate=disagreement_rate,
            error_correlation_matrix=error_corr_matrix,
            mean_error_correlation=mean_error_corr,
            kuncheva_index=kuncheva_index,
            q_statistics=q_statistics,
            mean_q=mean_q,
            is_diverse=is_diverse,
            model_names=model_names,
            n_samples=n,
        )
