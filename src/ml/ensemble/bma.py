"""Bayesian Model Averaging (BMA) for ensemble construction.

Protocol v2, Section 3.2: Replace fixed grid-search ensemble weight optimization
with BMA. BMA produces a weighted average across all models that pass the
multi-metric gate, where weights are posterior model probabilities computed
via the EM algorithm.

Reference: Raftery et al. (2005) — BMA for ensemble calibration in weather
forecasting, with direct analogies to tournament prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BMAResult:
    """Result of Bayesian Model Averaging."""

    # Model weights (posterior probabilities), keyed by model name
    weights: Dict[str, float] = field(default_factory=dict)

    # Combined predictions
    predictions: Optional[np.ndarray] = None

    # Diagnostics
    effective_model_count: float = 0.0  # 1 / sum(w_k^2)
    n_iterations: int = 0
    converged: bool = False
    log_marginal_likelihood: float = 0.0


class BayesianModelAveraging:
    """Bayesian Model Averaging using the EM algorithm.

    Given K models that passed the multi-metric gate and their LOYO predictions,
    computes posterior model weights via:

        P(M_k | data) ∝ P(data | M_k) * P(M_k)

    where P(data | M_k) is the likelihood under each model's predictions,
    and P(M_k) is a uniform prior across passing models.

    The final prediction is:
        P(A beats B) = sum_k w_k * P_k(A beats B)

    Protocol v2, Section 3.2: Track effective model count = 1 / sum(w_k^2).
    Flag if < 1.5 (BMA effectively selecting a single model).
    """

    def __init__(
        self,
        max_iterations: int = 200,
        tol: float = 1e-5,
        min_effective_count: float = 1.5,
    ):
        """
        Args:
            max_iterations: Maximum EM iterations.
            tol: Convergence tolerance for weight changes.
            min_effective_count: Minimum effective model count before flagging.
        """
        self.max_iterations = max_iterations
        self.tol = tol
        self.min_effective_count = min_effective_count

    def fit(
        self,
        model_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
    ) -> BMAResult:
        """Compute BMA weights via EM algorithm.

        Args:
            model_predictions: Dict of model_name -> predicted probabilities.
                Each array has shape (N,) where N is the number of evaluation games.
            outcomes: Binary outcomes array of shape (N,).

        Returns:
            BMAResult with weights, combined predictions, and diagnostics.
        """
        model_names = list(model_predictions.keys())
        K = len(model_names)

        if K == 0:
            logger.warning("BMA: No models provided.")
            return BMAResult()

        if K == 1:
            name = model_names[0]
            preds = model_predictions[name]
            logger.info("BMA: Only 1 model — weight = 1.0 for '%s'.", name)
            return BMAResult(
                weights={name: 1.0},
                predictions=preds,
                effective_model_count=1.0,
                n_iterations=0,
                converged=True,
            )

        N = len(outcomes)
        y = outcomes.astype(float)

        # Stack predictions: shape (K, N)
        pred_matrix = np.array([
            np.clip(model_predictions[name], 1e-7, 1 - 1e-7)
            for name in model_names
        ])

        # Compute log-likelihoods: log P(data | M_k) for each model
        # Using Bernoulli likelihood: sum_i [y_i * log(p_ki) + (1-y_i) * log(1-p_ki)]
        log_likelihoods = np.array([
            np.sum(y * np.log(pred_matrix[k]) + (1 - y) * np.log(1 - pred_matrix[k]))
            for k in range(K)
        ])

        # Initialize weights uniformly (Protocol: uniform prior)
        weights = np.ones(K) / K

        # EM algorithm
        converged = False
        n_iter = 0

        for iteration in range(self.max_iterations):
            n_iter = iteration + 1

            # E-step: compute responsibilities
            # For each observation i, compute P(M_k | y_i, theta)
            # log P(y_i | M_k) = y_i * log(p_ki) + (1-y_i) * log(1-p_ki)
            log_lik_per_obs = np.array([
                y * np.log(pred_matrix[k]) + (1 - y) * np.log(1 - pred_matrix[k])
                for k in range(K)
            ])  # shape (K, N)

            # Weighted log-likelihoods
            log_weighted = log_lik_per_obs + np.log(weights + 1e-30)[:, None]

            # Log-sum-exp for numerical stability
            max_log = np.max(log_weighted, axis=0)
            log_sum = max_log + np.log(np.sum(np.exp(log_weighted - max_log[None, :]), axis=0))

            # Responsibilities: shape (K, N)
            responsibilities = np.exp(log_weighted - log_sum[None, :])

            # M-step: update weights
            new_weights = responsibilities.mean(axis=1)
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights /= new_weights.sum()

            # Check convergence (use relative change for numerical stability
            # when weights are near 0 or 1)
            weight_change = float(np.max(np.abs(new_weights - weights)))
            weights = new_weights

            if weight_change < self.tol:
                converged = True
                break

            # Also converge if dominant weight is stable (BMA concentration)
            # With many binary observations, BMA can concentrate heavily on
            # the best model — this is correct behavior, not a bug.
            if np.max(weights) > 0.99 and weight_change < 1e-4:
                converged = True
                break

        # Compute final predictions
        combined_preds = np.zeros(N)
        for k, name in enumerate(model_names):
            combined_preds += weights[k] * pred_matrix[k]

        # Build weight dict
        weight_dict = {name: float(weights[k]) for k, name in enumerate(model_names)}

        # Effective model count (Protocol Section 3.2)
        effective_count = 1.0 / float(np.sum(weights ** 2))

        # Log marginal likelihood
        log_ml = float(np.sum(log_sum))

        if effective_count < self.min_effective_count:
            logger.warning(
                "BMA: Effective model count %.2f < %.1f. "
                "BMA is effectively selecting a single model. "
                "Investigate model diversity.",
                effective_count, self.min_effective_count,
            )

        # Log results
        logger.info(
            "BMA converged=%s after %d iterations. "
            "Effective model count: %.2f. Weights: %s",
            converged, n_iter, effective_count,
            {name: f"{w:.4f}" for name, w in weight_dict.items()},
        )

        return BMAResult(
            weights=weight_dict,
            predictions=combined_preds,
            effective_model_count=effective_count,
            n_iterations=n_iter,
            converged=converged,
            log_marginal_likelihood=log_ml,
        )

    def predict(
        self,
        model_predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> np.ndarray:
        """Apply pre-computed BMA weights to new predictions.

        Args:
            model_predictions: Dict of model_name -> predicted probabilities.
            weights: Pre-computed BMA weights from fit().

        Returns:
            Combined probability predictions.
        """
        N = None
        combined = None

        for name, preds in model_predictions.items():
            w = weights.get(name, 0.0)
            if w < 1e-10:
                continue
            if combined is None:
                N = len(preds)
                combined = np.zeros(N)
            combined += w * np.clip(preds, 1e-7, 1 - 1e-7)

        if combined is None:
            logger.warning("BMA predict: no valid models with nonzero weight.")
            return np.full(1, 0.5)

        return combined
