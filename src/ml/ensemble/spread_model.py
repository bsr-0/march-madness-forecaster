"""Point-spread regression model with logistic CDF probability conversion.

Predicts the expected scoring margin (team1 - team2), then converts to
win probability via:

    P(team1 wins) = 1 / (1 + exp(-predicted_spread / sigma))

where sigma is calibrated from historical residuals (~11 points for NCAA
basketball).  This preserves more statistical information than binary
classification — the margin carries richer gradient signal.

Reference: NCAA game-level spread standard deviation is ~11-12 points
(Stern 1991, Glickman & Stern 1998).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class SpreadRegressor:
    """Predicts point spread, converts to probability via logistic CDF.

    The model trains a LightGBM regression on actual point margins
    (team1_score - team2_score).  At inference time, the predicted spread
    is passed through a logistic CDF to produce a win probability.

    The logistic scale parameter ``sigma`` controls how quickly probability
    transitions from 0 to 1 as spread increases.  sigma=11 means a team
    predicted to win by 11 points has ~73% win probability — matching
    empirical NCAA upset rates.
    """

    def __init__(
        self,
        sigma: float = 11.0,
        params: Optional[Dict] = None,
    ):
        self.sigma = sigma
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 16,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 30,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "verbose": -1,
            "num_threads": 1,
        }
        self.model = None
        self.feature_names: Optional[List[str]] = None

    def train(
        self,
        X: np.ndarray,
        margins: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_rounds: int = 200,
        sample_weight: Optional[np.ndarray] = None,
        valid_X: Optional[np.ndarray] = None,
        valid_margins: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train regression model on point spreads.

        Args:
            X: Feature matrix [N, D].
            margins: Actual point margins (team1 - team2) [N].
            feature_names: Optional feature names for LightGBM.
            num_rounds: Number of boosting rounds.
            sample_weight: Optional per-sample weights.
            valid_X: Optional validation features for early stopping.
            valid_margins: Optional validation margins.

        Returns:
            Dict with training stats.
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available; spread model disabled.")
            return {"trained": False, "reason": "lightgbm_unavailable"}

        self.feature_names = feature_names
        train_data = lgb.Dataset(
            X,
            label=margins,
            feature_name=feature_names,
            weight=sample_weight,
        )

        callbacks = [lgb.log_evaluation(period=0)]
        valid_sets = []
        if valid_X is not None and valid_margins is not None and len(valid_margins) > 0:
            valid_data = lgb.Dataset(valid_X, label=valid_margins, reference=train_data)
            valid_sets = [valid_data]
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

        # Calibrate sigma on validation data if available
        if valid_X is not None and valid_margins is not None and len(valid_margins) >= 20:
            self.calibrate_sigma(valid_X, valid_margins)

        train_preds = self.model.predict(X)
        train_rmse = float(np.sqrt(np.mean((train_preds - margins) ** 2)))

        stats = {
            "trained": True,
            "n_samples": len(margins),
            "train_rmse": round(train_rmse, 4),
            "sigma": round(self.sigma, 4),
            "mean_margin": round(float(np.mean(margins)), 2),
            "std_margin": round(float(np.std(margins)), 2),
        }

        if valid_X is not None and valid_margins is not None and len(valid_margins) > 0:
            val_preds = self.model.predict(valid_X)
            val_rmse = float(np.sqrt(np.mean((val_preds - valid_margins) ** 2)))
            stats["val_rmse"] = round(val_rmse, 4)

        logger.info(
            "SpreadRegressor: trained on %d samples, train_rmse=%.2f, sigma=%.2f",
            len(margins),
            train_rmse,
            self.sigma,
        )
        return stats

    def predict_spread(self, X: np.ndarray) -> np.ndarray:
        """Predict expected point spread (team1 - team2).

        Args:
            X: Feature matrix [N, D] or single sample [D].

        Returns:
            Predicted margins as np.ndarray.
        """
        if self.model is None:
            return np.zeros(1 if X.ndim == 1 else len(X))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_probability(
        self,
        X: np.ndarray,
        sigma_override: Optional[float] = None,
    ) -> np.ndarray:
        """Convert predicted spread to win probability via logistic CDF.

        P(team1 wins) = 1 / (1 + exp(-spread / sigma))

        Args:
            X: Feature matrix [N, D] or single sample [D].
            sigma_override: Optional tournament-calibrated sigma to use
                instead of the model's default. This enables per-round
                tournament sigma calibration without modifying model state.

        Returns:
            Win probabilities as np.ndarray.
        """
        spreads = self.predict_spread(X)
        effective_sigma = sigma_override if sigma_override is not None else self.sigma
        return _logistic_cdf(spreads, effective_sigma)

    def calibrate_sigma(
        self,
        X: np.ndarray,
        actual_margins: np.ndarray,
        metric: str = "brier",
    ) -> float:
        """Calibrate sigma from validation data to minimize Brier score.

        Since Kaggle switched to Brier score in 2023, we optimize sigma
        directly for Brier rather than log-likelihood.  Brier-optimal sigma
        is typically slightly larger (less confident) than MLE sigma because
        Brier penalizes overconfidence quadratically rather than logarithmically.

        Also supports "logloss" mode for backwards compatibility.

        Args:
            X: Validation features.
            actual_margins: Actual point margins.
            metric: "brier" (default, Kaggle 2023+) or "logloss" (pre-2023).

        Returns:
            Calibrated sigma value.
        """
        if self.model is None or len(actual_margins) < 10:
            return self.sigma

        predicted_spreads = self.predict_spread(X)
        outcomes = (actual_margins > 0).astype(float)

        best_sigma = self.sigma
        best_score = float("inf") if metric == "brier" else -float("inf")

        for sigma_candidate in np.linspace(5.0, 20.0, 61):
            probs = _logistic_cdf(predicted_spreads, sigma_candidate)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)

            if metric == "brier":
                # Minimize Brier score = mean((p - y)^2)
                score = float(np.mean((probs - outcomes) ** 2))
                if score < best_score:
                    best_score = score
                    best_sigma = sigma_candidate
            else:
                # Maximize log-likelihood (legacy)
                ll = float(np.sum(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))
                if ll > best_score:
                    best_score = ll
                    best_sigma = sigma_candidate

        self.sigma = float(best_sigma)
        logger.info(
            "SpreadRegressor: calibrated sigma=%.2f (%s=%.4f)",
            self.sigma,
            metric,
            best_score,
        )
        return self.sigma


def _logistic_cdf(spreads: np.ndarray, sigma: float) -> np.ndarray:
    """Logistic CDF: P(win) = 1 / (1 + exp(-spread / sigma))."""
    return 1.0 / (1.0 + np.exp(-spreads / max(sigma, 0.01)))
