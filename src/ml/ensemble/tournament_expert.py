"""Tournament Expert sub-model.

Trained exclusively on NCAA tournament game data (2003-2025).
Captures tournament-specific dynamics that regular-season models miss:
- No home court advantage (all neutral sites)
- Single-elimination pressure
- Coach preparation time (week between rounds)
- Higher opponent quality baseline

Blended at 0.30 weight with the regular-season model.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class TournamentExpert:
    """Model trained exclusively on tournament games.

    This captures signals unique to tournament basketball:
    - Teams with tournament experience perform differently
    - Coaching preparation matters more (extended prep time)
    - Seed psychology affects play
    - No home court = different offensive efficiency distributions

    Blend weight: 0.30 with regular-season model
    """

    BLEND_WEIGHT = 0.30

    def __init__(
        self,
        sigma: float = 11.0,
        blend_weight: float = 0.30,
    ):
        self.sigma = sigma
        self.blend_weight = blend_weight
        self.spread_model = None
        self.classifier = None
        self.feature_names: Optional[List[str]] = None
        self.trained = False
        self._training_stats = {}

    def train(
        self,
        X: np.ndarray,
        margins: np.ndarray,
        outcomes: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_weights: Optional[np.ndarray] = None,
        round_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train on tournament-only data.

        Args:
            X: Feature matrix from tournament games only [N, D]
            margins: Actual point margins (team1 - team2) [N]
            outcomes: Binary outcomes (1 = team1 won) [N]
            feature_names: Feature names
            sample_weights: Optional temporal decay weights
            round_weights: Optional Kaggle round weights (F4=16x, NCG=32x)

        Returns:
            Training statistics dict
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM unavailable; TournamentExpert disabled.")
            return {"trained": False, "reason": "lightgbm_unavailable"}

        self.feature_names = feature_names
        n_samples = len(margins)

        if n_samples < 50:
            logger.warning(
                "TournamentExpert: only %d samples (need >= 50). Skipping.",
                n_samples,
            )
            return {"trained": False, "reason": f"insufficient_data_{n_samples}"}

        # Combine weights
        weights = np.ones(n_samples)
        if sample_weights is not None:
            weights *= sample_weights
        if round_weights is not None:
            weights *= round_weights

        # Train spread regression model (primary)
        spread_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 8,  # Very conservative for small sample
            "learning_rate": 0.03,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 2.0,
            "lambda_l2": 2.0,
            "verbose": -1,
            "num_threads": 1,
        }

        train_data = lgb.Dataset(
            X, label=margins,
            feature_name=feature_names,
            weight=weights,
        )

        self.spread_model = lgb.train(
            spread_params,
            train_data,
            num_boost_round=150,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        # FIX-LEAKAGE: Calibrate sigma on a held-out split to avoid
        # in-sample sigma optimization.  Use last 30% as calibration holdout.
        n_cal = max(20, int(n_samples * 0.3))
        if n_samples >= 80:
            # Enough data to hold out a calibration split
            cal_X = X[-n_cal:]
            cal_margins = margins[-n_cal:]
            self._calibrate_sigma(cal_X, cal_margins)
        else:
            # Too little data to split — use conservative default sigma
            self.sigma = 11.0

        # Also train a lightweight classifier for ensemble diversity
        cls_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 8,
            "learning_rate": 0.03,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 2.0,
            "lambda_l2": 2.0,
            "verbose": -1,
            "num_threads": 1,
        }

        cls_data = lgb.Dataset(
            X, label=outcomes,
            feature_name=feature_names,
            weight=weights,
        )

        self.classifier = lgb.train(
            cls_params,
            cls_data,
            num_boost_round=150,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        self.trained = True

        # Compute training stats
        pred_spreads = self.spread_model.predict(X)
        train_rmse = float(np.sqrt(np.mean((pred_spreads - margins) ** 2)))
        pred_probs = self._spread_to_prob(pred_spreads)
        train_brier = float(np.mean((pred_probs - outcomes) ** 2))

        self._training_stats = {
            "trained": True,
            "n_tournament_games": n_samples,
            "train_rmse": round(train_rmse, 4),
            "train_brier": round(train_brier, 4),
            "calibrated_sigma": round(self.sigma, 2),
            "mean_margin": round(float(np.mean(margins)), 2),
        }

        logger.info(
            "TournamentExpert trained on %d games: RMSE=%.2f, Brier=%.4f, sigma=%.1f",
            n_samples, train_rmse, train_brier, self.sigma,
        )

        return self._training_stats

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict win probability from tournament expert.

        Uses 70% spread-to-prob + 30% classifier for robustness.

        Args:
            X: Feature matrix [N, D]

        Returns:
            Win probabilities [N]
        """
        if not self.trained:
            return np.full(len(X) if X.ndim > 1 else 1, 0.5)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Spread-based probability (primary)
        spreads = self.spread_model.predict(X)
        spread_probs = self._spread_to_prob(spreads)

        # Classifier probability (secondary)
        cls_probs = self.classifier.predict(X)

        # Blend: 70% spread-based, 30% classifier
        return 0.70 * spread_probs + 0.30 * cls_probs

    def blend_with_regular(
        self,
        regular_probs: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Blend tournament expert with regular-season model.

        Final = (1 - blend_weight) * regular + blend_weight * tournament_expert

        Args:
            regular_probs: Probabilities from regular-season model [N]
            X: Feature matrix for tournament expert prediction [N, D]

        Returns:
            Blended probabilities [N]
        """
        if not self.trained:
            return regular_probs

        expert_probs = self.predict_probability(X)

        blended = (
            (1.0 - self.blend_weight) * regular_probs
            + self.blend_weight * expert_probs
        )

        return blended

    def _spread_to_prob(self, spreads: np.ndarray) -> np.ndarray:
        """Convert predicted spreads to probabilities via logistic CDF."""
        return 1.0 / (1.0 + np.exp(-spreads / max(self.sigma, 0.01)))

    def _calibrate_sigma(self, X: np.ndarray, actual_margins: np.ndarray) -> None:
        """Calibrate sigma for Brier score optimization."""
        if self.spread_model is None or len(actual_margins) < 20:
            return

        pred_spreads = self.spread_model.predict(X)
        outcomes = (actual_margins > 0).astype(float)

        best_sigma = self.sigma
        best_brier = float("inf")

        for sigma_candidate in np.linspace(5.0, 20.0, 61):
            probs = 1.0 / (1.0 + np.exp(-pred_spreads / sigma_candidate))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            brier = float(np.mean((probs - outcomes) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_sigma = sigma_candidate

        self.sigma = float(best_sigma)
