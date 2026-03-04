"""Margin-First Ensemble Architecture.

Implements the "Raddar-style" modeling approach where the primary
prediction path is:
1. Predict point margin (SpreadRegressor)
2. Convert margin to probability via Logistic CDF
3. Calibrate CDF parameter (k, sigma) per tournament round

Ensemble weights:
- SpreadRegressor: 55% (primary)
- LightGBM: 15%
- XGBoost: 15%
- Logistic Regression: 15%

This is blended with the TournamentExpert at 0.30 weight.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default ensemble weights (Margin-First architecture)
_MARGIN_FIRST_WEIGHTS = {
    "spread": 0.55,
    "lgb": 0.15,
    "xgb": 0.15,
    "logistic": 0.15,
}

# Round-specific sigma values (tighter in later rounds)
# Later rounds have closer matchups -> need different calibration
DEFAULT_ROUND_SIGMAS = {
    "R64": 11.0,
    "R32": 10.5,
    "S16": 10.0,
    "E8":  9.5,
    "F4":  9.0,
    "NCG": 8.5,
}


class RoundSpecificCalibrator:
    """Calibrate logistic CDF parameters per tournament round.

    Later rounds feature closer matchups between better teams,
    requiring tighter sigma values. Uses Optuna for optimization
    when available.
    """

    def __init__(self):
        self.round_params: Dict[str, Dict[str, float]] = {}
        self.calibrated = False

    def calibrate(
        self,
        spreads_by_round: Dict[str, np.ndarray],
        outcomes_by_round: Dict[str, np.ndarray],
        use_optuna: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Calibrate k and sigma per round.

        P(Win) = 1 / (1 + exp(-(k * Margin / sigma)))

        Args:
            spreads_by_round: Dict of round -> predicted spreads [N]
            outcomes_by_round: Dict of round -> binary outcomes [N]
            use_optuna: Whether to use Optuna for optimization

        Returns:
            Dict of round -> {"k": float, "sigma": float, "brier": float}
        """
        for round_name in spreads_by_round:
            spreads = spreads_by_round[round_name]
            outcomes = outcomes_by_round[round_name]

            if len(spreads) < 10:
                # Too few samples; use defaults
                self.round_params[round_name] = {
                    "k": 1.0,
                    "sigma": DEFAULT_ROUND_SIGMAS.get(round_name, 11.0),
                    "brier": None,
                    "n_samples": len(spreads),
                }
                continue

            if use_optuna:
                params = self._optimize_with_optuna(spreads, outcomes, round_name)
            else:
                params = self._optimize_grid(spreads, outcomes, round_name)

            self.round_params[round_name] = params

        self.calibrated = True
        logger.info("Round-specific calibration: %s", {
            r: f"k={p['k']:.3f}, sigma={p['sigma']:.1f}"
            for r, p in self.round_params.items()
        })

        return self.round_params

    def convert_spread_to_prob(
        self,
        spread: float,
        round_name: str = "R64",
    ) -> float:
        """Convert a single spread to probability using round-specific params.

        Args:
            spread: Predicted point spread
            round_name: Tournament round ("R64", "R32", "S16", "E8", "F4", "NCG")

        Returns:
            Win probability
        """
        params = self.round_params.get(round_name, {})
        k = params.get("k", 1.0)
        sigma = params.get("sigma", DEFAULT_ROUND_SIGMAS.get(round_name, 11.0))

        return 1.0 / (1.0 + math.exp(-(k * spread / max(sigma, 0.01))))

    def convert_spreads_to_probs(
        self,
        spreads: np.ndarray,
        round_name: str = "R64",
    ) -> np.ndarray:
        """Batch version of spread-to-prob conversion."""
        params = self.round_params.get(round_name, {})
        k = params.get("k", 1.0)
        sigma = params.get("sigma", DEFAULT_ROUND_SIGMAS.get(round_name, 11.0))

        return 1.0 / (1.0 + np.exp(-(k * spreads / max(sigma, 0.01))))

    def _optimize_with_optuna(
        self,
        spreads: np.ndarray,
        outcomes: np.ndarray,
        round_name: str,
    ) -> Dict:
        """Use Optuna to find optimal k and sigma for a round."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return self._optimize_grid(spreads, outcomes, round_name)

        def objective(trial):
            k = trial.suggest_float("k", 0.5, 2.0)
            sigma = trial.suggest_float("sigma", 4.0, 18.0)

            probs = 1.0 / (1.0 + np.exp(-(k * spreads / max(sigma, 0.01))))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            brier = float(np.mean((probs - outcomes) ** 2))
            return brier

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, timeout=30)

        best = study.best_params
        best_probs = 1.0 / (1.0 + np.exp(
            -(best["k"] * spreads / max(best["sigma"], 0.01))
        ))
        best_brier = float(np.mean((np.clip(best_probs, 1e-7, 1-1e-7) - outcomes) ** 2))

        return {
            "k": best["k"],
            "sigma": best["sigma"],
            "brier": best_brier,
            "n_samples": len(spreads),
        }

    def _optimize_grid(
        self,
        spreads: np.ndarray,
        outcomes: np.ndarray,
        round_name: str,
    ) -> Dict:
        """Grid search fallback for k and sigma optimization."""
        best_brier = float("inf")
        best_k = 1.0
        best_sigma = DEFAULT_ROUND_SIGMAS.get(round_name, 11.0)

        for k in np.linspace(0.5, 2.0, 31):
            for sigma in np.linspace(5.0, 18.0, 27):
                probs = 1.0 / (1.0 + np.exp(-(k * spreads / max(sigma, 0.01))))
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                brier = float(np.mean((probs - outcomes) ** 2))
                if brier < best_brier:
                    best_brier = brier
                    best_k = k
                    best_sigma = sigma

        return {
            "k": float(best_k),
            "sigma": float(best_sigma),
            "brier": float(best_brier),
            "n_samples": len(spreads),
        }


class MarginFirstEnsemble:
    """Margin-First ensemble with SpreadRegressor dominance.

    Architecture:
    1. SpreadRegressor predicts margin -> convert to P(win) via logistic CDF
    2. LightGBM, XGBoost, Logistic provide complementary signals
    3. Fixed-weight combination (55/15/15/15)
    4. TournamentExpert blend at 0.30
    5. Round-specific sigma calibration
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        tournament_expert_weight: float = 0.30,
    ):
        self.weights = weights or dict(_MARGIN_FIRST_WEIGHTS)
        self.tournament_expert_weight = tournament_expert_weight
        self.models: Dict[str, object] = {}
        self.round_calibrator = RoundSpecificCalibrator()
        self.tournament_expert = None

    def set_models(
        self,
        spread_model=None,
        lgb_model=None,
        xgb_model=None,
        logistic_model=None,
        tournament_expert=None,
    ):
        """Register trained models."""
        if spread_model is not None:
            self.models["spread"] = spread_model
        if lgb_model is not None:
            self.models["lgb"] = lgb_model
        if xgb_model is not None:
            self.models["xgb"] = xgb_model
        if logistic_model is not None:
            self.models["logistic"] = logistic_model
        if tournament_expert is not None:
            self.tournament_expert = tournament_expert

    def predict(
        self,
        X: np.ndarray,
        round_name: str = "R64",
    ) -> np.ndarray:
        """Predict win probabilities using margin-first ensemble.

        Args:
            X: Feature matrix [N, D]
            round_name: Tournament round for sigma calibration

        Returns:
            Win probabilities [N]
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = len(X)
        result = np.zeros(n)
        total_weight = 0.0

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue

            if name == "spread":
                # Use round-specific calibration if available
                if hasattr(model, 'predict_spread'):
                    spreads = model.predict_spread(X)
                    if self.round_calibrator.calibrated:
                        preds = self.round_calibrator.convert_spreads_to_probs(
                            spreads, round_name
                        )
                    else:
                        preds = model.predict_probability(X)
                else:
                    preds = model.predict_probability(X)
            elif name == "logistic":
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)

            result += w * np.asarray(preds).flatten()
            total_weight += w

        if total_weight > 0:
            result /= total_weight

        # Blend with TournamentExpert if available
        if self.tournament_expert is not None and self.tournament_expert.trained:
            expert_probs = self.tournament_expert.predict_probability(X)
            result = (
                (1.0 - self.tournament_expert_weight) * result
                + self.tournament_expert_weight * expert_probs
            )

        return result

    def predict_single(
        self,
        features: np.ndarray,
        round_name: str = "R64",
    ) -> float:
        """Predict single matchup."""
        return float(self.predict(features.reshape(1, -1), round_name)[0])

    def get_diagnostics(self) -> Dict:
        """Return ensemble diagnostics."""
        return {
            "weights": dict(self.weights),
            "n_models": len(self.models),
            "models_available": list(self.models.keys()),
            "tournament_expert_active": (
                self.tournament_expert is not None
                and self.tournament_expert.trained
            ),
            "round_calibrated": self.round_calibrator.calibrated,
            "round_params": dict(self.round_calibrator.round_params),
        }
