"""Women's NCAA Tournament prediction pipeline.

Strictly independent from Men's pipeline.
Women's basketball has different dynamics:
- Higher seed predictability
- Different scoring distributions
- Rebounding margin and A/TO ratio more predictive of upsets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WomensPipelineConfig:
    """Configuration for women's tournament pipeline."""
    year: int = 2026
    model_complexity: str = "simple"
    tournament_shrinkage: float = 0.04  # Lower than men's (more predictable)
    seed_prior_weight: float = 0.40  # Stronger seed prior for women's
    seed_prior_slope: float = 0.19  # Steeper (fewer upsets historically)

    # Ensemble weights (same Margin-First architecture)
    spread_weight: float = 0.55
    lgb_weight: float = 0.15
    xgb_weight: float = 0.15
    logistic_weight: float = 0.15

    # Tournament expert blend
    tournament_expert_weight: float = 0.30

    # Calibration
    calibration_method: str = "temperature"

    # Probability bounds - women's is slightly tighter (more predictable)
    clip_lo: float = 0.001
    clip_hi: float = 0.999

    # Scoring metric
    scoring_metric: str = "brier"

    # Women's-specific feature priorities
    prioritize_rebounding: bool = True
    prioritize_ato_ratio: bool = True


class WomensPipeline:
    """Independent women's tournament prediction pipeline.

    Key differences from Men's:
    - Stronger seed priors (women's bracket is more predictable)
    - Rebounding margin and A/TO ratio weighted more heavily
    - Different scoring distributions
    - Steeper seed-probability curve
    """

    def __init__(self, config: WomensPipelineConfig):
        self.config = config
        self.trained = False
        self.models = {}
        self.feature_names: List[str] = []
        self.scaler = None
        self.calibrator = None
        self.tournament_expert = None
        self.seed_matchup_prior = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        margins_train: np.ndarray,
        feature_names: List[str],
        sample_weights: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        margins_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train the women's ensemble pipeline.

        Same architecture as men's (SpreadRegressor-dominant) but with
        women's-specific calibration and stronger seed priors.
        """
        from ..ml.ensemble.cfa import LightGBMRanker, XGBoostRanker
        from ..ml.ensemble.spread_model import SpreadRegressor

        stats = {"pipeline": "womens"}
        self.feature_names = feature_names

        # StandardScaler
        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        except ImportError:
            X_scaled = X_train
            X_val_scaled = X_val

        # Train SpreadRegressor (primary, 55%)
        try:
            spread = SpreadRegressor(sigma=11.0)
            spread_stats = spread.train(
                X_scaled, margins_train,
                feature_names=feature_names,
                num_rounds=200,
                sample_weight=sample_weights,
                valid_X=X_val_scaled,
                valid_margins=margins_val,
            )
            self.models["spread"] = spread
            stats["spread"] = spread_stats
        except Exception as e:
            logger.warning("Women's SpreadRegressor failed: %s", e)

        # Train LightGBM (15%)
        try:
            lgb = LightGBMRanker()
            valid_set = (X_val_scaled, y_val) if X_val_scaled is not None and y_val is not None else None
            lgb.train(X_scaled, y_train, feature_names=feature_names,
                      num_rounds=500, valid_set=valid_set, sample_weight=sample_weights)
            self.models["lgb"] = lgb
            stats["lgb_trained"] = True
        except Exception as e:
            logger.warning("Women's LightGBM failed: %s", e)

        # Train XGBoost (15%)
        try:
            xgb = XGBoostRanker()
            valid_set = (X_val_scaled, y_val) if X_val_scaled is not None and y_val is not None else None
            xgb.train(X_scaled, y_train, feature_names=feature_names,
                      num_rounds=500, valid_set=valid_set, sample_weight=sample_weights)
            self.models["xgb"] = xgb
            stats["xgb_trained"] = True
        except Exception as e:
            logger.warning("Women's XGBoost failed: %s", e)

        # Train Logistic Regression (15%)
        try:
            from sklearn.linear_model import LogisticRegression
            logit = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', penalty='l2')
            logit.fit(X_scaled, y_train, sample_weight=sample_weights)
            self.models["logistic"] = logit
            stats["logistic_trained"] = True
        except Exception as e:
            logger.warning("Women's Logistic failed: %s", e)

        self.trained = True
        stats["n_models"] = len(self.models)
        return stats

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win probabilities for women's matchups."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X) if self.scaler else X

        weights = {
            "spread": self.config.spread_weight,
            "lgb": self.config.lgb_weight,
            "xgb": self.config.xgb_weight,
            "logistic": self.config.logistic_weight,
        }

        total_weight = 0.0
        result = np.zeros(len(X_scaled))

        for name, model in self.models.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            if name == "spread":
                preds = model.predict_probability(X_scaled)
            elif name == "logistic":
                preds = model.predict_proba(X_scaled)[:, 1]
            else:
                preds = model.predict(X_scaled)
            result += w * preds
            total_weight += w

        if total_weight > 0:
            result /= total_weight

        return np.clip(result, self.config.clip_lo, self.config.clip_hi)

    def predict_single(self, team1_features: np.ndarray, team2_features: np.ndarray) -> float:
        """Predict single matchup probability (team1 wins)."""
        diff = team1_features - team2_features
        return float(self.predict(diff.reshape(1, -1))[0])
