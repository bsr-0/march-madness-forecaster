"""Men's NCAA Tournament prediction pipeline.

Strictly independent from Women's pipeline - no shared features,
no shared models, no shared calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MensPipelineConfig:
    """Configuration for men's tournament pipeline."""
    year: int = 2026
    model_complexity: str = "simple"
    tournament_shrinkage: float = 0.06  # Increased from 0.02 for domain shift
    seed_prior_weight: float = 0.15
    seed_prior_slope: float = 0.175

    # Ensemble weights (Margin-First architecture)
    spread_weight: float = 0.55  # SpreadRegressor dominance
    lgb_weight: float = 0.15
    xgb_weight: float = 0.15
    logistic_weight: float = 0.15

    # Tournament expert blend
    tournament_expert_weight: float = 0.30

    # Calibration
    calibration_method: str = "temperature"

    # Probability bounds
    clip_lo: float = 0.001  # Aggressive for Brier
    clip_hi: float = 0.999

    # Feature set
    feature_set: str = "simple"  # "simple" or "standard"

    # Scoring metric
    scoring_metric: str = "brier"

    # Data paths
    teams_json: Optional[str] = None
    torvik_json: Optional[str] = None
    historical_games_json: Optional[str] = None

    # Multi-year training
    enable_multi_year_training: bool = True
    training_year_decay: float = 0.85

    # LOYO validation
    enable_loyo_cv: bool = True
    loyo_years: Optional[List[int]] = None


class MensPipeline:
    """Independent men's tournament prediction pipeline.

    Encapsulates all men's-specific logic:
    - Men's feature engineering
    - Men's model training (SpreadRegressor-dominant ensemble)
    - Men's calibration
    - Men's tournament domain adaptation

    This pipeline shares NO features, models, or calibration with
    the Women's pipeline.
    """

    def __init__(self, config: MensPipelineConfig):
        self.config = config
        self.trained = False
        self.models = {}
        self.feature_names: List[str] = []
        self.scaler = None
        self.calibrator = None
        self.tournament_expert = None
        self.seed_matchup_prior = None  # Historical seed-pairing win rates

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
        """Train the men's ensemble pipeline.

        Trains all base models and sets up the fixed-weight ensemble
        with SpreadRegressor as the dominant model (55% weight).

        Returns:
            Dict with training stats and diagnostics.
        """
        from ..ml.ensemble.cfa import LightGBMRanker, XGBoostRanker
        from ..ml.ensemble.spread_model import SpreadRegressor

        stats = {"pipeline": "mens"}
        self.feature_names = feature_names

        # 1. Optional StandardScaler
        try:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        except ImportError:
            X_scaled = X_train
            X_val_scaled = X_val

        # 2. Train SpreadRegressor (primary model, 55% weight)
        try:
            spread = SpreadRegressor(sigma=self.config.tournament_shrinkage * 100 + 5)
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
            logger.info("SpreadRegressor trained: %s", spread_stats)
        except Exception as e:
            logger.warning("SpreadRegressor training failed: %s", e)

        # 3. Train LightGBM classifier (15%)
        try:
            lgb = LightGBMRanker()
            valid_set = (X_val_scaled, y_val) if X_val_scaled is not None and y_val is not None else None
            lgb.train(
                X_scaled, y_train,
                feature_names=feature_names,
                num_rounds=500,
                valid_set=valid_set,
                sample_weight=sample_weights,
            )
            self.models["lgb"] = lgb
            stats["lgb_trained"] = True
        except Exception as e:
            logger.warning("LightGBM training failed: %s", e)

        # 4. Train XGBoost classifier (15%)
        try:
            xgb = XGBoostRanker()
            valid_set = (X_val_scaled, y_val) if X_val_scaled is not None and y_val is not None else None
            xgb.train(
                X_scaled, y_train,
                feature_names=feature_names,
                num_rounds=500,
                valid_set=valid_set,
                sample_weight=sample_weights,
            )
            self.models["xgb"] = xgb
            stats["xgb_trained"] = True
        except Exception as e:
            logger.warning("XGBoost training failed: %s", e)

        # 5. Train Logistic Regression (15%)
        try:
            from sklearn.linear_model import LogisticRegression
            logit = LogisticRegression(
                C=0.1, max_iter=1000, solver='lbfgs',
                penalty='l2'
            )
            logit.fit(X_scaled, y_train, sample_weight=sample_weights)
            self.models["logistic"] = logit
            stats["logistic_trained"] = True
        except Exception as e:
            logger.warning("Logistic regression training failed: %s", e)

        self.trained = True
        stats["n_models"] = len(self.models)
        return stats

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win probabilities using the margin-first ensemble.

        Args:
            X: Feature matrix [N, D] or single sample [D]

        Returns:
            Win probabilities [N]
        """
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

        # Clip to bounds
        result = np.clip(result, self.config.clip_lo, self.config.clip_hi)

        return result

    def predict_single(self, team1_features: np.ndarray, team2_features: np.ndarray) -> float:
        """Predict single matchup probability (team1 wins)."""
        diff = team1_features - team2_features
        return float(self.predict(diff.reshape(1, -1))[0])
