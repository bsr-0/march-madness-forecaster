"""
Combinatorial Fusion Analysis (CFA) for ensemble prediction.

Combines predictions from multiple models (GNN, Transformer, Baseline)
using dynamic weights based on model confidence.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    
    model_name: str
    win_probability: float
    confidence: float  # Model's confidence in its prediction (0-1)
    features: Optional[Dict[str, float]] = None


class LightGBMRanker:
    """
    LightGBM-based ranking model for matchup prediction.
    
    Uses gradient boosting on Four Factors and other features
    to predict game outcomes.
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize LightGBM ranker.
        
        Args:
            params: LightGBM parameters
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        # OOS-FIX: Conservative defaults — num_leaves=8 and
        # min_child_samples=50 force shallow, well-regularized trees
        # appropriate for ~400 training samples.
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 8,
            'learning_rate': 0.05,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'num_threads': 1,
        }
        
        self.model = None
        self.feature_names = None
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        num_rounds: int = 500,
        early_stopping_rounds: Optional[int] = 50,
        valid_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: np.ndarray = None,
    ) -> None:
        """
        Train LightGBM model.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N] (1 = team1 win)
            feature_names: Names of features
            num_rounds: Number of boosting rounds
            early_stopping_rounds: Early stopping patience (None to disable)
            valid_set: Validation set (X_val, y_val)
            sample_weight: Per-sample weights [N] for recency weighting
        """
        if X.shape[0] == 0:
            raise ValueError("Empty training set")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Training data contains NaN/Inf values")
        if X.shape[0] < 10:
            logger.warning("Very small training set (%d samples) for LightGBM", X.shape[0])
        self.feature_names = feature_names

        train_data = lgb.Dataset(X, label=y, feature_name=feature_names,
                                 weight=sample_weight)

        valid_sets = [train_data]
        valid_names = ['train']

        if valid_set is not None:
            valid_data = lgb.Dataset(
                valid_set[0],
                label=valid_set[1],
                feature_name=feature_names,
                reference=train_data
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')

        callbacks = []
        # Only add early stopping when a real validation set is provided.
        # Monitoring training loss alone is meaningless for early stopping.
        if early_stopping_rounds and valid_set is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=100))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict win probabilities.
        
        Args:
            X: Feature matrix [N, D]
            
        Returns:
            Predicted probabilities [N]
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def predict_matchup(
        self,
        team1_features: np.ndarray,
        team2_features: np.ndarray
    ) -> float:
        """
        Predict single matchup probability.
        
        Args:
            team1_features: Team 1 feature vector
            team2_features: Team 2 feature vector
            
        Returns:
            Probability that team 1 wins
        """
        # Compute differential features
        diff_features = team1_features - team2_features
        return float(self.predict(diff_features.reshape(1, -1))[0])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict of feature_name -> importance
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is not None:
            self.model.save_model(filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = lgb.Booster(model_file=filepath)


class XGBoostRanker:
    """
    XGBoost-based ranking model for matchup prediction.

    Uses gradient boosting on matchup differential features to predict
    game outcomes. XGBoost is a robust alternative/complement to LightGBM
    and often the top performer in Kaggle March Madness competitions.
    """

    def __init__(self, params: Dict = None):
        """
        Initialize XGBoost ranker.

        Args:
            params: XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")

        # OOS-FIX: Conservative defaults — max_depth=3 and
        # min_child_weight=10 prevent overfitting on small samples.
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 10,
            "gamma": 0.5,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "nthread": 1,
        }

        self.model = None
        self.feature_names = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        num_rounds: int = 500,
        early_stopping_rounds: Optional[int] = 50,
        valid_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: np.ndarray = None,
    ) -> None:
        """
        Train XGBoost model.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N] (1 = team1 win)
            feature_names: Names of features
            num_rounds: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            valid_set: Validation set (X_val, y_val)
            sample_weight: Per-sample weights [N] for recency weighting
        """
        if X.shape[0] == 0:
            raise ValueError("Empty training set")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Training data contains NaN/Inf values")
        if X.shape[0] < 10:
            logger.warning("Very small training set (%d samples) for XGBoost", X.shape[0])
        self.feature_names = feature_names

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names,
                             weight=sample_weight)

        evals = [(dtrain, "train")]
        if valid_set is not None:
            dval = xgb.DMatrix(valid_set[0], label=valid_set[1], feature_names=feature_names)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if valid_set is not None else None,
            verbose_eval=False,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict win probabilities.

        Args:
            X: Feature matrix [N, D]

        Returns:
            Predicted probabilities [N]
        """
        if self.model is None:
            raise ValueError("Model not trained")

        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dmat)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        importance = self.model.get_score(importance_type="gain")
        return importance

    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is not None:
            self.model.save_model(filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = xgb.Booster(model_file=filepath)


def create_matchup_features(
    team1_stats: Dict[str, float],
    team2_stats: Dict[str, float]
) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature vector for a matchup.
    
    Computes differential and interaction features.
    
    Args:
        team1_stats: Team 1 statistics
        team2_stats: Team 2 statistics
        
    Returns:
        Tuple of (feature_vector, feature_names)
    """
    features = []
    names = []
    
    # Standard features to include
    stat_keys = [
        'adj_efficiency_margin', 'adj_offensive_efficiency', 
        'adj_defensive_efficiency', 'adj_tempo',
        'effective_fg_pct', 'turnover_rate', 
        'offensive_reb_rate', 'free_throw_rate',
        'sos_adj_em', 'luck'
    ]
    
    # Differential features (team1 - team2)
    for key in stat_keys:
        val1 = team1_stats.get(key, 0.0)
        val2 = team2_stats.get(key, 0.0)
        
        features.append(val1 - val2)
        names.append(f"diff_{key}")
    
    # Raw features for each team
    for key in stat_keys[:4]:  # Just main efficiency metrics
        features.append(team1_stats.get(key, 0.0))
        names.append(f"team1_{key}")
        features.append(team2_stats.get(key, 0.0))
        names.append(f"team2_{key}")
    
    # Interaction features
    # Tempo matchup (faster vs slower)
    tempo1 = team1_stats.get('adj_tempo', 68)
    tempo2 = team2_stats.get('adj_tempo', 68)
    features.append(tempo1 * tempo2 / 4624)  # Normalized
    names.append('tempo_interaction')
    
    # Style matchup (offense-heavy vs defense-heavy)
    off1 = team1_stats.get('adj_offensive_efficiency', 100)
    def1 = team1_stats.get('adj_defensive_efficiency', 100)
    off2 = team2_stats.get('adj_offensive_efficiency', 100)
    def2 = team2_stats.get('adj_defensive_efficiency', 100)
    
    features.append((off1 - def2) / 10)  # Team1 offense vs Team2 defense
    names.append('t1_off_vs_t2_def')
    features.append((off2 - def1) / 10)  # Team2 offense vs Team1 defense
    names.append('t2_off_vs_t1_def')
    
    return np.array(features), names



class LightGBMMarginRegressor:
    """LightGBM regression model that predicts point margins then converts to win probabilities.

    Implements the "margin-first" training approach (Directive V7 S7-2):
    train a regression model on actual point margins, then convert
    predicted margins to win probabilities via a logistic CDF.

    This approach can outperform direct classification because:
    - Point margins carry more information than binary outcomes
    - The logistic CDF conversion naturally calibrates probabilities
    - Regression on margins is less sensitive to label noise

    Usage:
        model = LightGBMMarginRegressor()
        model.train(X, margins)
        probs = model.predict_proba(X)  # Returns win probabilities
    """

    # Logistic CDF scale parameter.  Empirically, tournament margins
    # have std ~10 points.  scale = std * sqrt(3) / pi ≈ 5.5.
    DEFAULT_LOGISTIC_SCALE = 5.5

    def __init__(self, params: Dict = None, logistic_scale: float = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        self.logistic_scale = logistic_scale or self.DEFAULT_LOGISTIC_SCALE
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 8,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "verbose": -1,
            "num_threads": 1,
        }
        self.model = None
        self.feature_names = None

    def train(
        self,
        X: np.ndarray,
        margins: np.ndarray,
        feature_names: List[str] = None,
        num_rounds: int = 500,
        early_stopping_rounds: Optional[int] = 50,
        valid_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: np.ndarray = None,
    ) -> None:
        """Train on point margins (positive = team1 win).

        Args:
            X: Feature matrix [N, D].
            margins: Point margins [N] (team1_score - team2_score).
            feature_names: Feature names for interpretability.
            num_rounds: Boosting rounds.
            early_stopping_rounds: Early stopping patience.
            valid_set: (X_val, margins_val) for early stopping.
            sample_weight: Per-sample weights.
        """
        self.feature_names = feature_names
        train_data = lgb.Dataset(
            X, label=margins, feature_name=feature_names,
            weight=sample_weight,
        )

        valid_sets = [train_data]
        valid_names = ["train"]
        if valid_set is not None:
            valid_data = lgb.Dataset(
                valid_set[0], label=valid_set[1],
                feature_name=feature_names, reference=train_data,
            )
            valid_sets.append(valid_data)
            valid_names.append("valid")

        callbacks = []
        if early_stopping_rounds and valid_set is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=100))

        self.model = lgb.train(
            self.params, train_data,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Predict raw point margins."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win probabilities via logistic CDF of predicted margins."""
        margins = self.predict_margin(X)
        return self._margin_to_probability(margins)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict() — returns win probabilities."""
        return self.predict(X)

    def _margin_to_probability(self, margins: np.ndarray) -> np.ndarray:
        """Convert predicted margins to win probabilities via logistic CDF.

        P(win) = 1 / (1 + exp(-margin / scale))
        """
        return 1.0 / (1.0 + np.exp(-margins / self.logistic_scale))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type="gain")
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return {f"feature_{i}": imp for i, imp in enumerate(importance)}


# FIX 1.2: SOTAEnsemble class REMOVED — was unused dead code with
# hardcoded weights for 5 models that diverged from the actual 3-model
# ensemble (baseline, gnn, transformer) used in the pipeline.  All
# ensemble logic now lives in CombinatorialFusionAnalysis + the
# EnsembleWeightOptimizer in hyperparameter_tuning.py.
