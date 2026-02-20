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


class CombinatorialFusionAnalysis:
    """
    Combines predictions from multiple models using CFA.
    
    Final Score = α * P_GNN + β * P_Trans + (1-α-β) * P_Baseline
    
    where α and β are dynamically determined based on model confidence.
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float] = None,
        confidence_scaling: float = 0.5
    ):
        """
        Initialize CFA ensemble.
        
        Args:
            base_weights: Default weights for each model
            confidence_scaling: How much confidence affects weight (0=ignore, 1=fully)
        """
        self.base_weights = base_weights or {
            "gnn": 0.35,
            "transformer": 0.35,
            "baseline": 0.30,
        }
        self.confidence_scaling = confidence_scaling
        
        # Historical performance tracking
        self.model_accuracy: Dict[str, List[Tuple[float, bool]]] = {}
    
    def predict(
        self,
        predictions: Dict[str, ModelPrediction]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combine model predictions using CFA.
        
        Args:
            predictions: Dict of model_name -> ModelPrediction
            
        Returns:
            Tuple of (combined_probability, model_weights)
        """
        # Calculate dynamic weights
        weights = self._calculate_weights(predictions)
        
        # Combine predictions
        combined_prob = 0.0
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.0)
            combined_prob += weight * pred.win_probability
        
        return combined_prob, weights
    
    def _calculate_weights(
        self,
        predictions: Dict[str, ModelPrediction]
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights via confidence-scaled base weights with a
        bounded diversity bonus.

        Weight calculation:
        1. Start from base weights (learned from historical Brier scores
           when available, else from config).
        2. Scale each weight by model confidence: w_i *= (0.5 + confidence_i).
           This maps confidence [0,1] → multiplier [0.5, 1.5], keeping the
           adjustment symmetric and bounded.
        3. Apply a *bounded* diversity bonus.  For each model, compute its
           deviation from the initial weighted mean.  The bonus is capped at
           ``max_diversity_ratio`` times the base weight, preventing any single
           outlier model from dominating.

        The key mathematical fix: the diversity bonus can never exceed
        ``max_diversity_ratio * base_weight``, ensuring convergent weights
        regardless of how extreme a model's prediction is.

        Args:
            predictions: Model predictions

        Returns:
            Weight dictionary
        """
        if not predictions:
            return {}

        # Stage 1: confidence-scaled base weights.
        # Multiplier maps confidence ∈ [0,1] → [0.5, 1.5] (symmetric, bounded).
        raw_weights: Dict[str, float] = {}
        for model_name, pred in predictions.items():
            base_weight = self.base_weights.get(model_name, 0.0)
            confidence_multiplier = 0.5 + pred.confidence
            raw_weights[model_name] = base_weight * confidence_multiplier

        # Stage 2: bounded diversity bonus.
        # Cap bonus at max_diversity_ratio * base_weight to prevent unbounded
        # upweighting of extreme-but-confident models.
        max_diversity_ratio = 0.20  # Max 20% bonus from diversity
        total_raw = sum(raw_weights.values())
        if total_raw > 0:
            norm_raw = {k: v / total_raw for k, v in raw_weights.items()}
            ensemble_mean = sum(
                norm_raw[m] * predictions[m].win_probability for m in predictions
            )

            for model_name, pred in predictions.items():
                deviation = abs(pred.win_probability - ensemble_mean)
                base_w = self.base_weights.get(model_name, 0.0)
                # Diversity bonus = deviation * confidence * scaling,
                # but hard-capped at max_diversity_ratio * base_weight.
                uncapped_bonus = deviation * pred.confidence * self.confidence_scaling
                max_bonus = max_diversity_ratio * base_w
                raw_weights[model_name] += min(uncapped_bonus, max_bonus)

        # Normalize to sum to 1
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in raw_weights.items()}
        return {k: 1.0 / len(raw_weights) for k in raw_weights}
    
    def update_accuracy(
        self,
        model_name: str,
        predicted_prob: float,
        actual_outcome: bool
    ) -> None:
        """
        Track model accuracy for weight optimization.
        
        Args:
            model_name: Name of model
            predicted_prob: Predicted probability
            actual_outcome: Actual result (True = team1 won)
        """
        if model_name not in self.model_accuracy:
            self.model_accuracy[model_name] = []
        
        self.model_accuracy[model_name].append((predicted_prob, actual_outcome))
    
    def optimize_weights(self) -> Dict[str, float]:
        """
        Optimize base weights using historical Brier scores.

        Uses inverse-Brier softmax weighting: each model's weight is
        proportional to exp(-beta * brier_i), where beta is a sharpness
        parameter. This is more principled than the naive ``1 - brier``
        because:

        1. It's always positive (no risk of negative weights).
        2. The exponential form corresponds to the Bayesian posterior over
           model weights under a Gaussian likelihood approximation.
        3. The beta parameter controls how aggressively we concentrate weight
           on the best model vs. spreading weight evenly.

        A minimum weight floor prevents any model from being zeroed out,
        which maintains ensemble diversity.

        Returns:
            Optimized weights
        """
        if not self.model_accuracy:
            return self.base_weights

        model_brier: Dict[str, float] = {}
        for model_name, history in self.model_accuracy.items():
            if not history:
                continue
            brier = float(np.mean([
                (pred - float(actual)) ** 2
                for pred, actual in history
            ]))
            model_brier[model_name] = brier

        if not model_brier:
            return self.base_weights

        # FIX #D: STATISTICAL SIGNIFICANCE GUARD
        #
        # Before applying inverse-Brier softmax, check whether the Brier
        # score differences across models are statistically meaningful.
        # The standard error of a Brier score is approximately:
        #   SE(brier) ≈ sqrt(brier * (1 - brier) / n) for binary outcomes.
        # If the RANGE of model Brier scores is smaller than the SE of the
        # best model's Brier, the differences are noise — concentrating
        # weight on one model is pure overfitting.  In that case, keep
        # existing weights (or uniform if not set).
        brier_vals = np.array([model_brier[m] for m in model_brier])
        brier_range = float(np.max(brier_vals) - np.min(brier_vals))
        best_brier = float(np.min(brier_vals))
        # Approximate sample size from first model's history
        sample_sizes = [len(h) for h in self.model_accuracy.values() if h]
        n_eval = min(sample_sizes) if sample_sizes else 0
        if n_eval > 0:
            se_brier = float(np.sqrt(best_brier * (1.0 - best_brier) / n_eval))
        else:
            se_brier = 1.0  # Huge SE → fall through to uniform

        if brier_range < se_brier:
            # Differences not significant — keep current weights unchanged.
            logger.debug(
                "Brier range (%.4f) < SE (%.4f); keeping existing weights.",
                brier_range, se_brier,
            )
            return self.base_weights

        # Inverse-Brier softmax: w_i ∝ exp(-beta * brier_i).
        # FIX #D (cont.): Reduce beta from 10→5 to avoid over-concentrating
        # weight based on small Brier differences that pass the significance
        # test but are still noisy.  beta=5 gives a model with Brier 0.20
        # about 1.65x the weight of one with Brier 0.30.
        beta = 5.0
        min_weight = 0.05  # Floor: no model drops below 5%
        # Shift for numerical stability (subtract min so max exponent ≈ 0)
        shifted = -beta * (brier_vals - np.min(brier_vals))
        exp_vals = np.exp(shifted)
        softmax_weights = exp_vals / np.sum(exp_vals)

        # Apply minimum weight floor, then renormalize
        names = list(model_brier.keys())
        raw = {name: max(min_weight, float(w))
               for name, w in zip(names, softmax_weights)}
        total = sum(raw.values())
        self.base_weights = {k: v / total for k, v in raw.items()}

        return self.base_weights


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
        
        self.params = params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
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

        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
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


class SOTAEnsemble:
    """
    State-of-the-art ensemble combining all prediction methods.
    
    Integrates:
    - GNN-based schedule analysis
    - Transformer-based temporal modeling
    - LightGBM ranking
    - Seed baseline
    - Elo ratings
    """
    
    def __init__(self):
        """Initialize SOTA ensemble."""
        self.cfa = CombinatorialFusionAnalysis()
        self.lgb_ranker = None
        
        # Sub-model weights
        self.model_weights = {
            "gnn": 0.25,
            "transformer": 0.20,
            "lightgbm": 0.25,
            "elo": 0.15,
            "seed_baseline": 0.15,
        }
    
    def predict(
        self,
        team1_id: str,
        team2_id: str,
        gnn_prob: float,
        transformer_prob: float,
        lgb_prob: float,
        elo_prob: float,
        seed_prob: float,
        confidence_scores: Dict[str, float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Generate ensemble prediction.
        
        Args:
            team1_id: Team 1 identifier
            team2_id: Team 2 identifier
            gnn_prob: GNN model probability (team1 wins)
            transformer_prob: Transformer model probability
            lgb_prob: LightGBM probability
            elo_prob: Elo-based probability
            seed_prob: Seed-based probability
            confidence_scores: Model confidence scores
            
        Returns:
            Tuple of (ensemble_probability, model_contributions)
        """
        confidence_scores = confidence_scores or {
            "gnn": 0.5, "transformer": 0.5, "lightgbm": 0.5,
            "elo": 0.5, "seed_baseline": 0.5
        }
        
        predictions = {
            "gnn": ModelPrediction("gnn", gnn_prob, confidence_scores.get("gnn", 0.5)),
            "transformer": ModelPrediction("transformer", transformer_prob, confidence_scores.get("transformer", 0.5)),
            "lightgbm": ModelPrediction("lightgbm", lgb_prob, confidence_scores.get("lightgbm", 0.5)),
            "elo": ModelPrediction("elo", elo_prob, confidence_scores.get("elo", 0.5)),
            "seed_baseline": ModelPrediction("seed_baseline", seed_prob, confidence_scores.get("seed_baseline", 0.5)),
        }
        
        # Use CFA with all models
        self.cfa.base_weights = self.model_weights
        combined_prob, weights = self.cfa.predict(predictions)
        
        return combined_prob, weights
