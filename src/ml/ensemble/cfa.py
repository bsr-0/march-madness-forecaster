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
        Calculate dynamic weights via confidence-scaled base weights.

        Weight calculation:
        1. Start from base weights (learned from historical Brier scores
           when available, else from config).
        2. Scale each weight by model confidence: w_i *= (0.5 + confidence_i).
           This maps confidence [0,1] → multiplier [0.5, 1.5], keeping the
           adjustment symmetric and bounded.
        3. Normalize to sum to 1.

        FIX 1.1: Per-prediction diversity bonus REMOVED.  With only 3 models,
        the diversity bonus effectively upweighted whichever model disagreed
        most on a specific matchup — rewarding noise rather than signal.
        Dataset-level weight optimization via EnsembleWeightOptimizer already
        captures optimal static weights based on Brier performance.

        Args:
            predictions: Model predictions

        Returns:
            Weight dictionary
        """
        if not predictions:
            return {}

        # Confidence-scaled base weights.
        # Multiplier maps confidence ∈ [0,1] → [0.5, 1.5] (symmetric, bounded).
        raw_weights: Dict[str, float] = {}
        for model_name, pred in predictions.items():
            base_weight = self.base_weights.get(model_name, 0.0)
            confidence_multiplier = 0.5 + pred.confidence
            raw_weights[model_name] = base_weight * confidence_multiplier

        # Normalize to sum to 1
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in raw_weights.items()}
        return {k: 1.0 / len(raw_weights) for k in raw_weights}
    
    def compute_diversity_metrics(
        self,
        predictions: Dict[str, ModelPrediction],
    ) -> Dict[str, float]:
        """
        Compute ensemble diversity metrics for a single matchup.

        Diversity measures how much base learners disagree.  Higher diversity
        → more potential for ensemble benefit (reducing variance), but only
        if the models are individually competent.

        Returns a dict with:
        - 'prediction_spread': max - min predicted probability
        - 'prediction_std': standard deviation across models
        - 'ensemble_mean': weighted mean prediction
        - Per-model 'deviation_<name>': |pred - ensemble_mean|
        """
        if len(predictions) < 2:
            return {"prediction_spread": 0.0, "prediction_std": 0.0}

        probs = [pred.win_probability for pred in predictions.values()]
        weights = self._calculate_weights(predictions)
        ensemble_mean = sum(
            weights.get(name, 0) * pred.win_probability
            for name, pred in predictions.items()
        )

        metrics = {
            "prediction_spread": float(max(probs) - min(probs)),
            "prediction_std": float(np.std(probs)),
            "ensemble_mean": float(ensemble_mean),
        }
        for name, pred in predictions.items():
            metrics[f"deviation_{name}"] = float(abs(pred.win_probability - ensemble_mean))

        return metrics

    @staticmethod
    def compute_pairwise_correlation(
        model_predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute pairwise Spearman rank correlations between base learner
        prediction vectors across all matchups.

        Useful for diagnosing ensemble diversity at the dataset level:
        - r ≈ 1.0: models are redundant (ensemble provides little benefit)
        - r ≈ 0.5-0.8: healthy diversity (ensemble reduces variance)
        - r < 0.3: models may be learning different signals (investigate)

        Args:
            model_predictions: Dict of model_name -> prediction_array [N]

        Returns:
            Dict of "model_a_vs_model_b" -> spearman_r
        """
        try:
            from scipy.stats import spearmanr
            scipy_available = True
        except ImportError:
            scipy_available = False

        names = sorted(model_predictions.keys())
        result = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = model_predictions[names[i]]
                b = model_predictions[names[j]]
                if len(a) != len(b) or len(a) < 5:
                    continue
                if scipy_available:
                    corr, _ = spearmanr(a, b)
                else:
                    corr = float(np.corrcoef(a, b)[0, 1])
                result[f"{names[i]}_vs_{names[j]}"] = round(float(corr), 4)

        return result

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



# FIX 1.2: SOTAEnsemble class REMOVED — was unused dead code with
# hardcoded weights for 5 models that diverged from the actual 3-model
# ensemble (baseline, gnn, transformer) used in the pipeline.  All
# ensemble logic now lives in CombinatorialFusionAnalysis + the
# EnsembleWeightOptimizer in hyperparameter_tuning.py.
