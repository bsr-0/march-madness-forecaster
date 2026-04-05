"""LambdaMART learning-to-rank model for tournament matchup prediction.

Uses LightGBM's ``lambdarank`` objective to optimize pairwise ordering of
teams rather than pointwise win probability.  This is particularly suited
to bracket optimization where *relative ranking* matters more than
calibrated probabilities.

The model produces relevance scores that are converted to probabilities
via a sigmoid transform for ensemble compatibility.

Implements Agent Directive V7 S7 (ranking model family).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LambdaMARTRanker:
    """LightGBM-based learning-to-rank model for matchup prediction.

    Groups matchups by tournament round so that the ranking loss
    optimizes ordering within each bracket region.  Falls back to
    binary classification when group structure is unavailable.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LambdaMARTRanker")

        self.params: Dict[str, Any] = params or {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [4, 8],
            "boosting_type": "gbdt",
            "num_leaves": 8,
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
        self.model: Any = None
        self.feature_names: Optional[List[str]] = None
        self._fallback_binary = False

    # ------------------------------------------------------------------
    # Group construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_groups(
        round_ids: Optional[np.ndarray] = None,
        n_samples: int = 0,
        min_group_size: int = 2,
    ) -> Optional[np.ndarray]:
        """Build query-group sizes from round/region identifiers.

        Args:
            round_ids: Per-sample group identifier (e.g. tournament round
                concatenated with region).  Samples must be sorted by group.
            n_samples: Total number of samples (used for fallback).
            min_group_size: Minimum samples per group.

        Returns:
            Array of group sizes suitable for LightGBM, or ``None`` if
            valid groups cannot be formed.
        """
        if round_ids is None:
            return None

        groups: List[int] = []
        current = round_ids[0]
        count = 1
        for rid in round_ids[1:]:
            if rid == current:
                count += 1
            else:
                groups.append(count)
                current = rid
                count = 1
        groups.append(count)

        if any(g < min_group_size for g in groups):
            return None

        return np.array(groups, dtype=np.int32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_rounds: int = 300,
        early_stopping_rounds: Optional[int] = 30,
        valid_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        valid_groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the LambdaMART model.

        Args:
            X: Feature matrix ``[N, D]``.
            y: Relevance labels ``[N]`` (1 = win, 0 = loss).
            feature_names: Optional feature names.
            num_rounds: Maximum boosting rounds.
            early_stopping_rounds: Patience for early stopping.
            valid_set: ``(X_val, y_val)`` tuple.
            sample_weight: Per-sample weights.
            groups: Query-group sizes for training data.
            valid_groups: Query-group sizes for validation data.

        Returns:
            Training statistics dict.
        """
        self.feature_names = feature_names

        # Decide whether to use ranking or fallback to binary
        if groups is None or len(groups) == 0:
            logger.info(
                "No valid groups provided — falling back to binary classification"
            )
            self._fallback_binary = True
            params = dict(self.params)
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
        else:
            self._fallback_binary = False
            params = dict(self.params)

        train_data = lgb.Dataset(
            X, label=y, feature_name=feature_names, weight=sample_weight,
            group=groups if not self._fallback_binary else None,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if valid_set is not None:
            valid_data = lgb.Dataset(
                valid_set[0],
                label=valid_set[1],
                feature_name=feature_names,
                reference=train_data,
                group=valid_groups if not self._fallback_binary else None,
            )
            valid_sets.append(valid_data)
            valid_names.append("valid")

        callbacks: list = []
        if early_stopping_rounds and valid_set is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=100))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        return {
            "model": "lambda_mart",
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_groups": len(groups) if groups is not None else 0,
            "fallback_binary": self._fallback_binary,
            "best_iteration": self.model.best_iteration,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw relevance scores."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return win probabilities via sigmoid of relevance scores.

        When the model fell back to binary classification, LightGBM
        already outputs probabilities — no transform needed.
        """
        raw = self.predict(X)
        if self._fallback_binary:
            return raw
        # Sigmoid transform: 1 / (1 + exp(-x))
        return 1.0 / (1.0 + np.exp(-raw))

    def predict_matchup(
        self, team1_features: np.ndarray, team2_features: np.ndarray
    ) -> float:
        """Predict single matchup probability (team 1 wins)."""
        diff = team1_features - team2_features
        return float(self.predict_proba(diff.reshape(1, -1))[0])

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance (gain-based)."""
        if self.model is None:
            return {}
        importance = self.model.feature_importance(importance_type="gain")
        if self.feature_names:
            return dict(zip(self.feature_names, importance.tolist()))
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is not None:
            self.model.save_model(filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = lgb.Booster(model_file=filepath)
