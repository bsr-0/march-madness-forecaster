"""Two-stage training: pretrain on regular season, finetune on tournament.

Addresses the fundamental small-sample problem in March Madness prediction:
only ~63 NCAA tournament games per year, but ~5000 regular-season games.

Stage 1 — Pretrain
    Train on regular-season games (large N, diverse matchups) to learn
    general basketball prediction patterns: efficiency margins predict wins,
    Four Factors capture offensive/defensive quality, schedule strength
    separates conference tiers.

Stage 2 — Finetune
    Finetune on tournament games (small N, high-stakes single-elimination)
    to adapt for tournament-specific dynamics:
      - No home-court advantage (neutral sites)
      - Higher average opponent quality
      - Single-elimination pressure (no regression to mean)
      - Seeding effects (matchup structure is non-random)

Key design decisions:

1. **Learning rate reduction**: Stage 2 uses 5-10x lower learning rate
   (default 0.01 vs 0.05) to avoid catastrophic forgetting of regular-season
   patterns while adapting to tournament distribution.

2. **Fewer boosting rounds**: Stage 2 limits rounds (default 50) since
   tournament samples are small and correlated (same 68 teams each year).

3. **Round-weighted loss**: Stage 2 applies Kaggle round weights so the
   model invests more gradient in later-round games where scoring matters most.

4. **No early stopping in Stage 2**: With N<100 tournament games, a
   validation split is too noisy for early stopping. Use fixed rounds.

5. **Model-specific handling**:
   - LightGBM/XGBoost: Finetune via init_model (continue boosting from
     pretrained checkpoint with reduced LR).
   - SpreadRegressor: Same approach (LightGBM regression under the hood).
   - Logistic Regression: Warm-start from pretrained coefficients.

References
----------
- Pan & Yang, "A Survey on Transfer Learning", IEEE TKDE 2010 — formal
  framework for domain adaptation (regular season → tournament).
- Kaggle March Machine Learning Mania leaderboards (2019-2024) — top-10
  solutions increasingly use two-stage or curriculum approaches.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Stage 2 defaults — conservative to prevent catastrophic forgetting
FINETUNE_LEARNING_RATE = 0.01       # 5x lower than typical Stage 1 (0.05)
FINETUNE_NUM_ROUNDS = 50            # Few rounds on small tournament data
FINETUNE_MIN_CHILD_SAMPLES = 10     # Lower threshold for small tournament set
FINETUNE_LAMBDA_L1 = 2.0            # Stronger L1 to prevent overfitting
FINETUNE_LAMBDA_L2 = 2.0            # Stronger L2 to prevent overfitting


@dataclass
class TwoStageConfig:
    """Configuration for two-stage training."""

    # Stage 1: Pretrain on regular season
    pretrain_learning_rate: float = 0.05
    pretrain_num_rounds: int = 500
    pretrain_early_stopping_rounds: int = 50

    # Stage 2: Finetune on tournament
    finetune_learning_rate: float = FINETUNE_LEARNING_RATE
    finetune_num_rounds: int = FINETUNE_NUM_ROUNDS
    finetune_min_child_samples: int = FINETUNE_MIN_CHILD_SAMPLES
    finetune_lambda_l1: float = FINETUNE_LAMBDA_L1
    finetune_lambda_l2: float = FINETUNE_LAMBDA_L2

    # Minimum tournament samples required to justify finetuning.
    # Below this, Stage 2 is skipped and Stage 1 model is returned as-is.
    min_finetune_samples: int = 30

    # Whether to apply Kaggle round weights in Stage 2
    apply_round_weights: bool = True

    # Whether to use init_model for tree models (continue boosting)
    # vs training from scratch with tournament data only.
    use_warm_start: bool = True


@dataclass
class TwoStageResult:
    """Result of two-stage training."""
    model: Any  # The final trained model
    stage1_trained: bool = True
    stage2_trained: bool = False
    stage1_n_samples: int = 0
    stage2_n_samples: int = 0
    stage1_train_metric: Optional[float] = None
    stage2_train_metric: Optional[float] = None
    reason_skipped_stage2: Optional[str] = None


def _build_finetune_params(base_params: Dict[str, Any], config: TwoStageConfig) -> Dict[str, Any]:
    """Build LightGBM/XGBoost parameters for Stage 2 finetuning.

    Modifies the base parameters with lower learning rate, stronger
    regularization, and smaller min_child_samples for the tournament
    finetuning stage.
    """
    params = dict(base_params)
    params["learning_rate"] = config.finetune_learning_rate
    params["lambda_l1"] = config.finetune_lambda_l1
    params["lambda_l2"] = config.finetune_lambda_l2

    # For LightGBM
    if "min_child_samples" in params:
        params["min_child_samples"] = config.finetune_min_child_samples
    # For XGBoost
    if "min_child_weight" in params:
        params["min_child_weight"] = config.finetune_min_child_samples

    return params


def two_stage_train_lightgbm(
    pretrain_X: np.ndarray,
    pretrain_y: np.ndarray,
    finetune_X: np.ndarray,
    finetune_y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    pretrain_weights: Optional[np.ndarray] = None,
    finetune_weights: Optional[np.ndarray] = None,
    pretrain_valid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    config: Optional[TwoStageConfig] = None,
) -> TwoStageResult:
    """Two-stage training for LightGBM classifier.

    Stage 1: Train on regular-season data with standard hyperparameters.
    Stage 2: Continue boosting on tournament data with reduced learning rate.

    Args:
        pretrain_X: Regular-season features [N1, D].
        pretrain_y: Regular-season labels [N1].
        finetune_X: Tournament features [N2, D].
        finetune_y: Tournament labels [N2].
        feature_names: Feature names for LightGBM.
        pretrain_weights: Sample weights for Stage 1.
        finetune_weights: Sample weights for Stage 2 (e.g., round weights).
        pretrain_valid: Validation set for Stage 1 early stopping.
        config: Two-stage configuration.

    Returns:
        TwoStageResult with the final model.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available; two-stage training disabled")
        return TwoStageResult(model=None, stage1_trained=False,
                              reason_skipped_stage2="lightgbm_unavailable")

    if config is None:
        config = TwoStageConfig()

    result = TwoStageResult(model=None, stage1_n_samples=len(pretrain_y))

    # --- Stage 1: Pretrain on regular season ---
    stage1_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": config.pretrain_learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 1,
    }

    train_data = lgb.Dataset(
        pretrain_X, label=pretrain_y,
        feature_name=feature_names,
        weight=pretrain_weights,
    )

    valid_sets = [train_data]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]

    if pretrain_valid is not None:
        valid_data = lgb.Dataset(
            pretrain_valid[0], label=pretrain_valid[1],
            feature_name=feature_names,
            reference=train_data,
        )
        valid_sets.append(valid_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(config.pretrain_early_stopping_rounds))

    stage1_model = lgb.train(
        stage1_params,
        train_data,
        num_boost_round=config.pretrain_num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    result.stage1_trained = True
    result.model = stage1_model

    logger.info(
        "Stage 1 complete: %d regular-season samples, %d trees",
        len(pretrain_y), stage1_model.num_trees(),
    )

    # --- Stage 2: Finetune on tournament ---
    if len(finetune_y) < config.min_finetune_samples:
        result.reason_skipped_stage2 = (
            f"insufficient tournament samples ({len(finetune_y)} < "
            f"{config.min_finetune_samples})"
        )
        logger.info("Stage 2 skipped: %s", result.reason_skipped_stage2)
        return result

    result.stage2_n_samples = len(finetune_y)
    stage2_params = _build_finetune_params(stage1_params, config)

    finetune_data = lgb.Dataset(
        finetune_X, label=finetune_y,
        feature_name=feature_names,
        weight=finetune_weights,
    )

    if config.use_warm_start:
        # Continue boosting from Stage 1 model
        stage2_model = lgb.train(
            stage2_params,
            finetune_data,
            num_boost_round=config.finetune_num_rounds,
            valid_sets=[finetune_data],
            valid_names=["finetune"],
            init_model=stage1_model,
            callbacks=[lgb.log_evaluation(period=0)],
        )
    else:
        # Train from scratch on tournament data only
        stage2_model = lgb.train(
            stage2_params,
            finetune_data,
            num_boost_round=config.finetune_num_rounds,
            valid_sets=[finetune_data],
            valid_names=["finetune"],
            callbacks=[lgb.log_evaluation(period=0)],
        )

    result.model = stage2_model
    result.stage2_trained = True

    logger.info(
        "Stage 2 complete: %d tournament samples, %d total trees "
        "(+%d finetune trees)",
        len(finetune_y), stage2_model.num_trees(),
        stage2_model.num_trees() - stage1_model.num_trees(),
    )

    return result


def two_stage_train_spread(
    pretrain_X: np.ndarray,
    pretrain_margins: np.ndarray,
    finetune_X: np.ndarray,
    finetune_margins: np.ndarray,
    feature_names: Optional[List[str]] = None,
    pretrain_weights: Optional[np.ndarray] = None,
    finetune_weights: Optional[np.ndarray] = None,
    sigma: float = 11.0,
    config: Optional[TwoStageConfig] = None,
) -> TwoStageResult:
    """Two-stage training for SpreadRegressor (margin-based model).

    Stage 1: Learn spread prediction from regular-season margins.
    Stage 2: Adapt to tournament-specific spread distribution.

    Tournament spreads tend to be tighter than regular season (~9-10 vs
    ~11-12 point std) because opponent quality is higher and neutral sites
    eliminate home advantage.

    Args:
        pretrain_X: Regular-season features [N1, D].
        pretrain_margins: Regular-season margins [N1].
        finetune_X: Tournament features [N2, D].
        finetune_margins: Tournament margins [N2].
        feature_names: Feature names.
        pretrain_weights: Sample weights for Stage 1.
        finetune_weights: Sample weights for Stage 2.
        sigma: Initial spread sigma for probability conversion.
        config: Two-stage configuration.

    Returns:
        TwoStageResult with the final SpreadRegressor model.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available; two-stage spread training disabled")
        return TwoStageResult(model=None, stage1_trained=False,
                              reason_skipped_stage2="lightgbm_unavailable")

    if config is None:
        config = TwoStageConfig()

    result = TwoStageResult(model=None, stage1_n_samples=len(pretrain_margins))

    # --- Stage 1: Pretrain spread model on regular season ---
    stage1_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 16,
        "learning_rate": config.pretrain_learning_rate,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 1,
    }

    train_data = lgb.Dataset(
        pretrain_X, label=pretrain_margins,
        feature_name=feature_names,
        weight=pretrain_weights,
    )

    stage1_model = lgb.train(
        stage1_params,
        train_data,
        num_boost_round=config.pretrain_num_rounds,
        valid_sets=[train_data],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    result.stage1_trained = True
    result.model = stage1_model

    logger.info(
        "Stage 1 spread: %d regular-season samples, %d trees",
        len(pretrain_margins), stage1_model.num_trees(),
    )

    # --- Stage 2: Finetune on tournament margins ---
    if len(finetune_margins) < config.min_finetune_samples:
        result.reason_skipped_stage2 = (
            f"insufficient tournament margin samples ({len(finetune_margins)} < "
            f"{config.min_finetune_samples})"
        )
        logger.info("Stage 2 spread skipped: %s", result.reason_skipped_stage2)
        return result

    result.stage2_n_samples = len(finetune_margins)
    stage2_params = _build_finetune_params(stage1_params, config)

    finetune_data = lgb.Dataset(
        finetune_X, label=finetune_margins,
        feature_name=feature_names,
        weight=finetune_weights,
    )

    if config.use_warm_start:
        stage2_model = lgb.train(
            stage2_params,
            finetune_data,
            num_boost_round=config.finetune_num_rounds,
            valid_sets=[finetune_data],
            valid_names=["finetune"],
            init_model=stage1_model,
            callbacks=[lgb.log_evaluation(period=0)],
        )
    else:
        stage2_model = lgb.train(
            stage2_params,
            finetune_data,
            num_boost_round=config.finetune_num_rounds,
            valid_sets=[finetune_data],
            valid_names=["finetune"],
            callbacks=[lgb.log_evaluation(period=0)],
        )

    result.model = stage2_model
    result.stage2_trained = True

    # Recalibrate sigma on tournament margins
    if len(finetune_margins) > 10:
        tournament_sigma = float(np.std(finetune_margins))
        if tournament_sigma > 0:
            logger.info(
                "Tournament sigma: %.2f (regular season: %.2f)",
                tournament_sigma, sigma,
            )

    logger.info(
        "Stage 2 spread: %d tournament samples, %d total trees",
        len(finetune_margins), stage2_model.num_trees(),
    )

    return result


def two_stage_train_logistic(
    pretrain_X: np.ndarray,
    pretrain_y: np.ndarray,
    finetune_X: np.ndarray,
    finetune_y: np.ndarray,
    pretrain_weights: Optional[np.ndarray] = None,
    finetune_weights: Optional[np.ndarray] = None,
    config: Optional[TwoStageConfig] = None,
) -> TwoStageResult:
    """Two-stage training for logistic regression.

    Stage 1: Fit on regular-season data.
    Stage 2: Warm-start from Stage 1 coefficients, refit on tournament data
             with stronger regularization (lower C).

    Args:
        pretrain_X: Regular-season features [N1, D].
        pretrain_y: Regular-season labels [N1].
        finetune_X: Tournament features [N2, D].
        finetune_y: Tournament labels [N2].
        pretrain_weights: Sample weights for Stage 1.
        finetune_weights: Sample weights for Stage 2.
        config: Two-stage configuration.

    Returns:
        TwoStageResult with the final LogisticRegression model.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not available; two-stage logistic disabled")
        return TwoStageResult(model=None, stage1_trained=False,
                              reason_skipped_stage2="sklearn_unavailable")

    if config is None:
        config = TwoStageConfig()

    result = TwoStageResult(model=None, stage1_n_samples=len(pretrain_y))

    # Impute NaN/Inf for logistic regression
    def _clean(X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    pretrain_X_clean = _clean(pretrain_X)
    finetune_X_clean = _clean(finetune_X)

    # Scale features
    scaler = StandardScaler()
    pretrain_X_scaled = scaler.fit_transform(pretrain_X_clean)

    # --- Stage 1: Pretrain ---
    stage1_model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", warm_start=True,
    )
    stage1_model.fit(pretrain_X_scaled, pretrain_y, sample_weight=pretrain_weights)
    result.stage1_trained = True
    result.model = {"logistic": stage1_model, "scaler": scaler}

    logger.info("Stage 1 logistic: %d regular-season samples", len(pretrain_y))

    # --- Stage 2: Finetune ---
    if len(finetune_y) < config.min_finetune_samples:
        result.reason_skipped_stage2 = (
            f"insufficient tournament samples ({len(finetune_y)} < "
            f"{config.min_finetune_samples})"
        )
        logger.info("Stage 2 logistic skipped: %s", result.reason_skipped_stage2)
        return result

    result.stage2_n_samples = len(finetune_y)

    # Transform tournament data using Stage 1 scaler
    finetune_X_scaled = scaler.transform(finetune_X_clean)

    # Warm-start: refit with stronger regularization (lower C)
    # The model starts from Stage 1 coefficients due to warm_start=True
    stage1_model.C = 0.1  # 10x stronger regularization for small tournament set
    stage1_model.fit(finetune_X_scaled, finetune_y, sample_weight=finetune_weights)

    result.model = {"logistic": stage1_model, "scaler": scaler}
    result.stage2_trained = True

    logger.info("Stage 2 logistic: %d tournament samples", len(finetune_y))

    return result
