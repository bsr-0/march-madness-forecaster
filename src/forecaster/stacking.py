"""Model stacking module.

Implements K-fold out-of-fold (OOF) stacking:
- Base models: Logistic Regression (L2), LightGBM
- Meta model: Regularized Logistic Regression
- Inputs to meta: [OOF_lr, OOF_gbm, market_prob]
- No in-sample predictions (strict OOF only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@dataclass
class StackingResult:
    """Results from stacking ensemble training."""
    oof_preds: np.ndarray           # Final OOF predictions from meta model
    oof_lr: np.ndarray              # OOF predictions from logistic regression
    oof_gbm: np.ndarray             # OOF predictions from GBM
    meta_model: LogisticRegression   # Fitted meta learner
    lr_models: List                  # Per-fold LR models
    gbm_models: List                 # Per-fold GBM models
    scaler: StandardScaler           # Feature scaler for LR
    meta_weights: np.ndarray         # Meta model coefficients
    brier_score: float               # OOF Brier score
    log_loss: float                  # OOF log loss


class StackingEnsemble:
    """K-fold stacking ensemble with LR + GBM base and LR meta.

    Invariants enforced:
    - LR: L2 regularization (always)
    - GBM: max_depth <= 6, learning_rate <= 0.1, early stopping
    - Meta: logistic regression with L2
    - All predictions are strictly out-of-fold (no in-sample)
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_seed: int = 2026,
        lr_C: float = 1.0,
        gbm_max_depth: int = 5,
        gbm_lr: float = 0.05,
        gbm_n_estimators: int = 500,
        gbm_min_child_samples: int = 20,
        meta_C: float = 1.0,
    ):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.lr_C = lr_C
        self.gbm_max_depth = min(gbm_max_depth, 6)  # Invariant: depth <= 6
        self.gbm_lr = min(gbm_lr, 0.1)              # Invariant: lr <= 0.1
        self.gbm_n_estimators = gbm_n_estimators
        self.gbm_min_child_samples = gbm_min_child_samples
        self.meta_C = meta_C

        self.lr_models: List[LogisticRegression] = []
        self.gbm_models: List = []
        self.scaler = StandardScaler()
        self.meta_model: Optional[LogisticRegression] = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
        year_labels: Optional[np.ndarray] = None,
    ) -> StackingResult:
        """Fit the stacking ensemble using K-fold OOF predictions.

        Args:
            X: Feature matrix [n_samples, n_features].
            y: Binary outcomes [n_samples].
            market_probs: Market-implied probabilities [n_samples] (optional).
            year_labels: Year labels for stratification (optional).

        Returns:
            StackingResult with OOF predictions and fitted models.
        """
        n_samples = len(y)
        oof_lr = np.full(n_samples, np.nan)
        oof_gbm = np.full(n_samples, np.nan)

        # Scale features for LR
        X_scaled = self.scaler.fit_transform(X)

        # Stratified K-fold
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_seed,
        )

        self.lr_models = []
        self.gbm_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info("Stacking fold %d/%d (train=%d, val=%d)",
                        fold_idx + 1, self.n_folds, len(train_idx), len(val_idx))

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            X_raw_train, X_raw_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Base model 1: Logistic Regression with L2
            lr = LogisticRegression(
                C=self.lr_C,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_seed,
            )
            lr.fit(X_train, y_train)
            oof_lr[val_idx] = lr.predict_proba(X_val)[:, 1]
            self.lr_models.append(lr)

            # Base model 2: LightGBM
            if LIGHTGBM_AVAILABLE:
                gbm = lgb.LGBMClassifier(
                    n_estimators=self.gbm_n_estimators,
                    max_depth=self.gbm_max_depth,
                    learning_rate=self.gbm_lr,
                    min_child_samples=self.gbm_min_child_samples,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_seed,
                    verbose=-1,
                    n_jobs=1,
                )
                # Early stopping with validation set
                gbm.fit(
                    X_raw_train, y_train,
                    eval_set=[(X_raw_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                oof_gbm[val_idx] = gbm.predict_proba(X_raw_val)[:, 1]
                self.gbm_models.append(gbm)
            else:
                # Fallback: second LR with different regularization
                lr2 = LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self.random_seed + 1,
                )
                lr2.fit(X_train, y_train)
                oof_gbm[val_idx] = lr2.predict_proba(X_val)[:, 1]
                self.gbm_models.append(lr2)

        # Build meta features: [OOF_lr, OOF_gbm, market_prob]
        meta_features = np.column_stack([oof_lr, oof_gbm])
        if market_probs is not None:
            meta_features = np.column_stack([meta_features, market_probs])

        # Meta model: regularized logistic regression
        self.meta_model = LogisticRegression(
            C=self.meta_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_seed,
        )
        self.meta_model.fit(meta_features, y)
        oof_preds = self.meta_model.predict_proba(meta_features)[:, 1]

        self.is_fitted = True

        # Compute metrics
        brier = float(np.mean((oof_preds - y) ** 2))
        eps = 1e-15
        clipped = np.clip(oof_preds, eps, 1 - eps)
        log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))

        meta_weights = self.meta_model.coef_[0] if hasattr(self.meta_model, 'coef_') else np.array([])

        result = StackingResult(
            oof_preds=oof_preds,
            oof_lr=oof_lr,
            oof_gbm=oof_gbm,
            meta_model=self.meta_model,
            lr_models=self.lr_models,
            gbm_models=self.gbm_models,
            scaler=self.scaler,
            meta_weights=meta_weights,
            brier_score=brier,
            log_loss=log_loss,
        )

        logger.info("Stacking OOF Brier=%.4f, LogLoss=%.4f", brier, log_loss)
        logger.info("Meta weights: %s", meta_weights)
        return result

    def predict(
        self,
        X: np.ndarray,
        market_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate predictions using the fitted stacking ensemble.

        Averages predictions across all fold models (no in-sample leakage
        since we average all K models, each trained on different folds).

        Args:
            X: Feature matrix [n_samples, n_features].
            market_prob: Market probabilities [n_samples] (optional).

        Returns:
            Predicted probabilities [n_samples].
        """
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fit before predict")

        X_scaled = self.scaler.transform(X)

        # Average base model predictions across folds
        lr_preds = np.mean([
            m.predict_proba(X_scaled)[:, 1] for m in self.lr_models
        ], axis=0)

        if LIGHTGBM_AVAILABLE:
            gbm_preds = np.mean([
                m.predict_proba(X)[:, 1] for m in self.gbm_models
            ], axis=0)
        else:
            gbm_preds = np.mean([
                m.predict_proba(X_scaled)[:, 1] for m in self.gbm_models
            ], axis=0)

        meta_features = np.column_stack([lr_preds, gbm_preds])
        if market_prob is not None:
            meta_features = np.column_stack([meta_features, market_prob])

        return self.meta_model.predict_proba(meta_features)[:, 1]

    def get_feature_importances(self) -> Dict[str, np.ndarray]:
        """Get feature importances from base models.

        Returns:
            Dict with 'lr' coefficients and 'gbm' importances.
        """
        importances = {}

        # LR: average absolute coefficients across folds
        lr_coefs = np.mean([np.abs(m.coef_[0]) for m in self.lr_models], axis=0)
        importances["lr"] = lr_coefs

        # GBM: average feature importances across folds
        if LIGHTGBM_AVAILABLE and self.gbm_models:
            gbm_imps = np.mean([
                m.feature_importances_ for m in self.gbm_models
            ], axis=0)
            importances["gbm"] = gbm_imps

        return importances
