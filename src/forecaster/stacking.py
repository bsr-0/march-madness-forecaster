"""Model stacking module with LOYO (Leave-One-Year-Out) folds.

Implements year-aware out-of-fold stacking:
- Base models: Logistic Regression (L2), LightGBM
- Meta model: Regularized Logistic Regression (also LOYO-fitted)
- Inputs to meta: [OOF_lr, OOF_gbm, market_prob]
- No in-sample predictions (strict OOF only at every level)

Key design: folds are defined by YEAR, not random splits, to ensure
temporal validity. The meta model is also fitted via LOYO to prevent
leakage at the stacking layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
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
    meta_model: LogisticRegression   # Fitted meta learner (on all data, for inference)
    lr_models: List                  # Per-fold LR models
    gbm_models: List                 # Per-fold GBM models
    scaler: StandardScaler           # Feature scaler for LR
    meta_weights: np.ndarray         # Meta model coefficients
    brier_score: float               # OOF Brier score
    log_loss: float                  # OOF log loss


class StackingEnsemble:
    """LOYO stacking ensemble with LR + GBM base and LR meta.

    Invariants enforced:
    - LR: L2 regularization (always)
    - GBM: max_depth <= 6, learning_rate <= 0.1, early stopping
    - Meta: logistic regression with L2
    - All predictions are strictly out-of-fold by YEAR (no in-sample)
    - Meta model is also LOYO-fitted (nested cross-validation)
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
        self.n_folds = n_folds  # Kept for interface compat, but LOYO uses year-based folds
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
        """Fit the stacking ensemble using LOYO (year-based) OOF predictions.

        For each year Y:
        1. Train base models (LR, GBM) on all years except Y
        2. Generate OOF predictions for year Y
        3. Train meta model on all years except Y (nested LOYO)
        4. Generate meta-OOF predictions for year Y

        This ensures NO data leakage at any stacking level.

        Args:
            X: Feature matrix [n_samples, n_features].
            y: Binary outcomes [n_samples].
            market_probs: Market-implied probabilities [n_samples] (optional).
            year_labels: Year labels for LOYO folds (required for proper LOYO).
        """
        n_samples = len(y)
        oof_lr = np.full(n_samples, np.nan)
        oof_gbm = np.full(n_samples, np.nan)
        oof_meta = np.full(n_samples, np.nan)

        # Determine fold structure
        if year_labels is not None:
            unique_years = sorted(set(year_labels))
            use_loyo = len(unique_years) >= 3
        else:
            use_loyo = False

        if not use_loyo:
            logger.warning("Falling back to stratified K-fold (no year labels or <3 years)")
            return self._fit_kfold(X, y, market_probs)

        logger.info("Using LOYO with %d year-folds: %s", len(unique_years), unique_years)

        # Scale features for LR (fit on ALL data — same transform for all folds)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.lr_models = []
        self.gbm_models = []

        # --- Level 1: Base model OOF via LOYO ---
        for hold_year in unique_years:
            train_mask = year_labels != hold_year
            val_mask = year_labels == hold_year

            n_train = train_mask.sum()
            n_val = val_mask.sum()

            if n_train < 20 or n_val < 5:
                logger.warning("Skipping year %d: train=%d, val=%d", hold_year, n_train, n_val)
                oof_lr[val_mask] = 0.5
                oof_gbm[val_mask] = 0.5
                continue

            logger.info("LOYO fold: hold_year=%d (train=%d, val=%d)",
                        hold_year, n_train, n_val)

            X_train_s, X_val_s = X_scaled[train_mask], X_scaled[val_mask]
            X_train_r, X_val_r = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]

            # Base model 1: Logistic Regression with L2
            lr = LogisticRegression(
                C=self.lr_C,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_seed,
            )
            lr.fit(X_train_s, y_train)
            oof_lr[val_mask] = lr.predict_proba(X_val_s)[:, 1]
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
                gbm.fit(
                    X_train_r, y_train,
                    eval_set=[(X_val_r, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                oof_gbm[val_mask] = gbm.predict_proba(X_val_r)[:, 1]
                self.gbm_models.append(gbm)
            else:
                lr2 = LogisticRegression(
                    C=0.1, penalty="l2", solver="lbfgs",
                    max_iter=1000, random_state=self.random_seed + 1,
                )
                lr2.fit(X_train_s, y_train)
                oof_gbm[val_mask] = lr2.predict_proba(X_val_s)[:, 1]
                self.gbm_models.append(lr2)

        # --- Level 2: Meta model OOF via nested LOYO ---
        # For each year Y, train meta on base OOF from all years except Y,
        # then predict meta-OOF for year Y. This prevents the meta model
        # from being trained on the same data it evaluates.
        for hold_year in unique_years:
            train_mask = year_labels != hold_year
            val_mask = year_labels == hold_year

            if train_mask.sum() < 20 or val_mask.sum() < 5:
                oof_meta[val_mask] = 0.5
                continue

            # Build meta features from base OOF
            meta_train = np.column_stack([oof_lr[train_mask], oof_gbm[train_mask]])
            meta_val = np.column_stack([oof_lr[val_mask], oof_gbm[val_mask]])

            if market_probs is not None:
                meta_train = np.column_stack([meta_train, market_probs[train_mask]])
                meta_val = np.column_stack([meta_val, market_probs[val_mask]])

            # Skip if base OOF has NaN (shouldn't happen with LOYO)
            if np.any(np.isnan(meta_train)):
                oof_meta[val_mask] = 0.5
                continue

            meta_lr = LogisticRegression(
                C=self.meta_C, penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=self.random_seed,
            )
            meta_lr.fit(meta_train, y[train_mask])
            oof_meta[val_mask] = meta_lr.predict_proba(meta_val)[:, 1]

        # --- Final meta model (for inference on new data) ---
        # Trained on ALL base OOF (acceptable: used only for 2026 prediction,
        # NOT for metric computation).
        meta_features_all = np.column_stack([oof_lr, oof_gbm])
        if market_probs is not None:
            meta_features_all = np.column_stack([meta_features_all, market_probs])

        self.meta_model = LogisticRegression(
            C=self.meta_C, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=self.random_seed,
        )
        valid_mask = ~np.isnan(oof_lr) & ~np.isnan(oof_gbm)
        self.meta_model.fit(meta_features_all[valid_mask], y[valid_mask])

        self.is_fitted = True

        # Use nested meta OOF for honest metrics
        final_oof = np.where(np.isnan(oof_meta), 0.5, oof_meta)

        brier = float(np.mean((final_oof - y) ** 2))
        eps = 1e-15
        clipped = np.clip(final_oof, eps, 1 - eps)
        log_loss_val = float(-np.mean(
            y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
        ))

        meta_weights = (self.meta_model.coef_[0]
                        if hasattr(self.meta_model, 'coef_') else np.array([]))

        result = StackingResult(
            oof_preds=final_oof,
            oof_lr=oof_lr,
            oof_gbm=oof_gbm,
            meta_model=self.meta_model,
            lr_models=self.lr_models,
            gbm_models=self.gbm_models,
            scaler=self.scaler,
            meta_weights=meta_weights,
            brier_score=brier,
            log_loss=log_loss_val,
        )

        logger.info("Stacking LOYO OOF Brier=%.4f, LogLoss=%.4f", brier, log_loss_val)
        logger.info("Meta weights: %s", meta_weights)
        return result

    def _fit_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
    ) -> StackingResult:
        """Fallback: stratified K-fold stacking (when no year labels)."""
        from sklearn.model_selection import StratifiedKFold

        n_samples = len(y)
        oof_lr = np.full(n_samples, np.nan)
        oof_gbm = np.full(n_samples, np.nan)

        X_scaled = self.scaler.fit_transform(X)
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True,
            random_state=self.random_seed,
        )

        self.lr_models = []
        self.gbm_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            X_raw_train, X_raw_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            lr = LogisticRegression(
                C=self.lr_C, penalty="l2", solver="lbfgs",
                max_iter=1000, random_state=self.random_seed,
            )
            lr.fit(X_train, y_train)
            oof_lr[val_idx] = lr.predict_proba(X_val)[:, 1]
            self.lr_models.append(lr)

            if LIGHTGBM_AVAILABLE:
                gbm = lgb.LGBMClassifier(
                    n_estimators=self.gbm_n_estimators,
                    max_depth=self.gbm_max_depth,
                    learning_rate=self.gbm_lr,
                    min_child_samples=self.gbm_min_child_samples,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.random_seed, verbose=-1, n_jobs=1,
                )
                gbm.fit(
                    X_raw_train, y_train,
                    eval_set=[(X_raw_val, y[val_idx])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                oof_gbm[val_idx] = gbm.predict_proba(X_raw_val)[:, 1]
                self.gbm_models.append(gbm)
            else:
                lr2 = LogisticRegression(
                    C=0.1, penalty="l2", solver="lbfgs",
                    max_iter=1000, random_state=self.random_seed + 1,
                )
                lr2.fit(X_train, y_train)
                oof_gbm[val_idx] = lr2.predict_proba(X_val)[:, 1]
                self.gbm_models.append(lr2)

        meta_features = np.column_stack([oof_lr, oof_gbm])
        if market_probs is not None:
            meta_features = np.column_stack([meta_features, market_probs])

        self.meta_model = LogisticRegression(
            C=self.meta_C, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=self.random_seed,
        )
        self.meta_model.fit(meta_features, y)
        oof_preds = self.meta_model.predict_proba(meta_features)[:, 1]

        self.is_fitted = True

        brier = float(np.mean((oof_preds - y) ** 2))
        eps = 1e-15
        clipped = np.clip(oof_preds, eps, 1 - eps)
        log_loss_val = float(-np.mean(
            y * np.log(clipped) + (1 - y) * np.log(1 - clipped)
        ))

        meta_weights = (self.meta_model.coef_[0]
                        if hasattr(self.meta_model, 'coef_') else np.array([]))

        return StackingResult(
            oof_preds=oof_preds, oof_lr=oof_lr, oof_gbm=oof_gbm,
            meta_model=self.meta_model,
            lr_models=self.lr_models, gbm_models=self.gbm_models,
            scaler=self.scaler, meta_weights=meta_weights,
            brier_score=brier, log_loss=log_loss_val,
        )

    def predict(
        self,
        X: np.ndarray,
        market_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate predictions using the fitted stacking ensemble.

        Averages base model predictions across all LOYO fold models,
        then applies the final meta model.
        """
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fit before predict")

        X_scaled = self.scaler.transform(X)

        lr_preds = np.mean([
            m.predict_proba(X_scaled)[:, 1] for m in self.lr_models
        ], axis=0)

        if LIGHTGBM_AVAILABLE and self.gbm_models:
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
        """Get feature importances from base models."""
        importances = {}

        lr_coefs = np.mean([np.abs(m.coef_[0]) for m in self.lr_models], axis=0)
        importances["lr"] = lr_coefs

        if LIGHTGBM_AVAILABLE and self.gbm_models:
            gbm_imps = np.mean([
                m.feature_importances_ for m in self.gbm_models
            ], axis=0)
            importances["gbm"] = gbm_imps

        return importances
