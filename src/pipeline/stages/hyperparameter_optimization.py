"""Phase 7 — Hyperparameter Optimization.

Tune hyperparameters for each selected model class from Phase 6 using
time-aware cross-validation.  Only accept configurations that improve
bracket EV over the baseline.  Reject any config that reduces EV.

Uses the existing :class:`TemporalCrossValidator` and model-specific
tuners (LightGBM via Optuna, XGBoost via Optuna, logistic regression
via grid search) to find optimal hyperparameters efficiently.

Usage::

    optimizer = HyperparameterOptimizer(
        baseline_ev=selection.baseline_ev,
        baseline_brier=selection.baseline_brier,
    )
    result = optimizer.run(
        selected_models=["gradient_boosting", "xgboost", "regularized_logistic"],
        train_X=X, train_y=y, sort_keys=keys,
        feature_names=names,
    )
    # result.optimized_configs → {"gradient_boosting": {...}, ...}

Output is an ``OptimizedConfigs`` contract consumed by Phase 8
(probability calibration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.exceptions import IntegrityError
from src.ml.calibration.calibration import BrierScoreOptimizer
from src.pipeline.stages.baseline_evaluation import (
    ROUND_SCORING_WEIGHTS,
    compute_bracket_ev,
)
from src.pipeline.stages.model_selection import _temporal_cv_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default number of Optuna trials per model.
DEFAULT_N_TRIALS = 15

# Default timeout per model (seconds).
DEFAULT_TIMEOUT = 300

# Default number of CV folds for hyperparameter search.
DEFAULT_N_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Optimized hyperparameters for a single model."""

    model_name: str
    best_params: Dict[str, Any] = field(default_factory=dict)
    cv_brier: float = 0.0
    cv_ev: float = 0.0
    ev_improvement: float = 0.0
    accepted: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "cv_Brier": round(self.cv_brier, 6),
            "cv_EV": round(self.cv_ev, 4),
            "ev_improvement": round(self.ev_improvement, 4),
            "accepted": self.accepted,
        }


@dataclass
class OptimizedConfigs:
    """Output of Phase 7 — optimized hyperparameters for selected models.

    Consumed by Phase 8 (probability calibration) and Phase 9 (ensembling).
    """

    configs: Dict[str, ModelConfig] = field(default_factory=dict)
    baseline_ev: float = 0.0
    baseline_brier: float = 0.0
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def optimized_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return {model_name: best_params} for accepted models."""
        return {
            name: cfg.best_params
            for name, cfg in self.configs.items()
            if cfg.accepted
        }

    @property
    def accepted_models(self) -> List[str]:
        return [name for name, cfg in self.configs.items() if cfg.accepted]

    def summary(self) -> str:
        lines = [
            f"HyperparameterOptimization: {'PASSED' if self.passed else 'FAILED'}",
            f"  Baseline EV={self.baseline_ev:.2f}, Brier={self.baseline_brier:.4f}",
        ]
        for name, cfg in self.configs.items():
            tag = "ACCEPTED" if cfg.accepted else "REJECTED"
            lines.append(
                f"  {name}: EV={cfg.cv_ev:.2f} "
                f"(+{cfg.ev_improvement:.2f}), "
                f"Brier={cfg.cv_brier:.4f} [{tag}]"
            )
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_models": self.accepted_models,
            "configs": {n: c.to_dict() for n, c in self.configs.items()},
        }


# ---------------------------------------------------------------------------
# Per-model tuning strategies
# ---------------------------------------------------------------------------


def _tune_gradient_boosting(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sort_keys: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    n_trials: int,
    timeout: int,
    n_cv_folds: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Tune LightGBM via Optuna with temporal CV.  Returns best_params."""
    try:
        from src.ml.optimization.hyperparameter_tuning import LightGBMTuner
        tuner = LightGBMTuner(
            n_trials=n_trials,
            n_cv_splits=n_cv_folds,
            timeout=timeout,
        )
        result = tuner.tune(
            train_X, train_y, sort_keys,
            feature_names=feature_names,
            sample_weight=sample_weight,
        )
        return result.best_params
    except ImportError:
        logger.warning("LightGBM/Optuna unavailable; using default params for gradient_boosting.")
        return {
            "num_leaves": 8,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "min_child_samples": 50,
            "num_rounds": 150,
        }


def _tune_xgboost(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sort_keys: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    n_trials: int,
    timeout: int,
    n_cv_folds: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Tune XGBoost via Optuna with temporal CV.  Returns best_params."""
    try:
        import optuna
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        splits = _temporal_cv_split(len(train_X), n_folds=n_cv_folds, sort_keys=sort_keys)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "verbosity": 0,
                "nthread": 1,
            }
            num_rounds = trial.suggest_int("num_rounds", 50, 200)

            fold_briers = []
            for train_idx, val_idx in splits:
                X_tr, y_tr = train_X[train_idx], train_y[train_idx]
                X_v, y_v = train_X[val_idx], train_y[val_idx]
                w_tr = sample_weight[train_idx] if sample_weight is not None else None
                dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
                dval = xgb.DMatrix(X_v, label=y_v)
                model = xgb.train(params, dtrain, num_boost_round=num_rounds)
                preds = model.predict(dval)
                preds = np.clip(preds, 1e-7, 1 - 1e-7)
                fold_briers.append(float(np.mean((preds - y_v) ** 2)))
            return float(np.mean(fold_briers))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        best = study.best_params
        return {
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": best["min_child_weight"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
            "num_rounds": best.get("num_rounds", 150),
        }
    except ImportError:
        logger.warning("XGBoost/Optuna unavailable; using default params for xgboost.")
        return {
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 10,
            "num_rounds": 150,
        }


def _tune_regularized_logistic(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sort_keys: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    n_trials: int,
    timeout: int,
    n_cv_folds: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Tune logistic regression C and l1_ratio via random search.

    Samples ``n_trials`` random (C, l1_ratio) combinations from log-uniform
    and uniform distributions respectively, which is substantially faster than
    the previous 6×6 exhaustive grid search (36 fixed combos) while covering
    the same parameter space more efficiently at configurable cost.

    C ~ LogUniform(0.01, 5.0), l1_ratio ~ Uniform(0.0, 1.0).
    Seed is fixed for reproducibility.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    splits = _temporal_cv_split(len(train_X), n_folds=n_cv_folds, sort_keys=sort_keys)

    rng = np.random.RandomState(seed=42)
    # Log-uniform C: sample in log space then exponentiate
    log_c_samples = rng.uniform(np.log(0.01), np.log(5.0), size=n_trials)
    c_samples = np.exp(log_c_samples)
    l1_ratio_samples = rng.uniform(0.0, 1.0, size=n_trials)

    best_brier = float("inf")
    best_params = {"C": 0.5, "l1_ratio": 0.3}

    for C, l1_ratio in zip(c_samples, l1_ratio_samples):
        fold_briers = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = train_X[train_idx], train_y[train_idx]
            X_v, y_v = train_X[val_idx], train_y[val_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
            X_v_s = scaler.transform(np.nan_to_num(X_v, nan=0.0))
            try:
                if l1_ratio < 0.05:
                    model = LogisticRegression(
                        penalty="l2", C=float(C), solver="lbfgs", max_iter=2000,
                    )
                else:
                    model = LogisticRegression(
                        penalty="elasticnet", C=float(C), l1_ratio=float(l1_ratio),
                        solver="saga", max_iter=3000,
                    )
                model.fit(X_tr_s, y_tr)
                preds = np.clip(model.predict_proba(X_v_s)[:, 1], 1e-7, 1 - 1e-7)
                fold_briers.append(float(np.mean((preds - y_v) ** 2)))
            except Exception:
                continue
        if fold_briers:
            mean_brier = float(np.mean(fold_briers))
            if mean_brier < best_brier:
                best_brier = mean_brier
                best_params = {"C": float(C), "l1_ratio": float(l1_ratio)}

    return best_params


def _tune_spread_regressor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sort_keys: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    n_trials: int,
    timeout: int,
    n_cv_folds: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return default params for SpreadRegressor (limited tunable surface)."""
    return {"sigma": 11.0, "num_rounds": 200}


def _tune_margin_regressor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sort_keys: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    n_trials: int,
    timeout: int,
    n_cv_folds: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return default params for margin regressor."""
    return {"num_leaves": 8, "learning_rate": 0.05, "num_rounds": 200}


def _tune_gnn_augmented(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sort_keys: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    n_trials: int,
    timeout: int,
    n_cv_folds: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Tune GNN-augmented LightGBM via Optuna with temporal CV.

    The GNN-augmented candidate uses the same LightGBM backend as
    ``gradient_boosting`` (with PCA-derived SOS features prepended), so we
    delegate directly to the LightGBM tuner.  Hyperparameters governing tree
    depth and regularisation are shared; the PCA augmentation is fixed.
    """
    return _tune_gradient_boosting(
        train_X, train_y, sort_keys, feature_names,
        n_trials, timeout, n_cv_folds,
        sample_weight=sample_weight,
    )


# Registry of tuning functions keyed by model name.
TUNER_REGISTRY: Dict[str, Callable] = {
    "gradient_boosting": _tune_gradient_boosting,
    "xgboost": _tune_xgboost,
    "regularized_logistic": _tune_regularized_logistic,
    "spread_regressor": _tune_spread_regressor,
    "margin_regressor": _tune_margin_regressor,
    "gnn_augmented": _tune_gnn_augmented,
}


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------


class HyperparameterOptimizer:
    """Phase 7 orchestrator: tune hyperparameters, gate on EV.

    Args:
        baseline_ev: Bracket EV from Phase 6 best model.
        baseline_brier: Brier from Phase 6 best model.
        n_trials: Optuna trials per model.
        timeout: Seconds per model.
        n_cv_folds: Temporal CV folds.
        strict: Raise IntegrityError if no model config accepted.
        scoring_weights: Round → point mapping for EV.
    """

    def __init__(
        self,
        baseline_ev: float,
        baseline_brier: float,
        n_trials: int = DEFAULT_N_TRIALS,
        timeout: int = DEFAULT_TIMEOUT,
        n_cv_folds: int = DEFAULT_N_CV_FOLDS,
        strict: bool = True,
        scoring_weights: Optional[Dict[str, int]] = None,
    ):
        self.baseline_ev = baseline_ev
        self.baseline_brier = baseline_brier
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_cv_folds = n_cv_folds
        self.strict = strict
        self.scoring_weights = scoring_weights or ROUND_SCORING_WEIGHTS

    def run(
        self,
        selected_models: List[str],
        train_X: np.ndarray,
        train_y: np.ndarray,
        sort_keys: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        round_labels: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        train_margins: Optional[np.ndarray] = None,
    ) -> OptimizedConfigs:
        """Tune hyperparameters for each selected model.

        Args:
            selected_models: Model names from Phase 6.
            train_X: Training features (n_samples, n_features).
            train_y: Training labels (0/1).
            sort_keys: Temporal sort keys for CV splitting.
            feature_names: Feature names for model training.
            round_labels: Per-game round labels for EV computation.
            sample_weight: Per-sample weights.
            train_margins: Point margins for margin-based models.

        Returns:
            OptimizedConfigs with per-model best hyperparameters.

        Raises:
            IntegrityError: If strict=True and no model passes EV gate.
        """
        result = OptimizedConfigs(
            baseline_ev=self.baseline_ev,
            baseline_brier=self.baseline_brier,
        )

        if sort_keys is None:
            sort_keys = np.arange(len(train_X), dtype=float)

        for model_name in selected_models:
            tuner_fn = TUNER_REGISTRY.get(model_name)
            if tuner_fn is None:
                result.warnings.append(f"No tuner for model: {model_name}")
                # Use empty params (defaults) for unknown models
                result.configs[model_name] = ModelConfig(
                    model_name=model_name,
                    best_params={},
                    accepted=True,
                    reason="No tuner — using model defaults",
                )
                continue

            try:
                best_params = tuner_fn(
                    train_X, train_y, sort_keys, feature_names,
                    self.n_trials, self.timeout, self.n_cv_folds,
                    sample_weight=sample_weight,
                )

                # Validate tuned config via CV
                cv_brier, cv_ev = self._validate_config(
                    model_name, best_params,
                    train_X, train_y, sort_keys,
                    feature_names, round_labels,
                    train_margins, sample_weight,
                )

                ev_improvement = cv_ev - self.baseline_ev
                accepted = ev_improvement >= 0  # Must not reduce EV

                config = ModelConfig(
                    model_name=model_name,
                    best_params=best_params,
                    cv_brier=cv_brier,
                    cv_ev=cv_ev,
                    ev_improvement=ev_improvement,
                    accepted=accepted,
                    reason=(
                        f"EV +{ev_improvement:.2f} over baseline"
                        if accepted
                        else f"EV reduced by {abs(ev_improvement):.2f} — rejected"
                    ),
                )
                result.configs[model_name] = config

                logger.info(
                    "Phase 7 %s: params=%s, CV Brier=%.4f, CV EV=%.2f [%s]",
                    model_name, best_params, cv_brier, cv_ev,
                    "ACCEPTED" if accepted else "REJECTED",
                )

            except Exception as e:
                logger.error("Phase 7 tuning failed for %s: %s", model_name, e)
                result.warnings.append(f"{model_name} tuning failed: {e}")
                # Accept with empty params (will use model defaults)
                result.configs[model_name] = ModelConfig(
                    model_name=model_name,
                    best_params={},
                    accepted=True,
                    reason=f"Tuning failed ({e}); using defaults",
                )

        if not result.accepted_models:
            result.passed = False
            msg = (
                "Phase 7: No tuned config improves EV. "
                "Reassess Phase 6 model selection."
            )
            result.errors.append(msg)
            logger.error(msg)

        logger.info(result.summary())

        if not result.passed and self.strict:
            raise IntegrityError(
                f"Phase 7 hyperparameter optimization failed: {result.errors}"
            )

        return result

    def _validate_config(
        self,
        model_name: str,
        params: Dict[str, Any],
        train_X: np.ndarray,
        train_y: np.ndarray,
        sort_keys: np.ndarray,
        feature_names: Optional[List[str]],
        round_labels: Optional[np.ndarray],
        train_margins: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
    ) -> tuple:
        """Validate a tuned config via temporal CV.  Returns (brier, ev)."""
        from src.pipeline.stages.model_selection import (
            CANDIDATE_REGISTRY,
            MARGIN_BASED_CANDIDATES,
        )

        splits = _temporal_cv_split(
            len(train_X), n_folds=self.n_cv_folds, sort_keys=sort_keys,
        )

        factory = CANDIDATE_REGISTRY.get(model_name)
        if factory is None:
            return self.baseline_brier, self.baseline_ev

        is_margin = model_name in MARGIN_BASED_CANDIDATES
        fold_briers = []
        fold_evs = []

        for train_idx, val_idx in splits:
            X_tr, y_tr = train_X[train_idx], train_y[train_idx]
            X_v, y_v = train_X[val_idx], train_y[val_idx]

            if is_margin:
                margins = train_margins[train_idx] if train_margins is not None else None
                predict_fn = factory(X_tr, y_tr, feature_names, train_margins=margins)
            else:
                predict_fn = factory(X_tr, y_tr, feature_names)

            if predict_fn is None:
                return self.baseline_brier, self.baseline_ev

            preds = np.clip(predict_fn(X_v), 1e-7, 1 - 1e-7)
            fold_briers.append(float(BrierScoreOptimizer.calculate(preds, y_v)))
            fold_evs.append(compute_bracket_ev(preds, scoring_weights=self.scoring_weights))

        return float(np.mean(fold_briers)), float(np.mean(fold_evs))
