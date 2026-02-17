"""
Hyperparameter optimization via Optuna with temporal cross-validation.

Provides:
- TemporalCrossValidator: time-series-aware k-fold splits for game data
- LightGBMTuner: Optuna-based hyperparameter search for LightGBM
- EnsembleWeightOptimizer: grid search over CFA base weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """A single temporal train/validation split."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    fold_id: int


@dataclass
class CVResult:
    """Cross-validation result for a single fold."""

    fold_id: int
    train_size: int
    val_size: int
    brier_score: float
    log_loss: float
    accuracy: float


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_params: Dict
    best_score: float
    cv_results: List[CVResult] = field(default_factory=list)
    n_trials: int = 0
    study_name: str = ""


class TemporalCrossValidator:
    """
    Time-series-aware cross-validation for game data.

    Uses expanding-window splits so training always precedes validation
    chronologically. This prevents data leakage from future games.

    For n_splits=5 with 1000 samples:
      Fold 0: train=[0:400],   val=[400:520]
      Fold 1: train=[0:520],   val=[520:640]
      Fold 2: train=[0:640],   val=[640:760]
      Fold 3: train=[0:760],   val=[760:880]
      Fold 4: train=[0:880],   val=[880:1000]
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 30):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(
        self, n_samples: int, sort_keys: Optional[np.ndarray] = None
    ) -> List[TemporalSplit]:
        """
        Generate temporal train/validation splits.

        Args:
            n_samples: Total number of samples
            sort_keys: Optional sort keys (e.g., game dates as ints).
                       If provided, indices are reordered by sort_keys.

        Returns:
            List of TemporalSplit objects
        """
        if sort_keys is not None:
            indices = np.argsort(sort_keys)
        else:
            indices = np.arange(n_samples)

        # Reserve at least 40% for initial training
        initial_train = max(self.min_train_size, int(0.4 * n_samples))
        remaining = n_samples - initial_train

        if remaining < self.n_splits:
            # Not enough data for requested splits; return single split
            split_point = max(self.min_train_size, int(0.8 * n_samples))
            return [
                TemporalSplit(
                    train_indices=indices[:split_point],
                    val_indices=indices[split_point:],
                    fold_id=0,
                )
            ]

        val_size = remaining // self.n_splits
        splits = []

        for fold in range(self.n_splits):
            val_start = initial_train + fold * val_size
            val_end = val_start + val_size if fold < self.n_splits - 1 else n_samples

            if val_start >= n_samples:
                break

            splits.append(
                TemporalSplit(
                    train_indices=indices[:val_start],
                    val_indices=indices[val_start:val_end],
                    fold_id=fold,
                )
            )

        return splits

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sort_keys: np.ndarray,
        train_fn,
        predict_fn,
    ) -> List[CVResult]:
        """
        Run temporal cross-validation with arbitrary train/predict functions.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            sort_keys: Chronological sort keys
            train_fn: Callable(X_train, y_train, X_val, y_val) -> model
            predict_fn: Callable(model, X) -> probabilities

        Returns:
            List of CVResult per fold
        """
        splits = self.split(len(y), sort_keys)
        results = []

        for split in splits:
            X_train = X[split.train_indices]
            y_train = y[split.train_indices]
            X_val = X[split.val_indices]
            y_val = y[split.val_indices]

            model = train_fn(X_train, y_train, X_val, y_val)
            preds = predict_fn(model, X_val)
            preds = np.clip(preds, 1e-7, 1 - 1e-7)

            brier = float(np.mean((preds - y_val) ** 2))
            eps = 1e-7
            ll = float(
                -np.mean(y_val * np.log(preds + eps) + (1 - y_val) * np.log(1 - preds + eps))
            )
            acc = float(np.mean((preds >= 0.5).astype(int) == y_val))

            results.append(
                CVResult(
                    fold_id=split.fold_id,
                    train_size=len(y_train),
                    val_size=len(y_val),
                    brier_score=brier,
                    log_loss=ll,
                    accuracy=acc,
                )
            )

        return results


class LightGBMTuner:
    """
    Optuna-based hyperparameter optimization for LightGBM.

    Searches over:
    - num_leaves: [15, 127]
    - learning_rate: [0.01, 0.3]
    - feature_fraction: [0.4, 1.0]
    - bagging_fraction: [0.4, 1.0]
    - min_child_samples: [5, 100]
    - lambda_l1: [1e-8, 10.0]
    - lambda_l2: [1e-8, 10.0]

    Uses TemporalCrossValidator internally to prevent leakage.
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_cv_splits: int = 5,
        timeout: Optional[int] = 300,
        random_seed: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for hyperparameter tuning")
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM required for LightGBMTuner")

        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.timeout = timeout
        self.random_seed = random_seed

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sort_keys: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> TuningResult:
        """
        Run Optuna hyperparameter search with temporal CV.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            sort_keys: Chronological sort keys for temporal splitting
            feature_names: Optional feature names

        Returns:
            TuningResult with best params and CV scores
        """
        cv = TemporalCrossValidator(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": 5,
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "verbose": -1,
            }
            num_rounds = trial.suggest_int("num_rounds", 50, 500)

            def train_fn(X_tr, y_tr, X_v, y_v):
                train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
                val_data = lgb.Dataset(X_v, label=y_v, feature_name=feature_names, reference=train_data)
                callbacks = [lgb.early_stopping(30), lgb.log_evaluation(period=0)]
                return lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_rounds,
                    valid_sets=[val_data],
                    valid_names=["valid"],
                    callbacks=callbacks,
                )

            def predict_fn(model, X_pred):
                return model.predict(X_pred)

            cv_results = cv.cross_validate(X, y, sort_keys, train_fn, predict_fn)
            mean_brier = float(np.mean([r.brier_score for r in cv_results]))
            return mean_brier

        study = optuna.create_study(
            direction="minimize",
            study_name="lgbm_tuning",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params
        best_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": best["num_leaves"],
            "learning_rate": best["learning_rate"],
            "feature_fraction": best["feature_fraction"],
            "bagging_fraction": best["bagging_fraction"],
            "bagging_freq": 5,
            "min_child_samples": best["min_child_samples"],
            "lambda_l1": best["lambda_l1"],
            "lambda_l2": best["lambda_l2"],
            "verbose": -1,
        }
        best_num_rounds = best.get("num_rounds", 200)

        # Final CV with best params to get detailed results
        def final_train(X_tr, y_tr, X_v, y_v):
            td = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
            vd = lgb.Dataset(X_v, label=y_v, feature_name=feature_names, reference=td)
            callbacks = [lgb.early_stopping(30), lgb.log_evaluation(period=0)]
            return lgb.train(
                best_params, td,
                num_boost_round=best_num_rounds,
                valid_sets=[vd], valid_names=["valid"],
                callbacks=callbacks,
            )

        def final_predict(model, X_pred):
            return model.predict(X_pred)

        final_cv = cv.cross_validate(X, y, sort_keys, final_train, final_predict)

        return TuningResult(
            best_params={**best_params, "num_rounds": best_num_rounds},
            best_score=study.best_value,
            cv_results=final_cv,
            n_trials=len(study.trials),
            study_name="lgbm_tuning",
        )


class XGBoostTuner:
    """
    Optuna-based hyperparameter optimization for XGBoost.

    Searches over:
    - max_depth: [3, 10]
    - learning_rate: [0.01, 0.3]
    - subsample: [0.5, 1.0]
    - colsample_bytree: [0.4, 1.0]
    - min_child_weight: [1, 20]
    - gamma: [0, 5.0]
    - reg_alpha: [1e-8, 10.0]
    - reg_lambda: [1e-8, 10.0]

    Uses TemporalCrossValidator internally to prevent leakage.
    """

    def __init__(
        self,
        n_trials: int = 50,
        n_cv_splits: int = 5,
        timeout: Optional[int] = 300,
        random_seed: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for hyperparameter tuning")
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost required for XGBoostTuner")

        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.timeout = timeout
        self.random_seed = random_seed

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sort_keys: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> TuningResult:
        """
        Run Optuna hyperparameter search with temporal CV.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            sort_keys: Chronological sort keys for temporal splitting
            feature_names: Optional feature names

        Returns:
            TuningResult with best params and CV scores
        """
        cv = TemporalCrossValidator(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbosity": 0,
            }
            num_rounds = trial.suggest_int("num_rounds", 50, 500)

            def train_fn(X_tr, y_tr, X_v, y_v):
                dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
                dval = xgb.DMatrix(X_v, label=y_v, feature_names=feature_names)
                return xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_rounds,
                    evals=[(dval, "valid")],
                    early_stopping_rounds=30,
                    verbose_eval=False,
                )

            def predict_fn(model, X_pred):
                dmat = xgb.DMatrix(X_pred, feature_names=feature_names)
                return model.predict(dmat)

            cv_results = cv.cross_validate(X, y, sort_keys, train_fn, predict_fn)
            mean_brier = float(np.mean([r.brier_score for r in cv_results]))
            return mean_brier

        study = optuna.create_study(
            direction="minimize",
            study_name="xgb_tuning",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params
        best_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": best["min_child_weight"],
            "gamma": best["gamma"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
            "verbosity": 0,
        }
        best_num_rounds = best.get("num_rounds", 200)

        # Final CV with best params
        def final_train(X_tr, y_tr, X_v, y_v):
            dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
            dval = xgb.DMatrix(X_v, label=y_v, feature_names=feature_names)
            return xgb.train(
                best_params, dtrain,
                num_boost_round=best_num_rounds,
                evals=[(dval, "valid")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )

        def final_predict(model, X_pred):
            dmat = xgb.DMatrix(X_pred, feature_names=feature_names)
            return model.predict(dmat)

        final_cv = cv.cross_validate(X, y, sort_keys, final_train, final_predict)

        return TuningResult(
            best_params={**best_params, "num_rounds": best_num_rounds},
            best_score=study.best_value,
            cv_results=final_cv,
            n_trials=len(study.trials),
            study_name="xgb_tuning",
        )


class LogisticTuner:
    """
    Optuna-based tuning for LogisticRegression (fallback when LightGBM unavailable).

    Searches over:
    - C (regularization): [0.001, 100]
    - penalty: l1, l2
    """

    def __init__(
        self,
        n_trials: int = 30,
        n_cv_splits: int = 5,
        timeout: Optional[int] = 120,
        random_seed: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for tuning")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for LogisticTuner")
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.timeout = timeout
        self.random_seed = random_seed

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sort_keys: np.ndarray,
    ) -> TuningResult:
        cv = TemporalCrossValidator(n_splits=self.n_cv_splits)

        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            solver = "saga" if penalty == "l1" else "lbfgs"

            def train_fn(X_tr, y_tr, X_v, y_v):
                model = LogisticRegression(
                    C=C, penalty=penalty, solver=solver, max_iter=2000, random_state=self.random_seed
                )
                model.fit(X_tr, y_tr)
                return model

            def predict_fn(model, X_pred):
                return model.predict_proba(X_pred)[:, 1]

            results = cv.cross_validate(X, y, sort_keys, train_fn, predict_fn)
            return float(np.mean([r.brier_score for r in results]))

        study = optuna.create_study(
            direction="minimize",
            study_name="logistic_tuning",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params
        return TuningResult(
            best_params=best,
            best_score=study.best_value,
            n_trials=len(study.trials),
            study_name="logistic_tuning",
        )


class LeaveOneYearOutCV:
    """
    Leave-One-Year-Out cross-validation for tournament prediction.

    The gold standard for March Madness modeling. Standard K-fold CV
    doesn't work because each tournament year has unique "chaos" patterns.
    LOYO tests whether the model generalizes across different years' dynamics.

    For years [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]:
      Fold 0: train=[2017-2024], test=[2025]
      Fold 1: train=[2017-2023, 2025], test=[2024]
      ... etc

    Note: 2020 is excluded (tournament cancelled due to COVID).
    """

    def __init__(self, years: Optional[List[int]] = None):
        """
        Args:
            years: Years to use for CV. Defaults to 2017-2025 (excluding 2020).
        """
        self.years = years or [y for y in range(2017, 2026) if y != 2020]

    def split(
        self,
        game_years: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate train/test splits by year.

        Args:
            game_years: Array of year labels for each sample

        Returns:
            List of (train_indices, test_indices, held_out_year)
        """
        splits = []
        for hold_out_year in self.years:
            test_mask = game_years == hold_out_year
            train_mask = ~test_mask

            if np.sum(test_mask) < 5:
                continue  # Skip years with too few games

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            splits.append((train_idx, test_idx, hold_out_year))

        return splits

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        game_years: np.ndarray,
        train_fn,
        predict_fn,
    ) -> List[CVResult]:
        """
        Run LOYO cross-validation with arbitrary train/predict functions.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            game_years: Year label for each sample
            train_fn: Callable(X_train, y_train, X_val, y_val) -> model
            predict_fn: Callable(model, X) -> probabilities

        Returns:
            List of CVResult per fold
        """
        splits = self.split(game_years)
        results = []

        for fold_id, (train_idx, test_idx, year) in enumerate(splits):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            model = train_fn(X_train, y_train, X_test, y_test)
            preds = predict_fn(model, X_test)
            preds = np.clip(preds, 1e-7, 1 - 1e-7)

            brier = float(np.mean((preds - y_test) ** 2))
            eps = 1e-7
            ll = float(
                -np.mean(y_test * np.log(preds + eps) + (1 - y_test) * np.log(1 - preds + eps))
            )
            acc = float(np.mean((preds >= 0.5).astype(int) == y_test))

            results.append(
                CVResult(
                    fold_id=fold_id,
                    train_size=len(y_train),
                    val_size=len(y_test),
                    brier_score=brier,
                    log_loss=ll,
                    accuracy=acc,
                )
            )

        return results


class EnsembleWeightOptimizer:
    """
    Grid search over CFA base weights to minimize Brier score.

    Tests weight combinations for (gnn, transformer, baseline) in increments
    of 0.05 that sum to 1.0, then returns the best combination.
    """

    def __init__(self, step: float = 0.05, min_weight: float = 0.05):
        self.step = step
        self.min_weight = min_weight

    def optimize(
        self,
        model_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
        model_confidences: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Find optimal weights by grid search over Brier score.

        Args:
            model_predictions: Dict of model_name -> predicted probabilities [N]
            outcomes: Actual outcomes [N]
            model_confidences: Per-model confidence scores (for CFA diversity bonus)

        Returns:
            Tuple of (best_weights, best_brier_score)
        """
        model_names = sorted(model_predictions.keys())
        if len(model_names) < 2:
            return {model_names[0]: 1.0} if model_names else {}, 0.25

        preds = {name: np.clip(model_predictions[name], 0.01, 0.99) for name in model_names}
        y = np.array(outcomes, dtype=float)

        best_weights = {name: 1.0 / len(model_names) for name in model_names}
        best_brier = float("inf")

        # Generate weight grid
        steps = int(round(1.0 / self.step))
        weight_grid = self._generate_weight_grid(len(model_names), steps)

        for combo in weight_grid:
            # Filter: each weight must be >= min_weight
            if any(w < self.min_weight for w in combo):
                continue

            weights = {name: w for name, w in zip(model_names, combo)}
            combined = np.zeros(len(y))
            for name in model_names:
                combined += weights[name] * preds[name]

            brier = float(np.mean((combined - y) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_weights = weights

        return best_weights, best_brier

    def _generate_weight_grid(
        self, n_models: int, steps: int
    ) -> List[Tuple[float, ...]]:
        """Generate all weight combinations summing to 1.0."""
        if n_models == 1:
            return [(1.0,)]
        if n_models == 2:
            return [
                (i / steps, 1.0 - i / steps)
                for i in range(steps + 1)
            ]

        combos = []
        self._weight_recurse(n_models, steps, steps, [], combos)
        return combos

    def _weight_recurse(
        self,
        remaining: int,
        steps: int,
        budget: int,
        current: list,
        combos: list,
    ) -> None:
        if remaining == 1:
            combos.append(tuple(current + [budget / steps]))
            return
        for i in range(budget + 1):
            self._weight_recurse(remaining - 1, steps, budget - i, current + [i / steps], combos)
