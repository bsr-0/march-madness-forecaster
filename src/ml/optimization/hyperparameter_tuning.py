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

    def __init__(self, n_splits: int = 5, min_train_size: int = 30, pair_size: int = 1):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        # FIX #A: pair_size > 1 snaps split boundaries to multiples of
        # pair_size so that symmetric sample pairs (e.g., game + reversed
        # game) always land in the same fold.  Set pair_size=2 for the
        # standard symmetric-augmented game data.
        self.pair_size = max(1, pair_size)

    def _snap_to_pair(self, idx: int) -> int:
        """Round index down to nearest pair boundary."""
        if self.pair_size <= 1:
            return idx
        return (idx // self.pair_size) * self.pair_size

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
        initial_train = self._snap_to_pair(initial_train)
        remaining = n_samples - initial_train

        if remaining < self.n_splits:
            # Not enough data for requested splits; return single split
            split_point = max(self.min_train_size, int(0.8 * n_samples))
            split_point = self._snap_to_pair(split_point)
            return [
                TemporalSplit(
                    train_indices=indices[:split_point],
                    val_indices=indices[split_point:],
                    fold_id=0,
                )
            ]

        val_size = remaining // self.n_splits
        val_size = max(self._snap_to_pair(val_size), self.pair_size)
        splits = []

        for fold in range(self.n_splits):
            val_start = initial_train + fold * val_size
            val_start = self._snap_to_pair(val_start)
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
        sample_weight: np.ndarray = None,
    ) -> List[CVResult]:
        """
        Run temporal cross-validation with arbitrary train/predict functions.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            sort_keys: Chronological sort keys
            train_fn: Callable(X_train, y_train, X_val, y_val, w_train) -> model
            predict_fn: Callable(model, X) -> probabilities
            sample_weight: Optional per-sample weights [N]. Sliced per fold
                and passed as the 5th argument to train_fn. None = uniform.

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
            w_train = sample_weight[split.train_indices] if sample_weight is not None else None

            model = train_fn(X_train, y_train, X_val, y_val, w_train)
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
        n_trials: int = 15,
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
        sample_weight: np.ndarray = None,
    ) -> TuningResult:
        """
        Run Optuna hyperparameter search with temporal CV.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            sort_keys: Chronological sort keys for temporal splitting
            feature_names: Optional feature names
            sample_weight: Optional per-sample weights for weighted training.
                Propagated through CV folds so hyperparameters are selected
                under the same weighting scheme used for final model training.

        Returns:
            TuningResult with best params and CV scores
        """
        # B3: pair_size=1 — symmetric augmentation was removed, so each game
        # is a single sample. pair_size=2 was vestigial.
        cv = TemporalCrossValidator(n_splits=self.n_cv_splits, pair_size=1)

        def objective(trial: optuna.Trial) -> float:
            # OOS-FIX: Heavily constrained search space to prevent overfitting
            # with ~400 training samples.  num_leaves<=16 and min_child_samples
            # >=30 force shallow, well-regularized trees.  15 trials with this
            # narrow space is sufficient — most configurations will generalize
            # similarly.
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 4, 16),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
                "bagging_freq": 5,
                "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0, log=True),
                "verbose": -1,
                "num_threads": 1,
            }
            # FIX M2: Use a fixed number of rounds during Optuna search
            # instead of early stopping on the val fold.
            num_rounds = trial.suggest_int("num_rounds", 50, 200)

            def train_fn(X_tr, y_tr, X_v, y_v, w_tr):
                train_data = lgb.Dataset(
                    X_tr, label=y_tr, feature_name=feature_names,
                    weight=w_tr,
                )
                callbacks = [lgb.log_evaluation(period=0)]
                return lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_rounds,
                    callbacks=callbacks,
                )

            def predict_fn(model, X_pred):
                return model.predict(X_pred)

            cv_results = cv.cross_validate(
                X, y, sort_keys, train_fn, predict_fn,
                sample_weight=sample_weight,
            )
            mean_brier = float(np.mean([r.brier_score for r in cv_results]))
            return mean_brier

        study = optuna.create_study(
            direction="minimize",
            study_name="lgbm_tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params
        # FIX: Extract Optuna-tuned num_rounds (was hardcoded to 500).
        best_num_rounds = best.get("num_rounds", 200)
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
            "num_threads": 1,
        }

        # Final CV with best params — use fixed num_rounds (no early
        # stopping) to keep val fold uncontaminated for evaluation.
        def final_train(X_tr, y_tr, X_v, y_v, w_tr):
            td = lgb.Dataset(
                X_tr, label=y_tr, feature_name=feature_names,
                weight=w_tr,
            )
            callbacks = [lgb.log_evaluation(period=0)]
            return lgb.train(
                best_params, td,
                num_boost_round=best_num_rounds,
                callbacks=callbacks,
            )

        def final_predict(model, X_pred):
            return model.predict(X_pred)

        final_cv = cv.cross_validate(
            X, y, sort_keys, final_train, final_predict,
            sample_weight=sample_weight,
        )

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
        n_trials: int = 15,
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
        sample_weight: np.ndarray = None,
    ) -> TuningResult:
        """
        Run Optuna hyperparameter search with temporal CV.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            sort_keys: Chronological sort keys for temporal splitting
            feature_names: Optional feature names
            sample_weight: Optional per-sample weights for weighted training.

        Returns:
            TuningResult with best params and CV scores
        """
        # B3: pair_size=1 — symmetric augmentation was removed.
        cv = TemporalCrossValidator(n_splits=self.n_cv_splits, pair_size=1)

        def objective(trial: optuna.Trial) -> float:
            # OOS-FIX: Constrained search space — max_depth<=4 and
            # min_child_weight>=5 prevent overfitting on ~400 samples.
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
                "gamma": trial.suggest_float("gamma", 0.1, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "verbosity": 0,
                "nthread": 1,
            }
            # FIX M2: Fixed rounds during Optuna search.
            num_rounds = trial.suggest_int("num_rounds", 50, 200)

            def train_fn(X_tr, y_tr, X_v, y_v, w_tr):
                dtrain = xgb.DMatrix(
                    X_tr, label=y_tr, feature_names=feature_names,
                    weight=w_tr,
                )
                return xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_rounds,
                    verbose_eval=False,
                )

            def predict_fn(model, X_pred):
                dmat = xgb.DMatrix(X_pred, feature_names=feature_names)
                return model.predict(dmat)

            cv_results = cv.cross_validate(
                X, y, sort_keys, train_fn, predict_fn,
                sample_weight=sample_weight,
            )
            mean_brier = float(np.mean([r.brier_score for r in cv_results]))
            return mean_brier

        study = optuna.create_study(
            direction="minimize",
            study_name="xgb_tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best = study.best_params
        # FIX: Extract Optuna-tuned num_rounds (was hardcoded to 500).
        best_num_rounds = best.get("num_rounds", 200)
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
            "nthread": 1,
        }

        # Final CV with best params — fixed rounds, no early stopping.
        def final_train(X_tr, y_tr, X_v, y_v, w_tr):
            dtrain = xgb.DMatrix(
                X_tr, label=y_tr, feature_names=feature_names,
                weight=w_tr,
            )
            return xgb.train(
                best_params, dtrain,
                num_boost_round=best_num_rounds,
                verbose_eval=False,
            )

        def final_predict(model, X_pred):
            dmat = xgb.DMatrix(X_pred, feature_names=feature_names)
            return model.predict(dmat)

        final_cv = cv.cross_validate(
            X, y, sort_keys, final_train, final_predict,
            sample_weight=sample_weight,
        )

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
    - C (regularization): [0.01, 10.0]
    - penalty: l1, l2, elasticnet
    - l1_ratio (for elasticnet): [0.1, 0.9]
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
        sample_weight: np.ndarray = None,
    ) -> TuningResult:
        # B3: pair_size=1 — symmetric augmentation was removed.
        cv = TemporalCrossValidator(n_splits=self.n_cv_splits, pair_size=1)

        def objective(trial: optuna.Trial) -> float:
            # FIX 7.2: Tightened C range [0.01, 10.0] (was [0.001, 100]).
            # High C values effectively disable regularization.
            # Added elasticnet with tunable l1_ratio for correlated features.
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            # saga solver required for l1 and elasticnet
            solver = "saga"
            l1_ratio = None
            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

            def train_fn(X_tr, y_tr, X_v, y_v, w_tr):
                kwargs = dict(
                    C=C, penalty=penalty, solver=solver, max_iter=2000,
                    random_state=self.random_seed,
                )
                if l1_ratio is not None:
                    kwargs["l1_ratio"] = l1_ratio
                model = LogisticRegression(**kwargs)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                return model

            def predict_fn(model, X_pred):
                return model.predict_proba(X_pred)[:, 1]

            results = cv.cross_validate(
                X, y, sort_keys, train_fn, predict_fn,
                sample_weight=sample_weight,
            )
            return float(np.mean([r.brier_score for r in results]))

        study = optuna.create_study(
            direction="minimize",
            study_name="logistic_tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
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

    Supports two temporal modes:

    ``rolling_window`` (default, recommended):
      Trains on all years **before** the held-out year only.  This is the
      honest causal evaluation — no future-year data leaks into training.
      Fold for 2023: train=[2017-2022], test=[2023].

    ``leave_one_out`` (original):
      Trains on all years **except** the held-out year (including future years).
      Fold for 2023: train=[2017-2022, 2024-2025], test=[2023].
      Overstates generalization because future information is available.

    Note: 2020 is excluded (tournament cancelled due to COVID).
    """

    def __init__(
        self,
        years: Optional[List[int]] = None,
        temporal_mode: str = "rolling_window",
    ):
        """
        Args:
            years: Years to use for CV. Defaults to 2017-2025 (excluding 2020).
            temporal_mode: ``"rolling_window"`` (train on past only) or
                ``"leave_one_out"`` (train on all other years).
        """
        self.years = years or [y for y in range(2017, 2026) if y != 2020]
        if temporal_mode not in ("rolling_window", "leave_one_out"):
            raise ValueError(
                f"temporal_mode must be 'rolling_window' or 'leave_one_out', "
                f"got '{temporal_mode}'"
            )
        self.temporal_mode = temporal_mode

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

            if self.temporal_mode == "rolling_window":
                # Train on strictly earlier years only — honest causal split
                train_mask = game_years < hold_out_year
            else:
                # Original: train on all years except held-out
                train_mask = ~test_mask

            if np.sum(test_mask) < 5:
                continue  # Skip years with too few games
            if np.sum(train_mask) < 5:
                continue  # Skip if not enough training data (early years in rolling mode)

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
        sample_weight: np.ndarray = None,
    ) -> List[CVResult]:
        """
        Run LOYO cross-validation with arbitrary train/predict functions.

        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            game_years: Year label for each sample
            train_fn: Callable(X_train, y_train, X_val, y_val, w_train) -> model
            predict_fn: Callable(model, X) -> probabilities
            sample_weight: Optional per-sample weights [N]. Sliced per fold.

        Returns:
            List of CVResult per fold
        """
        splits = self.split(game_years)
        results = []

        for fold_id, (train_idx, test_idx, year) in enumerate(splits):
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Hold out 15% of training data for early stopping to prevent
            # leaking test-year labels into the training process.
            n_tr = len(train_idx)
            es_size = max(10, int(0.15 * n_tr))
            es_idx = train_idx[-es_size:]   # chronologically last portion
            tr_idx = train_idx[:-es_size]

            X_train = X[tr_idx]
            y_train = y[tr_idx]
            X_es = X[es_idx]
            y_es = y[es_idx]
            w_train = sample_weight[tr_idx] if sample_weight is not None else None

            model = train_fn(X_train, y_train, X_es, y_es, w_train)
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
    Bootstrap-aggregated grid search over CFA base weights.

    Tests weight combinations for (gnn, transformer, baseline) in increments
    of 0.05 that sum to 1.0.  To prevent overfitting on small validation sets,
    the grid search is run on multiple bootstrap resamples and the selected
    weights are averaged across resamples (bootstrap aggregation / bagging).
    This stabilizes weight selection when the validation set has <100 samples,
    where a single grid search would overfit to noise in the specific holdout.
    """

    def __init__(
        self,
        step: float = 0.05,
        min_weight: float = 0.05,
        n_bootstrap: int = 100,
        random_seed: int = 42,
    ):
        self.step = step
        self.min_weight = min_weight
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed

    def optimize(
        self,
        model_predictions: Dict[str, np.ndarray],
        outcomes: np.ndarray,
        model_confidences: Optional[Dict[str, float]] = None,
        min_samples: int = 50,
        regularization_lambda: float = 0.1,
    ) -> Tuple[Dict[str, float], float]:
        """
        Find optimal weights by bootstrap-aggregated grid search over Brier score.

        Runs the grid search on ``n_bootstrap`` resamples (with replacement)
        of the validation data.  The final weights are the average of the
        per-resample best weights.  This is equivalent to bagging the weight
        selector, preventing it from latching onto noise in a small holdout.

        If ``n < min_samples``, optimization is skipped and uniform weights
        are returned to avoid fitting noise on tiny samples.

        When ``regularization_lambda > 0``, the objective includes an L2
        penalty toward uniform weights:
          penalized_brier = brier + lambda * ||w - w_uniform||^2

        Args:
            model_predictions: Dict of model_name -> predicted probabilities [N]
            outcomes: Actual outcomes [N]
            model_confidences: Per-model confidence scores (unused, kept for API compat)
            min_samples: Minimum samples required; returns uniform below this.
            regularization_lambda: L2 regularization toward uniform weights.

        Returns:
            Tuple of (best_weights, best_brier_score)
        """
        model_names = sorted(model_predictions.keys())
        if len(model_names) < 2:
            return {model_names[0]: 1.0} if model_names else {}, 0.25

        preds = {name: np.clip(model_predictions[name], 0.01, 0.99) for name in model_names}
        y = np.array(outcomes, dtype=float)
        n = len(y)

        # Fix 8: Minimum sample guard — skip optimization when data is too sparse
        n_models = len(model_names)
        uniform_weights = {name: 1.0 / n_models for name in model_names}
        if n < min_samples:
            import logging
            logging.getLogger(__name__).warning(
                "Ensemble weight optimization: only %d samples (minimum %d); "
                "returning uniform weights.",
                n, min_samples,
            )
            combined = np.zeros(n)
            for name in model_names:
                combined += uniform_weights[name] * preds[name]
            uniform_brier = float(np.mean((combined - y) ** 2))
            return uniform_weights, uniform_brier

        # Generate weight grid once (shared across bootstrap resamples)
        steps = int(round(1.0 / self.step))
        weight_grid = self._generate_weight_grid(len(model_names), steps)
        # Pre-filter: each weight must be >= min_weight
        weight_grid = [combo for combo in weight_grid if all(w >= self.min_weight for w in combo)]

        if not weight_grid:
            uniform = {name: 1.0 / len(model_names) for name in model_names}
            return uniform, 0.25

        rng = np.random.default_rng(self.random_seed)

        # Accumulate weights across bootstrap resamples
        weight_accum = {name: 0.0 for name in model_names}

        for _ in range(self.n_bootstrap):
            # Bootstrap resample (with replacement)
            idx = rng.choice(n, size=n, replace=True)
            y_boot = y[idx]
            preds_boot = {name: preds[name][idx] for name in model_names}

            best_brier_boot = float("inf")
            best_combo_boot = weight_grid[0]

            uniform_arr = np.array([1.0 / n_models] * n_models)
            for combo in weight_grid:
                combined = np.zeros(n)
                for name, w in zip(model_names, combo):
                    combined += w * preds_boot[name]
                brier = float(np.mean((combined - y_boot) ** 2))
                # Fix 8: L2 regularization toward uniform weights
                if regularization_lambda > 0:
                    combo_arr = np.array(combo)
                    brier += regularization_lambda * float(np.sum((combo_arr - uniform_arr) ** 2))
                if brier < best_brier_boot:
                    best_brier_boot = brier
                    best_combo_boot = combo

            for name, w in zip(model_names, best_combo_boot):
                weight_accum[name] += w

        # Average across bootstrap resamples
        best_weights = {name: weight_accum[name] / self.n_bootstrap for name in model_names}

        # Compute Brier score of the averaged weights on full data
        combined = np.zeros(n)
        for name in model_names:
            combined += best_weights[name] * preds[name]
        best_brier = float(np.mean((combined - y) ** 2))

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
