"""Baseline model training — models module."""

import logging

from ....ml.ensemble.cfa import (
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
    LightGBMRanker,
    ModelPrediction,
    XGBoostRanker,
)

# Optional imports — accessed via pipeline._optional_imports pattern
try:
    from ..._optional_imports import (
        BAYESIAN_BT_AVAILABLE,
        OPTUNA_AVAILABLE,
        SCALER_AVAILABLE,
        SKLEARN_AVAILABLE,
        SPREAD_MODEL_AVAILABLE,
        TOURNAMENT_SIGMA_AVAILABLE,
        BayesianBradleyTerry,
        EnsembleWeightOptimizer,
        LeaveOneYearOutCV,
        LightGBMTuner,
        LogisticRegression,
        LogisticTuner,
        SpreadRegressor,
        StandardScaler,
        TemporalCrossValidator,
        XGBoostTuner,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _train_all_models(
    pipeline,
    train_X,
    train_y,
    train_margins,
    eval_X,
    eval_y,
    eval_margins,
    feature_names,
    train_samples,
    valid_samples,
    train_sample_weight,
    valid_set,
    train_sort_keys,
    bt_game_triples,
):
    """Train LightGBM, XGBoost, logistic, spread, BT, and CalFirst models.

    Returns:
        Tuple of (trained_models, tuning_stats).
    """
    tuning_stats = {}

    # ====================================================================
    # MODEL TRAINING: Try LightGBM + XGBoost + Logistic, then optionally
    # stack them with a meta-learner for superior ensemble performance.
    # ====================================================================
    trained_models = []  # List of (name, model, predictions_on_eval)

    # Gap #3: In "simple" mode, skip LGB and XGB entirely.
    # A well-calibrated logistic regression + spread model on ~9 features
    # is competitive with (or better than) complex ensembles at ~600 samples.
    _use_tree_models = pipeline.config.model_complexity != "simple"

    # Phase 2: In production mode, LGB/XGB classifiers are disabled.
    # They remain available in experimental mode for research.
    _production_mode = pipeline.config.pipeline_mode == "production"
    _allow_lgb_classifier = pipeline.config.experimental_enable_lgb_classifier
    _allow_xgb_classifier = pipeline.config.experimental_enable_xgb_classifier

    # --- LightGBM training (experimental only in Phase 2) ---
    lgb_trained = False
    if LIGHTGBM_AVAILABLE and _use_tree_models and _allow_lgb_classifier:
        try:
            if (
                pipeline.config.enable_hyperparameter_tuning
                and OPTUNA_AVAILABLE
                and LightGBMTuner is not None
                and train_samples >= 60
            ):
                tuner = LightGBMTuner(
                    n_trials=pipeline.config.optuna_n_trials,
                    n_cv_splits=pipeline.config.temporal_cv_splits,
                    timeout=pipeline.config.optuna_timeout,
                    random_seed=pipeline.config.random_seed,
                )
                tuning_result = tuner.tune(
                    train_X,
                    train_y,
                    train_sort_keys,
                    feature_names=feature_names,
                    sample_weight=train_sample_weight,
                    development_years=list(pipeline.config.training_years or []),
                    year_split_policy=getattr(pipeline, "_year_split_policy", None),
                )

                _exclude_keys = {"num_rounds", "objective", "metric"}
                best_params = {k: v for k, v in tuning_result.best_params.items() if k not in _exclude_keys}
                best_num_rounds = tuning_result.best_params.get("num_rounds", 200)
                lgb_ranker = LightGBMRanker(params=best_params)
                lgb_ranker.train(
                    train_X,
                    train_y,
                    feature_names=feature_names,
                    num_rounds=best_num_rounds,
                    early_stopping_rounds=30 if valid_set is not None else None,
                    valid_set=valid_set,
                    sample_weight=train_sample_weight,
                )
                lgb_eval_preds = lgb_ranker.predict(eval_X)
                trained_models.append(("lgb", lgb_ranker, lgb_eval_preds))
                lgb_trained = True

                tuning_stats["lightgbm"] = {
                    "method": "optuna",
                    "n_trials": tuning_result.n_trials,
                    "best_optuna_score": round(tuning_result.best_score, 5),
                    "optuna_metric": "brier",
                    "best_params": {
                        k: round(v, 5) if isinstance(v, float) else v for k, v in tuning_result.best_params.items()
                    },
                    "cv_folds": len(tuning_result.cv_results),
                    "cv_brier_scores": [round(r.brier_score, 5) for r in tuning_result.cv_results],
                }
            else:
                lgb_ranker = LightGBMRanker()
                lgb_ranker.train(
                    train_X,
                    train_y,
                    feature_names=feature_names,
                    num_rounds=200,
                    early_stopping_rounds=30 if valid_set is not None else None,
                    valid_set=valid_set,
                    sample_weight=train_sample_weight,
                )
                lgb_eval_preds = lgb_ranker.predict(eval_X)
                trained_models.append(("lgb", lgb_ranker, lgb_eval_preds))
                lgb_trained = True
        except Exception as e:
            tuning_stats["lightgbm_error"] = str(e)

    # --- XGBoost training (experimental only in Phase 2) ---
    xgb_trained = False
    if XGBOOST_AVAILABLE and _use_tree_models and _allow_xgb_classifier:
        try:
            if (
                pipeline.config.enable_hyperparameter_tuning
                and OPTUNA_AVAILABLE
                and XGBoostTuner is not None
                and train_samples >= 60
            ):
                xgb_tuner = XGBoostTuner(
                    n_trials=pipeline.config.optuna_n_trials,
                    n_cv_splits=pipeline.config.temporal_cv_splits,
                    timeout=pipeline.config.optuna_timeout,
                    random_seed=pipeline.config.random_seed,
                )
                xgb_tuning_result = xgb_tuner.tune(
                    train_X,
                    train_y,
                    train_sort_keys,
                    feature_names=feature_names,
                    sample_weight=train_sample_weight,
                    development_years=list(pipeline.config.training_years or []),
                    year_split_policy=getattr(pipeline, "_year_split_policy", None),
                )

                xgb_best_params = {k: v for k, v in xgb_tuning_result.best_params.items() if k != "num_rounds"}
                xgb_best_rounds = xgb_tuning_result.best_params.get("num_rounds", 200)

                xgb_ranker = XGBoostRanker(params=xgb_best_params)
                xgb_ranker.train(
                    train_X,
                    train_y,
                    feature_names=feature_names,
                    num_rounds=xgb_best_rounds,
                    early_stopping_rounds=30 if valid_set is not None else None,
                    valid_set=valid_set,
                    sample_weight=train_sample_weight,
                )
                xgb_eval_preds = xgb_ranker.predict(eval_X)
                trained_models.append(("xgb", xgb_ranker, xgb_eval_preds))
                xgb_trained = True

                tuning_stats["xgboost"] = {
                    "method": "optuna",
                    "n_trials": xgb_tuning_result.n_trials,
                    "best_optuna_score": round(xgb_tuning_result.best_score, 5),
                    "optuna_metric": "brier",
                    "best_params": {
                        k: round(v, 5) if isinstance(v, float) else v for k, v in xgb_tuning_result.best_params.items()
                    },
                }
            else:
                xgb_ranker = XGBoostRanker()
                xgb_ranker.train(
                    train_X,
                    train_y,
                    feature_names=feature_names,
                    num_rounds=200,
                    early_stopping_rounds=30 if valid_set is not None else None,
                    valid_set=valid_set,
                    sample_weight=train_sample_weight,
                )
                xgb_eval_preds = xgb_ranker.predict(eval_X)
                trained_models.append(("xgb", xgb_ranker, xgb_eval_preds))
                xgb_trained = True
        except Exception as e:
            tuning_stats["xgboost_error"] = str(e)

    # --- Logistic regression ---
    # Phase 2: Always trained in both production and experimental mode.
    # In production mode, this is one of two sanctioned models.
    logit_trained = False
    if SKLEARN_AVAILABLE:
        try:
            if (
                pipeline.config.enable_hyperparameter_tuning
                and OPTUNA_AVAILABLE
                and LogisticTuner is not None
                and train_samples >= 60
            ):
                logit_tuner = LogisticTuner(
                    n_trials=min(pipeline.config.optuna_n_trials, 30),
                    n_cv_splits=pipeline.config.temporal_cv_splits,
                    timeout=min(pipeline.config.optuna_timeout, 120),
                    random_seed=pipeline.config.random_seed,
                )
                logit_tuning_result = logit_tuner.tune(
                    train_X,
                    train_y,
                    train_sort_keys,
                    sample_weight=train_sample_weight,
                    development_years=list(pipeline.config.training_years or []),
                    year_split_policy=getattr(pipeline, "_year_split_policy", None),
                )
                best_logit = logit_tuning_result.best_params
                _l1r = best_logit.get("l1_ratio", 0)
                logit = LogisticRegression(
                    C=best_logit["C"],
                    l1_ratio=_l1r,
                    solver="saga" if _l1r == 1 else "lbfgs",
                    max_iter=2000,
                    random_state=pipeline.config.random_seed,
                )
                tuning_stats["logistic"] = {
                    "method": "optuna",
                    "best_optuna_score": round(logit_tuning_result.best_score, 5),
                    "optuna_metric": "brier",
                    "best_params": best_logit,
                }
            else:
                logit = LogisticRegression(
                    C=1.0,
                    l1_ratio=0,
                    max_iter=2000,
                    random_state=pipeline.config.random_seed,
                )
            # FIX C1: LR cannot handle NaN — impute with column median
            # (training data only) before fitting.  Tree models see the
            # original NaN-preserving train_X; only LR gets imputed copy.
            _lr_train_X = train_X
            _lr_eval_X = eval_X if eval_X.shape[0] > 0 else None
            if int(np.isnan(train_X).sum()) > 0:
                from sklearn.impute import SimpleImputer

                _lr_imputer = SimpleImputer(strategy="median")
                _lr_train_X = _lr_imputer.fit_transform(train_X)
                if _lr_eval_X is not None:
                    _lr_eval_X = _lr_imputer.transform(eval_X)
                logger.debug("LR path: imputed %d NaN values with median.", int(np.isnan(train_X).sum()))
            logit.fit(_lr_train_X, train_y, sample_weight=train_sample_weight)

            # Coefficient stability diagnostic: large coefficient magnitudes
            # signal multicollinearity inflating LogisticRegression estimates.
            # Log a warning when detected so users can investigate.
            if hasattr(logit, "coef_") and feature_names is not None:
                coefs = np.abs(logit.coef_.ravel())
                if len(coefs) == len(feature_names):
                    max_coef_idx = int(np.argmax(coefs))
                    max_coef = float(coefs[max_coef_idx])
                    median_coef = float(np.median(coefs))
                    if max_coef > 10.0 and median_coef > 0:
                        ratio = max_coef / median_coef
                        if ratio > 20.0:
                            logger.warning(
                                "LogisticRegression coefficient instability: "
                                "'%s' has |coef|=%.2f (%.0fx median). "
                                "Possible residual multicollinearity.",
                                feature_names[max_coef_idx],
                                max_coef,
                                ratio,
                            )
                            tuning_stats["logistic_coef_warning"] = {
                                "feature": feature_names[max_coef_idx],
                                "abs_coef": round(max_coef, 3),
                                "ratio_to_median": round(ratio, 1),
                            }

            # FIX C1: Use imputed eval_X for LR predictions
            _lr_pred_X = _lr_eval_X if _lr_eval_X is not None else eval_X
            logit_eval_preds = logit.predict_proba(_lr_pred_X)[:, 1]
            # Store imputer on model object so prediction path can use it
            logit._nan_imputer = _lr_imputer if int(np.isnan(train_X).sum()) > 0 else None
            trained_models.append(("logit", logit, logit_eval_preds))
            logit_trained = True
        except Exception as e:
            tuning_stats["logistic_error"] = str(e)

    # --- Point-spread regression model ---
    # Trains LightGBM regression on actual margins, converts to P(win)
    # via logistic CDF.  Provides complementary signal to binary
    # classifiers — richer gradient from continuous target.
    spread_trained = False
    if (
        pipeline.config.enable_spread_model
        and SPREAD_MODEL_AVAILABLE
        and LIGHTGBM_AVAILABLE
        and _use_tree_models
        and train_samples >= 60
        and len(train_margins) == len(train_y)
    ):
        try:
            spread = SpreadRegressor(
                sigma=pipeline.config.spread_sigma_init,
            )
            # FIX-STACKING-LEAKAGE: Do NOT pass eval data for sigma
            # calibration.  calibrate_sigma() optimizes sigma to minimize
            # Brier on its input — passing eval data here contaminates all
            # spread probabilities used downstream for ensemble evaluation.
            # Instead, hold out the last 20% of training data for sigma
            # calibration.  This keeps sigma derivation strictly within
            # the training partition.
            _sigma_split = max(int(len(train_margins) * 0.8), 20)
            if len(train_margins) >= 40:
                _sigma_cal_X = train_X[_sigma_split:]
                _sigma_cal_margins = train_margins[_sigma_split:]
                _sigma_train_X = train_X[:_sigma_split]
                _sigma_train_margins = train_margins[:_sigma_split]
                _sigma_train_weight = train_sample_weight[:_sigma_split] if train_sample_weight is not None else None
            else:
                # Too few samples to split — skip sigma calibration
                _sigma_cal_X = None
                _sigma_cal_margins = None
                _sigma_train_X = train_X
                _sigma_train_margins = train_margins
                _sigma_train_weight = train_sample_weight

            spread_stats = spread.train(
                _sigma_train_X,
                _sigma_train_margins,
                feature_names=feature_names,
                num_rounds=200,
                sample_weight=_sigma_train_weight,
                valid_X=_sigma_cal_X,
                valid_margins=_sigma_cal_margins,
            )

            # Retrain on full training data with calibrated sigma
            if _sigma_cal_X is not None:
                _calibrated_sigma = spread.sigma
                spread_full = SpreadRegressor(sigma=_calibrated_sigma)
                spread_full.train(
                    train_X,
                    train_margins,
                    feature_names=feature_names,
                    num_rounds=200,
                    sample_weight=train_sample_weight,
                    valid_X=None,
                    valid_margins=None,
                )
                # Preserve calibrated sigma (train with no valid data won't recalibrate)
                spread_full.sigma = _calibrated_sigma
                spread = spread_full

            if spread_stats.get("trained"):
                pipeline.baseline_model.spread_model = spread
                spread_trained = True

                # Generate eval predictions (probability) for ensemble weighting
                if valid_samples > 0:
                    spread_eval_preds = spread.predict_probability(eval_X)
                else:
                    spread_eval_preds = np.array([])

                trained_models.append(("spread", spread, spread_eval_preds))
                tuning_stats["spread_model"] = spread_stats
                logger.info(
                    "SpreadRegressor trained: rmse=%.3f, sigma=%.2f",
                    spread_stats.get("train_rmse", 0),
                    spread_stats.get("sigma", 11.0),
                )
        except Exception as e:
            tuning_stats["spread_model_error"] = str(e)
            logger.warning("SpreadRegressor training failed: %s", e)

    # --- Tournament-specific sigma calibration ---
    # The spread model's sigma was calibrated on regular-season validation
    # data.  Tournament games have systematically tighter spread distributions
    # (neutral sites, better opponents, single-elimination pressure).  Using
    # regular-season sigma (≈11) in tournament predictions systematically
    # miscalibrates probabilities — especially in late rounds (F4/NCG) where
    # Kaggle applies 16-32× scoring weight.
    #
    # Solution: fit a TournamentSigmaCalibrator from historical tournament
    # data (Kaggle CSVs), then override the SpreadRegressor's sigma with the
    # tournament-calibrated value.  Per-round sigmas flow through the
    # _TrainedBaselineModel's tournament_sigma_calibrator.
    # Phase 2: TournamentSigmaCalibrator is experimental only.
    # In production mode, SpreadRegressor uses its validation-calibrated sigma
    # and TemperatureScaling is the sole final calibration layer.
    if spread_trained and TOURNAMENT_SIGMA_AVAILABLE:
        try:
            pipeline._fit_tournament_sigma(spread, tuning_stats)
        except Exception as e:
            logger.warning("Tournament sigma calibration failed: %s", e)
            tuning_stats["tournament_sigma_error"] = str(e)

    # --- Bayesian Bradley-Terry rating model ---
    # ID-based model: captures "who beat whom" without engineered features.
    # Fitted on current-year game triples (team1_id, team2_id, outcome).
    # Predictions are made via predict_probability(team1, team2) at
    # inference time — not through the feature-based ensemble.
    if pipeline.config.enable_bayesian_bt and BAYESIAN_BT_AVAILABLE and len(bt_game_triples) >= 50:
        try:
            bt_model = BayesianBradleyTerry(
                prior_std=pipeline.config.bayesian_bt_prior_std,
            )
            bt_stats = bt_model.fit(bt_game_triples)
            if bt_stats.get("fitted"):
                pipeline.bayesian_bt_model = bt_model
                tuning_stats["bayesian_bt"] = bt_stats
                logger.info(
                    "BayesianBT: fitted %d teams from %d games, mean_posterior_std=%.3f",
                    bt_stats.get("n_teams", 0),
                    bt_stats.get("n_games", 0),
                    bt_stats.get("mean_posterior_std", 0),
                )
        except Exception as e:
            tuning_stats["bayesian_bt_error"] = str(e)
            logger.warning("BayesianBT fitting failed: %s", e)

    return trained_models, tuning_stats
