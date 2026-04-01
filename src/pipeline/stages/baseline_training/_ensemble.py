"""Baseline model training — ensemble module."""


import logging
import math
import os

from ....data.models.game_flow import GameFlow
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
        BrierLightGBMTuner,
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

# BMA ensemble (Protocol v2, Section 3.2)
try:
    from ....ml.ensemble.bma import BayesianModelAveraging, BMAResult
    BMA_AVAILABLE = True
except ImportError:
    BMA_AVAILABLE = False

# Brier-objective LightGBM (Protocol Section 3.3, Phase 4)
try:
    from ....ml.ensemble.brier_objective import BrierLightGBMRanker
    BRIER_LGB_AVAILABLE = True
except ImportError:
    BRIER_LGB_AVAILABLE = False

# Calibration-first pipeline (Phase 4 research)
try:
    from ....ml.ensemble.calibration_first import CalibrationFirstPipeline
    CALIBRATION_FIRST_AVAILABLE = True
except ImportError:
    CALIBRATION_FIRST_AVAILABLE = False

logger = logging.getLogger(__name__)


from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _select_ensemble_and_evaluate(pipeline, trained_models, tuning_stats,
                                 train_X, train_y, train_margins, train_sort_keys,
                                 train_sample_weight, train_samples,
                                 eval_X, eval_y, eval_margins, valid_samples,
                                 feature_names, feature_names_full,
                                 _loyo_raw_feature_dim, n_unique_games, n,
                                 historical_training_stats, fs_stats, dist_shift_stats):
    """Select ensemble strategy, evaluate, run LOYO and audits.

    Returns:
        Result dict with model info, metrics, and diagnostics.
    """
    stacking_stats = {}
    _production_mode = pipeline.config.pipeline_mode == "production"

    # ====================================================================
    # MODEL SELECTION / ENSEMBLE
    #
    # OOS-FIX: When stacking is disabled (the new default), use a simple
    # fixed-weight average of all trained models.  This avoids fitting a
    # meta-learner on ~400 OOF samples.  The weights are based on typical
    # Kaggle March Madness competition performance:
    #   LGB: 0.45, XGB: 0.35, Logistic: 0.20
    # When stacking IS enabled (opt-in), the original learned meta-learner
    # path is preserved.
    # ====================================================================
    # Stacking: meta-learner over base model predictions.
    if (
        pipeline.config.enable_stacking
        and SKLEARN_AVAILABLE
        and len(trained_models) >= 2
        and valid_samples >= 20
    ):
        # --- Learned stacking path (opt-in, original behavior) ---
        stacking_cv = TemporalCrossValidator(n_splits=min(3, pipeline.config.temporal_cv_splits), pair_size=1)
        oof_preds = {name: np.full(train_samples, 0.5) for name, _, _ in trained_models}
        oof_counts = np.zeros(train_samples)

        _tuned_lgb_params = None
        _tuned_lgb_rounds = 200
        _tuned_xgb_params = None
        _tuned_xgb_rounds = 200
        _tuned_logit_params = {"C": 1.0, "penalty": "l2"}
        for name, model, _ in trained_models:
            if name == "lgb" and hasattr(model, 'params'):
                _tuned_lgb_params = model.params
                if tuning_stats.get("lightgbm", {}).get("best_params"):
                    _tuned_lgb_rounds = tuning_stats["lightgbm"]["best_params"].get("num_rounds", 200)
            elif name == "xgb" and hasattr(model, 'params'):
                _tuned_xgb_params = model.params
                if tuning_stats.get("xgboost", {}).get("best_params"):
                    _tuned_xgb_rounds = tuning_stats["xgboost"]["best_params"].get("num_rounds", 200)
            elif name == "logit" and hasattr(model, 'C'):
                _tuned_logit_params = {
                    "C": model.C,
                    "penalty": model.penalty if hasattr(model, 'penalty') else "l2",
                }

        for split in stacking_cv.split(train_samples, train_sort_keys):
            X_tr_fold = train_X[split.train_indices]
            y_tr_fold = train_y[split.train_indices]
            X_val_fold = train_X[split.val_indices]
            w_tr_fold = train_sample_weight[split.train_indices] if train_sample_weight is not None else None

            for name, model_template, _ in trained_models:
                if name == "lgb":
                    fold_model = LightGBMRanker(params=_tuned_lgb_params)
                    fold_model.train(X_tr_fold, y_tr_fold, feature_names=feature_names, num_rounds=_tuned_lgb_rounds, early_stopping_rounds=None, sample_weight=w_tr_fold)
                    fold_preds = fold_model.predict(X_val_fold)
                elif name == "xgb":
                    fold_model = XGBoostRanker(params=_tuned_xgb_params)
                    fold_model.train(X_tr_fold, y_tr_fold, feature_names=feature_names, num_rounds=_tuned_xgb_rounds, early_stopping_rounds=None, sample_weight=w_tr_fold)
                    fold_preds = fold_model.predict(X_val_fold)
                elif name == "logit":
                    solver = "saga" if _tuned_logit_params["penalty"] == "l1" else "lbfgs"
                    fold_model = LogisticRegression(
                        C=_tuned_logit_params["C"],
                        penalty=_tuned_logit_params["penalty"],
                        solver=solver,
                        max_iter=2000,
                        random_state=pipeline.config.random_seed,
                    )
                    fold_model.fit(X_tr_fold, y_tr_fold, sample_weight=w_tr_fold)
                    fold_preds = fold_model.predict_proba(X_val_fold)[:, 1]
                elif name == "spread" and SpreadRegressor is not None:
                    m_tr_fold = train_margins[split.train_indices]
                    fold_model = SpreadRegressor(sigma=pipeline.config.spread_sigma_init)
                    fold_model.train(X_tr_fold, m_tr_fold, feature_names=feature_names, num_rounds=200, sample_weight=w_tr_fold)
                    fold_preds = fold_model.predict_probability(X_val_fold)
                else:
                    continue
                oof_preds[name][split.val_indices] = fold_preds
                oof_counts[split.val_indices] += 1

        oof_mask = oof_counts > 0
        # Require >=100 OOF samples for stacking (was 20).  With 4 base
        # models the enriched meta-feature matrix has ~13 dimensions;
        # 100 samples gives a ~8:1 sample-to-feature ratio and avoids
        # overfitting the meta-learner on noisy OOF predictions.
        if np.sum(oof_mask) >= 100:
            base_meta_X = np.column_stack([oof_preds[name][oof_mask] for name, _, _ in trained_models])
            meta_y = train_y[oof_mask]
            meta_X = pipeline._build_enriched_meta(base_meta_X)

            meta_learner = LogisticRegression(
                C=1.0, penalty="l2", max_iter=2000,
                random_state=pipeline.config.random_seed,
            )
            meta_learner.fit(meta_X, meta_y)
            meta_learner_type = "logistic"

            # FIX-STACKING-LEAKAGE: Validate stacking via held-out OOF,
            # NOT by predicting on the training data (which was in-sample).
            # Use 3-fold CV on the meta features to get honest predictions.
            n_meta = len(meta_y)
            meta_oof_preds = np.full(n_meta, 0.5)
            _meta_fold_size = n_meta // 3
            for _mf in range(3):
                _mf_start = _mf * _meta_fold_size
                _mf_end = (_mf + 1) * _meta_fold_size if _mf < 2 else n_meta
                _mf_val = list(range(_mf_start, _mf_end))
                _mf_tr = [j for j in range(n_meta) if j not in _mf_val]
                if len(_mf_tr) < 20:
                    continue
                _mf_model = LogisticRegression(
                    C=1.0, penalty="l2", max_iter=2000,
                    random_state=pipeline.config.random_seed,
                )
                _mf_model.fit(meta_X[_mf_tr], meta_y[_mf_tr])
                meta_oof_preds[_mf_val] = _mf_model.predict_proba(
                    meta_X[_mf_val]
                )[:, 1]
            stacking_brier = float(np.mean((meta_oof_preds - meta_y) ** 2))
            n_models = base_meta_X.shape[1]
            ew_preds = np.mean(base_meta_X, axis=1)  # equal-weight baseline
            ew_brier = float(np.mean((ew_preds - meta_y) ** 2))

            if stacking_brier < ew_brier:
                pipeline.baseline_model.stacking_meta = meta_learner
                pipeline.baseline_model.stacking_meta_type = meta_learner_type
                pipeline.baseline_model.stacking_models = [(name, model) for name, model, _ in trained_models]

                stacking_stats = {
                    "enabled": True,
                    "meta_learner": meta_learner_type,
                    "base_models": [name for name, _, _ in trained_models],
                    "stacking_brier": round(stacking_brier, 5),
                    "equal_weight_brier": round(ew_brier, 5),
                }
                baseline_name = "stacking_ensemble"
            else:
                logger.info(
                    "Stacking meta-learner (Brier=%.5f) does not improve over "
                    "equal-weight baseline (Brier=%.5f). Falling back.",
                    stacking_brier, ew_brier,
                )
                stacking_stats = {
                    "enabled": False,
                    "reason": "no_improvement_over_equal_weight",
                    "stacking_brier": round(stacking_brier, 5),
                    "equal_weight_brier": round(ew_brier, 5),
                }
                baseline_name = pipeline._select_best_single_model(trained_models, eval_y)
        else:
            stacking_stats = {"enabled": False, "reason": "insufficient_oof_samples"}
            baseline_name = pipeline._select_best_single_model(trained_models, eval_y)

    elif len(trained_models) >= 2:
        # --- Fixed-weight average (default/fallback path) ---
        # Both production and experimental modes start with fallback weights.
        # In production mode, these may be overridden by LOYO-optimized
        # weights later (see FIX-CV-ENSEMBLE block below).
        if _production_mode:
            # Production: fallback weights from PRODUCTION_BASELINE spec.
            # These are literature-based priors, NOT fitted values.
            # The LOYO optimizer (below) may replace them with CV-derived
            # weights if it finds improvement over this fallback.
            from ...production_baseline import PRODUCTION_BASELINE
            _prod_weights = dict(PRODUCTION_BASELINE.fallback_weights)
            # Map production names to trained_models names
            _PROD_NAME_MAP = {"spread": "spread", "logistic": "logit", "lgb": "lgb", "xgb": "xgb"}
            model_names_present = [name for name, _, _ in trained_models]
            active_weights = {}
            for prod_name, weight in _prod_weights.items():
                internal_name = _PROD_NAME_MAP.get(prod_name, prod_name)
                if internal_name in model_names_present:
                    active_weights[internal_name] = weight
            # Normalize to sum to 1.0
            w_sum = sum(active_weights.values())
            if w_sum > 0:
                active_weights = {n: w / w_sum for n, w in active_weights.items()}
            else:
                # Fallback: equal weight if no sanctioned models present
                active_weights = {n: 1.0 / len(model_names_present) for n in model_names_present}
        else:
            # Experimental mode: legacy weights
            _FIXED_WEIGHTS = {
                "spread": 0.55,  # Primary: margin prediction via logistic CDF
                "lgb": 0.15,     # Secondary: LightGBM classifier
                "xgb": 0.15,     # Secondary: XGBoost classifier
                "logit": 0.15,   # Complementary: Logistic regression
            }
            model_names_present = [name for name, _, _ in trained_models]
            active_weights = {n: _FIXED_WEIGHTS.get(n, 0.25) for n in model_names_present}
            w_sum = sum(active_weights.values())
            active_weights = {n: w / w_sum for n, w in active_weights.items()}

        pipeline.baseline_model.fixed_weight_models = [(name, model) for name, model, _ in trained_models]
        pipeline.baseline_model.fixed_weights = active_weights

        stacking_stats = {
            "enabled": False,
            "method": "fixed_weight_average",
            "weights": {n: round(w, 3) for n, w in active_weights.items()},
        }
        baseline_name = "fixed_weight_ensemble"

    elif trained_models:
        baseline_name = pipeline._select_best_single_model(trained_models, eval_y)
    else:
        baseline_name = "none"

    pipeline.tuning_result = tuning_stats if tuning_stats else None

    # OOS-FIX: Eval set is now used ONLY for confidence estimation
    # (diagnostic reporting), NOT for model selection.  With fixed-weight
    # ensemble, no decisions depend on eval set performance.
    brier = 0.25  # uninformative default
    eval_roc_auc = None
    brier_ci = None
    if valid_samples > 0:
        y_pred = pipeline.baseline_model.predict_proba_batch(eval_X)
        brier = float(np.mean((y_pred - eval_y) ** 2))
        # Conservative confidence: discount by sqrt(n) uncertainty
        # Don't trust small eval sets to tightly estimate model quality
        confidence_discount = min(1.0, math.sqrt(valid_samples / 200.0))
        raw_confidence = float(np.clip(1.0 - brier, 0.05, 0.95))
        pipeline.model_confidence["baseline"] = 0.5 + (raw_confidence - 0.5) * confidence_discount

        if len(np.unique(eval_y)) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                eval_roc_auc = float(roc_auc_score(eval_y, y_pred))
            except Exception as _auc_exc:
                logger.debug("ROC-AUC computation failed: %s", _auc_exc)

        if valid_samples >= 20:
            _rng = np.random.default_rng(pipeline.config.random_seed)
            _n_boot = min(2000, max(500, valid_samples * 5))
            _boot_briers = np.empty(_n_boot)
            for _b in range(_n_boot):
                _idx = _rng.choice(valid_samples, size=valid_samples, replace=True)
                _boot_briers[_b] = float(np.mean((y_pred[_idx] - eval_y[_idx]) ** 2))
            brier_ci = (
                round(float(np.percentile(_boot_briers, 2.5)), 5),
                round(float(np.percentile(_boot_briers, 97.5)), 5),
            )

    # FIX-CV-ENSEMBLE: Cross-validated ensemble weight optimization.
    # Uses LOYO (leave-one-year-out) predictions to find weights that
    # generalize across tournament years, avoiding double-dipping.
    # Each year's weights are evaluated on data never used for training.
    #
    # Design: freeze the procedure, not the parameter values.
    # The optimizer runs with simplex bounds from PRODUCTION_BASELINE to
    # prevent overfitting (7 folds × ~63 games = ~440 OOS samples for
    # ~3 free parameters).  If LOYO improves over fallback, the derived
    # weights replace the fallback.  Otherwise, fallback weights stand.
    ensemble_weight_stats = {}

    # FIX-STACKING-LEAKAGE: BMA weights are now derived from LOYO OOS
    # predictions instead of the eval set.  This eliminates the critical
    # contamination pathway where weights optimized on eval_y biased all
    # downstream metrics.
    #
    # Protocol v2, Section 3.2: BMA is the ONLY supported ensemble strategy.
    _bma_cfg = getattr(pipeline, "config", None)
    _bma_flag = getattr(_bma_cfg, "bma_enabled", True)
    _use_bma = BMA_AVAILABLE and _bma_flag and len(trained_models) >= 2

    if _use_bma and len(trained_models) >= 2:
        # Derive BMA weights from LOYO OOS predictions (not eval set)
        loyo_bma_stats = _fit_bma_on_loyo(
            pipeline, trained_models, _loyo_raw_feature_dim, feature_names,
        )
        if loyo_bma_stats and loyo_bma_stats.get("optimized_weights"):
            pipeline.baseline_model.fixed_weights = loyo_bma_stats["optimized_weights"]
            ensemble_weight_stats = loyo_bma_stats
            logger.info(
                "FIX-STACKING-LEAKAGE: BMA weights from LOYO OOS applied: %s",
                loyo_bma_stats["optimized_weights"],
            )
        else:
            logger.info(
                "LOYO BMA unavailable (insufficient data). "
                "Keeping fixed baseline weights."
            )
            ensemble_weight_stats = {
                "method": "fixed_fallback",
                "weight_source": "fixed",
                "reason": "loyo_bma_insufficient_data",
            }

    # FIX-STACKING-LEAKAGE guard: ensure weights were NOT derived from eval set
    assert ensemble_weight_stats.get("weight_source") != "eval_set", (
        "BMA weights must not be derived from eval set (data leakage). "
        "Use _fit_bma_on_loyo() instead."
    )

    # Propagate tournament sigma calibrator to _TrainedBaselineModel so that
    # SpreadRegressor uses tournament-calibrated sigma at inference time.
    if hasattr(pipeline, '_tournament_sigma_calibrator') and pipeline._tournament_sigma_calibrator is not None:
        pipeline.baseline_model.tournament_sigma_calibrator = pipeline._tournament_sigma_calibrator
        logger.info(
            "Propagated TournamentSigmaCalibrator to baseline_model "
            "(global_sigma=%.2f)",
            pipeline._tournament_sigma_calibrator.global_tournament_sigma,
        )

    # ====================================================================
    # P0: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION — validates that the trained
    # model generalizes across different tournament years' "chaos" patterns.
    # Uses multi-year historical data (2015-2025) to run LOYO CV and report
    # per-year Brier scores.  This does NOT retrain the primary model — it
    # is a validation diagnostic only.
    # ====================================================================
    loyo_stats = {}
    _loyo_games_dir = getattr(pipeline, "_runtime_state", {}).get(
        "multi_year_games_dir", pipeline.config.multi_year_games_dir
    )
    if (
        pipeline.config.enable_loyo_cv
        and _loyo_games_dir
        and LeaveOneYearOutCV is not None
    ):
        # Use pre-pruning feature dimension and names since LOYO loads
        # raw historical data at the original matchup dimension, then
        # applies _pre_fs_keep_mask post-hoc for zero-variance pruning.
        _loyo_names = feature_names_full if feature_names_full is not None else feature_names
        loyo_stats = pipeline._run_loyo_validation(
            feature_dim=_loyo_raw_feature_dim,
            feature_names=_loyo_names,
        )

    # ====================================================================
    # MODEL COMPLEXITY AUDIT — verify effective parameter count stays
    # below 10% of training sample size.  This is the production-side
    # enforcement of the complexity guard defined in rdof_audit.py.
    # The audit runs at training time rather than only during offline
    # RDoF audits, ensuring every pipeline run catches violations.
    # ====================================================================
    complexity_stats = {}
    try:
        from src.ml.evaluation.rdof_audit import estimate_model_complexity
        complexity_audit = estimate_model_complexity(
            config=pipeline.config,
            n_training_samples=int(train_samples),
            n_features=int(train_X.shape[1]),
            gnn_embedding_dim=0,  # A1: GNN embeddings removed from ensemble
            transformer_embedding_dim=0,  # A1: Transformer removed from ensemble
        )
        complexity_stats = complexity_audit.to_dict()
        if not complexity_audit.passed:
            logger.warning(
                "COMPLEXITY GUARD: %d effective params / %d training samples "
                "= %.1f%% (target < %.0f%%). %s",
                complexity_audit.total_effective_params,
                train_samples,
                complexity_audit.actual_ratio * 100,
                complexity_audit.target_ratio * 100,
                "; ".join(complexity_audit.warnings),
            )
        else:
            logger.info(
                "Complexity guard PASSED: %d effective params / %d samples "
                "= %.1f%% (target < %.0f%%).",
                complexity_audit.total_effective_params,
                train_samples,
                complexity_audit.actual_ratio * 100,
                complexity_audit.target_ratio * 100,
            )
    except Exception as e:
        logger.debug("Model complexity audit skipped: %s", e)

    # ====================================================================
    # SAMPLE SIZE vs PARAMETER RATIO — explicit logging of the
    # fundamental sample-size concern (Concern A).  Reports the ratio
    # of active features + hyperparameters to training samples.
    # A healthy ratio should be well below 0.10.
    # ====================================================================
    n_active_features = int(train_X.shape[1])
    n_tier3_constants = 8  # From CONSTANT_REGISTRY Tier 3 count
    effective_dof = n_active_features + n_tier3_constants
    dof_ratio = effective_dof / max(train_samples, 1)
    logger.info(
        "Sample size audit: %d active features + %d Tier-3 constants "
        "= %d effective DoF / %d training samples = %.3f "
        "(target < 0.10).",
        n_active_features, n_tier3_constants, effective_dof,
        train_samples, dof_ratio,
    )
    if dof_ratio > 0.10:
        logger.warning(
            "SAMPLE SIZE WARNING: effective DoF / training samples = %.3f "
            "> 0.10. Consider enabling multi-year training pool or "
            "reducing features.",
            dof_ratio,
        )

    result = {
        "model": baseline_name,
        "unique_games": int(n_unique_games),
        "samples": int(n),
        "train_samples": int(train_samples),
        "validation_samples": int(valid_samples),
        "features": int(train_X.shape[1]),
        "brier": brier,
    }
    if eval_roc_auc is not None:
        result["roc_auc"] = round(eval_roc_auc, 4)
    if brier_ci is not None:
        result["brier_95ci"] = list(brier_ci)
    if tuning_stats:
        result["hyperparameter_tuning"] = tuning_stats
    if fs_stats:
        result["feature_selection"] = fs_stats
    if stacking_stats:
        result["stacking"] = stacking_stats
    if ensemble_weight_stats:
        result["ensemble_weight_optimization"] = ensemble_weight_stats
    if loyo_stats:
        result["loyo_cv"] = loyo_stats
    if dist_shift_stats:
        result["distribution_shift"] = dist_shift_stats
    if historical_training_stats:
        result["multi_year_training"] = historical_training_stats
    if complexity_stats:
        result["model_complexity_audit"] = complexity_stats
    result["sample_size_audit"] = {
        "active_features": n_active_features,
        "tier3_constants": n_tier3_constants,
        "effective_dof": effective_dof,
        "train_samples": int(train_samples),
        "dof_ratio": round(dof_ratio, 4),
        "passed": dof_ratio <= 0.10,
    }
    return result




def _optimize_ensemble_weights_loyo(
    pipeline,
    trained_models: list,
    feature_dim: int,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """FIX-STACKING-LEAKAGE: Nested LOYO ensemble weight optimization.

    Uses proper nested cross-validation to eliminate the self-evaluation
    contamination that existed in the previous pooled approach.

    For each outer fold (held-out year Y):
      1. Train all model types on years != Y → get OOS predictions for Y
      2. From years != Y, run inner LOYO to derive weights
      3. Apply inner-derived weights to year Y predictions
      4. Record year Y's Brier (honest — weights never saw year Y)

    The reported Brier is the mean across outer folds, guaranteed
    uncontaminated because each fold's weights are derived from data
    that excludes that fold.

    Returns:
        Dict with nested LOYO results, or {} if insufficient data.
    """
    import os
    import logging as _logging
    logger = _logging.getLogger(__name__)

    from ....ml.ensemble.stacking_weights import (
        StackingWeightOptimizer as _SWO,
    )

    games_dir = getattr(pipeline, "_runtime_state", {}).get(
        "multi_year_games_dir", pipeline.config.multi_year_games_dir
    )
    if not games_dir or not os.path.isdir(games_dir):
        return {}

    years = pipeline.config.loyo_years or [
        y for y in range(2015, pipeline.config.year) if y != 2020
    ]
    years = pipeline._filter_years(years, include_holdout=False)
    year_split_policy = getattr(pipeline, "_year_split_policy", None)
    if year_split_policy is not None:
        year_split_policy.assert_dev_artifact_years(
            list(years),
            context="LOYO ensemble weight optimization",
        )
    if len(years) < 3:
        return {}

    # Step 1: Load all years' data
    all_X: Dict[int, np.ndarray] = {}
    all_y: Dict[int, np.ndarray] = {}
    all_margins: Dict[int, np.ndarray] = {}

    for yr in years:
        gp = os.path.join(games_dir, f"historical_games_{yr}.json")
        mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
        if not os.path.isfile(gp) or not os.path.isfile(mp):
            continue
        try:
            yr_X, yr_y, yr_margins, _, _ = pipeline._load_year_samples_incremental(
                gp, mp, feature_dim, yr
            )
            if len(yr_y) >= 20:
                all_X[yr] = yr_X
                all_y[yr] = yr_y
                all_margins[yr] = yr_margins
        except Exception:
            continue

    valid_years = sorted(all_X.keys())
    if len(valid_years) < 3:
        return {}

    # Step 2: Generate per-year OOS predictions via LOYO
    model_names = [name for name, _, _ in trained_models]
    preds_by_year: Dict[int, Dict[str, np.ndarray]] = {}

    for hold_yr in valid_years:
        train_X_parts = [all_X[yr] for yr in valid_years if yr != hold_yr]
        train_y_parts = [all_y[yr] for yr in valid_years if yr != hold_yr]
        train_margin_parts = [all_margins[yr] for yr in valid_years if yr != hold_yr]
        if not train_X_parts:
            continue

        X_train = np.concatenate(train_X_parts, axis=0)
        y_train = np.concatenate(train_y_parts)
        margins_train = np.concatenate(train_margin_parts)
        X_val = all_X[hold_yr]
        y_val = all_y[hold_yr]

        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        X_train = _scaler.fit_transform(X_train)
        X_val = _scaler.transform(X_val)

        fold_preds: Dict[str, np.ndarray] = {}
        for name, _, _ in trained_models:
            try:
                if name == "lgb" and LIGHTGBM_AVAILABLE:
                    m = LightGBMRanker()
                    m.train(X_train, y_train, num_rounds=200)
                    fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                elif name == "xgb" and XGBOOST_AVAILABLE:
                    m = XGBoostRanker()
                    m.train(X_train, y_train, num_rounds=200)
                    fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                elif name == "logit" and SKLEARN_AVAILABLE:
                    m = LogisticRegression(
                        C=1.0, max_iter=2000,
                        random_state=pipeline.config.random_seed,
                    )
                    m.fit(X_train, y_train)
                    fold_preds[name] = np.clip(
                        m.predict_proba(X_val)[:, 1], 0.01, 0.99
                    )
                elif name == "spread" and SPREAD_MODEL_AVAILABLE:
                    m = SpreadRegressor(
                        sigma=pipeline.config.spread_sigma_init,
                    )
                    m.train(X_train, margins_train, num_rounds=200)
                    fold_preds[name] = np.clip(
                        m.predict_probability(X_val), 0.01, 0.99
                    )
                else:
                    fold_preds[name] = np.full(len(y_val), 0.5)
            except Exception:
                fold_preds[name] = np.full(len(y_val), 0.5)

        preds_by_year[hold_yr] = fold_preds

    if len(preds_by_year) < 3:
        return {}

    # Step 3: FIX-STACKING-LEAKAGE — Nested weight optimization.
    # Use StackingWeightOptimizer.fit_nested_loyo() which derives
    # weights per outer fold from inner folds only.
    optimizer = _SWO(regularization=0.1, random_seed=pipeline.config.random_seed)

    predictions_by_year = {}
    outcomes_by_year = {}
    for yr in preds_by_year:
        predictions_by_year[yr] = {
            name: preds_by_year[yr].get(name, np.full(len(all_y[yr]), 0.5))
            for name in model_names
        }
        outcomes_by_year[yr] = all_y[yr]

    nested_result = optimizer.fit_nested_loyo(
        predictions_by_year, outcomes_by_year,
    )

    logger.info(
        "FIX-STACKING-LEAKAGE: Nested LOYO weight optimization on %d folds "
        "(%d total samples). Honest mean Brier=%.5f (std=%.5f). "
        "Production weights: %s",
        nested_result.n_folds, nested_result.n_total_samples,
        nested_result.mean_brier, nested_result.std_brier,
        {n: round(w, 3) for n, w in nested_result.production_weights.items()},
    )

    return {
        "method": "nested_loyo",
        "weight_source": "nested_loyo",
        "years_used": list(preds_by_year.keys()),
        "n_folds": nested_result.n_folds,
        "oos_samples": nested_result.n_total_samples,
        "honest_mean_brier": round(nested_result.mean_brier, 5),
        "honest_std_brier": round(nested_result.std_brier, 5),
        "production_weights": {
            n: round(w, 3) for n, w in nested_result.production_weights.items()
        },
        "per_fold_brier": {
            str(yr): round(b, 5) for yr, b in nested_result.per_fold_brier.items()
        },
        "per_fold_weights": {
            str(yr): {n: round(w, 3) for n, w in ws.items()}
            for yr, ws in nested_result.per_fold_weights.items()
        },
    }




def _optimize_ensemble_weights_on_validation(
    pipeline,
    eval_X: np.ndarray,
    eval_y: np.ndarray,
    game_flows: Dict[str, List[GameFlow]],
) -> Dict:
    """
    Optimize CFA ensemble weights on held-out validation data only.

    Uses slice 1 of the 3-way validation split (Issue 5).  Slice 0 is
    used for embedding projections; slice 2 for calibration.
    FIX #5: Snapshots pre-optimization CFA weights before applying new ones,
    so that calibration can generate predictions with un-optimized weights.
    """
    # Snapshot current ensemble weights BEFORE optimization (Fix #5)
    pipeline._pre_optimization_cfa_weights = dict(pipeline.ensemble_base_weights)

    model_preds: Dict[str, List[float]] = {"baseline": [], "gnn": [], "transformer": []}
    outcomes: List[int] = []

    # Issue 5: Use slice 1 of the 3-way validation split.
    val_games = pipeline._get_validation_era_games_slice(game_flows, slice_index=1, n_slices=3)

    for g in val_games:
        outcome = pipeline._game_outcome(g)
        if outcome is None:
            continue
        outcomes.append(outcome)

        matchup = pipeline.feature_engineer.create_matchup_features(g.team1_id, g.team2_id, proprietary_engine=pipeline.proprietary_engine)
        feat_vec = matchup.to_vector()
        if pipeline.feature_selector is not None and pipeline.feature_selector.is_fitted:
            feat_vec = pipeline.feature_selector.transform(feat_vec.reshape(1, -1))[0]
        model_preds["baseline"].append(pipeline.baseline_model.predict_proba(feat_vec))
        model_preds["gnn"].append(
            pipeline._embedding_probability(pipeline.gnn_embeddings.get(g.team1_id), pipeline.gnn_embeddings.get(g.team2_id), model_type="gnn")
        )
        model_preds["transformer"].append(
            pipeline._embedding_probability(pipeline.transformer_embeddings.get(g.team1_id), pipeline.transformer_embeddings.get(g.team2_id), model_type="transformer")
        )

    if len(outcomes) < 10:
        return {}

    optimizer = EnsembleWeightOptimizer(step=0.05, min_weight=0.05, n_bootstrap=200, random_seed=pipeline.config.random_seed)
    pred_arrays = {name: np.array(preds) for name, preds in model_preds.items()}
    best_weights, best_brier = optimizer.optimize(
        pred_arrays,
        np.array(outcomes),
        min_samples=pipeline.config.min_ensemble_samples,
        regularization_lambda=pipeline.config.ensemble_weight_regularization,
    )

    # Apply optimized weights to ensemble
    pipeline.ensemble_base_weights = best_weights

    return {
        "optimized_weights": {k: round(v, 3) for k, v in best_weights.items()},
        "optimized_brier": round(best_brier, 5),
        "validation_samples": len(outcomes),
    }


def _fit_bma_on_loyo(
    pipeline,
    trained_models: list,
    feature_dim: int,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """FIX-STACKING-LEAKAGE: Fit BMA weights on LOYO OOS predictions.

    Replaces the prior approach of fitting BMA on the eval set, which
    contaminated any downstream metric computed using those weights.

    For each LOYO fold (held-out year Y):
      1. Train all base models on years != Y
      2. Generate OOS predictions for year Y
    Pool all OOS predictions and fit BMA on them.

    The resulting weights are derived from data that is out-of-sample
    for each individual year, making them suitable for production use.

    Returns:
        Dict with BMA weights and diagnostics, or {} if insufficient data.
    """
    import os

    games_dir = getattr(pipeline, "_runtime_state", {}).get(
        "multi_year_games_dir", pipeline.config.multi_year_games_dir
    )
    if not games_dir or not os.path.isdir(games_dir):
        return {}

    years = pipeline.config.loyo_years or [
        y for y in range(2015, pipeline.config.year) if y != 2020
    ]
    years = pipeline._filter_years(years, include_holdout=False)
    year_split_policy = getattr(pipeline, "_year_split_policy", None)
    if year_split_policy is not None:
        year_split_policy.assert_dev_artifact_years(
            list(years),
            context="LOYO BMA weight derivation",
        )
    if len(years) < 3:
        return {}

    # Step 1: Load all years' data
    all_X: Dict[int, np.ndarray] = {}
    all_y: Dict[int, np.ndarray] = {}
    all_margins: Dict[int, np.ndarray] = {}

    for yr in years:
        gp = os.path.join(games_dir, f"historical_games_{yr}.json")
        mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
        if not os.path.isfile(gp) or not os.path.isfile(mp):
            continue
        try:
            yr_X, yr_y, yr_margins, _, _ = pipeline._load_year_samples_incremental(
                gp, mp, feature_dim, yr
            )
            if len(yr_y) >= 20:
                all_X[yr] = yr_X
                all_y[yr] = yr_y
                all_margins[yr] = yr_margins
        except Exception:
            continue

    valid_years = sorted(all_X.keys())
    if len(valid_years) < 3:
        return {}

    # Step 2: LOYO — for each held-out year, train all model types
    model_names = [name for name, _, _ in trained_models]
    all_oos_preds: Dict[str, list] = {name: [] for name in model_names}
    all_oos_labels: list = []

    for hold_yr in valid_years:
        train_X_parts = [all_X[yr] for yr in valid_years if yr != hold_yr]
        train_y_parts = [all_y[yr] for yr in valid_years if yr != hold_yr]
        train_margin_parts = [all_margins[yr] for yr in valid_years if yr != hold_yr]
        if not train_X_parts:
            continue

        X_train = np.concatenate(train_X_parts, axis=0)
        y_train = np.concatenate(train_y_parts)
        margins_train = np.concatenate(train_margin_parts)
        X_val = all_X[hold_yr]
        y_val = all_y[hold_yr]

        # FIX-LEAKAGE-ENSEMBLE-WEIGHTS: fresh scaler per fold
        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        X_train = _scaler.fit_transform(X_train)
        X_val = _scaler.transform(X_val)

        fold_preds: Dict[str, np.ndarray] = {}
        for name, _, _ in trained_models:
            try:
                if name == "lgb" and LIGHTGBM_AVAILABLE:
                    m = LightGBMRanker()
                    m.train(X_train, y_train, num_rounds=200)
                    fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                elif name == "xgb" and XGBOOST_AVAILABLE:
                    m = XGBoostRanker()
                    m.train(X_train, y_train, num_rounds=200)
                    fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                elif name == "logit" and SKLEARN_AVAILABLE:
                    m = LogisticRegression(
                        C=1.0, max_iter=2000,
                        random_state=pipeline.config.random_seed,
                    )
                    m.fit(X_train, y_train)
                    fold_preds[name] = np.clip(
                        m.predict_proba(X_val)[:, 1], 0.01, 0.99
                    )
                elif name == "spread" and SPREAD_MODEL_AVAILABLE:
                    m = SpreadRegressor(
                        sigma=pipeline.config.spread_sigma_init,
                    )
                    m.train(X_train, margins_train, num_rounds=200)
                    fold_preds[name] = np.clip(
                        m.predict_probability(X_val), 0.01, 0.99
                    )
                else:
                    fold_preds[name] = np.full(len(y_val), 0.5)
            except Exception:
                fold_preds[name] = np.full(len(y_val), 0.5)

        for name in model_names:
            all_oos_preds[name].extend(
                fold_preds.get(name, np.full(len(y_val), 0.5)).tolist()
            )
        all_oos_labels.extend(y_val.tolist())

    if len(all_oos_labels) < 50:
        return {}

    # Step 3: Fit BMA on pooled OOS predictions
    oos_y = np.array(all_oos_labels)
    bma_preds = {}
    for name in model_names:
        arr = np.array(all_oos_preds[name])
        bma_preds[name] = np.clip(arr, 1e-7, 1 - 1e-7)

    if not BMA_AVAILABLE or len(bma_preds) < 2:
        return {}

    bma = BayesianModelAveraging()
    bma_result = bma.fit(bma_preds, oos_y)

    if not bma_result.weights:
        return {}

    logger.info(
        "FIX-STACKING-LEAKAGE: BMA weights derived from LOYO OOS "
        "(%d samples, %d years). Weights: %s",
        len(oos_y), len(valid_years),
        {k: round(v, 4) for k, v in bma_result.weights.items()},
    )

    return {
        "method": "bayesian_model_averaging",
        "weight_source": "loyo_oos",
        "optimized_weights": {
            k: round(v, 4) for k, v in bma_result.weights.items()
        },
        "effective_model_count": round(bma_result.effective_model_count, 2),
        "converged": bma_result.converged,
        "n_iterations": bma_result.n_iterations,
        "oos_samples": len(oos_y),
        "years_used": valid_years,
    }

