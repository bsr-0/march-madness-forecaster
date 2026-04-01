"""Baseline model training — loyo module."""


import logging
import os
import re
from datetime import date, timedelta

from ....data.features.feature_selection import FeatureSelector
from ....ml.ensemble.cfa import (
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
    LightGBMRanker,
    ModelPrediction,
    XGBoostRanker,
)
from ...config import (
    DATA_QUALITY_ERA_WEIGHTS,
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    SOTAPipelineConfig,
    TOURNAMENT_START_DATES,
    compute_year_data_quality,
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


def _run_loyo_validation(
    pipeline,
    feature_dim: int,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run Leave-One-Year-Out CV on multi-year historical data.

    Loads historical game results and team metrics for each year,
    constructs simplified differential feature vectors, and evaluates
    model generalization across different tournament years.

    This is a VALIDATION diagnostic — it does not modify the primary
    trained model. It answers: "Would our modeling approach have
    generalised to past tournaments?"

    Returns:
        Dict with per-year Brier scores, mean Brier, and sample counts.
    """
    import json
    import logging
    import os

    logger = logging.getLogger(__name__)

    try:
        from ....ml.evaluation.loyo_protocol import LOYOValidator
    except Exception as e:
        logger.warning("LOYO validator unavailable: %s", e)
        return {"enabled": False, "reason": f"loyo_validator_unavailable: {e}"}

    try:
        from ..pit_validation import PITValidator, SELECTION_SUNDAY_DATES
    except Exception as e:
        logger.warning("PIT validator unavailable in LOYO path: %s", e)
        PITValidator = None
        SELECTION_SUNDAY_DATES = {}

    def _parse_game_date(value) -> Optional[date]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        if "T" in raw:
            raw = raw.split("T", 1)[0]
        try:
            return date.fromisoformat(raw)
        except ValueError:
            m = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", raw)
            if not m:
                return None
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None

    def _latest_regular_season_game_date(games_path: str, year: int) -> Optional[date]:
        try:
            with open(games_path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.debug("LOYO PIT metadata: failed reading %s: %s", games_path, exc)
            return None

        if isinstance(payload, dict):
            team_games = payload.get("team_games", [])
        elif isinstance(payload, list):
            team_games = payload
        else:
            team_games = []

        tourney_start = TOURNAMENT_START_DATES.get(year)
        latest = None
        for game in team_games:
            if not isinstance(game, dict):
                continue
            gd = _parse_game_date(game.get("game_date") or game.get("date"))
            if gd is None:
                continue
            if tourney_start is not None and gd >= tourney_start:
                continue
            if latest is None or gd > latest:
                latest = gd
        return latest

    games_dir = getattr(pipeline, "_runtime_state", {}).get(
        "multi_year_games_dir", pipeline.config.multi_year_games_dir
    )
    if not games_dir or not os.path.isdir(games_dir):
        return {"enabled": False, "reason": f"directory_not_found: {games_dir}"}

    years = pipeline.config.loyo_years or [y for y in range(2015, 2026) if y != 2020]
    years = pipeline._filter_years(years, include_holdout=True)
    if not years:
        return {"enabled": False, "reason": "no_dev_years"}

    # ----------------------------------------------------------
    # Step 1: Load multi-year samples (year-keyed for LOYOValidator)
    # ----------------------------------------------------------
    data_by_year: Dict[int, Dict[str, np.ndarray]] = {}
    latest_game_date_by_year: Dict[int, Optional[date]] = {}

    # Training-time feature names must align with X column count.
    if feature_names is not None and len(feature_names) == feature_dim:
        model_feature_names = list(feature_names)
    else:
        model_feature_names = [f"f_{i}" for i in range(feature_dim)]
        if feature_names is not None and len(feature_names) != feature_dim:
            logger.warning(
                "LOYO: feature name mismatch (%d names vs %d columns). "
                "Using generic names for CV.",
                len(feature_names), feature_dim,
            )

    for year in years:
        games_path = os.path.join(games_dir, f"historical_games_{year}.json")
        metrics_path = os.path.join(games_dir, f"team_metrics_{year}.json")

        if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
            logger.info("LOYO: skipping year %d (missing data files)", year)
            continue

        year_X, year_y, _year_margins, _, _yr_rw = pipeline._load_year_samples_incremental(
            games_path, metrics_path, feature_dim, year
        )
        if len(year_y) < 10:
            logger.info("LOYO: skipping year %d (only %d samples)", year, len(year_y))
            continue

        # FIX-DQ: Apply the same early zero-variance pruning used in
        # the primary training path so LOYO folds operate on the same
        # reduced feature space.
        _loyo_keep_mask = getattr(pipeline, "_pre_fs_keep_mask", None)
        _year_feature_names = model_feature_names
        if _loyo_keep_mask is not None and year_X.shape[1] == len(_loyo_keep_mask):
            year_X = year_X[:, _loyo_keep_mask]
            # Slice feature names to match pruned dimension
            if len(model_feature_names) == len(_loyo_keep_mask):
                _year_feature_names = [
                    model_feature_names[i]
                    for i in range(len(_loyo_keep_mask))
                    if _loyo_keep_mask[i]
                ]

        data_by_year[year] = {
            "X": year_X,
            "y": year_y,
            "margins": _year_margins,
            "feature_names": _year_feature_names,
            "sample_weights": np.ones(len(year_y), dtype=np.float64),
        }
        latest_game_date_by_year[year] = _latest_regular_season_game_date(games_path, year)

    if not data_by_year:
        return {"enabled": False, "reason": "no_valid_year_data"}

    # ----------------------------------------------------------
    # Step 1.5: PIT validation (Protocol v2 Section 2)
    # ----------------------------------------------------------
    pit_summary = {
        "enabled": False,
        "folds_checked": 0,
        "folds_passed": 0,
        "violations": 0,
        "warnings": 0,
    }
    pit_enforcement = getattr(pipeline.config, "pit_enforcement", True)
    if PITValidator is not None and pit_enforcement:
        try:
            from src.data.features.feature_engineering import TeamFeatures

            pit_validator = PITValidator()
            team_feature_names = TeamFeatures.get_feature_names(include_embeddings=False)

            feature_metadata_by_year = {}
            for year in sorted(data_by_year):
                selection_sunday = SELECTION_SUNDAY_DATES.get(year)
                latest_game = latest_game_date_by_year.get(year)
                if selection_sunday is None:
                    continue
                if latest_game is None:
                    latest_game = selection_sunday - timedelta(days=1)
                elif latest_game > selection_sunday:
                    # Protocol v2 PIT rule: Tier 2 features must be frozen by
                    # Selection Sunday. If source files contain later games
                    # (e.g., play-in/postseason spillover), cap metadata at the
                    # protocol boundary and surface a warning.
                    logger.warning(
                        "LOYO PIT metadata: year %d latest regular-season date %s "
                        "is after Selection Sunday %s; capping at Selection Sunday.",
                        year, latest_game, selection_sunday,
                    )
                    latest_game = selection_sunday

                year_meta = {}
                for fname in pit_validator.get_tier2_features():
                    year_meta[fname] = {"latest_game_date": latest_game.isoformat()}
                for fname in pit_validator.get_tier3_features():
                    year_meta[fname] = {"snapshot_date": selection_sunday.isoformat()}
                feature_metadata_by_year[year] = year_meta

            pit_results = pit_validator.validate_loyo_folds(
                years=sorted(data_by_year.keys()),
                feature_names=team_feature_names,
                feature_metadata_by_year=feature_metadata_by_year,
                strict=True,
            )

            n_folds = len(pit_results)
            n_passed = sum(1 for r in pit_results if r.passed)
            n_violations = sum(len(r.violations) for r in pit_results)
            n_warnings = sum(len(r.warnings) for r in pit_results)
            pit_summary = {
                "enabled": True,
                "folds_checked": n_folds,
                "folds_passed": n_passed,
                "violations": n_violations,
                "warnings": n_warnings,
            }
            logger.info(
                "LOYO PIT summary: %d/%d folds passed, %d violations, %d warnings",
                n_passed, n_folds, n_violations, n_warnings,
            )
        except Exception as pit_exc:
            logger.error("LOYO PIT validation failed: %s", pit_exc)
            raise

    # FIX-LEAKAGE-LOYO: Do NOT apply the primary model's feature
    # selector or scaler here.  Both were fitted on training data that
    # includes the held-out year, so reusing them leaks information
    # (the feature selector's importance scores encode test-year labels;
    # the scaler's mean/std include test-year feature distributions).
    # Instead, each LOYO fold re-fits its own scaler and feature
    # selector below in train_fn, faithfully mirroring the production
    # pipeline's preprocessing stack.

    # ----------------------------------------------------------
    # Step 2: Run LOYOValidator (metrics: Brier, LogLoss, ECE, decomposition)
    # ----------------------------------------------------------
    loyo_validator = LOYOValidator(
        years=sorted(data_by_year.keys()),
        temporal_mode=pipeline.config.loyo_temporal_mode,
        enforce_pit=False,  # PIT is enforced above with explicit fold summaries.
    )

    # Per-fold state: scaler, feature selector, and ensemble models
    # re-fit each fold. Stored in a mutable container so predict_fn
    # can access them.
    #
    # FIX-STACKING-LEAKAGE: train_fn now trains ALL base models per fold
    # (not just LightGBM) and derives BMA weights from inner LOYO within
    # the training years. This means LOYO metrics reflect the actual
    # deployed ensemble, not a single model proxy.
    _fold_transforms = {
        "scaler": None,
        "selector": None,
        "selected_names": model_feature_names,
        "ensemble_models": {},    # {name: model}
        "ensemble_weights": {},   # {name: weight}
    }

    def train_fn(X_tr, y_tr, _margins_tr, fold_feature_names, w_tr):
        # Reset per-fold transforms
        _fold_transforms["scaler"] = None
        _fold_transforms["selector"] = None
        _fold_transforms["selected_names"] = list(fold_feature_names)
        _fold_transforms["ensemble_models"] = {}
        _fold_transforms["ensemble_weights"] = {}

        # Re-fit feature selector per fold (mirrors production pipeline)
        if pipeline.config.enable_feature_selection:
            try:
                fold_selector = FeatureSelector(
                    correlation_threshold=pipeline.config.correlation_threshold,
                    min_features=pipeline.config.min_features,
                    max_features=pipeline.config.max_features,
                )
                fold_selector.fit(X_tr, y_tr, fold_feature_names)
                X_tr = fold_selector.transform(X_tr)
                _fold_transforms["selector"] = fold_selector
                _fold_transforms["selected_names"] = fold_selector.get_selected_names()
            except Exception as _fs_exc:
                logger.debug("LOYO fold feature selection failed: %s", _fs_exc)

        # Re-fit scaler per fold (mirrors production pipeline)
        if pipeline.config.enable_feature_scaling and SCALER_AVAILABLE:
            fold_scaler = StandardScaler()
            X_tr = fold_scaler.fit_transform(X_tr)
            _fold_transforms["scaler"] = fold_scaler

        # FIX-STACKING-LEAKAGE: Train ALL base models, not just one
        trained = {}
        if LIGHTGBM_AVAILABLE:
            lgb = LightGBMRanker()
            lgb.train(
                X_tr, y_tr,
                feature_names=_fold_transforms["selected_names"],
                num_rounds=200,
                early_stopping_rounds=None,
                valid_set=None,
                sample_weight=w_tr,
            )
            trained["lgb"] = lgb

        if XGBOOST_AVAILABLE:
            try:
                xgb = XGBoostRanker()
                xgb.train(
                    X_tr, y_tr,
                    feature_names=_fold_transforms["selected_names"],
                    num_rounds=200,
                    early_stopping_rounds=None,
                    sample_weight=w_tr,
                )
                trained["xgb"] = xgb
            except Exception:
                pass

        if SKLEARN_AVAILABLE:
            logit = LogisticRegression(
                C=1.0, max_iter=2000,
                random_state=pipeline.config.random_seed,
            )
            logit.fit(X_tr, y_tr, sample_weight=w_tr)
            trained["logit"] = logit

        if SPREAD_MODEL_AVAILABLE and _margins_tr is not None:
            try:
                spread = SpreadRegressor(
                    sigma=pipeline.config.spread_sigma_init,
                )
                spread.train(X_tr, _margins_tr, num_rounds=200, sample_weight=w_tr)
                trained["spread"] = spread
            except Exception:
                pass

        _fold_transforms["ensemble_models"] = trained

        # Derive inner-fold BMA weights via cross-validation within
        # the training data.  Split training years into inner folds
        # and use BMA on pooled inner-OOS predictions.
        if BMA_AVAILABLE and len(trained) >= 2:
            try:
                # Use simple 3-fold temporal CV on training data for
                # inner BMA weight derivation
                n_tr = len(y_tr)
                inner_fold_size = n_tr // 3
                inner_preds = {name: [] for name in trained}
                inner_outcomes = []

                for i_fold in range(3):
                    start = i_fold * inner_fold_size
                    end = (i_fold + 1) * inner_fold_size if i_fold < 2 else n_tr
                    inner_val_idx = list(range(start, end))
                    inner_tr_idx = [j for j in range(n_tr) if j not in inner_val_idx]

                    if len(inner_tr_idx) < 20 or len(inner_val_idx) < 10:
                        continue

                    X_inner_tr = X_tr[inner_tr_idx]
                    y_inner_tr = y_tr[inner_tr_idx]
                    X_inner_val = X_tr[inner_val_idx]
                    y_inner_val = y_tr[inner_val_idx]
                    w_inner = w_tr[inner_tr_idx] if w_tr is not None else None
                    m_inner = _margins_tr[inner_tr_idx] if _margins_tr is not None else None

                    for name in trained:
                        try:
                            if name == "lgb" and LIGHTGBM_AVAILABLE:
                                m = LightGBMRanker()
                                m.train(X_inner_tr, y_inner_tr, num_rounds=200, sample_weight=w_inner)
                                inner_preds[name].extend(np.clip(m.predict(X_inner_val), 0.01, 0.99).tolist())
                            elif name == "xgb" and XGBOOST_AVAILABLE:
                                m = XGBoostRanker()
                                m.train(X_inner_tr, y_inner_tr, num_rounds=200, sample_weight=w_inner)
                                inner_preds[name].extend(np.clip(m.predict(X_inner_val), 0.01, 0.99).tolist())
                            elif name == "logit" and SKLEARN_AVAILABLE:
                                m = LogisticRegression(C=1.0, max_iter=2000, random_state=pipeline.config.random_seed)
                                m.fit(X_inner_tr, y_inner_tr, sample_weight=w_inner)
                                inner_preds[name].extend(np.clip(m.predict_proba(X_inner_val)[:, 1], 0.01, 0.99).tolist())
                            elif name == "spread" and SPREAD_MODEL_AVAILABLE and m_inner is not None:
                                m = SpreadRegressor(sigma=pipeline.config.spread_sigma_init)
                                m.train(X_inner_tr, m_inner, num_rounds=200, sample_weight=w_inner)
                                inner_preds[name].extend(np.clip(m.predict_probability(X_inner_val), 0.01, 0.99).tolist())
                        except Exception:
                            inner_preds[name].extend([0.5] * len(inner_val_idx))

                    inner_outcomes.extend(y_inner_val.tolist())

                if len(inner_outcomes) >= 30:
                    bma_inner_preds = {
                        name: np.clip(np.array(vals), 1e-7, 1 - 1e-7)
                        for name, vals in inner_preds.items()
                        if len(vals) == len(inner_outcomes)
                    }
                    if len(bma_inner_preds) >= 2:
                        bma = BayesianModelAveraging()
                        bma_result = bma.fit(bma_inner_preds, np.array(inner_outcomes))
                        if bma_result.weights:
                            _fold_transforms["ensemble_weights"] = bma_result.weights
            except Exception as _bma_exc:
                logger.debug("Inner BMA failed in LOYO fold: %s", _bma_exc)

        # Fallback: equal weights if BMA unavailable
        if not _fold_transforms["ensemble_weights"] and trained:
            _fold_transforms["ensemble_weights"] = {
                name: 1.0 / len(trained) for name in trained
            }

        # Return the primary model for interface compatibility
        if "lgb" in trained:
            return trained["lgb"]
        elif "logit" in trained:
            return trained["logit"]
        return None

    def predict_fn(model, X_pred):
        # Apply the same per-fold transforms used during training
        if _fold_transforms["selector"] is not None:
            try:
                X_pred = _fold_transforms["selector"].transform(X_pred)
            except (IndexError, ValueError):
                return np.full(len(X_pred), 0.5)
        if _fold_transforms["scaler"] is not None:
            X_pred = _fold_transforms["scaler"].transform(X_pred)

        # FIX-STACKING-LEAKAGE: Use the full ensemble with inner-derived
        # BMA weights, not just a single model
        ensemble_models = _fold_transforms.get("ensemble_models", {})
        ensemble_weights = _fold_transforms.get("ensemble_weights", {})

        if ensemble_models and ensemble_weights:
            weighted_pred = np.zeros(len(X_pred))
            total_weight = 0.0
            for name, weight in ensemble_weights.items():
                if name not in ensemble_models:
                    continue
                m = ensemble_models[name]
                try:
                    if isinstance(m, LightGBMRanker):
                        p = m.predict(X_pred)
                    elif isinstance(m, XGBoostRanker):
                        p = m.predict(X_pred)
                    elif hasattr(m, 'predict_probability'):
                        p = m.predict_probability(X_pred)
                    else:
                        p = m.predict_proba(X_pred)[:, 1]
                    weighted_pred += weight * np.clip(p, 0.01, 0.99)
                    total_weight += weight
                except Exception:
                    continue
            if total_weight > 0:
                return weighted_pred / total_weight
            # Fall through to single-model fallback

        # Fallback: single model prediction
        if model is None:
            return np.full(len(X_pred), 0.5)
        if isinstance(model, LightGBMRanker):
            return model.predict(X_pred)
        if isinstance(model, XGBoostRanker):
            return model.predict(X_pred)
        return model.predict_proba(X_pred)[:, 1]

    loyo_result_obj = loyo_validator.validate(data_by_year, train_fn, predict_fn)
    cv_results = loyo_result_obj.fold_results
    if not cv_results:
        return {"enabled": False, "reason": "no_cv_folds_completed"}

    per_year_brier = {}
    for result in cv_results:
        held_out_year = result.held_out_year
        year_entry = {
            "brier": round(result.brier_score, 5),
            "log_loss": round(result.log_loss, 5),
            "accuracy": round(result.accuracy, 4),
            "ece": round(result.calibration_error, 5),
            "brier_reliability": round(result.brier_reliability, 6),
            "brier_resolution": round(result.brier_resolution, 6),
            "brier_uncertainty": round(result.brier_uncertainty, 6),
            "brier_skill_score": round(result.brier_skill_score, 5),
            "train_size": result.n_train_games,
            "val_size": result.n_test_games,
        }
        per_year_brier[str(held_out_year)] = year_entry

    mean_brier = float(loyo_result_obj.mean_brier)
    mean_accuracy = float(loyo_result_obj.mean_accuracy)
    mean_logloss = float(loyo_result_obj.mean_logloss)
    mean_ece = float(np.mean([r.calibration_error for r in cv_results]))
    loyo_result = {
        "enabled": True,
        "years_evaluated": len(cv_results),
        "total_samples": int(sum(r.n_test_games for r in cv_results)),
        "mean_brier": round(mean_brier, 5),
        "mean_log_loss": round(mean_logloss, 5),
        "mean_logloss": round(mean_logloss, 5),
        "mean_accuracy": round(mean_accuracy, 4),
        "mean_ece": round(mean_ece, 5),
        "brier_decomposition": {
            "reliability": round(loyo_result_obj.mean_brier_reliability, 6),
            "resolution": round(loyo_result_obj.mean_brier_resolution, 6),
            "uncertainty": round(loyo_result_obj.mean_brier_uncertainty, 6),
        },
        "mean_brier_skill_score": round(loyo_result_obj.mean_brier_skill_score, 5),
        "brier_log_divergence": round(loyo_result_obj.brier_log_divergence, 5),
        "pit_validation": pit_summary,
        "per_year": per_year_brier,
    }

    # ----------------------------------------------------------
    # Step 3: AdmissionGate — formal model selection gate
    # ----------------------------------------------------------
    try:
        from ....ml.evaluation.admission_gate import (
            AdmissionGate,
            FoldMetrics as GateFoldMetrics,
            LOYOEvaluation,
        )

        baseline_folds = [
            GateFoldMetrics(
                year=r.held_out_year,
                brier_score=0.25,  # seed-only uninformed baseline
                calibration_error=0.0,
                n_games=r.n_test_games,
            )
            for r in cv_results
        ]
        candidate_folds = [
            GateFoldMetrics(
                year=r.held_out_year,
                brier_score=r.brier_score,
                calibration_error=r.calibration_error,
                n_games=r.n_test_games,
            )
            for r in cv_results
        ]

        # Use relaxed calibration threshold when comparing vs seed baseline
        # (seeds have ECE=0 by definition, so any model "degrades" calibration).
        # The 0.05 threshold ensures the model is reasonably well-calibrated
        # while still meaningfully gating on Brier improvement and fold rate.
        gate = AdmissionGate(
            min_mean_brier_improvement=0.0,
            min_fold_improvement_rate=0.60,
            max_calibration_degradation=0.05,
        )
        admission_result = gate.evaluate(
            "production_model",
            LOYOEvaluation("seed_baseline", baseline_folds),
            LOYOEvaluation("model", candidate_folds),
        )
        loyo_result["admission_gate"] = admission_result.to_dict()
        logger.info(
            "AdmissionGate: %s (Brier improvement=%.5f, fold rate=%.2f)",
            "PASSED" if admission_result.passed else "FAILED",
            admission_result.brier_improvement,
            admission_result.fold_improvement_rate,
        )
    except Exception as gate_exc:
        logger.warning("AdmissionGate evaluation failed: %s", gate_exc)
        loyo_result["admission_gate"] = {"error": str(gate_exc)}

    return loyo_result

