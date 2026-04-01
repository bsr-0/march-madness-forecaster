"""Baseline model training — orchestrator module."""


import logging
from datetime import date, timedelta

from ....data.features.proprietary_metrics import IncrementalMetricsEngine
from ....data.models.game_flow import GameFlow
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


# Imports from sibling sub-modules

from ._data import _load_historical_years, _apply_feature_preprocessing

from ._models import _train_all_models

from ._ensemble import _select_ensemble_and_evaluate



def _train_baseline_model(pipeline, game_flows: Dict[str, List[GameFlow]]) -> Dict:
    """Train baseline model: sample construction, multi-year augmentation,
    feature preprocessing, model training, and ensemble selection.
    """
    # Step 1: Build training samples from current-year games
    _sample_result = _build_current_year_samples(pipeline, game_flows)
    if _sample_result is None:
        return {"model": "none", "samples": 0}
    X_full, y_full, margins_full, sort_keys_full, bt_game_triples, n_unique_games = _sample_result
    n = len(y_full)

    # FEATURE MATRIX VALIDATION — catch NaN/inf/constant features that
    # indicate upstream data construction failures before they silently
    # degrade model quality.
    # ====================================================================
    # FIX C1: Preserve NaN for tree models (LightGBM/XGBoost) which
    # natively route missing values to the optimal split direction.
    # Only replace inf with NaN (inf is never valid).  The LR path
    # gets its own NaN imputation before fitting.
    _n_nan = int(np.isnan(X_full).sum())
    _n_inf = int(np.isinf(X_full).sum())
    if _n_inf > 0:
        logger.warning(
            "Feature matrix has %d inf values. Replacing inf with NaN.",
            _n_inf,
        )
        X_full = np.where(np.isinf(X_full), np.nan, X_full)
    if _n_nan > 0:
        logger.info(
            "Feature matrix has %d NaN values; preserved for tree-native handling.",
            _n_nan,
        )

    # Detect constant features (zero variance) that provide no signal
    _col_vars = np.var(X_full, axis=0)
    _constant_cols = int(np.sum(_col_vars < 1e-10))
    if _constant_cols > 0:
        logger.warning(
            "%d/%d features have near-zero variance in training data.",
            _constant_cols, X_full.shape[1],
        )

    # Log class balance for detecting systematic bias
    _pos_rate = float(np.mean(y_full))
    if abs(_pos_rate - 0.5) > 0.1:
        logger.warning(
            "Class imbalance detected: positive rate = %.3f (expected ~0.5). "
            "This may indicate systematic labeling bias.", _pos_rate,
        )

    # ====================================================================
    # LEAKAGE-SAFE ORDERING: split into train/val FIRST, then fit feature
    # selection and hyperparameter tuning on TRAINING data only.  This
    # prevents the validation set from influencing feature selection,
    # importance ranking, correlation pruning, or Optuna search.
    #
    # With symmetric augmentation enabled (default), each game produces
    # 2 interleaved samples: [orig, swap, orig, swap, ...].  Both
    # perspectives share the same game date / sort_key, so a simple
    # chronological split keeps pairs together — no leakage.
    # ====================================================================
    n = len(y_full)
    if getattr(pipeline.config, "enable_symmetric_augmentation", True):
        n_unique_games = n // 2  # Each game produces 2 samples
    else:
        n_unique_games = n  # Each game produces 1 sample
    train_samples = n
    valid_samples = 0

    # Reuse the pre-computed train/val boundary from
    # _compute_train_val_boundary() (called early in run()).
    if pipeline._validation_sort_key_boundary is not None and n >= 50:
        boundary = pipeline._validation_sort_key_boundary
        split_idx = n
        for i in range(n):
            if sort_keys_full[i] >= boundary:
                split_idx = i
                break
        # Option 2 fix: if the boundary split is degenerate (all samples
        # on one side), fall back to a local 80/20 chronological split on
        # the *current* filtered sample set rather than forcing empty eval.
        if split_idx <= 0 or split_idx >= n:
            logger.warning(
                "Boundary split degenerate (n=%d, split_idx=%d, boundary=%s); "
                "falling back to local 80/20 chronological split.",
                n,
                split_idx,
                str(boundary),
            )
            valid_count = max(5, int(0.2 * n))
            train_samples = n - valid_count
            valid_samples = valid_count
            if train_samples < 10:
                train_samples = n
                valid_samples = 0
        else:
            train_samples = split_idx
            valid_samples = n - split_idx
            if train_samples < 20:
                train_samples = n
                valid_samples = 0
    elif n >= 50:
        # Fallback: 80/20 chronological split
        valid_count = max(5, int(0.2 * n))
        train_samples = n - valid_count
        valid_samples = valid_count
        if train_samples < 10:
            train_samples = n
            valid_samples = 0

    train_X = X_full[:train_samples]
    train_y = y_full[:train_samples]
    train_margins = margins_full[:train_samples]
    train_sort_keys = sort_keys_full[:train_samples]
    if valid_samples > 0:
        eval_X = X_full[train_samples:]
        eval_y = y_full[train_samples:]
        eval_margins = margins_full[train_samples:]
    else:
        # FIX #6: Never use training data as eval — it inflates
        # confidence metrics and causes downstream leakage.  When we
        # can't split, we leave eval empty and skip eval-dependent steps.
        eval_X = np.empty((0, X_full.shape[1]))
        eval_y = np.array([], dtype=int)
        eval_margins = np.array([], dtype=np.float64)
        logger.warning(
            "Baseline split produced empty eval set: n=%d, train_samples=%d, "
            "valid_samples=%d, boundary=%s",
            n,
            train_samples,
            valid_samples,
            str(getattr(pipeline, "_validation_sort_key_boundary", None)),
        )

    # Step 2: Load multi-year historical training data
    n_current_year_train = train_samples
    feature_names = None
    (train_X, train_y, train_margins, train_sort_keys,
     train_samples, feature_names, feature_names_full,
     historical_training_stats) = _load_historical_years(
        pipeline, train_X, train_y, train_margins, train_sort_keys,
        X_full, n_current_year_train)

    # Step 3: Feature selection, scaling, and distribution shift detection
    (train_X, eval_X, X_full, feature_names, feature_names_full,
     fs_stats, dist_shift_stats, _loyo_raw_feature_dim) = _apply_feature_preprocessing(
        pipeline, train_X, eval_X, train_y, X_full,
        feature_names, feature_names_full, train_samples, valid_samples)


    # FIX M1: Split eval into dev (early stopping) and eval (final
    # evaluation).  Using the same data for both inflates eval metrics.
    # We use the first 40% of eval for early stopping and the rest for
    # final model selection / evaluation.  Align to even indices for
    # pair integrity.
    # Require >= 50 samples (25 games) so both dev and eval are large
    # enough: dev gets ~20 samples for early stopping, eval keeps ~30
    # for meaningful evaluation.
    if valid_samples >= 50:
        dev_count = int(valid_samples * 0.4)
        dev_count = max(dev_count, 10)
        dev_X = eval_X[:dev_count]
        dev_y = eval_y[:dev_count]
        eval_X = eval_X[dev_count:]
        eval_y = eval_y[dev_count:]
        eval_margins = eval_margins[dev_count:]
        valid_samples = len(eval_y)
        valid_set = (dev_X, dev_y)
        logger.info(
            "Eval split: %d dev samples (early stopping), %d eval samples (evaluation).",
            len(dev_y), valid_samples,
        )
    else:
        # Not enough eval data to split — use Optuna's tuned round
        # count without early stopping to avoid leakage.
        valid_set = None
        if valid_samples > 0:
            logger.info(
                "Eval set too small to split (%d samples); "
                "using fixed num_rounds (no early stopping).", valid_samples,
            )

    # FIX #3: Initialize round weights (populated during calibration with
    # tournament games; stays None for base training with regular-season only)
    pipeline._round_weights = None

    # ====================================================================
    # RECENCY WEIGHTING: late-season games receive higher sample weight.
    # Rationale: late-season games are played with settled rosters, against
    # tournament-caliber opponents, and their features more closely match
    # the end-of-season snapshot used at inference time.
    #
    # When multi-year training is active, year-based decay weights are
    # combined multiplicatively with intra-season recency weights.
    # Historical samples get year_weight * intra_weight, ensuring that
    # recent seasons' late-season games receive the highest overall weight.
    # ====================================================================
    train_sample_weight = None
    if pipeline.config.enable_recency_weighting and train_samples > 0:
        tk = train_sort_keys
        t_min, t_max = float(tk[0]), float(tk[-1])
        t_span = max(t_max - t_min, 1.0)
        progress = (tk - t_min) / t_span  # 0 = earliest, 1 = latest
        floor = pipeline.config.recency_decay_floor
        hl = max(pipeline.config.recency_half_life, 0.01)
        # Exponential ramp: earliest game → floor, latest game → 1.0
        raw_weight = floor + (1.0 - floor) * (1.0 - np.exp(-progress / hl))
        # Normalize so mean weight = 1.0 (preserves effective sample size)
        train_sample_weight = raw_weight / raw_weight.mean()

    # Combine year-based decay with intra-season recency
    if pipeline._historical_year_weights is not None and len(pipeline._historical_year_weights) == train_samples:
        if train_sample_weight is not None:
            train_sample_weight = train_sample_weight * pipeline._historical_year_weights
        else:
            train_sample_weight = pipeline._historical_year_weights.copy()
        # Re-normalize so mean = 1.0
        if train_sample_weight.mean() > 0:
            train_sample_weight = train_sample_weight / train_sample_weight.mean()

    # FIX #3: Apply round-weighted Brier training weights.
    # When tournament games are included in training (calibration mode),
    # weight them by the Kaggle round-weight schedule so the model
    # optimizes for the competition's actual scoring metric.
    if hasattr(pipeline, '_round_weights') and pipeline._round_weights is not None and len(pipeline._round_weights) == train_samples:
        if train_sample_weight is not None:
            train_sample_weight = train_sample_weight * pipeline._round_weights
        else:
            train_sample_weight = pipeline._round_weights.copy()
        if train_sample_weight.mean() > 0:
            train_sample_weight = train_sample_weight / train_sample_weight.mean()
        n_rw = int(np.sum(pipeline._round_weights > 1.0))
        if n_rw > 0:
            logger.info(
                "FIX #3: Applied round-weighted training: %d tournament "
                "games with Kaggle round weights (max=%.0f).",
                n_rw, float(np.max(pipeline._round_weights)),
            )
        else:
            logger.warning(
                "Round-weight verification: 0/%d training samples have "
                "weight > 1.0. No tournament games are receiving elevated "
                "weights — check that historical game files contain "
                "tournament games.",
                train_samples,
            )

    # Step 4b: Train individual models
    trained_models, tuning_stats = _train_all_models(
        pipeline, train_X, train_y, train_margins,
        eval_X, eval_y, eval_margins,
        feature_names, train_samples, valid_samples,
        train_sample_weight, valid_set, train_sort_keys,
        bt_game_triples)

    # Step 5: Ensemble selection, evaluation, and audits
    return _select_ensemble_and_evaluate(
        pipeline, trained_models, tuning_stats,
        train_X, train_y, train_margins, train_sort_keys,
        train_sample_weight, train_samples,
        eval_X, eval_y, eval_margins, valid_samples,
        feature_names, feature_names_full,
        _loyo_raw_feature_dim, n_unique_games, n,
        historical_training_stats, fs_stats, dist_shift_stats)




def _build_current_year_samples(pipeline, game_flows: Dict[str, List[GameFlow]]):
    """Build training samples from current-year games with PIT features.

    Returns:
        Tuple of (X_full, y_full, margins_full, sort_keys_full,
                  bt_game_triples, n_unique_games) or None if no samples.
    """
    samples: List[Tuple[int, np.ndarray, int]] = []

    # Exclude tournament games from baseline training to prevent leakage.
    # The model should only learn from regular-season game outcomes.
    all_games = [
        g for g in pipeline._unique_games(game_flows)
        if not pipeline._is_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"))
    ]

    # Hard tournament date cutoff — defense-in-depth guard that ensures
    # no game on or after the actual tournament start date survives.
    all_games = pipeline._exclude_tournament_games(all_games)

    # Late-season cutoff — with incremental PIT features this is no
    # longer strictly necessary (all games have accurate PIT features),
    # but retained as a configurable option.  Set cutoff_days=0 to
    # use all games.
    all_games_uncutoff = list(all_games)  # preserve for fallback
    if pipeline.config.late_season_training_cutoff_days > 0:
        tournament_start = TOURNAMENT_START_DATES.get(
            pipeline.config.year, date(pipeline.config.year, 3, 14)
        )
        cutoff_date = tournament_start - timedelta(days=pipeline.config.late_season_training_cutoff_days)
        cutoff_key = pipeline._game_sort_key(cutoff_date.isoformat())
        all_games = [
            g for g in all_games
            if pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")) >= cutoff_key
        ]
        # Fallback: if cutoff removes too many games, revert.
        # Threshold 60 balances the wider 45-day window against the
        # need for adequate training data (30 unique games minimum).
        if len(all_games) < 60:
            all_games = all_games_uncutoff

    # Build IncrementalMetricsEngine for current-year true PIT features.
    # Every training sample uses only data available before its game date,
    # eliminating all temporal leakage from season-end features.
    from src.data.features.proprietary_metrics import IncrementalMetricsEngine
    # Use prior-year Elo for cross-season carryover, matching what
    # historical training years get.  This eliminates the distribution
    # shift where historical Elo features are informative early-season
    # while current-year Elo starts at flat 1500.
    _prior_elo = getattr(pipeline, '_prior_year_elo', None)
    inc_engine = IncrementalMetricsEngine(
        pipeline._current_year_game_records,
        pipeline._current_year_conference_map or {},
        prior_elo=_prior_elo,
    )

    # Seed map for absolute features in matchup vector
    _seed_map: Dict[str, int] = {}
    # Roster overlay from current-year FeatureEngineer (RAPM, WARP, depth, etc.)
    _roster_overlay: Dict[str, Dict[int, float]] = {}
    for _tid, _tf in pipeline.feature_engineer.team_features.items():
        _seed_map[_tid] = _tf.seed if hasattr(_tf, "seed") and _tf.seed else 0
        _roster_overlay[_tid] = {
            11: _tf.total_rapm,
            12: _tf.top5_rapm,
            13: _tf.bench_rapm,
            14: _tf.total_warp,
            15: _tf.roster_continuity,
            17: _tf.avg_experience,
            18: _tf.bench_depth_score,
            55: _tf.top5_minutes_share,
            74: _tf.backcourt_rapm,
            75: _tf.frontcourt_rapm,
        }

    # SEED LEAKAGE FIX: Seeds are assigned on Selection Sunday (~March
    # 14-17) and must not appear in feature vectors for regular-season
    # training games.  This matches the guard in
    # _load_year_samples_incremental() at lines 3270-3274.
    _t_start = TOURNAMENT_START_DATES.get(
        pipeline.config.year, date(pipeline.config.year, 3, 14)
    )
    tournament_cutoff = _t_start.isoformat()

    for game in all_games:
        game_date = pipeline._coerce_game_date(
            getattr(game, "game_date", None),
            fallback_year=pipeline.config.year,
            game_id=getattr(game, "game_id", None),
            source="baseline_training",
        )
        game_key = pipeline._game_sort_key(game_date)

        # Compute true point-in-time metrics as of game date
        pit_metrics = inc_engine.compute_as_of(game_date)
        if game.team1_id not in pit_metrics or game.team2_id not in pit_metrics:
            continue

        m1 = pit_metrics[game.team1_id]
        m2 = pit_metrics[game.team2_id]
        if game_date > tournament_cutoff:
            s1 = _seed_map.get(game.team1_id, 0)
            s2 = _seed_map.get(game.team2_id, 0)
        else:
            s1, s2 = 0, 0
        # LEAKAGE FIX (Gap #1): External composites use end-of-season
        # ratings (latest RankingDayNum) and must not appear in feature
        # vectors for regular-season training games — the same temporal
        # constraint that applies to seeds above.
        if game_date > tournament_cutoff:
            _mc1 = pipeline._external_composites.get(game.team1_id, None) if hasattr(pipeline, '_external_composites') and pipeline._external_composites else None
            _mc2 = pipeline._external_composites.get(game.team2_id, None) if hasattr(pipeline, '_external_composites') and pipeline._external_composites else None
        else:
            _mc1, _mc2 = None, None
        _erc1 = _mc1.composite_rating if _mc1 is not None else 0.0
        _erc2 = _mc2.composite_rating if _mc2 is not None else 0.0
        _ers1 = _mc1.rating_spread if _mc1 is not None else 0.0
        _ers2 = _mc2.rating_spread if _mc2 is not None else 0.0
        # Massey multi-system features (individual system ratings)
        _mm1 = _mm2 = None
        if game_date > tournament_cutoff and hasattr(pipeline, '_massey_multi') and pipeline._massey_multi:
            _mm1 = pipeline._massey_multi.get(game.team1_id)
            _mm2 = pipeline._massey_multi.get(game.team2_id)
        v1 = IncrementalMetricsEngine.metrics_to_team_vector(
            m1, s1,
            external_rating_composite=_erc1,
            external_rating_spread=_ers1,
            massey_features=_mm1,
        )
        v2 = IncrementalMetricsEngine.metrics_to_team_vector(
            m2, s2,
            external_rating_composite=_erc2,
            external_rating_spread=_ers2,
            massey_features=_mm2,
        )
        # Overlay roster features (RAPM, WARP, depth, experience, etc.)
        for _v, _tid in ((v1, game.team1_id), (v2, game.team2_id)):
            _ov = _roster_overlay.get(_tid, {})
            for _idx, _val in _ov.items():
                _v[_idx] = _val
        vec = IncrementalMetricsEngine.build_matchup_vector(
            v1,
            v2,
            s1,
            s2,
            engine=inc_engine,
            team1_id=game.team1_id,
            team2_id=game.team2_id,
        )

        # S5 FIX: Use score-based label as primary (reliable), with
        # lead_history as fallback only if scores unavailable.
        _t1_score = getattr(game, "team1_score", None)
        _t2_score = getattr(game, "team2_score", None)
        if _t1_score is not None and _t2_score is not None and (_t1_score + _t2_score) > 0:
            _label = 1 if _t1_score > _t2_score else 0
            _margin = float(_t1_score - _t2_score)
        elif game.lead_history and len(game.lead_history) > 0:
            _label = 1 if game.lead_history[-1] > 0 else 0
            _margin = float(game.lead_history[-1])  # Approximate margin from final lead
        else:
            continue  # Skip games with no determinable outcome
        samples.append((game_key, vec, _label, _margin))

    # Collect BT game triples: (team1_id, team2_id, team1_won)
    bt_game_triples = []
    if pipeline.config.enable_bayesian_bt and BAYESIAN_BT_AVAILABLE:
        for game in all_games:
            _t1s = getattr(game, "team1_score", None)
            _t2s = getattr(game, "team2_score", None)
            if _t1s is not None and _t2s is not None and (_t1s + _t2s) > 0:
                bt_outcome = 1.0 if _t1s > _t2s else 0.0
            elif game.lead_history and len(game.lead_history) > 0:
                bt_outcome = 1.0 if game.lead_history[-1] > 0 else 0.0
            else:
                continue
            bt_game_triples.append((game.team1_id, game.team2_id, bt_outcome))

    if not samples:
        return None

    samples.sort(key=lambda x: x[0])
    X_full = np.stack([s[1] for s in samples])
    y_full = np.array([s[2] for s in samples], dtype=int)
    margins_full = np.array([s[3] for s in samples], dtype=np.float64)
    sort_keys_full = np.array([s[0] for s in samples])
    # (PIT metadata no longer needed — features computed incrementally)

    # Symmetric augmentation: double the dataset by adding the reverse-
    # perspective row for every game.  Historical years already get this
    # via sample_loading.py; current-year samples need it here.
    if getattr(pipeline.config, "enable_symmetric_augmentation", True):
        from ....ml.training.symmetric import symmetric_augment

        X_full, y_full, margins_full, _, sort_keys_full = symmetric_augment(
            X_full, y_full, margins_full, sort_keys=sort_keys_full,
        )

    # ====================================================================

    n = len(y_full)
    n_unique_games = n // 2 if getattr(pipeline.config, 'enable_symmetric_augmentation', True) else n
    return X_full, y_full, margins_full, sort_keys_full, bt_game_triples, n_unique_games




def _build_enriched_meta(base_X: np.ndarray) -> np.ndarray:
    """Build enriched meta-features from base model predictions.

    Given k base model columns, returns k + C(k,2) + 3 columns:
      - k base predictions
      - C(k,2) pairwise interactions
      - 3 aggregates: max, min, std
    """
    parts = [base_X]
    k = base_X.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            parts.append((base_X[:, i] * base_X[:, j]).reshape(-1, 1))
    parts.append(np.max(base_X, axis=1).reshape(-1, 1))
    parts.append(np.min(base_X, axis=1).reshape(-1, 1))
    parts.append(np.std(base_X, axis=1).reshape(-1, 1))
    return np.hstack(parts)




def _select_best_single_model(
    pipeline,
    trained_models: List[Tuple],
    eval_y: np.ndarray,
) -> str:
    """Select best single model using training-data cross-validation.

    FIX-STACKING-LEAKAGE: Model selection must NOT depend on eval_y.
    Using eval_y to select which model to deploy means the eval set
    influenced a pipeline decision, contaminating all downstream metrics.

    Instead, we use a priority-based selection that reflects the production
    baseline's model hierarchy (spread > logistic > lgb > xgb), which is
    determined by domain knowledge, not by fitting to evaluation data.

    The eval_y parameter is retained for backward API compatibility but
    is NOT used for selection decisions.
    """
    if not trained_models:
        return "none"

    name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression", "spread": "spread_regressor"}

    # FIX-STACKING-LEAKAGE: Use a fixed priority order based on domain
    # knowledge (production baseline hierarchy), NOT eval-set performance.
    # Spread is preferred because margin regression provides richer signal;
    # logistic regression provides strong regularization; tree models are
    # secondary diversity contributors.
    _PRIORITY = {"spread": 0, "logit": 1, "lgb": 2, "xgb": 3}

    # Select the highest-priority trained model
    best_name, best_model, best_preds = min(
        trained_models,
        key=lambda t: _PRIORITY.get(t[0], 99),
    )

    logger.info(
        "Single-model selection (priority-based, no eval_y dependence): %s",
        best_name,
    )

    pipeline._set_primary_model(best_name, best_model)
    return name_map.get(best_name, best_name)




def _set_primary_model(pipeline, name: str, model) -> None:
    """Set a single model as the primary baseline predictor."""
    if name == "lgb":
        pipeline.baseline_model.lgb_model = model
        pipeline.baseline_model.xgb_model = None
        pipeline.baseline_model.logit_model = None
    elif name == "xgb":
        pipeline.baseline_model.xgb_model = model
        pipeline.baseline_model.lgb_model = None
        pipeline.baseline_model.logit_model = None
    elif name == "logit":
        pipeline.baseline_model.logit_model = model
        pipeline.baseline_model.lgb_model = None
        pipeline.baseline_model.xgb_model = None
    elif name == "spread":
        pipeline.baseline_model.spread_model = model

# ------------------------------------------------------------------
# P0: Leave-One-Year-Out Cross-Validation (multi-year validation)
# ------------------------------------------------------------------

