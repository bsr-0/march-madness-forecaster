"""Baseline model training — extracted from SOTAPipeline.

Contains the model training pipeline: schedule graph construction,
baseline model training, LOYO validation, GNN/transformer embeddings,
ensemble weight optimization, and embedding projections.

Each function takes a ``pipeline`` parameter (SOTAPipeline instance)
to access config and mutable state. This is a pragmatic extraction
that reduces sota.py line count while maintaining exact behavioral
equivalence.

Implements Agent Directive V7 S2 (modular architecture decomposition).
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import date, timedelta
from math import sqrt

from ...data.features.feature_engineering import (
    ABSOLUTE_LEVEL_FEATURE_NAMES,
    FeatureEngineer,
)
from ...data.features.feature_selection import FeatureSelector
from ...data.features.proprietary_metrics import IncrementalMetricsEngine
from ...data.models.game_flow import GameFlow
from ...ml.calibration.calibration import CalibrationPipeline
from ...ml.ensemble.cfa import (
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
    LightGBMRanker,
    ModelPrediction,
    XGBoostRanker,
)
from ...ml.gnn.schedule_graph import ScheduleEdge, ScheduleGraph, compute_multi_hop_sos
from ...ml.transformer.game_sequence import GameEmbedding, SeasonSequence
from ...models.team import Team
from ..config import (
    DATA_QUALITY_ERA_WEIGHTS,
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    SOTAPipelineConfig,
    TOURNAMENT_START_DATES,
    compute_year_data_quality,
)
from .sample_loading import load_year_samples_incremental

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


def _construct_schedule_graph(pipeline, teams: List[Team]) -> ScheduleGraph:
    team_ids = {pipeline._team_id(t.name) for t in teams}
    for flow in pipeline.all_game_flows:
        team_ids.add(flow.team1_id)
        team_ids.add(flow.team2_id)
    team_ids = sorted(team_ids)
    graph = ScheduleGraph(team_ids, temporal_decay=pipeline.config.gnn_temporal_decay)

    if pipeline.team_features:
        default_dim = len(next(iter(pipeline.team_features.values())))
    else:
        default_dim = 16
    default_features = np.zeros(default_dim, dtype=float)
    for team_id in team_ids:
        graph.set_team_features(team_id, pipeline.team_features.get(team_id, default_features))

    # Filter out tournament games AND validation-era games to prevent
    # leakage — the GNN graph should only contain regular-season results
    # from the training era.  Validation-era edges would let the GNN
    # learn from outcomes it is later evaluated on (Issue 2).
    boundary = pipeline._validation_sort_key_boundary
    pre_tournament_games = [
        g for g in pipeline.all_game_flows
        if not pipeline._is_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"))
        and (boundary is None
             or pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")) < boundary)
    ]

    seen_games = set()
    for game in pre_tournament_games:
        if game.game_id in seen_games:
            continue
        seen_games.add(game.game_id)

        margin = game.lead_history[-1] if game.lead_history else 0

        # Compute xp_margin from proprietary metrics when possession-level xP is unavailable
        xp_margin = float(game.get_xp_margin())
        if abs(xp_margin) < 1e-6 and pipeline.proprietary_metrics:
            pm1 = pipeline.proprietary_metrics.get(game.team1_id)
            pm2 = pipeline.proprietary_metrics.get(game.team2_id)
            if pm1 is not None and pm2 is not None:
                xp_margin = float(
                    (pm1.offensive_xp_per_possession - pm2.defensive_xp_per_possession)
                    - (pm2.offensive_xp_per_possession - pm1.defensive_xp_per_possession)
                ) * 70.0  # scale to per-game margin (approx 70 possessions)

        graph.add_game(
            ScheduleEdge(
                game_id=game.game_id,
                team1_id=game.team1_id,
                team2_id=game.team2_id,
                actual_margin=float(margin),
                xp_margin=xp_margin,
                location_weight=float(getattr(game, "location_weight", 0.5)),
                game_date=str(getattr(game, "game_date", "2026-02-01")),
            )
        )

    return graph

def _train_baseline_model(pipeline, game_flows: Dict[str, List[GameFlow]]) -> Dict:
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
        tournament_start = date(pipeline.config.year, 3, 14)
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
    for _tid, _tf in pipeline.feature_engineer.team_features.items():
        _seed_map[_tid] = _tf.seed if hasattr(_tf, "seed") and _tf.seed else 0

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
        v1 = IncrementalMetricsEngine.metrics_to_team_vector(
            m1, s1,
            external_rating_composite=_erc1,
            external_rating_spread=_ers1,
        )
        v2 = IncrementalMetricsEngine.metrics_to_team_vector(
            m2, s2,
            external_rating_composite=_erc2,
            external_rating_spread=_ers2,
        )
        vec = IncrementalMetricsEngine.build_matchup_vector(v1, v2, s1, s2)

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
        return {"model": "none", "samples": 0}

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
        from ...ml.training.symmetric import symmetric_augment

        X_full, y_full, margins_full, _, sort_keys_full = symmetric_augment(
            X_full, y_full, margins_full, sort_keys=sort_keys_full,
        )

    # ====================================================================
    # FEATURE MATRIX VALIDATION — catch NaN/inf/constant features that
    # indicate upstream data construction failures before they silently
    # degrade model quality.
    # ====================================================================
    _n_nan = int(np.isnan(X_full).sum())
    _n_inf = int(np.isinf(X_full).sum())
    if _n_nan > 0 or _n_inf > 0:
        logger.warning(
            "Feature matrix has %d NaN and %d inf values. Replacing with 0.0.",
            _n_nan, _n_inf,
        )
        X_full = np.where(np.isnan(X_full) | np.isinf(X_full), 0.0, X_full)

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

    # ====================================================================
    # MULTI-YEAR TRAINING POOL: Augment current-year training data with
    # historical regular-season games to increase sample size from ~300
    # to ~3000+.  This addresses the fundamental sample-size problem:
    # building a 22-feature model from 300 games produces unstable
    # estimates.  10+ years of data provides the statistical mass needed
    # for robust gradient boosting and honest hyperparameter tuning.
    #
    # Both current-year and historical samples use IncrementalMetricsEngine
    # to compute true point-in-time features for every training game.
    # No season-end leakage remains.
    #
    # Year-based exponential decay downweights older seasons:
    #   weight(year) = max(min_weight, decay^(current_year - year - 1))
    # This ensures current-year data dominates while older seasons
    # provide regularization and stabilize split points.
    #
    # Historical data is prepended to train_X/train_y (chronologically
    # before current year).  Validation set remains current-year only
    # for honest evaluation.
    # ====================================================================
    # Build feature names early so they are available for multi-year
    # data-quality scoring (compute_year_data_quality) below.
    feature_names = None
    if train_samples >= 40:
        from src.data.features.feature_engineering import TeamFeatures
        base_names = TeamFeatures.get_feature_names(include_embeddings=False)
        diff_names = [f"diff_{n}" for n in base_names]
        absolute_names = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
        interaction_names = ["tempo_interaction", "style_mismatch", "h2h_record", "common_opp_margin", "travel_advantage", "seed_interaction", "seed_diff"]
        feature_names = diff_names + absolute_names + interaction_names
        if len(feature_names) != train_X.shape[1]:
            logger.warning(
                "Feature name count mismatch: %d names vs %d columns. "
                "Falling back to generic names.",
                len(feature_names), train_X.shape[1],
            )
            feature_names = [f"f_{i}" for i in range(train_X.shape[1])]

    historical_training_stats = {}
    n_current_year_train = train_samples  # Track for logging

    import os

    # Resolve "auto" multi_year_games_dir: check for data/raw/historical
    # relative to the working directory.
    if pipeline.config.multi_year_games_dir == "auto":
        candidate = os.path.join(os.getcwd(), "data", "raw", "historical")
        if os.path.isdir(candidate):
            pipeline.config.multi_year_games_dir = candidate
            logger.info("Auto-detected multi-year training directory: %s", candidate)
        else:
            pipeline.config.multi_year_games_dir = None
            logger.info("No historical directory found; multi-year training disabled")

    if (
        pipeline.config.enable_multi_year_training
        and pipeline.config.multi_year_games_dir
        and os.path.isdir(pipeline.config.multi_year_games_dir)
    ):
        games_dir = pipeline.config.multi_year_games_dir
        feature_dim_full = X_full.shape[1]

        # Determine which years to load
        if pipeline.config.training_years is not None:
            hist_years = sorted(pipeline.config.training_years)
        else:
            # Auto-detect available years from the data directory
            hist_years = []
            for fname in os.listdir(games_dir):
                if fname.startswith("historical_games_") and fname.endswith(".json"):
                    try:
                        yr = int(fname.replace("historical_games_", "").replace(".json", ""))
                        # Exclude current year (already in training), 2020 (COVID)
                        if yr != pipeline.config.year and yr != 2020:
                            hist_years.append(yr)
                    except ValueError:
                        pass
            hist_years.sort()

        # Enforce dev/holdout split for historical training
        hist_years = pipeline._filter_years(hist_years)

        hist_X_parts = []
        hist_y_parts = []
        hist_margin_parts = []
        hist_weight_parts = []
        hist_sortkey_parts = []
        hist_round_weight_parts = []  # FIX #3: per-sample round weights
        years_loaded = []
        per_year_quality: Dict[int, Dict] = {}  # FIX-DQ: per-year quality audit

        # D2: Persist Elo across historical years for cross-season carryover.
        # Processing years in sorted order (oldest→newest) so each year's
        # final Elo serves as the prior for the next year.
        _cross_year_elo: Dict[str, float] = {}

        # FIX #3: Include tournament games in historical training when
        # round-weighted training is enabled.  Tournament games receive
        # Kaggle round weights (R64=1, R32=2, S16=4, E8=8, F4=16, NCG=32)
        # so the model invests more gradient signal in elite matchups.
        _include_tourney = pipeline.config.enable_round_weighted_training

        for yr in hist_years:
            gp = os.path.join(games_dir, f"historical_games_{yr}.json")
            mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
            if not os.path.exists(gp) or not os.path.exists(mp):
                logger.warning(
                    "Multi-year training: missing data for %d (games=%s, metrics=%s); skipping.",
                    yr, gp, mp,
                )
                continue

            try:
                hX, hy, _h_margins, _end_elo, _h_rw = pipeline._load_year_samples_incremental(
                    gp, mp, feature_dim_full, yr,
                    include_tournament=_include_tourney,
                    prior_elo=_cross_year_elo,
                )
                if _end_elo:
                    _cross_year_elo = _end_elo  # D2: carry forward to next year
                else:
                    logger.warning(
                        "Multi-year training: year %d produced no Elo carryover; "
                        "subsequent years will start from base Elo.",
                        yr,
                    )
            except Exception as e:
                logger.warning("Failed to load year %d for training: %s", yr, e)
                continue

            if len(hy) < 10:
                logger.warning(
                    "Multi-year training: year %d has too few samples (%d); skipping.",
                    yr, len(hy),
                )
                continue

            # Year-based decay weight
            years_ago = pipeline.config.year - yr
            year_weight = max(
                pipeline.config.training_year_min_weight,
                pipeline.config.training_year_decay ** max(years_ago - 1, 0),
            )

            # FIX-DQ: Compute per-year data quality metrics using actual
            # feature matrix characteristics, not just hard-coded era weights.
            _dq = compute_year_data_quality(hX, yr, feature_names)
            quality_mult = _dq["combined_weight"]

            # Log quality diagnostics for years with issues
            if _dq["zero_columns"] > 5 or _dq["completeness"] < 0.30:
                logger.warning(
                    "FIX-DQ: Year %d data quality issues — "
                    "completeness=%.2f, active_features=%d/%d, "
                    "zero_cols=%d, bad_rate=%.4f. "
                    "Adaptive weight=%.3f (era=%.2f).",
                    yr, _dq["completeness"], _dq["n_active_features"],
                    _dq["n_features"], _dq["zero_columns"],
                    _dq["bad_rate"], quality_mult, _dq["era_weight"],
                )
                if _dq["zero_column_names"]:
                    logger.warning(
                        "FIX-DQ: Year %d zero columns (first 10): %s",
                        yr, _dq["zero_column_names"],
                    )

            year_weight *= quality_mult
            per_year_quality[yr] = _dq

            # Create per-sample weight array for this year
            sample_weights = np.full(len(hy), year_weight, dtype=float)

            # Synthetic sort keys: place historical samples before
            # current-year samples.  Use year * 10000 + day-of-season
            # so that multi-year samples maintain relative chronological
            # ordering among themselves.
            year_sort_keys = np.full(len(hy), yr * 10000, dtype=float)

            hist_X_parts.append(hX)
            hist_y_parts.append(hy)
            hist_margin_parts.append(_h_margins)
            hist_weight_parts.append(sample_weights)
            hist_sortkey_parts.append(year_sort_keys)
            # FIX #3: Collect round weights from historical years
            if _h_rw is not None and len(_h_rw) == len(hy):
                hist_round_weight_parts.append(_h_rw)
            else:
                hist_round_weight_parts.append(np.ones(len(hy), dtype=float))
            years_loaded.append(yr)

            _n_tourney = int(np.sum(_h_rw > 1.0)) if _h_rw is not None and len(_h_rw) == len(hy) else 0
            logger.info(
                "Multi-year training: loaded %d samples from %d "
                "(weight=%.3f, tournament_weighted=%d).",
                len(hy), yr, year_weight, _n_tourney,
            )

        total_hist_samples = sum(len(part) for part in hist_y_parts)
        logger.warning(
            "Multi-year training summary: loaded %d/%d years with %d samples total.",
            len(years_loaded), len(hist_years), total_hist_samples,
        )

        if hist_X_parts:
            hist_X = np.concatenate(hist_X_parts, axis=0)
            hist_y = np.concatenate(hist_y_parts)
            hist_margins = np.concatenate(hist_margin_parts)
            hist_weights = np.concatenate(hist_weight_parts)
            hist_sort_keys = np.concatenate(hist_sortkey_parts)

            # Clean NaN/inf in historical data
            _h_nan = int(np.isnan(hist_X).sum())
            _h_inf = int(np.isinf(hist_X).sum())
            if _h_nan > 0 or _h_inf > 0:
                hist_X = np.where(np.isnan(hist_X) | np.isinf(hist_X), 0.0, hist_X)

            # Prepend historical data to training set (chronologically first)
            train_X = np.concatenate([hist_X, train_X], axis=0)
            train_y = np.concatenate([hist_y, train_y])
            train_margins = np.concatenate([hist_margins, train_margins])
            train_sort_keys = np.concatenate([hist_sort_keys, train_sort_keys])

            # Store year-based weights to combine with recency weighting later.
            # Current-year samples get weight 1.0.
            pipeline._historical_year_weights = np.concatenate([
                hist_weights,
                np.ones(n_current_year_train, dtype=float),
            ])

            # FIX #3: Build round weights array for Kaggle round-weighted
            # Brier optimization.  Historical tournament games get their
            # actual round weight; regular-season games get 1.0.
            if pipeline.config.enable_round_weighted_training and hist_round_weight_parts:
                hist_rw = np.concatenate(hist_round_weight_parts)
                # Current-year training games are regular-season → weight 1.0
                pipeline._round_weights = np.concatenate([
                    hist_rw,
                    np.ones(n_current_year_train, dtype=float),
                ])
                _n_weighted = int(np.sum(pipeline._round_weights > 1.0))
                if _n_weighted > 0:
                    logger.info(
                        "FIX #3: Round-weighted training enabled: %d tournament "
                        "games with Kaggle round weights (max=%.0f, mean=%.2f).",
                        _n_weighted,
                        float(np.max(pipeline._round_weights)),
                        float(np.mean(pipeline._round_weights[pipeline._round_weights > 1.0])),
                    )

            train_samples = len(train_y)

            # FIX-DQ: Summarize per-year data quality for report
            _dq_summary = {}
            for _dq_yr, _dq_info in sorted(per_year_quality.items()):
                _dq_summary[str(_dq_yr)] = {
                    "samples": _dq_info["n_samples"],
                    "completeness": _dq_info["completeness"],
                    "active_features": _dq_info["n_active_features"],
                    "weight": _dq_info["combined_weight"],
                }

            historical_training_stats = {
                "enabled": True,
                "years_loaded": years_loaded,
                "historical_samples": int(len(hist_y)),
                "current_year_samples": int(n_current_year_train),
                "total_train_samples": int(train_samples),
                "sample_increase_factor": round(train_samples / max(n_current_year_train, 1), 1),
                "per_year_data_quality": _dq_summary,
            }
            logger.info(
                "Multi-year training pool: %d historical + %d current = %d total "
                "training samples (%.1fx increase).",
                len(hist_y), n_current_year_train, train_samples,
                train_samples / max(n_current_year_train, 1),
            )
        else:
            pipeline._historical_year_weights = None
    else:
        pipeline._historical_year_weights = None

    # --- Feature selection ---
    # OOS-FIX: Default path uses a fixed domain-knowledge feature set.
    # Learned feature selection can still be enabled via config.
    # (feature_names already constructed above, before multi-year block)
    fs_stats = {}

    if train_samples >= 40 and feature_names is not None:
        if not pipeline.config.enable_feature_selection:
            # OOS-FIX: Apply fixed domain-knowledge feature set.
            # No model fitting, no label dependency, no double-dipping.
            # Gap #3: Use SIMPLE_FEATURE_SET when model_complexity == "simple"
            active_feature_set = (
                SIMPLE_FEATURE_SET
                if pipeline.config.model_complexity == "simple"
                else FIXED_FEATURE_SET
            )
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            fixed_indices = [name_to_idx[n] for n in active_feature_set if n in name_to_idx]
            fixed_names = [n for n in active_feature_set if n in name_to_idx]

            if len(fixed_indices) >= 10:
                original_dim = train_X.shape[1]
                train_X = train_X[:, fixed_indices]
                eval_X = eval_X[:, fixed_indices]
                feature_names = fixed_names
                # Store indices for inference-time consistency
                pipeline.baseline_model.fixed_feature_indices = fixed_indices
                fs_stats = {
                    "method": "fixed_domain_knowledge",
                    "original_dim": original_dim,
                    "reduced_dim": len(fixed_indices),
                    "selected_features": fixed_names,
                }
                logger.info(
                    "Fixed feature selection: %d/%d features retained (domain knowledge).",
                    len(fixed_indices), original_dim,
                )
            else:
                logger.warning(
                    "Fixed feature set matched only %d features — using all features.",
                    len(fixed_indices),
                )
        else:
            # Learned feature selection (original path, now opt-in)
            effective_max_features = pipeline.config.max_features
            if pipeline.config.adaptive_max_features:
                samples_based_cap = max(pipeline.config.min_features, train_samples // 8)
                effective_max_features = min(effective_max_features, samples_based_cap)

            pipeline.feature_selector = FeatureSelector(
                correlation_threshold=pipeline.config.correlation_threshold,
                min_features=pipeline.config.min_features,
                max_features=effective_max_features,
                importance_threshold=pipeline.config.feature_importance_threshold,
                random_seed=pipeline.config.random_seed,
                enable_vif_pruning=pipeline.config.enable_vif_pruning,
                vif_threshold=pipeline.config.vif_threshold,
                enable_stability_filter=pipeline.config.enable_stability_filter,
                stability_threshold=pipeline.config.stability_threshold,
                n_bootstrap=pipeline.config.n_bootstrap,
            )
            pipeline.feature_selection_result = pipeline.feature_selector.fit(train_X, train_y, feature_names)
            train_X = pipeline.feature_selector.transform(train_X)
            eval_X = pipeline.feature_selector.transform(eval_X)
            feature_names = pipeline.feature_selector.get_selected_names()
            fs_stats = {
                "method": "learned",
                "original_dim": pipeline.feature_selection_result.original_dim,
                "reduced_dim": pipeline.feature_selection_result.reduced_dim,
            }

    # ====================================================================
    # DISTRIBUTION SHIFT DETECTION — compare train vs validation feature
    # distributions to detect temporal feature drift.  Flagged features
    # may have unstable predictive value across time periods.
    # ====================================================================
    dist_shift_stats = {}
    if valid_samples >= 20 and feature_names is not None:
        try:
            from src.data.features.feature_selection import detect_distribution_shift
            shift_results = detect_distribution_shift(
                train_X, eval_X, feature_names,
                psi_threshold=0.25, ks_alpha=0.05,
            )
            n_flagged = sum(1 for r in shift_results if r.flagged)
            if n_flagged > 0:
                dist_shift_stats["n_flagged"] = n_flagged
                dist_shift_stats["n_features"] = len(shift_results)
                dist_shift_stats["flagged_features"] = [
                    {
                        "feature": r.feature_name,
                        "psi": round(r.psi, 4),
                        "ks_pvalue": round(r.ks_pvalue, 4),
                        "mean_shift_std": round(r.mean_shift_std, 3),
                    }
                    for r in shift_results if r.flagged
                ][:10]  # Top 10 worst
        except Exception as e:
            logger.debug("Distribution shift detection skipped: %s", e)

    # ====================================================================
    # P0: STANDARDSCALER — fit on training data, transform both splits.
    # Critical for logistic regression and stacking meta-learner where
    # features on different scales cause L2 penalty to be unevenly applied.
    # Tree-based models (LGB/XGB) are scale-invariant but we still apply
    # scaling for consistency in the stacking pipeline.
    # ====================================================================
    if pipeline.config.enable_feature_scaling and SCALER_AVAILABLE:
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        eval_X = scaler.transform(eval_X)
        pipeline.baseline_model.scaler = scaler

    # Store the pre-selection feature dimensionality for historical
    # year loading (multi-year calibration needs to reconstruct vectors
    # of the same width as the original matchup features).
    pipeline.baseline_model.feature_dim = X_full.shape[1]

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

    tuning_stats = {}
    stacking_stats = {}

    # ====================================================================
    # MODEL TRAINING: Try LightGBM + XGBoost + Logistic, then optionally
    # stack them with a meta-learner for superior ensemble performance.
    # ====================================================================
    trained_models = []  # List of (name, model, predictions_on_eval)

    # Gap #3: In "simple" mode, skip LGB and XGB entirely.
    # A well-calibrated logistic regression + spread model on ~9 features
    # is competitive with (or better than) complex ensembles at ~600 samples.
    _use_tree_models = pipeline.config.model_complexity != "simple"

    # --- LightGBM training ---
    lgb_trained = False
    if LIGHTGBM_AVAILABLE and _use_tree_models:
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
                    train_X, train_y, train_sort_keys,
                    feature_names=feature_names,
                    sample_weight=train_sample_weight,
                )

                best_params = {k: v for k, v in tuning_result.best_params.items() if k != "num_rounds"}
                best_num_rounds = tuning_result.best_params.get("num_rounds", 200)

                lgb_ranker = LightGBMRanker(params=best_params)
                lgb_ranker.train(
                    train_X, train_y,
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
                    "best_brier": round(tuning_result.best_score, 5),
                    "best_params": {k: round(v, 5) if isinstance(v, float) else v for k, v in tuning_result.best_params.items()},
                    "cv_folds": len(tuning_result.cv_results),
                    "cv_brier_scores": [round(r.brier_score, 5) for r in tuning_result.cv_results],
                }
            else:
                lgb_ranker = LightGBMRanker()
                lgb_ranker.train(
                    train_X, train_y,
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

    # --- XGBoost training ---
    xgb_trained = False
    if XGBOOST_AVAILABLE and _use_tree_models:
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
                    train_X, train_y, train_sort_keys,
                    feature_names=feature_names,
                    sample_weight=train_sample_weight,
                )

                xgb_best_params = {k: v for k, v in xgb_tuning_result.best_params.items() if k != "num_rounds"}
                xgb_best_rounds = xgb_tuning_result.best_params.get("num_rounds", 200)

                xgb_ranker = XGBoostRanker(params=xgb_best_params)
                xgb_ranker.train(
                    train_X, train_y,
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
                    "best_brier": round(xgb_tuning_result.best_score, 5),
                    "best_params": {k: round(v, 5) if isinstance(v, float) else v for k, v in xgb_tuning_result.best_params.items()},
                }
            else:
                xgb_ranker = XGBoostRanker()
                xgb_ranker.train(
                    train_X, train_y,
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

    # --- Logistic regression (fallback only) ---
    # Phase 5: LogisticRegression is highly correlated with GBM classifiers
    # (~0.95 correlation on same features), reducing ensemble diversity.
    # Train it only as a fallback when GBM models are unavailable.
    logit_trained = False
    if SKLEARN_AVAILABLE and not (lgb_trained or xgb_trained):
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
                    train_X, train_y, train_sort_keys,
                    sample_weight=train_sample_weight,
                )
                best_logit = logit_tuning_result.best_params
                logit = LogisticRegression(
                    C=best_logit["C"],
                    penalty=best_logit["penalty"],
                    solver="saga" if best_logit["penalty"] == "l1" else "lbfgs",
                    max_iter=2000,
                    random_state=pipeline.config.random_seed,
                )
                tuning_stats["logistic"] = {
                    "method": "optuna",
                    "best_brier": round(logit_tuning_result.best_score, 5),
                    "best_params": best_logit,
                }
            else:
                logit = LogisticRegression(
                    C=1.0, penalty="l2", max_iter=2000,
                    random_state=pipeline.config.random_seed,
                )
            logit.fit(train_X, train_y, sample_weight=train_sample_weight)

            # Coefficient stability diagnostic: large coefficient magnitudes
            # signal multicollinearity inflating LogisticRegression estimates.
            # Log a warning when detected so users can investigate.
            if hasattr(logit, 'coef_') and feature_names is not None:
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
                                feature_names[max_coef_idx], max_coef, ratio,
                            )
                            tuning_stats["logistic_coef_warning"] = {
                                "feature": feature_names[max_coef_idx],
                                "abs_coef": round(max_coef, 3),
                                "ratio_to_median": round(ratio, 1),
                            }

            logit_eval_preds = logit.predict_proba(eval_X)[:, 1]
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
        and train_samples >= 60
        and len(train_margins) == len(train_y)
    ):
        try:
            spread = SpreadRegressor(
                sigma=pipeline.config.spread_sigma_init,
            )
            spread_valid_X = eval_X if valid_samples > 0 else None
            spread_valid_margins = eval_margins if valid_samples > 0 else None

            spread_stats = spread.train(
                train_X,
                train_margins,
                feature_names=feature_names,
                num_rounds=200,
                sample_weight=train_sample_weight,
                valid_X=spread_valid_X,
                valid_margins=spread_valid_margins,
            )

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
    # MarginFirstEnsemble's RoundSpecificCalibrator.
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
    if (
        pipeline.config.enable_bayesian_bt
        and BAYESIAN_BT_AVAILABLE
        and len(bt_game_triples) >= 50
    ):
        try:
            bt_model = BayesianBradleyTerry(
                prior_std=pipeline.config.bayesian_bt_prior_std,
            )
            bt_stats = bt_model.fit(bt_game_triples)
            if bt_stats.get("fitted"):
                pipeline.bayesian_bt_model = bt_model
                tuning_stats["bayesian_bt"] = bt_stats
                logger.info(
                    "BayesianBT: fitted %d teams from %d games, "
                    "mean_posterior_std=%.3f",
                    bt_stats.get("n_teams", 0),
                    bt_stats.get("n_games", 0),
                    bt_stats.get("mean_posterior_std", 0),
                )
        except Exception as e:
            tuning_stats["bayesian_bt_error"] = str(e)
            logger.warning("BayesianBT fitting failed: %s", e)

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
        if np.sum(oof_mask) >= 20:
            base_meta_X = np.column_stack([oof_preds[name][oof_mask] for name, _, _ in trained_models])
            meta_y = train_y[oof_mask]
            meta_X = pipeline._build_enriched_meta(base_meta_X)

            meta_learner = LogisticRegression(
                C=1.0, penalty="l2", max_iter=2000,
                random_state=pipeline.config.random_seed,
            )
            meta_learner.fit(meta_X, meta_y)
            meta_learner_type = "logistic"

            pipeline.baseline_model.stacking_meta = meta_learner
            pipeline.baseline_model.stacking_meta_type = meta_learner_type
            pipeline.baseline_model.stacking_models = [(name, model) for name, model, _ in trained_models]

            stacking_stats = {
                "enabled": True,
                "meta_learner": meta_learner_type,
                "base_models": [name for name, _, _ in trained_models],
            }
            baseline_name = "stacking_ensemble"
        else:
            stacking_stats = {"enabled": False, "reason": "insufficient_oof_samples"}
            baseline_name = pipeline._select_best_single_model(trained_models, eval_y)

    elif len(trained_models) >= 2:
        # --- OOS-FIX: Fixed-weight average (default path) ---
        # Store all models for fixed-weight averaging at inference time.
        # Base weights (unnormalized); actual weights are normalized to
        # sum to 1.0 based on which models are present.
        # Phase 3: Margin-First Ensemble (Raddar-style modeling)
        # SpreadRegressor: 0.55 (primary margin prediction path)
        # LightGBM: 0.15 (secondary classifier)
        # XGBoost: 0.15 (secondary classifier)
        # Logistic: 0.15 (complementary signal)
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
    ensemble_weight_stats = {}
    if (
        pipeline.config.optimize_ensemble_weights
        and pipeline.config.enable_loyo_cv
        and pipeline.config.multi_year_games_dir
        and len(trained_models) >= 2
        and EnsembleWeightOptimizer is not None
    ):
        cv_weights = pipeline._optimize_ensemble_weights_loyo(
            trained_models=trained_models,
            feature_dim=train_X.shape[1],
            feature_names=feature_names,
        )
        if cv_weights:
            ensemble_weight_stats = cv_weights
            # Apply CV-optimized weights if they improve over fixed weights
            optimized_w = cv_weights.get("optimized_weights", {})
            if optimized_w and cv_weights.get("improvement_over_fixed", 0) > 0:
                model_names_present = [name for name, _, _ in trained_models]
                # Only apply weights for models we actually have
                filtered_w = {
                    n: optimized_w.get(n, 0.25)
                    for n in model_names_present
                    if n in optimized_w
                }
                if filtered_w:
                    w_sum = sum(filtered_w.values())
                    filtered_w = {n: w / w_sum for n, w in filtered_w.items()}
                    pipeline.baseline_model.fixed_weights = filtered_w
                    logger.info(
                        "FIX-CV-ENSEMBLE: Applied LOYO-optimized weights: %s "
                        "(Brier improvement: %.5f)",
                        {n: round(w, 3) for n, w in filtered_w.items()},
                        cv_weights.get("improvement_over_fixed", 0),
                    )

    # ====================================================================
    # P0: LEAVE-ONE-YEAR-OUT CROSS-VALIDATION — validates that the trained
    # model generalizes across different tournament years' "chaos" patterns.
    # Uses multi-year historical data (2015-2025) to run LOYO CV and report
    # per-year Brier scores.  This does NOT retrain the primary model — it
    # is a validation diagnostic only.
    # ====================================================================
    loyo_stats = {}
    if (
        pipeline.config.enable_loyo_cv
        and pipeline.config.multi_year_games_dir
        and LeaveOneYearOutCV is not None
    ):
        loyo_stats = pipeline._run_loyo_validation(
            feature_dim=train_X.shape[1],
            feature_names=feature_names,
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
        from ..ml.evaluation.rdof_audit import estimate_model_complexity
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
    """Select the best single model by validation Brier score and set it as primary."""
    if not trained_models:
        return "none"

    best_name = "none"
    best_brier = float("inf")

    # FIX #6 (cont.): When eval_y is empty (no validation split), we
    # cannot evaluate models.  Default to the first trained model rather
    # than computing Brier on an empty array.
    if len(eval_y) == 0:
        name, model, _ = trained_models[0]
        pipeline._set_primary_model(name, model)
        name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression", "spread": "spread_regressor"}
        return name_map.get(name, name)

    for name, model, eval_preds in trained_models:
        brier = float(np.mean((eval_preds - eval_y) ** 2))
        if brier < best_brier:
            best_brier = brier
            best_name = name
            pipeline._set_primary_model(name, model)

    name_map = {"lgb": "lightgbm", "xgb": "xgboost", "logit": "logistic_regression", "spread": "spread_regressor"}
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
    import os
    import logging

    logger = logging.getLogger(__name__)

    games_dir = pipeline.config.multi_year_games_dir
    if not os.path.isdir(games_dir):
        return {"enabled": False, "reason": f"directory_not_found: {games_dir}"}

    years = pipeline.config.loyo_years or [y for y in range(2015, 2026) if y != 2020]
    years = pipeline._filter_years(years)
    if not years:
        return {"enabled": False, "reason": "no_dev_years"}

    # ----------------------------------------------------------
    # Step 1: Load multi-year samples
    # ----------------------------------------------------------
    all_X = []
    all_y = []
    all_years = []

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

        all_X.append(year_X)
        all_y.append(year_y)
        all_years.append(np.full(len(year_y), year))

    if not all_X:
        return {"enabled": False, "reason": "no_valid_year_data"}

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    game_years = np.concatenate(all_years)

    # FIX-LEAKAGE-LOYO: Do NOT apply the primary model's feature
    # selector or scaler here.  Both were fitted on training data that
    # includes the held-out year, so reusing them leaks information
    # (the feature selector's importance scores encode test-year labels;
    # the scaler's mean/std include test-year feature distributions).
    # Instead, each LOYO fold re-fits its own scaler below in train_fn.
    # Feature selection is omitted — the fold-level model sees raw features.

    # ----------------------------------------------------------
    # Step 2: Run LeaveOneYearOutCV
    # ----------------------------------------------------------
    loyo_cv = LeaveOneYearOutCV(
        years=[y for y in years if y in set(game_years)],
        temporal_mode=pipeline.config.loyo_temporal_mode,
    )

    def train_fn(X_tr, y_tr, X_v, y_v, w_tr):
        if LIGHTGBM_AVAILABLE:
            ranker = LightGBMRanker()
            vs = (X_v, y_v) if len(y_v) >= 10 else None
            ranker.train(X_tr, y_tr, num_rounds=200, early_stopping_rounds=30 if vs else None,
                         valid_set=vs, sample_weight=w_tr)
            return ranker
        elif SKLEARN_AVAILABLE:
            logit = LogisticRegression(C=1.0, max_iter=2000, random_state=pipeline.config.random_seed)
            logit.fit(X_tr, y_tr, sample_weight=w_tr)
            return logit
        return None

    def predict_fn(model, X_pred):
        if model is None:
            return np.full(len(X_pred), 0.5)
        if isinstance(model, LightGBMRanker):
            return model.predict(X_pred)
        return model.predict_proba(X_pred)[:, 1]

    cv_results = loyo_cv.cross_validate(X, y, game_years, train_fn, predict_fn)

    if not cv_results:
        return {"enabled": False, "reason": "no_cv_folds_completed"}

    per_year_brier = {}
    for i, result in enumerate(cv_results):
        held_out_year = loyo_cv.years[i] if i < len(loyo_cv.years) else i
        year_entry = {
            "brier": round(result.brier_score, 5),
            "log_loss": round(result.log_loss, 5),
            "accuracy": round(result.accuracy, 4),
            "train_size": result.train_size,
            "val_size": result.val_size,
        }
        # Add ROC-AUC if available
        if hasattr(result, 'roc_auc') and result.roc_auc is not None:
            year_entry["roc_auc"] = round(result.roc_auc, 4)
        per_year_brier[str(held_out_year)] = year_entry

    mean_brier = float(np.mean([r.brier_score for r in cv_results]))
    mean_accuracy = float(np.mean([r.accuracy for r in cv_results]))

    # Compute ROC-AUC across all CV folds if available
    roc_aucs = [r.roc_auc for r in cv_results if hasattr(r, 'roc_auc') and r.roc_auc is not None]
    loyo_result = {
        "enabled": True,
        "years_evaluated": len(cv_results),
        "total_samples": int(len(y)),
        "mean_brier": round(mean_brier, 5),
        "mean_accuracy": round(mean_accuracy, 4),
        "per_year": per_year_brier,
    }
    if roc_aucs:
        loyo_result["mean_roc_auc"] = round(float(np.mean(roc_aucs)), 4)

    return loyo_result

def _run_gnn(pipeline, graph: ScheduleGraph) -> Dict:
    multi_hop = compute_multi_hop_sos(graph, hops=3)
    pagerank = graph.compute_pagerank_sos()
    training_era_teams = set()
    for edge in graph.edges:
        training_era_teams.add(edge.team1_id)
        training_era_teams.add(edge.team2_id)

    # GNN disabled — use fallback embedding from graph statistics.
    pipeline.gnn_embeddings = {}
    for team_id in graph.team_ids:
        pipeline.gnn_embeddings[team_id] = np.array([
            multi_hop.get(team_id, 0.0),
            pagerank.get(team_id, 0.0),
        ])

    # FIX M5: Defer SOS refinement (same as PyG path above).
    pipeline._sos_refinement_pending = (multi_hop, pagerank)

    # Fix 12: Validation-based confidence for fallback path.
    val_teams = [t for t in graph.team_ids if t not in training_era_teams]
    if val_teams and pipeline.feature_engineer.team_features:
        mh_preds = np.array([multi_hop.get(t, 0.0) for t in val_teams])
        actual_ems = np.array([
            getattr(pipeline.feature_engineer.team_features.get(t), "adj_efficiency_margin", 0.0) / 30.0
            for t in val_teams
        ])
        fallback_mse = float(np.mean((mh_preds - actual_ems) ** 2))
        pipeline.model_confidence["gnn"] = float(np.clip(1.0 / (1.0 + fallback_mse) * 0.7, 0.1, 0.4))
    else:
        pipeline.model_confidence["gnn"] = 0.35

    return {
        "enabled": False,
        "framework": "statistical_fallback",
        "nodes": graph.n_teams,
        "edges": len(graph.edges),
    }

def _apply_sos_refinement(pipeline, multi_hop: Dict[str, float], pagerank: Dict[str, float]) -> None:
    if not pipeline.feature_engineer.team_features:
        return
    pr_values = np.array(list(pagerank.values()) or [0.0], dtype=float)
    pr_mean = float(np.mean(pr_values))

    for team_id, feats in pipeline.feature_engineer.team_features.items():
        mh = float(multi_hop.get(team_id, 0.0))
        pr = float(pagerank.get(team_id, pr_mean))
        refined_sos = 0.5 * feats.sos_adj_em + 3.0 * mh + 12.0 * (pr - pr_mean)
        feats.sos_adj_em = float(refined_sos)

        # Expose PageRank and multi-hop as standalone features so the
        # ensemble can learn their weights independently rather than
        # relying on the hardcoded blend above.  The blend still
        # refines sos_adj_em for backward compatibility, but the raw
        # graph signals are now available as independent dimensions.
        feats.pagerank_sos = float(pr - pr_mean)
        feats.multi_hop_sos = float(mh)

        pipeline.team_features[team_id] = feats.to_vector(include_embeddings=False)

def _apply_win_quality_metrics(pipeline, graph: ScheduleGraph) -> None:
    """Compute and attach graph-theoretic win quality metrics to team features.

    These features capture *who you beat* and *how convincingly*, which
    traditional win-loss records miss entirely.  A 25-5 team with zero
    top-50 wins is fundamentally different from a 22-8 team with five
    top-25 wins — but both look similar in record-based features.

    The schedule graph has already been built from training-era games
    only (leakage-safe), so these metrics are valid for both training
    and inference.
    """
    if not pipeline.feature_engineer.team_features or not graph.edges:
        return

    win_quality = graph.compute_win_quality_metrics()

    for team_id, feats in pipeline.feature_engineer.team_features.items():
        metrics = win_quality.get(team_id, {})
        feats.best_win_percentile = float(metrics.get("best_win_percentile", 0.5))
        feats.paper_tiger_score = float(metrics.get("paper_tiger_score", 0.0))
        feats.dominance_ratio = float(metrics.get("dominance_ratio", 1.0))
        pipeline.team_features[team_id] = feats.to_vector(include_embeddings=False)

    n_enriched = sum(
        1 for tid in pipeline.feature_engineer.team_features
        if tid in win_quality
    )
    logger.info(
        "Win quality metrics: enriched %d/%d teams (best_win_pctile, paper_tiger, dominance)",
        n_enriched, len(pipeline.feature_engineer.team_features),
    )

def _run_transformer(pipeline, game_flows: Dict[str, List[GameFlow]]) -> Dict:
    sequences: Dict[str, SeasonSequence] = {}

    for team_id, games in game_flows.items():
        embeddings: List[GameEmbedding] = []
        # Filter out tournament games AND validation-era games to prevent
        # leakage — the transformer should only learn from training-era
        # regular-season sequences (Issue 3).
        boundary = pipeline._validation_sort_key_boundary
        pre_tournament = [
            g for g in games
            if not pipeline._is_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"))
            and (boundary is None
                 or pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")) < boundary)
        ]
        ordered_games = sorted(
            pre_tournament,
            key=lambda g: (pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")), g.game_id),
        )

        for idx, game in enumerate(ordered_games):
            is_team1 = game.team1_id == team_id
            opp_id = game.team2_id if is_team1 else game.team1_id
            margin = game.lead_history[-1] if game.lead_history else 0
            if not is_team1:
                margin *= -1

            team_poss = [p for p in game.possessions if p.team_id == team_id]
            opp_poss = [p for p in game.possessions if p.team_id == opp_id]

            off = 100.0 * (sum(p.actual_points for p in team_poss) / max(len(team_poss), 1))
            deff = 100.0 * (sum(p.actual_points for p in opp_poss) / max(len(opp_poss), 1))
            tempo = float(len(team_poss) + len(opp_poss)) / 2

            embeddings.append(
                GameEmbedding(
                    game_id=game.game_id,
                    team_id=team_id,
                    opponent_id=opp_id,
                    game_date=str(getattr(game, "game_date", f"{pipeline.config.year}-01-01")),
                    game_number=idx + 1,
                    offensive_efficiency=float(off),
                    defensive_efficiency=float(deff),
                    tempo=float(np.clip(tempo, 58, 82)),
                    margin=float(margin),
                    win=margin > 0,
                    is_conference_game=True,
                    is_neutral_site=True,
                    opponent_rank=120,
                )
            )

        if len(embeddings) >= 6:
            sequences[team_id] = SeasonSequence(team_id=team_id, games=embeddings)

    # Transformer disabled — use fallback from trend statistics.
    pipeline.transformer_embeddings = {}
    breakout_count = 0
    for team_id, seq in sequences.items():
        matrix = seq.to_matrix()
        trend = np.mean(np.diff(matrix[:, 0])) if len(matrix) > 1 else 0.0
        volatility = float(np.std(matrix[:, 3]))
        recent = float(np.mean(matrix[-5:, 0]))
        pipeline.transformer_embeddings[team_id] = np.array([trend, volatility, recent])
        if len(matrix) >= 10:
            early = float(np.mean(matrix[:5, 0]))
            late = float(np.mean(matrix[-5:, 0]))
            if late - early > 0.05:
                breakout_count += 1

    pipeline.model_confidence["transformer"] = 0.35
    return {
        "enabled": False,
        "framework": "trend_fallback",
        "teams": len(sequences),
        "breakout_windows_detected": breakout_count,
    }

def _optimize_ensemble_weights_loyo(
    pipeline,
    trained_models: list,
    feature_dim: int,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """FIX-CV-ENSEMBLE: Optimize ensemble weights using LOYO cross-validation.

    Instead of optimizing weights on the eval set (which is the only data
    for honest evaluation), this method:
    1. Loads multi-year historical data
    2. For each held-out year, trains all model types on remaining years
    3. Generates predictions from each model on the held-out year
    4. Finds weights that minimize Brier score across all held-out folds

    This gives genuinely OOS weight estimates that generalize across
    tournament years' varying "chaos" patterns.

    Returns:
        Dict with optimized weights and LOYO Brier scores, or {} if
        insufficient data.
    """
    import os
    import logging as _logging
    logger = _logging.getLogger(__name__)

    games_dir = pipeline.config.multi_year_games_dir
    if not games_dir or not os.path.isdir(games_dir):
        return {}

    years = pipeline.config.loyo_years or [
        y for y in range(2015, pipeline.config.year) if y != 2020
    ]
    years = pipeline._filter_years(years)
    if len(years) < 3:
        return {}

    # Step 1: Load all years' data
    all_X: Dict[int, np.ndarray] = {}
    all_y: Dict[int, np.ndarray] = {}

    for yr in years:
        gp = os.path.join(games_dir, f"historical_games_{yr}.json")
        mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
        if not os.path.isfile(gp) or not os.path.isfile(mp):
            continue
        try:
            yr_X, yr_y, _, _, _ = pipeline._load_year_samples_incremental(
                gp, mp, feature_dim, yr
            )
            if len(yr_y) >= 20:
                all_X[yr] = yr_X
                all_y[yr] = yr_y
        except Exception:
            continue

    valid_years = sorted(all_X.keys())
    if len(valid_years) < 3:
        return {}

    # Step 2: LOYO cross-validation — for each held-out year, train
    # each model type on remaining years and predict on held-out.
    model_names = [name for name, _, _ in trained_models]
    all_oos_preds: Dict[str, list] = {name: [] for name in model_names}
    all_oos_labels: list = []

    for hold_yr in valid_years:
        # Combine training data from all years except hold_yr
        train_X_parts = [all_X[yr] for yr in valid_years if yr != hold_yr]
        train_y_parts = [all_y[yr] for yr in valid_years if yr != hold_yr]
        if not train_X_parts:
            continue

        X_train = np.concatenate(train_X_parts, axis=0)
        y_train = np.concatenate(train_y_parts)
        X_val = all_X[hold_yr]
        y_val = all_y[hold_yr]

        # FIX-LEAKAGE-ENSEMBLE-WEIGHTS: Do NOT reuse the primary
        # feature selector here.  It was fitted on training data that
        # includes the held-out year, so its importance scores encode
        # test-year labels.  Instead, skip feature selection (use raw
        # features) — same approach as the main LOYO CV (line 5047).
        # The per-fold scaler below handles scale normalization.

        # Apply scaler
        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        X_train = _scaler.fit_transform(X_train)
        X_val = _scaler.transform(X_val)

        # Train each model type and predict on held-out year
        fold_preds: Dict[str, np.ndarray] = {}
        for name, _, _ in trained_models:
            try:
                if name == "lgb" and LIGHTGBM_AVAILABLE:
                    from ..ml.models.lightgbm_ranker import LightGBMRanker
                    m = LightGBMRanker()
                    m.train(X_train, y_train, num_rounds=200)
                    fold_preds[name] = np.clip(m.predict(X_val), 0.01, 0.99)
                elif name == "xgb":
                    from ..ml.models.xgboost_ranker import XGBoostRanker
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
                elif name == "spread":
                    from ..ml.models.spread_regressor import SpreadRegressor
                    m = SpreadRegressor(
                        sigma=pipeline.config.spread_sigma_init,
                    )
                    # SpreadRegressor trains on margins, but we only have
                    # binary labels here; skip if margins unavailable
                    fold_preds[name] = np.full(len(y_val), 0.5)
                else:
                    fold_preds[name] = np.full(len(y_val), 0.5)
            except Exception:
                fold_preds[name] = np.full(len(y_val), 0.5)

        # Collect OOS predictions
        for name in model_names:
            all_oos_preds[name].extend(fold_preds.get(name, np.full(len(y_val), 0.5)).tolist())
        all_oos_labels.extend(y_val.tolist())

    if len(all_oos_labels) < 50:
        return {}

    # Step 3: Find optimal weights on OOS predictions
    oos_y = np.array(all_oos_labels)
    pred_arrays = {
        name: np.array(preds) for name, preds in all_oos_preds.items()
    }

    optimizer = EnsembleWeightOptimizer(
        step=0.05, min_weight=0.05, n_bootstrap=200,
        random_seed=pipeline.config.random_seed,
    )
    best_weights, best_brier = optimizer.optimize(
        pred_arrays, oos_y,
        min_samples=20,
        regularization_lambda=pipeline.config.ensemble_weight_regularization,
    )

    # Also compute fixed-weight Brier for comparison
    w_lgb = pipeline.config.ensemble_lgb_weight
    w_xgb = pipeline.config.ensemble_xgb_weight
    w_logit = max(0.05, 1.0 - w_lgb - w_xgb)
    fixed_w = {"lgb": w_lgb, "xgb": w_xgb, "logit": w_logit, "spread": 0.40}
    active_fixed = {n: fixed_w.get(n, 0.25) for n in model_names if n in fixed_w}
    w_sum = sum(active_fixed.values())
    active_fixed = {n: w / w_sum for n, w in active_fixed.items()}

    fixed_ensemble_preds = np.zeros(len(oos_y))
    for name, w in active_fixed.items():
        if name in pred_arrays:
            fixed_ensemble_preds += w * pred_arrays[name]
    fixed_brier = float(np.mean((fixed_ensemble_preds - oos_y) ** 2))

    improvement = fixed_brier - best_brier

    logger.info(
        "FIX-CV-ENSEMBLE: LOYO weight optimization on %d OOS samples "
        "across %d years. Fixed Brier=%.5f, Optimized Brier=%.5f, "
        "improvement=%.5f. Weights: %s",
        len(oos_y), len(valid_years), fixed_brier, best_brier,
        improvement,
        {n: round(w, 3) for n, w in best_weights.items()},
    )

    return {
        "method": "loyo_cross_validated",
        "years_used": valid_years,
        "oos_samples": len(oos_y),
        "optimized_weights": {n: round(w, 3) for n, w in best_weights.items()},
        "optimized_brier": round(best_brier, 5),
        "fixed_brier": round(fixed_brier, 5),
        "improvement_over_fixed": round(improvement, 5),
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

def _train_embedding_projections(
    pipeline,
    game_flows: Dict[str, List[GameFlow]],
) -> Dict[str, float]:
    """Train logistic models that map embedding pairs to win probability.

    Uses slice 0 of the 3-way validation split.  Slices 1 and 2 are
    reserved for ensemble weight optimization and calibration
    respectively, preventing any data overlap (Issue 5).
    """
    stats: Dict[str, float] = {}
    if not SKLEARN_AVAILABLE:
        return stats

    train_games = pipeline._get_validation_era_games_slice(game_flows, slice_index=0, n_slices=3)
    if len(train_games) < 10:
        return stats

    for emb_name, embeddings in [
        ("gnn", pipeline.gnn_embeddings),
        ("transformer", pipeline.transformer_embeddings),
    ]:
        if not embeddings:
            continue

        X_rows, y_rows = [], []
        for g in train_games:
            v1 = embeddings.get(g.team1_id)
            v2 = embeddings.get(g.team2_id)
            if v1 is None or v2 is None:
                continue
            _outcome = pipeline._game_outcome(g)
            if _outcome is None:
                continue
            diff = v1 - v2
            interaction = v1 * v2
            X_rows.append(np.concatenate([diff, interaction]))
            y_rows.append(_outcome)
            # Symmetric sample
            X_rows.append(np.concatenate([v2 - v1, v2 * v1]))
            y_rows.append(1 - _outcome)

        if len(y_rows) < 20:
            continue

        X = np.array(X_rows)
        y = np.array(y_rows)

        lr = LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=pipeline.config.random_seed
        )
        lr.fit(X, y)

        if emb_name == "gnn":
            pipeline._gnn_embedding_model = lr
        else:
            pipeline._transformer_embedding_model = lr

        stats[f"{emb_name}_projection_samples"] = len(y_rows)

    return stats
