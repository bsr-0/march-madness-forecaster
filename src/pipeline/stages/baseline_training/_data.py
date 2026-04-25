"""Baseline model training — data module."""

import logging
import os

from ....data.features.feature_engineering import (
    ABSOLUTE_LEVEL_FEATURE_NAMES,
    FeatureEngineer,
)
from ....governance.feature_manifest import build_feature_manifest
from ...config import (
    DATA_QUALITY_ERA_WEIGHTS,
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    ForecastConfig,
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


def _load_historical_years(pipeline, train_X, train_y, train_margins, train_sort_keys, X_full, n_current_year_train):
    """Build feature names and load multi-year historical training data.

    Returns:
        Tuple of (train_X, train_y, train_margins, train_sort_keys,
                  train_samples, feature_names, feature_names_full,
                  historical_training_stats).
    """
    train_samples = len(train_y)
    feature_names = None
    feature_names_full = None  # pre-selection names (91-dim) for LOYO
    if train_samples >= 40:
        from src.data.features.feature_engineering import TeamFeatures

        base_names = TeamFeatures.get_feature_names(include_embeddings=False)
        diff_names = [f"diff_{n}" for n in base_names]
        absolute_names = [f"abs_{n}" for n in ABSOLUTE_LEVEL_FEATURE_NAMES]
        interaction_names = [
            "tempo_interaction",
            "style_mismatch",
            "seed_em_residual",
            "sos_seed_interaction",
            "three_pt_var_seed_interaction",
            "seed_interaction",
            "seed_diff",
        ]
        feature_names = diff_names + absolute_names + interaction_names
        feature_names_full = list(feature_names)  # preserve pre-selection names for LOYO
        if len(feature_names) != train_X.shape[1]:
            logger.warning(
                "Feature name count mismatch: %d names vs %d columns. Falling back to generic names.",
                len(feature_names),
                train_X.shape[1],
            )
            feature_names = [f"f_{i}" for i in range(train_X.shape[1])]

    historical_training_stats = {}
    n_current_year_train = train_samples  # Track for logging

    import os

    # Resolve "auto" multi_year_games_dir via runtime state (never mutate config).
    _rs = getattr(pipeline, "_runtime_state", {})
    _resolved_games_dir = _rs.get("multi_year_games_dir", pipeline.config.multi_year_games_dir)
    if _resolved_games_dir == "auto":
        candidate = os.path.join(os.getcwd(), "data", "raw", "historical")
        if os.path.isdir(candidate):
            _resolved_games_dir = candidate
            _rs["multi_year_games_dir"] = candidate
            logger.info("Auto-detected multi-year training directory: %s", candidate)
        else:
            _resolved_games_dir = None
            _rs["multi_year_games_dir"] = None
            logger.info("No historical directory found; multi-year training disabled")

    if pipeline.config.enable_multi_year_training and _resolved_games_dir and os.path.isdir(_resolved_games_dir):
        games_dir = _resolved_games_dir
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

        # Phase 3: Apply optimal training window if computed.
        # Use the broadest (max) window across all model types to ensure
        # enough data for every model.  Individual models that prefer
        # shorter windows still benefit from year-decay weighting which
        # naturally downweights older data.
        _optimal_windows = getattr(pipeline.config, "optimal_training_windows", None)
        if _optimal_windows and hist_years:
            # Collect numeric window sizes (None = all available)
            _window_sizes = [v for v in _optimal_windows.values() if v is not None]
            if _window_sizes:
                _max_window = max(_window_sizes)
                _n_before = len(hist_years)
                hist_years = hist_years[-_max_window:]
                if len(hist_years) < _n_before:
                    logger.info(
                        "Phase 3 window optimization: trimmed historical years "
                        "from %d to %d (max optimal window=%d across %s).",
                        _n_before,
                        len(hist_years),
                        _max_window,
                        _optimal_windows,
                    )

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

        # FIX-DQ: Pre-scan current-year data to identify architecturally-
        # zero features (e.g. roster/player metrics unavailable in the
        # incremental engine).  These features are zero across ALL years
        # and should not penalize per-year data quality scores.
        _arch_zero_cols = np.all(np.abs(X_full) < 1e-8, axis=0)

        for yr in hist_years:
            gp = os.path.join(games_dir, f"historical_games_{yr}.json")
            mp = os.path.join(games_dir, f"team_metrics_{yr}.json")
            if not os.path.exists(gp) or not os.path.exists(mp):
                logger.warning(
                    "Multi-year training: missing data for %d (games=%s, metrics=%s); skipping.",
                    yr,
                    gp,
                    mp,
                )
                continue

            try:
                hX, hy, _h_margins, _end_elo, _h_rw = pipeline._load_year_samples_incremental(
                    gp,
                    mp,
                    feature_dim_full,
                    yr,
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
                    yr,
                    len(hy),
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
            # Exclude architecturally-zero columns so they don't penalize
            # quality scores for features that are always unavailable.
            _dq = compute_year_data_quality(
                hX,
                yr,
                feature_names,
                exclude_cols=_arch_zero_cols,
            )
            quality_mult = _dq["combined_weight"]

            # Log quality diagnostics for years with issues
            if _dq["zero_columns"] > 5 or _dq["completeness"] < 0.30:
                logger.warning(
                    "FIX-DQ: Year %d data quality issues — "
                    "completeness=%.2f, active_features=%d/%d, "
                    "zero_cols=%d, bad_rate=%.4f. "
                    "Adaptive weight=%.3f (era=%.2f).",
                    yr,
                    _dq["completeness"],
                    _dq["n_active_features"],
                    _dq["n_features"],
                    _dq["zero_columns"],
                    _dq["bad_rate"],
                    quality_mult,
                    _dq["era_weight"],
                )
                if _dq["zero_column_names"]:
                    logger.warning(
                        "FIX-DQ: Year %d zero columns (first 10): %s",
                        yr,
                        _dq["zero_column_names"],
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
                "Multi-year training: loaded %d samples from %d (weight=%.3f, tournament_weighted=%d).",
                len(hy),
                yr,
                year_weight,
                _n_tourney,
            )

        total_hist_samples = sum(len(part) for part in hist_y_parts)
        logger.warning(
            "Multi-year training summary: loaded %d/%d years with %d samples total.",
            len(years_loaded),
            len(hist_years),
            total_hist_samples,
        )

        if hist_X_parts:
            hist_X = np.concatenate(hist_X_parts, axis=0)
            hist_y = np.concatenate(hist_y_parts)
            hist_margins = np.concatenate(hist_margin_parts)
            hist_weights = np.concatenate(hist_weight_parts)
            hist_sort_keys = np.concatenate(hist_sortkey_parts)

            # FIX C1: Clean inf→NaN only; preserve NaN for tree-native handling
            _h_inf = int(np.isinf(hist_X).sum())
            if _h_inf > 0:
                hist_X = np.where(np.isinf(hist_X), np.nan, hist_X)

            # Prepend historical data to training set (chronologically first)
            train_X = np.concatenate([hist_X, train_X], axis=0)
            train_y = np.concatenate([hist_y, train_y])
            train_margins = np.concatenate([hist_margins, train_margins])
            train_sort_keys = np.concatenate([hist_sort_keys, train_sort_keys])

            # Store year-based weights to combine with recency weighting later.
            # Current-year samples get weight 1.0.
            pipeline._historical_year_weights = np.concatenate(
                [
                    hist_weights,
                    np.ones(n_current_year_train, dtype=float),
                ]
            )

            # FIX #3: Build round weights array for Kaggle round-weighted
            # Brier optimization.  Historical tournament games get their
            # actual round weight; regular-season games get 1.0.
            if pipeline.config.enable_round_weighted_training and hist_round_weight_parts:
                hist_rw = np.concatenate(hist_round_weight_parts)
                # Current-year training games are regular-season → weight 1.0
                pipeline._round_weights = np.concatenate(
                    [
                        hist_rw,
                        np.ones(n_current_year_train, dtype=float),
                    ]
                )
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
                "Multi-year training pool: %d historical + %d current = %d total training samples (%.1fx increase).",
                len(hist_y),
                n_current_year_train,
                train_samples,
                train_samples / max(n_current_year_train, 1),
            )
        else:
            pipeline._historical_year_weights = None
    else:
        pipeline._historical_year_weights = None

    return (
        train_X,
        train_y,
        train_margins,
        train_sort_keys,
        train_samples,
        feature_names,
        feature_names_full,
        historical_training_stats,
    )


def _apply_feature_preprocessing(
    pipeline, train_X, eval_X, train_y, X_full, feature_names, feature_names_full, train_samples, valid_samples
):
    """Apply zero-variance pruning, feature selection, scaling, and distribution shift detection.

    Returns:
        Tuple of (train_X, eval_X, X_full, feature_names, feature_names_full,
                  fs_stats, dist_shift_stats, _loyo_raw_feature_dim).
    """
    # --- Early zero-variance pruning ---
    # FIX-DQ: Remove columns that are all-zero across the entire training
    # set BEFORE feature selection and LOYO.  These are architecturally
    # zero features (roster/player metrics unavailable in incremental
    # engine) that inflate dimensionality and slow downstream steps
    # (VIF, bootstrap stability, LOYO re-fitting) without adding signal.
    _loyo_raw_feature_dim = X_full.shape[1]  # Pre-pruning dim for LOYO data loading
    _pre_fs_zero_mask = None
    if train_samples >= 40 and feature_names is not None:
        _combined_vars = np.var(train_X, axis=0)
        _zero_mask = _combined_vars < 1e-10
        _n_zero = int(np.sum(_zero_mask))
        if _n_zero > 0:
            _keep_mask = ~_zero_mask
            _dropped = [feature_names[i] for i in range(len(feature_names)) if i < len(_zero_mask) and _zero_mask[i]]
            logger.info(
                "FIX-DQ: Early zero-variance pruning removed %d/%d features before feature selection: %s",
                _n_zero,
                len(feature_names),
                _dropped[:15],
            )
            train_X = train_X[:, _keep_mask]
            eval_X = eval_X[:, _keep_mask]
            X_full = X_full[:, _keep_mask]
            feature_names = [
                feature_names[i] for i in range(len(feature_names)) if i < len(_keep_mask) and _keep_mask[i]
            ]
            # NOTE: Do NOT overwrite feature_names_full here — it must
            # preserve the original (pre-pruning) names so that LOYO can
            # load raw historical data at the full matchup dimension and
            # then apply _pre_fs_keep_mask post-hoc.
            _pre_fs_zero_mask = _keep_mask
            # Store for LOYO to apply the same pruning
            pipeline._pre_fs_keep_mask = _keep_mask

    # --- Feature selection ---
    # OOS-FIX: Default path uses a fixed domain-knowledge feature set.
    # Learned feature selection can still be enabled via config.
    # (feature_names already constructed above, before multi-year block)
    fs_stats = {}
    feature_selection_method = "all_features"

    if train_samples >= 40 and feature_names is not None:
        if not pipeline.config.enable_feature_selection:
            # OOS-FIX: Apply fixed domain-knowledge feature set.
            # No model fitting, no label dependency, no double-dipping.
            # Gap #3: Use SIMPLE_FEATURE_SET when model_complexity == "simple"
            active_feature_set = (
                SIMPLE_FEATURE_SET if pipeline.config.model_complexity == "simple" else FIXED_FEATURE_SET
            )
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            fixed_indices = [name_to_idx[n] for n in active_feature_set if n in name_to_idx]
            fixed_names = [n for n in active_feature_set if n in name_to_idx]

            min_required = 6 if pipeline.config.model_complexity == "simple" else 10
            if len(fixed_indices) >= min_required:
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
                feature_selection_method = "fixed_domain_knowledge"
                logger.info(
                    "Fixed feature selection: %d/%d features retained (domain knowledge).",
                    len(fixed_indices),
                    original_dim,
                )
            else:
                logger.warning(
                    "Fixed feature set matched only %d features (required %d) — using all features.",
                    len(fixed_indices),
                    min_required,
                )

    manifest_original_features = list(feature_names_full or feature_names or [])
    manifest_selected_features = list(feature_names or [])
    if manifest_original_features and manifest_selected_features:
        pipeline._feature_manifest = build_feature_manifest(
            target_year=pipeline.config.year,
            model_complexity=pipeline.config.model_complexity,
            selection_method=fs_stats.get("method", feature_selection_method),
            original_features=manifest_original_features,
            selected_features=manifest_selected_features,
            training_years=list(pipeline.config.training_years or []),
            dev_years=list(pipeline.config.dev_years or []),
            holdout_years=list(pipeline.config.holdout_years or []),
            metadata={
                "enable_feature_selection": bool(pipeline.config.enable_feature_selection),
                "strict_leakage_mode": bool(pipeline.config.strict_leakage_mode),
                "original_pre_selection_dim": len(manifest_original_features),
                "final_selected_dim": len(manifest_selected_features),
            },
        ).to_dict()
        fs_stats["feature_manifest_hash"] = pipeline._feature_manifest["manifest_hash"]

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
                train_X,
                eval_X,
                feature_names,
                psi_threshold=0.25,
                ks_alpha=0.05,
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
                    for r in shift_results
                    if r.flagged
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
        # FIX C1: Compute scaling stats ignoring NaN, then scale non-NaN
        # values while preserving NaN for tree-native handling.
        _has_nans = int(np.isnan(train_X).sum()) > 0
        if _has_nans:
            _means = np.nanmean(train_X, axis=0)
            _stds = np.nanstd(train_X, axis=0)
            _stds[_stds < 1e-10] = 1.0  # Avoid division by zero
            scaler = StandardScaler()
            scaler.mean_ = _means
            scaler.scale_ = _stds
            scaler.var_ = _stds**2
            scaler.n_features_in_ = train_X.shape[1]
            scaler.n_samples_seen_ = np.sum(~np.isnan(train_X), axis=0)
            # Scale non-NaN values, preserve NaN
            train_X = np.where(np.isnan(train_X), np.nan, (train_X - _means) / _stds)
        else:
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X)
        if eval_X.shape[0] > 0:
            if _has_nans and int(np.isnan(eval_X).sum()) > 0:
                eval_X = np.where(np.isnan(eval_X), np.nan, (eval_X - scaler.mean_) / scaler.scale_)
            else:
                eval_X = scaler.transform(eval_X)
        else:
            logger.warning(
                "Skipping scaler.transform(eval_X) because eval split is empty (shape=%s).",
                eval_X.shape,
            )
        pipeline.baseline_model.scaler = scaler

    # Store the pre-selection feature dimensionality for historical
    # year loading (multi-year calibration needs to reconstruct vectors
    # of the same width as the original matchup features).
    pipeline.baseline_model.feature_dim = X_full.shape[1]
    # Also store the raw (pre-zero-variance-pruning) matchup dimension
    # so calibration can load data at full width then prune post-hoc.
    pipeline.baseline_model.raw_feature_dim = _loyo_raw_feature_dim
    return (
        train_X,
        eval_X,
        X_full,
        feature_names,
        feature_names_full,
        fs_stats,
        dist_shift_stats,
        _loyo_raw_feature_dim,
    )
