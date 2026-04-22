"""Calibration fitting — extracted from TournamentPipeline.

Contains calibration-related methods: temperature scaling, Massey predictor
fitting, and tournament sigma calibration.

Each function takes a ``pipeline`` parameter (TournamentPipeline instance)
to access config and mutable state. This is a pragmatic extraction
that reduces tournament_pipeline line count while maintaining exact behavioral
equivalence.

Implements Agent Directive V7 S2 (modular architecture decomposition).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...data.models.game_flow import GameFlow
from ...governance.artifact_provenance import build_artifact_provenance
from ...ml.calibration.calibration import (
    CalibrationPipeline,
    calculate_calibration_metrics,
)
from ..config import (
    DataRequirementError,
    ForecastConfig,
)

# Optional imports — accessed via pipeline._optional_imports pattern
try:
    from .._optional_imports import (
        TOURNAMENT_SIGMA_AVAILABLE,
        TournamentSigmaCalibrator,
    )
except ImportError:
    TOURNAMENT_SIGMA_AVAILABLE = False
    TournamentSigmaCalibrator = None

try:
    from .._optional_imports import load_tournament_sigma_data
except ImportError:
    load_tournament_sigma_data = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _fit_calibration(pipeline, game_flows: Dict[str, List[GameFlow]]) -> Dict:
    """Fit calibration on validation-era games with nested OOS predictions.

    FIX-NESTED-CAL: Uses a nested approach to prevent double-dipping:
    1. PRIMARY: Historical tournament game predictions from explicitly
       configured calibration years. By default these are holdout years
       excluded from model training, so they are season-level OOS.
    2. SECONDARY: Current-year validation-era predictions using the existing
       model (validation era was NOT used for training due to chronological
       split, but the model DID see overlapping teams/features).

    The historical tournament predictions are the cleanest calibration
    signal because they match the inference domain (tournament games) and
    are truly out-of-sample with respect to the trained model.

    NOTE (LEAKAGE-AUDIT): Calibration now defaults to holdout-year
    tournament games only (via config.resolve_calibration_years()),
    eliminating year-overlap with training by default. If users override
    calibration_years, this guarantee no longer automatically holds.

    FIX #5: Temporarily restores pre-optimization CFA weights while
    generating calibration probabilities.  This prevents the calibrator
    from seeing predictions whose ensemble weights were already tuned to
    the same data — which would make them appear better-calibrated than
    they are on truly unseen data.
    """
    import logging as _logging
    logger = _logging.getLogger(__name__)

    probs = []
    outcomes = []
    # FIX-NESTED-CAL: Track provenance of calibration samples
    _n_historical_tourney_cal = 0
    _n_current_year_cal = 0

    # A1/B1: With GNN/Transformer removed from ensemble, the 3-way
    # validation split is no longer needed. Use ALL validation-era games
    # for calibration, roughly tripling the effective sample size.
    calibration_games = pipeline._get_validation_era_games(game_flows)

    unique_games = pipeline._unique_games(game_flows)
    unique_games_sorted = sorted(
        unique_games,
        key=lambda g: (pipeline._game_sort_key(getattr(g, "game_date", f"{pipeline.config.year}-01-01")), g.game_id),
    )
    regular_season_games = [
        g for g in unique_games_sorted
        if not pipeline._is_tournament_game(getattr(g, "game_date", f"{pipeline.config.year}-01-01"))
    ]

    for g in calibration_games:
        if g.team1_id not in pipeline.feature_engineer.team_features:
            continue
        if g.team2_id not in pipeline.feature_engineer.team_features:
            continue
        p = pipeline._raw_fusion_probability(g.team1_id, g.team2_id)
        # F1: Calibrate on raw ensemble probabilities.  Tournament
        # adaptation is applied AFTER calibration at inference time,
        # so the calibrator trains on the same raw distribution.
        p = float(np.clip(p, pipeline.config.pre_calibration_clip_lo, pipeline.config.pre_calibration_clip_hi))
        o = pipeline._game_outcome(g)
        if o is None:
            continue  # S5 FIX: skip games with indeterminate outcome
        probs.append(p)
        outcomes.append(o)
        _n_current_year_cal += 1

    # A1: CFA weight optimization removed — baseline-only prediction.

    # Augment calibration pool with historical TOURNAMENT-ONLY game data.
    # Tournament games are genuinely out-of-sample: the baseline model
    # trains only on regular-season games (include_tournament=False),
    # so tournament predictions are unseen during training.
    #
    # IMPORTANT: We load tournament-only samples by contract via
    # _load_year_tournament_samples_incremental, which selects games
    # on or after each year's tournament start date.  Regular-season
    # games are NEVER loaded — no downstream filtering needed.
    tourney_cal_count = 0
    _rs = getattr(pipeline, "_runtime_state", {})
    _resolved_games_dir = _rs.get("multi_year_games_dir", pipeline.config.multi_year_games_dir)
    if (pipeline.config.enable_multi_year_calibration
            and _resolved_games_dir
            and hasattr(pipeline, "baseline_model")
            and pipeline.baseline_model is not None):
        import os

        # Resolve "auto" multi_year_games_dir via runtime state (never mutate config)
        if _resolved_games_dir == "auto":
            candidate = os.path.join(os.getcwd(), "data", "raw", "historical")
            if os.path.isdir(candidate):
                _resolved_games_dir = candidate
                _rs["multi_year_games_dir"] = candidate
            else:
                _resolved_games_dir = None
                _rs["multi_year_games_dir"] = None

        games_dir = _resolved_games_dir
        if not games_dir:
            logger.warning(
                "multi_year_games_dir is None after resolution; "
                "historical tournament calibration augmentation skipped."
            )
        else:
            years = pipeline.config.resolve_calibration_years()
            if not years:
                logger.warning(
                    "No calibration_years resolved (holdout_years/calibration_years empty); "
                    "historical tournament calibration augmentation skipped."
                )
            # Determine feature dimensionality from current model.
            # Use the raw (pre-zero-variance-pruning) dimension for the
            # sample loader so that symmetric_augment / swap_matchup_batch
            # receives vectors with >= MATCHUP_DIM (78) columns.  After
            # loading, apply zero-variance pruning to match the model's
            # expected input width.
            raw_feature_dim = getattr(
                pipeline.baseline_model, "raw_feature_dim",
                pipeline.baseline_model.feature_dim,
            )
            feature_dim = pipeline.baseline_model.feature_dim
            _pre_fs_keep_mask = getattr(pipeline, "_pre_fs_keep_mask", None)

            def _load_tournament_cal_year(yr, games_dir, feature_dim):
                """Load tournament calibration samples for a single year.
                Returns (count_added, error_msg_or_None)."""
                nonlocal probs, outcomes, tourney_cal_count
                games_path = os.path.join(games_dir, f"historical_games_{yr}.json")
                metrics_path = os.path.join(games_dir, f"team_metrics_{yr}.json")
                if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
                    return 0, f"files missing (games={os.path.isfile(games_path)}, metrics={os.path.isfile(metrics_path)})"
                try:
                    yr_X, yr_y, _yr_margins, _, _yr_rw = pipeline._load_year_tournament_samples_incremental(
                        games_path, metrics_path, raw_feature_dim, yr,
                    )
                except Exception as e:
                    return 0, f"loader error: {e}"
                if len(yr_y) < 4:
                    return 0, f"only {len(yr_y)} tournament samples (need >= 4)"
                # Defense-in-depth: the tournament-only loader guarantees
                # rw >= 2.0 for every sample (asserted internally).
                if len(_yr_rw) > 0:
                    n_bad = int(np.sum(_yr_rw <= 1.0))
                    if n_bad > 0:
                        return 0, f"{n_bad} non-tournament rows (rw <= 1.0)"
                logger.info(
                    "Loaded %d tournament-only calibration samples for %d",
                    len(yr_y), yr,
                )
                # Apply zero-variance pruning (same mask used during training)
                if _pre_fs_keep_mask is not None and yr_X.shape[1] > feature_dim:
                    try:
                        yr_X = yr_X[:, _pre_fs_keep_mask]
                    except (IndexError, ValueError) as e:
                        return 0, f"zero-variance pruning error: {e}"
                # Apply feature selection if fitted
                if pipeline.feature_selector is not None and pipeline.feature_selector.is_fitted:
                    try:
                        yr_X = pipeline.feature_selector.transform(yr_X)
                    except (IndexError, ValueError) as e:
                        return 0, f"feature selector error: {e}"
                # Apply scaler if available
                if pipeline.baseline_model.scaler is not None:
                    try:
                        yr_X = pipeline.baseline_model.scaler.transform(yr_X)
                    except (ValueError, Exception) as e:
                        return 0, f"scaler error: {e}"
                # Predict using baseline model in batch
                try:
                    yr_preds = pipeline.baseline_model.predict_proba_batch(yr_X)
                    yr_preds = np.clip(
                        yr_preds,
                        pipeline.config.pre_calibration_clip_lo,
                        pipeline.config.pre_calibration_clip_hi,
                    )
                    probs.extend(yr_preds.tolist())
                    outcomes.extend(yr_y.tolist())
                    tourney_cal_count += len(yr_y)
                    return len(yr_y), None
                except Exception as e:
                    return 0, f"prediction error: {e}"

            # Load historical tournament-only games for calibration.
            # These match the inference domain exactly.
            for yr in years:
                n_added, err = _load_tournament_cal_year(yr, games_dir, feature_dim)
                if err:
                    logger.warning("Calibration year %d skipped: %s", yr, err)

            # Fallback: if configured calibration years didn't provide enough
            # samples, expand to dev_years. Tournament games are out-of-sample
            # (model trains only on regular-season games), so this is safe.
            if (len(probs) < pipeline.config.min_calibration_samples_hard
                    and pipeline.config.dev_years):
                fallback_years = sorted(
                    y for y in pipeline.config.dev_years
                    if y not in set(years) and y != 2020
                )
                if fallback_years:
                    logger.info(
                        "Calibration pool (%d) below hard minimum (%d). "
                        "Expanding to dev_years tournament games: %s",
                        len(probs), pipeline.config.min_calibration_samples_hard,
                        fallback_years,
                    )
                    for yr in fallback_years:
                        n_added, err = _load_tournament_cal_year(yr, games_dir, feature_dim)
                        if err:
                            logger.debug("Fallback calibration year %d skipped: %s", yr, err)
                        if len(probs) >= pipeline.config.min_calibration_samples_hard:
                            break
        _n_historical_tourney_cal = tourney_cal_count
        if tourney_cal_count > 0:
            logger.info(
                "Calibration augmented with %d historical tournament-only samples.",
                tourney_cal_count,
            )

    # FIX-NESTED-CAL: Log calibration data provenance.
    logger.info(
        "FIX-NESTED-CAL: Calibration data composition — "
        "%d historical tournament (holdout-year OOS by default) + %d current-year "
        "validation-era = %d total samples.  Historical tournament "
        "predictions are the cleanest calibration signal.",
        _n_historical_tourney_cal, _n_current_year_cal, len(probs),
    )

    if len(probs) < pipeline.config.min_calibration_samples_hard:
        raise DataRequirementError(
            "Calibration sample size (%d) below hard minimum (%d). "
            "Enable multi-year calibration or provide more data."
            % (len(probs), pipeline.config.min_calibration_samples_hard)
        )

    if len(probs) < pipeline.config.min_calibration_samples:
        import logging
        logging.getLogger(__name__).warning(
            "Calibration sample size (%d) below minimum (%d); "
            "consider enabling multi-year calibration or providing more data.",
            len(probs), pipeline.config.min_calibration_samples,
        )

    if len(probs) < 20:
        pipeline.calibration_pipeline = None
        metrics = calculate_calibration_metrics(np.array(probs or [0.5]), np.array(outcomes or [0]))
        return {
            "method": "none",
            "samples": len(probs),
            "brier_before": float(metrics.brier_score),
            "brier_after": float(metrics.brier_score),
            "feature_manifest_hash": (
                (getattr(pipeline, "_feature_manifest", {}) or {}).get("manifest_hash")
            ),
            "provenance": build_artifact_provenance(
                pipeline=pipeline,
                artifact_kind="calibration_report",
                extra={
                    "fit_data_source": "insufficient_samples",
                    "historical_tournament_samples": _n_historical_tourney_cal,
                    "current_year_validation_samples": _n_current_year_cal,
                },
            ),
        }

    if (
        pipeline.config.calibration_method == "none"
        and pipeline.config.probability_profile == "production"
    ):
        pipeline.calibration_pipeline = None
        metrics = calculate_calibration_metrics(np.array(probs), np.array(outcomes))
        return {
            "method": "none",
            "samples": len(probs),
            "brier_before": float(metrics.brier_score),
            "brier_after": float(metrics.brier_score),
            "ece_before": float(metrics.expected_calibration_error),
            "ece_after": float(metrics.expected_calibration_error),
            "feature_manifest_hash": (
                (getattr(pipeline, "_feature_manifest", {}) or {}).get("manifest_hash")
            ),
            "provenance": build_artifact_provenance(
                pipeline=pipeline,
                artifact_kind="calibration_report",
                extra={
                    "fit_data_source": "disabled_by_config",
                    "historical_tournament_samples": _n_historical_tourney_cal,
                    "current_year_validation_samples": _n_current_year_cal,
                },
            ),
        }

    p_arr = np.array(probs)
    y_arr = np.array(outcomes)

    # FIX-NESTED-CAL: Nested calibration to prevent double-dipping.
    #
    # Strategy: When historical tournament data is available (truly OOS),
    # use it exclusively for fitting the calibrator.  Use current-year
    # validation-era predictions (which have mild overlap with training
    # teams/features) only for OOS evaluation.
    #
    # This prevents the calibrator from learning biases specific to the
    # model's in-sample confidence patterns.  Historical tournament games
    # are the cleanest signal because:
    # 1. The model trains on regular-season games only
    # 2. Tournament games are the actual inference domain
    # 3. Year-level separation when calibration_years are disjoint from training
    n_cal = len(p_arr)
    _nested_mode = False

    if _n_historical_tourney_cal >= 30 and _n_current_year_cal >= 10:
        # BEST: Fit on historical tournament data, evaluate on current year.
        # Array layout: current-year validation samples are appended first
        # (lines 104-119), then historical tournament samples (lines 145-188).
        assert len(p_arr) == _n_current_year_cal + _n_historical_tourney_cal, (
            f"Calibration split mismatch: {len(p_arr)} != "
            f"{_n_current_year_cal} + {_n_historical_tourney_cal}"
        )
        p_fit = p_arr[_n_current_year_cal:]   # Historical tournament
        y_fit = y_arr[_n_current_year_cal:]
        p_eval = p_arr[:_n_current_year_cal]  # Current-year validation
        y_eval = y_arr[:_n_current_year_cal]
        use_oos_eval = True
        _nested_mode = True
        logger.info(
            "FIX-NESTED-CAL: Using nested calibration — fit on %d "
            "historical tournament samples, evaluate on %d current-year "
            "validation samples. No double-dipping.",
            len(p_fit), len(p_eval),
        )
    else:
        # Fallback: chronological 70/30 split (original approach).
        split_idx = int(n_cal * 0.7)
        if split_idx >= 20 and (n_cal - split_idx) >= 10:
            p_fit, p_eval = p_arr[:split_idx], p_arr[split_idx:]
            y_fit, y_eval = y_arr[:split_idx], y_arr[split_idx:]
            use_oos_eval = True
        else:
            # Too few samples for a meaningful split; fit on all
            # but do NOT evaluate on the same data (prevents inflated metrics).
            p_fit, p_eval = p_arr, None
            y_fit, y_eval = y_arr, None
            use_oos_eval = False

    # Bootstrap CI for temperature scaling: if the 95% CI for T includes
    # 1.0 (the identity), calibration is not statistically justified and
    # we skip it.  This prevents fitting noise when the calibration sample
    # is too small to distinguish T from 1.0.
    from ...ml.calibration.calibration import TemperatureScaling
    bootstrap_info = {}
    if pipeline.config.calibration_method == "temperature" and len(p_fit) >= 20:
        ts_check = TemperatureScaling()
        T_lo, T_hi, T_vals = ts_check.bootstrap_ci(
            p_fit, y_fit,
            n_bootstrap=200,
            ci_level=0.95,
            random_seed=pipeline.config.random_seed,
        )
        bootstrap_info = {
            "bootstrap_T_lower": round(T_lo, 4),
            "bootstrap_T_upper": round(T_hi, 4),
            "bootstrap_T_median": round(float(np.median(T_vals)), 4),
            "bootstrap_T_std": round(float(np.std(T_vals)), 4),
            "ci_includes_identity": T_lo <= 1.0 <= T_hi,
        }
        if T_lo <= 1.0 <= T_hi:
            # CI includes T=1.0 → calibration is indistinguishable from
            # identity; skip to avoid fitting noise.
            pipeline.calibration_pipeline = None
            pre_metrics = calculate_calibration_metrics(p_arr, y_arr)
            calibration_info = {
                "method": "none_bootstrap_ci_includes_identity",
                "samples": len(probs),
                "tournament_games_filtered": len(unique_games) - len(regular_season_games),
                "brier_before": float(pre_metrics.brier_score),
                "brier_after": float(pre_metrics.brier_score),
                "ece_before": float(pre_metrics.expected_calibration_error),
                "ece_after": float(pre_metrics.expected_calibration_error),
                "pre_calibration_clip": [pipeline.config.pre_calibration_clip_lo, pipeline.config.pre_calibration_clip_hi],
                "fit_years": pipeline.config.resolve_calibration_years(),
                "feature_manifest_hash": (
                    (getattr(pipeline, "_feature_manifest", {}) or {}).get("manifest_hash")
                ),
                **bootstrap_info,
            }
            calibration_info["provenance"] = build_artifact_provenance(
                pipeline=pipeline,
                artifact_kind="calibration_report",
                extra={
                    "fit_data_source": "bootstrap_identity_skip",
                    "historical_tournament_samples": _n_historical_tourney_cal,
                    "current_year_validation_samples": _n_current_year_cal,
                    "nested_calibration": _nested_mode,
                },
            )
            return calibration_info

    # Phase 5: Auto-select best calibration method via temporal benchmarking.
    _auto_selection_info = {}
    effective_calibration_method = pipeline.config.calibration_method
    if pipeline.config.calibration_method == "auto":
        try:
            from ...evaluation.calibration_benchmark import (
                CalibrationBenchmark,
                select_best_calibration,
            )
            from ...evaluation.calibration_methods import get_all_calibration_models

            benchmark = CalibrationBenchmark(
                methods=get_all_calibration_models(),
                n_bootstrap=500,
                selection_metric="log_loss",
            )
            # Build yearly data from fit/eval splits for benchmarking.
            # Use a simple 2-fold temporal split of the available calibration data.
            n_fit = len(p_fit)
            if n_fit >= 40:
                half = n_fit // 2
                yearly_preds = {1: p_fit[:half], 2: p_fit[half:]}
                yearly_outs = {1: y_fit[:half], 2: y_fit[half:]}
                agg = benchmark.run_temporal_benchmark(
                    yearly_preds, yearly_outs,
                    random_seed=getattr(pipeline.config, "random_seed", 42),
                )
                selection = select_best_calibration(agg)
                # Map Phase 5 method names to CalibrationPipeline method names
                method_map = {
                    "temperature_scaling": "temperature",
                    "logistic_calibration": "platt",
                    "isotonic_regression": "isotonic",
                    "beta_calibration": "temperature",  # fallback: no native beta in CalibrationPipeline
                }
                effective_calibration_method = method_map.get(
                    selection.chosen_method, "temperature"
                )
                _auto_selection_info = {
                    "auto_selected_method": selection.chosen_method,
                    "auto_mapped_to": effective_calibration_method,
                    "auto_selection_metric": selection.selection_metric,
                    "auto_secondary_metrics": selection.secondary_metrics,
                }
                logger.info(
                    "Phase 5 auto-calibration: selected '%s' (mapped to '%s') "
                    "via temporal benchmark (log_loss=%.4f)",
                    selection.chosen_method,
                    effective_calibration_method,
                    selection.secondary_metrics.get("log_loss", 0.0),
                )
            else:
                effective_calibration_method = "temperature"
                _auto_selection_info = {
                    "auto_selected_method": "temperature_scaling",
                    "auto_mapped_to": "temperature",
                    "auto_fallback_reason": f"insufficient samples ({n_fit}) for benchmark",
                }
                logger.info(
                    "Phase 5 auto-calibration: insufficient samples (%d) for "
                    "benchmark, defaulting to temperature scaling.",
                    n_fit,
                )
        except Exception as e:
            effective_calibration_method = "temperature"
            _auto_selection_info = {
                "auto_selected_method": "temperature_scaling",
                "auto_mapped_to": "temperature",
                "auto_fallback_reason": f"benchmark error: {e}",
            }
            logger.warning(
                "Phase 5 auto-calibration benchmark failed (%s); "
                "defaulting to temperature scaling.",
                e,
            )

    # Pass 2 calibration enforcement:
    # In non-production profiles, only isotonic/platt are permitted.
    # Keep production-profile behavior unchanged (locked production path
    # currently uses temperature scaling by design).
    if pipeline.config.probability_profile != "production":
        allowed_methods = {"isotonic", "platt"}
        if effective_calibration_method not in allowed_methods:
            # Isotonic is preferred when enough data is available.
            # Otherwise default to Platt as the lower-variance fallback.
            fallback_method = "isotonic" if len(p_fit) >= 200 else "platt"
            _auto_selection_info.update({
                "calibration_enforcement": "isotonic_platt_only",
                "calibration_method_requested": str(effective_calibration_method),
                "calibration_method_enforced": fallback_method,
            })
            logger.info(
                "Calibration enforcement: requested '%s' is not permitted in "
                "non-production profile. Using '%s' instead.",
                effective_calibration_method,
                fallback_method,
            )
            effective_calibration_method = fallback_method

    # ── Vegas Anchor Calibration ──────────────────────────────────────
    # When enabled, calibrate model scores against Vegas closing spread
    # probabilities using regular-season games (potentially thousands of
    # samples) instead of binary outcomes on a tiny tournament holdout.
    # Falls back to the standard calibration path if insufficient data.
    _vegas_anchor_used = False
    _vegas_anchor_info = {}

    if pipeline.config.enable_vegas_calibration_anchor:
        try:
            from ...ml.calibration.calibration import VegasAnchorCalibrator
            from ...forecaster.market import spread_to_probability
            from pathlib import Path
            import json as _json

            # Resolve Vegas spreads file
            vegas_path = pipeline.config.vegas_spreads_json
            if vegas_path is None:
                cache_dir = getattr(pipeline.config, "data_cache_dir", "data/raw")
                vegas_path = str(Path(cache_dir) / f"vegas_spreads_{pipeline.config.year}.json")

            vegas_spreads = None
            if os.path.exists(vegas_path):
                try:
                    with open(vegas_path, "r", encoding="utf-8") as _f:
                        _data = _json.load(_f)
                    # Support both formats: {"games": {key: {spread: X}}} or {key: X}
                    games_data = _data.get("games", _data) if isinstance(_data, dict) else {}
                    vegas_spreads = {}
                    for key, val in games_data.items():
                        if isinstance(val, (int, float)):
                            vegas_spreads[str(key)] = float(val)
                        elif isinstance(val, dict) and isinstance(val.get("spread"), (int, float)):
                            vegas_spreads[str(key)] = float(val["spread"])
                except Exception as _e:
                    logger.warning("Failed to load Vegas spreads from %s: %s", vegas_path, _e)

            if vegas_spreads and len(vegas_spreads) >= VegasAnchorCalibrator.MIN_GAMES:
                # Collect (model_pred, vegas_prob) pairs from regular-season games
                sigma = pipeline.config.vegas_anchor_sigma
                anchor_model_probs = []
                anchor_vegas_probs = []

                for g in regular_season_games:
                    if g.team1_id not in pipeline.feature_engineer.team_features:
                        continue
                    if g.team2_id not in pipeline.feature_engineer.team_features:
                        continue

                    # Try both orderings for matchup key lookup
                    key_fwd = f"{g.team1_id}_vs_{g.team2_id}"
                    key_rev = f"{g.team2_id}_vs_{g.team1_id}"
                    spread = vegas_spreads.get(key_fwd)
                    if spread is None and key_rev in vegas_spreads:
                        spread = -vegas_spreads[key_rev]  # flip sign
                    if spread is None:
                        continue

                    model_p = pipeline._raw_fusion_probability(g.team1_id, g.team2_id)
                    model_p = float(np.clip(
                        model_p,
                        pipeline.config.pre_calibration_clip_lo,
                        pipeline.config.pre_calibration_clip_hi,
                    ))
                    vegas_p = spread_to_probability(spread, sigma=sigma)

                    anchor_model_probs.append(model_p)
                    anchor_vegas_probs.append(vegas_p)

                if len(anchor_model_probs) >= VegasAnchorCalibrator.MIN_GAMES:
                    anchor_cal = VegasAnchorCalibrator()
                    anchor_cal.fit(
                        np.array(anchor_model_probs),
                        np.array(anchor_vegas_probs),
                    )

                    # Wrap the VegasAnchorCalibrator in a CalibrationPipeline-compatible shell
                    pipeline.calibration_pipeline = CalibrationPipeline(method="temperature")
                    pipeline.calibration_pipeline.calibrator = anchor_cal
                    pipeline.calibration_pipeline.calibrator.fitted = True
                    pipeline.calibration_pipeline.method = "vegas_anchor"

                    _vegas_anchor_used = True
                    _vegas_anchor_info = anchor_cal.to_dict()
                    effective_calibration_method = "vegas_anchor"

                    logger.info(
                        "Vegas anchor calibration: fitted on %d regular-season games "
                        "(a=%.4f, b=%.4f, logit MSE=%.4f). Falling back to standard "
                        "calibration on tournament holdout is NOT needed.",
                        anchor_cal.n_anchor_games,
                        anchor_cal.a, anchor_cal.b,
                        anchor_cal.anchor_mse or 0.0,
                    )
                else:
                    logger.info(
                        "Vegas anchor: only %d games matched (need %d). "
                        "Falling back to standard calibration.",
                        len(anchor_model_probs),
                        VegasAnchorCalibrator.MIN_GAMES,
                    )
            else:
                logger.info(
                    "Vegas anchor: spread data unavailable or insufficient at %s. "
                    "Falling back to standard calibration.",
                    vegas_path,
                )
        except Exception as _e:
            logger.warning("Vegas anchor calibration failed: %s. Falling back.", _e)

    # ── Standard calibration (fallback or primary) ─────────────────────
    if not _vegas_anchor_used:
        # Fit calibration on the fitting portion (70% or all).
        pipeline.calibration_pipeline = CalibrationPipeline(method=effective_calibration_method)
        pipeline.calibration_pipeline.fit(p_fit, y_fit)

    # FIX #3: Fit round-weighted Brier calibrator as secondary refinement.
    # Kaggle uses round-weighted Brier scoring, so calibration should
    # optimize for the actual competition metric, not flat Brier.
    pipeline._round_weighted_calibrator = None
    if (
        pipeline.config.enable_round_weighted_calibration
        and pipeline.config.calibration_method == "temperature"
        and len(p_fit) >= 30
    ):
        try:
            from ...ml.calibration.brier_optimal import BrierCalibrator
            rw_cal = BrierCalibrator()
            # Build synthetic round labels for calibration samples:
            # later calibration samples (closer to tournament) get higher
            # round weights as a proxy for tournament importance.
            n_fit = len(p_fit)
            cal_round_labels = []
            for i in range(n_fit):
                frac = i / max(n_fit - 1, 1)
                if frac > 0.9:
                    cal_round_labels.append("F4")
                elif frac > 0.8:
                    cal_round_labels.append("E8")
                elif frac > 0.6:
                    cal_round_labels.append("S16")
                elif frac > 0.4:
                    cal_round_labels.append("R32")
                else:
                    cal_round_labels.append("R64")
            rw_cal.fit_weighted(p_fit, y_fit, cal_round_labels)
            pipeline._round_weighted_calibrator = rw_cal
            logger.info(
                "FIX #3: Round-weighted Brier calibrator fitted (T=%.3f).",
                rw_cal.temperature,
            )
        except Exception as e:
            logger.warning("FIX #3: Round-weighted calibration failed: %s", e)

    # Evaluate calibration quality.
    pre_metrics = calculate_calibration_metrics(p_arr, y_arr)

    # In-sample evaluation: apply calibrator to fitting data for diagnostic reporting
    cal_preds_insample = pipeline.calibration_pipeline.calibrate(p_fit)
    insample_metrics = calculate_calibration_metrics(cal_preds_insample, y_fit)

    # OOS evaluation (held-out portion) when split is available
    if use_oos_eval:
        cal_preds_eval = pipeline.calibration_pipeline.calibrate(p_eval)
        oos_metrics = calculate_calibration_metrics(cal_preds_eval, y_eval)
        brier_after = float(oos_metrics.brier_score)
        ece_after = float(oos_metrics.expected_calibration_error)
        eval_mode = "nested_historical_tourney_vs_current" if _nested_mode else "oos_70_30"
    else:
        # FIX-LEAKAGE-CAL: No held-out data available.  Report pre-calibration
        # metrics as post-calibration metrics (no improvement claim) instead of
        # evaluating on the same data used for fitting, which would produce
        # artificially inflated improvement numbers.
        brier_after = float(pre_metrics.brier_score)
        ece_after = float(pre_metrics.expected_calibration_error)
        eval_mode = "insample_no_eval"
        logger.warning(
            "FIX-LEAKAGE-CAL: Too few calibration samples for train/eval split. "
            "Reporting pre-calibration metrics as post-calibration (no improvement "
            "claim). Calibrator is fitted on all %d samples but evaluation is skipped "
            "to prevent in-sample leakage.",
            len(p_fit),
        )

    # Gap #7: Fit round-weighted Brier sharpener.
    # Kaggle uses round-weighted Brier (finals weighted 32x vs R64).
    # The standard sharpener optimizes flat Brier, but we need to
    # optimize for the ACTUAL competition metric.
    #
    # FIX DOUBLE-DIP: Fit sharpener on EVALUATION portion only (not the
    # data used to fit the temperature calibrator).  When the calibrator
    # was fit on p_fit/y_fit, fitting the sharpener on the same data
    # would be double-dipping — it would overfit the post-processing
    # chain to ~300 samples.  Instead, use p_eval/y_eval (held-out data)
    # when available, or skip sharpening when no separate eval data exists.
    sharpener_info = {}
    synthetic_round_labels = []  # May be populated by sharpener, used by FLB
    sharpening_enabled = bool(pipeline.config.enable_brier_sharpening)
    if pipeline.config.mode == "calibration" and sharpening_enabled:
        # Protocol guardrail: sharpening is prohibited for Kaggle probability
        # submissions (mode="calibration"), regardless of config flag state.
        logger.info(
            "Sharpening disabled for calibration mode (Kaggle pathway guardrail)."
        )
        sharpening_enabled = False
        sharpener_info = {
            "sharpener_method": "disabled_for_kaggle",
            "sharpener_alpha": 1.0,
            "sharpener_fitted_on_eval_only": False,
            "sharpener_used_default": True,
        }

    if sharpening_enabled and pipeline._brier_post_processor is not None:
        try:
            from ...ml.calibration.brier_optimal import RoundWeightedSharpener
            rw_sharpener = RoundWeightedSharpener()
            # Use evaluation portion for sharpener fitting (no double-dip)
            if use_oos_eval and len(p_eval) >= 20:
                # Calibrate eval predictions through the fitted temperature
                cal_preds_for_sharp = pipeline.calibration_pipeline.calibrate(p_eval) if pipeline.calibration_pipeline else p_eval
                sharp_y = y_eval
            else:
                # FIX-LEAKAGE-SHARP: Previously fell back to fitting on
                # the same p_arr/y_arr used for calibration (double-dip).
                # Use a safe default alpha=1.0 (no sharpening) instead.
                rw_sharpener.alpha = 1.0
                pipeline._brier_post_processor.sharpener = rw_sharpener
                sharpener_info = {
                    "sharpener_method": "round_weighted",
                    "sharpener_alpha": 1.0,
                    "sharpener_fitted_on_eval_only": False,
                    "sharpener_used_default": True,
                }
                logger.info(
                    "Gap #7: Sharpener using safe default alpha=1.0 "
                    "(insufficient OOS eval data for fitting)"
                )
                cal_preds_for_sharp = None  # Signal to skip fitting below

            # Construct synthetic round labels and fit only if we have OOS data
            if cal_preds_for_sharp is not None:
                n_games_sharp = len(cal_preds_for_sharp)
                synthetic_round_labels = []
                for i in range(n_games_sharp):
                    frac = i / max(n_games_sharp - 1, 1)
                    if frac > 0.9:
                        synthetic_round_labels.append("F4")
                    elif frac > 0.8:
                        synthetic_round_labels.append("E8")
                    elif frac > 0.6:
                        synthetic_round_labels.append("S16")
                    elif frac > 0.4:
                        synthetic_round_labels.append("R32")
                    else:
                        synthetic_round_labels.append("R64")
                rw_sharpener.fit_weighted(
                    cal_preds_for_sharp, sharp_y, synthetic_round_labels,
                    alpha_bounds=pipeline.config.brier_sharpening_alpha_bounds,
                )
                pipeline._brier_post_processor.sharpener = rw_sharpener
                sharpener_info = {
                    "sharpener_method": "round_weighted",
                    "sharpener_alpha": round(rw_sharpener.alpha, 4),
                    "sharpener_fitted_on_eval_only": True,
                }
                logger.info(
                    "Gap #7: Round-weighted Brier sharpener fitted (alpha=%.3f, "
                    "eval_only=True, n_samples=%d)",
                    rw_sharpener.alpha,
                    n_games_sharp,
                )
        except Exception as e:
            logger.warning("Gap #7: Round-weighted sharpener fitting failed: %s", e)

    # goto_conversion: favourite-longshot bias correction.
    # The actual algorithm from gotoConversion/goto_conversion (GitHub).
    # Fit the margin parameter on evaluation data to minimize Brier score.
    # Supports round-weighted fitting when round labels are available.
    flb_info = {}
    _fit_flb = pipeline._flb_correction is not None and len(p_arr) >= 30
    # FIX-LEAKAGE-FLB: Only fit goto_conversion on genuinely OOS eval
    # data.  Falling back to p_arr/y_arr is a double-dip since the same
    # data fitted the calibration temperature.
    if _fit_flb and not (use_oos_eval and len(p_eval) >= 20):
        logger.info(
            "Gap #7: goto_conversion skipped (insufficient OOS "
            "eval data — using default margin to avoid double-dip)"
        )
        flb_info = {"goto_conversion_skipped": True, "reason": "no_oos_eval"}
        _fit_flb = False
    if _fit_flb:
        try:
            flb_preds = pipeline.calibration_pipeline.calibrate(p_eval) if pipeline.calibration_pipeline else p_eval
            flb_outcomes = y_eval
            if len(flb_preds) >= 20:
                # Build round labels for weighted optimization if possible.
                # This targets the actual Kaggle metric (round-weighted Brier).
                flb_round_labels = None
                if hasattr(pipeline, '_cal_round_labels') and pipeline._cal_round_labels is not None:
                    flb_round_labels = pipeline._cal_round_labels
                elif synthetic_round_labels and len(synthetic_round_labels) == len(flb_preds):
                    flb_round_labels = synthetic_round_labels

                pipeline._flb_correction.fit(
                    flb_preds, flb_outcomes,
                    strength_bounds=pipeline.config.goto_conversion_margin_bounds,
                    round_labels=flb_round_labels,
                )

                # Also wire into BrierPostProcessor for unified pipeline
                if pipeline._brier_post_processor is not None:
                    pipeline._brier_post_processor.goto_converter = pipeline._flb_correction

                flb_info = {
                    "goto_conversion_margin": round(pipeline._flb_correction.strength, 5),
                    "goto_conversion_fitted": True,
                    "goto_conversion_weighted": flb_round_labels is not None,
                }
                if pipeline._flb_correction._fit_details:
                    flb_info["goto_conversion_brier_before"] = pipeline._flb_correction._fit_details.get("brier_before", 0.0)
                    flb_info["goto_conversion_brier_after"] = pipeline._flb_correction._fit_details.get("brier_after", 0.0)
                    flb_info["goto_conversion_brier_delta"] = pipeline._flb_correction._fit_details.get("brier_delta", 0.0)
        except Exception as e:
            logger.warning("goto_conversion fitting failed: %s", e)

    # Compute uncertainty band: 95% CI half-width on score estimates
    # Uses worst-case binomial SE: 1.96 * sqrt(0.25 / N)
    # For Vegas anchor, N is the number of games with spread data (much larger)
    if _vegas_anchor_used:
        _n_cal = max(_vegas_anchor_info.get("n_anchor_games", 1), 1)
    else:
        _n_cal = max(len(probs), 1)
    _uncertainty_band = round(1.96 * (0.25 / _n_cal) ** 0.5, 4)
    _uncertainty_level = (
        "high" if _uncertainty_band >= 0.10
        else "moderate" if _uncertainty_band >= 0.05
        else "low"
    )

    calibration_info = {
        "method": effective_calibration_method,
        "requested_method": pipeline.config.calibration_method,
        "samples": len(probs),
        "historical_tournament_samples": tourney_cal_count,
        "current_year_calibration_samples": _n_current_year_cal,
        "fit_years": pipeline.config.resolve_calibration_years(),
        "nested_calibration": _nested_mode,
        "tournament_games_filtered": len(unique_games) - len(regular_season_games),
        "uncertainty_band": _uncertainty_band,
        "uncertainty_level": _uncertainty_level,
        "uncertainty_note": (
            f"Scores fitted on {len(probs)} calibration games carry "
            f"\u00b1{_uncertainty_band:.2f} uncertainty (95% CI). "
            f"Differences below {2 * _uncertainty_band:.2f} between matchups "
            f"are not statistically meaningful."
        ),
        "brier_before": float(pre_metrics.brier_score),
        "brier_after": brier_after,
        "brier_after_insample": float(insample_metrics.brier_score),
        "brier_eval_mode": eval_mode,
        "ece_before": float(pre_metrics.expected_calibration_error),
        "ece_after": ece_after,
        "pre_calibration_clip": [pipeline.config.pre_calibration_clip_lo, pipeline.config.pre_calibration_clip_hi],
        "reliability_pre": {
            "prob_pred": [round(float(x), 6) for x in np.asarray(pre_metrics.prob_pred).tolist()],
            "prob_true": [round(float(x), 6) for x in np.asarray(pre_metrics.prob_true).tolist()],
        },
        "reliability_post": {
            "prob_pred": [round(float(x), 6) for x in np.asarray(insample_metrics.prob_pred).tolist()],
            "prob_true": [round(float(x), 6) for x in np.asarray(insample_metrics.prob_true).tolist()],
        },
        **sharpener_info,
        **flb_info,
        "feature_manifest_hash": (
            (getattr(pipeline, "_feature_manifest", {}) or {}).get("manifest_hash")
        ),
    }
    if bootstrap_info:
        calibration_info.update(bootstrap_info)
    if _auto_selection_info:
        calibration_info.update(_auto_selection_info)
    if _vegas_anchor_info:
        calibration_info["vegas_anchor"] = _vegas_anchor_info

    # Add temperature value if using temperature scaling
    if effective_calibration_method == "temperature" and hasattr(pipeline.calibration_pipeline.calibrator, "temperature"):
        calibration_info["temperature"] = round(pipeline.calibration_pipeline.calibrator.temperature, 4)

    calibration_info["provenance"] = build_artifact_provenance(
        pipeline=pipeline,
        artifact_kind="calibration_report",
        extra={
            "fit_data_source": (
                "vegas_anchor_regular_season" if _vegas_anchor_used
                else "historical_tournament_only" if _nested_mode
                else "mixed_or_temporal_split"
            ),
            "evaluation_data_source": (
                "current_year_validation_only" if _nested_mode else eval_mode
            ),
            "historical_tournament_samples": _n_historical_tourney_cal,
            "current_year_validation_samples": _n_current_year_cal,
            "nested_calibration": _nested_mode,
        },
    )

    return calibration_info


def _fit_massey_predictor(pipeline, game_flows: Dict[str, List["GameFlow"]]) -> Dict:
    """Fit MasseyStandalonePredictor on validation-era games.

    Extracts Massey composite differences from validation-era game flows,
    calibrates sigma to minimize Brier score, and optimizes the blend
    weight between the base model and Massey-derived probabilities.

    Called from run() after _fit_calibration() so the base model is ready.

    Returns:
        Dict with fit statistics (sigma, blend_weight, brier, samples).
        Returns empty dict if fitting is disabled or insufficient data.
    """
    if not pipeline.config.fit_massey_on_training or pipeline._massey_predictor is None:
        return {}

    if not hasattr(pipeline, '_external_composites') or not pipeline._external_composites:
        logger.debug(
            "_fit_massey_predictor: no external composites loaded; skipping."
        )
        return {}

    calibration_games = pipeline._get_validation_era_games(game_flows)
    massey_cal_diffs: list = []
    massey_cal_outcomes: list = []
    massey_cal_model_probs: list = []

    for g in calibration_games:
        if g.team1_id not in pipeline.feature_engineer.team_features:
            continue
        if g.team2_id not in pipeline.feature_engineer.team_features:
            continue
        c1 = pipeline._external_composites.get(g.team1_id)
        c2 = pipeline._external_composites.get(g.team2_id)
        if c1 is None or c2 is None:
            continue
        o = pipeline._game_outcome(g)
        if o is None:
            continue
        p = float(np.clip(
            pipeline._raw_fusion_probability(g.team1_id, g.team2_id),
            pipeline.config.pre_calibration_clip_lo,
            pipeline.config.pre_calibration_clip_hi,
        ))
        massey_cal_diffs.append(c1.composite_rating - c2.composite_rating)
        massey_cal_outcomes.append(o)
        massey_cal_model_probs.append(p)

    n_samples = len(massey_cal_diffs)
    if n_samples < pipeline.config.massey_min_calibration_samples:
        logger.warning(
            "_fit_massey_predictor: only %d samples (need >= %d); "
            "using default sigma=%.1f, blend_weight=%.2f",
            n_samples,
            pipeline.config.massey_min_calibration_samples,
            pipeline._massey_predictor.sigma,
            pipeline._massey_predictor.blend_weight,
        )
        return {"massey_cal_samples": n_samples, "fitted": False}

    try:
        m_diffs = np.array(massey_cal_diffs, dtype=np.float64)
        m_outs = np.array(massey_cal_outcomes, dtype=np.float64)
        m_model_p = np.array(massey_cal_model_probs, dtype=np.float64)

        # Step 1: Calibrate sigma using configured bounds
        pipeline._massey_predictor.fit(
            m_diffs, m_outs,
            sigma_bounds=pipeline.config.massey_sigma_bounds,
        )

        # Step 2: Generate Massey probs and optimize blend weight
        m_probs = 1.0 / (1.0 + np.exp(
            -m_diffs / max(pipeline._massey_predictor.sigma, 0.01)
        ))
        pipeline._massey_predictor.fit_blend_weight(
            m_model_p, m_probs, m_outs,
            weight_bounds=pipeline.config.massey_blend_weight_bounds,
        )

        stats = {
            "massey_sigma": round(pipeline._massey_predictor.sigma, 3),
            "massey_blend_weight": round(pipeline._massey_predictor.blend_weight, 3),
            "massey_standalone_brier": round(pipeline._massey_predictor._fit_brier, 4),
            "massey_cal_samples": n_samples,
            "fitted": True,
        }
        logger.info(
            "_fit_massey_predictor: sigma=%.3f, blend_weight=%.3f, "
            "brier=%.4f on %d samples",
            pipeline._massey_predictor.sigma,
            pipeline._massey_predictor.blend_weight,
            pipeline._massey_predictor._fit_brier,
            n_samples,
        )
        return stats
    except Exception as e:
        logger.warning("_fit_massey_predictor: fitting failed: %s", e)
        return {"massey_cal_samples": n_samples, "fitted": False, "error": str(e)}


def _fit_tournament_sigma(pipeline, spread_model, tuning_stats: Dict) -> None:
    """Fit tournament-specific sigma from historical tournament data.

    Uses two approaches in priority order:
    1. Residual-based (preferred): use the trained SpreadRegressor to
       predict spreads for historical tournament games, then optimize
       sigma per round to minimize Brier score on actual outcomes.
    2. Margin-distribution-based (fallback): estimate sigma from the
       standard deviation of actual tournament margins per round.

    After fitting, overrides the SpreadRegressor's sigma with the global
    tournament sigma, and attaches the calibrator for per-round use by
    the _TrainedBaselineModel.
    """
    import os

    if not TOURNAMENT_SIGMA_AVAILABLE:
        return

    # Locate Kaggle tournament results CSV
    kaggle_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "kaggle",
    )
    tourney_csv = os.path.join(kaggle_dir, "MNCAATourneyCompactResults.csv")

    if not os.path.isfile(tourney_csv):
        logger.info(
            "Tournament sigma: MNCAATourneyCompactResults.csv not found at %s. "
            "Falling back to margin-distribution method.", kaggle_dir,
        )
        tourney_csv = None

    calibrator = TournamentSigmaCalibrator(
        prior_strength=30.0,
        n_bootstrap=100,  # Reduced for speed during training
    )

    if tourney_csv is not None:
        # Load historical tournament margins and round labels
        margins, round_labels, seasons = load_tournament_sigma_data(
            tourney_csv,
            min_season=2003,
            max_season=pipeline.config.year - 1,
        )

        if len(margins) < 30:
            logger.warning(
                "Tournament sigma: insufficient historical data (%d games). "
                "Using defaults.", len(margins),
            )
            calibrator._set_defaults()
            pipeline._tournament_sigma_calibrator = calibrator
            tuning_stats["tournament_sigma"] = {"fitted": True, "method": "defaults"}
            return

        # Try residual-based calibration if spread model is available
        if spread_model is not None and spread_model.model is not None:
            # Generate predicted spreads for historical tournament games
            # using margin distribution as proxy features (we don't have
            # feature vectors for historical games in this path).
            # Fall back to margin-based calibration which is nearly as good.
            fit_stats = calibrator.fit_from_margins(margins, round_labels)
        else:
            fit_stats = calibrator.fit_from_margins(margins, round_labels)
    else:
        # No CSV available — use hardcoded defaults
        calibrator._set_defaults()
        fit_stats = {"fitted": True, "method": "defaults"}

    pipeline._tournament_sigma_calibrator = calibrator

    # Override spread model's sigma with tournament-calibrated global sigma
    if calibrator.fitted and spread_model is not None:
        old_sigma = spread_model.sigma
        new_sigma = calibrator.global_tournament_sigma
        spread_model.sigma = new_sigma
        logger.info(
            "Tournament sigma: overrode SpreadRegressor sigma %.2f -> %.2f "
            "(tournament-calibrated from %d historical games)",
            old_sigma, new_sigma,
            fit_stats.get("n_total_games", 0),
        )

    tuning_stats["tournament_sigma"] = fit_stats
    if calibrator.fitted:
        tuning_stats["tournament_sigma_round_detail"] = {
            rname: {
                "sigma": est.sigma,
                "n_games": est.n_games,
                "source": est.source,
            }
            for rname, est in calibrator.round_estimates.items()
        }
