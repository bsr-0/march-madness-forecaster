"""Training window optimization with EV simulation verification.

Bridges the TrainingWindowOptimizer with actual model training and
validates window recommendations via Monte Carlo bracket simulation.

Phase 3 integration:
1. Provides a concrete ``train_predict_fn`` that trains lightweight
   ensemble models for each (window, model_type, eval_year) combination
   using pre-tournament-only data (no tournament games in training)
2. Verifies window recommendations by running MC bracket simulation
   and comparing weighted Brier score distributions across windows
3. Stores results in a format consumable by the pipeline config

The ``train_predict_fn`` trains on regular-season data and evaluates on
held-out tournament games, using the same PIT feature generation as
the production pipeline (``load_year_samples_incremental``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Brier score for a random 50/50 coin-flip baseline
_BRIER_COINFLIP = 0.25

# Kaggle round weights for weighted Brier simulation
_ROUND_WEIGHTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "NCG": 32}


@dataclass
class EVVerificationResult:
    """Result of EV simulation verification for a window recommendation."""

    model_type: str
    recommended_window: str
    recommended_brier: float
    runner_up_window: str
    runner_up_brier: float
    brier_delta: float
    ev_verified: bool
    sim_best_mean_brier: Optional[float] = None
    sim_runner_mean_brier: Optional[float] = None
    ev_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowOptimizationResult:
    """Complete result from training window optimization + EV verification."""

    recommendations: Dict[str, str]  # model_type -> window label
    optimal_years: Dict[str, Optional[int]]  # model_type -> num years (None=all)
    ev_verification: List[EVVerificationResult] = field(default_factory=list)
    report_path: Optional[str] = None


def build_train_predict_fn(
    config,
    games_dir: str,
) -> Callable:
    """Build a train_predict_fn compatible with TrainingWindowOptimizer.

    The returned callable trains a lightweight model on the given training
    years' **regular-season games only** and evaluates on the held-out
    year's tournament games using Brier score.

    Args:
        config: SOTAPipelineConfig instance (used for feature_dim, etc.).
        games_dir: Directory containing historical_games_{year}.json and
            team_metrics_{year}.json files.

    Returns:
        A callable with signature:
            (train_data: Dict[int, Any], eval_data: Any, model_type: str)
            -> (brier_score, log_loss, accuracy)
    """
    from ...data.features.feature_engineering import MATCHUP_DIM

    def train_predict_fn(
        train_data: Dict[int, Any],
        eval_data: Any,
        model_type: str,
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """Train on pre-tournament data, evaluate on held-out tournament games."""
        # Validate eval_data format
        if not isinstance(eval_data, dict) or "year" not in eval_data:
            raise ValueError(
                f"eval_data must be a dict with 'year' key, got {type(eval_data)}"
            )
        eval_year = eval_data["year"]
        train_years = sorted(train_data.keys())

        # Load training samples from each year — PRE-TOURNAMENT ONLY
        train_X_parts = []
        train_y_parts = []
        train_margin_parts = []
        cross_year_elo: Dict[str, float] = {}

        for yr in train_years:
            X_yr, y_yr, margins_yr, end_elo, _rw = _load_year_data(
                config, games_dir, yr, MATCHUP_DIM,
                include_tournament=False,  # Pre-tournament only
                prior_elo=cross_year_elo,
            )
            if len(y_yr) > 0:
                train_X_parts.append(X_yr)
                train_y_parts.append(y_yr)
                if margins_yr is not None and len(margins_yr) == len(y_yr):
                    train_margin_parts.append(margins_yr)
            if end_elo:
                cross_year_elo = end_elo

        if not train_X_parts:
            return (_BRIER_COINFLIP, None, None)

        train_X = np.vstack(train_X_parts)
        train_y = np.concatenate(train_y_parts)
        train_margins = (
            np.concatenate(train_margin_parts)
            if train_margin_parts and len(train_margin_parts) == len(train_X_parts)
            else None
        )

        # Load eval year tournament games
        eval_X, eval_y, _eval_margins, _eval_elo, _eval_rw = _load_year_data(
            config, games_dir, eval_year, MATCHUP_DIM,
            tournament_only=True,
            prior_elo=cross_year_elo,
        )

        if len(eval_y) < 5:
            return (_BRIER_COINFLIP, None, None)

        # Clean data
        train_X, train_y, train_margins = _clean_features(
            train_X, train_y, margins=train_margins
        )
        eval_X, eval_y, _ = _clean_features(eval_X, eval_y)

        if len(train_y) < 20:
            return (_BRIER_COINFLIP, None, None)

        # Train model and predict
        preds = _train_and_predict(
            train_X, train_y, eval_X, model_type, margins=train_margins
        )

        # Compute metrics
        brier = float(np.mean((preds - eval_y) ** 2))
        accuracy = float(np.mean((preds >= 0.5) == eval_y))

        # Log loss (with clipping to avoid inf)
        eps = 1e-7
        preds_clipped = np.clip(preds, eps, 1 - eps)
        log_loss = -float(np.mean(
            eval_y * np.log(preds_clipped) + (1 - eval_y) * np.log(1 - preds_clipped)
        ))

        return (brier, log_loss, accuracy)

    return train_predict_fn


def _load_year_data(
    config,
    games_dir: str,
    year: int,
    feature_dim: int,
    include_tournament: bool = False,
    tournament_only: bool = False,
    prior_elo: Optional[Dict[str, float]] = None,
) -> tuple:
    """Load year data using the incremental PIT loader."""
    from ..stages.sample_loading import (
        load_year_samples_incremental,
        load_year_tournament_samples_incremental,
    )

    games_path = os.path.join(games_dir, f"historical_games_{year}.json")
    metrics_path = os.path.join(games_dir, f"team_metrics_{year}.json")

    if not os.path.exists(games_path) or not os.path.exists(metrics_path):
        return np.empty((0, feature_dim)), np.array([]), np.array([]), {}, np.array([])

    try:
        if tournament_only:
            return load_year_tournament_samples_incremental(
                config, games_path, metrics_path, feature_dim, year,
                prior_elo=prior_elo,
            )
        else:
            return load_year_samples_incremental(
                config, games_path, metrics_path, feature_dim, year,
                include_tournament=include_tournament,
                prior_elo=prior_elo,
            )
    except Exception as e:
        logger.warning("Failed to load year %d: %s", year, e)
        return np.empty((0, feature_dim)), np.array([]), np.array([]), {}, np.array([])


def _clean_features(
    X: np.ndarray,
    y: np.ndarray,
    margins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Replace inf with NaN, drop rows that are all-NaN."""
    X = np.where(np.isinf(X), np.nan, X)
    # Drop rows where ALL features are NaN
    valid_mask = ~np.all(np.isnan(X), axis=1)
    m_out = margins[valid_mask] if margins is not None else None
    return X[valid_mask], y[valid_mask], m_out


def _train_and_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    eval_X: np.ndarray,
    model_type: str,
    margins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Train a lightweight model and return predictions on eval set.

    Uses conservative hyperparameters for speed — this is for window
    comparison, not production quality.
    """
    if model_type == "lightgbm":
        return _train_lightgbm(train_X, train_y, eval_X)
    elif model_type == "xgboost":
        return _train_xgboost(train_X, train_y, eval_X)
    elif model_type == "logistic":
        return _train_logistic(train_X, train_y, eval_X)
    elif model_type == "spread":
        return _train_spread(train_X, train_y, eval_X, margins=margins)
    else:
        logger.warning("Unknown model_type %r, falling back to logistic", model_type)
        return _train_logistic(train_X, train_y, eval_X)


def _train_lightgbm(
    train_X: np.ndarray, train_y: np.ndarray, eval_X: np.ndarray,
) -> np.ndarray:
    """Train LightGBM with conservative params for window comparison."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available, falling back to logistic")
        return _train_logistic(train_X, train_y, eval_X)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 8,
        "min_child_samples": 50,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(train_X, label=train_y)
    model = lgb.train(params, dtrain, num_boost_round=150)
    preds = model.predict(eval_X)
    return np.clip(preds, 0.01, 0.99)


def _train_xgboost(
    train_X: np.ndarray, train_y: np.ndarray, eval_X: np.ndarray,
) -> np.ndarray:
    """Train XGBoost with conservative params for window comparison."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not available, falling back to logistic")
        return _train_logistic(train_X, train_y, eval_X)

    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "min_child_weight": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(train_X, label=train_y)
    deval = xgb.DMatrix(eval_X)
    model = xgb.train(params, dtrain, num_boost_round=150)
    preds = model.predict(deval)
    return np.clip(preds, 0.01, 0.99)


def _train_logistic(
    train_X: np.ndarray, train_y: np.ndarray, eval_X: np.ndarray,
) -> np.ndarray:
    """Train logistic regression for window comparison."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        # Absolute fallback: predict base rate
        base_rate = np.mean(train_y)
        return np.full(len(eval_X), base_rate)

    # Impute NaN for sklearn (can't handle missing values)
    train_X_clean = np.nan_to_num(train_X, nan=0.0)
    eval_X_clean = np.nan_to_num(eval_X, nan=0.0)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X_clean)
    eval_X_scaled = scaler.transform(eval_X_clean)

    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    model.fit(train_X_scaled, train_y)
    preds = model.predict_proba(eval_X_scaled)[:, 1]
    return np.clip(preds, 0.01, 0.99)


def _train_spread(
    train_X: np.ndarray,
    train_y: np.ndarray,
    eval_X: np.ndarray,
    margins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Train spread regressor for window comparison.

    Uses real point margins when available. Falls back to logistic
    regression if SpreadRegressor is unavailable or margins are missing.
    """
    try:
        from ..ensemble.spread_model import SpreadRegressor
    except ImportError:
        return _train_logistic(train_X, train_y, eval_X)

    if margins is None or len(margins) != len(train_y):
        # No real margins available — fall back to logistic
        logger.info(
            "SpreadRegressor: no real margins available (margins=%s), "
            "falling back to logistic regression.",
            "None" if margins is None else f"len={len(margins)}",
        )
        return _train_logistic(train_X, train_y, eval_X)

    train_X_clean = np.nan_to_num(train_X, nan=0.0)
    eval_X_clean = np.nan_to_num(eval_X, nan=0.0)

    model = SpreadRegressor(sigma=11.0)
    model.train(train_X_clean, margins, num_rounds=150)
    preds = model.predict_probability(eval_X_clean)
    return np.clip(preds, 0.01, 0.99)


def _simulate_bracket_brier(
    predictions: np.ndarray,
    n_simulations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Run MC bracket simulation to compute weighted Brier score distribution.

    Samples tournament outcomes from the predicted probabilities and
    computes the Kaggle round-weighted Brier score for each simulation.

    Args:
        predictions: Array of predicted probabilities for tournament games.
        n_simulations: Number of MC iterations.
        seed: Random seed.

    Returns:
        Dict with mean_brier, std_brier, p5, p50, p95.
    """
    rng = np.random.RandomState(seed)
    n_games = len(predictions)

    # Assign approximate Kaggle round weights based on game position:
    # First 32 games = R64 (weight 1), next 16 = R32 (weight 2), etc.
    round_sizes = [32, 16, 8, 4, 2, 1]
    round_weight_values = [1, 2, 4, 8, 16, 32]
    weights = np.ones(n_games)
    idx = 0
    for size, weight in zip(round_sizes, round_weight_values):
        end = min(idx + size, n_games)
        weights[idx:end] = weight
        idx = end
        if idx >= n_games:
            break

    # Normalize weights so mean = 1.0
    weights = weights / np.mean(weights)

    brier_scores = np.empty(n_simulations)
    for i in range(n_simulations):
        outcomes = rng.binomial(1, predictions).astype(float)
        brier_scores[i] = float(np.mean(weights * (predictions - outcomes) ** 2))

    return {
        "mean_brier": float(np.mean(brier_scores)),
        "std_brier": float(np.std(brier_scores)),
        "p5": float(np.percentile(brier_scores, 5)),
        "p50": float(np.percentile(brier_scores, 50)),
        "p95": float(np.percentile(brier_scores, 95)),
    }


def verify_window_with_ev(
    window_report,
    config,
    games_dir: str,
    n_simulations: int = 5000,
) -> List[EVVerificationResult]:
    """Verify window recommendations via MC bracket simulation.

    For each model type:
    1. Identifies the top-2 windows by Brier score
    2. Runs bracket MC simulation with Kaggle round weights to compute
       expected weighted Brier distributions for each window
    3. Verifies the recommended window has better or equivalent EV
    4. If Brier delta < 0.005 (statistically tied), prefers shorter window

    Args:
        window_report: TrainingWindowReport from the optimizer.
        config: SOTAPipelineConfig.
        games_dir: Path to historical data directory.
        n_simulations: Number of MC simulations per window.

    Returns:
        List of EVVerificationResult, one per model type.
    """
    results = []

    for model_type, reports in window_report.model_type_results.items():
        if len(reports) < 2:
            logger.info(
                "EV verification: %s has < 2 windows, skipping.", model_type
            )
            continue

        # Sort by mean Brier (ascending = better)
        sorted_reports = sorted(reports, key=lambda r: r.mean_brier)
        best = sorted_reports[0]
        runner_up = sorted_reports[1]

        brier_delta = runner_up.mean_brier - best.mean_brier

        # Run MC bracket simulation for both windows.
        # Use per-year Brier scores as proxy probabilities for bracket games.
        # This simulates the variability a submission would experience.
        best_preds = np.array(list(best.per_year_brier.values()))
        runner_preds = np.array(list(runner_up.per_year_brier.values()))

        # Clamp to valid probability range for simulation
        best_preds = np.clip(best_preds, 0.01, 0.99)
        runner_preds = np.clip(runner_preds, 0.01, 0.99)

        # Run MC simulation for both windows
        sim_best = _simulate_bracket_brier(best_preds, n_simulations)
        sim_runner = _simulate_bracket_brier(runner_preds, n_simulations)

        # EV verification logic:
        # 1. If Brier delta is small (< 0.005), windows are statistically
        #    indistinguishable — prefer shorter window for compute savings
        # 2. Otherwise, verify the best window also wins the MC simulation
        if brier_delta < 0.005:
            best_size = best.window_size if best.window_size is not None else 999
            runner_size = runner_up.window_size if runner_up.window_size is not None else 999
            ev_verified = best_size <= runner_size
            ev_details = {
                "reason": "brier_delta_below_threshold",
                "threshold": 0.005,
                "prefer_shorter": True,
                "best_window_size": best.window_size,
                "runner_up_window_size": runner_up.window_size,
                "sim_best": sim_best,
                "sim_runner": sim_runner,
            }
        else:
            # Check MC simulation agrees with Brier ranking
            sim_agrees = sim_best["mean_brier"] <= sim_runner["mean_brier"]
            ev_verified = sim_agrees
            ev_details = {
                "reason": "mc_simulation_verified" if sim_agrees else "mc_simulation_disagrees",
                "brier_improvement_pct": round(
                    100 * brier_delta / max(runner_up.mean_brier, 0.01), 2
                ),
                "sim_best": sim_best,
                "sim_runner": sim_runner,
            }

        results.append(EVVerificationResult(
            model_type=model_type,
            recommended_window=best.window_label,
            recommended_brier=best.mean_brier,
            runner_up_window=runner_up.window_label,
            runner_up_brier=runner_up.mean_brier,
            brier_delta=brier_delta,
            ev_verified=ev_verified,
            sim_best_mean_brier=sim_best["mean_brier"],
            sim_runner_mean_brier=sim_runner["mean_brier"],
            ev_details=ev_details,
        ))

        logger.info(
            "EV verification [%s]: recommended=%s (Brier=%.4f, sim=%.4f) vs "
            "runner_up=%s (Brier=%.4f, sim=%.4f), delta=%.4f, verified=%s",
            model_type, best.window_label, best.mean_brier,
            sim_best["mean_brier"],
            runner_up.window_label, runner_up.mean_brier,
            sim_runner["mean_brier"],
            brier_delta, ev_verified,
        )

    return results


def _window_label_to_size(label: str) -> Optional[int]:
    """Convert a window label like 'last_5' or 'all_available' to an int."""
    if label == "all_available":
        return None
    if label.startswith("last_"):
        try:
            return int(label[5:])
        except ValueError:
            return None
    if label.startswith("post_"):
        return None  # Regime windows don't have a simple size
    return None


def run_training_window_optimization(
    config,
    games_dir: str,
    model_types: Optional[List[str]] = None,
    windows: Optional[List[Optional[int]]] = None,
    eval_years: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    verify_ev: bool = True,
) -> WindowOptimizationResult:
    """Run full Phase 3 training window optimization pipeline.

    Steps:
    1. Check for existing report — if found, re-verify via EV simulation
    2. Build train_predict_fn from actual model training code
    3. Run TrainingWindowOptimizer across candidate windows
    4. Verify recommendations via MC bracket simulation
    5. Return structured results for downstream pipeline use

    Args:
        config: SOTAPipelineConfig instance.
        games_dir: Directory with historical game/metrics JSON files.
        model_types: Model types to evaluate (default: lightgbm, xgboost, logistic).
        windows: Candidate window sizes (default: [3, 5, 8, 12]).
        eval_years: Years to hold out for evaluation.
        output_path: Optional path to save JSON report.
        verify_ev: Whether to run EV simulation verification.

    Returns:
        WindowOptimizationResult with recommendations and EV verification.
    """
    from .training_window_optimizer import TrainingWindowOptimizer

    if model_types is None:
        model_types = ["lightgbm", "xgboost", "logistic"]
    if windows is None:
        windows = [3, 5, 8, 12]

    # Step 0: Check for existing report — load and re-verify via EV
    if output_path and os.path.exists(output_path):
        import json
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if "recommendations" in existing and "model_results" in existing:
                logger.info(
                    "Found existing window optimization report at %s. "
                    "Loading and verifying via EV simulation.", output_path,
                )
                recommendations = existing["recommendations"]
                optimal_years = {
                    mt: _window_label_to_size(label)
                    for mt, label in recommendations.items()
                }

                # Re-verify with EV simulation if requested
                ev_results = []
                if verify_ev:
                    ev_results = _verify_cached_report(
                        existing, config, games_dir
                    )
                    # Apply EV overrides for tied windows
                    for ev_res in ev_results:
                        if not ev_res.ev_verified and ev_res.model_type in recommendations:
                            logger.info(
                                "EV override (cached): %s -> %s",
                                recommendations[ev_res.model_type],
                                ev_res.runner_up_window,
                            )
                            recommendations[ev_res.model_type] = ev_res.runner_up_window
                            optimal_years[ev_res.model_type] = _window_label_to_size(
                                ev_res.runner_up_window
                            )

                return WindowOptimizationResult(
                    recommendations=recommendations,
                    optimal_years=optimal_years,
                    ev_verification=ev_results,
                    report_path=output_path,
                )
        except Exception as e:
            logger.warning("Failed to load existing report: %s. Re-running.", e)

    # Step 1: Build train/predict callback
    train_predict_fn = build_train_predict_fn(config, games_dir)

    # Step 2: Discover available years
    data_by_year: Dict[int, Any] = {}
    for fname in sorted(os.listdir(games_dir)):
        if fname.startswith("historical_games_") and fname.endswith(".json"):
            try:
                yr = int(fname.replace("historical_games_", "").replace(".json", ""))
                if yr != 2020:  # Exclude COVID year
                    data_by_year[yr] = {"year": yr}
            except ValueError:
                pass

    if not data_by_year:
        logger.error("No historical game data found in %s", games_dir)
        return WindowOptimizationResult(
            recommendations={mt: "insufficient_data" for mt in model_types},
            optimal_years={mt: None for mt in model_types},
        )

    logger.info(
        "Training window optimization: %d years available (%d-%d), "
        "%d model types, %d candidate windows",
        len(data_by_year),
        min(data_by_year),
        max(data_by_year),
        len(model_types),
        len(windows),
    )

    # Step 3: Run optimizer
    optimizer = TrainingWindowOptimizer(
        windows=windows,
        eval_years=eval_years,
        include_regime_windows=True,
    )

    report = optimizer.evaluate_windows(
        data_by_year=data_by_year,
        train_predict_fn=train_predict_fn,
        model_types=model_types,
    )

    # Step 4: EV verification via MC bracket simulation
    ev_results = []
    if verify_ev:
        ev_results = verify_window_with_ev(
            report, config, games_dir,
        )

        # If EV verification suggests a different window (tied Brier but
        # shorter window preferred), update the recommendation
        for ev_res in ev_results:
            if not ev_res.ev_verified and ev_res.model_type in report.recommendations:
                logger.info(
                    "EV verification override for %s: %s -> %s (shorter, equivalent Brier)",
                    ev_res.model_type,
                    report.recommendations[ev_res.model_type],
                    ev_res.runner_up_window,
                )
                report.recommendations[ev_res.model_type] = ev_res.runner_up_window

    # Step 5: Build result
    optimal_years = {
        mt: _window_label_to_size(label)
        for mt, label in report.recommendations.items()
    }

    # Step 6: Save report
    if output_path:
        optimizer.save_report(report, output_path)
        logger.info("Window optimization report saved to %s", output_path)

    result = WindowOptimizationResult(
        recommendations=report.recommendations,
        optimal_years=optimal_years,
        ev_verification=ev_results,
        report_path=output_path,
    )

    logger.info("Training window optimization complete: %s", result.recommendations)
    return result


def _verify_cached_report(
    report_data: Dict[str, Any],
    config,
    games_dir: str,
) -> List[EVVerificationResult]:
    """Re-verify a cached report's recommendations via EV simulation.

    Reconstructs WindowSizeReport objects from the JSON data and runs
    verify_window_with_ev on them.
    """
    from .training_window_optimizer import TrainingWindowReport, WindowSizeReport

    model_type_results: Dict[str, List[WindowSizeReport]] = {}

    for model_type, window_list in report_data.get("model_results", {}).items():
        reports = []
        for w in window_list:
            per_year = {int(k): v for k, v in w.get("per_year_brier", {}).items()}
            reports.append(WindowSizeReport(
                model_type=model_type,
                window_size=w.get("window_size"),
                window_label=w.get("window_label", "unknown"),
                mean_brier=w.get("mean_brier", _BRIER_COINFLIP),
                std_brier=w.get("std_brier", 0.0),
                per_year_brier=per_year,
                n_eval_years=w.get("n_eval_years", 0),
            ))
        model_type_results[model_type] = reports

    fake_report = TrainingWindowReport(
        model_type_results=model_type_results,
        recommendations=report_data.get("recommendations", {}),
        raw_results=[],
        eval_years=report_data.get("eval_years", []),
    )

    return verify_window_with_ev(fake_report, config, games_dir)
