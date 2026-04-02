"""Seed baseline vs model: LOYO tournament comparison.

Trains on regular-season games from prior years, evaluates on
tournament games from the held-out year.  Computes per-fold Brier
scores for both the full model and a seed-only baseline, then runs
a paired t-test to determine if the model adds value beyond seeds.

Usage:
    python -m src.evaluation.seed_baseline_loyo
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

GAMES_DIR = "data/raw/historical"

# Years with tournament seed files (no 2020 — COVID cancellation).
EVAL_YEARS = [
    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025,
]


# ── Seed baseline ────────────────────────────────────────────────────────

# Historical first-round upset rates (from baselines.py).
SEED_WIN_RATES = {
    (1, 16): 0.985, (2, 15): 0.940, (3, 14): 0.850, (4, 13): 0.790,
    (5, 12): 0.640, (6, 11): 0.620, (7, 10): 0.610, (8, 9): 0.510,
}


def seed_baseline_prob(seed1: int, seed2: int) -> float:
    """P(team1 beats team2) using only seeds."""
    if seed1 == seed2:
        return 0.5
    fav, dog = min(seed1, seed2), max(seed1, seed2)
    key = (fav, dog)
    if key in SEED_WIN_RATES:
        p_fav = SEED_WIN_RATES[key]
    else:
        p_fav = 1.0 / (1.0 + math.exp(-0.175 * (dog - fav)))
    return p_fav if seed1 <= seed2 else 1.0 - p_fav


# ── Data loading ─────────────────────────────────────────────────────────

def _norm_id(raw: str) -> str:
    """Normalize team ID to lowercase with underscores."""
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def _load_seeds(year: int) -> Dict[str, int]:
    """Load tournament seeds for a year."""
    path = os.path.join(GAMES_DIR, f"tournament_seeds_{year}.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("teams", [])
    seeds: Dict[str, int] = {}
    for entry in data:
        seed = int(entry.get("seed", 0))
        if not seed:
            continue
        for key in ("team_id", "school_slug"):
            tid = entry.get(key, "")
            if tid:
                seeds[_norm_id(tid)] = seed
    return seeds


def _load_tournament_games(year: int) -> List[dict]:
    """Load tournament games (games after tournament start date)."""
    from src.pipeline.config import TOURNAMENT_START_DATES

    path = os.path.join(GAMES_DIR, f"historical_games_{year}.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        payload = json.load(f)
    team_games = payload.get("team_games", payload) if isinstance(payload, dict) else payload

    t_start = TOURNAMENT_START_DATES.get(year, date(year, 3, 14))
    cutoff = t_start.isoformat()

    games = []
    seen = set()
    for g in team_games:
        gdate = g.get("date") or g.get("game_date") or ""
        if gdate < cutoff:
            continue
        # Deduplicate: each game appears twice (once per team perspective).
        gid = g.get("game_id", "")
        if gid and gid in seen:
            continue
        if gid:
            seen.add(gid)

        tid = _norm_id(g.get("team_id", ""))
        oid = _norm_id(g.get("opponent_id", ""))
        t_score = g.get("team_score", 0)
        o_score = g.get("opponent_score", 0)
        if not tid or not oid or t_score == o_score:
            continue
        games.append({
            "team_id": tid,
            "opponent_id": oid,
            "team_score": t_score,
            "opponent_score": o_score,
            "date": gdate,
        })
    return games


def _build_seed_prefix_aliases(
    seeds: Dict[str, int], game_tids: set,
) -> Dict[str, int]:
    """Match game team IDs to seed entries via prefix matching."""
    aliases: Dict[str, int] = {}
    for gtid in game_tids:
        if gtid in seeds:
            continue
        matches = [(sid, s) for sid, s in seeds.items() if gtid.startswith(sid + "_")]
        if len(matches) == 1:
            aliases[gtid] = matches[0][1]
    return aliases


# ── Model training ───────────────────────────────────────────────────────

def _load_regular_season_data(
    config, year: int, feature_dim: int, prior_elo: Optional[Dict] = None,
) -> tuple:
    """Load regular-season data for a single year."""
    from src.pipeline.stages.sample_loading import load_year_samples_incremental

    games_path = os.path.join(GAMES_DIR, f"historical_games_{year}.json")
    metrics_path = os.path.join(GAMES_DIR, f"team_metrics_{year}.json")
    if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
        return np.empty((0, feature_dim)), np.array([]), np.array([]), {}, np.array([])
    return load_year_samples_incremental(
        config, games_path, metrics_path, feature_dim, year,
        include_tournament=False, prior_elo=prior_elo,
    )


def _load_tournament_data(
    config, year: int, feature_dim: int, prior_elo: Optional[Dict] = None,
) -> tuple:
    """Load tournament-only data for a single year."""
    from src.pipeline.stages.sample_loading import load_year_tournament_samples_incremental

    games_path = os.path.join(GAMES_DIR, f"historical_games_{year}.json")
    metrics_path = os.path.join(GAMES_DIR, f"team_metrics_{year}.json")
    if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
        return np.empty((0, feature_dim)), np.array([]), np.array([]), {}, np.array([])
    return load_year_tournament_samples_incremental(
        config, games_path, metrics_path, feature_dim, year,
        prior_elo=prior_elo,
    )


def _train_logistic(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Train logistic regression, return predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_tr = np.nan_to_num(X_train, nan=0.0)
    X_ev = np.nan_to_num(X_eval, nan=0.0)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)
    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    model.fit(X_tr, y_train)
    preds = model.predict_proba(X_ev)[:, 1]
    return np.clip(preds, 1e-7, 1 - 1e-7)


def _train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Train LightGBM, return predictions."""
    try:
        import lightgbm as lgb
    except ImportError:
        return _train_logistic(X_train, y_train, X_eval)
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
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=150)
    preds = model.predict(X_eval)
    return np.clip(preds, 1e-7, 1 - 1e-7)


# ── Main comparison ──────────────────────────────────────────────────────

def run_comparison(
    model_types: Optional[List[str]] = None,
) -> dict:
    """Run the seed baseline vs model LOYO comparison.

    For each held-out year:
      1. Train on regular-season games from all prior years
      2. Evaluate on tournament games from the held-out year
      3. Compute model Brier and seed baseline Brier on same games

    Returns dict with per-fold and aggregate results.
    """
    from src.pipeline.config import SOTAPipelineConfig
    from src.data.features.feature_engineering import MATCHUP_DIM

    if model_types is None:
        model_types = ["logistic", "lightgbm"]

    config = SOTAPipelineConfig(
        year=2026,
        multi_year_games_dir=GAMES_DIR,
        kaggle_dir="data/kaggle",
        external_ratings_dir="data/external_ratings",
    )
    feature_dim = MATCHUP_DIM

    # ── Pre-load all years (cache to avoid redundant loading) ──────
    logger.info("Pre-loading regular-season data for all years...")
    rs_cache: Dict[int, tuple] = {}
    cross_year_elo: Dict[str, float] = {}
    for year in sorted(EVAL_YEARS):
        X_yr, y_yr, m_yr, end_elo, _ = _load_regular_season_data(
            config, year, feature_dim, prior_elo=cross_year_elo,
        )
        rs_cache[year] = (X_yr, y_yr, m_yr)
        if end_elo:
            cross_year_elo = end_elo
        logger.info("  Year %d: %d regular-season samples", year, len(y_yr))

    logger.info("Pre-loading tournament data for all years...")
    tourney_cache: Dict[int, tuple] = {}
    for year in EVAL_YEARS:
        tX, ty, tm, te, trw = _load_tournament_data(config, year, feature_dim)
        tourney_cache[year] = (tX, ty, tm)
        logger.info("  Year %d: %d tournament samples", year, len(ty))

    logger.info("Data loading complete.\n")

    # Results storage
    results_by_model = {mt: [] for mt in model_types}
    seed_briers_per_fold = []
    fold_details = []

    for eval_year in EVAL_YEARS:
        logger.info("=" * 60)
        logger.info("FOLD: held-out year = %d", eval_year)

        # ------ Tournament eval data (features + outcomes) ------
        eval_X, eval_y, _ = tourney_cache[eval_year]
        if len(eval_y) < 5:
            logger.warning("  Year %d: < 5 tournament games, skipping", eval_year)
            continue

        # ------ Load tournament games + seeds for baseline ------
        tournament_games = _load_tournament_games(eval_year)
        seeds = _load_seeds(eval_year)

        # Build prefix aliases for fuzzy team ID matching
        game_tids = set()
        for g in tournament_games:
            game_tids.add(g["team_id"])
            game_tids.add(g["opponent_id"])
        aliases = _build_seed_prefix_aliases(seeds, game_tids)
        seeds.update(aliases)

        # Compute seed baseline predictions for each tournament game
        seed_preds = []
        seed_outcomes = []
        n_missing_seeds = 0
        for g in tournament_games:
            s1 = seeds.get(g["team_id"], 0)
            s2 = seeds.get(g["opponent_id"], 0)
            if s1 == 0 or s2 == 0:
                n_missing_seeds += 1
                continue
            seed_preds.append(seed_baseline_prob(s1, s2))
            seed_outcomes.append(1 if g["team_score"] > g["opponent_score"] else 0)

        if len(seed_preds) < 5:
            logger.warning(
                "  Year %d: only %d games with valid seeds (missing: %d), skipping",
                eval_year, len(seed_preds), n_missing_seeds,
            )
            continue

        seed_preds_arr = np.array(seed_preds)
        seed_outcomes_arr = np.array(seed_outcomes)
        seed_brier = float(np.mean((seed_preds_arr - seed_outcomes_arr) ** 2))

        logger.info(
            "  Seed baseline: %d games, Brier=%.4f (missing seeds: %d)",
            len(seed_preds), seed_brier, n_missing_seeds,
        )

        # ------ Assemble training data from all prior years ------
        train_X_parts = []
        train_y_parts = []
        for train_year in sorted(y for y in EVAL_YEARS if y < eval_year):
            X_yr, y_yr, _ = rs_cache[train_year]
            if len(y_yr) > 0:
                train_X_parts.append(X_yr)
                train_y_parts.append(y_yr)

        if not train_X_parts:
            logger.warning("  Year %d: no training data, skipping", eval_year)
            continue

        train_X = np.vstack(train_X_parts)
        train_y = np.concatenate(train_y_parts)

        # Clean: replace inf/nan
        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
        eval_X_clean = np.nan_to_num(eval_X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            "  Training: %d games from %d years | Eval: %d tournament games",
            len(train_y), len(train_X_parts), len(eval_y),
        )

        # ------ Model predictions on tournament data ------
        fold_info = {
            "year": eval_year,
            "n_train": len(train_y),
            "n_eval_model": len(eval_y),
            "n_eval_seed": len(seed_preds),
            "seed_brier": seed_brier,
        }

        for mt in model_types:
            if mt == "logistic":
                model_preds = _train_logistic(train_X, train_y, eval_X_clean)
            elif mt == "lightgbm":
                model_preds = _train_lightgbm(train_X, train_y, eval_X_clean)
            else:
                model_preds = _train_logistic(train_X, train_y, eval_X_clean)

            model_brier = float(np.mean((model_preds - eval_y) ** 2))
            fold_info[f"{mt}_brier"] = model_brier
            results_by_model[mt].append(model_brier)

            logger.info(
                "  %s model: Brier=%.4f | seed Brier=%.4f | delta=%.4f",
                mt, model_brier, seed_brier, model_brier - seed_brier,
            )

        seed_briers_per_fold.append(seed_brier)
        fold_details.append(fold_info)

    # ── Aggregate and statistical test ───────────────────────────────
    if not fold_details:
        logger.error("No valid folds. Cannot compute comparison.")
        return {"error": "no_valid_folds"}

    n_folds = len(fold_details)
    seed_arr = np.array(seed_briers_per_fold)

    print("\n" + "=" * 70)
    print("SEED BASELINE vs MODEL — LOYO TOURNAMENT COMPARISON")
    print("=" * 70)
    print(f"Folds: {n_folds} years | Eval: tournament games only")
    print(f"Training: regular-season games from prior years (rolling window)")
    print()

    # Per-fold table
    header = f"{'Year':>6} {'N_eval':>6} {'Seed':>8}"
    for mt in model_types:
        header += f" {mt:>12} {'Delta':>8}"
    print(header)
    print("-" * len(header))

    for fd in fold_details:
        row = f"{fd['year']:>6} {fd['n_eval_seed']:>6} {fd['seed_brier']:>8.4f}"
        for mt in model_types:
            mb = fd.get(f"{mt}_brier", float("nan"))
            delta = mb - fd["seed_brier"]
            row += f" {mb:>12.4f} {delta:>+8.4f}"
        print(row)

    print("-" * len(header))

    # Aggregate means
    row_mean = f"{'MEAN':>6} {'':>6} {float(np.mean(seed_arr)):>8.4f}"
    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        row_mean += f" {float(np.mean(m_arr)):>12.4f} {float(np.mean(m_arr - seed_arr)):>+8.4f}"
    print(row_mean)

    row_std = f"{'STD':>6} {'':>6} {float(np.std(seed_arr, ddof=1)):>8.4f}"
    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        row_std += f" {float(np.std(m_arr, ddof=1)):>12.4f} {float(np.std(m_arr - seed_arr, ddof=1)):>+8.4f}"
    print(row_std)

    print()

    # Paired t-test for each model type
    from scipy import stats

    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        diffs = m_arr - seed_arr  # positive = model worse than seed

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        se_diff = std_diff / np.sqrt(n_folds)

        # Two-sided paired t-test
        t_stat, p_two = stats.ttest_rel(m_arr, seed_arr)
        # One-sided: is model significantly BETTER (lower Brier)?
        p_one_better = p_two / 2 if t_stat < 0 else 1 - p_two / 2

        bss = 1.0 - float(np.mean(m_arr)) / float(np.mean(seed_arr))

        print(f"── {mt.upper()} vs SEED BASELINE ──")
        print(f"  Mean model Brier:  {float(np.mean(m_arr)):.6f}")
        print(f"  Mean seed Brier:   {float(np.mean(seed_arr)):.6f}")
        print(f"  Mean difference:   {mean_diff:+.6f} ({'model worse' if mean_diff > 0 else 'model better'})")
        print(f"  Brier Skill Score: {bss:+.4f}")
        print(f"  SE(difference):    {se_diff:.6f}")
        print(f"  Paired t-stat:     {t_stat:.4f}")
        print(f"  p-value (two-sided): {p_two:.4f}")
        print(f"  p-value (one-sided, model better): {p_one_better:.4f}")
        if p_one_better < 0.05:
            print(f"  >>> MODEL IS SIGNIFICANTLY BETTER THAN SEED BASELINE (p={p_one_better:.4f})")
        elif p_one_better > 0.95:
            print(f"  >>> SEED BASELINE IS SIGNIFICANTLY BETTER THAN MODEL (p={1-p_one_better:.4f})")
        else:
            print(f"  >>> NO SIGNIFICANT DIFFERENCE (p={p_one_better:.4f})")
        print()

    # ── Note on sample sizes ─────────────────────────────────────────
    total_eval_games = sum(fd["n_eval_seed"] for fd in fold_details)
    print("── DIAGNOSTICS ──")
    print(f"  Total tournament games evaluated: {total_eval_games}")
    print(f"  Mean games per fold: {total_eval_games / n_folds:.1f}")
    print(f"  Note: model evaluated on {sum(fd['n_eval_model'] for fd in fold_details)} games")
    print(f"        seed baseline on {total_eval_games} games")
    print(f"        (difference due to game deduplication/filtering)")
    print()
    print(
        "  CAVEAT: Seed baseline Brier is computed on deduplicated\n"
        "  tournament games from the JSON. Model Brier is computed on\n"
        "  the tournament samples from the incremental loader, which\n"
        "  may include duplicate perspectives (team A vs B and B vs A).\n"
        "  This means the sample sizes may differ. Both are valid\n"
        "  measurements, but use per-fold paired comparison with caution\n"
        "  if sample counts diverge significantly."
    )

    return {
        "n_folds": n_folds,
        "eval_years": [fd["year"] for fd in fold_details],
        "seed_briers": seed_briers_per_fold,
        "model_briers": {mt: results_by_model[mt] for mt in model_types},
        "fold_details": fold_details,
    }


if __name__ == "__main__":
    run_comparison()
