"""Seed baseline vs model: LOYO tournament comparison.

Trains on regular-season games from prior years, evaluates on NCAA
tournament games from the held-out year.  Both the model and the
seed baseline are evaluated on the EXACT SAME game set: only games
where both teams have valid tournament seeds.

Usage:
    python -m src.evaluation.seed_baseline_loyo
"""

from __future__ import annotations

import json
import logging
import math
import os
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


# ── Unified NCAA tournament loader ──────────────────────────────────────
#
# Produces aligned (X, y, seed1, seed2) arrays for games where both
# teams have valid tournament seeds.  Uses IncrementalMetricsEngine
# for PIT features — same as the production loader, but also tracks
# seeds and does NOT apply symmetric augmentation (each game once).


def _norm_id(raw: str) -> str:
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def _load_seeds(year: int) -> Dict[str, int]:
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


def load_ncaa_tournament_with_seeds(
    config,
    year: int,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NCAA tournament games with aligned features AND seeds.

    Only returns games where BOTH teams have valid tournament seeds.
    No symmetric augmentation — each game appears exactly once.

    Returns:
        (X, y, seeds1, seeds2) where all arrays have the same length N:
          X:      [N, feature_dim] feature matrix
          y:      [N] binary outcomes (1 = team1 won)
          seeds1: [N] seed of team1
          seeds2: [N] seed of team2
    """
    from src.pipeline.config import TOURNAMENT_START_DATES
    from src.data.features.proprietary_metrics import (
        IncrementalMetricsEngine,
        _team_id,
        team_games_to_game_records,
    )
    from src.pipeline.stages.data_loader import load_roster_overlay

    games_path = os.path.join(GAMES_DIR, f"historical_games_{year}.json")
    metrics_path = os.path.join(GAMES_DIR, f"team_metrics_{year}.json")
    empty = (np.empty((0, feature_dim)), np.array([]), np.array([]), np.array([]))

    if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
        return empty

    # ── 1. Load game data ────────────────────────────────────────────
    with open(games_path) as f:
        payload = json.load(f)
    team_games_raw = payload.get("team_games", []) if isinstance(payload, dict) else payload
    if not team_games_raw:
        return empty

    game_records = team_games_to_game_records(team_games_raw, year)
    if len(game_records) < 100:
        return empty

    # ── 2. Load seeds ────────────────────────────────────────────────
    team_seeds = _load_seeds(year)
    if not team_seeds:
        return empty

    # Prefix aliases: game team_ids like "akron_zips" → seed "akron"
    game_tids = set()
    for row in team_games_raw:
        game_tids.add(_norm_id(str(row.get("team_id", ""))))
        game_tids.add(_norm_id(str(row.get("opponent_id", ""))))
    for gtid in game_tids:
        if gtid in team_seeds:
            continue
        matches = [(sid, s) for sid, s in team_seeds.items() if gtid.startswith(sid + "_")]
        if len(matches) == 1:
            team_seeds[gtid] = matches[0][1]

    # ── 3. Load auxiliary data (conference map, Massey, rosters) ─────
    conference_map = None
    try:
        with open(metrics_path) as f:
            metrics_payload = json.load(f)
        if isinstance(metrics_payload, dict):
            for tid, info in metrics_payload.items():
                if isinstance(info, dict) and "conference" in info:
                    if conference_map is None:
                        conference_map = {}
                    conference_map[tid] = info["conference"]
    except Exception:
        pass

    if not conference_map and config.kaggle_dir:
        try:
            from src.data.kaggle_loader import KaggleDataLoader
            _kloader = KaggleDataLoader(config.kaggle_dir)
            _kconf = _kloader.load_team_conferences(year)
            if _kconf:
                conference_map = _kconf
        except Exception:
            pass

    team_massey_composite: Dict[str, float] = {}
    massey_cache_path = os.path.join(GAMES_DIR, f"external_massey_composite_{year}.json")
    if os.path.isfile(massey_cache_path):
        try:
            with open(massey_cache_path) as f:
                massey_data = json.load(f)
            for entry in massey_data:
                tid = entry.get("team_id", "")
                if tid:
                    team_massey_composite[_team_id(tid)] = entry.get("normalized", float('nan'))
        except Exception:
            pass

    team_massey_spread: Dict[str, float] = {}
    team_massey_multi: Dict = {}

    roster_path = os.path.join(GAMES_DIR, f"cbbpy_rosters_{year}.json")
    team_roster_overlay = load_roster_overlay(roster_path, year=year)

    # ── 4. Create incremental engine ─────────────────────────────────
    inc_engine = IncrementalMetricsEngine(
        game_records,
        conference_map=conference_map,
    )

    # ── 5. Identify NCAA tournament games ────────────────────────────
    t_start = TOURNAMENT_START_DATES.get(year, date(year, 3, 14))
    tournament_cutoff = t_start.isoformat()

    seen_gids: set = set()
    all_games = []
    for g in sorted(game_records, key=lambda r: r.game_date):
        if g.game_id in seen_gids:
            continue
        seen_gids.add(g.game_id)
        all_games.append(g)

    tournament_games = [g for g in all_games if g.game_date >= tournament_cutoff]

    # Data quality filter (same as production loader)
    tournament_games = [
        g for g in tournament_games
        if g.points > 0 and g.opp_points > 0 and abs(g.points - g.opp_points) <= 80
    ]

    # ── 6. Build features ONLY for seeded games ──────────────────────
    min_games = getattr(config, "game_level_min_games_per_team", 5)

    X_list = []
    y_list = []
    s1_list = []
    s2_list = []
    skipped = 0
    no_seeds = 0

    for g in tournament_games:
        # Only keep games where BOTH teams have tournament seeds
        seed1 = team_seeds.get(g.team_id, 0)
        seed2 = team_seeds.get(g.opponent_id, 0)
        if seed1 == 0 or seed2 == 0:
            no_seeds += 1
            continue

        if min_games > 0:
            n1 = inc_engine.games_played_before(g.team_id, g.game_date)
            n2 = inc_engine.games_played_before(g.opponent_id, g.game_date)
            if n1 < min_games or n2 < min_games:
                skipped += 1
                continue

        metrics = inc_engine.compute_as_of(g.game_date)
        if not metrics:
            skipped += 1
            continue

        m1 = metrics.get(g.team_id)
        m2 = metrics.get(g.opponent_id)
        if m1 is None or m2 is None:
            skipped += 1
            continue

        # Massey: only attach after selection cutoff
        _mc1 = team_massey_composite.get(g.team_id, float('nan'))
        _mc2 = team_massey_composite.get(g.opponent_id, float('nan'))
        _ms1 = team_massey_spread.get(g.team_id, float('nan'))
        _ms2 = team_massey_spread.get(g.opponent_id, float('nan'))
        _mm1 = team_massey_multi.get(g.team_id)
        _mm2 = team_massey_multi.get(g.opponent_id)

        v1 = IncrementalMetricsEngine.metrics_to_team_vector(
            m1, seed=seed1,
            external_rating_composite=_mc1, external_rating_spread=_ms1,
            massey_features=_mm1,
        )
        v2 = IncrementalMetricsEngine.metrics_to_team_vector(
            m2, seed=seed2,
            external_rating_composite=_mc2, external_rating_spread=_ms2,
            massey_features=_mm2,
        )

        for _v, _tid in ((v1, g.team_id), (v2, g.opponent_id)):
            _overlay = team_roster_overlay.get(_tid, {})
            for _idx, _val in _overlay.items():
                _v[_idx] = _val

        matchup = IncrementalMetricsEngine.build_matchup_vector(
            v1, v2, seed1, seed2,
            engine=inc_engine,
            team1_id=g.team_id,
            team2_id=g.opponent_id,
        )

        if len(matchup) < feature_dim:
            padded = np.zeros(feature_dim, dtype=np.float64)
            padded[:len(matchup)] = matchup
            matchup = padded
        elif len(matchup) > feature_dim:
            matchup = matchup[:feature_dim]

        X_list.append(matchup)
        y_list.append(1 if g.points > g.opp_points else 0)
        s1_list.append(seed1)
        s2_list.append(seed2)

    if not X_list:
        return empty

    logger.info(
        "  Year %d NCAA tournament: %d seeded games loaded "
        "(%d no seeds, %d skipped metrics). feature_dim=%d.",
        year, len(X_list), no_seeds, skipped, feature_dim,
    )

    return (
        np.stack(X_list),
        np.array(y_list, dtype=int),
        np.array(s1_list, dtype=int),
        np.array(s2_list, dtype=int),
    )


# ── Regular-season loader (for training) ────────────────────────────────

def _load_regular_season_data(
    config, year: int, feature_dim: int, prior_elo: Optional[Dict] = None,
) -> tuple:
    from src.pipeline.stages.sample_loading import load_year_samples_incremental

    games_path = os.path.join(GAMES_DIR, f"historical_games_{year}.json")
    metrics_path = os.path.join(GAMES_DIR, f"team_metrics_{year}.json")
    if not os.path.isfile(games_path) or not os.path.isfile(metrics_path):
        return np.empty((0, feature_dim)), np.array([]), np.array([]), {}, np.array([])
    return load_year_samples_incremental(
        config, games_path, metrics_path, feature_dim, year,
        include_tournament=False, prior_elo=prior_elo,
    )


# ── Model training ───────────────────────────────────────────────────────

def _train_logistic(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
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

    Both model and seed baseline are evaluated on the EXACT SAME
    game set: NCAA tournament games where both teams have seeds.
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

    # ── Pre-load all years ───────────────────────────────────────────
    logger.info("Pre-loading regular-season data for all years...")
    rs_cache: Dict[int, tuple] = {}
    cross_year_elo: Dict[str, float] = {}
    for year in sorted(EVAL_YEARS):
        X_yr, y_yr, m_yr, end_elo, _ = _load_regular_season_data(
            config, year, feature_dim, prior_elo=cross_year_elo,
        )
        rs_cache[year] = (X_yr, y_yr)
        if end_elo:
            cross_year_elo = end_elo
        logger.info("  Year %d: %d regular-season samples", year, len(y_yr))

    logger.info("Pre-loading NCAA tournament data (seeded games only)...")
    tourney_cache: Dict[int, tuple] = {}
    for year in EVAL_YEARS:
        tX, ty, ts1, ts2 = load_ncaa_tournament_with_seeds(config, year, feature_dim)
        tourney_cache[year] = (tX, ty, ts1, ts2)

    logger.info("Data loading complete.\n")

    # ── LOYO evaluation ──────────────────────────────────────────────
    results_by_model: Dict[str, list] = {mt: [] for mt in model_types}
    seed_briers_per_fold: list = []
    fold_details: list = []

    for eval_year in EVAL_YEARS:
        logger.info("=" * 60)
        logger.info("FOLD: held-out year = %d", eval_year)

        eval_X, eval_y, eval_s1, eval_s2 = tourney_cache[eval_year]
        n_eval = len(eval_y)
        if n_eval < 5:
            logger.warning("  Year %d: < 5 seeded tournament games, skipping", eval_year)
            continue

        # Seed baseline — SAME games as model
        seed_preds = np.array([
            seed_baseline_prob(int(s1), int(s2))
            for s1, s2 in zip(eval_s1, eval_s2)
        ])
        seed_brier = float(np.mean((seed_preds - eval_y) ** 2))

        logger.info("  Seed baseline: %d games, Brier=%.4f", n_eval, seed_brier)

        # Assemble training data from prior years
        train_X_parts = []
        train_y_parts = []
        for train_year in sorted(y for y in EVAL_YEARS if y < eval_year):
            X_yr, y_yr = rs_cache[train_year]
            if len(y_yr) > 0:
                train_X_parts.append(X_yr)
                train_y_parts.append(y_yr)

        if not train_X_parts:
            logger.warning("  Year %d: no training data, skipping", eval_year)
            continue

        train_X = np.nan_to_num(np.vstack(train_X_parts), nan=0.0, posinf=0.0, neginf=0.0)
        train_y = np.concatenate(train_y_parts)
        eval_X_clean = np.nan_to_num(eval_X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            "  Training: %d games from %d years | Eval: %d NCAA tournament games (same for both)",
            len(train_y), len(train_X_parts), n_eval,
        )

        fold_info: dict = {
            "year": eval_year,
            "n_train": len(train_y),
            "n_eval": n_eval,
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
                "  %s: Brier=%.4f | seed=%.4f | delta=%+.4f",
                mt, model_brier, seed_brier, model_brier - seed_brier,
            )

        seed_briers_per_fold.append(seed_brier)
        fold_details.append(fold_info)

    # ── Aggregate and statistical test ───────────────────────────────
    if not fold_details:
        logger.error("No valid folds.")
        return {"error": "no_valid_folds"}

    n_folds = len(fold_details)
    seed_arr = np.array(seed_briers_per_fold)

    print("\n" + "=" * 70)
    print("SEED BASELINE vs MODEL — LOYO TOURNAMENT COMPARISON")
    print("=" * 70)
    print(f"Folds: {n_folds} years | Eval: NCAA tournament games (seeded only)")
    print(f"Training: regular-season games from prior years (rolling window)")
    print(f"Both model and seed baseline evaluated on IDENTICAL game sets.")
    print()

    header = f"{'Year':>6} {'N':>4} {'Seed':>8}"
    for mt in model_types:
        header += f" {mt:>12} {'Delta':>8}"
    print(header)
    print("-" * len(header))

    for fd in fold_details:
        row = f"{fd['year']:>6} {fd['n_eval']:>4} {fd['seed_brier']:>8.4f}"
        for mt in model_types:
            mb = fd.get(f"{mt}_brier", float("nan"))
            delta = mb - fd["seed_brier"]
            row += f" {mb:>12.4f} {delta:>+8.4f}"
        print(row)

    print("-" * len(header))

    row_mean = f"{'MEAN':>6} {'':>4} {float(np.mean(seed_arr)):>8.4f}"
    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        row_mean += f" {float(np.mean(m_arr)):>12.4f} {float(np.mean(m_arr - seed_arr)):>+8.4f}"
    print(row_mean)

    row_std = f"{'STD':>6} {'':>4} {float(np.std(seed_arr, ddof=1)):>8.4f}"
    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        row_std += f" {float(np.std(m_arr, ddof=1)):>12.4f} {float(np.std(m_arr - seed_arr, ddof=1)):>+8.4f}"
    print(row_std)

    print()

    from scipy import stats

    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        diffs = m_arr - seed_arr

        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))
        se_diff = std_diff / np.sqrt(n_folds)

        t_stat, p_two = stats.ttest_rel(m_arr, seed_arr)
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
            print(f"  >>> MODEL SIGNIFICANTLY BETTER (p={p_one_better:.4f})")
        elif p_one_better > 0.95:
            print(f"  >>> SEED BASELINE SIGNIFICANTLY BETTER (p={1 - p_one_better:.4f})")
        else:
            print(f"  >>> NO SIGNIFICANT DIFFERENCE (p={p_one_better:.4f})")
        print()

    total_eval = sum(fd["n_eval"] for fd in fold_details)
    print("── DIAGNOSTICS ──")
    print(f"  Total NCAA tournament games evaluated: {total_eval}")
    print(f"  Mean games per fold: {total_eval / n_folds:.1f}")
    print(f"  Model and seed baseline evaluated on identical game sets.")
    print(f"  No symmetric augmentation on evaluation data.")

    return {
        "n_folds": n_folds,
        "eval_years": [fd["year"] for fd in fold_details],
        "seed_briers": seed_briers_per_fold,
        "model_briers": {mt: results_by_model[mt] for mt in model_types},
        "fold_details": fold_details,
    }


if __name__ == "__main__":
    run_comparison()
