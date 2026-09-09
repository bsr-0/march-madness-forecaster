"""Upset classifier: LOYO tournament evaluation.

Reuses the exact same feature/data plumbing as seed_baseline_loyo.py, but
re-targets the label to "did the lower (numerically higher) seed win?"
(an upset) instead of "did team1 win?". The features are unaffected by
this relabeling since they are already computed relative to team1/team2
in a fixed orientation per game — only the label changes.

Baseline: the seed-implied upset probability P(upset) = 1 - P(favorite
wins), using the same empirical/logistic seed model as seed_baseline_loyo.
A useful upset classifier must beat this baseline out-of-sample; games
where both teams share a seed (no favorite) are excluded.

Usage:
    python -m src.evaluation.upset_classifier_loyo
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.seed_baseline_loyo import (
    EVAL_YEARS,
    GAMES_DIR,
    _load_regular_season_data,
    _train_lightgbm,
    _train_logistic,
    load_ncaa_tournament_with_seeds,
    seed_baseline_prob,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def _upset_labels(
    y: np.ndarray, seeds1: np.ndarray, seeds2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (team1-won label, seeds) into (keep_mask, upset_label).

    upset_label[i] = 1 iff the higher-numbered (weaker) seed won.
    Games where seeds1[i] == seeds2[i] have no favorite and are dropped.
    """
    keep = seeds1 != seeds2
    team1_is_dog = seeds1 > seeds2
    upset = np.where(team1_is_dog, y, 1 - y)
    return keep, upset.astype(int)


def _seed_upset_prob(seed1: int, seed2: int) -> float:
    """Seed-baseline P(upset) for a given matchup, order-independent."""
    if seed1 == seed2:
        return float("nan")
    fav, dog = min(seed1, seed2), max(seed1, seed2)
    p_fav = seed_baseline_prob(fav, dog)
    return 1.0 - p_fav


def _train_tree(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Heavily regularized shallow decision tree — kept interpretable and
    resistant to overfitting on a few hundred training games."""
    from sklearn.tree import DecisionTreeClassifier

    X_tr = np.nan_to_num(X_train, nan=0.0)
    X_ev = np.nan_to_num(X_eval, nan=0.0)
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=30,
        ccp_alpha=0.005,
        random_state=0,
    )
    model.fit(X_tr, y_train)
    preds = model.predict_proba(X_ev)[:, 1] if model.n_classes_ > 1 else np.full(len(X_ev), y_train.mean())
    return np.clip(preds, 1e-7, 1 - 1e-7)


def run_comparison(model_types: Optional[List[str]] = None) -> dict:
    from src.pipeline.config import ForecastConfig
    from src.data.features.feature_engineering import MATCHUP_DIM

    if model_types is None:
        model_types = ["logistic", "tree", "lightgbm"]

    config = ForecastConfig(
        year=2026,
        multi_year_games_dir=GAMES_DIR,
        kaggle_dir="data/kaggle",
        external_ratings_dir="data/external_ratings",
    )
    feature_dim = MATCHUP_DIM

    logger.info("Pre-loading regular-season data for all years...")
    rs_cache: Dict[int, tuple] = {}
    cross_year_elo: Dict[str, float] = {}
    for year in sorted(EVAL_YEARS):
        X_yr, y_yr, m_yr, end_elo, _ = _load_regular_season_data(
            config, year, feature_dim, prior_elo=cross_year_elo
        )
        rs_cache[year] = (X_yr, y_yr)
        if end_elo:
            cross_year_elo = end_elo

    logger.info("Pre-loading NCAA tournament data (seeded games only)...")
    tourney_cache: Dict[int, tuple] = {}
    for year in EVAL_YEARS:
        tX, ty, ts1, ts2 = load_ncaa_tournament_with_seeds(config, year, feature_dim)
        tourney_cache[year] = (tX, ty, ts1, ts2)

    # Regular-season training targets stay "team1 won" (favorite/underdog
    # framing is a tournament-seed concept only) — the tree/logistic/GBM
    # models below just learn P(team1 wins) from the same features as
    # always, and we relabel ONLY the evaluation side into upset terms.
    results_by_model: Dict[str, list] = {mt: [] for mt in model_types}
    seed_briers_per_fold: list = []
    fold_details: list = []

    for eval_year in EVAL_YEARS:
        eval_X, eval_y, eval_s1, eval_s2 = tourney_cache[eval_year]
        keep, upset_y = _upset_labels(eval_y, eval_s1, eval_s2)
        n_dropped = int((~keep).sum())
        eval_X, upset_y = eval_X[keep], upset_y[keep]
        eval_s1, eval_s2 = eval_s1[keep], eval_s2[keep]
        n_eval = len(upset_y)
        if n_eval < 5:
            logger.warning("  Year %d: < 5 usable games (dropped %d no-favorite), skipping", eval_year, n_dropped)
            continue

        seed_preds = np.array([_seed_upset_prob(int(s1), int(s2)) for s1, s2 in zip(eval_s1, eval_s2)])
        seed_brier = float(np.mean((seed_preds - upset_y) ** 2))

        train_X_parts, train_y_parts = [], []
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

        # Reorient regular-season training labels/features into "upset"
        # terms too, using seed_diff proxy already baked into features is
        # not available for regular season games (no tournament seed), so
        # we instead train models directly on "team1 won" and flip the
        # predicted probability at eval time based on which side (team1
        # or team2) is the tournament underdog. This avoids inventing a
        # regular-season notion of "upset" that doesn't exist.
        team1_is_dog = eval_s1 > eval_s2

        fold_info: dict = {"year": eval_year, "n_train": len(train_y), "n_eval": n_eval, "seed_brier": seed_brier}

        for mt in model_types:
            if mt == "logistic":
                p_team1 = _train_logistic(train_X, train_y, eval_X_clean)
            elif mt == "tree":
                p_team1 = _train_tree(train_X, train_y, eval_X_clean)
            elif mt == "lightgbm":
                p_team1 = _train_lightgbm(train_X, train_y, eval_X_clean)
            else:
                p_team1 = _train_logistic(train_X, train_y, eval_X_clean)

            model_preds = np.where(team1_is_dog, p_team1, 1.0 - p_team1)
            model_brier = float(np.mean((model_preds - upset_y) ** 2))
            fold_info[f"{mt}_brier"] = model_brier
            results_by_model[mt].append(model_brier)

        seed_briers_per_fold.append(seed_brier)
        fold_details.append(fold_info)

    if not fold_details:
        logger.error("No valid folds.")
        return {"error": "no_valid_folds"}

    n_folds = len(fold_details)
    seed_arr = np.array(seed_briers_per_fold)

    print("\n" + "=" * 74)
    print("UPSET CLASSIFIER — LOYO TOURNAMENT COMPARISON")
    print("=" * 74)
    print(f"Folds: {n_folds} years | Label: higher-numbered seed wins (upset)")
    print(f"Baseline: seed-implied P(upset) = 1 - P(favorite wins)")
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
            row += f" {mb:>12.4f} {mb - fd['seed_brier']:>+8.4f}"
        print(row)
    print("-" * len(header))

    row_mean = f"{'MEAN':>6} {'':>4} {float(np.mean(seed_arr)):>8.4f}"
    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        row_mean += f" {float(np.mean(m_arr)):>12.4f} {float(np.mean(m_arr - seed_arr)):>+8.4f}"
    print(row_mean)
    print()

    from scipy import stats

    for mt in model_types:
        m_arr = np.array(results_by_model[mt])
        diffs = m_arr - seed_arr
        t_stat, p_two = stats.ttest_rel(m_arr, seed_arr)
        p_one_better = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        bss = 1.0 - float(np.mean(m_arr)) / float(np.mean(seed_arr))
        print(f"── {mt.upper()} vs SEED-IMPLIED BASELINE ──")
        print(f"  Mean model Brier: {float(np.mean(m_arr)):.6f} | Mean seed Brier: {float(np.mean(seed_arr)):.6f}")
        print(f"  Brier Skill Score: {bss:+.4f}")
        print(f"  Paired t-stat: {t_stat:.4f} | p (one-sided, model better): {p_one_better:.4f}")
        if p_one_better < 0.05:
            print(f"  >>> MODEL SIGNIFICANTLY BETTER")
        elif p_one_better > 0.95:
            print(f"  >>> SEED BASELINE SIGNIFICANTLY BETTER")
        else:
            print(f"  >>> NO SIGNIFICANT DIFFERENCE")
        print()

    total_eval = sum(fd["n_eval"] for fd in fold_details)
    print("── DIAGNOSTICS ──")
    print(f"  Total upset-labeled games evaluated: {total_eval}")
    print(f"  No-favorite (equal-seed) games are dropped from evaluation.")

    return {
        "n_folds": n_folds,
        "eval_years": [fd["year"] for fd in fold_details],
        "seed_briers": seed_briers_per_fold,
        "model_briers": {mt: results_by_model[mt] for mt in model_types},
        "fold_details": fold_details,
    }


if __name__ == "__main__":
    run_comparison()
