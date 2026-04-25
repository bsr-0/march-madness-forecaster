"""Feature importance analysis via LOYO tournament evaluation.

Reuses the data caches from seed_baseline_loyo.py to answer:
  1. Which individual features beat the seed baseline?
  2. Which feature GROUPS matter most?
  3. What do the logistic regression coefficients tell us?

Usage:
    python -m src.evaluation.feature_importance_loyo
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.seed_baseline_loyo import (
    EVAL_YEARS,
    GAMES_DIR,
    _load_regular_season_data,
    _train_logistic,
    load_ncaa_tournament_with_seeds,
    seed_baseline_prob,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Feature name map (must match TeamFeatures.get_feature_names) ────────

DIFF_FEATURE_NAMES = [
    "adj_off_eff",
    "adj_def_eff",
    "adj_tempo",
    "efg_pct",
    "to_rate",
    "orb_rate",
    "ft_rate",
    "opp_efg_pct",
    "opp_to_rate",
    "drb_rate",
    "opp_ft_rate",
    "total_rapm",
    "top5_rapm",
    "bench_rapm",
    "total_warp",
    "roster_continuity",
    "transfer_impact",
    "avg_experience",
    "bench_depth",
    "injury_risk",
    "lead_volatility",
    "entropy",
    "lead_sustainability",
    "comeback_factor",
    "xp_per_poss",
    "shot_distribution",
    "sos_adj_em",
    "sos_opp_o",
    "sos_opp_d",
    "ncsos_adj_em",
    "luck",
    "sor",
    "wab_poisson",
    "momentum",
    "three_pt_variance",
    "pace_adj_variance",
    "elo_rating",
    "free_throw_pct",
    "assist_to_turnover",
    "assist_rate",
    "steal_rate",
    "block_rate",
    "opp_two_pt_pct_allowed",
    "opp_three_pt_attempt_rate",
    "conference_adj_em",
    "three_pt_pct",
    "three_pt_rate",
    "def_xp_per_poss",
    "win_pct",
    "elite_sos",
    "q1_win_pct",
    "foul_rate",
    "three_pt_regression",
    "rest_days",
    "top5_minutes_share",
    "preseason_ap_rank",
    "coach_tournament_exp",
    "coach_tournament_win_rate",
    "coach_postseason_score",
    "pagerank_sos",
    "multi_hop_sos",
    "best_win_percentile",
    "paper_tiger_score",
    "dominance_ratio",
    "pace_variance",
    "conf_tourney_champ",
    "neutral_site_win_pct",
    "home_court_dependence",
    "tournament_resume",
    "backcourt_rapm",
    "frontcourt_rapm",
    "external_rating_composite",
    "external_rating_spread",
    "seed_strength",
    "massey_pom",
    "massey_sag",
    "massey_mor",
    "massey_dol",
    "massey_col",
    "massey_wol",
    "massey_rth",
    "massey_ap",
    "massey_usa",
    "massey_rpi",
    "massey_rank_mean",
    "massey_rank_std",
]  # 86 features — differential block [0:86]

ABS_FEATURE_NAMES = [
    "avg_adj_off_eff",
    "avg_adj_def_eff",
    "avg_sos_adj_em",
    "avg_elo_rating",
    "avg_win_pct",
]  # [86:91]

INTERACTION_FEATURE_NAMES = [
    "tempo_interaction",
    "style_mismatch",
    "seed_em_residual_diff",
    "sos_seed_interaction",
    "three_pt_var_seed_int",
    "seed_interaction",
    "seed_diff",
]  # [91:98]

ALL_FEATURE_NAMES = ["diff_" + n for n in DIFF_FEATURE_NAMES] + ABS_FEATURE_NAMES + INTERACTION_FEATURE_NAMES
assert len(ALL_FEATURE_NAMES) == 98

# ── SIMPLE_FEATURE_SET index mapping ────────────────────────────────────

SIMPLE_FEATURE_INDICES = {
    "diff_adj_off_eff": 0,
    "diff_adj_def_eff": 1,
    "diff_sos_adj_em": 26,
    "diff_elo_rating": 36,
    "diff_win_pct": 48,
    "diff_free_throw_pct": 37,
    "diff_momentum": 33,
}

# ── Feature groups for group-level analysis ─────────────────────────────

FEATURE_GROUPS = {
    "efficiency_core": [0, 1, 2],  # adj_off/def/tempo
    "four_factors_off": [3, 4, 5, 6],  # efg, to, orb, ft
    "four_factors_def": [7, 8, 9, 10],  # opp_efg, opp_to, drb, opp_ft
    "player_metrics": [11, 12, 13, 14, 15, 16],  # rapm, warp, continuity, transfer
    "roster_depth": [17, 18, 19],  # experience, bench, injury
    "volatility": [20, 21, 22, 23, 35, 64],  # lead_vol, entropy, sustainability, comeback, pace_adj_var, pace_var
    "shot_quality": [24, 25, 47],  # xp, distribution, def_xp
    "schedule_strength": [26, 27, 28, 29, 49, 59, 60],  # sos variants, elite_sos, pagerank, multi_hop
    "resume": [
        30,
        31,
        32,
        48,
        50,
        61,
        65,
        66,
        68,
    ],  # luck, sor, wab, win_pct, q1, best_win, conf_champ, neutral, tourn_resume
    "momentum": [33],
    "shooting": [34, 37, 45, 46, 52],  # 3pt_var, ft_pct, 3pt_pct, 3pt_rate, 3pt_regression
    "elo": [36],
    "ball_movement": [38, 39],
    "defense_disruption": [40, 41, 42, 43],  # steal, block, opp_2pt, opp_3pt_rate
    "context": [44, 51, 53, 54, 55, 67],  # conf_adj, foul, rest, top5_min, ap_rank, home_dep
    "coaching": [56, 57, 58],
    "quality_metrics": [62, 63],  # paper_tiger, dominance
    "positional_rapm": [69, 70],
    "external_ratings": [71, 72],
    "seed_strength": [73],
    "massey_systems": list(range(74, 86)),  # 12 massey features
    "absolute_level": list(range(86, 91)),
    "interactions": list(range(91, 98)),
    # Composite groups
    "SIMPLE_7": [0, 1, 26, 33, 36, 37, 48],
    "seed_features": [73, 91, 92, 93, 94, 95, 96, 97],
}


def _brier(preds: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((preds - y) ** 2))


def _run_loyo_subset(
    rs_cache: Dict[int, tuple],
    tourney_cache: Dict[int, tuple],
    feature_indices: List[int],
    seed_briers: Dict[int, float],
) -> Tuple[float, float, List[float]]:
    """Run LOYO with only the given feature indices. Returns (mean_brier, bss, per_fold_briers)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fold_briers = []
    for eval_year in EVAL_YEARS:
        eval_X, eval_y, _, _ = tourney_cache[eval_year]
        if len(eval_y) < 5:
            continue

        train_parts_X = []
        train_parts_y = []
        for yr in sorted(y for y in EVAL_YEARS if y < eval_year):
            X_yr, y_yr = rs_cache[yr]
            if len(y_yr) > 0:
                train_parts_X.append(X_yr)
                train_parts_y.append(y_yr)
        if not train_parts_X:
            continue

        train_X = np.nan_to_num(np.vstack(train_parts_X), nan=0.0, posinf=0.0, neginf=0.0)
        train_y = np.concatenate(train_parts_y)
        eval_X_clean = np.nan_to_num(eval_X, nan=0.0, posinf=0.0, neginf=0.0)

        # Subset features
        tr_sub = train_X[:, feature_indices]
        ev_sub = eval_X_clean[:, feature_indices]

        scaler = StandardScaler()
        tr_sub = scaler.fit_transform(tr_sub)
        ev_sub = scaler.transform(ev_sub)

        model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        model.fit(tr_sub, train_y)
        preds = np.clip(model.predict_proba(ev_sub)[:, 1], 1e-7, 1 - 1e-7)
        fold_briers.append(_brier(preds, eval_y))

    if not fold_briers:
        return float("nan"), float("nan"), []

    mean_brier = float(np.mean(fold_briers))
    mean_seed = float(np.mean([seed_briers[y] for y in EVAL_YEARS if y in seed_briers][: len(fold_briers)]))
    bss = 1.0 - mean_brier / mean_seed
    return mean_brier, bss, fold_briers


def _extract_coefficients(
    rs_cache: Dict[int, tuple],
    tourney_cache: Dict[int, tuple],
) -> np.ndarray:
    """Train logistic on ALL data, return standardized coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Use all years for training, get average coefficients across folds
    all_coefs = []
    for eval_year in EVAL_YEARS:
        eval_X, eval_y, _, _ = tourney_cache[eval_year]
        if len(eval_y) < 5:
            continue

        train_parts_X = []
        train_parts_y = []
        for yr in sorted(y for y in EVAL_YEARS if y < eval_year):
            X_yr, y_yr = rs_cache[yr]
            if len(y_yr) > 0:
                train_parts_X.append(X_yr)
                train_parts_y.append(y_yr)
        if not train_parts_X:
            continue

        train_X = np.nan_to_num(np.vstack(train_parts_X), nan=0.0, posinf=0.0, neginf=0.0)
        train_y = np.concatenate(train_parts_y)

        scaler = StandardScaler()
        tr_scaled = scaler.fit_transform(train_X)

        model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        model.fit(tr_scaled, train_y)
        all_coefs.append(model.coef_[0])

    return np.mean(all_coefs, axis=0)


def run_analysis():
    from src.pipeline.config import ForecastConfig
    from src.data.features.feature_engineering import MATCHUP_DIM

    config = ForecastConfig(
        year=2026,
        multi_year_games_dir=GAMES_DIR,
        kaggle_dir="data/kaggle",
        external_ratings_dir="data/external_ratings",
    )
    feature_dim = MATCHUP_DIM

    # ── Pre-load data ───────────────────────────────────────────────
    logger.info("Loading regular-season data...")
    rs_cache: Dict[int, tuple] = {}
    cross_year_elo: Dict[str, float] = {}
    for year in sorted(EVAL_YEARS):
        X_yr, y_yr, m_yr, end_elo, _ = _load_regular_season_data(
            config,
            year,
            feature_dim,
            prior_elo=cross_year_elo,
        )
        rs_cache[year] = (X_yr, y_yr)
        if end_elo:
            cross_year_elo = end_elo
        logger.info("  Year %d: %d samples", year, len(y_yr))

    logger.info("Loading NCAA tournament data...")
    tourney_cache: Dict[int, tuple] = {}
    for year in EVAL_YEARS:
        tX, ty, ts1, ts2 = load_ncaa_tournament_with_seeds(config, year, feature_dim)
        tourney_cache[year] = (tX, ty, ts1, ts2)

    # ── Compute seed baseline per fold ──────────────────────────────
    seed_briers: Dict[int, float] = {}
    for year in EVAL_YEARS:
        _, eval_y, eval_s1, eval_s2 = tourney_cache[year]
        if len(eval_y) < 5:
            continue
        seed_preds = np.array([seed_baseline_prob(int(s1), int(s2)) for s1, s2 in zip(eval_s1, eval_s2)])
        seed_briers[year] = _brier(seed_preds, eval_y)

    mean_seed_brier = float(np.mean(list(seed_briers.values())))

    logger.info("Data loaded. Seed baseline mean Brier = %.4f\n", mean_seed_brier)

    # ══════════════════════════════════════════════════════════════════
    # 1. FULL MODEL (all 98 features) — reference
    # ══════════════════════════════════════════════════════════════════
    full_brier, full_bss, full_folds = _run_loyo_subset(
        rs_cache,
        tourney_cache,
        list(range(98)),
        seed_briers,
    )
    print("=" * 74)
    print("FEATURE IMPORTANCE ANALYSIS — LOYO NCAA TOURNAMENT")
    print("=" * 74)
    print(f"Seed baseline mean Brier: {mean_seed_brier:.4f}")
    print(f"Full model (98 feat):     {full_brier:.4f}  BSS={full_bss:+.4f}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 2. INDIVIDUAL FEATURE BSS — which single features beat seeds?
    # ══════════════════════════════════════════════════════════════════
    print("─" * 74)
    print("INDIVIDUAL FEATURE BSS (single feature vs seed baseline)")
    print("─" * 74)

    individual_results = []
    for i in range(98):
        brier, bss, _ = _run_loyo_subset(rs_cache, tourney_cache, [i], seed_briers)
        individual_results.append((i, ALL_FEATURE_NAMES[i], brier, bss))

    individual_results.sort(key=lambda x: -x[3])  # Sort by BSS descending

    print(f"{'Rank':>4} {'Idx':>3} {'Feature':>35} {'Brier':>8} {'BSS':>8} {'vs Seed':>8}")
    print("-" * 74)
    for rank, (idx, name, brier, bss) in enumerate(individual_results[:30], 1):
        delta = brier - mean_seed_brier
        marker = " ***" if bss > 0 else ""
        print(f"{rank:>4} {idx:>3} {name:>35} {brier:>8.4f} {bss:>+8.4f} {delta:>+8.4f}{marker}")

    n_positive = sum(1 for _, _, _, bss in individual_results if bss > 0)
    print(f"\n{n_positive} of 98 features individually beat seed baseline.")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 3. FEATURE GROUP BSS
    # ══════════════════════════════════════════════════════════════════
    print("─" * 74)
    print("FEATURE GROUP BSS")
    print("─" * 74)

    group_results = []
    for group_name, indices in FEATURE_GROUPS.items():
        brier, bss, _ = _run_loyo_subset(rs_cache, tourney_cache, indices, seed_briers)
        group_results.append((group_name, len(indices), brier, bss))

    group_results.sort(key=lambda x: -x[3])

    print(f"{'Group':>25} {'N':>3} {'Brier':>8} {'BSS':>8}")
    print("-" * 50)
    for gname, n, brier, bss in group_results:
        marker = " ***" if bss > 0 else ""
        print(f"{gname:>25} {n:>3} {brier:>8.4f} {bss:>+8.4f}{marker}")

    print()

    # ══════════════════════════════════════════════════════════════════
    # 4. FORWARD SELECTION — greedy best-first
    # ══════════════════════════════════════════════════════════════════
    print("─" * 74)
    print("FORWARD SELECTION (greedy best-first, logistic regression)")
    print("─" * 74)

    remaining = set(range(98))
    selected: List[int] = []
    best_brier_so_far = float("inf")

    print(f"{'Step':>4} {'Feature Added':>35} {'Brier':>8} {'BSS':>8} {'Delta':>8}")
    print("-" * 70)

    for step in range(min(15, 98)):  # Stop at 15 features
        best_candidate = -1
        best_candidate_brier = float("inf")
        best_candidate_bss = float("-inf")

        for idx in remaining:
            trial = selected + [idx]
            brier, bss, _ = _run_loyo_subset(rs_cache, tourney_cache, trial, seed_briers)
            if brier < best_candidate_brier:
                best_candidate = idx
                best_candidate_brier = brier
                best_candidate_bss = bss

        if best_candidate < 0:
            break

        delta = best_candidate_brier - best_brier_so_far if best_brier_so_far < float("inf") else 0.0
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_brier_so_far = best_candidate_brier

        print(
            f"{step + 1:>4} {ALL_FEATURE_NAMES[best_candidate]:>35} {best_candidate_brier:>8.4f} {best_candidate_bss:>+8.4f} {delta:>+8.4f}"
        )

    print()

    # ══════════════════════════════════════════════════════════════════
    # 5. STANDARDIZED LOGISTIC COEFFICIENTS
    # ══════════════════════════════════════════════════════════════════
    print("─" * 74)
    print("STANDARDIZED LOGISTIC COEFFICIENTS (averaged across folds)")
    print("─" * 74)

    coefs = _extract_coefficients(rs_cache, tourney_cache)
    coef_ranked = sorted(
        [(i, ALL_FEATURE_NAMES[i], coefs[i]) for i in range(98)],
        key=lambda x: -abs(x[2]),
    )

    print(f"{'Rank':>4} {'Idx':>3} {'Feature':>35} {'Coef':>10} {'|Coef|':>8}")
    print("-" * 66)
    for rank, (idx, name, coef) in enumerate(coef_ranked[:30], 1):
        print(f"{rank:>4} {idx:>3} {name:>35} {coef:>+10.4f} {abs(coef):>8.4f}")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"Full 98-feature model BSS: {full_bss:+.4f}")
    print(f"Top individual feature:    {individual_results[0][1]} (BSS={individual_results[0][3]:+.4f})")

    simple_result = next((g for g in group_results if g[0] == "SIMPLE_7"), None)
    if simple_result:
        print(f"SIMPLE_7 features BSS:     {simple_result[3]:+.4f}")

    print(f"Top coefficient:           {coef_ranked[0][1]} (|coef|={abs(coef_ranked[0][2]):.4f})")
    print(f"\nForward-selected top 5:    {', '.join(ALL_FEATURE_NAMES[i] for i in selected[:5])}")


if __name__ == "__main__":
    run_analysis()
