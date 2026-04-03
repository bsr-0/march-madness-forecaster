#!/usr/bin/env python3
"""Unified evaluation: measures BOTH Brier score AND ESPN pool points
for all three probability modes (seed, noseed, blend).

This resolves the gap where the ablation script only measured Brier and
the EV backtest only measured pool points. Here we measure both metrics
side by side so we can make apples-to-apples comparisons.

Metrics:
  - Brier score: mean squared error of predicted win prob vs actual outcome.
    Lower is better. Measures raw prediction accuracy.
  - BSS (Brier Skill Score): 1 - (model_brier / seed_brier).
    Positive = better than seed baseline.
  - ESPN points: total bracket points from leverage picks fed through
    the pool optimizer. Higher is better. Measures pool EV.

For each test year (LOYO), trains noseed model on prior years, then
evaluates all three modes on both metrics.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.seed_pick_model import _win_rate
from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.prediction.noseed_model import (
    NoseedModel,
    _load_team_stats,
    build_blend_probabilities,
    build_blend_round_probabilities,
    build_noseed_probabilities,
    build_noseed_round_probabilities,
    train_noseed_model,
)
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.ratings_opponent_model import build_opponent_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIST_DIR = Path("data/raw/historical")

# Years with full data (seeds, results, Torvik, opponent model)
BACKTEST_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
RESULT_ROUND_TO_SCORING = {
    "R64": "R64",
    "R32": "R32",
    "S16": "S16",
    "E8": "E8",
    "F4": "F4",
    "NCG": "CHAMP",
}
SCOREABLE_ROUNDS = {"R64", "R32", "S16", "E8", "F4", "NCG"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_tournament_results(year):
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("games", [])


def load_seeds(year):
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "teams" in data:
        return {t["team_id"]: t["seed"] for t in data["teams"]}
    return {}


# ---------------------------------------------------------------------------
# Brier score measurement
# ---------------------------------------------------------------------------


def measure_brier(pairwise_probs, games, seeds):
    """Compute Brier score for a set of pairwise probabilities against
    actual game outcomes.

    For each game, looks up P(team1 wins) from the pairwise dict and
    compares to the actual binary outcome.

    Returns (brier_score, n_games).
    """
    sq_errors = []
    for g in games:
        if g.get("round_name") == "FF":
            continue
        t1, t2 = g["team1_id"], g["team2_id"]
        actual = 1.0 if g["team1_won"] else 0.0

        p = pairwise_probs.get((t1, t2))
        if p is None:
            # Fall back to seed-based if pair not in dict
            s1, s2 = seeds.get(t1, 8), seeds.get(t2, 8)
            p = _win_rate(s1, s2)

        sq_errors.append((p - actual) ** 2)

    if not sq_errors:
        return 0.25, 0  # coin-flip baseline
    return float(np.mean(sq_errors)), len(sq_errors)


# ---------------------------------------------------------------------------
# ESPN pool points measurement
# ---------------------------------------------------------------------------


def determine_winners(games):
    winners = {r: set() for r in ESPN_SCORING}
    for game in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(game["round_name"])
        if scoring_round is None:
            continue
        if game["team1_won"]:
            winners[scoring_round].add(game["team1_id"])
        else:
            winners[scoring_round].add(game["team2_id"])
    return winners


def score_bracket(picks, winners):
    total = 0
    for team_id, round_name in picks.items():
        if team_id in winners.get(round_name, set()):
            total += ESPN_SCORING[round_name]
    return total


def build_chalk_picks(games, seeds):
    picks = {}
    for game in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(game["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = game["team1_id"], game["team2_id"]
        s1 = seeds.get(t1, 16)
        s2 = seeds.get(t2, 16)
        picks[t1 if s1 <= s2 else t2] = scoring_round
    return picks


def build_leverage_bracket(games, seeds, leverage_picks):
    leverage_set = {}
    for lp in leverage_picks:
        leverage_set[(lp["team_id"], lp["round"])] = lp.get("leverage_ratio", 0.0)

    picks = {}
    for game in games:
        scoring_round = RESULT_ROUND_TO_SCORING.get(game["round_name"])
        if scoring_round is None:
            continue
        t1, t2 = game["team1_id"], game["team2_id"]
        s1, s2 = seeds.get(t1, 16), seeds.get(t2, 16)
        lev1 = leverage_set.get((t1, scoring_round), 0.0)
        lev2 = leverage_set.get((t2, scoring_round), 0.0)
        if lev1 > 0 and lev1 >= lev2:
            picks[t1] = scoring_round
        elif lev2 > 0:
            picks[t2] = scoring_round
        elif s1 <= s2:
            picks[t1] = scoring_round
        else:
            picks[t2] = scoring_round
    return picks


def measure_pool_points(pairwise_probs, round_probs, games, seeds, year):
    """Run pool optimizer and score the resulting bracket against actuals.

    Returns (optimizer_score, chalk_score).
    """
    scoreable_games = [g for g in games if g["round_name"] in SCOREABLE_ROUNDS]
    winners = determine_winners(scoreable_games)

    opponent_picks = build_opponent_model(year, seeds)

    env = PoolEnvironment(
        pool_size=100,
        scoring_rules=ESPN_SCORING,
        payout_structure="winner_take_all",
        public_pick_distribution=opponent_picks,
    )
    optimizer = PoolOptimizer(pairwise_probs, env, model_round_probs=round_probs)
    result = optimizer.optimize()

    chalk = build_chalk_picks(scoreable_games, seeds)
    leverage = build_leverage_bracket(scoreable_games, seeds, result.leverage_picks)

    chalk_score = score_bracket(chalk, winners)
    leverage_score = score_bracket(leverage, winners)

    return leverage_score, chalk_score


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print("UNIFIED MODE EVALUATION: Brier Score + ESPN Pool Points")
    print("=" * 100)
    print()
    print("Metrics:")
    print("  Brier  = mean squared error of P(team1 wins) vs actual (lower is better)")
    print("  BSS    = 1 - model_brier/seed_brier (positive = better than seed baseline)")
    print("  Points = ESPN bracket points from pool optimizer leverage picks")
    print("  Edge   = points above chalk (pure seed picks)")
    print()

    header = (
        f"  {'Year':<6}  {'--- Brier Score ---':^30}  {'------ BSS ------':^22}  {'------- ESPN Points -------':^34}"
    )
    subheader = (
        f"  {'':6}"
        f"  {'seed':>8} {'noseed':>8} {'blend':>8}"
        f"  {'noseed':>8} {'blend':>8}"
        f"  {'chalk':>7} {'seed':>7} {'noseed':>7} {'blend':>7}"
    )
    print(header)
    print(subheader)
    print(f"  {'-' * 94}")

    results = []

    for test_year in BACKTEST_YEARS:
        try:
            seeds = load_seeds(test_year)
            if not seeds:
                print(f"  {test_year:<6} SKIP — no seeds")
                continue

            games = load_tournament_results(test_year)
            stats = _load_team_stats(test_year)

            # Train noseed model on prior years
            model = train_noseed_model(max_year=test_year)

            # --- Build probabilities for each mode ---

            # Seed
            seed_pw = build_seed_probabilities(seeds)
            seed_rp = build_seed_round_probabilities(seeds)

            # Noseed
            noseed_pw = build_noseed_probabilities(model, seeds, stats)
            noseed_rp = build_noseed_round_probabilities(model, seeds, stats)

            # Blend (50/50)
            blend_pw = build_blend_probabilities(seed_pw, noseed_pw, alpha=0.5)
            blend_rp = build_blend_round_probabilities(seed_rp, noseed_rp, alpha=0.5)

            # --- Measure Brier scores ---
            seed_brier, n = measure_brier(seed_pw, games, seeds)
            noseed_brier, _ = measure_brier(noseed_pw, games, seeds)
            blend_brier, _ = measure_brier(blend_pw, games, seeds)

            noseed_bss = 1.0 - noseed_brier / seed_brier if seed_brier > 0 else 0.0
            blend_bss = 1.0 - blend_brier / seed_brier if seed_brier > 0 else 0.0

            # --- Measure ESPN pool points ---
            seed_pts, chalk_pts = measure_pool_points(seed_pw, seed_rp, games, seeds, test_year)
            noseed_pts, _ = measure_pool_points(noseed_pw, noseed_rp, games, seeds, test_year)
            blend_pts, _ = measure_pool_points(blend_pw, blend_rp, games, seeds, test_year)

            print(
                f"  {test_year:<6}"
                f"  {seed_brier:8.4f} {noseed_brier:8.4f} {blend_brier:8.4f}"
                f"  {noseed_bss:+8.3f} {blend_bss:+8.3f}"
                f"  {chalk_pts:7d} {seed_pts:7d} {noseed_pts:7d} {blend_pts:7d}"
            )

            results.append(
                {
                    "year": test_year,
                    "n_games": n,
                    "seed_brier": seed_brier,
                    "noseed_brier": noseed_brier,
                    "blend_brier": blend_brier,
                    "noseed_bss": noseed_bss,
                    "blend_bss": blend_bss,
                    "chalk_pts": chalk_pts,
                    "seed_pts": seed_pts,
                    "noseed_pts": noseed_pts,
                    "blend_pts": blend_pts,
                }
            )

        except Exception as exc:
            print(f"  {test_year:<6} ERROR: {exc}")
            import traceback

            traceback.print_exc()

    if not results:
        print("\nNo results produced.")
        return 1

    # --- Aggregates ---
    n = len(results)
    print(f"\n{'=' * 100}")
    print(f"AGGREGATE ({n} years)")
    print(f"{'=' * 100}")

    def mean(key):
        return np.mean([r[key] for r in results])

    print(f"\n  Brier Score (lower = better prediction accuracy):")
    print(f"    Seed:   {mean('seed_brier'):.4f}")
    print(f"    Noseed: {mean('noseed_brier'):.4f}")
    print(f"    Blend:  {mean('blend_brier'):.4f}")

    best_brier = min(
        [("seed", mean("seed_brier")), ("noseed", mean("noseed_brier")), ("blend", mean("blend_brier"))],
        key=lambda x: x[1],
    )
    print(f"    → Best for accuracy: {best_brier[0]} ({best_brier[1]:.4f})")

    print(f"\n  BSS vs seed baseline (positive = better than seeds):")
    print(f"    Noseed: {mean('noseed_bss'):+.4f}")
    print(f"    Blend:  {mean('blend_bss'):+.4f}")

    print(f"\n  ESPN Pool Points (higher = better pool EV):")
    print(f"    Chalk:  {mean('chalk_pts'):.0f}")
    print(f"    Seed:   {mean('seed_pts'):.0f}  (edge: {mean('seed_pts') - mean('chalk_pts'):+.0f})")
    print(f"    Noseed: {mean('noseed_pts'):.0f}  (edge: {mean('noseed_pts') - mean('chalk_pts'):+.0f})")
    print(f"    Blend:  {mean('blend_pts'):.0f}  (edge: {mean('blend_pts') - mean('chalk_pts'):+.0f})")

    best_pts = max(
        [("seed", mean("seed_pts")), ("noseed", mean("noseed_pts")), ("blend", mean("blend_pts"))],
        key=lambda x: x[1],
    )
    print(f"    → Best for pool EV: {best_pts[0]} ({best_pts[1]:.0f})")

    # --- Head-to-head ---
    print(f"\n  Head-to-head wins ({n} years):")
    noseed_brier_wins = sum(1 for r in results if r["noseed_brier"] < r["seed_brier"])
    blend_brier_wins = sum(1 for r in results if r["blend_brier"] < r["seed_brier"])
    noseed_pts_wins = sum(1 for r in results if r["noseed_pts"] > r["seed_pts"])
    blend_pts_wins = sum(1 for r in results if r["blend_pts"] > r["seed_pts"])
    print(f"    Brier: noseed beats seed {noseed_brier_wins}/{n}, blend beats seed {blend_brier_wins}/{n}")
    print(f"    Points: noseed beats seed {noseed_pts_wins}/{n}, blend beats seed {blend_pts_wins}/{n}")

    # --- Verdict ---
    print(f"\n{'=' * 100}")
    print("VERDICT")
    print(f"{'=' * 100}")
    print(f"  Best for prediction accuracy (Brier): {best_brier[0]}")
    print(f"  Best for pool EV (ESPN points):       {best_pts[0]}")

    if best_brier[0] != best_pts[0]:
        print(f"\n  Different modes win on different metrics.")
        print(f"  → Use '--mode {best_pts[0]}' for pool optimization (maximize leverage)")
        print(f"  → Use '--mode {best_brier[0]}' for Kaggle/prediction contests (maximize accuracy)")
    else:
        winner = best_brier[0]
        print(f"\n  '{winner}' wins on both metrics — use it as default.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
