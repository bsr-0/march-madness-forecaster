#!/usr/bin/env python3
"""Backtest: does using no-seed model probabilities improve pool EV?

Compares three strategies for the pool optimizer's "truth" probabilities:
  1. Seed-only: current approach (seed_probabilities.py)
  2. No-seed model: ML model trained without seed features
  3. Blend: 50/50 seed baseline + no-seed model (best for raw accuracy)

For each year (2018-2025), trains the no-seed model on all prior years,
generates probabilities, feeds them to the pool optimizer, and scores
the resulting leverage brackets against actual outcomes.

The key question: does the no-seed model's structural disagreement with
the public (who thinks in seeds) translate to better pool EV?
"""

import sys
from pathlib import Path

from scripts._common import (  # noqa: F401
    build_chalk_picks as _cm_build_chalk_picks,
    build_leverage_bracket as _cm_build_leverage_bracket,
    determine_winners as _cm_determine_winners,
    load_seeds,
    load_tournament_results,
    score_bracket_espn,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.prediction.noseed_model import (
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

# Years where we have all needed data: seeds, results, Torvik, opponent model
BACKTEST_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
RESULT_ROUND_TO_SCORING = {"R64": "R64", "R32": "R32", "S16": "S16", "E8": "E8", "F4": "F4", "NCG": "CHAMP"}
SCOREABLE_ROUNDS = {"R64", "R32", "S16", "E8", "F4", "NCG"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def determine_winners(games):
    return _cm_determine_winners(games, ESPN_SCORING, RESULT_ROUND_TO_SCORING)


def score_bracket(picks, winners):
    return score_bracket_espn(picks, winners, ESPN_SCORING)


def build_chalk_picks(games, seeds):
    return _cm_build_chalk_picks(games, seeds, RESULT_ROUND_TO_SCORING)


def build_leverage_bracket(games, seeds, leverage_picks):
    return _cm_build_leverage_bracket(games, seeds, leverage_picks, RESULT_ROUND_TO_SCORING)


def run_optimizer(pairwise, round_probs, opponent_picks):
    """Run pool optimizer and return leverage picks."""
    env = PoolEnvironment(
        pool_size=100,
        scoring_rules=ESPN_SCORING,
        payout_structure="winner_take_all",
        public_pick_distribution=opponent_picks,
    )
    optimizer = PoolOptimizer(pairwise, env, model_round_probs=round_probs)
    result = optimizer.optimize()
    return result.leverage_picks


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def main():
    print("=" * 90)
    print("POOL EV BACKTEST: Seed-only vs No-seed vs Blend probabilities")
    print("=" * 90)
    print(
        f"\n  {'Year':<6} {'Chalk':>6} {'Seed-EV':>8} {'NoSeed-EV':>10} {'Blend-EV':>9} {'Seed Edge':>10} {'NoSeed Edge':>12} {'Blend Edge':>11}"
    )
    print(f"  {'-' * 4:<6} {'-' * 6} {'-' * 8} {'-' * 10} {'-' * 9} {'-' * 10} {'-' * 12} {'-' * 11}")

    results = []

    for test_year in BACKTEST_YEARS:
        try:
            seeds = load_seeds(test_year)
            if not seeds:
                print(f"  {test_year:<6} SKIP — no seeds")
                continue

            games = [g for g in load_tournament_results(test_year) if g["round_name"] in SCOREABLE_ROUNDS]
            winners = determine_winners(games)
            stats = _load_team_stats(test_year)

            # Train no-seed model on prior years (max_year=test_year → strict
            # walk-forward; no future years leak into the fold).
            try:
                model = train_noseed_model(max_year=test_year)
            except ValueError as e:
                print(f"  {test_year:<6} SKIP — {e}")
                continue

            # Build opponent model (same for all conditions)
            opponent_picks = build_opponent_model(test_year, seeds)

            # --- Condition 1: Seed-only probabilities (current approach) ---
            seed_pairwise = build_seed_probabilities(seeds)
            seed_round = build_seed_round_probabilities(seeds)
            seed_leverage = run_optimizer(seed_pairwise, seed_round, opponent_picks)

            # --- Condition 2: No-seed model probabilities ---
            noseed_pairwise = build_noseed_probabilities(model, seeds, stats)
            noseed_round = build_noseed_round_probabilities(model, seeds, stats)
            noseed_leverage = run_optimizer(noseed_pairwise, noseed_round, opponent_picks)

            # --- Condition 3: Blend (50/50 seed + no-seed) ---
            blend_pairwise = build_blend_probabilities(seed_pairwise, noseed_pairwise, alpha=0.5)
            blend_round = build_blend_round_probabilities(seed_round, noseed_round, alpha=0.5)
            blend_leverage = run_optimizer(blend_pairwise, blend_round, opponent_picks)

            # Score all brackets
            chalk = build_chalk_picks(games, seeds)
            seed_bracket = build_leverage_bracket(games, seeds, seed_leverage)
            noseed_bracket = build_leverage_bracket(games, seeds, noseed_leverage)
            blend_bracket = build_leverage_bracket(games, seeds, blend_leverage)

            chalk_score = score_bracket(chalk, winners)
            seed_score = score_bracket(seed_bracket, winners)
            noseed_score = score_bracket(noseed_bracket, winners)
            blend_score = score_bracket(blend_bracket, winners)

            seed_edge = seed_score - chalk_score
            noseed_edge = noseed_score - chalk_score
            blend_edge = blend_score - chalk_score

            print(
                f"  {test_year:<6} {chalk_score:6d} {seed_score:8d} {noseed_score:10d} {blend_score:9d} {seed_edge:+10d} {noseed_edge:+12d} {blend_edge:+11d}"
            )

            results.append(
                {
                    "year": test_year,
                    "chalk": chalk_score,
                    "seed_ev": seed_score,
                    "noseed_ev": noseed_score,
                    "blend_ev": blend_score,
                    "seed_edge": seed_edge,
                    "noseed_edge": noseed_edge,
                    "blend_edge": blend_edge,
                }
            )

        except Exception as exc:
            print(f"  {test_year:<6} ERROR: {exc}")
            import traceback

            traceback.print_exc()

    if not results:
        print("\nNo results produced.")
        return 1

    # Aggregate
    n = len(results)
    print(f"\n{'=' * 90}")
    print(f"AGGREGATE ({n} years)")
    print(f"{'=' * 90}")

    mean_chalk = np.mean([r["chalk"] for r in results])
    mean_seed = np.mean([r["seed_ev"] for r in results])
    mean_noseed = np.mean([r["noseed_ev"] for r in results])
    mean_blend = np.mean([r["blend_ev"] for r in results])

    print(f"  Mean chalk score:      {mean_chalk:.0f}")
    print(f"  Mean seed-EV score:    {mean_seed:.0f}  (edge: {mean_seed - mean_chalk:+.0f})")
    print(f"  Mean no-seed-EV score: {mean_noseed:.0f}  (edge: {mean_noseed - mean_chalk:+.0f})")
    print(f"  Mean blend-EV score:   {mean_blend:.0f}  (edge: {mean_blend - mean_chalk:+.0f})")

    seed_wins = sum(1 for r in results if r["seed_edge"] > 0)
    noseed_wins = sum(1 for r in results if r["noseed_edge"] > 0)
    blend_wins = sum(1 for r in results if r["blend_edge"] > 0)
    print(f"\n  Years beating chalk: seed={seed_wins}/{n}, no-seed={noseed_wins}/{n}, blend={blend_wins}/{n}")

    # Head-to-head: no-seed vs seed
    noseed_beats_seed = sum(1 for r in results if r["noseed_ev"] > r["seed_ev"])
    blend_beats_seed = sum(1 for r in results if r["blend_ev"] > r["seed_ev"])
    print(f"  No-seed beats seed:  {noseed_beats_seed}/{n}")
    print(f"  Blend beats seed:    {blend_beats_seed}/{n}")

    # Per-round leverage pick analysis
    print(f"\n{'=' * 90}")
    print("VERDICT")
    print(f"{'=' * 90}")
    best = max(
        [("Seed-only", mean_seed), ("No-seed", mean_noseed), ("Blend", mean_blend)],
        key=lambda x: x[1],
    )
    print(f"  Best pool EV strategy: {best[0]} (mean score {best[1]:.0f})")
    if best[0] == "No-seed":
        print("  → Use no-seed model for pool optimization (EV mode)")
        print("  → Use blend for Kaggle-style prediction accuracy")
    elif best[0] == "Blend":
        print("  → Blend wins for both EV and accuracy")
    else:
        print("  → Seed-only remains best — no-seed model doesn't improve pool EV")

    return 0


if __name__ == "__main__":
    sys.exit(main())
