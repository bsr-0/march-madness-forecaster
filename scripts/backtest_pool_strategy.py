"""Backtest the bracket pool optimization strategy against historical outcomes.

Tests years 2018-2025 (excluding 2020) where we have ESPN public pick data
and actual tournament results. Compares a leverage-optimized bracket against
a pure chalk bracket to measure the strategy's historical edge.
"""

from __future__ import annotations
from scripts._common import (  # noqa: F401
    build_chalk_picks as _cm_build_chalk_picks,
    determine_winners as _cm_determine_winners,
    load_seeds,
    load_tournament_results,
    score_bracket_espn,
)

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.historical_picks import load_historical_public_picks
from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)
from src.simulation.ratings_opponent_model import build_opponent_model

YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

ESPN_SCORING: Dict[str, int] = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "CHAMP": 320,
}

# Map from result round names to scoring round names
RESULT_ROUND_TO_SCORING = {
    "R64": "R64",
    "R32": "R32",
    "S16": "S16",
    "E8": "E8",
    "F4": "F4",
    "NCG": "CHAMP",
}

SCOREABLE_ROUNDS = {"R64", "R32", "S16", "E8", "F4", "NCG"}


def load_results(year: int) -> List[Dict]:
    """Load tournament results, filtering to scoreable rounds only."""
    return [g for g in load_tournament_results(year) if g["round_name"] in SCOREABLE_ROUNDS]


def determine_winners(games: List[Dict]) -> Dict[str, set]:
    return _cm_determine_winners(games, ESPN_SCORING, RESULT_ROUND_TO_SCORING)


def score_bracket(picks: Dict[str, str], winners: Dict[str, set]) -> int:
    return score_bracket_espn(picks, winners, ESPN_SCORING)


def build_chalk_picks(games: List[Dict], seeds: Dict[str, int]) -> Dict[str, str]:
    return _cm_build_chalk_picks(games, seeds, RESULT_ROUND_TO_SCORING)


def build_leverage_picks_bracket(
    games: List[Dict],
    seeds: Dict[str, int],
    leverage_picks: List[Dict],
) -> Dict[str, str]:
    """Build a leverage bracket: use leverage picks when available, else chalk.

    For each game, check if either team has a leverage pick for that round.
    If so, pick the leveraged team. Otherwise, pick the lower seed.

    Returns dict of team_id -> round_name for each pick.
    """
    # Index leverage picks by (team_id, round)
    leverage_set: Dict[Tuple[str, str], float] = {}
    for lp in leverage_picks:
        key = (lp["team_id"], lp["round"])
        leverage_set[key] = lp.get("leverage_ratio", 0.0)

    picks: Dict[str, str] = {}
    for game in games:
        result_round = game["round_name"]
        scoring_round = RESULT_ROUND_TO_SCORING.get(result_round)
        if scoring_round is None:
            continue

        t1, t2 = game["team1_id"], game["team2_id"]
        s1 = seeds.get(t1, game.get("team1_seed", 16))
        s2 = seeds.get(t2, game.get("team2_seed", 16))

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


def format_hit_miss(
    leverage_picks: List[Dict],
    winners: Dict[str, set],
    top_n: int = 5,
) -> str:
    """Format top leverage picks with hit/miss indicators."""
    parts = []
    for lp in leverage_picks[:top_n]:
        team = lp["team_id"]
        rnd = lp["round"]
        hit = team in winners.get(rnd, set())
        mark = "\u2713" if hit else "\u2717"
        parts.append(f"{team} {rnd} {mark}")
    return ", ".join(parts)


def run_backtest() -> None:
    """Run the full historical backtest and print results."""
    print("HISTORICAL POOL STRATEGY BACKTEST")
    print("=" * 80)
    print(f"{'Year':<6} {'Leverage':>8} {'Chalk':>7} {'Edge':>6}  Top Leverage Picks (hit/miss)")
    print("-" * 80)

    yearly_results: List[Dict] = []

    for year in YEARS:
        try:
            # 1. Load data
            seeds = load_seeds(year)
            games = load_results(year)
            winners = determine_winners(games)

            # 2. Build seed-based probabilities
            pairwise_probs = build_seed_probabilities(seeds)
            round_probs = build_seed_round_probabilities(seeds)

            # 3. Build opponent model
            opponent_picks = build_opponent_model(year, seeds)

            # 4. Run pool optimizer
            env = PoolEnvironment(
                pool_size=100,
                scoring_rules=ESPN_SCORING,
                payout_structure="winner_take_all",
                public_pick_distribution=opponent_picks,
            )
            optimizer = PoolOptimizer(
                pairwise_probs,
                env,
                model_round_probs=round_probs,
            )
            result = optimizer.optimize()
            leverage_list = result.leverage_picks

            # 5. Score brackets
            chalk_bracket = build_chalk_picks(games, seeds)
            leverage_bracket = build_leverage_picks_bracket(games, seeds, leverage_list)
            chalk_score = score_bracket(chalk_bracket, winners)
            leverage_score = score_bracket(leverage_bracket, winners)

            edge = leverage_score - chalk_score
            hit_miss_str = format_hit_miss(leverage_list, winners, top_n=5)

            print(f"{year:<6} {leverage_score:>8} {chalk_score:>7} {edge:>+6}  {hit_miss_str}")

            yearly_results.append(
                {
                    "year": year,
                    "leverage_score": leverage_score,
                    "chalk_score": chalk_score,
                    "edge": edge,
                }
            )

        except Exception as exc:
            print(f"{year:<6} {'ERROR':>8}  {exc}")

    # Aggregate summary
    if yearly_results:
        n = len(yearly_results)
        mean_lev = sum(r["leverage_score"] for r in yearly_results) / n
        mean_chalk = sum(r["chalk_score"] for r in yearly_results) / n
        mean_edge = sum(r["edge"] for r in yearly_results) / n
        wins = sum(1 for r in yearly_results if r["edge"] > 0)

        print()
        print("=" * 80)
        print(f"AGGREGATE ({n} years):")
        print(f"  Mean leverage score: {mean_lev:.0f}")
        print(f"  Mean chalk score:    {mean_chalk:.0f}")
        print(f"  Mean edge:           {mean_edge:+.0f}")
        print(f"  Years where leverage > chalk: {wins}/{n}")
        print("=" * 80)


if __name__ == "__main__":
    run_backtest()
