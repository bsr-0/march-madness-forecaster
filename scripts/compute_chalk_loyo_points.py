"""Compute chalk's real ESPN points for every LOYO backtest year.

chalk (always the lower seed, barthag tiebreak) is deterministic and needs
no training data or Monte Carlo — it's a pure function of a single year's
own seeds/regions/actual results, so this bypasses run_backtest()'s full
opponent-simulation machinery entirely and just scores the one bracket
chalk would have submitted against that year's real outcome.

Region pairing doesn't need to match reality here: R64-E8 chalk picks are
region-internal (invariant to region_order), F4/CHAMP picks are invariant
to which regions get paired first (seed+barthag tiebreak is a total order,
so the bracket winner is the transitive max regardless of pairing), and
team-identity scoring only checks team-set membership per round, not
game-level structure — so encode/decode consistency is all that matters,
not matching the real F4 pairing.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import load_tournament_results
from scripts.mc_pool_backtest import (
    BACKTEST_YEARS,
    ESPN_SCORING,
    _load_torvik_barthag,
    build_first_round_matchups,
    load_seeds_and_regions,
    resolve_first_four,
)
from src.simulation.pool_competition import actual_winners_by_round, score_brackets_team_identity

OUT_PATH = PROJECT_ROOT / "artifacts" / "backtest_runs" / "chalk_loyo_points.json"


def chalk_bracket_vector(first_round, seeds, barthag):
    """Deterministic chalk walk: lower seed wins, tie-break by barthag."""
    result = np.zeros(63, dtype=bool)
    current = list(first_round)
    game_idx = 0
    for _ in range(6):
        next_round = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            s1, s2 = seeds.get(t1, 99), seeds.get(t2, 99)
            if s1 != s2:
                t1_wins = s1 < s2
            else:
                t1_wins = barthag.get(t1, 0.5) >= barthag.get(t2, 0.5)
            result[game_idx] = t1_wins
            next_round.append(t1 if t1_wins else t2)
            game_idx += 1
        current = next_round
    return result


def main():
    points_by_year = {}
    for year in BACKTEST_YEARS:
        seeds, regions = load_seeds_and_regions(year)
        games = load_tournament_results(year)
        if not seeds or not games:
            print(f"  {year}  SKIP — no seeds or results")
            continue

        resolve_first_four(games, seeds, regions)
        first_round = build_first_round_matchups(seeds, regions)
        if len(first_round) != 64:
            print(f"  {year}  SKIP — {len(first_round)} teams (need 64)")
            continue

        barthag = _load_torvik_barthag(year, seeds)
        bvec = chalk_bracket_vector(first_round, seeds, barthag)
        winners_by_round = actual_winners_by_round(games)

        score = score_brackets_team_identity(
            bvec.reshape(1, 63), winners_by_round, first_round, dict(ESPN_SCORING)
        )[0]
        points_by_year[str(year)] = float(score)
        print(f"  {year}  chalk  {score:.0f} pts")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"mode": "chalk", "points_by_year": points_by_year}, f, indent=2)
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
