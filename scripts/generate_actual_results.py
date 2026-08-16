"""Export the real 2026 tournament outcome to docs/data/ for the "how it
actually went" retrospective panel in the Bracket Picker UI.

The 2026 tournament concluded months before this was written (real
champion: Michigan, beat Connecticut 69-63 in the title game). This
script reads the ground-truth results already on disk
(data/raw/historical/tournament_results_2026.json) and reshapes them into
a simple per-round "who actually won" lookup so the frontend can grade
each strategy's picks against reality without re-implementing bracket
scoring logic in JS.

Excludes First Four (play-in) games — the bracket JSON's Round of 64 is
already the 64-team field, so play-in games have no corresponding slot.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._bracket_export_common import load_team_names
from scripts._common import load_tournament_results
from scripts.mc_pool_backtest import ESPN_SCORING

YEAR = 2026
OUT_PATH = PROJECT_ROOT / "docs" / "data" / f"actual_{YEAR}.json"

# Real results use "NCG" for the championship game; the bracket JSON /
# ESPN_SCORING use "CHAMP". Everything else matches.
RESULT_ROUND_TO_KEY = {"R64": "R64", "R32": "R32", "S16": "S16", "E8": "E8", "F4": "F4", "NCG": "CHAMP"}
ROUND_KEYS = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]


def main():
    games = load_tournament_results(YEAR)

    results_by_round = {k: [] for k in ROUND_KEYS}
    for game in games:
        key = RESULT_ROUND_TO_KEY.get(game["round_name"])
        if key is None:
            continue  # First Four play-in game, no bracket slot
        winner_id = game["team1_id"] if game["team1_won"] else game["team2_id"]
        results_by_round[key].append(winner_id)

    team_names = load_team_names()
    champion_id = results_by_round["CHAMP"][0]
    runner_up_id = [t for t in results_by_round["F4"] if t != champion_id][0]
    final_four_ids = results_by_round["E8"]

    output = {
        "season": YEAR,
        "status": "completed",
        "champion_id": champion_id,
        "champion_name": team_names.get(champion_id, champion_id),
        "runner_up_id": runner_up_id,
        "runner_up_name": team_names.get(runner_up_id, runner_up_id),
        # E8 *winners* = the 4 teams that reached the Final Four (ROUND_KEYS
        # names a round by the game being won, not the round being entered).
        "final_four_ids": final_four_ids,
        "final_four_names": [team_names.get(t, t) for t in final_four_ids],
        "results_by_round": results_by_round,
        "scoring": ESPN_SCORING,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Champion: {output['champion_name']}")
    print(f"Runner-up: {team_names.get(output['runner_up_id'], output['runner_up_id'])}")
    print(f"Final Four: {[team_names.get(t, t) for t in output['final_four_ids']]}")
    for k in ROUND_KEYS:
        print(f"  {k}: {len(results_by_round[k])} winners")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
