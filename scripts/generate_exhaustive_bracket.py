"""Generate the meta_exhaustive bracket for 2026 and export to docs/data/.

Uses exhaustive_champion construction mode (tries all 64 possible champions,
picks the bracket with the highest expected points) on Torvik round probs.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    ESPN_SCORING,
    REGION_ORDER,
    SEED_MATCHUP_ORDER,
    _log5,
    _load_torvik_barthag,
    build_torvik_round_probabilities,
    build_espn_pick_distribution,
    load_seeds_and_regions,
)
from src.optimization.bracket_construction import construct_bracket

YEAR = 2026
OUT_PATH = PROJECT_ROOT / "docs" / "data" / "bracket_2026_exhaustive.json"
TEAM_PROFILES_PATH = PROJECT_ROOT / "docs" / "data" / "team_profiles.json"

ROUND_DISPLAY = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
ROUND_KEYS = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

F4_REGION_LABELS = [
    f"{REGION_ORDER[0]} vs {REGION_ORDER[1]}",
    f"{REGION_ORDER[2]} vs {REGION_ORDER[3]}",
]
CHAMP_LABEL = "Championship"


def load_team_names():
    with open(TEAM_PROFILES_PATH) as f:
        data = json.load(f)
    return {t["team_id"]: t["team_name"] for t in data["teams"]}


def fmt_team(seed, name):
    return f"({seed}) {name}"


def build_bracket_json(seeds, regions, barthag, round_probs, picks, team_names):
    """Walk the picks dict and construct the full 6-round JSON structure."""

    # Build region → {seed → team_id} map
    teams_by_region = defaultdict(dict)
    for tid, seed in seeds.items():
        region = regions.get(tid, "")
        teams_by_region[region][seed] = tid

    # Round winners from picks for quick lookup
    round_winners_by_key = {k: v for k, v in picks.items()}

    # We need to simulate forward through the bracket to know which teams
    # meet in each game. Walk round by round in bracket order.
    current = []
    for region in REGION_ORDER:
        rt = teams_by_region[region]
        for high, low in SEED_MATCHUP_ORDER:
            current.append(rt.get(high, f"unknown_{region}_{high}"))
            current.append(rt.get(low, f"unknown_{region}_{low}"))

    rounds_output = []

    for round_idx, (display_name, rkey) in enumerate(zip(ROUND_DISPLAY, ROUND_KEYS)):
        games = []
        next_round = []

        for g_idx in range(0, len(current), 2):
            t1, t2 = current[g_idx], current[g_idx + 1]

            seed1 = seeds.get(t1, 0)
            seed2 = seeds.get(t2, 0)
            name1 = team_names.get(t1, t1)
            name2 = team_names.get(t2, t2)
            rating1 = barthag.get(t1, 0.5)
            rating2 = barthag.get(t2, 0.5)
            win_prob = round(_log5(rating1, rating2), 4)

            # Determine game key
            game_num = g_idx // 2 + 1
            if rkey == "R64":
                region = regions.get(t1, regions.get(t2, ""))
                game_key = f"R64_{region}_{seed1}v{seed2}"
            elif rkey in ("R32", "S16"):
                region = regions.get(t1, regions.get(t2, ""))
                # game_num resets per region; each region occupies 8 slots in R64
                region_idx = REGION_ORDER.index(region) if region in REGION_ORDER else 0
                games_per_region_prev = 8 >> (ROUND_KEYS.index(rkey) - 1)  # 8→4→2
                region_game = (g_idx // 2) - region_idx * games_per_region_prev + 1
                game_key = f"{rkey}_{region}_{region_game}"
            elif rkey == "E8":
                region = regions.get(t1, regions.get(t2, ""))
                game_key = f"E8_{region}"
            elif rkey == "F4":
                # 2 F4 games
                if game_num == 1:
                    game_key = f"F4_{REGION_ORDER[0]}_{REGION_ORDER[1]}"
                else:
                    game_key = f"F4_{REGION_ORDER[2]}_{REGION_ORDER[3]}"
            else:
                game_key = "CHAMP"

            winner_id = round_winners_by_key.get(game_key)
            if winner_id is None:
                # Fall back: pick t1 if win_prob >= 0.5
                winner_id = t1 if win_prob >= 0.5 else t2

            winner_seed = seeds.get(winner_id, 0)
            loser_id = t2 if winner_id == t1 else t1
            loser_seed = seeds.get(loser_id, 0)
            is_upset = bool(winner_seed > loser_seed)

            # Region label for the game
            if rkey in ("R64", "R32", "S16", "E8"):
                game_region = regions.get(t1, regions.get(t2, ""))
            elif rkey == "F4":
                game_region = F4_REGION_LABELS[0] if game_num == 1 else F4_REGION_LABELS[1]
            else:
                game_region = CHAMP_LABEL

            games.append(
                {
                    "team1": fmt_team(seed1, name1),
                    "team2": fmt_team(seed2, name2),
                    "team1_id": t1,
                    "team2_id": t2,
                    "team1_seed": seed1,
                    "team2_seed": seed2,
                    "team1_rating": round(rating1, 4),
                    "team2_rating": round(rating2, 4),
                    "winner": team_names.get(winner_id, winner_id),
                    "winner_id": winner_id,
                    "winner_seed": winner_seed,
                    "win_prob": win_prob,
                    "is_upset": is_upset,
                    "region": game_region,
                    "round": display_name,
                }
            )
            next_round.append(winner_id)

        rounds_output.append({"round_name": display_name, "games": games})
        current = next_round

    return rounds_output


def main():
    seeds, regions = load_seeds_and_regions(YEAR)
    if not seeds:
        print(f"ERROR: no seeds found for {YEAR}")
        sys.exit(1)

    barthag = _load_torvik_barthag(YEAR, seeds)
    round_probs = build_torvik_round_probabilities(seeds, regions, barthag)

    try:
        pick_dist = build_espn_pick_distribution(YEAR, seeds)
    except FileNotFoundError:
        pick_dist = {}

    picks, champion, final_four, ev, _var = construct_bracket(
        mode="exhaustive_champion",
        seeds=seeds,
        regions=regions,
        round_probs=round_probs,
        public_picks=pick_dist,
        risk_level=0.5,
        pool_size=30,
        scoring_system=dict(ESPN_SCORING),
    )

    team_names = load_team_names()

    rounds = build_bracket_json(seeds, regions, barthag, round_probs, picks, team_names)

    output = {
        "season": YEAR,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "model": "Torvik Barthag — Exhaustive Champion Search",
        "n_simulations": 10000,
        "rounds": rounds,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Champion: {team_names.get(champion, champion)}")
    print(f"Final Four: {[team_names.get(t, t) for t in final_four]}")
    print(f"Expected points: {ev:.1f}")
    print()
    for rnd in rounds:
        print(f"  {rnd['round_name']}: {len(rnd['games'])} games")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
