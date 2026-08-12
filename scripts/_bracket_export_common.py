"""Shared helpers for the live bracket export scripts.

generate_poolaware_bracket.py, generate_region_bracket.py, and
generate_exhaustive_bracket.py all build the same docs/data/bracket_*.json
shape from a seeds/regions/barthag/picks tuple. This module is the one
place that shape gets built, so the three scripts can't drift from each
other.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mc_pool_backtest import (
    POOL_HIST_PATH,
    REGION_ORDER,
    SEED_MATCHUP_ORDER,
    _log5,
    build_espn_pick_distribution,
)
from src.simulation.pool_history_opponent_model import (
    build_pool_pick_distribution,
    load_pool_brackets,
)

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


def resolve_pool_consensus(seeds: Dict[str, int], year: int) -> Tuple[Dict[str, Dict[str, float]], Optional[str]]:
    """Pick distribution for "what did the opponent field actually pick" display.

    Deliberately independent of whatever pick_dist a given construction
    algorithm uses internally for its own optimization — this is always the
    best available real-world reference population (your actual pool's
    history when we have it, ESPN public picks otherwise), shown consistently
    across every strategy tab so the percentages are comparable apples-to-apples.
    """
    try:
        pool_brackets, group_size = load_pool_brackets(POOL_HIST_PATH, year)
        pick_dist = build_pool_pick_distribution(pool_brackets, seeds)
        return pick_dist, f"your pool history (N={group_size})"
    except (FileNotFoundError, KeyError):
        pass
    try:
        pick_dist = build_espn_pick_distribution(year, seeds)
        return pick_dist, "ESPN public picks"
    except FileNotFoundError:
        return {}, None


def build_bracket_json(seeds, regions, barthag, round_probs, picks, team_names, pool_pick_dist=None):
    """Walk the picks dict and construct the full 6-round JSON structure.

    pool_pick_dist (if given) is a Dict[team_id, Dict[round_key, float]] of
    what fraction of the opponent field advanced that team past that round —
    used to annotate each game with what the field actually picked, alongside
    the model's own win_prob for that game.
    """

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

            team1_pool_pct = None
            team2_pool_pct = None
            if pool_pick_dist:
                v1 = pool_pick_dist.get(t1, {}).get(rkey)
                v2 = pool_pick_dist.get(t2, {}).get(rkey)
                team1_pool_pct = round(v1, 4) if v1 is not None else None
                team2_pool_pct = round(v2, 4) if v2 is not None else None

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
                    "team1_pool_pct": team1_pool_pct,
                    "team2_pool_pct": team2_pool_pct,
                }
            )
            next_round.append(winner_id)

        rounds_output.append({"round_name": display_name, "games": games})
        current = next_round

    return rounds_output
