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

    round_probs drives each game's displayed win_prob (normalized against
    the two teams' round_probs for that round — the same quantity the
    construction algorithm's EV scorer compares) so the display can never
    contradict the pick for reasons unrelated to strategic risk_level
    trade-offs. barthag is only a fallback for degenerate/missing coverage.

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

            # win_prob is derived from round_probs (the same quantity the
            # construction algorithm's EV scorer actually compares — see
            # _make_ev_scorer / _decide_winner in bracket_construction.py),
            # not an independent Log5 on scalar ratings. Those two views can
            # genuinely disagree: round_probs encode a team's probability of
            # winning across its WHOLE bracket path (survival to this round
            # included), while scalar-rating Log5 only answers "if these two
            # teams played in isolation, who wins." A team can be the
            # round_probs favorite (and get picked) while trailing in a
            # naive head-to-head Log5 comparison, if its path here was
            # otherwise stronger — visible under roster_adj/coach_adj, whose
            # capped per-team adjustment factor is applied post-hoc to
            # Torvik's already-simulated round_probs and can easily flip a
            # close isolated matchup without flipping the larger structural
            # gap that round_probs reflects. Falls back to scalar Log5 only
            # if round_probs has no usable data for either team (e.g. a
            # base with missing round_probs coverage).
            rp1 = round_probs.get(t1, {}).get(rkey, 0.0)
            rp2 = round_probs.get(t2, {}).get(rkey, 0.0)
            if rp1 + rp2 > 1e-9:
                win_prob = round(rp1 / (rp1 + rp2), 4)
            else:
                win_prob = round(_log5(rating1, rating2), 4)

            # Determine game key
            game_num = g_idx // 2 + 1
            if rkey == "R64":
                region = regions.get(t1, regions.get(t2, ""))
                game_key = f"R64_{region}_{seed1}v{seed2}"
            elif rkey in ("R32", "S16"):
                region = regions.get(t1, regions.get(t2, ""))
                # region_game must reset to 1 at the start of every region's
                # games within this round — bracket_construction.py's own
                # key format (_walk_bracket / _enumerate_region_outcomes)
                # is f"{rkey}_{region}_{game_within_region}", 1-indexed.
                # games_per_region is the number of GAMES per region THIS
                # round (4 for R32, 2 for S16) — half the number of teams
                # entering the round per region (8, 4 respectively). This
                # was previously off by one bit (used teams-per-region, not
                # games-per-region), producing negative region_game numbers
                # for every region but the first and silently falling
                # through to the win_prob-guess fallback below for 12/16
                # R32 games and 6/8 S16 games — i.e. most of the bracket
                # outside the first region was never actually showing the
                # optimizer's real pick. Fixed 2026-08-15.
                region_idx = REGION_ORDER.index(region) if region in REGION_ORDER else 0
                games_per_region = 8 >> ROUND_KEYS.index(rkey)  # R32:4, S16:2
                region_game = (g_idx // 2) - region_idx * games_per_region + 1
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
