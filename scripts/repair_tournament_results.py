"""Repair transposed games in tournament_context_{year}.json.

These files are the backtest's ground truth. Eight games across six years
record the wrong winner — see memory/tournament_results_ground_truth_defect.md
for the full investigation.

Corrections are DERIVED, not hardcoded: for every game flagged by the bracket
integrity check (a team recorded as losing twice, which single-elimination
forbids), the true result is looked up in `historical_games_{year}.json` — an
independent cbbpy scrape — and the record is rewritten to match. A game is
only touched when the game log actually disagrees with it.

Dry-run by default. Pass --apply to write.

Usage:
    python3 scripts/repair_tournament_results.py            # show the diff
    python3 scripts/repair_tournament_results.py --apply    # write it
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import HIST_DIR
from src.data.normalize import bridge_cbbpy_id

ROUNDS = ("R64", "R32", "S16", "E8", "F4", "NCG")
YEARS = [y for y in range(2010, 2027) if y != 2020]


def _load_context(year: int):
    path = HIST_DIR / f"tournament_context_{year}.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        return path, json.load(f)


def _game_log_result(year: int, a: str, b: str):
    """True (score_a, score_b) from the independent cbbpy scrape, or None."""
    gpath = HIST_DIR / f"historical_games_{year}.json"
    tpath = HIST_DIR / f"torvik_{year}.json"
    if not gpath.exists() or not tpath.exists():
        return None
    with open(tpath) as f:
        canonical = {t["team_id"] for t in json.load(f)["teams"]}
    with open(gpath) as f:
        games = json.load(f).get("games", [])
    cache: dict = {}

    def bridge(x):
        if x not in cache:
            cache[x] = bridge_cbbpy_id(x, canonical)
        return cache[x]

    for g in games:
        t1, t2 = bridge(g.get("team1_id", "")), bridge(g.get("team2_id", ""))
        s1, s2 = g.get("team1_score"), g.get("team2_score")
        if s1 is None or s2 is None or s1 == s2:
            continue
        if (t1, t2) == (a, b):
            return s1, s2
        if (t1, t2) == (b, a):
            return s2, s1
    return None


def find_repairs(year: int):
    """Return (path, ctx, repairs).

    The parsed ``ctx`` is returned alongside the repairs on purpose: the game
    dicts in ``repairs`` are references INTO that object graph, so the caller
    must serialize this same ctx for the mutations to land. Re-parsing the
    file in the caller would silently write back an untouched copy.
    """
    path, ctx = _load_context(year)
    if not ctx:
        return None, None, []
    games = [
        g
        for g in ctx.get("results", {}).get("games", [])
        if g.get("round_name") in ROUNDS and g.get("team1_won") is not None
    ]

    losses: dict[str, list] = {}
    for g in games:
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        losses.setdefault(loser, []).append(g)

    repairs = []
    for team, gs in losses.items():
        if len(gs) < 2:
            continue
        # The earliest recorded loss is the suspect: the team demonstrably
        # played on afterwards, so it cannot have lost that one.
        suspect = min(gs, key=lambda g: ROUNDS.index(g["round_name"]))
        t1, t2 = suspect["team1_id"], suspect["team2_id"]
        truth = _game_log_result(year, t1, t2)
        if truth is None:
            print(f"  {year} {suspect['round_name']} {t1} vs {t2}: NO game-log entry — skipped")
            continue
        s1, s2 = truth
        won = s1 > s2
        if (suspect.get("team1_score"), suspect.get("team2_score"), suspect["team1_won"]) == (s1, s2, won):
            continue  # already correct
        repairs.append((suspect, s1, s2, won))
    return path, ctx, repairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the corrections")
    args = ap.parse_args()

    total = 0
    for year in YEARS:
        path, ctx, repairs = find_repairs(year)
        if not repairs:
            continue
        print(f"\n{year}:")
        for game, s1, s2, won in repairs:
            old_w = game["team1_id"] if game["team1_won"] else game["team2_id"]
            new_w = game["team1_id"] if won else game["team2_id"]
            print(
                f"  {game['round_name']:4s} {game['team1_id']} vs {game['team2_id']}\n"
                f"       before: {game.get('team1_score')}-{game.get('team2_score')}  "
                f"team1_won={game['team1_won']}  winner={old_w}\n"
                f"       after : {s1}-{s2}  team1_won={won}  winner={new_w}"
            )
            total += 1
            if args.apply:
                # Mutating the dict mutates the object inside `ctx`.
                game["team1_score"] = s1
                game["team2_score"] = s2
                game["team1_won"] = won
        if args.apply:
            # indent=2 with no trailing newline reproduces these files
            # byte-for-byte, so the diff shows only the corrected games.
            with open(path, "w") as f:
                json.dump(ctx, f, indent=2)
            print(f"  -> wrote {path}")

    print(f"\n{'=' * 60}")
    if args.apply:
        print(f"Applied {total} correction(s). Re-run scripts/audit_tournament_results.py to verify.")
    else:
        print(f"{total} correction(s) pending. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
