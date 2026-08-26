#!/usr/bin/env python3
"""Does travel distance to the venue actually predict tournament outcomes?

WHAT THIS USES
data/kaggle/tournament_locations.json ships pre-computed, round-by-round
travel distance (campus -> that round's venue, great-circle miles) for every
tournament team, 2008-2025 (excluding the cancelled 2020 tournament). It is
authoritative and requires no geocoding: Kaggle already resolved campus and
venue coordinates and computed distance_mi per (year, team, round).

WHY NOT THE UI'S VARIABLE SYSTEM
scripts/build_ui_payload.py's VARIABLES are one fixed number per team per
SEASON (e.g. barthag), diffed once and reused for every round. Travel distance
is round-dependent -- a team's venue (and therefore its travel distance)
changes at every round it survives to. It cannot be represented as a single
season-level column without either picking one round arbitrarily or leaking
how far the team advanced (a team only has a Final Four row if it made the
Final Four). This script instead measures the raw historical relationship
directly, one row per (team, round actually played).

METHOD
For each tournament game, join both teams' distance_mi for that round from
tournament_locations.json, and their actual scoring margin from
tournament_context_{year}.json. Two rows per game (one per team's
perspective), so the relationship is symmetric by construction: own_distance
minus opponent_distance vs. own margin.

Usage:
    python3 scripts/travel_distance_backtest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.normalize import normalize_team_id  # noqa: E402

LOCATIONS_PATH = REPO / "data" / "kaggle" / "tournament_locations.json"
CONTEXT_DIR = REPO / "data" / "raw" / "historical"

# tournament_context_*.json round_name -> tournament_locations.json current_round.
# The championship game gets its own current_round (2) even though it is
# played at the same site as the Final Four (4); First Four ("FF") has no
# entry in the locations file and is excluded, matching the UI's own
# SKIP_ROUNDS (play-in games are not part of the 63-game bracket).
ROUND_TO_CURRENT_ROUND = {
    "R64": 64,
    "R32": 32,
    "S16": 16,
    "E8": 8,
    "F4": 4,
    "NCG": 2,
}

# A handful of team-name spellings in tournament_locations.json normalize to a
# canonical id that differs from the one tournament_context_*.json uses for
# the same school. Both are real ids used elsewhere in the codebase; this is
# just two sources disagreeing about which alias is canonical.
NAME_OVERRIDES = {
    "usc": "southern_california",
    "sam_houston_state": "sam_houston",
    "liu_brooklyn": "long_island_university",
    "southern_mississippi": "southern_miss",
    "detroit": "detroit_mercy",
    "louisiana_lafayette": "louisiana",
    "massachusetts": "umass",
}


def _team_key(name: str) -> str:
    key = normalize_team_id(name)
    return NAME_OVERRIDES.get(key, key)


def load_distances() -> Dict[tuple, float]:
    """Return {(year, team_id, current_round): distance_mi}."""
    payload = json.loads(LOCATIONS_PATH.read_text())
    cols = payload["columns"]
    idx = {c: i for i, c in enumerate(cols)}
    out: Dict[tuple, float] = {}
    for row in payload["data"]:
        year = row[idx["year"]]
        team_id = _team_key(row[idx["team"]])
        current_round = row[idx["current_round"]]
        out[(year, team_id, current_round)] = row[idx["distance_mi"]]
    return out


def iter_games() -> List[Dict[str, Any]]:
    games = []
    for path in sorted(CONTEXT_DIR.glob("tournament_context_*.json")):
        ctx = json.loads(path.read_text())
        for g in ctx.get("results", {}).get("games", []):
            games.append(g)
    return games


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx**0.5 * vy**0.5)


def main() -> int:
    distances = load_distances()
    games = iter_games()

    # One row per team's perspective: (distance_diff, margin, closer_won, round_name).
    rows = []
    skipped_no_round = 0
    skipped_no_distance = 0

    for g in games:
        round_name = g.get("round_name")
        current_round = ROUND_TO_CURRENT_ROUND.get(round_name)
        if current_round is None:
            skipped_no_round += 1  # FF (First Four), not part of the 63-game bracket
            continue
        year = g.get("year")
        a, b = g.get("team1_id"), g.get("team2_id")
        s1, s2 = g.get("team1_score"), g.get("team2_score")
        if s1 is None or s2 is None:
            continue

        da = distances.get((year, a, current_round))
        db = distances.get((year, b, current_round))
        if da is None or db is None:
            skipped_no_distance += 1
            continue

        rows.append((da - db, s1 - s2, round_name))
        rows.append((db - da, s2 - s1, round_name))

    print(f"{len(games)} bracket games available")
    print(f"skipped (First Four / no round mapping): {skipped_no_round}")
    print(f"skipped (missing distance for a team): {skipped_no_distance}")
    print(f"{len(rows) // 2} games with both teams' travel distance resolved")
    print()

    diffs = [r[0] for r in rows]
    margins = [r[1] for r in rows]
    r = pearson(diffs, margins)
    print(f"corr(distance_diff, margin), all rounds: {r:.4f}" if r is not None else "insufficient data")

    closer_wins = sum(1 for d, m, _ in rows if d < 0 and m > 0)
    closer_games = sum(1 for d, _, _ in rows if d < 0)
    if closer_games:
        print(f"win rate when strictly closer to the venue: {closer_wins / closer_games:.1%} ({closer_games} rows)")

    print()
    print(f"{'round':6s} {'n_games':>8s} {'corr':>8s} {'closer_win%':>12s}")
    for rname in ("R64", "R32", "S16", "E8", "F4", "NCG"):
        sub = [r for r in rows if r[2] == rname]
        n_games = len(sub) // 2
        sub_diffs = [s[0] for s in sub]
        sub_margins = [s[1] for s in sub]
        rr = pearson(sub_diffs, sub_margins)
        sub_closer = [s for s in sub if s[0] < 0]
        win_pct = (sum(1 for s in sub_closer if s[1] > 0) / len(sub_closer)) if sub_closer else None
        print(
            f"{rname:6s} {n_games:8d} "
            f"{(f'{rr:.3f}' if rr is not None else '   n/a'):>8s} "
            f"{(f'{win_pct:.1%}' if win_pct is not None else '   n/a'):>12s}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
