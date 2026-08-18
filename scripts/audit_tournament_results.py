"""Audit tournament_context_{year}.json for bracket-integrity violations.

These files are the backtest's GROUND TRUTH — `actual_winners_by_round` and
`build_actual_outcome` read `team1_won` straight out of them, so a single
transposed game silently mis-scores every bracket in that year's pool.

Checks three invariants that a single-elimination bracket cannot violate:

  A. Every team except the champion has exactly one recorded loss.
  B. Every round-R winner appears again in round R+1 (the champion excepted).
  C. Round sizes are 32/16/8/4/2/1.

A and B are independent views of the same underlying defect, so when they
agree on a game the diagnosis is solid. Where a game log
(`historical_games_{year}.json`) is available the audit also cross-checks the
disputed games against it — that is an entirely separate scrape and settles
which side is right.

Exits non-zero if any violation is found, so it can be wired into CI once the
known defects below are repaired.

Usage:
    python3 scripts/audit_tournament_results.py [--year YYYY]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import HIST_DIR, load_tournament_results
from src.data.normalize import resolve_cbbpy_bridge

ROUNDS = ("R64", "R32", "S16", "E8", "F4", "NCG")
EXPECTED_SIZES = (32, 16, 8, 4, 2, 1)
YEARS = [y for y in range(2010, 2027) if y != 2020]


def _game_log_winner(year: int, a: str, b: str):
    """Independent verdict from the cbbpy game log, or None if unavailable."""
    path = HIST_DIR / f"historical_games_{year}.json"
    tv = HIST_DIR / f"torvik_{year}.json"
    if not path.exists() or not tv.exists():
        return None
    with open(tv) as f:
        canonical = {t["team_id"] for t in json.load(f)["teams"]}
    with open(path) as f:
        games = json.load(f).get("games", [])

    # Resolve the whole log at once so a lower-division school that shares a
    # D1 school's name prefix cannot impersonate it (see
    # memory/cbbpy_team_id_bridge_defect.md). `canonical` is already the full
    # Torvik D1 list, which doubles as the disambiguating universe.
    appearances: dict[str, int] = {}
    for g in games:
        for key in ("team1_id", "team2_id"):
            if g.get(key):
                appearances[g[key]] = appearances.get(g[key], 0) + 1
    bridge_map = resolve_cbbpy_bridge(appearances, canonical)

    def bridge(x):
        return bridge_map.get(x)

    for g in games:
        t1, t2 = bridge(g.get("team1_id", "")), bridge(g.get("team2_id", ""))
        if {t1, t2} == {a, b} and g.get("team1_score") is not None:
            if g["team1_score"] == g["team2_score"]:
                return None
            return (t1, g["team1_score"], t2, g["team2_score"])
    return None


def audit_year(year: int) -> list[str]:
    games = [
        g
        for g in load_tournament_results(year)
        if g.get("round_name") in ROUNDS and g.get("team1_won") is not None
    ]
    if not games:
        return []

    by_round = {r: [g for g in games if g["round_name"] == r] for r in ROUNDS}
    plays = {r: {t for g in by_round[r] for t in (g["team1_id"], g["team2_id"])} for r in ROUNDS}
    wins = {r: {(g["team1_id"] if g["team1_won"] else g["team2_id"]) for g in by_round[r]} for r in ROUNDS}

    losses: dict[str, list] = {}
    for g in games:
        loser = g["team2_id"] if g["team1_won"] else g["team1_id"]
        losses.setdefault(loser, []).append(g)

    issues = []

    # A — nobody loses twice in single elimination.
    for team, gs in sorted(losses.items()):
        if len(gs) > 1:
            rounds = ", ".join(g["round_name"] for g in gs)
            issues.append(f"A  {team}: {len(gs)} recorded losses ({rounds})")
            # The earliest of those losses is the suspect game.
            suspect = min(gs, key=lambda g: ROUNDS.index(g["round_name"]))
            other = suspect["team2_id"] if suspect["team1_id"] == team else suspect["team1_id"]
            verdict = _game_log_winner(year, team, other)
            if verdict:
                t1, s1, t2, s2 = verdict
                real = t1 if s1 > s2 else t2
                recorded = suspect["team1_id"] if suspect["team1_won"] else suspect["team2_id"]
                mark = "CONFIRMS the file is wrong" if real != recorded else "agrees with the file"
                issues.append(
                    f"   game log: {t1} {s1}-{s2} {t2} -> winner {real}; "
                    f"file says {recorded} won {suspect['round_name']} — {mark}"
                )

    # B — a winner must show up in the next round.
    for i, rnd in enumerate(ROUNDS[:-1]):
        nxt = ROUNDS[i + 1]
        for w in sorted(wins[rnd] - plays[nxt]):
            issues.append(f"B  {w}: recorded as winning {rnd} but never appears in {nxt}")

    # C — structural sizes.
    sizes = tuple(len(by_round[r]) for r in ROUNDS)
    if sizes != EXPECTED_SIZES:
        issues.append(f"C  round sizes {sizes}, expected {EXPECTED_SIZES}")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, help="audit a single year")
    args = ap.parse_args()

    years = [args.year] if args.year else YEARS
    total = 0
    for year in years:
        issues = audit_year(year)
        if issues:
            total += sum(1 for i in issues if i[:1] in "ABC")
            print(f"\n{year}:")
            for i in issues:
                print(f"  {i}")

    print(f"\n{'=' * 60}")
    if total:
        print(f"FAIL — {total} bracket-integrity violation(s) across {len(years)} year(s).")
        print("These files are backtest ground truth; see")
        print("memory/next_steps_pretournament_player_data.md for the verified corrections.")
        return 1
    print(f"OK — no violations across {len(years)} year(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
