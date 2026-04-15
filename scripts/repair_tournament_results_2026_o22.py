#!/usr/bin/env python3
"""Repair malformed data/raw/historical/tournament_results_2026.json.

Closes COUNCIL_LESSONS.md §2 O22.

The 2026 file was ingested with corrupted ``round_name`` labels (49×NCG,
4×R68, 0×R64, 0×R32, correct 8/4/2 for S16/E8/F4). Root cause: the ESPN
event-note headline matcher in ``src/data/scrapers/tournament_results.py``
collapsed every round whose headline contains "championship" but not a
more-specific round phrase onto ``NCG``. The R64 → R32 → S16 → E8 → F4
ordering inside the file is intact (verified via team-continuity audit —
R32 teams are exactly the R64 winners; S16 teams are exactly the R32
winners; F4 winners go to NCG). This lets us re-label deterministically
without re-scraping.

Repair:
    - idx 0..3   round_name R68 → FF        (canonical first-four label)
    - idx 4..35  round_name NCG → R64       (32 games)
    - idx 36..51 round_name NCG → R32       (16 games)
    - idx 52..59 round_name S16             (unchanged)
    - idx 60..63 round_name E8              (unchanged)
    - idx 64..65 round_name F4              (unchanged); region "" → National
    - idx 66     round_name NCG             (unchanged); region "" → National

Team-continuity invariants asserted BEFORE the re-label is written:
    - FF winners ⊆ R64 teams
    - R64 winners = R32 teams     (exactly 32, bijection)
    - R32 winners = S16 teams     (exactly 16, bijection)
    - S16 winners = E8 teams      (exactly 8,  bijection)
    - E8  winners = F4 teams      (exactly 4,  bijection)
    - F4  winners = NCG teams     (exactly 2,  bijection)

If any invariant fails, the script raises without touching disk.

Usage:
    python scripts/repair_tournament_results_2026_o22.py
    python scripts/repair_tournament_results_2026_o22.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_FILE = _PROJECT_ROOT / "data" / "raw" / "historical" / "tournament_results_2026.json"

EXPECTED_MALFORMED_COUNTS = {"R68": 4, "NCG": 49, "S16": 8, "E8": 4, "F4": 2}
CANONICAL_COUNTS_POST_2011 = {"FF": 4, "R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}


def _winner(game: dict) -> str:
    return game["team1_id"] if game["team1_won"] else game["team2_id"]


def _teams(game: dict) -> set[str]:
    return {game["team1_id"], game["team2_id"]}


def _assert_malformed_shape(games: list[dict]) -> None:
    """Refuse to run against anything other than the exact 2026 defect."""
    if len(games) != 67:
        raise SystemExit(f"expected 67 games, got {len(games)}")

    counts = Counter(g["round_name"] for g in games)
    if dict(counts) != EXPECTED_MALFORMED_COUNTS:
        raise SystemExit(
            f"file is not the expected O22 defect shape. got {dict(counts)}, expected {EXPECTED_MALFORMED_COUNTS}"
        )

    # Run-structure check: malformed file must have contiguous runs
    # R68(4) | NCG(48) | S16(8) | E8(4) | F4(2) | NCG(1) in that order.
    runs = []
    start = 0
    for i in range(1, len(games)):
        if games[i]["round_name"] != games[start]["round_name"]:
            runs.append((games[start]["round_name"], start, i - 1))
            start = i
    runs.append((games[start]["round_name"], start, len(games) - 1))

    expected_runs = [
        ("R68", 0, 3),
        ("NCG", 4, 51),
        ("S16", 52, 59),
        ("E8", 60, 63),
        ("F4", 64, 65),
        ("NCG", 66, 66),
    ]
    if runs != expected_runs:
        raise SystemExit(f"unexpected run structure: {runs}")


def _assert_team_continuity(games: list[dict], boundaries: dict[str, tuple[int, int]]) -> None:
    """Assert winner-of-round-N = teams-of-round-N+1 for every bracket edge."""

    def _round_winners(lo: int, hi: int) -> set[str]:
        return {_winner(games[i]) for i in range(lo, hi + 1)}

    def _round_teams(lo: int, hi: int) -> set[str]:
        out: set[str] = set()
        for i in range(lo, hi + 1):
            out |= _teams(games[i])
        return out

    ff_lo, ff_hi = boundaries["FF"]
    r64_lo, r64_hi = boundaries["R64"]
    r32_lo, r32_hi = boundaries["R32"]
    s16_lo, s16_hi = boundaries["S16"]
    e8_lo, e8_hi = boundaries["E8"]
    f4_lo, f4_hi = boundaries["F4"]
    ncg_lo, ncg_hi = boundaries["NCG"]

    # FF winners advance into R64 (subset, not bijection — only 4 of 64).
    ff_winners = _round_winners(ff_lo, ff_hi)
    r64_teams = _round_teams(r64_lo, r64_hi)
    missing = ff_winners - r64_teams
    if missing:
        raise SystemExit(f"FF winners not found in R64: {missing}")

    # R64 → R32 bijection
    r64_winners = _round_winners(r64_lo, r64_hi)
    r32_teams = _round_teams(r32_lo, r32_hi)
    if r64_winners != r32_teams:
        raise SystemExit(
            f"R64 winners ≠ R32 teams. "
            f"Only in R64 winners: {r64_winners - r32_teams}; "
            f"only in R32 teams: {r32_teams - r64_winners}"
        )

    # R32 → S16 bijection
    r32_winners = _round_winners(r32_lo, r32_hi)
    s16_teams = _round_teams(s16_lo, s16_hi)
    if r32_winners != s16_teams:
        raise SystemExit("R32 winners ≠ S16 teams")

    # S16 → E8 bijection
    s16_winners = _round_winners(s16_lo, s16_hi)
    e8_teams = _round_teams(e8_lo, e8_hi)
    if s16_winners != e8_teams:
        raise SystemExit("S16 winners ≠ E8 teams")

    # E8 → F4 bijection
    e8_winners = _round_winners(e8_lo, e8_hi)
    f4_teams = _round_teams(f4_lo, f4_hi)
    if e8_winners != f4_teams:
        raise SystemExit("E8 winners ≠ F4 teams")

    # F4 → NCG bijection
    f4_winners = _round_winners(f4_lo, f4_hi)
    ncg_teams = _round_teams(ncg_lo, ncg_hi)
    if f4_winners != ncg_teams:
        raise SystemExit("F4 winners ≠ NCG teams")


def _apply_relabel(games: list[dict]) -> list[dict]:
    """Write the canonical taxonomy + fix F4/NCG region."""
    out = [dict(g) for g in games]

    # Round-name relabel
    for i in range(0, 4):
        out[i]["round_name"] = "FF"
    for i in range(4, 36):
        out[i]["round_name"] = "R64"
    for i in range(36, 52):
        out[i]["round_name"] = "R32"
    # idx 52..65 already correct (S16/E8/F4); idx 66 already NCG.

    # Region normalization for F4 + NCG (match 2025 canonical "National").
    for i in range(64, 66):
        if not out[i].get("region"):
            out[i]["region"] = "National"
    if not out[66].get("region"):
        out[66]["region"] = "National"

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Audit + print the planned repair; do not write.")
    parser.add_argument("--path", default=str(_DATA_FILE), help="Path to tournament_results_2026.json")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1

    with path.open() as f:
        doc = json.load(f)

    games = doc["games"]
    _assert_malformed_shape(games)

    # Post-repair boundaries — passed into the continuity check as the
    # *intended* layout. If the continuity check passes under this layout,
    # the relabel is safe.
    boundaries = {
        "FF": (0, 3),
        "R64": (4, 35),
        "R32": (36, 51),
        "S16": (52, 59),
        "E8": (60, 63),
        "F4": (64, 65),
        "NCG": (66, 66),
    }
    _assert_team_continuity(games, boundaries)

    repaired = _apply_relabel(games)

    # Post-condition: canonical taxonomy exactly matches.
    counts = Counter(g["round_name"] for g in repaired)
    if dict(counts) != CANONICAL_COUNTS_POST_2011:
        raise SystemExit(f"post-relabel sanity check failed: got {dict(counts)}, expected {CANONICAL_COUNTS_POST_2011}")

    print("pre-repair counts :", EXPECTED_MALFORMED_COUNTS)
    print("post-repair counts:", dict(counts))
    print("team-continuity invariants: PASSED (FF→R64 subset; R64/R32/S16/E8/F4→NCG bijections)")

    if args.dry_run:
        print("dry-run: no file written")
        return 0

    doc["games"] = repaired
    with path.open("w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
