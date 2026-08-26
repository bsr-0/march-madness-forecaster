#!/usr/bin/env python3
"""Assert the per-date Kaggle features equal the shipping season-final ones.

WHY THIS IS THE TEST THAT MATTERS. point_in_time_kaggle mirrors
generate_team_stats_table formula for formula so that regular-season rows and
tournament rows sit on one scale. Mirroring by hand is exactly the kind of
thing that drifts silently: a different rounding, population stdev vs sample,
a None where the original writes 0.0. Evaluated at the last day of a season
the two must agree exactly, so this compares them and fails loudly if not.

A disagreement here is not cosmetic. It would put a scale discontinuity at the
boundary between a March row and a February row -- the same failure class as
the torvik vintage split that check B caught, and just as invisible in any
downstream metric.

These features come from the Kaggle CSVs, not from Torvik, so this comparison
is unaffected by the torvik vintage reconciliation and can run before
team_stats_by_year.json is regenerated.

Run: python3 scripts/validate_point_in_time_kaggle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.features.custom_ratings import ratings_to_canonical  # noqa: E402
from src.data.features.point_in_time_kaggle import (  # noqa: E402
    box_profile,
    detailed_before,
    form_stats,
    load_detailed_games,
    season_is_complete,
)
from src.data.features.point_in_time_ratings import (  # noqa: E402
    games_before,
    load_season_games,
)

STATS_PATH = REPO / "docs" / "data" / "team_stats_by_year.json"

FORM_FIELDS = [
    "games_played",
    "reg_season_margin_avg",
    "reg_season_margin_std",
    "close_game_rate",
    "close_game_win_rate",
]
BOX_FIELDS = [
    "three_pt_rate",
    "three_pt_pct",
    "opp_three_pt_pct",
    "ast_to_ratio",
    "havoc_rate",
    "true_road_win_pct",
]


def canonicalise(table: dict) -> dict:
    """Kaggle-id keyed dict -> canonical-id keyed, preserving the values.

    ratings_to_canonical takes {id: scalar}, so the id is smuggled through as
    the value to recover the mapping in ONE call -- it reads MTeams.csv on
    every invocation, and calling it per team would re-read the file thousands
    of times.
    """
    canon_to_kid = ratings_to_canonical({k: float(k) for k in table})
    return {canon: table[int(kid)] for canon, kid in canon_to_kid.items() if int(kid) in table}


def main() -> int:
    stats = json.loads(STATS_PATH.read_text())["stats_by_year"]
    years = sorted(int(y) for y in stats)

    mismatches: list[str] = []
    compared = 0
    covered_teams = 0
    skipped: list[str] = []

    for year in years:
        compact = load_season_games(year)
        detailed = load_detailed_games(year)
        if not compact:
            continue
        # An incompletely-ingested season is not a mismatch, it is a season
        # this comparison cannot make. generate_team_stats_table falls back to
        # the cbbpy log for these, so the two sides read different sources and
        # disagreeing tells us nothing about the formulas.
        if not season_is_complete(compact):
            skipped.append(f"{year} (Kaggle ends day {max(g.day for g in compact)})")
            continue
        # "the whole season" = one past the last regular-season day
        last = max(g.day for g in compact) + 1
        form = canonicalise(form_stats(games_before(compact, last)))
        box = canonicalise(box_profile(detailed_before(detailed, last))) if detailed else {}

        for row in stats[str(year)]:
            tid = row["team_id"]
            mine = {}
            if tid in form:
                mine.update(form[tid])
            if tid in box:
                mine.update(box[tid])
            if not mine:
                continue
            covered_teams += 1
            for field in FORM_FIELDS + BOX_FIELDS:
                if field not in mine or field not in row:
                    continue
                a, b = mine[field], row[field]
                if a is None and b is None:
                    continue
                compared += 1
                if a is None or b is None:
                    mismatches.append(f"{year} {tid}.{field}: pit={a} final={b}")
                elif abs(float(a) - float(b)) > 1e-9:
                    mismatches.append(f"{year} {tid}.{field}: pit={a} final={b}")

    print(f"\n{compared:,} field comparisons across {covered_teams:,} team-seasons")
    if skipped:
        print(f"skipped (Kaggle not fully ingested): {', '.join(skipped)}")
    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES); first 15:")
        for m in mismatches[:15]:
            print(f"  {m}")
        # group by field so a systematic formula difference is obvious
        by_field: dict[str, int] = {}
        for m in mismatches:
            by_field[m.split(".")[1].split(":")[0]] = by_field.get(m.split(".")[1].split(":")[0], 0) + 1
        print("\n  by field:", dict(sorted(by_field.items(), key=lambda kv: -kv[1])))
        return 1

    print("per-date features reproduce the season-final values exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
