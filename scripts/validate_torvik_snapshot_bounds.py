#!/usr/bin/env python3
"""Prove each Torvik snapshot reflects only games played before its own date.

WHY CHECK C IS NOT ENOUGH. audit_snapshot_boundary check C shows that a
season's first and last snapshots differ, which rules out the crude failure --
one scrape copied under many date labels. It does not establish the thing that
actually matters: that the snapshot dated 2023-12-09 is bounded by 2023-12-09.
A source that silently ignored the `end` parameter, or a scraper that passed
the wrong window, would still produce snapshots that differ from each other
while every one of them carried end-of-season information.

THE TEST. Torvik's effective_fg_pct is a raw rate, so it is independently
computable from the Kaggle box scores: (FGM + 0.5 * FGM3) / FGA. For each
snapshot we compute that same rate two ways -- over games before the snapshot
date, and over the whole season -- and ask which one the snapshot resembles.

A correctly bounded November snapshot must match the November computation far
more closely than the full-season one. If the two match equally well the test
is uninformative (by March they converge by construction, which is why the
verdict is driven by the EARLY snapshots where the two computations genuinely
disagree).

This is a stronger statement than "the values move", and it is the assertion
the point-in-time story actually rests on.

Run: python3 scripts/validate_torvik_snapshot_bounds.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.features.custom_ratings import ratings_to_canonical  # noqa: E402

KAGGLE = REPO / "data" / "kaggle"
HIST = REPO / "data" / "raw" / "historical"

# A snapshot this early in the season is where bounded and unbounded values
# differ most, so it carries the signal. Later ones converge on the full season
# no matter what, and would dilute the verdict.
EARLY_DAY_MAX = 60


def load_shooting_rows() -> dict[int, list[tuple]]:
    """season -> [(day, team, fgm, fga, fgm3)], both teams per game."""
    path = KAGGLE / "MRegularSeasonDetailedResults.csv"
    out: dict[int, list[tuple]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            s, day = int(r["Season"]), int(r["DayNum"])
            out[s].append((day, int(r["WTeamID"]), int(r["WFGM"]), int(r["WFGA"]), int(r["WFGM3"])))
            out[s].append((day, int(r["LTeamID"]), int(r["LFGM"]), int(r["LFGA"]), int(r["LFGM3"])))
    return out


def efg_through(rows: list[tuple], day_limit: int | None) -> dict[int, float]:
    """eFG% per team over games before day_limit (None = whole season)."""
    fgm: dict[int, int] = defaultdict(int)
    fga: dict[int, int] = defaultdict(int)
    fgm3: dict[int, int] = defaultdict(int)
    for day, team, m, a, m3 in rows:
        if day_limit is not None and day >= day_limit:
            continue
        fgm[team] += m
        fga[team] += a
        fgm3[team] += m3
    return {t: (fgm[t] + 0.5 * fgm3[t]) / fga[t] for t in fga if fga[t]}


def main() -> int:
    dayzero = {}
    with open(KAGGLE / "MSeasons.csv") as f:
        for r in csv.DictReader(f):
            dayzero[int(r["Season"])] = dt.datetime.strptime(r["DayZero"], "%m/%d/%Y").date()

    shooting = load_shooting_rows()
    seasons = sorted(s for s in shooting if (HIST / f"torvik_{s}.json").exists())

    print(
        f"\n{'season':>7} {'snap':>11} {'day':>4} {'n':>4}   "
        f"{'|Δ| vs bounded':>14} {'|Δ| vs full-season':>19}   verdict"
    )

    early_bounded: list[float] = []
    early_full: list[float] = []
    rows_shown = 0
    failures: list[str] = []

    for season in seasons:
        data = json.loads((HIST / f"torvik_{season}.json").read_text())
        snaps = sorted(data.get("four_factors_snapshots") or [], key=lambda s: s["date"])
        if not snaps:
            continue
        rows = shooting[season]
        full = efg_through(rows, None)

        # kaggle ids -> canonical, once per season
        canon_of_kid = {}
        for canon, kid in ratings_to_canonical({k: float(k) for k in full}).items():
            canon_of_kid[int(kid)] = canon

        for snap in snaps:
            day = (dt.date.fromisoformat(snap["date"]) - dayzero[season]).days
            if day > EARLY_DAY_MAX:
                continue
            bounded = efg_through(rows, day)
            tv = {
                k: v["effective_fg_pct"]
                for k, v in snap["data"].items()
                if isinstance(v, dict) and isinstance(v.get("effective_fg_pct"), (int, float))
            }
            db, df = [], []
            for kid, canon in canon_of_kid.items():
                if canon not in tv or kid not in bounded or kid not in full:
                    continue
                db.append(abs(tv[canon] - bounded[kid]))
                df.append(abs(tv[canon] - full[kid]))
            if len(db) < 50:
                continue
            mb, mf = statistics.fmean(db), statistics.fmean(df)
            early_bounded.append(mb)
            early_full.append(mf)
            ok = mb < mf
            if not ok:
                failures.append(f"{season} {snap['date']}: bounded {mb:.4f} >= full {mf:.4f}")
            if rows_shown < 18:
                print(
                    f"{season:>7} {snap['date']:>11} {day:>4} {len(db):>4}   "
                    f"{mb:>14.4f} {mf:>19.4f}   {'bounded' if ok else 'FULL-SEASON'}"
                )
                rows_shown += 1

    if not early_bounded:
        print("\nno early snapshots available to test")
        return 1

    mb, mf = statistics.fmean(early_bounded), statistics.fmean(early_full)
    print(f"\n  {len(early_bounded)} early snapshots (day <= {EARLY_DAY_MAX}) across {len(seasons)} seasons")
    print(f"  mean |Δ| vs games-before-date : {mb:.4f}")
    print(f"  mean |Δ| vs whole season      : {mf:.4f}")
    print(f"  ratio                         : {mf / mb:.2f}x")

    if failures:
        print(f"\n{len(failures)} SNAPSHOT(S) resemble the full season more than their own date:")
        for line in failures[:10]:
            print(f"  {line}")
        return 1

    print("\nevery early snapshot matches its own date more closely than the full season")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
