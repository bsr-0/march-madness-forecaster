"""Read-only status check for the PBP historical backfill.

Summarizes what's already cached per season -- games fetched, whether that
season is complete, and where it left off -- so a restart after a crash (or
just checking progress) doesn't require hand-running snippets against each
pbp_{year}.json. Safe to run at any time, including while
backfill_pbp_history.py is actively running: this only reads files, and a
running Python process isn't affected by files changing on disk underneath
the module code it already loaded.

Usage:
    python3 scripts/pbp_backfill_status.py
    python3 scripts/pbp_backfill_status.py --start-year 2008 --end-year 2026
"""

import argparse
import json
from pathlib import Path

CACHE_DIR = Path("data/raw/historical")


def season_status(year: int) -> dict:
    pbp_path = CACHE_DIR / f"pbp_{year}.json"
    clutch_path = CACHE_DIR / f"clutch_features_{year}.json"

    if not pbp_path.exists():
        return {"year": year, "state": "not started"}

    try:
        with open(pbp_path) as f:
            pbp = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"year": year, "state": f"UNREADABLE ({e})"}

    meta = pbp.get("metadata", {})
    n_games = len(pbp.get("games", []))
    complete = meta.get("complete", False)
    last_date = meta.get("last_completed_date", "?")
    window_end = (meta.get("date_window") or [None, None])[1]

    n_teams = None
    if clutch_path.exists():
        try:
            with open(clutch_path) as f:
                n_teams = len(json.load(f).get("teams", []))
        except (json.JSONDecodeError, OSError):
            n_teams = "UNREADABLE"

    return {
        "year": year,
        "state": "complete" if complete else "in progress",
        "games": n_games,
        "through": last_date,
        "window_end": window_end,
        "clutch_teams": n_teams,
    }


def main(start_year: int, end_year: int) -> None:
    rows = [season_status(y) for y in range(end_year, start_year - 1, -1)]

    print(f"{'year':>6}  {'state':<12} {'games':>7}  {'through':<12} {'clutch teams':>12}")
    print("-" * 60)
    for r in rows:
        # "games" is absent for both "not started" and "UNREADABLE" rows. The
        # latter happens transiently when this runs while the backfill is
        # mid-checkpoint, so it must not crash the summary.
        if "games" not in r:
            print(f"{r['year']:>6}  {r['state']:<12}")
            continue
        clutch = r["clutch_teams"] if r["clutch_teams"] is not None else "-"
        print(f"{r['year']:>6}  {r['state']:<12} {r['games']:>7}  {r['through']:<12} {clutch!s:>12}")

    started = [r for r in rows if r["state"] != "not started"]
    complete = [r for r in rows if r.get("state") == "complete"]
    total_games = sum(r.get("games", 0) or 0 for r in started)
    print("-" * 60)
    print(
        f"{len(started)}/{len(rows)} seasons started, "
        f"{len(complete)}/{len(rows)} complete, "
        f"{total_games} total games fetched so far"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    main(args.start_year, args.end_year)
