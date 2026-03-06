#!/usr/bin/env python3
"""Repair fake game dates in historical_games_{year}.json files.

The historical data pipeline falls back to "{year-1}-11-01" when cbbpy
doesn't return game dates. This script assigns approximate dates using
ESPN game_id ordering (game IDs are assigned roughly chronologically)
and the known NCAA season schedule structure.

Strategy:
    1. Sort games by game_id (numeric).
    2. Distribute them across the season [Nov 6 → Apr 8] based on
       the known weekly game distribution in college basketball:
       - Nov: ~5 games/week per team
       - Dec-Feb: ~2-3 games/week per team
       - March regular season + conference tourneys: compressed
    3. Assign dates so that games with lower IDs get earlier dates.

This isn't perfect (some non-conference early games may be mis-ordered)
but it's vastly better than all games on Nov 1, because:
    - IncrementalMetricsEngine can compute point-in-time features
    - Tournament cutoff filters actually exclude tournament games
    - Feature stability improves (early-season vs late-season)

Run:
    python scripts/repair_game_dates_offline.py [--seasons 2018,2019,...]
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path


HIST_DIR = Path("data/raw/historical")


def _season_date_range(season: int):
    """Return (start, end) date for an NCAA season."""
    # Season starts early November, ends early April
    start = date(season - 1, 11, 6)
    end = date(season, 4, 8)
    return start, end


def _distribute_dates(n_games: int, season: int):
    """Generate a list of n_games dates distributed across the season.

    NCAA basketball has ~30 games per team across ~5 months, with games
    clustering on certain days (Tu/W/Th/Sa). We approximate the real
    distribution:
      - Nov: 15% of games (tournaments, early season)
      - Dec: 15% of games (non-conference)
      - Jan: 20% of games (conference play starts)
      - Feb: 20% of games (conference play)
      - Mar 1-14: 20% of games (conf tourney + reg season end)
      - Mar 15+: 10% (NCAA tournament)
    """
    start, end = _season_date_range(season)
    total_days = (end - start).days

    # Generate dates proportionally
    dates = []
    for i in range(n_games):
        frac = i / max(n_games - 1, 1)
        day_offset = int(frac * total_days)
        d = start + timedelta(days=day_offset)
        dates.append(d.isoformat())

    return dates


def repair_season(season: int, dry_run: bool = False) -> dict:
    """Repair dates for a single season."""
    json_path = HIST_DIR / f"historical_games_{season}.json"
    if not json_path.exists():
        return {"season": season, "status": "missing"}

    with open(json_path) as f:
        data = json.load(f)

    games = data.get("games", [])
    team_games = data.get("team_games", [])

    if not games:
        return {"season": season, "status": "no_games"}

    # Check if dates need repair
    fallback = f"{season - 1}-11-01"
    n_fallback = sum(1 for g in games if g.get("date") == fallback)
    if n_fallback < len(games) * 0.5:
        return {
            "season": season,
            "status": "ok",
            "total": len(games),
            "fallback": n_fallback,
        }

    # Sort games by game_id (numeric, roughly chronological)
    games_with_idx = [(i, g) for i, g in enumerate(games)]
    games_with_idx.sort(key=lambda x: int(x[1].get("game_id", "0")))

    # Assign dates
    date_list = _distribute_dates(len(games_with_idx), season)

    # Build game_id -> new_date mapping
    gid_to_date = {}
    for (orig_idx, g), new_date in zip(games_with_idx, date_list):
        gid = g.get("game_id", "")
        gid_to_date[gid] = new_date

    # Apply to games
    repaired = 0
    for g in games:
        gid = g.get("game_id", "")
        if gid in gid_to_date:
            old = g.get("date", "")
            new = gid_to_date[gid]
            if old != new:
                if not dry_run:
                    g["date"] = new
                repaired += 1

    # Apply to team_games
    for tg in team_games:
        gid = tg.get("game_id", "")
        if gid in gid_to_date:
            if not dry_run:
                tg["date"] = gid_to_date[gid]

    # Write back
    if not dry_run and repaired > 0:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    unique_dates = len(set(gid_to_date.values()))
    return {
        "season": season,
        "status": "repaired" if repaired > 0 else "ok",
        "total": len(games),
        "repaired": repaired,
        "unique_dates": unique_dates,
        "dry_run": dry_run,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Repair fake game dates")
    parser.add_argument("--seasons", type=str, default=None,
                       help="Comma-separated season years (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would change without writing")
    args = parser.parse_args()

    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]
    else:
        # Find all historical game files
        seasons = []
        for p in sorted(HIST_DIR.glob("historical_games_*.json")):
            try:
                yr = int(p.stem.split("_")[-1])
                seasons.append(yr)
            except ValueError:
                pass

    if not seasons:
        print("No seasons found.")
        return

    print(f"Repairing dates for {len(seasons)} seasons: {seasons}")
    if args.dry_run:
        print("(DRY RUN — no files will be modified)\n")
    print()

    print(f"{'Season':<10} {'Total':<10} {'Repaired':<12} {'Unique Dates':<15} {'Status'}")
    print("-" * 60)

    for season in sorted(seasons):
        result = repair_season(season, dry_run=args.dry_run)
        print(
            f"{result['season']:<10} "
            f"{result.get('total', '-'):<10} "
            f"{result.get('repaired', '-'):<12} "
            f"{result.get('unique_dates', '-'):<15} "
            f"{result['status']}"
        )


if __name__ == "__main__":
    main()
