"""Build unified odds JSON files from all raw sources.

Merges SBRO (2008-2022), Covers (2023-2026), and SBR (2025-2026)
into one normalized file per season under data/processed/betting_odds/.

Usage:
    python scripts/build_unified_odds.py
    python scripts/build_unified_odds.py --season 2023
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scrapers.unified_odds import (  # noqa: E402
    UnifiedGameOdds,
    normalize_from_covers,
    normalize_from_sbr,
    normalize_from_sbro,
)

RAW_DIR = "data/raw/betting_odds"
OUT_DIR = "data/processed/betting_odds"

# Source priority for dedup (lower = preferred)
SOURCE_PRIORITY = {"sbro": 1, "covers": 2, "sbr": 3}


def _load_raw_games(filepath: str, source: str) -> list[dict]:
    """Load raw games from a source JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("games", [])


def _find_raw_files(season: int) -> list[tuple[str, str]]:
    """Find all raw odds files for a season. Returns [(filepath, source), ...]."""
    files = []

    # SBRO: full season or tournament
    for pattern in [f"sbro_game_odds_{season}.json", f"sbro_game_odds_{season}_tournament.json"]:
        path = os.path.join(RAW_DIR, pattern)
        if os.path.isfile(path):
            files.append((path, "sbro"))
            break  # prefer full season over tournament-only

    # Covers: full season or tournament
    for pattern in [f"covers_game_odds_{season}_fullseason.json", f"covers_game_odds_{season}.json"]:
        path = os.path.join(RAW_DIR, pattern)
        if os.path.isfile(path):
            files.append((path, "covers"))
            break

    # SBR
    path = os.path.join(RAW_DIR, f"sbr_game_odds_{season}.json")
    if os.path.isfile(path):
        files.append((path, "sbr"))

    return files


def _build_season(season: int) -> int:
    """Build unified odds for one season. Returns game count."""
    raw_files = _find_raw_files(season)
    if not raw_files:
        print(f"  {season}: no raw files found")
        return 0

    # Normalize all games
    normalizers = {
        "sbro": normalize_from_sbro,
        "covers": normalize_from_covers,
        "sbr": normalize_from_sbr,
    }

    all_games: list[UnifiedGameOdds] = []
    source_counts: dict[str, int] = defaultdict(int)

    for filepath, source in raw_files:
        raw_games = _load_raw_games(filepath, source)
        normalizer = normalizers[source]
        for raw in raw_games:
            try:
                game = normalizer(raw)
                all_games.append(game)
                source_counts[source] += 1
            except Exception as e:
                pass  # skip malformed records

    # Dedup by (game_date, home_team_id, away_team_id), keeping highest-priority source
    seen: dict[tuple, UnifiedGameOdds] = {}
    for game in all_games:
        key = (game.game_date, game.home_team_id, game.away_team_id)
        existing = seen.get(key)
        if existing is None:
            seen[key] = game
        elif SOURCE_PRIORITY.get(game.source, 99) < SOURCE_PRIORITY.get(existing.source, 99):
            seen[key] = game

    deduped = sorted(seen.values(), key=lambda g: (g.game_date, g.home_team_id))

    # Write output
    os.makedirs(OUT_DIR, exist_ok=True)
    output = {
        "season": season,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_games": len(deduped),
        "sources": dict(source_counts),
        "games": [asdict(g) for g in deduped],
    }

    outpath = os.path.join(OUT_DIR, f"unified_odds_{season}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    unique_teams = set()
    for g in deduped:
        unique_teams.add(g.home_team_id)
        unique_teams.add(g.away_team_id)

    spreads = sum(1 for g in deduped if g.spread != 0)
    mls = sum(1 for g in deduped if g.moneyline_home != 0)

    sources_str = ", ".join(f"{s}={c}" for s, c in sorted(source_counts.items()))
    print(
        f"  {season}: {len(deduped)} games ({sources_str}), "
        f"{len(unique_teams)} teams, {spreads} spreads, {mls} MLs"
    )
    return len(deduped)


def main():
    parser = argparse.ArgumentParser(description="Build unified odds files")
    parser.add_argument("--season", type=int, help="Single season")
    args = parser.parse_args()

    if args.season:
        seasons = [args.season]
    else:
        # Auto-detect available seasons from raw files
        seasons = set()
        for f in os.listdir(RAW_DIR):
            m = re.search(r'(\d{4})', f)
            if m:
                seasons.add(int(m.group(1)))
        seasons = sorted(s for s in seasons if 2008 <= s <= 2026)

    print(f"Building unified odds → {OUT_DIR}/")
    print(f"  Seasons: {seasons}")
    print()

    total = 0
    for season in seasons:
        total += _build_season(season)

    print(f"\nDone. {total} total games across {len(seasons)} seasons.")


if __name__ == "__main__":
    main()
