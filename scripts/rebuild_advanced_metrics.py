#!/usr/bin/env python3
"""Rebuild advanced_metrics_2026.json from existing historical game data.

Reads team_games from historical_games_2026.json, runs the
PublicAdvancedMetricsBuilder, then merges real t_rank values from
torvik_2026.json using GameToTorvikResolver for mascot-suffixed IDs.
"""

import json
import sys
from pathlib import Path

# Add project root to path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features.public_advanced_metrics import PublicAdvancedMetricsBuilder
from src.data.normalize import normalize_team_id
from src.data.team_id_resolver import GameToTorvikResolver


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"

    # 1. Load historical game records (team-perspective rows)
    games_path = data_dir / "historical_games_2026.json"
    print(f"Loading games from {games_path}...")
    with open(games_path) as f:
        games_data = json.load(f)

    team_games = games_data.get("team_games", [])
    if not team_games:
        print("ERROR: No team_games found in historical_games_2026.json")
        sys.exit(1)
    print(f"  Found {len(team_games)} team-game records")

    # 2. Load torvik data for real t_rank values
    torvik_path = data_dir / "torvik_2026.json"
    print(f"Loading Torvik data from {torvik_path}...")
    with open(torvik_path) as f:
        torvik_data = json.load(f)

    torvik_teams = torvik_data.get("teams", [])
    torvik_ids = set()
    torvik_trank = {}
    for t in torvik_teams:
        tid = t.get("team_id", t.get("team_name", ""))
        torvik_ids.add(tid)
        trank = t.get("t_rank", 0)
        if trank and trank > 0:
            torvik_trank[tid] = trank
    print(f"  Found {len(torvik_trank)} teams with valid t_rank")

    # 3. Build advanced metrics
    print("Building advanced metrics...")
    builder = PublicAdvancedMetricsBuilder()
    result = builder.build(team_games, teams=torvik_teams)

    built_teams = result.get("teams", [])
    print(f"  Built metrics for {len(built_teams)} teams")

    if not built_teams:
        print("ERROR: Builder returned no teams. Check input data format.")
        sys.exit(1)

    # 4. Merge real t_rank from Torvik where available
    #    Use GameToTorvikResolver to handle mascot-suffixed IDs from cbbpy
    resolver = GameToTorvikResolver(torvik_ids)
    merged_count = 0
    for team in built_teams:
        raw_id = team.get("team_id", "")
        # Try resolver first (handles mascot stripping via prefix matching)
        torvik_id = resolver.resolve(raw_id)
        if torvik_id is None:
            # Fallback: normalize both sides
            torvik_id = normalize_team_id(raw_id)
        if torvik_id in torvik_trank:
            team["t_rank"] = torvik_trank[torvik_id]
            merged_count += 1
    print(f"  Merged real t_rank for {merged_count}/{len(built_teams)} teams")

    # 5. Write output
    out_path = data_dir / "advanced_metrics_2026.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path} ({len(built_teams)} teams)")


if __name__ == "__main__":
    main()
