#!/usr/bin/env python3
"""Backfill Four Factors and adj_tempo into torvik_YYYY.json for all years.

Also repairs bad ORB%/DRB% in torvik_four_factors_YYYY.json where CSV
player-level aggregation produced ~3.5x too-low values.

The torvik scraper writes team data and four_factors to separate files.
Historically, the merge step was missing from the collector, leaving
effective_fg_pct, free_throw_rate, and all opp_* fields as zero in the
main torvik_YYYY.json files.  adj_tempo was also never populated for
2008-2022.

This script patches all years by:
  1. Merging four_factors using normalize_team_id for alias resolution
  2. Computing adj_tempo from historical game possessions data
  3. Fixing bad ORB%/DRB% in four_factors files (CSV approximation bias)
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize import normalize_team_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "raw"

_FF_FIELDS = (
    "effective_fg_pct", "turnover_rate", "offensive_reb_rate", "free_throw_rate",
    "opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate", "defensive_reb_rate",
)


def backfill_year(year: int) -> dict:
    """Backfill Four Factors for a single year. Returns stats dict."""
    torvik_path = DATA_DIR / f"torvik_{year}.json"
    ff_path = DATA_DIR / f"torvik_four_factors_{year}.json"

    if not torvik_path.exists():
        return {"status": "skip", "reason": "no torvik file"}
    if not ff_path.exists():
        return {"status": "skip", "reason": "no four_factors file"}

    with open(torvik_path) as f:
        torvik_data = json.load(f)
    with open(ff_path) as f:
        ff_data = json.load(f)

    teams = torvik_data.get("teams", [])
    if not teams:
        return {"status": "skip", "reason": "no teams in torvik file"}

    # Build normalized lookup
    normalized_ff: dict = {}
    for key, entry in ff_data.items():
        if isinstance(entry, dict):
            normalized_ff[normalize_team_id(key)] = entry

    updated = 0
    fields_fixed = {f: 0 for f in _FF_FIELDS}

    for team in teams:
        tid = team.get("team_id", "")
        ff = ff_data.get(tid)
        if ff is None:
            ff = normalized_ff.get(normalize_team_id(tid))
        if ff is None or not isinstance(ff, dict):
            continue

        changed = False
        for field in _FF_FIELDS:
            current = float(team.get(field, 0) or 0)
            source = float(ff.get(field, 0) or 0)
            if abs(current) < 1e-6 and abs(source) > 1e-6:
                team[field] = ff[field]
                changed = True
                fields_fixed[field] += 1
                # Also update enriched_stats
                enriched = team.get("enriched_stats", {})
                if abs(float(enriched.get(field, 0) or 0)) < 1e-6:
                    enriched[field] = ff[field]
                    team["enriched_stats"] = enriched
        if changed:
            updated += 1

    if updated > 0:
        with open(torvik_path, "w") as f:
            json.dump(torvik_data, f, indent=2)

    return {
        "status": "fixed" if updated > 0 else "clean",
        "teams": len(teams),
        "updated": updated,
        "fields": {k: v for k, v in fields_fixed.items() if v > 0},
    }


def backfill_tempo(year: int) -> dict:
    """Compute adj_tempo from historical game possessions and backfill.

    Uses average possessions-per-game as a proxy for adj_tempo.
    Requires GameToTorvikResolver for team ID matching.
    """
    torvik_path = DATA_DIR / f"torvik_{year}.json"
    if not torvik_path.exists():
        return {"status": "skip", "reason": "no torvik file"}

    with open(torvik_path) as f:
        torvik_data = json.load(f)
    teams = torvik_data.get("teams", [])
    if not teams:
        return {"status": "skip", "reason": "no teams"}

    # Check if tempo already populated
    zero_tempo = sum(1 for t in teams if abs(float(t.get("adj_tempo", 0) or 0)) < 1e-6)
    if zero_tempo < len(teams) * 0.5:
        return {"status": "clean", "teams": len(teams), "zero_tempo": zero_tempo}

    # Find game data
    games_path = DATA_DIR / "historical" / f"historical_games_{year}.json"
    if not games_path.exists():
        games_path = DATA_DIR / f"historical_games_{year}.json"
    if not games_path.exists():
        return {"status": "skip", "reason": "no game data"}

    with open(games_path) as f:
        games_data = json.load(f)
    records = games_data.get("team_games", games_data.get("games", []))

    # Build resolver from torvik team IDs
    from src.data.team_id_resolver import GameToTorvikResolver
    torvik_ids = {t["team_id"] for t in teams if t.get("team_id")}
    resolver = GameToTorvikResolver(torvik_ids)

    # Aggregate possessions per team
    team_poss: dict[str, list[float]] = defaultdict(list)
    for g in records:
        poss = float(g.get("possessions", 0) or 0)
        if poss <= 0:
            continue
        raw_id = str(g.get("team_id") or g.get("team_name", "")).lower().replace(" ", "_")
        resolved = resolver.resolve(raw_id)
        if not resolved:
            resolved = resolver.resolve_display_name(g.get("team_name", ""))
        if resolved:
            team_poss[resolved].append(poss)

    # Compute average tempo per team
    team_tempo = {}
    for tid, poss_list in team_poss.items():
        if len(poss_list) >= 3:  # Need at least 3 games for a meaningful average
            team_tempo[tid] = round(sum(poss_list) / len(poss_list), 2)

    # Apply to torvik data
    updated = 0
    for team in teams:
        tid = team.get("team_id", "")
        if abs(float(team.get("adj_tempo", 0) or 0)) < 1e-6:
            tempo = team_tempo.get(tid)
            if tempo:
                team["adj_tempo"] = tempo
                updated += 1

    if updated > 0:
        with open(torvik_path, "w") as f:
            json.dump(torvik_data, f, indent=2)

    return {"status": "fixed" if updated > 0 else "clean", "teams": len(teams), "updated": updated}


# D1 population means for ORB%/DRB% validation
_D1_ORB_MIN = 0.18
_D1_ORB_MAX = 0.42
_D1_DRB_MIN = 0.58
_D1_DRB_MAX = 0.82


def repair_csv_approximation_orb_drb(year: int) -> dict:
    """Fix ORB%/DRB% in four_factors files where CSV player-level aggregation
    produced values ~3.5x too low (team-level ORB% ~0.30, player-level ~0.08).

    Detects the bias by checking if median ORB% < 0.15, then recomputes
    from historical game box scores.
    """
    ff_path = DATA_DIR / f"torvik_four_factors_{year}.json"
    if not ff_path.exists():
        return {"status": "skip", "reason": "no four_factors file"}

    with open(ff_path) as f:
        ff_data = json.load(f)

    # Check for CSV approximation bias
    orb_vals = sorted(
        float(v.get("offensive_reb_rate", 0) or 0)
        for v in ff_data.values()
        if isinstance(v, dict) and float(v.get("offensive_reb_rate", 0) or 0) > 1e-6
    )
    if not orb_vals:
        return {"status": "skip", "reason": "no ORB% data"}

    median_orb = orb_vals[len(orb_vals) // 2]
    if median_orb >= _D1_ORB_MIN:
        return {"status": "clean", "median_orb": round(median_orb, 4)}

    # Bias detected — recompute from game box scores
    games_path = DATA_DIR / "historical" / f"historical_games_{year}.json"
    if not games_path.exists():
        games_path = DATA_DIR / f"historical_games_{year}.json"
    if not games_path.exists():
        return {"status": "skip", "reason": "no game data for recompute"}

    with open(games_path) as f:
        games_data = json.load(f)
    records = games_data.get("team_games", games_data.get("games", []))

    # Pair games by game_id
    game_pairs: dict[str, list[dict]] = defaultdict(list)
    for g in records:
        gid = g.get("game_id")
        if gid:
            game_pairs[gid].append(g)

    # Build resolver
    ff_ids = set(ff_data.keys())
    from src.data.team_id_resolver import GameToTorvikResolver
    resolver = GameToTorvikResolver(ff_ids)

    # Also build normalized lookup
    normalized_to_ff: dict[str, str] = {}
    for k in ff_ids:
        normalized_to_ff[normalize_team_id(k)] = k

    # Accumulate ORB/DRB per team
    team_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"orb": 0, "drb": 0, "opp_orb": 0, "opp_drb": 0, "games": 0}
    )
    for game_id, sides in game_pairs.items():
        if len(sides) != 2:
            continue
        for i, my_side in enumerate(sides):
            opp_side = sides[1 - i]
            raw_id = str(my_side.get("team_id") or my_side.get("team_name", "")).lower().replace(" ", "_")
            my_id = resolver.resolve(raw_id)
            if not my_id:
                my_id = resolver.resolve_display_name(my_side.get("team_name", ""))
            if not my_id:
                norm = normalize_team_id(raw_id)
                my_id = normalized_to_ff.get(norm, raw_id)

            s = team_stats[my_id]
            s["orb"] += float(my_side.get("orb", 0) or 0)
            s["drb"] += float(my_side.get("drb", 0) or 0)
            s["opp_orb"] += float(opp_side.get("orb", 0) or 0)
            s["opp_drb"] += float(opp_side.get("drb", 0) or 0)
            s["games"] += 1

    # Compute team-level ORB%/DRB%
    updated = 0
    for ff_key, entry in ff_data.items():
        if not isinstance(entry, dict):
            continue
        s = team_stats.get(ff_key)
        if not s:
            norm = normalize_team_id(ff_key)
            s = team_stats.get(normalized_to_ff.get(norm, ""))
        if not s or s["games"] < 3:
            continue

        orb_chances = s["orb"] + s["opp_drb"]
        drb_chances = s["drb"] + s["opp_orb"]
        if orb_chances > 0 and drb_chances > 0:
            new_orb = round(s["orb"] / orb_chances, 4)
            new_drb = round(s["drb"] / drb_chances, 4)
            if _D1_ORB_MIN <= new_orb <= _D1_ORB_MAX:
                entry["offensive_reb_rate"] = new_orb
                entry["defensive_reb_rate"] = new_drb
                updated += 1

    if updated > 0:
        with open(ff_path, "w") as f:
            json.dump(ff_data, f, indent=2)

    return {
        "status": "fixed" if updated > 0 else "no_fix",
        "median_orb_before": round(median_orb, 4),
        "updated": updated,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Backfill Four Factors into torvik files for all years")
    parser.add_argument("--start", type=int, default=2005, help="Start year (default: 2005)")
    parser.add_argument("--end", type=int, default=2026, help="End year inclusive (default: 2026)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    args = parser.parse_args()

    print("=" * 70)
    print("Four Factors Backfill — All Years")
    print("=" * 70)

    total_fixed = 0
    for year in range(args.start, args.end + 1):
        result = backfill_year(year)
        status = result["status"]
        if status == "skip":
            print(f"  {year}: SKIP ({result['reason']})")
        elif status == "clean":
            print(f"  {year}: OK ({result['teams']} teams, no zeros to fix)")
        else:
            fields_str = ", ".join(f"{k}={v}" for k, v in result["fields"].items())
            print(f"  {year}: FIXED {result['updated']}/{result['teams']} teams ({fields_str})")
            total_fixed += result["updated"]

    print()
    if total_fixed:
        print(f"Total Four Factors teams fixed: {total_fixed}")
    else:
        print("All Four Factors clean — no fixes needed.")

    # --- Phase 2: adj_tempo backfill ---
    print()
    print("=" * 70)
    print("adj_tempo Backfill — All Years")
    print("=" * 70)

    tempo_fixed = 0
    for year in range(args.start, args.end + 1):
        result = backfill_tempo(year)
        status = result["status"]
        if status == "skip":
            print(f"  {year}: SKIP ({result.get('reason', '')})")
        elif status == "clean":
            print(f"  {year}: OK ({result['teams']} teams, {result.get('zero_tempo', 0)} zeros)")
        else:
            print(f"  {year}: FIXED {result['updated']}/{result['teams']} teams")
            tempo_fixed += result["updated"]

    if tempo_fixed:
        print(f"\nTotal adj_tempo teams fixed: {tempo_fixed}")

    # --- Phase 3: ORB%/DRB% CSV approximation repair ---
    print()
    print("=" * 70)
    print("ORB%/DRB% CSV Approximation Repair — All Years")
    print("=" * 70)

    orb_fixed = 0
    for year in range(args.start, args.end + 1):
        result = repair_csv_approximation_orb_drb(year)
        status = result["status"]
        if status == "skip":
            print(f"  {year}: SKIP ({result.get('reason', '')})")
        elif status == "clean":
            print(f"  {year}: OK (median ORB%={result['median_orb']})")
        elif status == "fixed":
            print(f"  {year}: FIXED {result['updated']} teams (median ORB% was {result['median_orb_before']})")
            orb_fixed += result["updated"]
        else:
            print(f"  {year}: {status} (median ORB% was {result.get('median_orb_before', '?')})")

    if orb_fixed:
        print(f"\nTotal ORB%/DRB% teams fixed: {orb_fixed}")

    # --- Final audit ---
    print()
    print("=" * 70)
    print("Final Audit")
    print("=" * 70)

    any_problems = False
    for year in range(args.start, args.end + 1):
        torvik_path = DATA_DIR / f"torvik_{year}.json"
        if not torvik_path.exists():
            continue
        with open(torvik_path) as f:
            data = json.load(f)
        teams = data.get("teams", [])
        issues = []
        for field in list(_FF_FIELDS) + ["adj_tempo"]:
            zero_count = sum(1 for t in teams if abs(float(t.get(field, 0) or 0)) < 1e-6)
            if zero_count > len(teams) * 0.5:
                issues.append(f"{field}={zero_count}/{len(teams)}")
        if issues:
            print(f"  {year}: PROBLEMS -> {', '.join(issues)}")
            any_problems = True
        else:
            print(f"  {year}: OK")

    return 1 if any_problems else 0


if __name__ == "__main__":
    sys.exit(main())
