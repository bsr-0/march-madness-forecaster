#!/usr/bin/env python3
"""One-time repair script: re-normalize all team_id fields in cached historical data.

Applies the current normalize_team_id() function (with generic _st -> _state rule
and updated alias table) to all team_id fields in data/raw/historical/*.json files.

Idempotent: running twice produces zero changes the second time.

Usage:
    python scripts/repair_team_ids.py [--dry-run]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.normalize import normalize_team_id  # noqa: E402

HIST_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "historical"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def repair_team_list(teams: list, id_field: str = "team_id") -> tuple[int, list[tuple[str, str]]]:
    """Re-normalize team_id in a list of team dicts. Returns (count_changed, [(old, new)])."""
    changed = 0
    changes = []
    for team in teams:
        if isinstance(team, dict) and id_field in team:
            old = team[id_field]
            new = normalize_team_id(old)
            if new != old:
                team[id_field] = new
                changes.append((old, new))
                changed += 1
    return changed, changes


def repair_game_ids(games: list) -> tuple[int, list[tuple[str, str]]]:
    """Re-normalize team IDs in game records."""
    changed = 0
    changes = []
    id_fields = ["team1_id", "team2_id", "winner_id", "loser_id"]
    for game in games:
        if not isinstance(game, dict):
            continue
        for field in id_fields:
            if field in game and game[field]:
                old = game[field]
                new = normalize_team_id(old)
                if new != old:
                    game[field] = new
                    changes.append((old, new))
                    changed += 1
    return changed, changes


def repair_dict_keys(data: dict) -> tuple[dict, int, list[tuple[str, str]]]:
    """Re-normalize dict keys that are team IDs. Returns (new_dict, count, changes)."""
    new_data = {}
    changed = 0
    changes = []
    for key, value in data.items():
        new_key = normalize_team_id(key)
        if new_key != key:
            changes.append((key, new_key))
            changed += 1
        new_data[new_key] = value
    return new_data, changed, changes


def process_file(filepath: Path, dry_run: bool = False) -> tuple[int, list[tuple[str, str]]]:
    """Process a single JSON file. Returns (changes_count, change_list)."""
    with open(filepath) as f:
        data = json.load(f)

    fname = filepath.name
    total_changed = 0
    all_changes: list[tuple[str, str]] = []

    # Determine file type and repair accordingly
    if fname.startswith("torvik_four_factors_"):
        # Dict keyed by team IDs
        if isinstance(data, dict):
            data, changed, changes = repair_dict_keys(data)
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("torvik_") and not fname.startswith("torvik_shooting_"):
        # {"teams": [{"team_id": ...}, ...]}
        if isinstance(data, dict) and "teams" in data and isinstance(data["teams"], list):
            changed, changes = repair_team_list(data["teams"])
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("tournament_seeds_"):
        # {"season": N, "teams": [{"team_id": ...}, ...]}
        if isinstance(data, dict) and "teams" in data:
            changed, changes = repair_team_list(data["teams"])
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("tournament_results_"):
        # {"games": [{"team1_id": ..., "team2_id": ..., "winner_id": ..., "loser_id": ...}, ...]}
        if isinstance(data, dict) and "games" in data:
            changed, changes = repair_game_ids(data["games"])
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("team_metrics_"):
        # {"season": N, "teams": [{"team_id": ...}, ...]}
        if isinstance(data, dict) and "teams" in data:
            changed, changes = repair_team_list(data["teams"])
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("historical_games_"):
        # {"games": [{"team1_id": ..., "team2_id": ...}, ...]}
        if isinstance(data, dict) and "games" in data:
            changed, changes = repair_game_ids(data["games"])
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("external_massey_composite_") or fname.startswith("external_"):
        # List of dicts with team_id
        if isinstance(data, list):
            changed, changes = repair_team_list(data)
            total_changed += changed
            all_changes.extend(changes)

    elif fname.startswith("cbbpy_rosters_"):
        # {"teams": [{"team_id": ...}, ...]} or list
        if isinstance(data, dict) and "teams" in data:
            changed, changes = repair_team_list(data["teams"])
            total_changed += changed
            all_changes.extend(changes)
        elif isinstance(data, list):
            changed, changes = repair_team_list(data)
            total_changed += changed
            all_changes.extend(changes)

    # Write back if changed
    if total_changed > 0 and not dry_run:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return total_changed, all_changes


def cross_validate(years: list[int]) -> list[str]:
    """Cross-validate team IDs across sources for given years."""
    issues = []
    for year in years:
        # Load seeds
        seeds_path = HIST_DIR / f"tournament_seeds_{year}.json"
        if not seeds_path.exists():
            continue
        with open(seeds_path) as f:
            seeds_data = json.load(f)
        teams = seeds_data.get("teams", []) if isinstance(seeds_data, dict) else seeds_data
        seed_ids = {normalize_team_id(t["team_id"]) for t in teams if isinstance(t, dict) and "team_id" in t}

        # Load torvik
        torvik_path = HIST_DIR / f"torvik_{year}.json"
        if not torvik_path.exists():
            continue
        with open(torvik_path) as f:
            torvik_data = json.load(f)
        torvik_teams = torvik_data.get("teams", []) if isinstance(torvik_data, dict) else torvik_data
        torvik_ids = {normalize_team_id(t["team_id"]) for t in torvik_teams if isinstance(t, dict) and "team_id" in t}

        # Check seeds vs torvik
        missing = seed_ids - torvik_ids
        if missing:
            issues.append(f"{year}: {len(missing)} seed teams not in Torvik: {sorted(missing)}")

        # Load results
        results_path = HIST_DIR / f"tournament_results_{year}.json"
        if results_path.exists():
            with open(results_path) as f:
                results_data = json.load(f)
            result_ids = set()
            for g in results_data.get("games", []):
                for key in ["team1_id", "team2_id", "winner_id", "loser_id"]:
                    if g.get(key):
                        result_ids.add(normalize_team_id(g[key]))
            if result_ids:
                results_not_in_seeds = result_ids - seed_ids
                if results_not_in_seeds:
                    issues.append(f"{year}: {len(results_not_in_seeds)} result teams not in seeds: {sorted(results_not_in_seeds)}")

    return issues


def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=== DRY RUN MODE (no files will be modified) ===\n")

    # Process all files in historical directory
    total_files = 0
    total_changes = 0
    unique_changes: dict[str, str] = {}

    # Process historical directory
    for filepath in sorted(HIST_DIR.glob("*.json")):
        changed, changes = process_file(filepath, dry_run=dry_run)
        if changed > 0:
            total_files += 1
            total_changes += changed
            for old, new in changes:
                unique_changes[old] = new
            examples = changes[:3]
            example_str = ", ".join(f"{o}->{n}" for o, n in examples)
            suffix = f" (+{len(changes) - 3} more)" if len(changes) > 3 else ""
            print(f"  {filepath.name}: {changed} IDs changed ({example_str}{suffix})")

    # Process symlinked files in raw/ that aren't symlinks (avoid double-processing)
    for filepath in sorted(RAW_DIR.glob("*.json")):
        if filepath.is_symlink():
            continue  # Skip symlinks — target in historical/ already processed
        changed, changes = process_file(filepath, dry_run=dry_run)
        if changed > 0:
            total_files += 1
            total_changes += changed
            for old, new in changes:
                unique_changes[old] = new
            examples = changes[:3]
            example_str = ", ".join(f"{o}->{n}" for o, n in examples)
            print(f"  raw/{filepath.name}: {changed} IDs changed ({example_str})")

    print(f"\n=== Summary ===")
    print(f"Files modified: {total_files}")
    print(f"Total ID changes: {total_changes}")
    print(f"Unique ID mappings: {len(unique_changes)}")

    if unique_changes:
        print(f"\nAll unique ID changes:")
        for old, new in sorted(unique_changes.items()):
            print(f"  {old:40s} -> {new}")

    # Cross-validate training years
    training_years = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    print(f"\n=== Cross-validation (training + holdout years) ===")
    issues = cross_validate(training_years)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("All seed teams found in Torvik for all training years.")
        print("All result teams found in seeds for all years with results.")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
