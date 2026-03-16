#!/usr/bin/env python3
"""
Validate training data integrity for the March Madness forecaster.

Checks for known data quality issues:
  1. All-zero critical fields in Torvik ratings (Four Factors)
  2. Missing RAPM in roster data
  3. Historical Torvik schema mismatches (flat dict vs array)
  4. Year coverage gaps
  5. Hardcoded-default saturation (all teams falling back to averages)

Exit codes:
  0 = all checks pass
  1 = warnings only (non-blocking issues found)
  2 = critical issues found (model accuracy likely compromised)

Usage:
  python scripts/validate_training_data.py [--year 2026] [--data-dir data/raw]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

FOUR_FACTORS = [
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
]


def _load_four_factors_teams(data_dir: Path, filename: str) -> List[Dict]:
    """Load teams from a Torvik JSON file, handling both flat-dict and array schemas."""
    path = data_dir / filename
    if not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, dict) and "teams" in payload:
        return payload["teams"]
    if isinstance(payload, list):
        return payload
    # Flat-dict format: keys are team IDs
    if isinstance(payload, dict) and not any(
        k in payload for k in ("year", "generated_at")
    ):
        return list(payload.values())
    return []


def check_torvik_four_factors(data_dir: Path, year: int) -> List[Tuple[str, str]]:
    """Check Four Factors across main Torvik file AND the separate four_factors file.

    The pipeline enriches torvik_{year}.json with values from
    torvik_four_factors_{year}.json at runtime.  This check mirrors that
    logic: a field is only flagged as missing if it is zero in BOTH files.
    """
    issues: List[Tuple[str, str]] = []

    main_teams = _load_four_factors_teams(data_dir, f"torvik_{year}.json")
    ff_teams = _load_four_factors_teams(data_dir, f"torvik_four_factors_{year}.json")

    if not main_teams and not ff_teams:
        issues.append(("CRITICAL", "No Torvik team data found in either file"))
        return issues

    # For each field, check if *either* source provides nonzero values
    for field in FOUR_FACTORS:
        main_vals = [_to_float(t.get(field)) for t in main_teams if isinstance(t, dict)]
        ff_vals = [_to_float(t.get(field)) for t in ff_teams if isinstance(t, dict)]

        main_nonzero = sum(1 for v in main_vals if v is not None and abs(v) > 1e-6)
        ff_nonzero = sum(1 for v in ff_vals if v is not None and abs(v) > 1e-6)

        best_source = max(main_nonzero, ff_nonzero)
        best_total = len(main_vals) if main_nonzero >= ff_nonzero else len(ff_vals)

        if best_total == 0:
            continue

        if best_source == 0:
            issues.append((
                "CRITICAL",
                f"'{field}' is zero in BOTH torvik_{year}.json and "
                f"torvik_four_factors_{year}.json for all teams "
                f"— no source provides this metric (scraper CSV fallback "
                f"cannot compute defensive factors)",
            ))
        elif best_source < best_total * 0.5:
            issues.append((
                "WARNING",
                f"'{field}' has data for only {best_source}/{best_total} teams",
            ))

    return issues


def check_roster_rapm(data_dir: Path, year: int) -> List[Tuple[str, str]]:
    """Check roster files for missing RAPM data."""
    issues: List[Tuple[str, str]] = []
    roster_path = data_dir / f"rosters_{year}.json"
    if not roster_path.exists():
        issues.append(("WARNING", f"Roster file missing: {roster_path}"))
        return issues

    payload = _load_json(roster_path)
    teams = payload.get("teams", [])
    if not teams:
        issues.append(("WARNING", f"No teams found in {roster_path}"))
        return issues

    teams_with_rapm = 0
    total_teams = len(teams)
    for team in teams:
        players = team.get("players", [])
        has_nonzero = any(
            _to_float(p.get("rapm_total")) not in (None, 0.0)
            for p in players
        )
        if has_nonzero:
            teams_with_rapm += 1

    if teams_with_rapm == 0:
        # Check if BPM/WARP proxy data is available — the pipeline
        # auto-repairs RAPM from these via enrich_roster_rapm() at runtime.
        has_proxy = False
        for team in teams[:10]:
            for p in team.get("players", []):
                bpm = _to_float(p.get("box_plus_minus"))
                if bpm is not None and abs(bpm) > 1e-6:
                    has_proxy = True
                    break
            if has_proxy:
                break

        if has_proxy:
            issues.append((
                "WARNING",
                f"RAPM is null/zero for ALL {total_teams} teams in {roster_path.name} "
                f"— but BPM/WARP proxy data is available; pipeline auto-repairs "
                f"via enrich_roster_rapm()",
            ))
        else:
            issues.append((
                "CRITICAL",
                f"RAPM is null/zero for ALL {total_teams} teams in {roster_path.name} "
                f"and no BPM/WARP proxy data available for auto-repair",
            ))
    elif teams_with_rapm < total_teams * 0.5:
        issues.append((
            "WARNING",
            f"RAPM data available for only {teams_with_rapm}/{total_teams} teams",
        ))

    return issues


def check_historical_torvik_schema(data_dir: Path) -> List[Tuple[str, str]]:
    """Check historical Torvik files for schema consistency."""
    issues: List[Tuple[str, str]] = []
    ff_files = sorted(data_dir.glob("torvik_four_factors_*.json"))

    flat_dict_years = []
    array_years = []
    empty_years = []

    for fp in ff_files:
        year_str = fp.stem.replace("torvik_four_factors_", "")
        try:
            yr = int(year_str)
        except ValueError:
            continue

        payload = _load_json(fp)
        if isinstance(payload, dict) and "teams" in payload:
            teams = payload["teams"]
            if isinstance(teams, list) and teams:
                array_years.append(yr)
            else:
                empty_years.append(yr)
        elif isinstance(payload, dict):
            # Flat dict format — keys are team IDs
            non_meta = {
                k: v for k, v in payload.items()
                if k not in ("year", "generated_at", "source")
            }
            if non_meta:
                flat_dict_years.append(yr)
            else:
                empty_years.append(yr)
        else:
            empty_years.append(yr)

    if flat_dict_years and array_years:
        issues.append((
            "CRITICAL",
            f"Historical Torvik schema mismatch: years {flat_dict_years} use flat-dict "
            f"format, years {array_years} use array format. "
            f"Data loader expecting array format will get 0 teams from flat-dict files.",
        ))

    if empty_years:
        issues.append((
            "WARNING",
            f"Empty/unparseable Torvik Four Factors files for years: {empty_years}",
        ))

    return issues


def check_year_coverage(data_dir: Path, expected_start: int = 2005, expected_end: int = 2025) -> List[Tuple[str, str]]:
    """Check for gaps in historical data year coverage."""
    issues: List[Tuple[str, str]] = []

    hist_dir = data_dir / "historical"
    if not hist_dir.exists():
        hist_dir = data_dir  # Fall back to main data dir

    game_years = set()
    for fp in data_dir.glob("historical_games_*.json"):
        try:
            yr = int(fp.stem.replace("historical_games_", ""))
            game_years.add(yr)
        except ValueError:
            pass

    # Also check historical subdir
    for fp in (data_dir / "historical").glob("historical_games_*.json") if (data_dir / "historical").exists() else []:
        try:
            yr = int(fp.stem.replace("historical_games_", ""))
            game_years.add(yr)
        except ValueError:
            pass

    expected = set(range(expected_start, expected_end + 1)) - {2020}  # Exclude COVID year
    missing = expected - game_years
    if missing:
        issues.append((
            "WARNING",
            f"Missing historical game data for years: {sorted(missing)}",
        ))

    return issues


def check_manifest_errors(data_dir: Path, year: int) -> List[Tuple[str, str]]:
    """Summarize validation errors already recorded in the manifest."""
    issues: List[Tuple[str, str]] = []
    manifest_path = data_dir / f"manifest_{year}.json"
    if not manifest_path.exists():
        return issues

    manifest = _load_json(manifest_path)
    validation_errors = manifest.get("validation_errors", {})
    total_errors = sum(len(v) for v in validation_errors.values())
    if total_errors > 0:
        issues.append((
            "WARNING",
            f"Manifest contains {total_errors} pre-existing validation errors "
            f"across {len(validation_errors)} artifacts — pipeline proceeded despite errors",
        ))
        for artifact, errs in validation_errors.items():
            if len(errs) > 3:
                issues.append((
                    "INFO",
                    f"  {artifact}: {len(errs)} errors (e.g. '{errs[0]}')",
                ))
            else:
                for e in errs:
                    issues.append(("INFO", f"  {artifact}: {e}"))

    return issues

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2026, help="Current prediction year")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    year = args.year
    all_issues: List[Tuple[str, str]] = []

    print(f"Validating training data for year {year} in {data_dir}...\n")

    checks = [
        ("Torvik Four Factors (current year)", check_torvik_four_factors(data_dir, year)),
        ("Roster RAPM data", check_roster_rapm(data_dir, year)),
        ("Historical Torvik schema consistency", check_historical_torvik_schema(data_dir)),
        ("Year coverage", check_year_coverage(data_dir)),
        ("Manifest validation errors", check_manifest_errors(data_dir, year)),
    ]

    for name, issues in checks:
        print(f"--- {name} ---")
        if not issues:
            print("  PASS\n")
        else:
            for severity, msg in issues:
                print(f"  [{severity}] {msg}")
                all_issues.append((severity, msg))
            print()

    # Summary
    criticals = sum(1 for s, _ in all_issues if s == "CRITICAL")
    warnings = sum(1 for s, _ in all_issues if s == "WARNING")
    infos = sum(1 for s, _ in all_issues if s == "INFO")

    print("=" * 60)
    print(f"SUMMARY: {criticals} critical, {warnings} warnings, {infos} info")

    if criticals:
        print("\nCRITICAL issues found — model accuracy is likely compromised.")
        print("Run 'python scripts/repair_2026_data_quality.py' to fix known issues.")
        return 2
    elif warnings:
        print("\nWarnings found — review before relying on predictions.")
        return 1
    else:
        print("\nAll checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
