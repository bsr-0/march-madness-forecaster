#!/usr/bin/env python3
"""Scrape historical ESPN "Who Picked Whom" data from multiple sources.

Populates the historical public pick data archive used by the MC pool
backtest and LOYO evaluation.  Data sources (tried in order):

1. **Kaggle** — Nishaan Amin's March Madness dataset (2008-2025).
2. **Wayback Machine** — Internet Archive snapshots of ESPN's WPW page.
3. **ESPN live** — Current-season endpoint (Gambit API → WPW → People's Bracket).
4. **CSV import** — Manual data entry via ``--from-csv path.csv``.

Usage:
    # Import from Kaggle (requires KAGGLE_USERNAME and KAGGLE_KEY env vars)
    python scripts/scrape_historical_espn_picks.py --from-kaggle

    # Auto-scrape from Wayback Machine (default)
    python scripts/scrape_historical_espn_picks.py --years 2024 2025

    # Import from a CSV file (columns: team,seed,R64,R32,S16,E8,F4,CHAMP)
    python scripts/scrape_historical_espn_picks.py --from-csv picks_2024.csv --year 2024

    # Validate existing data files
    python scripts/scrape_historical_espn_picks.py --validate

Output:
    data/raw/historical_public_picks/espn_picks_{year}.json

Notes:
    - Rate-limited to 1 request per 2 seconds to be respectful to archive.org
    - Falls back gracefully when a year is not archived
    - 2020 is excluded (COVID, no tournament)
    - CSV percentages can be 0-100 or 0-1 scale (auto-detected)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/historical_public_picks")

WAYBACK_API = (
    "https://web.archive.org/web/{year}*/fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"
)
WAYBACK_URL_TEMPLATE = "https://web.archive.org/web/{timestamp}/https://fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom"

DEFAULT_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]
ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]


def scrape_year(year: int) -> dict | None:
    """Attempt to scrape ESPN public pick data for a given year.

    Tries multiple URL patterns against the Wayback Machine:
    - fantasy.espn.com (2019+)
    - games.espn.com (2014-2018, legacy domain)
    - gambit-api.fantasy.espn.com propositions endpoint (JSON, 2019+)

    Returns:
        Dict with team pick percentages, or None if unavailable.
    """
    try:
        import urllib.error
        import urllib.request

        # Try multiple URL patterns — ESPN changed domains over the years
        url_patterns = [
            f"fantasy.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom",
            f"games.espn.com/tournament-challenge-bracket/{year}/en/whopickedwhom",
            f"gambit-api.fantasy.espn.com/apis/v1/propositions-sets/tournament-challenge-bracket-{year}-en-whopickedwhom",
        ]

        best_snapshot = None

        for url_pattern in url_patterns:
            cdx_url = (
                f"https://web.archive.org/cdx/search/cdx?url={url_pattern}&output=json&limit=5&filter=statuscode:200"
            )

            logger.info("Checking Wayback Machine for %d: %s", year, url_pattern.split("/")[0])
            req = urllib.request.Request(cdx_url, headers={"User-Agent": "MarchMadnessForecaster/1.0"})

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                logger.debug("CDX query failed for %s: %s", url_pattern, e)
                continue

            if len(data) <= 1:
                continue

            # Found snapshots — pick the best one (prefer March)
            snapshots = data[1:]
            best_timestamp = snapshots[-1][1]
            for snap in snapshots:
                ts = snap[1]
                if ts[4:6] == "03":
                    best_timestamp = ts
                    break

            # Reconstruct the original URL from the CDX result
            original_url = snapshots[0][0] if snapshots[0][0].startswith("http") else f"https://{url_pattern}"
            best_snapshot = (best_timestamp, original_url)
            logger.info("Found snapshot for %d via %s: %s", year, url_pattern.split("/")[0], best_timestamp)
            break

            time.sleep(1)  # Rate limit between CDX queries

        if best_snapshot is None:
            logger.warning("No Wayback Machine snapshots found for %d across any URL pattern", year)
            return None

        best_timestamp, original_url = best_snapshot

        # Step 2: Fetch the archived page
        archive_url = f"https://web.archive.org/web/{best_timestamp}/{original_url}"
        time.sleep(2)  # Rate limit

        req = urllib.request.Request(archive_url, headers={"User-Agent": "MarchMadnessForecaster/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.warning("Failed to fetch archived page for %d: %s", year, e)
            return None

        # Step 3: Parse pick percentages from HTML
        teams = _parse_espn_picks_html(html, year)

        if teams and len(teams) >= 32:
            logger.info("Extracted %d teams for %d", len(teams), year)
            return {
                "year": year,
                "source": "wayback_machine",
                "timestamp": best_timestamp,
                "original_url": original_url,
                "teams": teams,
            }
        else:
            logger.warning(
                "Could not extract sufficient team data for %d (got %d)",
                year,
                len(teams) if teams else 0,
            )
            return None

    except ImportError:
        logger.error("urllib not available")
        return None


def _parse_espn_picks_html(html: str, year: int) -> dict:
    """Extract pick percentages from ESPN WPW HTML.

    Delegates to ESPNPicksScraper.parse_wpw_html() which handles both
    __NEXT_DATA__ JSON embeds and HTML table formats.

    Returns dict of team_id -> {round: pick_pct (0-1 scale)}.
    """
    from src.data.scrapers.espn_picks import ESPNPicksScraper

    parsed = ESPNPicksScraper.parse_wpw_html(html)
    if not parsed:
        logger.warning("parse_wpw_html returned no data for %d", year)
        return {}

    # Convert to {team_id: {R64: pct, ...}} format with 0-1 scale
    round_map = {
        "round_of_64_pct": "R64",
        "round_of_32_pct": "R32",
        "sweet_16_pct": "S16",
        "elite_8_pct": "E8",
        "final_four_pct": "F4",
        "champion_pct": "CHAMP",
    }
    teams = {}
    for t in parsed:
        tid = t["name"].lower().replace(" ", "_").replace(".", "").replace("'", "")
        teams[tid] = {rnd: t.get(field, 0.0) / 100.0 for field, rnd in round_map.items()}
    return teams


def import_from_csv(csv_path: str, year: int) -> dict | None:
    """Import ESPN pick data from a CSV file.

    Expected CSV format (header required):
        team,seed,R64,R32,S16,E8,F4,CHAMP

    Percentages can be on 0-100 scale (e.g. 97.5) or 0-1 scale (e.g. 0.975).
    Auto-detected: if any value > 1.0, assumes 0-100 scale.

    Args:
        csv_path: Path to CSV file.
        year: Tournament year to associate with this data.

    Returns:
        Dict in standard output format, or None on error.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return None

    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)

            # Validate header
            required_cols = {"team"}
            round_cols = set(ROUND_NAMES)
            if reader.fieldnames is None:
                logger.error("CSV file has no header row")
                return None

            header_set = set(reader.fieldnames)
            if not required_cols.issubset(header_set):
                logger.error(
                    "CSV missing required column 'team'. Found: %s",
                    reader.fieldnames,
                )
                return None

            available_rounds = round_cols & header_set
            if not available_rounds:
                logger.error(
                    "CSV has no round columns (%s). Found: %s",
                    ROUND_NAMES,
                    reader.fieldnames,
                )
                return None

            rows = list(reader)

        if not rows:
            logger.error("CSV file is empty (no data rows)")
            return None

        # Collect all numeric values to detect scale
        all_values = []
        for row in rows:
            for rnd in available_rounds:
                val = row.get(rnd, "").strip()
                if val:
                    try:
                        all_values.append(float(val))
                    except ValueError:
                        pass

        # Auto-detect scale: if any value > 1.0, assume 0-100
        is_pct_scale = any(v > 1.0 for v in all_values)
        scale_factor = 100.0 if is_pct_scale else 1.0
        if is_pct_scale:
            logger.info("Detected 0-100 percentage scale, converting to 0-1")
        else:
            logger.info("Detected 0-1 scale, using as-is")

        # Parse teams
        teams = {}
        for row in rows:
            team_name = row.get("team", "").strip()
            if not team_name:
                continue

            # Normalize team ID
            tid = team_name.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")

            picks = {}
            for rnd in ROUND_NAMES:
                val = row.get(rnd, "").strip()
                if val:
                    try:
                        pct = float(val) / scale_factor
                        picks[rnd] = max(0.0, min(1.0, pct))
                    except ValueError:
                        logger.warning(
                            "Non-numeric value '%s' for %s/%s, using 0.0",
                            val,
                            team_name,
                            rnd,
                        )
                        picks[rnd] = 0.0
                else:
                    picks[rnd] = 0.0

            teams[tid] = picks

        if len(teams) < 16:
            logger.error(
                "CSV has only %d teams (need at least 16 for a bracket region)",
                len(teams),
            )
            return None

        logger.info(
            "Imported %d teams from CSV for %d (%d round columns: %s)",
            len(teams),
            year,
            len(available_rounds),
            sorted(available_rounds),
        )

        return {
            "year": year,
            "source": "csv_import",
            "source_file": str(path.name),
            "teams": teams,
        }

    except (csv.Error, OSError) as e:
        logger.error("Failed to read CSV %s: %s", csv_path, e)
        return None


def import_from_kaggle(output_dir: Path | None = None, years: list[int] | None = None) -> int:
    """Download and import public picks from Kaggle's March Madness dataset.

    Uses the Kaggle API (requires KAGGLE_USERNAME and KAGGLE_KEY env vars)
    to download Nishaan Amin's 'march-madness-data' dataset, then converts
    the 'Public Picks.csv' file into per-year JSON files.

    The Kaggle CSV has columns: YEAR, TEAMNO, TEAM, R64, R32, S16, E8, F4, FINALS
    Values are on 0-100 scale.

    Returns:
        Number of years that failed to import (0 = all OK).
    """
    output_dir = output_dir or OUTPUT_DIR

    # Check for Kaggle credentials
    kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
    kaggle_key = os.environ.get("KAGGLE_KEY", "")
    if not kaggle_user or not kaggle_key:
        logger.error(
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables are required. "
            "Set them as repo secrets or export them locally."
        )
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download dataset via kaggle CLI
        logger.info("Downloading dataset from Kaggle (nishaanamin/march-madness-data)...")
        try:
            result = subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    "nishaanamin/march-madness-data",
                    "-p",
                    tmpdir,
                    "--unzip",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "KAGGLE_USERNAME": kaggle_user, "KAGGLE_KEY": kaggle_key},
            )
            if result.returncode != 0:
                logger.error("Kaggle download failed: %s", result.stderr.strip())
                return 1
        except FileNotFoundError:
            logger.error("kaggle CLI not found. Install with: pip install kaggle")
            return 1
        except subprocess.TimeoutExpired:
            logger.error("Kaggle download timed out after 120 seconds")
            return 1

        # Find the Public Picks CSV
        csv_path = _find_kaggle_picks_csv(Path(tmpdir))
        if csv_path is None:
            logger.error(
                "Could not find 'Public Picks' CSV in downloaded dataset. Files found: %s",
                [f.name for f in Path(tmpdir).rglob("*.csv")],
            )
            return 1

        logger.info("Found picks CSV: %s", csv_path.name)
        return _convert_kaggle_csv(csv_path, output_dir, years)


def _find_kaggle_picks_csv(search_dir: Path) -> Path | None:
    """Locate the public picks CSV in the downloaded Kaggle dataset."""
    for f in search_dir.rglob("*.csv"):
        name_lower = f.name.lower().replace(" ", "").replace("_", "")
        if "publicpick" in name_lower:
            return f
    return None


def _convert_kaggle_csv(csv_path: Path, output_dir: Path, years: list[int] | None) -> int:
    """Convert Kaggle's Public Picks CSV into per-year JSON files.

    The CSV has columns: YEAR, TEAMNO, TEAM, R64, R32, S16, E8, F4, FINALS
    Values are percentages on 0-100 scale.
    """
    # Kaggle CSV uses FINALS for the championship round
    kaggle_round_map = {
        "R64": "R64",
        "R32": "R32",
        "S16": "S16",
        "E8": "E8",
        "F4": "F4",
        "FINALS": "CHAMP",
    }

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.error("CSV has no header row")
                return 1

            # Normalize header names (strip whitespace)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]

            rows = list(reader)
    except (csv.Error, OSError) as e:
        logger.error("Failed to read Kaggle CSV: %s", e)
        return 1

    if not rows:
        logger.error("Kaggle CSV is empty")
        return 1

    # Group rows by year
    by_year: dict[int, list[dict]] = {}
    for row in rows:
        try:
            yr = int(row.get("YEAR", "0"))
        except ValueError:
            continue
        if yr == 2020:
            continue
        by_year.setdefault(yr, []).append(row)

    available_years = sorted(by_year.keys())
    logger.info("Kaggle CSV contains years: %s", available_years)

    if years:
        target_years = [y for y in years if y in by_year]
        skipped = [y for y in years if y not in by_year]
        if skipped:
            logger.warning("Requested years not in Kaggle data: %s", skipped)
    else:
        target_years = available_years

    # Detect scale: check if any round value > 1.0
    sample_values = []
    for row in rows[:50]:
        for col in kaggle_round_map:
            val = row.get(col, "").strip()
            if val:
                try:
                    sample_values.append(float(val))
                except ValueError:
                    pass
    is_pct_scale = any(v > 1.0 for v in sample_values)
    scale_factor = 100.0 if is_pct_scale else 1.0
    logger.info("Detected %s scale", "0-100" if is_pct_scale else "0-1")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for yr in target_years:
        year_rows = by_year[yr]
        teams = {}

        for row in year_rows:
            team_name = row.get("TEAM", "").strip()
            if not team_name:
                continue

            tid = team_name.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")

            picks = {}
            for kaggle_col, our_col in kaggle_round_map.items():
                val = row.get(kaggle_col, "").strip()
                if val:
                    try:
                        pct = float(val) / scale_factor
                        picks[our_col] = max(0.0, min(1.0, round(pct, 4)))
                    except ValueError:
                        picks[our_col] = 0.0
                else:
                    picks[our_col] = 0.0

            teams[tid] = picks

        if len(teams) < 32:
            logger.warning("Year %d: only %d teams (need >= 32), skipping", yr, len(teams))
            failures += 1
            continue

        data = {
            "year": yr,
            "source": "kaggle_espn",
            "source_dataset": "nishaanamin/march-madness-data",
            "teams": teams,
        }

        out_path = output_dir / f"espn_picks_{yr}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %s (%d teams)", out_path.name, len(teams))

    logger.info(
        "\nKaggle import complete: %d years written, %d failures",
        len(target_years) - failures,
        failures,
    )
    return failures


def validate_existing_files(output_dir: Path | None = None) -> int:
    """Validate all existing ESPN pick data files.

    Checks:
    - JSON is parseable
    - Has required fields (year, source, teams)
    - Teams dict has at least 32 entries
    - Each team has all 6 round columns
    - Pick percentages are in [0, 1] range
    - Monotonic decrease: R64 >= R32 >= ... >= CHAMP for each team
    - Flags synthetic/placeholder data

    Returns:
        Number of files with issues (0 = all clean).
    """
    output_dir = output_dir or OUTPUT_DIR
    if not output_dir.exists():
        logger.error("Directory does not exist: %s", output_dir)
        return 1

    files = sorted(output_dir.glob("*.json"))
    if not files:
        logger.warning("No JSON files found in %s", output_dir)
        return 0

    issues_total = 0

    for filepath in files:
        issues = []
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("  FAIL %s: cannot parse: %s", filepath.name, e)
            issues_total += 1
            continue

        year = data.get("year", "?")
        source = data.get("source", "unknown")
        teams = data.get("teams", {})

        # Check source — flag synthetic data
        synthetic_sources = {"seed_model_v2", "seed_model", "synthetic", "placeholder"}
        if source in synthetic_sources:
            issues.append(f"SYNTHETIC data (source={source!r}) — needs real ESPN data")

        # Check team count
        if not isinstance(teams, dict):
            issues.append("'teams' is not a dict")
        elif len(teams) < 32:
            issues.append(f"only {len(teams)} teams (need >= 32)")
        else:
            # Check individual teams
            missing_rounds = 0
            out_of_range = 0
            non_monotonic = 0

            for tid, picks in teams.items():
                if not isinstance(picks, dict):
                    issues.append(f"team {tid!r} has non-dict picks")
                    continue

                for rnd in ROUND_NAMES:
                    if rnd not in picks:
                        missing_rounds += 1
                    else:
                        v = picks[rnd]
                        if not isinstance(v, (int, float)):
                            out_of_range += 1
                        elif v < 0 or v > 1.0:
                            out_of_range += 1

                # Check monotonic decrease
                prev = 1.0
                for rnd in ROUND_NAMES:
                    v = picks.get(rnd, 0.0)
                    if isinstance(v, (int, float)) and v > prev + 0.01:
                        non_monotonic += 1
                        break
                    if isinstance(v, (int, float)):
                        prev = v

            if missing_rounds:
                issues.append(f"{missing_rounds} missing round values")
            if out_of_range:
                issues.append(f"{out_of_range} values out of [0,1] range")
            if non_monotonic:
                issues.append(f"{non_monotonic} teams with non-monotonic picks")

        # Report
        status = "WARN" if issues else "OK"
        icon = "!" if issues else "✓"
        logger.info(
            "  %s %s: year=%s, source=%s, teams=%d",
            icon,
            filepath.name,
            year,
            source,
            len(teams) if isinstance(teams, dict) else 0,
        )
        for issue in issues:
            logger.warning("      %s", issue)

        if issues:
            issues_total += 1

    logger.info("")
    if issues_total:
        logger.warning("%d / %d files have issues", issues_total, len(files))
    else:
        logger.info("All %d files validated OK", len(files))

    return issues_total


def main():
    parser = argparse.ArgumentParser(description="Scrape/import historical ESPN pick data")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Tournament years to scrape (default: 2018-2025 excl. 2020)",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Import pick data from CSV instead of scraping",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Tournament year for CSV import (required with --from-csv)",
    )
    parser.add_argument(
        "--from-kaggle",
        action="store_true",
        help="Download and import from Kaggle (nishaanamin/march-madness-data). "
        "Requires KAGGLE_USERNAME and KAGGLE_KEY env vars.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing data files and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # --- Validate mode ---
    if args.validate:
        logger.info("Validating existing ESPN pick data files...")
        logger.info("Directory: %s\n", output_dir)
        n_issues = validate_existing_files(output_dir)
        return 1 if n_issues > 0 else 0

    # --- Kaggle import mode ---
    if args.from_kaggle:
        failures = import_from_kaggle(output_dir, args.years if args.years != DEFAULT_YEARS else None)
        if failures == 0:
            logger.info("\nValidating imported data...")
            validate_existing_files(output_dir)
        return 1 if failures > 0 else 0

    # --- CSV import mode ---
    if args.from_csv:
        if args.year is None:
            logger.error("--year is required when using --from-csv")
            return 1

        data = import_from_csv(args.from_csv, args.year)
        if data is None:
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"espn_picks_{args.year}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %s", output_path)

        # Validate the written file
        logger.info("\nValidating written file...")
        validate_existing_files(output_dir)
        return 0

    # --- Scrape mode (default) ---
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for year in args.years:
        if year == 2020:
            logger.info("Skipping 2020 (COVID, no tournament)")
            continue

        data = scrape_year(year)
        if data is not None:
            output_path = output_dir / f"espn_picks_{year}.json"
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved %s", output_path)
            results[year] = "success"
        else:
            logger.warning("No data for %d", year)
            results[year] = "failed"

        time.sleep(2)  # Rate limit between years

    logger.info("\nSummary:")
    for year, status in sorted(results.items()):
        logger.info("  %d: %s", year, status)

    return 0


if __name__ == "__main__":
    sys.exit(main())
