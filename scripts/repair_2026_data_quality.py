#!/usr/bin/env python3
"""Repair season data quality issues.

Addresses the following gaps in scraped season data:

1. Coach tournament teams arrays are empty — populated from Barttorvik
   team-coach mappings, and head_coach injected into roster data.
2. Player RAPM values are null — estimated from BPM/WARP/usage priors.
3. Manifest is stale — updated with current timestamp and cleared errors.

Note: Four Factors are now sourced exclusively from barttorvik.com trank.php
CSV with date filtering. Box-score computation of Four Factors has been removed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features.public_advanced_metrics import PublicAdvancedMetricsBuilder
from src.data.normalize import normalize_team_id, _raw_normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_YEAR = 2026



def load_json(filename: str, required: bool = False) -> dict | list:
    path = DATA_DIR / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required data file missing: {path}")
        logger.warning("Data file not found, using empty default: %s", path)
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if required:
            raise
        logger.warning("Failed to read %s: %s — using empty default", path, exc)
        return {}


def save_json(filename: str, data: dict | list) -> None:
    path = DATA_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s", path)






# ---------------------------------------------------------------------------
# 1d. Backfill offensive eFG% and FT rate from four_factors into torvik teams
# ---------------------------------------------------------------------------

def backfill_offensive_ff_from_four_factors(
    torvik_data: dict,
    ff_data: dict,
) -> int:
    """Backfill effective_fg_pct and free_throw_rate from four_factors file.

    The Torvik scraper may leave these fields as zero on the team entries
    while the four_factors file (computed from game box scores) has correct
    values.  This patches the torvik team entries from the authoritative
    four_factors source.

    Uses normalize_team_id to resolve _st/_state, abbreviation, and alias
    mismatches between torvik team IDs and four_factors keys.

    Returns count of torvik teams updated.
    """
    if not ff_data:
        return 0

    # Build multiple lookup indices for four_factors entries
    # 1. Direct key lookup
    # 2. Collapsed (no underscores) lookup
    # 3. Normalized ID lookup (handles _st -> _state, byu -> brigham_young, etc.)
    collapsed_ff: dict[str, dict] = {}
    normalized_ff: dict[str, dict] = {}
    for k, v in ff_data.items():
        if isinstance(v, dict):
            collapsed_ff[k.replace("_", "")] = v
            norm_k = normalize_team_id(k)
            normalized_ff[norm_k] = v

    teams = torvik_data.get("teams", [])
    updated = 0
    fields = ("effective_fg_pct", "free_throw_rate")

    for team in teams:
        tid = team.get("team_id", "")
        ff_entry = ff_data.get(tid)
        if ff_entry is None:
            ff_entry = collapsed_ff.get(tid.replace("_", ""))
        if ff_entry is None:
            # Use normalize_team_id to resolve aliases (_st -> _state, etc.)
            norm_tid = normalize_team_id(tid)
            ff_entry = normalized_ff.get(norm_tid)
        if ff_entry is None or not isinstance(ff_entry, dict):
            continue

        changed = False
        for field in fields:
            current = float(team.get(field, 0) or 0)
            source = float(ff_entry.get(field, 0) or 0)
            if abs(current) < 1e-6 and abs(source) > 1e-6:
                team[field] = ff_entry[field]
                changed = True
                # Also update enriched_stats
                enriched = team.get("enriched_stats", {})
                if abs(float(enriched.get(field, 0) or 0)) < 1e-6:
                    enriched[field] = ff_entry[field]
                    team["enriched_stats"] = enriched
        if changed:
            updated += 1

    logger.info("Backfilled effective_fg_pct/free_throw_rate for %d torvik teams from four_factors", updated)
    return updated


# ---------------------------------------------------------------------------
# 1b. Rebuild advanced_metrics from historical game box scores
# ---------------------------------------------------------------------------

def rebuild_advanced_metrics(
    torvik_data: dict,
    year: int = DEFAULT_YEAR,
) -> int:
    """Regenerate advanced_metrics JSON by re-running PublicAdvancedMetricsBuilder.

    The builder pairs game records by game_id to fill opponent box score stats
    from the companion row, computes SOS-adjusted efficiency ratings, Four
    Factors, and Barthag.  Returns the number of teams in the output.
    """
    games_data = load_json(f"historical_games_{year}.json")
    game_records = games_data.get("games", [])
    if not game_records:
        logger.warning("No game records found — skipping advanced metrics rebuild")
        return 0

    torvik_teams = torvik_data.get("teams", [])
    builder = PublicAdvancedMetricsBuilder()
    result = builder.build(game_records, teams=torvik_teams)

    team_count = len(result.get("teams", []))
    save_json(f"advanced_metrics_{year}.json", result)
    return team_count


# ---------------------------------------------------------------------------
# 2. Map coaches to current teams
# ---------------------------------------------------------------------------

def fix_coach_tournament_teams(
    coach_data: dict,
    roster_data: dict,
    team_coach_map: dict[str, str],
) -> tuple[int, int]:
    """Populate coach tournament teams arrays and inject head_coach into rosters.

    Uses a team_coach_map (team_id -> coach_name) scraped from Barttorvik to:
    1. Set head_coach on each roster team block (enables data_loader.py lookup)
    2. Populate the teams array on coach tournament entries via reverse lookup

    Returns (roster_teams_updated, coaches_updated) counts.
    """
    # -- Inject head_coach into roster team blocks --
    roster_teams = roster_data.get("teams", [])
    roster_updated = 0

    # Build collapsed lookup for fuzzy matching
    collapsed_map: dict[str, tuple[str, str]] = {}
    for tid, coach in team_coach_map.items():
        collapsed_map[tid.replace("_", "")] = (tid, coach)

    for team in roster_teams:
        tid = team.get("team_id", "")
        coach = team_coach_map.get(tid)
        if not coach:
            # Try fuzzy match via collapsed form
            entry = collapsed_map.get(tid.replace("_", ""))
            if entry:
                coach = entry[1]
        if not coach:
            # Try normalizing team_name
            tname = team.get("team_name", "")
            norm = normalize_team_id(tname)
            coach = team_coach_map.get(norm)
            if not coach:
                entry = collapsed_map.get(norm.replace("_", ""))
                if entry:
                    coach = entry[1]
        if coach:
            team["head_coach"] = coach
            roster_updated += 1

    # -- Populate teams arrays on coach tournament entries --
    coaches = coach_data.get("coaches", {})

    # Build reverse map: normalized coach name -> [team_ids]
    coach_to_teams: dict[str, list[str]] = defaultdict(list)
    for tid, coach_name in team_coach_map.items():
        norm_coach = _normalize_coach_name(coach_name)
        coach_to_teams[norm_coach].append(tid)

    coaches_updated = 0
    for cid, info in coaches.items():
        coach_name = info.get("name", "")
        norm = _normalize_coach_name(coach_name)
        matched_teams = coach_to_teams.get(norm, [])

        # Fallback: try last-name-only matching
        if not matched_teams and " " in coach_name:
            last_name = coach_name.split()[-1].lower()
            for norm_key, teams_list in coach_to_teams.items():
                if norm_key.endswith(last_name) or last_name in norm_key.split("_"):
                    matched_teams = teams_list
                    break

        if matched_teams:
            info["teams"] = matched_teams
            info["teams_source"] = "barttorvik_team_coaches"
            coaches_updated += 1
        elif not info.get("teams"):
            info["teams_source"] = "unavailable_from_scrape"

    logger.info(
        "Coach mapping: %d roster teams updated, %d coaches mapped to teams",
        roster_updated, coaches_updated,
    )
    return roster_updated, coaches_updated


def _normalize_coach_name(name: str) -> str:
    """Normalize a coach name for fuzzy matching."""
    return "".join(
        c.lower() if c.isalnum() else "_" for c in (name or "")
    ).strip("_")


# ---------------------------------------------------------------------------
# 3. Estimate RAPM from BPM/WARP priors
# ---------------------------------------------------------------------------

def estimate_rapm_from_priors(roster_data: dict) -> int:
    """Fill null/zero RAPM values using BPM/WARP/usage proxy.

    Uses the same formula as enrich_roster_rapm() in data_loader.py:
      proxy = 0.6 * BPM + 0.3 * (4.0 * WARP) + 0.1 * ((usage - 20) / 25)
      off_share = 0.6 if usage >= 20 else 0.45

    Returns count of players updated.
    """
    teams = roster_data.get("teams", [])
    updated = 0

    for team in teams:
        players = team.get("players", [])
        for player in players:
            rapm_off = player.get("rapm_offensive")
            rapm_def = player.get("rapm_defensive")

            # Check if RAPM is null or zero
            if rapm_off is not None and rapm_def is not None:
                if abs(float(rapm_off or 0)) > 1e-8 or abs(float(rapm_def or 0)) > 1e-8:
                    continue

            bpm = float(player.get("box_plus_minus") or 0.0)
            warp = float(player.get("warp") or 0.0)
            usage = float(player.get("usage_rate") or 0.0)

            warp_signal = 4.0 * warp
            usage_signal = (usage - 20.0) / 25.0
            proxy = 0.6 * bpm + 0.3 * warp_signal + 0.1 * usage_signal

            off_share = 0.6 if usage >= 20.0 else 0.45
            player["rapm_offensive"] = round(proxy * off_share, 4)
            player["rapm_defensive"] = round(proxy * (1.0 - off_share), 4)
            updated += 1

    logger.info("Estimated RAPM for %d players from BPM/WARP priors", updated)
    return updated


# ---------------------------------------------------------------------------
# 4. Update manifest
# ---------------------------------------------------------------------------

def update_manifest(
    torvik_ff_fixed: int,
    rapm_fixed: int,
    def_ff_count: int,
    coaches_mapped: int = 0,
    year: int = DEFAULT_YEAR,
) -> None:
    """Update manifest with current timestamp and revised validation errors."""
    manifest = load_json(f"manifest_{year}.json")

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Revise validation errors based on what we fixed
    errors = manifest.get("validation_errors", {})

    # Torvik: remove Four Factors errors if we populated them
    torvik_errors = errors.get("torvik_json", [])
    if torvik_ff_fixed > 0 or def_ff_count > 0:
        # Remove the "all zero" errors for fields we've now populated
        fixed_fields = set()
        if torvik_ff_fixed > 0:
            fixed_fields.update([
                "effective_fg_pct", "turnover_rate",
                "offensive_reb_rate", "free_throw_rate",
            ])
        if def_ff_count > 0:
            fixed_fields.update([
                "opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate",
            ])
        torvik_errors = [
            e for e in torvik_errors
            if not any(f"'{f}'" in e for f in fixed_fields)
        ]
        errors["torvik_json"] = torvik_errors

    # Rosters: update RAPM errors
    if rapm_fixed > 0:
        errors["rosters_json"] = [
            f"RAPM estimated from BPM/WARP priors for {rapm_fixed} players "
            f"(no raw RAPM available from source)"
        ]

    manifest["validation_errors"] = errors

    # Add provenance for the repair
    provenance = manifest.get("provenance", {})
    provenance["data_quality_repair"] = {
        "script": "scripts/repair_2026_data_quality.py",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "defensive_four_factors_teams": def_ff_count,
        "rapm_players_estimated": rapm_fixed,
        "coaches_mapped_to_teams": coaches_mapped,
    }
    manifest["provenance"] = provenance

    save_json(f"manifest_{year}.json", manifest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Repair season data quality issues")
    parser.add_argument(
        "--year", type=int, default=DEFAULT_YEAR,
        help=f"Season year to repair (default: {DEFAULT_YEAR})",
    )
    args = parser.parse_args()
    year = args.year

    logger.info("=== %d Data Quality Repair ===", year)

    # Load data files (torvik is required; others degrade gracefully)
    torvik_data = load_json(f"torvik_{year}.json", required=True)
    ff_data = load_json(f"torvik_four_factors_{year}.json")
    coach_data = load_json(f"coach_tournament_{year}.json")
    roster_data = load_json(f"rosters_{year}.json")

    # Backfill offensive eFG% and FT rate from four_factors
    logger.info("--- Step 1d: Backfill eFG%%/FTR from Four Factors ---")
    efg_ftr_updated = backfill_offensive_ff_from_four_factors(torvik_data, ff_data)

    # 1b. Rebuild advanced_metrics from historical games
    logger.info("--- Step 1b: Rebuild Advanced Metrics ---")
    adv_metrics_updated = rebuild_advanced_metrics(torvik_data, year=year)
    logger.info("Rebuilt advanced_metrics with %d teams", adv_metrics_updated)

    # 2. Fix coach tournament teams
    logger.info("--- Step 2: Coach Tournament Teams ---")
    from src.data.scrapers.tournament_context import TournamentContextScraper
    try:
        team_coach_map = TournamentContextScraper(
            cache_dir=str(DATA_DIR / "cache"),
        ).fetch_team_coaches(year)
    except Exception as exc:
        logger.warning("Failed to fetch team coaches: %s", exc)
        team_coach_map = {}
    logger.info("Fetched %d team-coach mappings from Barttorvik", len(team_coach_map))

    # Fallback: if scraper returned empty, try coach fields from torvik data
    if not team_coach_map:
        logger.info("Scraper returned empty; trying torvik_data coach fallback")
        for team in torvik_data.get("teams", []):
            coach = team.get("coach", "")
            tid = team.get("team_id", "")
            if coach and tid:
                team_coach_map[tid] = coach
        logger.info("Recovered %d mappings from torvik_data", len(team_coach_map))

    roster_coaches_updated = 0
    coaches_updated = 0
    if team_coach_map:
        roster_coaches_updated, coaches_updated = fix_coach_tournament_teams(
            coach_data, roster_data, team_coach_map,
        )
    else:
        logger.warning(
            "No team-coach mappings available — skipping coach team "
            "population to preserve any existing data"
        )

    # 3. Estimate RAPM from priors
    logger.info("--- Step 3: RAPM Estimation ---")
    rapm_updated = estimate_rapm_from_priors(roster_data)

    # Save updated files (skip empty dicts to avoid overwriting existing data)
    logger.info("--- Saving repaired data ---")
    save_json(f"torvik_{year}.json", torvik_data)
    if ff_data:
        save_json(f"torvik_four_factors_{year}.json", ff_data)
    if coach_data:
        save_json(f"coach_tournament_{year}.json", coach_data)
    if roster_data:
        save_json(f"rosters_{year}.json", roster_data)

    # 4. Update manifest
    logger.info("--- Step 4: Update Manifest ---")
    update_manifest(
        torvik_ff_fixed=torvik_updated,
        rapm_fixed=rapm_updated,
        def_ff_count=len(def_ff),
        coaches_mapped=coaches_updated,
        year=year,
    )

    # Summary
    logger.info("=== Repair Complete ===")
    logger.info("  Defensive Four Factors computed for %d teams", len(def_ff))
    logger.info("  Defensive: torvik teams updated: %d, ff entries: %d", torvik_updated, ff_updated)
    logger.info("  Offensive: torvik teams updated: %d, ff entries: %d", off_torvik_updated, off_ff_updated)
    logger.info("  Offensive eFG%%/FTR backfilled: %d torvik teams", efg_ftr_updated)
    logger.info("  Advanced metrics rebuilt for %d teams", adv_metrics_updated)
    logger.info("  Roster teams with head_coach: %d", roster_coaches_updated)
    logger.info("  Coach entries mapped to teams: %d", coaches_updated)
    logger.info("  Player RAPM estimated: %d", rapm_updated)

    # --- Data Quality Checks ---
    logger.info("--- Data Quality Checks ---")
    warnings = []

    adv_data = load_json(f"advanced_metrics_{year}.json")
    adv_teams = adv_data.get("teams", [])
    if adv_teams:
        barthag_ones = sum(
            1 for t in adv_teams if t.get("barthag", 0) >= 1.0
        )
        if barthag_ones > 10:
            warnings.append(
                f"{barthag_ones} teams have barthag >= 1.0 (expected <= 10)"
            )
        zero_opp_efg = sum(
            1 for t in adv_teams
            if t.get("opp_effective_fg_pct", 0) == 0.0
        )
        if zero_opp_efg > 5:
            warnings.append(
                f"{zero_opp_efg} teams have opp_effective_fg_pct == 0.0 "
                f"(expected <= 5)"
            )

    # Check torvik offensive FF after repair
    torvik_teams = torvik_data.get("teams", [])
    orb_vals = sorted(
        float(t.get("offensive_reb_rate", 0) or 0)
        for t in torvik_teams
        if float(t.get("offensive_reb_rate", 0) or 0) > 1e-6
    )
    if orb_vals:
        median_orb = orb_vals[len(orb_vals) // 2]
        if median_orb < 0.18:
            warnings.append(
                f"Torvik ORB% median = {median_orb:.4f} (expected >= 0.18) — "
                f"CSV player-level bias may persist"
            )
        elif median_orb > 0.40:
            warnings.append(
                f"Torvik ORB% median = {median_orb:.4f} (expected <= 0.40) — "
                f"suspiciously high"
            )
        else:
            logger.info("Torvik ORB%% median = %.4f (within [0.18, 0.40] range)", median_orb)

    efg_zero_count = sum(
        1 for t in torvik_teams
        if abs(float(t.get("effective_fg_pct", 0) or 0)) < 1e-6
    )
    if efg_zero_count > 5:
        warnings.append(
            f"{efg_zero_count}/{len(torvik_teams)} torvik teams still have "
            f"effective_fg_pct == 0.0 after repair (expected <= 5)"
        )

    ftr_zero_count = sum(
        1 for t in torvik_teams
        if abs(float(t.get("free_throw_rate", 0) or 0)) < 1e-6
    )
    if ftr_zero_count > 5:
        warnings.append(
            f"{ftr_zero_count}/{len(torvik_teams)} torvik teams still have "
            f"free_throw_rate == 0.0 after repair (expected <= 5)"
        )

    def_zero_count = sum(
        1 for t in torvik_teams
        if abs(float(t.get("opp_effective_fg_pct", 0) or 0)) < 1e-6
    )
    if def_zero_count > 5:
        warnings.append(
            f"{def_zero_count}/{len(torvik_teams)} torvik teams still have "
            f"opp_effective_fg_pct == 0.0 after repair (expected <= 5)"
        )

    coach_entries = coach_data.get("coaches", {})
    if coach_entries:
        coaches_with_teams = sum(
            1 for c in coach_entries.values() if c.get("teams")
        )
        pct = coaches_with_teams / len(coach_entries) if coach_entries else 0
        if pct < 0.5:
            warnings.append(
                f"Only {coaches_with_teams}/{len(coach_entries)} "
                f"({pct:.0%}) coaches have non-empty teams"
            )

    if warnings:
        for w in warnings:
            logger.warning("DATA QUALITY: %s", w)
    else:
        logger.info("All data quality checks passed")


if __name__ == "__main__":
    main()
