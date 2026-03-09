"""Data enrichment for conference tournament predictions.

Merges additional statistics from separate Torvik data files
(Four Factors, shooting splits) into the base team data, filling
in fields that are missing or zero in the main torvik_YYYY.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _try_load_json(path: str) -> Optional[dict]:
    """Load a JSON file, returning None if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _find_data_file(base_dir: str, prefix: str, year: int) -> Tuple[Optional[dict], int]:
    """Find data file for the given year, falling back to most recent prior year.

    Args:
        base_dir: Directory containing data files.
        prefix: File prefix (e.g. "torvik_four_factors").
        year: Target year.

    Returns:
        (data_dict, actual_year) or (None, 0) if not found.
    """
    # Try exact year first, then fall back through recent years
    for y in range(year, max(year - 3, 2004), -1):
        path = Path(base_dir) / f"{prefix}_{y}.json"
        data = _try_load_json(str(path))
        if data is not None:
            if y != year:
                logger.info(
                    "No %s_%d.json found; using %d data as fallback",
                    prefix, year, y,
                )
            return data, y
    return None, 0


def enrich_torvik_teams(
    torvik_data: dict,
    data_dir: str = "data/raw",
    year: int = 2026,
) -> dict:
    """Enrich Torvik team data with Four Factors and shooting stats.

    The main torvik_YYYY.json often has zeros for Four Factors and
    shooting splits.  This function merges real values from the
    separate torvik_four_factors_YYYY.json and torvik_shooting_YYYY.json
    files.

    Args:
        torvik_data: Loaded torvik_YYYY.json dict with "teams" key.
        data_dir: Directory containing data files.
        year: Season year to look up.

    Returns:
        Enriched copy of torvik_data with merged statistics.
    """
    teams = torvik_data.get("teams", [])
    if not teams:
        return torvik_data

    # Load supplementary data
    ff_data, ff_year = _find_data_file(data_dir, "torvik_four_factors", year)
    shooting_data, shooting_year = _find_data_file(data_dir, "torvik_shooting", year)

    ff_matched = 0
    shooting_matched = 0

    for team in teams:
        team_id = team.get("team_id", "")
        if not team_id:
            continue

        # Initialize enriched stats dict
        stats = team.get("enriched_stats", {})

        # Merge Four Factors
        if ff_data is not None:
            ff = ff_data.get(team_id)
            if ff is None:
                # Try common ID variations
                ff = _fuzzy_lookup(ff_data, team_id)
            if ff:
                ff_matched += 1
                # Only overwrite if the existing value is zero/missing
                _merge_if_zero(team, ff, "effective_fg_pct")
                _merge_if_zero(team, ff, "turnover_rate")
                _merge_if_zero(team, ff, "offensive_reb_rate")
                _merge_if_zero(team, ff, "free_throw_rate")
                _merge_if_zero(team, ff, "opp_effective_fg_pct")
                _merge_if_zero(team, ff, "opp_turnover_rate")
                _merge_if_zero(team, ff, "defensive_reb_rate")
                _merge_if_zero(team, ff, "opp_free_throw_rate")
                # Also store in enriched stats
                for k, v in ff.items():
                    stats[k] = v

        # Merge shooting stats
        if shooting_data is not None:
            shooting = shooting_data.get(team_id)
            if shooting is None:
                shooting = _fuzzy_lookup(shooting_data, team_id)
            if shooting:
                shooting_matched += 1
                _merge_if_zero(team, shooting, "ft_pct")
                _merge_if_zero(team, shooting, "three_pt_pct")
                for k, v in shooting.items():
                    stats[k] = v

        if stats:
            team["enriched_stats"] = stats

    total = len(teams)
    if ff_data is not None:
        logger.info(
            "Four Factors enrichment (%d): matched %d/%d teams",
            ff_year, ff_matched, total,
        )
    else:
        logger.warning("No Four Factors data found for year %d or recent fallbacks", year)

    if shooting_data is not None:
        logger.info(
            "Shooting enrichment (%d): matched %d/%d teams",
            shooting_year, shooting_matched, total,
        )
    else:
        logger.warning("No shooting data found for year %d or recent fallbacks", year)

    return torvik_data


def _merge_if_zero(team: dict, source: dict, field: str):
    """Copy field from source to team if team's value is zero or missing."""
    current = team.get(field, 0.0)
    if current == 0.0 or current == 1.0:  # 1.0 is also a sentinel for ORB%/DRB%
        new_val = source.get(field)
        if new_val is not None and new_val != 0.0:
            team[field] = new_val


def _fuzzy_lookup(data: dict, team_id: str) -> Optional[dict]:
    """Try common team ID variations to find a match.

    Handles cases like:
    - "miami_fl" vs "miami__fl" (double underscore)
    - "n_c_state" vs "nc_state"
    - "st_john_s" vs "st_johns"
    """
    # Strip trailing state qualifiers and retry
    variations = [
        team_id,
        team_id.replace("__", "_"),
        team_id.replace("_", ""),
        # Also try expanding single underscores to double (reverse of collapse)
    ]
    # Check if any data keys with collapsed underscores match
    collapsed_id = team_id.replace("_", "")
    for key in data:
        if key.replace("_", "") == collapsed_id:
            return data[key]
    # Handle "n_c_" -> "nc_" pattern
    if "_c_" in team_id:
        variations.append(team_id.replace("_c_", "c_"))
    # Handle apostrophe stripping: "st_john_s" -> "st_johns"
    if team_id.endswith("_s"):
        variations.append(team_id[:-2] + "s")

    for v in variations:
        if v in data:
            return data[v]
    return None
