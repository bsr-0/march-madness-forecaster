"""Data enrichment for conference tournament predictions.

Merges additional statistics from separate Torvik data files
(Four Factors, shooting splits) into the base team data, filling
in fields that are missing or zero in the main torvik_YYYY.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.data.normalize import normalize_team_id as _canonical_id

logger = logging.getLogger(__name__)


def _try_load_json(path: str) -> Optional[dict]:
    """Load a JSON file, returning None if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _find_data_file(
    base_dir: str,
    prefix: str,
    year: int,
    strict_leakage: bool = False,
) -> Tuple[Optional[dict], int]:
    """Find data file for the given year, falling back to most recent prior year.

    Args:
        base_dir: Directory containing data files.
        prefix: File prefix (e.g. "torvik_four_factors").
        year: Target year.
        strict_leakage: When True, reject files whose ``data_as_of`` or
            ``scraped_at`` is on/after the tournament start date.

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
            # Leakage guard: reject data scraped after tournament start
            if strict_leakage and isinstance(data, dict):
                _ts_fields = ["data_as_of", "timestamp", "generated_at", "fetched_at", "scraped_at"]
                _ts = next((data.get(f) for f in _ts_fields if data.get(f)), None)
                if _ts:
                    try:
                        from datetime import date as _date
                        from src.pipeline.config import TOURNAMENT_START_DATES
                        _ts_date = _date.fromisoformat(str(_ts)[:10])
                        _cutoff = TOURNAMENT_START_DATES.get(y)
                        if _cutoff and _ts_date >= _cutoff:
                            logger.warning(
                                "%s has timestamp %s on/after tournament start %s "
                                "— rejecting to prevent leakage.",
                                path, _ts, _cutoff,
                            )
                            continue
                    except (ValueError, TypeError):
                        pass
            return data, y
    return None, 0


def enrich_torvik_teams(
    torvik_data: dict,
    data_dir: str = "data/raw",
    year: int = 2026,
    strict_leakage: bool = False,
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
    ff_data, ff_year = _find_data_file(data_dir, "torvik_four_factors", year, strict_leakage=strict_leakage)
    shooting_data, shooting_year = _find_data_file(data_dir, "torvik_shooting", year, strict_leakage=strict_leakage)

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

    Uses canonical normalization to bridge different ID schemes:
    - "michigan_st" vs "michigan_state" (abbreviation vs full)
    - "miami_fl" vs "miami__fl" (single vs double underscore)
    - "smu" vs "southern_methodist" (abbreviation expansion)
    - "queens" vs "queens__nc" (with/without state qualifier)
    - "prairie_view_a_m" vs "prairie_view" (with/without A&M suffix)
    """
    # Strategy 1: Normalize the lookup key and try to find a match
    # in a canonicalized index of the data keys.
    canonical_id = _canonical_id(team_id)
    if canonical_id in data:
        return data[canonical_id]

    # Strategy 2: Build reverse index — normalize each data key and match.
    # This handles the case where *data* keys use non-canonical IDs.
    for key in data:
        if _canonical_id(key) == canonical_id:
            return data[key]

    # Strategy 3: Legacy heuristics for edge cases the normalizer might miss.
    collapsed_id = team_id.replace("_", "")
    for key in data:
        if key.replace("_", "") == collapsed_id:
            return data[key]

    # Strategy 4: Disambiguation — match keys that start with our ID
    # (e.g. "miami" matches "miami__fl" or "miami__oh").  Only use
    # if there's exactly one such match to avoid ambiguity.
    prefix_matches = [k for k in data if k.startswith(team_id + "_") or k.startswith(team_id + "__")]
    if len(prefix_matches) == 1:
        return data[prefix_matches[0]]

    # Strategy 5: Reverse prefix — our ID starts with a data key
    # (e.g. "miami__fl" matches data key "miami")
    for key in data:
        if team_id.startswith(key + "_") or team_id.startswith(key + "__"):
            return data[key]

    return None
