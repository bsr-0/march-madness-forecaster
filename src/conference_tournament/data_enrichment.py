"""Data enrichment for conference tournament predictions.

Merges additional statistics from separate Torvik data files
(Four Factors, shooting splits) into the base team data, filling
in fields that are missing or zero in the main torvik_YYYY.json.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.normalize import normalize_team_id as _canonical_id

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


def compute_defensive_four_factors_from_games(
    games: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute defensive Four Factors from game box scores.

    When the Torvik HTML scraper fails (e.g. JavaScript wall) and the CSV
    fallback cannot provide defensive metrics, this function reconstructs
    them from historical game records.  Each game appears as two records
    (one per team) paired by ``game_id``.

    Returns a dict mapping team_id -> {opp_effective_fg_pct, opp_turnover_rate,
    opp_free_throw_rate}.
    """
    # Pair game records by game_id
    game_pairs: Dict[str, List[Dict]] = defaultdict(list)
    for g in games:
        gid = g.get("game_id")
        if gid:
            game_pairs[gid].append(g)

    # Accumulate opponent stats per team
    team_opp: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"fgm": 0, "fga": 0, "fg3m": 0, "fta": 0, "turnovers": 0, "games": 0}
    )

    for gid, sides in game_pairs.items():
        if len(sides) != 2:
            continue
        for i, my_side in enumerate(sides):
            opp_side = sides[1 - i]
            my_id = (
                my_side.get("team_id")
                or my_side.get("team_name", "")
            )
            if not my_id:
                continue
            # Normalize team_id: lowercase, spaces to underscores
            my_id = str(my_id).lower().replace(" ", "_")

            opp_fga = float(opp_side.get("fga") or 0)
            if opp_fga < 1:
                continue

            s = team_opp[my_id]
            s["fgm"] += float(opp_side.get("fgm") or 0)
            s["fga"] += opp_fga
            s["fg3m"] += float(opp_side.get("fg3m") or 0)
            s["fta"] += float(opp_side.get("fta") or 0)
            s["turnovers"] += float(opp_side.get("turnovers") or 0)
            s["games"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for team_id, s in team_opp.items():
        if s["fga"] < 10:
            continue
        opp_efg = (s["fgm"] + 0.5 * s["fg3m"]) / s["fga"]
        denom_to = s["fga"] + 0.44 * s["fta"] + s["turnovers"]
        opp_to_rate = s["turnovers"] / denom_to if denom_to > 0 else 0.0
        opp_ftr = s["fta"] / s["fga"]
        result[team_id] = {
            "opp_effective_fg_pct": round(opp_efg, 4),
            "opp_turnover_rate": round(opp_to_rate, 4),
            "opp_free_throw_rate": round(opp_ftr, 4),
        }

    logger.info(
        "Computed defensive Four Factors from game box scores for %d teams",
        len(result),
    )
    return result


def compute_offensive_four_factors_from_games(
    games: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute offensive rebound/turnover rates from game box scores.

    When the Torvik CSV fallback is used, individual player ORB%/DRB%/TO%
    cannot be converted to team-level rates (they are not additive).  This
    function reconstructs team-level offensive Four Factors from paired
    game records that contain counting stats (fga, fta, turnovers, orb, drb).

    Returns a dict mapping team_id -> {turnover_rate, offensive_reb_rate,
    defensive_reb_rate}.
    """
    game_pairs: Dict[str, List[Dict]] = defaultdict(list)
    for g in games:
        gid = g.get("game_id")
        if gid:
            game_pairs[gid].append(g)

    team_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"fga": 0, "fta": 0, "turnovers": 0, "orb": 0, "drb": 0,
                 "opp_orb": 0, "opp_drb": 0, "games": 0}
    )

    for gid, sides in game_pairs.items():
        if len(sides) != 2:
            continue
        for i, my_side in enumerate(sides):
            opp_side = sides[1 - i]
            my_id = str(my_side.get("team_id") or my_side.get("team_name", "")).lower().replace(" ", "_")
            if not my_id:
                continue

            my_fga = float(my_side.get("fga") or 0)
            my_orb = float(my_side.get("orb") or 0)
            my_drb = float(my_side.get("drb") or 0)
            my_to = float(my_side.get("turnovers") or 0)
            my_fta = float(my_side.get("fta") or 0)
            opp_orb = float(opp_side.get("orb") or 0)
            opp_drb = float(opp_side.get("drb") or 0)

            if my_fga < 1:
                continue

            s = team_stats[my_id]
            s["fga"] += my_fga
            s["fta"] += my_fta
            s["turnovers"] += my_to
            s["orb"] += my_orb
            s["drb"] += my_drb
            s["opp_orb"] += opp_orb
            s["opp_drb"] += opp_drb
            s["games"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for team_id, s in team_stats.items():
        if s["games"] < 3 or s["fga"] < 30:
            continue

        # TO% = turnovers / (FGA + 0.44*FTA + turnovers)
        denom_to = s["fga"] + 0.44 * s["fta"] + s["turnovers"]
        to_rate = s["turnovers"] / denom_to if denom_to > 0 else 0.0

        # ORB% = team ORB / (team ORB + opponent DRB)
        orb_chances = s["orb"] + s["opp_drb"]
        orb_rate = s["orb"] / orb_chances if orb_chances > 0 else 0.0

        # DRB% = team DRB / (team DRB + opponent ORB)
        drb_chances = s["drb"] + s["opp_orb"]
        drb_rate = s["drb"] / drb_chances if drb_chances > 0 else 0.0

        result[team_id] = {
            "turnover_rate": round(to_rate, 4),
            "offensive_reb_rate": round(orb_rate, 4),
            "defensive_reb_rate": round(drb_rate, 4),
        }

    logger.info(
        "Computed offensive Four Factors (TO%%/ORB%%/DRB%%) from game box scores for %d teams",
        len(result),
    )
    return result
