"""Historical public pick data loader for ESPN LOYO backtesting.

Provides two data tracks for historical public pick distributions:

Track A (seed-based, always available):
    Uses the ``SEED_PICK_RATES`` table from ``optimization.leverage``
    (calibrated from ESPN 2015-2024 aggregate data) to generate
    team-level pick distributions from seed assignments alone.

Track B (archived real data, higher fidelity):
    Loads per-team pick distributions from JSON files in
    ``data/raw/historical_public_picks/espn_picks_{year}.json``,
    sourced from ESPN "Who Picked Whom" via Wayback Machine or
    manual curation.

The loader tries Track B first and falls back to Track A seamlessly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from src.data.normalize import normalize_team_id

logger = logging.getLogger(__name__)

# Default directory for archived ESPN pick data
_DEFAULT_PICKS_DIR = Path("data/raw/historical_public_picks")

# Round names matching the rest of the codebase
_ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

# Aliases for picks-file team names that don't resolve through normalize_team_id.
# These cover abbreviations, disambiguation, and play-in artifacts found in
# the ESPN/Kaggle scraped data (diagnosed via scripts/diagnose_picks_team_matching.py).
_PICKS_TEAM_ALIAS: Dict[str, str] = {
    "miami": "miami__fl",
    "umass": "massachusetts",
    "s_dakota_state": "south_dakota_state",
    "j_ville_state": "jacksonville_state",
    "western_ky": "western_kentucky",
    "mount_st_marys": "mount_st__mary_s",
    "norf_app": "norfolk_state",
    "virginia_commonwealth": "vcu",
    "louisiana": "louisiana_lafayette",
    "siue": "siu_edwardsville",
}


def load_historical_public_picks(
    year: int,
    bracket_teams: Dict[str, int],
    picks_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Load public pick distribution for a historical tournament year.

    Tries Track B (archived real data) first, then falls back to Track A
    (seed-based approximation).

    Args:
        year: Tournament year (e.g. 2023).
        bracket_teams: team_id -> seed (1-16) for all 64 tournament teams.
        picks_dir: Directory containing archived JSON files.  Defaults to
            ``data/raw/historical_public_picks/``.

    Returns:
        Dict mapping team_id -> {round_name: pick_pct} where pick_pct
        is in [0, 1].  Contains entries for all 64 teams and 6 rounds.
    """
    picks_dir = picks_dir or _DEFAULT_PICKS_DIR

    # Track B: try archived real data
    result = _load_archived_picks(year, bracket_teams, picks_dir)
    if result is not None:
        logger.info(
            "Loaded archived ESPN public picks for %d (%d teams)",
            year,
            len(result),
        )
        return result

    # Track A: seed-based fallback
    logger.info(
        "No archived public picks for %d, using seed-based approximation",
        year,
    )
    return _build_seed_based_picks(bracket_teams)


def _resolve_picks_team_id(
    raw_id: str,
    bracket_teams: Dict[str, int],
) -> str:
    """Resolve a picks-file team name to a canonical bracket_teams key.

    Tries in order: direct match, picks alias table, normalize_team_id,
    normalize + picks alias.  Returns the original raw_id if nothing matches
    (it will be kept in results but won't collide with bracket_teams keys,
    so it effectively gets ignored during the fill-in step).
    """
    if raw_id in bracket_teams:
        return raw_id
    if raw_id in _PICKS_TEAM_ALIAS and _PICKS_TEAM_ALIAS[raw_id] in bracket_teams:
        return _PICKS_TEAM_ALIAS[raw_id]
    norm = normalize_team_id(raw_id)
    if norm in bracket_teams:
        return norm
    if norm in _PICKS_TEAM_ALIAS and _PICKS_TEAM_ALIAS[norm] in bracket_teams:
        return _PICKS_TEAM_ALIAS[norm]
    return raw_id


def _load_archived_picks(
    year: int,
    bracket_teams: Dict[str, int],
    picks_dir: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load archived ESPN public pick data from JSON.

    Expected JSON schema::

        {
            "year": 2023,
            "source": "espn_who_picked_whom",
            "teams": {
                "team_id": {
                    "R64": 0.97,
                    "R32": 0.85,
                    "S16": 0.60,
                    "E8": 0.35,
                    "F4": 0.18,
                    "CHAMP": 0.08
                },
                ...
            }
        }

    Returns:
        Dict mapping team_id -> {round: pick_pct}, or None if file
        not found or invalid.
    """
    candidates = [
        picks_dir / f"espn_picks_{year}.json",
        picks_dir / f"public_picks_{year}.json",
        picks_dir / f"{year}.json",
    ]

    for filepath in candidates:
        if not filepath.exists():
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)

            teams_data = data.get("teams", data)
            if not isinstance(teams_data, dict):
                logger.warning("Invalid format in %s: 'teams' is not a dict", filepath)
                continue

            result: Dict[str, Dict[str, float]] = {}
            for raw_id, rounds in teams_data.items():
                if not isinstance(rounds, dict):
                    continue
                pick_data = {r: float(rounds.get(r, 0.0)) for r in _ROUND_NAMES}
                # Normalize picks team name to match bracket_teams keys
                team_id = _resolve_picks_team_id(raw_id, bracket_teams)
                result[team_id] = pick_data

            # Fill in any missing teams with seed-based defaults
            n_matched = sum(1 for t in bracket_teams if t in result)
            for team_id, seed in bracket_teams.items():
                if team_id not in result:
                    result[team_id] = _seed_pick_rates(seed)

            if n_matched < len(bracket_teams):
                logger.info(
                    "Picks for %d: %d/%d bracket teams matched, %d fell back to seed-based",
                    year,
                    n_matched,
                    len(bracket_teams),
                    len(bracket_teams) - n_matched,
                )

            if len(result) >= 32:  # Sanity: at least half the bracket
                return result
            else:
                logger.warning(
                    "Archived picks for %d have only %d teams, discarding",
                    year,
                    len(result),
                )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to load %s: %s", filepath, exc)

    return None


def _build_seed_based_picks(
    bracket_teams: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Build public pick distribution from seed data alone (Track A).

    Uses the ``SEED_PICK_RATES`` table calibrated from ESPN aggregate
    data (2015-2024).  This is the fallback when no archived real
    data is available.
    """
    result: Dict[str, Dict[str, float]] = {}
    for team_id, seed in bracket_teams.items():
        result[team_id] = _seed_pick_rates(seed)
    return result


def _seed_pick_rates(seed: int) -> Dict[str, float]:
    """Return approximate public pick rates for a given seed.

    Uses the principled model from ``src.data.seed_pick_model`` which
    derives rates from historical advancement probabilities (1985-2025)
    and a chalk bias transformation calibrated against ESPN BTC data.
    """
    from src.data.seed_pick_model import SEED_PICK_RATES

    rates = SEED_PICK_RATES.get(seed, SEED_PICK_RATES[8])
    return dict(rates)


def get_available_years(picks_dir: Optional[Path] = None) -> list[int]:
    """List years for which archived public pick data is available."""
    picks_dir = picks_dir or _DEFAULT_PICKS_DIR
    if not picks_dir.exists():
        return []

    years = []
    for f in picks_dir.glob("*_picks_*.json"):
        # Extract year from filename like espn_picks_2023.json
        stem = f.stem
        parts = stem.split("_")
        for part in parts:
            if part.isdigit() and 2010 <= int(part) <= 2030:
                years.append(int(part))
                break

    for f in picks_dir.glob("[0-9][0-9][0-9][0-9].json"):
        year = int(f.stem)
        if 2010 <= year <= 2030 and year not in years:
            years.append(year)

    return sorted(years)
