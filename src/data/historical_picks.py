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

logger = logging.getLogger(__name__)

# Default directory for archived ESPN pick data
_DEFAULT_PICKS_DIR = Path("data/raw/historical_public_picks")

# Round names matching the rest of the codebase
_ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]


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
            year, len(result),
        )
        return result

    # Track A: seed-based fallback
    logger.info(
        "No archived public picks for %d, using seed-based approximation",
        year,
    )
    return _build_seed_based_picks(bracket_teams)


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
            for team_id, rounds in teams_data.items():
                if isinstance(rounds, dict):
                    result[team_id] = {
                        r: float(rounds.get(r, 0.0))
                        for r in _ROUND_NAMES
                    }

            # Fill in any missing teams with seed-based defaults
            for team_id, seed in bracket_teams.items():
                if team_id not in result:
                    result[team_id] = _seed_pick_rates(seed)

            if len(result) >= 32:  # Sanity: at least half the bracket
                return result
            else:
                logger.warning(
                    "Archived picks for %d have only %d teams, discarding",
                    year, len(result),
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

    Duplicated from ``optimization.leverage.SEED_PICK_RATES`` to avoid
    circular imports.  Values are calibrated from ESPN Tournament
    Challenge aggregate data (2015-2024).
    """
    rates = _SEED_PICK_RATES.get(seed, _SEED_PICK_RATES[8])
    return dict(rates)


# Calibrated from ESPN Tournament Challenge aggregate data (2015-2024).
# Source: optimization/leverage.py SEED_PICK_RATES
_SEED_PICK_RATES: Dict[int, Dict[str, float]] = {
    1:  {"R64": 0.97, "R32": 0.90, "S16": 0.75, "E8": 0.55, "F4": 0.35, "CHAMP": 0.18},
    2:  {"R64": 0.94, "R32": 0.82, "S16": 0.58, "E8": 0.35, "F4": 0.18, "CHAMP": 0.08},
    3:  {"R64": 0.85, "R32": 0.65, "S16": 0.38, "E8": 0.18, "F4": 0.08, "CHAMP": 0.03},
    4:  {"R64": 0.80, "R32": 0.55, "S16": 0.28, "E8": 0.12, "F4": 0.05, "CHAMP": 0.02},
    5:  {"R64": 0.65, "R32": 0.38, "S16": 0.18, "E8": 0.07, "F4": 0.03, "CHAMP": 0.01},
    6:  {"R64": 0.63, "R32": 0.35, "S16": 0.15, "E8": 0.06, "F4": 0.02, "CHAMP": 0.008},
    7:  {"R64": 0.60, "R32": 0.30, "S16": 0.12, "E8": 0.05, "F4": 0.02, "CHAMP": 0.006},
    8:  {"R64": 0.50, "R32": 0.22, "S16": 0.08, "E8": 0.03, "F4": 0.01, "CHAMP": 0.003},
    9:  {"R64": 0.50, "R32": 0.20, "S16": 0.07, "E8": 0.02, "F4": 0.008, "CHAMP": 0.002},
    10: {"R64": 0.40, "R32": 0.15, "S16": 0.05, "E8": 0.02, "F4": 0.006, "CHAMP": 0.001},
    11: {"R64": 0.37, "R32": 0.15, "S16": 0.06, "E8": 0.02, "F4": 0.007, "CHAMP": 0.001},
    12: {"R64": 0.35, "R32": 0.15, "S16": 0.05, "E8": 0.02, "F4": 0.005, "CHAMP": 0.001},
    13: {"R64": 0.20, "R32": 0.06, "S16": 0.02, "E8": 0.005, "F4": 0.001, "CHAMP": 0.0003},
    14: {"R64": 0.15, "R32": 0.04, "S16": 0.01, "E8": 0.003, "F4": 0.0005, "CHAMP": 0.0001},
    15: {"R64": 0.06, "R32": 0.02, "S16": 0.005, "E8": 0.001, "F4": 0.0002, "CHAMP": 0.00005},
    16: {"R64": 0.03, "R32": 0.005, "S16": 0.001, "E8": 0.0002, "F4": 0.00003, "CHAMP": 0.00001},
}


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
