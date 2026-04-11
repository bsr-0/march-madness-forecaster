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

The loader tries Track B first and falls back to Track A only when no
archived file exists for a year.  Within Track B, any team that cannot
be resolved to a bracket_teams key is logged as a WARNING (not silently
filled with seed defaults).  Individual bracket teams missing from the
picks file are explicitly seed-filled and logged.
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
    # Abbreviations / disambiguation
    "miami": "miami__fl",
    "umass": "massachusetts",
    "s_dakota_state": "south_dakota_state",
    "j_ville_state": "jacksonville_state",
    "western_ky": "western_kentucky",
    "mount_st_marys": "mount_st__mary_s",
    "norf_app": "norfolk_state",
    "virginia_commonwealth": "vcu",
    "louisiana": "louisiana_lafayette",
    "louisiana_lafayette": "louisiana",  # Bracket ID varies by year
    "siue": "siu_edwardsville",
    "sam_houston_state": "sam_houston",
    "liu_brooklyn": "long_island_university",
    # Play-in combined entries: ESPN lumps both play-in teams into one row
    # before the play-in game is decided. Map to the team that won the
    # play-in and entered the main bracket.
    "_playin_11_pr_sc": "southern_california",  # 2017: Providence/USC
    "_playin_16_nc_ud": "uc_davis",  # 2017: NC Central/UC Davis
    "playin_11_pr_sc": "southern_california",  # normalized (leading _ stripped)
    "playin_16_nc_ud": "uc_davis",
    "_playin_11_asu_sju": "arizona_state",  # 2019: Arizona St/St. John's
    "playin_11_asu_sju": "arizona_state",
    "michigan_state_ucla": "ucla",  # 2021
    "msm_texas_southern": "texas_southern",  # 2021: Mt St Mary's/TX Southern
    "wichita_state_drake": "drake",  # 2021
    "asu_nev": "arizona_state",  # 2023: Arizona St/Nevada
    "msst_pitt": "pittsburgh",  # 2023: Miss St/Pittsburgh
    "txso_fdu": "fairleigh_dickinson",  # 2023: TX Southern/FDU
}


def load_historical_public_picks(
    year: int,
    bracket_teams: Dict[str, int],
    picks_dir: Optional[Path] = None,
    require_archived: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Load public pick distribution for a historical tournament year.

    Tries Track B (archived real data) first, then falls back to Track A
    (seed-based approximation).

    Args:
        year: Tournament year (e.g. 2023).
        bracket_teams: team_id -> seed (1-16) for all 64 tournament teams.
        picks_dir: Directory containing archived JSON files.  Defaults to
            ``data/raw/historical_public_picks/``.
        require_archived: If True, raise FileNotFoundError when no archived
            file exists instead of falling back to the seed-based track.
            Pool-optimizer callers set this to True so missing real picks
            fail loudly.

    Returns:
        Dict mapping team_id -> {round_name: pick_pct} where pick_pct
        is in [0, 1].  Contains entries for all 64 teams and 6 rounds.

    Raises:
        FileNotFoundError: If ``require_archived`` is True and no archived
            picks file exists for ``year``.
    """
    picks_dir = Path(picks_dir) if picks_dir else _DEFAULT_PICKS_DIR

    # Track B: try archived real data
    result = _load_archived_picks(
        year, bracket_teams, picks_dir, strict=require_archived
    )
    if result is not None:
        logger.info(
            "Loaded archived ESPN public picks for %d (%d teams)",
            year,
            len(result),
        )
        return result

    if require_archived:
        raise FileNotFoundError(
            f"No archived ESPN public picks for {year} in {picks_dir}. "
            f"Expected one of: espn_picks_{year}.json, public_picks_{year}.json, "
            f"{year}.json. Pool optimizer requires real public pick data — "
            f"refresh via .github/workflows/espn-picks-ingestion.yml or "
            f"scripts/scrape_historical_espn_picks.py."
        )

    # Track A: seed-based fallback
    logger.info(
        "No archived public picks for %d, using seed-based approximation",
        year,
    )
    return _build_seed_based_picks(bracket_teams)


def _resolve_picks_team_id(
    raw_id: str,
    bracket_teams: Dict[str, int],
) -> Optional[str]:
    """Resolve a picks-file team name to a canonical bracket_teams key.

    Tries in order: direct match, picks alias table, normalize_team_id,
    normalize + picks alias.  Returns None if nothing matches — callers
    must handle unresolved names explicitly (no silent fallback).
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
    return None


def _load_archived_picks(
    year: int,
    bracket_teams: Dict[str, int],
    picks_dir: Path,
    strict: bool = False,
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
                if strict:
                    raise ValueError(
                        f"Invalid format in {filepath}: 'teams' is not a dict"
                    )
                logger.warning("Invalid format in %s: 'teams' is not a dict", filepath)
                continue

            result: Dict[str, Dict[str, float]] = {}
            unresolved: list[str] = []
            for raw_id, rounds in teams_data.items():
                if not isinstance(rounds, dict):
                    continue
                pick_data = {r: float(rounds.get(r, 0.0)) for r in _ROUND_NAMES}
                team_id = _resolve_picks_team_id(raw_id, bracket_teams)
                if team_id is not None:
                    result[team_id] = pick_data
                else:
                    unresolved.append(raw_id)

            n_matched = sum(1 for t in bracket_teams if t in result)
            n_bracket = len(bracket_teams)
            missing_bracket = [t for t in bracket_teams if t not in result]

            if unresolved:
                logger.warning(
                    "Picks %d: %d team(s) in %s could NOT be resolved to "
                    "bracket_teams and were DROPPED (no silent fallback): %s",
                    year,
                    len(unresolved),
                    filepath.name,
                    unresolved,
                )
            if missing_bracket:
                logger.warning(
                    "Picks %d: %d bracket team(s) have NO ESPN pick data (will use seed-based rates): %s",
                    year,
                    len(missing_bracket),
                    missing_bracket,
                )

            # Fill missing bracket teams with seed-based defaults.
            # This is now explicit and logged — not a silent fallback.
            for team_id in missing_bracket:
                seed = bracket_teams[team_id]
                result[team_id] = _seed_pick_rates(seed)

            match_pct = n_matched / n_bracket * 100 if n_bracket else 0
            logger.info(
                "Picks for %d: %d/%d matched (%.0f%%), %d unresolved, %d bracket-only (seed-filled)",
                year,
                n_matched,
                n_bracket,
                match_pct,
                len(unresolved),
                len(missing_bracket),
            )

            if match_pct < 75:
                msg = (
                    f"Picks {year}: match rate {match_pct:.0f}% is below 75% "
                    f"threshold. ESPN data for this year is unreliable. Fix "
                    f"team-name aliases in _PICKS_TEAM_ALIAS or exclude this year."
                )
                if strict:
                    raise ValueError(msg)
                logger.error(msg)

            if len(result) >= 32:
                return result
            else:
                if strict:
                    raise ValueError(
                        f"Archived picks for {year} have only {len(result)} "
                        f"teams (need >=32), discarding"
                    )
                logger.warning(
                    "Archived picks for %d have only %d teams, discarding",
                    year,
                    len(result),
                )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if strict:
                raise
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
