"""Regression tests for team ID normalization consistency across cached data.

Ensures that:
1. All tournament seed teams match Torvik teams after normalization
2. All tournament result teams match seed teams after normalization
3. normalize_team_id is idempotent (stable under re-application)
4. The generic _st suffix rule works for all known State schools
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.normalize import normalize_team_id

HIST_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "historical"

TRAINING_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
HOLDOUT_YEARS = [2025]
ALL_YEARS = TRAINING_YEARS + HOLDOUT_YEARS


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _extract_seed_ids(year: int) -> set[str]:
    path = HIST_DIR / f"tournament_seeds_{year}.json"
    if not path.exists():
        return set()
    data = _load_json(path)
    teams = data.get("teams", []) if isinstance(data, dict) else data
    return {normalize_team_id(t["team_id"]) for t in teams if isinstance(t, dict) and t.get("team_id")}


def _extract_torvik_ids(year: int) -> set[str]:
    path = HIST_DIR / f"torvik_{year}.json"
    if not path.exists():
        return set()
    data = _load_json(path)
    teams = data.get("teams", []) if isinstance(data, dict) else data
    return {normalize_team_id(t["team_id"]) for t in teams if isinstance(t, dict) and t.get("team_id")}


def _extract_result_ids(year: int) -> set[str]:
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        return set()
    data = _load_json(path)
    ids = set()
    for g in data.get("games", []):
        if not isinstance(g, dict):
            continue
        for key in ("team1_id", "team2_id", "winner_id", "loser_id"):
            tid = g.get(key)
            if tid:
                ids.add(normalize_team_id(tid))
    return ids


# ---------------------------------------------------------------------------
# Test 1: Cached data cross-source consistency (data_contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("year", ALL_YEARS)
def test_seed_teams_exist_in_torvik(year: int):
    """All tournament seed teams must have matching Torvik entries."""
    seed_ids = _extract_seed_ids(year)
    torvik_ids = _extract_torvik_ids(year)
    if not seed_ids or not torvik_ids:
        pytest.skip(f"Missing data files for {year}")
    missing = seed_ids - torvik_ids
    assert not missing, f"{year}: {len(missing)} seed team(s) not in Torvik: {sorted(missing)}"


@pytest.mark.parametrize("year", ALL_YEARS)
def test_result_teams_exist_in_seeds(year: int):
    """All tournament result teams must appear in seeds (minus First Four losers)."""
    seed_ids = _extract_seed_ids(year)
    result_ids = _extract_result_ids(year)
    if not seed_ids or not result_ids:
        pytest.skip(f"Missing data files for {year}")
    missing = result_ids - seed_ids
    # Allow up to 4 missing (First Four losers are in seeds but not results, not vice versa)
    assert len(missing) <= 4, f"{year}: {len(missing)} result team(s) not in seeds: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Test 2: normalize_team_id idempotency (unit)
# ---------------------------------------------------------------------------

def test_normalize_team_id_idempotent():
    """normalize_team_id(normalize_team_id(x)) == normalize_team_id(x) for all cached IDs."""
    all_ids: set[str] = set()

    # Collect team IDs from torvik and seeds files for all years
    for year in range(2005, 2027):
        for pattern in [f"torvik_{year}.json", f"tournament_seeds_{year}.json"]:
            path = HIST_DIR / pattern
            if not path.exists():
                continue
            data = _load_json(path)
            teams = data.get("teams", []) if isinstance(data, dict) else data
            if isinstance(teams, list):
                for t in teams:
                    if isinstance(t, dict) and t.get("team_id"):
                        all_ids.add(t["team_id"])

    assert len(all_ids) > 100, "Should have found at least 100 unique team IDs"

    failures = []
    for tid in sorted(all_ids):
        once = normalize_team_id(tid)
        twice = normalize_team_id(once)
        if once != twice:
            failures.append(f"{tid} -> {once} -> {twice}")

    assert not failures, f"normalize_team_id is not idempotent for {len(failures)} IDs:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------
# Test 3: Generic _st suffix rule (unit)
# ---------------------------------------------------------------------------

def test_st_suffix_generic_rule():
    """Any team ID ending in _st should normalize to _state (unless aliased otherwise)."""
    st_teams = [
        "iowa_st", "michigan_st", "ohio_st", "tarleton_st", "chicago_st",
        "mississippi_valley_st", "southeast_missouri_st", "nicholls_st",
        "winston_salem_st",
    ]
    for tid in st_teams:
        result = normalize_team_id(tid)
        assert result.endswith("_state"), f"{tid} -> {result} (expected _state suffix)"

    # grambling_st is explicitly aliased to 'grambling' — verify the override
    assert normalize_team_id("grambling_st") == "grambling"


# ---------------------------------------------------------------------------
# Test 4: Tournament results completeness (data_contract)
# ---------------------------------------------------------------------------

EXPECTED_ROUNDS = {"FF": 4, "R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "NCG": 1}


@pytest.mark.parametrize("year", ALL_YEARS)
def test_tournament_results_completeness(year: int):
    """Each training/holdout year must have exactly 67 games with correct round distribution."""
    path = HIST_DIR / f"tournament_results_{year}.json"
    if not path.exists():
        pytest.skip(f"No tournament_results file for {year}")
    data = _load_json(path)
    games = data.get("games", [])
    assert len(games) == 67, f"{year}: Expected 67 games, got {len(games)}"
    round_counts: dict[str, int] = {}
    for g in games:
        rnd = g.get("round_name", "")
        round_counts[rnd] = round_counts.get(rnd, 0) + 1
    for rnd, expected in EXPECTED_ROUNDS.items():
        actual = round_counts.get(rnd, 0)
        assert actual == expected, f"{year}: {rnd} has {actual} games, expected {expected}"


@pytest.mark.parametrize("year", [2016, 2017])
def test_2016_2017_results_not_empty(year: int):
    """Guard against regression to placeholder files for 2016 and 2017."""
    path = HIST_DIR / f"tournament_results_{year}.json"
    assert path.exists(), f"tournament_results_{year}.json missing"
    data = _load_json(path)
    games = data.get("games", [])
    assert len(games) == 67, f"{year}: Must have 67 games, got {len(games)} (placeholder regression?)"
    ncg = [g for g in games if g.get("round_name") == "NCG"]
    assert len(ncg) == 1, f"{year}: Expected exactly 1 NCG game"
