"""Tests for historical public pick data loader."""

import json
import pytest
from pathlib import Path

from src.data.historical_picks import (
    load_historical_public_picks,
    _build_seed_based_picks,
    get_available_years,
    _SEED_PICK_RATES,
)


@pytest.fixture
def bracket_teams():
    """64 teams with seeds 1-16 (4 per seed)."""
    teams = {}
    for seed in range(1, 17):
        for region_idx, region in enumerate(["east", "west", "south", "midwest"]):
            team_id = f"team_{seed}_{region}"
            teams[team_id] = seed
    return teams


class TestSeedBasedPicks:
    def test_returns_64_teams(self, bracket_teams):
        result = _build_seed_based_picks(bracket_teams)
        assert len(result) == 64

    def test_each_team_has_6_rounds(self, bracket_teams):
        result = _build_seed_based_picks(bracket_teams)
        for team_id, rounds in result.items():
            assert len(rounds) == 6, f"{team_id} has {len(rounds)} rounds"
            for r in ["R64", "R32", "S16", "E8", "F4", "CHAMP"]:
                assert r in rounds

    def test_1_seed_r64_high(self, bracket_teams):
        """1-seeds should have R64 pick rate > 0.90."""
        result = _build_seed_based_picks(bracket_teams)
        for team_id, seed in bracket_teams.items():
            if seed == 1:
                assert result[team_id]["R64"] > 0.90

    def test_16_seed_r64_low(self, bracket_teams):
        """16-seeds should have R64 pick rate < 0.10."""
        result = _build_seed_based_picks(bracket_teams)
        for team_id, seed in bracket_teams.items():
            if seed == 16:
                assert result[team_id]["R64"] < 0.10

    def test_rates_decrease_by_round(self, bracket_teams):
        """Pick rates should decrease for later rounds."""
        result = _build_seed_based_picks(bracket_teams)
        for team_id, rounds in result.items():
            for i in range(len(["R64", "R32", "S16", "E8", "F4", "CHAMP"]) - 1):
                r1 = ["R64", "R32", "S16", "E8", "F4", "CHAMP"][i]
                r2 = ["R64", "R32", "S16", "E8", "F4", "CHAMP"][i + 1]
                assert rounds[r1] >= rounds[r2], (
                    f"{team_id}: {r1}={rounds[r1]} < {r2}={rounds[r2]}"
                )


class TestLoadHistoricalPicks:
    def test_fallback_to_seed_based(self, bracket_teams, tmp_path):
        """When no archived data exists, should fall back to seed-based."""
        result = load_historical_public_picks(2023, bracket_teams, picks_dir=tmp_path)
        assert len(result) == 64

    def test_loads_archived_data(self, bracket_teams, tmp_path):
        """Should load real data when JSON file exists."""
        archived = {
            "year": 2023,
            "source": "espn_who_picked_whom",
            "teams": {
                f"team_1_east": {
                    "R64": 0.98, "R32": 0.92, "S16": 0.80,
                    "E8": 0.60, "F4": 0.40, "CHAMP": 0.20,
                },
            },
        }
        filepath = tmp_path / "espn_picks_2023.json"
        filepath.write_text(json.dumps(archived))

        result = load_historical_public_picks(2023, bracket_teams, picks_dir=tmp_path)
        # Should have the archived team + seed-based fallbacks
        assert len(result) == 64
        assert result["team_1_east"]["R64"] == 0.98

    def test_invalid_json_falls_back(self, bracket_teams, tmp_path):
        """Invalid JSON should fall back gracefully."""
        filepath = tmp_path / "espn_picks_2023.json"
        filepath.write_text("not valid json{{{")

        result = load_historical_public_picks(2023, bracket_teams, picks_dir=tmp_path)
        assert len(result) == 64  # Falls back to seed-based


class TestGetAvailableYears:
    def test_empty_dir(self, tmp_path):
        assert get_available_years(tmp_path) == []

    def test_finds_years(self, tmp_path):
        (tmp_path / "espn_picks_2023.json").write_text("{}")
        (tmp_path / "espn_picks_2024.json").write_text("{}")
        years = get_available_years(tmp_path)
        assert 2023 in years
        assert 2024 in years


class TestSeedPickRatesTable:
    def test_all_seeds_present(self):
        for seed in range(1, 17):
            assert seed in _SEED_PICK_RATES

    def test_rates_in_valid_range(self):
        for seed, rates in _SEED_PICK_RATES.items():
            for round_name, rate in rates.items():
                assert 0 <= rate <= 1, f"Seed {seed} {round_name}: {rate}"
