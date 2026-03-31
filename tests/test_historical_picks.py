"""Tests for historical public pick data loader."""

import json
import pytest
from pathlib import Path

from src.data.historical_picks import (
    load_historical_public_picks,
    _build_seed_based_picks,
    get_available_years,
)
from src.data.seed_pick_model import SEED_PICK_RATES as _SEED_PICK_RATES


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


class TestSeedPickRatesEspnAnchors:
    """Validate SEED_PICK_RATES against known ESPN BTC anchor points.

    These anchors are from ESPN "Who Picked Whom" aggregate data
    (2015-2024).  The model must stay within tolerance of these
    empirical values to ensure the leverage/differentiation
    calculation uses realistic public pick estimates.
    """

    # (seed, round) -> (espn_value, max_absolute_error)
    ESPN_ANCHORS = {
        (1, "R64"):   (0.97,  0.02),
        (2, "R64"):   (0.93,  0.02),
        (5, "R64"):   (0.64,  0.02),
        (8, "R64"):   (0.52,  0.02),
        (12, "R64"):  (0.36,  0.02),
        (16, "R64"):  (0.02,  0.02),
        (1, "F4"):    (0.42,  0.03),
        (2, "F4"):    (0.22,  0.03),
        (1, "CHAMP"): (0.18,  0.02),
        (2, "CHAMP"): (0.06,  0.02),
    }

    def test_anchors_within_tolerance(self):
        for (seed, rnd), (target, tol) in self.ESPN_ANCHORS.items():
            model_val = _SEED_PICK_RATES[seed][rnd]
            assert abs(model_val - target) <= tol, (
                f"Seed {seed} {rnd}: model={model_val:.4f}, "
                f"ESPN={target:.4f}, tolerance={tol}"
            )

    def test_champ_sums_to_one(self):
        """Championship picks across all 64 teams must sum to ~100%."""
        total = sum(_SEED_PICK_RATES[s]["CHAMP"] * 4 for s in range(1, 17))
        assert 0.98 <= total <= 1.02, f"CHAMP sum = {total:.4f}"

    def test_1_seed_champ_dominates(self):
        """1-seeds should have highest championship pick rate."""
        for seed in range(2, 17):
            assert _SEED_PICK_RATES[1]["CHAMP"] > _SEED_PICK_RATES[seed]["CHAMP"], (
                f"Seed 1 CHAMP ({_SEED_PICK_RATES[1]['CHAMP']:.4f}) should exceed "
                f"seed {seed} ({_SEED_PICK_RATES[seed]['CHAMP']:.4f})"
            )

    def test_f4_concentration_on_top_seeds(self):
        """F4 pick rates for 1-2 seeds should exceed their advancement rates."""
        from src.data.seed_pick_model import _compute_advancement_rates
        adv = _compute_advancement_rates()
        for seed in (1, 2):
            assert _SEED_PICK_RATES[seed]["F4"] > adv[seed]["F4"], (
                f"Seed {seed} F4 pick rate should exceed advancement rate "
                f"(public over-picks top seeds for F4)"
            )


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
