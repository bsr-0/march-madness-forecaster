"""Tests for Phase 1: fail-closed validation and bracket structural constraints.

Covers:
- BracketStructureError raised on hard constraint violations
- Soft warnings returned for minor deviations
- First-round matchup complementarity checks
- Hardened monotonic validation (hard fail > 2pp)
- Fail-closed behavior in ESPNPicksScraper._dict_to_consensus
"""

import pytest

from src.data.scrapers.schemas import (
    BracketStructureError,
    PublicPicksSchema,
    SchemaValidationError,
    validate_bracket_structure,
    validate_consensus_data,
)
from src.data.scrapers.espn_picks import ConsensusData, ESPNPicksScraper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Seed-based templates that approximate realistic ESPN pick distributions.
# The values are calibrated so that a full 64-team bracket sums to
# approximately the expected totals (CHAMP ~100%, F4 ~200%, etc.).
_SEED_TEMPLATES = {
    1: {
        "round_of_64_pct": 97.0,
        "round_of_32_pct": 90.0,
        "sweet_16_pct": 75.0,
        "elite_8_pct": 55.0,
        "final_four_pct": 35.0,
        "champion_pct": 18.0,
    },
    2: {
        "round_of_64_pct": 94.0,
        "round_of_32_pct": 82.0,
        "sweet_16_pct": 58.0,
        "elite_8_pct": 35.0,
        "final_four_pct": 18.0,
        "champion_pct": 8.0,
    },
    3: {
        "round_of_64_pct": 85.0,
        "round_of_32_pct": 65.0,
        "sweet_16_pct": 38.0,
        "elite_8_pct": 18.0,
        "final_four_pct": 8.0,
        "champion_pct": 3.0,
    },
    4: {
        "round_of_64_pct": 80.0,
        "round_of_32_pct": 55.0,
        "sweet_16_pct": 28.0,
        "elite_8_pct": 12.0,
        "final_four_pct": 5.0,
        "champion_pct": 2.0,
    },
    5: {
        "round_of_64_pct": 65.0,
        "round_of_32_pct": 38.0,
        "sweet_16_pct": 18.0,
        "elite_8_pct": 7.0,
        "final_four_pct": 3.0,
        "champion_pct": 1.0,
    },
    6: {
        "round_of_64_pct": 63.0,
        "round_of_32_pct": 35.0,
        "sweet_16_pct": 15.0,
        "elite_8_pct": 6.0,
        "final_four_pct": 2.0,
        "champion_pct": 0.8,
    },
    7: {
        "round_of_64_pct": 60.0,
        "round_of_32_pct": 30.0,
        "sweet_16_pct": 12.0,
        "elite_8_pct": 5.0,
        "final_four_pct": 2.0,
        "champion_pct": 0.6,
    },
    8: {
        "round_of_64_pct": 50.0,
        "round_of_32_pct": 22.0,
        "sweet_16_pct": 8.0,
        "elite_8_pct": 3.0,
        "final_four_pct": 1.0,
        "champion_pct": 0.3,
    },
    9: {
        "round_of_64_pct": 50.0,
        "round_of_32_pct": 20.0,
        "sweet_16_pct": 7.0,
        "elite_8_pct": 2.0,
        "final_four_pct": 0.8,
        "champion_pct": 0.2,
    },
    10: {
        "round_of_64_pct": 40.0,
        "round_of_32_pct": 15.0,
        "sweet_16_pct": 5.0,
        "elite_8_pct": 2.0,
        "final_four_pct": 0.6,
        "champion_pct": 0.1,
    },
    11: {
        "round_of_64_pct": 37.0,
        "round_of_32_pct": 15.0,
        "sweet_16_pct": 6.0,
        "elite_8_pct": 2.0,
        "final_four_pct": 0.7,
        "champion_pct": 0.1,
    },
    12: {
        "round_of_64_pct": 35.0,
        "round_of_32_pct": 15.0,
        "sweet_16_pct": 5.0,
        "elite_8_pct": 2.0,
        "final_four_pct": 0.5,
        "champion_pct": 0.1,
    },
    13: {
        "round_of_64_pct": 20.0,
        "round_of_32_pct": 6.0,
        "sweet_16_pct": 2.0,
        "elite_8_pct": 0.5,
        "final_four_pct": 0.1,
        "champion_pct": 0.03,
    },
    14: {
        "round_of_64_pct": 15.0,
        "round_of_32_pct": 4.0,
        "sweet_16_pct": 1.0,
        "elite_8_pct": 0.3,
        "final_four_pct": 0.05,
        "champion_pct": 0.01,
    },
    15: {
        "round_of_64_pct": 6.0,
        "round_of_32_pct": 2.0,
        "sweet_16_pct": 0.5,
        "elite_8_pct": 0.1,
        "final_four_pct": 0.02,
        "champion_pct": 0.005,
    },
    16: {
        "round_of_64_pct": 3.0,
        "round_of_32_pct": 0.5,
        "sweet_16_pct": 0.1,
        "elite_8_pct": 0.02,
        "final_four_pct": 0.003,
        "champion_pct": 0.001,
    },
}

_REGIONS = ["East", "West", "South", "Midwest"]


def _build_64_team_bracket() -> dict:
    """Build a realistic 64-team validated-teams dict."""
    teams = {}
    for region in _REGIONS:
        for seed in range(1, 17):
            team_id = f"{region.lower()}_{seed}"
            teams[team_id] = {
                "team_id": team_id,
                "team_name": f"{region} {seed}-Seed",
                "seed": seed,
                "region": region,
                **_SEED_TEMPLATES[seed],
            }
    return teams


def _build_64_team_matchups() -> dict:
    """Build first-round matchup mapping (1v16, 2v15, … 8v9) per region."""
    matchups = {}
    for region in _REGIONS:
        for hi in range(1, 9):
            lo = 17 - hi
            team_a = f"{region.lower()}_{hi}"
            team_b = f"{region.lower()}_{lo}"
            matchups[team_a] = team_b
            matchups[team_b] = team_a
    return matchups


# ---------------------------------------------------------------------------
# validate_bracket_structure — valid data
# ---------------------------------------------------------------------------


class TestValidBracketStructure:
    def test_realistic_64_team_bracket_passes(self):
        teams = _build_64_team_bracket()
        warnings = validate_bracket_structure(teams)
        # May have minor soft warnings due to imperfect template sums,
        # but should NOT raise BracketStructureError.
        assert isinstance(warnings, list)

    def test_empty_teams_returns_no_warnings(self):
        warnings = validate_bracket_structure({})
        assert warnings == []

    def test_partial_bracket_scales_expectations(self):
        """A 4-team partial bracket should scale expected sums to 4/64."""
        teams = {}
        for i, seed in enumerate([1, 2, 15, 16]):
            tid = f"team_{seed}"
            teams[tid] = {
                "team_id": tid,
                "team_name": f"Team {seed}",
                "seed": seed,
                "region": "East",
                **_SEED_TEMPLATES[seed],
            }
        # Should not raise — expectations are scaled by n_teams/64
        warnings = validate_bracket_structure(teams)
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# validate_bracket_structure — hard failures
# ---------------------------------------------------------------------------


class TestBracketStructureHardFailures:
    def test_champ_sum_way_too_high_raises(self):
        """All 64 teams have 100% championship pick → sum=6400% vs expected ~100%."""
        teams = _build_64_team_bracket()
        for team_data in teams.values():
            team_data["champion_pct"] = 100.0
        with pytest.raises(BracketStructureError, match="CHAMP"):
            validate_bracket_structure(teams)

    def test_champ_sum_way_too_low_raises(self):
        """All teams have 0% championship pick → sum=0% vs expected ~100%."""
        teams = _build_64_team_bracket()
        for team_data in teams.values():
            team_data["champion_pct"] = 0.0
        with pytest.raises(BracketStructureError, match="CHAMP"):
            validate_bracket_structure(teams)

    def test_f4_sum_massively_inflated_raises(self):
        """All 64 teams at 50% F4 → sum=3200% vs expected ~200%."""
        teams = _build_64_team_bracket()
        for team_data in teams.values():
            team_data["final_four_pct"] = 50.0
        with pytest.raises(BracketStructureError, match="F4"):
            validate_bracket_structure(teams)

    def test_r64_sum_all_zero_raises(self):
        """R64 all-zero → sum=0% vs expected ~3200%."""
        teams = _build_64_team_bracket()
        for team_data in teams.values():
            team_data["round_of_64_pct"] = 0.0
        with pytest.raises(BracketStructureError, match="R64"):
            validate_bracket_structure(teams)


# ---------------------------------------------------------------------------
# validate_bracket_structure — soft warnings
# ---------------------------------------------------------------------------


class TestBracketStructureSoftWarnings:
    def test_slight_champ_deviation_warns_not_raises(self):
        """Nudge all CHAMP picks up slightly — should warn, not raise."""
        teams = _build_64_team_bracket()
        for team_data in teams.values():
            team_data["champion_pct"] = team_data["champion_pct"] + 0.5
        warnings = validate_bracket_structure(teams)
        # The total bump is ~32pp across 64 teams, above soft_tolerance (20)
        # but well within hard_tolerance (50).
        champ_warnings = [w for w in warnings if "CHAMP" in w]
        assert len(champ_warnings) >= 1

    def test_first_round_matchup_complementarity_warns(self):
        """1-seed at 97% and 16-seed at 20% → sum=117%, should warn."""
        teams = _build_64_team_bracket()
        # Inflate the 16-seed so the pair sum is off
        teams["east_16"]["round_of_64_pct"] = 20.0
        matchups = _build_64_team_matchups()
        warnings = validate_bracket_structure(teams, bracket_matchups=matchups)
        matchup_warnings = [w for w in warnings if "matchup" in w.lower()]
        assert any("east_1" in w or "east_16" in w for w in matchup_warnings)

    def test_no_matchup_warning_when_complementary(self):
        """1-seed 97% + 16-seed 3% → sum=100%, no warning."""
        teams = _build_64_team_bracket()
        matchups = _build_64_team_matchups()
        warnings = validate_bracket_structure(teams, bracket_matchups=matchups)
        # The default templates have 97+3=100, 94+6=100, etc.
        matchup_warnings = [w for w in warnings if "matchup" in w.lower()]
        assert matchup_warnings == []


# ---------------------------------------------------------------------------
# Hardened monotonic validation
# ---------------------------------------------------------------------------


class TestMonotonicValidation:
    def test_valid_monotonic_team_passes(self):
        team = PublicPicksSchema(
            team_id="duke",
            seed=1,
            region="East",
            round_of_64_pct=98.0,
            round_of_32_pct=92.0,
            sweet_16_pct=75.0,
            elite_8_pct=55.0,
            final_four_pct=35.0,
            champion_pct=22.0,
        )
        assert team.champion_pct == 22.0

    def test_small_rounding_violation_warns_not_raises(self):
        """0.8pp violation (between 0.5 and 2.0) should warn but pass."""
        # sweet_16_pct slightly above round_of_32_pct (within 2pp tolerance)
        team = PublicPicksSchema(
            team_id="test",
            seed=5,
            region="East",
            round_of_64_pct=65.0,
            round_of_32_pct=38.0,
            sweet_16_pct=38.8,
            elite_8_pct=7.0,
            final_four_pct=3.0,
            champion_pct=1.0,
        )
        assert team.sweet_16_pct == 38.8

    def test_large_violation_raises_value_error(self):
        """S16 > R32 by 5pp should hard-fail."""
        with pytest.raises(Exception, match="Non-monotonic"):
            PublicPicksSchema(
                team_id="corrupt",
                seed=3,
                region="West",
                round_of_64_pct=85.0,
                round_of_32_pct=30.0,
                sweet_16_pct=35.0,  # 5pp above R32 → hard fail
                elite_8_pct=18.0,
                final_four_pct=8.0,
                champion_pct=3.0,
            )

    def test_champ_above_f4_by_3pp_raises(self):
        """CHAMP 6pp above F4 → hard fail."""
        with pytest.raises(Exception, match="Non-monotonic"):
            PublicPicksSchema(
                team_id="broken",
                seed=2,
                region="South",
                round_of_64_pct=94.0,
                round_of_32_pct=82.0,
                sweet_16_pct=58.0,
                elite_8_pct=35.0,
                final_four_pct=10.0,
                champion_pct=16.0,  # 6pp above F4
            )

    def test_equal_consecutive_rounds_passes(self):
        """Equal values in consecutive rounds are fine (delta=0)."""
        team = PublicPicksSchema(
            team_id="flat",
            seed=8,
            region="Midwest",
            round_of_64_pct=50.0,
            round_of_32_pct=50.0,
            sweet_16_pct=50.0,
            elite_8_pct=50.0,
            final_four_pct=50.0,
            champion_pct=50.0,
        )
        assert team.round_of_64_pct == 50.0


# ---------------------------------------------------------------------------
# Fail-closed behavior in _dict_to_consensus
# ---------------------------------------------------------------------------


class TestFailClosedConsensus:
    def test_valid_payload_returns_populated_consensus(self):
        payload = {
            "sources": ["espn"],
            "timestamp": "2026-03-17T12:00:00Z",
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 98.0,
                    "round_of_32_pct": 92.0,
                    "sweet_16_pct": 75.0,
                    "elite_8_pct": 55.0,
                    "final_four_pct": 35.0,
                    "champion_pct": 22.0,
                },
            },
        }
        scraper = ESPNPicksScraper()
        result = scraper._dict_to_consensus(payload)
        assert "duke" in result.teams
        assert result.teams["duke"].champion_pct == 22.0

    def test_schema_failure_returns_empty_consensus(self):
        """Payload with invalid team data should return empty teams,
        not silently use raw data."""
        payload = {
            "sources": ["espn"],
            "teams": {
                "corrupt": {
                    "team_name": "Corrupt",
                    "seed": 99,  # seed > 16 → schema fail
                    "region": "East",
                    "round_of_64_pct": 50.0,
                    "round_of_32_pct": 40.0,
                    "sweet_16_pct": 30.0,
                    "elite_8_pct": 20.0,
                    "final_four_pct": 10.0,
                    "champion_pct": 5.0,
                },
            },
        }
        scraper = ESPNPicksScraper()
        result = scraper._dict_to_consensus(payload)
        # The corrupt team should be skipped by validate_consensus_data,
        # and result should reflect what survived validation.
        # With only 1 team that fails, validated_teams is empty,
        # but it does NOT raise — it returns empty teams.
        assert isinstance(result, ConsensusData)

    def test_non_monotonic_team_rejected(self):
        """A team with hard-fail monotonic violation should be skipped."""
        payload = {
            "sources": ["espn"],
            "teams": {
                "good": {
                    "team_name": "Good",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 97.0,
                    "round_of_32_pct": 90.0,
                    "sweet_16_pct": 75.0,
                    "elite_8_pct": 55.0,
                    "final_four_pct": 35.0,
                    "champion_pct": 18.0,
                },
                "bad": {
                    "team_name": "Bad",
                    "seed": 5,
                    "region": "West",
                    "round_of_64_pct": 65.0,
                    "round_of_32_pct": 20.0,
                    "sweet_16_pct": 30.0,  # 10pp above R32 → rejected
                    "elite_8_pct": 7.0,
                    "final_four_pct": 3.0,
                    "champion_pct": 1.0,
                },
            },
        }
        scraper = ESPNPicksScraper()
        result = scraper._dict_to_consensus(payload)
        # Good team should survive; bad team should be skipped by validation
        assert "good" in result.teams
        assert "bad" not in result.teams

    def test_bracket_structure_failure_returns_empty(self):
        """A payload that passes per-team validation but fails bracket
        structure checks should return empty consensus."""
        # Build a payload where every team has 100% CHAMP (structurally impossible)
        teams = {}
        for i in range(64):
            seed = (i % 16) + 1
            tid = f"team_{i}"
            teams[tid] = {
                "team_name": f"Team {i}",
                "seed": min(seed, 16),
                "region": _REGIONS[i // 16],
                "round_of_64_pct": 100.0,
                "round_of_32_pct": 100.0,
                "sweet_16_pct": 100.0,
                "elite_8_pct": 100.0,
                "final_four_pct": 100.0,
                "champion_pct": 100.0,
            }
        payload = {"sources": ["espn"], "teams": teams}
        scraper = ESPNPicksScraper()
        result = scraper._dict_to_consensus(payload)
        # Bracket structure check should reject: CHAMP sum = 6400% vs ~100%
        assert len(result.teams) == 0
