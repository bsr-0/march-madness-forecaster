"""Tests for the upset probability adjuster module."""

import numpy as np
import pytest

from src.prediction.upset_probability_adjuster import (
    adjust_r32_for_upset_teams,
    adjust_round_probs_for_upsets,
    get_upset_game_mapping,
)


def _make_bracket_setup():
    """Create a minimal 64-team bracket with seeds and round_probs."""
    seeds = {}
    first_round = []
    round_probs = {}
    for region_idx in range(4):
        for high, low in [
            (1, 16),
            (8, 9),
            (5, 12),
            (4, 13),
            (6, 11),
            (3, 14),
            (7, 10),
            (2, 15),
        ]:
            t_high = f"team_{region_idx}_{high}"
            t_low = f"team_{region_idx}_{low}"
            seeds[t_high] = high
            seeds[t_low] = low
            first_round.extend([t_high, t_low])
            round_probs[t_high] = {"R64": 0.70, "R32": 0.40, "S16": 0.20}
            round_probs[t_low] = {"R64": 0.30, "R32": 0.10, "S16": 0.05}
    return seeds, first_round, round_probs


class TestAdjustRoundProbsForUpsets:
    """Test R64 probability adjustment."""

    def test_calibrated_uses_model_confidence(self):
        """Calibrated mode sets R64 to max(base, p_upset)."""
        seeds, first_round, rp = _make_bracket_setup()
        # Game 2 is 5v12 in region 0 (team_0_5 vs team_0_12)
        upset_probs = {2: ("team_0_5", "team_0_12", 0.65)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="calibrated",
        )
        assert result["team_0_12"]["R64"] == pytest.approx(0.65)

    def test_calibrated_does_not_downgrade(self):
        """If base prob > p_upset, keeps base prob."""
        seeds, first_round, rp = _make_bracket_setup()
        rp["team_0_12"]["R64"] = 0.75  # already high
        upset_probs = {2: ("team_0_5", "team_0_12", 0.60)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="calibrated",
        )
        assert result["team_0_12"]["R64"] == pytest.approx(0.75)

    def test_aggressive_adds_boost(self):
        """Aggressive mode adds +0.10 to p_upset."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.60)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="aggressive",
        )
        assert result["team_0_12"]["R64"] == pytest.approx(0.70)

    def test_aggressive_capped_at_095(self):
        """Aggressive mode caps at 0.95."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.90)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="aggressive",
        )
        assert result["team_0_12"]["R64"] == pytest.approx(0.95)

    def test_low_p_upset_unchanged(self):
        """Games with p_upset <= 0.50 are not adjusted."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.45)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="calibrated",
        )
        assert result["team_0_12"]["R64"] == pytest.approx(0.30)

    def test_non_eligible_games_untouched(self):
        """Teams not in upset_probs are not modified."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.65)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="calibrated",
        )
        # 1v16 game (game 0) should be untouched
        assert result["team_0_16"]["R64"] == pytest.approx(0.30)
        assert result["team_0_1"]["R64"] == pytest.approx(0.70)

    def test_round_probs_structure_preserved(self):
        """All teams and rounds present in output."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.65)}
        result = adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="calibrated",
        )
        assert set(result.keys()) == set(rp.keys())
        for tid in rp:
            assert set(result[tid].keys()) == set(rp[tid].keys())

    def test_does_not_mutate_input(self):
        """Input round_probs is not modified."""
        seeds, first_round, rp = _make_bracket_setup()
        original_val = rp["team_0_12"]["R64"]
        upset_probs = {2: ("team_0_5", "team_0_12", 0.65)}
        adjust_round_probs_for_upsets(
            rp,
            upset_probs,
            first_round,
            seeds,
            boost_mode="calibrated",
        )
        assert rp["team_0_12"]["R64"] == pytest.approx(original_val)


class TestAdjustR32ForUpsetTeams:
    """Test R32 probability boost for upset-predicted teams."""

    def test_12v4_gets_r32_boost(self):
        """12-seed beating 5-seed gets R32 boost for facing 4-seed."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.65)}
        result = adjust_r32_for_upset_teams(
            rp,
            upset_probs,
            first_round,
            seeds,
        )
        # R32 prob should be boosted by 1.3x: 0.10 * 1.3 = 0.13
        assert result["team_0_12"]["R32"] == pytest.approx(0.13)

    def test_11v3_gets_r32_boost(self):
        """11-seed beating 6-seed gets R32 boost for facing 3-seed."""
        seeds, first_round, rp = _make_bracket_setup()
        # Game 4 is 6v11 in region 0
        upset_probs = {4: ("team_0_6", "team_0_11", 0.60)}
        result = adjust_r32_for_upset_teams(
            rp,
            upset_probs,
            first_round,
            seeds,
        )
        assert result["team_0_11"]["R32"] > rp["team_0_11"]["R32"]

    def test_10v2_no_r32_boost(self):
        """10-seed beating 7-seed does NOT get R32 boost (faces 2-seed)."""
        seeds, first_round, rp = _make_bracket_setup()
        # Game 6 is 7v10 in region 0
        upset_probs = {6: ("team_0_7", "team_0_10", 0.65)}
        result = adjust_r32_for_upset_teams(
            rp,
            upset_probs,
            first_round,
            seeds,
        )
        assert result["team_0_10"]["R32"] == pytest.approx(rp["team_0_10"]["R32"])

    def test_9v1_no_r32_boost(self):
        """9-seed beating 8-seed does NOT get R32 boost (faces 1-seed)."""
        seeds, first_round, rp = _make_bracket_setup()
        # Game 1 is 8v9 in region 0
        upset_probs = {1: ("team_0_8", "team_0_9", 0.65)}
        result = adjust_r32_for_upset_teams(
            rp,
            upset_probs,
            first_round,
            seeds,
        )
        assert result["team_0_9"]["R32"] == pytest.approx(rp["team_0_9"]["R32"])

    def test_r32_boost_capped_at_050(self):
        """R32 boost never exceeds 0.50."""
        seeds, first_round, rp = _make_bracket_setup()
        rp["team_0_12"]["R32"] = 0.45
        upset_probs = {2: ("team_0_5", "team_0_12", 0.70)}
        result = adjust_r32_for_upset_teams(
            rp,
            upset_probs,
            first_round,
            seeds,
        )
        # 0.45 * 1.3 = 0.585, should be capped at 0.50
        assert result["team_0_12"]["R32"] == pytest.approx(0.50)

    def test_low_p_upset_no_boost(self):
        """Games with p_upset <= 0.50 get no R32 boost."""
        seeds, first_round, rp = _make_bracket_setup()
        upset_probs = {2: ("team_0_5", "team_0_12", 0.45)}
        result = adjust_r32_for_upset_teams(
            rp,
            upset_probs,
            first_round,
            seeds,
        )
        assert result["team_0_12"]["R32"] == pytest.approx(rp["team_0_12"]["R32"])


class TestGetUpsetGameMapping:
    """Test the R64-to-R32 game mapping helper."""

    def test_only_5v12_and_6v11_included(self):
        """Only 5v12 and 6v11 games appear in mapping."""
        seeds, first_round, _ = _make_bracket_setup()
        mapping = get_upset_game_mapping(first_round, seeds)
        for game_idx in mapping:
            t1 = first_round[game_idx * 2]
            t2 = first_round[game_idx * 2 + 1]
            s1 = seeds[t1]
            s2 = seeds[t2]
            pair = tuple(sorted([s1, s2]))
            assert pair in {(5, 12), (6, 11)}

    def test_8v9_7v10_excluded(self):
        """8v9 and 7v10 games are NOT in the mapping."""
        seeds, first_round, _ = _make_bracket_setup()
        mapping = get_upset_game_mapping(first_round, seeds)
        for game_idx in mapping:
            t1 = first_round[game_idx * 2]
            t2 = first_round[game_idx * 2 + 1]
            s1 = seeds[t1]
            s2 = seeds[t2]
            pair = tuple(sorted([s1, s2]))
            assert pair not in {(7, 10), (8, 9)}

    def test_mapping_has_valid_r32_indices(self):
        """R32 game indices are in range [32, 48)."""
        seeds, first_round, _ = _make_bracket_setup()
        mapping = get_upset_game_mapping(first_round, seeds)
        for _, (r32_idx, _) in mapping.items():
            assert 32 <= r32_idx < 48
