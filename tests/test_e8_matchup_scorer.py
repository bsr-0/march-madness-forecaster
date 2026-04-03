"""Tests for Elite 8 matchup interaction scoring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.optimization.e8_matchup_scorer import (
    E8MatchupInteractionScorer,
    apply_e8_adjustments,
    predict_e8_matchups,
)


def _make_team(**overrides):
    """Build a minimal team features object with D1-average defaults."""
    defaults = {
        "opp_turnover_rate": 0.18,
        "turnover_rate": 0.18,
        "offensive_reb_rate": 0.28,
        "defensive_reb_rate": 0.72,
        "adj_tempo": 68.0,
        "three_pt_pct": 0.34,
        "opp_effective_fg_pct": 0.48,
        "coach_e8_appearances": 0,
        "coach_deep_run_rate": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSignalBounds:
    """Each signal and the composite must stay in [0, 1]."""

    def test_neutral_inputs_produce_neutral_score(self):
        scorer = E8MatchupInteractionScorer()
        avg_team = _make_team()
        result = scorer.score(avg_team, avg_team)
        assert abs(result.composite - 0.5) < 0.01
        assert abs(result.adjustment) < 0.01

    def test_all_signals_in_unit_range(self):
        scorer = E8MatchupInteractionScorer()
        # Extreme teams: one very strong, one very weak
        strong = _make_team(
            opp_turnover_rate=0.24,
            turnover_rate=0.12,
            offensive_reb_rate=0.38,
            defensive_reb_rate=0.80,
            adj_tempo=78.0,
            three_pt_pct=0.42,
            opp_effective_fg_pct=0.42,
            coach_e8_appearances=8,
            coach_deep_run_rate=0.4,
        )
        weak = _make_team(
            opp_turnover_rate=0.12,
            turnover_rate=0.24,
            offensive_reb_rate=0.22,
            defensive_reb_rate=0.64,
            adj_tempo=60.0,
            three_pt_pct=0.28,
            opp_effective_fg_pct=0.54,
            coach_e8_appearances=0,
            coach_deep_run_rate=0.0,
        )
        result = scorer.score(strong, weak)
        for name, val in result.signals.items():
            assert 0.0 <= val <= 1.0, f"{name} = {val}"
        assert 0.0 <= result.composite <= 1.0


class TestSymmetry:
    """Swapping teams should negate the adjustment."""

    def test_swap_negates_adjustment(self):
        scorer = E8MatchupInteractionScorer()
        a = _make_team(opp_turnover_rate=0.22, turnover_rate=0.14, adj_tempo=74.0)
        b = _make_team(opp_turnover_rate=0.14, turnover_rate=0.22, adj_tempo=62.0)

        result_ab = scorer.score(a, b)
        result_ba = scorer.score(b, a)

        assert abs(result_ab.adjustment + result_ba.adjustment) < 0.001


class TestAdjustmentBounds:
    """Adjustment must never exceed ±0.08."""

    def test_max_adjustment_capped(self):
        scorer = E8MatchupInteractionScorer()
        # Maximally divergent teams
        strong = _make_team(
            opp_turnover_rate=0.30,
            turnover_rate=0.10,
            offensive_reb_rate=0.45,
            defensive_reb_rate=0.85,
            adj_tempo=85.0,
            three_pt_pct=0.45,
            opp_effective_fg_pct=0.40,
            coach_e8_appearances=15,
            coach_deep_run_rate=0.6,
        )
        weak = _make_team(
            opp_turnover_rate=0.10,
            turnover_rate=0.30,
            offensive_reb_rate=0.18,
            defensive_reb_rate=0.55,
            adj_tempo=55.0,
            three_pt_pct=0.25,
            opp_effective_fg_pct=0.56,
            coach_e8_appearances=0,
            coach_deep_run_rate=0.0,
        )
        result = scorer.score(strong, weak)
        assert -0.08 <= result.adjustment <= 0.08


class TestDeadZone:
    """Near-neutral composites should have dampened adjustments."""

    def test_small_difference_dampened(self):
        scorer = E8MatchupInteractionScorer()
        # Slightly different teams
        a = _make_team(opp_turnover_rate=0.185)
        b = _make_team(opp_turnover_rate=0.175)
        result = scorer.score(a, b)
        # Very close to neutral — dead zone should dampen
        assert abs(result.adjustment) < 0.02


class TestApplyE8Adjustments:
    """Integration tests for apply_e8_adjustments."""

    def test_does_not_mutate_input(self):
        round_probs = {
            "team_1": {"R64": 0.95, "R32": 0.80, "S16": 0.60, "E8": 0.40, "F4": 0.25, "CHAMP": 0.12},
            "team_2": {"R64": 0.85, "R32": 0.65, "S16": 0.45, "E8": 0.30, "F4": 0.18, "CHAMP": 0.08},
        }
        original_e8_1 = round_probs["team_1"]["E8"]

        features = {
            "team_1": _make_team(opp_turnover_rate=0.22, coach_e8_appearances=5),
            "team_2": _make_team(opp_turnover_rate=0.14),
        }

        result = apply_e8_adjustments(round_probs, [("team_1", "team_2")], features)

        # Original unchanged
        assert round_probs["team_1"]["E8"] == original_e8_1
        # Result is a different object
        assert result is not round_probs

    def test_preserves_non_e8_rounds(self):
        round_probs = {
            "team_1": {"R64": 0.95, "R32": 0.80, "S16": 0.60, "E8": 0.40, "F4": 0.25, "CHAMP": 0.12},
            "team_2": {"R64": 0.85, "R32": 0.65, "S16": 0.45, "E8": 0.30, "F4": 0.18, "CHAMP": 0.08},
        }
        features = {
            "team_1": _make_team(opp_turnover_rate=0.22),
            "team_2": _make_team(opp_turnover_rate=0.14),
        }

        result = apply_e8_adjustments(round_probs, [("team_1", "team_2")], features)

        # R64, R32, S16 should be unchanged
        for rnd in ("R64", "R32", "S16"):
            assert result["team_1"][rnd] == round_probs["team_1"][rnd]
            assert result["team_2"][rnd] == round_probs["team_2"][rnd]

    def test_f4_champ_scale_proportionally(self):
        round_probs = {
            "team_1": {"R64": 0.95, "R32": 0.80, "S16": 0.60, "E8": 0.40, "F4": 0.20, "CHAMP": 0.10},
            "team_2": {"R64": 0.85, "R32": 0.65, "S16": 0.45, "E8": 0.30, "F4": 0.15, "CHAMP": 0.05},
        }
        features = {
            "team_1": _make_team(opp_turnover_rate=0.24, coach_e8_appearances=6),
            "team_2": _make_team(opp_turnover_rate=0.12),
        }

        result = apply_e8_adjustments(round_probs, [("team_1", "team_2")], features)

        # If E8 went up for team_1, F4 and CHAMP should also go up
        if result["team_1"]["E8"] > round_probs["team_1"]["E8"]:
            assert result["team_1"]["F4"] >= round_probs["team_1"]["F4"]
            assert result["team_1"]["CHAMP"] >= round_probs["team_1"]["CHAMP"]

    def test_missing_features_skipped(self):
        round_probs = {
            "team_1": {"E8": 0.40, "F4": 0.20, "CHAMP": 0.10},
            "team_2": {"E8": 0.30, "F4": 0.15, "CHAMP": 0.05},
        }
        # team_2 has no features — should be skipped gracefully
        features = {"team_1": _make_team()}
        result = apply_e8_adjustments(round_probs, [("team_1", "team_2")], features)
        assert result["team_1"]["E8"] == round_probs["team_1"]["E8"]


class TestPredictE8Matchups:
    """Test bracket structure prediction."""

    def test_eight_team_bracket(self):
        seeds = {f"team_{s}": s for s in range(1, 9)}
        matchups = predict_e8_matchups(seeds)
        # With 8 teams (1 region), should produce 1 matchup
        assert len(matchups) >= 1
        for a, b in matchups:
            assert a in seeds
            assert b in seeds

    def test_returns_tuples(self):
        seeds = {f"team_{s}": s for s in range(1, 9)}
        matchups = predict_e8_matchups(seeds)
        for m in matchups:
            assert isinstance(m, tuple)
            assert len(m) == 2
