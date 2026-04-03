"""Synthetic optimizer audit: verifies leverage detection on hand-crafted distributions.

Council recommendation #1: before any Monte Carlo work, prove the optimizer
correctly identifies disagreement between model probabilities and public picks.

Tests at two levels:
  - Unit: LeverageCalculator directly (Tests 2-5)
  - Integration: Full PoolOptimizer.optimize() pipeline (Test 1)
"""

import copy

import pytest

from src.data.seed_pick_model import SEED_PICK_RATES
from src.optimization.leverage import LeverageCalculator
from src.optimization.pool_optimizer import PoolEnvironment, PoolOptimizer
from src.prediction.seed_probabilities import (
    build_seed_probabilities,
    build_seed_round_probabilities,
)

ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


@pytest.fixture
def eight_team_seeds():
    return {f"team_{s}": s for s in range(1, 9)}


@pytest.fixture
def seed_based_public_picks(eight_team_seeds):
    """Chalk baseline from SEED_PICK_RATES."""
    return {tid: dict(SEED_PICK_RATES.get(s, SEED_PICK_RATES[8])) for tid, s in eight_team_seeds.items()}


@pytest.fixture
def pairwise_probs(eight_team_seeds):
    return build_seed_probabilities(eight_team_seeds)


@pytest.fixture
def seed_round_probs(eight_team_seeds):
    return build_seed_round_probabilities(eight_team_seeds)


class TestInflatedUnderdogDetected:
    """Integration: PoolOptimizer detects leverage when model inflates an underdog."""

    def test_inflated_underdog_in_leverage_picks(
        self, eight_team_seeds, pairwise_probs, seed_round_probs, seed_based_public_picks
    ):
        # Inflate team_5 (5-seed) well above public chalk expectations
        model_probs = copy.deepcopy(seed_round_probs)
        model_probs["team_5"]["E8"] = 0.40  # public ~0.15
        model_probs["team_5"]["F4"] = 0.20  # public ~0.06
        model_probs["team_5"]["CHAMP"] = 0.15  # public ~0.04

        env = PoolEnvironment(
            pool_size=100,
            scoring_rules=ESPN_SCORING,
            payout_structure="winner_take_all",
            public_pick_distribution=seed_based_public_picks,
        )
        optimizer = PoolOptimizer(pairwise_probs, env, model_round_probs=model_probs)
        result = optimizer.optimize()

        assert len(result.leverage_picks) > 0, "Expected non-empty leverage picks"

        team5_picks = [p for p in result.leverage_picks if p["team_id"] == "team_5"]
        assert len(team5_picks) > 0, "team_5 should appear in leverage picks"

        for pick in team5_picks:
            assert pick["ev_differential"] > 0
            assert pick["leverage_ratio"] >= 1.5


class TestLeverageOrdering:
    """Unit: leverage picks sorted by EV differential, not leverage ratio."""

    def test_sorted_by_ev_differential_not_ratio(self):
        # team_a: leverage=4.0, ev_diff = (0.20 - 0.05) * 320 = 48.0
        # team_b: leverage=5.0, ev_diff = (0.10 - 0.02) * 320 = 25.6
        model_probs = {
            "team_a": {"CHAMP": 0.20},
            "team_b": {"CHAMP": 0.10},
        }
        public_picks = {
            "team_a": {"CHAMP": 0.05},
            "team_b": {"CHAMP": 0.02},
        }
        calc = LeverageCalculator(model_probs, public_picks, scoring_system=ESPN_SCORING)
        picks = calc.find_leverage_picks(min_leverage=1.5, min_probability=0.01)

        assert len(picks) == 2
        assert picks[0].team_id == "team_a", "Higher EV diff should sort first"
        assert picks[1].team_id == "team_b"
        # team_b has higher leverage ratio but lower EV diff
        assert picks[1].leverage_ratio > picks[0].leverage_ratio
        assert picks[0].expected_value_differential > picks[1].expected_value_differential


class TestFadeDetection:
    """Unit: over-picked favorites appear in fade_picks."""

    def test_deflated_favorite_faded(self):
        # Model says 5% champ, public says 18% → leverage = 0.28
        model_probs = {
            "team_fav": {"R64": 0.60, "R32": 0.30, "CHAMP": 0.05},
        }
        public_picks = {
            "team_fav": {"R64": 0.95, "R32": 0.70, "CHAMP": 0.18},
        }
        calc = LeverageCalculator(model_probs, public_picks, scoring_system=ESPN_SCORING)
        fades = calc.find_fade_picks(max_leverage=0.7)

        assert len(fades) > 0, "Expected non-empty fade picks"

        fav_fades = [f for f in fades if f.team_id == "team_fav"]
        assert len(fav_fades) > 0, "team_fav should be faded"
        for fade in fav_fades:
            assert fade.leverage_ratio <= 0.7

        # CHAMP fade should be most negative EV diff → sorted first
        assert fades[0].round_name == "CHAMP"


class TestSeedModeZeroLeverage:
    """Unit: identical seed probs for model and public produce no leverage picks."""

    def test_identical_probs_no_leverage(self):
        # Use SEED_PICK_RATES for both — exactly identical
        identical_probs = {f"team_{s}": dict(SEED_PICK_RATES[s]) for s in range(1, 9)}
        calc = LeverageCalculator(
            model_probs=identical_probs,
            public_picks=copy.deepcopy(identical_probs),
            scoring_system=ESPN_SCORING,
        )
        picks = calc.find_leverage_picks(min_leverage=1.5, min_probability=0.05)

        assert len(picks) == 0, (
            "Identical model and public probs should produce zero leverage picks. "
            "This confirms seed-mode zero-leverage is by design."
        )


class TestMagnitudeMatters:
    """Unit: late-round disagreements dominate EV differential over early rounds."""

    def test_champ_outranks_r64(self):
        # team_early: R64 leverage=1.8, ev_diff = (0.90-0.50)*10 = 4.0
        # team_late: CHAMP leverage=2.5, ev_diff = (0.10-0.04)*320 = 19.2
        model_probs = {
            "team_early": {"R64": 0.90},
            "team_late": {"CHAMP": 0.10},
        }
        public_picks = {
            "team_early": {"R64": 0.50},
            "team_late": {"CHAMP": 0.04},
        }
        calc = LeverageCalculator(model_probs, public_picks, scoring_system=ESPN_SCORING)
        picks = calc.find_leverage_picks(min_leverage=1.5, min_probability=0.05)

        assert len(picks) == 2
        assert picks[0].team_id == "team_late", "CHAMP pick should rank first by EV diff"
        assert picks[0].expected_value_differential > picks[1].expected_value_differential
