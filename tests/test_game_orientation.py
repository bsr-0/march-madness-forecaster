"""Tests for canonical game orientation.

Guards the fix for the winner-first ordering trap: tournament_context files
store the winner first for 100% of 2005-2015 games, which silently biases
any consumer scoring P(team1 wins) against outcome. See
src/data/game_orientation.py.
"""

import json
from pathlib import Path

import pytest

from src.data.game_orientation import (
    favorite_won,
    orient_prediction_record,
    orient_result_game,
    should_flip,
)

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "artifacts" / "loyo_pergame_predictions.json"


class TestShouldFlip:
    def test_better_seed_first_is_left_alone(self):
        assert should_flip(1, 16, "duke", "siena") is False

    def test_worse_seed_first_is_flipped(self):
        assert should_flip(16, 1, "siena", "duke") is True

    def test_equal_seeds_break_alphabetically_not_by_outcome(self):
        assert should_flip(1, 1, "arizona", "michigan") is False
        assert should_flip(1, 1, "michigan", "arizona") is True

    def test_decision_never_consults_the_outcome(self):
        # Same pair, opposite results -> identical orientation decision.
        assert should_flip(3, 7, "a_team", "b_team") == should_flip(3, 7, "a_team", "b_team")


class TestOrientResultGame:
    def _game(self, **kw):
        base = {
            "team1_id": "siena",
            "team1_seed": 16,
            "team1_score": 60,
            "team2_id": "duke",
            "team2_seed": 1,
            "team2_score": 80,
            "team1_won": False,
            "round_name": "R64",
        }
        base.update(kw)
        return base

    def test_flip_swaps_fields_and_inverts_winner(self):
        out = orient_result_game(self._game())
        assert out["team1_id"] == "duke"
        assert out["team1_seed"] == 1
        assert out["team1_score"] == 80
        assert out["team2_id"] == "siena"
        assert out["team1_won"] is True

    def test_winner_identity_is_preserved_by_the_flip(self):
        g = self._game()
        before = g["team1_id"] if g["team1_won"] else g["team2_id"]
        out = orient_result_game(g)
        after = out["team1_id"] if out["team1_won"] else out["team2_id"]
        assert before == after == "duke"

    def test_input_is_not_mutated(self):
        g = self._game()
        orient_result_game(g)
        assert g["team1_id"] == "siena" and g["team1_won"] is False

    def test_already_oriented_game_is_unchanged(self):
        g = self._game(
            team1_id="duke",
            team1_seed=1,
            team1_score=80,
            team2_id="siena",
            team2_seed=16,
            team2_score=60,
            team1_won=True,
        )
        assert orient_result_game(g) == g


class TestOrientPredictionRecord:
    def _rec(self, **kw):
        base = {
            "team1": "siena",
            "team2": "duke",
            "seed1": 16,
            "seed2": 1,
            "round": "R64",
            "outcome": 0,
            "torvik": 0.03,
            "seed": 0.01,
            "closing_market": 0.02,
            "market_movement": 0.005,
        }
        base.update(kw)
        return base

    def test_probabilities_invert_with_the_label(self):
        out = orient_prediction_record(self._rec())
        assert out["team1"] == "duke"
        assert out["outcome"] == 1
        assert out["torvik"] == pytest.approx(0.97)
        assert out["seed"] == pytest.approx(0.99)
        assert out["closing_market"] == pytest.approx(0.98)

    def test_market_movement_negates_rather_than_complements(self):
        # A closing-minus-opening delta is a difference, not a probability.
        out = orient_prediction_record(self._rec())
        assert out["market_movement"] == pytest.approx(-0.005)

    def test_none_probabilities_stay_none(self):
        out = orient_prediction_record(self._rec(closing_market=None))
        assert out["closing_market"] is None

    def test_brier_is_invariant_under_orientation(self):
        # The property that makes existing Brier figures trustworthy.
        r = self._rec()
        o = orient_prediction_record(r)
        assert (r["torvik"] - r["outcome"]) ** 2 == pytest.approx((o["torvik"] - o["outcome"]) ** 2)

    def test_orientation_is_idempotent(self):
        once = orient_prediction_record(self._rec())
        assert orient_prediction_record(once) == once


class TestFavoriteWon:
    def test_reports_favorite_result_for_seeded_matchups(self):
        assert favorite_won({"seed1": 1, "seed2": 16, "outcome": 1}) is True
        assert favorite_won({"seed1": 1, "seed2": 16, "outcome": 0}) is False

    def test_same_seed_matchups_have_no_favorite(self):
        assert favorite_won({"seed1": 1, "seed2": 1, "outcome": 1}) is None


@pytest.mark.skipif(not ARTIFACT.exists(), reason="prediction artifact not present")
class TestArtifactIsOriented:
    """Regression guard: the shipped artifact must stay oriented.

    If a regeneration ever drops the orientation step, the base rate jumps
    back to ~90% and every calibration/accuracy figure downstream silently
    breaks. This is the cheapest place to catch that.
    """

    @staticmethod
    def _rows():
        with open(ARTIFACT) as f:
            data = json.load(f)
        return [g for games in data.values() for g in games]

    def test_team1_is_always_the_better_or_equal_seed(self):
        offenders = [g for g in self._rows() if g["seed1"] > g["seed2"]]
        assert not offenders, f"{len(offenders)} records stored worse-seed-first"

    def test_base_rate_is_plausible_not_winner_first(self):
        rows = self._rows()
        rate = sum(g["outcome"] for g in rows) / len(rows)
        # True favourite win rate is ~72%; the winner-first defect showed ~90%.
        assert 0.65 < rate < 0.80, f"base rate {rate:.3f} suggests winner-first ordering"

    def test_no_single_year_is_degenerate(self):
        with open(ARTIFACT) as f:
            data = json.load(f)
        for year, games in data.items():
            rate = sum(g["outcome"] for g in games) / len(games)
            assert rate < 1.0, f"{year} has outcome=1 for every game"
