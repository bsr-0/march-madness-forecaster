"""Tests for path protection scoring and quadrant correlation constraints."""

import pytest

from src.optimization.path_protection import (
    PathProtectionScorer,
    QuadrantCorrelationConstraint,
    PathConflict,
    ROUND_NAMES,
)
from src.optimization.bracket_search import BracketPick


@pytest.fixture
def predict_fn():
    """Simple seed-based prediction function."""
    def _extract_seed(name: str) -> int:
        # Parse "seed1_east" -> 1, "seed12_east" -> 12
        import re
        m = re.search(r"seed(\d+)", name)
        return int(m.group(1)) if m else 8

    def _predict(team1: str, team2: str) -> float:
        s1 = _extract_seed(team1)
        s2 = _extract_seed(team2)
        if s1 == s2:
            return 0.5
        diff = s2 - s1  # Positive = team1 is favored
        return 1.0 / (1.0 + 2.71828 ** (-0.175 * diff))
    return _predict


@pytest.fixture
def team_regions():
    return {
        "seed1_east": "East",
        "seed4_east": "East",
        "seed5_east": "East",
        "seed12_east": "East",
        "seed1_west": "West",
        "seed3_west": "West",
        "seed2_south": "South",
        "seed2_midwest": "Midwest",
    }


@pytest.fixture
def team_seeds():
    return {
        "seed1_east": 1,
        "seed4_east": 4,
        "seed5_east": 5,
        "seed12_east": 12,
        "seed1_west": 1,
        "seed3_west": 3,
        "seed2_south": 2,
        "seed2_midwest": 2,
    }


class TestPathProtectionScorer:
    def test_empty_bracket_returns_one(self, predict_fn, team_regions):
        scorer = PathProtectionScorer()
        assert scorer.compute_path_protection_score([], predict_fn, team_regions) == 1.0

    def test_chalk_bracket_high_score(self, predict_fn, team_regions):
        """A chalk bracket (all favorites win) should have score close to 1.0."""
        scorer = PathProtectionScorer()
        # Simple 3-game chalk bracket: R64 and R32
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="seed1_east",
                        loser_id="seed12_east", win_probability=0.95),
            BracketPick(round_num=0, game_idx=1, winner_id="seed4_east",
                        loser_id="seed5_east", win_probability=0.6),
            BracketPick(round_num=1, game_idx=0, winner_id="seed1_east",
                        loser_id="seed4_east", win_probability=0.75),
        ]
        score = scorer.compute_path_protection_score(
            picks, predict_fn, team_regions,
        )
        assert score >= 0.85, f"Chalk bracket score {score} below 0.85"

    def test_conflicting_bracket_low_score(self, predict_fn, team_regions):
        """A bracket with path conflicts should have lower score."""
        scorer = PathProtectionScorer()
        # Conflict: team loses in R64 but appears as winner in R32
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="seed12_east",
                        loser_id="seed1_east", win_probability=0.05),  # Upset!
            BracketPick(round_num=0, game_idx=1, winner_id="seed4_east",
                        loser_id="seed5_east", win_probability=0.6),
            # seed1_east was eliminated but is picked to win R32
            BracketPick(round_num=1, game_idx=0, winner_id="seed1_east",
                        loser_id="seed4_east", win_probability=0.75),
        ]
        score = scorer.compute_path_protection_score(
            picks, predict_fn, team_regions,
        )
        # Score should be penalized because seed1_east lost in R64
        assert score < 1.0, "Conflicting bracket should have score < 1.0"

    def test_identify_path_conflicts(self, team_regions):
        """Should detect when a team loses early but wins later."""
        scorer = PathProtectionScorer()
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="seed12_east",
                        loser_id="seed1_east", win_probability=0.05),
            BracketPick(round_num=1, game_idx=0, winner_id="seed1_east",
                        loser_id="seed4_east", win_probability=0.75),
        ]
        conflicts = scorer.identify_path_conflicts(picks, team_regions)
        assert len(conflicts) >= 1
        assert conflicts[0].team_id == "seed1_east"

    def test_no_conflicts_in_valid_bracket(self, team_regions):
        """A valid bracket should have no structural conflicts."""
        scorer = PathProtectionScorer()
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="seed1_east",
                        loser_id="seed12_east", win_probability=0.95),
            BracketPick(round_num=1, game_idx=0, winner_id="seed1_east",
                        loser_id="seed4_east", win_probability=0.75),
        ]
        conflicts = scorer.identify_path_conflicts(picks, team_regions)
        assert len(conflicts) == 0


class TestQuadrantCorrelationConstraint:
    def test_compliant_bracket(self, predict_fn, team_regions, team_seeds):
        """Bracket with safe F4 pick and no disruptions should be compliant."""
        constraint = QuadrantCorrelationConstraint(max_disruption_cost=0.03)
        picks = [
            BracketPick(round_num=0, game_idx=0, winner_id="seed1_east",
                        loser_id="seed12_east", win_probability=0.95),
        ]
        report = constraint.evaluate(
            picks=picks,
            final_four_picks=["seed1_east", "seed1_west", "seed2_south", "seed2_midwest"],
            champion_id="seed1_east",
            team_regions=team_regions,
            team_seeds=team_seeds,
            predict_fn=predict_fn,
        )
        assert report.penalty > 0.0
        assert report.penalty <= 1.0

    def test_no_safe_f4_violation(self, predict_fn, team_regions, team_seeds):
        """Should flag when no F4 pick has seed <= 3."""
        # Override seeds so all F4 picks are high seeds
        high_seeds = {k: 8 for k in team_seeds}
        constraint = QuadrantCorrelationConstraint(require_safe_f4=True)
        picks = []
        report = constraint.evaluate(
            picks=picks,
            final_four_picks=["seed1_east", "seed1_west"],
            champion_id="seed1_east",
            team_regions=team_regions,
            team_seeds=high_seeds,
            predict_fn=predict_fn,
        )
        assert not report.is_compliant
        assert any("seed <= 3" in v for v in report.violations)

    def test_penalty_range(self, predict_fn, team_regions, team_seeds):
        """Penalty should always be in [0, 1]."""
        constraint = QuadrantCorrelationConstraint()
        report = constraint.evaluate(
            picks=[],
            final_four_picks=["seed1_east"],
            champion_id="seed1_east",
            team_regions=team_regions,
            team_seeds=team_seeds,
            predict_fn=predict_fn,
        )
        assert 0.0 <= report.penalty <= 1.0
