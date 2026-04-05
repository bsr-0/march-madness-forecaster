"""Tests for Elo temporal model."""

import numpy as np
import pytest

from src.ml.time_series.elo_temporal import EloTemporalModel


@pytest.fixture
def sample_games():
    """Simple chronological game list."""
    return [
        {"team_a": "Duke", "team_b": "UNC", "outcome": 1.0, "margin": 10.0, "season": 2023},
        {"team_a": "UNC", "team_b": "Kentucky", "outcome": 1.0, "margin": 5.0, "season": 2023},
        {"team_a": "Kentucky", "team_b": "Duke", "outcome": 0.0, "margin": -8.0, "season": 2023},
        {"team_a": "Gonzaga", "team_b": "Duke", "outcome": 0.0, "margin": -3.0, "season": 2023},
        {"team_a": "Gonzaga", "team_b": "UNC", "outcome": 1.0, "margin": 7.0, "season": 2023},
    ]


@pytest.fixture
def multi_season_games():
    """Games spanning two seasons for testing mean reversion."""
    return [
        {"team_a": "A", "team_b": "B", "outcome": 1.0, "margin": 20.0, "season": 2022},
        {"team_a": "A", "team_b": "B", "outcome": 1.0, "margin": 15.0, "season": 2022},
        {"team_a": "A", "team_b": "B", "outcome": 1.0, "margin": 10.0, "season": 2022},
        # New season — mean reversion should pull ratings toward center
        {"team_a": "A", "team_b": "B", "outcome": 1.0, "margin": 5.0, "season": 2023},
    ]


class TestEloTemporalModel:
    def test_train_returns_stats(self, sample_games):
        model = EloTemporalModel()
        stats = model.train(sample_games)
        assert stats["model"] == "elo_temporal"
        assert stats["games_processed"] == 5
        assert stats["teams_rated"] == 4

    def test_ratings_update(self, sample_games):
        model = EloTemporalModel()
        model.train(sample_games)
        ratings = model.get_ratings()
        # Duke won 2 games (beat UNC, beat Kentucky) — should be higher rated
        assert ratings["Duke"] > 1500.0
        # All 4 teams should have ratings
        assert len(ratings) == 4

    def test_predict_proba_range(self, sample_games):
        model = EloTemporalModel()
        model.train(sample_games)
        prob = model.predict_proba("Duke", "UNC")
        assert 0.0 < prob < 1.0

    def test_prediction_symmetry(self, sample_games):
        model = EloTemporalModel()
        model.train(sample_games)
        p_ab = model.predict_proba("Duke", "UNC")
        p_ba = model.predict_proba("UNC", "Duke")
        np.testing.assert_almost_equal(p_ab + p_ba, 1.0)

    def test_equal_teams_predict_half(self):
        model = EloTemporalModel()
        # No games — both at default rating
        prob = model.predict_proba("X", "Y")
        np.testing.assert_almost_equal(prob, 0.5)

    def test_predict_matchup_array(self, sample_games):
        model = EloTemporalModel()
        model.train(sample_games)
        probs = model.predict_matchup_array(
            ["Duke", "UNC"], ["UNC", "Kentucky"]
        )
        assert probs.shape == (2,)
        assert np.all(probs > 0) and np.all(probs < 1)

    def test_mean_reversion(self, multi_season_games):
        # Train without mean reversion
        no_mr = EloTemporalModel(mean_reversion=0.0)
        no_mr.train(multi_season_games)
        gap_no_mr = no_mr.get_rating("A") - no_mr.get_rating("B")

        # Train with mean reversion
        with_mr = EloTemporalModel(mean_reversion=0.5)
        with_mr.train(multi_season_games)
        gap_with_mr = with_mr.get_rating("A") - with_mr.get_rating("B")

        # Mean reversion should shrink the gap
        assert gap_with_mr < gap_no_mr

    def test_unknown_team_gets_default(self, sample_games):
        model = EloTemporalModel()
        model.train(sample_games)
        rating = model.get_rating("NeverPlayed")
        assert rating == 1500.0

    def test_feature_importance_empty(self, sample_games):
        model = EloTemporalModel()
        model.train(sample_games)
        assert model.get_feature_importance() == {}

    def test_serialization_roundtrip(self, sample_games):
        model = EloTemporalModel(k_factor=25, mean_reversion=0.4)
        model.train(sample_games)
        data = model.to_dict()

        restored = EloTemporalModel.from_dict(data)
        assert restored.config.k_factor == 25
        assert restored.config.mean_reversion == 0.4
        assert restored.get_ratings() == model.get_ratings()

    def test_no_temporal_leakage(self):
        """Verify that ratings only incorporate past games."""
        games = [
            {"team_a": "A", "team_b": "B", "outcome": 1.0, "season": 2023},
            {"team_a": "A", "team_b": "B", "outcome": 1.0, "season": 2023},
            {"team_a": "A", "team_b": "B", "outcome": 1.0, "season": 2023},
        ]
        model = EloTemporalModel()

        # Process only first game
        model.train(games[:1])
        rating_after_1 = model.get_rating("A")

        # Process all three
        model.train(games)
        rating_after_3 = model.get_rating("A")

        # More wins = higher rating (later games don't leak back)
        assert rating_after_3 > rating_after_1

    def test_mov_adjustment_effect(self, sample_games):
        no_mov = EloTemporalModel(mov_adjustment=False)
        no_mov.train(sample_games)

        with_mov = EloTemporalModel(mov_adjustment=True)
        with_mov.train(sample_games)

        # Ratings should differ when MOV is enabled
        assert no_mov.get_rating("Duke") != with_mov.get_rating("Duke")
