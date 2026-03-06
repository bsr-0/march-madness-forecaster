"""Tests for historical tournament results loading and backtesting."""

import json

import pytest

from src.data.historical_tournament_results import (
    TournamentGame,
    get_available_years,
    get_tournament_games_for_eval,
    load_tournament_results,
    save_tournament_results,
)


class TestTournamentGame:
    """Tests for the TournamentGame dataclass."""

    def test_winner_id(self):
        g = TournamentGame(
            year=2024, round_name="R64",
            team1_id="duke", team2_id="vermont",
            team1_seed=4, team2_seed=13,
            team1_score=64, team2_score=47,
            team1_won=True,
        )
        assert g.winner_id == "duke"
        assert g.loser_id == "vermont"

    def test_upset_lower_seed_loses(self):
        g = TournamentGame(
            year=2024, round_name="R64",
            team1_id="houston", team2_id="longshot",
            team1_seed=1, team2_seed=16,
            team1_score=50, team2_score=60,
            team1_won=False,
        )
        assert g.upset is True

    def test_no_upset_chalk(self):
        g = TournamentGame(
            year=2024, round_name="R64",
            team1_id="houston", team2_id="longshot",
            team1_seed=1, team2_seed=16,
            team1_score=80, team2_score=50,
            team1_won=True,
        )
        assert g.upset is False

    def test_to_eval_dict(self):
        g = TournamentGame(
            year=2024, round_name="R32",
            team1_id="connecticut", team2_id="florida_atlantic",
            team1_seed=1, team2_seed=8,
            team1_score=75, team2_score=35,
            team1_won=True,
        )
        d = g.to_eval_dict()
        assert d["team1_id"] == "connecticut"
        assert d["round"] == "R32"
        assert d["team1_won"] is True
        assert d["team1_seed"] == 1


class TestLoadSaveTournamentResults:
    """Tests for loading/saving tournament results JSON files."""

    def test_save_and_load_roundtrip(self, tmp_path):
        games = [
            TournamentGame(
                year=2024, round_name="NCG",
                team1_id="connecticut", team2_id="purdue",
                team1_seed=1, team2_seed=1,
                team1_score=75, team2_score=60,
                team1_won=True,
            ),
        ]
        save_tournament_results(games, 2024, str(tmp_path))

        loaded = load_tournament_results(2024, str(tmp_path))
        assert len(loaded) == 1
        assert loaded[0].team1_id == "connecticut"
        assert loaded[0].team1_won is True
        assert loaded[0].round_name == "NCG"

    def test_load_missing_year(self, tmp_path):
        result = load_tournament_results(2099, str(tmp_path))
        assert result == []

    def test_get_available_years(self, tmp_path):
        # Save data for two years
        for year in [2023, 2024]:
            games = [
                TournamentGame(
                    year=year, round_name="NCG",
                    team1_id="a", team2_id="b",
                    team1_seed=1, team2_seed=1,
                    team1_score=75, team2_score=60,
                    team1_won=True,
                ),
            ]
            save_tournament_results(games, year, str(tmp_path))

        avail = get_available_years(str(tmp_path))
        assert 2023 in avail
        assert 2024 in avail

    def test_get_tournament_games_for_eval(self, tmp_path):
        games = [
            TournamentGame(
                year=2024, round_name="R64",
                team1_id="duke", team2_id="vermont",
                team1_seed=4, team2_seed=13,
                team1_score=64, team2_score=47,
                team1_won=True,
            ),
            TournamentGame(
                year=2024, round_name="R64",
                team1_id="houston", team2_id="longwood",
                team1_seed=1, team2_seed=16,
                team1_score=87, team2_score=56,
                team1_won=True,
            ),
        ]
        save_tournament_results(games, 2024, str(tmp_path))

        eval_games = get_tournament_games_for_eval(2024, str(tmp_path))
        assert len(eval_games) == 2
        assert eval_games[0]["team1_id"] == "duke"
        assert eval_games[0]["round"] == "R64"


class TestKaggleBacktesterIntegration:
    """Test that tournament results work with KaggleBacktester."""

    def test_evaluate_seed_baseline(self, tmp_path):
        from src.ml.evaluation.kaggle_backtest import KaggleBacktester

        # Create minimal tournament results
        games = [
            TournamentGame(
                year=2024, round_name="R64",
                team1_id="duke", team2_id="vermont",
                team1_seed=4, team2_seed=13,
                team1_score=64, team2_score=47,
                team1_won=True,
            ),
            TournamentGame(
                year=2024, round_name="R64",
                team1_id="houston", team2_id="longwood",
                team1_seed=1, team2_seed=16,
                team1_score=87, team2_score=56,
                team1_won=True,
            ),
        ]
        save_tournament_results(games, 2024, str(tmp_path))

        actual = get_tournament_games_for_eval(2024, str(tmp_path))

        # Seed-based predictions: chalk (lower seed always wins)
        predictions = {
            ("duke", "vermont"): 0.8,
            ("houston", "longwood"): 0.99,
        }

        backtester = KaggleBacktester(historical_results_dir=str(tmp_path))
        result = backtester.evaluate_predictions(predictions, actual, 2024)

        assert result.year == 2024
        assert result.n_games == 2
        assert result.accuracy == 1.0  # Both chalk picks correct
        assert result.brier_score < 0.25  # Better than coin flip
