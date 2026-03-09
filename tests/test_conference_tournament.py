"""Tests for conference tournament prediction module."""

import json
import math
import os
import tempfile

import pytest

from src.models.conference_tournament import (
    ConferenceTournamentBracket,
    ConferenceTournamentGame,
    ConferenceTeam,
    _CONFERENCE_FULL_NAMES,
)
from src.conference_tournament.predictor import (
    ConferenceTournamentPredictor,
    _DEFAULT_CONF_TOURNAMENT_SIZES,
)


# ---------------------------------------------------------------------------
# ConferenceTeam
# ---------------------------------------------------------------------------


class TestConferenceTeam:
    def test_str_representation(self):
        team = ConferenceTeam(
            team_id="duke", name="Duke", conf_seed=1, conference="ACC"
        )
        assert str(team) == "[1] Duke"

    def test_adj_em_default(self):
        team = ConferenceTeam(
            team_id="duke", name="Duke", conf_seed=1, conference="ACC"
        )
        assert team.adj_em == 0.0


# ---------------------------------------------------------------------------
# ConferenceTournamentGame
# ---------------------------------------------------------------------------


class TestConferenceTournamentGame:
    def _make_teams(self):
        t1 = ConferenceTeam("duke", "Duke", 1, "ACC", adj_em=30.0)
        t2 = ConferenceTeam("unc", "North Carolina", 8, "ACC", adj_em=10.0)
        return t1, t2

    def test_set_prediction(self):
        t1, t2 = self._make_teams()
        game = ConferenceTournamentGame(game_id=1, round=1, round_name="QF", team1=t1, team2=t2)
        game.set_prediction(t1, 0.75)
        assert game.winner == t1
        assert game.win_probability == 0.75

    def test_set_prediction_invalid_prob(self):
        t1, t2 = self._make_teams()
        game = ConferenceTournamentGame(game_id=1, round=1, round_name="QF", team1=t1, team2=t2)
        with pytest.raises(ValueError, match="Probability"):
            game.set_prediction(t1, 1.5)

    def test_is_upset_false_when_higher_seed_wins(self):
        t1, t2 = self._make_teams()
        game = ConferenceTournamentGame(game_id=1, round=1, round_name="QF", team1=t1, team2=t2)
        game.set_prediction(t1, 0.8)
        assert not game.is_upset

    def test_is_upset_true_when_lower_seed_wins(self):
        t1, t2 = self._make_teams()
        game = ConferenceTournamentGame(game_id=1, round=1, round_name="QF", team1=t1, team2=t2)
        game.set_prediction(t2, 0.55)
        assert game.is_upset

    def test_to_dict(self):
        t1, t2 = self._make_teams()
        game = ConferenceTournamentGame(game_id=1, round=1, round_name="QF", team1=t1, team2=t2)
        game.set_prediction(t1, 0.7)
        d = game.to_dict()
        assert d["game_id"] == 1
        assert d["round_name"] == "QF"
        assert d["winner"] == "[1] Duke"
        assert d["win_probability"] == 0.7


# ---------------------------------------------------------------------------
# ConferenceTournamentBracket
# ---------------------------------------------------------------------------


class TestConferenceTournamentBracket:
    def _make_teams(self, n):
        return [
            ConferenceTeam(
                team_id=f"team_{i}",
                name=f"Team {i}",
                conf_seed=i,
                conference="TEST",
                adj_em=30.0 - i * 2,
            )
            for i in range(1, n + 1)
        ]

    def test_4_team_bracket(self):
        teams = self._make_teams(4)
        bracket = ConferenceTournamentBracket("TEST", teams)
        assert bracket.total_rounds == 2
        assert bracket.num_byes == 0
        assert len(bracket.games) == 2
        assert len(bracket.games[0]) == 2  # Semifinals
        assert len(bracket.games[1]) == 1  # Championship

    def test_8_team_bracket(self):
        teams = self._make_teams(8)
        bracket = ConferenceTournamentBracket("TEST", teams)
        assert bracket.total_rounds == 3
        assert bracket.num_byes == 0
        assert len(bracket.games[0]) == 4  # Quarterfinals

    def test_6_team_bracket_with_byes(self):
        teams = self._make_teams(6)
        bracket = ConferenceTournamentBracket("TEST", teams)
        assert bracket.total_rounds == 3
        assert bracket.num_byes == 2  # 8 - 6 = 2 byes

    def test_12_team_bracket_with_byes(self):
        teams = self._make_teams(12)
        bracket = ConferenceTournamentBracket("TEST", teams)
        assert bracket.total_rounds == 4
        assert bracket.num_byes == 4  # 16 - 12 = 4 byes

    def test_2_team_bracket(self):
        teams = self._make_teams(2)
        bracket = ConferenceTournamentBracket("TEST", teams)
        assert bracket.total_rounds == 1
        assert len(bracket.games) == 1
        assert len(bracket.games[0]) == 1

    def test_too_few_teams_raises(self):
        teams = self._make_teams(1)
        with pytest.raises(ValueError, match="at least 2"):
            ConferenceTournamentBracket("TEST", teams)

    def test_get_all_games(self):
        teams = self._make_teams(8)
        bracket = ConferenceTournamentBracket("TEST", teams)
        all_games = bracket.get_all_games()
        assert len(all_games) == 7  # 4 + 2 + 1

    def test_summary_contains_conference_name(self):
        teams = self._make_teams(4)
        bracket = ConferenceTournamentBracket("ACC", teams)
        summary = bracket.summary()
        assert "ACC" in summary

    def test_to_dict(self):
        teams = self._make_teams(4)
        bracket = ConferenceTournamentBracket("ACC", teams)
        d = bracket.to_dict()
        assert d["conference"] == "ACC"
        assert d["num_teams"] == 4
        assert len(d["rounds"]) == 2


# ---------------------------------------------------------------------------
# ConferenceTournamentPredictor
# ---------------------------------------------------------------------------


class TestConferenceTournamentPredictor:
    def _make_predictor(self, n_teams=8):
        teams = [
            ConferenceTeam(
                team_id=f"team_{i}",
                name=f"Team {i}",
                conf_seed=i,
                conference="TEST",
                t_rank=i * 10,
                adj_em=30.0 - i * 3,
            )
            for i in range(1, n_teams + 1)
        ]
        return ConferenceTournamentPredictor(
            teams_by_conference={"TEST": teams},
        )

    def test_list_conferences(self):
        predictor = self._make_predictor()
        assert predictor.list_conferences() == ["TEST"]

    def test_get_conference_teams(self):
        predictor = self._make_predictor()
        teams = predictor.get_conference_teams("TEST")
        assert len(teams) == 8
        assert teams[0].conf_seed == 1

    def test_predict_matchup_standalone(self):
        predictor = self._make_predictor()
        teams = predictor.get_conference_teams("TEST")
        # Team 1 (AdjEM=27) should beat Team 8 (AdjEM=6) with high probability
        prob = predictor.predict_matchup(teams[0], teams[-1])
        assert prob > 0.7

    def test_predict_matchup_symmetry(self):
        predictor = self._make_predictor()
        teams = predictor.get_conference_teams("TEST")
        p_ab = predictor.predict_matchup(teams[0], teams[1])
        p_ba = predictor.predict_matchup(teams[1], teams[0])
        # Standalone logistic model is exactly symmetric
        assert abs(p_ab + p_ba - 1.0) < 1e-10

    def test_predict_matchup_clipped(self):
        """Predictions should be clipped to [0.02, 0.98]."""
        predictor = self._make_predictor()
        # Create extreme teams
        strong = ConferenceTeam("strong", "Strong", 1, "T", adj_em=100.0)
        weak = ConferenceTeam("weak", "Weak", 2, "T", adj_em=-50.0)
        prob = predictor.predict_matchup(strong, weak)
        assert 0.02 <= prob <= 0.98

    def test_predict_conference(self):
        predictor = self._make_predictor()
        bracket = predictor.predict_conference("TEST")
        assert bracket.champion is not None
        # The 1-seed (highest AdjEM) should be predicted champion
        # in a pure efficiency model
        assert bracket.champion.conf_seed == 1

    def test_predict_conference_invalid(self):
        predictor = self._make_predictor()
        with pytest.raises(ValueError, match="not found"):
            predictor.predict_conference("INVALID")

    def test_predict_all(self):
        predictor = self._make_predictor()
        results = predictor.predict_all()
        assert "TEST" in results
        assert results["TEST"].champion is not None

    def test_generate_report(self):
        predictor = self._make_predictor()
        report = predictor.generate_report()
        assert "CONFERENCE TOURNAMENT PREDICTIONS" in report
        assert "Team 1" in report  # Champion should appear

    def test_to_json(self):
        predictor = self._make_predictor()
        output = predictor.to_json()
        data = json.loads(output)
        assert "TEST" in data
        assert data["TEST"]["champion"] is not None

    def test_from_torvik_json(self):
        """Test loading from a Torvik JSON file."""
        torvik_data = {
            "teams": [
                {
                    "team_id": "duke",
                    "team_name": "Duke",
                    "conference": "ACC",
                    "t_rank": 3,
                    "adj_offensive_efficiency": 120.0,
                    "adj_defensive_efficiency": 90.0,
                },
                {
                    "team_id": "unc",
                    "team_name": "North Carolina",
                    "conference": "ACC",
                    "t_rank": 12,
                    "adj_offensive_efficiency": 110.0,
                    "adj_defensive_efficiency": 95.0,
                },
                {
                    "team_id": "gonzaga",
                    "team_name": "Gonzaga",
                    "conference": "WCC",
                    "t_rank": 5,
                    "adj_offensive_efficiency": 118.0,
                    "adj_defensive_efficiency": 92.0,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(torvik_data, f)
            tmp_path = f.name

        try:
            predictor = ConferenceTournamentPredictor.from_torvik_json(tmp_path)
            assert "ACC" in predictor.list_conferences()
            assert "WCC" in predictor.list_conferences()

            acc_teams = predictor.get_conference_teams("ACC")
            assert len(acc_teams) == 2
            # Duke has higher AdjEM (120-90=30) vs UNC (110-95=15)
            assert acc_teams[0].team_id == "duke"
            assert acc_teams[0].conf_seed == 1
            assert acc_teams[1].team_id == "unc"
            assert acc_teams[1].conf_seed == 2
        finally:
            os.unlink(tmp_path)

    def test_predict_with_actual_torvik_data(self):
        """Integration test using actual 2026 Torvik data if available."""
        torvik_path = "data/raw/torvik_2026.json"
        if not os.path.exists(torvik_path):
            pytest.skip("Torvik 2026 data not available")

        predictor = ConferenceTournamentPredictor.from_torvik_json(torvik_path)
        conferences = predictor.list_conferences()

        assert len(conferences) >= 25  # Should have ~31 conferences

        # Predict a major conference
        bracket = predictor.predict_conference("B10")
        assert bracket.champion is not None
        assert bracket.conference == "B10"

        # Predict all conferences
        results = predictor.predict_all()
        assert len(results) >= 25

        # Generate report should not crash
        report = predictor.generate_report()
        assert len(report) > 100


# ---------------------------------------------------------------------------
# Standalone logistic model calibration check
# ---------------------------------------------------------------------------


class TestStandaloneModel:
    """Verify the standalone AdjEM-based logistic model is well-calibrated."""

    def test_equal_teams_50_50(self):
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        t1 = ConferenceTeam("a", "A", 1, "T", adj_em=10.0)
        t2 = ConferenceTeam("b", "B", 2, "T", adj_em=10.0)
        prob = predictor.predict_matchup(t1, t2)
        assert abs(prob - 0.5) < 1e-10

    def test_10pt_gap_roughly_82pct(self):
        """A 10-point AdjEM gap should yield ~82% win probability."""
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        t1 = ConferenceTeam("a", "A", 1, "T", adj_em=20.0)
        t2 = ConferenceTeam("b", "B", 2, "T", adj_em=10.0)
        prob = predictor.predict_matchup(t1, t2)
        # 1/(1+exp(-0.15*10)) ≈ 0.817
        assert 0.75 < prob < 0.90

    def test_20pt_gap_roughly_95pct(self):
        """A 20-point AdjEM gap (e.g. 1-seed vs 16-seed) should be ~95%."""
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        t1 = ConferenceTeam("a", "A", 1, "T", adj_em=30.0)
        t2 = ConferenceTeam("b", "B", 2, "T", adj_em=10.0)
        prob = predictor.predict_matchup(t1, t2)
        # 1/(1+exp(-0.15*20)) ≈ 0.953
        assert 0.90 < prob < 0.98


# ---------------------------------------------------------------------------
# Tournament size completeness
# ---------------------------------------------------------------------------


class TestTournamentSizeCompleteness:
    """Verify all 31 conferences have correct tournament size entries."""

    def test_all_conferences_have_tournament_sizes(self):
        """Every conference in the full-name mapping must have a tournament size."""
        missing = set(_CONFERENCE_FULL_NAMES) - set(_DEFAULT_CONF_TOURNAMENT_SIZES)
        assert not missing, f"Conferences missing tournament sizes: {sorted(missing)}"

    def test_tournament_sizes_are_valid(self):
        """All tournament sizes must be between 4 and 18."""
        for conf, size in _DEFAULT_CONF_TOURNAMENT_SIZES.items():
            assert 4 <= size <= 18, (
                f"{conf} has invalid tournament size {size}"
            )

    def test_ivy_league_4_team_bracket(self):
        """Ivy League tournament should use exactly 4 teams (top 4 qualify)."""
        teams = [
            ConferenceTeam(f"ivy_{i}", f"Ivy {i}", i, "Ivy", adj_em=20.0 - i * 2)
            for i in range(1, 9)
        ]
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"Ivy": teams},
        )
        bracket = predictor.predict_conference("Ivy")
        assert bracket.num_teams == 4
        assert bracket.total_rounds == 2  # Semis + Championship

    def test_all_conferences_correct_team_count(self):
        """Integration: each bracket should have the configured number of teams."""
        torvik_path = "data/raw/torvik_2026.json"
        if not os.path.exists(torvik_path):
            pytest.skip("Torvik 2026 data not available")

        predictor = ConferenceTournamentPredictor.from_torvik_json(torvik_path)
        results = predictor.predict_all()

        for conf, bracket in results.items():
            expected = _DEFAULT_CONF_TOURNAMENT_SIZES.get(conf, bracket.num_teams)
            assert bracket.num_teams == expected, (
                f"{conf}: expected {expected} teams, got {bracket.num_teams}"
            )
