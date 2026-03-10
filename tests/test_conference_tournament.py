"""Tests for conference tournament prediction module."""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from src.models.conference_tournament import (
    ConferenceTournamentBracket,
    ConferenceTournamentGame,
    ConferenceTeam,
    _CONFERENCE_FULL_NAMES,
    bracket_seed_order,
)
from src.conference_tournament.bracket_formats import (
    BracketFormat,
    get_bracket_format,
    get_all_formats,
    _standard_format,
)
from src.conference_tournament.predictor import (
    ConferenceTournamentPredictor,
    _DEFAULT_CONF_TOURNAMENT_SIZES,
)
from src.conference_tournament.data_enrichment import (
    enrich_torvik_teams,
    _fuzzy_lookup,
    _merge_if_zero,
)
from src.conference_tournament.seeding import (
    apply_seed_overrides,
    load_seed_overrides,
    seed_by_adj_em,
)
from src.conference_tournament.simulator import (
    ConferenceTournamentSimulator,
    ConferenceTournamentResult,
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

    def test_stats_dict(self):
        team = ConferenceTeam(
            team_id="duke", name="Duke", conf_seed=1, conference="ACC",
            stats={"effective_fg_pct": 0.55, "turnover_rate": 0.15},
        )
        assert team.stats["effective_fg_pct"] == 0.55

    def test_adj_o_adj_d_tempo(self):
        team = ConferenceTeam(
            team_id="duke", name="Duke", conf_seed=1, conference="ACC",
            adj_o=120.0, adj_d=90.0, tempo=70.0,
        )
        assert team.adj_o == 120.0
        assert team.adj_d == 90.0
        assert team.tempo == 70.0


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
# Bracket Seed Ordering
# ---------------------------------------------------------------------------


class TestBracketSeedOrder:
    """Verify bracket_seed_order produces proper tournament bracket positions."""

    def test_2_teams(self):
        assert bracket_seed_order(2) == [0, 1]

    def test_4_teams(self):
        # Should produce matchups: 1v4, 2v3
        order = bracket_seed_order(4)
        assert order == [0, 3, 1, 2]

    def test_8_teams(self):
        # Should produce matchups: 1v8, 4v5, 2v7, 3v6
        # Seeds 1 and 2 on opposite halves
        order = bracket_seed_order(8)
        assert order == [0, 7, 3, 4, 1, 6, 2, 5]

    def test_16_teams(self):
        order = bracket_seed_order(16)
        # First pair: seed 1 (idx 0) vs seed 16 (idx 15)
        assert order[0] == 0
        assert order[1] == 15
        # Seed 2 (idx 1) should be in the bottom half (positions 8-15)
        seed2_pos = order.index(1)
        assert seed2_pos >= 8, "Seed 2 must be in bottom half of bracket"

    def test_1_and_2_on_opposite_halves(self):
        """#1 and #2 seeds must always be on opposite halves."""
        for n in [4, 8, 16]:
            order = bracket_seed_order(n)
            pos_seed1 = order.index(0)
            pos_seed2 = order.index(1)
            half = n // 2
            seed1_half = pos_seed1 // half
            seed2_half = pos_seed2 // half
            assert seed1_half != seed2_half, (
                f"n={n}: seeds 1 and 2 are in the same half "
                f"(positions {pos_seed1} and {pos_seed2})"
            )

    def test_empty_and_single(self):
        assert bracket_seed_order(0) == []
        assert bracket_seed_order(1) == [0]


class TestBracketSeedingIntegrity:
    """Integration tests verifying #1 and #2 seeds cannot meet before the final."""

    def _make_teams(self, n):
        return [
            ConferenceTeam(
                team_id=f"team_{i}",
                name=f"Team {i}",
                conf_seed=i,
                conference="TEST",
                adj_em=30.0 - i * 2,
                adj_o=120.0 - i,
                adj_d=90.0 + i,
            )
            for i in range(1, n + 1)
        ]

    def test_8_team_bracket_seeds_opposite_halves(self):
        """In an 8-team bracket, #1 and #2 should only meet in the Championship."""
        teams = self._make_teams(8)
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"TEST": teams},
        )
        bracket = predictor.predict_conference("TEST")

        # With deterministic predictions, top seeds always win.
        # #1 and #2 should meet only in the final.
        final = bracket.games[-1][0]
        assert final.round_name == "Championship"
        seeds_in_final = {final.team1.conf_seed, final.team2.conf_seed}
        assert seeds_in_final == {1, 2}, (
            f"Expected seeds 1 and 2 in final, got {seeds_in_final}"
        )

    def test_acc_15_team_bracket_seeds_opposite_halves(self):
        """In a 15-team bracket (like ACC), #1 and #2 must be on opposite sides."""
        teams = self._make_teams(15)
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"TEST": teams},
        )
        bracket = predictor.predict_conference("TEST")

        # The Championship game should feature seeds 1 and 2 when
        # higher seeds always win (deterministic).
        final = bracket.games[-1][0]
        assert final.round_name == "Championship"
        seeds_in_final = {final.team1.conf_seed, final.team2.conf_seed}
        assert seeds_in_final == {1, 2}, (
            f"Expected seeds 1 and 2 in final, got {seeds_in_final}"
        )

    def test_16_team_bracket_top4_separation(self):
        """In a 16-team bracket, seeds 1-4 should be in separate quarter-brackets."""
        teams = self._make_teams(16)
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"TEST": teams},
        )
        bracket = predictor.predict_conference("TEST")

        # Semifinals should feature seeds 1, 2, 3, 4 when top seeds always win
        semis = bracket.games[-2]
        semi_seeds = set()
        for game in semis:
            semi_seeds.add(game.team1.conf_seed)
            semi_seeds.add(game.team2.conf_seed)
        assert semi_seeds == {1, 2, 3, 4}, (
            f"Expected seeds 1-4 in semis, got {semi_seeds}"
        )

    def test_1v2_never_before_final(self):
        """Seed 1 and 2 must never play each other before the Championship."""
        for n in [4, 8, 12, 15, 16]:
            teams = self._make_teams(n)
            predictor = ConferenceTournamentPredictor(
                teams_by_conference={"TEST": teams},
            )
            bracket = predictor.predict_conference("TEST")

            # Check all rounds except the final
            for round_games in bracket.games[:-1]:
                for game in round_games:
                    if game.team1 and game.team2:
                        seeds = {game.team1.conf_seed, game.team2.conf_seed}
                        assert seeds != {1, 2}, (
                            f"n={n}: Seeds 1 and 2 met in {game.round_name}!"
                        )


# ---------------------------------------------------------------------------
# Data Enrichment
# ---------------------------------------------------------------------------


class TestDataEnrichment:
    def test_merge_if_zero_replaces_zero(self):
        team = {"effective_fg_pct": 0.0}
        source = {"effective_fg_pct": 0.55}
        _merge_if_zero(team, source, "effective_fg_pct")
        assert team["effective_fg_pct"] == 0.55

    def test_merge_if_zero_preserves_nonzero(self):
        team = {"effective_fg_pct": 0.48}
        source = {"effective_fg_pct": 0.55}
        _merge_if_zero(team, source, "effective_fg_pct")
        assert team["effective_fg_pct"] == 0.48

    def test_merge_if_zero_replaces_sentinel_1(self):
        """1.0 is a sentinel for ORB%/DRB% in bad data."""
        team = {"offensive_reb_rate": 1.0}
        source = {"offensive_reb_rate": 0.33}
        _merge_if_zero(team, source, "offensive_reb_rate")
        assert team["offensive_reb_rate"] == 0.33

    def test_fuzzy_lookup_exact_match(self):
        data = {"michigan": {"efg": 0.55}}
        assert _fuzzy_lookup(data, "michigan") is not None

    def test_fuzzy_lookup_double_underscore(self):
        data = {"miami__fl": {"efg": 0.50}}
        assert _fuzzy_lookup(data, "miami_fl") is not None

    def test_fuzzy_lookup_nc_pattern(self):
        data = {"nc_state": {"efg": 0.52}}
        assert _fuzzy_lookup(data, "n_c_state") is not None

    def test_fuzzy_lookup_no_match(self):
        data = {"duke": {"efg": 0.55}}
        assert _fuzzy_lookup(data, "nonexistent") is None

    def test_enrich_torvik_teams_merges_four_factors(self):
        """Test enrichment with mock data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write mock four factors file
            ff = {"duke": {"effective_fg_pct": 0.55, "turnover_rate": 0.15}}
            with open(os.path.join(tmpdir, "torvik_four_factors_2025.json"), "w") as f:
                json.dump(ff, f)

            # Write mock shooting file
            shooting = {"duke": {"ft_pct": 0.78, "three_pt_pct": 0.38}}
            with open(os.path.join(tmpdir, "torvik_shooting_2025.json"), "w") as f:
                json.dump(shooting, f)

            # Base torvik data with zero four factors
            torvik = {
                "teams": [{
                    "team_id": "duke",
                    "team_name": "Duke",
                    "conference": "ACC",
                    "effective_fg_pct": 0.0,
                    "turnover_rate": 0.0,
                }]
            }

            enriched = enrich_torvik_teams(torvik, data_dir=tmpdir, year=2026)
            team = enriched["teams"][0]
            assert team["effective_fg_pct"] == 0.55
            assert team["turnover_rate"] == 0.15
            assert team["enriched_stats"]["ft_pct"] == 0.78

    def test_enrich_torvik_teams_fallback_year(self):
        """Falls back to 2025 when 2026 file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ff = {"duke": {"effective_fg_pct": 0.53}}
            with open(os.path.join(tmpdir, "torvik_four_factors_2025.json"), "w") as f:
                json.dump(ff, f)

            torvik = {
                "teams": [{"team_id": "duke", "effective_fg_pct": 0.0}]
            }

            enriched = enrich_torvik_teams(torvik, data_dir=tmpdir, year=2026)
            assert enriched["teams"][0]["effective_fg_pct"] == 0.53


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeeding:
    def _make_teams(self, n):
        return [
            ConferenceTeam(
                team_id=f"team_{i}", name=f"Team {i}",
                conf_seed=0, conference="TEST", adj_em=30.0 - i * 3,
            )
            for i in range(1, n + 1)
        ]

    def test_seed_by_adj_em(self):
        teams = self._make_teams(5)
        result = seed_by_adj_em(teams)
        assert result[0].conf_seed == 1
        assert result[0].adj_em == 27.0  # Highest AdjEM
        assert result[-1].conf_seed == 5

    def test_apply_seed_overrides(self):
        teams = self._make_teams(4)
        overrides = {"team_3": 1, "team_1": 3}  # Swap 1 and 3 seeds
        result = apply_seed_overrides(teams, overrides)
        # team_3 should be seed 1 now
        assert result[0].team_id == "team_3"
        assert result[0].conf_seed == 1
        # team_1 should be seed 3
        assert next(t for t in result if t.team_id == "team_1").conf_seed == 3

    def test_apply_seed_overrides_partial(self):
        """Override only some teams; rest assigned by AdjEM."""
        teams = self._make_teams(4)
        overrides = {"team_2": 1}
        result = apply_seed_overrides(teams, overrides)
        assert result[0].team_id == "team_2"
        assert result[0].conf_seed == 1

    def test_load_seed_overrides_from_file(self):
        data = {"ACC": {"duke": 1, "unc": 4}, "SEC": {"florida": 1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            overrides = load_seed_overrides(path)
            assert "ACC" in overrides
            assert overrides["ACC"]["duke"] == 1
            assert "SEC" in overrides
        finally:
            os.unlink(path)

    def test_load_seed_overrides_missing_file(self):
        overrides = load_seed_overrides("/nonexistent/path.json")
        assert overrides == {}


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
                adj_o=120.0 - i * 2,
                adj_d=90.0 + i * 1,
                stats={"effective_fg_pct": 0.55 - i * 0.01, "turnover_rate": 0.12 + i * 0.005},
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

    def test_predict_matchup_standalone_multi_feature(self):
        predictor = self._make_predictor()
        teams = predictor.get_conference_teams("TEST")
        # Team 1 should beat Team 8 with high probability
        prob = predictor.predict_matchup(teams[0], teams[-1])
        assert prob > 0.6

    def test_predict_matchup_symmetry(self):
        predictor = self._make_predictor()
        teams = predictor.get_conference_teams("TEST")
        p_ab = predictor.predict_matchup(teams[0], teams[1])
        p_ba = predictor.predict_matchup(teams[1], teams[0])
        # Logistic model is exactly symmetric
        assert abs(p_ab + p_ba - 1.0) < 1e-10

    def test_predict_matchup_clipped(self):
        """Predictions should be clipped to [0.02, 0.98]."""
        predictor = self._make_predictor()
        strong = ConferenceTeam("strong", "Strong", 1, "T", adj_em=100.0, adj_o=200.0, adj_d=100.0)
        weak = ConferenceTeam("weak", "Weak", 2, "T", adj_em=-50.0, adj_o=50.0, adj_d=100.0)
        prob = predictor.predict_matchup(strong, weak)
        assert 0.02 <= prob <= 0.98

    def test_predict_conference(self):
        predictor = self._make_predictor()
        bracket = predictor.predict_conference("TEST")
        assert bracket.champion is not None

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
                    "adj_tempo": 70.0,
                    "effective_fg_pct": 0.0,
                },
                {
                    "team_id": "unc",
                    "team_name": "North Carolina",
                    "conference": "ACC",
                    "t_rank": 12,
                    "adj_offensive_efficiency": 110.0,
                    "adj_defensive_efficiency": 95.0,
                    "adj_tempo": 68.0,
                    "effective_fg_pct": 0.0,
                },
                {
                    "team_id": "gonzaga",
                    "team_name": "Gonzaga",
                    "conference": "WCC",
                    "t_rank": 5,
                    "adj_offensive_efficiency": 118.0,
                    "adj_defensive_efficiency": 92.0,
                    "adj_tempo": 72.0,
                    "effective_fg_pct": 0.0,
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
            assert acc_teams[0].conf_seed == 1
            assert acc_teams[0].adj_em == pytest.approx(30.0)
            assert acc_teams[1].conf_seed == 2
            assert acc_teams[1].adj_em == pytest.approx(15.0)

            # Verify AdjO/AdjD/Tempo stored
            assert acc_teams[0].adj_o == 120.0
            assert acc_teams[0].adj_d == 90.0
            assert acc_teams[0].tempo == 70.0
        finally:
            os.unlink(tmp_path)

    def test_from_torvik_json_with_seed_overrides(self):
        """Test that seed overrides are applied."""
        torvik_data = {
            "teams": [
                {"team_id": "duke", "team_name": "Duke", "conference": "ACC",
                 "adj_offensive_efficiency": 120.0, "adj_defensive_efficiency": 90.0},
                {"team_id": "virginia", "team_name": "Virginia", "conference": "ACC",
                 "adj_offensive_efficiency": 110.0, "adj_defensive_efficiency": 95.0},
            ],
        }
        # Override: Virginia is 1-seed despite lower AdjEM
        seed_overrides = {"ACC": {"virginia": 1, "duke": 2}}

        with tempfile.TemporaryDirectory() as tmpdir:
            torvik_path = os.path.join(tmpdir, "torvik.json")
            seeds_path = os.path.join(tmpdir, "seeds.json")
            with open(torvik_path, "w") as f:
                json.dump(torvik_data, f)
            with open(seeds_path, "w") as f:
                json.dump(seed_overrides, f)

            predictor = ConferenceTournamentPredictor.from_torvik_json(
                torvik_path, seed_overrides_path=seeds_path,
            )
            acc = predictor.get_conference_teams("ACC")
            assert acc[0].team_id == "virginia"
            assert acc[0].conf_seed == 1
            assert acc[1].team_id == "duke"
            assert acc[1].conf_seed == 2

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

    def test_enriched_data_loaded(self):
        """Test that Four Factors enrichment works with real data."""
        torvik_path = "data/raw/torvik_2026.json"
        ff_path = "data/raw/torvik_four_factors_2025.json"
        if not os.path.exists(torvik_path) or not os.path.exists(ff_path):
            pytest.skip("Required data files not available")

        predictor = ConferenceTournamentPredictor.from_torvik_json(
            torvik_path, data_dir="data/raw", year=2026,
        )
        # Check that some teams have enriched stats
        b10_teams = predictor.get_conference_teams("B10")
        teams_with_stats = [t for t in b10_teams if t.stats.get("effective_fg_pct", 0) > 0]
        # Most teams should get enriched (fuzzy matching covers edge cases)
        assert len(teams_with_stats) > 0


# ---------------------------------------------------------------------------
# Monte Carlo Simulator
# ---------------------------------------------------------------------------


class TestConferenceTournamentSimulator:
    def _make_predictor(self, n_teams=8):
        teams = [
            ConferenceTeam(
                team_id=f"team_{i}", name=f"Team {i}",
                conf_seed=i, conference="TEST",
                adj_em=30.0 - i * 3, adj_o=120.0 - i * 2, adj_d=90.0 + i,
                stats={"effective_fg_pct": 0.55 - i * 0.01},
            )
            for i in range(1, n_teams + 1)
        ]
        return ConferenceTournamentPredictor(
            teams_by_conference={"TEST": teams},
        )

    def test_simulate_conference(self):
        predictor = self._make_predictor()
        sim = ConferenceTournamentSimulator(predictor, num_simulations=1000, random_seed=42)
        result = sim.simulate_conference("TEST")

        assert result.conference == "TEST"
        assert result.num_simulations == 1000

        # Championship probs should sum to ~1.0
        total_prob = sum(result.championship_probs.values())
        assert abs(total_prob - 1.0) < 0.01

        # Top seed should have highest championship probability
        assert result.most_likely_champion == "team_1"
        assert result.most_likely_champion_prob > 0.1

    def test_simulate_championship_probs_all_positive(self):
        predictor = self._make_predictor(4)
        sim = ConferenceTournamentSimulator(predictor, num_simulations=5000, random_seed=42)
        result = sim.simulate_conference("TEST")

        # All teams should have some chance (with noise)
        for team_id, prob in result.championship_probs.items():
            assert prob > 0, f"{team_id} has zero championship probability"

    def test_simulate_noise_increases_upsets(self):
        """Higher noise should increase lower-seed chances."""
        predictor = self._make_predictor(4)

        sim_low = ConferenceTournamentSimulator(
            predictor, num_simulations=5000, noise_std=0.05, random_seed=42
        )
        sim_high = ConferenceTournamentSimulator(
            predictor, num_simulations=5000, noise_std=0.30, random_seed=42
        )

        result_low = sim_low.simulate_conference("TEST")
        result_high = sim_high.simulate_conference("TEST")

        # With high noise, the top seed should win less often
        assert result_high.championship_probs["team_1"] < result_low.championship_probs["team_1"]

    def test_simulate_all(self):
        predictor = self._make_predictor()
        sim = ConferenceTournamentSimulator(predictor, num_simulations=500, random_seed=42)
        results = sim.simulate_all()
        assert "TEST" in results

    def test_simulate_summary(self):
        predictor = self._make_predictor(4)
        sim = ConferenceTournamentSimulator(predictor, num_simulations=1000, random_seed=42)
        result = sim.simulate_conference("TEST")
        summary = result.summary()
        assert "Championship Probabilities" in summary
        assert "team_1" in summary

    def test_simulate_conference_not_found(self):
        predictor = self._make_predictor()
        sim = ConferenceTournamentSimulator(predictor)
        with pytest.raises(ValueError, match="not found"):
            sim.simulate_conference("INVALID")

    def test_deterministic_with_seed(self):
        """Same random seed should produce same results."""
        predictor = self._make_predictor(4)
        sim1 = ConferenceTournamentSimulator(predictor, num_simulations=1000, random_seed=42)
        sim2 = ConferenceTournamentSimulator(predictor, num_simulations=1000, random_seed=42)

        r1 = sim1.simulate_conference("TEST")
        r2 = sim2.simulate_conference("TEST")

        for team_id in r1.championship_probs:
            assert abs(r1.championship_probs[team_id] - r2.championship_probs[team_id]) < 1e-10


# ---------------------------------------------------------------------------
# Standalone logistic model calibration check
# ---------------------------------------------------------------------------


class TestStandaloneModel:
    """Verify the multi-feature logistic model is well-calibrated."""

    def test_equal_teams_50_50(self):
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        t1 = ConferenceTeam("a", "A", 1, "T", adj_em=10.0, adj_o=110.0, adj_d=100.0)
        t2 = ConferenceTeam("b", "B", 2, "T", adj_em=10.0, adj_o=110.0, adj_d=100.0)
        prob = predictor.predict_matchup(t1, t2)
        assert abs(prob - 0.5) < 1e-10

    def test_stronger_team_wins_more(self):
        """Team with better AdjO and AdjD should have >50% win prob."""
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        t1 = ConferenceTeam("a", "A", 1, "T", adj_o=120.0, adj_d=90.0, adj_em=30.0)
        t2 = ConferenceTeam("b", "B", 2, "T", adj_o=110.0, adj_d=95.0, adj_em=15.0)
        prob = predictor.predict_matchup(t1, t2)
        assert prob > 0.5

    def test_four_factors_contribute(self):
        """Four Factors should influence predictions when available."""
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        # Same AdjO/AdjD but different Four Factors
        t1 = ConferenceTeam(
            "a", "A", 1, "T", adj_o=110.0, adj_d=95.0, adj_em=15.0,
            stats={"effective_fg_pct": 0.55, "turnover_rate": 0.12},
        )
        t2 = ConferenceTeam(
            "b", "B", 2, "T", adj_o=110.0, adj_d=95.0, adj_em=15.0,
            stats={"effective_fg_pct": 0.48, "turnover_rate": 0.18},
        )
        prob = predictor.predict_matchup(t1, t2)
        # Team with better eFG% and lower TO% should be favored
        assert prob > 0.5

    def test_four_factors_missing_graceful(self):
        """Model should work when Four Factors are missing (all zeros)."""
        predictor = ConferenceTournamentPredictor(teams_by_conference={})
        t1 = ConferenceTeam("a", "A", 1, "T", adj_o=120.0, adj_d=90.0, adj_em=30.0)
        t2 = ConferenceTeam("b", "B", 2, "T", adj_o=110.0, adj_d=95.0, adj_em=15.0)
        prob = predictor.predict_matchup(t1, t2)
        # logit = 0.075*(120-110) + (-0.075)*(90-95) = 0.75+0.375 = 1.125
        # 1/(1+exp(-1.125)) ≈ 0.755
        assert 0.70 < prob < 0.85


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


# ---------------------------------------------------------------------------
# Bracket format definitions tests
# ---------------------------------------------------------------------------


class TestBracketFormats:
    """Validate all bracket format definitions."""

    def test_all_formats_validate(self):
        """Every defined format must pass validation (seeds complete, games = N-1)."""
        all_fmts = get_all_formats()
        assert len(all_fmts) >= 31, f"Expected >= 31 formats, got {len(all_fmts)}"
        for conf, fmt in all_fmts.items():
            errors = fmt.validate()
            assert errors == [], f"{conf}: {errors}"

    def test_acc_format_is_5_rounds(self):
        """ACC with 15 teams must have 5 rounds, NOT the mathematical 4."""
        fmt = get_bracket_format("ACC", 15)
        assert fmt.total_rounds == 5
        assert fmt.num_teams == 15

    def test_acc_round1_has_only_bottom_6_seeds(self):
        """ACC round 1 should have seeds 10-15 (NOT seeds 2-15)."""
        fmt = get_bracket_format("ACC", 15)
        r1_seeds = set(fmt.round_entry[1])
        assert r1_seeds == {10, 11, 12, 13, 14, 15}

    def test_acc_round1_matchups(self):
        """ACC first-round matchups must be 10v15, 11v14, 12v13."""
        fmt = get_bracket_format("ACC", 15)
        assert fmt.first_round_matchups is not None
        matchups = set(fmt.first_round_matchups)
        assert matchups == {(10, 15), (11, 14), (12, 13)}

    def test_acc_round2_seeds_5_through_9(self):
        """ACC round 2 should have seeds 5-9 entering (single byes)."""
        fmt = get_bracket_format("ACC", 15)
        r2_seeds = set(fmt.round_entry[2])
        assert r2_seeds == {5, 6, 7, 8, 9}

    def test_acc_round3_seeds_1_through_4(self):
        """ACC round 3 (quarterfinals) should have seeds 1-4 (double byes)."""
        fmt = get_bracket_format("ACC", 15)
        r3_seeds = set(fmt.round_entry[3])
        assert r3_seeds == {1, 2, 3, 4}

    def test_b10_format_is_6_rounds(self):
        """Big Ten with 18 teams should have 6 rounds (stepladder)."""
        fmt = get_bracket_format("B10", 18)
        assert fmt.total_rounds == 6

    def test_b10_round1_bottom_4(self):
        """Big Ten round 1 should have seeds 15-18."""
        fmt = get_bracket_format("B10", 18)
        r1_seeds = set(fmt.round_entry[1])
        assert r1_seeds == {15, 16, 17, 18}

    def test_sec_16_team_stepladder_format(self):
        """SEC 16-team bracket should have 5 rounds (stepladder)."""
        fmt = get_bracket_format("SEC", 16)
        assert fmt.total_rounds == 5
        r1_seeds = set(fmt.round_entry[1])
        assert r1_seeds == {9, 10, 11, 12, 13, 14, 15, 16}
        assert set(fmt.round_entry[2]) == {5, 6, 7, 8}
        assert set(fmt.round_entry[3]) == {1, 2, 3, 4}

    def test_standard_8_team_format(self):
        """8-team brackets should have 3 rounds, 0 byes."""
        fmt = get_bracket_format("AE", 8)
        assert fmt.total_rounds == 3
        r1_seeds = set(fmt.round_entry[1])
        assert r1_seeds == set(range(1, 9))

    def test_standard_format_fallback(self):
        """Unknown conferences should get a mathematical default."""
        fmt = get_bracket_format("UNKNOWN", 12)
        assert fmt.total_rounds == 4
        assert fmt.num_teams == 12

    def test_all_seeds_covered(self):
        """For every format, all seeds 1..N must appear exactly once."""
        all_fmts = get_all_formats()
        for conf, fmt in all_fmts.items():
            all_seeds = []
            for seeds in fmt.round_entry.values():
                all_seeds.extend(seeds)
            assert sorted(all_seeds) == list(range(1, fmt.num_teams + 1)), (
                f"{conf}: seed coverage mismatch"
            )


class TestACCBracketIntegration:
    """Integration tests verifying the ACC bracket produces correct matchups."""

    @staticmethod
    def _make_acc_teams(n=15):
        return [
            ConferenceTeam(
                team_id=f"team_{i}",
                name=f"Team {i}",
                conf_seed=i,
                conference="ACC",
                adj_em=30.0 - i * 1.5,
                adj_o=110.0 - i * 0.5,
                adj_d=80.0 + i * 0.5,
            )
            for i in range(1, n + 1)
        ]

    def test_acc_bracket_has_5_rounds(self):
        """ACC bracket should have 5 rounds."""
        teams = self._make_acc_teams()
        bracket = ConferenceTournamentBracket("ACC", teams)
        assert len(bracket.games) == 5
        assert bracket.total_rounds == 5

    def test_acc_first_round_has_3_games(self):
        """ACC round 1 should have exactly 3 games (not 7)."""
        teams = self._make_acc_teams()
        bracket = ConferenceTournamentBracket("ACC", teams)
        r1_games = bracket.games[0]
        assert len(r1_games) == 3

    def test_acc_first_round_matchups_correct(self):
        """ACC R1 must be 10v15, 11v14, 12v13."""
        teams = self._make_acc_teams()
        bracket = ConferenceTournamentBracket("ACC", teams)
        r1_matchups = set()
        for game in bracket.games[0]:
            r1_matchups.add((game.team1.conf_seed, game.team2.conf_seed))
        assert r1_matchups == {(10, 15), (11, 14), (12, 13)}

    def test_acc_seeds_1_to_4_not_in_rounds_1_or_2(self):
        """Seeds 1-4 should not appear in any R1 or R2 game (double byes)."""
        teams = self._make_acc_teams()
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"ACC": teams},
        )
        bracket = predictor.predict_conference("ACC")

        top4_seeds = {1, 2, 3, 4}
        for round_idx in range(2):  # R1 and R2
            for game in bracket.games[round_idx]:
                if game.team1:
                    assert game.team1.conf_seed not in top4_seeds, (
                        f"Seed {game.team1.conf_seed} found in round {round_idx + 1}"
                    )
                if game.team2:
                    assert game.team2.conf_seed not in top4_seeds, (
                        f"Seed {game.team2.conf_seed} found in round {round_idx + 1}"
                    )

    def test_acc_seeds_5_to_9_enter_in_round_2(self):
        """Seeds 5-9 should first appear in round 2 (single byes)."""
        teams = self._make_acc_teams()
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"ACC": teams},
        )
        bracket = predictor.predict_conference("ACC")

        r2_seeds = set()
        for game in bracket.games[1]:
            if game.team1:
                r2_seeds.add(game.team1.conf_seed)
            if game.team2:
                r2_seeds.add(game.team2.conf_seed)

        # R2 should include seeds 5-9 (new entrants) plus R1 winners
        for seed in [5, 6, 7, 8, 9]:
            assert seed in r2_seeds, f"Seed {seed} not found in round 2"

    def test_acc_round2_has_4_games(self):
        """ACC R2 should have exactly 4 games."""
        teams = self._make_acc_teams()
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"ACC": teams},
        )
        bracket = predictor.predict_conference("ACC")
        assert len(bracket.games[1]) == 4

    def test_acc_quarterfinals_has_4_games(self):
        """ACC QF (R3) should have exactly 4 games."""
        teams = self._make_acc_teams()
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"ACC": teams},
        )
        bracket = predictor.predict_conference("ACC")
        assert len(bracket.games[2]) == 4

    def test_acc_champion_is_seed_1_deterministic(self):
        """With deterministic predictions, seed 1 should win the ACC."""
        teams = self._make_acc_teams()
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"ACC": teams},
        )
        bracket = predictor.predict_conference("ACC")
        assert bracket.champion is not None
        assert bracket.champion.conf_seed == 1

    def test_acc_seeds_1_and_2_meet_only_in_final(self):
        """Seeds 1 and 2 must not meet before the championship."""
        teams = self._make_acc_teams()
        predictor = ConferenceTournamentPredictor(
            teams_by_conference={"ACC": teams},
        )
        bracket = predictor.predict_conference("ACC")

        for round_games in bracket.games[:-1]:
            for game in round_games:
                if game.team1 and game.team2:
                    seeds = {game.team1.conf_seed, game.team2.conf_seed}
                    assert seeds != {1, 2}, (
                        f"Seeds 1 and 2 met before the final in round {game.round}"
                    )

    def test_acc_total_games_equals_n_minus_1(self):
        """Total games in ACC bracket should be 14 (15-1)."""
        teams = self._make_acc_teams()
        bracket = ConferenceTournamentBracket("ACC", teams)
        total_games = sum(len(rnd) for rnd in bracket.games)
        assert total_games == 14

    def test_acc_round_names(self):
        """ACC round names should be Play-in, First Round, QF, SF, Championship."""
        teams = self._make_acc_teams()
        bracket = ConferenceTournamentBracket("ACC", teams)
        round_names = [bracket.games[i][0].round_name for i in range(5)]
        assert round_names == [
            "Play-in", "First Round", "Quarterfinals",
            "Semifinals", "Championship",
        ]


class TestMultiLevelByes:
    """Test that multi-level byes work correctly for various sizes."""

    @staticmethod
    def _make_teams(n, conf="TEST"):
        return [
            ConferenceTeam(
                team_id=f"t{i}", name=f"T{i}", conf_seed=i,
                conference=conf, adj_em=30.0 - i,
                adj_o=110.0 - i * 0.3, adj_d=80.0 + i * 0.3,
            )
            for i in range(1, n + 1)
        ]

    def test_15_team_bracket_structure(self):
        """15-team bracket (ACC style) has correct round sizes: 3, 4, 4, 2, 1."""
        fmt = get_bracket_format("ACC", 15)
        teams = self._make_teams(15, "ACC")
        bracket = ConferenceTournamentBracket("ACC", teams, bracket_format=fmt)
        round_sizes = [len(rnd) for rnd in bracket.games]
        assert round_sizes == [3, 4, 4, 2, 1]

    def test_18_team_bracket_structure(self):
        """18-team bracket (B10 style) has correct round sizes: 2, 4, 4, 4, 2, 1 (6 rounds)."""
        fmt = get_bracket_format("B10", 18)
        teams = self._make_teams(18, "B10")
        bracket = ConferenceTournamentBracket("B10", teams, bracket_format=fmt)
        round_sizes = [len(rnd) for rnd in bracket.games]
        assert round_sizes == [2, 4, 4, 4, 2, 1]

    def test_11_team_bracket_structure(self):
        """11-team bracket (BE style) has correct round sizes: 3, 4, 2, 1."""
        fmt = get_bracket_format("BE", 11)
        teams = self._make_teams(11, "BE")
        bracket = ConferenceTournamentBracket("BE", teams, bracket_format=fmt)
        round_sizes = [len(rnd) for rnd in bracket.games]
        assert round_sizes == [3, 4, 2, 1]

    def test_12_team_bracket_structure(self):
        """12-team bracket has correct round sizes: 4, 4, 2, 1."""
        teams = self._make_teams(12, "MWC")
        bracket = ConferenceTournamentBracket("MWC", teams)
        round_sizes = [len(rnd) for rnd in bracket.games]
        assert round_sizes == [4, 4, 2, 1]
