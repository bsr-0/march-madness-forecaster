"""Tests for conference tournament performance features."""

import numpy as np
import pytest

from src.data.features.proprietary_metrics import (
    GameRecord,
    IncrementalMetricsEngine,
    ProprietaryTeamMetrics,
)
from src.data.features.feature_engineering import (
    TEAM_FEATURE_DIM,
    TeamFeatures,
)


def _make_game(
    game_id: str,
    game_date: str,
    team_id: str,
    opponent_id: str,
    points: int,
    opp_points: int,
) -> GameRecord:
    return GameRecord(
        game_id=game_id,
        game_date=game_date,
        team_id=team_id,
        team_name=team_id,
        opponent_id=opponent_id,
        points=points,
        opp_points=opp_points,
        possessions=70,
    )


class TestComputeConfTourneyFeatures:
    """Test the date-window heuristic for conference tournament features."""

    def _make_conf_map(self):
        return {
            "duke": "ACC",
            "unc": "ACC",
            "nc_state": "ACC",
            "kansas": "Big 12",
            "baylor": "Big 12",
        }

    def test_same_conference_in_window(self):
        """Games between same-conference teams in the conf tourney window are counted."""
        # 2024 tournament starts March 19. Window: March 7-18.
        results = {
            "duke": ProprietaryTeamMetrics(team_id="", team_name=""),
            "unc": ProprietaryTeamMetrics(team_id="", team_name=""),
        }
        by_team = {
            "duke": [
                _make_game("g1", "2024-03-10", "duke", "unc", 75, 70),
                _make_game("g2", "2024-03-12", "duke", "nc_state", 80, 65),
            ],
            "unc": [
                _make_game("g1r", "2024-03-10", "unc", "duke", 70, 75),
            ],
        }
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-19"
        )

        assert results["duke"].conf_tourney_games == 2
        assert results["duke"].conf_tourney_margin == pytest.approx((5 + 15) / 2)
        assert results["unc"].conf_tourney_games == 1
        assert results["unc"].conf_tourney_margin == pytest.approx(-5.0)

    def test_cross_conference_not_counted(self):
        """Games against teams from other conferences are not counted."""
        results = {
            "duke": ProprietaryTeamMetrics(team_id="", team_name=""),
            "kansas": ProprietaryTeamMetrics(team_id="", team_name=""),
        }
        by_team = {
            "duke": [
                _make_game("g1", "2024-03-10", "duke", "kansas", 80, 75),
            ],
            "kansas": [
                _make_game("g1r", "2024-03-10", "kansas", "duke", 75, 80),
            ],
        }
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-19"
        )

        assert results["duke"].conf_tourney_games == 0
        assert results["duke"].conf_tourney_margin == 0.0
        assert results["kansas"].conf_tourney_games == 0

    def test_outside_window_not_counted(self):
        """Games before the 12-day window are not counted."""
        results = {"duke": ProprietaryTeamMetrics(team_id="", team_name="")}
        by_team = {
            "duke": [
                _make_game("g1", "2024-03-01", "duke", "unc", 75, 70),  # before window
                _make_game("g2", "2024-03-20", "duke", "unc", 80, 70),  # after window
            ],
        }
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-19"
        )

        assert results["duke"].conf_tourney_games == 0

    def test_as_of_before_window(self):
        """If as_of_date is before window start, all features stay at defaults."""
        results = {"duke": ProprietaryTeamMetrics(team_id="", team_name="")}
        by_team = {
            "duke": [
                _make_game("g1", "2024-03-10", "duke", "unc", 75, 70),
            ],
        }
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-01"  # before window
        )

        assert results["duke"].conf_tourney_games == 0
        assert results["duke"].conf_tourney_margin == 0.0

    def test_games_capped_at_5(self):
        """conf_tourney_games is capped at 5."""
        results = {"duke": ProprietaryTeamMetrics(team_id="", team_name="")}
        games = [
            _make_game(f"g{i}", f"2024-03-{10+i}", "duke", "unc", 75, 70)
            for i in range(7)
        ]
        by_team = {"duke": games}
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-19"
        )

        assert results["duke"].conf_tourney_games == 5

    def test_late_season_includes_all_games(self):
        """Late-season features count all games in window, not just same-conference."""
        results = {
            "duke": ProprietaryTeamMetrics(team_id="duke", team_name="Duke"),
        }
        by_team = {
            "duke": [
                _make_game("g1", "2024-03-10", "duke", "unc", 75, 70),       # same conf
                _make_game("g2", "2024-03-12", "duke", "kansas", 80, 60),     # cross conf
                _make_game("g3", "2024-03-14", "duke", "baylor", 65, 70),     # cross conf, loss
            ],
        }
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-19"
        )

        # Conf tourney: only same-conference (duke vs unc)
        assert results["duke"].conf_tourney_games == 1
        assert results["duke"].conf_tourney_margin == pytest.approx(5.0)

        # Late season: all 3 games
        assert results["duke"].late_season_games == 3
        assert results["duke"].late_season_margin == pytest.approx((5 + 20 - 5) / 3)
        assert results["duke"].late_season_win_pct == pytest.approx(2 / 3)

    def test_respects_as_of_date_within_window(self):
        """Games on or after as_of_date within the window are excluded."""
        results = {"duke": ProprietaryTeamMetrics(team_id="", team_name="")}
        by_team = {
            "duke": [
                _make_game("g1", "2024-03-10", "duke", "unc", 75, 70),
                _make_game("g2", "2024-03-15", "duke", "unc", 80, 65),
            ],
        }
        conf_map = self._make_conf_map()

        IncrementalMetricsEngine._compute_conf_tourney_features(
            results, by_team, conf_map, "2024-03-15"  # only g1 should count
        )

        assert results["duke"].conf_tourney_games == 1
        assert results["duke"].conf_tourney_margin == pytest.approx(5.0)


class TestFeatureVectorDimensions:
    """Verify the 6 new features are wired into the vectors."""

    def test_team_feature_dim_is_52(self):
        assert TEAM_FEATURE_DIM == 56

    def test_to_vector_length(self):
        tf = TeamFeatures(team_id="duke", team_name="Duke", seed=1, region="East")
        v = tf.to_vector()
        assert len(v) == 56

    def test_conf_tourney_indices(self):
        tf = TeamFeatures(
            team_id="duke", team_name="Duke", seed=1, region="East",
            conf_tourney_champion=1.0,
            conf_tourney_games=3.0,
            conf_tourney_margin=7.5,
            late_season_games=5.0,
            late_season_margin=10.0,
            late_season_win_pct=0.8,
        )
        v = tf.to_vector()
        assert v[46] == 1.0
        assert v[47] == 3.0
        assert v[48] == pytest.approx(7.5)
        assert v[49] == 5.0
        assert v[50] == pytest.approx(10.0)
        assert v[51] == pytest.approx(0.8)

    def test_get_feature_names_length(self):
        names = TeamFeatures.get_feature_names()
        assert len(names) == 56
        assert names[46] == 'conf_tourney_champion'
        assert names[47] == 'conf_tourney_games'
        assert names[48] == 'conf_tourney_margin'
        assert names[49] == 'late_season_games'
        assert names[50] == 'late_season_margin'
        assert names[51] == 'late_season_win_pct'

    def test_metrics_to_team_vector_maps_all_recency(self):
        m = ProprietaryTeamMetrics(team_id="duke", team_name="Duke")
        m.conf_tourney_champion = True
        m.conf_tourney_games = 4
        m.conf_tourney_margin = 12.5
        m.late_season_games = 6
        m.late_season_margin = 8.3
        m.late_season_win_pct = 0.833

        v = IncrementalMetricsEngine.metrics_to_team_vector(m, seed=1)
        assert v[46] == 1.0
        assert v[47] == 4.0
        assert v[48] == pytest.approx(12.5)
        assert v[49] == 6.0
        assert v[50] == pytest.approx(8.3)
        assert v[51] == pytest.approx(0.833)
        assert len(v) == 56
