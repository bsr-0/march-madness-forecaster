"""Tests for box-score stats derived from play-by-play.

Fixtures mirror the *confirmed live* ESPN play shape (2026 season, game
401808890) — in particular the counterintuitive cases that would silently
corrupt shooting splits if mis-parsed. See pbp_box_scores.py's docstring.
"""

from src.data.features.pbp_box_scores import (
    aggregate_team_season_box,
    derive_game_box_scores,
)

HOME = "tcu_horned_frogs"
AWAY = "lsu_new_orleans_privateers"


def _play(**kw):
    base = {
        "period": 1,
        "seconds_remaining": 1000.0,
        "home_score": 0,
        "away_score": 0,
        "home_away": "home",
        "scoring_play": False,
        "shooting_play": False,
        "points_attempted": None,
        "play_type": None,
        "text": "",
        "athlete_team": HOME,
    }
    base.update(kw)
    return base


def _game(plays):
    return {
        "game_id": "g1",
        "game_date": "2025-11-04",
        "home_team_raw": HOME,
        "away_team_raw": AWAY,
        "plays": plays,
    }


class TestFreeThrowParsing:
    def test_missed_free_throw_is_not_counted_as_made(self):
        # ESPN labels BOTH made and missed FTs 'MadeFreeThrow' -- only
        # scoring_play distinguishes them. Trusting play_type would give 100% FT.
        plays = [
            _play(
                play_type="MadeFreeThrow",
                shooting_play=True,
                points_attempted=1,
                scoring_play=True,
                text="A made Free Throw.",
            ),
            _play(
                play_type="MadeFreeThrow",
                shooting_play=True,
                points_attempted=1,
                scoring_play=False,
                text="A missed Free Throw.",
            ),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert box.fta == 2
        assert box.ftm == 1

    def test_free_throws_do_not_inflate_field_goal_attempts(self):
        # shooting_play is True for FTs too; FGA must filter on points_attempted.
        plays = [
            _play(
                play_type="MadeFreeThrow",
                shooting_play=True,
                points_attempted=1,
                scoring_play=True,
            ),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert box.fga == 0
        assert box.fta == 1


class TestFieldGoalParsing:
    def test_three_pointer_counts_toward_both_fg_and_fg3(self):
        plays = [
            _play(
                play_type="JumpShot",
                shooting_play=True,
                points_attempted=3,
                scoring_play=True,
            ),
            _play(
                play_type="JumpShot",
                shooting_play=True,
                points_attempted=3,
                scoring_play=False,
            ),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert (box.fgm, box.fga) == (1, 2)
        assert (box.fg3m, box.fg3a) == (1, 2)

    def test_points_derive_from_shot_counts(self):
        plays = [
            _play(play_type="JumpShot", shooting_play=True, points_attempted=3, scoring_play=True),
            _play(play_type="LayUpShot", shooting_play=True, points_attempted=2, scoring_play=True),
            _play(
                play_type="MadeFreeThrow",
                shooting_play=True,
                points_attempted=1,
                scoring_play=True,
            ),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert box.pts == 3 + 2 + 1


class TestTeamAttribution:
    def test_steal_without_athlete_team_falls_back_to_home_away(self):
        # Confirmed live: every Steal play has athlete_team=None but a
        # populated home_away naming the STEALING team.
        plays = [
            _play(play_type="Steal", athlete_team=None, home_away="away", text="X Steal."),
        ]
        boxes = derive_game_box_scores(_game(plays))
        assert boxes[AWAY].stl == 1
        assert boxes[HOME].stl == 0

    def test_play_with_neither_attribution_is_skipped(self):
        plays = [_play(play_type="End Game", athlete_team=None, home_away=None)]
        boxes = derive_game_box_scores(_game(plays))
        assert boxes[HOME].pts == 0 and boxes[AWAY].pts == 0


class TestReboundsAndCounting:
    def test_dead_ball_rebound_tracked_apart_from_player_rebounds(self):
        plays = [
            _play(play_type="Offensive Rebound"),
            _play(play_type="Defensive Rebound"),
            _play(play_type="Dead Ball Rebound", athlete_team=None, home_away="home"),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert (box.orb, box.drb, box.team_reb) == (1, 1, 1)

    def test_assists_counted_from_text_marker(self):
        plays = [
            _play(
                play_type="LayUpShot",
                shooting_play=True,
                points_attempted=2,
                scoring_play=True,
                text="A made Layup. Assisted by B.",
            ),
            _play(
                play_type="LayUpShot",
                shooting_play=True,
                points_attempted=2,
                scoring_play=True,
                text="A made Layup.",
            ),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert box.ast == 1

    def test_turnovers_and_fouls_matched_by_substring(self):
        plays = [
            _play(play_type="Lost Ball Turnover"),
            _play(play_type="PersonalFoul"),
            _play(play_type="Technical Foul"),
        ]
        box = derive_game_box_scores(_game(plays))[HOME]
        assert box.tov == 1
        assert box.pf == 2


class TestSeasonAggregation:
    def test_shooting_rates_use_season_totals_not_per_game_means(self):
        from src.data.features.pbp_box_scores import TeamGameBox

        boxes = [
            TeamGameBox(
                game_id="g1", team_id="t", opponent_id="o", is_home=True, fg3m=1, fg3a=10, ftm=1, fta=1, fgm=1, fga=10
            ),
            TeamGameBox(
                game_id="g2", team_id="t", opponent_id="o", is_home=True, fg3m=9, fg3a=10, ftm=0, fta=1, fgm=9, fga=10
            ),
        ]
        agg = aggregate_team_season_box(boxes)["t"]
        # Pooled: 10/20 = 0.5 (not the mean of 0.1 and 0.9, which coincides
        # here -- ft_pct is the discriminating case: 1/2, not mean(1.0, 0.0)).
        assert agg["three_pt_pct"] == 0.5
        assert agg["ft_pct"] == 0.5
        assert agg["games_with_box_data"] == 2

    def test_zero_attempts_yields_none_not_zero(self):
        from src.data.features.pbp_box_scores import TeamGameBox

        boxes = [TeamGameBox(game_id="g1", team_id="t", opponent_id="o", is_home=True)]
        agg = aggregate_team_season_box(boxes)["t"]
        assert agg["three_pt_pct"] is None
        assert agg["ft_pct"] is None
