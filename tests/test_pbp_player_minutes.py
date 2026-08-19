"""Tests for player minutes reconstructed from PBP substitution events."""

from src.data.features.pbp_player_minutes import (
    derive_game_player_minutes,
    elapsed_seconds,
    period_length_seconds,
    validate_game_minutes,
)

TEAM = "tcu_horned_frogs"
OPP = "lsu_new_orleans_privateers"


def _sub(athlete_id, name, period, secs_left, direction, team=TEAM):
    return {
        "period": period,
        "seconds_remaining": float(secs_left),
        "home_score": 0,
        "away_score": 0,
        "play_type": "Substitution",
        "text": f"{name} subbing {direction} for Team",
        "athlete_id": athlete_id,
        "athlete_name": name,
        "athlete_team": team,
    }


def _action(athlete_id, name, period, secs_left, team=TEAM):
    return {
        "period": period,
        "seconds_remaining": float(secs_left),
        "home_score": 0,
        "away_score": 0,
        "play_type": "JumpShot",
        "text": f"{name} made Jumper.",
        "athlete_id": athlete_id,
        "athlete_name": name,
        "athlete_team": team,
    }


def _game(plays):
    return {
        "game_id": "g1",
        "home_team_raw": TEAM,
        "away_team_raw": OPP,
        "plays": plays,
    }


class TestClockMath:
    def test_regulation_and_overtime_period_lengths(self):
        assert period_length_seconds(1) == 1200
        assert period_length_seconds(2) == 1200
        assert period_length_seconds(3) == 300  # OT

    def test_elapsed_accumulates_prior_periods(self):
        # 19:39 left in the 1st == 21s elapsed
        assert elapsed_seconds(1, 1179) == 21
        # start of 2nd half == 1200s elapsed
        assert elapsed_seconds(2, 1200) == 1200
        # 2:00 left in first OT == 40 min + 3 min
        assert elapsed_seconds(3, 120) == 2400 + 180


class TestStarterInference:
    def test_player_whose_first_event_is_sub_out_is_a_starter(self):
        plays = [_sub("p1", "Starter One", 1, 900, "out")]
        players = derive_game_player_minutes(_game(plays))
        p = next(p for p in players if p.athlete_id == "p1")
        assert p.started is True
        # On court from tip (0s) until 5:00 elapsed.
        assert p.seconds == 300

    def test_player_with_action_before_first_sub_in_is_a_starter(self):
        plays = [
            _action("p1", "Starter One", 1, 1100),
            _sub("p1", "Starter One", 1, 900, "out"),
        ]
        p = next(p for p in derive_game_player_minutes(_game(plays)) if p.athlete_id == "p1")
        assert p.started is True

    def test_player_whose_first_event_is_sub_in_is_a_bench_player(self):
        plays = [
            _sub("p2", "Bench Two", 1, 900, "in"),
            _sub("p2", "Bench Two", 1, 600, "out"),
        ]
        p = next(p for p in derive_game_player_minutes(_game(plays)) if p.athlete_id == "p2")
        assert p.started is False
        assert p.seconds == 300  # 15:00 -> 10:00 remaining


class TestMinutesAccumulation:
    def test_multiple_stints_sum(self):
        plays = [
            _sub("p2", "Bench", 1, 1000, "in"),
            _sub("p2", "Bench", 1, 800, "out"),  # 200s
            _sub("p2", "Bench", 2, 600, "in"),
            _sub("p2", "Bench", 2, 300, "out"),  # 300s
        ]
        p = next(p for p in derive_game_player_minutes(_game(plays)) if p.athlete_id == "p2")
        assert p.seconds == 500

    def test_player_never_subbed_out_counts_through_end_of_game(self):
        plays = [
            _sub("p1", "Iron Man", 1, 1200, "out"),  # marks him a starter
            _sub("p1", "Iron Man", 1, 1200, "in"),
            _action("p9", "Someone", 2, 0),  # establishes game length
        ]
        p = next(p for p in derive_game_player_minutes(_game(plays)) if p.athlete_id == "p1")
        # In at tip, never out -> full 40 minutes.
        assert p.seconds == 2400

    def test_overtime_extends_game_length(self):
        plays = [
            _sub("p1", "OT Player", 1, 1200, "out"),
            _sub("p1", "OT Player", 1, 1200, "in"),
            _action("p9", "Someone", 3, 0),  # game reached OT
        ]
        p = next(p for p in derive_game_player_minutes(_game(plays)) if p.athlete_id == "p1")
        assert p.seconds == 2700  # 40 min + 5 min OT


class TestValidation:
    def test_complete_feed_passes_budget_check(self):
        # 5 starters each playing the full 40 minutes == exactly 200 team minutes.
        plays = []
        for i in range(5):
            plays.append(_sub(f"p{i}", f"P{i}", 1, 1200, "out"))
            plays.append(_sub(f"p{i}", f"P{i}", 1, 1200, "in"))
        plays.append(_action("p0", "P0", 2, 0))
        players = derive_game_player_minutes(_game(plays))
        assert validate_game_minutes(_game(plays), players)[TEAM] is True

    def test_missing_substitution_feed_fails_budget_check(self):
        # Only one player's worth of minutes for a whole team -> implausible.
        plays = [
            _sub("p1", "Lonely", 1, 1200, "out"),
            _sub("p1", "Lonely", 1, 1200, "in"),
            _action("p1", "Lonely", 2, 0),
        ]
        players = derive_game_player_minutes(_game(plays))
        assert validate_game_minutes(_game(plays), players)[TEAM] is False
