"""Tests for src.pipeline.stages.game_utils — extracted utility functions."""

from datetime import date, datetime, timezone

import pytest

from src.pipeline.stages.game_utils import (
    coerce_game_date,
    compute_game_outcome,
    compute_game_sort_key,
    detect_tournament_game,
    game_outcome,
    is_target_season_game,
    is_tournament_game,
    normalize_key,
    normalize_team_key,
    parse_game_date,
    parse_timestamp,
    validate_source_coverage,
)


# --- normalize_team_key ---

def test_normalize_team_key_basic():
    assert normalize_team_key("  Duke  ") == "duke"

def test_normalize_team_key_removes_suffix():
    assert normalize_team_key("Michigan State") == "michigan"

def test_normalize_team_key_collapses_whitespace():
    assert normalize_team_key("North  Carolina") == "north carolina"


# --- normalize_key ---

def test_normalize_key_replaces_special():
    assert normalize_key("Texas A&M") == "texas_aandm"

def test_normalize_key_strips_underscores():
    assert normalize_key("_test_") == "test"


# --- is_tournament_game ---

def test_is_tournament_game_before():
    assert not is_tournament_game(date(2024, 3, 1), 2024)

def test_is_tournament_game_after():
    assert is_tournament_game(date(2024, 3, 20), 2024)

def test_is_tournament_game_unknown_year():
    # Defaults to March 15
    assert is_tournament_game(date(2030, 3, 16), 2030)


# --- detect_tournament_game ---

def test_detect_tournament_game_true():
    assert detect_tournament_game("2024-03-20", fallback_year=2024)

def test_detect_tournament_game_false():
    assert not detect_tournament_game("2024-02-15", fallback_year=2024)

def test_detect_tournament_game_unparseable():
    assert not detect_tournament_game("invalid", fallback_year=2024)


# --- parse_game_date ---

def test_parse_game_date_iso():
    assert parse_game_date("2024-03-15") == date(2024, 3, 15)

def test_parse_game_date_us():
    assert parse_game_date("03/15/2024") == date(2024, 3, 15)

def test_parse_game_date_invalid():
    assert parse_game_date("not-a-date") is None


# --- parse_timestamp ---

def test_parse_timestamp_iso():
    result = parse_timestamp("2024-03-15T10:30:00")
    assert result is not None
    assert result.year == 2024

def test_parse_timestamp_with_z():
    result = parse_timestamp("2024-03-15T10:30:00Z")
    assert result is not None

def test_parse_timestamp_empty():
    assert parse_timestamp("") is None

def test_parse_timestamp_none():
    assert parse_timestamp(None) is None


# --- game_outcome ---

class _MockGame:
    def __init__(self, t1=None, t2=None, lead_history=None):
        self.team1_score = t1
        self.team2_score = t2
        self.lead_history = lead_history

def test_game_outcome_by_score():
    assert game_outcome(_MockGame(t1=80, t2=70)) == 1
    assert game_outcome(_MockGame(t1=70, t2=80)) == 0

def test_game_outcome_by_lead_history():
    assert game_outcome(_MockGame(lead_history=[3, 5, 2])) == 1
    assert game_outcome(_MockGame(lead_history=[3, -1, -2])) == 0

def test_game_outcome_unknown():
    assert game_outcome(_MockGame()) is None


# --- compute_game_outcome ---

def test_compute_game_outcome_win():
    is_win, margin = compute_game_outcome(80, 70)
    assert is_win is True
    assert margin == 10

def test_compute_game_outcome_loss():
    is_win, margin = compute_game_outcome(65, 78)
    assert is_win is False
    assert margin == -13


# --- coerce_game_date ---

def test_coerce_game_date_iso():
    assert coerce_game_date("2024-03-15") == "2024-03-15"

def test_coerce_game_date_with_time():
    assert coerce_game_date("2024-03-15T10:30:00") == "2024-03-15"

def test_coerce_game_date_empty_fallback():
    assert coerce_game_date("", fallback_year=2024) == "2024-01-01"

def test_coerce_game_date_none_fallback():
    assert coerce_game_date(None, fallback_year=2024) == "2024-01-01"


# --- compute_game_sort_key ---

def test_compute_game_sort_key_normal():
    assert compute_game_sort_key("2024-03-15", fallback_year=2024) == 20240315

def test_compute_game_sort_key_empty():
    assert compute_game_sort_key("", fallback_year=2024) == 20240101


# --- is_target_season_game ---

def test_is_target_season_in():
    assert is_target_season_game("2024-02-15", 2024)

def test_is_target_season_out():
    assert not is_target_season_game("2024-06-15", 2024)

def test_is_target_season_prior_year():
    assert is_target_season_game("2023-11-15", 2024)


# --- validate_source_coverage ---

def test_validate_source_coverage_ok():
    validate_source_coverage("test", {"a": 1, "b": 2}, 3, 0.5)

def test_validate_source_coverage_too_low():
    with pytest.raises(ValueError, match="too low"):
        validate_source_coverage("test", {"a": 1}, 10, 0.5)

def test_validate_source_coverage_no_teams():
    with pytest.raises(ValueError, match="No tournament teams"):
        validate_source_coverage("test", {}, 0, 0.5)
