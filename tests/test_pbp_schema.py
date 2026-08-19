"""Tests for the play-by-play payload schema validation."""

from src.data.scrapers.schemas import validate_pbp_payload


def _valid_play(**overrides):
    play = {
        "game_id": "g1",
        "period": 2,
        "seconds_remaining": 120.0,
        "home_score": 55,
        "away_score": 50,
    }
    play.update(overrides)
    return play


def test_valid_payload_passes_through():
    payload = {
        "season": 2024,
        "games": [
            {
                "game_id": "g1",
                "game_date": "2024-02-10",
                "home_team_raw": "duke_blue_devils",
                "away_team_raw": "unc_tar_heels",
                "plays": [_valid_play()],
            }
        ],
    }
    result = validate_pbp_payload(payload)
    assert len(result["games"]) == 1
    assert len(result["games"][0]["plays"]) == 1


def test_negative_score_play_is_dropped_not_fatal():
    payload = {
        "season": 2024,
        "games": [
            {
                "game_id": "g1",
                "plays": [_valid_play(), _valid_play(home_score=-5)],
            }
        ],
    }
    result = validate_pbp_payload(payload)
    assert len(result["games"][0]["plays"]) == 1


def test_game_with_no_valid_plays_is_dropped():
    payload = {
        "season": 2024,
        "games": [
            {"game_id": "g1", "plays": [_valid_play(home_score=-1)]},
            {"game_id": "g2", "plays": [_valid_play()]},
        ],
    }
    result = validate_pbp_payload(payload)
    assert len(result["games"]) == 1
    assert result["games"][0]["game_id"] == "g2"


def test_extra_unmapped_columns_are_kept():
    payload = {
        "season": 2024,
        "games": [
            {
                "game_id": "g1",
                "plays": [_valid_play(play_desc="Jones makes 3-pt jump shot")],
            }
        ],
    }
    result = validate_pbp_payload(payload)
    assert result["games"][0]["plays"][0]["play_desc"] == "Jones makes 3-pt jump shot"


def test_empty_payload_returned_as_is():
    assert validate_pbp_payload({}) == {}
    assert validate_pbp_payload({"season": 2024, "games": []}) == {"season": 2024, "games": []}
