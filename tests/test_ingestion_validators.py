"""Unit tests for ingestion payload validators."""

from src.data.ingestion.validators import (
    validate_games_payload,
    validate_public_picks_payload,
    validate_ratings_payload,
    validate_teams_payload,
    validate_transfer_payload,
)


def test_validate_teams_payload_ok():
    payload = {"teams": [{"name": "Duke", "seed": 1, "region": "East"}]}
    assert validate_teams_payload(payload) == []


def test_validate_teams_payload_missing_field():
    payload = {"teams": [{"name": "Duke", "seed": 1}]}
    errors = validate_teams_payload(payload)
    assert errors
    assert "missing fields" in errors[0]


def test_validate_ratings_payload_ok():
    payload = {"teams": [{"team_id": "duke", "name": "Duke"}]}
    assert validate_ratings_payload(payload) == []


def test_validate_games_payload_ok():
    payload = {"games": [{"game_id": "g1", "team1_id": "duke", "team2_id": "unc"}]}
    assert validate_games_payload(payload) == []


def test_validate_public_picks_payload_ok():
    payload = {
        "teams": {
            "duke": {
                "team_name": "Duke",
                "seed": 1,
                "region": "East",
                "round_of_64_pct": 99,
                "round_of_32_pct": 90,
                "sweet_16_pct": 70,
                "elite_8_pct": 45,
                "final_four_pct": 25,
                "champion_pct": 12,
            }
        }
    }
    assert validate_public_picks_payload(payload) == []


def test_validate_transfer_payload_requires_player_identity():
    payload = {"entries": [{"destination_team_name": "Duke"}]}
    errors = validate_transfer_payload(payload)
    assert errors
    assert "player id/name" in errors[0]
