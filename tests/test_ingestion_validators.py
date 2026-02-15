"""Unit tests for ingestion payload validators."""

from src.data.ingestion.validators import (
    validate_games_payload,
    validate_odds_payload,
    validate_public_picks_payload,
    validate_ratings_payload,
    validate_rosters_payload,
    validate_shotquality_games_payload,
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


def test_validate_ratings_payload_rejects_missing_required_numeric():
    payload = {"teams": [{"team_id": "duke", "name": "Duke"}]}
    errors = validate_ratings_payload(payload, required_numeric_fields=["adj_offensive_efficiency"])
    assert errors
    assert "missing/invalid numeric field" in errors[0]


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


def test_validate_rosters_payload_accepts_rapm_inputs():
    payload = {
        "teams": [
            {
                "team_id": "duke",
                "players": [
                    {"player_id": "p1", "name": "A", "rapm_offensive": 1.1, "rapm_defensive": 0.5}
                ],
            }
        ]
    }
    assert validate_rosters_payload(payload) == []


def test_validate_rosters_payload_rejects_all_zero_rapm():
    payload = {
        "teams": [
            {
                "team_id": "duke",
                "players": [
                    {"player_id": "p1", "name": "A", "rapm_offensive": 0.0, "rapm_defensive": 0.0},
                    {"player_id": "p2", "name": "B", "rapm_offensive": 0.0, "rapm_defensive": 0.0},
                ],
            }
        ]
    }
    errors = validate_rosters_payload(payload)
    assert errors
    assert "all RAPM values are zero" in errors[0]


def test_validate_shotquality_games_payload_checks_xp_coverage():
    payload = {
        "games": [
            {
                "game_id": "g1",
                "team_id": "duke",
                "opponent_id": "unc",
                "possessions": [
                    {"team_id": "duke", "xp": 1.1},
                    {"team_id": "unc"},
                ],
            }
        ]
    }
    errors = validate_shotquality_games_payload(payload, min_xp_coverage=0.8)
    assert errors
    assert "coverage too low" in errors[-1]


def test_validate_odds_payload_requires_probability_or_odds():
    payload = {"teams": [{"team_name": "Duke"}]}
    errors = validate_odds_payload(payload)
    assert errors
    assert "implied_win_probability/title_odds" in errors[0]
