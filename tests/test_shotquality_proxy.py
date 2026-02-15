"""Tests for open-data ShotQuality proxy construction."""

from src.data.scrapers.shotquality_proxy import OpenShotQualityProxyBuilder


def test_proxy_builder_creates_games_and_team_metrics():
    rows = [
        {
            "game_id": "g1",
            "team_id": "duke",
            "team_name": "Duke",
            "opponent_id": "unc",
            "opponent_name": "UNC",
            "team_score": 78,
            "opponent_score": 70,
            "possessions": 69,
            "fga": 56,
            "fg3a": 20,
            "fta": 18,
            "turnovers": 9,
            "orb": 10,
            "date": "2026-01-10",
        },
        {
            "game_id": "g1",
            "team_id": "unc",
            "team_name": "UNC",
            "opponent_id": "duke",
            "opponent_name": "Duke",
            "team_score": 70,
            "opponent_score": 78,
            "possessions": 69,
            "fga": 58,
            "fg3a": 22,
            "fta": 14,
            "turnovers": 11,
            "orb": 9,
            "date": "2026-01-10",
        },
    ]
    payload = OpenShotQualityProxyBuilder().build(rows)
    assert payload is not None
    assert len(payload["games"]) == 1
    assert len(payload["teams"]) == 2

    game = payload["games"][0]
    possessions = game["possessions"]
    assert possessions
    assert all("xp" in p for p in possessions)
    assert all(p.get("team_id") in {"duke", "unc"} for p in possessions)

    team = next(t for t in payload["teams"] if t["team_id"] == "duke")
    assert team["offensive_xp_per_possession"] > 0
    assert 0 <= team["three_rate"] <= 1
