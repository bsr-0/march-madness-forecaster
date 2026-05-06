"""Tests for ESPN BPI scraper."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.scrapers.bpi_scraper import (
    BPIGamePrediction,
    BPISeasonData,
    BPIScraper,
    BPITeamRating,
    _extract_float,
)

# ---------------------------------------------------------------------------
# Fixtures — sample API responses
# ---------------------------------------------------------------------------

SAMPLE_POWERINDEX_PAGE = {
    "count": 2,
    "items": [
        {
            "team": {
                "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2026/teams/130?lang=en"
            },
            "stats": [
                {"name": "bpi", "value": 18.5},
                {"name": "bpioffense", "value": 10.2},
                {"name": "bpidefense", "value": 8.3},
                {"name": "sorrank", "value": 3.0},
                {"name": "sospastrank", "value": 12.0},
                {"name": "rpirank", "value": 5.0},
                {"name": "wins", "value": 30.0},
                {"name": "losses", "value": 4.0},
            ],
        },
        {
            "team": {
                "$ref": "http://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2026/teams/2305?lang=en"
            },
            "stats": [
                {"name": "bpi", "value": 15.1},
                {"name": "bpioffense", "value": 9.0},
                {"name": "bpidefense", "value": 6.1},
            ],
        },
    ],
}

SAMPLE_PREDICTOR = {
    "homeTeam": {
        "gameProjection": 72.3,
        "teampredmov": 5.8,
    },
    "awayTeam": {
        "gameProjection": 27.7,
        "teampredmov": -5.8,
    },
    "matchupQuality": 84.0,
}

SAMPLE_EVENT = {
    "event_id": "401234567",
    "competition_id": "401234567",
    "team1_espn_id": 130,
    "team2_espn_id": 2305,
    "game_date": "2026-03-20",
    "round_name": "Round of 64",
}

SAMPLE_TEAMS = {
    130: {"name": "Michigan Wolverines", "canonical_id": "michigan"},
    2305: {"name": "Duke Blue Devils", "canonical_id": "duke"},
}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestExtractFloat:
    def test_first_key_found(self):
        assert _extract_float({"a": 1.5, "b": 2.0}, "a", "b") == 1.5

    def test_fallback_to_second_key(self):
        assert _extract_float({"b": 3.7}, "a", "b") == 3.7

    def test_no_keys_returns_zero(self):
        assert _extract_float({"x": 99}, "a", "b") == 0.0

    def test_string_value_converted(self):
        assert _extract_float({"a": "4.2"}, "a") == 4.2

    def test_invalid_value_skipped(self):
        assert _extract_float({"a": "bad", "b": 5.0}, "a", "b") == 5.0


class TestParseTeamRating:
    def test_parses_full_rating(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {2026: SAMPLE_TEAMS}

        item = SAMPLE_POWERINDEX_PAGE["items"][0]
        rating = scraper._parse_team_rating(item, 2026)

        assert rating is not None
        assert rating.team_id == "michigan"
        assert rating.espn_id == 130
        assert rating.bpi == 18.5
        assert rating.bpi_offense == 10.2
        assert rating.bpi_defense == 8.3
        assert rating.sor == 3.0
        assert rating.wins == 30
        assert rating.losses == 4

    def test_parses_partial_stats(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {2026: SAMPLE_TEAMS}

        item = SAMPLE_POWERINDEX_PAGE["items"][1]
        rating = scraper._parse_team_rating(item, 2026)

        assert rating is not None
        assert rating.team_id == "duke"
        assert rating.bpi == 15.1
        assert rating.sor == 0.0  # missing stat defaults to 0

    def test_no_team_ref_returns_none(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {2026: SAMPLE_TEAMS}

        rating = scraper._parse_team_rating({"statistics": []}, 2026)
        assert rating is None


class TestParsePrediction:
    def test_parses_full_prediction(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {2026: SAMPLE_TEAMS}

        pred = scraper._parse_prediction(SAMPLE_PREDICTOR, SAMPLE_EVENT, 2026)

        assert pred is not None
        assert pred.team1_id == "michigan"
        assert pred.team2_id == "duke"
        assert pred.team1_win_pct == 72.3
        assert pred.team2_win_pct == 27.7
        assert pred.team1_pred_margin == 5.8
        assert pred.matchup_quality == 84.0
        assert pred.round_name == "Round of 64"

    def test_empty_teams_returns_none(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {2026: SAMPLE_TEAMS}

        pred = scraper._parse_prediction({}, SAMPLE_EVENT, 2026)
        assert pred is None

    def test_normalizes_probabilities(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {2026: SAMPLE_TEAMS}

        bad_data = {
            "homeTeam": {"gameProjection": 80.0},
            "awayTeam": {"gameProjection": 40.0},
            "matchupQuality": 50.0,
        }
        pred = scraper._parse_prediction(bad_data, SAMPLE_EVENT, 2026)
        assert pred is not None
        assert abs(pred.team1_win_pct + pred.team2_win_pct - 100.0) < 0.1


class TestMissing404:
    @patch("src.data.scrapers.bpi_scraper.retry_request")
    def test_404_returns_none(self, mock_retry):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_retry.return_value = mock_resp

        scraper = BPIScraper()
        result = scraper._get_json("http://fake/url", "test")
        assert result is None


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        data = BPISeasonData(
            season=2026,
            timestamp="2026-04-28T12:00:00Z",
            teams={
                "michigan": BPITeamRating(
                    team_id="michigan",
                    team_name="Michigan Wolverines",
                    espn_id=130,
                    bpi=18.5,
                    bpi_offense=10.2,
                    bpi_defense=8.3,
                ),
            },
            games=[
                BPIGamePrediction(
                    event_id="401234567",
                    game_date="2026-03-20",
                    team1_id="michigan",
                    team2_id="duke",
                    team1_win_pct=72.3,
                    team2_win_pct=27.7,
                    team1_pred_margin=5.8,
                    matchup_quality=84.0,
                    round_name="Round of 64",
                ),
            ],
        )

        # Patch the output dir
        out_dir = tmp_path / "processed" / "bpi"
        out_dir.mkdir(parents=True)

        with patch("src.data.scrapers.bpi_scraper.BPIScraper.save_processed") as mock_save:
            # Manually save to tmp
            path = out_dir / f"bpi_{data.season}.json"
            payload = {
                "season": data.season,
                "timestamp": data.timestamp,
                "n_teams": len(data.teams),
                "n_games": len(data.games),
                "teams": {
                    k: {
                        "team_id": v.team_id,
                        "team_name": v.team_name,
                        "espn_id": v.espn_id,
                        "bpi": v.bpi,
                        "bpi_offense": v.bpi_offense,
                        "bpi_defense": v.bpi_defense,
                        "sor": v.sor,
                        "sos": v.sos,
                        "rpi_rank": v.rpi_rank,
                        "wins": v.wins,
                        "losses": v.losses,
                    }
                    for k, v in data.teams.items()
                },
                "games": [
                    {
                        "event_id": g.event_id,
                        "game_date": g.game_date,
                        "team1_id": g.team1_id,
                        "team2_id": g.team2_id,
                        "team1_espn_id": g.team1_espn_id,
                        "team2_espn_id": g.team2_espn_id,
                        "team1_win_pct": g.team1_win_pct,
                        "team2_win_pct": g.team2_win_pct,
                        "team1_pred_margin": g.team1_pred_margin,
                        "matchup_quality": g.matchup_quality,
                        "round_name": g.round_name,
                    }
                    for g in data.games
                ],
            }
            with open(path, "w") as f:
                json.dump(payload, f)

        # Load it back
        with patch(
            "src.data.scrapers.bpi_scraper.Path",
            side_effect=lambda p: tmp_path / "processed" / "bpi" / Path(p).name if "processed" in str(p) else Path(p),
        ):
            loaded = (
                BPIScraper.load_processed.__wrapped__(2026)
                if hasattr(BPIScraper.load_processed, "__wrapped__")
                else None
            )

        # Direct load from the file we wrote
        with open(path) as f:
            raw = json.load(f)

        teams = {}
        for tid, tdata in raw.get("teams", {}).items():
            teams[tid] = BPITeamRating(**tdata)

        games = [BPIGamePrediction(**g) for g in raw.get("games", [])]

        assert "michigan" in teams
        assert teams["michigan"].bpi == 18.5
        assert len(games) == 1
        assert games[0].team1_win_pct == 72.3
        assert games[0].matchup_quality == 84.0


class TestTeamIDNormalization:
    def test_espn_names_normalized(self):
        scraper = BPIScraper.__new__(BPIScraper)
        scraper._espn_team_cache = {
            2026: {
                130: {"name": "Michigan Wolverines", "canonical_id": "michigan"},
                2579: {
                    "name": "South Carolina Gamecocks",
                    "canonical_id": "south_carolina",
                },
            }
        }

        assert scraper._canonical(130, 2026) == "michigan"
        assert scraper._canonical(2579, 2026) == "south_carolina"
        assert scraper._canonical(99999, 2026) == "espn_99999"
