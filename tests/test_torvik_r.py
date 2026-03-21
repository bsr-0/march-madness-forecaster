"""Unit tests for the toRvik R package wrapper (torvik_r.py).

These tests cover:
- JSON parsing and column mapping (no R required)
- Graceful degradation when R/toRvik is unavailable
- Integration with the providers.py fallback chain
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.data.scrapers.torvik_r import TorvikRWrapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TORVIK_RATINGS_JSON = json.dumps([
    {
        "team": "Duke",
        "conf": "ACC",
        "year": 2026,
        "rk": 1,
        "barthag": 0.97,
        "adj_o": 122.5,
        "adj_d": 88.3,
        "adj_t": 72.1,
        "off_efg": 55.2,
        "off_to": 14.1,
        "off_or": 33.5,
        "off_ftr": 37.2,
        "def_efg": 43.1,
        "def_to": 18.3,
        "def_or": 71.2,
        "def_ftr": 25.0,
        "wab": 12.3,
        "rec": "32-4",
    },
    {
        "team": "Kansas",
        "conf": "B12",
        "year": 2026,
        "rk": 2,
        "barthag": 0.95,
        "adj_o": 118.0,
        "adj_d": 90.2,
        "adj_t": 69.8,
        "off_efg": 52.1,
        "off_to": 16.3,
        "off_or": 29.7,
        "off_ftr": 34.0,
        "def_efg": 45.0,
        "def_to": 17.1,
        "def_or": 68.4,
        "def_ftr": 26.5,
        "wab": 10.1,
        "rec": "29-6",
    },
])

SAMPLE_SCHEDULE_JSON = json.dumps([
    {
        "game_id": "g001",
        "date": "2026-01-15",
        "home": "Duke",
        "away": "Kansas",
        "home_score": 78,
        "away_score": 72,
    },
])


# ---------------------------------------------------------------------------
# Helper: mock _run_r to return sample JSON without launching Rscript
# ---------------------------------------------------------------------------

def _mock_run_r_ratings(_script: str):
    return SAMPLE_TORVIK_RATINGS_JSON


def _mock_run_r_schedule(_script: str):
    return SAMPLE_SCHEDULE_JSON


# ---------------------------------------------------------------------------
# Tests: column mapping
# ---------------------------------------------------------------------------

class TestMapRatingsRow:
    def test_maps_all_core_fields(self):
        wrapper = TorvikRWrapper()
        row = json.loads(SAMPLE_TORVIK_RATINGS_JSON)[0]
        result = wrapper._map_ratings_row(row)

        assert result is not None
        assert result["team_name"] == "Duke"
        assert result["conference"] == "ACC"
        assert result["t_rank"] == 1
        assert abs(result["barthag"] - 0.97) < 1e-6
        assert abs(result["adj_offensive_efficiency"] - 122.5) < 1e-6
        assert abs(result["adj_defensive_efficiency"] - 88.3) < 1e-6
        assert abs(result["adj_tempo"] - 72.1) < 1e-6

    def test_normalises_rates_from_percentage(self):
        """Fields like off_efg=55.2 should be stored as 0.552."""
        wrapper = TorvikRWrapper()
        row = json.loads(SAMPLE_TORVIK_RATINGS_JSON)[0]
        result = wrapper._map_ratings_row(row)

        assert result["effective_fg_pct"] < 1.0, "eFG% should be in 0-1 range"
        assert abs(result["effective_fg_pct"] - 0.552) < 0.001
        assert abs(result["opp_effective_fg_pct"] - 0.431) < 0.001

    def test_parses_wins_losses_from_rec(self):
        wrapper = TorvikRWrapper()
        row = json.loads(SAMPLE_TORVIK_RATINGS_JSON)[0]
        result = wrapper._map_ratings_row(row)

        assert result["wins"] == 32
        assert result["losses"] == 4

    def test_returns_none_for_empty_team_name(self):
        wrapper = TorvikRWrapper()
        result = wrapper._map_ratings_row({"team": "", "rk": 1})
        assert result is None

    def test_uses_canonical_team_id(self):
        wrapper = TorvikRWrapper()
        row = {"team": "North Carolina", "rk": 5, "barthag": 0.9, "adj_o": 115.0, "adj_d": 95.0}
        result = wrapper._map_ratings_row(row)
        # team_id should be normalised (lowercase, no spaces)
        assert result is not None
        assert " " not in result["team_id"]


class TestMapScheduleRow:
    def test_maps_home_team_row(self):
        wrapper = TorvikRWrapper()
        row = json.loads(SAMPLE_SCHEDULE_JSON)[0]
        result = wrapper._map_schedule_row(row)

        assert result is not None
        assert result["date"] == "2026-01-15"
        assert result["team_score"] == 78
        assert result["opponent_score"] == 72

    def test_returns_none_for_empty_teams(self):
        wrapper = TorvikRWrapper()
        result = wrapper._map_schedule_row({"game_id": "x", "date": "2026-01-01"})
        assert result is None


# ---------------------------------------------------------------------------
# Tests: JSON parsing
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_parses_clean_array(self):
        wrapper = TorvikRWrapper()
        rows = wrapper._parse_json(SAMPLE_TORVIK_RATINGS_JSON)
        assert len(rows) == 2
        assert rows[0]["team"] == "Duke"

    def test_skips_r_warnings_before_json(self):
        wrapper = TorvikRWrapper()
        noisy = "Warning message: something happened\n" + SAMPLE_TORVIK_RATINGS_JSON
        rows = wrapper._parse_json(noisy)
        assert len(rows) == 2

    def test_returns_empty_for_empty_array(self):
        wrapper = TorvikRWrapper()
        assert wrapper._parse_json("[]") == []

    def test_returns_empty_for_no_json(self):
        wrapper = TorvikRWrapper()
        assert wrapper._parse_json("Error: package not found") == []


# ---------------------------------------------------------------------------
# Tests: fetch_ratings (mocked Rscript)
# ---------------------------------------------------------------------------

class TestFetchRatings:
    def test_returns_records_when_r_available(self):
        with (
            patch("src.data.scrapers.torvik_r.is_available", return_value=True),
            patch("src.data.scrapers.torvik_r._run_r", side_effect=_mock_run_r_ratings),
        ):
            wrapper = TorvikRWrapper()
            records = wrapper.fetch_ratings(2026)

        assert len(records) == 2
        teams = {r["team_name"] for r in records}
        assert "Duke" in teams
        assert "Kansas" in teams

    def test_returns_empty_when_r_unavailable(self):
        with patch("src.data.scrapers.torvik_r.is_available", return_value=False):
            wrapper = TorvikRWrapper()
            records = wrapper.fetch_ratings(2026)
        assert records == []

    def test_returns_empty_when_rscript_fails(self):
        with (
            patch("src.data.scrapers.torvik_r.is_available", return_value=True),
            patch("src.data.scrapers.torvik_r._run_r", return_value=None),
        ):
            wrapper = TorvikRWrapper()
            records = wrapper.fetch_ratings(2026)
        assert records == []

    def test_skips_malformed_rows(self):
        malformed = json.dumps([{"team": "Duke", "rk": "not-a-number"}, {"team": "", "rk": 2}])
        with (
            patch("src.data.scrapers.torvik_r.is_available", return_value=True),
            patch("src.data.scrapers.torvik_r._run_r", return_value=malformed),
        ):
            wrapper = TorvikRWrapper()
            # Duke row should survive (rk defaults to 999 on parse error), empty team skipped
            records = wrapper.fetch_ratings(2026)
        # only rows with non-empty team_name survive
        assert all(r["team_name"] for r in records)


# ---------------------------------------------------------------------------
# Tests: provider integration
# ---------------------------------------------------------------------------

class TestProviderIntegration:
    """Verify LibraryProviderHub.fetch_torvik_ratings uses torvik_r first."""

    def test_torvik_r_is_first_priority(self):
        from src.data.ingestion.providers import LibraryProviderHub
        hub = LibraryProviderHub()
        priorities = hub.DEFAULT_PRIORITIES["torvik"]
        assert priorities[0] == "torvik_r"

    def test_falls_back_when_torvik_r_empty(self):
        """When torvik_r returns no records, the hub tries the next provider."""
        from src.data.ingestion.providers import LibraryProviderHub, ProviderResult

        hub = LibraryProviderHub()

        fallback_records = [{"team_id": "duke", "team_name": "Duke", "barthag": 0.97,
                              "adj_offensive_efficiency": 120.0, "adj_defensive_efficiency": 90.0,
                              "adj_tempo": 70.0, "effective_fg_pct": 0.55, "turnover_rate": 0.14,
                              "offensive_reb_rate": 0.33, "free_throw_rate": 0.37,
                              "t_rank": 1, "conference": "ACC"}]

        with (
            patch.object(hub, "_from_torvik_r", return_value=ProviderResult("torvik_r", [])),
            patch.object(hub, "_from_barttorvik_csv", return_value=ProviderResult("barttorvik", fallback_records)),
        ):
            result = hub.fetch_torvik_ratings(2026)

        assert result.provider == "barttorvik"
        assert len(result.records) == 1

    def test_uses_torvik_r_when_available(self):
        from src.data.ingestion.providers import LibraryProviderHub, ProviderResult

        hub = LibraryProviderHub()
        torvik_records = [{"team_id": "duke", "team_name": "Duke", "barthag": 0.97,
                            "adj_offensive_efficiency": 122.0, "adj_defensive_efficiency": 88.0,
                            "adj_tempo": 72.0, "effective_fg_pct": 0.55, "turnover_rate": 0.14,
                            "offensive_reb_rate": 0.33, "free_throw_rate": 0.37,
                            "t_rank": 1, "conference": "ACC"}]

        with patch.object(hub, "_from_torvik_r", return_value=ProviderResult("torvik_r", torvik_records)):
            result = hub.fetch_torvik_ratings(2026)

        assert result.provider == "torvik_r"
        assert len(result.records) == 1
