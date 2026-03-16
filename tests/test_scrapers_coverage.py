"""Coverage tests for scraper modules with low or zero test coverage.

Tests class existence, constructors, pure utility functions, constants,
and basic method behavior with mocked network calls.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ======================================================================
# 1. ncaa_stats.py (21% coverage)
# ======================================================================


class TestNCAAStatsScraper:
    """Tests for src.data.scrapers.ncaa_stats.NCAAStatsScraper."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        assert NCAAStatsScraper is not None

    def test_constructor_no_cache(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        assert scraper.cache_dir is None
        assert scraper.session is not None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        cache = str(tmp_path / "ncaa_cache")
        scraper = NCAAStatsScraper(cache_dir=cache)
        assert scraper.cache_dir == Path(cache)
        assert scraper.cache_dir.exists()

    def test_load_cache_no_cache_dir(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        result = scraper._load_cache("anything.json")
        assert result is None

    def test_save_cache_no_cache_dir(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        # Should not raise
        scraper._save_cache("anything.json", {"test": True})

    def test_load_cache_file_not_found(self, tmp_path):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper(cache_dir=str(tmp_path))
        result = scraper._load_cache("nonexistent.json")
        assert result is None

    def test_save_and_load_cache(self, tmp_path):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper(cache_dir=str(tmp_path))
        data = {"teams": [{"name": "Duke"}]}
        scraper._save_cache("test.json", data)
        loaded = scraper._load_cache("test.json")
        assert loaded == data

    def test_load_cache_bad_json(self, tmp_path):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper(cache_dir=str(tmp_path))
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")
        result = scraper._load_cache("bad.json")
        assert result is None

    def test_fetch_tournament_teams_from_cache(self, tmp_path):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper(cache_dir=str(tmp_path))
        cached = {"teams": [{"name": "Duke"}, {"name": "UNC"}]}
        scraper._save_cache("ncaa_teams_2025.json", cached)
        result = scraper.fetch_tournament_teams(2025, "http://fake.url")
        assert len(result) == 2

    def test_fetch_tournament_teams_network(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"teams": [{"name": "Kansas"}]}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_tournament_teams(2025, "http://fake.url")
        assert result == [{"name": "Kansas"}]

    def test_fetch_tournament_teams_empty_raises(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"teams": []}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="no teams"):
            scraper.fetch_tournament_teams(2025, "http://fake.url")

    def test_fetch_historical_games_network(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"games": [{"id": 1}]}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_historical_games(2025, "http://fake.url")
        assert result == [{"id": 1}]

    def test_fetch_historical_games_empty_raises(self):
        from src.data.scrapers.ncaa_stats import NCAAStatsScraper
        scraper = NCAAStatsScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"games": []}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="no games"):
            scraper.fetch_historical_games(2025, "http://fake.url")


# ======================================================================
# 2. open_data_feed.py (30% coverage)
# ======================================================================


class TestOpenDataFeedScraper:
    """Tests for src.data.scrapers.open_data_feed.OpenDataFeedScraper."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        assert OpenDataFeedScraper is not None

    def test_constructor_no_cache(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        assert scraper.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper(cache_dir=str(tmp_path / "odf_cache"))
        assert scraper.cache_dir.exists()

    def test_extract_records_list_payload(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        result = scraper._extract_records([{"a": 1}, {"b": 2}, "skip"], None)
        assert result == [{"a": 1}, {"b": 2}]

    def test_extract_records_dict_with_key(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        payload = {"my_records": [{"x": 1}]}
        result = scraper._extract_records(payload, "my_records")
        assert result == [{"x": 1}]

    def test_extract_records_dict_auto_detect(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        payload = {"teams": [{"name": "Duke"}]}
        result = scraper._extract_records(payload, None)
        assert result == [{"name": "Duke"}]

    def test_extract_records_non_dict_non_list(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        result = scraper._extract_records("string_payload", None)
        assert result == []

    def test_extract_records_dict_no_match(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        result = scraper._extract_records({"unknown_key": [1, 2]}, None)
        assert result == []

    def test_parse_csv(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        csv_text = "name,score\nDuke,95\nUNC,88"
        result = scraper._parse_csv(csv_text)
        assert len(result) == 2
        assert result[0]["name"] == "Duke"
        assert result[1]["score"] == "88"

    def test_fetch_records_json(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": 1}]
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_records("test.json", "http://fake.url")
        assert result == [{"id": 1}]

    def test_fetch_records_csv(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        mock_resp = MagicMock()
        mock_resp.text = "name,score\nDuke,95"
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_records("test.csv", "http://fake.url", fmt="csv")
        assert len(result) == 1
        assert result[0]["name"] == "Duke"

    def test_fetch_records_empty_raises(self):
        from src.data.scrapers.open_data_feed import OpenDataFeedScraper
        scraper = OpenDataFeedScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"unknown": "structure"}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="no records"):
            scraper.fetch_records("test.json", "http://fake.url")


# ======================================================================
# 3. transfer_portal.py (24% coverage)
# ======================================================================


class TestTransferPortalScraper:
    """Tests for src.data.scrapers.transfer_portal.TransferPortalScraper."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        assert TransferPortalScraper is not None

    def test_constructor_no_cache(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper()
        assert scraper.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper(cache_dir=str(tmp_path))
        assert scraper.cache_dir.exists()

    def test_parse_csv(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper()
        csv_text = "player_id,player_name,source_team_name,destination_team_name,entry_date\np1,John Doe,Duke,UNC,2025-04-01"
        result = scraper._parse_csv(csv_text)
        assert len(result) == 1
        assert result[0]["player_name"] == "John Doe"
        assert result[0]["source_team_name"] == "Duke"
        assert result[0]["destination_team_name"] == "UNC"

    def test_parse_csv_alternate_columns(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper()
        csv_text = "id,name,from_team,to_team,entry_date\np2,Jane Smith,Kentucky,Alabama,2025-05-01"
        result = scraper._parse_csv(csv_text)
        assert len(result) == 1
        assert result[0]["player_id"] == "p2"
        assert result[0]["player_name"] == "Jane Smith"
        assert result[0]["source_team_name"] == "Kentucky"

    def test_fetch_entries_json(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"entries": [{"player": "A"}]}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_entries(2025, "http://fake.url")
        assert result == [{"player": "A"}]

    def test_fetch_entries_csv_format(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper()
        mock_resp = MagicMock()
        mock_resp.text = "player_id,player_name,source_team_name,destination_team_name,entry_date\np1,J,D,U,2025-01-01"
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_entries(2025, "http://fake.url", fmt="csv")
        assert len(result) == 1

    def test_fetch_entries_empty_raises(self):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"entries": []}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        with pytest.raises(ValueError, match="no entries"):
            scraper.fetch_entries(2025, "http://fake.url")

    def test_cache_round_trip(self, tmp_path):
        from src.data.scrapers.transfer_portal import TransferPortalScraper
        scraper = TransferPortalScraper(cache_dir=str(tmp_path))
        scraper._save_cache("tp.json", {"entries": [{"id": "1"}]})
        loaded = scraper._load_cache("tp.json")
        assert loaded["entries"] == [{"id": "1"}]


# ======================================================================
# 4. player_metrics.py (29% coverage)
# ======================================================================


class TestPlayerMetricsScraper:
    """Tests for src.data.scrapers.player_metrics.PlayerMetricsScraper."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        assert PlayerMetricsScraper is not None

    def test_constructor_no_cache(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        assert scraper.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper(cache_dir=str(tmp_path))
        assert scraper.cache_dir.exists()

    def test_to_float(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        assert PlayerMetricsScraper._to_float("3.14") == 3.14
        assert PlayerMetricsScraper._to_float(42) == 42.0
        assert PlayerMetricsScraper._to_float(None) == 0.0
        assert PlayerMetricsScraper._to_float("bad") == 0.0

    def test_to_int(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        assert PlayerMetricsScraper._to_int("10") == 10
        assert PlayerMetricsScraper._to_int("3.7") == 3
        assert PlayerMetricsScraper._to_int(None) == 0
        assert PlayerMetricsScraper._to_int("bad") == 0

    def test_to_bool(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        assert PlayerMetricsScraper._to_bool(True) is True
        assert PlayerMetricsScraper._to_bool(False) is False
        assert PlayerMetricsScraper._to_bool(None) is False
        assert PlayerMetricsScraper._to_bool("1") is True
        assert PlayerMetricsScraper._to_bool("true") is True
        assert PlayerMetricsScraper._to_bool("yes") is True
        assert PlayerMetricsScraper._to_bool("y") is True
        assert PlayerMetricsScraper._to_bool("0") is False
        assert PlayerMetricsScraper._to_bool("no") is False

    def test_team_id(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        assert PlayerMetricsScraper._team_id("Duke Blue Devils") == "duke_blue_devils"
        assert PlayerMetricsScraper._team_id("UNC") == "unc"
        assert PlayerMetricsScraper._team_id("St. John's") == "st__john_s"
        assert PlayerMetricsScraper._team_id("") == ""

    def test_team_player_id(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        result = PlayerMetricsScraper._team_player_id("duke", "John Smith")
        assert result == "duke_john_smith"
        result_empty = PlayerMetricsScraper._team_player_id("duke", "")
        assert result_empty == "duke_player"

    def test_normalize_json_payload_list(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        result = scraper._normalize_json_payload([{"team": "Duke"}])
        assert result == {"teams": [{"team": "Duke"}]}

    def test_normalize_json_payload_dict_with_teams(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        raw = {"teams": [{"team": "Duke"}], "source": "test", "timestamp": "2025-01-01"}
        result = scraper._normalize_json_payload(raw)
        assert result["teams"] == [{"team": "Duke"}]
        assert result["source"] == "test"

    def test_normalize_json_payload_dict_without_teams(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        result = scraper._normalize_json_payload({"other": "data"})
        assert result == {"teams": []}

    def test_normalize_json_payload_non_dict_non_list(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        result = scraper._normalize_json_payload("string")
        assert result == {"teams": []}

    def test_fetch_rosters_no_url(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        with patch.dict("os.environ", {}, clear=True):
            result = scraper.fetch_rosters(2025)
        assert result == {}

    def test_fetch_rosters_json(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"teams": [{"team_id": "duke"}]}
        mock_resp.raise_for_status = MagicMock()
        scraper.session.get = MagicMock(return_value=mock_resp)
        result = scraper.fetch_rosters(2025, source_url="http://fake.url")
        assert result["teams"] == [{"team_id": "duke"}]

    def test_parse_csv_basic(self):
        from src.data.scrapers.player_metrics import PlayerMetricsScraper
        scraper = PlayerMetricsScraper()
        csv_text = (
            "team_id,team_name,name,player_id,position,minutes_per_game,"
            "games_played,games_started,rapm_offensive,rapm_defensive,"
            "warp,box_plus_minus,usage_rate,injury_status,is_transfer,"
            "transfer_from,eligibility_year\n"
            "duke,Duke,John Doe,p1,PG,30,32,30,1.5,-0.5,2.0,3.0,20.5,"
            "healthy,false,,2\n"
        )
        result = scraper._parse_csv(csv_text)
        assert len(result) == 1
        assert result[0]["team_id"] == "duke"
        assert len(result[0]["players"]) == 1
        assert result[0]["players"][0]["name"] == "John Doe"


# ======================================================================
# 5. sports_reference.py (31% coverage)
# ======================================================================


class TestSportsReferenceScraper:
    """Tests for src.data.scrapers.sports_reference.SportsReferenceScraper."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert SportsReferenceScraper is not None

    def test_base_url_constant(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert "sports-reference.com" in SportsReferenceScraper.BASE_URL

    def test_required_fields_constant(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert "team_name" in SportsReferenceScraper._REQUIRED_FIELDS
        assert "pace" in SportsReferenceScraper._REQUIRED_FIELDS
        assert "off_rtg" in SportsReferenceScraper._REQUIRED_FIELDS
        assert "def_rtg" in SportsReferenceScraper._REQUIRED_FIELDS

    def test_constructor_no_cache(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        scraper = SportsReferenceScraper()
        assert scraper.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        scraper = SportsReferenceScraper(cache_dir=str(tmp_path))
        assert scraper.cache_dir.exists()

    def test_to_float(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert SportsReferenceScraper._to_float("3.14") == 3.14
        assert SportsReferenceScraper._to_float("bad") == 0.0

    def test_to_int(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert SportsReferenceScraper._to_int("10") == 10
        assert SportsReferenceScraper._to_int("bad") == 0

    def test_has_critical_zeros_empty(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert SportsReferenceScraper._has_critical_zeros([]) is True

    def test_has_critical_zeros_all_zero(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        teams = [{"def_rtg": 0}, {"def_rtg": 0}, {"def_rtg": 0}]
        assert SportsReferenceScraper._has_critical_zeros(teams) is True

    def test_has_critical_zeros_all_nonzero(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        teams = [{"def_rtg": 100}, {"def_rtg": 95}, {"def_rtg": 105}]
        assert SportsReferenceScraper._has_critical_zeros(teams) is False

    def test_has_degraded_schema_empty(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        assert SportsReferenceScraper._has_degraded_schema([]) is True

    def test_has_degraded_schema_valid(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        teams = [{
            "team_name": "Duke", "pace": 70, "off_rtg": 115,
            "def_rtg": 95, "wins": 30, "losses": 5, "srs": 20, "sos": 10,
        }]
        assert SportsReferenceScraper._has_degraded_schema(teams) is False

    def test_has_degraded_schema_missing_fields(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        teams = [{"team_name": "Duke", "pace": 70}]
        assert SportsReferenceScraper._has_degraded_schema(teams) is True

    def test_ncaa_suffix_regex(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        import re
        result = SportsReferenceScraper._NCAA_SUFFIX_RE.sub("", "Duke NCAA").rstrip()
        assert result == "Duke"
        result2 = SportsReferenceScraper._NCAA_SUFFIX_RE.sub("", "Duke").rstrip()
        assert result2 == "Duke"

    def test_compute_def_rtg_from_games_empty(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        result = SportsReferenceScraper._compute_def_rtg_from_games([])
        assert result == {}

    def test_compute_def_rtg_from_games_with_possessions(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        games = [
            {
                "team1_id": "Duke",
                "team2_id": "UNC",
                "team1_score": 80,
                "team2_score": 70,
                "possessions": 70,
            },
        ]
        result = SportsReferenceScraper._compute_def_rtg_from_games(games)
        # Duke allowed 70 pts in 70 possessions -> def_rtg = 100
        assert len(result) >= 1

    def test_compute_def_rtg_from_games_no_possessions(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        games = [
            {
                "team1_id": "Duke",
                "team2_id": "UNC",
                "team1_score": 80,
                "team2_score": 70,
                "possessions": 0,
            },
        ]
        result = SportsReferenceScraper._compute_def_rtg_from_games(games)
        # Uses approximate 70 possessions per game
        assert len(result) >= 1

    def test_compute_def_rtg_non_dict_games_skipped(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        games = ["not a dict", 42, None]
        result = SportsReferenceScraper._compute_def_rtg_from_games(games)
        assert result == {}

    def test_parse_team_table_empty_html(self):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        scraper = SportsReferenceScraper()
        result = scraper._parse_team_table("<html><body></body></html>")
        assert result == []

    def test_cache_round_trip(self, tmp_path):
        from src.data.scrapers.sports_reference import SportsReferenceScraper
        scraper = SportsReferenceScraper(cache_dir=str(tmp_path))
        data = {"teams": [{"team_name": "Duke"}]}
        scraper._save_cache("sr.json", data)
        loaded = scraper._load_cache("sr.json")
        assert loaded == data


# ======================================================================
# 6. conference_seeds.py (20% coverage)
# ======================================================================


class TestConferenceSeedsScraper:
    """Tests for src.data.scrapers.conference_seeds.ConferenceSeedsScraper."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        assert ConferenceSeedsScraper is not None

    def test_espn_conf_ids_constant(self):
        from src.data.scrapers.conference_seeds import _ESPN_CONF_IDS
        assert isinstance(_ESPN_CONF_IDS, dict)
        assert "ACC" in _ESPN_CONF_IDS
        assert "SEC" in _ESPN_CONF_IDS
        assert "B10" in _ESPN_CONF_IDS
        assert "B12" in _ESPN_CONF_IDS

    def test_espn_api_base_constant(self):
        from src.data.scrapers.conference_seeds import _ESPN_API_BASE
        assert "espn.com" in _ESPN_API_BASE

    def test_default_tourney_window_constant(self):
        from src.data.scrapers.conference_seeds import _DEFAULT_TOURNEY_WINDOW
        assert len(_DEFAULT_TOURNEY_WINDOW) == 4

    def test_constructor_no_cache(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper()
        assert scraper.cache_dir is None
        assert scraper.timeout == 30

    def test_constructor_with_cache_and_timeout(self, tmp_path):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper(cache_dir=str(tmp_path), timeout=60)
        assert scraper.cache_dir.exists()
        assert scraper.timeout == 60

    def test_map_espn_team_to_id_location(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        team_obj = {"location": "Duke", "name": "Blue Devils"}
        result = ConferenceSeedsScraper._map_espn_team_to_id(team_obj)
        assert result  # Should produce a non-empty ID
        assert isinstance(result, str)

    def test_map_espn_team_to_id_empty(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        result = ConferenceSeedsScraper._map_espn_team_to_id({})
        assert result == ""

    def test_map_espn_team_to_id_fallback_to_short_display(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        team_obj = {"shortDisplayName": "Duke"}
        result = ConferenceSeedsScraper._map_espn_team_to_id(team_obj)
        assert result != ""

    def test_map_espn_team_to_id_fallback_to_abbreviation(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        team_obj = {"abbreviation": "DUKE"}
        result = ConferenceSeedsScraper._map_espn_team_to_id(team_obj)
        assert result != ""

    def test_save_to_file(self, tmp_path):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper()
        seeds = {"ACC": {"duke": 1, "unc": 2}}
        output_path = str(tmp_path / "subdir" / "seeds.json")
        scraper.save_to_file(seeds, output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded == seeds

    def test_scrape_seeds_from_cache(self, tmp_path):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper(cache_dir=str(tmp_path))
        cached_data = {"seeds": {"ACC": {"duke": 1}}}
        scraper._save_cache("conference_seeds_2026.json", cached_data)
        result = scraper.scrape_seeds(2026)
        assert result == {"ACC": {"duke": 1}}

    def test_cache_round_trip(self, tmp_path):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper(cache_dir=str(tmp_path))
        data = {"seeds": {"B10": {"purdue": 1}}}
        scraper._save_cache("test.json", data)
        loaded = scraper._load_cache("test.json")
        assert loaded == data

    def test_load_cache_no_dir(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper()
        assert scraper._load_cache("anything.json") is None

    def test_save_cache_no_dir(self):
        from src.data.scrapers.conference_seeds import ConferenceSeedsScraper
        scraper = ConferenceSeedsScraper()
        # Should not raise
        scraper._save_cache("anything.json", {})


# ======================================================================
# 7. conference_bracket_validator.py (0% coverage)
# ======================================================================


class TestConferenceBracketValidator:
    """Tests for src.data.scrapers.conference_bracket_validator.ConferenceBracketValidator."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.conference_bracket_validator import (
            ConferenceBracketValidator,
        )
        assert ConferenceBracketValidator is not None

    def test_espn_conf_ids_constant(self):
        from src.data.scrapers.conference_bracket_validator import _ESPN_CONF_IDS
        assert isinstance(_ESPN_CONF_IDS, dict)
        assert "ACC" in _ESPN_CONF_IDS
        assert "SEC" in _ESPN_CONF_IDS

    def test_espn_api_base_constant(self):
        from src.data.scrapers.conference_bracket_validator import _ESPN_API_BASE
        assert "espn.com" in _ESPN_API_BASE

    def test_constructor_default(self):
        from src.data.scrapers.conference_bracket_validator import (
            ConferenceBracketValidator,
        )
        validator = ConferenceBracketValidator()
        assert validator.timeout == 30
        assert validator.session is not None

    def test_constructor_custom_timeout(self):
        from src.data.scrapers.conference_bracket_validator import (
            ConferenceBracketValidator,
        )
        validator = ConferenceBracketValidator(timeout=60)
        assert validator.timeout == 60

    def test_fetch_tournament_games_unknown_conf(self):
        from src.data.scrapers.conference_bracket_validator import (
            ConferenceBracketValidator,
        )
        validator = ConferenceBracketValidator()
        result = validator.fetch_tournament_games(2026, "FAKE_CONF")
        assert result == []

    def test_fetch_tournament_games_mocked(self):
        from src.data.scrapers.conference_bracket_validator import (
            ConferenceBracketValidator,
        )
        validator = ConferenceBracketValidator()
        # Mock the session to return no events for every date
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": []}
        mock_resp.raise_for_status = MagicMock()
        validator.session.get = MagicMock(return_value=mock_resp)
        with patch("src.data.scrapers.conference_bracket_validator.time.sleep"):
            result = validator.fetch_tournament_games(2026, "ACC")
        assert result == []

    def test_session_has_user_agent(self):
        from src.data.scrapers.conference_bracket_validator import (
            ConferenceBracketValidator,
        )
        validator = ConferenceBracketValidator()
        ua = validator.session.headers.get("User-Agent", "")
        assert "Mozilla" in ua


# ======================================================================
# 8. torvik.py (28% coverage)
# ======================================================================


class TestTorVikTeam:
    """Tests for the TorVikTeam dataclass."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.torvik import TorVikTeam
        assert TorVikTeam is not None

    def test_construct_torvik_team(self):
        from src.data.scrapers.torvik import TorVikTeam
        team = TorVikTeam(
            team_id="duke",
            name="Duke",
            conference="ACC",
            t_rank=1,
            barthag=0.97,
            adj_offensive_efficiency=122.0,
            adj_defensive_efficiency=93.0,
            adj_tempo=70.0,
            effective_fg_pct=0.55,
            turnover_rate=0.16,
            offensive_reb_rate=0.32,
            free_throw_rate=0.35,
            opp_effective_fg_pct=0.45,
            opp_turnover_rate=0.20,
            defensive_reb_rate=0.72,
            opp_free_throw_rate=0.28,
        )
        assert team.team_id == "duke"
        assert team.t_rank == 1

    def test_to_dict(self):
        from src.data.scrapers.torvik import TorVikTeam
        team = TorVikTeam(
            team_id="duke", name="Duke", conference="ACC",
            t_rank=1, barthag=0.97,
            adj_offensive_efficiency=122.0, adj_defensive_efficiency=93.0,
            adj_tempo=70.0,
            effective_fg_pct=0.55, turnover_rate=0.16,
            offensive_reb_rate=0.32, free_throw_rate=0.35,
            opp_effective_fg_pct=0.45, opp_turnover_rate=0.20,
            defensive_reb_rate=0.72, opp_free_throw_rate=0.28,
        )
        d = team.to_dict()
        assert d["team_id"] == "duke"
        assert d["name"] == "Duke"
        assert d["conference"] == "ACC"
        assert d["barthag"] == 0.97
        assert d["wins"] == 0  # default
        assert d["wab"] == 0.0  # default
        assert "three_pt_pct" in d
        assert "opp_three_pt_rate" in d

    def test_default_values(self):
        from src.data.scrapers.torvik import TorVikTeam
        team = TorVikTeam(
            team_id="unc", name="UNC", conference="ACC",
            t_rank=5, barthag=0.90,
            adj_offensive_efficiency=115.0, adj_defensive_efficiency=98.0,
            adj_tempo=68.0,
            effective_fg_pct=0.50, turnover_rate=0.18,
            offensive_reb_rate=0.30, free_throw_rate=0.30,
            opp_effective_fg_pct=0.48, opp_turnover_rate=0.18,
            defensive_reb_rate=0.70, opp_free_throw_rate=0.30,
        )
        assert team.two_pt_pct == 0.0
        assert team.three_pt_pct == 0.0
        assert team.ft_pct == 0.0
        assert team.block_pct == 0.0
        assert team.steal_pct == 0.0
        assert team.wins == 0
        assert team.losses == 0
        assert team.conf_wins == 0
        assert team.conf_losses == 0


class TestBartTorvikScraper:
    """Tests for BartTorvikScraper class."""

    def test_import_and_class_exists(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        assert BartTorvikScraper is not None

    def test_base_url_constant(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        assert BartTorvikScraper.BASE_URL == "https://barttorvik.com"

    def test_constructor_no_cache(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        assert scraper.cache_dir is None
        assert scraper.session is not None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(cache_dir=str(tmp_path))
        assert scraper.cache_dir.exists()

    def test_safe_float(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        assert scraper._safe_float("3.14") == 3.14
        assert scraper._safe_float("50.2%") == 50.2
        assert scraper._safe_float("bad") == 0.0
        assert scraper._safe_float("") == 0.0

    def test_normalize_team_name_to_id(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        assert BartTorvikScraper._normalize_team_name_to_id("Duke") == "duke"
        assert BartTorvikScraper._normalize_team_name_to_id("North Carolina") == "north_carolina"
        assert BartTorvikScraper._normalize_team_name_to_id("St. John's") == "st_john_s"
        assert BartTorvikScraper._normalize_team_name_to_id("Texas A&M") == "texas_a_m"

    def test_dict_to_team(self):
        from src.data.scrapers.torvik import BartTorvikScraper, TorVikTeam
        scraper = BartTorvikScraper()
        data = {
            "team_id": "duke", "name": "Duke", "conference": "ACC",
            "t_rank": 1, "barthag": 0.97,
            "adj_offensive_efficiency": 122.0,
            "adj_defensive_efficiency": 93.0,
            "adj_tempo": 70.0,
        }
        team = scraper._dict_to_team(data)
        assert isinstance(team, TorVikTeam)
        assert team.team_id == "duke"
        assert team.t_rank == 1

    def test_dict_to_team_defaults(self):
        from src.data.scrapers.torvik import BartTorvikScraper, TorVikTeam
        scraper = BartTorvikScraper()
        team = scraper._dict_to_team({})
        assert team.team_id == ""
        assert team.t_rank == 999
        assert team.barthag == 0.5
        assert team.adj_tempo == 68.0

    def test_cache_round_trip(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(cache_dir=str(tmp_path))
        data = {"teams": [{"name": "Duke"}]}
        scraper._save_to_cache("test.json", data)
        loaded = scraper._load_from_cache("test.json")
        assert loaded == data

    def test_load_from_cache_no_dir(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        assert scraper._load_from_cache("anything.json") is None

    def test_load_from_cache_bad_json(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(cache_dir=str(tmp_path))
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{{invalid json")
        assert scraper._load_from_cache("bad.json") is None

    def test_save_to_cache_no_dir(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        # Should not raise
        scraper._save_to_cache("anything.json", {})

    def test_fetch_current_rankings_from_cache(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(cache_dir=str(tmp_path))
        cached = {
            "teams": [{
                "team_id": "duke", "name": "Duke", "conference": "ACC",
                "t_rank": 1, "barthag": 0.97,
                "adj_offensive_efficiency": 122.0,
                "adj_defensive_efficiency": 93.0,
                "adj_tempo": 70.0,
            }]
        }
        scraper._save_to_cache("torvik_rankings_2026.json", cached)
        result = scraper.fetch_current_rankings(year=2026)
        assert len(result) == 1
        assert result[0].team_id == "duke"

    def test_fetch_current_rankings_network_failure(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        scraper.session.get = MagicMock(side_effect=Exception("Connection error"))
        result = scraper.fetch_current_rankings(year=2026)
        assert result == []

    def test_parse_rankings_page_no_table(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        result = scraper._parse_rankings_page("<html><body><p>No table</p></body></html>")
        assert result == []

    def test_aggregate_player_csv_empty(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        result = scraper._aggregate_player_csv("")
        assert result == {}

    def test_aggregate_player_csv_basic(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        # Build a minimal CSV line with at least 22 columns
        cols = [""] * 22
        cols[0] = "Player1"
        cols[1] = "Duke"
        cols[2] = "ACC"
        cols[4] = "25.0"  # min_pct
        cols[9] = "10.0"  # orb_pct
        cols[10] = "20.0"  # drb_pct
        cols[12] = "15.0"  # to_pct
        cols[13] = "50"  # ftm
        cols[14] = "60"  # fta
        cols[16] = "100"  # fg2m
        cols[17] = "200"  # fg2a
        cols[19] = "30"  # fg3m
        cols[20] = "80"  # fg3a
        csv_text = ",".join(cols)
        result = scraper._aggregate_player_csv(csv_text)
        assert "duke" in result
        assert result["duke"]["fg3m"] == 30.0
        assert result["duke"]["ftm"] == 50.0

    def test_fetch_four_factors_from_cache(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(cache_dir=str(tmp_path))
        cached = {"duke": {"effective_fg_pct": 0.55}}
        scraper._save_to_cache("torvik_four_factors_2026.json", cached)
        result = scraper.fetch_four_factors(year=2026)
        assert result == cached

    def test_fetch_shooting_stats_from_cache(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper(cache_dir=str(tmp_path))
        cached = {"duke": {"three_pt_pct": 0.36, "ft_pct": 0.78}}
        scraper._save_to_cache("torvik_shooting_2026.json", cached)
        result = scraper.fetch_shooting_stats(year=2026)
        assert result == cached

    def test_load_from_json(self, tmp_path):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        data = {
            "teams": [{
                "team_id": "duke", "name": "Duke", "conference": "ACC",
                "t_rank": 1, "barthag": 0.97,
                "adj_offensive_efficiency": 122.0,
                "adj_defensive_efficiency": 93.0,
                "adj_tempo": 70.0,
            }]
        }
        filepath = tmp_path / "torvik_data.json"
        with open(filepath, "w") as f:
            json.dump(data, f)
        result = scraper.load_from_json(str(filepath))
        assert len(result) == 1
        assert result[0].name == "Duke"

    def test_parse_four_factors_page_no_table(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        result = scraper._parse_four_factors_page("<html><body></body></html>")
        assert result == {}

    def test_parse_shooting_page_no_table(self):
        from src.data.scrapers.torvik import BartTorvikScraper
        scraper = BartTorvikScraper()
        result = scraper._parse_shooting_page("<html><body></body></html>")
        assert result == {}
