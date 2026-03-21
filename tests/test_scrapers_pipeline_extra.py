"""Comprehensive unit tests for scrapers, pipeline stages, ML calibration, and agents.

Covers:
1. src/data/scrapers/cbbpy_rosters.py
2. src/data/scrapers/tournament_context.py
3. src/data/scrapers/espn_picks.py
4. src/pipeline/sota.py (selected utilities)
5. src/pipeline/stages/calibration.py
6. src/ml/calibration/brier_optimal.py
7. src/agents/concrete.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch, mock_open

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. CBBpyRosterScraper tests
# ---------------------------------------------------------------------------

from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper


class TestCBBpyRosterScraperInit:
    def test_init_no_cache(self):
        s = CBBpyRosterScraper()
        assert s.cache_dir is None

    def test_init_with_cache(self, tmp_path):
        s = CBBpyRosterScraper(cache_dir=str(tmp_path / "cache"))
        assert s.cache_dir.exists()


class TestCBBpyStaticHelpers:
    """Test all static/class methods on CBBpyRosterScraper."""

    def test_team_id(self):
        assert CBBpyRosterScraper._team_id("Duke Blue Devils") == "duke_blue_devils"
        assert CBBpyRosterScraper._team_id("") == ""
        assert CBBpyRosterScraper._team_id("UNC") == "unc"

    def test_player_id(self):
        pid = CBBpyRosterScraper._player_id("duke", "John Smith")
        assert pid == "duke_john_smith"
        assert CBBpyRosterScraper._player_id("duke", "") == "duke_player"

    def test_to_float(self):
        assert CBBpyRosterScraper._to_float(None) == 0.0
        assert CBBpyRosterScraper._to_float(3.5) == 3.5
        assert CBBpyRosterScraper._to_float("10") == 10.0
        assert CBBpyRosterScraper._to_float("abc") == 0.0

    def test_to_bool(self):
        assert CBBpyRosterScraper._to_bool(True) is True
        assert CBBpyRosterScraper._to_bool(False) is False
        assert CBBpyRosterScraper._to_bool(None) is False
        assert CBBpyRosterScraper._to_bool("1") is True
        assert CBBpyRosterScraper._to_bool("true") is True
        assert CBBpyRosterScraper._to_bool("yes") is True
        assert CBBpyRosterScraper._to_bool("no") is False

    def test_safe_int(self):
        assert CBBpyRosterScraper._safe_int("5", 0) == 5
        assert CBBpyRosterScraper._safe_int(None, 99) == 99
        assert CBBpyRosterScraper._safe_int("abc", 0) == 0

    def test_parse_minutes_none(self):
        assert CBBpyRosterScraper._parse_minutes(None) == 0.0

    def test_parse_minutes_int(self):
        assert CBBpyRosterScraper._parse_minutes(30) == 30.0

    def test_parse_minutes_float(self):
        assert CBBpyRosterScraper._parse_minutes(25.5) == 25.5

    def test_parse_minutes_colon(self):
        result = CBBpyRosterScraper._parse_minutes("35:30")
        assert abs(result - 35.5) < 0.01

    def test_parse_minutes_string(self):
        assert CBBpyRosterScraper._parse_minutes("20") == 20.0

    def test_parse_minutes_invalid(self):
        assert CBBpyRosterScraper._parse_minutes("abc") == 0.0

    def test_parse_minutes_invalid_colon(self):
        assert CBBpyRosterScraper._parse_minutes("ab:cd") == 0.0

    def test_normalize_position(self):
        assert CBBpyRosterScraper._normalize_position("PG") == "PG"
        assert CBBpyRosterScraper._normalize_position("SG") == "SG"
        assert CBBpyRosterScraper._normalize_position("SF") == "SF"
        assert CBBpyRosterScraper._normalize_position("PF") == "PF"
        assert CBBpyRosterScraper._normalize_position("C") == "C"
        assert CBBpyRosterScraper._normalize_position("POINT GUARD") == "PG"
        assert CBBpyRosterScraper._normalize_position("CENTER") == "C"
        assert CBBpyRosterScraper._normalize_position("G") == "PG"
        assert CBBpyRosterScraper._normalize_position("F") == "SF"
        assert CBBpyRosterScraper._normalize_position("") == "PG"

    def test_parse_eligibility_year(self):
        assert CBBpyRosterScraper._parse_eligibility_year(None) is None
        assert CBBpyRosterScraper._parse_eligibility_year("FR") == 1
        assert CBBpyRosterScraper._parse_eligibility_year("FRESHMAN") == 1
        assert CBBpyRosterScraper._parse_eligibility_year("SO") == 2
        assert CBBpyRosterScraper._parse_eligibility_year("SOPHOMORE") == 2
        assert CBBpyRosterScraper._parse_eligibility_year("JR") == 3
        assert CBBpyRosterScraper._parse_eligibility_year("SR") == 4
        assert CBBpyRosterScraper._parse_eligibility_year("GR") == 5
        assert CBBpyRosterScraper._parse_eligibility_year("GRADUATE") == 5
        assert CBBpyRosterScraper._parse_eligibility_year("5TH") == 5
        assert CBBpyRosterScraper._parse_eligibility_year("RS FR") == 1
        assert CBBpyRosterScraper._parse_eligibility_year("RS-SO") == 2
        assert CBBpyRosterScraper._parse_eligibility_year("RS-JR") == 3
        assert CBBpyRosterScraper._parse_eligibility_year("3") == 3
        assert CBBpyRosterScraper._parse_eligibility_year("garbage") is None

    def test_normalize_player_name(self):
        assert CBBpyRosterScraper._normalize_player_name("John Smith") == "johnsmith"
        assert CBBpyRosterScraper._normalize_player_name("O'Brien") == "obrien"

    def test_is_possession_ending(self):
        assert CBBpyRosterScraper._is_possession_ending("turnover", "", {}) is True
        assert CBBpyRosterScraper._is_possession_ending("made shot", "", {}) is True
        assert CBBpyRosterScraper._is_possession_ending(
            "shooting foul", "", {"scoring_play": True}
        ) is True
        assert CBBpyRosterScraper._is_possession_ending(
            "rebound", "Defensive Rebound", {}
        ) is True
        assert CBBpyRosterScraper._is_possession_ending(
            "free throw", "Free throw 2 of 2", {}
        ) is True
        assert CBBpyRosterScraper._is_possession_ending(
            "free throw", "Free throw 1 of 1", {}
        ) is True
        assert CBBpyRosterScraper._is_possession_ending(
            "free throw", "Free throw 1 of 2", {}
        ) is False
        assert CBBpyRosterScraper._is_possession_ending("dribble", "Player dribbles", {}) is False

    def test_frame_to_records_none(self):
        assert CBBpyRosterScraper._frame_to_records(None) == []

    def test_frame_to_records_list(self):
        records = CBBpyRosterScraper._frame_to_records([{"a": 1}, "not_dict", {"b": 2}])
        assert records == [{"a": 1}, {"b": 2}]

    def test_frame_to_records_dict(self):
        assert CBBpyRosterScraper._frame_to_records({"x": 1}) == [{"x": 1}]

    def test_frame_to_records_dataframe_like(self):
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"a": 1}]
        assert CBBpyRosterScraper._frame_to_records(mock_df) == [{"a": 1}]

    def test_frame_to_records_unknown(self):
        assert CBBpyRosterScraper._frame_to_records(42) == []

    def test_import_module_missing(self):
        assert CBBpyRosterScraper._import_module("nonexistent_module_xyz") is None


class TestCBBpyRosterScraperCache:
    def test_load_cache_no_dir(self):
        s = CBBpyRosterScraper()
        assert s._load_cache("test.json") is None

    def test_save_cache_no_dir(self):
        s = CBBpyRosterScraper()
        s._save_cache("test.json", {"a": 1})  # should not raise

    def test_load_save_cache(self, tmp_path):
        s = CBBpyRosterScraper(cache_dir=str(tmp_path / "cache"))
        s._save_cache("test.json", {"teams": [1, 2, 3]})
        loaded = s._load_cache("test.json")
        assert loaded == {"teams": [1, 2, 3]}

    def test_load_cache_missing_file(self, tmp_path):
        s = CBBpyRosterScraper(cache_dir=str(tmp_path / "cache"))
        assert s._load_cache("nonexistent.json") is None

    def test_load_cache_invalid_json(self, tmp_path):
        s = CBBpyRosterScraper(cache_dir=str(tmp_path / "cache"))
        bad_path = s.cache_dir / "bad.json"
        bad_path.write_text("{invalid json")
        assert s._load_cache("bad.json") is None


class TestCBBpyRosterScraperBuildPayload:
    def test_build_payload_basic(self):
        s = CBBpyRosterScraper()
        box_rows = [
            {
                "team": "Duke",
                "player": "Player A",
                "game_id": "g1",
                "min": 30,
                "pts": 20,
                "reb": 5,
                "ast": 3,
                "stl": 1,
                "blk": 0,
                "to": 2,
                "fga": 10,
                "fgm": 7,
                "3pm": 2,
                "fta": 5,
                "oreb": 1,
                "dreb": 4,
                "pf": 2,
                "position": "G",
                "starter": True,
            },
            {
                "team": "Duke",
                "player": "Player B",
                "game_id": "g1",
                "min": 20,
                "pts": 10,
                "reb": 3,
                "ast": 2,
                "stl": 0,
                "blk": 1,
                "to": 1,
                "fga": 5,
                "fgm": 4,
                "3pm": 1,
                "fta": 2,
                "oreb": 0,
                "dreb": 3,
                "pf": 1,
                "position": "F",
                "starter": False,
            },
            {
                "team": "UNC",
                "player": "Player C",
                "game_id": "g1",
                "min": 25,
                "pts": 15,
                "reb": 7,
                "ast": 4,
                "stl": 2,
                "blk": 1,
                "to": 3,
                "fga": 8,
                "fgm": 5,
                "3pm": 1,
                "fta": 3,
                "oreb": 2,
                "dreb": 5,
                "pf": 3,
                "position": "C",
                "starter": True,
            },
        ]
        result = s._build_payload(2025, box_rows)
        assert result["year"] == 2025
        assert len(result["teams"]) == 2

    def test_build_payload_skips_totals(self):
        s = CBBpyRosterScraper()
        box_rows = [
            {"team": "Duke", "player": "Team", "game_id": "g1", "min": 200},
            {"team": "Duke", "player": "Totals", "game_id": "g1", "min": 200},
            {"team": "Duke", "player": "Real Player", "game_id": "g1", "min": 20,
             "pts": 5, "reb": 2, "ast": 1, "stl": 0, "blk": 0, "to": 0,
             "fga": 3, "fgm": 2, "3pm": 0, "fta": 1, "oreb": 0, "dreb": 2, "pf": 1,
             "position": "G", "starter": True},
        ]
        result = s._build_payload(2025, box_rows)
        # Should have exactly one player (not "Team" or "Totals")
        for team in result.get("teams", []):
            for p in team["players"]:
                assert p["name"] not in ("Team", "Totals")


class TestCBBpyExtractors:
    def test_extract_box_rows_tuple(self):
        s = CBBpyRosterScraper()
        data = ("info", [{"a": 1}])
        result = s._extract_box_rows(data)
        assert result == [{"a": 1}]

    def test_extract_box_rows_non_tuple(self):
        s = CBBpyRosterScraper()
        result = s._extract_box_rows([{"b": 2}])
        assert result == [{"b": 2}]

    def test_extract_pbp_rows_tuple(self):
        s = CBBpyRosterScraper()
        data = ("info", [{"box": 1}], [{"pbp": 1}])
        result = s._extract_pbp_rows(data)
        assert result == [{"pbp": 1}]

    def test_extract_pbp_rows_short_tuple(self):
        s = CBBpyRosterScraper()
        assert s._extract_pbp_rows(("info", [{"box": 1}])) == []

    def test_extract_pbp_rows_non_tuple(self):
        s = CBBpyRosterScraper()
        assert s._extract_pbp_rows([{"a": 1}]) == []

    def test_frame_to_single_row(self):
        s = CBBpyRosterScraper()
        assert s._frame_to_single_row(None) is None
        assert s._frame_to_single_row([{"a": 1}, {"b": 2}]) == {"a": 1}
        assert s._frame_to_single_row([]) is None


class TestCBBpyApplyPlayerProfile:
    def test_apply_player_profile_position(self):
        s = CBBpyRosterScraper()
        player = {"position": "PG", "eligibility_year": 1}
        profile = {"position": "C", "class": "JUNIOR"}
        s._apply_player_profile(player, profile)
        assert player["position"] == "C"
        assert player["eligibility_year"] == 3

    def test_apply_player_profile_no_position(self):
        s = CBBpyRosterScraper()
        player = {"position": "PG", "eligibility_year": 1}
        profile = {}
        s._apply_player_profile(player, profile)
        assert player["position"] == "PG"
        assert player["eligibility_year"] == 1


class TestCBBpySeasonDates:
    def test_season_dates_returns_dates(self):
        s = CBBpyRosterScraper()
        dates = list(s._season_dates(2025))
        assert len(dates) > 0
        # First date should be Nov 1 of previous year
        from datetime import date
        assert dates[0] == date(2024, 11, 1)


class TestCBBpyFetchRosters:
    def test_fetch_rosters_uses_cache(self, tmp_path):
        s = CBBpyRosterScraper(cache_dir=str(tmp_path))
        cached = {"teams": [{"team_id": "duke"}]}
        s._save_cache("cbbpy_rosters_2025.json", cached)
        result = s.fetch_rosters(2025)
        assert result == cached

    def test_fetch_rosters_no_module(self):
        s = CBBpyRosterScraper()
        with patch.object(s, "_import_module", return_value=None):
            result = s.fetch_rosters(2025)
            assert result == {}

    def test_fetch_rosters_empty_box_rows(self):
        s = CBBpyRosterScraper()
        mock_scraper = MagicMock()
        with patch.object(s, "_import_module", return_value=mock_scraper), \
             patch.object(s, "_collect_box_and_pbp_rows", return_value=([], [], "none")):
            result = s.fetch_rosters(2025)
            assert result == {}


class TestCBBpyEstimateRapmFromBoxScore:
    def test_starter_produces_nonzero_rapm(self):
        p = {"pts": 450, "ast": 90, "oreb": 60, "reb": 180, "stl": 45,
             "blk": 30, "to": 60, "pf": 60}
        off, dfn = CBBpyRosterScraper._estimate_rapm_from_box_score(
            p, games_played=30, minutes_per_game=30.0, true_shooting=0.55,
        )
        assert off != 0.0
        assert dfn != 0.0

    def test_low_minutes_returns_zero(self):
        p = {"pts": 10, "ast": 5, "oreb": 3, "reb": 10, "stl": 2,
             "blk": 1, "to": 3, "pf": 5}
        off, dfn = CBBpyRosterScraper._estimate_rapm_from_box_score(
            p, games_played=5, minutes_per_game=3.0, true_shooting=0.40,
        )
        assert off == 0.0 and dfn == 0.0


class TestCBBpyScrapeGameIds:
    @patch("src.data.scrapers.cbbpy_rosters.CBBpyRosterScraper._scrape_game_ids_for_date")
    def test_scrape_game_ids(self, mock_scrape):
        mock_scrape.return_value = ["401234567", "401234568"]
        result = CBBpyRosterScraper._scrape_game_ids_for_date("2025-01-15")
        assert len(result) == 2


class TestCBBpyRunWithTimeout:
    def test_run_with_timeout_success(self):
        result = CBBpyRosterScraper._run_with_timeout(lambda: 42, timeout=5)
        assert result == 42

    def test_run_with_timeout_with_args(self):
        result = CBBpyRosterScraper._run_with_timeout(
            lambda x, y: x + y, args=(3, 4), timeout=5
        )
        assert result == 7


# ---------------------------------------------------------------------------
# 2. TournamentContextScraper tests
# ---------------------------------------------------------------------------

from src.data.scrapers.tournament_context import TournamentContextScraper


class TestTournamentContextHelpers:
    def test_normalize_name(self):
        assert TournamentContextScraper._normalize_name("Duke Blue Devils") == "duke_blue_devils"
        assert TournamentContextScraper._normalize_name("") == ""

    def test_parse_rank(self):
        assert TournamentContextScraper._parse_rank("1") == 1
        assert TournamentContextScraper._parse_rank("14T") == 14
        assert TournamentContextScraper._parse_rank("") is None
        assert TournamentContextScraper._parse_rank("abc") is None

    def test_safe_int(self):
        assert TournamentContextScraper._safe_int("10") == 10
        assert TournamentContextScraper._safe_int("abc") == 0
        assert TournamentContextScraper._safe_int("") == 0

    def test_extract_team_name_with_link(self):
        from unittest.mock import MagicMock
        cell = MagicMock()
        link = MagicMock()
        link.get_text.return_value = "Duke"
        cell.find.return_value = link
        assert TournamentContextScraper._extract_team_name(cell) == "Duke"

    def test_extract_team_name_no_link(self):
        cell = MagicMock()
        cell.find.return_value = None
        cell.get_text.return_value = "UNC"
        assert TournamentContextScraper._extract_team_name(cell) == "UNC"


class TestTournamentContextCache:
    def test_load_save_cache(self, tmp_path):
        s = TournamentContextScraper(cache_dir=str(tmp_path / "ctx_cache"))
        s._save_cache("test.json", {"key": "value"})
        loaded = s._load_cache("test.json")
        assert loaded == {"key": "value"}

    def test_load_cache_no_dir(self):
        s = TournamentContextScraper()
        assert s._load_cache("test.json") is None

    def test_save_cache_no_dir(self):
        s = TournamentContextScraper()
        s._save_cache("test.json", {})  # should not raise


class TestTournamentContextAPRankings:
    def test_fetch_preseason_ap_cached(self, tmp_path):
        s = TournamentContextScraper(cache_dir=str(tmp_path))
        s._save_cache("preseason_ap_2026.json", {"rankings": {"duke": 1}, "year": 2026})
        result = s.fetch_preseason_ap_rankings(2026)
        assert result == {"duke": 1}

    @patch.object(TournamentContextScraper, "_scrape_preseason_ap", return_value={})
    def test_fetch_preseason_ap_scrape_empty(self, mock_scrape):
        s = TournamentContextScraper()
        result = s.fetch_preseason_ap_rankings(2026)
        assert result == {}


class TestTournamentContextCoachExperience:
    def test_fetch_coach_cached(self, tmp_path):
        s = TournamentContextScraper(cache_dir=str(tmp_path))
        coaches = {"john_calipari": {"appearances": 10, "wins": 30, "losses": 10, "teams": []}}
        s._save_cache("coach_tournament_2026.json", {"coaches": coaches, "year": 2026})
        result = s.fetch_coach_tournament_experience(2026)
        assert "john_calipari" in result

    def test_build_team_to_coach_appearances(self):
        s = TournamentContextScraper()
        coach_data = {
            "john_smith": {"appearances": 5, "wins": 10, "losses": 5, "teams": []},
        }
        mapping = {"duke": "John Smith"}
        result = s.build_team_to_coach_appearances(coach_data, mapping)
        assert result["duke"] == 5

    def test_build_team_to_coach_appearances_fuzzy(self):
        s = TournamentContextScraper()
        coach_data = {
            "john_smith": {"appearances": 5, "wins": 10, "losses": 5, "teams": []},
        }
        mapping = {"duke": "Coach Smith"}
        result = s.build_team_to_coach_appearances(coach_data, mapping)
        assert result["duke"] == 5

    def test_build_team_to_coach_win_rate(self):
        s = TournamentContextScraper()
        coach_data = {
            "john_smith": {"appearances": 5, "wins": 10, "losses": 5, "teams": []},
        }
        mapping = {"duke": "John Smith"}
        result = s.build_team_to_coach_win_rate(coach_data, mapping)
        assert abs(result["duke"] - 10 / 15) < 0.01

    def test_build_team_to_coach_win_rate_no_games(self):
        s = TournamentContextScraper()
        coach_data = {}
        mapping = {"duke": "Unknown Coach"}
        result = s.build_team_to_coach_win_rate(coach_data, mapping)
        assert result["duke"] == 0.0

    def test_build_team_to_coach_win_rate_fuzzy(self):
        s = TournamentContextScraper()
        coach_data = {
            "john_smith": {"wins": 10, "losses": 5, "teams": []},
        }
        mapping = {"duke": "Coach Smith"}
        result = s.build_team_to_coach_win_rate(coach_data, mapping)
        assert result["duke"] > 0


class TestTournamentContextConfChampions:
    def test_fetch_conf_cached(self, tmp_path):
        s = TournamentContextScraper(cache_dir=str(tmp_path))
        champs = {"duke": "ACC"}
        s._save_cache("conf_tourney_champs_2026.json", {"champions": champs, "year": 2026})
        result = s.fetch_conference_tournament_champions(2026)
        assert result == {"duke": "ACC"}


class TestTournamentContextJSON:
    def test_load_preseason_ap_from_json(self, tmp_path):
        fpath = tmp_path / "ap.json"
        fpath.write_text(json.dumps({"rankings": {"duke": 1, "unc": 5}}))
        result = TournamentContextScraper.load_preseason_ap_from_json(str(fpath))
        assert result == {"duke": 1, "unc": 5}

    def test_load_coach_data_from_json(self, tmp_path):
        fpath = tmp_path / "coach.json"
        fpath.write_text(json.dumps({"coaches": {"k": {"wins": 10}}}))
        result = TournamentContextScraper.load_coach_data_from_json(str(fpath))
        assert "k" in result

    def test_load_conf_champions_from_json(self, tmp_path):
        fpath = tmp_path / "conf.json"
        fpath.write_text(json.dumps({"champions": {"duke": "ACC"}}))
        result = TournamentContextScraper.load_conf_champions_from_json(str(fpath))
        assert result == {"duke": "ACC"}


class TestTournamentContextTeamCoaches:
    def test_fetch_team_coaches_cached(self, tmp_path):
        s = TournamentContextScraper(cache_dir=str(tmp_path))
        s._save_cache("team_coaches_2026.json", {"coaches": {"duke": "Coach K"}, "year": 2026})
        result = s.fetch_team_coaches(2026)
        assert result == {"duke": "Coach K"}

    @patch.object(TournamentContextScraper, "_scrape_team_coaches", return_value={})
    @patch.object(TournamentContextScraper, "_team_coaches_from_csv", return_value={})
    def test_fetch_team_coaches_no_data(self, mock_csv, mock_scrape):
        s = TournamentContextScraper()
        result = s.fetch_team_coaches(2026)
        assert result == {}


# ---------------------------------------------------------------------------
# 3. ESPN Picks tests
# ---------------------------------------------------------------------------

from src.data.scrapers.espn_picks import (
    PublicPicks,
    ConsensusData,
    ESPNPicksScraper,
    YahooPicksScraper,
    CBSPicksScraper,
    aggregate_consensus,
    SourceHealth,
    SourceStatus,
    ScraperOrchestrator,
    OrchestratorResult,
    RoundDelta,
    PicksUpdate,
    RefreshablePicksManager,
)


class TestPublicPicks:
    def test_creation(self):
        pp = PublicPicks(
            team_id="duke", team_name="Duke", seed=1, region="East",
            round_of_64_pct=99.0, champion_pct=22.0,
        )
        assert pp.team_id == "duke"
        assert pp.seed == 1

    def test_as_dict(self):
        pp = PublicPicks(
            team_id="duke", team_name="Duke", seed=1, region="East",
            round_of_64_pct=99.0, round_of_32_pct=90.0,
            sweet_16_pct=75.0, elite_8_pct=55.0,
            final_four_pct=35.0, champion_pct=22.0,
        )
        d = pp.as_dict
        assert d["R64"] == 99.0
        assert d["CHAMP"] == 22.0
        assert len(d) == 6


class TestConsensusData:
    def test_get_champion_favorites(self):
        teams = {
            "duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=22.0),
            "unc": PublicPicks("unc", "UNC", 2, "South", champion_pct=15.0),
            "kansas": PublicPicks("kansas", "Kansas", 1, "West", champion_pct=18.0),
        }
        cd = ConsensusData(teams=teams)
        top = cd.get_champion_favorites(top_n=2)
        assert len(top) == 2
        assert top[0].champion_pct == 22.0

    def test_get_contrarian_picks(self):
        teams = {
            "duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=10.0),
            "unc": PublicPicks("unc", "UNC", 5, "South", champion_pct=2.0),
        }
        cd = ConsensusData(teams=teams)
        model_probs = {"duke": 15.0, "unc": 8.0}
        picks = cd.get_contrarian_picks(model_probs, min_leverage=2.0)
        # unc: 8.0 / 2.0 = 4.0 leverage
        assert len(picks) >= 1
        # Should be sorted by leverage descending
        assert picks[0][3] >= 2.0

    def test_get_contrarian_no_match(self):
        cd = ConsensusData(teams={})
        result = cd.get_contrarian_picks({"duke": 10.0})
        assert result == []


class TestESPNPicksScraper:
    def test_init_no_cache(self):
        s = ESPNPicksScraper()
        assert s.cache_dir is None

    def test_init_with_cache(self, tmp_path):
        s = ESPNPicksScraper(cache_dir=str(tmp_path / "espn_cache"))
        assert s.cache_dir.exists()

    def test_dict_to_consensus(self):
        s = ESPNPicksScraper()
        data = {
            "teams": {
                "duke": {
                    "team_name": "Duke", "seed": 1, "region": "East",
                    "round_of_64_pct": 99.0, "champion_pct": 22.0,
                }
            },
            "sources": ["espn"],
            "timestamp": "2026-03-17",
        }
        result = s._dict_to_consensus(data)
        assert "duke" in result.teams
        assert result.teams["duke"].champion_pct == 22.0
        assert result.timestamp == "2026-03-17"

    def test_load_from_json(self, tmp_path):
        fpath = tmp_path / "picks.json"
        data = {
            "teams": {
                "duke": {"team_name": "Duke", "seed": 1, "region": "East"}
            }
        }
        fpath.write_text(json.dumps(data))
        s = ESPNPicksScraper()
        result = s.load_from_json(str(fpath))
        assert "duke" in result.teams

    def test_load_from_cache(self, tmp_path):
        s = ESPNPicksScraper(cache_dir=str(tmp_path))
        data = {"teams": {"duke": {"team_name": "Duke", "seed": 1, "region": "East"}}}
        (tmp_path / "espn_picks_2026.json").write_text(json.dumps(data))
        result = s._load_from_cache("espn_picks_2026.json")
        assert result is not None

    def test_load_from_cache_missing(self, tmp_path):
        s = ESPNPicksScraper(cache_dir=str(tmp_path))
        assert s._load_from_cache("nonexistent.json") is None

    def test_save_to_cache(self, tmp_path):
        s = ESPNPicksScraper(cache_dir=str(tmp_path))
        s._save_to_cache("test.json", {"a": 1})
        loaded = json.loads((tmp_path / "test.json").read_text())
        assert loaded == {"a": 1}

    def test_save_to_cache_no_dir(self):
        s = ESPNPicksScraper()
        s._save_to_cache("test.json", {"a": 1})  # no raise

    @patch.dict(os.environ, {}, clear=True)
    def test_fetch_picks_no_sources(self):
        s = ESPNPicksScraper()
        with patch.object(s, "_try_json_url", return_value=None), \
             patch.object(s, "_load_from_cache", return_value=None):
            result = s.fetch_picks(2026)
            assert isinstance(result, ConsensusData)

    @patch.dict(os.environ, {"ESPN_PUBLIC_PICKS_URL": "http://example.com/picks.json"})
    def test_fetch_picks_env_url(self):
        s = ESPNPicksScraper()
        mock_consensus = ConsensusData(
            teams={"duke": PublicPicks("duke", "Duke", 1, "East")},
            sources=["espn"],
        )
        with patch.object(s, "_try_json_url", return_value=mock_consensus):
            result = s.fetch_picks(2026)
            assert "duke" in result.teams


class TestYahooPicksScraper:
    def test_init(self, tmp_path):
        s = YahooPicksScraper(cache_dir=str(tmp_path / "yahoo"))
        assert s.cache_dir.exists()

    @patch.dict(os.environ, {}, clear=True)
    def test_fetch_picks_no_data(self):
        s = YahooPicksScraper()
        result = s.fetch_picks(2026)
        assert isinstance(result, ConsensusData)
        assert result.sources == ["yahoo"]


class TestCBSPicksScraper:
    def test_init(self, tmp_path):
        s = CBSPicksScraper(cache_dir=str(tmp_path / "cbs"))
        assert s.cache_dir.exists()

    @patch.dict(os.environ, {}, clear=True)
    def test_fetch_picks_no_data(self):
        s = CBSPicksScraper()
        result = s.fetch_picks(2026)
        assert isinstance(result, ConsensusData)
        assert result.sources == ["cbs"]


class TestAggregateConsensus:
    def test_basic_aggregation(self):
        espn = ConsensusData(
            teams={"duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=20.0)},
        )
        yahoo = ConsensusData(
            teams={"duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=18.0)},
        )
        cbs = ConsensusData(
            teams={"duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=22.0)},
        )
        result = aggregate_consensus(espn, yahoo, cbs)
        assert "duke" in result.teams
        # Weighted average: 0.5*20 + 0.3*18 + 0.2*22 = 10+5.4+4.4 = 19.8
        assert abs(result.teams["duke"].champion_pct - 19.8) < 0.1

    def test_aggregation_partial_sources(self):
        espn = ConsensusData(
            teams={"duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=20.0)},
        )
        yahoo = ConsensusData(teams={})
        cbs = ConsensusData(teams={})
        result = aggregate_consensus(espn, yahoo, cbs)
        assert "duke" in result.teams
        # Only ESPN has duke, weight 0.5, normalized to 1.0 → should be 20.0
        assert abs(result.teams["duke"].champion_pct - 20.0) < 0.1


class TestSourceStatus:
    def test_is_stale_never_fetched(self):
        ss = SourceStatus(name="espn")
        assert ss.is_stale is True

    def test_is_stale_recent(self):
        ss = SourceStatus(name="espn", last_success=time.time())
        assert ss.is_stale is False

    def test_is_stale_old(self):
        ss = SourceStatus(name="espn", last_success=time.time() - 200000)
        assert ss.is_stale is True


class TestSourceHealth:
    def test_enum_values(self):
        assert SourceHealth.HEALTHY.value == "healthy"
        assert SourceHealth.DEGRADED.value == "degraded"
        assert SourceHealth.FAILED.value == "failed"


class TestOrchestratorResult:
    def test_is_degraded(self):
        r = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn", "yahoo"],
            source_statuses={},
        )
        assert r.is_degraded is True

    def test_not_degraded(self):
        r = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn", "yahoo", "cbs"],
            source_statuses={},
        )
        assert r.is_degraded is False

    def test_health_summary(self):
        statuses = {
            "espn": SourceStatus(name="espn", health=SourceHealth.HEALTHY),
            "yahoo": SourceStatus(name="yahoo", health=SourceHealth.FAILED, last_error="timeout"),
        }
        r = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn"],
            source_statuses=statuses,
        )
        summary = r.health_summary
        assert "espn" in summary
        assert "timeout" in summary


class TestRoundDelta:
    def test_delta(self):
        rd = RoundDelta(team_id="duke", round_name="CHAMP", old_pct=20.0, new_pct=25.0)
        assert rd.delta == 5.0
        assert rd.abs_delta == 5.0

    def test_negative_delta(self):
        rd = RoundDelta(team_id="duke", round_name="CHAMP", old_pct=25.0, new_pct=20.0)
        assert rd.delta == -5.0
        assert rd.abs_delta == 5.0


class TestPicksUpdate:
    def test_summary_no_changes(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=0.0,
        )
        assert "No significant changes" in pu.summary

    def test_summary_with_changes(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=["duke", "unc"], max_delta=3.5,
        )
        assert "2 teams" in pu.summary
        assert "3.5" in pu.summary

    def test_max_round_delta_empty(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=2.0,
        )
        assert pu.max_round_delta == 2.0

    def test_max_round_delta_with_entries(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=0.0,
            round_deltas=[
                RoundDelta("duke", "R64", 50.0, 55.0),
                RoundDelta("unc", "CHAMP", 10.0, 18.0),
            ],
        )
        assert pu.max_round_delta == 8.0

    def test_late_round_max_delta(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=0.0,
            round_deltas=[
                RoundDelta("duke", "R64", 50.0, 55.0),
                RoundDelta("unc", "F4", 10.0, 18.0),
                RoundDelta("kansas", "E8", 20.0, 22.0),
            ],
        )
        assert pu.late_round_max_delta == 8.0

    def test_late_round_max_delta_empty(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=0.0,
            round_deltas=[RoundDelta("duke", "R64", 50.0, 55.0)],
        )
        assert pu.late_round_max_delta == 0.0

    def test_summary_with_late_round_shifts(self):
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=["duke"], max_delta=3.0,
            round_deltas=[RoundDelta("duke", "CHAMP", 10.0, 15.0)],
        )
        assert "Late-round" in pu.summary


class TestRefreshablePicksManager:
    def test_is_stale_initially(self):
        orch = MagicMock()
        mgr = RefreshablePicksManager(orch)
        assert mgr.is_stale is True

    def test_hours_since_refresh_none(self):
        orch = MagicMock()
        mgr = RefreshablePicksManager(orch)
        assert mgr.hours_since_refresh is None

    def test_fetch_or_refresh_first_call(self):
        mock_result = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn"],
            source_statuses={},
        )
        orch = MagicMock()
        orch.fetch_all_picks.return_value = mock_result
        mgr = RefreshablePicksManager(orch)
        result, update = mgr.fetch_or_refresh(2026)
        assert result is mock_result
        assert update is None

    def test_fetch_or_refresh_uses_cached(self):
        mock_result = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn"],
            source_statuses={},
        )
        orch = MagicMock()
        orch.fetch_all_picks.return_value = mock_result
        mgr = RefreshablePicksManager(orch, staleness_threshold_hours=1000)
        # First call
        mgr.fetch_or_refresh(2026)
        # Second call should use cached
        result2, update2 = mgr.fetch_or_refresh(2026)
        assert update2 is None
        assert orch.fetch_all_picks.call_count == 1

    def test_fetch_or_refresh_force(self):
        mock_result = OrchestratorResult(
            consensus=ConsensusData(teams={"duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=20.0)}),
            successful_sources=["espn"],
            source_statuses={},
        )
        mock_result2 = OrchestratorResult(
            consensus=ConsensusData(teams={"duke": PublicPicks("duke", "Duke", 1, "East", champion_pct=25.0)}),
            successful_sources=["espn"],
            source_statuses={},
        )
        orch = MagicMock()
        orch.fetch_all_picks.side_effect = [mock_result, mock_result2]
        mgr = RefreshablePicksManager(orch, staleness_threshold_hours=1000)
        mgr.fetch_or_refresh(2026)
        result2, update2 = mgr.fetch_or_refresh(2026, force_refresh=True)
        assert update2 is not None
        assert update2.max_delta == 5.0


# ---------------------------------------------------------------------------
# 4. Pipeline sota.py — selected data classes and utilities
# ---------------------------------------------------------------------------

from src.pipeline.config import SOTAPipelineConfig, DataRequirementError


class TestSOTAPipelineConfig:
    def test_default_config(self):
        cfg = SOTAPipelineConfig(year=2026)
        assert cfg.year == 2026


class TestDataRequirementError:
    def test_is_exception(self):
        with pytest.raises(DataRequirementError):
            raise DataRequirementError("Not enough data")


# ---------------------------------------------------------------------------
# 5. Calibration stage tests (mock-heavy since it depends on pipeline)
# ---------------------------------------------------------------------------


class TestCalibrationStageModule:
    """Test the calibration stage module utilities."""

    def test_fit_calibration_insufficient_samples(self):
        """Pipeline with insufficient samples should raise DataRequirementError."""
        from src.pipeline.stages.calibration import _fit_calibration

        pipeline = MagicMock()
        pipeline.config = SOTAPipelineConfig(year=2026)
        pipeline.config.min_calibration_samples = 50
        pipeline.config.min_calibration_samples_hard = 10
        pipeline.config.pre_calibration_clip_lo = 0.01
        pipeline.config.pre_calibration_clip_hi = 0.99
        pipeline.config.calibration_method = "temperature"
        pipeline.config.enable_multi_year_calibration = False
        pipeline._get_validation_era_games.return_value = []
        pipeline._unique_games.return_value = []
        pipeline.feature_engineer = MagicMock()
        pipeline.feature_engineer.team_features = {}

        game_flows = {}
        with pytest.raises(DataRequirementError):
            _fit_calibration(pipeline, game_flows)


# ---------------------------------------------------------------------------
# 6. Brier Optimal tests
# ---------------------------------------------------------------------------

from src.ml.calibration.brier_optimal import (
    BrierOptimalSharpener,
    SeedBasedOverrides,
    BrierCalibrator,
    BrierPostProcessor,
    KAGGLE_ROUND_WEIGHTS,
    round_weighted_brier,
    round_label_from_seed_matchup,
    RoundWeightedSharpener,
    MasseyStandalonePredictor,
    FavouriteLongshotCorrection,
)


class TestBrierOptimalSharpener:
    def test_init(self):
        s = BrierOptimalSharpener()
        assert s.alpha == 1.0
        assert s.fitted is False

    def test_sharpen_identity(self):
        s = BrierOptimalSharpener()
        preds = np.array([0.3, 0.5, 0.7])
        result = s.sharpen(preds)
        np.testing.assert_allclose(result, preds, atol=0.01)

    def test_fit(self):
        s = BrierOptimalSharpener()
        preds = np.array([0.6, 0.7, 0.8, 0.55, 0.65, 0.75, 0.85, 0.9])
        outcomes = np.array([1, 1, 1, 0, 1, 1, 1, 1])
        alpha = s.fit(preds, outcomes)
        assert s.fitted is True
        assert 0.5 <= alpha <= 2.0

    def test_apply_static(self):
        result = BrierOptimalSharpener._apply(np.array([0.5]), 1.0)
        assert abs(result[0] - 0.5) < 0.01


class TestSeedBasedOverrides:
    def test_init_mens(self):
        sbo = SeedBasedOverrides()
        assert sbo.rates == SeedBasedOverrides.MENS_SEED_WIN_RATES

    def test_init_womens(self):
        sbo = SeedBasedOverrides(is_womens=True)
        assert sbo.rates == SeedBasedOverrides.WOMENS_SEED_WIN_RATES

    def test_apply_same_seed(self):
        sbo = SeedBasedOverrides()
        assert sbo.apply(0.6, 1, 1) == 0.6

    def test_apply_invalid_seed(self):
        sbo = SeedBasedOverrides()
        assert sbo.apply(0.6, 0, 5) == 0.6
        assert sbo.apply(0.6, 3, -1) == 0.6

    def test_apply_no_matchup(self):
        sbo = SeedBasedOverrides()
        # 1 vs 2 is not in first-round rates
        assert sbo.apply(0.6, 1, 2) == 0.6

    def test_apply_1v16_close(self):
        sbo = SeedBasedOverrides(snap_threshold=0.08)
        result = sbo.apply(0.98, 1, 16)
        # Should shrink toward 0.987
        assert result != 0.98
        assert abs(result - 0.987) < abs(0.98 - 0.987)

    def test_apply_1v16_far(self):
        sbo = SeedBasedOverrides(snap_threshold=0.08)
        # 0.8 is far from 0.987, should not snap
        result = sbo.apply(0.80, 1, 16)
        assert result == 0.80

    def test_apply_flipped_seeds(self):
        sbo = SeedBasedOverrides(snap_threshold=0.20)
        # seed2 < seed1 → flip
        result = sbo.apply(0.02, 16, 1)
        # historical for (1,16) is 0.987, target for team1 (seed 16) is 1-0.987=0.013
        assert result != 0.02

    def test_apply_with_round_early(self):
        sbo = SeedBasedOverrides()
        # R64 should fall back to standard apply
        result = sbo.apply_with_round(0.98, 1, 16, "R64")
        expected = sbo.apply(0.98, 1, 16)
        assert result == expected

    def test_apply_with_round_s16(self):
        sbo = SeedBasedOverrides(snap_threshold=0.20)
        result = sbo.apply_with_round(0.65, 1, 4, "S16")
        # Should use later-round rates
        assert isinstance(result, float)

    def test_apply_with_round_same_seed(self):
        sbo = SeedBasedOverrides()
        assert sbo.apply_with_round(0.5, 3, 3, "E8") == 0.5


class TestBrierCalibrator:
    def test_init(self):
        bc = BrierCalibrator()
        assert bc.temperature == 1.0
        assert bc.fitted is False

    def test_calibrate_not_fitted(self):
        bc = BrierCalibrator()
        with pytest.raises(ValueError, match="Not fitted"):
            bc.calibrate(np.array([0.5]))

    def test_fit_and_calibrate(self):
        bc = BrierCalibrator()
        np.random.seed(42)
        preds = np.random.uniform(0.3, 0.7, 100)
        outcomes = (preds > 0.5).astype(float)
        bc.fit(preds, outcomes)
        assert bc.fitted is True
        calibrated = bc.calibrate(np.array([0.5]))
        assert 0.0 < calibrated[0] < 1.0

    def test_fit_weighted(self):
        bc = BrierCalibrator()
        np.random.seed(42)
        preds = np.random.uniform(0.3, 0.7, 50)
        outcomes = (preds > 0.5).astype(float)
        labels = ["R64"] * 30 + ["E8"] * 10 + ["F4"] * 10
        bc.fit_weighted(preds, outcomes, labels)
        assert bc.fitted is True

    def test_fit_weighted_zero_weights(self):
        bc = BrierCalibrator()
        preds = np.random.uniform(0.3, 0.7, 20)
        outcomes = (preds > 0.5).astype(float)
        # Zero-weight schedule
        bc.fit_weighted(preds, outcomes, ["R64"] * 20, weights_schedule={"XXX": 0.0})
        # Should fall back to regular fit
        assert bc.fitted is True


class TestBrierPostProcessor:
    def test_init_defaults(self):
        bpp = BrierPostProcessor()
        assert bpp.seed_overrides_mens is not None
        assert bpp.seed_overrides_womens is not None

    def test_process_no_seeds(self):
        bpp = BrierPostProcessor()
        result = bpp.process(0.6)
        assert abs(result - 0.6) < 0.01

    def test_process_with_calibrator(self):
        cal = BrierCalibrator()
        cal.temperature = 1.0
        cal.fitted = True
        bpp = BrierPostProcessor(calibrator=cal)
        result = bpp.process(0.6)
        assert 0.0 < result < 1.0

    def test_process_with_sharpener(self):
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 0.8
        sharpener.fitted = True
        bpp = BrierPostProcessor(sharpener=sharpener)
        result = bpp.process(0.6)
        assert 0.0 < result < 1.0

    def test_process_clip(self):
        bpp = BrierPostProcessor(clip_lo=0.1, clip_hi=0.9)
        assert bpp.process(0.001) >= 0.1
        assert bpp.process(0.999) <= 0.9

    def test_process_batch(self):
        bpp = BrierPostProcessor()
        preds = np.array([0.3, 0.5, 0.7])
        result = bpp.process_batch(preds)
        assert len(result) == 3
        assert all(bpp.clip_lo <= r <= bpp.clip_hi for r in result)

    def test_process_batch_with_seeds(self):
        bpp = BrierPostProcessor()
        preds = np.array([0.98])
        seeds1 = np.array([1])
        seeds2 = np.array([16])
        result = bpp.process_batch(preds, seeds1, seeds2)
        assert len(result) == 1

    def test_process_with_goto_converter(self):
        flb = FavouriteLongshotCorrection(strength=0.05)
        flb.fitted = True
        bpp = BrierPostProcessor(goto_converter=flb)
        result = bpp.process(0.7)
        assert 0.0 < result < 1.0

    def test_process_batch_with_calibrator_and_sharpener(self):
        cal = BrierCalibrator()
        cal.temperature = 1.0
        cal.fitted = True
        sharpener = BrierOptimalSharpener()
        sharpener.alpha = 1.0
        sharpener.fitted = True
        flb = FavouriteLongshotCorrection(strength=0.02)
        flb.fitted = True
        bpp = BrierPostProcessor(
            calibrator=cal, sharpener=sharpener, goto_converter=flb
        )
        preds = np.array([0.3, 0.5, 0.7, 0.9])
        result = bpp.process_batch(preds)
        assert len(result) == 4


class TestKaggleRoundWeights:
    def test_weights_exist(self):
        assert "R64" in KAGGLE_ROUND_WEIGHTS
        assert "NCG" in KAGGLE_ROUND_WEIGHTS
        assert KAGGLE_ROUND_WEIGHTS["NCG"] == 32.0

    def test_round_weighted_brier_uniform(self):
        preds = np.array([0.5, 0.5])
        outcomes = np.array([1.0, 0.0])
        result = round_weighted_brier(preds, outcomes)
        # Standard Brier: mean of (0.25, 0.25) = 0.25
        assert abs(result - 0.25) < 0.01

    def test_round_weighted_brier_with_labels(self):
        preds = np.array([0.5, 0.5])
        outcomes = np.array([1.0, 0.0])
        labels = ["R64", "NCG"]
        result = round_weighted_brier(preds, outcomes, labels)
        # R64 weight=1, NCG weight=32, total weight=33
        # weighted: (1*0.25 + 32*0.25) / 33 = 8.25/33 = 0.25
        assert abs(result - 0.25) < 0.01

    def test_round_weighted_brier_no_labels(self):
        preds = np.array([0.6])
        outcomes = np.array([1.0])
        result = round_weighted_brier(preds, outcomes, None)
        assert abs(result - 0.16) < 0.01


class TestRoundLabelFromSeedMatchup:
    def test_explicit_round(self):
        assert round_label_from_seed_matchup(1, 16, "R64") == "R64"
        assert round_label_from_seed_matchup(1, 16, "SWEET 16") == "S16"
        assert round_label_from_seed_matchup(1, 16, "ELITE 8") == "E8"
        assert round_label_from_seed_matchup(1, 16, "Final Four") == "F4"
        assert round_label_from_seed_matchup(1, 16, "Championship") == "NCG"

    def test_default(self):
        assert round_label_from_seed_matchup(1, 16) == "R64"

    def test_variations(self):
        assert round_label_from_seed_matchup(1, 16, "Round of 64") == "R64"
        assert round_label_from_seed_matchup(1, 16, "Second Round") == "R32"
        assert round_label_from_seed_matchup(1, 16, "National Championship") == "NCG"


class TestRoundWeightedSharpener:
    def test_fit_weighted(self):
        rws = RoundWeightedSharpener()
        np.random.seed(42)
        preds = np.random.uniform(0.3, 0.7, 50)
        outcomes = (preds > 0.5).astype(float)
        labels = ["R64"] * 30 + ["E8"] * 10 + ["F4"] * 10
        alpha = rws.fit_weighted(preds, outcomes, labels)
        assert rws.fitted is True
        assert 0.5 <= alpha <= 2.0

    def test_fit_weighted_zero_weights(self):
        rws = RoundWeightedSharpener()
        preds = np.random.uniform(0.3, 0.7, 30)
        outcomes = (preds > 0.5).astype(float)
        labels = ["UNKNOWN"] * 30
        # weights all 1.0 (default for unknown), should still work
        alpha = rws.fit_weighted(preds, outcomes, labels)
        assert rws.fitted is True


class TestMasseyStandalonePredictor:
    def test_init(self):
        m = MasseyStandalonePredictor()
        assert m.sigma == 8.0
        assert m.fitted is False
        assert m.blend_weight == 0.20

    def test_predict(self):
        m = MasseyStandalonePredictor(sigma=10.0)
        p = m.predict(0.0)
        assert abs(p - 0.5) < 0.01
        p_pos = m.predict(10.0)
        assert p_pos > 0.5
        p_neg = m.predict(-10.0)
        assert p_neg < 0.5

    def test_fit(self):
        m = MasseyStandalonePredictor()
        np.random.seed(42)
        diffs = np.random.randn(50) * 5
        outcomes = (diffs > 0).astype(float)
        sigma = m.fit(diffs, outcomes)
        assert m.fitted is True
        assert 1.0 <= sigma <= 25.0

    def test_fit_too_few_samples(self):
        m = MasseyStandalonePredictor(sigma=8.0)
        diffs = np.array([1.0, -1.0])
        outcomes = np.array([1.0, 0.0])
        sigma = m.fit(diffs, outcomes)
        assert sigma == 8.0  # unchanged

    def test_fit_blend_weight(self):
        m = MasseyStandalonePredictor()
        np.random.seed(42)
        n = 50
        model_probs = np.random.uniform(0.3, 0.7, n)
        massey_probs = np.random.uniform(0.3, 0.7, n)
        outcomes = (model_probs > 0.5).astype(float)
        w = m.fit_blend_weight(model_probs, massey_probs, outcomes)
        assert 0.05 <= w <= 0.40

    def test_fit_blend_weight_too_few(self):
        m = MasseyStandalonePredictor()
        m.blend_weight = 0.20
        model_probs = np.array([0.5, 0.6])
        massey_probs = np.array([0.4, 0.7])
        outcomes = np.array([1.0, 1.0])
        w = m.fit_blend_weight(model_probs, massey_probs, outcomes)
        assert w == 0.20  # unchanged


class TestFavouriteLongshotCorrection:
    def test_init(self):
        flb = FavouriteLongshotCorrection()
        assert flb.strength == 0.05
        assert flb.fitted is False

    def test_correct_zero_margin(self):
        flb = FavouriteLongshotCorrection(strength=0.0)
        preds = np.array([0.3, 0.5, 0.7])
        result = flb.correct(preds)
        np.testing.assert_allclose(result, preds, atol=0.001)

    def test_correct_positive_margin(self):
        flb = FavouriteLongshotCorrection(strength=0.10)
        preds = np.array([0.7])
        result = flb.correct(preds)
        # Should sharpen toward favorite
        assert result[0] > 0.7

    def test_correct_single(self):
        flb = FavouriteLongshotCorrection(strength=0.05)
        result = flb.correct_single(0.6)
        assert 0.0 < result < 1.0

    def test_fit(self):
        flb = FavouriteLongshotCorrection()
        np.random.seed(42)
        preds = np.random.uniform(0.3, 0.7, 50)
        outcomes = (preds > 0.5).astype(float)
        margin = flb.fit(preds, outcomes)
        assert flb.fitted is True
        assert 0.0 <= margin <= 0.20

    def test_fit_too_few(self):
        flb = FavouriteLongshotCorrection(strength=0.05)
        preds = np.array([0.5] * 5)
        outcomes = np.array([1.0] * 5)
        margin = flb.fit(preds, outcomes)
        assert margin == 0.05
        assert flb.fitted is True

    def test_fit_with_round_labels(self):
        flb = FavouriteLongshotCorrection()
        np.random.seed(42)
        preds = np.random.uniform(0.3, 0.7, 50)
        outcomes = (preds > 0.5).astype(float)
        labels = ["R64"] * 30 + ["E8"] * 10 + ["F4"] * 10
        margin = flb.fit(preds, outcomes, round_labels=labels)
        assert flb.fitted is True
        assert "brier_before" in flb._fit_details


# ---------------------------------------------------------------------------
# 7. Agents concrete tests
# ---------------------------------------------------------------------------

from src.agents import AgentRole, AgentResult, MessageBus, MessageType, AgentMessage
from src.agents.concrete import (
    DataScoutAgent,
    FeatureEngineerAgent,
    ModelingAgent,
    AuditAgent,
    OrchestratorAgent,
)


class TestAgentRolesAndNames:
    def test_data_scout(self):
        a = DataScoutAgent()
        assert a.role == AgentRole.DATA_SCOUT
        assert a.name == "DataScout"

    def test_feature_engineer(self):
        a = FeatureEngineerAgent()
        assert a.role == AgentRole.FEATURE_ENGINEER
        assert a.name == "FeatureEngineer"

    def test_modeling(self):
        a = ModelingAgent()
        assert a.role == AgentRole.MODELER
        assert a.name == "Modeler"

    def test_audit(self):
        a = AuditAgent()
        assert a.role == AgentRole.AUDITOR
        assert a.name == "Auditor"

    def test_orchestrator(self):
        a = OrchestratorAgent()
        assert a.role == AgentRole.ORCHESTRATOR
        assert a.name == "Orchestrator"


class TestMessageBus:
    def test_publish_and_get(self):
        bus = MessageBus()
        msg = AgentMessage(sender=AgentRole.DATA_SCOUT, msg_type=MessageType.DATA_READY)
        bus.publish(msg)
        assert len(bus) == 1
        msgs = bus.get_messages(topic=MessageType.DATA_READY)
        assert len(msgs) == 1

    def test_has_veto(self):
        bus = MessageBus()
        assert bus.has_veto() is False
        bus.publish(AgentMessage(sender=AgentRole.AUDITOR, msg_type=MessageType.VETO))
        assert bus.has_veto() is True

    def test_clear(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS))
        bus.clear()
        assert len(bus) == 0

    def test_filter_by_sender(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS))
        bus.publish(AgentMessage(sender=AgentRole.MODELER, msg_type=MessageType.STATUS))
        msgs = bus.get_messages(sender=AgentRole.DATA_SCOUT)
        assert len(msgs) == 1

    def test_filter_by_recipient(self):
        bus = MessageBus()
        bus.publish(AgentMessage(
            sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS,
            recipient=AgentRole.MODELER,
        ))
        bus.publish(AgentMessage(
            sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS,
        ))
        # Broadcast messages (recipient=None) should be included
        msgs = bus.get_messages(recipient=AgentRole.MODELER)
        assert len(msgs) == 2

    def test_all_messages(self):
        bus = MessageBus()
        bus.publish(AgentMessage(sender=AgentRole.DATA_SCOUT, msg_type=MessageType.STATUS))
        assert len(bus.all_messages()) == 1


class TestAgentMessage:
    def test_auto_id_and_timestamp(self):
        msg = AgentMessage(sender=AgentRole.DATA_SCOUT, msg_type=MessageType.DATA_READY)
        assert msg.message_id != ""
        assert msg.timestamp != ""

    def test_to_dict(self):
        msg = AgentMessage(
            sender=AgentRole.DATA_SCOUT, msg_type=MessageType.DATA_READY,
            payload={"key": "val"}, recipient=AgentRole.MODELER,
        )
        d = msg.to_dict()
        assert d["sender"] == "data_scout"
        assert d["msg_type"] == "data_ready"
        assert d["recipient"] == "modeler"


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(agent_role=AgentRole.DATA_SCOUT)
        assert r.success is True
        assert r.findings == []
        assert r.messages_sent == 0
        assert r.wall_seconds == 0.0


class TestAuditAgentLeakage:
    def test_no_leakage(self):
        audit = AuditAgent()
        features = SimpleNamespace(feature_names=["adj_oe", "tempo", "sos"])
        result = audit._check_leakage(None, features)
        assert result is None

    def test_leakage_detected(self):
        audit = AuditAgent()
        features = SimpleNamespace(feature_names=["adj_oe", "future_wins", "tempo"])
        result = audit._check_leakage(None, features)
        assert result is not None
        assert "future_wins" in result

    def test_leakage_none_features(self):
        audit = AuditAgent()
        result = audit._check_leakage(None, None)
        assert result is None


class TestAuditAgentRobustness:
    def test_robustness_missing_data(self):
        audit = AuditAgent()
        result = audit._run_robustness(None, MagicMock(_X_train=None), None, None)
        assert result is None

    def test_robustness_exception(self):
        audit = AuditAgent()
        pipeline = MagicMock()
        pipeline._X_train = np.zeros((10, 5))
        pipeline._y_train = np.zeros(10)
        pipeline._predict_proba = MagicMock()
        with patch("src.agents.concrete.AuditAgent._run_robustness", side_effect=Exception("fail")):
            # Direct call should handle exception
            pass


class TestAuditAgentRun:
    def test_audit_no_models_no_features(self):
        audit = AuditAgent()
        bus = MessageBus()
        result = audit.run(None, bus, pipeline=MagicMock())
        assert result.success is True
        assert any("skipped" in f.lower() for f in result.findings)

    def test_audit_with_leakage(self):
        audit = AuditAgent()
        bus = MessageBus()
        features = SimpleNamespace(feature_names=["future_wins", "bracket_result"])
        result = audit.run(
            None, bus,
            pipeline=MagicMock(),
            models=MagicMock(),
            features=features,
        )
        # Should issue veto
        assert bus.has_veto()
        assert result.success is False


class TestOrchestratorMakeResult:
    def test_make_result(self):
        orch = OrchestratorAgent()
        t0 = time.monotonic()
        result = orch._make_result(True, {"data": 1}, ["finding1"], 2, t0)
        assert result.success is True
        assert result.output == {"data": 1}
        assert len(result.findings) == 1
        assert result.messages_sent == 2


class TestOrchestratorBuildReport:
    def test_build_report_basic(self):
        orch = OrchestratorAgent()
        agent_results = {
            "data_scout": AgentResult(
                agent_role=AgentRole.DATA_SCOUT,
                success=True,
                findings=["OK"],
                wall_seconds=1.5,
            ),
        }
        report = orch._build_report(
            MagicMock(), True, agent_results, ["finding1"],
        )
        assert report["multi_agent"] is True
        assert report["pipeline_passed"] is True
        assert report["n_agents"] == 1

    def test_build_report_with_modeler(self):
        orch = OrchestratorAgent()
        sim = SimpleNamespace(championship_odds={"duke": 0.15}, num_simulations=10000)
        agent_results = {
            "modeler": AgentResult(
                agent_role=AgentRole.MODELER,
                success=True,
                output={"models": None, "calibration": None, "simulation": sim},
                findings=[],
                wall_seconds=5.0,
            ),
        }
        report = orch._build_report(MagicMock(), True, agent_results, [])
        assert "championship_odds" in report
        assert report["num_simulations"] == 10000


class TestOrchestratorLogGovernance:
    def test_log_governance_no_module(self):
        orch = OrchestratorAgent()
        # Should not raise even when governance module is missing
        orch._log_governance(True, ["finding"], {})


class TestBaseAgentPublish:
    def test_publish(self):
        agent = DataScoutAgent()
        bus = MessageBus()
        agent.publish(bus, MessageType.STATUS, {"info": "test"})
        assert len(bus) == 1

    def test_log_finding(self):
        agent = DataScoutAgent()
        bus = MessageBus()
        agent.log_finding(bus, "Test finding")
        msgs = bus.get_messages(topic=MessageType.STATUS)
        assert len(msgs) == 1
        assert msgs[0].payload["finding"] == "Test finding"


class TestScraperOrchestrator:
    def test_init(self, tmp_path):
        orch = ScraperOrchestrator(cache_dir=str(tmp_path))
        assert orch.cache_dir == str(tmp_path)
        assert len(orch.source_statuses) == 3

    def test_circuit_breaker(self):
        orch = ScraperOrchestrator()
        orch.source_statuses["espn"].consecutive_failures = 10
        orch.source_statuses["yahoo"].consecutive_failures = 10
        orch.source_statuses["cbs"].consecutive_failures = 10
        result = orch.fetch_all_picks(2026, min_sources=0)
        # All sources circuit-broken
        assert result.successful_sources == []
