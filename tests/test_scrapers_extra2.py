"""Comprehensive tests for scraper modules not covered by test_scrapers_coverage.py.

Covers:
- _retry.py (retry_request, rate_limited_call)
- espn_picks.py (PublicPicks, ConsensusData, ESPNPicksScraper, YahooPicksScraper,
                  CBSPicksScraper, aggregate_consensus, SourceHealth, SourceStatus,
                  ScraperOrchestrator, OrchestratorResult, RoundDelta, PicksUpdate,
                  RefreshablePicksManager)
- injury_report.py (InjuryReport, TeamInjuryReport, InjuryReportScraper,
                     InjurySeverityEstimator, PositionalDepthChart,
                     apply_injury_reports_to_roster)
- roster_enrichment.py (RosterEnrichment)
- bracket_ingestion.py (BracketTeam, TournamentBracketData, BracketIngestionPipeline)
- cbbpy_rosters.py (CBBpyRosterScraper static helpers)
- tournament_results.py (TournamentResultsScraper)
- womens/herhoopstats.py (WomensTeamStats, HerHoopStatsScraper, _estimate_from_seed)
- womens/historical_results.py (HistoricalTournamentGame, WomensHistoricalResults)
- womens/ncaa_net.py (NETRanking, WomensNETScraper)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ======================================================================
# 1. _retry.py
# ======================================================================


class TestRetryRequest:
    """Tests for retry_request function."""

    def test_success_on_first_attempt(self):
        from src.data.scrapers._retry import retry_request
        mock_func = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_func.return_value = mock_resp
        result = retry_request(mock_func, "http://example.com", max_retries=2)
        assert result is mock_resp
        assert mock_func.call_count == 1

    @patch("src.data.scrapers._retry.time.sleep")
    def test_retry_on_429(self, mock_sleep):
        from src.data.scrapers._retry import retry_request
        mock_func = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {}
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_func.side_effect = [resp_429, resp_200]
        result = retry_request(mock_func, max_retries=2)
        assert result is resp_200
        assert mock_sleep.called

    @patch("src.data.scrapers._retry.time.sleep")
    def test_retry_on_500(self, mock_sleep):
        from src.data.scrapers._retry import retry_request
        mock_func = MagicMock()
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.headers = {}
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_func.side_effect = [resp_500, resp_200]
        result = retry_request(mock_func, max_retries=2)
        assert result is resp_200

    @patch("src.data.scrapers._retry.time.sleep")
    def test_retry_respects_retry_after_header(self, mock_sleep):
        from src.data.scrapers._retry import retry_request
        mock_func = MagicMock()
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "10"}
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_func.side_effect = [resp_429, resp_200]
        retry_request(mock_func, max_retries=2)
        # Sleep should use at least the Retry-After value
        call_args = mock_sleep.call_args[0][0]
        assert call_args >= 10.0

    @patch("src.data.scrapers._retry.time.sleep")
    def test_retry_exhausted_raises(self, mock_sleep):
        import requests
        from src.data.scrapers._retry import retry_request
        mock_func = MagicMock()
        mock_func.side_effect = requests.exceptions.ConnectionError("fail")
        with pytest.raises(requests.exceptions.ConnectionError):
            retry_request(mock_func, max_retries=1)

    def test_defaults_are_set(self):
        from src.data.scrapers._retry import (
            DEFAULT_MAX_RETRIES, DEFAULT_BACKOFF_BASE, DEFAULT_RATE_LIMIT_DELAY,
        )
        assert DEFAULT_MAX_RETRIES == 3
        assert DEFAULT_BACKOFF_BASE == 2.0
        assert DEFAULT_RATE_LIMIT_DELAY == 3.0


class TestRateLimitedCall:
    """Tests for rate_limited_call function."""

    @patch("src.data.scrapers._retry.time.sleep")
    def test_success_returns_result(self, mock_sleep):
        from src.data.scrapers._retry import rate_limited_call
        func = MagicMock(return_value=42)
        result = rate_limited_call(func, delay=0.01, max_retries=1)
        assert result == 42
        func.assert_called_once()

    @patch("src.data.scrapers._retry.time.sleep")
    def test_retries_on_exception(self, mock_sleep):
        from src.data.scrapers._retry import rate_limited_call
        func = MagicMock(side_effect=[ValueError("oops"), "ok"])
        result = rate_limited_call(func, delay=0.01, max_retries=2)
        assert result == "ok"
        assert func.call_count == 2

    @patch("src.data.scrapers._retry.time.sleep")
    def test_exhausted_raises(self, mock_sleep):
        from src.data.scrapers._retry import rate_limited_call
        func = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            rate_limited_call(func, delay=0.01, max_retries=1)


# ======================================================================
# 2. espn_picks.py — PublicPicks and ConsensusData
# ======================================================================


class TestPublicPicks:

    def test_as_dict_keys(self):
        from src.data.scrapers.espn_picks import PublicPicks
        pp = PublicPicks(
            team_id="duke", team_name="Duke", seed=1, region="East",
            round_of_64_pct=98.0, champion_pct=22.0,
        )
        d = pp.as_dict
        assert set(d.keys()) == {"R64", "R32", "S16", "E8", "F4", "CHAMP"}
        assert d["R64"] == 98.0
        assert d["CHAMP"] == 22.0

    def test_defaults(self):
        from src.data.scrapers.espn_picks import PublicPicks
        pp = PublicPicks(team_id="t", team_name="T", seed=1, region="R")
        assert pp.round_of_64_pct == 0.0
        assert pp.champion_pct == 0.0


class TestConsensusData:

    def test_get_champion_favorites(self):
        from src.data.scrapers.espn_picks import PublicPicks, ConsensusData
        teams = {
            "a": PublicPicks("a", "A", 1, "E", champion_pct=30.0),
            "b": PublicPicks("b", "B", 2, "E", champion_pct=20.0),
            "c": PublicPicks("c", "C", 3, "E", champion_pct=10.0),
        }
        cd = ConsensusData(teams=teams)
        top2 = cd.get_champion_favorites(top_n=2)
        assert len(top2) == 2
        assert top2[0].team_id == "a"
        assert top2[1].team_id == "b"

    def test_get_contrarian_picks(self):
        from src.data.scrapers.espn_picks import PublicPicks, ConsensusData
        teams = {
            "a": PublicPicks("a", "A", 1, "E", champion_pct=5.0),
            "b": PublicPicks("b", "B", 2, "E", champion_pct=20.0),
        }
        cd = ConsensusData(teams=teams)
        model_probs = {"a": 15.0, "b": 25.0}
        contrarian = cd.get_contrarian_picks(model_probs, min_leverage=2.0)
        # a: 15/5 = 3.0 >= 2.0 → included
        # b: 25/20 = 1.25 < 2.0 → excluded
        assert len(contrarian) == 1
        assert contrarian[0][0] == "a"
        assert contrarian[0][3] == pytest.approx(3.0)

    def test_get_contrarian_picks_zero_public_pct_skipped(self):
        from src.data.scrapers.espn_picks import PublicPicks, ConsensusData
        teams = {"a": PublicPicks("a", "A", 1, "E", champion_pct=0.0)}
        cd = ConsensusData(teams=teams)
        result = cd.get_contrarian_picks({"a": 10.0}, min_leverage=1.0)
        assert result == []

    def test_get_contrarian_picks_unknown_team_skipped(self):
        from src.data.scrapers.espn_picks import PublicPicks, ConsensusData
        cd = ConsensusData(teams={})
        result = cd.get_contrarian_picks({"unknown": 10.0})
        assert result == []


class TestESPNPicksScraper:

    def test_constructor_no_cache(self):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper()
        assert s.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper(cache_dir=str(tmp_path / "cache"))
        assert s.cache_dir.exists()

    def test_dict_to_consensus(self):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper()
        data = {
            "teams": {
                "duke": {
                    "team_name": "Duke", "seed": 1, "region": "East",
                    "round_of_64_pct": 98.0, "champion_pct": 22.0,
                }
            },
            "sources": ["espn"],
            "timestamp": "2026-03-17",
        }
        cd = s._dict_to_consensus(data)
        assert "duke" in cd.teams
        assert cd.teams["duke"].champion_pct == 22.0
        assert cd.sources == ["espn"]
        assert cd.timestamp == "2026-03-17"

    def test_load_from_json(self, tmp_path):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper()
        data = {"teams": {"a": {"team_name": "A", "seed": 1, "region": "E", "champion_pct": 5.0}}}
        fp = tmp_path / "picks.json"
        fp.write_text(json.dumps(data))
        cd = s.load_from_json(str(fp))
        assert "a" in cd.teams

    def test_load_from_cache_no_cache_dir(self):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper()
        assert s._load_from_cache("anything") is None

    def test_save_and_load_cache(self, tmp_path):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper(cache_dir=str(tmp_path / "c"))
        s._save_to_cache("test.json", {"teams": {}})
        loaded = s._load_from_cache("test.json")
        assert loaded == {"teams": {}}

    @patch.dict("os.environ", {}, clear=True)
    def test_fetch_picks_returns_empty_on_failure(self):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper()
        with patch.object(s, "_try_json_url", return_value=None):
            result = s.fetch_picks(year=2026)
            assert result.sources == ["espn"]
            assert len(result.teams) == 0

    def test_try_json_url_403_returns_none(self):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        s.session.get = MagicMock(return_value=mock_resp)
        result = s._try_json_url("http://example.com", 2026, "test")
        assert result is None

    def test_try_json_url_success(self, tmp_path):
        from src.data.scrapers.espn_picks import ESPNPicksScraper
        s = ESPNPicksScraper(cache_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "teams": {"duke": {"team_name": "Duke", "seed": 1, "region": "E"}}
        }
        s.session.get = MagicMock(return_value=mock_resp)
        result = s._try_json_url("http://example.com", 2026, "test")
        assert result is not None
        assert "duke" in result.teams


class TestYahooPicksScraper:

    def test_constructor(self):
        from src.data.scrapers.espn_picks import YahooPicksScraper
        s = YahooPicksScraper()
        assert s.cache_dir is None
        assert s.BASE_URL == "https://tournament.fantasysports.yahoo.com"

    @patch.dict("os.environ", {}, clear=True)
    def test_fetch_picks_no_url_returns_empty(self):
        from src.data.scrapers.espn_picks import YahooPicksScraper
        s = YahooPicksScraper()
        result = s.fetch_picks(2026)
        assert result.sources == ["yahoo"]

    def test_cache_save_and_load(self, tmp_path):
        from src.data.scrapers.espn_picks import YahooPicksScraper
        s = YahooPicksScraper(cache_dir=str(tmp_path / "yc"))
        s._save_cache("test.json", {"x": 1})
        loaded = s._load_cache("test.json")
        assert loaded == {"x": 1}

    def test_load_cache_no_dir(self):
        from src.data.scrapers.espn_picks import YahooPicksScraper
        s = YahooPicksScraper()
        assert s._load_cache("any") is None


class TestCBSPicksScraper:

    def test_constructor(self):
        from src.data.scrapers.espn_picks import CBSPicksScraper
        s = CBSPicksScraper()
        assert s.cache_dir is None

    @patch.dict("os.environ", {}, clear=True)
    def test_fetch_picks_no_url_returns_empty(self):
        from src.data.scrapers.espn_picks import CBSPicksScraper
        s = CBSPicksScraper()
        result = s.fetch_picks(2026)
        assert result.sources == ["cbs"]


class TestAggregateConsensus:

    def test_single_source(self):
        from src.data.scrapers.espn_picks import (
            PublicPicks, ConsensusData, aggregate_consensus,
        )
        espn = ConsensusData(
            teams={"a": PublicPicks("a", "A", 1, "E", champion_pct=10.0)},
        )
        yahoo = ConsensusData()
        cbs = ConsensusData()
        result = aggregate_consensus(espn, yahoo, cbs)
        assert "a" in result.teams
        assert result.teams["a"].champion_pct == pytest.approx(10.0)

    def test_weighted_average(self):
        from src.data.scrapers.espn_picks import (
            PublicPicks, ConsensusData, aggregate_consensus,
        )
        espn = ConsensusData(
            teams={"a": PublicPicks("a", "A", 1, "E", champion_pct=20.0)},
        )
        yahoo = ConsensusData(
            teams={"a": PublicPicks("a", "A", 1, "E", champion_pct=10.0)},
        )
        cbs = ConsensusData()
        weights = {"espn": 0.6, "yahoo": 0.4, "cbs": 0.0}
        result = aggregate_consensus(espn, yahoo, cbs, weights=weights)
        # (0.6*20 + 0.4*10) / (0.6+0.4) = 16.0
        assert result.teams["a"].champion_pct == pytest.approx(16.0)


class TestSourceHealthAndStatus:

    def test_source_health_enum(self):
        from src.data.scrapers.espn_picks import SourceHealth
        assert SourceHealth.HEALTHY.value == "healthy"
        assert SourceHealth.FAILED.value == "failed"

    def test_source_status_is_stale_never_succeeded(self):
        from src.data.scrapers.espn_picks import SourceStatus
        s = SourceStatus(name="test")
        assert s.is_stale is True

    def test_source_status_is_stale_recent(self):
        from src.data.scrapers.espn_picks import SourceStatus
        s = SourceStatus(name="test", last_success=time.time())
        assert s.is_stale is False

    def test_source_status_is_stale_old(self):
        from src.data.scrapers.espn_picks import SourceStatus
        s = SourceStatus(name="test", last_success=time.time() - 100000)
        assert s.is_stale is True


class TestOrchestratorResult:

    def test_is_degraded(self):
        from src.data.scrapers.espn_picks import OrchestratorResult, ConsensusData
        r = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn", "yahoo"],
            source_statuses={},
        )
        assert r.is_degraded is True

    def test_not_degraded(self):
        from src.data.scrapers.espn_picks import OrchestratorResult, ConsensusData
        r = OrchestratorResult(
            consensus=ConsensusData(),
            successful_sources=["espn", "yahoo", "cbs"],
            source_statuses={},
        )
        assert r.is_degraded is False

    def test_health_summary(self):
        from src.data.scrapers.espn_picks import (
            OrchestratorResult, ConsensusData, SourceStatus, SourceHealth,
        )
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
        assert "yahoo" in summary
        assert "timeout" in summary


class TestRoundDeltaAndPicksUpdate:

    def test_round_delta_properties(self):
        from src.data.scrapers.espn_picks import RoundDelta
        rd = RoundDelta(team_id="a", round_name="CHAMP", old_pct=10.0, new_pct=15.0)
        assert rd.delta == 5.0
        assert rd.abs_delta == 5.0

    def test_round_delta_negative(self):
        from src.data.scrapers.espn_picks import RoundDelta
        rd = RoundDelta(team_id="a", round_name="R64", old_pct=20.0, new_pct=15.0)
        assert rd.delta == -5.0
        assert rd.abs_delta == 5.0

    def test_picks_update_summary_no_changes(self):
        from src.data.scrapers.espn_picks import PicksUpdate, ConsensusData
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=0.0,
        )
        assert "No significant changes" in pu.summary

    def test_picks_update_summary_with_changes(self):
        from src.data.scrapers.espn_picks import PicksUpdate, ConsensusData
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=["duke", "unc"], max_delta=3.5,
        )
        assert "2 teams changed" in pu.summary

    def test_max_round_delta_empty(self):
        from src.data.scrapers.espn_picks import PicksUpdate, ConsensusData
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=1.0, round_deltas=[],
        )
        assert pu.max_round_delta == 1.0

    def test_late_round_max_delta(self):
        from src.data.scrapers.espn_picks import PicksUpdate, ConsensusData, RoundDelta
        rds = [
            RoundDelta("a", "R64", 10.0, 12.0),
            RoundDelta("a", "CHAMP", 5.0, 9.0),
        ]
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=["a"], max_delta=4.0, round_deltas=rds,
        )
        assert pu.late_round_max_delta == 4.0

    def test_late_round_max_delta_no_late_rounds(self):
        from src.data.scrapers.espn_picks import PicksUpdate, ConsensusData, RoundDelta
        rds = [RoundDelta("a", "R64", 10.0, 12.0)]
        pu = PicksUpdate(
            previous=ConsensusData(), current=ConsensusData(),
            changed_teams=[], max_delta=2.0, round_deltas=rds,
        )
        assert pu.late_round_max_delta == 0.0


class TestRefreshablePicksManager:

    def test_is_stale_initially(self):
        from src.data.scrapers.espn_picks import RefreshablePicksManager, ScraperOrchestrator
        mgr = RefreshablePicksManager(MagicMock(spec=ScraperOrchestrator))
        assert mgr.is_stale is True
        assert mgr.hours_since_refresh is None

    @patch("src.data.scrapers.espn_picks.time.time")
    def test_fetch_or_refresh_first_call(self, mock_time):
        from src.data.scrapers.espn_picks import (
            RefreshablePicksManager, ScraperOrchestrator, OrchestratorResult,
            ConsensusData,
        )
        mock_time.return_value = 1000.0
        mock_orch = MagicMock(spec=ScraperOrchestrator)
        mock_orch.fetch_all_picks.return_value = OrchestratorResult(
            consensus=ConsensusData(), successful_sources=["espn"],
            source_statuses={},
        )
        mgr = RefreshablePicksManager(mock_orch)
        result, update = mgr.fetch_or_refresh(2026)
        assert result is not None
        assert update is None  # No previous data

    @patch("src.data.scrapers.espn_picks.time.time")
    def test_not_stale_returns_cached(self, mock_time):
        from src.data.scrapers.espn_picks import (
            RefreshablePicksManager, OrchestratorResult, ConsensusData,
        )
        mock_time.return_value = 1000.0
        mock_orch = MagicMock()
        cached = OrchestratorResult(
            consensus=ConsensusData(), successful_sources=["espn"],
            source_statuses={},
        )
        mock_orch.fetch_all_picks.return_value = cached
        mgr = RefreshablePicksManager(mock_orch, staleness_threshold_hours=24.0)
        # First call: fetch
        mgr.fetch_or_refresh(2026)
        # Second call (not stale): should return cached
        result, update = mgr.fetch_or_refresh(2026)
        assert result is cached
        assert update is None
        assert mock_orch.fetch_all_picks.call_count == 1


# ======================================================================
# 3. injury_report.py
# ======================================================================


class TestInjuryReport:

    def test_dataclass_creation(self):
        from src.data.scrapers.injury_report import InjuryReport
        from src.data.models.player import InjuryStatus
        r = InjuryReport(
            player_name="Test Player", team_id="duke",
            status=InjuryStatus.QUESTIONABLE, injury_type="ankle",
        )
        assert r.player_name == "Test Player"
        assert r.team_id == "duke"
        assert r.injury_type == "ankle"


class TestTeamInjuryReport:

    def test_has_injuries_false_when_all_healthy(self):
        from src.data.scrapers.injury_report import InjuryReport, TeamInjuryReport
        from src.data.models.player import InjuryStatus
        reports = [
            InjuryReport("P1", "duke", InjuryStatus.HEALTHY),
            InjuryReport("P2", "duke", InjuryStatus.HEALTHY),
        ]
        tir = TeamInjuryReport(team_id="duke", reports=reports)
        assert tir.has_injuries is False

    def test_has_injuries_true(self):
        from src.data.scrapers.injury_report import InjuryReport, TeamInjuryReport
        from src.data.models.player import InjuryStatus
        reports = [
            InjuryReport("P1", "duke", InjuryStatus.HEALTHY),
            InjuryReport("P2", "duke", InjuryStatus.OUT),
        ]
        tir = TeamInjuryReport(team_id="duke", reports=reports)
        assert tir.has_injuries is True


class TestInjuryReportScraper:

    def test_constructor_no_cache(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        s = InjuryReportScraper()
        assert s.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.injury_report import InjuryReportScraper
        s = InjuryReportScraper(cache_dir=str(tmp_path / "inj_cache"))
        assert s.cache_dir.exists()

    def test_normalize_status_healthy(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        from src.data.models.player import InjuryStatus
        assert InjuryReportScraper._normalize_status("healthy") == InjuryStatus.HEALTHY
        assert InjuryReportScraper._normalize_status("active") == InjuryStatus.HEALTHY
        assert InjuryReportScraper._normalize_status("probable") == InjuryStatus.HEALTHY

    def test_normalize_status_questionable(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        from src.data.models.player import InjuryStatus
        assert InjuryReportScraper._normalize_status("questionable") == InjuryStatus.QUESTIONABLE
        assert InjuryReportScraper._normalize_status("game-time decision") == InjuryStatus.QUESTIONABLE
        assert InjuryReportScraper._normalize_status("day-to-day") == InjuryStatus.QUESTIONABLE

    def test_normalize_status_out(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        from src.data.models.player import InjuryStatus
        assert InjuryReportScraper._normalize_status("out") == InjuryStatus.OUT
        assert InjuryReportScraper._normalize_status("doubtful") == InjuryStatus.DOUBTFUL

    def test_normalize_status_season_ending(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        from src.data.models.player import InjuryStatus
        assert InjuryReportScraper._normalize_status("out for season") == InjuryStatus.SEASON_ENDING
        assert InjuryReportScraper._normalize_status("season-ending") == InjuryStatus.SEASON_ENDING

    def test_normalize_status_unknown_defaults_questionable(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        from src.data.models.player import InjuryStatus
        assert InjuryReportScraper._normalize_status("unknown_status") == InjuryStatus.QUESTIONABLE

    def test_load_from_json_dict_format(self, tmp_path):
        from src.data.scrapers.injury_report import InjuryReportScraper
        from src.data.models.player import InjuryStatus
        data = {
            "source": "espn",
            "timestamp": "2026-03-15",
            "teams": {
                "duke": {
                    "players": [
                        {"name": "Player A", "status": "out", "injury_type": "knee"},
                        {"name": "Player B", "status": "healthy"},
                    ]
                }
            },
        }
        fp = tmp_path / "injuries.json"
        fp.write_text(json.dumps(data))
        s = InjuryReportScraper()
        reports = s.load_from_json(str(fp))
        assert "duke" in reports
        assert len(reports["duke"].reports) == 2
        assert reports["duke"].reports[0].status == InjuryStatus.OUT

    def test_load_from_json_list_format(self, tmp_path):
        from src.data.scrapers.injury_report import InjuryReportScraper
        data = {
            "teams": [
                {
                    "team_id": "unc",
                    "players": [{"name": "P1", "status": "questionable"}],
                }
            ]
        }
        fp = tmp_path / "inj2.json"
        fp.write_text(json.dumps(data))
        s = InjuryReportScraper()
        reports = s.load_from_json(str(fp))
        assert "unc" in reports

    def test_parse_team_block_alternative_keys(self):
        from src.data.scrapers.injury_report import InjuryReportScraper
        s = InjuryReportScraper()
        block = {
            "injuries": [
                {"player_name": "John", "injury_status": "doubtful", "injury": "ankle", "return": "day-to-day"},
            ]
        }
        result = s._parse_team_block("t1", block, "espn", "2026-03-15")
        assert len(result.reports) == 1
        assert result.reports[0].player_name == "John"
        assert result.reports[0].injury_type == "ankle"


class TestInjurySeverityEstimator:

    def test_healthy_player(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator()
        r = InjuryReport("P", "t", InjuryStatus.HEALTHY)
        mean, std = est.estimate_availability(r)
        assert mean == 1.0
        assert std == 0.0

    def test_season_ending(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator()
        r = InjuryReport("P", "t", InjuryStatus.SEASON_ENDING)
        mean, std = est.estimate_availability(r)
        assert mean == 0.0
        assert std == 0.0

    def test_out_no_timeline(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator()
        r = InjuryReport("P", "t", InjuryStatus.OUT, expected_return="")
        mean, std = est.estimate_availability(r)
        assert mean == pytest.approx(0.0, abs=0.1)

    def test_out_with_dtd_timeline(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator()
        r = InjuryReport("P", "t", InjuryStatus.OUT, expected_return="day-to-day")
        mean, std = est.estimate_availability(r)
        assert 0.0 < mean < 1.0

    def test_questionable_ankle(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator()
        r = InjuryReport("P", "t", InjuryStatus.QUESTIONABLE, injury_type="ankle")
        mean, std = est.estimate_availability(r)
        assert 0.0 < mean <= 1.0
        assert std > 0

    def test_doubtful_lower_than_questionable(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator()
        r_q = InjuryReport("P", "t", InjuryStatus.QUESTIONABLE, injury_type="ankle")
        r_d = InjuryReport("P", "t", InjuryStatus.DOUBTFUL, injury_type="ankle")
        mean_q, _ = est.estimate_availability(r_q)
        mean_d, _ = est.estimate_availability(r_d)
        assert mean_d < mean_q

    def test_sample_availability_deterministic(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator(random_seed=42)
        r = InjuryReport("P", "t", InjuryStatus.HEALTHY)
        samples = est.sample_availability(r, n_samples=10)
        assert len(samples) == 10
        assert all(s == 1.0 for s in samples)

    def test_sample_availability_questionable(self):
        from src.data.scrapers.injury_report import InjurySeverityEstimator, InjuryReport
        from src.data.models.player import InjuryStatus
        est = InjurySeverityEstimator(random_seed=42)
        r = InjuryReport("P", "t", InjuryStatus.QUESTIONABLE, injury_type="knee")
        samples = est.sample_availability(r, n_samples=100)
        assert len(samples) == 100
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_backward_compat_alias(self):
        from src.data.scrapers.injury_report import InjurySeverityModel, InjurySeverityEstimator
        assert InjurySeverityModel is InjurySeverityEstimator


class TestApplyInjuryReportsToRoster:

    def test_apply_updates_player_status(self):
        from src.data.scrapers.injury_report import (
            InjuryReport, TeamInjuryReport, apply_injury_reports_to_roster,
        )
        from src.data.models.player import Player, Position, InjuryStatus, Roster
        p = Player("p1", "John Doe", "duke", Position.POINT_GUARD, injury_status=InjuryStatus.HEALTHY)
        roster = Roster(team_id="duke", players=[p])
        report = TeamInjuryReport(
            team_id="duke",
            reports=[InjuryReport("John Doe", "duke", InjuryStatus.OUT, injury_type="knee")],
        )
        count = apply_injury_reports_to_roster(roster, report)
        assert count == 1
        assert p.injury_status == InjuryStatus.OUT
        assert p.injury_details == "knee"

    def test_apply_no_matching_player(self):
        from src.data.scrapers.injury_report import (
            InjuryReport, TeamInjuryReport, apply_injury_reports_to_roster,
        )
        from src.data.models.player import Player, Position, InjuryStatus, Roster
        p = Player("p1", "John Doe", "duke", Position.POINT_GUARD)
        roster = Roster(team_id="duke", players=[p])
        report = TeamInjuryReport(
            team_id="duke",
            reports=[InjuryReport("Jane Smith", "duke", InjuryStatus.OUT)],
        )
        count = apply_injury_reports_to_roster(roster, report)
        assert count == 0

    def test_apply_same_status_not_counted(self):
        from src.data.scrapers.injury_report import (
            InjuryReport, TeamInjuryReport, apply_injury_reports_to_roster,
        )
        from src.data.models.player import Player, Position, InjuryStatus, Roster
        p = Player("p1", "John Doe", "duke", Position.POINT_GUARD, injury_status=InjuryStatus.OUT)
        roster = Roster(team_id="duke", players=[p])
        report = TeamInjuryReport(
            team_id="duke",
            reports=[InjuryReport("John Doe", "duke", InjuryStatus.OUT)],
        )
        count = apply_injury_reports_to_roster(roster, report)
        assert count == 0


# ======================================================================
# 4. roster_enrichment.py
# ======================================================================


class TestRosterEnrichment:

    def test_normalize_player_name_basic(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        assert RosterEnrichment.normalize_player_name("John Smith") == "john_smith"
        assert RosterEnrichment.normalize_player_name("D.J. Burns Jr.") == "dj_burns"
        assert RosterEnrichment.normalize_player_name("Player III") == "player"
        assert RosterEnrichment.normalize_player_name("") == ""

    def test_player_key_with_player_id(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        assert RosterEnrichment.player_key({"player_id": "12345", "name": "John"}) == "espn:12345"

    def test_player_key_without_player_id(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        assert RosterEnrichment.player_key({"name": "John Smith"}) == "john_smith"

    def test_compute_eligibility_year_freshman(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        history = {"p1": [(2025, "duke", 10.0)]}
        elig = RosterEnrichment.compute_eligibility_year("p1", 2025, history)
        assert elig == 1  # No prior years

    def test_compute_eligibility_year_sophomore(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        history = {"p1": [(2024, "duke", 10.0), (2025, "duke", 20.0)]}
        elig = RosterEnrichment.compute_eligibility_year("p1", 2025, history)
        assert elig == 2

    def test_compute_eligibility_year_capped_at_5(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        history = {"p1": [(y, "t", 10.0) for y in range(2020, 2027)]}
        elig = RosterEnrichment.compute_eligibility_year("p1", 2026, history)
        assert elig == 5

    def test_compute_eligibility_year_unknown_player(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        elig = RosterEnrichment.compute_eligibility_year("unknown", 2026, {})
        assert elig == 1

    def test_compute_is_transfer_true(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        history = {"p1": [(2025, "unc", 10.0), (2026, "duke", 15.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2026, "duke", history) is True

    def test_compute_is_transfer_false_same_team(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        history = {"p1": [(2025, "duke", 10.0), (2026, "duke", 15.0)]}
        assert RosterEnrichment.compute_is_transfer("p1", 2026, "duke", history) is False

    def test_compute_is_transfer_false_freshman(self):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        assert RosterEnrichment.compute_is_transfer("p1", 2026, "duke", {}) is False

    def test_load_all_rosters(self, tmp_path):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        roster_data = {"teams": [{"team_id": "duke", "players": []}]}
        (tmp_path / "cbbpy_rosters_2025.json").write_text(json.dumps(roster_data))
        enrichment = RosterEnrichment(str(tmp_path))
        result = enrichment.load_all_rosters(2024, 2026)
        assert 2025 in result
        assert 2024 not in result

    def test_build_player_history(self, tmp_path):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        all_rosters = {
            2025: {"teams": [{"team_id": "duke", "players": [{"name": "John", "minutes_per_game": 20}]}]},
            2026: {"teams": [{"team_id": "duke", "players": [{"name": "John", "minutes_per_game": 25}]}]},
        }
        enrichment = RosterEnrichment(str(tmp_path))
        history = enrichment.build_player_history(all_rosters)
        key = RosterEnrichment.normalize_player_name("John")
        assert key in history
        assert len(history[key]) == 2

    def test_enrich_all_empty_dir(self, tmp_path):
        from src.data.scrapers.roster_enrichment import RosterEnrichment
        enrichment = RosterEnrichment(str(tmp_path))
        result = enrichment.enrich_all(2024, 2026)
        assert result["years_processed"] == 0
        assert result["total_players_enriched"] == 0


# ======================================================================
# 5. bracket_ingestion.py
# ======================================================================


class TestBracketTeam:

    def test_creation(self):
        from src.data.scrapers.bracket_ingestion import BracketTeam
        t = BracketTeam(
            canonical_id="duke", display_name="Duke", seed=1, region="East",
        )
        assert t.canonical_id == "duke"
        assert t.seed == 1
        assert t.name_confidence == 1.0


class TestTournamentBracketData:

    def test_n_teams(self):
        from src.data.scrapers.bracket_ingestion import BracketTeam, TournamentBracketData
        bd = TournamentBracketData(
            season=2026,
            teams=[BracketTeam("a", "A", 1, "East"), BracketTeam("b", "B", 16, "East")],
        )
        assert bd.n_teams == 2

    def test_get_first_round_matchups(self):
        from src.data.scrapers.bracket_ingestion import BracketTeam, TournamentBracketData
        teams = []
        for seed in range(1, 17):
            teams.append(BracketTeam(f"t{seed}", f"Team{seed}", seed, "East"))
        bd = TournamentBracketData(season=2026, teams=teams)
        matchups = bd.get_first_round_matchups()
        assert len(matchups) == 8
        # 1 vs 16
        assert matchups[0][0].seed == 1
        assert matchups[0][1].seed == 16

    def test_to_pipeline_json(self):
        from src.data.scrapers.bracket_ingestion import BracketTeam, TournamentBracketData
        bd = TournamentBracketData(
            season=2026,
            teams=[BracketTeam("duke", "Duke", 1, "East")],
            source="test",
        )
        pj = bd.to_pipeline_json()
        assert pj["season"] == 2026
        assert len(pj["teams"]) == 1
        assert pj["teams"][0]["team_id"] == "duke"
        assert pj["teams"][0]["school_slug"] == "duke"

    def test_to_espn_bracket(self):
        from src.data.scrapers.bracket_ingestion import BracketTeam, TournamentBracketData
        teams = [BracketTeam(f"t{s}", f"T{s}", s, "East") for s in range(1, 17)]
        bd = TournamentBracketData(season=2026, teams=teams)
        espn = bd.to_espn_bracket()
        assert espn["season"] == 2026
        assert "East" in espn["regions"]
        assert len(espn["first_round_matchups"]) == 8


class TestBracketIngestionPipeline:

    @patch("src.data.scrapers.bracket_ingestion.BIGDANCE_AVAILABLE", False)
    @patch("src.data.scrapers.bracket_ingestion.SPORTS_REF_AVAILABLE", False)
    def test_fetch_auto_no_source_raises(self, tmp_path):
        from src.data.scrapers.bracket_ingestion import BracketIngestionPipeline
        pipe = BracketIngestionPipeline(season=2026, cache_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="No bracket source"):
            pipe.fetch(source="auto")

    def test_fetch_manual_json(self, tmp_path):
        from src.data.scrapers.bracket_ingestion import BracketIngestionPipeline
        data = {
            "season": 2026,
            "teams": [
                {"team_name": "Duke", "seed": 1, "region": "East"},
            ],
        }
        fp = tmp_path / "bracket.json"
        fp.write_text(json.dumps(data))
        pipe = BracketIngestionPipeline(season=2026, cache_dir=str(tmp_path))
        result = pipe.fetch(source=str(fp))
        assert result.n_teams == 1
        assert result.source.startswith("manual:")

    def test_save(self, tmp_path):
        from src.data.scrapers.bracket_ingestion import (
            BracketIngestionPipeline, BracketTeam, TournamentBracketData,
        )
        pipe = BracketIngestionPipeline(season=2026, cache_dir=str(tmp_path))
        bd = TournamentBracketData(
            season=2026,
            teams=[BracketTeam("duke", "Duke", 1, "East")],
        )
        path = pipe.save(bd)
        assert Path(path).exists()
        with open(path) as f:
            saved = json.load(f)
        assert saved["season"] == 2026


# ======================================================================
# 6. cbbpy_rosters.py — static helper methods
# ======================================================================


class TestCBBpyRosterScraperHelpers:

    def test_normalize_position(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._normalize_position("PG") == "PG"
        assert CBBpyRosterScraper._normalize_position("POINT GUARD") == "PG"
        assert CBBpyRosterScraper._normalize_position("SG") == "SG"
        assert CBBpyRosterScraper._normalize_position("SF") == "SF"
        assert CBBpyRosterScraper._normalize_position("PF") == "PF"
        assert CBBpyRosterScraper._normalize_position("C") == "C"
        assert CBBpyRosterScraper._normalize_position("G") == "PG"
        assert CBBpyRosterScraper._normalize_position("F") == "SF"
        assert CBBpyRosterScraper._normalize_position("") == "PG"

    def test_parse_eligibility_year(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._parse_eligibility_year("FR") == 1
        assert CBBpyRosterScraper._parse_eligibility_year("FRESHMAN") == 1
        assert CBBpyRosterScraper._parse_eligibility_year("SO") == 2
        assert CBBpyRosterScraper._parse_eligibility_year("JR") == 3
        assert CBBpyRosterScraper._parse_eligibility_year("SR") == 4
        assert CBBpyRosterScraper._parse_eligibility_year("GR") == 5
        assert CBBpyRosterScraper._parse_eligibility_year("GRADUATE") == 5
        assert CBBpyRosterScraper._parse_eligibility_year(None) is None
        assert CBBpyRosterScraper._parse_eligibility_year("3") == 3
        assert CBBpyRosterScraper._parse_eligibility_year("RS-JR") == 3

    def test_parse_minutes(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._parse_minutes(None) == 0.0
        assert CBBpyRosterScraper._parse_minutes(30) == 30.0
        assert CBBpyRosterScraper._parse_minutes("25.5") == 25.5
        assert CBBpyRosterScraper._parse_minutes("30:30") == pytest.approx(30.5)
        assert CBBpyRosterScraper._parse_minutes("invalid") == 0.0

    def test_to_float(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._to_float(None) == 0.0
        assert CBBpyRosterScraper._to_float(3.14) == 3.14
        assert CBBpyRosterScraper._to_float("5.5") == 5.5
        assert CBBpyRosterScraper._to_float("bad") == 0.0

    def test_to_bool(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._to_bool(True) is True
        assert CBBpyRosterScraper._to_bool(False) is False
        assert CBBpyRosterScraper._to_bool(None) is False
        assert CBBpyRosterScraper._to_bool("1") is True
        assert CBBpyRosterScraper._to_bool("yes") is True
        assert CBBpyRosterScraper._to_bool("no") is False

    def test_safe_int(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._safe_int("5") == 5
        assert CBBpyRosterScraper._safe_int(None) == 0
        assert CBBpyRosterScraper._safe_int("bad", 99) == 99

    def test_team_id(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._team_id("Duke Blue Devils") == "duke_blue_devils"
        assert CBBpyRosterScraper._team_id("North Carolina") == "north_carolina"

    def test_player_id(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        pid = CBBpyRosterScraper._player_id("duke", "John Smith")
        assert pid == "duke_john_smith"

    def test_normalize_player_name(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._normalize_player_name("John Smith") == "johnsmith"

    def test_frame_to_records_none(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._frame_to_records(None) == []

    def test_frame_to_records_list(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._frame_to_records([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]

    def test_frame_to_records_dict(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._frame_to_records({"a": 1}) == [{"a": 1}]

    def test_frame_to_records_dataframe_mock(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"x": 1}]
        result = CBBpyRosterScraper._frame_to_records(mock_df)
        assert result == [{"x": 1}]

    def test_is_possession_ending(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._is_possession_ending("turnover", "", {}) is True
        assert CBBpyRosterScraper._is_possession_ending("made shot", "", {"scoring_play": True}) is True
        assert CBBpyRosterScraper._is_possession_ending("rebound", "defensive rebound", {}) is True
        assert CBBpyRosterScraper._is_possession_ending("free throw", "Free Throw 2 of 2", {}) is True
        assert CBBpyRosterScraper._is_possession_ending("pass", "assist", {}) is False

    def test_constructor_no_cache(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        s = CBBpyRosterScraper()
        assert s.cache_dir is None

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        s = CBBpyRosterScraper(cache_dir=str(tmp_path / "cache"))
        assert s.cache_dir.exists()

    def test_cache_roundtrip(self, tmp_path):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        s = CBBpyRosterScraper(cache_dir=str(tmp_path))
        s._save_cache("test.json", {"teams": [{"team_id": "duke"}]})
        loaded = s._load_cache("test.json")
        assert loaded["teams"][0]["team_id"] == "duke"

    def test_load_cache_missing_file(self, tmp_path):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        s = CBBpyRosterScraper(cache_dir=str(tmp_path))
        assert s._load_cache("nonexistent.json") is None

    def test_import_module_nonexistent(self):
        from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper
        assert CBBpyRosterScraper._import_module("nonexistent_module_xyz") is None


# ======================================================================
# 7. tournament_results.py
# ======================================================================


class TestTournamentResultsScraper:

    def test_constructor(self):
        from src.data.scrapers.tournament_results import TournamentResultsScraper
        s = TournamentResultsScraper()
        assert s.cache_dir is None
        assert s.BASE_URL == "https://www.sports-reference.com/cbb/postseason/men"

    def test_constructor_with_cache(self, tmp_path):
        from src.data.scrapers.tournament_results import TournamentResultsScraper
        s = TournamentResultsScraper(cache_dir=str(tmp_path / "res_cache"))
        assert s.cache_dir.exists()

    def test_cache_roundtrip(self, tmp_path):
        from src.data.scrapers.tournament_results import TournamentResultsScraper
        s = TournamentResultsScraper(cache_dir=str(tmp_path))
        s._save_cache("test.json", {"games": []})
        loaded = s._load_cache("test.json")
        assert loaded == {"games": []}

    def test_load_cache_no_dir(self):
        from src.data.scrapers.tournament_results import TournamentResultsScraper
        s = TournamentResultsScraper()
        assert s._load_cache("any") is None

    def test_scrape_year_uses_cache(self, tmp_path):
        from src.data.scrapers.tournament_results import TournamentResultsScraper
        s = TournamentResultsScraper(cache_dir=str(tmp_path))
        games = [{"game": i} for i in range(63)]
        s._save_cache("tournament_results_2024.json", {"games": games})
        result = s.scrape_year(2024)
        assert len(result) == 63

    def test_round_labels(self):
        from src.data.scrapers.tournament_results import _ROUND_LABELS
        assert _ROUND_LABELS[0] == "R64"
        assert _ROUND_LABELS[5] == "NCG"


# ======================================================================
# 8. womens/herhoopstats.py
# ======================================================================


class TestWomensTeamStats:

    def test_creation_and_to_dict(self):
        from src.data.scrapers.womens.herhoopstats import WomensTeamStats
        stats = WomensTeamStats(
            team_name="UConn", team_id="uconn",
            adj_efficiency_margin=25.0, seed=1,
        )
        d = stats.to_dict()
        assert d["team_name"] == "UConn"
        assert d["adj_efficiency_margin"] == 25.0
        assert d["seed"] == 1

    def test_from_dict(self):
        from src.data.scrapers.womens.herhoopstats import WomensTeamStats
        d = {"team_name": "SC", "team_id": "sc", "seed": 1, "wins": 30}
        stats = WomensTeamStats.from_dict(d)
        assert stats.team_name == "SC"
        assert stats.wins == 30

    def test_from_dict_ignores_unknown_keys(self):
        from src.data.scrapers.womens.herhoopstats import WomensTeamStats
        d = {"team_name": "SC", "team_id": "sc", "unknown_key": 99}
        stats = WomensTeamStats.from_dict(d)
        assert stats.team_name == "SC"

    def test_defaults(self):
        from src.data.scrapers.womens.herhoopstats import WomensTeamStats
        stats = WomensTeamStats(team_name="T")
        assert stats.adj_tempo == 68.0
        assert stats.elo_rating == 1500.0
        assert stats.win_pct == 0.5


class TestEstimateFromSeed:

    def test_seed_1(self):
        from src.data.scrapers.womens.herhoopstats import _estimate_from_seed
        stats = _estimate_from_seed("uconn", 1)
        assert stats.adj_efficiency_margin == 28.0
        assert stats.seed == 1
        assert stats.elo_rating == 1750

    def test_seed_16(self):
        from src.data.scrapers.womens.herhoopstats import _estimate_from_seed
        stats = _estimate_from_seed("t16", 16)
        assert stats.adj_efficiency_margin == -15.0
        assert stats.elo_rating == 1250

    def test_win_pct_decreases_with_seed(self):
        from src.data.scrapers.womens.herhoopstats import _estimate_from_seed
        s1 = _estimate_from_seed("t1", 1)
        s8 = _estimate_from_seed("t8", 8)
        assert s1.win_pct > s8.win_pct


class TestHerHoopStatsScraper:

    def test_constructor(self):
        from src.data.scrapers.womens.herhoopstats import HerHoopStatsScraper
        s = HerHoopStatsScraper()
        assert s.cache_dir == Path("data/raw")

    def test_load_cached_no_file(self, tmp_path):
        from src.data.scrapers.womens.herhoopstats import HerHoopStatsScraper
        s = HerHoopStatsScraper(cache_dir=str(tmp_path))
        result = s.load_cached(2026)
        assert result == {}

    def test_load_cached_with_file(self, tmp_path):
        from src.data.scrapers.womens.herhoopstats import HerHoopStatsScraper
        data = [{"team_name": "UConn", "team_id": "uconn", "seed": 1}]
        (tmp_path / "womens_stats_2026.json").write_text(json.dumps(data))
        s = HerHoopStatsScraper(cache_dir=str(tmp_path))
        result = s.load_cached(2026)
        assert "uconn" in result

    def test_generate_seed_based_estimates(self):
        from src.data.scrapers.womens.herhoopstats import HerHoopStatsScraper
        s = HerHoopStatsScraper()
        seeds = {"uconn": 1, "team2": 8}
        result = s.generate_seed_based_estimates(seeds)
        assert "uconn" in result
        assert "team2" in result
        assert result["uconn"].adj_efficiency_margin > result["team2"].adj_efficiency_margin

    def test_save_cache(self, tmp_path):
        from src.data.scrapers.womens.herhoopstats import HerHoopStatsScraper, WomensTeamStats
        s = HerHoopStatsScraper(cache_dir=str(tmp_path))
        teams = {"uconn": WomensTeamStats(team_name="UConn", team_id="uconn")}
        s.save_cache(2026, teams)
        assert (tmp_path / "womens_stats_2026.json").exists()

    def test_upset_rates_constant(self):
        from src.data.scrapers.womens.herhoopstats import WOMENS_HISTORICAL_UPSET_RATES
        assert (1, 16) in WOMENS_HISTORICAL_UPSET_RATES
        assert WOMENS_HISTORICAL_UPSET_RATES[(1, 16)] < WOMENS_HISTORICAL_UPSET_RATES[(8, 9)]


# ======================================================================
# 9. womens/historical_results.py
# ======================================================================


class TestHistoricalTournamentGame:

    def test_upset_lower_seed_wins(self):
        from src.data.scrapers.womens.historical_results import HistoricalTournamentGame
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=12, team2_seed=5,
            team1_name="Underdog", team2_name="Favorite",
            team1_won=True,
        )
        assert g.upset is True

    def test_no_upset(self):
        from src.data.scrapers.womens.historical_results import HistoricalTournamentGame
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=1, team2_seed=16,
            team1_name="Top", team2_name="Bot",
            team1_won=True,
        )
        assert g.upset is False

    def test_seed_matchup(self):
        from src.data.scrapers.womens.historical_results import HistoricalTournamentGame
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=8, team2_seed=9,
            team1_name="A", team2_name="B", team1_won=True,
        )
        assert g.seed_matchup == (8, 9)

    def test_favored_won_true(self):
        from src.data.scrapers.womens.historical_results import HistoricalTournamentGame
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=1, team2_seed=16,
            team1_name="A", team2_name="B", team1_won=True,
        )
        assert g.favored_won is True

    def test_favored_won_false(self):
        from src.data.scrapers.womens.historical_results import HistoricalTournamentGame
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=1, team2_seed=16,
            team1_name="A", team2_name="B", team1_won=False,
        )
        assert g.favored_won is False

    def test_same_seed_favored_always_true(self):
        from src.data.scrapers.womens.historical_results import HistoricalTournamentGame
        g = HistoricalTournamentGame(
            year=2024, round_name="R64",
            team1_seed=5, team2_seed=5,
            team1_name="A", team2_name="B", team1_won=False,
        )
        assert g.favored_won is True


class TestWomensHistoricalResults:

    def test_constructor(self):
        from src.data.scrapers.womens.historical_results import WomensHistoricalResults
        whr = WomensHistoricalResults()
        assert whr.cache_dir == Path("data/raw")
        assert whr.games == []

    def test_generate_synthetic_history(self):
        from src.data.scrapers.womens.historical_results import WomensHistoricalResults
        whr = WomensHistoricalResults()
        games = whr._generate_synthetic_history(years=[2024])
        # 4 regions x 8 matchups = 32 games per year
        assert len(games) == 32

    def test_compute_upset_rates(self):
        from src.data.scrapers.womens.historical_results import WomensHistoricalResults
        whr = WomensHistoricalResults()
        whr._generate_synthetic_history(years=[2024, 2025])
        rates = whr.compute_upset_rates()
        assert (1, 16) in rates
        # 1v16 upsets should be very rare
        assert rates[(1, 16)] < 0.5

    def test_load_cached_no_file(self, tmp_path):
        from src.data.scrapers.womens.historical_results import WomensHistoricalResults
        whr = WomensHistoricalResults(cache_dir=str(tmp_path))
        games = whr.load_cached()
        assert len(games) > 0  # Falls back to synthetic

    def test_load_cached_with_file(self, tmp_path):
        from src.data.scrapers.womens.historical_results import WomensHistoricalResults
        data = [
            {
                "year": 2024, "round_name": "R64",
                "team1_seed": 1, "team2_seed": 16,
                "team1_name": "A", "team2_name": "B",
                "team1_won": True,
            }
        ]
        (tmp_path / "womens_tournament_history.json").write_text(json.dumps(data))
        whr = WomensHistoricalResults(cache_dir=str(tmp_path))
        games = whr.load_cached()
        assert len(games) == 1

    def test_save_cache(self, tmp_path):
        from src.data.scrapers.womens.historical_results import (
            WomensHistoricalResults, HistoricalTournamentGame,
        )
        whr = WomensHistoricalResults(cache_dir=str(tmp_path))
        games = [
            HistoricalTournamentGame(
                year=2024, round_name="R64",
                team1_seed=1, team2_seed=16,
                team1_name="A", team2_name="B", team1_won=True,
            )
        ]
        whr.save_cache(games)
        assert (tmp_path / "womens_tournament_history.json").exists()


# ======================================================================
# 10. womens/ncaa_net.py
# ======================================================================


class TestNETRanking:

    def test_creation(self):
        from src.data.scrapers.womens.ncaa_net import NETRanking
        r = NETRanking(team_name="UConn", team_id="uconn", net_ranking=3)
        assert r.net_ranking == 3
        assert r.net_rating == 0.0

    def test_to_dict(self):
        from src.data.scrapers.womens.ncaa_net import NETRanking
        r = NETRanking("UConn", "uconn", 3, record="30-2", conference="Big East")
        d = r.to_dict()
        assert d["team_name"] == "UConn"
        assert d["record"] == "30-2"
        assert d["conference"] == "Big East"


class TestWomensNETScraper:

    def test_constructor(self):
        from src.data.scrapers.womens.ncaa_net import WomensNETScraper
        s = WomensNETScraper()
        assert s.cache_dir == Path("data/raw")

    def test_load_cached_no_file(self, tmp_path):
        from src.data.scrapers.womens.ncaa_net import WomensNETScraper
        s = WomensNETScraper(cache_dir=str(tmp_path))
        result = s.load_cached(2026)
        assert result == {}

    def test_load_cached_with_file(self, tmp_path):
        from src.data.scrapers.womens.ncaa_net import WomensNETScraper
        data = [{"team_name": "UConn", "team_id": "uconn", "net_ranking": 3}]
        (tmp_path / "womens_net_rankings_2026.json").write_text(json.dumps(data))
        s = WomensNETScraper(cache_dir=str(tmp_path))
        result = s.load_cached(2026)
        assert "uconn" in result
        assert result["uconn"].net_ranking == 3

    def test_estimate_from_seeds(self):
        from src.data.scrapers.womens.ncaa_net import WomensNETScraper
        s = WomensNETScraper()
        seeds = {"uconn": 1, "team16": 16}
        result = s.estimate_from_seeds(seeds)
        assert result["uconn"].net_ranking < result["team16"].net_ranking
        assert result["uconn"].net_rating > result["team16"].net_rating

    def test_save_cache(self, tmp_path):
        from src.data.scrapers.womens.ncaa_net import WomensNETScraper, NETRanking
        s = WomensNETScraper(cache_dir=str(tmp_path))
        rankings = {"uconn": NETRanking("UConn", "uconn", 3)}
        s.save_cache(2026, rankings)
        assert (tmp_path / "womens_net_rankings_2026.json").exists()


# ======================================================================
# 11. Injury constants
# ======================================================================


class TestInjuryConstants:

    def test_severity_priors_keys(self):
        from src.data.scrapers.injury_report import INJURY_SEVERITY_PRIORS
        assert "ankle" in INJURY_SEVERITY_PRIORS
        assert "knee" in INJURY_SEVERITY_PRIORS
        assert "acl" in INJURY_SEVERITY_PRIORS
        assert "illness" in INJURY_SEVERITY_PRIORS
        # ACL = zero availability
        mean, std = INJURY_SEVERITY_PRIORS["acl"]
        assert mean == 0.0

    def test_return_timeline_factors(self):
        from src.data.scrapers.injury_report import RETURN_TIMELINE_FACTORS
        assert RETURN_TIMELINE_FACTORS["day-to-day"] == 0.80
        assert RETURN_TIMELINE_FACTORS["season"] == 0.0

    def test_position_groups(self):
        from src.data.scrapers.injury_report import POSITION_GROUPS
        from src.data.models.player import Position
        assert Position.POINT_GUARD in POSITION_GROUPS["guard"]
        assert Position.CENTER in POSITION_GROUPS["big"]


# ======================================================================
# 12. PositionalDepthChart
# ======================================================================


class TestPositionalDepthChart:

    def _make_roster(self):
        from src.data.models.player import Player, Position, InjuryStatus, Roster
        players = [
            Player("p1", "PG1", "duke", Position.POINT_GUARD,
                   minutes_per_game=30, rapm_offensive=3.0, warp=0.5, box_plus_minus=5.0),
            Player("p2", "SG1", "duke", Position.SHOOTING_GUARD,
                   minutes_per_game=28, rapm_offensive=2.0, warp=0.3, box_plus_minus=3.0),
            Player("p3", "SF1", "duke", Position.SMALL_FORWARD,
                   minutes_per_game=25, rapm_offensive=1.5, warp=0.2, box_plus_minus=2.0),
            Player("p4", "PF1", "duke", Position.POWER_FORWARD,
                   minutes_per_game=22, rapm_offensive=1.0, warp=0.15, box_plus_minus=1.5),
            Player("p5", "C1", "duke", Position.CENTER,
                   minutes_per_game=20, rapm_offensive=0.8, warp=0.1, box_plus_minus=1.0),
        ]
        return Roster(team_id="duke", players=players)

    def test_analyze_returns_all_groups(self):
        from src.data.scrapers.injury_report import PositionalDepthChart
        dc = PositionalDepthChart()
        roster = self._make_roster()
        result = dc.analyze(roster)
        assert "guard" in result
        assert "wing" in result
        assert "forward" in result
        assert "big" in result

    def test_compute_injury_impact_healthy(self):
        from src.data.scrapers.injury_report import PositionalDepthChart
        dc = PositionalDepthChart()
        roster = self._make_roster()
        impacts = dc.compute_injury_impact(roster)
        assert "total_impact" in impacts
        assert "positional_vulnerability" in impacts
        # Healthy roster should have minimal impact
        assert impacts["total_impact"] >= 0.0
