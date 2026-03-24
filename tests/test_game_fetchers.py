"""Integration tests for the Historical Game Pipeline refactor.

Tests cover:
  - ESPN event parsing (box scores extracted from statistics array)
  - IngestionGameRecord schema validation
  - to_team_game_rows() contract (compatible with team_games_to_game_records)
  - dedup_records() logging date inconsistencies
  - Season window validation
  - HistoricalGameFetcher provider fallback behaviour
  - IncrementalGameFetcher ESPN-first fetch
  - Full pipeline: IngestionGameRecord → team_games → GameRecord
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.data.ingestion.game_fetchers import (
    HistoricalGameFetcher,
    IncrementalGameFetcher,
    _parse_espn_stats,
    dedup_records,
    is_in_season_window,
    parse_espn_event,
    season_window,
)
from src.data.ingestion.schemas import IngestionGameRecord

# ---------------------------------------------------------------------------
# Fixtures — real ESPN scoreboard payloads (truncated for test speed)
# ---------------------------------------------------------------------------

ESPN_STATS_HOME = [
    {"name": "fieldGoalsMade-fieldGoalsAttempted", "abbreviation": "FG", "displayValue": "28-60"},
    {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "abbreviation": "3PT", "displayValue": "8-22"},
    {"name": "freeThrowsMade-freeThrowsAttempted", "abbreviation": "FT", "displayValue": "10-15"},
    {"name": "offensiveRebounds", "abbreviation": "OREB", "displayValue": "8"},
    {"name": "defensiveRebounds", "abbreviation": "DREB", "displayValue": "27"},
    {"name": "assists", "abbreviation": "AST", "displayValue": "18"},
    {"name": "steals", "abbreviation": "STL", "displayValue": "7"},
    {"name": "blocks", "abbreviation": "BLK", "displayValue": "4"},
    {"name": "turnovers", "abbreviation": "TO", "displayValue": "11"},
    {"name": "fouls", "abbreviation": "PF", "displayValue": "18"},
]

ESPN_STATS_AWAY = [
    {"name": "fieldGoalsMade-fieldGoalsAttempted", "abbreviation": "FG", "displayValue": "25-58"},
    {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "abbreviation": "3PT", "displayValue": "6-18"},
    {"name": "freeThrowsMade-freeThrowsAttempted", "abbreviation": "FT", "displayValue": "17-20"},
    {"name": "offensiveRebounds", "abbreviation": "OREB", "displayValue": "6"},
    {"name": "defensiveRebounds", "abbreviation": "DREB", "displayValue": "25"},
    {"name": "assists", "abbreviation": "AST", "displayValue": "15"},
    {"name": "steals", "abbreviation": "STL", "displayValue": "5"},
    {"name": "blocks", "abbreviation": "BLK", "displayValue": "2"},
    {"name": "turnovers", "abbreviation": "TO", "displayValue": "13"},
    {"name": "fouls", "abbreviation": "PF", "displayValue": "20"},
]

ESPN_EVENT = {
    "id": "401583234",
    "date": "2024-03-20T21:30Z",
    "competitions": [{
        "id": "401583234",
        "neutralSite": True,
        "conferenceCompetition": False,
        "competitors": [
            {
                "id": "57",
                "homeAway": "home",
                "winner": True,
                "team": {"id": "57", "displayName": "Duke Blue Devils"},
                "score": "82",
                "statistics": ESPN_STATS_HOME,
            },
            {
                "id": "153",
                "homeAway": "away",
                "winner": False,
                "team": {"id": "153", "displayName": "North Carolina Tar Heels"},
                "score": "79",
                "statistics": ESPN_STATS_AWAY,
            },
        ],
    }],
}

ESPN_SCOREBOARD_RESPONSE = {"events": [ESPN_EVENT]}

# ---------------------------------------------------------------------------
# _parse_espn_stats
# ---------------------------------------------------------------------------

class TestParseEspnStats:
    def test_fg_extraction(self):
        stats = _parse_espn_stats(ESPN_STATS_HOME)
        assert stats["fgm"] == 28.0
        assert stats["fga"] == 60.0

    def test_three_point_extraction(self):
        stats = _parse_espn_stats(ESPN_STATS_HOME)
        assert stats["fg3m"] == 8.0
        assert stats["fg3a"] == 22.0

    def test_ft_extraction(self):
        stats = _parse_espn_stats(ESPN_STATS_HOME)
        assert stats["ftm"] == 10.0
        assert stats["fta"] == 15.0

    def test_rebounds_and_other(self):
        stats = _parse_espn_stats(ESPN_STATS_HOME)
        assert stats["orb"] == 8.0
        assert stats["drb"] == 27.0
        assert stats["ast"] == 18.0
        assert stats["stl"] == 7.0
        assert stats["blk"] == 4.0
        assert stats["tov"] == 11.0
        assert stats["pf"] == 18.0

    def test_empty_statistics(self):
        stats = _parse_espn_stats([])
        assert stats == {}

    def test_malformed_display_value(self):
        """Non-numeric display values should produce 0.0, not raise."""
        stats = _parse_espn_stats([
            {"name": "fieldGoalsMade-fieldGoalsAttempted", "abbreviation": "FG", "displayValue": "N/A"},
        ])
        assert stats.get("fgm", 0.0) == 0.0

    def test_abbreviation_fallback(self):
        """Should match by abbreviation when name differs."""
        stats = _parse_espn_stats([
            {"name": "someOtherName", "abbreviation": "FG", "displayValue": "20-45"},
        ])
        assert stats["fgm"] == 20.0
        assert stats["fga"] == 45.0


# ---------------------------------------------------------------------------
# parse_espn_event
# ---------------------------------------------------------------------------

class TestParseEspnEvent:
    def test_creates_valid_record(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        assert rec is not None
        assert rec.game_id == "401583234"
        assert rec.date == "2024-03-20"
        assert rec.season == 2024

    def test_home_away_scores(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        assert rec.home_score == 82
        assert rec.away_score == 79

    def test_home_team_id_normalized(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        # normalize_team_id("Duke Blue Devils") → some snake_case id
        assert rec.home_team_id  # non-empty
        assert "duke" in rec.home_team_id.lower()

    def test_box_score_extracted(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        assert rec.home_fgm == 28.0
        assert rec.home_fga == 60.0
        assert rec.home_fg3m == 8.0
        assert rec.home_fg3a == 22.0
        assert rec.home_fta == 15.0
        assert rec.home_tov == 11.0
        assert rec.home_orb == 8.0
        assert rec.home_drb == 27.0

    def test_away_box_score_extracted(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        assert rec.away_fgm == 25.0
        assert rec.away_fga == 58.0
        assert rec.away_fg3m == 6.0
        assert rec.away_tov == 13.0

    def test_neutral_site_flag(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        assert rec.neutral_site is True

    def test_provider_tag(self):
        rec = parse_espn_event(ESPN_EVENT, season=2024, fallback_date="2024-03-20")
        assert rec.provider == "espn_scoreboard"

    def test_fallback_date_used_when_missing(self):
        event_no_date = {**ESPN_EVENT, "date": ""}
        rec = parse_espn_event(event_no_date, season=2024, fallback_date="2024-03-15")
        assert rec.date == "2024-03-15"

    def test_returns_none_when_no_competitors(self):
        bad_event = {**ESPN_EVENT, "competitions": [{"competitors": []}]}
        rec = parse_espn_event(bad_event, season=2024, fallback_date="2024-03-20")
        assert rec is None

    def test_returns_none_when_both_scores_zero(self):
        """Upcoming / cancelled games should be skipped."""
        event = json.loads(json.dumps(ESPN_EVENT))  # deep copy
        event["competitions"][0]["competitors"][0]["score"] = "0"
        event["competitions"][0]["competitors"][1]["score"] = "0"
        rec = parse_espn_event(event, season=2024, fallback_date="2024-03-20")
        assert rec is None


# ---------------------------------------------------------------------------
# IngestionGameRecord validation & conversion
# ---------------------------------------------------------------------------

class TestIngestionGameRecord:
    def _make_record(self, **kwargs) -> IngestionGameRecord:
        defaults = dict(
            game_id="g1", date="2024-01-15", season=2024,
            home_team_id="duke", home_team_name="Duke",
            away_team_id="unc", away_team_name="North Carolina",
            home_score=82, away_score=79,
            home_fgm=28, home_fga=60, home_fg3m=8, home_fg3a=22,
            home_fta=15, home_tov=11, home_orb=8, home_drb=27,
            away_fgm=25, away_fga=58, away_fg3m=6, away_fg3a=18,
            away_fta=20, away_tov=13, away_orb=6, away_drb=25,
        )
        defaults.update(kwargs)
        return IngestionGameRecord(**defaults)

    def test_validate_valid_record(self):
        rec = self._make_record()
        assert rec.validate() == []

    def test_validate_empty_game_id(self):
        rec = self._make_record(game_id="")
        errors = rec.validate()
        assert any("game_id" in e for e in errors)

    def test_validate_bad_date(self):
        rec = self._make_record(date="20240115")
        errors = rec.validate()
        assert any("date" in e for e in errors)

    def test_validate_both_zero_scores(self):
        rec = self._make_record(home_score=0, away_score=0)
        errors = rec.validate()
        assert any("0" in e for e in errors)

    def test_validate_negative_score(self):
        rec = self._make_record(home_score=-1)
        errors = rec.validate()
        assert any("negative" in e for e in errors)

    def test_to_team_game_rows_returns_two_rows(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        assert len(rows) == 2

    def test_to_team_game_rows_team_ids(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        team_ids = {r["team_id"] for r in rows}
        assert team_ids == {"duke", "unc"}

    def test_to_team_game_rows_scores_mirror(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        home_row = next(r for r in rows if r["team_id"] == "duke")
        away_row = next(r for r in rows if r["team_id"] == "unc")
        assert home_row["team_score"] == 82
        assert home_row["opponent_score"] == 79
        assert away_row["team_score"] == 79
        assert away_row["opponent_score"] == 82

    def test_to_team_game_rows_opponent_ids(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        home_row = next(r for r in rows if r["team_id"] == "duke")
        assert home_row["opponent_id"] == "unc"

    def test_to_team_game_rows_box_scores(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        home_row = next(r for r in rows if r["team_id"] == "duke")
        assert home_row["fgm"] == 28
        assert home_row["fga"] == 60
        assert home_row["fg3m"] == 8
        assert home_row["fta"] == 15
        assert home_row["turnovers"] == 11
        assert home_row["orb"] == 8
        assert home_row["drb"] == 27

    def test_to_team_game_rows_possessions_computed(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        home_row = next(r for r in rows if r["team_id"] == "duke")
        # possessions = FGA - ORB + TOV + 0.475 * FTA = 60 - 8 + 11 + 0.475*15 = 70.125
        assert home_row["possessions"] == pytest.approx(70.125, rel=1e-3)

    def test_to_team_game_rows_possession_fallback_when_no_box(self):
        """When box scores are all zero, use score-average fallback."""
        rec = self._make_record(
            home_fga=0, home_orb=0, home_tov=0, home_fta=0,
            away_fga=0, away_orb=0, away_tov=0, away_fta=0,
        )
        rows = rec.to_team_game_rows()
        assert all(r["possessions"] > 0 for r in rows)

    def test_to_team_game_rows_date_and_season_present(self):
        rec = self._make_record()
        rows = rec.to_team_game_rows()
        assert all(r["date"] == "2024-01-15" for r in rows)
        assert all(r["season"] == 2024 for r in rows)

    def test_to_game_row(self):
        rec = self._make_record()
        game_row = rec.to_game_row()
        assert game_row["game_id"] == "g1"
        assert game_row["team1_id"] == "duke"
        assert game_row["team2_id"] == "unc"
        assert game_row["team1_score"] == 82
        assert game_row["team2_score"] == 79
        assert game_row["date"] == "2024-01-15"


# ---------------------------------------------------------------------------
# Overtime inference
# ---------------------------------------------------------------------------

class TestOvertimeInference:
    """Tests for _infer_overtime and _inject_overtime_from_periods."""

    def test_infer_overtime_explicit_field(self):
        from src.data.ingestion.game_fetchers import _infer_overtime
        t1 = {"overtime": True, "fga": 60, "orb": 8, "turnovers": 11, "fta": 15}
        t2 = {"fga": 58, "orb": 6, "turnovers": 13, "fta": 20}
        assert _infer_overtime(t1, t2) is True

    def test_infer_overtime_explicit_string(self):
        from src.data.ingestion.game_fetchers import _infer_overtime
        t1 = {"ot": "true", "fga": 60, "orb": 8, "turnovers": 11, "fta": 15}
        t2 = {"fga": 58, "orb": 6, "turnovers": 13, "fta": 20}
        assert _infer_overtime(t1, t2) is True

    def test_infer_overtime_num_periods(self):
        from src.data.ingestion.game_fetchers import _infer_overtime
        t1 = {"num_periods": 3, "fga": 60, "orb": 8, "turnovers": 11, "fta": 15}
        t2 = {"fga": 58, "orb": 6, "turnovers": 13, "fta": 20}
        assert _infer_overtime(t1, t2) is True

    def test_infer_overtime_regulation(self):
        """Regulation game with normal possessions → False."""
        from src.data.ingestion.game_fetchers import _infer_overtime
        t1 = {"fga": 60, "orb": 8, "turnovers": 11, "fta": 15}
        t2 = {"fga": 58, "orb": 6, "turnovers": 13, "fta": 20}
        # t1 poss est: 60 - 8 + 11 + 0.475*15 = 70.125 (< 82)
        assert _infer_overtime(t1, t2) is False

    def test_infer_overtime_high_possessions(self):
        """Very high possessions → heuristic triggers."""
        from src.data.ingestion.game_fetchers import _infer_overtime
        t1 = {"fga": 80, "orb": 8, "turnovers": 15, "fta": 20}
        t2 = {"fga": 58, "orb": 6, "turnovers": 13, "fta": 20}
        # t1 poss est: 80 - 8 + 15 + 0.475*20 = 96.5 (> 82)
        assert _infer_overtime(t1, t2) is True

    def test_inject_overtime_from_periods(self):
        from src.data.ingestion.game_fetchers import _inject_overtime_from_periods
        rows = [
            {"game_id": "g1", "period_number": 2, "team_id": "duke"},
            {"game_id": "g2", "period_number": 3, "team_id": "unc"},
        ]
        _inject_overtime_from_periods(rows)
        assert rows[0].get("overtime") is None  # g1 only has period 2 (regulation)
        assert rows[1].get("overtime") is True   # g2 has period 3 (OT)

    def test_overtime_propagated_to_team_game_rows(self):
        """overtime field is included in to_team_game_rows() output."""
        rec = IngestionGameRecord(
            game_id="g1", date="2024-01-15", season=2024,
            home_team_id="duke", home_team_name="Duke",
            away_team_id="unc", away_team_name="North Carolina",
            home_score=95, away_score=90,
            home_fgm=32, home_fga=68, home_fg3m=8, home_fg3a=22,
            home_fta=18, home_tov=12, home_orb=8, home_drb=27,
            away_fgm=30, away_fga=65, away_fg3m=6, away_fg3a=18,
            away_fta=22, away_tov=14, away_orb=6, away_drb=25,
            overtime=True,
        )
        rows = rec.to_team_game_rows()
        assert all(r["overtime"] is True for r in rows)

    def test_overtime_propagated_to_game_row(self):
        """overtime field is included in to_game_row() output."""
        rec = IngestionGameRecord(
            game_id="g1", date="2024-01-15", season=2024,
            home_team_id="duke", home_team_name="Duke",
            away_team_id="unc", away_team_name="North Carolina",
            home_score=95, away_score=90,
            overtime=True,
        )
        game_row = rec.to_game_row()
        assert game_row["overtime"] is True

    def test_overtime_default_false_in_game_row(self):
        rec = IngestionGameRecord(
            game_id="g1", date="2024-01-15", season=2024,
            home_team_id="duke", home_team_name="Duke",
            away_team_id="unc", away_team_name="North Carolina",
            home_score=82, away_score=79,
        )
        game_row = rec.to_game_row()
        assert game_row["overtime"] is False


# ---------------------------------------------------------------------------
# season_window and is_in_season_window
# ---------------------------------------------------------------------------

class TestSeasonWindow:
    def test_season_2024_window(self):
        start, end = season_window(2024)
        assert start.year == 2023 and start.month == 11
        assert end.year == 2024 and end.month == 4

    def test_inside_window(self):
        assert is_in_season_window("2024-01-15", 2024)
        assert is_in_season_window("2023-11-01", 2024)
        assert is_in_season_window("2024-04-08", 2024)

    def test_outside_window_before(self):
        assert not is_in_season_window("2022-01-15", 2024)

    def test_outside_window_after(self):
        assert not is_in_season_window("2024-05-01", 2024)

    def test_malformed_date(self):
        assert not is_in_season_window("not-a-date", 2024)


# ---------------------------------------------------------------------------
# dedup_records
# ---------------------------------------------------------------------------

class TestDedupRecords:
    def _make_record(self, game_id: str, date: str, fga: float = 0.0, provider: str = "espn") -> IngestionGameRecord:
        return IngestionGameRecord(
            game_id=game_id, date=date, season=2024,
            home_team_id="a", home_team_name="A",
            away_team_id="b", away_team_name="B",
            home_score=80, away_score=75,
            home_fga=fga, away_fga=fga,
            provider=provider,
        )

    def test_no_duplicates_passthrough(self):
        records = [self._make_record("g1", "2024-01-01"), self._make_record("g2", "2024-01-02")]
        result = dedup_records(records)
        assert len(result) == 2

    def test_duplicate_same_date_deduplicated(self):
        records = [
            self._make_record("g1", "2024-01-01"),
            self._make_record("g1", "2024-01-01"),
        ]
        result = dedup_records(records)
        assert len(result) == 1

    def test_richer_record_preferred(self):
        """When duplicates, keep the one with more box-score data (higher FGA)."""
        records = [
            self._make_record("g1", "2024-01-01", fga=0.0, provider="espn"),
            self._make_record("g1", "2024-01-01", fga=60.0, provider="sportsdataverse"),
        ]
        result = dedup_records(records)
        assert len(result) == 1
        assert result[0].home_fga == 60.0

    def test_date_inconsistency_logged(self, caplog):
        """Two records with the same game_id but different dates should log a warning."""
        records = [
            self._make_record("g1", "2024-01-15", provider="espn"),
            self._make_record("g1", "2024-01-14", provider="sportsdataverse"),
        ]
        with caplog.at_level(logging.WARNING):
            result = dedup_records(records)
        assert len(result) == 1
        assert any("inconsistency" in m.lower() or "inconsistent" in m.lower() for m in caplog.messages)

    def test_sorted_by_date(self):
        records = [
            self._make_record("g3", "2024-01-10"),
            self._make_record("g1", "2024-01-01"),
            self._make_record("g2", "2024-01-05"),
        ]
        result = dedup_records(records)
        dates = [r.date for r in result]
        assert dates == sorted(dates)

    def test_dedup_preserves_overtime_when_replacing(self):
        """When the kept record has overtime=True but the richer record doesn't,
        the overtime flag should be preserved."""
        espn = self._make_record("g1", "2024-01-01", fga=60, provider="espn")
        espn.overtime = True
        sdv = self._make_record("g1", "2024-01-01", fga=70, provider="sportsdataverse")
        sdv.overtime = False
        result = dedup_records([espn, sdv])
        assert len(result) == 1
        # sdv wins on FGA, but should inherit ESPN's overtime=True
        assert result[0].overtime is True
        assert result[0].provider == "sportsdataverse"

    def test_dedup_preserves_overtime_when_keeping(self):
        """When the discarded record has overtime=True, the kept record
        should inherit the flag."""
        espn = self._make_record("g1", "2024-01-01", fga=70, provider="espn")
        espn.overtime = False
        sdv = self._make_record("g1", "2024-01-01", fga=60, provider="sportsdataverse")
        sdv.overtime = True
        result = dedup_records([espn, sdv])
        assert len(result) == 1
        assert result[0].overtime is True
        assert result[0].provider == "espn"


# ---------------------------------------------------------------------------
# HistoricalGameFetcher — provider fallback
# ---------------------------------------------------------------------------

class TestHistoricalGameFetcherFallback:
    """Tests that the provider cascade works correctly using monkeypatching."""

    def test_espn_primary_used_when_sufficient(self, tmp_path, monkeypatch):
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        espn_records = [
            IngestionGameRecord(
                game_id=f"g{i}", date=f"2024-01-{i:02d}", season=2024,
                home_team_id="a", home_team_name="A",
                away_team_id="b", away_team_name="B",
                home_score=80, away_score=75, provider="espn_scoreboard",
            )
            for i in range(1, 600)
        ]
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: espn_records)
        sdv_called = []
        monkeypatch.setattr(fetcher, "_fetch_via_sportsdataverse", lambda s: sdv_called.append(s) or [])

        result = fetcher.fetch_season(2024, strict=False)
        assert len(result) >= 500
        assert sdv_called == []  # sportsdataverse NOT called when ESPN is sufficient

    def test_sportsdataverse_fallback_when_espn_insufficient(self, tmp_path, monkeypatch):
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: [])
        sdv_records = [
            IngestionGameRecord(
                game_id=f"g{i}", date=f"2024-01-{i:02d}", season=2024,
                home_team_id="a", home_team_name="A",
                away_team_id="b", away_team_name="B",
                home_score=80, away_score=75, provider="sportsdataverse",
            )
            for i in range(1, 600)
        ]
        monkeypatch.setattr(fetcher, "_fetch_via_sportsdataverse", lambda s: sdv_records)
        cbbpy_called = []
        monkeypatch.setattr(fetcher, "_fetch_via_cbbpy", lambda s: cbbpy_called.append(s) or [])

        result = fetcher.fetch_season(2024, strict=False)
        assert len(result) >= 500
        assert all(r.provider == "sportsdataverse" for r in result)
        assert cbbpy_called == []

    def test_cbbpy_fallback_when_both_fail(self, tmp_path, monkeypatch):
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: [])
        monkeypatch.setattr(fetcher, "_fetch_via_sportsdataverse", lambda s: [])
        cbbpy_records = [
            IngestionGameRecord(
                game_id=f"g{i}", date=f"2024-01-{i:02d}", season=2024,
                home_team_id="a", home_team_name="A",
                away_team_id="b", away_team_name="B",
                home_score=80, away_score=75, provider="cbbpy",
            )
            for i in range(1, 600)
        ]
        monkeypatch.setattr(fetcher, "_fetch_via_cbbpy", lambda s: cbbpy_records)

        result = fetcher.fetch_season(2024, strict=False)
        assert len(result) >= 500
        assert all(r.provider == "cbbpy" for r in result)

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        """fetch_season should write and re-read from cache."""
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        records_returned = [
            IngestionGameRecord(
                game_id=f"g{i}", date=f"2024-01-{i:02d}", season=2024,
                home_team_id="a", home_team_name="A",
                away_team_id="b", away_team_name="B",
                home_score=80, away_score=75, provider="espn_scoreboard",
            )
            for i in range(1, 600)
        ]
        fetch_count = {"n": 0}

        def mock_espn(s):
            fetch_count["n"] += 1
            return records_returned

        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", mock_espn)

        # First call — fetches and caches
        result1 = fetcher.fetch_season(2024, strict=False)
        assert fetch_count["n"] == 1

        # Second call — from cache
        result2 = fetcher.fetch_season(2024, strict=False)
        assert fetch_count["n"] == 1  # NOT called again
        assert len(result2) == len(result1)


# ---------------------------------------------------------------------------
# HistoricalGameFetcher — fetch_day with mocked HTTP
# ---------------------------------------------------------------------------

class TestHistoricalGameFetcherDay:
    def test_fetch_day_parses_event(self, tmp_path):
        from datetime import date as _date
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path), http_timeout=5)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = ESPN_SCOREBOARD_RESPONSE

        with patch("requests.Session.get", return_value=mock_resp):
            session = __import__("requests").Session()
            records = fetcher._fetch_day(_date(2024, 3, 20), season=2024, session=session)

        assert len(records) == 1
        assert records[0].game_id == "401583234"
        assert records[0].home_score == 82
        assert records[0].away_score == 79

    def test_fetch_day_deduplicates_seasontype(self, tmp_path):
        """Same game returned for seasontype=2 and seasontype=3 should yield 1 record."""
        from datetime import date as _date
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path), http_timeout=5)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = ESPN_SCOREBOARD_RESPONSE

        with patch("requests.Session.get", return_value=mock_resp):
            session = __import__("requests").Session()
            records = fetcher._fetch_day(_date(2024, 3, 20), season=2024, session=session)

        game_ids = [r.game_id for r in records]
        assert len(game_ids) == len(set(game_ids))  # no duplicates

    def test_fetch_day_rejects_out_of_window(self, tmp_path):
        """Games with dates outside the season window should be filtered."""
        from datetime import date as _date
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path), http_timeout=5)

        # Event with a date in 2022, but we're fetching for season 2024
        bad_event = json.loads(json.dumps(ESPN_EVENT))
        bad_event["date"] = "2022-01-15T00:00Z"
        bad_event["id"] = "999999"

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"events": [bad_event]}

        with patch("requests.Session.get", return_value=mock_resp):
            session = __import__("requests").Session()
            records = fetcher._fetch_day(_date(2022, 1, 15), season=2024, session=session)

        assert records == []

    def test_fetch_day_handles_http_error(self, tmp_path):
        from datetime import date as _date
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path), http_timeout=5)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = __import__("requests").HTTPError("404")

        with patch("requests.Session.get", return_value=mock_resp):
            session = __import__("requests").Session()
            records = fetcher._fetch_day(_date(2024, 3, 20), season=2024, session=session)

        assert records == []


# ---------------------------------------------------------------------------
# IncrementalGameFetcher
# ---------------------------------------------------------------------------

class TestIncrementalGameFetcher:
    def test_fetch_since_calls_espn(self, monkeypatch):
        fetcher = IncrementalGameFetcher()

        espn_records = [
            IngestionGameRecord(
                game_id=f"g{i}", date=f"2026-03-{i + 8:02d}", season=2026,
                home_team_id="a", home_team_name="A",
                away_team_id="b", away_team_name="B",
                home_score=80, away_score=75, provider="espn_scoreboard",
            )
            for i in range(1, 5)
        ]

        # Mock the underlying _fetch_day to return records
        from datetime import date as _date
        def mock_fetch_day(d, season, session=None):
            day_str = d.isoformat()
            matching = [r for r in espn_records if r.date == day_str]
            return matching

        monkeypatch.setattr(fetcher._hist, "_fetch_day", mock_fetch_day)

        result = fetcher.fetch_since(year=2026, since="2026-03-09")
        # Should have records from 2026-03-09 onwards
        assert len(result) >= 1
        assert all(r.date >= "2026-03-09" for r in result)

    def test_fetch_since_invalid_date(self):
        fetcher = IncrementalGameFetcher()
        result = fetcher.fetch_since(year=2026, since="not-a-date")
        assert result == []

    def test_fetch_since_deduplicates(self, monkeypatch):
        fetcher = IncrementalGameFetcher()
        dup_record = IngestionGameRecord(
            game_id="dup1", date="2026-03-15", season=2026,
            home_team_id="a", home_team_name="A",
            away_team_id="b", away_team_name="B",
            home_score=80, away_score=75, provider="espn_scoreboard",
        )

        def mock_fetch_day(d, season, session=None):
            if d.isoformat() == "2026-03-15":
                return [dup_record, dup_record]  # duplicate from two seasontypes
            return []

        monkeypatch.setattr(fetcher._hist, "_fetch_day", mock_fetch_day)

        result = fetcher.fetch_since(year=2026, since="2026-03-15")
        game_ids = [r.game_id for r in result]
        assert len(game_ids) == len(set(game_ids))


# ---------------------------------------------------------------------------
# Full pipeline: IngestionGameRecord → team_games → GameRecord
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """Ensure to_team_game_rows() output is compatible with team_games_to_game_records."""

    def test_team_games_compatible_with_incremental_engine(self):
        from src.data.features.proprietary_metrics import team_games_to_game_records

        records = [
            IngestionGameRecord(
                game_id=f"game_{i}",
                date=f"2024-01-{i:02d}",
                season=2024,
                home_team_id="duke",
                home_team_name="Duke",
                away_team_id="unc",
                away_team_name="North Carolina",
                home_score=80 + i,
                away_score=75,
                home_fgm=28, home_fga=60, home_fg3m=8, home_fg3a=22,
                home_fta=15, home_tov=11, home_orb=8, home_drb=27,
                away_fgm=25, away_fga=58, away_fg3m=6, away_fg3a=18,
                away_fta=20, away_tov=13, away_orb=6, away_drb=25,
            )
            for i in range(1, 10)
        ]

        team_games: list = []
        for r in records:
            team_games.extend(r.to_team_game_rows())

        game_records = team_games_to_game_records(team_games, 2024)

        # Each IngestionGameRecord → 2 team-perspective GameRecords
        assert len(game_records) == len(records) * 2

        # Duke records: fgm=28, fga=60, tov=11, orb=8, drb=27
        duke_records = [g for g in game_records if g.team_id == "duke"]
        assert len(duke_records) == len(records)
        for gr in duke_records:
            assert gr.fgm == pytest.approx(28.0)
            assert gr.fga == pytest.approx(60.0)
            assert gr.tov == pytest.approx(11.0)
            assert gr.orb == pytest.approx(8.0)
            assert gr.drb == pytest.approx(27.0)

        # UNC records: fgm=25, fga=58, tov=13, orb=6, drb=25
        unc_records = [g for g in game_records if g.team_id in ("unc", "north_carolina")]
        assert len(unc_records) == len(records)
        for gr in unc_records:
            assert gr.fgm == pytest.approx(25.0)
            assert gr.tov == pytest.approx(13.0)

    def test_opponent_fields_populated_from_mirror_row(self):
        """team_games_to_game_records should cross-populate opp_* from the mirror row."""
        from src.data.features.proprietary_metrics import team_games_to_game_records

        rec = IngestionGameRecord(
            game_id="g1", date="2024-01-15", season=2024,
            home_team_id="duke", home_team_name="Duke",
            away_team_id="unc", away_team_name="UNC",
            home_score=82, away_score=79,
            home_fgm=28, home_fga=60, home_fg3m=8, home_fg3a=22,
            home_fta=15, home_tov=11, home_orb=8, home_drb=27,
            away_fgm=25, away_fga=58, away_fg3m=6, away_fg3a=18,
            away_fta=20, away_tov=13, away_orb=6, away_drb=25,
        )
        team_games = rec.to_team_game_rows()
        game_records = team_games_to_game_records(team_games, 2024)

        duke_rec = next(g for g in game_records if g.team_id == "duke")
        # Duke's opp_* should be populated from UNC's row
        assert duke_rec.opp_fgm == pytest.approx(25.0)
        assert duke_rec.opp_fga == pytest.approx(58.0)
        assert duke_rec.opp_tov == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# HistoricalGameFetcher — quality gate tests
# ---------------------------------------------------------------------------

class TestHistoricalGameFetcherQualityGate:
    """Tests that fetch_season raises IngestionQualityError on bad data."""

    @staticmethod
    def _make_records(n, home_id="a", away_id="b", fga=60.0, provider="espn"):
        return [
            IngestionGameRecord(
                game_id=f"g{i}", date=f"2024-01-{(i % 28) + 1:02d}", season=2024,
                home_team_id=home_id, home_team_name=home_id.upper(),
                away_team_id=away_id, away_team_name=away_id.upper(),
                home_score=80, away_score=75,
                home_fga=fga, away_fga=fga,
                provider=provider,
            )
            for i in range(n)
        ]

    def test_raises_on_insufficient_games(self, tmp_path, monkeypatch):
        from src.data.ingestion.game_fetchers import IngestionQualityError

        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: self._make_records(10))
        monkeypatch.setattr(fetcher, "_fetch_via_sportsdataverse", lambda s: [])
        monkeypatch.setattr(fetcher, "_fetch_via_cbbpy", lambda s: [])

        with pytest.raises(IngestionQualityError, match="insufficient games"):
            fetcher.fetch_season(2024)

    def test_raises_on_insufficient_teams(self, tmp_path, monkeypatch):
        from src.data.ingestion.game_fetchers import IngestionQualityError

        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        # 600 games but only 2 teams — fails team coverage gate
        records = self._make_records(600, home_id="duke", away_id="unc")
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: records)

        with pytest.raises(IngestionQualityError, match="team coverage"):
            fetcher.fetch_season(2024)

    def test_strict_false_allows_insufficient(self, tmp_path, monkeypatch):
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: self._make_records(10))
        monkeypatch.setattr(fetcher, "_fetch_via_sportsdataverse", lambda s: [])
        monkeypatch.setattr(fetcher, "_fetch_via_cbbpy", lambda s: [])

        result = fetcher.fetch_season(2024, strict=False)
        assert len(result) == 10

    def test_provider_contributions_logged(self, tmp_path, monkeypatch, caplog):
        fetcher = HistoricalGameFetcher(cache_dir=str(tmp_path))
        espn = self._make_records(5, provider="espn_scoreboard")
        sdv = self._make_records(5, provider="sportsdataverse")
        # Use different game_ids to avoid dedup
        for i, r in enumerate(sdv):
            r.game_id = f"sdv_{i}"
        monkeypatch.setattr(fetcher, "_fetch_via_espn_scoreboard", lambda s: espn)
        monkeypatch.setattr(fetcher, "_fetch_via_sportsdataverse", lambda s: sdv)
        monkeypatch.setattr(fetcher, "_fetch_via_cbbpy", lambda s: [])

        with caplog.at_level(logging.INFO):
            fetcher.fetch_season(2024, strict=False)

        assert any("provider contributions" in m for m in caplog.messages)
