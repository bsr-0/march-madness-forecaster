"""Tests for game date preservation, validation, and repair."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig
from src.data.ingestion.providers import LibraryProviderHub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_info_df(games):
    """Create a cbbpy-style info DataFrame with game_id and game_day columns."""
    rows = []
    for gid, day_str in games:
        rows.append({
            "game_id": gid,
            "game_day": day_str,  # e.g. "January 15, 2024"
            "game_time": "07:00 PM PST",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_score": 70,
            "away_score": 65,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _make_box_df(game_ids):
    """Create a cbbpy-style box score DataFrame for given game IDs."""
    rows = []
    for gid in game_ids:
        # Two players per team, two teams
        rows.extend([
            {"game_id": gid, "team": "Team A", "player": "Player 1", "pts": 10, "fgm": 4, "fga": 8, "3pm": 1, "3pa": 3, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
            {"game_id": gid, "team": "Team A", "player": "Player 2", "pts": 8, "fgm": 3, "fga": 6, "3pm": 0, "3pa": 2, "fta": 2, "to": 0, "oreb": 0, "dreb": 1},
            {"game_id": gid, "team": "Team B", "player": "Player 3", "pts": 12, "fgm": 5, "fga": 9, "3pm": 1, "3pa": 3, "fta": 1, "to": 2, "oreb": 1, "dreb": 3},
            {"game_id": gid, "team": "Team B", "player": "Player 4", "pts": 6, "fgm": 2, "fga": 5, "3pm": 1, "3pa": 2, "fta": 1, "to": 1, "oreb": 0, "dreb": 2},
        ])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Test: Fast path preserves dates from info DataFrame
# ---------------------------------------------------------------------------

class TestFastPathPreservesDates:
    """Verify _collect_season_games_fast extracts dates from info DataFrame."""

    def _make_pipeline(self, tmp_path):
        config = HistoricalIngestionConfig(
            start_season=2024,
            end_season=2024,
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
        )
        return HistoricalDataPipeline(config)

    def test_dates_extracted_from_info_df(self, tmp_path):
        """When info DataFrame has game_day, dates should be parsed and applied."""
        pipeline = self._make_pipeline(tmp_path)

        info_df = _make_info_df([
            ("401", "November 06, 2023"),
            ("402", "December 15, 2023"),
            ("403", "February 20, 2024"),
        ])
        box_df = _make_box_df(["401", "402", "403"])

        class MockScraper:
            @staticmethod
            def get_games_season(season, info=True, box=True, pbp=False):
                return (info_df, box_df, pd.DataFrame())

        result = pipeline._collect_season_games_fast(2024, MockScraper)
        assert result is not None

        games = result["games"]
        assert len(games) == 3

        date_map = {g["game_id"]: g["date"] for g in games}
        assert date_map["401"] == "2023-11-06"
        assert date_map["402"] == "2023-12-15"
        assert date_map["403"] == "2024-02-20"

        # Also check team_games
        for tg in result["team_games"]:
            gid = tg["game_id"]
            assert tg["date"] == date_map[gid]

    def test_fallback_when_info_empty(self, tmp_path):
        """When info DataFrame is empty, fallback date should be used."""
        pipeline = self._make_pipeline(tmp_path)

        box_df = _make_box_df(["401"])

        class MockScraper:
            @staticmethod
            def get_games_season(season, info=True, box=True, pbp=False):
                return (pd.DataFrame(), box_df, pd.DataFrame())

        result = pipeline._collect_season_games_fast(2024, MockScraper)
        assert result is not None
        # After the fix, missing dates produce empty string (not fake Nov 1)
        # so the offline date repair script can detect and fix them.
        assert result["games"][0]["date"] == ""

    def test_partial_info_uses_fallback_for_missing(self, tmp_path):
        """Games without info entries get fallback; those with info get real dates."""
        pipeline = self._make_pipeline(tmp_path)

        info_df = _make_info_df([("401", "January 10, 2024")])
        box_df = _make_box_df(["401", "402"])

        class MockScraper:
            @staticmethod
            def get_games_season(season, info=True, box=True, pbp=False):
                return (info_df, box_df, pd.DataFrame())

        result = pipeline._collect_season_games_fast(2024, MockScraper)
        assert result is not None

        date_map = {g["game_id"]: g["date"] for g in result["games"]}
        assert date_map["401"] == "2024-01-10"
        assert date_map["402"] == ""  # missing date (no longer uses fake Nov 1 fallback)


# ---------------------------------------------------------------------------
# Test: _aggregate_cbbpy_box_rows preserves date field
# ---------------------------------------------------------------------------

class TestAggregateBoxRowsDate:
    """Verify _aggregate_cbbpy_box_rows passes through date field if present."""

    def test_date_preserved_from_input_rows(self):
        hub = LibraryProviderHub()
        rows = [
            {"game_id": "501", "team": "X", "player": "p1", "pts": 5, "fgm": 2, "fga": 4, "3pm": 0, "3pa": 1, "fta": 1, "to": 0, "oreb": 0, "dreb": 1, "date": "2024-01-15"},
            {"game_id": "501", "team": "Y", "player": "p2", "pts": 7, "fgm": 3, "fga": 5, "3pm": 1, "3pa": 2, "fta": 0, "to": 1, "oreb": 1, "dreb": 2, "date": "2024-01-15"},
        ]
        result = hub._aggregate_cbbpy_box_rows(rows)
        assert len(result) == 2
        for rec in result:
            assert rec.get("date") == "2024-01-15"

    def test_no_date_in_input_means_no_date_in_output(self):
        hub = LibraryProviderHub()
        rows = [
            {"game_id": "501", "team": "X", "player": "p1", "pts": 5, "fgm": 2, "fga": 4, "3pm": 0, "3pa": 1, "fta": 1, "to": 0, "oreb": 0, "dreb": 1},
            {"game_id": "501", "team": "Y", "player": "p2", "pts": 7, "fgm": 3, "fga": 5, "3pm": 1, "3pa": 2, "fta": 0, "to": 1, "oreb": 1, "dreb": 2},
        ]
        result = hub._aggregate_cbbpy_box_rows(rows)
        assert len(result) == 2
        # date key should not be present when no input date
        for rec in result:
            assert rec.get("date", "") == ""


# ---------------------------------------------------------------------------
# Test: Date validation
# ---------------------------------------------------------------------------

class TestDateValidation:
    """Verify _validate_game_dates catches suspicious date distributions."""

    def test_catches_all_fallback_dates(self):
        games = [{"date": "2023-11-01"} for _ in range(100)]
        warnings = HistoricalDataPipeline._validate_game_dates(games, 2024)
        assert len(warnings) >= 1
        assert "CRITICAL" in warnings[0]

    def test_catches_low_diversity(self):
        # 5 unique dates across 200 games
        games = [{"date": f"2024-0{(i % 5) + 1}-01"} for i in range(200)]
        warnings = HistoricalDataPipeline._validate_game_dates(games, 2025)
        assert any("unique dates" in w.lower() for w in warnings)

    def test_passes_good_data(self):
        # 150 unique dates
        games = [{"date": f"2023-{((i % 5) + 11):02d}-{(i % 28) + 1:02d}"} for i in range(150)]
        # Ensure at least 10 unique
        unique = len(set(g["date"] for g in games))
        assert unique >= 10
        warnings = HistoricalDataPipeline._validate_game_dates(games, 2024)
        assert len(warnings) == 0

    def test_catches_empty_dates(self):
        games = [{"date": ""}, {"date": "2024-01-15"}, {"date": ""}, {}]
        warnings = HistoricalDataPipeline._validate_game_dates(games, 2025)
        assert any("empty or missing" in w.lower() for w in warnings)

    def test_empty_games_no_warnings(self):
        warnings = HistoricalDataPipeline._validate_game_dates([], 2024)
        assert warnings == []


# ---------------------------------------------------------------------------
# Test: repair_historical_dates
# ---------------------------------------------------------------------------

class TestRepairHistoricalDates:
    """Verify the repair method correctly patches dates in JSON files."""

    def _write_historical_file(self, tmp_path, season, games, team_games=None):
        path = tmp_path / f"historical_games_{season}.json"
        data = {
            "season": season,
            "provider": "cbbpy",
            "games": games,
            "team_games": team_games or [],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_repair_patches_fallback_dates(self, tmp_path):
        """Repair should replace fallback dates with real ones."""
        games = [
            {"game_id": "401", "season": 2024, "date": "2023-11-01"},
            {"game_id": "402", "season": 2024, "date": "2023-11-01"},
        ]
        team_games = [
            {"game_id": "401", "date": "2023-11-01", "team_id": "a"},
            {"game_id": "402", "date": "2023-11-01", "team_id": "b"},
        ]
        self._write_historical_file(tmp_path, 2024, games, team_games)

        config = HistoricalIngestionConfig(output_dir=str(tmp_path))
        pipeline = HistoricalDataPipeline(config)

        # Mock the date fetcher
        date_map = {"401": "2024-01-10", "402": "2024-02-15"}
        pipeline._fetch_date_map_for_season = lambda s, scr, force_slow=False: date_map
        pipeline.providers._import_module = lambda m: object()  # dummy scraper

        results = pipeline.repair_historical_dates(seasons=[2024])
        assert results[2024]["repaired"] == 2
        assert results[2024]["unique_dates"] == 2

        # Verify file was written with correct dates
        with open(tmp_path / "historical_games_2024.json") as f:
            data = json.load(f)
        assert data["games"][0]["date"] == "2024-01-10"
        assert data["games"][1]["date"] == "2024-02-15"
        assert data["team_games"][0]["date"] == "2024-01-10"
        assert data["team_games"][1]["date"] == "2024-02-15"

    def test_dry_run_does_not_modify(self, tmp_path):
        """Dry run should report changes but not write."""
        games = [{"game_id": "401", "season": 2024, "date": "2023-11-01"}]
        self._write_historical_file(tmp_path, 2024, games)

        config = HistoricalIngestionConfig(output_dir=str(tmp_path))
        pipeline = HistoricalDataPipeline(config)

        date_map = {"401": "2024-01-10"}
        pipeline._fetch_date_map_for_season = lambda s, scr, force_slow=False: date_map
        pipeline.providers._import_module = lambda m: object()

        results = pipeline.repair_historical_dates(seasons=[2024], dry_run=True)
        assert results[2024]["repaired"] == 1

        # File should still have old date
        with open(tmp_path / "historical_games_2024.json") as f:
            data = json.load(f)
        assert data["games"][0]["date"] == "2023-11-01"

    def test_repair_skips_already_correct(self, tmp_path):
        """Games with correct dates should not be counted as repaired."""
        games = [{"game_id": "401", "season": 2024, "date": "2024-01-10"}]
        self._write_historical_file(tmp_path, 2024, games)

        config = HistoricalIngestionConfig(output_dir=str(tmp_path))
        pipeline = HistoricalDataPipeline(config)

        date_map = {"401": "2024-01-10"}
        pipeline._fetch_date_map_for_season = lambda s, scr, force_slow=False: date_map
        pipeline.providers._import_module = lambda m: object()

        results = pipeline.repair_historical_dates(seasons=[2024])
        assert results[2024]["repaired"] == 0

    def test_auto_discovers_seasons(self, tmp_path):
        """Without explicit seasons, should discover all historical files."""
        for yr in [2022, 2023]:
            self._write_historical_file(
                tmp_path, yr, [{"game_id": "1", "date": f"{yr-1}-11-01"}]
            )

        config = HistoricalIngestionConfig(output_dir=str(tmp_path))
        pipeline = HistoricalDataPipeline(config)

        date_map = {"1": "2023-01-15"}
        pipeline._fetch_date_map_for_season = lambda s, scr, force_slow=False: date_map
        pipeline.providers._import_module = lambda m: object()

        results = pipeline.repair_historical_dates()
        assert 2022 in results
        assert 2023 in results


# ---------------------------------------------------------------------------
# Test: Fast path rejects bad dates
# ---------------------------------------------------------------------------

class TestFastPathRejectsBadDates:
    """Verify _collect_season_games_fast returns None when dates are all fallback."""

    def test_fast_path_returns_none_on_all_fallback_dates(self, tmp_path):
        """When info DataFrame is empty (no dates), fast path should return None."""
        config = HistoricalIngestionConfig(
            start_season=2024, end_season=2024,
            output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"),
        )
        pipeline = HistoricalDataPipeline(config)

        # Create box data with many games but no info DataFrame dates.
        # With enough games (> 100), _validate_game_dates will trigger the
        # low-diversity warning (only 1 unique empty date).
        game_ids = [str(i) for i in range(401, 601)]  # 200 games
        box_rows = []
        for gid in game_ids:
            box_rows.extend([
                {"game_id": gid, "team": "A", "player": "p1", "pts": 10, "fgm": 4, "fga": 8, "3pm": 1, "3pa": 3, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
                {"game_id": gid, "team": "B", "player": "p2", "pts": 8, "fgm": 3, "fga": 6, "3pm": 0, "3pa": 2, "fta": 2, "to": 0, "oreb": 0, "dreb": 1},
            ])
        box_df = pd.DataFrame(box_rows)

        class MockScraper:
            @staticmethod
            def get_games_season(season, info=True, box=True, pbp=False):
                return (pd.DataFrame(), box_df, pd.DataFrame())

        result = pipeline._collect_season_games_fast(2024, MockScraper)
        # With 200 games all having empty dates (only 1 unique date value),
        # validation should flag this as critically low diversity and reject.
        assert result is None


# ---------------------------------------------------------------------------
# Test: Cache save blocks completion on bad dates
# ---------------------------------------------------------------------------

class TestCacheSaveBlocksBadDates:
    """Verify _save_season_cache refuses to mark complete with bad dates."""

    def test_save_cache_downgrades_complete_on_critical_dates(self, tmp_path):
        config = HistoricalIngestionConfig(
            start_season=2024, end_season=2024,
            output_dir=str(tmp_path), cache_dir=str(tmp_path / "cache"),
        )
        pipeline = HistoricalDataPipeline(config)

        # All games with fallback date
        games = [{"game_id": str(i), "date": "2023-11-01", "season": 2024}
                 for i in range(200)]

        cache_file = Path(tmp_path / "cache" / "test_cache.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        result = pipeline._save_season_cache(
            cache_file, 2024, games, [], [], [], complete=True,
        )
        # Should have downgraded complete to False due to critical date issue
        assert result["complete"] is False

        # Verify the file on disk also has complete=False
        with open(cache_file) as f:
            saved = json.load(f)
        assert saved["complete"] is False


# ---------------------------------------------------------------------------
# Test: _extract_date_map_from_info
# ---------------------------------------------------------------------------

class TestExtractDateMapFromInfo:
    """Verify providers._extract_date_map_from_info extracts dates correctly."""

    def test_extracts_dates(self):
        info_df = pd.DataFrame([
            {"game_id": "100", "game_day": "November 10, 2023"},
            {"game_id": "200", "game_day": "March 15, 2024"},
        ])
        result = LibraryProviderHub._extract_date_map_from_info(info_df)
        assert result == {"100": "2023-11-10", "200": "2024-03-15"}

    def test_empty_df_returns_empty(self):
        result = LibraryProviderHub._extract_date_map_from_info(pd.DataFrame())
        assert result == {}

    def test_none_returns_empty(self):
        result = LibraryProviderHub._extract_date_map_from_info(None)
        assert result == {}


# ---------------------------------------------------------------------------
# Regression tests: scan on-disk historical data files for date corruption
# ---------------------------------------------------------------------------

import glob
import re
from collections import Counter


def _collect_historical_game_files():
    """Find all historical_games_*.json files across data directories."""
    patterns = [
        "data/raw/historical/historical_games_*.json",
        "data/raw/historical_full/historical_games_*.json",
    ]
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    return files


def _load_games_file(path):
    """Load a historical games JSON and return (season, games, team_games)."""
    with open(path) as f:
        data = json.load(f)
    season = data.get("season")
    games = data.get("games", [])
    team_games = data.get("team_games", [])
    return season, games, team_games


class TestHistoricalDataFileDateRegression:
    """Regression tests that scan committed data files for the date corruption
    bug where all games shared a single fallback date ({season-1}-11-01).

    These tests act as a tripwire: even if a code-level guard is bypassed,
    corrupted data in the repository will cause an immediate test failure.
    """

    def _get_files_or_skip(self):
        files = _collect_historical_game_files()
        if not files:
            pytest.skip("No historical game data files found on disk")
        return files

    def test_no_historical_games_have_nov1_fallback_date(self):
        """No game should have the {season-1}-11-01 fallback date.

        This was the signature of the original bug: cbbpy was called with
        info=False, so game dates defaulted to Nov 1 of the prior year.
        """
        files = self._get_files_or_skip()
        violations = []
        for path in files:
            season, games, team_games = _load_games_file(path)
            if not games or season is None:
                continue
            fallback = f"{season - 1}-11-01"
            bad_games = [g for g in games if g.get("date") == fallback]
            if bad_games:
                violations.append(
                    f"{path}: {len(bad_games)}/{len(games)} games have "
                    f"fallback date {fallback}"
                )
            bad_tg = [tg for tg in team_games if tg.get("date") == fallback]
            if bad_tg:
                violations.append(
                    f"{path}: {len(bad_tg)}/{len(team_games)} team_games have "
                    f"fallback date {fallback}"
                )
        assert not violations, (
            "Games with fallback Nov 1 dates found (date bug regression):\n"
            + "\n".join(violations)
        )

    def test_historical_games_have_sufficient_date_diversity(self):
        """Seasons with 100+ games must have at least 50 unique dates.

        Real NCAA seasons have 154-155 game days. Anything below 50 indicates
        dates are missing, duplicated, or fabricated.
        """
        files = self._get_files_or_skip()
        violations = []
        for path in files:
            season, games, _ = _load_games_file(path)
            if not games or len(games) < 100:
                continue
            unique_dates = len(set(g.get("date", "") for g in games))
            if unique_dates < 50:
                violations.append(
                    f"{path}: season {season} has only {unique_dates} unique "
                    f"dates across {len(games)} games (minimum: 50)"
                )
        assert not violations, (
            "Insufficient date diversity (possible date corruption):\n"
            + "\n".join(violations)
        )

    def test_no_historical_games_have_empty_dates(self):
        """No committed game data should have empty or missing date fields."""
        files = self._get_files_or_skip()
        violations = []
        for path in files:
            season, games, team_games = _load_games_file(path)
            if not games:
                continue
            empty_games = [g for g in games if not g.get("date")]
            if empty_games:
                violations.append(
                    f"{path}: {len(empty_games)}/{len(games)} games have "
                    f"empty/missing date"
                )
            empty_tg = [tg for tg in team_games if not tg.get("date")]
            if empty_tg:
                violations.append(
                    f"{path}: {len(empty_tg)}/{len(team_games)} team_games "
                    f"have empty/missing date"
                )
        assert not violations, (
            "Games with empty dates found:\n" + "\n".join(violations)
        )

    def test_all_historical_game_dates_are_valid_iso_format(self):
        """Every game date must be a valid YYYY-MM-DD string."""
        files = self._get_files_or_skip()
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        violations = []
        for path in files:
            season, games, _ = _load_games_file(path)
            if not games:
                continue
            for g in games:
                d = g.get("date", "")
                if not d:
                    continue  # caught by the empty-date test
                if not iso_re.match(d):
                    violations.append(
                        f"{path}: game {g.get('game_id')} has non-ISO date "
                        f"'{d}'"
                    )
                    continue
                try:
                    datetime.strptime(d, "%Y-%m-%d")
                except ValueError:
                    violations.append(
                        f"{path}: game {g.get('game_id')} has unparseable "
                        f"date '{d}'"
                    )
        assert not violations, (
            "Invalid date formats found:\n" + "\n".join(violations[:20])
        )

    def test_all_historical_game_dates_fall_within_season_range(self):
        """Game dates must fall within the season window (Nov 1 to May 1).

        A date outside this range indicates a parsing error or data corruption.
        """
        files = self._get_files_or_skip()
        violations = []
        for path in files:
            season, games, _ = _load_games_file(path)
            if not games or season is None:
                continue
            season_start = datetime(season - 1, 11, 1)
            season_end = datetime(season, 5, 1)
            for g in games:
                d = g.get("date", "")
                if not d:
                    continue
                try:
                    parsed = datetime.strptime(d, "%Y-%m-%d")
                except ValueError:
                    continue  # caught by iso-format test
                if parsed < season_start or parsed > season_end:
                    violations.append(
                        f"{path}: game {g.get('game_id')} date {d} is outside "
                        f"season {season} range "
                        f"({season_start.date()} to {season_end.date()})"
                    )
        assert not violations, (
            "Dates outside season range:\n" + "\n".join(violations[:20])
        )

    def test_no_single_date_has_excessive_game_concentration(self):
        """No single date should have more than 5% of a season's games.

        The original bug put 100% of games on one date. Real NCAA schedules
        peak at ~0.7% per date. A 5% threshold catches corruption while
        allowing for heavy tournament days.
        """
        files = self._get_files_or_skip()
        violations = []
        for path in files:
            season, games, _ = _load_games_file(path)
            if not games or len(games) < 100:
                continue
            date_counts = Counter(g.get("date", "") for g in games)
            threshold = len(games) * 0.05
            for d, count in date_counts.most_common(5):
                if count > threshold:
                    violations.append(
                        f"{path}: date {d} has {count}/{len(games)} games "
                        f"({count/len(games)*100:.1f}%), exceeds 5% threshold"
                    )
        assert not violations, (
            "Excessive game concentration on single dates:\n"
            + "\n".join(violations)
        )
