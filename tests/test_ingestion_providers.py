"""Unit tests for provider-priority behavior."""

import pandas as pd

from src.data.ingestion.providers import LibraryProviderHub, ProviderResult


class StubProviderHub(LibraryProviderHub):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _from_sportsdataverse_pbp(self, year, since=None):
        self.calls.append("sportsdataverse")
        return ProviderResult("sportsdataverse", [])

    def _from_cbbpy_pbp(self, year, since=None):
        self.calls.append("cbbpy")
        return ProviderResult("cbbpy", [{"game_id": "g1"}])

    def _from_cbbdata_games_api(self, year, since=None):
        self.calls.append("cbbdata")
        return ProviderResult("cbbdata", [{"game_id": "g2"}])


def test_provider_priority_uses_custom_order():
    hub = StubProviderHub()
    result = hub.fetch_historical_games(2026, priority=["cbbdata", "cbbpy", "sportsdataverse"])

    assert result.provider == "cbbdata"
    assert hub.calls == ["cbbdata"]


def test_provider_priority_falls_through_to_next():
    class FallthroughHub(StubProviderHub):
        def _from_cbbdata_games_api(self, year, since=None):
            self.calls.append("cbbdata")
            return ProviderResult("cbbdata", [])

    hub = FallthroughHub()
    result = hub.fetch_historical_games(2026, priority=["cbbdata", "cbbpy"])

    assert result.provider == "cbbpy"
    assert hub.calls == ["cbbdata", "cbbpy"]


def test_unknown_provider_in_priority_is_ignored():
    hub = StubProviderHub()
    result = hub.fetch_historical_games(2026, priority=["unknown", "cbbpy"])

    assert result.provider == "cbbpy"
    assert hub.calls == ["cbbpy"]


def test_cbbpy_provider_normalizes_tuple_boxscore(monkeypatch):
    class DummyCBBpy:
        @staticmethod
        def get_games_season(year, info=True, box=True, pbp=False):
            info_df = pd.DataFrame([
                {"game_id": "401", "game_day": "January 15, 2025", "home_team": "Duke", "away_team": "UNC", "home_score": 18, "away_score": 21},
            ])
            box_df = pd.DataFrame(
                [
                    {"game_id": "401", "team": "Duke", "pts": 10, "fgm": 4, "fga": 8, "3pm": 2, "3pa": 4, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
                    {"game_id": "401", "team": "Duke", "pts": 8, "fgm": 3, "fga": 7, "3pm": 1, "3pa": 3, "fta": 2, "to": 2, "oreb": 1, "dreb": 3},
                    {"game_id": "401", "team": "UNC", "pts": 12, "fgm": 5, "fga": 9, "3pm": 1, "3pa": 3, "fta": 3, "to": 3, "oreb": 2, "dreb": 4},
                    {"game_id": "401", "team": "UNC", "pts": 9, "fgm": 4, "fga": 8, "3pm": 2, "3pa": 4, "fta": 2, "to": 1, "oreb": 0, "dreb": 2},
                ]
            )
            return (info_df, box_df, pd.DataFrame())

    hub = LibraryProviderHub()
    monkeypatch.setattr(hub, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None)

    result = hub.fetch_historical_games(2025, priority=["cbbpy"])

    assert result.provider == "cbbpy"
    assert len(result.records) == 2
    assert result.records[0]["game_id"] == "401"
    assert {"duke", "north_carolina"} == {result.records[0]["team_id"], result.records[1]["team_id"]}
    assert all("opponent_id" in row for row in result.records)
    # Verify dates are extracted from info DataFrame
    assert all(row.get("date") == "2025-01-15" for row in result.records)


def test_cbbpy_provider_incremental_skips_season_fallback(monkeypatch):
    """When since is set and get_games_range fails, cbbpy should NOT fall
    back to get_games_season (which scrapes from November).  Instead it
    returns empty so the next provider can handle it."""
    class DummyCBBpy:
        @staticmethod
        def get_games_range(start, end, info=True, box=True, pbp=False):
            raise Exception("Network error")

        @staticmethod
        def get_games_season(year, info=True, box=True, pbp=False):
            raise AssertionError("get_games_season should not be called for incremental fetches")

    hub = LibraryProviderHub()
    monkeypatch.setattr(hub, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None)

    result = hub.fetch_historical_games(2025, priority=["cbbpy"], since="2025-03-01")
    # cbbpy should return empty (get_games_range failed, no season fallback)
    assert result.provider == "none"
    assert result.records == []


def test_cbbpy_provider_incremental_keeps_records_with_empty_dates(monkeypatch):
    """Records with empty dates should be kept during incremental fetches,
    not silently dropped — they likely represent recent games."""
    class DummyCBBpy:
        @staticmethod
        def get_games_range(start, end, info=True, box=True, pbp=False):
            # Info only has a date for game 601, not 602
            info_df = pd.DataFrame([
                {"game_id": "601", "game_day": "January 15, 2025"},
            ])
            box_df = pd.DataFrame([
                {"game_id": "601", "team": "A", "player": "p1", "pts": 10, "fgm": 4, "fga": 8, "3pm": 1, "3pa": 3, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
                {"game_id": "601", "team": "B", "player": "p2", "pts": 8, "fgm": 3, "fga": 6, "3pm": 0, "3pa": 2, "fta": 2, "to": 0, "oreb": 0, "dreb": 1},
                {"game_id": "602", "team": "C", "player": "p3", "pts": 12, "fgm": 5, "fga": 9, "3pm": 1, "3pa": 3, "fta": 1, "to": 2, "oreb": 1, "dreb": 3},
                {"game_id": "602", "team": "D", "player": "p4", "pts": 6, "fgm": 2, "fga": 5, "3pm": 1, "3pa": 2, "fta": 1, "to": 1, "oreb": 0, "dreb": 2},
            ])
            return (info_df, box_df, pd.DataFrame())

    hub = LibraryProviderHub()
    monkeypatch.setattr(hub, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None)

    result = hub.fetch_historical_games(2025, priority=["cbbpy"], since="2025-03-01")
    assert result.provider == "cbbpy"
    # Game 602 has no date — should still be included (not filtered out)
    game_ids = {r["game_id"] for r in result.records}
    assert "602" in game_ids, "Records with empty dates should be kept during incremental fetch"


def test_cbbpy_provider_extracts_dates_from_info(monkeypatch):
    """Verify _from_cbbpy_pbp extracts dates from info DataFrame."""
    class DummyCBBpy:
        @staticmethod
        def get_games_season(year, info=True, box=True, pbp=False):
            info_df = pd.DataFrame([
                {"game_id": "501", "game_day": "November 06, 2024", "home_team": "A", "away_team": "B", "home_score": 70, "away_score": 65},
                {"game_id": "502", "game_day": "February 10, 2025", "home_team": "C", "away_team": "D", "home_score": 80, "away_score": 75},
            ])
            box_df = pd.DataFrame([
                {"game_id": "501", "team": "A", "player": "p1", "pts": 10, "fgm": 4, "fga": 8, "3pm": 1, "3pa": 3, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
                {"game_id": "501", "team": "B", "player": "p2", "pts": 8, "fgm": 3, "fga": 6, "3pm": 0, "3pa": 2, "fta": 2, "to": 0, "oreb": 0, "dreb": 1},
                {"game_id": "502", "team": "C", "player": "p3", "pts": 12, "fgm": 5, "fga": 9, "3pm": 1, "3pa": 3, "fta": 1, "to": 2, "oreb": 1, "dreb": 3},
                {"game_id": "502", "team": "D", "player": "p4", "pts": 6, "fgm": 2, "fga": 5, "3pm": 1, "3pa": 2, "fta": 1, "to": 1, "oreb": 0, "dreb": 2},
            ])
            return (info_df, box_df, pd.DataFrame())

    hub = LibraryProviderHub()
    monkeypatch.setattr(hub, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None)

    result = hub.fetch_historical_games(2025, priority=["cbbpy"])
    assert result.provider == "cbbpy"

    date_by_game = {}
    for row in result.records:
        date_by_game[row["game_id"]] = row.get("date", "")

    assert date_by_game["501"] == "2024-11-06"
    assert date_by_game["502"] == "2025-02-10"


def test_map_barttorvik_row_extracts_coach():
    row = {
        "Team": "Duke",
        "Conf": "ACC",
        "AdjOE": "120.5",
        "AdjDE": "95.2",
        "AdjT": "70.1",
        "Barthag": "0.95",
        "Coach": "Jon Scheyer",
    }
    result = LibraryProviderHub._map_barttorvik_row(row)
    assert result is not None
    assert result["coach"] == "Jon Scheyer"


def test_map_barttorvik_row_no_coach_column():
    row = {
        "Team": "Duke",
        "Conf": "ACC",
        "AdjOE": "120.5",
        "AdjDE": "95.2",
    }
    result = LibraryProviderHub._map_barttorvik_row(row)
    assert result is not None
    assert "coach" not in result


class FullStubProviderHub(LibraryProviderHub):
    """Stub with all providers returning empty except those overridden."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def _from_sportsdataverse_pbp(self, year, since=None):
        self.calls.append("sportsdataverse")
        return ProviderResult("sportsdataverse", [])

    def _from_cbbpy_pbp(self, year, since=None):
        self.calls.append("cbbpy")
        return ProviderResult("cbbpy", [{"game_id": "g1"}])

    def _from_espn_scoreboard_api(self, year, since=None):
        self.calls.append("espn_scoreboard")
        return ProviderResult("espn_scoreboard", [{"game_id": "g2"}])

    def _from_sportsipy_games_api(self, year, since=None):
        self.calls.append("sportsipy")
        return ProviderResult("sportsipy", [])

    def _from_cbbdata_games_api(self, year, since=None):
        self.calls.append("cbbdata")
        return ProviderResult("cbbdata", [])


def test_incremental_fetch_prefers_espn_over_cbbpy():
    """When since is provided, espn_scoreboard should be tried before cbbpy."""
    hub = FullStubProviderHub()
    result = hub.fetch_historical_games(2026, since="2026-03-05")

    assert result.provider == "espn_scoreboard"
    assert hub.calls == ["sportsdataverse", "espn_scoreboard"]


def test_full_season_fetch_keeps_cbbpy_before_espn():
    """Without since, cbbpy should be tried before espn_scoreboard."""
    hub = FullStubProviderHub()
    result = hub.fetch_historical_games(2026)

    assert result.provider == "cbbpy"
    assert hub.calls == ["sportsdataverse", "cbbpy"]


def test_explicit_priority_overrides_incremental_default():
    """Explicit priority should be honored even with since set."""
    hub = FullStubProviderHub()
    result = hub.fetch_historical_games(2026, priority=["cbbpy"], since="2026-03-05")

    assert result.provider == "cbbpy"
    assert hub.calls == ["cbbpy"]
