"""Unit tests for provider-priority behavior."""

import pandas as pd

from src.data.ingestion.providers import LibraryProviderHub, ProviderResult


class StubProviderHub(LibraryProviderHub):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _from_sportsdataverse_pbp(self, year):
        self.calls.append("sportsdataverse")
        return ProviderResult("sportsdataverse", [])

    def _from_cbbpy_pbp(self, year):
        self.calls.append("cbbpy")
        return ProviderResult("cbbpy", [{"game_id": "g1"}])

    def _from_cbbdata_games_api(self, year):
        self.calls.append("cbbdata")
        return ProviderResult("cbbdata", [{"game_id": "g2"}])


def test_provider_priority_uses_custom_order():
    hub = StubProviderHub()
    result = hub.fetch_historical_games(2026, priority=["cbbdata", "cbbpy", "sportsdataverse"])

    assert result.provider == "cbbdata"
    assert hub.calls == ["cbbdata"]


def test_provider_priority_falls_through_to_next():
    class FallthroughHub(StubProviderHub):
        def _from_cbbdata_games_api(self, year):
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
        def get_games_season(year, info=False, box=True, pbp=True):
            box_df = pd.DataFrame(
                [
                    {"game_id": "401", "team": "Duke", "pts": 10, "fgm": 4, "fga": 8, "3pm": 2, "3pa": 4, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
                    {"game_id": "401", "team": "Duke", "pts": 8, "fgm": 3, "fga": 7, "3pm": 1, "3pa": 3, "fta": 2, "to": 2, "oreb": 1, "dreb": 3},
                    {"game_id": "401", "team": "UNC", "pts": 12, "fgm": 5, "fga": 9, "3pm": 1, "3pa": 3, "fta": 3, "to": 3, "oreb": 2, "dreb": 4},
                    {"game_id": "401", "team": "UNC", "pts": 9, "fgm": 4, "fga": 8, "3pm": 2, "3pa": 4, "fta": 2, "to": 1, "oreb": 0, "dreb": 2},
                ]
            )
            return (pd.DataFrame(), box_df, pd.DataFrame())

    hub = LibraryProviderHub()
    monkeypatch.setattr(hub, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None)

    result = hub.fetch_historical_games(2025, priority=["cbbpy"])

    assert result.provider == "cbbpy"
    assert len(result.records) == 2
    assert result.records[0]["game_id"] == "401"
    assert {"duke", "unc"} == {result.records[0]["team_id"], result.records[1]["team_id"]}
    assert all("opponent_id" in row for row in result.records)
