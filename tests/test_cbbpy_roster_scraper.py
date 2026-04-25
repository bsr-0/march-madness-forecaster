"""Tests for cbbpy roster scraper."""

import pandas as pd

from src.data.scrapers.cbbpy_rosters import CBBpyRosterScraper


# -- Shared fixtures --------------------------------------------------------

_DUKE_PLAYERS = [
    {
        "game_id": "g1",
        "team": "Duke",
        "player": "A Guard",
        "player_id": "1001",
        "position": "G",
        "starter": True,
        "min": 34,
        "pts": 18,
        "reb": 4,
        "ast": 5,
        "stl": 1,
        "blk": 0,
        "to": 2,
        "fga": 12,
        "fgm": 6,
        "3pm": 2,
        "fg3a": 5,
        "fta": 3,
        "oreb": 1,
        "dreb": 3,
        "pf": 2,
    },
    {
        "game_id": "g1",
        "team": "Duke",
        "player": "B Wing",
        "player_id": "1002",
        "position": "F",
        "starter": True,
        "min": 30,
        "pts": 12,
        "reb": 6,
        "ast": 2,
        "stl": 0,
        "blk": 1,
        "to": 1,
        "fga": 10,
        "fgm": 4,
        "3pm": 1,
        "fg3a": 3,
        "fta": 4,
        "oreb": 2,
        "dreb": 4,
        "pf": 3,
    },
    {
        "game_id": "g1",
        "team": "Duke",
        "player": "E Bench",
        "player_id": "1003",
        "position": "G",
        "starter": False,
        "min": 15,
        "pts": 6,
        "reb": 1,
        "ast": 1,
        "stl": 0,
        "blk": 0,
        "to": 1,
        "fga": 5,
        "fgm": 2,
        "3pm": 0,
        "fg3a": 1,
        "fta": 2,
        "oreb": 0,
        "dreb": 1,
        "pf": 1,
    },
]

_UNC_PLAYERS = [
    {
        "game_id": "g1",
        "team": "UNC",
        "player": "C Guard",
        "player_id": "2001",
        "position": "G",
        "starter": True,
        "min": 35,
        "pts": 20,
        "reb": 3,
        "ast": 6,
        "stl": 1,
        "blk": 0,
        "to": 3,
        "fga": 14,
        "fgm": 7,
        "3pm": 3,
        "fg3a": 6,
        "fta": 5,
        "oreb": 1,
        "dreb": 2,
        "pf": 2,
    },
    {
        "game_id": "g1",
        "team": "UNC",
        "player": "D Big",
        "player_id": "2002",
        "position": "C",
        "starter": True,
        "min": 31,
        "pts": 10,
        "reb": 8,
        "ast": 1,
        "stl": 0,
        "blk": 2,
        "to": 2,
        "fga": 9,
        "fgm": 4,
        "3pm": 0,
        "fg3a": 0,
        "fta": 2,
        "oreb": 3,
        "dreb": 5,
        "pf": 3,
    },
]

_ALL_BOX_ROWS = _DUKE_PLAYERS + _UNC_PLAYERS


def test_cbbpy_roster_scraper_builds_rosters_with_rapm(monkeypatch, tmp_path):
    class DummyCBBpy:
        @staticmethod
        def get_games_season(year, info=False, box=True, pbp=False):
            box_df = pd.DataFrame(_ALL_BOX_ROWS)
            return (pd.DataFrame(), box_df, pd.DataFrame())

        @staticmethod
        def get_player_info(player_id):
            return pd.DataFrame([{"position": "SG", "class": "JR"}])

    scraper = CBBpyRosterScraper(cache_dir=str(tmp_path))
    monkeypatch.setattr(
        scraper, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None
    )

    payload = scraper.fetch_rosters(2026)
    assert payload
    assert payload["source"] == "cbbpy_schedule_boxscore_player_endpoint"
    assert len(payload["teams"]) == 2

    duke = next(t for t in payload["teams"] if t["team_id"] == "duke")
    assert duke["players"]
    guard = next(p for p in duke["players"] if p["player_id"] == "1001")
    assert guard["position"] == "SG"
    assert guard["eligibility_year"] == 3
    # RAPM should be computed from box-score stats (not None)
    assert guard["rapm_offensive"] is not None
    assert guard["rapm_defensive"] is not None


def test_pbp_disabled_by_default(monkeypatch, tmp_path):
    """PBP is disabled by default; no PBP rows collected."""

    class DummyCBBpy:
        @staticmethod
        def get_games_season(year, info=False, box=True, pbp=False):
            assert pbp is False, "PBP should be disabled by default"
            box_df = pd.DataFrame(_ALL_BOX_ROWS)
            return (pd.DataFrame(), box_df, pd.DataFrame())

        @staticmethod
        def get_player_info(player_id):
            return pd.DataFrame([{"position": "SG", "class": "JR"}])

    scraper = CBBpyRosterScraper(cache_dir=str(tmp_path))
    monkeypatch.setattr(
        scraper, "_import_module", lambda module_name: DummyCBBpy if module_name == "cbbpy.mens_scraper" else None
    )
    # Ensure env var is not set.
    monkeypatch.delenv("CBBPY_ROSTER_ENABLE_PBP", raising=False)

    payload = scraper.fetch_rosters(2026)
    assert payload
    assert payload["metadata"]["pbp_enabled"] is False
    assert payload["metadata"]["raw_pbp_rows"] == 0
