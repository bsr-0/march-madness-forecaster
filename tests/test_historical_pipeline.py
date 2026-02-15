"""Tests for the 2022-2025 historical ingestion pipeline."""

import json

import pandas as pd

from src.data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig
from src.data.ingestion.providers import ProviderResult


def test_historical_pipeline_writes_season_artifacts(tmp_path):
    class DummyCBBpy:
        @staticmethod
        def get_game_ids(day):
            return ["401"] if day == "2021-11-01" else []

        @staticmethod
        def get_game(game_id, info=False, box=True, pbp=False):
            box_df = pd.DataFrame(
                [
                    {"game_id": game_id, "team": "Duke", "pts": 10, "fgm": 4, "fga": 8, "3pm": 2, "3pa": 4, "fta": 1, "to": 1, "oreb": 1, "dreb": 2},
                    {"game_id": game_id, "team": "Duke", "pts": 8, "fgm": 3, "fga": 7, "3pm": 1, "3pa": 3, "fta": 2, "to": 2, "oreb": 1, "dreb": 3},
                    {"game_id": game_id, "team": "UNC", "pts": 12, "fgm": 5, "fga": 9, "3pm": 1, "3pa": 3, "fta": 3, "to": 3, "oreb": 2, "dreb": 4},
                    {"game_id": game_id, "team": "UNC", "pts": 9, "fgm": 4, "fga": 8, "3pm": 2, "3pa": 4, "fta": 2, "to": 1, "oreb": 0, "dreb": 2},
                ]
            )
            return (pd.DataFrame(), box_df, pd.DataFrame())

    config = HistoricalIngestionConfig(
        start_season=2022,
        end_season=2022,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        max_games_per_season=1,
        include_tournament_context=False,
        strict_validation=True,
    )
    pipeline = HistoricalDataPipeline(config)
    pipeline.providers._import_module = lambda module: DummyCBBpy if module == "cbbpy.mens_scraper" else None
    pipeline.providers.fetch_team_box_metrics = lambda season, priority=None: ProviderResult(
        "sportsipy",
        [{"team_id": "duke", "team_name": "Duke", "adj_offensive_efficiency": 112.0, "adj_defensive_efficiency": 96.0, "adj_tempo": 68.0}],
    )

    manifest = pipeline.run()

    assert manifest["manifest_path"]
    assert "2022" in manifest["artifacts"]
    assert manifest["season_counts"]["2022"]["games"] == 1
    assert manifest["season_counts"]["2022"]["team_games"] == 2
    assert manifest["providers"]["2022"]["team_metrics_json"] == "sportsipy"


def test_historical_pipeline_falls_back_to_sports_reference(tmp_path):
    class DummyCBBpy:
        @staticmethod
        def get_game_ids(day):
            return ["401"] if day == "2021-11-01" else []

        @staticmethod
        def get_game(game_id, info=False, box=True, pbp=False):
            box_df = pd.DataFrame(
                [
                    {"game_id": game_id, "team": "A", "pts": 1, "fgm": 1, "fga": 2, "3pm": 0, "3pa": 0, "fta": 0, "to": 1, "oreb": 0, "dreb": 1},
                    {"game_id": game_id, "team": "B", "pts": 2, "fgm": 1, "fga": 3, "3pm": 0, "3pa": 1, "fta": 0, "to": 1, "oreb": 1, "dreb": 1},
                ]
            )
            return (pd.DataFrame(), box_df, pd.DataFrame())

    config = HistoricalIngestionConfig(
        start_season=2022,
        end_season=2022,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        max_games_per_season=1,
        include_tournament_context=False,
        strict_validation=True,
    )
    pipeline = HistoricalDataPipeline(config)
    pipeline.providers._import_module = lambda module: DummyCBBpy if module == "cbbpy.mens_scraper" else None
    pipeline.providers.fetch_team_box_metrics = lambda season, priority=None: ProviderResult("sportsipy", [])
    pipeline.sports_reference.fetch_team_season_stats = lambda season: [
        {"team_name": "A", "pace": 68.0, "off_rtg": 102.0, "def_rtg": 99.0, "wins": 20, "losses": 10}
    ]

    manifest = pipeline.run()

    assert manifest["providers"]["2022"]["team_metrics_json"] == "sports_reference_scraper"


def test_historical_pipeline_includes_tournament_context_when_available(tmp_path):
    class DummyCBBpy:
        @staticmethod
        def get_game_ids(day):
            return ["401"] if day == "2021-11-01" else []

        @staticmethod
        def get_game(game_id, info=False, box=True, pbp=False):
            box_df = pd.DataFrame(
                [
                    {"game_id": game_id, "team": "A", "pts": 1, "fgm": 1, "fga": 2, "3pm": 0, "3pa": 0, "fta": 0, "to": 1, "oreb": 0, "dreb": 1},
                    {"game_id": game_id, "team": "B", "pts": 2, "fgm": 1, "fga": 3, "3pm": 0, "3pa": 1, "fta": 0, "to": 1, "oreb": 1, "dreb": 1},
                ]
            )
            return (pd.DataFrame(), box_df, pd.DataFrame())

    config = HistoricalIngestionConfig(
        start_season=2022,
        end_season=2022,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        max_games_per_season=1,
        include_tournament_context=True,
        strict_validation=True,
    )
    pipeline = HistoricalDataPipeline(config)
    pipeline.providers._import_module = lambda module: DummyCBBpy if module == "cbbpy.mens_scraper" else None
    pipeline.providers.fetch_team_box_metrics = lambda season, priority=None: ProviderResult(
        "sportsipy",
        [{"team_id": "a", "team_name": "A", "adj_offensive_efficiency": 101.0, "adj_defensive_efficiency": 99.0, "adj_tempo": 68.0}],
    )
    pipeline.tournament_seed_scraper.fetch_tournament_seeds = lambda season: [
        {"season": season, "team_name": "A", "team_id": "a", "seed": 1, "region": "East", "school_slug": "a"}
    ]

    manifest = pipeline.run()

    assert "tournament_seeds_json" in manifest["artifacts"]["2022"]
    assert manifest["season_counts"]["2022"]["tournament_seed_teams"] == 1


def test_historical_pipeline_ignores_capped_cache_when_running_uncapped(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    capped_cache = cache_dir / "cbbpy_historical_games_2022.json"
    with open(capped_cache, "w") as f:
        f.write(
            '{"season": 2022, "provider": "cbbpy", "games": [{"game_id":"old"}], "team_games": [{"game_id":"old"}], '
            '"failed_game_ids": [], "complete": true, "max_games_per_season": 1}'
        )

    class DummyCBBpy:
        @staticmethod
        def get_game_ids(day):
            return ["401"] if day == "2021-11-01" else []

        @staticmethod
        def get_game(game_id, info=False, box=True, pbp=False):
            box_df = pd.DataFrame(
                [
                    {"game_id": game_id, "team": "A", "pts": 1, "fgm": 1, "fga": 2, "3pm": 0, "3pa": 0, "fta": 0, "to": 1, "oreb": 0, "dreb": 1},
                    {"game_id": game_id, "team": "B", "pts": 2, "fgm": 1, "fga": 3, "3pm": 0, "3pa": 1, "fta": 0, "to": 1, "oreb": 1, "dreb": 1},
                ]
            )
            return (pd.DataFrame(), box_df, pd.DataFrame())

    pipeline = HistoricalDataPipeline(
        HistoricalIngestionConfig(
            start_season=2022,
            end_season=2022,
            output_dir=str(tmp_path),
            cache_dir=str(cache_dir),
            max_games_per_season=None,
            include_tournament_context=False,
            strict_validation=True,
        )
    )
    pipeline.providers._import_module = lambda module: DummyCBBpy if module == "cbbpy.mens_scraper" else None
    pipeline.providers.fetch_team_box_metrics = lambda season, priority=None: ProviderResult(
        "sportsipy",
        [{"team_id": "a", "team_name": "A", "adj_offensive_efficiency": 101.0, "adj_defensive_efficiency": 99.0, "adj_tempo": 68.0}],
    )

    manifest = pipeline.run()
    games_path = manifest["artifacts"]["2022"]["historical_games_json"]
    with open(games_path, "r") as f:
        payload = json.load(f)

    assert payload["games"]
    assert payload["games"][0]["game_id"] == "401"


def test_historical_pipeline_fast_path_uses_get_games_season(tmp_path):
    class DummyCBBpy:
        @staticmethod
        def get_games_season(season, info=False, box=True, pbp=False):
            box_df = pd.DataFrame(
                [
                    {"game_id": "g1", "team": "A", "player": "P1", "pts": 10, "fgm": 4, "fga": 8, "3pm": 1, "3pa": 3, "fta": 2, "to": 1, "oreb": 1, "dreb": 2},
                    {"game_id": "g1", "team": "B", "player": "P2", "pts": 8, "fgm": 3, "fga": 7, "3pm": 1, "3pa": 4, "fta": 1, "to": 2, "oreb": 0, "dreb": 2},
                ]
            )
            return (pd.DataFrame(), box_df, pd.DataFrame())

    pipeline = HistoricalDataPipeline(
        HistoricalIngestionConfig(
            start_season=2022,
            end_season=2022,
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            max_games_per_season=None,
            include_tournament_context=False,
            strict_validation=True,
        )
    )
    pipeline.providers._import_module = lambda module: DummyCBBpy if module == "cbbpy.mens_scraper" else None
    pipeline.providers.fetch_team_box_metrics = lambda season, priority=None: ProviderResult(
        "sportsipy",
        [{"team_id": "a", "team_name": "A", "adj_offensive_efficiency": 101.0, "adj_defensive_efficiency": 99.0, "adj_tempo": 68.0}],
    )

    manifest = pipeline.run()
    games_path = manifest["artifacts"]["2022"]["historical_games_json"]
    with open(games_path, "r") as f:
        payload = json.load(f)
    assert payload["games"]
    assert payload["games"][0]["game_id"] == "g1"
