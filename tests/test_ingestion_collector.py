"""Tests for real-data collector behavior."""

import json

import src.data.ingestion.collector as collector_mod
from src.data.ingestion.collector import IngestionConfig, RealDataCollector
from src.data.ingestion.providers import ProviderResult
from src.data.scrapers.espn_picks import ConsensusData, PublicPicks


class _StubProviders:
    def fetch_historical_games(self, year, priority=None):
        return ProviderResult(
            "cbbpy",
            [
                {
                    "game_id": "g1",
                    "team_id": "duke",
                    "team_name": "Duke",
                    "opponent_id": "unc",
                    "opponent_name": "UNC",
                    "team_score": 75,
                    "opponent_score": 70,
                    "possessions": 68,
                    "fga": 55,
                    "fgm": 27,
                    "fg3a": 18,
                    "fg3m": 7,
                    "fta": 14,
                    "turnovers": 9,
                    "orb": 10,
                    "drb": 20,
                },
                {
                    "game_id": "g1",
                    "team_id": "unc",
                    "team_name": "UNC",
                    "opponent_id": "duke",
                    "opponent_name": "Duke",
                    "team_score": 70,
                    "opponent_score": 75,
                    "possessions": 68,
                    "fga": 54,
                    "fgm": 24,
                    "fg3a": 20,
                    "fg3m": 6,
                    "fta": 15,
                    "turnovers": 12,
                    "orb": 8,
                    "drb": 18,
                },
            ],
        )

    def fetch_team_box_metrics(self, year, priority=None):
        return ProviderResult("none", [])

    def fetch_torvik_ratings(self, year, priority=None):
        return ProviderResult("none", [])

    def fetch_kenpom_ratings(self, year, priority=None):
        return ProviderResult("none", [])

    def credential_requirements(self):
        return {}


def test_collector_skips_shotquality_when_real_source_missing(tmp_path):
    config = IngestionConfig(
        year=2025,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        scrape_torvik=False,
        scrape_kenpom=False,
        scrape_public_picks=False,
        scrape_sports_reference=False,
        scrape_shotquality=True,
        scrape_rosters=False,
    )
    collector = RealDataCollector(config)
    collector.providers = _StubProviders()

    manifest = collector.run()
    artifacts = manifest["artifacts"]

    assert "historical_games_json" in artifacts
    assert "kenpom_json" in artifacts
    assert "shotquality_teams_json" in artifacts
    assert "shotquality_games_json" in artifacts
    assert manifest["providers"]["shotquality_games_json"] == "open_boxscore_proxy"


def test_collector_aggregates_public_pick_sources(tmp_path, monkeypatch):
    def _mk_pick(champion_pct: float) -> ConsensusData:
        return ConsensusData(
            teams={
                "duke": PublicPicks(
                    team_id="duke",
                    team_name="Duke",
                    seed=1,
                    region="East",
                    round_of_64_pct=98.0,
                    round_of_32_pct=92.0,
                    sweet_16_pct=74.0,
                    elite_8_pct=52.0,
                    final_four_pct=31.0,
                    champion_pct=champion_pct,
                )
            }
        )

    monkeypatch.setattr(collector_mod.ESPNPicksScraper, "fetch_picks", lambda self, year: _mk_pick(20.0))
    monkeypatch.setattr(collector_mod.YahooPicksScraper, "fetch_picks", lambda self, year: _mk_pick(10.0))
    monkeypatch.setattr(collector_mod.CBSPicksScraper, "fetch_picks", lambda self, year: _mk_pick(5.0))

    config = IngestionConfig(
        year=2025,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        scrape_torvik=False,
        scrape_kenpom=False,
        scrape_public_picks=True,
        scrape_sports_reference=False,
        scrape_shotquality=False,
        scrape_rosters=False,
    )
    collector = RealDataCollector(config)
    collector.providers = _StubProviders()

    manifest = collector.run()
    artifacts = manifest["artifacts"]

    assert "public_picks_json" in artifacts
    with open(artifacts["public_picks_json"], "r") as f:
        payload = json.load(f)

    assert sorted(payload["sources"]) == ["cbs", "espn", "yahoo"]
    # Weighted source blend: 0.5*20 + 0.3*10 + 0.2*5 = 14.
    assert abs(payload["teams"]["duke"]["champion_pct"] - 14.0) < 1e-9


def test_collector_writes_roster_artifact_when_player_feed_available(tmp_path, monkeypatch):
    monkeypatch.setattr(collector_mod.CBBpyRosterScraper, "fetch_rosters", lambda self, year: {})
    monkeypatch.setattr(
        collector_mod.PlayerMetricsScraper,
        "fetch_rosters",
        lambda self, year, source_url=None, fmt="json": {
            "timestamp": "2026-03-17T12:00:00Z",
            "source": "player_metrics",
            "teams": [
                {
                    "team_id": "duke",
                    "team_name": "Duke",
                    "players": [
                        {
                            "player_id": "duke_1",
                            "name": "Player 1",
                            "rapm_offensive": 1.2,
                            "rapm_defensive": 0.4,
                            "minutes_per_game": 30,
                            "games_played": 30,
                        }
                    ],
                }
            ],
        },
    )

    config = IngestionConfig(
        year=2025,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        scrape_torvik=False,
        scrape_kenpom=False,
        scrape_public_picks=False,
        scrape_sports_reference=False,
        scrape_shotquality=False,
        scrape_rosters=True,
        roster_url="https://example.com/rosters.json",
    )
    collector = RealDataCollector(config)
    collector.providers = _StubProviders()

    manifest = collector.run()
    assert "rosters_json" in manifest["artifacts"]


def test_collector_merges_cbbpy_rosters_with_external_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        collector_mod.CBBpyRosterScraper,
        "fetch_rosters",
        lambda self, year: {
            "source": "cbbpy_schedule_boxscore",
            "teams": [
                {
                    "team_id": "duke",
                    "team_name": "Duke",
                    "players": [
                        {
                            "player_id": "duke_1",
                            "name": "Player 1",
                            "position": "PG",
                            "minutes_per_game": 32.0,
                            "games_played": 30,
                            "box_plus_minus": 1.0,
                        }
                    ],
                    "stints": [{"players": ["duke_1"], "plus_minus": 2.0, "possessions": 10.0}],
                }
            ],
        },
    )
    monkeypatch.setattr(
        collector_mod.PlayerMetricsScraper,
        "fetch_rosters",
        lambda self, year, source_url=None, fmt="json": {
            "source": "player_metrics",
            "teams": [
                {
                    "team_id": "duke",
                    "team_name": "Duke",
                    "players": [
                        {
                            "player_id": "duke_1",
                            "name": "Player 1",
                            "rapm_offensive": 1.5,
                            "rapm_defensive": 0.7,
                            "warp": 0.4,
                            "injury_status": "questionable",
                        },
                        {
                            "player_id": "duke_2",
                            "name": "Player 2",
                            "rapm_offensive": 0.2,
                            "rapm_defensive": 0.1,
                        },
                    ],
                }
            ],
        },
    )

    config = IngestionConfig(
        year=2025,
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        scrape_torvik=False,
        scrape_kenpom=False,
        scrape_public_picks=False,
        scrape_sports_reference=False,
        scrape_shotquality=False,
        scrape_rosters=True,
        roster_url="https://example.com/rosters.json",
    )
    collector = RealDataCollector(config)
    collector.providers = _StubProviders()
    manifest = collector.run()

    with open(manifest["artifacts"]["rosters_json"], "r") as f:
        payload = json.load(f)

    assert payload["source"] == "cbbpy_schedule_boxscore+player_metrics"
    duke = payload["teams"][0]
    player1 = next(p for p in duke["players"] if p["player_id"] == "duke_1")
    assert abs(player1["rapm_offensive"] - 1.5) < 1e-9
    assert player1["injury_status"] == "questionable"
    assert any(p["player_id"] == "duke_2" for p in duke["players"])
    assert isinstance(duke.get("stints"), list) and duke["stints"]
