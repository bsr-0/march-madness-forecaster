"""Tests for real-data collector behavior."""

from src.data.ingestion.collector import IngestionConfig, RealDataCollector
from src.data.ingestion.providers import ProviderResult


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
    )
    collector = RealDataCollector(config)
    collector.providers = _StubProviders()

    manifest = collector.run()
    artifacts = manifest["artifacts"]

    assert "historical_games_json" in artifacts
    assert "kenpom_json" in artifacts
    assert "shotquality_teams_json" not in artifacts
    assert "shotquality_games_json" not in artifacts
