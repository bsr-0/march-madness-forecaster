"""The ingest path must import, and say what is missing when asked for a deleted provider.

`src/data/ingestion/collector.py` imported five scrapers by name that commit
44b048f (2026-04-21) deleted, so `src.data.ingestion` -- and with it the
README's `ingest` and `ingest-historical` commands, the first step of the
March 2027 runbook -- raised ImportError on import for five months. The
providers are optional now: absent unless restored, and any config option
that needs one fails with a message naming the commit.
"""

import pytest


def test_ingestion_package_imports():
    import src.data.ingestion  # noqa: F401
    from src.data.ingestion import collector

    assert collector.NCAAStatsScraper is None or callable(collector.NCAAStatsScraper)


def test_default_ingestion_config_does_not_need_deleted_providers():
    from src.data.ingestion.collector import IngestionConfig

    cfg = IngestionConfig(year=2030)
    for opt in ("ncaa_teams_url", "ncaa_games_url", "roster_url", "transfer_portal_url", "odds_url"):
        assert not getattr(cfg, opt, None), f"{opt} defaults to a value that needs a deleted provider"


def test_requesting_a_deleted_provider_names_the_commit():
    from src.data.ingestion.collector import _require

    with pytest.raises(RuntimeError) as exc:
        _require(None, "NCAAStatsScraper", "ncaa_teams_url")
    assert "44b048f" in str(exc.value)
    assert "ncaa_teams_url" in str(exc.value)
    sentinel = object()
    assert _require(sentinel, "x", "y") is sentinel


def test_historical_pipeline_constructs_without_sports_reference(tmp_path):
    from src.data.ingestion.historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig

    cfg = HistoricalIngestionConfig(start_season=2030, end_season=2030, cache_dir=str(tmp_path))
    pipe = HistoricalDataPipeline(cfg)
    assert pipe is not None
