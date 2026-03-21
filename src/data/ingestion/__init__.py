"""Data ingestion workflows for real-world sources."""

from .collector import IngestionConfig, RealDataCollector
from .game_fetchers import HistoricalGameFetcher, IncrementalGameFetcher, dedup_records, is_in_season_window, season_window
from .historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig
from .schemas import IngestionGameRecord

__all__ = [
    "IngestionConfig",
    "RealDataCollector",
    "HistoricalDataPipeline",
    "HistoricalIngestionConfig",
    "HistoricalGameFetcher",
    "IncrementalGameFetcher",
    "IngestionGameRecord",
    "dedup_records",
    "is_in_season_window",
    "season_window",
]
