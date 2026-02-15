"""Data ingestion workflows for real-world sources."""

from .collector import IngestionConfig, RealDataCollector
from .historical_pipeline import HistoricalDataPipeline, HistoricalIngestionConfig

__all__ = [
    "IngestionConfig",
    "RealDataCollector",
    "HistoricalDataPipeline",
    "HistoricalIngestionConfig",
]
