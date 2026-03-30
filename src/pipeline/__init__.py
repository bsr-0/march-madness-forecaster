"""Pipeline exports."""

# New production-grade names
from .tournament_pipeline import TournamentPipeline, ForecastConfig, run_pipeline_to_file

# Backward-compatible aliases — prefer the names above in new code
from .sota import SOTAPipeline, SOTAPipelineConfig, run_sota_pipeline_to_file  # noqa: F401
from .womens import WomensPipeline, WomensPipelineConfig

__all__ = [
    "TournamentPipeline",
    "ForecastConfig",
    "run_pipeline_to_file",
    "SOTAPipeline",
    "SOTAPipelineConfig",
    "run_sota_pipeline_to_file",
    "WomensPipeline",
    "WomensPipelineConfig",
]
