"""Backward-compatible re-exports.  Import from tournament_pipeline directly.

This module is a facade — all implementation has moved to:
  src/pipeline/tournament_pipeline.py  (TournamentPipeline, public API)
  src/pipeline/pipeline_runner.py      (_PipelineRunner, orchestration)
  src/pipeline/probability_predictor.py(_ProbabilityPredictor, inference)
"""

# noqa: F401
from .tournament_pipeline import (
    TournamentPipeline,
    TournamentPipeline as SOTAPipeline,
    ForecastConfig,
    ForecastConfig as SOTAPipelineConfig,
    BT_BLEND_WEIGHT,
    # Config-level constants re-exported by tournament_pipeline
    DataRequirementError,
    EVModeReport,
    _TrainedBaselineModel,
    TOURNAMENT_START_DATES,
    FIXED_FEATURE_SET,
    SIMPLE_FEATURE_SET,
    KAGGLE_ROUND_WEIGHTS,
    DATA_QUALITY_ERA_WEIGHTS,
    MIN_SEASON_FEATURE_COMPLETENESS,
    compute_year_data_quality,
    _infer_tournament_round_weight,
    TEAM_FEATURE_DIM,
    ABSOLUTE_LEVEL_FEATURE_NAMES,
    run_pipeline_to_file,
    run_pipeline_to_file as run_sota_pipeline_to_file,
)
