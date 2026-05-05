"""Tests for pipeline stage protocol, data contracts, and inter-stage validation."""

import numpy as np
import pytest

from src.pipeline.stages import (
    CalibratedPipeline,
    EngineeredFeatures,
    LoadedData,
    PipelineReport,
    PipelineStage,
    SimulationResults,
    TrainedModels,
)
from src.pipeline.stages.context import PipelineContext
from src.pipeline.stages.game_utils import (
    TOURNAMENT_START_DATES,
    compute_game_outcome,
    is_tournament_game,
    normalize_team_key,
    parse_game_date,
)
from src.data.schemas import (
    validate_engineered_features,
    validate_loaded_data,
    validate_trained_models,
)


# ---------------------------------------------------------------------------
# PipelineStage protocol compliance
# ---------------------------------------------------------------------------


class _DummyStage:
    name = "dummy"

    def run(self):
        return {"ok": True}


def test_pipeline_stage_protocol_compliance():
    """A class with name and run() should satisfy PipelineStage."""
    stage = _DummyStage()
    assert isinstance(stage, PipelineStage)


def test_pipeline_stage_protocol_rejects_non_compliant():
    """A class without name should NOT satisfy PipelineStage."""

    class _BadStage:
        def run(self):
            pass

    assert not isinstance(_BadStage(), PipelineStage)


# ---------------------------------------------------------------------------
# Data contract construction and properties
# ---------------------------------------------------------------------------


class TestLoadedData:
    def test_summary(self):
        ld = LoadedData(teams=["a", "b", "c"])
        s = ld.summary()
        assert "3 teams" in s

    def test_empty(self):
        ld = LoadedData(teams=[])
        assert ld.summary().startswith("LoadedData")


class TestEngineeredFeatures:
    def test_n_teams_and_dim(self):
        ef = EngineeredFeatures(
            team_features={"t1": np.zeros(23), "t2": np.zeros(23)},
        )
        assert ef.n_teams == 2
        assert ef.feature_dim == 23

    def test_empty_features(self):
        ef = EngineeredFeatures()
        assert ef.n_teams == 0
        assert ef.feature_dim == 0


class TestTrainedModels:
    def test_has_gnn(self):
        tm = TrainedModels(gnn_embeddings={"t1": np.zeros(16)})
        assert tm.has_gnn
        assert not tm.has_transformer

    def test_no_embeddings(self):
        tm = TrainedModels()
        assert not tm.has_gnn
        assert not tm.has_transformer


class TestPipelineReport:
    def test_to_dict(self):
        pr = PipelineReport(mode="ev", year=2026, brier_score=0.189)
        d = pr.to_dict()
        assert d["mode"] == "ev"
        assert d["brier_score"] == 0.189


# ---------------------------------------------------------------------------
# Inter-stage validation
# ---------------------------------------------------------------------------


class TestValidateLoadedData:
    def test_valid(self):
        ld = LoadedData(
            teams=list(range(100)),
            torvik_map={"t": {}},
            rosters={"t": {}},
        )
        result = validate_loaded_data(ld)
        assert result.passed

    def test_empty_teams_fails(self):
        ld = LoadedData(teams=[])
        result = validate_loaded_data(ld)
        assert not result.passed

    def test_few_teams_warns(self):
        ld = LoadedData(teams=list(range(10)))
        result = validate_loaded_data(ld)
        assert result.passed  # Not an error, just a warning
        assert len(result.warnings) > 0


class TestValidateEngineeredFeatures:
    def test_valid(self):
        ef = EngineeredFeatures(
            team_features={"t1": np.zeros(23), "t2": np.zeros(23)},
            team_struct={"t1": "A"},
        )
        result = validate_engineered_features(ef)
        assert result.passed

    def test_empty_fails(self):
        ef = EngineeredFeatures()
        result = validate_engineered_features(ef)
        assert not result.passed

    def test_inconsistent_dims_fails(self):
        ef = EngineeredFeatures(
            team_features={"t1": np.zeros(23), "t2": np.zeros(10)},
        )
        result = validate_engineered_features(ef)
        assert not result.passed


class TestValidateTrainedModels:
    def test_valid(self):
        tm = TrainedModels(baseline_model=object())
        result = validate_trained_models(tm)
        assert result.passed

    def test_no_model_fails(self):
        tm = TrainedModels()
        result = validate_trained_models(tm)
        assert not result.passed

    def test_mismatched_oof(self):
        tm = TrainedModels(
            baseline_model=object(),
            oof_predictions=np.zeros(10),
            oof_actuals=np.zeros(5),
        )
        result = validate_trained_models(tm)
        assert not result.passed


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------


class TestPipelineContext:
    def test_from_pipeline(self):
        """PipelineContext.from_pipeline should extract attrs from a pipeline-like obj."""

        class FakePipeline:
            config = {"year": 2026}
            _phase_timer = None
            _resource_tracker = None
            _experiment_registry = None

        ctx = PipelineContext.from_pipeline(FakePipeline())
        assert ctx.config == {"year": 2026}


# ---------------------------------------------------------------------------
# Game utilities
# ---------------------------------------------------------------------------


class TestGameUtils:
    def test_normalize_team_key(self):
        assert normalize_team_key("Duke ") == "duke"
        assert normalize_team_key("Ohio State") == "ohio"

    def test_is_tournament_game(self):
        from datetime import date

        assert is_tournament_game(date(2025, 3, 20), 2025)
        assert not is_tournament_game(date(2025, 3, 1), 2025)

    def test_parse_game_date(self):
        from datetime import date

        assert parse_game_date("2025-03-18") == date(2025, 3, 18)
        assert parse_game_date("03/18/2025") == date(2025, 3, 18)
        assert parse_game_date("invalid") is None

    def test_compute_game_outcome(self):
        win, margin = compute_game_outcome(75, 60)
        assert win is True
        assert margin == 15

        loss, margin = compute_game_outcome(60, 75)
        assert loss is False
        assert margin == -15

    def test_tournament_start_dates_coverage(self):
        assert 2025 in TOURNAMENT_START_DATES
        assert 2026 in TOURNAMENT_START_DATES
        assert 2027 in TOURNAMENT_START_DATES
