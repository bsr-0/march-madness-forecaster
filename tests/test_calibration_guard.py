import pytest

from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig, DataRequirementError


def test_calibration_hard_min_raises(monkeypatch):
    config = SOTAPipelineConfig(year=2026, enable_multi_year_calibration=False)
    pipeline = SOTAPipeline(config)

    monkeypatch.setattr(pipeline, "_get_validation_era_games", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline, "_unique_games", lambda *_args, **_kwargs: [])

    with pytest.raises(DataRequirementError):
        pipeline._fit_calibration({})
