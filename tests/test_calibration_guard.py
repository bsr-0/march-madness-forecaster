import logging

import numpy as np
import pytest

from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig, DataRequirementError


def test_calibration_hard_min_raises(monkeypatch):
    config = SOTAPipelineConfig(year=2026, enable_multi_year_calibration=False)
    pipeline = SOTAPipeline(config)

    monkeypatch.setattr(pipeline, "_get_validation_era_games", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline, "_unique_games", lambda *_args, **_kwargs: [])

    with pytest.raises(DataRequirementError):
        pipeline._fit_calibration({})


# ---------------------------------------------------------------------------
# C3: Calibration train/test separation enforcement
# ---------------------------------------------------------------------------


class TestCalibrationTrainTestSeparation:
    """Verify that CalibrationPipeline guards against same-data fit+evaluate."""

    def test_same_data_raises_in_strict_mode(self):
        """Fitting and evaluating on same data must raise CalibrationLeakageError in strict mode."""
        from src.ml.calibration.calibration import CalibrationPipeline, CalibrationLeakageError

        pipeline = CalibrationPipeline(method="temperature", strict_mode=True)
        preds = np.array([0.3, 0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.4])
        outcomes = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=float)

        pipeline.fit(preds, outcomes)

        with pytest.raises(CalibrationLeakageError, match="same data used for fit"):
            pipeline.evaluate(preds, outcomes)

    def test_same_data_warns_in_non_strict_mode(self, caplog):
        """Fitting and evaluating on same data must produce a warning when strict_mode=False."""
        from src.ml.calibration.calibration import CalibrationPipeline

        pipeline = CalibrationPipeline(method="temperature", strict_mode=False)
        preds = np.array([0.3, 0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.4])
        outcomes = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=float)

        pipeline.fit(preds, outcomes)

        with caplog.at_level(logging.WARNING):
            pipeline.evaluate(preds, outcomes)

        assert any(
            "same data used for fit" in msg
            for msg in caplog.messages
        ), f"Expected in-sample warning, got: {caplog.messages}"

    def test_different_data_no_warning(self, caplog):
        """Evaluating on different data must NOT produce the warning."""
        from src.ml.calibration.calibration import CalibrationPipeline

        pipeline = CalibrationPipeline(method="temperature")
        train_preds = np.array([0.3, 0.5, 0.7, 0.9, 0.2, 0.6, 0.8, 0.4])
        train_outcomes = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=float)
        test_preds = np.array([0.4, 0.6, 0.8, 0.1])
        test_outcomes = np.array([0, 1, 1, 0], dtype=float)

        pipeline.fit(train_preds, train_outcomes)

        with caplog.at_level(logging.WARNING):
            pipeline.evaluate(test_preds, test_outcomes)

        assert not any(
            "same data used for fit" in msg
            for msg in caplog.messages
        ), f"Unexpected in-sample warning on different data: {caplog.messages}"
