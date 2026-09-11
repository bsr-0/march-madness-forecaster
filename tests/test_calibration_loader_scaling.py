"""Historical calibration rows must be scaled exactly once, by the model.

`predict_proba_batch` applies the model's fixed feature indices and then its
scaler (`_scale_batch`). The calibration stage's historical-tournament loader
also called `scaler.transform` itself, first. Two consequences, one per
feature-selection mode:

* fixed set (the harness): the scaler saw the full pruned width and raised
  "X has 60 features, but StandardScaler is expecting 9", every historical
  year was skipped, and the temperature fit fell to ~47 current-year rows --
  below the hard minimum, so every LOYO fold died in calibration.
* learned selector (the production configs): rows were standardized twice,
  so the temperature was fit on distorted probabilities.
"""

import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.pipeline.config import _TrainedBaselineModel as Model

ROOT = Path(__file__).resolve().parents[1]


def _fixed_set_model():
    rng = np.random.default_rng(0)
    idx = [2, 7, 11]
    X_wide = rng.normal(size=(200, 60))
    X_fit = X_wide[:, idx]
    y = (X_fit[:, 0] + 0.5 * X_fit[:, 1] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    scaler = StandardScaler().fit(X_fit)
    logit = LogisticRegression().fit(scaler.transform(X_fit), y)
    m = Model()
    m.fixed_feature_indices = idx
    m.scaler = scaler
    m.logit_model = logit
    return m, X_wide, idx


def test_model_selects_fixed_columns_then_scales_itself():
    m, X_wide, idx = _fixed_set_model()
    p_wide = m.predict_proba_batch(X_wide)
    p_narrow = m.predict_proba_batch(X_wide[:, idx])
    assert p_wide.shape == (200,)
    assert np.allclose(p_wide, p_narrow)


def test_pre_scaling_the_wide_matrix_is_the_bug():
    m, X_wide, _ = _fixed_set_model()
    try:
        m.scaler.transform(X_wide)
    except ValueError as exc:
        assert "expecting 3 features" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("scaler accepted the wide matrix; the regression this guards has changed shape")


def test_double_scaling_changes_the_prediction():
    m, X_wide, idx = _fixed_set_model()
    once = m.predict_proba_batch(X_wide[:, idx])
    twice = m.predict_proba_batch(m.scaler.transform(X_wide[:, idx]))
    assert not np.allclose(once, twice), "double scaling must be observable, or this test guards nothing"


def test_calibration_loader_does_not_scale_before_predicting():
    src = (ROOT / "src" / "pipeline" / "stages" / "calibration.py").read_text()
    start = src.index("def _load_tournament_cal_year")
    end = src.index("return len(yr_y), None", start)
    body = src[start:end]
    assert "scaler.transform(" not in body, "the historical calibration loader scales before predict_proba_batch again"
    assert "predict_proba_batch(" in body
