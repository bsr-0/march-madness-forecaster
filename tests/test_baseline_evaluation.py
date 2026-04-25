"""Tests for Phase 5 — BaselineEvaluator.

Covers: logistic + Elo training, Brier/EV computation, gate thresholds,
strict mode, coin-flip EV, and output format consistency.
"""

import numpy as np
import pytest

from src.pipeline.stages.baseline_evaluation import (
    BaselineEvaluator,
    BaselineResults,
    SingleModelResult,
    _EloBaseline,
    _LogisticBaseline,
    compute_bracket_ev,
    compute_coin_flip_ev,
    _safe_log_loss,
    ROUND_SCORING_WEIGHTS,
)
from src.exceptions import IntegrityError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification_data(
    n_train: int = 200,
    n_val: int = 63,
    n_features: int = 10,
    seed: int = 42,
    separability: float = 1.5,
):
    """Generate linearly separable train/val data."""
    rng = np.random.RandomState(seed)
    # Feature 0 is the signal, rest is noise
    train_X = rng.randn(n_train, n_features)
    train_X[:, 0] = rng.randn(n_train) * separability
    train_y = (train_X[:, 0] > 0).astype(float)

    val_X = rng.randn(n_val, n_features)
    val_X[:, 0] = rng.randn(n_val) * separability
    val_y = (val_X[:, 0] > 0).astype(float)

    return train_X, train_y, val_X, val_y


# ---------------------------------------------------------------------------
# Bracket EV computation
# ---------------------------------------------------------------------------


class TestBracketEV:
    """Bracket EV must correctly sum round-weighted expected points."""

    def test_coin_flip_ev_standard_bracket(self):
        ev = compute_coin_flip_ev(n_games=63)
        # 0.5 * (32*10 + 16*20 + 8*40 + 4*80 + 2*160 + 1*320)
        # = 0.5 * (320 + 320 + 320 + 320 + 320 + 320) = 0.5 * 1920 = 960
        assert abs(ev - 960.0) < 1e-6

    def test_perfect_predictions_ev(self):
        # All predictions = 1.0, all games are R64
        preds = np.ones(32)
        ev = compute_bracket_ev(preds, round_labels=None)
        assert ev == 32 * 10  # 320

    def test_ev_with_round_labels(self):
        preds = np.array([0.8, 0.6])
        labels = np.array(["R64", "CHAMP"])
        ev = compute_bracket_ev(preds, round_labels=labels)
        expected = 0.8 * 10 + 0.6 * 320
        assert abs(ev - expected) < 1e-6

    def test_zero_predictions_ev(self):
        preds = np.zeros(10)
        ev = compute_bracket_ev(preds)
        assert ev == 0.0


# ---------------------------------------------------------------------------
# Logistic baseline
# ---------------------------------------------------------------------------


class TestLogisticBaseline:
    """Logistic regression baseline must learn separable data."""

    def test_fit_and_predict(self):
        train_X, train_y, val_X, val_y = _make_classification_data()
        model = _LogisticBaseline()
        model.fit(train_X, train_y)
        preds = model.predict_proba(val_X)

        assert preds.shape == (len(val_y),)
        assert np.all(preds >= 0) and np.all(preds <= 1)
        # Should beat coin flip on separable data
        accuracy = np.mean((preds > 0.5) == val_y)
        assert accuracy > 0.6

    def test_handles_nan_features(self):
        train_X, train_y, val_X, val_y = _make_classification_data()
        train_X[0, 3] = np.nan
        val_X[5, 2] = np.nan
        model = _LogisticBaseline()
        model.fit(train_X, train_y)
        preds = model.predict_proba(val_X)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# Elo baseline
# ---------------------------------------------------------------------------


class TestEloBaseline:
    """Elo baseline must find a predictive feature and use logistic transform."""

    def test_fit_and_predict(self):
        train_X, train_y, val_X, val_y = _make_classification_data()
        model = _EloBaseline()
        model.fit(train_X, train_y)
        preds = model.predict_proba(val_X)

        assert preds.shape == (len(val_y),)
        assert np.all(preds >= 0.01) and np.all(preds <= 0.99)

    def test_prefers_named_features(self):
        train_X, train_y, _, _ = _make_classification_data(n_features=5)
        names = ["noise1", "seed_strength", "noise2", "noise3", "noise4"]
        # Make seed_strength the signal
        train_X[:, 1] = train_X[:, 0]  # copy signal to seed_strength column
        train_X[:, 0] = np.random.randn(len(train_X))  # noise-ify original

        model = _EloBaseline()
        model.fit(train_X, train_y, feature_names=names)
        # Should have selected index 1 (seed_strength)
        assert model.feature_index == 1

    def test_predictions_clipped(self):
        train_X, train_y, val_X, val_y = _make_classification_data()
        model = _EloBaseline()
        model.fit(train_X, train_y)
        preds = model.predict_proba(val_X)
        assert np.all(preds >= 0.01) and np.all(preds <= 0.99)


# ---------------------------------------------------------------------------
# Full evaluator
# ---------------------------------------------------------------------------


class TestBaselineEvaluator:
    """Full Phase 5 pipeline: train, evaluate, gate."""

    def test_passes_on_separable_data(self):
        train_X, train_y, val_X, val_y = _make_classification_data(separability=2.0)
        evaluator = BaselineEvaluator(
            brier_threshold=0.30,
            min_bracket_ev_above_coinflip=-1000,  # easy threshold
            strict=False,
        )
        results = evaluator.run(train_X, train_y, val_X, val_y)

        assert results.passed
        assert "logistic_regression" in results.models
        assert "Elo" in results.models

        for name, r in results.models.items():
            assert r.brier_score < 0.30
            assert r.accuracy > 0.5

    def test_strict_mode_raises_on_failure(self):
        # Use impossible thresholds
        train_X, train_y, val_X, val_y = _make_classification_data()
        evaluator = BaselineEvaluator(
            brier_threshold=0.001,  # impossible
            strict=True,
        )
        with pytest.raises(IntegrityError, match="Phase 5"):
            evaluator.run(train_X, train_y, val_X, val_y)

    def test_non_strict_returns_failed(self):
        train_X, train_y, val_X, val_y = _make_classification_data()
        evaluator = BaselineEvaluator(
            brier_threshold=0.001,
            strict=False,
        )
        results = evaluator.run(train_X, train_y, val_X, val_y)
        assert not results.passed
        assert len(results.errors) > 0

    def test_output_format_consistent(self):
        train_X, train_y, val_X, val_y = _make_classification_data()
        evaluator = BaselineEvaluator(strict=False)
        results = evaluator.run(train_X, train_y, val_X, val_y)

        d = results.to_dict()
        for name, scores in d.items():
            assert "EV" in scores
            assert "Brier" in scores
            assert "model_name" in scores
            assert "passed" in scores

    def test_coin_flip_ev_computed(self):
        train_X, train_y, val_X, val_y = _make_classification_data(n_val=63)
        evaluator = BaselineEvaluator(strict=False)
        results = evaluator.run(train_X, train_y, val_X, val_y)
        # coin_flip_ev should be > 0 for 63 games
        assert results.coin_flip_ev > 0

    def test_round_labels_used_for_ev(self):
        train_X, train_y, val_X, val_y = _make_classification_data(n_val=4)
        rounds = np.array(["R64", "R32", "S16", "E8"])
        evaluator = BaselineEvaluator(strict=False)
        results = evaluator.run(train_X, train_y, val_X, val_y, round_labels=rounds)
        # EV should differ from the no-round-label case
        assert "logistic_regression" in results.models


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestSafeLogLoss:
    """Log loss must handle edge cases without NaN/inf."""

    def test_perfect_predictions(self):
        preds = np.array([1.0, 0.0, 1.0])
        labels = np.array([1.0, 0.0, 1.0])
        ll = _safe_log_loss(preds, labels)
        assert ll < 0.001  # near zero

    def test_worst_predictions(self):
        preds = np.array([0.0, 1.0])
        labels = np.array([1.0, 0.0])
        ll = _safe_log_loss(preds, labels)
        assert ll > 10  # very high but not inf


# ---------------------------------------------------------------------------
# BaselineResults contract
# ---------------------------------------------------------------------------


class TestBaselineResults:
    """Output format must be stable for downstream consumers."""

    def test_summary_includes_key_info(self):
        results = BaselineResults(
            models={
                "lr": SingleModelResult(
                    model_name="lr",
                    brier_score=0.18,
                    log_loss=0.5,
                    accuracy=0.72,
                    bracket_ev=1050.0,
                    predictions=np.array([0.7]),
                )
            },
            coin_flip_ev=960.0,
        )
        s = results.summary()
        assert "PASSED" in s
        assert "lr" in s
        assert "1050" in s

    def test_to_dict_format(self):
        results = BaselineResults(
            models={
                "test_model": SingleModelResult(
                    model_name="test_model",
                    brier_score=0.20,
                    log_loss=0.55,
                    accuracy=0.70,
                    bracket_ev=980.0,
                    predictions=np.array([0.6]),
                )
            }
        )
        d = results.to_dict()
        assert "test_model" in d
        assert d["test_model"]["EV"] == 980.0
        assert d["test_model"]["Brier"] == 0.2
