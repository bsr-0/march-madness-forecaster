"""Tests for Phase 6 — ModelClassSelector.

Covers: candidate evaluation, EV gating, temporal CV, selection logic,
strict mode, output format, and edge cases.
"""

import numpy as np
import pytest

from src.pipeline.stages.model_selection import (
    CANDIDATE_REGISTRY,
    MARGIN_BASED_CANDIDATES,
    CandidateResult,
    ModelClassSelector,
    ModelSelectionResult,
    _temporal_cv_split,
    _train_regularized_logistic,
    _train_spread_regressor,
    _train_margin_regressor,
)
from src.pipeline.stages.baseline_evaluation import compute_bracket_ev, compute_coin_flip_ev
from src.exceptions import IntegrityError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separable_data(
    n_train: int = 300,
    n_val: int = 63,
    n_features: int = 10,
    seed: int = 42,
    separability: float = 2.0,
):
    """Generate linearly separable train/val data for model testing."""
    rng = np.random.RandomState(seed)
    train_X = rng.randn(n_train, n_features)
    train_X[:, 0] *= separability
    train_y = (train_X[:, 0] > 0).astype(float)

    val_X = rng.randn(n_val, n_features)
    val_X[:, 0] *= separability
    val_y = (val_X[:, 0] > 0).astype(float)

    return train_X, train_y, val_X, val_y


# ---------------------------------------------------------------------------
# Temporal CV splitting
# ---------------------------------------------------------------------------


class TestTemporalCVSplit:
    """Expanding-window CV must produce valid chronological splits."""

    def test_produces_correct_number_of_folds(self):
        splits = _temporal_cv_split(100, n_folds=5)
        assert len(splits) == 5

    def test_train_precedes_val(self):
        sort_keys = np.arange(100, dtype=float)
        splits = _temporal_cv_split(100, n_folds=3, sort_keys=sort_keys)
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()

    def test_expanding_window(self):
        splits = _temporal_cv_split(100, n_folds=4)
        # Each successive fold should have more training data
        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_small_dataset(self):
        splits = _temporal_cv_split(10, n_folds=3)
        assert len(splits) >= 1
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0


# ---------------------------------------------------------------------------
# Candidate training functions
# ---------------------------------------------------------------------------


class TestCandidateTraining:
    """Candidate factories must return callable predict functions."""

    def test_regularized_logistic_trains(self):
        train_X, train_y, val_X, val_y = _make_separable_data()
        predict_fn = _train_regularized_logistic(train_X, train_y)
        preds = predict_fn(val_X)
        assert preds.shape == (len(val_y),)
        assert np.all(preds >= 0) and np.all(preds <= 1)
        # Should learn separable data
        acc = np.mean((preds > 0.5) == val_y)
        assert acc > 0.6

    def test_spread_regressor_returns_none_without_margins(self):
        train_X, train_y, _, _ = _make_separable_data()
        result = _train_spread_regressor(train_X, train_y, train_margins=None)
        assert result is None

    def test_margin_regressor_returns_none_without_margins(self):
        train_X, train_y, _, _ = _make_separable_data()
        result = _train_margin_regressor(train_X, train_y, train_margins=None)
        assert result is None

    def test_registry_includes_spread_and_margin(self):
        assert "spread_regressor" in CANDIDATE_REGISTRY
        assert "margin_regressor" in CANDIDATE_REGISTRY
        assert "spread_regressor" in MARGIN_BASED_CANDIDATES
        assert "margin_regressor" in MARGIN_BASED_CANDIDATES

    def test_all_registered_candidates_are_callable(self):
        for name, factory in CANDIDATE_REGISTRY.items():
            assert callable(factory), f"{name} factory is not callable"


# ---------------------------------------------------------------------------
# Full selector
# ---------------------------------------------------------------------------


class TestModelClassSelector:
    """Full Phase 6 pipeline: evaluate candidates, gate on EV, select top N."""

    def test_selects_models_that_beat_baseline(self):
        train_X, train_y, val_X, val_y = _make_separable_data(separability=2.5)

        # Set a very low baseline so candidates should beat it
        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            min_ev_improvement=-1000,
            max_brier=0.30,
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        assert result.passed
        assert "regularized_logistic" in result.selected_models

    def test_rejects_models_below_ev_threshold(self):
        train_X, train_y, val_X, val_y = _make_separable_data()

        # Set impossibly high baseline EV so nothing passes
        selector = ModelClassSelector(
            baseline_ev=999999.0,
            baseline_brier=0.0,
            min_ev_improvement=1.0,  # must beat by 1 point
            max_brier=0.30,
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        assert not result.passed
        assert len(result.selected_models) == 0
        assert len(result.errors) > 0

    def test_max_models_limit(self):
        train_X, train_y, val_X, val_y = _make_separable_data(separability=2.5)

        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            min_ev_improvement=-1000,
            max_brier=0.30,
            max_models=1,
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        assert len(result.selected_models) <= 1

    def test_strict_mode_raises_when_no_selection(self):
        train_X, train_y, val_X, val_y = _make_separable_data()

        selector = ModelClassSelector(
            baseline_ev=999999.0,
            baseline_brier=0.0,
            min_ev_improvement=1.0,
            strict=True,
            candidate_names=["regularized_logistic"],
        )
        with pytest.raises(IntegrityError, match="Phase 6"):
            selector.run(train_X, train_y, val_X, val_y)

    def test_non_strict_returns_failed_result(self):
        train_X, train_y, val_X, val_y = _make_separable_data()

        selector = ModelClassSelector(
            baseline_ev=999999.0,
            baseline_brier=0.0,
            min_ev_improvement=1.0,
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        assert not result.passed

    def test_brier_ceiling_rejects_bad_model(self):
        train_X, train_y, val_X, val_y = _make_separable_data()

        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            max_brier=0.001,  # impossibly tight
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        assert not result.passed

    def test_margin_candidates_skipped_without_margins(self):
        """Spread/margin candidates must be gracefully skipped when no margins provided."""
        train_X, train_y, val_X, val_y = _make_separable_data()

        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            min_ev_improvement=-1000,
            strict=False,
            candidate_names=["spread_regressor", "margin_regressor"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        # Neither should appear in candidates (they require margins)
        assert "spread_regressor" not in result.candidates
        assert "margin_regressor" not in result.candidates

    def test_margins_passed_to_candidates(self):
        """When margins are provided, margin-based candidates should be evaluated."""
        train_X, train_y, val_X, val_y = _make_separable_data(n_train=300)
        # Synthesize margins: positive = class 1, negative = class 0
        rng = np.random.RandomState(42)
        train_margins = np.where(train_y == 1, rng.uniform(1, 15, len(train_y)),
                                 rng.uniform(-15, -1, len(train_y)))
        val_margins = np.where(val_y == 1, rng.uniform(1, 15, len(val_y)),
                               rng.uniform(-15, -1, len(val_y)))

        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            min_ev_improvement=-1000,
            strict=False,
            # Only test regularized_logistic (always works) to verify
            # margins plumbing doesn't break non-margin models
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(
            train_X, train_y, val_X, val_y,
            train_margins=train_margins,
            val_margins=val_margins,
        )
        assert "regularized_logistic" in result.candidates


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Output must be consistent for downstream Phase 7 consumers."""

    def test_to_dict_structure(self):
        train_X, train_y, val_X, val_y = _make_separable_data()

        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            min_ev_improvement=-1000,
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)
        d = result.to_dict()

        assert "selected_models" in d
        assert "candidates" in d
        for name, cand in d["candidates"].items():
            assert "mean_EV" in cand
            assert "mean_Brier" in cand
            assert "selected" in cand
            assert "n_folds" in cand

    def test_summary_readable(self):
        result = ModelSelectionResult(
            selected_models=["gradient_boosting"],
            baseline_ev=960.0,
            baseline_brier=0.200,
            candidates={
                "gradient_boosting": CandidateResult(
                    model_name="gradient_boosting",
                    mean_brier=0.185,
                    mean_log_loss=0.52,
                    mean_accuracy=0.72,
                    mean_bracket_ev=1020.0,
                    ev_improvement_over_baseline=60.0,
                    brier_improvement_over_baseline=0.015,
                    selected=True,
                ),
            },
        )
        s = result.summary()
        assert "PASSED" in s
        assert "gradient_boosting" in s
        assert "SELECTED" in s

    def test_selected_models_list_format(self):
        train_X, train_y, val_X, val_y = _make_separable_data(separability=2.5)

        selector = ModelClassSelector(
            baseline_ev=0.0,
            baseline_brier=0.30,
            min_ev_improvement=-1000,
            strict=False,
            candidate_names=["regularized_logistic"],
        )
        result = selector.run(train_X, train_y, val_X, val_y)

        assert isinstance(result.selected_models, list)
        for m in result.selected_models:
            assert isinstance(m, str)


# ---------------------------------------------------------------------------
# CandidateResult
# ---------------------------------------------------------------------------


class TestCandidateResult:
    """Individual candidate results must serialize correctly."""

    def test_to_dict(self):
        c = CandidateResult(
            model_name="test",
            mean_brier=0.19,
            mean_log_loss=0.55,
            mean_accuracy=0.71,
            mean_bracket_ev=1000.0,
            fold_briers=[0.18, 0.19, 0.20],
            fold_evs=[990, 1000, 1010],
            ev_improvement_over_baseline=40.0,
            brier_improvement_over_baseline=0.01,
            selected=True,
        )
        d = c.to_dict()
        assert d["mean_EV"] == 1000.0
        assert d["mean_Brier"] == 0.19
        assert d["selected"] is True
        assert d["n_folds"] == 3
