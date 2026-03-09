"""Tests for src.ml.meta_learning — regime-based ensemble weight adjustment."""

import pytest

from src.ml.meta_learning import MetaLearner, MetaLearningDecision


class TestMetaLearner:
    def setup_method(self):
        self.learner = MetaLearner()

    def test_predict_regime_known_year(self):
        regime = self.learner.predict_regime(2018)
        assert regime in ("chalk", "upset_heavy", "moderate")

    def test_predict_regime_unknown_year(self):
        regime = self.learner.predict_regime(2030)
        assert regime == "moderate"  # default

    def test_adjust_weights_returns_decision(self):
        base = {"lgb": 0.25, "xgb": 0.25, "spread": 0.30, "logistic": 0.20}
        decision = self.learner.adjust_weights(base, 2023)
        assert isinstance(decision, MetaLearningDecision)
        assert decision.regime in ("chalk", "upset_heavy", "moderate")
        assert isinstance(decision.weight_adjustments, dict)

    def test_apply_adjustments_preserves_sum(self):
        base = {"lgb": 0.25, "xgb": 0.25, "spread": 0.30, "logistic": 0.20}
        decision = self.learner.adjust_weights(base, 2023)
        adjusted = self.learner.apply_adjustments(base, decision)
        total = sum(adjusted.values())
        assert abs(total - 1.0) < 0.01

    def test_apply_adjustments_nonnegative(self):
        base = {"lgb": 0.1, "xgb": 0.1, "spread": 0.5, "logistic": 0.3}
        decision = self.learner.adjust_weights(base, 2019)
        adjusted = self.learner.apply_adjustments(base, decision)
        for v in adjusted.values():
            assert v >= 0.0

    def test_no_registry_still_works(self):
        learner = MetaLearner(registry=None)
        base = {"lgb": 0.25, "xgb": 0.25, "spread": 0.25, "logistic": 0.25}
        decision = learner.adjust_weights(base, 2024)
        assert decision is not None
