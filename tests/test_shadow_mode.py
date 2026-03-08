"""Tests for shadow mode deployment."""

from src.deployment.shadow_mode import ShadowRunner, ShadowResult


class TestShadowPromote:
    """Shadow is meaningfully better with low divergence -> promote_shadow."""

    def test_shadow_promote(self):
        runner = ShadowRunner()
        # Shadow predictions are closer to actual outcomes
        outcomes = {"g1": 1.0, "g2": 0.0, "g3": 1.0, "g4": 0.0, "g5": 1.0}
        primary = {"g1": 0.7, "g2": 0.3, "g3": 0.6, "g4": 0.4, "g5": 0.7}
        shadow = {"g1": 0.8, "g2": 0.2, "g3": 0.72, "g4": 0.28, "g5": 0.82}

        result = runner.run(primary, shadow, outcomes)

        assert result.delta < -runner.PROMOTE_THRESHOLD
        assert result.max_divergence < runner.MAX_DIVERGENCE_THRESHOLD
        assert result.recommendation == "promote_shadow"
        assert result.shadow_brier < result.primary_brier


class TestShadowKeep:
    """Shadow not meaningfully better -> keep_primary."""

    def test_shadow_keep(self):
        runner = ShadowRunner()
        outcomes = {"g1": 1.0, "g2": 0.0, "g3": 1.0}
        # Nearly identical predictions
        primary = {"g1": 0.8, "g2": 0.2, "g3": 0.8}
        shadow = {"g1": 0.8, "g2": 0.2, "g3": 0.805}

        result = runner.run(primary, shadow, outcomes)

        assert abs(result.delta) < 0.001
        assert result.recommendation == "keep_primary"


class TestShadowNeedsReview:
    """Shadow is better but divergence is too high -> needs_review."""

    def test_shadow_needs_review(self):
        runner = ShadowRunner()
        outcomes = {"g1": 1.0, "g2": 0.0, "g3": 1.0, "g4": 0.0}
        primary = {"g1": 0.6, "g2": 0.4, "g3": 0.6, "g4": 0.4}
        # Shadow is better overall but has high divergence on some games
        shadow = {"g1": 0.95, "g2": 0.05, "g3": 0.95, "g4": 0.2}

        result = runner.run(primary, shadow, outcomes)

        assert result.delta < -runner.PROMOTE_THRESHOLD
        assert result.max_divergence >= runner.MAX_DIVERGENCE_THRESHOLD
        assert result.recommendation == "needs_review"


class TestShadowResultFields:
    """Verify all ShadowResult fields are populated correctly."""

    def test_shadow_result_fields(self):
        runner = ShadowRunner()
        outcomes = {"g1": 1.0, "g2": 0.0, "g3": 1.0}
        primary = {"g1": 0.7, "g2": 0.3, "g3": 0.7}
        shadow = {"g1": 0.8, "g2": 0.2, "g3": 0.8}

        result = runner.run(primary, shadow, outcomes)

        assert isinstance(result, ShadowResult)
        assert isinstance(result.primary_brier, float)
        assert isinstance(result.shadow_brier, float)
        assert isinstance(result.delta, float)
        assert isinstance(result.per_game_diffs, list)
        assert len(result.per_game_diffs) == 3
        assert isinstance(result.max_divergence, float)
        assert isinstance(result.correlation, float)
        assert result.recommendation in ("promote_shadow", "keep_primary", "needs_review")
        # delta should equal shadow_brier - primary_brier
        assert abs(result.delta - (result.shadow_brier - result.primary_brier)) < 1e-10
