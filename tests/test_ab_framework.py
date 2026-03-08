"""Tests for A/B testing framework."""

from src.deployment.ab_framework import ABFramework, ABExperiment


class TestRunComparison:
    """Verify ABExperiment populated with per-game scores."""

    def test_run_comparison(self):
        framework = ABFramework()

        predictions_a = {"g1": 0.7, "g2": 0.3, "g3": 0.6}
        predictions_b = {"g1": 0.8, "g2": 0.2, "g3": 0.7}
        outcomes = {"g1": 1.0, "g2": 0.0, "g3": 1.0}

        experiment = framework.run_comparison(
            predictions_a, predictions_b, outcomes,
            strategy_a="ModelA", strategy_b="ModelB",
        )

        assert isinstance(experiment, ABExperiment)
        assert experiment.strategy_a == "ModelA"
        assert experiment.strategy_b == "ModelB"
        assert len(experiment.results_a) == 3
        assert len(experiment.results_b) == 3
        # Verify Brier scores are computed correctly for first game
        # g1: A -> (0.7 - 1.0)^2 = 0.09, B -> (0.8 - 1.0)^2 = 0.04
        assert abs(experiment.results_a[0] - 0.09) < 1e-10
        assert abs(experiment.results_b[0] - 0.04) < 1e-10


class TestStatisticalComparison:
    """Verify p_value and winner fields present."""

    def test_statistical_comparison(self):
        framework = ABFramework()

        # Strategy B is clearly better
        predictions_a = {f"g{i}": 0.5 for i in range(30)}
        predictions_b = {f"g{i}": 0.9 for i in range(30)}
        outcomes = {f"g{i}": 1.0 for i in range(30)}

        experiment = framework.run_comparison(
            predictions_a, predictions_b, outcomes,
            strategy_a="Baseline", strategy_b="Improved",
        )
        result = framework.statistical_comparison(experiment)

        assert "p_value" in result
        assert "winner" in result
        assert "mean_diff" in result
        assert "confidence_interval" in result
        assert "significant" in result
        assert isinstance(result["p_value"], float)
        assert isinstance(result["significant"], bool)
        # B should be the winner (lower Brier)
        assert result["winner"] == "Improved"
        assert result["significant"] is True


class TestIdenticalPredictions:
    """Same predictions -> winner 'neither', not significant."""

    def test_identical_predictions(self):
        framework = ABFramework()

        predictions = {f"g{i}": 0.6 for i in range(20)}
        outcomes = {f"g{i}": 1.0 for i in range(20)}

        experiment = framework.run_comparison(
            predictions, predictions, outcomes,
            strategy_a="Same1", strategy_b="Same2",
        )
        result = framework.statistical_comparison(experiment)

        assert result["winner"] == "neither"
        assert result["significant"] is False
        assert result["mean_diff"] == 0.0
