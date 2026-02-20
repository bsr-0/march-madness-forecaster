"""Tests for ablation study framework."""

import numpy as np
import pytest

from src.ml.evaluation.ablation import (
    ABLATABLE_COMPONENTS,
    AblationResult,
    AblationStudy,
    FullAblationReport,
)


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_helps_when_significant_and_positive_delta(self):
        result = AblationResult(
            component="gnn",
            baseline_brier=0.20,
            ablated_brier=0.25,
            brier_delta=0.05,
            t_stat=2.5,
            p_value=0.01,
            cohens_d=0.3,
            n_games=100,
            significant_at_05=True,
        )
        assert result.helps is True

    def test_not_helps_when_not_significant(self):
        result = AblationResult(
            component="travel_distance",
            baseline_brier=0.20,
            ablated_brier=0.201,
            brier_delta=0.001,
            t_stat=0.5,
            p_value=0.60,
            cohens_d=0.02,
            n_games=100,
            significant_at_05=False,
        )
        assert result.helps is False

    def test_not_helps_when_component_hurts(self):
        """Even if significant, removing the component improves things."""
        result = AblationResult(
            component="gnn",
            baseline_brier=0.25,
            ablated_brier=0.20,
            brier_delta=-0.05,
            t_stat=-2.5,
            p_value=0.01,
            cohens_d=-0.3,
            n_games=100,
            significant_at_05=False,  # delta is negative, so not "helps"
        )
        assert result.helps is False


class TestFullAblationReport:
    """Tests for FullAblationReport."""

    def test_helpful_and_unhelpful_classification(self):
        report = FullAblationReport(baseline_brier=0.20, n_games=100)
        report.results["gnn"] = AblationResult(
            "gnn", 0.20, 0.25, 0.05, 2.5, 0.01, 0.3, 100, True
        )
        report.results["travel_distance"] = AblationResult(
            "travel_distance", 0.20, 0.201, 0.001, 0.5, 0.6, 0.02, 100, False
        )

        assert "gnn" in report.helpful_components
        assert "travel_distance" in report.unhelpful_components

    def test_to_dict_structure(self):
        report = FullAblationReport(baseline_brier=0.20, n_games=100)
        report.results["gnn"] = AblationResult(
            "gnn", 0.20, 0.25, 0.05, 2.5, 0.01, 0.3, 100, True
        )

        d = report.to_dict()
        assert "baseline_brier" in d
        assert "helpful_components" in d
        assert "per_component" in d
        assert "gnn" in d["per_component"]
        assert "p_value" in d["per_component"]["gnn"]


class TestAblationStudy:
    """Tests for AblationStudy using a mock pipeline."""

    def _make_mock_pipeline(self, base_quality: float = 0.8):
        """Create a minimal mock pipeline object."""

        class MockConfig:
            enable_tournament_adaptation = True
            enable_injury_severity_model = True

        class MockPipeline:
            def __init__(self, quality):
                self.config = MockConfig()
                self.model_confidence = {
                    "baseline": 0.7,
                    "gnn": 0.6,
                    "transformer": 0.5,
                }
                self._quality = quality
                self._rng = np.random.default_rng(42)

            def predict_probability(self, team1: str, team2: str) -> float:
                """Deterministic mock: better team always gets high prob."""
                # Use team names to create deterministic predictions
                seed = hash((team1, team2)) % 10000
                rng = np.random.default_rng(seed)
                base = rng.uniform(0.3, 0.7)
                # Quality modulates how close to truth
                noise = rng.normal(0, 1.0 - self._quality)
                return float(np.clip(base + noise * 0.1, 0.01, 0.99))

        return MockPipeline(base_quality)

    def _make_games(self, n: int = 50, rng_seed: int = 42):
        """Generate synthetic validation games."""
        rng = np.random.default_rng(rng_seed)
        games = []
        for i in range(n):
            games.append({
                "team1": f"team_a_{i}",
                "team2": f"team_b_{i}",
                "team1_won": bool(rng.random() > 0.5),
            })
        return games

    def test_single_ablation_returns_valid_result(self):
        pipeline = self._make_mock_pipeline()
        games = self._make_games(50)
        study = AblationStudy(pipeline, games)

        result = study.run_single_ablation("gnn")

        assert isinstance(result, AblationResult)
        assert result.component == "gnn"
        assert 0 <= result.p_value <= 1.0
        assert result.n_games == 50

    def test_unknown_component_raises(self):
        pipeline = self._make_mock_pipeline()
        games = self._make_games(10)
        study = AblationStudy(pipeline, games)

        with pytest.raises(ValueError, match="Unknown component"):
            study.run_single_ablation("nonexistent")

    def test_full_ablation_runs_all_components(self):
        pipeline = self._make_mock_pipeline()
        games = self._make_games(30)
        study = AblationStudy(pipeline, games)

        report = study.run_full_ablation(
            components=["gnn", "transformer", "tournament_adaptation"]
        )

        assert isinstance(report, FullAblationReport)
        assert len(report.results) == 3
        assert report.baseline_brier >= 0

    def test_state_restored_after_ablation(self):
        """Pipeline state should be fully restored after ablation."""
        pipeline = self._make_mock_pipeline()
        games = self._make_games(20)
        study = AblationStudy(pipeline, games)

        original_gnn_conf = pipeline.model_confidence["gnn"]
        original_adaptation = pipeline.config.enable_tournament_adaptation

        study.run_single_ablation("gnn")
        assert pipeline.model_confidence["gnn"] == original_gnn_conf

        study.run_single_ablation("tournament_adaptation")
        assert pipeline.config.enable_tournament_adaptation == original_adaptation

    def test_ablatable_components_constant(self):
        """Verify all expected components are in the constant."""
        expected = {"gnn", "transformer", "travel_distance",
                    "injury_model", "tournament_adaptation", "stacking"}
        assert set(ABLATABLE_COMPONENTS) == expected
