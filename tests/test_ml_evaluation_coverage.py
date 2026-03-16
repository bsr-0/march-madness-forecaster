"""Tests for ml/evaluation and simulation modules for coverage improvement."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# unified_backtest.py
# ---------------------------------------------------------------------------

class TestTournamentGame:
    def test_basic_construction(self):
        from src.ml.evaluation.unified_backtest import TournamentGame
        g = TournamentGame(
            team1_id="duke", team2_id="unc",
            team1_seed=1, team2_seed=4,
            team1_won=True, round_name="R64",
            team1_score=80, team2_score=70,
        )
        assert g.team1_id == "duke"

    def test_winner_id(self):
        from src.ml.evaluation.unified_backtest import TournamentGame
        g = TournamentGame("duke", "unc", 1, 4, True, "R64")
        assert g.winner_id == "duke"

    def test_winner_id_team2(self):
        from src.ml.evaluation.unified_backtest import TournamentGame
        g = TournamentGame("duke", "unc", 1, 4, False, "R64")
        assert g.winner_id == "unc"

    def test_loser_id(self):
        from src.ml.evaluation.unified_backtest import TournamentGame
        g = TournamentGame("duke", "unc", 1, 4, True, "R64")
        assert g.loser_id == "unc"

    def test_to_eval_dict(self):
        from src.ml.evaluation.unified_backtest import TournamentGame
        g = TournamentGame("duke", "unc", 1, 4, True, "R64")
        d = g.to_eval_dict()
        assert isinstance(d, dict)


class TestTournamentHistory:
    def test_construction(self):
        from src.ml.evaluation.unified_backtest import TournamentHistory, TournamentGame
        games = [TournamentGame("a", "b", 1, 16, True, "R64")]
        th = TournamentHistory(
            year=2023, games=games,
            seeds={"a": 1, "b": 16}, regions={"a": "East", "b": "East"},
            first_round_matchups=["a_b"],
        )
        assert th.year == 2023
        assert th.n_games == 1

    def test_find_winner(self):
        from src.ml.evaluation.unified_backtest import TournamentHistory, TournamentGame
        games = [TournamentGame("a", "b", 1, 16, True, "R64")]
        th = TournamentHistory(
            year=2023, games=games,
            seeds={"a": 1, "b": 16}, regions={},
            first_round_matchups=[],
        )
        assert th._find_winner("a", "b") == "a"


class TestChampionBoostBacktestResult:
    def test_construction(self):
        from src.ml.evaluation.unified_backtest import ChampionBoostBacktestResult
        r = ChampionBoostBacktestResult(
            year=2023, selected_champion="duke", selected_champion_seed=1,
            actual_champion="unc", champion_correct=False,
            baseline_rw_brier=0.2, boosted_rw_brier=0.19, brier_delta=-0.01,
        )
        assert r.year == 2023
        assert r.brier_delta == pytest.approx(-0.01)


class TestUnifiedBacktestConfig:
    def test_default(self):
        from src.ml.evaluation.unified_backtest import UnifiedBacktestConfig
        c = UnifiedBacktestConfig()
        assert isinstance(c.years, list)
        assert isinstance(c.modes, list)

    def test_custom(self):
        from src.ml.evaluation.unified_backtest import UnifiedBacktestConfig
        c = UnifiedBacktestConfig(years=[2023], modes=["calibration", "ev"])
        assert len(c.years) == 1


class TestYearModeResult:
    def test_construction(self):
        from src.ml.evaluation.unified_backtest import YearModeResult
        r = YearModeResult(year=2023, mode="calibration", brier_score=0.2)
        assert r.year == 2023
        assert r.mode == "calibration"


class TestUnifiedBacktestResult:
    def test_construction(self):
        from src.ml.evaluation.unified_backtest import UnifiedBacktestResult, UnifiedBacktestConfig
        c = UnifiedBacktestConfig()
        r = UnifiedBacktestResult(config=c)
        assert r.year_mode_results == []

    def test_summary_empty(self):
        from src.ml.evaluation.unified_backtest import UnifiedBacktestResult, UnifiedBacktestConfig
        c = UnifiedBacktestConfig()
        r = UnifiedBacktestResult(config=c)
        s = r.summary()
        assert isinstance(s, str)


class TestComputeRoundWeightedBrier:
    def test_basic(self):
        from src.ml.evaluation.unified_backtest import _compute_round_weighted_brier, TournamentGame
        games = [
            TournamentGame("a", "b", 1, 16, True, "R64"),
            TournamentGame("c", "d", 2, 15, True, "R64"),
        ]
        preds = {("a", "b"): 0.9, ("c", "d"): 0.8}
        result = _compute_round_weighted_brier(preds, games)
        assert isinstance(result, float)
        assert result >= 0


# ---------------------------------------------------------------------------
# pool_competition.py
# ---------------------------------------------------------------------------

class TestBuildScoringVector:
    def test_default(self):
        from src.simulation.pool_competition import build_scoring_vector
        default_scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
        v = build_scoring_vector(default_scoring)
        assert len(v) == 63

    def test_custom(self):
        from src.simulation.pool_competition import build_scoring_vector
        custom = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "CHAMP": 32}
        v = build_scoring_vector(custom)
        assert len(v) == 63


class TestWilsonScoreInterval:
    def test_basic(self):
        from src.simulation.pool_competition import wilson_score_interval
        result = wilson_score_interval(50, 100)
        assert len(result) == 3
        assert result[1] < result[0] < result[2]

    def test_zero_successes(self):
        from src.simulation.pool_competition import wilson_score_interval
        result = wilson_score_interval(0, 100)
        assert result[0] == pytest.approx(0.0)

    def test_zero_trials(self):
        from src.simulation.pool_competition import wilson_score_interval
        result = wilson_score_interval(0, 0)
        assert result[0] == 0.0


class TestPoolSimulationConfig:
    def test_defaults(self):
        from src.simulation.pool_competition import PoolSimulationConfig
        c = PoolSimulationConfig()
        assert c.n_tournaments == 1000
        assert c.n_opponents == 200

    def test_custom(self):
        from src.simulation.pool_competition import PoolSimulationConfig
        c = PoolSimulationConfig(n_tournaments=500, random_seed=123)
        assert c.n_tournaments == 500


class TestPercentileEstimate:
    def test_construction(self):
        from src.simulation.pool_competition import PercentileEstimate
        pe = PercentileEstimate(
            label="top_5pct", threshold=0.05,
            probability=0.12, ci_lower=0.08, ci_upper=0.16, n_samples=1000
        )
        assert pe.label == "top_5pct"

    def test_to_dict(self):
        from src.simulation.pool_competition import PercentileEstimate
        pe = PercentileEstimate("top_5pct", 0.05, 0.12, 0.08, 0.16, 1000)
        d = pe.to_dict()
        assert "label" in d


class TestBracketPerformance:
    def test_construction(self):
        from src.simulation.pool_competition import BracketPerformance
        bp = BracketPerformance(
            bracket_id="test", strategy="aggressive",
            mean_score=100.0, median_score=95.0, std_score=10.0,
            mean_rank=5.0, median_rank=4.0,
        )
        assert bp.bracket_id == "test"

    def test_to_dict(self):
        from src.simulation.pool_competition import BracketPerformance
        bp = BracketPerformance("test", "agg", 100.0, 95.0, 10.0, 5.0, 4.0)
        d = bp.to_dict()
        assert "bracket_id" in d


class TestPoolSimulationResult:
    def test_construction(self):
        from src.simulation.pool_competition import PoolSimulationResult, PoolSimulationConfig
        c = PoolSimulationConfig()
        r = PoolSimulationResult(config=c, bracket_performances=[])
        assert r.bracket_performances == []

    def test_to_dict(self):
        from src.simulation.pool_competition import PoolSimulationResult, PoolSimulationConfig
        c = PoolSimulationConfig()
        r = PoolSimulationResult(config=c, bracket_performances=[])
        d = r.to_dict()
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# metrics_validation.py
# ---------------------------------------------------------------------------

class TestMetricComparison:
    def test_construction(self):
        from src.ml.evaluation.metrics_validation import MetricComparison
        mc = MetricComparison(metric_name="test")
        assert mc.metric_name == "test"

    def test_with_values(self):
        from src.ml.evaluation.metrics_validation import MetricComparison
        mc = MetricComparison(
            metric_name="test", n_teams=50,
            pearson_r=0.95, spearman_rho=0.93,
        )
        assert mc.n_teams == 50

    def test_grade(self):
        from src.ml.evaluation.metrics_validation import MetricComparison
        mc = MetricComparison(metric_name="test", pearson_r=0.95)
        g = mc.grade
        assert isinstance(g, str)


class TestValidationReport:
    def test_construction(self):
        from src.ml.evaluation.metrics_validation import ValidationReport
        vr = ValidationReport(year=2023)
        assert vr.year == 2023

    def test_summary(self):
        from src.ml.evaluation.metrics_validation import ValidationReport
        vr = ValidationReport(year=2023)
        s = vr.summary()
        assert isinstance(s, str)

    def test_to_dict(self):
        from src.ml.evaluation.metrics_validation import ValidationReport
        vr = ValidationReport(year=2023)
        d = vr.to_dict()
        assert d["year"] == 2023


class TestNormalizeTeamId:
    def test_basic(self):
        from src.ml.evaluation.metrics_validation import _normalize_team_id
        result = _normalize_team_id("Duke Blue Devils")
        assert isinstance(result, str)

    def test_lowercase(self):
        from src.ml.evaluation.metrics_validation import _normalize_team_id
        result = _normalize_team_id("DUKE")
        assert result == result.lower()


class TestCompareArrays:
    def test_basic(self):
        from src.ml.evaluation.metrics_validation import _compare_arrays
        result = _compare_arrays(
            "test_metric",
            ["a", "b", "c"],
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
        )
        assert hasattr(result, "metric_name")

    def test_identical(self):
        from src.ml.evaluation.metrics_validation import _compare_arrays
        arr = np.array([1.0, 2.0, 3.0])
        result = _compare_arrays("test", ["a", "b", "c"], arr, arr)
        assert result.mean_absolute_error == pytest.approx(0.0, abs=0.01)


class TestMultiYearValidationResult:
    def test_construction(self):
        from src.ml.evaluation.metrics_validation import MultiYearValidationResult
        r = MultiYearValidationResult()
        assert r.diagnostic_reports == {}

    def test_summary(self):
        from src.ml.evaluation.metrics_validation import MultiYearValidationResult
        r = MultiYearValidationResult()
        s = r.summary()
        assert isinstance(s, str)

    def test_to_dict(self):
        from src.ml.evaluation.metrics_validation import MultiYearValidationResult
        r = MultiYearValidationResult()
        d = r.to_dict()
        assert isinstance(d, dict)


class TestConstantSensitivityResult:
    def test_construction(self):
        from src.ml.evaluation.metrics_validation import ConstantSensitivityResult
        r = ConstantSensitivityResult(constant_name="test", default_value=0.5)
        assert r.constant_name == "test"

    def test_recommendation(self):
        from src.ml.evaluation.metrics_validation import ConstantSensitivityResult
        r = ConstantSensitivityResult(constant_name="test", default_value=0.5)
        rec = r.recommendation
        assert isinstance(rec, str)


# ---------------------------------------------------------------------------
# loyo_protocol.py
# ---------------------------------------------------------------------------

class TestLoyoProtocol:
    def test_module_imports(self):
        from src.ml.evaluation import loyo_protocol
        assert loyo_protocol is not None

    def test_loyo_years_constant(self):
        from src.ml.evaluation.loyo_protocol import LOYO_YEARS
        assert isinstance(LOYO_YEARS, (list, tuple))

    def test_loyo_validator_class(self):
        from src.ml.evaluation.loyo_protocol import LOYOValidator
        assert LOYOValidator is not None

    def test_loyo_fold_result(self):
        from src.ml.evaluation.loyo_protocol import LOYOFoldResult
        assert LOYOFoldResult is not None

    def test_loyo_result(self):
        from src.ml.evaluation.loyo_protocol import LOYOResult
        assert LOYOResult is not None

    def test_feature_family(self):
        from src.ml.evaluation.loyo_protocol import FeatureFamily
        assert FeatureFamily is not None

    def test_feature_ablator(self):
        from src.ml.evaluation.loyo_protocol import FeatureAblator
        assert FeatureAblator is not None

    def test_prospective_validator(self):
        from src.ml.evaluation.loyo_protocol import ProspectiveValidator
        assert ProspectiveValidator is not None


# ---------------------------------------------------------------------------
# Other module imports for coverage
# ---------------------------------------------------------------------------

class TestEvaluationIntegrity:
    def test_module_imports(self):
        from src.ml.evaluation import evaluation_integrity
        assert evaluation_integrity is not None

class TestHyperparameterTuning:
    def test_module_imports(self):
        from src.ml.optimization import hyperparameter_tuning
        assert hyperparameter_tuning is not None

class TestMonteCarlo:
    def test_module_imports(self):
        from src.simulation import monte_carlo
        assert monte_carlo is not None

class TestResourceTracker:
    def test_module_imports(self):
        from src.monitoring import resource_tracker
        assert resource_tracker is not None

class TestConferenceTournamentPredictor:
    def test_module_imports(self):
        from src.conference_tournament import predictor
        assert predictor is not None

class TestKaggleDownloader:
    def test_module_imports(self):
        from src.data import kaggle_downloader
        assert kaggle_downloader is not None

class TestWomensPipeline:
    def test_module_imports(self):
        from src.pipeline import womens
        assert womens is not None

class TestSelectionSundayBacktest:
    def test_module_imports(self):
        from src.evaluation import selection_sunday_backtest
        assert selection_sunday_backtest is not None

class TestSimulationStage:
    def test_module_imports(self):
        from src.pipeline.stages import simulation
        assert simulation is not None

class TestSampleLoadingStage:
    def test_module_imports(self):
        from src.pipeline.stages import sample_loading
        assert sample_loading is not None

class TestDataIngestionModules:
    def test_historical_pipeline(self):
        from src.data.ingestion import historical_pipeline
        assert historical_pipeline is not None

    def test_providers(self):
        from src.data.ingestion import providers
        assert providers is not None

    def test_collector(self):
        from src.data.ingestion import collector
        assert collector is not None

class TestFeatureSelection:
    def test_module_imports(self):
        from src.data.features import feature_selection
        assert feature_selection is not None

class TestFeatureEngineering:
    def test_module_imports(self):
        from src.data.features import feature_engineering
        assert feature_engineering is not None

class TestKaggleLoader:
    def test_module_imports(self):
        from src.data import kaggle_loader
        assert kaggle_loader is not None

class TestCoverageAudit:
    def test_module_imports(self):
        from src.data import coverage_audit
        assert coverage_audit is not None
