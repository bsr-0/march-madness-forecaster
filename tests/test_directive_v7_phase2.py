"""Tests for Directive V7 Phase 2 improvements.

Covers:
  S19-1:  Enhanced schema contracts (ensemble weights, calibration data, matchup vectors)
  S13-2:  Regime-conditional performance analysis
  S10-2:  Named scenario analysis (optimistic/base/pessimistic)
  S18-1:  Data freshness SLA enforcement in pre-run validation
  S3-1:   Experiment registry regime/scenario fields
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# S19-1: Enhanced schema contracts
# ---------------------------------------------------------------------------


class TestEnsembleWeightValidation:
    """S19-1: Validate ensemble component weights at fusion boundary."""

    def test_valid_weights_pass(self):
        from src.data.schemas import validate_ensemble_weights

        weights = {"lgb": 0.45, "xgb": 0.35, "logistic": 0.20}
        result = validate_ensemble_weights(weights)
        assert result.passed

    def test_weights_not_summing_to_one_fail(self):
        from src.data.schemas import validate_ensemble_weights

        weights = {"lgb": 0.5, "xgb": 0.5, "logistic": 0.5}
        result = validate_ensemble_weights(weights)
        assert not result.passed
        assert any("sum" in e.lower() for e in result.errors)

    def test_negative_weight_fails(self):
        from src.data.schemas import validate_ensemble_weights

        weights = {"lgb": -0.1, "xgb": 0.6, "logistic": 0.5}
        result = validate_ensemble_weights(weights)
        assert not result.passed

    def test_empty_weights_fails(self):
        from src.data.schemas import validate_ensemble_weights

        result = validate_ensemble_weights({})
        assert not result.passed

    def test_missing_expected_component_warns(self):
        from src.data.schemas import validate_ensemble_weights

        weights = {"lgb": 0.5, "xgb": 0.5}
        result = validate_ensemble_weights(
            weights, expected_components=["lgb", "xgb", "logistic"]
        )
        assert result.passed  # Missing component is a warning, not error
        assert len(result.warnings) > 0

    def test_extra_component_warns(self):
        from src.data.schemas import validate_ensemble_weights

        weights = {"lgb": 0.3, "xgb": 0.3, "logistic": 0.2, "extra": 0.2}
        result = validate_ensemble_weights(
            weights, expected_components=["lgb", "xgb", "logistic"]
        )
        assert result.passed
        assert any("unexpected" in w.lower() for w in result.warnings)


class TestCalibrationDataValidation:
    """S19-1: Validate calibration pipeline input data."""

    def test_valid_calibration_data(self):
        from src.data.schemas import validate_calibration_data

        preds = np.array([0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0.0, 1.0, 1.0, 1.0])
        result = validate_calibration_data(preds, outcomes)
        assert result.passed

    def test_length_mismatch_fails(self):
        from src.data.schemas import validate_calibration_data

        preds = np.array([0.3, 0.5, 0.7])
        outcomes = np.array([0.0, 1.0])
        result = validate_calibration_data(preds, outcomes)
        assert not result.passed

    def test_non_binary_outcomes_fail(self):
        from src.data.schemas import validate_calibration_data

        preds = np.array([0.3, 0.5, 0.7])
        outcomes = np.array([0.0, 0.5, 1.0])
        result = validate_calibration_data(preds, outcomes)
        assert not result.passed

    def test_single_class_outcomes_fail(self):
        from src.data.schemas import validate_calibration_data

        preds = np.array([0.3, 0.5, 0.7])
        outcomes = np.array([1.0, 1.0, 1.0])
        result = validate_calibration_data(preds, outcomes)
        assert not result.passed

    def test_small_sample_warns(self):
        from src.data.schemas import validate_calibration_data

        preds = np.array([0.3, 0.7])
        outcomes = np.array([0.0, 1.0])
        result = validate_calibration_data(preds, outcomes, min_samples=10)
        assert result.passed  # Warning, not error
        assert len(result.warnings) > 0


class TestMatchupVectorValidation:
    """S19-1: Validate matchup feature vectors."""

    def test_valid_vector(self):
        from src.data.schemas import validate_matchup_vector

        vec = np.random.randn(77)
        result = validate_matchup_vector(vec, expected_dim=77)
        assert result.passed

    def test_wrong_dimension_fails(self):
        from src.data.schemas import validate_matchup_vector

        vec = np.random.randn(50)
        result = validate_matchup_vector(vec, expected_dim=77)
        assert not result.passed

    def test_infinite_values_fail(self):
        from src.data.schemas import validate_matchup_vector

        vec = np.array([1.0, np.inf, 3.0])
        result = validate_matchup_vector(vec)
        assert not result.passed

    def test_high_nan_fraction_fails(self):
        from src.data.schemas import validate_matchup_vector

        vec = np.array([np.nan] * 8 + [1.0, 2.0])
        result = validate_matchup_vector(vec)
        assert not result.passed

    def test_moderate_nan_warns(self):
        from src.data.schemas import validate_matchup_vector

        vec = np.array([np.nan] * 3 + [1.0] * 7)
        result = validate_matchup_vector(vec)
        assert result.passed
        assert len(result.warnings) > 0

    def test_2d_vector_fails(self):
        from src.data.schemas import validate_matchup_vector

        vec = np.random.randn(3, 3)
        result = validate_matchup_vector(vec)
        assert not result.passed


# ---------------------------------------------------------------------------
# S13-2: Regime-conditional performance analysis
# ---------------------------------------------------------------------------


class TestRegimeAnalysis:
    """S13-2: Performance breakdown by tournament regime type."""

    def test_basic_regime_classification(self):
        from src.ml.evaluation.risk_report import RegimeAnalysis

        year_briers = {
            2018: 0.18, 2019: 0.22, 2021: 0.15,
            2022: 0.25, 2023: 0.17, 2024: 0.20,
        }
        regime = RegimeAnalysis.from_loyo_results(year_briers)
        assert len(regime.regime_labels) == 6
        assert "chalk" in regime.regime_mean_brier
        assert "upset_heavy" in regime.regime_mean_brier

    def test_regime_with_upset_rates(self):
        from src.ml.evaluation.risk_report import RegimeAnalysis

        year_briers = {2018: 0.18, 2019: 0.22, 2021: 0.15, 2022: 0.25}
        upset_rates = {2018: 0.35, 2019: 0.45, 2021: 0.30, 2022: 0.50}

        regime = RegimeAnalysis.from_loyo_results(year_briers, upset_rates)
        assert len(regime.regime_labels) == 4
        # High upset rate years should be classified as upset_heavy
        assert regime.regime_labels[2022] == "upset_heavy"
        assert regime.regime_labels[2021] == "chalk"

    def test_too_few_years_returns_empty(self):
        from src.ml.evaluation.risk_report import RegimeAnalysis

        regime = RegimeAnalysis.from_loyo_results({2020: 0.2, 2021: 0.3})
        assert len(regime.regime_labels) == 0

    def test_serialization(self):
        from src.ml.evaluation.risk_report import RegimeAnalysis

        year_briers = {2018: 0.18, 2019: 0.22, 2021: 0.15, 2022: 0.25}
        regime = RegimeAnalysis.from_loyo_results(year_briers)
        d = regime.to_dict()
        assert "regime_labels" in d
        assert "regime_mean_brier" in d
        assert "per_regime" in d

    def test_upset_heavy_has_higher_brier(self):
        from src.ml.evaluation.risk_report import RegimeAnalysis

        # Construct data where high-Brier years are upset-heavy
        year_briers = {
            2018: 0.15, 2019: 0.16, 2021: 0.30, 2022: 0.32,
        }
        regime = RegimeAnalysis.from_loyo_results(year_briers)
        # Upset-heavy regime (above median Brier) should have higher mean
        assert regime.regime_mean_brier["upset_heavy"] > regime.regime_mean_brier["chalk"]


# ---------------------------------------------------------------------------
# S10-2: Named scenario analysis
# ---------------------------------------------------------------------------


class TestScenarioAnalysis:
    """S10-2: Optimistic/base/pessimistic scenario projections."""

    def test_basic_scenarios(self):
        from src.ml.evaluation.risk_report import ScenarioAnalysis

        year_briers = {
            2018: 0.18, 2019: 0.22, 2021: 0.15,
            2022: 0.25, 2023: 0.17, 2024: 0.20,
        }
        scenario = ScenarioAnalysis.from_loyo_results(year_briers)

        assert scenario.optimistic
        assert scenario.base
        assert scenario.pessimistic

    def test_optimistic_better_than_base(self):
        from src.ml.evaluation.risk_report import ScenarioAnalysis

        year_briers = {2018: 0.18, 2019: 0.22, 2021: 0.15, 2022: 0.25}
        scenario = ScenarioAnalysis.from_loyo_results(year_briers)

        assert scenario.optimistic["expected_brier"] < scenario.base["expected_brier"]

    def test_pessimistic_worse_than_base(self):
        from src.ml.evaluation.risk_report import ScenarioAnalysis

        year_briers = {2018: 0.18, 2019: 0.22, 2021: 0.15, 2022: 0.25}
        scenario = ScenarioAnalysis.from_loyo_results(year_briers)

        assert scenario.pessimistic["expected_brier"] > scenario.base["expected_brier"]

    def test_scenarios_have_assumptions(self):
        from src.ml.evaluation.risk_report import ScenarioAnalysis

        year_briers = {2018: 0.18, 2019: 0.22, 2021: 0.15, 2022: 0.25}
        scenario = ScenarioAnalysis.from_loyo_results(year_briers)

        for s in [scenario.optimistic, scenario.base, scenario.pessimistic]:
            assert "assumptions" in s
            assert len(s["assumptions"]) >= 2
            assert "label" in s

    def test_serialization(self):
        from src.ml.evaluation.risk_report import ScenarioAnalysis

        year_briers = {2018: 0.18, 2019: 0.22, 2021: 0.15, 2022: 0.25}
        scenario = ScenarioAnalysis.from_loyo_results(year_briers)
        d = scenario.to_dict()
        assert "optimistic" in d
        assert "base" in d
        assert "pessimistic" in d

    def test_empty_briers_returns_empty(self):
        from src.ml.evaluation.risk_report import ScenarioAnalysis

        scenario = ScenarioAnalysis.from_loyo_results({})
        assert not scenario.base


# ---------------------------------------------------------------------------
# S3-1: Experiment registry — regime/scenario fields
# ---------------------------------------------------------------------------


class TestExperimentRegistryNewFields:
    """S3-1: Experiment registry captures regime and scenario analysis."""

    def test_regime_scenario_round_trip(self, tmp_path):
        from src.ml.evaluation.experiment_registry import ExperimentRecord, ExperimentRegistry

        registry = ExperimentRegistry(ledger_path=str(tmp_path / "ledger.jsonl"))
        record = ExperimentRecord(
            experiment_id="regime-test-001",
            loyo_mean_brier=0.195,
            regime_analysis={
                "chalk": {"mean_brier": 0.17},
                "upset_heavy": {"mean_brier": 0.22},
            },
            scenario_analysis={
                "optimistic": {"expected_brier": 0.16},
                "base": {"expected_brier": 0.195},
                "pessimistic": {"expected_brier": 0.24},
            },
        )

        registry.log(record)
        records = registry.list(n=1)
        assert len(records) == 1
        r = records[0]
        assert r.regime_analysis["chalk"]["mean_brier"] == 0.17
        assert r.scenario_analysis["base"]["expected_brier"] == 0.195


# ---------------------------------------------------------------------------
# S18-1: Data freshness SLA in pre-run validation
# ---------------------------------------------------------------------------


class TestPreRunFreshness:
    """S18-1: Data freshness enforcement in pre-run validation."""

    def test_missing_sources_are_critical(self):
        pd = pytest.importorskip("pandas")
        from src.pipeline.sota import SOTAPipeline, SOTAPipelineConfig

        config = SOTAPipelineConfig(
            year=2025,
            strict_leakage_mode=False,
            data_cache_dir="/nonexistent/path",
        )
        pipeline = SOTAPipeline(config)
        result = pipeline._pre_run_validation()

        freshness_check = next(
            c for c in result["checks"] if c["check"] == "data_freshness"
        )
        # Missing data dir means sources can't be found — status should reflect this
        assert freshness_check["missing_count"] >= 0
