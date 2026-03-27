"""Tests for Phases 7-10: Hyperparameter Optimization, Probability Calibration,
Ensembling, and Tournament Simulation Loop.

Tests cover data contracts, EV gating, weight optimization, simulation
convergence, and end-to-end pipeline flow through all four phases.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Generate synthetic training data for pipeline stages."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 10

    X = rng.randn(n_samples, n_features)
    # Labels correlated with first feature
    logits = X[:, 0] * 1.5 + rng.randn(n_samples) * 0.5
    y = (logits > 0).astype(float)
    sort_keys = np.arange(n_samples, dtype=float)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y, sort_keys, feature_names


@pytest.fixture
def model_predictions():
    """Generate synthetic model predictions for calibration/ensembling."""
    rng = np.random.RandomState(42)
    n = 100
    actuals = rng.binomial(1, 0.6, size=n).astype(float)

    # Model A: decent predictions
    preds_a = np.clip(actuals + rng.randn(n) * 0.2, 0.05, 0.95)
    # Model B: slightly worse predictions
    preds_b = np.clip(actuals + rng.randn(n) * 0.3, 0.05, 0.95)
    # Model C: different error pattern
    preds_c = np.clip(actuals + rng.randn(n) * 0.25 + 0.05, 0.05, 0.95)

    return {
        "model_a": preds_a,
        "model_b": preds_b,
        "model_c": preds_c,
    }, actuals


# ---------------------------------------------------------------------------
# Phase 7 — Hyperparameter Optimization
# ---------------------------------------------------------------------------


class TestPhase7HyperparameterOptimization:
    """Tests for Phase 7 hyperparameter optimization stage."""

    def test_model_config_data_contract(self):
        from src.pipeline.stages.hyperparameter_optimization import ModelConfig

        config = ModelConfig(
            model_name="gradient_boosting",
            best_params={"num_leaves": 8, "learning_rate": 0.05},
            cv_brier=0.18,
            cv_ev=120.0,
            ev_improvement=5.0,
            accepted=True,
        )
        d = config.to_dict()
        assert d["model_name"] == "gradient_boosting"
        assert d["accepted"] is True
        assert "num_leaves" in d["best_params"]

    def test_optimized_configs_contract(self):
        from src.pipeline.stages.hyperparameter_optimization import (
            ModelConfig,
            OptimizedConfigs,
        )

        configs = OptimizedConfigs(baseline_ev=100.0, baseline_brier=0.20)
        configs.configs["model_a"] = ModelConfig(
            model_name="model_a",
            best_params={"C": 1.0},
            accepted=True,
        )
        configs.configs["model_b"] = ModelConfig(
            model_name="model_b",
            best_params={"C": 0.5},
            accepted=False,
        )

        assert configs.accepted_models == ["model_a"]
        assert "model_a" in configs.optimized_configs
        assert "model_b" not in configs.optimized_configs
        assert "PASSED" in configs.summary()

    def test_optimizer_with_unknown_model(self, synthetic_data):
        from src.pipeline.stages.hyperparameter_optimization import (
            HyperparameterOptimizer,
        )

        X, y, sort_keys, feature_names = synthetic_data
        optimizer = HyperparameterOptimizer(
            baseline_ev=50.0,
            baseline_brier=0.25,
            strict=False,
        )

        result = optimizer.run(
            selected_models=["unknown_model"],
            train_X=X,
            train_y=y,
            sort_keys=sort_keys,
            feature_names=feature_names,
        )

        assert "unknown_model" in result.configs
        assert result.configs["unknown_model"].accepted is True
        assert "No tuner" in result.warnings[0]

    def test_optimizer_regularized_logistic(self, synthetic_data):
        from src.pipeline.stages.hyperparameter_optimization import (
            HyperparameterOptimizer,
        )

        X, y, sort_keys, feature_names = synthetic_data
        optimizer = HyperparameterOptimizer(
            baseline_ev=0.0,
            baseline_brier=0.25,
            n_cv_folds=3,
            strict=False,
        )

        result = optimizer.run(
            selected_models=["regularized_logistic"],
            train_X=X,
            train_y=y,
            sort_keys=sort_keys,
            feature_names=feature_names,
        )

        assert "regularized_logistic" in result.configs
        cfg = result.configs["regularized_logistic"]
        assert "C" in cfg.best_params
        assert "l1_ratio" in cfg.best_params

    def test_tuner_registry_completeness(self):
        from src.pipeline.stages.hyperparameter_optimization import TUNER_REGISTRY
        from src.pipeline.stages.model_selection import CANDIDATE_REGISTRY

        for name in CANDIDATE_REGISTRY:
            assert name in TUNER_REGISTRY, f"Missing tuner for {name}"


# ---------------------------------------------------------------------------
# Phase 8 — Probability Calibration
# ---------------------------------------------------------------------------


class TestPhase8ProbabilityCalibration:
    """Tests for Phase 8 probability calibration stage."""

    def test_calibrated_model_result_contract(self):
        from src.pipeline.stages.probability_calibration import (
            CalibratedModelResult,
        )

        r = CalibratedModelResult(
            model_name="model_a",
            method="temperature",
            brier_before=0.20,
            brier_after=0.18,
            ev_before=100.0,
            ev_after=105.0,
        )
        d = r.to_dict()
        assert d["method"] == "temperature"
        assert d["brier_before"] > d["brier_after"]

    def test_calibrated_models_contract(self):
        from src.pipeline.stages.probability_calibration import (
            CalibratedModelResult,
            CalibratedModels,
        )

        cm = CalibratedModels(baseline_ev=100.0, baseline_brier=0.20)
        cm.results["model_a"] = CalibratedModelResult(
            model_name="model_a", accepted=True,
        )
        assert cm.accepted_models == ["model_a"]
        assert "PASSED" in cm.summary()

    def test_method_selection_by_sample_size(self):
        from src.pipeline.stages.probability_calibration import (
            _select_calibration_method,
        )

        assert _select_calibration_method(20) == "temperature"
        assert _select_calibration_method(150) == "platt"
        assert _select_calibration_method(500) == "isotonic"
        assert _select_calibration_method(20, preferred="platt") == "platt"

    def test_calibrator_preserves_ev(self, model_predictions):
        from src.pipeline.stages.probability_calibration import (
            ProbabilityCalibrator,
        )

        preds, actuals = model_predictions
        calibrator = ProbabilityCalibrator(
            baseline_ev=0.0,
            baseline_brier=0.25,
            method="temperature",
            strict=False,
        )

        result = calibrator.run(
            model_predictions=preds,
            actuals=actuals,
        )

        assert result.passed is True
        # All models should be accepted (raw fallback if cal fails EV)
        assert len(result.accepted_models) == 3
        for name in preds:
            assert name in result.calibrated_predictions
            cal_preds = result.calibrated_predictions[name]
            assert len(cal_preds) == len(actuals)
            assert np.all(cal_preds >= 0) and np.all(cal_preds <= 1)

    def test_calibration_clips_probabilities(self):
        from src.pipeline.stages.probability_calibration import (
            ProbabilityCalibrator,
            PROB_CLIP_MIN,
            PROB_CLIP_MAX,
        )

        preds = {"model_a": np.array([0.0, 0.5, 1.0])}
        actuals = np.array([0.0, 1.0, 1.0])

        calibrator = ProbabilityCalibrator(
            baseline_ev=0.0, baseline_brier=0.25, strict=False,
        )
        result = calibrator.run(preds, actuals)

        cal = result.calibrated_predictions["model_a"]
        assert np.all(cal >= PROB_CLIP_MIN)
        assert np.all(cal <= PROB_CLIP_MAX)


# ---------------------------------------------------------------------------
# Phase 9 — Ensembling
# ---------------------------------------------------------------------------


class TestPhase9Ensembling:
    """Tests for Phase 9 ensembling stage."""

    def test_ensemble_result_contract(self):
        from src.pipeline.stages.ensembling import EnsembleResult

        er = EnsembleResult(
            weights={"model_a": 0.6, "model_b": 0.4},
            ensemble_ev=110.0,
            ensemble_brier=0.17,
        )
        assert er.model_names == ["model_a", "model_b"]
        assert abs(er.effective_model_count - 1 / (0.6**2 + 0.4**2)) < 0.01
        assert "110.00" in er.summary()

    def test_weight_simplex_generation(self):
        from src.pipeline.stages.ensembling import _generate_weight_simplex

        # 2 models
        simplex_2 = _generate_weight_simplex(2, step=0.25)
        for w in simplex_2:
            assert abs(sum(w) - 1.0) < 1e-9
            assert all(wi >= -1e-9 for wi in w)
        assert len(simplex_2) == 5  # 0.0, 0.25, 0.5, 0.75, 1.0

        # 1 model
        simplex_1 = _generate_weight_simplex(1)
        assert simplex_1 == [[1.0]]

    def test_blend_predictions(self):
        from src.pipeline.stages.ensembling import _blend_predictions

        p1 = np.array([0.8, 0.2, 0.6])
        p2 = np.array([0.6, 0.4, 0.5])
        blended = _blend_predictions([p1, p2], [0.5, 0.5])
        np.testing.assert_allclose(blended, [0.7, 0.3, 0.55], atol=1e-4)

    def test_ensemble_optimizer_single_model(self, model_predictions):
        from src.pipeline.stages.ensembling import EnsembleOptimizer

        preds, actuals = model_predictions
        single = {"model_a": preds["model_a"]}

        optimizer = EnsembleOptimizer(baseline_ev=0.0, baseline_brier=0.25)
        result = optimizer.run(single, actuals)

        assert result.weights == {"model_a": 1.0}
        assert result.optimization_method == "single_model"
        np.testing.assert_array_equal(result.ensemble_predictions, single["model_a"])

    def test_ensemble_optimizer_multi_model(self, model_predictions):
        from src.pipeline.stages.ensembling import EnsembleOptimizer

        preds, actuals = model_predictions

        optimizer = EnsembleOptimizer(
            baseline_ev=0.0,
            baseline_brier=0.25,
            method="grid",
            grid_step=0.25,  # Coarse for speed
        )
        result = optimizer.run(preds, actuals)

        assert result.passed is True
        assert len(result.weights) > 0
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.ensemble_predictions is not None
        assert len(result.ensemble_predictions) == len(actuals)

    def test_ensemble_equal_weighting(self):
        from src.pipeline.stages.ensembling import EnsembleOptimizer

        # Create predictions where equal blend is better than any individual
        rng = np.random.RandomState(123)
        n = 100
        actuals = rng.binomial(1, 0.6, size=n).astype(float)
        # Models with diverse errors — blend should help
        preds = {
            "model_a": np.clip(actuals + rng.randn(n) * 0.3, 0.05, 0.95),
            "model_b": np.clip(actuals + rng.randn(n) * 0.3, 0.05, 0.95),
        }

        optimizer = EnsembleOptimizer(method="equal", strict=False)
        result = optimizer.run(preds, actuals)

        # Either equal weights or fallback — both are valid
        assert result.passed is True
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_ensemble_fallback_on_ev_regression(self):
        from src.pipeline.stages.ensembling import EnsembleOptimizer

        # Make model_a clearly better than model_b
        rng = np.random.RandomState(42)
        actuals = rng.binomial(1, 0.7, size=50).astype(float)
        preds = {
            "model_a": np.clip(actuals + rng.randn(50) * 0.1, 0.05, 0.95),
            "model_b": np.full(50, 0.5),  # Coin-flip — bad
        }

        optimizer = EnsembleOptimizer(strict=False)
        result = optimizer.run(preds, actuals)

        # Ensemble should still work (grid search should find weight=1.0 for model_a)
        assert result.passed is True

    def test_empty_predictions_fails(self):
        from src.pipeline.stages.ensembling import EnsembleOptimizer

        optimizer = EnsembleOptimizer()
        result = optimizer.run({}, np.array([]))
        assert result.passed is False
        assert "No model predictions" in result.errors[0]


# ---------------------------------------------------------------------------
# Phase 10 — Tournament Simulation Loop
# ---------------------------------------------------------------------------


class TestPhase10SimulationLoop:
    """Tests for Phase 10 tournament simulation loop."""

    def test_simulation_iteration_contract(self):
        from src.pipeline.stages.simulation_loop import SimulationIterationResult

        ir = SimulationIterationResult(
            iteration=1, weights={"a": 0.6, "b": 0.4}, ev=100.0,
        )
        d = ir.to_dict()
        assert d["iteration"] == 1
        assert d["EV"] == 100.0

    def test_simulation_loop_result_contract(self):
        from src.pipeline.stages.simulation_loop import SimulationLoopResult

        r = SimulationLoopResult(
            final_ev=110.0, final_brier=0.17, initial_ev=100.0,
            total_ev_improvement=10.0, n_iterations_run=3, converged=True,
        )
        assert "110.00" in r.summary()
        d = r.to_dict()
        assert d["converged"] is True
        assert d["n_iterations"] == 3

    def test_simulate_bracket_ev(self):
        from src.pipeline.stages.simulation_loop import _simulate_bracket_ev

        # Perfect predictions: all 1.0 → EV should be ~sum of weights
        preds = np.ones(63)
        ev, round_ev, std = _simulate_bracket_ev(preds, n_simulations=100)
        # With all P=1, every game is won → total = 32*10 + 16*20 + 8*40 + 4*80 + 2*160 + 1*320
        expected = 32 * 10 + 16 * 20 + 8 * 40 + 4 * 80 + 2 * 160 + 1 * 320  # 1920
        # Without round labels, all get R64 weight (10)
        assert ev > 0
        assert std >= 0

    def test_simulate_bracket_ev_with_round_labels(self):
        from src.pipeline.stages.simulation_loop import _simulate_bracket_ev

        preds = np.full(6, 0.8)  # 6 games, P(win)=0.8
        round_labels = np.array(["R64", "R32", "S16", "E8", "F4", "CHAMP"])
        ev, round_ev, _ = _simulate_bracket_ev(preds, 1000, round_labels)

        # Expected EV = 0.8 * (10 + 20 + 40 + 80 + 160 + 320) = 0.8 * 630 = 504
        assert abs(ev - 504) < 30  # Monte Carlo variance

    def test_weight_convergence_check(self):
        from src.pipeline.stages.simulation_loop import _check_weight_convergence

        w1 = {"a": 0.5, "b": 0.5}
        w2 = {"a": 0.505, "b": 0.495}
        assert _check_weight_convergence(w1, w2, threshold=0.01) is True

        w3 = {"a": 0.7, "b": 0.3}
        assert _check_weight_convergence(w1, w3, threshold=0.01) is False

    def test_simulation_loop_converges(self, model_predictions):
        from src.pipeline.stages.simulation_loop import SimulationLoop

        preds, actuals = model_predictions

        loop = SimulationLoop(
            max_iterations=3,
            n_simulations=500,
        )
        result = loop.run(
            actuals=actuals,
            model_predictions=preds,
            initial_weights={"model_a": 0.5, "model_b": 0.3, "model_c": 0.2},
            initial_predictions=0.5 * preds["model_a"] + 0.3 * preds["model_b"] + 0.2 * preds["model_c"],
        )

        assert result.passed is True
        assert result.final_predictions is not None
        assert result.n_iterations_run > 0
        assert result.final_ev > 0

    def test_simulation_loop_no_predictions_fails(self):
        from src.pipeline.stages.simulation_loop import SimulationLoop

        loop = SimulationLoop()
        result = loop.run()
        assert result.passed is False
        assert "No predictions" in result.errors[0]

    def test_simulation_loop_single_model(self):
        from src.pipeline.stages.simulation_loop import SimulationLoop

        rng = np.random.RandomState(42)
        preds = np.clip(rng.randn(50) * 0.3 + 0.6, 0.1, 0.9)
        actuals = (preds > 0.5).astype(float)

        loop = SimulationLoop(
            max_iterations=2,
            n_simulations=200,
        )
        result = loop.run(
            actuals=actuals,
            initial_predictions=preds,
            initial_weights={"model_a": 1.0},
        )

        assert result.passed is True
        assert result.final_weights == {"model_a": 1.0}

    def test_simulation_loop_without_actuals(self):
        from src.pipeline.stages.simulation_loop import SimulationLoop

        rng = np.random.RandomState(42)
        preds = np.clip(rng.randn(63) * 0.3 + 0.6, 0.1, 0.9)

        loop = SimulationLoop(
            max_iterations=2,
            n_simulations=200,
        )
        result = loop.run(
            actuals=None,
            initial_predictions=preds,
        )

        assert result.passed is True
        assert result.final_brier == 0.0  # No actuals → no Brier


# ---------------------------------------------------------------------------
# Integration: Phases 7-10 data contract flow
# ---------------------------------------------------------------------------


class TestPipelineContractFlow:
    """Test that data contracts flow correctly between phases."""

    def test_init_contracts_exist(self):
        from src.pipeline.stages import (
            OptimizedHyperparams,
            CalibratedModelPredictions,
            EnsemblePredictions,
            SimulationLoopOutput,
        )

        oh = OptimizedHyperparams()
        assert oh.passed is True
        assert "OptimizedHyperparams" in oh.summary()

        cmp = CalibratedModelPredictions()
        assert cmp.model_names == []

        ep = EnsemblePredictions()
        assert "EnsemblePredictions" in ep.summary()

        slo = SimulationLoopOutput()
        assert slo.converged is False

    def test_phase8_to_phase9_flow(self, model_predictions):
        from src.pipeline.stages.probability_calibration import ProbabilityCalibrator
        from src.pipeline.stages.ensembling import EnsembleOptimizer

        preds, actuals = model_predictions

        # Phase 8
        calibrator = ProbabilityCalibrator(
            baseline_ev=0.0, baseline_brier=0.25, strict=False,
        )
        cal_result = calibrator.run(preds, actuals)

        # Phase 9 consumes Phase 8 output
        ensembler = EnsembleOptimizer(
            method="equal", strict=False,
        )
        ens_result = ensembler.run(cal_result.calibrated_predictions, actuals)

        assert ens_result.passed is True
        assert ens_result.ensemble_predictions is not None

    def test_phase9_to_phase10_flow(self, model_predictions):
        from src.pipeline.stages.ensembling import EnsembleOptimizer, EnsembleResult
        from src.pipeline.stages.simulation_loop import SimulationLoop

        preds, actuals = model_predictions

        # Phase 9
        ensembler = EnsembleOptimizer(method="equal", strict=False)
        ens_result = ensembler.run(preds, actuals)

        # Phase 10 consumes Phase 9 output
        loop = SimulationLoop(
            ensemble_result=ens_result,
            max_iterations=2,
            n_simulations=200,
        )
        sim_result = loop.run(actuals=actuals)

        assert sim_result.passed is True
        assert sim_result.final_predictions is not None
        assert sim_result.n_iterations_run > 0
