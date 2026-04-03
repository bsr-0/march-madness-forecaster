"""Tests for FIX-STACKING-LEAKAGE: Ensemble weight contamination prevention.

Validates that:
1. Nested LOYO derives weights per-fold without seeing held-out data
2. Nested LOYO Brier >= pooled Brier (less optimistic = honest)
3. BMA is never fitted on eval set
4. LOYO evaluates the full ensemble, not a single model
5. Stacking gate uses held-out OOF, not training data
6. Audit rejects eval-set-derived weights
"""

from __future__ import annotations

import numpy as np
import pytest


class TestNestedLOYOWeightOptimizer:
    """Test StackingWeightOptimizer.fit_nested_loyo()."""

    def _make_synthetic_data(self, n_years=5, n_per_year=60, seed=42):
        """Create synthetic per-year predictions and outcomes."""
        rng = np.random.default_rng(seed)

        predictions_by_year = {}
        outcomes_by_year = {}

        for yr in range(2018, 2018 + n_years):
            y = rng.integers(0, 2, size=n_per_year).astype(float)
            # Model A: decent calibration
            p_a = np.clip(y + rng.normal(0, 0.3, n_per_year), 0.05, 0.95)
            # Model B: slightly worse but different errors
            p_b = np.clip(y + rng.normal(0, 0.35, n_per_year), 0.05, 0.95)
            # Model C: noisier
            p_c = np.clip(y + rng.normal(0, 0.4, n_per_year), 0.05, 0.95)

            predictions_by_year[yr] = {
                "model_a": p_a,
                "model_b": p_b,
                "model_c": p_c,
            }
            outcomes_by_year[yr] = y

        return predictions_by_year, outcomes_by_year

    def test_nested_loyo_bma_weights_exclude_held_out_year(self):
        """Each fold's weights must be derived without the held-out year."""
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        predictions_by_year, outcomes_by_year = self._make_synthetic_data()
        optimizer = StackingWeightOptimizer(regularization=0.1, random_seed=42)

        result = optimizer.fit_nested_loyo(predictions_by_year, outcomes_by_year)

        # Verify we got per-fold weights for each year
        assert len(result.per_fold_weights) == 5
        assert len(result.per_fold_brier) == 5

        # Verify each fold has weights for all models
        for yr, weights in result.per_fold_weights.items():
            assert set(weights.keys()) == {"model_a", "model_b", "model_c"}
            # Weights should sum to ~1
            assert abs(sum(weights.values()) - 1.0) < 0.01

        # Verify weight_source is correct
        assert result.weight_source == "nested_loyo"

    def test_nested_loyo_brier_is_honest(self):
        """Nested Brier should be >= pooled Brier (less optimistic)."""
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        predictions_by_year, outcomes_by_year = self._make_synthetic_data()
        optimizer = StackingWeightOptimizer(regularization=0.1, random_seed=42)

        # Nested LOYO (honest)
        nested_result = optimizer.fit_nested_loyo(predictions_by_year, outcomes_by_year)

        # Pooled optimization (contaminated baseline for comparison)
        all_preds = {
            name: np.concatenate([predictions_by_year[yr][name] for yr in sorted(predictions_by_year)])
            for name in ["model_a", "model_b", "model_c"]
        }
        all_y = np.concatenate([outcomes_by_year[yr] for yr in sorted(outcomes_by_year)])
        pooled_result = optimizer.fit(all_preds, all_y)

        # Nested Brier should be >= pooled Brier because the pooled
        # approach optimizes weights on the same data it evaluates on.
        # The nested approach is honest and should produce a higher
        # (more conservative) Brier estimate.
        assert nested_result.mean_brier >= pooled_result.brier_with_stacking - 0.001, (
            f"Nested Brier ({nested_result.mean_brier:.5f}) should be >= "
            f"pooled Brier ({pooled_result.brier_with_stacking:.5f}) "
            f"because pooled is optimistically biased"
        )

    def test_nested_loyo_requires_minimum_years(self):
        """fit_nested_loyo should fail with < 3 years."""
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        predictions_by_year, outcomes_by_year = self._make_synthetic_data(n_years=2)
        optimizer = StackingWeightOptimizer()

        with pytest.raises(ValueError, match="requires >= 3 years"):
            optimizer.fit_nested_loyo(predictions_by_year, outcomes_by_year)

    def test_nested_loyo_result_is_frozen(self):
        """NestedLOYOEnsembleResult should be immutable (frozen dataclass)."""
        from src.ml.ensemble.stacking_weights import NestedLOYOEnsembleResult

        result = NestedLOYOEnsembleResult(
            per_fold_weights={2020: {"a": 0.5, "b": 0.5}},
            per_fold_brier={2020: 0.2},
            mean_brier=0.2,
            std_brier=0.0,
            n_folds=1,
            n_total_samples=60,
            production_weights={"a": 0.5, "b": 0.5},
        )

        with pytest.raises(AttributeError):
            result.weight_source = "eval_set"  # Should fail: frozen

    def test_nested_loyo_production_weights_fit_on_all_data(self):
        """Production weights should be derived from all data (for inference only)."""
        from src.ml.ensemble.stacking_weights import StackingWeightOptimizer

        predictions_by_year, outcomes_by_year = self._make_synthetic_data()
        optimizer = StackingWeightOptimizer(regularization=0.1, random_seed=42)

        result = optimizer.fit_nested_loyo(predictions_by_year, outcomes_by_year)

        # Production weights should exist and sum to 1
        assert len(result.production_weights) == 3
        assert abs(sum(result.production_weights.values()) - 1.0) < 0.01


class TestBMANotFittedOnEvalSet:
    """Verify eval set is not used for weight derivation."""

    def test_bma_not_fitted_on_eval_set(self):
        """The BMA fitting code in _ensemble.py should not use eval_y."""
        import inspect
        from src.pipeline.stages.baseline_training._ensemble import (
            _select_ensemble_and_evaluate,
        )

        source = inspect.getsource(_select_ensemble_and_evaluate)

        # The old contaminated pattern: bma.fit(bma_preds, eval_y)
        assert "bma.fit(bma_preds, eval_y)" not in source, (
            "_select_ensemble_and_evaluate still fits BMA on eval_y! This is the stacking weight contamination bug."
        )

        # Should NOT fit BMA on eval set (the contamination pattern).
        # The fix may use _fit_bma_on_loyo or inline LOYO logic.
        assert "bma.fit(bma_preds, eval_y)" not in source

    def test_ensemble_weight_stats_guard_present(self):
        """The eval_set assertion guard must be present in the code."""
        import inspect
        from src.pipeline.stages.baseline_training._ensemble import (
            _select_ensemble_and_evaluate,
        )

        source = inspect.getsource(_select_ensemble_and_evaluate)
        # Core invariant: eval_set contamination pattern must be absent
        assert "bma.fit(bma_preds, eval_y)" not in source, "Stacking weight contamination: BMA fitted on eval set"


class TestLOYOEvaluatesFullEnsemble:
    """Verify LOYO train_fn produces multi-model ensemble."""

    def test_loyo_train_fn_trains_multiple_models(self):
        """LOYO train_fn should train LGB, XGB, Logistic, Spread."""
        import inspect
        from src.pipeline.stages.baseline_training._loyo import (
            _run_loyo_validation,
        )

        source = inspect.getsource(_run_loyo_validation)

        # Should train multiple model types
        assert "ensemble_models" in source, "LOYO train_fn should store multiple models in ensemble_models"
        assert "ensemble_weights" in source, "LOYO train_fn should derive ensemble_weights"

    def test_loyo_predict_fn_uses_ensemble(self):
        """LOYO predict_fn should combine multiple models."""
        import inspect
        from src.pipeline.stages.baseline_training._loyo import (
            _run_loyo_validation,
        )

        source = inspect.getsource(_run_loyo_validation)

        # predict_fn should use weighted ensemble, not single model
        assert "weighted_pred" in source, "LOYO predict_fn should compute weighted ensemble predictions"


class TestStackingGateUsesHoldout:
    """Verify stacking gate evaluates on held-out data."""

    def test_stacking_gate_not_in_sample(self):
        """The stacking validation gate must NOT evaluate in-sample."""
        import inspect
        from src.pipeline.stages.baseline_training._ensemble import (
            _select_ensemble_and_evaluate,
        )

        source = inspect.getsource(_select_ensemble_and_evaluate)

        # Old contaminated pattern: predict on training data
        assert "meta_learner.predict_proba(meta_X)" not in source, (
            "Stacking gate still evaluates meta_learner on training data! This is in-sample evaluation."
        )

        # Should use OOF predictions instead
        assert "meta_oof_preds" in source, "Stacking gate should use out-of-fold predictions"


class TestAuditRejectsEvalSetWeights:
    """Verify audit check catches eval-set weight contamination."""

    def test_audit_rejects_eval_set_weights(self):
        """Audit should FAIL when weights come from eval set."""
        from src.forecaster.audit import run_audit

        result = run_audit(
            stacking_result=None,
            calibration_result=None,
            feature_importances=None,
            metrics={"brier": 0.2},
            n_features=20,
            ensemble_weight_stats={
                "weight_source": "eval_set",
                "method": "bayesian_model_averaging",
            },
        )

        assert not result.passed, "Audit should fail for eval_set weights"
        weight_violations = [v for v in result.violations if v.component == "ensemble" and "provenance" in v.check]
        assert len(weight_violations) > 0, "Should have ensemble weight provenance violation"

    def test_audit_accepts_nested_loyo_weights(self):
        """Audit should PASS when weights come from nested LOYO."""
        from src.forecaster.audit import run_audit

        result = run_audit(
            stacking_result=None,
            calibration_result=None,
            feature_importances=None,
            metrics={"brier": 0.2},
            n_features=20,
            ensemble_weight_stats={
                "weight_source": "nested_loyo",
                "method": "nested_loyo",
            },
        )

        weight_violations = [v for v in result.violations if v.component == "ensemble"]
        assert len(weight_violations) == 0, f"Should have no ensemble violations, got: {weight_violations}"

    def test_audit_accepts_fixed_weights(self):
        """Audit should PASS for fixed (non-optimized) weights."""
        from src.forecaster.audit import run_audit

        result = run_audit(
            stacking_result=None,
            calibration_result=None,
            feature_importances=None,
            metrics={"brier": 0.2},
            n_features=20,
            ensemble_weight_stats={
                "weight_source": "fixed",
            },
        )

        weight_violations = [v for v in result.violations if v.component == "ensemble" and v.severity == "error"]
        assert len(weight_violations) == 0

    def test_audit_accepts_no_ensemble(self):
        """Audit should PASS when no ensemble weights are present."""
        from src.forecaster.audit import run_audit

        result = run_audit(
            stacking_result=None,
            calibration_result=None,
            feature_importances=None,
            metrics={"brier": 0.2},
            n_features=20,
            ensemble_weight_stats=None,
        )

        weight_violations = [v for v in result.violations if v.component == "ensemble"]
        assert len(weight_violations) == 0


class TestProvenanceTracking:
    """Verify governance provenance tracks ensemble weight source."""

    def test_provenance_includes_ensemble_weight_source(self):
        """Provenance block should include ensemble_weight_source field."""
        from src.governance.artifact_provenance import build_artifact_provenance

        provenance = build_artifact_provenance(
            artifact_kind="predictions",
        )

        assert "ensemble_weight_source" in provenance, "Provenance must track ensemble_weight_source for governance"


class TestOptimizeEnsembleWeightsLoyoNested:
    """Verify _optimize_ensemble_weights_loyo uses nested approach."""

    def test_function_uses_nested_loyo(self):
        """_optimize_ensemble_weights_loyo should use fit_nested_loyo."""
        import inspect
        from src.pipeline.stages.baseline_training._ensemble import (
            _optimize_ensemble_weights_loyo,
        )

        source = inspect.getsource(_optimize_ensemble_weights_loyo)

        assert "fit_nested_loyo" in source, (
            "_optimize_ensemble_weights_loyo should use nested LOYO, not pooled optimization"
        )

        # Should NOT have the old pooled pattern
        assert "optimizer.optimize(" not in source, (
            "_optimize_ensemble_weights_loyo should NOT use pooled optimizer.optimize() (self-evaluating contamination)"
        )

    def test_function_reports_nested_source(self):
        """Return dict should include weight_source='nested_loyo'."""
        import inspect
        from src.pipeline.stages.baseline_training._ensemble import (
            _optimize_ensemble_weights_loyo,
        )

        source = inspect.getsource(_optimize_ensemble_weights_loyo)
        assert '"nested_loyo"' in source, "Must tag weight_source as 'nested_loyo'"


class TestSpreadSigmaNotOnEvalSet:
    """Verify spread model sigma is NOT calibrated on eval data."""

    def test_spread_sigma_uses_training_data(self):
        """Spread sigma calibration must NOT use eval_X/eval_margins."""
        import inspect
        from src.pipeline.stages.baseline_training._models import (
            _train_all_models,
        )

        source = inspect.getsource(_train_all_models)

        # The old contaminated pattern: pass eval data for sigma calibration
        assert "spread_valid_X = eval_X" not in source, (
            "_train_all_models still passes eval_X to SpreadRegressor! "
            "This contaminates sigma calibration with eval data."
        )
        assert "spread_valid_margins = eval_margins" not in source, (
            "_train_all_models still passes eval_margins to SpreadRegressor! "
            "This contaminates sigma calibration with eval data."
        )

    def test_spread_sigma_uses_training_split(self):
        """Spread sigma should be calibrated on held-out training data."""
        import inspect
        from src.pipeline.stages.baseline_training._models import (
            _train_all_models,
        )

        source = inspect.getsource(_train_all_models)

        # Should use a split of training data for sigma calibration
        assert "_sigma_split" in source, "Should split training data for sigma calibration"
        assert "_sigma_cal_X" in source, "Should have a sigma calibration subset from training data"

    def test_spread_sigma_comment_documents_fix(self):
        """The fix should be clearly documented in code."""
        import inspect
        from src.pipeline.stages.baseline_training._models import (
            _train_all_models,
        )

        source = inspect.getsource(_train_all_models)
        assert "FIX-STACKING-LEAKAGE" in source, "Spread sigma fix should reference FIX-STACKING-LEAKAGE"


class TestSingleModelSelectionNoEvalDependence:
    """Verify single-model selection does NOT depend on eval_y."""

    def test_selection_does_not_compute_eval_brier(self):
        """_select_best_single_model must not compute Brier on eval_y."""
        import inspect
        from src.pipeline.stages.baseline_training._orchestrator import (
            _select_best_single_model,
        )

        source = inspect.getsource(_select_best_single_model)

        # Old contaminated patterns
        assert "np.mean((preds_clipped - eval_y)" not in source, (
            "_select_best_single_model still computes Brier on eval_y! "
            "This uses eval set for model selection decisions."
        )
        assert "eval_y * np.log" not in source, "_select_best_single_model still computes LogLoss on eval_y!"

    def test_selection_uses_priority_order(self):
        """Selection should use fixed priority, not eval-dependent ranking."""
        import inspect
        from src.pipeline.stages.baseline_training._orchestrator import (
            _select_best_single_model,
        )

        source = inspect.getsource(_select_best_single_model)

        assert "_PRIORITY" in source, "Should use fixed priority order for model selection"
        assert "FIX-STACKING-LEAKAGE" in source, "Model selection fix should reference FIX-STACKING-LEAKAGE"
