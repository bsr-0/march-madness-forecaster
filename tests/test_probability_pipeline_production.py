"""Tests for the production probability pipeline.

Verifies that:
- Production paths use only: raw → calibrate → shrink → clip
- No banned experimental components leak into production
- Shared helpers behave correctly
- Config hard-fails on invalid production settings
"""

from __future__ import annotations

import inspect
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.probability_pipeline import (
    apply_calibration,
    apply_final_clip,
    apply_tournament_shrinkage,
    compute_raw_probability,
    run_ablation_ladder,
)


# ───────────────────────────────────────────────────────────────────────
# Task 8: compute_raw_probability validates inputs
# ───────────────────────────────────────────────────────────────────────


class TestComputeRawProbability:
    def test_finite_passthrough(self):
        assert compute_raw_probability(0.75) == 0.75
        assert compute_raw_probability(0.0) == 0.0
        assert compute_raw_probability(1.0) == 1.0

    def test_nan_returns_half(self):
        assert compute_raw_probability(float("nan")) == 0.5

    def test_inf_returns_half(self):
        assert compute_raw_probability(float("inf")) == 0.5
        assert compute_raw_probability(float("-inf")) == 0.5


# ───────────────────────────────────────────────────────────────────────
# Task 3: production output equals clip(shrink(calibrate(raw)))
# ───────────────────────────────────────────────────────────────────────


class TestProductionPipelineFormula:
    def test_production_output_equals_pipeline(self):
        """Manually walk the 4-stage pipeline and verify equivalence."""
        raw = 0.72
        clip_lo, clip_hi = 0.005, 0.995
        shrinkage = 0.06

        # Build a mock calibrator: temperature=1.2
        calibrator = MagicMock()
        calibrator.fitted = True
        calibrator.temperature = 1.2
        # Remove .calibrate so the BrierCalibrator branch is used
        del calibrator.calibrate

        stage1 = apply_calibration(raw, calibrator, clip_lo, clip_hi)
        stage2 = apply_tournament_shrinkage(stage1, shrinkage)
        stage3 = apply_final_clip(stage2, clip_lo, clip_hi)

        # Manually compute expected
        logit = np.log(raw / (1.0 - raw))
        cal = 1.0 / (1.0 + np.exp(-logit / 1.2))
        cal = float(np.clip(cal, clip_lo, clip_hi))
        shrunk = shrinkage * 0.5 + (1.0 - shrinkage) * cal
        expected = float(np.clip(shrunk, clip_lo, clip_hi))

        assert stage3 == pytest.approx(expected, abs=1e-10)

    def test_no_calibrator_path(self):
        """When calibrator is None, output equals clip(shrink(raw))."""
        raw = 0.65
        clip_lo, clip_hi = 0.01, 0.99
        shrinkage = 0.04

        stage1 = apply_calibration(raw, None, clip_lo, clip_hi)
        assert stage1 == raw  # identity

        stage2 = apply_tournament_shrinkage(stage1, shrinkage)
        expected_shrunk = shrinkage * 0.5 + (1.0 - shrinkage) * raw
        assert stage2 == pytest.approx(expected_shrunk, abs=1e-10)

        stage3 = apply_final_clip(stage2, clip_lo, clip_hi)
        assert stage3 == pytest.approx(expected_shrunk, abs=1e-10)


# ───────────────────────────────────────────────────────────────────────
# Task 5: _tournament_adapt performs shrinkage only
# ───────────────────────────────────────────────────────────────────────


class TestTournamentShrinkage:
    def test_shrinkage_formula(self):
        prob = 0.80
        shrinkage = 0.10
        result = apply_tournament_shrinkage(prob, shrinkage)
        expected = 0.10 * 0.5 + 0.90 * 0.80
        assert result == pytest.approx(expected, abs=1e-10)

    def test_zero_shrinkage_identity(self):
        assert apply_tournament_shrinkage(0.73, 0.0) == pytest.approx(0.73)

    def test_full_shrinkage_gives_half(self):
        assert apply_tournament_shrinkage(0.9, 1.0) == pytest.approx(0.5)


# ───────────────────────────────────────────────────────────────────────
# Task 6: invalid production config raises hard errors
# ───────────────────────────────────────────────────────────────────────


class TestProductionConfigValidation:
    def test_sota_seed_overrides_allowed_in_production(self):
        from src.pipeline.config import SOTAPipelineConfig

        # Seed overrides are now sanctioned in production
        cfg = SOTAPipelineConfig(
            probability_profile="production",
            enable_seed_overrides=True,
        )
        assert cfg.enable_seed_overrides is True

    def test_sota_sharpening_allowed_in_production(self):
        from src.pipeline.config import SOTAPipelineConfig

        # Brier sharpening is now sanctioned in production
        cfg = SOTAPipelineConfig(
            probability_profile="production",
            enable_brier_sharpening=True,
        )
        assert cfg.enable_brier_sharpening is True

    def test_sota_goto_allowed_in_production(self):
        from src.pipeline.config import SOTAPipelineConfig

        # GoTo conversion is now sanctioned in production
        cfg = SOTAPipelineConfig(
            probability_profile="production",
            enable_goto_conversion=True,
        )
        assert cfg.enable_goto_conversion is True

    def test_sota_round_weighted_allowed_in_production(self):
        from src.pipeline.config import SOTAPipelineConfig

        # Round-weighted calibration is now sanctioned in production
        cfg = SOTAPipelineConfig(
            probability_profile="production",
            enable_round_weighted_calibration=True,
        )
        assert cfg.enable_round_weighted_calibration is True

    def test_sota_seed_prior_allowed_in_production(self):
        from src.pipeline.config import SOTAPipelineConfig

        # Seed prior weight is now allowed in production
        cfg = SOTAPipelineConfig(
            probability_profile="production",
            seed_prior_weight=0.1,
        )
        assert cfg.seed_prior_weight == 0.1

    def test_sota_consistency_bonus_allowed_in_production(self):
        from src.pipeline.config import SOTAPipelineConfig

        # Consistency bonus is now allowed in production
        cfg = SOTAPipelineConfig(
            probability_profile="production",
            consistency_bonus_max=0.01,
        )
        assert cfg.consistency_bonus_max == 0.01

    def test_womens_sharpening_raises(self):
        from src.pipeline.womens import WomensPipelineConfig

        with pytest.raises(ValueError, match="enable_brier_sharpening"):
            WomensPipelineConfig(
                probability_profile="production",
                enable_brier_sharpening=True,
            )

    def test_womens_goto_raises(self):
        from src.pipeline.womens import WomensPipelineConfig

        with pytest.raises(ValueError, match="enable_goto_conversion"):
            WomensPipelineConfig(
                probability_profile="production",
                enable_goto_conversion=True,
            )

    def test_womens_seed_prior_raises(self):
        from src.pipeline.womens import WomensPipelineConfig

        with pytest.raises(ValueError, match="seed_prior_weight"):
            WomensPipelineConfig(
                probability_profile="production",
                seed_prior_weight=0.1,
            )

    def test_locked_path_rejects_unsanctioned_complexity(self):
        from src.pipeline.config import SOTAPipelineConfig

        with pytest.raises(ValueError, match="model_complexity"):
            SOTAPipelineConfig(
                probability_profile="production",
                model_complexity="experimental",
            )

    def test_locked_path_requires_calibration_mode(self):
        from src.pipeline.config import SOTAPipelineConfig

        with pytest.raises(ValueError, match="mode=ev"):
            SOTAPipelineConfig(
                probability_profile="production",
                mode="ev",
            )

    def test_locked_path_requires_calibration_year_alignment(self):
        from src.pipeline.config import SOTAPipelineConfig

        with pytest.raises(ValueError, match="calibration_years"):
            SOTAPipelineConfig(
                probability_profile="production",
                calibration_years=[2024],
            )

    def test_resolve_calibration_years_defaults_to_holdout(self):
        from src.pipeline.config import SOTAPipelineConfig

        cfg = SOTAPipelineConfig()
        expected = [
            2008,
            2009,
            2010,
            2011,
            2012,
            2013,
            2014,
            2015,
            2016,
            2017,
            2018,
            2019,
            2021,
            2022,
            2023,
            2024,
            2025,
        ]
        assert cfg.resolve_calibration_years() == expected

    def test_invalid_profile_raises(self):
        from src.pipeline.config import SOTAPipelineConfig

        with pytest.raises(ValueError, match="Invalid probability_profile"):
            SOTAPipelineConfig(probability_profile="bogus")

    def test_experimental_profile_allows_all(self):
        from src.pipeline.config import SOTAPipelineConfig

        # Should not raise
        cfg = SOTAPipelineConfig(
            probability_profile="experimental",
            enable_seed_overrides=True,
            enable_brier_sharpening=True,
            enable_goto_conversion=True,
            enable_round_weighted_calibration=True,
            seed_prior_weight=0.2,
            consistency_bonus_max=0.02,
        )
        assert cfg.probability_profile == "experimental"


# ───────────────────────────────────────────────────────────────────────
# Task 1 & 2: men's/women's production path never calls banned components
# ───────────────────────────────────────────────────────────────────────


class TestMenProductionNoBannedComponents:
    def test_production_source_has_no_banned_imports(self):
        """Verify predict_probability_production source does not reference
        banned experimental components."""
        from src.pipeline.sota import SOTAPipeline

        source = inspect.getsource(SOTAPipeline.predict_probability_production)
        banned = [
            "_brier_post_processor",
            "_round_weighted_calibrator",
            "_flb_correction",
            "seed_prior",
            "consistency_bonus",
            "apply_experimental_",
            "BrierPostProcessor",
        ]
        for term in banned:
            assert term not in source, f"Production path references banned component: {term}"

    def test_production_uses_shared_helpers(self):
        """Verify predict_probability_production imports from probability_pipeline."""
        from src.pipeline.sota import SOTAPipeline

        source = inspect.getsource(SOTAPipeline.predict_probability_production)
        assert "probability_pipeline" in source
        assert "apply_calibration" in source
        assert "apply_tournament_shrinkage" in source
        assert "apply_final_clip" in source


class TestWomenProductionNoBannedComponents:
    def test_production_source_has_no_banned_calls(self):
        """Verify predict_probability_production does not call
        banned experimental components (ignoring docstrings/comments)."""
        from src.pipeline.womens import WomensPipeline

        source = inspect.getsource(WomensPipeline.predict_probability_production)
        # Strip docstrings and comments to avoid false positives
        code_lines = []
        in_docstring = False
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                # Single-line docstring
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    continue
                in_docstring = True
                continue
            if in_docstring:
                continue
            if stripped.startswith("#"):
                continue
            code_lines.append(line)
        code_only = "\n".join(code_lines)

        banned = [
            "post_processor",
            "seed_prior",
            "apply_experimental_",
        ]
        for term in banned:
            assert term not in code_only, f"Women's production path references banned component: {term}"

    def test_production_uses_shared_helpers(self):
        """Verify predict_probability_production imports from probability_pipeline."""
        from src.pipeline.womens import WomensPipeline

        source = inspect.getsource(WomensPipeline.predict_probability_production)
        assert "probability_pipeline" in source
        assert "apply_calibration" in source
        assert "apply_tournament_shrinkage" in source
        assert "apply_final_clip" in source


# ───────────────────────────────────────────────────────────────────────
# Task 7: audit code uses shared helpers
# ───────────────────────────────────────────────────────────────────────


class TestAuditUsesSharedHelpers:
    def test_tournament_adapt_imports_shared(self):
        """Verify rdof_audit._tournament_adapt uses probability_pipeline."""
        from src.ml.evaluation.rdof_audit import HoldoutEvaluator

        source = inspect.getsource(HoldoutEvaluator._tournament_adapt)
        assert "probability_pipeline" in source
        assert "apply_tournament_shrinkage" in source

    def test_apply_posthoc_imports_shared(self):
        """Verify _apply_posthoc_and_score uses probability_pipeline."""
        from src.ml.evaluation.rdof_audit import HoldoutEvaluator

        source = inspect.getsource(HoldoutEvaluator._apply_posthoc_and_score)
        assert "probability_pipeline" in source
        assert "apply_calibration" in source


# ───────────────────────────────────────────────────────────────────────
# Task 9: ablation ladder has max_movement
# ───────────────────────────────────────────────────────────────────────


class TestAblationLadderMaxMovement:
    def test_ablation_ladder_returns_max_movement(self):
        raw_probs = np.array([0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 1, 1, 1])
        shrinkage = 0.06

        result = run_ablation_ladder(
            raw_probs,
            outcomes,
            calibrator=None,
            shrinkage=shrinkage,
        )

        for stage_name, metrics in result.items():
            assert "max_movement" in metrics, f"Stage '{stage_name}' missing max_movement"
            assert "mean_movement" in metrics
            assert metrics["max_movement"] >= metrics["mean_movement"]

    def test_raw_stage_has_zero_movement(self):
        raw_probs = np.array([0.4, 0.6])
        outcomes = np.array([0, 1])
        result = run_ablation_ladder(
            raw_probs,
            outcomes,
            calibrator=None,
            shrinkage=0.05,
        )
        assert result["raw"]["max_movement"] == 0.0
        assert result["raw"]["mean_movement"] == 0.0

    def test_ablation_ladder_has_all_required_metrics(self):
        raw_probs = np.array([0.2, 0.8])
        outcomes = np.array([0, 1])
        result = run_ablation_ladder(
            raw_probs,
            outcomes,
            calibrator=None,
            shrinkage=0.06,
        )
        required_metrics = {"brier", "ece", "log_loss", "mean_movement", "max_movement"}
        for stage_name, metrics in result.items():
            missing = required_metrics - set(metrics.keys())
            assert not missing, f"Stage '{stage_name}' missing metrics: {missing}"
