"""Tests for parameter optimization workflow.

Covers:
  - LOCKED_FIELDS completeness
  - TUNABLE_PARAM_SPACE validation
  - Search space building and filtering
  - ParameterChange / ConfigDiffReport serialization
  - Config diff generation
  - Dry-run mode
  - Recommendation level thresholds
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

import pytest


class TestLockedFields:
    """Verify LOCKED_FIELDS covers critical production config keys."""

    def test_locked_fields_contains_identity_fields(self):
        from src.research.param_optimizer import LOCKED_FIELDS

        for f in ["year", "probability_profile", "mode", "model_complexity"]:
            assert f in LOCKED_FIELDS, f"Missing locked field: {f}"

    def test_locked_fields_contains_disabled_modules(self):
        from src.research.param_optimizer import LOCKED_FIELDS

        for f in ["enable_gnn", "enable_transformer", "enable_embedding_projections"]:
            assert f in LOCKED_FIELDS, f"Missing locked field: {f}"

    def test_locked_fields_contains_year_partitions(self):
        from src.research.param_optimizer import LOCKED_FIELDS

        for f in ["training_years", "dev_years", "holdout_years"]:
            assert f in LOCKED_FIELDS, f"Missing locked field: {f}"

    def test_locked_fields_contains_data_paths(self):
        from src.research.param_optimizer import LOCKED_FIELDS

        for f in ["teams_json", "torvik_json", "historical_games_json", "roster_json"]:
            assert f in LOCKED_FIELDS, f"Missing locked field: {f}"

    def test_locked_fields_contains_leakage_guards(self):
        from src.research.param_optimizer import LOCKED_FIELDS

        assert "strict_leakage_mode" in LOCKED_FIELDS
        assert "require_freeze_file" in LOCKED_FIELDS

    def test_locked_fields_is_frozenset(self):
        from src.research.param_optimizer import LOCKED_FIELDS

        assert isinstance(LOCKED_FIELDS, frozenset)

    def test_no_overlap_between_locked_and_tunable(self):
        from src.research.param_optimizer import LOCKED_FIELDS, TUNABLE_PARAM_SPACE

        overlap = set(TUNABLE_PARAM_SPACE.keys()) & LOCKED_FIELDS
        assert not overlap, f"Overlap between locked and tunable: {overlap}"


class TestTunableParamSpace:
    """Verify TUNABLE_PARAM_SPACE entries are well-formed."""

    def test_all_entries_have_required_keys(self):
        from src.research.param_optimizer import TUNABLE_PARAM_SPACE

        for name, spec in TUNABLE_PARAM_SPACE.items():
            assert "min" in spec, f"{name} missing 'min'"
            assert "max" in spec, f"{name} missing 'max'"
            assert "step" in spec, f"{name} missing 'step'"
            assert "description" in spec, f"{name} missing 'description'"

    def test_min_less_than_max(self):
        from src.research.param_optimizer import TUNABLE_PARAM_SPACE

        for name, spec in TUNABLE_PARAM_SPACE.items():
            assert spec["min"] < spec["max"], f"{name}: min >= max"

    def test_step_positive(self):
        from src.research.param_optimizer import TUNABLE_PARAM_SPACE

        for name, spec in TUNABLE_PARAM_SPACE.items():
            assert spec["step"] > 0, f"{name}: step <= 0"

    def test_params_exist_on_config(self):
        from src.pipeline.config import SOTAPipelineConfig
        from src.research.param_optimizer import TUNABLE_PARAM_SPACE

        config = SOTAPipelineConfig()
        for name in TUNABLE_PARAM_SPACE:
            assert hasattr(config, name), f"{name} not on SOTAPipelineConfig"

    def test_expected_params_present(self):
        from src.research.param_optimizer import TUNABLE_PARAM_SPACE

        expected = [
            "tournament_shrinkage", "massey_blend_weight", "massey_sigma",
            "seed_prior_weight", "mc_noise_std", "training_year_decay",
        ]
        for name in expected:
            assert name in TUNABLE_PARAM_SPACE, f"Missing expected param: {name}"


class TestBuildSearchSpace:
    """Test search space building and filtering."""

    def test_returns_all_tunable_by_default(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer, TUNABLE_PARAM_SPACE

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        space = optimizer.build_search_space()
        assert len(space) > 0
        assert len(space) <= len(TUNABLE_PARAM_SPACE)

    def test_filters_to_specified_params(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        space = optimizer.build_search_space(params=["tournament_shrinkage"])
        assert list(space.keys()) == ["tournament_shrinkage"]

    def test_rejects_locked_fields(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        space = optimizer.build_search_space(params=["year"])
        assert len(space) == 0

    def test_rejects_unknown_params(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        space = optimizer.build_search_space(params=["nonexistent_param_xyz"])
        assert len(space) == 0


class TestParameterChange:
    """Test ParameterChange dataclass."""

    def test_to_dict_roundtrip(self):
        from src.research.param_optimizer import ParameterChange

        change = ParameterChange(
            name="tournament_shrinkage",
            current_value=0.06,
            recommended_value=0.08,
            brier_impact=-0.001,
            p_value=0.03,
            confidence="high",
            rationale="Test rationale",
        )
        d = change.to_dict()
        assert d["name"] == "tournament_shrinkage"
        assert d["current_value"] == 0.06
        assert d["recommended_value"] == 0.08
        assert d["brier_impact"] == -0.001
        assert d["confidence"] == "high"


class TestConfigDiffReport:
    """Test ConfigDiffReport dataclass."""

    def test_to_json_valid(self):
        from src.research.param_optimizer import ConfigDiffReport

        report = ConfigDiffReport(
            production_config_path="configs/production_2026.json",
            baseline_loyo_brier=0.185,
            recommended_loyo_brier=0.180,
            brier_improvement=0.005,
            recommendation="APPLY",
        )
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["recommendation"] == "APPLY"
        assert parsed["baseline_loyo_brier"] == 0.185

    def test_save_creates_file(self, tmp_path):
        from src.research.param_optimizer import ConfigDiffReport

        report = ConfigDiffReport(recommendation="REVIEW")
        path = str(tmp_path / "report.json")
        report.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["recommendation"] == "REVIEW"

    def test_auto_timestamp(self):
        from src.research.param_optimizer import ConfigDiffReport

        report = ConfigDiffReport()
        assert report.timestamp  # Should be auto-populated

    def test_default_recommendation_is_reject(self):
        from src.research.param_optimizer import ConfigDiffReport

        report = ConfigDiffReport()
        assert report.recommendation == "REJECT"


class TestGenerateConfigDiff:
    """Test config diff generation logic."""

    def test_apply_recommendation(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer, ParameterChange

        optimizer = ParamOptimizer(production_config_path=str(config_path))

        change = ParameterChange(
            name="tournament_shrinkage",
            current_value=0.06,
            recommended_value=0.08,
            brier_impact=-0.003,
            p_value=0.02,
            confidence="high",
            rationale="test",
        )

        baseline_year = {2018: 0.20, 2019: 0.21, 2021: 0.19, 2022: 0.22, 2023: 0.20, 2024: 0.21}
        candidate_year = {2018: 0.17, 2019: 0.17, 2021: 0.16, 2022: 0.18, 2023: 0.16, 2024: 0.17}

        report = optimizer.generate_config_diff(
            baseline_brier=0.205,
            baseline_year_briers=baseline_year,
            combined_brier=0.168,
            combined_year_briers=candidate_year,
            changes=[change],
            wall_clock_seconds=100.0,
        )

        assert report.recommendation == "APPLY"
        assert report.brier_improvement > 0
        assert len(report.parameter_changes) == 1

    def test_reject_when_not_significant(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer, ParameterChange

        optimizer = ParamOptimizer(production_config_path=str(config_path))

        change = ParameterChange(
            name="mc_noise_std",
            current_value=0.16,
            recommended_value=0.17,
            brier_impact=-0.0001,
            p_value=0.5,
            confidence="low",
            rationale="test",
        )

        # Near-identical Brier scores with high variance across folds
        baseline_year = {2018: 0.20, 2019: 0.15, 2021: 0.22, 2022: 0.14}
        candidate_year = {2018: 0.19, 2019: 0.16, 2021: 0.23, 2022: 0.13}

        report = optimizer.generate_config_diff(
            baseline_brier=0.185,
            baseline_year_briers=baseline_year,
            combined_brier=0.184,
            combined_year_briers=candidate_year,
            changes=[change],
            wall_clock_seconds=50.0,
        )

        # With such small improvement and mixed folds, should not be APPLY
        assert report.recommendation in ("REVIEW", "REJECT")

    def test_per_year_breakdown_populated(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer, ParameterChange

        optimizer = ParamOptimizer(production_config_path=str(config_path))

        change = ParameterChange("x", 1, 2, -0.01, 0.01, "high", "test")
        baseline_year = {2018: 0.20, 2019: 0.19}
        candidate_year = {2018: 0.18, 2019: 0.17}

        report = optimizer.generate_config_diff(
            0.195, baseline_year, 0.175, candidate_year, [change], 10.0,
        )

        assert "2018" in report.per_year_briers
        assert "2019" in report.per_year_briers
        assert report.per_year_briers["2018"]["delta"] < 0


class TestDryRun:
    """Test dry-run mode."""

    def test_dry_run_returns_zero_evaluations(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        report = optimizer.run(dry_run=True)
        assert report.n_configs_evaluated == 0
        assert report.recommendation == "REJECT"

    def test_dry_run_with_specific_params(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        report = optimizer.run(dry_run=True, params=["tournament_shrinkage"])
        assert report.n_configs_evaluated == 0


class TestFoldPValue:
    """Test fold-level p-value computation."""

    def test_identical_folds_give_pvalue_one(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        baseline = {2018: 0.20, 2019: 0.19, 2021: 0.18}
        p = optimizer._compute_fold_p_value(baseline, baseline)
        assert p == 1.0

    def test_clearly_different_folds_give_low_pvalue(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        baseline = {2018: 0.20, 2019: 0.20, 2021: 0.20, 2022: 0.20, 2023: 0.20, 2024: 0.20}
        candidate = {2018: 0.15, 2019: 0.15, 2021: 0.15, 2022: 0.15, 2023: 0.15, 2024: 0.15}
        p = optimizer._compute_fold_p_value(baseline, candidate)
        assert p < 0.01

    def test_too_few_folds_returns_one(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        p = optimizer._compute_fold_p_value({2018: 0.2}, {2018: 0.1})
        assert p == 1.0


class TestCohensD:
    """Test Cohen's d computation."""

    def test_large_effect(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        baseline = {2018: 0.20, 2019: 0.20, 2021: 0.20}
        candidate = {2018: 0.15, 2019: 0.15, 2021: 0.15}
        d = optimizer._compute_cohens_d(baseline, candidate)
        # All deltas identical → std=0 → d=0 (degenerate case)
        # This is expected behavior
        assert d == 0.0

    def test_nonzero_effect(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        baseline = {2018: 0.20, 2019: 0.22, 2021: 0.18}
        candidate = {2018: 0.17, 2019: 0.19, 2021: 0.16}
        d = optimizer._compute_cohens_d(baseline, candidate)
        assert d > 0


class TestLoadProductionBaseline:
    """Test loading production config."""

    def test_loads_valid_config(self, tmp_path):
        config_path = _write_minimal_config(tmp_path)
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(config_path))
        config = optimizer.load_production_baseline()
        assert config.year == 2026
        assert config.tournament_shrinkage == 0.06

    def test_missing_config_raises(self, tmp_path):
        from src.research.param_optimizer import ParamOptimizer

        optimizer = ParamOptimizer(production_config_path=str(tmp_path / "nope.json"))
        with pytest.raises(FileNotFoundError):
            optimizer.load_production_baseline()


# ── Helpers ────────────────────────────────────────────────────────────────

def _write_minimal_config(tmp_path):
    """Write a minimal production config for testing."""
    config = {
        "year": 2026,
        "probability_profile": "production",
        "mode": "calibration",
        "model_complexity": "standard",
        "tournament_shrinkage": 0.06,
        "massey_blend_weight": 0.25,
        "massey_sigma": 4.5,
        "seed_prior_weight": 0.10,
        "consistency_bonus_max": 0.02,
        "mc_noise_std": 0.16,
        "num_simulations": 50000,
        "training_year_decay": 0.85,
        "training_year_min_weight": 0.15,
        "spread_sigma_init": 11.0,
        "random_seed": 2026,
    }
    path = tmp_path / "production_2026.json"
    path.write_text(json.dumps(config, indent=2))
    return path
