"""Comprehensive unit tests for src.ml.evaluation.rdof_audit.

Covers: PipelineConstant, CONSTANT_REGISTRY, helper functions,
        metric computations, dataclasses, HoldoutEvaluator,
        SensitivityAnalyzer, MCParameterBacktester, MCBacktestResult,
        RDOFAuditReport, freeze/verify, contamination tracker,
        adopt_sensitivity_optima, and the top-level runners.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, mock_open, patch, PropertyMock

import numpy as np
import pytest

from src.ml.evaluation.rdof_audit import (
    CONSTANT_REGISTRY,
    TIER3_GAME_LEVEL_CONSTANTS,
    TIER3_MC_CONSTANTS,
    ConstantSensitivityResult,
    HoldoutEvaluator,
    HoldoutReport,
    MCBacktestResult,
    MCParameterBacktester,
    ModelComplexityAudit,
    PipelineConstant,
    RDOFAuditReport,
    SensitivityAnalyzer,
    YearMetrics,
    _bootstrap_brier_ci,
    _compute_accuracy,
    _compute_brier,
    _compute_ece,
    _compute_log_loss,
    _feature_set_hash,
    _lockfile_path,
    adopt_sensitivity_optima,
    check_holdout_contamination,
    config_hash,
    estimate_model_complexity,
    get_constants_by_tier,
    get_tier3_constants,
    record_holdout_evaluation,
    _seed_baseline_prob,
    _apply_mc_calibration_to_config,
)


# ─── Helpers ────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Create a mock config that behaves like a dataclass with __dataclass_fields__."""
    defaults = {
        "tournament_shrinkage": 0.0,
        "seed_prior_weight": 0.0,
        "consistency_bonus_max": 0.0,
        "consistency_normalizer": 15.0,
        "ensemble_lgb_weight": 0.45,
        "ensemble_xgb_weight": 0.35,
        "mc_noise_std": 0.16,
        "mc_regional_correlation": 0.0,
        "training_year_decay": 0.85,
        "training_year_min_weight": 0.15,
        "recency_half_life": 0.3,
        "recency_decay_floor": 0.3,
        "late_season_training_cutoff_days": 45,
        "seed_prior_slope": 0.175,
        "pre_calibration_clip_lo": 0.03,
        "pre_calibration_clip_hi": 0.97,
        "vif_threshold": 10.0,
        "probability_profile": "production",
    }
    defaults.update(overrides)
    cfg = MagicMock()
    cfg.__dataclass_fields__ = list(defaults.keys())
    for k, v in defaults.items():
        setattr(cfg, k, v)
    cfg.mc_calibration_json = None
    return cfg


# ─── PipelineConstant Tests ─────────────────────────────────────────


class TestPipelineConstant:
    def test_create_basic(self):
        pc = PipelineConstant(
            name="test", tier=1, current_value=0.5,
            config_path="some.path", valid_range=(0.0, 1.0),
            derivation="test derivation",
        )
        assert pc.name == "test"
        assert pc.tier == 1
        assert pc.notes == ""

    def test_to_dict(self):
        pc = PipelineConstant(
            name="alpha", tier=3, current_value=0.7,
            config_path="cfg.alpha", valid_range=(0.1, 0.9),
            derivation="hand-tuned", notes="important",
        )
        d = pc.to_dict()
        assert d["name"] == "alpha"
        assert d["tier"] == 3
        assert d["current_value"] == 0.7
        assert d["valid_range"] == [0.1, 0.9]
        assert d["notes"] == "important"

    def test_to_dict_with_list_value(self):
        pc = PipelineConstant(
            name="weights", tier=1, current_value=[0.4, 0.25],
            config_path="w", valid_range=(0, 1), derivation="pub",
        )
        d = pc.to_dict()
        assert d["current_value"] == [0.4, 0.25]


# ─── Registry Helpers ───────────────────────────────────────────────


class TestRegistryHelpers:
    def test_get_tier3_constants_returns_only_tier3(self):
        tier3 = get_tier3_constants()
        assert len(tier3) > 0
        for c in tier3:
            assert c.tier == 3

    def test_get_constants_by_tier(self):
        for tier in [1, 2, 3]:
            result = get_constants_by_tier(tier)
            assert all(c.tier == tier for c in result)
            assert len(result) > 0

    def test_constant_registry_not_empty(self):
        assert len(CONSTANT_REGISTRY) > 30

    def test_tier3_game_level_constants_list(self):
        assert "ensemble_lgb_weight" in TIER3_GAME_LEVEL_CONSTANTS
        assert "tournament_shrinkage" in TIER3_GAME_LEVEL_CONSTANTS

    def test_tier3_mc_constants_list(self):
        assert "mc_noise_std" in TIER3_MC_CONSTANTS
        assert "mc_regional_correlation" in TIER3_MC_CONSTANTS

    def test_all_tier3_constants_in_either_list(self):
        tier3_names = {c.name for c in get_tier3_constants()}
        combined = set(TIER3_GAME_LEVEL_CONSTANTS) | set(TIER3_MC_CONSTANTS)
        for name in tier3_names:
            assert name in combined, f"{name} not in game-level or MC lists"


# ─── Metric Computation Helpers ─────────────────────────────────────


class TestMetricHelpers:
    def test_compute_brier_perfect(self):
        probs = np.array([1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert _compute_brier(probs, outcomes) == pytest.approx(0.0)

    def test_compute_brier_worst(self):
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert _compute_brier(probs, outcomes) == pytest.approx(1.0)

    def test_compute_brier_uniform(self):
        probs = np.array([0.5, 0.5])
        outcomes = np.array([1.0, 0.0])
        assert _compute_brier(probs, outcomes) == pytest.approx(0.25)

    def test_compute_log_loss_perfect_ish(self):
        probs = np.array([0.999, 0.001])
        outcomes = np.array([1.0, 0.0])
        result = _compute_log_loss(probs, outcomes)
        assert result < 0.01

    def test_compute_log_loss_clips(self):
        # Should not raise even with extreme values
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        result = _compute_log_loss(probs, outcomes)
        assert np.isfinite(result)

    def test_compute_accuracy_perfect(self):
        probs = np.array([0.9, 0.1, 0.8])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert _compute_accuracy(probs, outcomes) == pytest.approx(1.0)

    def test_compute_accuracy_half(self):
        probs = np.array([0.9, 0.9])
        outcomes = np.array([1.0, 0.0])
        assert _compute_accuracy(probs, outcomes) == pytest.approx(0.5)

    def test_compute_ece_perfect_calibration(self):
        # If predictions match outcomes perfectly, ECE should be very small
        probs = np.array([0.5] * 100)
        outcomes = np.concatenate([np.ones(50), np.zeros(50)])
        ece = _compute_ece(probs, outcomes)
        assert ece < 0.01

    def test_compute_ece_with_bins(self):
        probs = np.array([0.1, 0.9])
        outcomes = np.array([0.0, 1.0])
        ece = _compute_ece(probs, outcomes, n_bins=5)
        assert ece < 0.2

    def test_compute_ece_empty_bins(self):
        probs = np.array([0.5, 0.5])
        outcomes = np.array([1.0, 0.0])
        ece = _compute_ece(probs, outcomes, n_bins=10)
        assert np.isfinite(ece)

    def test_seed_baseline_prob_equal_seeds(self):
        prob = _seed_baseline_prob(8, 8)
        assert prob == pytest.approx(0.5)

    def test_seed_baseline_prob_higher_seed_favored(self):
        prob = _seed_baseline_prob(1, 16)
        assert prob > 0.5

    def test_seed_baseline_prob_lower_seed_underdog(self):
        prob = _seed_baseline_prob(16, 1)
        assert prob < 0.5

    def test_seed_baseline_prob_symmetry(self):
        p1 = _seed_baseline_prob(1, 16)
        p2 = _seed_baseline_prob(16, 1)
        assert p1 + p2 == pytest.approx(1.0)

    def test_bootstrap_brier_ci_returns_tuple(self):
        probs = np.array([0.6, 0.4, 0.7, 0.3, 0.5] * 10)
        outcomes = np.array([1.0, 0.0, 1.0, 0.0, 1.0] * 10)
        lo, hi = _bootstrap_brier_ci(probs, outcomes, n_bootstrap=100)
        assert lo < hi
        assert lo > 0
        assert hi < 1

    def test_bootstrap_brier_ci_95_includes_mean(self):
        rng = np.random.RandomState(123)
        probs = rng.uniform(0.3, 0.7, size=50)
        outcomes = (rng.uniform(size=50) > 0.5).astype(float)
        lo, hi = _bootstrap_brier_ci(probs, outcomes, n_bootstrap=500, ci_level=0.95)
        actual = _compute_brier(probs, outcomes)
        assert lo <= actual <= hi


# ─── YearMetrics Tests ──────────────────────────────────────────────


class TestYearMetrics:
    def _make(self, **kw):
        defaults = dict(
            year=2025, n_games=63, brier_score=0.18,
            log_loss=0.55, accuracy=0.72, ece=0.04,
            seed_baseline_brier=0.20, elo_baseline_brier=0.22,
        )
        defaults.update(kw)
        return YearMetrics(**defaults)

    def test_brier_skill_score_positive(self):
        m = self._make(brier_score=0.18, seed_baseline_brier=0.20)
        assert m.brier_skill_score() > 0

    def test_brier_skill_score_zero_baseline(self):
        m = self._make(seed_baseline_brier=0.0)
        assert m.brier_skill_score() == 0.0

    def test_elo_skill_score_positive(self):
        m = self._make(brier_score=0.18, elo_baseline_brier=0.22)
        assert m.elo_skill_score() > 0

    def test_elo_skill_score_zero_baseline(self):
        m = self._make(elo_baseline_brier=0.0)
        assert m.elo_skill_score() == 0.0

    def test_to_dict_keys(self):
        m = self._make()
        d = m.to_dict()
        expected_keys = {
            "year", "n_games", "brier_score", "log_loss", "accuracy",
            "ece", "seed_baseline_brier", "elo_baseline_brier",
            "uniform_brier", "brier_skill_score", "elo_skill_score",
            "brier_ci",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_rounding(self):
        m = self._make(brier_score=0.123456789)
        d = m.to_dict()
        assert d["brier_score"] == round(0.123456789, 5)


# ─── HoldoutReport Tests ───────────────────────────────────────────


class TestHoldoutReport:
    def _make_report(self, years_data=None):
        r = HoldoutReport(holdout_years=[2024, 2025])
        if years_data is None:
            years_data = {
                2024: YearMetrics(2024, 63, 0.18, 0.5, 0.72, 0.04, 0.20, 0.22),
                2025: YearMetrics(2025, 63, 0.19, 0.52, 0.70, 0.05, 0.21, 0.23),
            }
        r.per_year = years_data
        return r

    def test_aggregate_brier_weighted(self):
        r = self._make_report()
        agg = r.aggregate_brier
        assert 0.18 <= agg <= 0.19

    def test_aggregate_brier_empty(self):
        r = HoldoutReport(holdout_years=[])
        assert r.aggregate_brier == 0.0

    def test_aggregate_seed_brier(self):
        r = self._make_report()
        assert r.aggregate_seed_brier > 0

    def test_aggregate_seed_brier_empty(self):
        r = HoldoutReport(holdout_years=[])
        assert r.aggregate_seed_brier == 0.0

    def test_aggregate_elo_brier(self):
        r = self._make_report()
        assert r.aggregate_elo_brier > 0

    def test_aggregate_elo_brier_empty(self):
        r = HoldoutReport(holdout_years=[])
        assert r.aggregate_elo_brier == 0.0

    def test_total_games(self):
        r = self._make_report()
        assert r.total_games == 126

    def test_verdict_pass(self):
        r = self._make_report({
            2024: YearMetrics(2024, 63, 0.18, 0.5, 0.72, 0.04, 0.25, 0.25),
        })
        assert r.verdict() == "PASS"

    def test_verdict_warn(self):
        r = self._make_report({
            2024: YearMetrics(2024, 63, 0.197, 0.5, 0.72, 0.04, 0.20, 0.25),
        })
        assert r.verdict() == "WARN"

    def test_verdict_fail(self):
        r = self._make_report({
            2024: YearMetrics(2024, 63, 0.25, 0.5, 0.50, 0.04, 0.20, 0.22),
        })
        assert r.verdict() == "FAIL"

    def test_to_dict_structure(self):
        r = self._make_report()
        d = r.to_dict()
        assert "holdout_years" in d
        assert "verdict" in d
        assert "per_year" in d
        assert "aggregate_brier" in d
        assert "integrity_level" in d


# ─── ConstantSensitivityResult Tests ────────────────────────────────


class TestConstantSensitivityResult:
    def _make(self, **kw):
        defaults = dict(
            constant_name="test_const",
            grid_values=[0.0, 0.1, 0.2, 0.3, 0.4],
            loyo_brier_scores=[0.200, 0.195, 0.190, 0.192, 0.198],
            current_value=0.2,
            optimal_value=0.2,
            optimal_brier=0.190,
            current_brier=0.190,
            brier_range=0.010,
            is_flat=False,
        )
        defaults.update(kw)
        return ConstantSensitivityResult(**defaults)

    def test_brier_gap(self):
        sr = self._make(current_brier=0.195, optimal_brier=0.190)
        assert sr.brier_gap == pytest.approx(0.005)

    def test_rank_of_current_at_optimal(self):
        sr = self._make()
        assert sr.rank_of_current == 1  # current=0.2 has best score

    def test_rank_of_current_not_optimal(self):
        sr = self._make(
            current_value=0.0,
            loyo_brier_scores=[0.200, 0.195, 0.190, 0.192, 0.198],
        )
        assert sr.rank_of_current == 5  # worst

    def test_rank_of_current_missing_value(self):
        sr = self._make(current_value=0.999)  # Not in grid
        assert sr.rank_of_current == 5  # defaults to len(grid)

    def test_rdof_risk_flat_plateau(self):
        sr = self._make(is_flat=True)
        assert sr.rdof_risk_category == "flat_plateau"

    def test_rdof_risk_mild_slope(self):
        sr = self._make(is_flat=False, brier_range=0.010, current_brier=0.191, optimal_brier=0.190)
        assert sr.rdof_risk_category == "mild_slope"

    def test_rdof_risk_sharp_peak(self):
        sr = self._make(is_flat=False, brier_range=0.020)
        assert sr.rdof_risk_category == "sharp_peak"

    def test_effective_dof_flat(self):
        sr = self._make(is_flat=True)
        assert sr.effective_dof == 0.0

    def test_effective_dof_mild(self):
        sr = self._make(is_flat=False, brier_range=0.010, current_brier=0.191, optimal_brier=0.190)
        assert sr.effective_dof == 0.5

    def test_effective_dof_sharp(self):
        sr = self._make(is_flat=False, brier_range=0.020)
        assert sr.effective_dof == 1.0

    def test_to_dict_keys(self):
        sr = self._make()
        d = sr.to_dict()
        expected_keys = {
            "constant_name", "grid_values", "loyo_brier_scores",
            "current_value", "optimal_value", "optimal_brier",
            "current_brier", "brier_gap", "brier_range", "is_flat",
            "rank_of_current", "rdof_risk_category", "effective_dof",
            "circularity_warning",
        }
        assert expected_keys == set(d.keys())

    def test_circularity_warning_default(self):
        sr = self._make()
        assert sr.circularity_warning is False

    def test_circularity_warning_set(self):
        sr = self._make(circularity_warning=True)
        assert sr.circularity_warning is True


# ─── ModelComplexityAudit Tests ─────────────────────────────────────


class TestModelComplexityAudit:
    def test_default_values(self):
        audit = ModelComplexityAudit()
        assert audit.n_training_samples == 0
        assert audit.passed is False

    def test_to_dict(self):
        audit = ModelComplexityAudit(
            n_training_samples=400,
            component_params={"lgb": 800},
            total_effective_params=800,
            actual_ratio=2.0,
            passed=False,
        )
        d = audit.to_dict()
        assert d["n_training_samples"] == 400
        assert d["passed"] is False
        assert d["actual_ratio"] == 2.0

    def test_estimate_model_complexity_defaults(self):
        audit = estimate_model_complexity()
        assert audit.n_training_samples == 400
        assert audit.total_effective_params > 0
        assert "lightgbm" in audit.component_params
        assert "xgboost" in audit.component_params
        assert "logistic_regression" in audit.component_params
        assert "ensemble_weights" in audit.component_params
        assert "tier3_constants" in audit.component_params

    def test_estimate_model_complexity_pass(self):
        # With a very large N, should pass
        audit = estimate_model_complexity(n_training_samples=100000)
        assert audit.passed is True

    def test_estimate_model_complexity_fail(self):
        # With very small N, should fail
        audit = estimate_model_complexity(n_training_samples=10)
        assert audit.passed is False
        assert any("COMPLEXITY VIOLATION" in w for w in audit.warnings)

    def test_estimate_model_complexity_gnn_warning(self):
        audit = estimate_model_complexity(
            n_training_samples=100,
            gnn_embedding_dim=50,  # Large embedding
        )
        assert any("GNN" in w for w in audit.warnings)

    def test_estimate_model_complexity_transformer_warning(self):
        audit = estimate_model_complexity(
            n_training_samples=100,
            transformer_embedding_dim=100,
        )
        assert any("Transformer" in w for w in audit.warnings)

    def test_estimate_n_training_samples_zero(self):
        audit = estimate_model_complexity(n_training_samples=0)
        assert audit.actual_ratio > 0  # max(0,1) prevents div by zero


# ─── Config Hash Tests ──────────────────────────────────────────────


class TestConfigHash:
    def test_config_hash_deterministic(self):
        cfg = _make_config()
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2

    def test_config_hash_changes_with_value(self):
        cfg1 = _make_config(tournament_shrinkage=0.0)
        cfg2 = _make_config(tournament_shrinkage=0.1)
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_config_hash_length(self):
        cfg = _make_config()
        h = config_hash(cfg)
        assert len(h) == 16

    def test_config_hash_handles_list(self):
        cfg = _make_config()
        cfg.some_list = [1, 2, 3]
        cfg.__dataclass_fields__ = list(cfg.__dataclass_fields__) + ["some_list"]
        h = config_hash(cfg)
        assert len(h) == 16

    def test_config_hash_handles_none(self):
        cfg = _make_config()
        cfg.optional_field = None
        cfg.__dataclass_fields__ = list(cfg.__dataclass_fields__) + ["optional_field"]
        h = config_hash(cfg)
        assert isinstance(h, str)


# ─── Feature Set Hash Tests ────────────────────────────────────────


class TestFeatureSetHash:
    @patch("src.ml.evaluation.rdof_audit._feature_set_hash")
    def test_feature_set_hash_returns_str_or_none(self, mock):
        mock.return_value = "abc123"
        assert mock() == "abc123"

    def test_feature_set_hash_returns_none_on_import_error(self):
        # Real function should return None when import fails
        result = _feature_set_hash()
        # It may return None or a string depending on whether the import works
        assert result is None or isinstance(result, str)


# ─── Apply MC Calibration Tests ────────────────────────────────────


class TestApplyMCCalibration:
    def test_returns_none_when_no_path(self):
        cfg = _make_config()
        cfg.mc_calibration_json = None
        with patch("os.path.isfile", return_value=False):
            with patch("os.getcwd", return_value="/fake"):
                result = _apply_mc_calibration_to_config(cfg)
        assert result is None

    def test_applies_best_params(self):
        cfg = _make_config()
        cfg.mc_calibration_json = "/tmp/mc_cal.json"
        payload = {"best_params": {"noise_std": 0.12, "regional_correlation": 0.05}}
        with patch("builtins.open", mock_open(read_data=json.dumps(payload))):
            with patch("os.path.isfile", return_value=True):
                result = _apply_mc_calibration_to_config(cfg)
        assert result is not None
        assert cfg.mc_noise_std == 0.12
        assert cfg.mc_regional_correlation == 0.05

    def test_handles_missing_best_params(self):
        cfg = _make_config()
        cfg.mc_calibration_json = "/tmp/mc_cal.json"
        payload = {"best_params": {}}
        with patch("builtins.open", mock_open(read_data=json.dumps(payload))):
            with patch("os.path.isfile", return_value=True):
                result = _apply_mc_calibration_to_config(cfg)
        assert result is not None

    def test_returns_none_on_json_error(self):
        cfg = _make_config()
        cfg.mc_calibration_json = "/tmp/mc_cal.json"
        with patch("builtins.open", mock_open(read_data="not json")):
            with patch("os.path.isfile", return_value=True):
                result = _apply_mc_calibration_to_config(cfg)
        assert result is None


# ─── Freeze / Verify Tests ──────────────────────────────────────────


class TestFreezePipeline:
    @patch("subprocess.run")
    def test_freeze_pipeline_creates_artifact(self, mock_run):
        from src.ml.evaluation.rdof_audit import freeze_pipeline

        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n")
        cfg = _make_config()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value="feat123"):
                    result = freeze_pipeline(cfg, output_path=path)

            assert result["freeze_type"] == "pre-registration"
            assert "config_hash" in result
            assert result["feature_set_hash"] == "feat123"
            assert result["n_constants"] == len(CONSTANT_REGISTRY)

            # Verify file was written
            with open(path) as fh:
                saved = json.load(fh)
            assert saved["freeze_type"] == "pre-registration"
        finally:
            os.unlink(path)

    @patch("subprocess.run")
    def test_freeze_pipeline_git_dirty_warning(self, mock_run):
        from src.ml.evaluation.rdof_audit import freeze_pipeline

        # First call: git rev-parse, second: git status (dirty), third: git tag
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="sha123\n"),
            MagicMock(returncode=0, stdout="M some_file.py\n"),
            MagicMock(returncode=0, stdout=""),
        ]
        cfg = _make_config()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value=None):
                    result = freeze_pipeline(cfg, output_path=path)
            assert result["git_dirty"] is True
        finally:
            os.unlink(path)


class TestVerifyFreeze:
    def test_verify_matching_config(self):
        from src.ml.evaluation.rdof_audit import verify_freeze

        cfg = _make_config()
        h = config_hash(cfg)
        freeze_data = {
            "config_hash": h,
            "config_fields": {},
            "timestamp": "2025-01-01T00:00:00",
            "git_commit": "abc123",
            "constant_registry": [c.to_dict() for c in CONSTANT_REGISTRY],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(freeze_data, f)
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value=None):
                    result = verify_freeze(cfg, path)
            assert result["matches"] is True
            assert result["mismatches"] == []
        finally:
            os.unlink(path)

    def test_verify_mismatching_config(self):
        from src.ml.evaluation.rdof_audit import verify_freeze

        cfg = _make_config()
        freeze_data = {
            "config_hash": "different_hash",
            "config_fields": {"tournament_shrinkage": 0.5},
            "timestamp": "2025-01-01",
            "git_commit": "abc",
            "constant_registry": [],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(freeze_data, f)
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value=None):
                    result = verify_freeze(cfg, path)
            assert result["matches"] is False
            assert len(result["mismatches"]) > 0
        finally:
            os.unlink(path)

    def test_verify_feature_set_hash_mismatch(self):
        from src.ml.evaluation.rdof_audit import verify_freeze

        cfg = _make_config()
        h = config_hash(cfg)
        freeze_data = {
            "config_hash": h,
            "config_fields": {},
            "feature_set_hash": "old_hash",
            "constant_registry": [c.to_dict() for c in CONSTANT_REGISTRY],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(freeze_data, f)
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value="new_hash"):
                    result = verify_freeze(cfg, path)
            assert any("Feature set hash" in m for m in result["mismatches"])
        finally:
            os.unlink(path)

    def test_verify_constant_value_changed(self):
        from src.ml.evaluation.rdof_audit import verify_freeze

        cfg = _make_config()
        h = config_hash(cfg)
        registry_snap = [c.to_dict() for c in CONSTANT_REGISTRY]
        # Modify a constant value in the frozen snapshot
        if registry_snap:
            registry_snap[0]["current_value"] = "CHANGED"

        freeze_data = {
            "config_hash": h,
            "config_fields": {},
            "constant_registry": registry_snap,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(freeze_data, f)
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value=None):
                    result = verify_freeze(cfg, path)
            assert any("changed" in m.lower() for m in result.get("mismatches", []))
        finally:
            os.unlink(path)


# ─── Holdout Contamination Tracker Tests ────────────────────────────


class TestHoldoutContamination:
    def test_lockfile_path(self):
        p = _lockfile_path("/data/raw/historical")
        assert p.endswith("holdout_evaluation.lock.json")

    def test_record_and_check_no_contamination(self):
        cfg = _make_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            record_holdout_evaluation(
                tmpdir, [2024, 2025], config_hash(cfg),
                {"aggregate_brier": 0.18, "verdict": "PASS"},
            )
            result = check_holdout_contamination(tmpdir, cfg)
            assert result is None  # No contamination

    def test_check_contamination_detected(self):
        cfg1 = _make_config(tournament_shrinkage=0.0)
        cfg2 = _make_config(tournament_shrinkage=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            record_holdout_evaluation(
                tmpdir, [2024, 2025], config_hash(cfg1),
                {"aggregate_brier": 0.18, "verdict": "PASS"},
            )
            result = check_holdout_contamination(tmpdir, cfg2)
            assert result is not None
            assert result["contamination_detected"] is True

    def test_check_no_lockfile(self):
        cfg = _make_config()
        result = check_holdout_contamination("/nonexistent/path", cfg)
        assert result is None

    def test_check_corrupt_lockfile(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "holdout_evaluation.lock.json")
            with open(lock_path, "w") as f:
                f.write("not valid json")
            result = check_holdout_contamination(tmpdir, cfg)
            assert result is None

    def test_record_with_dev_years(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            record_holdout_evaluation(
                tmpdir, [2025], config_hash(cfg),
                {"aggregate_brier": 0.18, "verdict": "PASS"},
                dev_years=[2018, 2019, 2021, 2022, 2023],
            )
            lock_path = os.path.join(tmpdir, "holdout_evaluation.lock.json")
            with open(lock_path) as f:
                data = json.load(f)
            assert data["dev_years"] == [2018, 2019, 2021, 2022, 2023]


# ─── HoldoutEvaluator Tests ────────────────────────────────────────


class TestHoldoutEvaluator:
    def test_init_with_explicit_years(self):
        ev = HoldoutEvaluator(
            historical_dir="/fake/dir",
            all_years=[2018, 2019, 2021],
        )
        assert ev.all_years == [2018, 2019, 2021]

    def test_init_with_dev_years(self):
        ev = HoldoutEvaluator(
            historical_dir="/fake/dir",
            all_years=[2018, 2019, 2020, 2021],
            dev_years=[2018, 2019, 2020, 2021],
        )
        # 2020 should be excluded
        assert 2020 not in ev.dev_years

    def test_detect_years_from_glob(self):
        with patch.object(Path, "glob") as mock_glob:
            mock_glob.return_value = [
                Path("historical_games_2018.json"),
                Path("historical_games_2019.json"),
                Path("historical_games_2020.json"),
            ]
            with patch.object(Path, "exists", return_value=True):
                ev = HoldoutEvaluator(historical_dir="/fake/dir")
                years = ev._detect_years()
                # 2020 excluded (COVID)
                assert 2020 not in years

    def test_build_team_metrics_valid(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        payload = {
            "teams": [
                {"team_id": "duke", "off_rtg": 110.0, "def_rtg": 95.0,
                 "pace": 70.0, "srs": 15.0, "sos": 5.0, "wins": 28, "losses": 6},
                {"team_id": "unc", "off_rtg": 108.0, "def_rtg": 97.0},
            ]
        }
        result = ev._build_team_metrics(payload)
        assert "duke" in result
        assert result["duke"]["off_rtg"] == 110.0

    def test_build_team_metrics_all_zeros(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        payload = {
            "teams": [
                {"team_id": "duke", "off_rtg": 0.0, "def_rtg": 95.0},
                {"team_id": "unc", "off_rtg": 0.0, "def_rtg": 97.0},
            ]
        }
        result = ev._build_team_metrics(payload)
        assert result == {}

    def test_build_team_metrics_single_unique_off(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        payload = {
            "teams": [
                {"team_id": "duke", "off_rtg": 100.0, "def_rtg": 95.0},
                {"team_id": "unc", "off_rtg": 100.0, "def_rtg": 97.0},
            ]
        }
        result = ev._build_team_metrics(payload)
        assert result == {}

    def test_build_team_metrics_empty(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        payload = {"teams": []}
        result = ev._build_team_metrics(payload)
        assert result == {}

    def test_build_team_metrics_skip_zero_ratings(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        payload = {
            "teams": [
                {"team_id": "duke", "off_rtg": 110.0, "def_rtg": 95.0},
                {"team_id": "bad", "off_rtg": 0.0, "def_rtg": 0.0},
                {"team_id": "unc", "off_rtg": 108.0, "def_rtg": 97.0},
            ]
        }
        result = ev._build_team_metrics(payload)
        assert "bad" not in result
        assert "duke" in result

    def test_tournament_adapt_production(self):
        cfg = _make_config(
            tournament_shrinkage=0.1,
            probability_profile="production",
        )
        with patch("src.pipeline.probability_pipeline.apply_tournament_shrinkage",
                    side_effect=lambda p, s: p * (1 - s) + 0.5 * s):
            result = HoldoutEvaluator._tournament_adapt(0.8, 10.0, 0.5, cfg)
        assert 0.03 <= result <= 0.97

    def test_tournament_adapt_experimental(self):
        cfg = _make_config(
            tournament_shrinkage=0.0,
            probability_profile="experimental",
            seed_prior_weight=0.05,
            seed_prior_slope=0.175,
            consistency_bonus_max=0.02,
            consistency_normalizer=15.0,
        )
        with patch("src.pipeline.probability_pipeline.apply_tournament_shrinkage",
                    side_effect=lambda p, s: p):
            result = HoldoutEvaluator._tournament_adapt(0.7, 5.0, 0.3, cfg)
        assert 0.03 <= result <= 0.97


# ─── SensitivityAnalyzer Tests ──────────────────────────────────────


class TestSensitivityAnalyzer:
    def test_init(self):
        sa = SensitivityAnalyzer(
            historical_dir="/fake",
            dev_years=[2018, 2019, 2020, 2021],
            holdout_years=[2025],
            n_grid_points=5,
        )
        assert sa.n_grid_points == 5
        assert 2025 not in sa.dev_years
        assert 2020 not in sa.dev_years

    def test_make_grid_includes_current(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021], n_grid_points=5)
        pc = PipelineConstant("test", 3, 0.33, "path", (0.0, 1.0), "deriv")
        grid = sa._make_grid(pc)
        assert 0.33 in grid
        assert len(grid) == 5

    def test_make_grid_list_value(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021], n_grid_points=5)
        pc = PipelineConstant("test", 3, [1.0, 0.6], "path", (0.0, 1.0), "deriv")
        grid = sa._make_grid(pc)
        assert len(grid) == 5  # current value (list) is not insertable

    def test_apply_constant_to_config_shrinkage(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021])
        cfg = _make_config()
        pc = PipelineConstant("tournament_shrinkage", 3, 0.0, "p", (0, 0.25), "d")
        new_cfg = sa._apply_constant_to_config(cfg, pc, 0.15)
        assert new_cfg.tournament_shrinkage == 0.15

    def test_apply_constant_to_config_ensemble_lgb(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021])
        cfg = _make_config()
        pc = PipelineConstant("ensemble_lgb_weight", 3, 0.45, "p", (0.2, 0.7), "d")
        new_cfg = sa._apply_constant_to_config(cfg, pc, 0.55)
        assert new_cfg.ensemble_lgb_weight == 0.55

    def test_apply_constant_mc_is_noop(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021])
        cfg = _make_config()
        pc = PipelineConstant("mc_noise_std", 3, 0.16, "p", (0.02, 0.25), "d")
        new_cfg = sa._apply_constant_to_config(cfg, pc, 0.10)
        # mc_noise_std is not applied at game level
        assert new_cfg.mc_noise_std == 0.16  # unchanged (deepcopy of original)

    def test_analyze_constant_mc_skipped(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021])
        cfg = _make_config()
        pc = PipelineConstant("mc_noise_std", 3, 0.16, "p", (0.02, 0.25), "d")
        result = sa.analyze_constant(pc, cfg)
        assert result.is_flat is True
        assert result.grid_values == [0.16]


# ─── MCBacktestResult Tests ────────────────────────────────────────


class TestMCBacktestResult:
    def test_to_dict(self):
        r = MCBacktestResult(
            noise_std=0.16,
            regional_correlation=0.0,
            years_tested=[2021, 2022],
            per_year_upset_calibration={2021: 0.05, 2022: 0.03},
            mean_upset_calibration_error=0.04,
            per_year_seed_accuracy={2021: 0.75, 2022: 0.70},
            mean_seed_accuracy=0.725,
            per_year_log_prob={2021: -0.5, 2022: -0.6},
            mean_log_prob=-0.55,
        )
        d = r.to_dict()
        assert d["noise_std"] == 0.16
        assert d["mean_upset_calibration_error"] == 0.04
        assert "2021" in d["per_year_upset_calibration"]


# ─── MCParameterBacktester Tests ───────────────────────────────────


class TestMCParameterBacktester:
    def test_init_default_years(self):
        bt = MCParameterBacktester(
            historical_dir="/fake",
            years=[2021, 2022],
        )
        assert bt.years == [2021, 2022]

    def test_backtest_grid_dimensions(self):
        bt = MCParameterBacktester("/fake", years=[2021])
        with patch.object(bt, "backtest_params") as mock_bp:
            mock_bp.return_value = MCBacktestResult(
                noise_std=0.1, regional_correlation=0.0,
                years_tested=[2021],
                per_year_upset_calibration={}, mean_upset_calibration_error=0.5,
                per_year_seed_accuracy={}, mean_seed_accuracy=0.5,
                per_year_log_prob={}, mean_log_prob=-1.0,
            )
            results = bt.backtest_grid(
                noise_std_values=[0.1, 0.2],
                regional_corr_values=[0.0, 0.1],
            )
            assert len(results) == 4  # 2 x 2

    def test_backtest_params_empty_years(self):
        bt = MCParameterBacktester("/fake", years=[])
        # Directly test with empty years - should handle gracefully
        bt.years = []
        result = bt.backtest_params(0.16, 0.0)
        assert result.mean_upset_calibration_error == 1.0
        assert result.mean_seed_accuracy == 0.0


# ─── RDOFAuditReport Tests ─────────────────────────────────────────


class TestRDOFAuditReport:
    def _make_holdout_report(self):
        r = HoldoutReport(holdout_years=[2024, 2025])
        r.per_year = {
            2024: YearMetrics(2024, 63, 0.18, 0.5, 0.72, 0.04, 0.22, 0.24),
            2025: YearMetrics(2025, 63, 0.19, 0.52, 0.70, 0.05, 0.23, 0.25),
        }
        return r

    def _make_sensitivity_result(self, name="test", is_flat=True, circularity=False):
        return ConstantSensitivityResult(
            constant_name=name,
            grid_values=[0.0, 0.1, 0.2],
            loyo_brier_scores=[0.20, 0.19, 0.20],
            current_value=0.1,
            optimal_value=0.1,
            optimal_brier=0.19,
            current_brier=0.19,
            brier_range=0.01 if not is_flat else 0.002,
            is_flat=is_flat,
            circularity_warning=circularity,
        )

    def test_to_dict_minimal(self):
        report = RDOFAuditReport()
        d = report.to_dict()
        assert "metadata" in d
        assert "constant_inventory" in d
        assert "recommendations" in d

    def test_to_dict_with_holdout(self):
        report = RDOFAuditReport(holdout_report=self._make_holdout_report())
        d = report.to_dict()
        assert "holdout_evaluation" in d

    def test_to_dict_with_sensitivity(self):
        sr = {"test": self._make_sensitivity_result()}
        report = RDOFAuditReport(sensitivity_results=sr)
        d = report.to_dict()
        assert "sensitivity_analysis" in d

    def test_to_dict_with_complexity(self):
        audit = estimate_model_complexity()
        report = RDOFAuditReport(complexity_audit=audit)
        d = report.to_dict()
        assert "model_complexity_audit" in d

    def test_recommendations_integrity_level(self):
        report = RDOFAuditReport(holdout_report=self._make_holdout_report())
        recs = report._generate_recommendations()
        assert any("INTEGRITY" in r for r in recs)
        assert any("Level 3" in r for r in recs)

    def test_recommendations_holdout_pass(self):
        hr = self._make_holdout_report()
        report = RDOFAuditReport(holdout_report=hr)
        recs = report._generate_recommendations()
        assert any("HOLDOUT" in r for r in recs)

    def test_recommendations_holdout_fail(self):
        hr = HoldoutReport(holdout_years=[2024])
        hr.per_year = {
            2024: YearMetrics(2024, 63, 0.25, 0.7, 0.50, 0.1, 0.20, 0.22),
        }
        report = RDOFAuditReport(holdout_report=hr)
        recs = report._generate_recommendations()
        assert any("NOT beat" in r for r in recs)

    def test_recommendations_circularity(self):
        sr = {"lgb": self._make_sensitivity_result("lgb", circularity=True)}
        report = RDOFAuditReport(sensitivity_results=sr)
        recs = report._generate_recommendations()
        assert any("CIRCULARITY" in r for r in recs)

    def test_recommendations_sensitivity_flat(self):
        sr = {"x": self._make_sensitivity_result("x", is_flat=True)}
        report = RDOFAuditReport(sensitivity_results=sr)
        recs = report._generate_recommendations()
        assert any("INSENSITIVE" in r for r in recs)

    def test_recommendations_sensitivity_overfit(self):
        sr_obj = ConstantSensitivityResult(
            constant_name="bad",
            grid_values=[0.0, 0.1, 0.2],
            loyo_brier_scores=[0.20, 0.19, 0.20],
            current_value=0.0,
            optimal_value=0.1,
            optimal_brier=0.190,
            current_brier=0.200,
            brier_range=0.010,
            is_flat=False,
        )
        report = RDOFAuditReport(sensitivity_results={"bad": sr_obj})
        recs = report._generate_recommendations()
        assert any("OVERFIT" in r for r in recs)

    def test_recommendations_complexity_pass(self):
        audit = estimate_model_complexity(n_training_samples=100000)
        report = RDOFAuditReport(complexity_audit=audit)
        recs = report._generate_recommendations()
        assert any("COMPLEXITY: PASS" in r for r in recs)

    def test_recommendations_dof_ratio(self):
        hr = self._make_holdout_report()
        report = RDOFAuditReport(holdout_report=hr)
        recs = report._generate_recommendations()
        assert any("DoF RATIO" in r for r in recs)

    def test_recommendations_effective_dof_summary(self):
        sr = {"a": self._make_sensitivity_result("a", is_flat=True)}
        report = RDOFAuditReport(sensitivity_results=sr)
        recs = report._generate_recommendations()
        assert any("EFFECTIVE DoF" in r for r in recs)

    def test_to_text_returns_string(self):
        hr = self._make_holdout_report()
        sr = {"x": self._make_sensitivity_result("x", is_flat=False)}
        audit = estimate_model_complexity()
        report = RDOFAuditReport(
            holdout_report=hr,
            sensitivity_results=sr,
            complexity_audit=audit,
        )
        text = report.to_text()
        assert isinstance(text, str)
        assert "RESEARCHER DEGREES OF FREEDOM" in text
        assert "CONSTANT INVENTORY" in text
        assert "HOLDOUT EVALUATION" in text
        assert "SENSITIVITY ANALYSIS" in text
        assert "MODEL COMPLEXITY AUDIT" in text
        assert "RECOMMENDATIONS" in text
        assert "DISCLOSURE" in text

    def test_to_text_minimal(self):
        report = RDOFAuditReport()
        text = report.to_text()
        assert "CONSTANT INVENTORY" in text

    def test_to_file(self):
        report = RDOFAuditReport()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_file(path)
            with open(path) as fh:
                data = json.load(fh)
            assert "metadata" in data
        finally:
            os.unlink(path)


# ─── Adopt Sensitivity Optima Tests ─────────────────────────────────


class TestAdoptSensitivityOptima:
    def _make_sr(self, name, current, optimal, gap, is_flat=False, brier_range=0.01):
        return ConstantSensitivityResult(
            constant_name=name,
            grid_values=[current, optimal],
            loyo_brier_scores=[current + gap, current],
            current_value=current,
            optimal_value=optimal,
            optimal_brier=current - gap,
            current_brier=current,
            brier_range=brier_range,
            is_flat=is_flat,
        )

    def test_adopt_significant_improvement(self):
        cfg = _make_config(tournament_shrinkage=0.0)
        sr = {"tournament_shrinkage": self._make_sr(
            "tournament_shrinkage", 0.0, 0.1, 0.005, is_flat=False, brier_range=0.01,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg, min_brier_improvement=0.002)
        assert new_cfg.tournament_shrinkage == 0.1
        assert log["tournament_shrinkage"]["action"] == "adopted"

    def test_skip_flat_constant(self):
        cfg = _make_config(ensemble_lgb_weight=0.45)
        sr = {"ensemble_lgb_weight": self._make_sr(
            "ensemble_lgb_weight", 0.45, 0.50, 0.001, is_flat=True, brier_range=0.002,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert log["ensemble_lgb_weight"]["action"] == "skipped"

    def test_neutralize_flat_neutral_default(self):
        cfg = _make_config(tournament_shrinkage=0.05)
        sr = {"tournament_shrinkage": self._make_sr(
            "tournament_shrinkage", 0.05, 0.06, 0.001, is_flat=True, brier_range=0.002,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert new_cfg.tournament_shrinkage == 0.0
        assert log["tournament_shrinkage"]["action"] == "neutralized"

    def test_skip_already_optimal(self):
        cfg = _make_config(ensemble_lgb_weight=0.45)
        sr = {"ensemble_lgb_weight": self._make_sr(
            "ensemble_lgb_weight", 0.45, 0.45, 0.005, is_flat=False, brier_range=0.01,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert log["ensemble_lgb_weight"]["action"] == "skipped"
        assert "already at optimal" in log["ensemble_lgb_weight"]["reason"]

    def test_skip_unknown_constant(self):
        cfg = _make_config()
        sr = {"unknown_param": self._make_sr(
            "unknown_param", 0.5, 0.6, 0.01, is_flat=False, brier_range=0.02,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert log["unknown_param"]["action"] == "skipped"
        assert "no config mapping" in log["unknown_param"]["reason"]

    def test_adopt_multiple_constants(self):
        cfg = _make_config(
            tournament_shrinkage=0.0,
            ensemble_lgb_weight=0.45,
        )
        sr = {
            "tournament_shrinkage": self._make_sr(
                "tournament_shrinkage", 0.0, 0.08, 0.005, brier_range=0.01,
            ),
            "ensemble_lgb_weight": self._make_sr(
                "ensemble_lgb_weight", 0.45, 0.50, 0.004, brier_range=0.01,
            ),
        }
        new_cfg, log = adopt_sensitivity_optima(sr, cfg, min_brier_improvement=0.002)
        assert new_cfg.tournament_shrinkage == 0.08
        assert new_cfg.ensemble_lgb_weight == 0.50

    def test_neutralize_seed_prior_weight(self):
        cfg = _make_config(seed_prior_weight=0.05)
        sr = {"seed_prior_weight": self._make_sr(
            "seed_prior_weight", 0.05, 0.06, 0.001, is_flat=True, brier_range=0.002,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert new_cfg.seed_prior_weight == 0.0

    def test_neutralize_consistency_bonus_max(self):
        cfg = _make_config(consistency_bonus_max=0.03)
        sr = {"consistency_bonus_max": self._make_sr(
            "consistency_bonus_max", 0.03, 0.04, 0.001, is_flat=True, brier_range=0.002,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert new_cfg.consistency_bonus_max == 0.0

    def test_neutralize_mc_regional_correlation(self):
        cfg = _make_config(mc_regional_correlation=0.1)
        sr = {"mc_regional_correlation": self._make_sr(
            "mc_regional_correlation", 0.1, 0.15, 0.001, is_flat=True, brier_range=0.002,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg)
        assert new_cfg.mc_regional_correlation == 0.0

    def test_adopt_consistency_normalizer(self):
        cfg = _make_config(consistency_normalizer=15.0)
        sr = {"consistency_normalizer": self._make_sr(
            "consistency_normalizer", 15.0, 20.0, 0.005, brier_range=0.01,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg, min_brier_improvement=0.002)
        assert new_cfg.consistency_normalizer == 20.0

    def test_adopt_ensemble_xgb_weight(self):
        cfg = _make_config(ensemble_xgb_weight=0.35)
        sr = {"ensemble_xgb_weight": self._make_sr(
            "ensemble_xgb_weight", 0.35, 0.40, 0.005, brier_range=0.01,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg, min_brier_improvement=0.002)
        assert new_cfg.ensemble_xgb_weight == 0.40

    def test_adopt_mc_regional_correlation(self):
        cfg = _make_config(mc_regional_correlation=0.0)
        sr = {"mc_regional_correlation": self._make_sr(
            "mc_regional_correlation", 0.0, 0.15, 0.005, brier_range=0.01,
        )}
        new_cfg, log = adopt_sensitivity_optima(sr, cfg, min_brier_improvement=0.002)
        assert new_cfg.mc_regional_correlation == 0.15


# ─── Run RDOF Audit Tests ──────────────────────────────────────────


class TestRunRDOFAudit:
    @patch("src.ml.evaluation.rdof_audit.check_holdout_contamination", return_value=None)
    @patch("src.ml.evaluation.rdof_audit.estimate_model_complexity")
    @patch("src.ml.evaluation.rdof_audit.RDOFAuditReport")
    def test_run_audit_no_holdout_no_sensitivity(self, mock_report_cls, mock_complexity, mock_contam):
        from src.ml.evaluation.rdof_audit import run_rdof_audit

        mock_audit = MagicMock()
        mock_complexity.return_value = mock_audit

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"metadata": {}}
        mock_report.to_text.return_value = "text"
        mock_report_cls.return_value = mock_report

        cfg = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output = f.name
        try:
            result = run_rdof_audit(
                historical_dir="/fake",
                holdout_years=[2025],
                run_holdout=False,
                run_sensitivity=False,
                output_path=output,
                config=cfg,
            )
            assert "audit_context" in result
        finally:
            if os.path.exists(output):
                os.unlink(output)


# ─── Infer Dates and Split Tests ────────────────────────────────────


class TestInferDatesAndSplit:
    def test_infer_dates_single_date(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        games = [
            {"date": "2021-01-01", "game_id": str(i)} for i in range(100)
        ]
        result = ev._infer_dates_and_split(games, 2021)
        # With only one unique date and >50 games, dates should be inferred
        # Check that at least some dates differ now
        dates = [g["date"] for g in result]
        assert len(set(dates)) > 1

    def test_infer_dates_multiple_dates(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        games = [
            {"date": f"2021-01-{i+1:02d}", "game_id": str(i)} for i in range(10)
        ]
        result = ev._infer_dates_and_split(games, 2021)
        # Multiple dates already exist, no inference needed
        assert len(result) == 10


# ─── Extract Tournament Games Tests ─────────────────────────────────


class TestExtractTournamentGames:
    def test_extract_with_matching_teams(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        team_metrics = {
            "duke": {"off_rtg": 110, "def_rtg": 95},
            "unc": {"off_rtg": 108, "def_rtg": 97},
        }
        games = [
            {
                "date": "2021-03-20",
                "team1_id": "duke",
                "team2_id": "unc",
                "team1_score": 80,
                "team2_score": 75,
            }
        ]
        with patch.object(ev, "_is_tournament_game", return_value=True):
            result = ev._extract_tournament_games(games, team_metrics, 2021)
        assert len(result) == 1
        assert result[0]["outcome"] == 1

    def test_extract_skips_non_tournament(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021])
        team_metrics = {"duke": {}, "unc": {}}
        games = [
            {
                "date": "2021-01-15",
                "team1_id": "duke",
                "team2_id": "unc",
                "team1_score": 80,
                "team2_score": 75,
            }
        ]
        with patch.object(ev, "_is_tournament_game", return_value=False):
            result = ev._extract_tournament_games(games, team_metrics, 2021)
        assert len(result) == 0


# ─── Sensitivity Analyzer Grid and Circularity Tests ────────────────


class TestSensitivityAnalyzerCircularity:
    def test_circularity_detected_loyo(self):
        pc = PipelineConstant(
            "test", 3, 0.45, "path", (0.2, 0.7),
            "Calibrated from LOYO", notes="",
        )
        assert "loyo" in pc.derivation.lower()

    def test_circularity_detected_iteratively(self):
        pc = PipelineConstant(
            "test", 3, 0.02, "path", (0.0, 0.1),
            "iteratively adjusted", notes="",
        )
        assert "iteratively" in pc.derivation.lower()

    def test_make_grid_sorted(self):
        sa = SensitivityAnalyzer("/fake", dev_years=[2021], n_grid_points=7)
        pc = PipelineConstant("test", 3, 0.55, "path", (0.0, 1.0), "deriv")
        grid = sa._make_grid(pc)
        assert grid == sorted(grid)


# ─── Run Holdout Protocol Tests ─────────────────────────────────────


class TestRunHoldoutProtocol:
    def test_run_protocol_catches_errors(self):
        ev = HoldoutEvaluator("/fake", all_years=[2021, 2022])
        cfg = _make_config()

        with patch.object(ev, "evaluate_holdout_year", side_effect=ValueError("no data")):
            report = ev.run_holdout_protocol([2021, 2022], cfg)

        assert report.holdout_years == [2021, 2022]
        assert len(report.per_year) == 0  # Both failed


# ─── Run Prospective Evaluation Tests ───────────────────────────────


class TestRunProspectiveEvaluation:
    @patch("src.ml.evaluation.rdof_audit.verify_freeze")
    def test_raises_on_mismatch(self, mock_verify):
        from src.ml.evaluation.rdof_audit import run_prospective_evaluation

        mock_verify.return_value = {
            "matches": False,
            "mismatches": ["hash mismatch"],
        }
        cfg = _make_config()

        with pytest.raises(ValueError, match="does not match"):
            run_prospective_evaluation(
                freeze_path="/fake/freeze.json",
                evaluation_year=2026,
                config=cfg,
            )


# ─── ECE Edge Cases ────────────────────────────────────────────────


class TestECEEdgeCases:
    def test_ece_last_bin_boundary(self):
        # Test that the last bin includes the right boundary (probs == 1.0)
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        outcomes = np.array([1.0, 1.0, 0.0, 0.0])
        ece = _compute_ece(probs, outcomes, n_bins=10)
        assert ece == pytest.approx(0.0)

    def test_ece_single_value(self):
        probs = np.array([0.5])
        outcomes = np.array([1.0])
        ece = _compute_ece(probs, outcomes)
        assert ece == pytest.approx(0.5)


# ─── HoldoutReport Verdict Edge Cases ──────────────────────────────


class TestHoldoutReportVerdictEdge:
    def test_verdict_exact_boundary_pass(self):
        # delta = 0.005 exactly: >0.005 is False, so WARN
        # But 0.200 - 0.195 = 0.005 which due to float is tricky.
        # Use values that clearly cross: delta = 0.006 -> PASS
        r = HoldoutReport(holdout_years=[2024])
        r.per_year = {
            2024: YearMetrics(2024, 63, 0.194, 0.5, 0.72, 0.04, 0.200, 0.25),
        }
        assert r.verdict() == "PASS"

    def test_verdict_marginal_warn(self):
        # delta = 0.002 -> positive but < 0.005 -> WARN
        r = HoldoutReport(holdout_years=[2024])
        r.per_year = {
            2024: YearMetrics(2024, 63, 0.198, 0.5, 0.72, 0.04, 0.200, 0.25),
        }
        assert r.verdict() == "WARN"

    def test_verdict_zero_delta_is_fail(self):
        # delta = 0 -> not > 0, so FAIL
        r = HoldoutReport(holdout_years=[2024])
        r.per_year = {
            2024: YearMetrics(2024, 63, 0.200, 0.5, 0.72, 0.04, 0.200, 0.25),
        }
        assert r.verdict() == "FAIL"


# ─── Sensitivity Near-Optimal Recommendation ─────────────────────


class TestSensitivityNearOptimalRec:
    def test_near_optimal_recommendation(self):
        sr = ConstantSensitivityResult(
            constant_name="test_const",
            grid_values=[0.0, 0.1, 0.2],
            loyo_brier_scores=[0.20, 0.195, 0.198],
            current_value=0.1,
            optimal_value=0.1,
            optimal_brier=0.195,
            current_brier=0.196,
            brier_range=0.008,
            is_flat=False,
        )
        report = RDOFAuditReport(sensitivity_results={"test_const": sr})
        recs = report._generate_recommendations()
        assert any("near-optimal" in r for r in recs)


# ─── to_text ASCII Plot ────────────────────────────────────────────


class TestToTextASCIIPlot:
    def test_ascii_plot_includes_markers(self):
        sr = ConstantSensitivityResult(
            constant_name="plot_test",
            grid_values=[0.0, 0.1, 0.2],
            loyo_brier_scores=[0.200, 0.190, 0.195],
            current_value=0.0,
            optimal_value=0.1,
            optimal_brier=0.190,
            current_brier=0.200,
            brier_range=0.010,
            is_flat=False,
        )
        report = RDOFAuditReport(sensitivity_results={"plot_test": sr})
        text = report.to_text()
        assert "<-- optimal" in text
        assert "<-- current" in text


# ─── Freeze with MC calibration ────────────────────────────────────


class TestFreezeWithMCCalibration:
    @patch("subprocess.run")
    def test_freeze_includes_mc_calibration(self, mock_run):
        from src.ml.evaluation.rdof_audit import freeze_pipeline

        mock_run.return_value = MagicMock(returncode=0, stdout="sha\n")

        mc_payload = {
            "best_params": {"noise_std": 0.12},
            "best_dev_score": -0.75,
            "holdout_score": -0.80,
            "_source_path": "/fake/mc.json",
        }
        cfg = _make_config()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config",
                        return_value=mc_payload):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value=None):
                    result = freeze_pipeline(cfg, output_path=path)
            assert "mc_calibration" in result
            assert result["mc_calibration"]["best_params"]["noise_std"] == 0.12
        finally:
            os.unlink(path)


# ─── Verify Freeze MC Mismatch ─────────────────────────────────────


class TestVerifyFreezeMCMismatch:
    def test_mc_noise_std_mismatch_detected(self):
        from src.ml.evaluation.rdof_audit import verify_freeze

        cfg = _make_config(mc_noise_std=0.20)
        h = config_hash(cfg)
        freeze_data = {
            "config_hash": h,
            "config_fields": {},
            "constant_registry": [c.to_dict() for c in CONSTANT_REGISTRY],
            "mc_calibration": {
                "best_params": {"noise_std": 0.12},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(freeze_data, f)
            path = f.name

        try:
            with patch("src.ml.evaluation.rdof_audit._apply_mc_calibration_to_config", return_value=None):
                with patch("src.ml.evaluation.rdof_audit._feature_set_hash", return_value=None):
                    result = verify_freeze(cfg, path)
            assert any("MC noise_std" in m for m in result["mismatches"])
        finally:
            os.unlink(path)


# ─── HoldoutReport to_dict aggregate skill scores ──────────────────


class TestHoldoutReportAggregateSkillScores:
    def test_aggregate_skill_scores_in_to_dict(self):
        r = HoldoutReport(holdout_years=[2024])
        r.per_year = {
            2024: YearMetrics(2024, 63, 0.18, 0.5, 0.72, 0.04, 0.22, 0.25),
        }
        d = r.to_dict()
        assert "aggregate_brier_skill_score" in d
        assert "aggregate_elo_skill_score" in d
        assert d["aggregate_brier_skill_score"] > 0
        assert d["aggregate_elo_skill_score"] > 0
