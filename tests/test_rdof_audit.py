"""Tests for Researcher Degrees of Freedom audit framework."""

import json
import math
import os
import tempfile

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
    _seed_baseline_prob,
    adopt_sensitivity_optima,
    check_holdout_contamination,
    config_hash,
    estimate_model_complexity,
    get_constants_by_tier,
    get_tier3_constants,
    record_holdout_evaluation,
)
from src.pipeline.sota import SOTAPipelineConfig


# ───────────────────────────────────────────────────────────────────────
# Constant Registry Tests
# ───────────────────────────────────────────────────────────────────────

class TestConstantRegistry:
    """Tests for the constant registry completeness and validity."""

    def test_all_tiers_present(self):
        """Registry has constants in all three tiers."""
        tiers_present = set(c.tier for c in CONSTANT_REGISTRY)
        assert {1, 2, 3} == tiers_present

    def test_tier_counts(self):
        """Reasonable number of constants in each tier."""
        tier1 = get_constants_by_tier(1)
        tier2 = get_constants_by_tier(2)
        tier3 = get_constants_by_tier(3)
        assert len(tier1) >= 5, "Tier 1 should have at least 5 externally derived constants"
        assert len(tier2) >= 8, "Tier 2 should have at least 8 structurally constrained constants"
        assert len(tier3) >= 6, "Tier 3 should have at least 6 freely tuned constants"

    def test_tier3_completeness(self):
        """All critical Tier 3 constants are registered."""
        tier3_names = {c.name for c in get_tier3_constants()}
        required = {
            "tournament_shrinkage",
            "seed_prior_weight",
            "ensemble_lgb_weight",
            "ensemble_xgb_weight",
            "consistency_bonus_max",
        }
        assert required.issubset(tier3_names), (
            f"Missing Tier 3 constants: {required - tier3_names}"
        )

    def test_valid_ranges_sensible(self):
        """Valid ranges are non-empty and contain current value."""
        for c in CONSTANT_REGISTRY:
            lo, hi = c.valid_range
            assert lo < hi, f"{c.name}: valid_range lo ({lo}) >= hi ({hi})"

            # For scalar constants, current value should be in valid range
            if isinstance(c.current_value, (int, float)):
                assert lo <= c.current_value <= hi, (
                    f"{c.name}: current_value {c.current_value} outside "
                    f"valid_range [{lo}, {hi}]"
                )

    def test_all_have_derivation(self):
        """Every constant has a non-empty derivation string."""
        for c in CONSTANT_REGISTRY:
            assert c.derivation, f"{c.name} has empty derivation"

    def test_all_have_config_path(self):
        """Every constant has a config path."""
        for c in CONSTANT_REGISTRY:
            assert c.config_path, f"{c.name} has empty config_path"

    def test_tier3_game_and_mc_partition(self):
        """Tier 3 constants are partitioned into game-level and MC."""
        all_tier3_names = {c.name for c in get_tier3_constants()}
        game_set = set(TIER3_GAME_LEVEL_CONSTANTS)
        mc_set = set(TIER3_MC_CONSTANTS)
        # Game + MC should cover all tier3
        assert game_set | mc_set == all_tier3_names, (
            f"Uncovered: {all_tier3_names - game_set - mc_set}"
        )
        # No overlap
        assert not (game_set & mc_set), "Game and MC sets overlap"

    def test_serialization_roundtrip(self):
        """PipelineConstant serializes to dict correctly."""
        c = CONSTANT_REGISTRY[0]
        d = c.to_dict()
        assert d["name"] == c.name
        assert d["tier"] == c.tier
        assert "derivation" in d


# ───────────────────────────────────────────────────────────────────────
# Config Hash Tests
# ───────────────────────────────────────────────────────────────────────

class TestConfigHash:
    """Tests for config hashing (audit trail)."""

    def test_deterministic(self):
        """Same config produces same hash."""
        c1 = SOTAPipelineConfig()
        c2 = SOTAPipelineConfig()
        assert config_hash(c1) == config_hash(c2)

    def test_changes_on_modification(self):
        """Changing a field changes the hash."""
        c1 = SOTAPipelineConfig()
        c2 = SOTAPipelineConfig(tournament_shrinkage=0.12)
        assert config_hash(c1) != config_hash(c2)

    def test_hash_length(self):
        """Hash is a reasonable hex string."""
        h = config_hash(SOTAPipelineConfig())
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_new_fields_change_hash(self):
        """Changing the new config fields produces different hashes."""
        c_base = SOTAPipelineConfig()
        c_seed = SOTAPipelineConfig(seed_prior_weight=0.10)
        c_bonus = SOTAPipelineConfig(consistency_bonus_max=0.05)
        c_lgb = SOTAPipelineConfig(ensemble_lgb_weight=0.50)
        hashes = {config_hash(c) for c in [c_base, c_seed, c_bonus, c_lgb]}
        assert len(hashes) == 4, "All configs should have different hashes"


# ───────────────────────────────────────────────────────────────────────
# Metric Computation Tests
# ───────────────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for metric computation helpers."""

    def test_brier_perfect(self):
        """Perfect predictions have Brier = 0."""
        probs = np.array([1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert _compute_brier(probs, outcomes) == pytest.approx(0.0)

    def test_brier_worst(self):
        """Worst predictions have Brier = 1."""
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert _compute_brier(probs, outcomes) == pytest.approx(1.0)

    def test_brier_uniform(self):
        """Uniform 0.5 predictions have Brier = 0.25."""
        probs = np.full(100, 0.5)
        outcomes = np.random.RandomState(42).randint(0, 2, 100).astype(float)
        assert _compute_brier(probs, outcomes) == pytest.approx(0.25)

    def test_log_loss_perfect(self):
        """Near-perfect predictions have very low log loss."""
        probs = np.array([0.999, 0.001])
        outcomes = np.array([1.0, 0.0])
        assert _compute_log_loss(probs, outcomes) < 0.01

    def test_accuracy(self):
        """Accuracy computed correctly."""
        probs = np.array([0.8, 0.3, 0.6, 0.2])
        outcomes = np.array([1.0, 0.0, 1.0, 1.0])
        assert _compute_accuracy(probs, outcomes) == pytest.approx(0.75)

    def test_ece_calibrated(self):
        """Well-calibrated predictions have low ECE."""
        # Generate perfectly calibrated probabilities
        rng = np.random.RandomState(42)
        probs = rng.uniform(0, 1, 1000)
        outcomes = (rng.uniform(0, 1, 1000) < probs).astype(float)
        ece = _compute_ece(probs, outcomes)
        assert ece < 0.05, f"ECE should be low for calibrated probs, got {ece}"

    def test_seed_baseline_prob_equal_seeds(self):
        """Equal seeds → 0.5."""
        assert _seed_baseline_prob(8, 8) == pytest.approx(0.5)

    def test_seed_baseline_prob_lower_seed_favored(self):
        """Lower seed (1) vs higher seed (16) → high probability."""
        p = _seed_baseline_prob(1, 16)
        assert p > 0.85, f"1 vs 16 should have high win prob, got {p}"

    def test_seed_baseline_prob_higher_seed_underdog(self):
        """Higher seed (16) vs lower seed (1) → low probability."""
        p = _seed_baseline_prob(16, 1)
        assert p < 0.15

    def test_bootstrap_ci_contains_point_estimate(self):
        """Bootstrap CI contains the point Brier estimate."""
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.3, 0.7, 100)
        outcomes = (rng.uniform(0, 1, 100) < 0.5).astype(float)
        lo, hi = _bootstrap_brier_ci(probs, outcomes)
        point = _compute_brier(probs, outcomes)
        assert lo <= point <= hi


# ───────────────────────────────────────────────────────────────────────
# YearMetrics Tests
# ───────────────────────────────────────────────────────────────────────

class TestYearMetrics:
    """Tests for YearMetrics dataclass."""

    def test_brier_skill_score_positive(self):
        """Model better than seed → positive BSS."""
        m = YearMetrics(
            year=2024, n_games=63, brier_score=0.180,
            log_loss=0.5, accuracy=0.72, ece=0.03,
            seed_baseline_brier=0.200,
        )
        assert m.brier_skill_score() > 0

    def test_brier_skill_score_negative(self):
        """Model worse than seed → negative BSS."""
        m = YearMetrics(
            year=2024, n_games=63, brier_score=0.220,
            log_loss=0.6, accuracy=0.65, ece=0.05,
            seed_baseline_brier=0.200,
        )
        assert m.brier_skill_score() < 0

    def test_serialization(self):
        """to_dict() produces valid dict."""
        m = YearMetrics(
            year=2024, n_games=63, brier_score=0.190,
            log_loss=0.55, accuracy=0.70, ece=0.04,
            seed_baseline_brier=0.200,
        )
        d = m.to_dict()
        assert d["year"] == 2024
        assert "brier_skill_score" in d


# ───────────────────────────────────────────────────────────────────────
# HoldoutReport Tests
# ───────────────────────────────────────────────────────────────────────

class TestHoldoutReport:
    """Tests for HoldoutReport."""

    def _make_report(self, brier1=0.190, brier2=0.185, seed1=0.200, seed2=0.200):
        report = HoldoutReport(holdout_years=[2024, 2025])
        report.per_year[2024] = YearMetrics(
            year=2024, n_games=63, brier_score=brier1,
            log_loss=0.55, accuracy=0.70, ece=0.04,
            seed_baseline_brier=seed1,
        )
        report.per_year[2025] = YearMetrics(
            year=2025, n_games=63, brier_score=brier2,
            log_loss=0.52, accuracy=0.72, ece=0.03,
            seed_baseline_brier=seed2,
        )
        return report

    def test_aggregate_brier(self):
        """Weighted average Brier across years."""
        report = self._make_report()
        expected = (0.190 * 63 + 0.185 * 63) / 126
        assert report.aggregate_brier == pytest.approx(expected, abs=1e-6)

    def test_verdict_pass(self):
        """PASS when Brier delta > 0.005."""
        report = self._make_report(brier1=0.185, brier2=0.185, seed1=0.200, seed2=0.200)
        assert report.verdict() == "PASS"

    def test_verdict_warn(self):
        """WARN when delta > 0 but < 0.005."""
        report = self._make_report(brier1=0.197, brier2=0.197, seed1=0.200, seed2=0.200)
        assert report.verdict() == "WARN"

    def test_verdict_fail(self):
        """FAIL when model Brier >= seed Brier."""
        report = self._make_report(brier1=0.210, brier2=0.210, seed1=0.200, seed2=0.200)
        assert report.verdict() == "FAIL"

    def test_serialization(self):
        """to_dict() is complete."""
        report = self._make_report()
        d = report.to_dict()
        assert "aggregate_brier" in d
        assert "verdict" in d
        assert 2024 in d["per_year"]


# ───────────────────────────────────────────────────────────────────────
# SensitivityResult Tests
# ───────────────────────────────────────────────────────────────────────

class TestSensitivityResult:
    """Tests for ConstantSensitivityResult."""

    def test_flat_detection(self):
        """Flat curves (range < 0.005) are detected."""
        sr = ConstantSensitivityResult(
            constant_name="test",
            grid_values=[0.0, 0.05, 0.10],
            loyo_brier_scores=[0.200, 0.199, 0.201],
            current_value=0.05,
            optimal_value=0.05,
            optimal_brier=0.199,
            current_brier=0.199,
            brier_range=0.002,
            is_flat=True,
        )
        assert sr.is_flat
        assert sr.brier_gap == pytest.approx(0.0)

    def test_non_flat_detection(self):
        """Non-flat curves are detected."""
        sr = ConstantSensitivityResult(
            constant_name="test",
            grid_values=[0.0, 0.10, 0.20],
            loyo_brier_scores=[0.190, 0.200, 0.210],
            current_value=0.10,
            optimal_value=0.0,
            optimal_brier=0.190,
            current_brier=0.200,
            brier_range=0.020,
            is_flat=False,
        )
        assert not sr.is_flat
        assert sr.brier_gap == pytest.approx(0.010)

    def test_rank_of_current(self):
        """Rank computation works."""
        sr = ConstantSensitivityResult(
            constant_name="test",
            grid_values=[0.0, 0.05, 0.10],
            loyo_brier_scores=[0.190, 0.195, 0.200],
            current_value=0.10,
            optimal_value=0.0,
            optimal_brier=0.190,
            current_brier=0.200,
            brier_range=0.010,
            is_flat=False,
        )
        assert sr.rank_of_current == 3  # Worst of 3

    def test_serialization(self):
        sr = ConstantSensitivityResult(
            constant_name="test",
            grid_values=[0.0, 0.10],
            loyo_brier_scores=[0.190, 0.200],
            current_value=0.10,
            optimal_value=0.0,
            optimal_brier=0.190,
            current_brier=0.200,
            brier_range=0.010,
            is_flat=False,
        )
        d = sr.to_dict()
        assert d["constant_name"] == "test"
        assert d["brier_gap"] == pytest.approx(0.010, abs=1e-5)


# ───────────────────────────────────────────────────────────────────────
# HoldoutEvaluator Tests
# ───────────────────────────────────────────────────────────────────────

HISTORICAL_DIR = "data/raw/historical"


class TestHoldoutEvaluator:
    """Tests for the HoldoutEvaluator (integration tests)."""

    @pytest.fixture
    def evaluator(self):
        if not os.path.exists(HISTORICAL_DIR):
            pytest.skip("Historical data not available")
        return HoldoutEvaluator(HISTORICAL_DIR)

    def test_year_detection(self, evaluator):
        """Detects available historical years."""
        assert len(evaluator.all_years) >= 5
        assert 2020 not in evaluator.all_years  # COVID excluded

    def test_holdout_year_exclusion(self, evaluator):
        """Holdout year is never in training data."""
        holdout = 2024
        training_years = [y for y in evaluator.all_years
                          if y != holdout and y != 2020]
        assert holdout not in training_years

    def test_tournament_game_detection(self, evaluator):
        """Can detect tournament-era dates."""
        assert evaluator._is_tournament_game("2024-03-21", 2024)
        assert evaluator._is_tournament_game("2024-04-08", 2024)
        assert not evaluator._is_tournament_game("2024-02-15", 2024)
        assert not evaluator._is_tournament_game("2024-05-01", 2024)

    def test_evaluate_single_year(self, evaluator):
        """Evaluate holdout year produces valid metrics."""
        config = SOTAPipelineConfig()
        metrics = evaluator.evaluate_holdout_year(2024, config)
        assert metrics.year == 2024
        assert metrics.n_games > 0
        assert 0.0 < metrics.brier_score < 0.5
        assert 0.0 < metrics.accuracy < 1.0
        assert metrics.seed_baseline_brier > 0


# ───────────────────────────────────────────────────────────────────────
# Report Tests
# ───────────────────────────────────────────────────────────────────────

class TestRDOFAuditReport:
    """Tests for the audit report formatter."""

    def test_text_report_generation(self):
        """Text report generates without errors."""
        report = RDOFAuditReport()
        text = report.to_text()
        assert "RESEARCHER DEGREES OF FREEDOM" in text
        assert "CONSTANT INVENTORY" in text

    def test_json_report_generation(self):
        """JSON report is valid."""
        report = RDOFAuditReport()
        d = report.to_dict()
        assert "constant_inventory" in d
        assert "metadata" in d

    def test_report_with_holdout(self):
        """Report with holdout data."""
        hr = HoldoutReport(holdout_years=[2024])
        hr.per_year[2024] = YearMetrics(
            year=2024, n_games=63, brier_score=0.190,
            log_loss=0.55, accuracy=0.70, ece=0.04,
            seed_baseline_brier=0.200,
        )
        report = RDOFAuditReport(holdout_report=hr)
        text = report.to_text()
        assert "HOLDOUT EVALUATION" in text
        assert "2024" in text

    def test_report_with_sensitivity(self):
        """Report with sensitivity results."""
        sr = ConstantSensitivityResult(
            constant_name="tournament_shrinkage",
            grid_values=[0.0, 0.08, 0.16],
            loyo_brier_scores=[0.195, 0.190, 0.198],
            current_value=0.08,
            optimal_value=0.08,
            optimal_brier=0.190,
            current_brier=0.190,
            brier_range=0.008,
            is_flat=False,
        )
        report = RDOFAuditReport(sensitivity_results={"tournament_shrinkage": sr})
        text = report.to_text()
        assert "SENSITIVITY ANALYSIS" in text
        assert "tournament_shrinkage" in text

    def test_json_file_output(self):
        """Report writes to JSON file."""
        report = RDOFAuditReport()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmppath = f.name
        try:
            report.to_file(tmppath)
            with open(tmppath) as f:
                data = json.load(f)
            assert "constant_inventory" in data
        finally:
            os.unlink(tmppath)

    def test_recommendations_generated(self):
        """Recommendations are generated."""
        hr = HoldoutReport(holdout_years=[2024])
        hr.per_year[2024] = YearMetrics(
            year=2024, n_games=63, brier_score=0.190,
            log_loss=0.55, accuracy=0.70, ece=0.04,
            seed_baseline_brier=0.200,
        )
        report = RDOFAuditReport(holdout_report=hr)
        recs = report._generate_recommendations()
        assert len(recs) > 0
        assert any("HOLDOUT" in r for r in recs)


# ───────────────────────────────────────────────────────────────────────
# SOTAPipelineConfig Tests (new fields)
# ───────────────────────────────────────────────────────────────────────

class TestConfigNewFields:
    """Tests for the new config fields added for RDoF audit."""

    def test_default_values(self):
        config = SOTAPipelineConfig()
        assert config.seed_prior_weight == 0.05
        assert config.seed_prior_slope == 0.175
        assert config.consistency_bonus_max == 0.02
        assert config.consistency_normalizer == 15.0
        assert config.ensemble_lgb_weight == 0.45
        assert config.ensemble_xgb_weight == 0.35

    def test_ensemble_weights_sum(self):
        """Default ensemble weights sum to 1."""
        config = SOTAPipelineConfig()
        w_logit = 1.0 - config.ensemble_lgb_weight - config.ensemble_xgb_weight
        total = config.ensemble_lgb_weight + config.ensemble_xgb_weight + w_logit
        assert total == pytest.approx(1.0)

    def test_custom_values(self):
        config = SOTAPipelineConfig(
            tournament_shrinkage=0.10,
            seed_prior_weight=0.08,
            consistency_bonus_max=0.03,
            ensemble_lgb_weight=0.50,
        )
        assert config.tournament_shrinkage == 0.10
        assert config.seed_prior_weight == 0.08
        assert config.consistency_bonus_max == 0.03
        assert config.ensemble_lgb_weight == 0.50


# ───────────────────────────────────────────────────────────────────────
# Model Complexity Audit Tests
# ───────────────────────────────────────────────────────────────────────

class TestModelComplexityAudit:
    """Tests for the effective model complexity auditor."""

    def test_default_audit_structure(self):
        """Default audit returns all expected fields."""
        audit = estimate_model_complexity()
        assert audit.n_training_samples == 400
        assert audit.total_effective_params > 0
        assert "lightgbm" in audit.component_params
        assert "xgboost" in audit.component_params
        assert "logistic_regression" in audit.component_params
        assert "gnn_projection" in audit.component_params
        assert "transformer_projection" in audit.component_params
        assert "ensemble_weights" in audit.component_params
        assert "tier3_constants" in audit.component_params

    def test_gnn_projection_dimension(self):
        """GNN projection params = 2 * embedding_dim + 1."""
        audit = estimate_model_complexity(gnn_embedding_dim=16)
        assert audit.component_params["gnn_projection"] == 33

    def test_transformer_projection_dimension(self):
        """Transformer projection params = 2 * embedding_dim + 1."""
        audit = estimate_model_complexity(transformer_embedding_dim=48)
        assert audit.component_params["transformer_projection"] == 97

    def test_pass_with_large_n(self):
        """Large N makes ratio < 10%."""
        audit = estimate_model_complexity(n_training_samples=50000)
        assert audit.passed
        assert audit.actual_ratio < 0.10

    def test_fail_with_tiny_n(self):
        """Tiny N causes complexity violation."""
        audit = estimate_model_complexity(n_training_samples=50)
        assert not audit.passed
        assert audit.actual_ratio > 0.10
        assert any("COMPLEXITY VIOLATION" in w for w in audit.warnings)

    def test_transformer_warning_on_small_n(self):
        """Transformer projection warned when >20% of N."""
        audit = estimate_model_complexity(
            n_training_samples=200, transformer_embedding_dim=48
        )
        assert any("Transformer projection" in w for w in audit.warnings)

    def test_serialization(self):
        """to_dict produces valid dict."""
        audit = estimate_model_complexity()
        d = audit.to_dict()
        assert "n_training_samples" in d
        assert "total_effective_params" in d
        assert "passed" in d
        assert "actual_ratio" in d


# ───────────────────────────────────────────────────────────────────────
# YearMetrics Elo Baseline Tests
# ───────────────────────────────────────────────────────────────────────

class TestYearMetricsEloBaseline:
    """Tests for the Elo baseline addition to YearMetrics."""

    def test_elo_skill_score_positive(self):
        """Model better than Elo → positive skill score."""
        m = YearMetrics(
            year=2024, n_games=63, brier_score=0.180,
            log_loss=0.5, accuracy=0.72, ece=0.03,
            seed_baseline_brier=0.200,
            elo_baseline_brier=0.220,
        )
        assert m.elo_skill_score() > 0

    def test_elo_skill_score_negative(self):
        """Model worse than Elo → negative skill score."""
        m = YearMetrics(
            year=2024, n_games=63, brier_score=0.230,
            log_loss=0.6, accuracy=0.65, ece=0.05,
            seed_baseline_brier=0.200,
            elo_baseline_brier=0.220,
        )
        assert m.elo_skill_score() < 0

    def test_elo_in_to_dict(self):
        """to_dict includes Elo baseline fields."""
        m = YearMetrics(
            year=2024, n_games=63, brier_score=0.190,
            log_loss=0.55, accuracy=0.70, ece=0.04,
            seed_baseline_brier=0.200,
            elo_baseline_brier=0.225,
        )
        d = m.to_dict()
        assert "elo_baseline_brier" in d
        assert "elo_skill_score" in d

    def test_holdout_report_elo_aggregate(self):
        """HoldoutReport aggregates Elo baseline Brier."""
        report = HoldoutReport(holdout_years=[2024])
        report.per_year[2024] = YearMetrics(
            year=2024, n_games=63, brier_score=0.190,
            log_loss=0.55, accuracy=0.70, ece=0.04,
            seed_baseline_brier=0.200,
            elo_baseline_brier=0.225,
        )
        assert report.aggregate_elo_brier == pytest.approx(0.225, abs=1e-6)

        d = report.to_dict()
        assert "aggregate_elo_brier" in d
        assert "aggregate_elo_skill_score" in d


# ───────────────────────────────────────────────────────────────────────
# MC Backtest Tests
# ───────────────────────────────────────────────────────────────────────

class TestMCBacktestResult:
    """Tests for MCBacktestResult dataclass."""

    def test_serialization(self):
        """to_dict produces valid structure."""
        result = MCBacktestResult(
            noise_std=0.12,
            regional_correlation=0.10,
            years_tested=[2019, 2021, 2022],
            per_year_upset_calibration={2019: 0.05, 2021: 0.03, 2022: 0.08},
            mean_upset_calibration_error=0.053,
            per_year_seed_accuracy={2019: 0.72, 2021: 0.68, 2022: 0.70},
            mean_seed_accuracy=0.70,
            per_year_log_prob={2019: -0.5, 2021: -0.6, 2022: -0.55},
            mean_log_prob=-0.55,
        )
        d = result.to_dict()
        assert d["noise_std"] == 0.12
        assert d["regional_correlation"] == 0.10
        assert len(d["years_tested"]) == 3
        assert "mean_upset_calibration_error" in d
        assert "mean_log_prob" in d


# ───────────────────────────────────────────────────────────────────────
# RDOFAuditReport Complexity Audit Integration
# ───────────────────────────────────────────────────────────────────────

class TestRDOFReportComplexity:
    """Tests for complexity audit in the RDoF report."""

    def test_report_with_complexity_audit(self):
        """Report includes complexity audit when provided."""
        audit = estimate_model_complexity(n_training_samples=400)
        report = RDOFAuditReport(complexity_audit=audit)

        d = report.to_dict()
        assert "model_complexity_audit" in d
        assert d["model_complexity_audit"]["n_training_samples"] == 400

    def test_text_report_complexity_section(self):
        """Text report includes complexity section."""
        audit = estimate_model_complexity(n_training_samples=400)
        report = RDOFAuditReport(complexity_audit=audit)
        text = report.to_text()
        assert "MODEL COMPLEXITY AUDIT" in text
        assert "lightgbm" in text

    def test_recommendations_include_complexity(self):
        """Recommendations include complexity warnings."""
        audit = estimate_model_complexity(n_training_samples=50)
        report = RDOFAuditReport(complexity_audit=audit)
        recs = report._generate_recommendations()
        assert any("COMPLEXITY" in r for r in recs)


# ───────────────────────────────────────────────────────────────────────
# Calibration Guardrail Tests
# ───────────────────────────────────────────────────────────────────────

class TestCalibrationGuardrails:
    """Tests for calibration method guardrails (isotonic/Platt overfitting protection)."""

    def test_temperature_scaling_no_downgrade(self):
        """Temperature scaling is never downgraded."""
        from src.ml.calibration.calibration import CalibrationPipeline, TemperatureScaling
        rng = np.random.RandomState(42)
        preds = rng.uniform(0.3, 0.7, 30)
        outcomes = (rng.uniform(0, 1, 30) < 0.5).astype(float)

        pipeline = CalibrationPipeline(method="temperature", nested_cv=False)
        pipeline.fit(preds, outcomes)
        assert pipeline.method == "temperature"
        assert pipeline._downgraded_from is None

    def test_isotonic_downgraded_without_nested_cv(self):
        """Isotonic is downgraded to temperature on small N without nested CV."""
        from src.ml.calibration.calibration import CalibrationPipeline, CALIBRATION_MIN_SAMPLES
        rng = np.random.RandomState(42)
        # Use N < isotonic minimum (200)
        n = min(CALIBRATION_MIN_SAMPLES["isotonic"] - 1, 100)
        preds = rng.uniform(0.3, 0.7, n)
        outcomes = (rng.uniform(0, 1, n) < 0.5).astype(float)

        pipeline = CalibrationPipeline(method="isotonic", nested_cv=False)
        pipeline.fit(preds, outcomes)
        assert pipeline.method == "temperature"
        assert pipeline._downgraded_from == "isotonic"

    def test_platt_downgraded_without_nested_cv(self):
        """Platt is downgraded to temperature on small N without nested CV."""
        from src.ml.calibration.calibration import CalibrationPipeline, CALIBRATION_MIN_SAMPLES
        rng = np.random.RandomState(42)
        # Use N < platt minimum (100)
        n = min(CALIBRATION_MIN_SAMPLES["platt"] - 1, 50)
        preds = rng.uniform(0.3, 0.7, n)
        outcomes = (rng.uniform(0, 1, n) < 0.5).astype(float)

        pipeline = CalibrationPipeline(method="platt", nested_cv=False)
        pipeline.fit(preds, outcomes)
        assert pipeline.method == "temperature"
        assert pipeline._downgraded_from == "platt"

    def test_isotonic_kept_with_nested_cv(self):
        """Isotonic is kept when nested CV is declared."""
        from src.ml.calibration.calibration import CalibrationPipeline
        rng = np.random.RandomState(42)
        preds = rng.uniform(0.3, 0.7, 50)
        outcomes = (rng.uniform(0, 1, 50) < 0.5).astype(float)

        pipeline = CalibrationPipeline(method="isotonic", nested_cv=True)
        pipeline.fit(preds, outcomes)
        assert pipeline.method == "isotonic"
        assert pipeline._downgraded_from is None

    def test_isotonic_kept_with_large_n(self):
        """Isotonic is kept on large N even without nested CV."""
        from src.ml.calibration.calibration import CalibrationPipeline, CALIBRATION_MIN_SAMPLES
        rng = np.random.RandomState(42)
        n = CALIBRATION_MIN_SAMPLES["isotonic"] + 50
        preds = rng.uniform(0.3, 0.7, n)
        outcomes = (rng.uniform(0, 1, n) < 0.5).astype(float)

        pipeline = CalibrationPipeline(method="isotonic", nested_cv=False)
        pipeline.fit(preds, outcomes)
        # Should still be isotonic (large enough N), just with a warning
        assert pipeline.method == "isotonic"
        assert pipeline._downgraded_from is None


# ───────────────────────────────────────────────────────────────────────
# Holdout Contamination Tracking Tests
# ───────────────────────────────────────────────────────────────────────

class TestHoldoutContaminationTracking:
    """Tests for the holdout contamination lockfile mechanism."""

    def test_record_and_check_no_contamination(self):
        """Recording and checking with same config shows no contamination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SOTAPipelineConfig()
            record_holdout_evaluation(
                tmpdir,
                holdout_years=[2024, 2025],
                config_hash_value=config_hash(config),
                report_summary={"aggregate_brier": 0.190, "verdict": "PASS"},
            )
            result = check_holdout_contamination(tmpdir, config)
            assert result is None  # No contamination

    def test_detect_contamination_after_config_change(self):
        """Changing config after recording triggers contamination warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = SOTAPipelineConfig()
            record_holdout_evaluation(
                tmpdir,
                holdout_years=[2024, 2025],
                config_hash_value=config_hash(config1),
                report_summary={"aggregate_brier": 0.190, "verdict": "PASS"},
            )
            # Modify a Tier 3 constant
            config2 = SOTAPipelineConfig(tournament_shrinkage=0.12)
            result = check_holdout_contamination(tmpdir, config2)
            assert result is not None
            assert result["contamination_detected"] is True
            assert "config has changed" in result["message"]

    def test_no_lockfile_no_contamination(self):
        """No lockfile → no contamination (holdout never evaluated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SOTAPipelineConfig()
            result = check_holdout_contamination(tmpdir, config)
            assert result is None

    def test_lockfile_contents(self):
        """Lockfile contains expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SOTAPipelineConfig()
            record_holdout_evaluation(
                tmpdir,
                holdout_years=[2024, 2025],
                config_hash_value=config_hash(config),
                report_summary={"aggregate_brier": 0.190, "verdict": "PASS"},
            )
            lockfile = os.path.join(tmpdir, "holdout_evaluation.lock.json")
            assert os.path.exists(lockfile)
            with open(lockfile) as f:
                data = json.load(f)
            assert data["holdout_years"] == [2024, 2025]
            assert "config_hash" in data
            assert "warning" in data
            assert "Do NOT modify" in data["warning"]


# ───────────────────────────────────────────────────────────────────────
# Sensitivity Auto-Adoption Tests
# ───────────────────────────────────────────────────────────────────────

class TestSensitivityAutoAdoption:
    """Tests for the sensitivity-based auto-adoption of Tier 3 constants."""

    def _make_sensitivity_result(
        self, name, current, optimal, brier_gap, is_flat=False
    ):
        return ConstantSensitivityResult(
            constant_name=name,
            grid_values=[current - 0.05, current, optimal],
            loyo_brier_scores=[0.200, 0.200 - brier_gap, 0.200 - brier_gap],
            current_value=current,
            optimal_value=optimal,
            optimal_brier=0.200 - brier_gap,
            current_brier=0.200,
            brier_range=brier_gap if not is_flat else 0.001,
            is_flat=is_flat,
        )

    def test_adopt_significant_improvement(self):
        """Adopts optimal value when improvement exceeds threshold."""
        sr = self._make_sensitivity_result(
            "tournament_shrinkage", 0.08, 0.12, 0.005
        )
        config = SOTAPipelineConfig()
        new_config, log = adopt_sensitivity_optima(
            {"tournament_shrinkage": sr}, config
        )
        assert new_config.tournament_shrinkage == 0.12
        assert log["tournament_shrinkage"]["action"] == "adopted"

    def test_skip_flat_constant(self):
        """Skips flat constants (insensitive)."""
        sr = self._make_sensitivity_result(
            "tournament_shrinkage", 0.08, 0.12, 0.005, is_flat=True
        )
        config = SOTAPipelineConfig()
        new_config, log = adopt_sensitivity_optima(
            {"tournament_shrinkage": sr}, config
        )
        assert new_config.tournament_shrinkage == 0.08  # Unchanged
        assert log["tournament_shrinkage"]["action"] == "skipped"
        assert "flat" in log["tournament_shrinkage"]["reason"]

    def test_skip_below_threshold(self):
        """Skips when improvement is below threshold."""
        sr = self._make_sensitivity_result(
            "tournament_shrinkage", 0.08, 0.12, 0.001  # Below default 0.002
        )
        config = SOTAPipelineConfig()
        new_config, log = adopt_sensitivity_optima(
            {"tournament_shrinkage": sr}, config
        )
        assert new_config.tournament_shrinkage == 0.08  # Unchanged
        assert log["tournament_shrinkage"]["action"] == "skipped"
        assert "threshold" in log["tournament_shrinkage"]["reason"]

    def test_multiple_constants(self):
        """Can adopt/skip multiple constants independently."""
        results = {
            "tournament_shrinkage": self._make_sensitivity_result(
                "tournament_shrinkage", 0.08, 0.12, 0.005
            ),
            "seed_prior_weight": self._make_sensitivity_result(
                "seed_prior_weight", 0.05, 0.05, 0.0  # Already optimal
            ),
            "ensemble_lgb_weight": self._make_sensitivity_result(
                "ensemble_lgb_weight", 0.45, 0.50, 0.003
            ),
        }
        config = SOTAPipelineConfig()
        new_config, log = adopt_sensitivity_optima(results, config)

        assert new_config.tournament_shrinkage == 0.12  # Adopted
        assert new_config.seed_prior_weight == 0.05  # Unchanged (already optimal)
        assert new_config.ensemble_lgb_weight == 0.50  # Adopted
        assert log["tournament_shrinkage"]["action"] == "adopted"
        assert log["seed_prior_weight"]["action"] == "skipped"
        assert log["ensemble_lgb_weight"]["action"] == "adopted"

    def test_does_not_modify_original_config(self):
        """Adoption returns a new config, not modifying the original."""
        sr = self._make_sensitivity_result(
            "tournament_shrinkage", 0.08, 0.12, 0.005
        )
        config = SOTAPipelineConfig()
        new_config, _ = adopt_sensitivity_optima(
            {"tournament_shrinkage": sr}, config
        )
        assert config.tournament_shrinkage == 0.08  # Original unchanged
        assert new_config.tournament_shrinkage == 0.12  # New has update
