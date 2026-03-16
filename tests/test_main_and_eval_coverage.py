"""Tests for uncovered functions in src/main.py and src/evaluation/ modules.

Covers:
- main.py utility functions: _resolve_multi_year_dir, _parse_year_list,
  _parse_float_list, _build_pipeline_config, _guard_production_2026
- evaluation/calibration_methods.py: CalibrationModel ABC, _EPS
- evaluation/baselines.py: SEED_WIN_RATES, seed_baseline_probability
- evaluation/bootstrap_metrics.py: brier_score, log_loss, expected_calibration_error
- evaluation/calibration_gate.py: module import, DEFAULT_THRESHOLDS
- evaluation/evaluation_report.py: module import
- evaluation/round_analysis.py: ROUND_NAME_MAP, SEED_MATCHUP_SEGMENTS, ROUND_SEGMENTS
- evaluation/seed_gap_calibration.py: module import
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.main import (
    _resolve_multi_year_dir,
    _parse_year_list,
    _parse_float_list,
    _build_pipeline_config,
    _guard_production_2026,
)
from src.governance.production_validator import ProductionValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Return a SimpleNamespace with default CLI-like args, applying overrides."""
    defaults = dict(
        year=2025, simulations=1000, pool_size=10, input=None, torvik=None,
        historical_games=None, sports_reference=None, public_picks=None,
        rosters=None, transfer_portal=None, scoring_rules=None,
        calibration="temperature", seed=42, scrape_live=False,
        cache_dir="data/raw", injury_noise_samples=100,
        allow_stale_feeds=False, max_feed_age_hours=168,
        min_public_sources=2, min_rapm_players_per_team=5,
        bracket_source="auto", bracket_json=None,
        multi_year_games_dir="auto", require_freeze=False,
        freeze_file=None, mc_calibration=None,
        enable_gnn=False, enable_transformer=False,
        enable_embedding_projections=False, kaggle_dir=None,
        model_complexity="simple", enable_bracket_portfolio=False,
        probability_profile="production", mode="calibration",
        output="output.json",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ===================================================================
# _resolve_multi_year_dir
# ===================================================================

class TestResolveMultiYearDir:
    def test_none_returns_none(self):
        assert _resolve_multi_year_dir(None) is None

    def test_string_none_returns_none(self):
        assert _resolve_multi_year_dir("none") is None

    def test_string_none_upper_returns_none(self):
        assert _resolve_multi_year_dir("None") is None

    def test_string_none_mixed_case_returns_none(self):
        assert _resolve_multi_year_dir("NONE") is None

    def test_auto_passes_through(self):
        assert _resolve_multi_year_dir("auto") == "auto"

    def test_path_passes_through(self):
        assert _resolve_multi_year_dir("/some/path") == "/some/path"

    def test_empty_string_passes_through(self):
        # empty string is not None and not "none", so it passes through
        assert _resolve_multi_year_dir("") == ""

    def test_relative_path_passes_through(self):
        assert _resolve_multi_year_dir("data/games") == "data/games"

    def test_integer_passes_through(self):
        # Non-string, non-None values pass through unchanged
        assert _resolve_multi_year_dir(42) == 42


# ===================================================================
# _parse_year_list
# ===================================================================

class TestParseYearList:
    def test_none_returns_none(self):
        assert _parse_year_list(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_year_list("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_year_list("   ") is None

    def test_single_year_string(self):
        assert _parse_year_list("2021") == [2021]

    def test_comma_separated_string(self):
        assert _parse_year_list("2021,2022") == [2021, 2022]

    def test_comma_separated_with_spaces(self):
        assert _parse_year_list("2021 , 2022 , 2023") == [2021, 2022, 2023]

    def test_list_of_ints(self):
        assert _parse_year_list([2021, 2022]) == [2021, 2022]

    def test_list_of_strings(self):
        assert _parse_year_list(["2021", "2022"]) == [2021, 2022]

    def test_trailing_comma_ignored(self):
        result = _parse_year_list("2021,2022,")
        assert result == [2021, 2022]

    def test_single_element_list(self):
        assert _parse_year_list([2025]) == [2025]

    def test_many_years(self):
        result = _parse_year_list("2018,2019,2020,2021,2022,2023,2024,2025")
        assert len(result) == 8
        assert result[0] == 2018
        assert result[-1] == 2025


# ===================================================================
# _parse_float_list
# ===================================================================

class TestParseFloatList:
    def test_none_returns_none(self):
        assert _parse_float_list(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_float_list("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_float_list("   ") is None

    def test_single_value(self):
        assert _parse_float_list("0.1") == [0.1]

    def test_comma_separated(self):
        result = _parse_float_list("0.1,0.2")
        assert len(result) == 2
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.2)

    def test_comma_separated_with_spaces(self):
        result = _parse_float_list("0.1 , 0.2 , 0.3")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_trailing_comma_ignored(self):
        result = _parse_float_list("0.5,0.6,")
        assert result == pytest.approx([0.5, 0.6])

    def test_integer_values_as_floats(self):
        result = _parse_float_list("1,2,3")
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_scientific_notation(self):
        result = _parse_float_list("1e-3,2.5e-2")
        assert result == pytest.approx([0.001, 0.025])


# ===================================================================
# _build_pipeline_config
# ===================================================================

class TestBuildPipelineConfig:
    def test_basic_config_fields(self):
        args = _make_args()
        config = _build_pipeline_config(args)
        assert config.year == 2025
        assert config.num_simulations == 1000
        assert config.random_seed == 42
        assert config.calibration_method == "temperature"

    def test_pool_size(self):
        args = _make_args(pool_size=50)
        config = _build_pipeline_config(args)
        assert config.pool_size == 50

    def test_scrape_live_false(self):
        args = _make_args(scrape_live=False)
        config = _build_pipeline_config(args)
        assert config.scrape_live is False

    def test_path_overrides_year(self):
        args = _make_args(year=2025)
        config = _build_pipeline_config(args, path_overrides={"year": 2024})
        assert config.year == 2024

    def test_path_overrides_teams_json(self):
        args = _make_args(input=None)
        config = _build_pipeline_config(args, path_overrides={"teams_json": "/data/teams.json"})
        assert config.teams_json == "/data/teams.json"

    def test_multi_year_dir_auto(self):
        args = _make_args(multi_year_games_dir="auto")
        config = _build_pipeline_config(args)
        assert config.multi_year_games_dir == "auto"

    def test_multi_year_dir_none_string(self):
        args = _make_args(multi_year_games_dir="none")
        config = _build_pipeline_config(args)
        assert config.multi_year_games_dir is None

    def test_gnn_disabled_by_default(self):
        args = _make_args()
        config = _build_pipeline_config(args)
        assert config.enable_gnn is False

    def test_mode_default_calibration(self):
        args = _make_args(mode="calibration")
        config = _build_pipeline_config(args)
        assert config.mode == "calibration"

    def test_enforce_feed_freshness_default(self):
        args = _make_args(allow_stale_feeds=False)
        config = _build_pipeline_config(args)
        assert config.enforce_feed_freshness is True

    def test_enforce_feed_freshness_stale_allowed(self):
        args = _make_args(allow_stale_feeds=True)
        config = _build_pipeline_config(args)
        assert config.enforce_feed_freshness is False

    def test_path_overrides_empty_dict(self):
        args = _make_args()
        config = _build_pipeline_config(args, path_overrides={})
        assert config.year == 2025

    def test_injury_noise_samples(self):
        args = _make_args(injury_noise_samples=500)
        config = _build_pipeline_config(args)
        assert config.injury_noise_samples == 500

    def test_bracket_source(self):
        args = _make_args(bracket_source="manual")
        config = _build_pipeline_config(args)
        assert config.bracket_source == "manual"

    def test_model_complexity(self):
        # Use research profile to avoid production path lock
        args = _make_args(model_complexity="ensemble", probability_profile="experimental")
        config = _build_pipeline_config(args)
        assert config.model_complexity == "ensemble"

    def test_enable_transformer(self):
        args = _make_args(enable_transformer=True, probability_profile="experimental")
        config = _build_pipeline_config(args)
        assert config.enable_transformer is True

    def test_enable_embedding_projections(self):
        args = _make_args(enable_embedding_projections=True, probability_profile="experimental")
        config = _build_pipeline_config(args)
        assert config.enable_embedding_projections is True

    def test_enable_bracket_portfolio(self):
        args = _make_args(enable_bracket_portfolio=True, probability_profile="experimental")
        config = _build_pipeline_config(args)
        assert config.enable_bracket_portfolio is True

    def test_probability_profile(self):
        args = _make_args(probability_profile="experimental")
        config = _build_pipeline_config(args)
        assert config.probability_profile == "experimental"

    def test_max_feed_age_hours(self):
        args = _make_args(max_feed_age_hours=72)
        config = _build_pipeline_config(args)
        assert config.max_feed_age_hours == 72

    def test_min_public_sources(self):
        args = _make_args(min_public_sources=3)
        config = _build_pipeline_config(args)
        assert config.min_public_sources == 3

    def test_require_freeze_file(self):
        args = _make_args(require_freeze=True)
        config = _build_pipeline_config(args)
        assert config.require_freeze_file is True

    def test_kaggle_dir(self):
        args = _make_args(kaggle_dir="/data/kaggle")
        config = _build_pipeline_config(args)
        assert config.kaggle_dir == "/data/kaggle"


# ===================================================================
# _guard_production_2026
# ===================================================================

class TestGuardProduction2026:
    def test_raises_on_production_2026(self):
        config = SimpleNamespace(probability_profile="production", year=2026)
        with pytest.raises(ProductionValidationError):
            _guard_production_2026(config)

    def test_no_raise_on_non_production(self):
        config = SimpleNamespace(probability_profile="experimental", year=2026)
        _guard_production_2026(config)  # should not raise

    def test_no_raise_on_non_2026(self):
        config = SimpleNamespace(probability_profile="production", year=2025)
        _guard_production_2026(config)  # should not raise

    def test_no_raise_on_both_different(self):
        config = SimpleNamespace(probability_profile="experimental", year=2025)
        _guard_production_2026(config)  # should not raise

    def test_error_message_content(self):
        config = SimpleNamespace(probability_profile="production", year=2026)
        with pytest.raises(ProductionValidationError, match="dedicated entrypoint"):
            _guard_production_2026(config)

    def test_no_raise_when_profile_missing(self):
        config = SimpleNamespace(year=2026)
        # No probability_profile attr; getattr returns None, not "production"
        _guard_production_2026(config)

    def test_no_raise_when_year_missing(self):
        config = SimpleNamespace(probability_profile="production")
        # No year attr; getattr returns None, not 2026
        _guard_production_2026(config)


# ===================================================================
# evaluation/calibration_methods.py
# ===================================================================

class TestCalibrationMethods:
    def test_eps_constant(self):
        from src.evaluation.calibration_methods import _EPS
        assert _EPS == pytest.approx(1e-7)

    def test_eps_is_positive(self):
        from src.evaluation.calibration_methods import _EPS
        assert _EPS > 0

    def test_calibration_model_abc_exists(self):
        from src.evaluation.calibration_methods import CalibrationModel
        assert hasattr(CalibrationModel, "fit")
        assert hasattr(CalibrationModel, "predict")

    def test_calibration_model_has_fitted_attr(self):
        from src.evaluation.calibration_methods import CalibrationModel
        assert CalibrationModel._fitted is False

    def test_calibration_model_cannot_instantiate(self):
        from src.evaluation.calibration_methods import CalibrationModel
        with pytest.raises(TypeError):
            CalibrationModel()

    def test_scipy_available_flag_exists(self):
        from src.evaluation.calibration_methods import SCIPY_AVAILABLE
        assert isinstance(SCIPY_AVAILABLE, bool)


# ===================================================================
# evaluation/baselines.py
# ===================================================================

class TestBaselines:
    def test_seed_win_rates_exists(self):
        from src.evaluation.baselines import SEED_WIN_RATES
        assert isinstance(SEED_WIN_RATES, dict)
        assert len(SEED_WIN_RATES) > 0

    def test_seed_win_rates_1v16(self):
        from src.evaluation.baselines import SEED_WIN_RATES
        assert SEED_WIN_RATES[(1, 16)] == pytest.approx(0.985)

    def test_seed_win_rates_8v9(self):
        from src.evaluation.baselines import SEED_WIN_RATES
        assert SEED_WIN_RATES[(8, 9)] == pytest.approx(0.510)

    def test_seed_win_rates_all_above_half(self):
        from src.evaluation.baselines import SEED_WIN_RATES
        for key, rate in SEED_WIN_RATES.items():
            assert rate >= 0.5, f"Rate for {key} should be >= 0.5, got {rate}"

    def test_seed_baseline_equal_seeds(self):
        from src.evaluation.baselines import seed_baseline_probability
        assert seed_baseline_probability(5, 5) == 0.5

    def test_seed_baseline_known_matchup(self):
        from src.evaluation.baselines import seed_baseline_probability
        assert seed_baseline_probability(1, 16) == pytest.approx(0.985)

    def test_seed_baseline_reverse_matchup(self):
        from src.evaluation.baselines import seed_baseline_probability
        assert seed_baseline_probability(16, 1) == pytest.approx(1.0 - 0.985)

    def test_seed_baseline_non_standard_matchup(self):
        from src.evaluation.baselines import seed_baseline_probability
        # Not in lookup table, uses logistic approximation
        p = seed_baseline_probability(1, 8)
        assert 0.5 < p < 1.0

    def test_seed_baseline_symmetry(self):
        from src.evaluation.baselines import seed_baseline_probability
        p12 = seed_baseline_probability(3, 11)
        p21 = seed_baseline_probability(11, 3)
        assert p12 + p21 == pytest.approx(1.0)


# ===================================================================
# evaluation/bootstrap_metrics.py
# ===================================================================

class TestBootstrapMetrics:
    def test_brier_score_perfect(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0])
        assert brier_score(preds, outcomes) == pytest.approx(0.0)

    def test_brier_score_worst(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        assert brier_score(preds, outcomes) == pytest.approx(1.0)

    def test_brier_score_midpoint(self):
        from src.evaluation.bootstrap_metrics import brier_score
        preds = np.array([0.5, 0.5])
        outcomes = np.array([1.0, 0.0])
        assert brier_score(preds, outcomes) == pytest.approx(0.25)

    def test_log_loss_perfect_approx(self):
        from src.evaluation.bootstrap_metrics import log_loss
        preds = np.array([0.999, 0.001])
        outcomes = np.array([1.0, 0.0])
        result = log_loss(preds, outcomes)
        assert result < 0.01

    def test_log_loss_bad_predictions(self):
        from src.evaluation.bootstrap_metrics import log_loss
        preds = np.array([0.1, 0.9])
        outcomes = np.array([1.0, 0.0])
        result = log_loss(preds, outcomes)
        assert result > 1.0

    def test_log_loss_clips_extreme_values(self):
        from src.evaluation.bootstrap_metrics import log_loss
        # Should not raise even with 0.0 and 1.0 predictions
        preds = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        result = log_loss(preds, outcomes)
        assert math.isfinite(result)

    def test_ece_perfectly_calibrated(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        preds = np.array([0.5] * 100)
        outcomes = np.array([1.0] * 50 + [0.0] * 50)
        ece = expected_calibration_error(preds, outcomes)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_ece_empty_arrays(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        preds = np.array([])
        outcomes = np.array([])
        assert expected_calibration_error(preds, outcomes) == 0.0

    def test_ece_non_negative(self):
        from src.evaluation.bootstrap_metrics import expected_calibration_error
        rng = np.random.RandomState(42)
        preds = rng.uniform(0, 1, 200)
        outcomes = (rng.uniform(0, 1, 200) > 0.5).astype(float)
        ece = expected_calibration_error(preds, outcomes)
        assert ece >= 0.0

    def test_accuracy_function(self):
        from src.evaluation.bootstrap_metrics import accuracy
        preds = np.array([0.9, 0.1, 0.8, 0.3])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert accuracy(preds, outcomes) == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        from src.evaluation.bootstrap_metrics import accuracy
        preds = np.array([0.1, 0.9])
        outcomes = np.array([1.0, 0.0])
        assert accuracy(preds, outcomes) == pytest.approx(0.0)

    def test_bootstrap_result_dataclass(self):
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(
            estimate=0.25, ci_lower=0.20, ci_upper=0.30,
            ci_level=0.95, n_bootstrap=1000, n_samples=63,
        )
        d = br.to_dict()
        assert d["estimate"] == 0.25
        assert d["ci_level"] == 0.95
        assert d["n_samples"] == 63

    def test_bootstrap_result_frozen(self):
        from src.evaluation.bootstrap_metrics import BootstrapResult
        br = BootstrapResult(
            estimate=0.25, ci_lower=0.20, ci_upper=0.30,
            ci_level=0.95, n_bootstrap=1000, n_samples=63,
        )
        with pytest.raises(AttributeError):
            br.estimate = 0.5

    def test_upset_recall_no_upsets(self):
        from src.evaluation.bootstrap_metrics import upset_recall
        preds = np.array([0.9, 0.8])
        outcomes = np.array([1.0, 1.0])  # no upsets
        is_fav = np.array([True, True])
        assert upset_recall(preds, outcomes, is_fav) == 0.0

    def test_upset_recall_all_caught(self):
        from src.evaluation.bootstrap_metrics import upset_recall
        preds = np.array([0.3, 0.4])  # both predict upset
        outcomes = np.array([0.0, 0.0])  # both are upsets
        is_fav = np.array([True, True])
        assert upset_recall(preds, outcomes, is_fav) == pytest.approx(1.0)


# ===================================================================
# evaluation/calibration_gate.py
# ===================================================================

class TestCalibrationGateImport:
    def test_module_imports(self):
        import src.evaluation.calibration_gate  # noqa: F401

    def test_calibration_gate_result_exists(self):
        from src.evaluation.calibration_gate import CalibrationGateResult
        assert CalibrationGateResult is not None

    def test_default_thresholds_exist(self):
        from src.evaluation.calibration_gate import DEFAULT_THRESHOLDS
        assert isinstance(DEFAULT_THRESHOLDS, dict)
        assert "max_brier" in DEFAULT_THRESHOLDS
        assert "max_log_loss" in DEFAULT_THRESHOLDS
        assert "max_ece" in DEFAULT_THRESHOLDS

    def test_gate_check_result_exists(self):
        from src.evaluation.calibration_gate import GateCheckResult
        assert GateCheckResult is not None


# ===================================================================
# evaluation/evaluation_report.py
# ===================================================================

class TestEvaluationReportImport:
    def test_module_imports(self):
        import src.evaluation.evaluation_report  # noqa: F401

    def test_collect_library_versions(self):
        from src.evaluation.evaluation_report import collect_library_versions
        versions = collect_library_versions()
        assert isinstance(versions, dict)
        assert "python" in versions


# ===================================================================
# evaluation/round_analysis.py
# ===================================================================

class TestRoundAnalysis:
    def test_round_name_map_exists(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert isinstance(ROUND_NAME_MAP, dict)

    def test_round_name_map_r64(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["R64"] == "R64"
        assert ROUND_NAME_MAP["Round of 64"] == "R64"

    def test_round_name_map_r32(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["R32"] == "R32"
        assert ROUND_NAME_MAP["Round of 32"] == "R32"

    def test_round_name_map_sweet16(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["S16"] == "S16"
        assert ROUND_NAME_MAP["Sweet 16"] == "S16"

    def test_round_name_map_elite8(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["E8"] == "E8"
        assert ROUND_NAME_MAP["Elite 8"] == "E8"

    def test_round_name_map_final_four(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["F4"] == "F4"
        assert ROUND_NAME_MAP["Final Four"] == "F4"

    def test_round_name_map_championship(self):
        from src.evaluation.round_analysis import ROUND_NAME_MAP
        assert ROUND_NAME_MAP["NCG"] == "NCG"
        assert ROUND_NAME_MAP["Championship"] == "NCG"
        assert ROUND_NAME_MAP["CHAMP"] == "NCG"

    def test_seed_matchup_segments_exist(self):
        from src.evaluation.round_analysis import SEED_MATCHUP_SEGMENTS
        assert "heavy_favorites" in SEED_MATCHUP_SEGMENTS
        assert "mid_tier" in SEED_MATCHUP_SEGMENTS
        assert "toss_ups" in SEED_MATCHUP_SEGMENTS

    def test_round_segments_exist(self):
        from src.evaluation.round_analysis import ROUND_SEGMENTS
        assert "round_of_64" in ROUND_SEGMENTS
        assert "championship" in ROUND_SEGMENTS

    def test_eval_game_dataclass(self):
        from src.evaluation.round_analysis import EvalGame
        game = EvalGame(
            team1_id="Duke", team2_id="UNC",
            team1_seed=2, team2_seed=3,
            round_name="S16", prediction=0.55, outcome=1.0,
        )
        assert game.canonical_round == "S16"


# ===================================================================
# evaluation/seed_gap_calibration.py
# ===================================================================

class TestSeedGapCalibrationImport:
    def test_module_imports(self):
        import src.evaluation.seed_gap_calibration  # noqa: F401

    def test_calibration_bucket_exists(self):
        from src.evaluation.seed_gap_calibration import CalibrationBucket
        assert CalibrationBucket is not None

    def test_seed_gap_calibration_report_exists(self):
        from src.evaluation.seed_gap_calibration import SeedGapCalibrationReport
        assert SeedGapCalibrationReport is not None
