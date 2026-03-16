"""Comprehensive unit tests for src/main.py to maximize coverage.

Tests all utility/helper functions, argument parsing, configuration building,
data validation, and command dispatch logic using extensive mocking to avoid I/O.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: import the functions under test
# ---------------------------------------------------------------------------

from src.main import (
    _resolve_multi_year_dir,
    _parse_year_list,
    _parse_float_list,
    _build_pipeline_config,
    _run_pipeline_and_report,
    _guard_production_2026,
    _load_manifest,
    _resolve_manifest_paths,
    run_sota,
    run_production_2026_cmd,
    run_sota_from_manifest,
    run_calibrate_mc,
    freeze_pipeline_cmd,
    verify_freeze_cmd,
    run_scrape_conference_seeds,
    run_conference_tournaments,
    scrape_rosters,
    enrich_rosters,
    download_kaggle,
    scrape_tournament_results,
    repair_dates,
    main,
)


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_multi_year_dir
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveMultiYearDir:
    def test_none_returns_none(self):
        assert _resolve_multi_year_dir(None) is None

    def test_none_string_returns_none(self):
        assert _resolve_multi_year_dir("none") is None

    def test_none_string_case_insensitive(self):
        assert _resolve_multi_year_dir("None") is None
        assert _resolve_multi_year_dir("NONE") is None
        assert _resolve_multi_year_dir("NoNe") is None

    def test_auto_passes_through(self):
        assert _resolve_multi_year_dir("auto") == "auto"

    def test_literal_path_passes_through(self):
        assert _resolve_multi_year_dir("/some/path") == "/some/path"

    def test_empty_string_passes_through(self):
        # Empty string is not "none" and not None
        assert _resolve_multi_year_dir("") == ""

    def test_integer_passes_through(self):
        assert _resolve_multi_year_dir(42) == 42


# ═══════════════════════════════════════════════════════════════════════════
# _parse_year_list
# ═══════════════════════════════════════════════════════════════════════════

class TestParseYearList:
    def test_none_returns_none(self):
        assert _parse_year_list(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_year_list("") is None

    def test_whitespace_returns_none(self):
        assert _parse_year_list("   ") is None

    def test_single_year(self):
        assert _parse_year_list("2025") == [2025]

    def test_comma_separated(self):
        assert _parse_year_list("2022,2023,2024") == [2022, 2023, 2024]

    def test_spaces_around_commas(self):
        assert _parse_year_list(" 2022 , 2023 , 2024 ") == [2022, 2023, 2024]

    def test_list_input(self):
        assert _parse_year_list([2022, 2023]) == [2022, 2023]

    def test_list_of_strings(self):
        assert _parse_year_list(["2022", "2023"]) == [2022, 2023]

    def test_trailing_comma_handled(self):
        assert _parse_year_list("2022,2023,") == [2022, 2023]

    def test_leading_comma_handled(self):
        assert _parse_year_list(",2022") == [2022]

    def test_integer_input(self):
        # str(2025) = "2025"
        assert _parse_year_list(2025) == [2025]


# ═══════════════════════════════════════════════════════════════════════════
# _parse_float_list
# ═══════════════════════════════════════════════════════════════════════════

class TestParseFloatList:
    def test_none_returns_none(self):
        assert _parse_float_list(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_float_list("") is None

    def test_whitespace_returns_none(self):
        assert _parse_float_list("   ") is None

    def test_single_float(self):
        assert _parse_float_list("0.5") == [0.5]

    def test_comma_separated(self):
        result = _parse_float_list("0.06,0.08,0.10")
        assert len(result) == 3
        assert abs(result[0] - 0.06) < 1e-9
        assert abs(result[1] - 0.08) < 1e-9
        assert abs(result[2] - 0.10) < 1e-9

    def test_spaces_around_commas(self):
        result = _parse_float_list(" 1.0 , 2.5 , 3.0 ")
        assert result == [1.0, 2.5, 3.0]

    def test_trailing_comma(self):
        result = _parse_float_list("1.0,2.0,")
        assert result == [1.0, 2.0]

    def test_integer_input_as_string(self):
        assert _parse_float_list("5") == [5.0]


# ═══════════════════════════════════════════════════════════════════════════
# _build_pipeline_config
# ═══════════════════════════════════════════════════════════════════════════

def _make_args(**overrides):
    """Create a minimal args namespace for _build_pipeline_config."""
    defaults = dict(
        year=2026,
        simulations=1000,
        pool_size=100,
        input=None,
        torvik=None,
        historical_games=None,
        sports_reference=None,
        public_picks=None,
        rosters=None,
        transfer_portal=None,
        scoring_rules=None,
        calibration="temperature",
        seed=2026,
        scrape_live=False,
        cache_dir="data/raw",
        injury_noise_samples=10000,
        allow_stale_feeds=False,
        max_feed_age_hours=168,
        min_public_sources=2,
        min_rapm_players_per_team=5,
        bracket_source="auto",
        bracket_json=None,
        multi_year_games_dir="auto",
        require_freeze=False,
        freeze_file=None,
        mc_calibration=None,
        enable_gnn=False,
        enable_transformer=False,
        enable_embedding_projections=False,
        kaggle_dir=None,
        model_complexity="simple",
        enable_bracket_portfolio=False,
        probability_profile="production",
        mode="calibration",
        dev_years=None,
        holdout_years=None,
        calibration_years=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildPipelineConfig:
    @patch("src.main.SOTAPipelineConfig")
    def test_basic_config_build(self, mock_config_cls):
        args = _make_args()
        _build_pipeline_config(args)
        mock_config_cls.assert_called_once()
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["year"] == 2026
        assert kwargs["num_simulations"] == 1000

    @patch("src.main.SOTAPipelineConfig")
    def test_path_overrides_take_precedence(self, mock_config_cls):
        args = _make_args(year=2025)
        overrides = {"year": 2024, "teams_json": "/override/teams.json"}
        _build_pipeline_config(args, path_overrides=overrides)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["year"] == 2024
        assert kwargs["teams_json"] == "/override/teams.json"

    @patch("src.main.SOTAPipelineConfig")
    def test_dev_years_parsed(self, mock_config_cls):
        args = _make_args(dev_years="2022,2023")
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["dev_years"] == [2022, 2023]

    @patch("src.main.SOTAPipelineConfig")
    def test_holdout_years_parsed(self, mock_config_cls):
        args = _make_args(holdout_years="2025")
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["holdout_years"] == [2025]

    @patch("src.main.SOTAPipelineConfig")
    def test_calibration_years_parsed(self, mock_config_cls):
        args = _make_args(calibration_years="2023,2024")
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["calibration_years"] == [2023, 2024]

    @patch("src.main.SOTAPipelineConfig")
    def test_none_dev_years_not_in_kwargs(self, mock_config_cls):
        args = _make_args(dev_years=None)
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert "dev_years" not in kwargs

    @patch("src.main.SOTAPipelineConfig")
    def test_multi_year_games_dir_none(self, mock_config_cls):
        args = _make_args(multi_year_games_dir="none")
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["multi_year_games_dir"] is None

    @patch("src.main.SOTAPipelineConfig")
    def test_allow_stale_feeds_inverts(self, mock_config_cls):
        args = _make_args(allow_stale_feeds=True)
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["enforce_feed_freshness"] is False

    @patch("src.main.SOTAPipelineConfig")
    def test_extra_path_overrides_applied(self, mock_config_cls):
        args = _make_args()
        overrides = {
            "preseason_ap_json": "/ap.json",
            "coach_tournament_json": "/coach.json",
            "conf_champions_json": "/conf.json",
            "betting_odds_json": "/odds.json",
        }
        _build_pipeline_config(args, path_overrides=overrides)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["preseason_ap_json"] == "/ap.json"
        assert kwargs["coach_tournament_json"] == "/coach.json"
        assert kwargs["conf_champions_json"] == "/conf.json"
        assert kwargs["betting_odds_json"] == "/odds.json"

    @patch("src.main.SOTAPipelineConfig")
    def test_enable_gnn_true(self, mock_config_cls):
        args = _make_args(enable_gnn=True)
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["enable_gnn"] is True

    @patch("src.main.SOTAPipelineConfig")
    def test_enable_transformer_true(self, mock_config_cls):
        args = _make_args(enable_transformer=True)
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["enable_transformer"] is True

    @patch("src.main.SOTAPipelineConfig")
    def test_model_complexity_full(self, mock_config_cls):
        args = _make_args(model_complexity="full")
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["model_complexity"] == "full"

    @patch("src.main.SOTAPipelineConfig")
    def test_enable_bracket_portfolio(self, mock_config_cls):
        args = _make_args(enable_bracket_portfolio=True)
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["enable_bracket_portfolio"] is True

    @patch("src.main.SOTAPipelineConfig")
    def test_require_freeze_true(self, mock_config_cls):
        args = _make_args(require_freeze=True, freeze_file="freeze.json")
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["require_freeze_file"] is True
        assert kwargs["freeze_file"] == "freeze.json"

    @patch("src.main.SOTAPipelineConfig")
    def test_missing_optional_attrs_use_getattr_defaults(self, mock_config_cls):
        # args with NO pool_size attribute
        args = argparse.Namespace(
            year=2026, simulations=500, calibration="none", seed=42,
            scrape_live=False, cache_dir="cache",
        )
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        assert kwargs["pool_size"] == 100  # getattr default


# ═══════════════════════════════════════════════════════════════════════════
# _run_pipeline_and_report
# ═══════════════════════════════════════════════════════════════════════════

class TestRunPipelineAndReport:
    @patch("src.main.run_sota_pipeline_to_file")
    def test_success(self, mock_run):
        mock_run.return_value = {
            "artifacts": {
                "pool_recommendation": "chalk",
                "simulation": {"num_simulations": 50000},
            }
        }
        config = MagicMock()
        exit_code, report = _run_pipeline_and_report(config, "out.json")
        assert exit_code == 0
        assert report is not None
        assert report["artifacts"]["pool_recommendation"] == "chalk"

    @patch("src.main.run_sota_pipeline_to_file")
    def test_data_requirement_error(self, mock_run):
        from src.pipeline.sota import DataRequirementError
        mock_run.side_effect = DataRequirementError("missing data")
        config = MagicMock()
        exit_code, report = _run_pipeline_and_report(config, "out.json")
        assert exit_code == 1
        assert report is None


# ═══════════════════════════════════════════════════════════════════════════
# _guard_production_2026
# ═══════════════════════════════════════════════════════════════════════════

class TestGuardProduction2026:
    def test_production_2026_raises(self):
        from src.governance.production_validator import ProductionValidationError
        config = SimpleNamespace(probability_profile="production", year=2026)
        with pytest.raises(ProductionValidationError):
            _guard_production_2026(config)

    def test_experimental_2026_ok(self):
        config = SimpleNamespace(probability_profile="experimental", year=2026)
        _guard_production_2026(config)  # should not raise

    def test_production_2025_ok(self):
        config = SimpleNamespace(probability_profile="production", year=2025)
        _guard_production_2026(config)  # should not raise

    def test_no_profile_ok(self):
        config = SimpleNamespace(year=2026)
        _guard_production_2026(config)  # should not raise

    def test_no_year_ok(self):
        config = SimpleNamespace(probability_profile="production")
        _guard_production_2026(config)  # should not raise

    def test_both_none_ok(self):
        config = SimpleNamespace()
        _guard_production_2026(config)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# _load_manifest
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadManifest:
    def test_file_not_found(self, tmp_path):
        manifest, base_dir = _load_manifest(str(tmp_path / "nonexistent.json"))
        assert manifest is None
        assert base_dir is None

    def test_valid_manifest(self, tmp_path):
        manifest_data = {"artifacts": {"teams_json": "teams.json"}, "year": 2026}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        manifest, base_dir = _load_manifest(str(manifest_path))
        assert manifest is not None
        assert manifest["year"] == 2026
        assert base_dir == tmp_path

    def test_missing_artifacts_key_defaults_to_empty(self, tmp_path):
        manifest_data = {"year": 2026}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        manifest, base_dir = _load_manifest(str(manifest_path))
        # .get("artifacts", {}) returns {} which IS a dict, so it passes
        assert manifest is not None
        assert base_dir == tmp_path

    def test_artifacts_not_dict(self, tmp_path):
        manifest_data = {"artifacts": "not_a_dict", "year": 2026}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        manifest, base_dir = _load_manifest(str(manifest_path))
        assert manifest is None
        assert base_dir is None


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_manifest_paths
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveManifestPaths:
    def test_basic_resolution(self, tmp_path):
        args = argparse.Namespace(
            year=None, input=None, torvik=None, historical_games=None,
            sports_reference=None, public_picks=None, rosters=None,
            transfer_portal=None, preseason_ap=None, coach_tournament=None,
            conf_champions=None, betting_odds=None, scoring_rules=None,
        )
        manifest = {
            "year": 2026,
            "artifacts": {
                "teams_json": "teams.json",
                "torvik_json": "torvik.json",
            },
        }
        result = _resolve_manifest_paths(args, manifest, tmp_path)
        assert result["year"] == 2026
        assert result["teams_json"] == str((tmp_path / "teams.json").resolve())
        assert result["torvik_json"] == str((tmp_path / "torvik.json").resolve())

    def test_cli_override_takes_precedence(self, tmp_path):
        args = argparse.Namespace(
            year=2025, input="/override/teams.json", torvik=None,
            historical_games=None, sports_reference=None, public_picks=None,
            rosters=None, transfer_portal=None, preseason_ap=None,
            coach_tournament=None, conf_champions=None, betting_odds=None,
            scoring_rules=None,
        )
        manifest = {"year": 2026, "artifacts": {"teams_json": "teams.json"}}
        result = _resolve_manifest_paths(args, manifest, tmp_path)
        assert result["year"] == 2025
        assert result["teams_json"] == str(Path("/override/teams.json").resolve())

    def test_absolute_path_in_manifest(self, tmp_path):
        args = argparse.Namespace(
            year=None, input=None, torvik=None, historical_games=None,
            sports_reference=None, public_picks=None, rosters=None,
            transfer_portal=None, preseason_ap=None, coach_tournament=None,
            conf_champions=None, betting_odds=None, scoring_rules=None,
        )
        manifest = {
            "year": 2026,
            "artifacts": {"teams_json": "/absolute/path/teams.json"},
        }
        result = _resolve_manifest_paths(args, manifest, tmp_path)
        assert result["teams_json"] == "/absolute/path/teams.json"

    def test_empty_artifacts(self, tmp_path):
        args = argparse.Namespace(
            year=None, input=None, torvik=None, historical_games=None,
            sports_reference=None, public_picks=None, rosters=None,
            transfer_portal=None, preseason_ap=None, coach_tournament=None,
            conf_champions=None, betting_odds=None, scoring_rules=None,
        )
        manifest = {"year": 2024, "artifacts": {}}
        result = _resolve_manifest_paths(args, manifest, tmp_path)
        assert result["year"] == 2024
        assert result["teams_json"] is None
        assert result["torvik_json"] is None

    def test_odds_json_maps_to_betting_odds(self, tmp_path):
        args = argparse.Namespace(
            year=None, input=None, torvik=None, historical_games=None,
            sports_reference=None, public_picks=None, rosters=None,
            transfer_portal=None, preseason_ap=None, coach_tournament=None,
            conf_champions=None, betting_odds=None, scoring_rules=None,
        )
        manifest = {"year": 2026, "artifacts": {"odds_json": "odds.json"}}
        result = _resolve_manifest_paths(args, manifest, tmp_path)
        assert result["betting_odds_json"] == str((tmp_path / "odds.json").resolve())


# ═══════════════════════════════════════════════════════════════════════════
# run_sota
# ═══════════════════════════════════════════════════════════════════════════

class TestRunSota:
    @patch("src.main._run_pipeline_and_report")
    @patch("src.main._guard_production_2026")
    @patch("src.main._build_pipeline_config")
    def test_run_sota_normal(self, mock_build, mock_guard, mock_run):
        mock_build.return_value = MagicMock()
        mock_run.return_value = (0, {"result": True})
        args = _make_args(output="out.json", multi_agent=False)
        code = run_sota(args)
        assert code == 0
        mock_guard.assert_called_once()
        mock_run.assert_called_once()

    @patch("src.main.SOTAPipeline")
    @patch("src.main._guard_production_2026")
    @patch("src.main._build_pipeline_config")
    def test_run_sota_multi_agent(self, mock_build, mock_guard, mock_pipeline_cls):
        mock_build.return_value = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.run_multi_agent.return_value = {"key": "value"}
        mock_pipeline_cls.return_value = mock_pipeline
        args = _make_args(output="out.json", multi_agent=True)
        with patch.dict("sys.modules", {"src.pipeline.sota": MagicMock(SOTAPipeline=mock_pipeline_cls)}):
            code = run_sota(args)
        assert code == 0


# ═══════════════════════════════════════════════════════════════════════════
# run_production_2026_cmd
# ═══════════════════════════════════════════════════════════════════════════

class TestRunProduction2026Cmd:
    def test_wrong_config_path(self, tmp_path):
        args = argparse.Namespace(
            config=str(tmp_path / "wrong.json"),
            output="out.json",
            freeze_manifest="freeze.json",
            governance_report="gov.json",
        )
        code = run_production_2026_cmd(args)
        assert code == 1

    @patch("src.main.run_production_2026")
    def test_successful_run(self, mock_run, tmp_path):
        # Create the blessed config path
        blessed = Path("configs/production_2026.json").resolve()
        mock_run.return_value = (
            {"production_path_verification": {"probability_profile": "production"}},
            {},
            {},
        )
        args = argparse.Namespace(
            config=str(blessed),
            output="out.json",
            freeze_manifest="freeze.json",
            governance_report="gov.json",
        )
        code = run_production_2026_cmd(args)
        assert code == 0

    @patch("src.main.run_production_2026")
    def test_production_validation_error(self, mock_run):
        from src.governance.production_validator import ProductionValidationError
        blessed = Path("configs/production_2026.json").resolve()
        mock_run.side_effect = ProductionValidationError("bad config")
        args = argparse.Namespace(
            config=str(blessed),
            output="out.json",
            freeze_manifest="freeze.json",
            governance_report="gov.json",
        )
        code = run_production_2026_cmd(args)
        assert code == 1

    @patch("src.main.run_production_2026")
    def test_data_requirement_error(self, mock_run):
        from src.pipeline.sota import DataRequirementError
        blessed = Path("configs/production_2026.json").resolve()
        mock_run.side_effect = DataRequirementError("missing feed")
        args = argparse.Namespace(
            config=str(blessed),
            output="out.json",
            freeze_manifest="freeze.json",
            governance_report="gov.json",
        )
        code = run_production_2026_cmd(args)
        assert code == 1


# ═══════════════════════════════════════════════════════════════════════════
# run_sota_from_manifest
# ═══════════════════════════════════════════════════════════════════════════

class TestRunSotaFromManifest:
    @patch("src.main._run_pipeline_and_report")
    @patch("src.main._guard_production_2026")
    @patch("src.main._build_pipeline_config")
    @patch("src.main._resolve_manifest_paths")
    @patch("src.main._load_manifest")
    def test_success(self, mock_load, mock_resolve, mock_build, mock_guard, mock_run):
        mock_load.return_value = ({"artifacts": {}, "year": 2026}, Path("/base"))
        mock_resolve.return_value = {"year": 2026}
        mock_build.return_value = MagicMock()
        mock_run.return_value = (0, {})
        args = _make_args(manifest="m.json", output="out.json")
        code = run_sota_from_manifest(args)
        assert code == 0

    @patch("src.main._load_manifest")
    def test_manifest_not_found(self, mock_load):
        mock_load.return_value = (None, None)
        args = _make_args(manifest="bad.json", output="out.json")
        code = run_sota_from_manifest(args)
        assert code == 1


# ═══════════════════════════════════════════════════════════════════════════
# run_calibrate_mc
# ═══════════════════════════════════════════════════════════════════════════

class TestRunCalibrateMc:
    @patch("builtins.open", mock_open())
    @patch("json.dump")
    @patch("src.simulation.mc_calibration.calibrate_mc_parameters")
    def test_basic_run(self, mock_calibrate, mock_json_dump):
        mock_calibrate.return_value = {
            "best_params": {"noise_std": 0.08, "regional_correlation": 0.0},
            "best_dev_score": 0.2,
            "holdout_score": 0.21,
        }
        args = argparse.Namespace(
            dev_years="2022,2023",
            holdout_years="2025",
            noise_grid=None,
            corr_grid=None,
            tune_regional_correlation=False,
            historical_dir="data",
            simulations=100,
            em_slope=0.17,
            seed_slope=0.175,
            seed=42,
            parallel_workers=None,
            output="mc.json",
        )
        code = run_calibrate_mc(args)
        assert code == 0
        mock_calibrate.assert_called_once()

    @patch("builtins.open", mock_open())
    @patch("json.dump")
    @patch("src.simulation.mc_calibration.calibrate_mc_parameters")
    def test_with_regional_correlation(self, mock_calibrate, mock_json_dump):
        mock_calibrate.return_value = {
            "best_params": {"noise_std": 0.08, "regional_correlation": 0.05},
            "best_dev_score": 0.2,
            "holdout_score": 0.21,
        }
        args = argparse.Namespace(
            dev_years=None,
            holdout_years=None,
            noise_grid="0.06,0.08",
            corr_grid="0.0,0.05",
            tune_regional_correlation=True,
            historical_dir="data",
            simulations=100,
            em_slope=0.17,
            seed_slope=0.175,
            seed=42,
            parallel_workers=2,
            output="mc.json",
        )
        code = run_calibrate_mc(args)
        assert code == 0
        call_kwargs = mock_calibrate.call_args[1]
        assert call_kwargs["corr_grid"] == [0.0, 0.05]


# ═══════════════════════════════════════════════════════════════════════════
# freeze_pipeline_cmd / verify_freeze_cmd
# ═══════════════════════════════════════════════════════════════════════════

class TestFreezePipelineCmd:
    @patch("src.main.freeze_pipeline")
    @patch("src.main.SOTAPipelineConfig")
    def test_basic_freeze(self, mock_config, mock_freeze):
        mock_freeze.return_value = {
            "config_hash": "abc123",
            "git_tag": "v1.0",
            "git_dirty": False,
        }
        args = argparse.Namespace(output="freeze.json", mc_calibration=None)
        code = freeze_pipeline_cmd(args)
        assert code == 0

    @patch("src.main.freeze_pipeline")
    @patch("src.main.SOTAPipelineConfig")
    def test_freeze_with_mc_calibration(self, mock_config, mock_freeze):
        mock_freeze.return_value = {
            "config_hash": "abc123",
            "git_tag": None,
            "git_dirty": True,
        }
        args = argparse.Namespace(output="freeze.json", mc_calibration="mc.json")
        code = freeze_pipeline_cmd(args)
        assert code == 0
        mock_config.assert_called_once_with(mc_calibration_json="mc.json")


class TestVerifyFreezeCmd:
    @patch("src.main.verify_freeze")
    @patch("src.main.SOTAPipelineConfig")
    def test_verified_match(self, mock_config, mock_verify):
        mock_verify.return_value = {
            "matches": True,
            "frozen_timestamp": "2026-01-01",
            "current_hash": "abc",
        }
        args = argparse.Namespace(freeze_file="freeze.json")
        code = verify_freeze_cmd(args)
        assert code == 0

    @patch("src.main.verify_freeze")
    @patch("src.main.SOTAPipelineConfig")
    def test_mismatch(self, mock_config, mock_verify):
        mock_verify.return_value = {
            "matches": False,
            "frozen_timestamp": "2026-01-01",
            "current_hash": "xyz",
            "mismatches": ["field1 changed from A to B"],
        }
        args = argparse.Namespace(freeze_file="freeze.json")
        code = verify_freeze_cmd(args)
        assert code == 1


# ═══════════════════════════════════════════════════════════════════════════
# scrape_rosters
# ═══════════════════════════════════════════════════════════════════════════

class TestScrapeRosters:
    @patch("src.main.CBBpyRosterScraper")
    def test_cached_skip(self, mock_scraper_cls, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create cached files
        (cache_dir / "cbbpy_rosters_2025.json").write_text("{}")
        args = argparse.Namespace(
            start_year=2025, end_year=2025, cache_dir=str(cache_dir),
            delay=0, force=False,
        )
        code = scrape_rosters(args)
        assert code == 0

    @patch("src.main.CBBpyRosterScraper")
    def test_scrape_success(self, mock_scraper_cls, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_scraper = MagicMock()
        mock_scraper.fetch_rosters.return_value = {
            "teams": [{"name": "Duke", "players": [{"name": "Player1"}]}]
        }
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            start_year=2025, end_year=2025, cache_dir=str(cache_dir),
            delay=0, force=True,
        )
        code = scrape_rosters(args)
        assert code == 0

    @patch("src.main.CBBpyRosterScraper")
    def test_scrape_zero_teams(self, mock_scraper_cls, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_scraper = MagicMock()
        mock_scraper.fetch_rosters.return_value = {"teams": []}
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            start_year=2025, end_year=2025, cache_dir=str(cache_dir),
            delay=0, force=True,
        )
        code = scrape_rosters(args)
        assert code == 1  # failed because 0 teams

    @patch("src.main.CBBpyRosterScraper")
    def test_scrape_exception(self, mock_scraper_cls, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_scraper = MagicMock()
        mock_scraper.fetch_rosters.side_effect = RuntimeError("network error")
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            start_year=2025, end_year=2025, cache_dir=str(cache_dir),
            delay=0, force=True,
        )
        code = scrape_rosters(args)
        assert code == 1


# ═══════════════════════════════════════════════════════════════════════════
# enrich_rosters
# ═══════════════════════════════════════════════════════════════════════════

class TestEnrichRosters:
    @patch("src.main.RosterEnrichment")
    def test_enrich_basic(self, mock_enrichment_cls):
        mock_enricher = MagicMock()
        mock_enricher.enrich_all.return_value = {
            "total_players_enriched": 100,
            "years_processed": 5,
            "total_transfers": 20,
            "eligibility_distribution": {"FR": 40, "SO": 30, "JR": 20, "SR": 10},
        }
        mock_enrichment_cls.return_value = mock_enricher
        args = argparse.Namespace(
            roster_dir="data/rosters",
            output_dir="data/enriched",
            start_year=2020,
            end_year=2025,
        )
        code = enrich_rosters(args)
        assert code == 0
        mock_enricher.enrich_all.assert_called_once_with(start_year=2020, end_year=2025)


# ═══════════════════════════════════════════════════════════════════════════
# download_kaggle
# ═══════════════════════════════════════════════════════════════════════════

class TestDownloadKaggle:
    @patch("src.data.kaggle_downloader.download_competition_data")
    def test_download_success(self, mock_download):
        mock_download.return_value = "/data/kaggle"
        args = argparse.Namespace(
            output_dir="/data/kaggle", force=False,
            verify_season=None, competition=None,
        )
        code = download_kaggle(args)
        assert code == 0

    @patch("src.data.kaggle_downloader.download_competition_data")
    def test_download_failure(self, mock_download):
        mock_download.return_value = None
        args = argparse.Namespace(
            output_dir="/data/kaggle", force=False,
            verify_season=None, competition=None,
        )
        code = download_kaggle(args)
        assert code == 1

    @patch("src.data.kaggle_downloader.verify_massey_ordinals")
    @patch("src.data.kaggle_downloader.download_competition_data")
    def test_download_with_verify_ok(self, mock_download, mock_verify):
        mock_download.return_value = "/data/kaggle"
        mock_verify.return_value = {"status": "ok", "system_names": ["RPI", "SAG"]}
        args = argparse.Namespace(
            output_dir="/data/kaggle", force=False,
            verify_season=2025, competition=None,
        )
        code = download_kaggle(args)
        assert code == 0

    @patch("src.data.kaggle_downloader.verify_massey_ordinals")
    @patch("src.data.kaggle_downloader.download_competition_data")
    def test_download_with_verify_fail(self, mock_download, mock_verify):
        mock_download.return_value = "/data/kaggle"
        mock_verify.return_value = {"status": "fail", "system_names": []}
        args = argparse.Namespace(
            output_dir="/data/kaggle", force=False,
            verify_season=2025, competition=None,
        )
        code = download_kaggle(args)
        assert code == 1

    @patch("src.data.kaggle_downloader.verify_massey_ordinals")
    @patch("src.data.kaggle_downloader.download_competition_data")
    def test_download_with_verify_partial(self, mock_download, mock_verify):
        mock_download.return_value = "/data/kaggle"
        mock_verify.return_value = {"status": "partial", "system_names": ["RPI"]}
        args = argparse.Namespace(
            output_dir="/data/kaggle", force=False,
            verify_season=2025, competition=None,
        )
        code = download_kaggle(args)
        assert code == 0


# ═══════════════════════════════════════════════════════════════════════════
# scrape_tournament_results
# ═══════════════════════════════════════════════════════════════════════════

class TestScrapeTournamentResults:
    @patch("src.data.historical_tournament_results.save_tournament_results")
    @patch("src.data.scrapers.tournament_results.TournamentResultsScraper")
    def test_single_year(self, mock_scraper_cls, mock_save, tmp_path):
        mock_scraper = MagicMock()
        mock_scraper.scrape_year.return_value = [
            {
                "year": 2025, "round_name": "R64",
                "team1_id": "Duke", "team2_id": "UNC",
                "team1_seed": 1, "team2_seed": 16,
                "team1_score": 80, "team2_score": 60,
                "team1_won": True, "region": "East",
            }
        ]
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=2025, years=None,
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "out"),
        )
        code = scrape_tournament_results(args)
        assert code == 0
        mock_save.assert_called_once()

    @patch("src.data.scrapers.tournament_results.TournamentResultsScraper")
    def test_scrape_no_games(self, mock_scraper_cls, tmp_path):
        mock_scraper = MagicMock()
        mock_scraper.scrape_year.return_value = []
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=2020, years=None,
            cache_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        code = scrape_tournament_results(args)
        assert code == 0

    @patch("src.data.scrapers.tournament_results.TournamentResultsScraper")
    def test_scrape_exception_year(self, mock_scraper_cls, tmp_path):
        mock_scraper = MagicMock()
        mock_scraper.scrape_year.side_effect = RuntimeError("timeout")
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=None, years="2025",
            cache_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        code = scrape_tournament_results(args)
        assert code == 0  # does not fail, just prints error

    @patch("src.data.scrapers.tournament_results.TournamentResultsScraper")
    def test_default_years(self, mock_scraper_cls, tmp_path):
        mock_scraper = MagicMock()
        mock_scraper.scrape_year.return_value = []
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=None, years=None,
            cache_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        code = scrape_tournament_results(args)
        assert code == 0
        # Default is [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
        assert mock_scraper.scrape_year.call_count == 8


# ═══════════════════════════════════════════════════════════════════════════
# repair_dates
# ═══════════════════════════════════════════════════════════════════════════

class TestRepairDates:
    @patch("src.main.HistoricalDataPipeline")
    @patch("src.main.HistoricalIngestionConfig")
    def test_repair_range(self, mock_config_cls, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.repair_historical_dates.return_value = {
            2022: {"total": 100, "repaired": 5, "unique_dates": 30},
        }
        mock_pipeline_cls.return_value = mock_pipeline
        args = argparse.Namespace(
            seasons="2022-2024",
            historical_dir="data/hist",
            dry_run=False,
            force_slow=False,
        )
        code = repair_dates(args)
        assert code == 0
        call_kwargs = mock_pipeline.repair_historical_dates.call_args[1]
        assert call_kwargs["seasons"] == [2022, 2023, 2024]

    @patch("src.main.HistoricalDataPipeline")
    @patch("src.main.HistoricalIngestionConfig")
    def test_repair_csv_seasons(self, mock_config_cls, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.repair_historical_dates.return_value = {
            2022: {"total": 100, "repaired": 5, "unique_dates": 30},
        }
        mock_pipeline_cls.return_value = mock_pipeline
        args = argparse.Namespace(
            seasons="2017,2018",
            historical_dir="data/hist",
            dry_run=False,
            force_slow=False,
        )
        code = repair_dates(args)
        assert code == 0
        call_kwargs = mock_pipeline.repair_historical_dates.call_args[1]
        assert call_kwargs["seasons"] == [2017, 2018]

    @patch("src.main.HistoricalDataPipeline")
    @patch("src.main.HistoricalIngestionConfig")
    def test_repair_no_seasons(self, mock_config_cls, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.repair_historical_dates.return_value = {}
        mock_pipeline_cls.return_value = mock_pipeline
        args = argparse.Namespace(
            seasons=None,
            historical_dir="data/hist",
            dry_run=True,
            force_slow=False,
        )
        code = repair_dates(args)
        assert code == 0
        call_kwargs = mock_pipeline.repair_historical_dates.call_args[1]
        assert call_kwargs["seasons"] is None


# ═══════════════════════════════════════════════════════════════════════════
# run_scrape_conference_seeds
# ═══════════════════════════════════════════════════════════════════════════

class TestRunScrapeConferenceSeeds:
    @patch("src.data.scrapers.conference_seeds.ConferenceSeedsScraper")
    def test_no_seeds_scraped(self, mock_scraper_cls):
        mock_scraper = MagicMock()
        mock_scraper.scrape_seeds.return_value = {}
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=2026, output=None, conferences=None, cache_dir="cache",
        )
        code = run_scrape_conference_seeds(args)
        assert code == 1

    @patch("src.data.scrapers.conference_seeds.ConferenceSeedsScraper")
    def test_seeds_scraped_success(self, mock_scraper_cls):
        mock_scraper = MagicMock()
        mock_scraper.scrape_seeds.return_value = {
            "ACC": {"Duke": 1, "UNC": 2},
            "SEC": {"Kentucky": 1, "Alabama": 2},
        }
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=2026, output="/out/seeds.json", conferences=["ACC", "SEC"],
            cache_dir="cache",
        )
        code = run_scrape_conference_seeds(args)
        assert code == 0
        mock_scraper.save_to_file.assert_called_once()

    @patch("src.data.scrapers.conference_seeds.ConferenceSeedsScraper")
    def test_default_output_path(self, mock_scraper_cls):
        mock_scraper = MagicMock()
        mock_scraper.scrape_seeds.return_value = {"B12": {"Kansas": 1}}
        mock_scraper_cls.return_value = mock_scraper
        args = argparse.Namespace(
            year=2026, output=None, conferences=None, cache_dir="cache",
        )
        code = run_scrape_conference_seeds(args)
        assert code == 0
        # Check the default path was used
        save_call = mock_scraper.save_to_file.call_args
        assert "2026" in save_call[0][1]


# ═══════════════════════════════════════════════════════════════════════════
# run_conference_tournaments
# ═══════════════════════════════════════════════════════════════════════════

class TestRunConferenceTournaments:
    @patch("src.conference_tournament.predictor.ConferenceTournamentPredictor")
    def test_list_conferences(self, mock_predictor_cls):
        mock_predictor = MagicMock()
        mock_predictor.list_conferences.return_value = ["ACC", "SEC"]
        mock_predictor.get_conference_teams.return_value = ["Duke", "UNC"]
        mock_predictor_cls.from_torvik_json.return_value = mock_predictor
        args = argparse.Namespace(
            torvik="torvik.json", use_pipeline=False,
            list_conferences=True, conference=None,
            output=None, year=2026, simulate=False,
            simulations=1000, data_dir=None, seeds=None,
        )
        code = run_conference_tournaments(args)
        assert code == 0

    @patch("src.conference_tournament.predictor.ConferenceTournamentPredictor")
    def test_single_conference_no_output(self, mock_predictor_cls):
        mock_predictor = MagicMock()
        mock_predictor.generate_report.return_value = "Conference Report"
        mock_predictor_cls.from_torvik_json.return_value = mock_predictor
        args = argparse.Namespace(
            torvik="torvik.json", use_pipeline=False,
            list_conferences=False, conference="ACC",
            output=None, year=2026, simulate=False,
            simulations=1000, data_dir=None, seeds=None,
        )
        code = run_conference_tournaments(args)
        assert code == 0

    @patch("builtins.open", mock_open())
    @patch("src.conference_tournament.predictor.ConferenceTournamentPredictor")
    def test_with_output(self, mock_predictor_cls):
        mock_predictor = MagicMock()
        mock_predictor.to_json.return_value = '{"test": true}'
        mock_predictor_cls.from_torvik_json.return_value = mock_predictor
        args = argparse.Namespace(
            torvik="torvik.json", use_pipeline=False,
            list_conferences=False, conference=None,
            output="conf.json", year=2026, simulate=False,
            simulations=1000, data_dir=None, seeds=None,
        )
        code = run_conference_tournaments(args)
        assert code == 0


# ═══════════════════════════════════════════════════════════════════════════
# main() - command dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestMainDispatch:
    def test_no_command_returns_1(self):
        with patch("sys.argv", ["main"]):
            code = main()
        assert code == 1

    @patch("src.main.run_sota")
    def test_sota_dispatches(self, mock_run_sota):
        mock_run_sota.return_value = 0
        with patch("sys.argv", ["main", "sota"]):
            code = main()
        assert code == 0
        mock_run_sota.assert_called_once()

    @patch("src.main.ingest_data")
    def test_ingest_dispatches(self, mock_ingest):
        mock_ingest.return_value = 0
        with patch("sys.argv", ["main", "ingest", "--year", "2026"]):
            code = main()
        assert code == 0

    @patch("src.main.ingest_historical")
    def test_ingest_historical_dispatches(self, mock_ingest):
        mock_ingest.return_value = 0
        with patch("sys.argv", ["main", "ingest-historical"]):
            code = main()
        assert code == 0

    @patch("src.main.materialize_features")
    def test_materialize_features_dispatches(self, mock_mat):
        mock_mat.return_value = 0
        with patch("sys.argv", ["main", "materialize-features"]):
            code = main()
        assert code == 0

    @patch("src.main.run_sota_from_manifest")
    def test_sota_from_manifest_dispatches(self, mock_run):
        mock_run.return_value = 0
        with patch("sys.argv", ["main", "sota-from-manifest", "--manifest", "m.json"]):
            code = main()
        assert code == 0

    @patch("src.main.audit_rdof")
    def test_audit_rdof_dispatches(self, mock_audit):
        mock_audit.return_value = 0
        with patch("sys.argv", ["main", "audit-rdof"]):
            code = main()
        assert code == 0

    @patch("src.main.freeze_pipeline_cmd")
    def test_freeze_pipeline_dispatches(self, mock_freeze):
        mock_freeze.return_value = 0
        with patch("sys.argv", ["main", "freeze-pipeline"]):
            code = main()
        assert code == 0

    @patch("src.main.verify_freeze_cmd")
    def test_verify_freeze_dispatches(self, mock_verify):
        mock_verify.return_value = 0
        with patch("sys.argv", ["main", "verify-freeze"]):
            code = main()
        assert code == 0

    @patch("src.main.prospective_eval")
    def test_prospective_eval_dispatches(self, mock_eval):
        mock_eval.return_value = 0
        with patch("sys.argv", ["main", "prospective-eval", "--freeze-file", "f.json", "--year", "2025"]):
            code = main()
        assert code == 0

    @patch("src.main.run_calibrate_mc")
    def test_calibrate_mc_dispatches(self, mock_cal):
        mock_cal.return_value = 0
        with patch("sys.argv", ["main", "calibrate-mc"]):
            code = main()
        assert code == 0

    @patch("src.main.run_validate_vs_market")
    def test_validate_vs_market_dispatches(self, mock_val):
        mock_val.return_value = 0
        with patch("sys.argv", ["main", "validate-vs-market",
                                 "--model-probs", "m.json", "--market-odds", "o.json"]):
            code = main()
        assert code == 0

    @patch("src.main.scrape_rosters")
    def test_scrape_rosters_dispatches(self, mock_scrape):
        mock_scrape.return_value = 0
        with patch("sys.argv", ["main", "scrape-rosters"]):
            code = main()
        assert code == 0

    @patch("src.main.enrich_rosters")
    def test_enrich_rosters_dispatches(self, mock_enrich):
        mock_enrich.return_value = 0
        with patch("sys.argv", ["main", "enrich-rosters"]):
            code = main()
        assert code == 0

    @patch("src.main.download_kaggle")
    def test_download_kaggle_dispatches(self, mock_dl):
        mock_dl.return_value = 0
        with patch("sys.argv", ["main", "download-kaggle"]):
            code = main()
        assert code == 0

    @patch("src.main.run_loyo_validate")
    def test_loyo_validate_dispatches(self, mock_loyo):
        mock_loyo.return_value = 0
        with patch("sys.argv", ["main", "loyo-validate"]):
            code = main()
        assert code == 0

    @patch("src.main.run_backtest_kaggle")
    def test_backtest_kaggle_dispatches(self, mock_bt):
        mock_bt.return_value = 0
        with patch("sys.argv", ["main", "backtest-kaggle"]):
            code = main()
        assert code == 0

    @patch("src.main.run_backtest_unified")
    def test_backtest_unified_dispatches(self, mock_bt):
        mock_bt.return_value = 0
        with patch("sys.argv", ["main", "backtest-unified"]):
            code = main()
        assert code == 0

    @patch("src.main.run_validate_metrics")
    def test_validate_metrics_dispatches(self, mock_vm):
        mock_vm.return_value = 0
        with patch("sys.argv", ["main", "validate-metrics"]):
            code = main()
        assert code == 0

    @patch("src.main.scrape_tournament_results")
    def test_scrape_tournament_results_dispatches(self, mock_str):
        mock_str.return_value = 0
        with patch("sys.argv", ["main", "scrape-tournament-results"]):
            code = main()
        assert code == 0

    @patch("src.main.repair_dates")
    def test_repair_dates_dispatches(self, mock_repair):
        mock_repair.return_value = 0
        with patch("sys.argv", ["main", "repair-dates"]):
            code = main()
        assert code == 0

    @patch("src.main.run_scrape_conference_seeds")
    def test_scrape_conference_seeds_dispatches(self, mock_scrape):
        mock_scrape.return_value = 0
        with patch("sys.argv", ["main", "scrape-conference-seeds"]):
            code = main()
        assert code == 0

    @patch("src.main.run_production_2026_cmd")
    def test_run_production_2026_dispatches(self, mock_prod):
        mock_prod.return_value = 0
        with patch("sys.argv", ["main", "run-production-2026"]):
            code = main()
        assert code == 0

    @patch("src.main.run_conference_tournaments")
    def test_conference_tournaments_dispatches(self, mock_conf):
        mock_conf.return_value = 0
        with patch("sys.argv", ["main", "conference-tournaments"]):
            code = main()
        assert code == 0

    @patch("src.main.run_research_loop")
    def test_research_loop_dispatches(self, mock_rl):
        mock_rl.return_value = 0
        with patch("sys.argv", ["main", "research-loop"]):
            code = main()
        assert code == 0


# ═══════════════════════════════════════════════════════════════════════════
# main() - Phase 7/8 workflow commands
# ═══════════════════════════════════════════════════════════════════════════

class TestMainPhase7Dispatch:
    @patch("src.workflows.reproducible.train_historical_snapshot")
    def test_train_historical_snapshot_success(self, mock_func):
        with patch("sys.argv", ["main", "train-historical-snapshot"]):
            code = main()
        assert code == 0

    @patch("src.workflows.reproducible.train_historical_snapshot", side_effect=RuntimeError("oops"))
    def test_train_historical_snapshot_error(self, mock_func):
        with patch("sys.argv", ["main", "train-historical-snapshot"]):
            code = main()
        assert code == 1

    @patch("src.workflows.reproducible.predict_selection_sunday")
    def test_predict_selection_sunday_success(self, mock_func):
        with patch("sys.argv", ["main", "predict-selection-sunday"]):
            code = main()
        assert code == 0

    @patch("src.workflows.reproducible.predict_selection_sunday", side_effect=Exception("err"))
    def test_predict_selection_sunday_error(self, mock_func):
        with patch("sys.argv", ["main", "predict-selection-sunday"]):
            code = main()
        assert code == 1

    @patch("src.workflows.reproducible.evaluate_prospective")
    def test_evaluate_prospective_success(self, mock_func):
        with patch("sys.argv", ["main", "evaluate-prospective"]):
            code = main()
        assert code == 0

    @patch("src.workflows.reproducible.export_kaggle")
    def test_export_kaggle_success(self, mock_func):
        with patch("sys.argv", ["main", "export-kaggle",
                                 "--manifest", "m.json",
                                 "--sample-submission", "s.csv",
                                 "--kaggle-teams", "t.csv"]):
            code = main()
        assert code == 0


class TestMainPhase8Dispatch:
    @patch("src.workflows.live_protocol.freeze_2026")
    def test_freeze_2026_success(self, mock_func):
        with patch("sys.argv", ["main", "freeze-2026"]):
            code = main()
        assert code == 0

    @patch("src.workflows.live_protocol.freeze_2026", side_effect=Exception("err"))
    def test_freeze_2026_error(self, mock_func):
        with patch("sys.argv", ["main", "freeze-2026"]):
            code = main()
        assert code == 1

    @patch("src.workflows.live_protocol.generate_predictions_2026")
    def test_generate_predictions_2026_success(self, mock_func):
        with patch("sys.argv", ["main", "generate-predictions-2026"]):
            code = main()
        assert code == 0

    @patch("src.workflows.live_protocol.export_forecasts")
    def test_export_forecasts_success(self, mock_func):
        with patch("sys.argv", ["main", "export-forecasts"]):
            code = main()
        assert code == 0

    @patch("src.workflows.live_protocol.score_tournament")
    def test_score_tournament_success(self, mock_func):
        mock_func.return_value = {"score": 0.2}
        with patch("sys.argv", ["main", "score-tournament"]):
            code = main()
        assert code == 0

    @patch("src.workflows.live_protocol.score_tournament")
    def test_score_tournament_with_error(self, mock_func):
        mock_func.return_value = {"error": "no data"}
        with patch("sys.argv", ["main", "score-tournament"]):
            code = main()
        assert code == 1

    @patch("src.workflows.live_protocol.production_readiness_check")
    def test_production_readiness_check_green(self, mock_func):
        mock_func.return_value = {"all_green": True}
        with patch("sys.argv", ["main", "production-readiness-check"]):
            code = main()
        assert code == 0

    @patch("src.workflows.live_protocol.production_readiness_check")
    def test_production_readiness_check_not_green(self, mock_func):
        mock_func.return_value = {"all_green": False}
        with patch("sys.argv", ["main", "production-readiness-check"]):
            code = main()
        assert code == 1


# ═══════════════════════════════════════════════════════════════════════════
# main() - monitor / save-baseline / list-experiments / governance / deploy
# ═══════════════════════════════════════════════════════════════════════════

class TestMainMonitorDispatch:
    @patch("src.monitoring.pipeline_monitor.PipelineMonitor")
    def test_monitor_basic(self, mock_monitor_cls):
        mock_monitor = MagicMock()
        mock_report = MagicMock()
        mock_report.summary.return_value = "All green"
        mock_monitor.generate_report.return_value = mock_report
        mock_monitor_cls.return_value = mock_monitor
        with patch("sys.argv", ["main", "monitor"]):
            code = main()
        assert code == 0

    @patch("src.monitoring.pipeline_monitor.PipelineMonitor")
    def test_monitor_with_baseline(self, mock_monitor_cls):
        mock_monitor = MagicMock()
        mock_report = MagicMock()
        mock_report.summary.return_value = "All green"
        mock_monitor.generate_report.return_value = mock_report
        mock_monitor_cls.return_value = mock_monitor
        mock_monitor_cls.load_baseline.return_value = {"stats": {}}
        with patch("sys.argv", ["main", "monitor", "--baseline", "b.json"]):
            code = main()
        assert code == 0


class TestMainGovernanceDispatch:
    @patch("src.governance.gate.GovernanceGate")
    def test_governance_status(self, mock_gate_cls):
        mock_gate = MagicMock()
        mock_gate.status.return_value = "No pending approvals"
        mock_gate_cls.return_value = mock_gate
        with patch("sys.argv", ["main", "governance", "status"]):
            code = main()
        assert code == 0

    @patch("src.governance.gate.GovernanceGate")
    def test_governance_approve(self, mock_gate_cls):
        mock_gate = MagicMock()
        mock_gate.approve.return_value = True
        mock_gate_cls.return_value = mock_gate
        with patch("sys.argv", ["main", "governance", "approve", "req123"]):
            code = main()
        assert code == 0

    @patch("src.governance.gate.GovernanceGate")
    def test_governance_deny(self, mock_gate_cls):
        mock_gate = MagicMock()
        mock_gate.deny.return_value = True
        mock_gate_cls.return_value = mock_gate
        with patch("sys.argv", ["main", "governance", "deny", "req123"]):
            code = main()
        assert code == 0

    @patch("src.governance.audit_trail.GovernanceAuditLog")
    @patch("src.governance.gate.GovernanceGate")
    def test_governance_audit(self, mock_gate_cls, mock_audit_cls):
        mock_gate = MagicMock()
        mock_gate_cls.return_value = mock_gate
        mock_audit = MagicMock()
        mock_audit.query.return_value = [
            {"timestamp": "2026-01-01T00:00:00", "type": "approval", "action": "approve"}
        ]
        mock_audit_cls.return_value = mock_audit
        with patch("sys.argv", ["main", "governance", "audit"]):
            code = main()
        assert code == 0

    @patch("src.governance.gate.GovernanceGate")
    def test_governance_no_subcommand(self, mock_gate_cls):
        mock_gate = MagicMock()
        mock_gate_cls.return_value = mock_gate
        with patch("sys.argv", ["main", "governance"]):
            code = main()
        assert code == 0


class TestMainDeployDispatch:
    @patch("src.deployment.model_store.ModelStore")
    def test_deploy_list_empty(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store.list_versions.return_value = []
        mock_store_cls.return_value = mock_store
        with patch("sys.argv", ["main", "deploy", "list"]):
            code = main()
        assert code == 0

    @patch("src.deployment.model_store.ModelStore")
    def test_deploy_list_versions(self, mock_store_cls):
        mock_store = MagicMock()
        v = MagicMock()
        v.version_id = "v1"
        v.model_name = "spread_logistic"
        v.brier_score = 0.2
        v.is_production = True
        v.created_at = "2026-01-01T00:00:00"
        mock_store.list_versions.return_value = [v]
        mock_store_cls.return_value = mock_store
        with patch("sys.argv", ["main", "deploy", "list"]):
            code = main()
        assert code == 0

    @patch("src.deployment.model_store.ModelStore")
    def test_deploy_promote(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        with patch("sys.argv", ["main", "deploy", "promote", "--version", "v1"]):
            code = main()
        assert code == 0
        mock_store.promote.assert_called_once_with("v1")

    @patch("src.deployment.model_store.ModelStore")
    def test_deploy_drift_check(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        with patch("sys.argv", ["main", "deploy", "drift-check"]):
            code = main()
        assert code == 0

    @patch("src.deployment.model_store.ModelStore")
    def test_deploy_no_subcommand(self, mock_store_cls):
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        with patch("sys.argv", ["main", "deploy"]):
            code = main()
        assert code == 0


# ═══════════════════════════════════════════════════════════════════════════
# main() - list-experiments / log-experiment / pre-tournament-check / etc.
# ═══════════════════════════════════════════════════════════════════════════

class TestMainExperimentDispatch:
    @patch("src.ml.evaluation.experiment_registry.ExperimentRegistry")
    def test_list_experiments(self, mock_registry_cls):
        mock_registry = MagicMock()
        mock_registry.summary.return_value = "3 experiments"
        mock_registry.list.return_value = []
        mock_registry_cls.return_value = mock_registry
        with patch("sys.argv", ["main", "list-experiments"]):
            code = main()
        assert code == 0

    @patch("builtins.open", mock_open(read_data='{"loyo_mean_brier": 0.2}'))
    @patch("src.ml.evaluation.experiment_registry.ExperimentRegistry")
    @patch("src.ml.evaluation.experiment_registry.ExperimentRecord")
    def test_log_experiment(self, mock_record_cls, mock_registry_cls):
        mock_registry = MagicMock()
        mock_registry.log.return_value = "exp_123"
        mock_registry_cls.return_value = mock_registry
        with patch("sys.argv", ["main", "log-experiment", "--result", "result.json"]):
            code = main()
        assert code == 0


class TestMainSnapshotDispatch:
    @patch("src.data.versioning.snapshot_data_dir")
    def test_snapshot(self, mock_snap):
        mock_snap.return_value = "snap_123"
        with patch("sys.argv", ["main", "snapshot"]):
            code = main()
        assert code == 0

    @patch("src.data.versioning.list_snapshots")
    def test_list_snapshots_empty(self, mock_list):
        mock_list.return_value = []
        with patch("sys.argv", ["main", "list-snapshots"]):
            code = main()
        assert code == 0

    @patch("src.data.versioning.list_snapshots")
    def test_list_snapshots_with_data(self, mock_list):
        s = MagicMock()
        s.snapshot_id = "snap_1"
        s.total_size_bytes = 1024 * 1024
        s.file_count = 10
        s.label = "test"
        mock_list.return_value = [s]
        with patch("sys.argv", ["main", "list-snapshots"]):
            code = main()
        assert code == 0

    @patch("src.data.versioning.restore_snapshot")
    def test_restore_snapshot(self, mock_restore):
        mock_restore.return_value = 5
        with patch("sys.argv", ["main", "restore-snapshot", "--id", "snap_1"]):
            code = main()
        assert code == 0


class TestMainRunHistory:
    @patch("src.monitoring.run_history.RunHistory")
    def test_run_history(self, mock_history_cls):
        mock_history = MagicMock()
        mock_history.summary.return_value = "10 runs"
        mock_history_cls.return_value = mock_history
        with patch("sys.argv", ["main", "run-history"]):
            code = main()
        assert code == 0


class TestMainPreTournamentCheck:
    @patch("src.monitoring.pre_tournament_checklist.PreTournamentChecklist")
    def test_pre_tournament_check_ready(self, mock_cls):
        mock_checklist = MagicMock()
        mock_report = MagicMock()
        mock_report.summary.return_value = "All checks pass"
        mock_report.ready.return_value = True
        mock_checklist.run.return_value = mock_report
        mock_cls.return_value = mock_checklist
        with patch("sys.argv", ["main", "pre-tournament-check"]):
            code = main()
        assert code == 0

    @patch("src.monitoring.pre_tournament_checklist.PreTournamentChecklist")
    def test_pre_tournament_check_not_ready(self, mock_cls):
        mock_checklist = MagicMock()
        mock_report = MagicMock()
        mock_report.summary.return_value = "NOT READY"
        mock_report.ready.return_value = False
        mock_checklist.run.return_value = mock_report
        mock_cls.return_value = mock_checklist
        with patch("sys.argv", ["main", "pre-tournament-check"]):
            code = main()
        assert code == 1


class TestMainResearchReport:
    @patch("src.ml.research.research_loop.ResearchLoop")
    @patch("src.ml.evaluation.experiment_registry.ExperimentRegistry")
    @patch("src.ml.research.hypothesis_registry.HypothesisRegistry")
    def test_research_report(self, mock_h_cls, mock_e_cls, mock_loop_cls):
        mock_loop = MagicMock()
        mock_loop.report.return_value = "Research summary"
        mock_loop_cls.return_value = mock_loop
        with patch("sys.argv", ["main", "research-report"]):
            code = main()
        assert code == 0


class TestMainAuditMetricsCoverage:
    @patch("src.data.coverage_audit.run_coverage_audit")
    def test_audit_metrics_coverage(self, mock_audit):
        with patch("sys.argv", ["main", "audit-metrics-coverage"]):
            code = main()
        assert code == 0
        mock_audit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases for _build_pipeline_config with getattr defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPipelineConfigEdgeCases:
    @patch("src.main.SOTAPipelineConfig")
    def test_empty_namespace(self, mock_config_cls):
        """Verify defaults are used when args has minimal attributes."""
        args = argparse.Namespace(year=2026, simulations=100)
        _build_pipeline_config(args)
        kwargs = mock_config_cls.call_args[1]
        # All getattr defaults should be applied
        assert kwargs["pool_size"] == 100
        assert kwargs["random_seed"] == 2026
        assert kwargs["scrape_live"] is False
        assert kwargs["enable_gnn"] is False
        assert kwargs["enable_transformer"] is False

    @patch("src.main.SOTAPipelineConfig")
    def test_path_overrides_empty_dict(self, mock_config_cls):
        args = _make_args()
        _build_pipeline_config(args, path_overrides={})
        mock_config_cls.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Additional edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveMultiYearDirEdgeCases:
    def test_false_value_passes_through(self):
        assert _resolve_multi_year_dir(False) is False

    def test_list_passes_through(self):
        val = ["/a", "/b"]
        assert _resolve_multi_year_dir(val) == val


class TestParseYearListEdgeCases:
    def test_single_element_list(self):
        assert _parse_year_list([2025]) == [2025]

    def test_empty_list(self):
        assert _parse_year_list([]) == []

    def test_string_with_only_commas(self):
        assert _parse_year_list(",,,") == []


class TestParseFloatListEdgeCases:
    def test_single_negative(self):
        assert _parse_float_list("-1.5") == [-1.5]

    def test_integer_value(self):
        assert _parse_float_list(42) == [42.0]

    def test_comma_only(self):
        assert _parse_float_list(",,,") == []
