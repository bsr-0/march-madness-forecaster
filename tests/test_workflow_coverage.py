"""Comprehensive tests for workflow and governance modules with mocking.

Covers:
    - src/workflows/live_protocol.py
    - src/workflows/reproducible.py
    - src/governance/production_runner.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from src.governance.production_runner import (
    FREEZE_ARTIFACT_REQUIRED_FIELDS,
    REQUIRED_EXPLICIT_PATH_FIELDS,
    REQUIRED_SOURCE_HASH_FILES,
    _assert_year_partition,
    _build_runtime_branch_audit,
    _count_samples_by_year,
    _ensure_required_keys,
    _git_commit_sha,
    _load_raw_config,
    _require_explicit_paths,
    _sha256_file,
    _sha256_path,
    _snapshot_config_hash,
    _validate_freeze_artifact,
)
from src.governance.production_validator import (
    EXPECTED_DEV_YEARS,
    EXPECTED_HOLDOUT_YEARS,
    EXPECTED_TRAINING_YEARS,
    ProductionValidationError,
)
from src.pipeline.config import SOTAPipelineConfig
from src.workflows.live_protocol import (
    _check_git_dirty,
    _sha256_str,
    export_forecasts,
    production_readiness_check,
)


# ---------------------------------------------------------------------------
# _sha256_str (pure function)
# ---------------------------------------------------------------------------

class TestSha256Str:
    def test_empty_string(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_str("") == expected

    def test_hello(self):
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _sha256_str("hello") == expected

    def test_unicode(self):
        s = "caf\u00e9"
        expected = hashlib.sha256(s.encode("utf-8")).hexdigest()
        assert _sha256_str(s) == expected

    def test_deterministic(self):
        assert _sha256_str("test") == _sha256_str("test")

    def test_different_inputs_differ(self):
        assert _sha256_str("a") != _sha256_str("b")


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_file_hash(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(f) == expected

    def test_binary_file(self, tmp_path):
        data = bytes(range(256))
        f = tmp_path / "binary.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256_file(f) == expected


# ---------------------------------------------------------------------------
# _sha256_path
# ---------------------------------------------------------------------------

class TestSha256Path:
    def test_single_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_bytes(b"content")
        assert _sha256_path(f) == _sha256_file(f)

    def test_directory(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "a.txt").write_bytes(b"aaa")
        (d / "b.txt").write_bytes(b"bbb")
        result = _sha256_path(d)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest length

    def test_directory_deterministic(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "a.txt").write_bytes(b"aaa")
        (d / "b.txt").write_bytes(b"bbb")
        assert _sha256_path(d) == _sha256_path(d)

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _sha256_path(tmp_path / "nope")


# ---------------------------------------------------------------------------
# _snapshot_config_hash
# ---------------------------------------------------------------------------

class TestSnapshotConfigHash:
    def test_deterministic(self):
        config = SOTAPipelineConfig(year=2026)
        assert _snapshot_config_hash(config) == _snapshot_config_hash(config)

    def test_different_configs_differ(self):
        c1 = SOTAPipelineConfig(year=2026)
        c2 = SOTAPipelineConfig(year=2025)
        assert _snapshot_config_hash(c1) != _snapshot_config_hash(c2)

    def test_returns_hex_digest(self):
        config = SOTAPipelineConfig(year=2026)
        h = _snapshot_config_hash(config)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# _load_raw_config
# ---------------------------------------------------------------------------

class TestLoadRawConfig:
    def test_valid_dict(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"year": 2026}))
        result = _load_raw_config(f)
        assert result == {"year": 2026}

    def test_non_dict_raises(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ProductionValidationError, match="JSON object"):
            _load_raw_config(f)

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text("not json")
        with pytest.raises(json.JSONDecodeError):
            _load_raw_config(f)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_raw_config(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# _git_commit_sha
# ---------------------------------------------------------------------------

class TestGitCommitSha:
    def test_no_git_dir(self, tmp_path):
        assert _git_commit_sha(tmp_path) == "unknown"

    def test_direct_sha_in_head(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("abc123def456\n")
        assert _git_commit_sha(tmp_path) == "abc123def456"

    def test_ref_pointer(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("deadbeef1234\n")
        assert _git_commit_sha(tmp_path) == "deadbeef1234"

    def test_ref_pointer_missing_ref_file(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/missing\n")
        # When ref file doesn't exist, returns the ref string itself
        assert _git_commit_sha(tmp_path) == "ref: refs/heads/missing"


# ---------------------------------------------------------------------------
# _ensure_required_keys
# ---------------------------------------------------------------------------

class TestEnsureRequiredKeys:
    def test_missing_keys_raises(self):
        with pytest.raises(ProductionValidationError, match="Missing required"):
            _ensure_required_keys({})

    def test_missing_single_key_raises(self):
        # Build a config that has everything except one key
        from src.governance.production_validator import REQUIRED_CONFIG_VALUES
        raw = dict(REQUIRED_CONFIG_VALUES)
        raw["training_years"] = EXPECTED_TRAINING_YEARS
        raw["dev_years"] = EXPECTED_DEV_YEARS
        raw["holdout_years"] = EXPECTED_HOLDOUT_YEARS
        raw["seed_prior_weight"] = 0.0
        raw["consistency_bonus_max"] = 0.0
        for field in REQUIRED_EXPLICIT_PATH_FIELDS:
            raw[field] = f"/fake/{field}"
        # Remove one key to trigger error
        del raw["year"]
        with pytest.raises(ProductionValidationError, match="year"):
            _ensure_required_keys(raw)

    def test_all_keys_present_passes(self):
        from src.governance.production_validator import REQUIRED_CONFIG_VALUES
        raw = dict(REQUIRED_CONFIG_VALUES)
        raw["training_years"] = EXPECTED_TRAINING_YEARS
        raw["dev_years"] = EXPECTED_DEV_YEARS
        raw["holdout_years"] = EXPECTED_HOLDOUT_YEARS
        raw["seed_prior_weight"] = 0.0
        raw["consistency_bonus_max"] = 0.0
        for field in REQUIRED_EXPLICIT_PATH_FIELDS:
            raw[field] = f"/fake/{field}"
        # Should not raise
        _ensure_required_keys(raw)


# ---------------------------------------------------------------------------
# _assert_year_partition
# ---------------------------------------------------------------------------

class TestAssertYearPartition:
    def test_correct_partition(self):
        raw = {
            "training_years": EXPECTED_TRAINING_YEARS,
            "dev_years": EXPECTED_DEV_YEARS,
            "holdout_years": EXPECTED_HOLDOUT_YEARS,
        }
        _assert_year_partition(raw)  # Should not raise

    def test_wrong_training_years(self):
        raw = {
            "training_years": [2016, 2017],
            "dev_years": EXPECTED_DEV_YEARS,
            "holdout_years": EXPECTED_HOLDOUT_YEARS,
        }
        with pytest.raises(ProductionValidationError, match="training_years"):
            _assert_year_partition(raw)

    def test_wrong_dev_years(self):
        raw = {
            "training_years": EXPECTED_TRAINING_YEARS,
            "dev_years": [2016, 2017],
            "holdout_years": EXPECTED_HOLDOUT_YEARS,
        }
        with pytest.raises(ProductionValidationError, match="dev_years"):
            _assert_year_partition(raw)

    def test_wrong_holdout_years(self):
        raw = {
            "training_years": EXPECTED_TRAINING_YEARS,
            "dev_years": EXPECTED_DEV_YEARS,
            "holdout_years": [2024],
        }
        with pytest.raises(ProductionValidationError, match="holdout_years"):
            _assert_year_partition(raw)

    def test_missing_training_years(self):
        raw = {
            "dev_years": EXPECTED_DEV_YEARS,
            "holdout_years": EXPECTED_HOLDOUT_YEARS,
        }
        with pytest.raises(ProductionValidationError, match="training_years"):
            _assert_year_partition(raw)


# ---------------------------------------------------------------------------
# _require_explicit_paths
# ---------------------------------------------------------------------------

class TestRequireExplicitPaths:
    def test_all_paths_exist(self, tmp_path):
        raw = {}
        for field in REQUIRED_EXPLICIT_PATH_FIELDS:
            p = tmp_path / field
            p.mkdir(exist_ok=True)
            raw[field] = str(p)
        result = _require_explicit_paths(raw, tmp_path)
        assert len(result) == len(REQUIRED_EXPLICIT_PATH_FIELDS)

    def test_missing_field_raises(self, tmp_path):
        raw = {}
        for field in REQUIRED_EXPLICIT_PATH_FIELDS:
            p = tmp_path / field
            p.mkdir(exist_ok=True)
            raw[field] = str(p)
        del raw[REQUIRED_EXPLICIT_PATH_FIELDS[0]]
        with pytest.raises(ProductionValidationError, match="explicitly set"):
            _require_explicit_paths(raw, tmp_path)

    def test_auto_value_raises(self, tmp_path):
        raw = {}
        for field in REQUIRED_EXPLICIT_PATH_FIELDS:
            p = tmp_path / field
            p.mkdir(exist_ok=True)
            raw[field] = str(p)
        raw[REQUIRED_EXPLICIT_PATH_FIELDS[0]] = "auto"
        with pytest.raises(ProductionValidationError, match="explicitly set"):
            _require_explicit_paths(raw, tmp_path)

    def test_nonexistent_path_raises(self, tmp_path):
        raw = {}
        for field in REQUIRED_EXPLICIT_PATH_FIELDS:
            raw[field] = str(tmp_path / field / "nonexistent")
        with pytest.raises(ProductionValidationError, match="missing"):
            _require_explicit_paths(raw, tmp_path)


# ---------------------------------------------------------------------------
# _validate_freeze_artifact
# ---------------------------------------------------------------------------

class TestValidateFreezeArtifact:
    def _make_freeze(self, tmp_path, config_file, raw_config, overrides=None):
        """Create a valid freeze artifact for testing."""
        config_hash = _sha256_file(config_file)
        # Create source files
        repo_root = tmp_path / "repo"
        repo_root.mkdir(exist_ok=True)
        source_hashes = {}
        for src_file in REQUIRED_SOURCE_HASH_FILES:
            src_path = repo_root / src_file
            src_path.parent.mkdir(parents=True, exist_ok=True)
            src_path.write_text(f"# {src_file}")
            source_hashes[src_file] = _sha256_file(src_path)

        # Create src tree
        src_tree = repo_root / "src"
        src_tree.mkdir(exist_ok=True)
        source_tree_hash = _sha256_path(src_tree)

        freeze = {
            "git_commit_sha": "abc123",
            "config_file_hash": config_hash,
            "training_years": raw_config.get("training_years"),
            "holdout_years": raw_config.get("holdout_years"),
            "target_year": raw_config.get("year"),
            "source_file_hashes": source_hashes,
            "source_tree_hash": source_tree_hash,
            "data_file_hashes": {},
            "dependency_lockfile_hash": "lockfile_hash",
            "run_timestamp_utc": "2026-03-01T00:00:00Z",
        }
        if overrides:
            freeze.update(overrides)

        freeze_path = tmp_path / "freeze.json"
        freeze_path.write_text(json.dumps(freeze))
        return freeze_path, repo_root

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ProductionValidationError, match="not found"):
            _validate_freeze_artifact(
                tmp_path / "missing.json",
                tmp_path / "config.json",
                {},
                tmp_path,
            )

    def test_missing_fields_raises(self, tmp_path):
        freeze_path = tmp_path / "freeze.json"
        freeze_path.write_text(json.dumps({"git_commit_sha": "abc"}))
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"year": 2026}))
        with pytest.raises(ProductionValidationError, match="missing required fields"):
            _validate_freeze_artifact(freeze_path, config_file, {}, tmp_path)

    def test_config_hash_mismatch_raises(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"year": 2026}))
        raw_config = {"year": 2026, "training_years": [2016], "holdout_years": [2025]}

        freeze_path, repo_root = self._make_freeze(
            tmp_path, config_file, raw_config,
            overrides={"config_file_hash": "wrong_hash"},
        )
        with pytest.raises(ProductionValidationError, match="config_hash mismatch"):
            _validate_freeze_artifact(freeze_path, config_file, raw_config, repo_root)

    def test_training_years_mismatch_raises(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"year": 2026}))
        raw_config = {"year": 2026, "training_years": [2016, 2017], "holdout_years": [2025]}

        freeze_path, repo_root = self._make_freeze(
            tmp_path, config_file, raw_config,
            overrides={"training_years": [2018, 2019]},
        )
        with pytest.raises(ProductionValidationError, match="training_years mismatch"):
            _validate_freeze_artifact(freeze_path, config_file, raw_config, repo_root)

    def test_holdout_years_mismatch_raises(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"year": 2026}))
        raw_config = {"year": 2026, "training_years": [2016], "holdout_years": [2025]}

        freeze_path, repo_root = self._make_freeze(
            tmp_path, config_file, raw_config,
            overrides={"holdout_years": [2024]},
        )
        with pytest.raises(ProductionValidationError, match="holdout_years mismatch"):
            _validate_freeze_artifact(freeze_path, config_file, raw_config, repo_root)

    def test_target_year_mismatch_raises(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"year": 2026}))
        raw_config = {"year": 2026, "training_years": [2016], "holdout_years": [2025]}

        freeze_path, repo_root = self._make_freeze(
            tmp_path, config_file, raw_config,
            overrides={"target_year": 2025},
        )
        with pytest.raises(ProductionValidationError, match="target_year mismatch"):
            _validate_freeze_artifact(freeze_path, config_file, raw_config, repo_root)

    def test_valid_freeze_passes(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"year": 2026}))
        raw_config = {"year": 2026, "training_years": [2016], "holdout_years": [2025]}
        freeze_path, repo_root = self._make_freeze(tmp_path, config_file, raw_config)
        result = _validate_freeze_artifact(freeze_path, config_file, raw_config, repo_root)
        assert isinstance(result, dict)
        assert result["git_commit_sha"] == "abc123"


# ---------------------------------------------------------------------------
# _build_runtime_branch_audit
# ---------------------------------------------------------------------------

class TestBuildRuntimeBranchAudit:
    def test_production_path(self):
        config = SOTAPipelineConfig(year=2026)
        prod = {"n": 10}
        exp = {"n": 0}
        audit = _build_runtime_branch_audit(config, prod, exp)
        assert audit["probability_profile"] == "production"
        assert audit["used_experimental_probability_path"] is False
        assert audit["production_inference_call_count"] == 10
        assert audit["experimental_inference_call_count"] == 0

    def test_experimental_path(self):
        config = SOTAPipelineConfig(year=2026)
        # Bypass post_init validation by setting attribute directly
        object.__setattr__(config, "enable_gnn", True)
        prod = {"n": 5}
        exp = {"n": 3}
        audit = _build_runtime_branch_audit(config, prod, exp)
        assert audit["used_experimental_probability_path"] is True
        assert audit["gnn_enabled"] is True


# ---------------------------------------------------------------------------
# _count_samples_by_year
# ---------------------------------------------------------------------------

class TestCountSamplesByYear:
    def test_missing_file(self, tmp_path):
        result = _count_samples_by_year(tmp_path, [2020], tournament_only=False)
        assert result == {"2020": 0}

    def test_count_all_games(self, tmp_path):
        games = {
            "games": [
                {"game_date": "2021-01-15", "team1": "A", "team2": "B"},
                {"game_date": "2021-02-20", "team1": "C", "team2": "D"},
                {"game_date": "2021-03-20", "team1": "E", "team2": "F"},
            ]
        }
        (tmp_path / "historical_games_2021.json").write_text(json.dumps(games))
        result = _count_samples_by_year(tmp_path, [2021], tournament_only=False)
        assert result["2021"] == 3

    def test_tournament_only(self, tmp_path):
        # Tournament start 2021 is March 18
        games = {
            "games": [
                {"game_date": "2021-01-15", "team1": "A", "team2": "B"},
                {"game_date": "2021-03-17", "team1": "C", "team2": "D"},
                {"game_date": "2021-03-20", "team1": "E", "team2": "F"},
                {"game_date": "2021-04-01", "team1": "G", "team2": "H"},
            ]
        }
        (tmp_path / "historical_games_2021.json").write_text(json.dumps(games))
        result = _count_samples_by_year(tmp_path, [2021], tournament_only=True)
        # Only games on or after 2021-03-18 should count
        assert result["2021"] == 2

    def test_game_missing_date(self, tmp_path):
        games = {"games": [{"team1": "A", "team2": "B"}]}
        (tmp_path / "historical_games_2022.json").write_text(json.dumps(games))
        result = _count_samples_by_year(tmp_path, [2022], tournament_only=False)
        assert result["2022"] == 0

    def test_multiple_years(self, tmp_path):
        for year in [2021, 2022]:
            games = {"games": [{"game_date": f"{year}-02-01"}]}
            (tmp_path / f"historical_games_{year}.json").write_text(json.dumps(games))
        result = _count_samples_by_year(tmp_path, [2021, 2022], tournament_only=False)
        assert result["2021"] == 1
        assert result["2022"] == 1


# ---------------------------------------------------------------------------
# _check_git_dirty
# ---------------------------------------------------------------------------

class TestCheckGitDirty:
    @patch("subprocess.run")
    def test_clean_repo(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout="")
        assert _check_git_dirty(tmp_path) is False

    @patch("subprocess.run")
    def test_dirty_repo(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout=" M src/file.py\n")
        assert _check_git_dirty(tmp_path) is True

    @patch("subprocess.run", side_effect=OSError("git not found"))
    def test_exception_returns_true(self, mock_run, tmp_path):
        assert _check_git_dirty(tmp_path) is True

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10))
    def test_timeout_returns_true(self, mock_run, tmp_path):
        assert _check_git_dirty(tmp_path) is True

    @patch("subprocess.run")
    def test_whitespace_only_is_clean(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout="   \n  \n")
        assert _check_git_dirty(tmp_path) is False


# ---------------------------------------------------------------------------
# export_forecasts
# ---------------------------------------------------------------------------

class TestExportForecasts:
    def test_creates_forecast_files(self, tmp_path):
        report = {
            "pairwise_probabilities": {"Duke_vs_UNC": 0.65},
            "bracket": {"R64": [{"team1": "Duke", "team2": "UNC"}]},
            "artifacts": {
                "calibration": {"method": "temperature", "temperature": 1.1},
                "baseline_training": {"teams": {"Duke": {}, "UNC": {}}},
                "pool_recommendation": {"strategy": "upset_heavy"},
                "simulation": {"num_simulations": 10000},
            },
            "bracket_picks": {"champion": "Duke"},
            "ev_analysis": {"ev": 1.5},
        }
        report_path = tmp_path / "predictions.json"
        report_path.write_text(json.dumps(report))

        result = export_forecasts(
            predictions_report=str(report_path),
            output_dir=str(tmp_path / "output"),
        )

        assert result["workflow"] == "export-forecasts"
        assert "forecast_only_path" in result
        assert "pool_strategy_path" in result

        # Verify forecast_only file
        fo_stable = tmp_path / "output" / "forecast_only_2026.json"
        assert fo_stable.exists()
        fo = json.loads(fo_stable.read_text())
        assert fo["type"] == "forecast_only"
        assert fo["pairwise_probabilities"] == {"Duke_vs_UNC": 0.65}
        assert fo["calibration_metadata"]["method"] == "temperature"
        assert fo["team_count"] == 2

        # Verify pool_strategy file
        ps_stable = tmp_path / "output" / "pool_strategy_2026.json"
        assert ps_stable.exists()
        ps = json.loads(ps_stable.read_text())
        assert ps["type"] == "pool_strategy"
        assert ps["bracket_picks"] == {"champion": "Duke"}

    def test_empty_report(self, tmp_path):
        report_path = tmp_path / "predictions.json"
        report_path.write_text(json.dumps({}))

        result = export_forecasts(
            predictions_report=str(report_path),
            output_dir=str(tmp_path / "output"),
        )
        assert result["workflow"] == "export-forecasts"

        fo = json.loads((tmp_path / "output" / "forecast_only_2026.json").read_text())
        assert fo["pairwise_probabilities"] == {}
        assert fo["team_count"] == 0

    def test_timestamped_files_created(self, tmp_path):
        report_path = tmp_path / "predictions.json"
        report_path.write_text(json.dumps({"artifacts": {}}))

        result = export_forecasts(
            predictions_report=str(report_path),
            output_dir=str(tmp_path / "output"),
        )

        # Verify timestamped files exist
        ts = result["timestamp"]
        assert (tmp_path / "output" / f"forecast_only_2026_{ts}.json").exists()
        assert (tmp_path / "output" / f"pool_strategy_2026_{ts}.json").exists()


# ---------------------------------------------------------------------------
# production_readiness_check
# ---------------------------------------------------------------------------

class TestProductionReadinessCheck:
    @patch("subprocess.run")
    def test_all_failing(self, mock_subprocess, tmp_path):
        """All checks fail when nothing exists."""
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="1 failed")
        result = production_readiness_check(
            config_path=str(tmp_path / "nonexistent_config.json"),
            freeze_file=str(tmp_path / "nonexistent_freeze.json"),
            predictions_report=str(tmp_path / "nonexistent_predictions.json"),
            output_dir=str(tmp_path / "output"),
        )
        assert result["all_green"] is False
        assert len(result["checks"]) == 6
        assert "NOT READY" in result["verdict"]

    @patch("subprocess.run")
    def test_freeze_file_present_check(self, mock_subprocess, tmp_path):
        """Freeze file present when it exists."""
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="")
        freeze_path = tmp_path / "freeze.json"
        freeze_path.write_text("{}")

        result = production_readiness_check(
            config_path=str(tmp_path / "nonexistent.json"),
            freeze_file=str(freeze_path),
            predictions_report=str(tmp_path / "nonexistent.json"),
            output_dir=str(tmp_path / "output"),
        )
        freeze_check = result["checks"][0]
        assert freeze_check["name"] == "Freeze file present"
        assert freeze_check["status"] == "pass"

    @patch("subprocess.run")
    def test_calibration_present_check(self, mock_subprocess, tmp_path):
        """Calibration check passes when predictions have calibration data."""
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="all passed")
        pred = {
            "artifacts": {
                "calibration": {"method": "temperature", "temperature": 1.05}
            }
        }
        pred_path = tmp_path / "predictions.json"
        pred_path.write_text(json.dumps(pred))

        result = production_readiness_check(
            config_path=str(tmp_path / "nonexistent.json"),
            freeze_file=str(tmp_path / "nonexistent.json"),
            predictions_report=str(pred_path),
            output_dir=str(tmp_path / "output"),
        )
        cal_check = result["checks"][2]
        assert cal_check["name"] == "Calibration report present"
        assert cal_check["status"] == "pass"

    @patch("subprocess.run")
    def test_report_written_to_output_dir(self, mock_subprocess, tmp_path):
        mock_subprocess.return_value = MagicMock(returncode=1, stdout="")
        result = production_readiness_check(
            config_path=str(tmp_path / "x.json"),
            freeze_file=str(tmp_path / "x.json"),
            predictions_report=str(tmp_path / "x.json"),
            output_dir=str(tmp_path / "out"),
        )
        report_file = tmp_path / "out" / "production_readiness_report.json"
        assert report_file.exists()
        saved = json.loads(report_file.read_text())
        assert saved["all_green"] == result["all_green"]

    @patch("subprocess.run", side_effect=Exception("boom"))
    def test_leakage_test_exception(self, mock_subprocess, tmp_path):
        """Leakage test check handles exceptions gracefully."""
        result = production_readiness_check(
            config_path=str(tmp_path / "x.json"),
            freeze_file=str(tmp_path / "x.json"),
            predictions_report=str(tmp_path / "x.json"),
            output_dir=str(tmp_path / "out"),
        )
        leakage_check = result["checks"][3]
        assert leakage_check["name"] == "No leakage assertions failed"
        assert leakage_check["status"] == "FAIL"
        assert "boom" in leakage_check["detail"]


# ---------------------------------------------------------------------------
# train_historical_snapshot (skip paths)
# ---------------------------------------------------------------------------

class TestTrainHistoricalSnapshot:
    def test_skip_both_stages(self, tmp_path):
        from src.workflows.reproducible import train_historical_snapshot

        result = train_historical_snapshot(
            skip_ingestion=True,
            skip_materialization=True,
            features_output_dir=str(tmp_path / "processed"),
        )
        assert result["workflow"] == "train-historical-snapshot"
        assert "ingestion_manifest" not in result
        assert "materialization_manifest" not in result
        manifest_path = tmp_path / "processed" / "train_snapshot_manifest.json"
        assert manifest_path.exists()
        saved = json.loads(manifest_path.read_text())
        assert saved["workflow"] == "train-historical-snapshot"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestFreezeArtifactRequiredFieldsConstant:
    def test_has_expected_fields(self):
        assert "git_commit_sha" in FREEZE_ARTIFACT_REQUIRED_FIELDS
        assert "config_file_hash" in FREEZE_ARTIFACT_REQUIRED_FIELDS
        assert "training_years" in FREEZE_ARTIFACT_REQUIRED_FIELDS
        assert "source_tree_hash" in FREEZE_ARTIFACT_REQUIRED_FIELDS

    def test_is_list(self):
        assert isinstance(FREEZE_ARTIFACT_REQUIRED_FIELDS, list)
        assert len(FREEZE_ARTIFACT_REQUIRED_FIELDS) >= 5


class TestRequiredExplicitPathFieldsConstant:
    def test_has_expected_fields(self):
        assert "multi_year_games_dir" in REQUIRED_EXPLICIT_PATH_FIELDS
        assert "freeze_file" in REQUIRED_EXPLICIT_PATH_FIELDS
        assert "teams_json" in REQUIRED_EXPLICIT_PATH_FIELDS

    def test_is_list(self):
        assert isinstance(REQUIRED_EXPLICIT_PATH_FIELDS, list)


class TestExpectedYearsConstants:
    def test_training_years_exclude_2020(self):
        assert 2020 not in EXPECTED_TRAINING_YEARS

    def test_dev_years_exclude_2020(self):
        assert 2020 not in EXPECTED_DEV_YEARS

    def test_holdout_is_2025(self):
        assert EXPECTED_HOLDOUT_YEARS == [2025]

    def test_training_and_dev_match(self):
        assert EXPECTED_TRAINING_YEARS == EXPECTED_DEV_YEARS


class TestSha256PathDirectory:
    def test_nested_directory(self, tmp_path):
        d = tmp_path / "root"
        d.mkdir()
        sub = d / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_bytes(b"deep content")
        (d / "top.txt").write_bytes(b"top content")
        result = _sha256_path(d)
        assert len(result) == 64

    def test_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        # Empty directory - sorted rglob returns nothing, so hash of empty string
        result = _sha256_path(d)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_directory_content_change_changes_hash(self, tmp_path):
        d = tmp_path / "dir"
        d.mkdir()
        (d / "file.txt").write_bytes(b"version1")
        hash1 = _sha256_path(d)
        (d / "file.txt").write_bytes(b"version2")
        hash2 = _sha256_path(d)
        assert hash1 != hash2
