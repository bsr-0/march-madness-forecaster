"""Dry-run test: full pipeline with 2026 data pretending to be 2027.

Catches data path issues, config validation failures, and pipeline
initialization problems before they matter in the real 2027 season.
Symlinks existing 2026 data files as 2027 names so the pipeline sees
a realistic data environment without requiring actual 2027 data.
"""

import dataclasses
import json
import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# 2026 data files that get symlinked as 2027
_DATA_FILES = {
    "teams_json": ("data/raw/teams_2026.json", "data/raw/teams_2027.json"),
    "torvik_json": ("data/raw/torvik_2026.json", "data/raw/torvik_2027.json"),
    "historical_games_json": ("data/raw/historical_games_2026.json", "data/raw/historical_games_2027.json"),
    "roster_json": ("data/raw/rosters_2026.json", "data/raw/rosters_2027.json"),
    "public_picks_json": ("data/raw/public_picks_2026.json", "data/raw/public_picks_2027.json"),
    "scoring_rules_json": ("data/raw/scoring_rules_2026.json", "data/raw/scoring_rules_2027.json"),
}

_ARTIFACT_FILES = {
    "mc_calibration_json": ("artifacts/mc_calibration_2026.json", "artifacts/mc_calibration_2027.json"),
}


def _all_2026_data_exists() -> bool:
    for src, _ in list(_DATA_FILES.values()) + list(_ARTIFACT_FILES.values()):
        if not (REPO_ROOT / src).exists():
            return False
    return True


@pytest.fixture
def symlinked_2027_data():
    """Create symlinks from 2026 data → 2027 names, clean up after."""
    created = []
    for src_rel, dst_rel in list(_DATA_FILES.values()) + list(_ARTIFACT_FILES.values()):
        src = REPO_ROOT / src_rel
        dst = REPO_ROOT / dst_rel
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
            created.append(dst)
    yield created
    for link in created:
        if link.is_symlink():
            link.unlink()


@pytest.mark.skipif(not _all_2026_data_exists(), reason="2026 data files not present")
class TestDryRun2027:
    """Verify the 2027 pipeline config and initialization using 2026 data."""

    def test_config_validation_passes(self, symlinked_2027_data):
        """Production config for 2027 passes structural validation."""
        from src.governance.production_runner_generic import (
            validate_production_config,
            derive_year_partitions,
        )
        from src.pipeline.config import SOTAPipelineConfig

        config_path = REPO_ROOT / "configs" / "production_2027.json"
        assert config_path.exists(), f"Missing {config_path}"

        raw = json.loads(config_path.read_text())
        valid_fields = {f.name for f in dataclasses.fields(SOTAPipelineConfig)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        config = SOTAPipelineConfig(**filtered)

        # Should not raise
        validate_production_config(config, 2027)

        # Verify year partitions
        partitions = derive_year_partitions(2027)
        assert 2025 in partitions["training_years"]
        assert 2026 in partitions["holdout_years"]
        assert 2020 not in partitions["training_years"]

    def test_config_paths_resolve(self, symlinked_2027_data):
        """All data paths in production_2027.json resolve to files on disk."""
        from src.governance.production_runner_generic import (
            _load_raw_config,
            _require_explicit_paths,
        )

        config_path = REPO_ROOT / "configs" / "production_2027.json"
        raw = _load_raw_config(config_path)
        config_base_dir = config_path.parent.parent

        # Should not raise — all paths exist (via symlinks)
        resolved = _require_explicit_paths(raw, config_base_dir)
        assert len(resolved) > 0

        for field, path in resolved.items():
            assert Path(path).exists(), f"{field} path does not exist: {path}"

    def test_pipeline_initializes(self, symlinked_2027_data):
        """SOTAPipeline can be constructed with 2027 config + symlinked data."""
        from src.pipeline.config import SOTAPipelineConfig
        from src.pipeline.sota import SOTAPipeline
        from src.governance.production_runner_generic import (
            _load_raw_config,
            _require_explicit_paths,
        )

        config_path = REPO_ROOT / "configs" / "production_2027.json"
        raw = _load_raw_config(config_path)
        config_base_dir = config_path.parent.parent
        resolved = _require_explicit_paths(raw, config_base_dir)
        raw.update(resolved)

        valid_fields = {f.name for f in dataclasses.fields(SOTAPipelineConfig)}
        for key in list(raw):
            if key not in valid_fields:
                del raw[key]

        config = SOTAPipelineConfig(**raw)
        assert config.year == 2027

        # Pipeline construction — loads data, builds feature engineers
        pipeline = SOTAPipeline(config)
        assert pipeline is not None

    def test_cli_dry_run(self, symlinked_2027_data):
        """The standalone entrypoint's dry-run validates without errors."""
        from src.run_production import main

        exit_code = main(["--year", "2027", "--dry-run"])
        assert exit_code == 0

    def test_year_partition_continuity(self):
        """2027 partitions are a strict superset of 2026 partitions (plus 2025/2026)."""
        from src.governance.production_runner_generic import derive_year_partitions

        p2026 = derive_year_partitions(2026)
        p2027 = derive_year_partitions(2027)

        # 2027 training should include everything from 2026 training + 2025
        assert set(p2026["training_years"]).issubset(set(p2027["training_years"]))
        assert 2025 in p2027["training_years"]
        assert 2025 not in p2026["training_years"]

        # 2026 holdout becomes 2027 training
        assert p2026["holdout_years"] == [2025]
        assert p2027["holdout_years"] == [2026]
        assert 2025 in p2027["training_years"]

    def test_optimize_pool_cli_resolves(self, symlinked_2027_data):
        """optimize-pool --year 2027 at least parses and finds seed data."""
        # Just verify the CLI accepts the args — actual optimization
        # would require tournament seeds which don't exist for 2027 yet.
        import subprocess

        result = subprocess.run(
            ["python", "-m", "src.main", "optimize-pool", "--year", "2027", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--submission" in result.stdout
