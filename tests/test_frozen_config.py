"""Tests for src.reproducibility.frozen_config."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from src.reproducibility.frozen_config import FrozenExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frozen(
    experiment_id: str = "exp-001",
    pipeline_config: Dict[str, Any] | None = None,
    dataset_hashes: Dict[str, str] | None = None,
    code_version: str = "abc123",
    reproducibility_hash: str = "deadbeef",
    frozen_at: str | None = None,
) -> FrozenExperimentConfig:
    return FrozenExperimentConfig(
        experiment_id=experiment_id,
        pipeline_config=pipeline_config or {"year": 2026, "random_seed": 2026},
        dataset_hashes=dataset_hashes or {},
        code_version=code_version,
        reproducibility_hash=reproducibility_hash,
        frozen_at=frozen_at or datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFreezeThawRoundtrip:
    """to_json -> from_json should preserve all fields."""

    def test_roundtrip(self) -> None:
        original = _make_frozen(
            pipeline_config={"year": 2025, "num_simulations": 10000},
            dataset_hashes={"data/teams.json": "aabb"},
        )
        json_str = original.to_json()
        restored = FrozenExperimentConfig.from_json(json_str)

        assert restored.experiment_id == original.experiment_id
        assert restored.pipeline_config == original.pipeline_config
        assert restored.dataset_hashes == original.dataset_hashes
        assert restored.code_version == original.code_version
        assert restored.reproducibility_hash == original.reproducibility_hash
        assert restored.frozen_at == original.frozen_at

    def test_json_is_valid(self) -> None:
        frozen = _make_frozen()
        parsed = json.loads(frozen.to_json())
        assert "experiment_id" in parsed
        assert "pipeline_config" in parsed


class TestRecreateConfig:
    """recreate_config should produce a SOTAPipelineConfig with correct values."""

    def test_recreate(self) -> None:
        from src.pipeline.sota import SOTAPipelineConfig

        original_cfg = SOTAPipelineConfig(year=2025, num_simulations=10000, random_seed=42)

        from dataclasses import asdict
        frozen = _make_frozen(
            pipeline_config=asdict(original_cfg),
        )

        recreated = frozen.recreate_config()
        assert isinstance(recreated, SOTAPipelineConfig)
        assert recreated.year == 2025
        assert recreated.num_simulations == 10000
        assert recreated.random_seed == 42

    def test_ignores_unknown_fields(self) -> None:
        """Extra keys in pipeline_config should be silently dropped."""
        from src.pipeline.sota import SOTAPipelineConfig

        frozen = _make_frozen(
            pipeline_config={"year": 2026, "unknown_future_field": True},
        )
        recreated = frozen.recreate_config()
        assert isinstance(recreated, SOTAPipelineConfig)
        assert recreated.year == 2026


class TestVerifyDataIntegrity:
    """verify_data_integrity should detect changed or missing files."""

    def test_returns_true_when_files_match(self, tmp_path: Path) -> None:
        data_file = tmp_path / "teams.json"
        data_file.write_text('{"teams": []}')

        import hashlib
        expected_hash = hashlib.sha256(b'{"teams": []}').hexdigest()

        frozen = _make_frozen(
            dataset_hashes={str(data_file): expected_hash},
        )
        assert frozen.verify_data_integrity() is True

    def test_returns_false_when_file_modified(self, tmp_path: Path) -> None:
        data_file = tmp_path / "teams.json"
        data_file.write_text('{"teams": []}')

        import hashlib
        original_hash = hashlib.sha256(b'{"teams": []}').hexdigest()

        frozen = _make_frozen(
            dataset_hashes={str(data_file): original_hash},
        )

        # Modify the file after freezing.
        data_file.write_text('{"teams": ["Duke"]}')
        assert frozen.verify_data_integrity() is False

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        frozen = _make_frozen(
            dataset_hashes={str(tmp_path / "nonexistent.json"): "aabbccdd"},
        )
        assert frozen.verify_data_integrity() is False

    def test_empty_hashes_returns_true(self) -> None:
        """No files to check means nothing can be wrong."""
        frozen = _make_frozen(dataset_hashes={})
        assert frozen.verify_data_integrity() is True
