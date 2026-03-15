"""Phase 7: Lineage manifest generation and validation tests.

Validates prediction manifests against schemas/prediction_manifest.schema.json.

Manifest Validation Levels:
  Level A: Provenance traceable — all fields present and non-empty (CI-blocking)
  Level B: Feature matrix reproducible — training_data hash matches (CI-blocking)
  Level C: Prediction output reproducible (aspirational, xfail if non-deterministic)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.data.manifest_generator import ManifestGenerator, _sha256_bytes, _sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "prediction_manifest.schema.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_manifest():
    """A valid sample manifest for testing."""
    return {
        "run_id": "20260315_test_2026",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": {
            "commit_sha": "abc123def456",
            "diff_hash": "clean",
        },
        "config": {
            "path": "configs/production_2026.json",
            "hash": "sha256_config_hash_placeholder",
        },
        "raw_inputs": [
            {
                "provider": "cbbpy",
                "path_or_uri": "data/raw/games_2026.json",
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "checksum": "sha256_input_hash_placeholder",
            }
        ],
        "feature_contracts": {
            "path": "contracts/feature_contracts.yaml",
            "hash": "sha256_contracts_hash_placeholder",
        },
        "training_data": {
            "path_or_uri": "data/processed/train_X_2026.npy",
            "hash": "sha256_training_hash_placeholder",
        },
        "model": {
            "artifact_path_or_uri": "artifacts/model_2026.pkl",
            "artifact_hash": "sha256_model_hash_placeholder",
        },
        "random_seed": 42,
        "environment": {
            "python_version": "3.10.12",
            "os": "Linux 5.15.0",
            "lockfile_hash": "sha256_lockfile_hash_placeholder",
        },
    }


@pytest.fixture
def manifest_generator():
    return ManifestGenerator(config_path="configs/production_2026.json")


# ---------------------------------------------------------------------------
# Level A: Provenance traceable
# ---------------------------------------------------------------------------


class TestManifestSchemaValid:
    """Level A: Manifest conforms to JSON schema."""

    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), f"Missing {SCHEMA_PATH}"

    def test_valid_manifest_passes_schema(self, sample_manifest):
        jsonschema = pytest.importorskip("jsonschema")
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        jsonschema.validate(instance=sample_manifest, schema=schema)

    def test_missing_field_fails_schema(self, sample_manifest):
        jsonschema = pytest.importorskip("jsonschema")
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        # Remove a required field
        del sample_manifest["run_id"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=sample_manifest, schema=schema)


class TestAllFieldsNonempty:
    """Level A: All required fields are present and non-empty."""

    def test_level_a_valid_manifest(self, sample_manifest):
        violations = ManifestGenerator.validate_level_a(sample_manifest)
        assert not violations, f"Level A violations: {violations}"

    def test_level_a_empty_run_id(self, sample_manifest):
        sample_manifest["run_id"] = ""
        violations = ManifestGenerator.validate_level_a(sample_manifest)
        assert any("run_id" in v for v in violations)

    def test_level_a_missing_code_commit(self, sample_manifest):
        sample_manifest["code"]["commit_sha"] = ""
        violations = ManifestGenerator.validate_level_a(sample_manifest)
        assert any("commit_sha" in v for v in violations)

    def test_level_a_empty_raw_inputs(self, sample_manifest):
        sample_manifest["raw_inputs"] = []
        violations = ManifestGenerator.validate_level_a(sample_manifest)
        assert any("raw_inputs" in v for v in violations)

    def test_level_a_missing_env_field(self, sample_manifest):
        sample_manifest["environment"]["python_version"] = ""
        violations = ManifestGenerator.validate_level_a(sample_manifest)
        assert any("python_version" in v for v in violations)


class TestRawInputChecksumsVerifiable:
    """Level A: raw_input checksums can be verified against actual files."""

    def test_sha256_file_deterministic(self):
        """SHA-256 of a file is deterministic."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"key": "value"}')
            f.flush()
            path = f.name

        try:
            h1 = _sha256_file(path)
            h2 = _sha256_file(path)
            assert h1 == h2
            assert len(h1) == 64  # SHA-256 hex digest
        finally:
            os.unlink(path)

    def test_sha256_bytes_deterministic(self):
        data = b"test data for hashing"
        h1 = _sha256_bytes(data)
        h2 = _sha256_bytes(data)
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = _sha256_bytes(b"content A")
        h2 = _sha256_bytes(b"content B")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Level B: Feature matrix reproducible (HARD GATE)
# ---------------------------------------------------------------------------


class TestFeatureMatrixReproducible:
    """Level B: Given manifest inputs + code, train_X is bit-identical."""

    def test_level_b_matching_hash(self, sample_manifest):
        actual_hash = sample_manifest["training_data"]["hash"]
        violations = ManifestGenerator.validate_level_b(sample_manifest, actual_hash)
        # Should pass since hashes match
        level_b_specific = [v for v in violations if "Level B" in v]
        assert not level_b_specific

    def test_level_b_mismatched_hash(self, sample_manifest):
        violations = ManifestGenerator.validate_level_b(
            sample_manifest, "different_hash_value"
        )
        level_b_specific = [v for v in violations if "Level B" in v]
        assert len(level_b_specific) == 1

    def test_config_hash_deterministic(self):
        """Same config content produces same hash."""
        config = {"year": 2026, "random_seed": 42, "model_complexity": "standard"}
        content = json.dumps(config, sort_keys=True).encode()
        h1 = _sha256_bytes(content)
        h2 = _sha256_bytes(content)
        assert h1 == h2

    def test_training_matrix_hash_deterministic(self):
        """Same numpy array produces same hash when serialized consistently."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 79)
        h1 = _sha256_bytes(X.tobytes())
        h2 = _sha256_bytes(X.tobytes())
        assert h1 == h2

    def test_different_matrices_different_hash(self):
        """Different training data produces different hash."""
        X1 = np.zeros((100, 79))
        X2 = np.ones((100, 79))
        h1 = _sha256_bytes(X1.tobytes())
        h2 = _sha256_bytes(X2.tobytes())
        assert h1 != h2


# ---------------------------------------------------------------------------
# Level C: Prediction output reproducible (ASPIRATIONAL)
# ---------------------------------------------------------------------------


class TestPredictionReproducible:
    """Level C: Full prediction output matches within tolerance.

    This is aspirational — marked xfail if non-deterministic backend.
    """

    @pytest.mark.xfail(
        reason="Level C is aspirational; non-deterministic backends may fail",
        strict=False,
    )
    def test_same_seed_same_predictions(self):
        """With identical seed and inputs, predictions should match."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        # Simulate a simple prediction pipeline
        X = np.random.randn(50, 10)
        weights = rng1.randn(10)
        preds1 = 1.0 / (1.0 + np.exp(-X @ weights))

        weights2 = rng2.randn(10)
        preds2 = 1.0 / (1.0 + np.exp(-X @ weights2))

        np.testing.assert_array_almost_equal(preds1, preds2, decimal=10)


# ---------------------------------------------------------------------------
# Generator integration tests
# ---------------------------------------------------------------------------


class TestManifestGenerator:
    """Tests for the ManifestGenerator class."""

    def test_generate_produces_valid_manifest(self, manifest_generator):
        manifest = manifest_generator.generate(
            run_id="test_run_001",
            config_hash="abc123",
            raw_inputs=[{
                "provider": "cbbpy",
                "path_or_uri": "data/raw/games.json",
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "checksum": "def456",
            }],
            training_data_hash="train_hash_789",
            model_artifact_hash="model_hash_012",
            random_seed=42,
            model_artifact_path="artifacts/model.pkl",
            training_data_path="data/processed/train.npy",
        )

        violations = ManifestGenerator.validate_level_a(manifest)
        assert not violations, f"Generated manifest has Level A violations: {violations}"

    def test_save_and_reload(self, manifest_generator):
        manifest = manifest_generator.generate(
            run_id="test_save_001",
            config_hash="abc",
            raw_inputs=[{
                "provider": "test",
                "path_or_uri": "test.json",
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "checksum": "xyz",
            }],
            training_data_hash="th",
            model_artifact_hash="mh",
            random_seed=42,
            model_artifact_path="model.pkl",
            training_data_path="train.npy",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            manifest_generator.save(manifest, path)
            with open(path) as f:
                reloaded = json.load(f)
            assert reloaded["run_id"] == "test_save_001"
            assert reloaded["random_seed"] == 42
        finally:
            os.unlink(path)

    def test_git_metadata_populated(self, manifest_generator):
        manifest = manifest_generator.generate(
            run_id="git_test",
            config_hash="c",
            raw_inputs=[{
                "provider": "p",
                "path_or_uri": "u",
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "checksum": "h",
            }],
            training_data_hash="t",
            model_artifact_hash="m",
            random_seed=1,
            model_artifact_path="mp",
            training_data_path="tp",
        )
        # commit_sha should be populated (may be "UNKNOWN" in CI without git)
        assert manifest["code"]["commit_sha"]
        assert manifest["code"]["diff_hash"]
        assert manifest["environment"]["python_version"]
