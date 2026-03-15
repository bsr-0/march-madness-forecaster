"""Prediction manifest generator for full lineage and reproducibility.

Generates manifests/predictions/<run_id>.manifest.json conforming to
schemas/prediction_manifest.schema.json.

Manifest Validation Levels:
  Level A: Provenance traceable — all fields present and non-empty
  Level B: Feature matrix reproducible — given manifest inputs + code,
           train_X matrix is bit-identical (HARD GATE)
  Level C: Prediction output reproducible — full prediction output matches
           within tolerance (ASPIRATIONAL, tracked but non-blocking)

Usage:
    from src.data.manifest_generator import ManifestGenerator
    gen = ManifestGenerator(config_path="configs/production_2026.json")
    manifest = gen.generate(
        run_id="20260315_prod_2026",
        config_hash=config_hash,
        raw_inputs=[...],
        training_data_hash=train_hash,
        model_artifact_hash=model_hash,
        random_seed=42,
    )
    gen.save(manifest, "manifests/predictions/20260315_prod_2026.manifest.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "FILE_NOT_FOUND"


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _git_commit_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            timeout=5,
        )
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _git_diff_hash() -> str:
    """Get hash of uncommitted changes, or 'clean' if working tree is clean."""
    try:
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True, cwd=str(REPO_ROOT),
            timeout=10,
        )
        if not result.stdout:
            return "clean"
        return _sha256_bytes(result.stdout)
    except Exception:
        return "UNKNOWN"


def _lockfile_hash() -> str:
    """Hash the requirements lockfile for environment reproducibility."""
    lockfile = REPO_ROOT / "requirements-production-lock.txt"
    if lockfile.exists():
        return _sha256_file(str(lockfile))
    # Fallback to requirements.txt
    reqfile = REPO_ROOT / "requirements.txt"
    if reqfile.exists():
        return _sha256_file(str(reqfile))
    return "NO_LOCKFILE"


class ManifestGenerator:
    """Generates prediction manifests with full lineage metadata."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or ""

    def generate(
        self,
        run_id: str,
        config_hash: str,
        raw_inputs: List[Dict[str, str]],
        training_data_hash: str,
        model_artifact_hash: str,
        random_seed: int,
        model_artifact_path: str = "",
        training_data_path: str = "",
        feature_contracts_path: str = "contracts/feature_contracts.yaml",
    ) -> Dict[str, Any]:
        """Generate a complete prediction manifest.

        Args:
            run_id: Unique identifier for this run.
            config_hash: SHA-256 of the config JSON used.
            raw_inputs: List of dicts with provider, path_or_uri, snapshot_timestamp, checksum.
            training_data_hash: SHA-256 of assembled training matrix.
            model_artifact_hash: SHA-256 of serialized model.
            random_seed: Random seed used.
            model_artifact_path: Path to model artifact file.
            training_data_path: Path to training data file.
            feature_contracts_path: Path to feature contracts YAML.

        Returns:
            Dict conforming to prediction_manifest.schema.json.
        """
        contracts_full = str(REPO_ROOT / feature_contracts_path)
        contracts_hash = _sha256_file(contracts_full) if os.path.exists(contracts_full) else "NOT_FOUND"

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code": {
                "commit_sha": _git_commit_sha(),
                "diff_hash": _git_diff_hash(),
            },
            "config": {
                "path": self.config_path,
                "hash": config_hash,
            },
            "raw_inputs": raw_inputs,
            "feature_contracts": {
                "path": feature_contracts_path,
                "hash": contracts_hash,
            },
            "training_data": {
                "path_or_uri": training_data_path,
                "hash": training_data_hash,
            },
            "model": {
                "artifact_path_or_uri": model_artifact_path,
                "artifact_hash": model_artifact_hash,
            },
            "random_seed": random_seed,
            "environment": {
                "python_version": platform.python_version(),
                "os": f"{platform.system()} {platform.release()}",
                "lockfile_hash": _lockfile_hash(),
            },
        }
        return manifest

    def save(self, manifest: Dict[str, Any], output_path: str) -> str:
        """Save manifest to JSON file, creating directories as needed.

        Returns the absolute path of the saved file.
        """
        full_path = Path(output_path)
        if not full_path.is_absolute():
            full_path = REPO_ROOT / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Manifest saved: %s", full_path)
        return str(full_path)

    @staticmethod
    def validate_level_a(manifest: Dict[str, Any]) -> List[str]:
        """Level A: Provenance traceable — all fields present and non-empty.

        Returns list of violations (empty = passed).
        """
        violations = []
        required_top = [
            "run_id", "created_at", "code", "config", "raw_inputs",
            "feature_contracts", "training_data", "model", "random_seed",
            "environment",
        ]
        for key in required_top:
            if key not in manifest:
                violations.append(f"Missing top-level field: {key}")
            elif manifest[key] in (None, "", [], {}):
                violations.append(f"Empty top-level field: {key}")

        # Nested checks
        code = manifest.get("code", {})
        for k in ("commit_sha", "diff_hash"):
            if not code.get(k):
                violations.append(f"Missing or empty code.{k}")

        config = manifest.get("config", {})
        for k in ("path", "hash"):
            if not config.get(k):
                violations.append(f"Missing or empty config.{k}")

        for i, inp in enumerate(manifest.get("raw_inputs", [])):
            for k in ("provider", "path_or_uri", "snapshot_timestamp", "checksum"):
                if not inp.get(k):
                    violations.append(f"Missing or empty raw_inputs[{i}].{k}")

        fc = manifest.get("feature_contracts", {})
        for k in ("path", "hash"):
            if not fc.get(k):
                violations.append(f"Missing or empty feature_contracts.{k}")

        td = manifest.get("training_data", {})
        for k in ("path_or_uri", "hash"):
            if not td.get(k):
                violations.append(f"Missing or empty training_data.{k}")

        model = manifest.get("model", {})
        for k in ("artifact_path_or_uri", "artifact_hash"):
            if not model.get(k):
                violations.append(f"Missing or empty model.{k}")

        env = manifest.get("environment", {})
        for k in ("python_version", "os", "lockfile_hash"):
            if not env.get(k):
                violations.append(f"Missing or empty environment.{k}")

        return violations

    @staticmethod
    def validate_level_b(
        manifest: Dict[str, Any],
        actual_training_data_hash: str,
    ) -> List[str]:
        """Level B: Feature matrix reproducible.

        Given the same inputs described in the manifest, the training data
        hash must match.

        Returns list of violations (empty = passed).
        """
        violations = ManifestGenerator.validate_level_a(manifest)
        expected = manifest.get("training_data", {}).get("hash", "")
        if expected != actual_training_data_hash:
            violations.append(
                f"Level B FAILED: training_data hash mismatch. "
                f"Expected {expected}, got {actual_training_data_hash}"
            )
        return violations
