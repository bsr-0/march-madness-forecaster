"""Frozen experiment configuration for exact reproducibility.

A ``FrozenExperimentConfig`` captures the full pipeline configuration,
data hashes, code version, and reproducibility hash at a point in time
so that an experiment can be exactly reproduced later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class FrozenExperimentConfig:
    """Immutable snapshot of everything needed to reproduce an experiment.

    Fields
    ------
    experiment_id : str
        Unique identifier for the experiment.
    pipeline_config : Dict[str, Any]
        Full pipeline configuration as a plain dict (from
        ``dataclasses.asdict(config)``).
    dataset_hashes : Dict[str, str]
        Mapping of data-file paths to their SHA-256 hex digests at
        freeze time.
    code_version : str
        Git SHA or tag identifying the code snapshot.
    reproducibility_hash : str
        Combined hash of code + data + config.
    frozen_at : str
        ISO 8601 timestamp of when the config was frozen.
    """

    experiment_id: str
    pipeline_config: Dict[str, Any]
    dataset_hashes: Dict[str, str]
    code_version: str
    reproducibility_hash: str
    frozen_at: str  # ISO 8601

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to a deterministic JSON string."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> FrozenExperimentConfig:
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def recreate_config(self) -> Any:
        """Reconstruct a ``SOTAPipelineConfig`` from the stored dict.

        Uses a lazy import to avoid circular dependencies between the
        reproducibility and pipeline packages.

        Returns
        -------
        SOTAPipelineConfig
            A new config instance with field values from
            ``self.pipeline_config``.
        """
        from src.pipeline.sota import SOTAPipelineConfig

        # Filter to known fields to handle schema evolution gracefully.
        known_fields = {f.name for f in SOTAPipelineConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in self.pipeline_config.items() if k in known_fields}
        return SOTAPipelineConfig(**filtered)

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_data_integrity(self) -> bool:
        """Re-hash data files and compare against stored hashes.

        Returns ``True`` if every file referenced in
        ``self.dataset_hashes`` still has the same SHA-256 digest.
        Returns ``False`` if any file has changed or is missing.
        """
        for file_path, expected_hash in self.dataset_hashes.items():
            path = Path(file_path)
            if not path.is_file():
                return False
            if self._sha256_file(path) != expected_hash:
                return False
        return True

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute the SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
