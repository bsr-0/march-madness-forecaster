"""Versioned model artifact storage.

Provides directory-based model versioning with JSON sidecar metadata.
Models are serialized via joblib and tracked with version IDs derived
from timestamps and config hashes.

Usage:
    from src.deployment.model_store import ModelStore

    store = ModelStore()
    version = store.save("ensemble_v1", model, config, {"brier_score": 0.189})
    model, version = store.load(version.version_id)
    store.promote(version.version_id)
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a versioned model artifact."""

    version_id: str = ""
    model_name: str = ""
    created_at: str = ""
    config_hash: str = ""
    brier_score: Optional[float] = None
    artifact_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_production: bool = False


class ModelStore:
    """Directory-based versioned model storage."""

    def __init__(self, store_dir: str = "data/models") -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_name: str,
        artifact: Any,
        config: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Serialize a model artifact and save with metadata sidecar."""
        try:
            import joblib
        except ImportError:
            logger.warning("joblib not available; model will not be serialized")
            joblib = None

        now = datetime.now(timezone.utc)
        config_hash = self._hash_config(config)
        version_id = f"{model_name}_{now.strftime('%Y%m%d_%H%M%S')}_{config_hash[:8]}"

        version_dir = self._store_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = str(version_dir / "model.joblib")
        if joblib is not None:
            joblib.dump(artifact, artifact_path)
        else:
            artifact_path = ""

        metrics = metrics or {}
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            created_at=now.isoformat(),
            config_hash=config_hash,
            brier_score=metrics.get("brier_score"),
            artifact_path=artifact_path,
            metadata=metrics,
            is_production=False,
        )

        sidecar_path = version_dir / "metadata.json"
        with open(sidecar_path, "w") as f:
            json.dump(asdict(version), f, indent=2, default=str)

        logger.info("Saved model version: %s", version_id)
        return version

    def load(self, version_id: str) -> Tuple[Any, ModelVersion]:
        """Load a model artifact and its metadata."""
        try:
            import joblib
        except ImportError:
            raise RuntimeError("joblib required to load models")

        version_dir = self._store_dir / version_id
        sidecar_path = version_dir / "metadata.json"

        if not sidecar_path.exists():
            raise FileNotFoundError(f"Model version not found: {version_id}")

        with open(sidecar_path) as f:
            data = json.load(f)
        version = ModelVersion(**{k: v for k, v in data.items()
                                  if k in ModelVersion.__dataclass_fields__})

        artifact_path = version_dir / "model.joblib"
        if artifact_path.exists():
            artifact = joblib.load(artifact_path)
        else:
            artifact = None

        return artifact, version

    def list_versions(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """List all model versions, optionally filtered by model name."""
        versions: List[ModelVersion] = []
        if not self._store_dir.exists():
            return versions

        for version_dir in sorted(self._store_dir.iterdir()):
            sidecar = version_dir / "metadata.json"
            if not sidecar.exists():
                continue
            try:
                with open(sidecar) as f:
                    data = json.load(f)
                v = ModelVersion(**{k: v for k, v in data.items()
                                    if k in ModelVersion.__dataclass_fields__})
                if model_name is None or v.model_name == model_name:
                    versions.append(v)
            except (json.JSONDecodeError, TypeError):
                continue

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_production(self, model_name: Optional[str] = None) -> Optional[ModelVersion]:
        """Get the current production model version."""
        for v in self.list_versions(model_name):
            if v.is_production:
                return v
        return None

    def promote(self, version_id: str) -> ModelVersion:
        """Mark a version as production, demoting any existing production version."""
        # Demote current production
        for v in self.list_versions():
            if v.is_production:
                v.is_production = False
                self._save_sidecar(v)

        # Promote target
        version_dir = self._store_dir / version_id
        sidecar_path = version_dir / "metadata.json"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Model version not found: {version_id}")

        with open(sidecar_path) as f:
            data = json.load(f)
        version = ModelVersion(**{k: v for k, v in data.items()
                                  if k in ModelVersion.__dataclass_fields__})
        version.is_production = True
        self._save_sidecar(version)

        logger.info("Promoted %s to production", version_id)
        return version

    def _save_sidecar(self, version: ModelVersion) -> None:
        """Write metadata sidecar JSON."""
        version_dir = self._store_dir / version.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = version_dir / "metadata.json"
        with open(sidecar_path, "w") as f:
            json.dump(asdict(version), f, indent=2, default=str)

    @staticmethod
    def _hash_config(config: Any) -> str:
        """Generate a hash from config for version deduplication."""
        if config is None:
            return "noconfig"
        if hasattr(config, "__dataclass_fields__"):
            config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {"repr": repr(config)}
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
