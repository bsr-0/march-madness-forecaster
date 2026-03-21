"""Content-addressable artifact store for experiment outputs.

Stores models, configs, and predictions as content-addressed files with
an append-only JSONL manifest for metadata tracking.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ArtifactManifest:
    """Metadata record for a single stored artifact."""

    artifact_id: str
    artifact_type: str  # "model", "config", "predictions", "features"
    created_at: str  # ISO 8601
    experiment_id: str
    file_path: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArtifactManifest:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class ArtifactStore:
    """Content-addressable store for experiment artifacts.

    Artifacts are stored under ``store_dir/<artifact_type>/<sha[:8]>.<ext>``
    and tracked via an append-only JSONL manifest.

    Parameters
    ----------
    store_dir : str
        Root directory for artifact storage.
    """

    def __init__(self, store_dir: str = "data/artifacts") -> None:
        self.store_dir = Path(store_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: Any,
        experiment_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Pickle and store a model artifact.  Returns the artifact_id."""
        raw = pickle.dumps(model)
        sha = hashlib.sha256(raw).hexdigest()
        artifact_id = sha[:8]

        # Check for dedup — same content already stored.
        existing = self._find_manifest(artifact_id)
        if existing is not None:
            return artifact_id

        rel_path = Path("model") / f"{artifact_id}.pkl"
        abs_path = self.store_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(raw)

        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="model",
            created_at=datetime.now(timezone.utc).isoformat(),
            experiment_id=experiment_id,
            file_path=str(rel_path),
            size_bytes=len(raw),
            metadata=metadata or {},
        )
        self._write_manifest(manifest)
        return artifact_id

    def save_config(self, config: Any, experiment_id: str) -> str:
        """Serialize and store a dataclass config as JSON.  Returns the artifact_id."""
        from dataclasses import asdict as _asdict

        config_dict = _asdict(config)
        raw = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        artifact_id = sha[:8]

        existing = self._find_manifest(artifact_id)
        if existing is not None:
            return artifact_id

        rel_path = Path("config") / f"{artifact_id}.json"
        abs_path = self.store_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(raw)

        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="config",
            created_at=datetime.now(timezone.utc).isoformat(),
            experiment_id=experiment_id,
            file_path=str(rel_path),
            size_bytes=len(raw),
        )
        self._write_manifest(manifest)
        return artifact_id

    def save_predictions(
        self, predictions: Dict[str, float], experiment_id: str
    ) -> str:
        """Serialize and store a predictions dict as JSON.  Returns the artifact_id."""
        return self._save_json_artifact(
            predictions,
            experiment_id=experiment_id,
            artifact_type="predictions",
        )

    def save_feature_manifest(
        self, manifest: Dict[str, Any], experiment_id: str
    ) -> str:
        """Serialize and store a feature manifest as a dedicated artifact."""
        return self._save_json_artifact(
            manifest,
            experiment_id=experiment_id,
            artifact_type="features",
        )

    def _save_json_artifact(
        self,
        payload: Dict[str, Any],
        *,
        experiment_id: str,
        artifact_type: str,
    ) -> str:
        """Serialize and store a JSON artifact.  Returns the artifact_id."""
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        artifact_id = sha[:8]

        existing = self._find_manifest(artifact_id)
        if existing is not None:
            return artifact_id

        rel_path = Path(artifact_type) / f"{artifact_id}.json"
        abs_path = self.store_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(raw)

        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            experiment_id=experiment_id,
            file_path=str(rel_path),
            size_bytes=len(raw),
        )
        self._write_manifest(manifest)
        return artifact_id

    def load_artifact(self, artifact_id: str) -> Any:
        """Load an artifact by its ID.

        Returns the deserialized object (unpickled model, parsed JSON
        dict, etc.).

        Raises
        ------
        FileNotFoundError
            If no artifact with the given ID exists.
        """
        manifest = self._find_manifest(artifact_id)
        if manifest is None:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")

        abs_path = self.store_dir / manifest.file_path
        if manifest.artifact_type == "model":
            return pickle.loads(abs_path.read_bytes())
        else:
            return json.loads(abs_path.read_text(encoding="utf-8"))

    def list_artifacts(
        self,
        experiment_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[ArtifactManifest]:
        """Return manifest entries, optionally filtered."""
        manifests = self._load_manifests()
        if experiment_id is not None:
            manifests = [m for m in manifests if m.experiment_id == experiment_id]
        if artifact_type is not None:
            manifests = [m for m in manifests if m.artifact_type == artifact_type]
        return manifests

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_manifest(self, manifest: ArtifactManifest) -> None:
        """Append a manifest entry to the JSONL ledger."""
        manifest_path = self.store_dir / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest.to_dict()) + "\n")

    def _load_manifests(self) -> List[ArtifactManifest]:
        """Read all manifest entries from the JSONL ledger."""
        manifest_path = self.store_dir / "manifest.jsonl"
        if not manifest_path.exists():
            return []
        manifests: List[ArtifactManifest] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    manifests.append(ArtifactManifest.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return manifests

    def _find_manifest(self, artifact_id: str) -> Optional[ArtifactManifest]:
        """Look up a single manifest entry by artifact_id."""
        for m in self._load_manifests():
            if m.artifact_id == artifact_id:
                return m
        return None
