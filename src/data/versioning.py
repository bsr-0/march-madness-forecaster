"""Lightweight data versioning via filesystem snapshots.

Provides snapshot/restore functionality for data directories, enabling
rollback and reproducibility without external version control.

Implements Agent Directive V7 S19 (data engineering — data versioning).

Usage:
    from src.data.versioning import snapshot_data_dir, list_snapshots, restore_snapshot

    snapshot_id = snapshot_data_dir("data/raw", label="pre-tournament")
    snapshots = list_snapshots()
    restore_snapshot(snapshot_id, "data/raw")
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_DIR = "data/snapshots"


@dataclass
class SnapshotManifest:
    """Manifest for a data snapshot."""

    snapshot_id: str
    timestamp: str
    label: Optional[str] = None
    source_dir: str = ""
    file_count: int = 0
    total_size_bytes: int = 0
    file_hashes: Dict[str, str] = field(default_factory=dict)  # relative_path → SHA-256

    def to_dict(self) -> Dict:
        return asdict(self)


def snapshot_data_dir(
    data_dir: str = "data/raw",
    label: Optional[str] = None,
    snapshot_base: str = DEFAULT_SNAPSHOT_DIR,
) -> str:
    """Create a snapshot of a data directory.

    Copies all files to ``snapshot_base/{timestamp}_{label}/`` and writes
    a manifest with file hashes for integrity verification.

    Args:
        data_dir: Source data directory to snapshot.
        label: Optional human-readable label (e.g., "pre-tournament").
        snapshot_base: Base directory for snapshots.

    Returns:
        Snapshot ID (directory name).
    """
    source = Path(data_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"{timestamp_str}_{label}" if label else timestamp_str
    # Sanitize label for filesystem
    snapshot_id = snapshot_id.replace(" ", "_").replace("/", "_")

    dest = Path(snapshot_base) / snapshot_id
    dest.mkdir(parents=True, exist_ok=True)

    file_hashes: Dict[str, str] = {}
    total_size = 0
    file_count = 0

    for src_file in sorted(source.rglob("*")):
        if not src_file.is_file():
            continue
        # Skip hidden files and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in src_file.parts):
            continue

        rel_path = src_file.relative_to(source)
        dst_file = dest / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(src_file), str(dst_file))
        file_hashes[str(rel_path)] = _sha256_file(src_file)
        total_size += src_file.stat().st_size
        file_count += 1

    manifest = SnapshotManifest(
        snapshot_id=snapshot_id,
        timestamp=now.isoformat(),
        label=label,
        source_dir=str(source),
        file_count=file_count,
        total_size_bytes=total_size,
        file_hashes=file_hashes,
    )

    manifest_path = dest / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(
        "Snapshot '%s' created: %d files, %.1f MB",
        snapshot_id,
        file_count,
        total_size / (1024 * 1024),
    )
    return snapshot_id


def list_snapshots(
    snapshot_base: str = DEFAULT_SNAPSHOT_DIR,
) -> List[SnapshotManifest]:
    """List all available snapshots.

    Args:
        snapshot_base: Base directory for snapshots.

    Returns:
        List of snapshot manifests, sorted by timestamp (newest first).
    """
    base = Path(snapshot_base)
    if not base.is_dir():
        return []

    manifests: List[SnapshotManifest] = []
    for entry in sorted(base.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                manifests.append(SnapshotManifest(**data))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("Skipping invalid manifest %s: %s", manifest_path, exc)

    return manifests


def restore_snapshot(
    snapshot_id: str,
    target_dir: str = "data/raw",
    snapshot_base: str = DEFAULT_SNAPSHOT_DIR,
    verify: bool = True,
) -> int:
    """Restore a data directory from a snapshot.

    Args:
        snapshot_id: Snapshot directory name.
        target_dir: Destination directory to restore into.
        snapshot_base: Base directory for snapshots.
        verify: If True, verify file hashes after restore.

    Returns:
        Number of files restored.

    Raises:
        FileNotFoundError: If snapshot doesn't exist.
        ValueError: If hash verification fails.
    """
    snapshot_dir = Path(snapshot_base) / snapshot_id
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")

    manifest_path = snapshot_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = SnapshotManifest(**json.load(f))

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    restored = 0
    for src_file in sorted(snapshot_dir.rglob("*")):
        if not src_file.is_file() or src_file.name == "manifest.json":
            continue

        rel_path = src_file.relative_to(snapshot_dir)
        dst_file = target / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(src_file), str(dst_file))
        restored += 1

        if verify and manifest and str(rel_path) in manifest.file_hashes:
            expected = manifest.file_hashes[str(rel_path)]
            actual = _sha256_file(dst_file)
            if actual != expected:
                raise ValueError(
                    f"Hash mismatch for {rel_path}: expected {expected[:16]}..., "
                    f"got {actual[:16]}..."
                )

    logger.info(
        "Restored snapshot '%s': %d files to %s",
        snapshot_id,
        restored,
        target_dir,
    )
    return restored


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
