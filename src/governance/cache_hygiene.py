"""Small cache-hygiene helpers used by ingestion and pipeline pre-checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


CACHE_PROTOCOL_VERSION = 1

_EPHEMERAL_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def fingerprint_context(context: Dict[str, Any]) -> str:
    """Return a stable fingerprint for a DAG execution context."""
    payload = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def purge_ephemeral_cache_paths(repo_root: Path) -> List[str]:
    """Best-effort removal of throwaway cache directories inside the repo."""
    removed: List[str] = []
    for path in repo_root.rglob("*"):
        if path.name not in _EPHEMERAL_NAMES:
            continue
        if not path.is_dir():
            continue
        try:
            shutil.rmtree(path)
            removed.append(str(path))
        except OSError:
            continue
    return removed
