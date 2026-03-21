"""Runtime cache hygiene helpers for leakage-sensitive workflows.

This module centralizes two related concerns:

1. **Ephemeral repo caches** such as ``__pycache__`` and ``.pytest_cache``
   should never be allowed to masquerade as durable modeling artifacts.
2. **Idempotency cache markers** should be bound to an execution context so
   stale markers from a prior season / split / config cannot be reused after
   the context changes.

The goal is not to treat Python bytecode as a statistical leakage source by
itself.  Rather, we enforce a clean separation between:
    - reproducible, explicitly versioned artifacts, and
    - throwaway local runtime caches.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CACHE_PROTOCOL_VERSION = 1

EPHEMERAL_CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}

EPHEMERAL_CACHE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}

_VOLATILE_CONTEXT_KEYS = {
    "timestamp",
    "run_timestamp",
    "started_at",
    "finished_at",
    "wall_seconds",
}

_SKIP_SCAN_DIRS = {
    ".git",
    ".venv",
    "node_modules",
}


def _normalize_for_hash(value: Any) -> Any:
    """Recursively normalize values into stable, JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _normalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_for_hash(v) for v in value)
    return value


def fingerprint_context(
    context: Dict[str, Any],
    *,
    include_keys: Sequence[str] | None = None,
) -> str:
    """Create a stable fingerprint for an execution context."""
    if include_keys is None:
        include_keys = sorted(
            key for key in context.keys()
            if key not in _VOLATILE_CONTEXT_KEYS
        )

    payload = {
        str(key): _normalize_for_hash(context.get(key))
        for key in include_keys
        if key in context
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def find_ephemeral_cache_paths(repo_root: Path) -> List[Path]:
    """Return repo-local ephemeral cache paths that should be purged."""
    repo_root = repo_root.resolve()
    found: List[Path] = []

    for current_root, dirnames, filenames in os.walk(repo_root):
        current = Path(current_root)
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_SCAN_DIRS
        ]

        for dirname in list(dirnames):
            if dirname in EPHEMERAL_CACHE_DIRS:
                found.append(current / dirname)
                dirnames.remove(dirname)

        for filename in filenames:
            path = current / filename
            if path.suffix in EPHEMERAL_CACHE_FILE_SUFFIXES:
                found.append(path)

    return sorted(set(found))


def purge_ephemeral_cache_paths(repo_root: Path) -> List[Path]:
    """Delete ephemeral repo-local caches and return the removed paths."""
    removed: List[Path] = []
    for path in find_ephemeral_cache_paths(repo_root):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed
