"""Read JSON files that may have been archived as ``<name>.json.gz``.

Large raw caches (play-by-play, box scores) are gzipped in place to save disk.
``resolve_json`` points a plain ``.json`` path at its ``.gz`` sibling when only
the compressed file exists, and ``load_json`` reads either form.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def resolve_json(path: PathLike) -> Path:
    """Return ``path`` if it exists, else its ``.gz`` sibling if that exists, else ``path``."""
    path = Path(path)
    if path.exists():
        return path
    gz = path.with_name(path.name + ".gz")
    return gz if gz.exists() else path


def load_json(path: PathLike) -> Any:
    """Load JSON from ``path``, transparently handling a ``.gz`` archive."""
    path = resolve_json(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)
