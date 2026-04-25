"""Strategy-prediction cache: deterministic, content-addressed storage of
per-strategy bracket predictions for the experiment loop.

Schema:
  artifacts/strategy_cache/
    manifest.json                       -- index + content hashes
    team_id_lookup.json                 -- team name <-> uint16 id
    brackets/{cache_key}.npz            -- (n_brackets, 63) uint16 team_ids
    probabilities/{cache_key}.npz       -- (n_teams, n_rounds) float32   [Phase 2+]
    opponent_pool/{year}_seed{N}.npz    -- (n_opponents, 63) uint16      [Phase 2+]
    outcomes/{year}.npz                 -- (63,) uint16                  [Phase 2+]

cache_key = sha256(strategy_id || year || rng_seed || code_version
                   || data_version || model_version)[:16]
Any input change shifts the key and forces a recompute. content_hash on the
on-disk NPZ catches tampering or stale state independent of the key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

CACHE_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "strategy_cache"
MANIFEST_PATH = CACHE_ROOT / "manifest.json"
TEAM_LOOKUP_PATH = CACHE_ROOT / "team_id_lookup.json"
SCHEMA_VERSION = 1
N_GAMES = 63  # 32 R64 + 16 R32 + 8 S16 + 4 E8 + 2 F4 + 1 CHAMP

SeedT = Union[int, str]


def compute_cache_key(
    strategy_id: str,
    year: int,
    rng_seed: SeedT,
    code_version: str,
    data_version: str,
    model_version: str,
) -> str:
    payload = "|".join(
        [
            f"schema={SCHEMA_VERSION}",
            f"strategy={strategy_id}",
            f"year={year}",
            f"rng_seed={rng_seed}",
            f"code={code_version}",
            f"data={data_version}",
            f"model={model_version}",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class CacheEntry:
    strategy_id: str
    year: int
    cache_key: str
    rng_seed: SeedT
    code_version: str
    data_version: str
    model_version: str
    artifact_kind: str
    artifact_path: str
    content_hash: str
    shape: tuple
    dtype: str
    generated_at: str
    provenance: dict = field(default_factory=dict)


@dataclass
class Manifest:
    schema_version: int
    entries: list[CacheEntry]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Manifest":
        path = path or MANIFEST_PATH
        if not path.exists():
            return cls(schema_version=SCHEMA_VERSION, entries=[])
        raw = json.loads(path.read_text())
        return cls(
            schema_version=raw["schema_version"],
            entries=[CacheEntry(**e) for e in raw["entries"]],
        )

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MANIFEST_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "entries": [_serialize_entry(e) for e in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2))

    def append(self, entry: CacheEntry) -> None:
        # Idempotent on (cache_key, artifact_kind): a re-run replaces the prior entry.
        self.entries = [
            e for e in self.entries if not (e.cache_key == entry.cache_key and e.artifact_kind == entry.artifact_kind)
        ]
        self.entries.append(entry)


def _serialize_entry(entry: CacheEntry) -> dict:
    d = asdict(entry)
    d["shape"] = list(entry.shape)
    return d


def load_team_lookup() -> dict[str, int]:
    if not TEAM_LOOKUP_PATH.exists():
        return {}
    return json.loads(TEAM_LOOKUP_PATH.read_text())


def save_team_lookup(lookup: dict[str, int]) -> None:
    TEAM_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEAM_LOOKUP_PATH.write_text(json.dumps(lookup, indent=2, sort_keys=True))


def save_brackets(
    cache_key: str,
    brackets: np.ndarray,
    *,
    strategy_id: str,
    year: int,
    rng_seed: SeedT,
    code_version: str,
    data_version: str,
    model_version: str,
    provenance: Optional[dict] = None,
) -> CacheEntry:
    if brackets.dtype != np.uint16:
        raise ValueError(f"brackets must be uint16, got {brackets.dtype}")
    if brackets.ndim != 2 or brackets.shape[1] != N_GAMES:
        raise ValueError(f"brackets must be (n_brackets, {N_GAMES}), got {brackets.shape}")
    rel_path = f"brackets/{cache_key}.npz"
    abs_path = CACHE_ROOT / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(abs_path, brackets=brackets)
    return CacheEntry(
        strategy_id=strategy_id,
        year=year,
        cache_key=cache_key,
        rng_seed=rng_seed,
        code_version=code_version,
        data_version=data_version,
        model_version=model_version,
        artifact_kind="brackets",
        artifact_path=rel_path,
        content_hash=content_hash(abs_path),
        shape=tuple(brackets.shape),
        dtype="uint16",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        provenance=provenance or {},
    )


def load_brackets(cache_key: str, manifest: Optional[Manifest] = None) -> np.ndarray:
    manifest = manifest or Manifest.load()
    matches = [e for e in manifest.entries if e.cache_key == cache_key and e.artifact_kind == "brackets"]
    if not matches:
        raise KeyError(f"No brackets cache entry for key {cache_key}")
    entry = matches[0]
    abs_path = CACHE_ROOT / entry.artifact_path
    if content_hash(abs_path) != entry.content_hash:
        raise ValueError(f"Content hash mismatch for {abs_path} — cache tampered or stale")
    with np.load(abs_path) as z:
        return z["brackets"]


def verify_manifest(manifest: Optional[Manifest] = None) -> list[str]:
    """Return list of integrity issues. Empty list = healthy. Used by lock test."""
    manifest = manifest or Manifest.load()
    issues: list[str] = []
    for entry in manifest.entries:
        abs_path = CACHE_ROOT / entry.artifact_path
        if not abs_path.exists():
            issues.append(f"missing artifact: {entry.artifact_path}")
            continue
        if content_hash(abs_path) != entry.content_hash:
            issues.append(f"content hash mismatch: {entry.artifact_path}")
    return issues
