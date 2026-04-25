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

# Manually-bumped sampler-logic version. Lives in the cache_key as the
# "code_version" component for new entries (Phase 2+ probability cache,
# Phase 3+ bracket cache, Phase 4 live-loop writes). Bump when ANY of:
#   - sampler RNG consumption pattern changes (sample_*_brackets in
#     scripts/mc_pool_backtest.py)
#   - build_torvik_round_probabilities or upstream Log5/MC math changes
#   - picks_by_round / brackets_to_array encoding changes
#   - make_mode_rng derivation below changes
# Pre-existing migration entries (rng_seed='frozen-2026-04-18') were
# hashed with a one-time git SHA; they remain loadable independently.
STRATEGY_CACHE_VERSION = 1

# Shared data/model version strings for the production torvik bracket
# cache. Both the offline builder (scripts/build_bracket_cache.py) and
# the live experiment loop's --write-cache path use these so cache keys
# match round-trip. The probability cache lives one layer up; its
# constants travel with it.
BRACKET_DATA_VERSION = "torvik+seed=42"
BRACKET_MODEL_VERSION = "torvik-barthag-v1"
PROBABILITY_DATA_VERSION = "tournament_seeds+torvik_barthag"
PROBABILITY_MODEL_VERSION = "torvik-barthag-v1"

# Parent seed feeding make_mode_rng — kept as a named constant so the
# cache key component is greppable. Matches build_bracket_cache.py's
# RNG_SEED.
CACHE_PARENT_SEED = 42

# Round layout in the (n_brackets, 63) alphabetical-within-round encoding.
# Sum of sizes == N_GAMES.
_ROUND_ORDER = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
_ROUND_SIZES = {"R64": 32, "R32": 16, "S16": 8, "E8": 4, "F4": 2, "CHAMP": 1}
_ROUND_OFFSETS = {"R64": 0, "R32": 32, "S16": 48, "E8": 56, "F4": 60, "CHAMP": 62}

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


def save_probabilities(
    cache_key: str,
    probabilities: np.ndarray,
    team_ids: np.ndarray,
    *,
    strategy_id: str,
    year: int,
    rng_seed: SeedT,
    code_version: str,
    data_version: str,
    model_version: str,
    provenance: Optional[dict] = None,
) -> CacheEntry:
    """Cache (n_teams, 6) round-advancement probabilities for one (source, year).

    `strategy_id` here names the probability source (base + adjustments only,
    no construction mode) — multiple bracket strategies can share one
    probability artifact. team_ids is the parallel uint16 column mapping
    rows back to the global team_id_lookup.
    """
    if probabilities.dtype != np.float32:
        raise ValueError(f"probabilities must be float32, got {probabilities.dtype}")
    if probabilities.ndim != 2 or probabilities.shape[1] != 6:
        raise ValueError(f"probabilities must be (n_teams, 6), got {probabilities.shape}")
    if team_ids.dtype != np.uint16:
        raise ValueError(f"team_ids must be uint16, got {team_ids.dtype}")
    if team_ids.shape != (probabilities.shape[0],):
        raise ValueError(f"team_ids must be ({probabilities.shape[0]},), got {team_ids.shape}")
    rel_path = f"probabilities/{cache_key}.npz"
    abs_path = CACHE_ROOT / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(abs_path, probabilities=probabilities, team_ids=team_ids)
    return CacheEntry(
        strategy_id=strategy_id,
        year=year,
        cache_key=cache_key,
        rng_seed=rng_seed,
        code_version=code_version,
        data_version=data_version,
        model_version=model_version,
        artifact_kind="probabilities",
        artifact_path=rel_path,
        content_hash=content_hash(abs_path),
        shape=tuple(probabilities.shape),
        dtype="float32",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        provenance=provenance or {},
    )


def load_probabilities(cache_key: str, manifest: Optional[Manifest] = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, team_ids). Raises on missing or hash mismatch."""
    manifest = manifest or Manifest.load()
    matches = [e for e in manifest.entries if e.cache_key == cache_key and e.artifact_kind == "probabilities"]
    if not matches:
        raise KeyError(f"No probabilities cache entry for key {cache_key}")
    entry = matches[0]
    abs_path = CACHE_ROOT / entry.artifact_path
    if content_hash(abs_path) != entry.content_hash:
        raise ValueError(f"Content hash mismatch for {abs_path} — cache tampered or stale")
    with np.load(abs_path) as z:
        return z["probabilities"], z["team_ids"]


def save_outcomes(
    year: int,
    outcomes: np.ndarray,
    *,
    code_version: str,
    data_version: str,
    provenance: Optional[dict] = None,
) -> CacheEntry:
    """Cache the (63,) uint16 ground-truth winners array for a single year.

    Outcomes are not strategy-dependent, so the cache_key is just the year-
    scoped path; the manifest entry still carries content_hash for tamper
    detection. Picks per round are stored alphabetically (matches the
    bracket-cache convention) so validation = (brackets == outcomes[None, :]).
    """
    if outcomes.dtype != np.uint16:
        raise ValueError(f"outcomes must be uint16, got {outcomes.dtype}")
    if outcomes.shape != (N_GAMES,):
        raise ValueError(f"outcomes must be ({N_GAMES},), got {outcomes.shape}")
    rel_path = f"outcomes/{year}.npz"
    abs_path = CACHE_ROOT / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(abs_path, winners=outcomes)
    return CacheEntry(
        strategy_id="GROUND_TRUTH",
        year=year,
        cache_key=f"outcomes-{year}",
        rng_seed="N/A",
        code_version=code_version,
        data_version=data_version,
        model_version="N/A",
        artifact_kind="outcomes",
        artifact_path=rel_path,
        content_hash=content_hash(abs_path),
        shape=tuple(outcomes.shape),
        dtype="uint16",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        provenance=provenance or {},
    )


def load_outcomes(year: int, manifest: Optional[Manifest] = None) -> np.ndarray:
    manifest = manifest or Manifest.load()
    matches = [e for e in manifest.entries if e.artifact_kind == "outcomes" and e.year == year]
    if not matches:
        raise KeyError(f"No outcomes cache entry for year {year}")
    entry = matches[0]
    abs_path = CACHE_ROOT / entry.artifact_path
    if content_hash(abs_path) != entry.content_hash:
        raise ValueError(f"Content hash mismatch for {abs_path} — cache tampered or stale")
    with np.load(abs_path) as z:
        return z["winners"]


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


def make_mode_rng(
    mode_name: str,
    year: int,
    parent_seed: int = 42,
) -> np.random.Generator:
    """Deterministic per-(mode, year) child rng derived from a parent seed.

    Decouples per-mode rng streams so caching one mode's brackets never
    perturbs another mode's draws — and so the live experiment loop and
    the offline cache builder produce identical brackets regardless of
    mode iteration order. The (mode_name, year) pair selects a unique
    child stream of the parent_seed entropy.
    """
    h = hashlib.sha256(f"{mode_name}|{year}".encode()).digest()
    spawn_key = tuple(int.from_bytes(h[i : i + 4], "big") for i in (0, 4, 8, 12))
    ss = np.random.SeedSequence(entropy=parent_seed, spawn_key=spawn_key)
    return np.random.default_rng(ss)


def brackets_to_array(
    model_brackets: np.ndarray,
    first_round: list,
    lookup: dict[str, int],
) -> np.ndarray:
    """Encode (n_brackets, 63) bool dense → (n_brackets, 63) uint16 alphabetical-within-round.

    Tree-walk: for each round, pair-walk the current matchup list using the
    dense bool to pick a winner, sort the round's winners alphabetically,
    and stamp them into the cache row. Self-contained — no dependency on
    picks_by_round in src.simulation.pool_competition (so the data layer
    doesn't pull scipy/sklearn at import time).

    Inverse of ``array_to_brackets`` below; round-trip lock-tested in
    tests/test_strategy_cache_loop_integration.py.
    """
    if model_brackets.ndim != 2 or model_brackets.shape[1] != N_GAMES:
        raise ValueError(f"model_brackets must be (n, {N_GAMES}), got {model_brackets.shape}")
    if len(first_round) != 64:
        raise ValueError(f"first_round must have 64 teams, got {len(first_round)}")

    n = model_brackets.shape[0]
    out = np.zeros((n, N_GAMES), dtype=np.uint16)
    for m in range(n):
        current = list(first_round)
        gi = 0
        per_round: dict[str, list[str]] = {}
        for rnd in _ROUND_ORDER:
            nxt: list[str] = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                winner = t1 if model_brackets[m, gi] else t2
                nxt.append(winner)
                gi += 1
            per_round[rnd] = sorted(nxt)
            current = nxt
        cursor = 0
        for rnd in _ROUND_ORDER:
            size = _ROUND_SIZES[rnd]
            for i, name in enumerate(per_round[rnd]):
                out[m, cursor + i] = lookup[name]
            cursor += size
    return out


def array_to_brackets(
    bracket_array: np.ndarray,
    first_round: list,
    lookup: dict[str, int],
) -> np.ndarray:
    """Decode (n_brackets, 63) uint16 alphabetical-within-round → (n_brackets, 63) bool.

    Inverse of ``brackets_to_array`` in scripts/build_bracket_cache.py. The
    cached form stores per-round winner team-ids alphabetically; the dense
    form stores one bool per game (True = top-listed team of the slot won).
    The dense form is what the experiment loop's scoring/ranking pipeline
    consumes, so cache hits must be decoded back to dense before use.

    Walk: for each round, look up that round's cached winner *set*, then
    pair-walk the current matchup list to set the per-game bool based on
    which side's team is in the set. Order within a round doesn't matter
    because we only need set membership.
    """
    if bracket_array.dtype != np.uint16:
        raise ValueError(f"bracket_array must be uint16, got {bracket_array.dtype}")
    if bracket_array.ndim != 2 or bracket_array.shape[1] != N_GAMES:
        raise ValueError(f"bracket_array must be (n_brackets, {N_GAMES}), got {bracket_array.shape}")
    if len(first_round) != 64:
        raise ValueError(f"first_round must have 64 teams, got {len(first_round)}")

    inverse_lookup = {v: k for k, v in lookup.items()}
    n = bracket_array.shape[0]
    out = np.zeros((n, N_GAMES), dtype=bool)

    for m in range(n):
        winners_by_round: dict[str, set[str]] = {}
        for rnd in _ROUND_ORDER:
            offset = _ROUND_OFFSETS[rnd]
            size = _ROUND_SIZES[rnd]
            ids = bracket_array[m, offset : offset + size]
            winners_by_round[rnd] = {inverse_lookup[int(tid)] for tid in ids}

        current = list(first_round)
        gi = 0
        for rnd in _ROUND_ORDER:
            nxt: list[str] = []
            for g in range(0, len(current), 2):
                t1, t2 = current[g], current[g + 1]
                if t1 in winners_by_round[rnd]:
                    out[m, gi] = True
                    nxt.append(t1)
                elif t2 in winners_by_round[rnd]:
                    out[m, gi] = False
                    nxt.append(t2)
                else:
                    raise ValueError(
                        f"bracket {m} round {rnd} game {g}: neither {t1!r} nor {t2!r} "
                        f"in cached winner set — cache inconsistent with first_round layout"
                    )
                gi += 1
            current = nxt
    return out
