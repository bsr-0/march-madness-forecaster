"""Phase-1 lock test for the strategy-prediction cache.

Verifies the cache primitives' invariants:
  - cache_key is deterministic and content-addressed (any input change shifts it)
  - manifest entries point at on-disk artifacts whose content_hash matches
  - load_brackets round-trips bit-exactly
  - save then load yields the same uint16 array
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.strategy_cache import (
    CACHE_ROOT,
    Manifest,
    compute_cache_key,
    load_brackets,
    save_brackets,
    verify_manifest,
)


def test_cache_key_is_deterministic() -> None:
    args = dict(
        strategy_id="torvik+f4_first_tv",
        year=2026,
        rng_seed=42,
        code_version="abc123",
        data_version="def456",
        model_version="prod-v1",
    )
    assert compute_cache_key(**args) == compute_cache_key(**args)


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("strategy_id", "torvik+champ_first_tv"),
        ("year", 2025),
        ("rng_seed", 43),
        ("code_version", "abc124"),
        ("data_version", "def457"),
        ("model_version", "prod-v2"),
    ],
)
def test_cache_key_changes_when_any_input_changes(field: str, new_value) -> None:
    base = dict(
        strategy_id="torvik+f4_first_tv",
        year=2026,
        rng_seed=42,
        code_version="abc123",
        data_version="def456",
        model_version="prod-v1",
    )
    perturbed = {**base, field: new_value}
    assert compute_cache_key(**base) != compute_cache_key(**perturbed)


def test_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("src.data.strategy_cache.MANIFEST_PATH", tmp_path / "manifest.json")
    rng = np.random.default_rng(0)
    brackets = rng.integers(0, 68, size=(50, 63), dtype=np.uint16)
    cache_key = compute_cache_key("torvik", 2026, 42, "code", "data", "model")
    entry = save_brackets(
        cache_key,
        brackets,
        strategy_id="torvik",
        year=2026,
        rng_seed=42,
        code_version="code",
        data_version="data",
        model_version="model",
    )
    manifest = Manifest(schema_version=1, entries=[entry])
    manifest.save(tmp_path / "manifest.json")
    loaded = load_brackets(cache_key, manifest)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (50, 63)
    np.testing.assert_array_equal(loaded, brackets)


def test_save_rejects_wrong_dtype(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    bad = np.zeros((50, 63), dtype=np.int32)
    with pytest.raises(ValueError, match="uint16"):
        save_brackets(
            "key",
            bad,
            strategy_id="x",
            year=2026,
            rng_seed=42,
            code_version="c",
            data_version="d",
            model_version="m",
        )


def test_save_rejects_wrong_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    bad = np.zeros((50, 64), dtype=np.uint16)
    with pytest.raises(ValueError, match="63"):
        save_brackets(
            "key",
            bad,
            strategy_id="x",
            year=2026,
            rng_seed=42,
            code_version="c",
            data_version="d",
            model_version="m",
        )


def test_content_hash_mismatch_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("src.data.strategy_cache.MANIFEST_PATH", tmp_path / "manifest.json")
    brackets = np.zeros((50, 63), dtype=np.uint16)
    cache_key = compute_cache_key("x", 2026, 42, "c", "d", "m")
    entry = save_brackets(
        cache_key,
        brackets,
        strategy_id="x",
        year=2026,
        rng_seed=42,
        code_version="c",
        data_version="d",
        model_version="m",
    )
    manifest = Manifest(schema_version=1, entries=[entry])
    # Tamper: rewrite the npz with different bytes; manifest's content_hash is now stale.
    tampered = np.ones((50, 63), dtype=np.uint16)
    np.savez_compressed(tmp_path / entry.artifact_path, brackets=tampered)
    with pytest.raises(ValueError, match="Content hash mismatch"):
        load_brackets(cache_key, manifest)


def test_repo_manifest_is_consistent_if_present() -> None:
    """If the repo's strategy_cache/manifest.json exists, every entry must verify."""
    if not (CACHE_ROOT / "manifest.json").exists():
        pytest.skip("no cache populated yet")
    issues = verify_manifest()
    assert issues == [], f"manifest integrity issues: {issues}"
