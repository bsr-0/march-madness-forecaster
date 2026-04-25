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
    load_outcomes,
    load_probabilities,
    save_brackets,
    save_outcomes,
    save_probabilities,
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


def test_outcomes_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("src.data.strategy_cache.MANIFEST_PATH", tmp_path / "manifest.json")
    rng = np.random.default_rng(1)
    winners = rng.integers(0, 68, size=63, dtype=np.uint16)
    entry = save_outcomes(2026, winners, code_version="c", data_version="d")
    manifest = Manifest(schema_version=1, entries=[entry])
    loaded = load_outcomes(2026, manifest)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (63,)
    np.testing.assert_array_equal(loaded, winners)


def test_outcomes_reject_bad_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    bad = np.zeros(64, dtype=np.uint16)
    with pytest.raises(ValueError, match="63"):
        save_outcomes(2026, bad, code_version="c", data_version="d")


def test_brackets_match_outcomes_per_round(tmp_path, monkeypatch) -> None:
    """Demonstrates the validation primitive the cache enables."""
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    rng = np.random.default_rng(2)
    outcomes = rng.integers(0, 68, size=63, dtype=np.uint16)
    brackets = np.tile(outcomes, (50, 1)).astype(np.uint16)
    brackets[0, 0] = (outcomes[0] + 1) % 68  # one wrong pick on bracket 0
    correct = brackets == outcomes[None, :]
    assert correct.shape == (50, 63)
    assert correct[0, 0] == False  # noqa: E712
    assert correct[0, 1:].all()
    assert correct[1:, :].all()


def test_probabilities_save_load_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("src.data.strategy_cache.MANIFEST_PATH", tmp_path / "manifest.json")
    rng = np.random.default_rng(3)
    probs = rng.random((68, 6), dtype=np.float32)
    team_ids = np.arange(68, dtype=np.uint16)
    cache_key = compute_cache_key("torvik", 2026, 42, "c", "d", "m")
    entry = save_probabilities(
        cache_key,
        probs,
        team_ids,
        strategy_id="torvik",
        year=2026,
        rng_seed=42,
        code_version="c",
        data_version="d",
        model_version="m",
    )
    manifest = Manifest(schema_version=1, entries=[entry])
    loaded_probs, loaded_ids = load_probabilities(cache_key, manifest)
    assert loaded_probs.dtype == np.float32
    assert loaded_probs.shape == (68, 6)
    np.testing.assert_array_equal(loaded_probs, probs)
    np.testing.assert_array_equal(loaded_ids, team_ids)


def test_probabilities_reject_bad_dtype(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    bad = np.zeros((68, 6), dtype=np.float64)
    team_ids = np.arange(68, dtype=np.uint16)
    with pytest.raises(ValueError, match="float32"):
        save_probabilities(
            "k",
            bad,
            team_ids,
            strategy_id="x",
            year=2026,
            rng_seed=42,
            code_version="c",
            data_version="d",
            model_version="m",
        )


def test_probabilities_reject_bad_n_rounds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    bad = np.zeros((68, 5), dtype=np.float32)
    team_ids = np.arange(68, dtype=np.uint16)
    with pytest.raises(ValueError, match="\\(n_teams, 6\\)"):
        save_probabilities(
            "k",
            bad,
            team_ids,
            strategy_id="x",
            year=2026,
            rng_seed=42,
            code_version="c",
            data_version="d",
            model_version="m",
        )


def test_probabilities_reject_team_id_length_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    probs = np.zeros((68, 6), dtype=np.float32)
    bad_ids = np.arange(67, dtype=np.uint16)
    with pytest.raises(ValueError, match="team_ids must be"):
        save_probabilities(
            "k",
            probs,
            bad_ids,
            strategy_id="x",
            year=2026,
            rng_seed=42,
            code_version="c",
            data_version="d",
            model_version="m",
        )


def test_torvik_round_probabilities_are_bit_exact() -> None:
    """build_torvik_round_probabilities seeds rng=42 internally; two calls
    on the same inputs must produce identical floats. This is the contract
    Phase 2 relies on for bit-exact reproducibility of cached probabilities."""
    pytest.importorskip("numpy")
    try:
        from scripts.mc_pool_backtest import (
            _load_torvik_barthag,
            build_torvik_round_probabilities,
            load_seeds_and_regions,
        )
    except ImportError:
        pytest.skip("project deps not available")
    try:
        seeds, regions = load_seeds_and_regions(2026)
        barthag = _load_torvik_barthag(2026, seeds)
    except Exception:
        pytest.skip("2026 data unavailable")
    if barthag is None:
        pytest.skip("torvik barthag for 2026 unavailable")
    a = build_torvik_round_probabilities(seeds, regions, barthag)
    b = build_torvik_round_probabilities(seeds, regions, barthag)
    assert set(a.keys()) == set(b.keys())
    for tid in a:
        for rnd in a[tid]:
            assert a[tid][rnd] == b[tid][rnd], f"non-deterministic at {tid}/{rnd}: {a[tid][rnd]} != {b[tid][rnd]}"


def test_repo_manifest_is_consistent_if_present() -> None:
    """If the repo's strategy_cache/manifest.json exists, every entry must verify."""
    if not (CACHE_ROOT / "manifest.json").exists():
        pytest.skip("no cache populated yet")
    issues = verify_manifest()
    assert issues == [], f"manifest integrity issues: {issues}"
