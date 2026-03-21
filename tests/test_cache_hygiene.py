from pathlib import Path

from src.governance.cache_hygiene import (
    EPHEMERAL_CACHE_DIRS,
    CACHE_PROTOCOL_VERSION,
    find_ephemeral_cache_paths,
    fingerprint_context,
    purge_ephemeral_cache_paths,
)


def test_fingerprint_context_ignores_volatile_keys():
    base = {"season": 2026, "holdout_years": [2025], "timestamp": "a"}
    changed = {"season": 2026, "holdout_years": [2025], "timestamp": "b"}

    assert fingerprint_context(base) == fingerprint_context(changed)


def test_fingerprint_context_changes_with_split_inputs():
    a = {"season": 2026, "holdout_years": [2025]}
    b = {"season": 2026, "holdout_years": [2024]}

    assert fingerprint_context(a) != fingerprint_context(b)


def test_find_ephemeral_cache_paths_detects_common_runtime_dirs(tmp_path):
    pycache = tmp_path / "src" / "__pycache__"
    pycache.mkdir(parents=True)
    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    pyc = tmp_path / "tests" / "foo.pyc"
    pyc.parent.mkdir()
    pyc.write_bytes(b"x")

    found = find_ephemeral_cache_paths(tmp_path)

    assert pycache in found
    assert pytest_cache in found
    assert pyc in found
    assert "__pycache__" in EPHEMERAL_CACHE_DIRS


def test_purge_ephemeral_cache_paths_removes_detected_entries(tmp_path):
    pycache = tmp_path / "tests" / "__pycache__"
    pycache.mkdir(parents=True)
    compiled = tmp_path / "src" / "module.pyc"
    compiled.parent.mkdir(exist_ok=True)
    compiled.write_bytes(b"x")

    removed = purge_ephemeral_cache_paths(tmp_path)

    assert pycache in removed
    assert compiled in removed
    assert not pycache.exists()
    assert not compiled.exists()
