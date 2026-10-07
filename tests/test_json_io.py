import gzip
import json

from src.data.json_io import load_json, resolve_json


def test_plain_json_preferred(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1}))
    (tmp_path / "x.json.gz").write_bytes(gzip.compress(json.dumps({"a": 2}).encode()))
    assert resolve_json(p) == p
    assert load_json(p) == {"a": 1}


def test_gz_fallback(tmp_path):
    p = tmp_path / "x.json"
    (tmp_path / "x.json.gz").write_bytes(gzip.compress(json.dumps({"a": 2}).encode()))
    assert resolve_json(p) == tmp_path / "x.json.gz"
    assert load_json(p) == {"a": 2}


def test_missing_returns_original_path(tmp_path):
    p = tmp_path / "x.json"
    assert resolve_json(p) == p
