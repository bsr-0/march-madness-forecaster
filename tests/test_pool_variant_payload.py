import json
from pathlib import Path


def test_2026_pool_variants_are_settings_bound():
    payload = json.loads((Path(__file__).parents[1] / "docs/data/season_2026.json").read_text())
    variants = {v["pool_size"]: v for v in payload["pool_variants"]}
    assert set(variants) == {10, 30, 50, 100}
    for size, v in variants.items():
        assert v["schema"] == 1
        assert v["scoring_id"] == "espn_standard"
        assert v["p1_pool_size"] == size
        assert len(v["strategies"]) == 2
        assert v["artifact_sha256"]
        assert all(len(s["picks"]) == 6 for s in v["strategies"])
