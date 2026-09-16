"""Unit tests for A5 Massey composite source.

Covers the loader, the Massey-specific ID bridge (audit 2026-09, Step 18
remediation: expanded from 6 to 45 edge-case aliases after coverage
measurement found up to 13 of 68 tournament teams per season silently
falling back to the seed-based barthag), the seed-based fallback for teams
not in the Massey composite, coverage gates, and the pipeline/backtest
registry wiring.

A6 ``massey_best`` is deferred — needs a per-system Brier-selection
harness that is its own phase of work. Not tested here.
"""

import json
from pathlib import Path

import pytest

from scripts._common import load_seeds_and_regions
from src.prediction.massey_probabilities import (
    _MASSEY_EDGE_CASES,
    _bridge_massey_id,
    _seed_fallback_barthag,
    load_massey_avg_barthag,
    massey_coverage,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# 2012 and 2020 have no tournament (no field / cancelled); this is the
# audited backtest year set (Steps 1-18).
BACKTEST_YEARS = [2011, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


def _load_2025_seeds():
    hist_dir = DATA_ROOT / "raw" / "historical"
    ctx_path = hist_dir / "tournament_context_2025.json"
    raw = None
    if ctx_path.exists():
        with open(ctx_path) as f:
            raw = json.load(f).get("seeds")
    if raw is None:
        raw = json.load(open(hist_dir / "tournament_seeds_2025.json"))
    teams = raw["teams"] if isinstance(raw, dict) and "teams" in raw else raw
    return {t["team_id"]: t["seed"] for t in teams}


# -------------------------------------------------------------------- ID bridge


def test_bridge_exact_match_passthrough():
    assert _bridge_massey_id("duke", frozenset({"duke", "kansas"})) == "duke"


def test_bridge_applies_edge_case_alias():
    canonical = frozenset(_MASSEY_EDGE_CASES.values())
    for massey_id, expected in _MASSEY_EDGE_CASES.items():
        assert _bridge_massey_id(massey_id, canonical) == expected


def test_bridge_returns_none_for_unknown_id():
    assert _bridge_massey_id("nonexistent_team", frozenset({"duke"})) is None


def test_seed_fallback_monotone():
    assert _seed_fallback_barthag(1) > _seed_fallback_barthag(8) > _seed_fallback_barthag(16)
    assert _seed_fallback_barthag(None) == 0.5


# -------------------------------------------------------------------- loader


def test_load_massey_avg_2025_covers_all_tournament_teams():
    """Direct + alias bridge should resolve every 2025 tournament team."""
    seeds = _load_2025_seeds()
    barthag = load_massey_avg_barthag(2025, seeds, DATA_ROOT)
    assert barthag is not None
    assert set(barthag.keys()) == set(seeds.keys())


def test_load_massey_avg_2025_coverage_gate():
    """All 68 teams should bridge — anything less is a regression."""
    seeds = _load_2025_seeds()
    cov = massey_coverage(2025, seeds.keys(), DATA_ROOT)
    assert cov["file_exists"] == 1
    assert cov["covered"] == 68, f"Expected 68/68 2025 coverage; got {cov['covered']}/{cov['total']}"


def test_load_massey_avg_values_in_unit_range():
    seeds = _load_2025_seeds()
    barthag = load_massey_avg_barthag(2025, seeds, DATA_ROOT)
    for tid, b in barthag.items():
        assert 0.10 <= b <= 0.99, f"Team {tid} barthag {b} outside [0.10, 0.99] catalog clip"


def test_load_massey_avg_1_seeds_rank_high():
    """Every 2025 1-seed should end up with barthag > 0.8 — sanity on the mapping."""
    seeds = _load_2025_seeds()
    barthag = load_massey_avg_barthag(2025, seeds, DATA_ROOT)
    one_seeds = [tid for tid, s in seeds.items() if s == 1]
    assert one_seeds, "Expected 2025 to have 1-seeds"
    for tid in one_seeds:
        assert barthag[tid] > 0.8, f"1-seed {tid} has barthag {barthag[tid]:.3f}; expected > 0.8"


def test_load_massey_avg_covers_all_teams_every_backtest_year():
    """audit 2026-09, Step 18 remediation.

    Before the alias table was expanded, coverage ranged 55-68 of 68 teams
    per season (up to 13 teams/season on the crude seed fallback instead of
    a real Massey rating). Every backtest year must now resolve the full
    field via exact match or the verified alias table.
    """
    for year in BACKTEST_YEARS:
        seeds, _regions = load_seeds_and_regions(year)
        cov = massey_coverage(year, seeds.keys(), DATA_ROOT)
        assert cov["covered"] == cov["total"] == 68, f"{year}: {cov['covered']}/{cov['total']} covered"


def test_massey_edge_cases_do_not_collide():
    """Two different Massey IDs must never alias to the same canonical team."""
    from collections import Counter

    counts = Counter(_MASSEY_EDGE_CASES.values())
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"multiple Massey IDs alias to the same canonical team: {dupes}"


def test_load_massey_avg_ignores_untracked_data_raw_override(tmp_path, monkeypatch):
    """audit 2026-09, Step 18: reproducibility regression.

    ``load_massey_avg_barthag`` used to check ``data/raw/`` before
    ``data/raw/historical/``. An untracked file at the top-level path
    silently overrode the committed, reproducible data for every caller in
    this working directory, and a fresh checkout (no untracked files) got
    different candidate artifacts than the one everyone had been
    developing against. The loader must read only the tracked path.
    """
    fake_root = tmp_path
    hist_dir = fake_root / "raw" / "historical"
    hist_dir.mkdir(parents=True)
    tracked = {"systems": {"massey_composite": [{"team_id": "duke", "team_name": "Duke", "normalized": 0.9}]}}
    json.dump(tracked, open(hist_dir / "external_ratings_2025.json", "w"))

    # An untracked override at data/raw/ (not data/raw/historical/) with
    # different content must NOT be read.
    untracked = {"systems": {"massey_composite": [{"team_id": "duke", "team_name": "Duke", "normalized": 0.1}]}}
    json.dump(untracked, open(fake_root / "raw" / "external_ratings_2025.json", "w"))

    result = load_massey_avg_barthag(2025, {"duke": 1}, fake_root)
    assert result["duke"] == pytest.approx(0.9), "loader read the untracked data/raw/ override instead of data/raw/historical/"


def test_load_massey_avg_returns_none_for_missing_year():
    """Years without composite file must return None, not crash."""
    seeds = {"duke": 1, "kansas": 2}
    # Any year without the file should return None
    assert load_massey_avg_barthag(1999, seeds, DATA_ROOT) is None
    assert load_massey_avg_barthag(2050, seeds, DATA_ROOT) is None


def test_load_massey_avg_clip_bounds():
    """Values are clipped to [0.10, 0.99] per catalog spec."""
    seeds = _load_2025_seeds()
    barthag = load_massey_avg_barthag(2025, seeds, DATA_ROOT)
    # Top cluster should saturate at 0.99 (the composite has several teams >0.99).
    assert max(barthag.values()) == pytest.approx(0.99, abs=1e-9)


# -------------------------------------------------------------------- pipeline wiring


def test_massey_avg_registered_in_pipeline():
    from src.prediction.strategy_pipeline import IMPLEMENTED_SOURCES

    assert "massey_avg" in IMPLEMENTED_SOURCES


def test_massey_avg_registered_in_probability_bases():
    from scripts.mc_pool_backtest import PROBABILITY_BASES

    assert "massey_avg" in PROBABILITY_BASES


def test_permutation_generator_enumerates_massey_avg():
    from src.prediction.strategy_pipeline import generate_all_permutations

    strategies = generate_all_permutations(implemented_only=True)
    massey = [s for s in strategies if "massey_avg" in s]
    # 10 constructions × 1 bare source + bare name + adjustment chains + blends.
    assert len(massey) >= 20
    # Phase 3 spot-check: the canonical massey_avg + F4-first variant exists.
    assert "massey_avg_f4_first" in strategies
    assert "massey_avg" in strategies  # bare source + forward (no suffix)
