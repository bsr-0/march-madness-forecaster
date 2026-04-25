"""Phase-5 lock test: cached vs live run_backtest produce identical results.

This is the integrity gate that justifies trusting --use-cache in any
budgeted run. The contract:

  Given the same year_rng (np.random.default_rng(42 + year)) and the
  same per-(mode, year) bracket_rng (make_mode_rng), running
  run_backtest with use_cache=True after a write_cache=True priming
  pass MUST produce bit-equal best_rank / mean_rank / p_first /
  best_score / mean_score per (year, mode) row.

Why this can't drift silently:
  - Phase 4.2 isolated bracket sampling onto bracket_rng; opponent
    generation + tournament simulation stay on year_rng. So whether
    a mode hits the cache or computes live, year_rng's state at the
    inner repeat-loop entry is identical.
  - Phase 4.3 keys cache entries on
    (mode, year, parent_seed, STRATEGY_CACHE_VERSION, data_version, model_version).
    Any change to sampler RNG consumption forces a key shift via the
    STRATEGY_CACHE_VERSION bump, so a cache hit is definitionally an
    artifact of the same sampler logic that a live miss would run.

The test isolates to a tmp cache root so it doesn't touch the repo's
committed artifacts/strategy_cache/. It needs a populated
team_id_lookup.json (copied from the repo) — without that, the
encode/decode path can't map team names to uint16 ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _have_repo_team_lookup() -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "artifacts" / "strategy_cache" / "team_id_lookup.json").exists()


@pytest.mark.parametrize("mode_name", ["torvik", "f4_first_tv", "e8_first_tv"])
def test_cached_run_matches_live_run_per_mode(mode_name: str, tmp_path, monkeypatch) -> None:
    """Cached vs live --use-cache equivalence for one production mode × 2026."""
    pytest.importorskip("numpy")
    try:
        from scripts.mc_pool_backtest import (
            PoolHyperparameters,
            _load_torvik_barthag,
            load_seeds_and_regions,
            load_tournament_results,
            run_backtest,
        )
    except ImportError:
        pytest.skip("project deps not available")

    if not _have_repo_team_lookup():
        pytest.skip("repo team_id_lookup not populated — run scripts/migrate_backtest_brackets_to_cache.py first")

    # 2026 prereqs: seeds, regions, torvik barthag, tournament results.
    try:
        seeds, _ = load_seeds_and_regions(2026)
        if _load_torvik_barthag(2026, seeds) is None:
            pytest.skip("torvik barthag for 2026 unavailable")
        if not load_tournament_results(2026):
            pytest.skip("2026 tournament results unavailable")
    except Exception as exc:
        pytest.skip(f"2026 data unavailable: {exc}")

    # Copy the repo's team_id_lookup into the tmp cache root so the
    # encode path has the full team-name -> uint16 mapping. Patching the
    # module-level paths in src.data.strategy_cache is sufficient — its
    # helper functions (load_team_lookup, save_brackets, Manifest.load)
    # all resolve those names at call time, so the imported references
    # in scripts.mc_pool_backtest see the patched values automatically.
    repo_root = Path(__file__).resolve().parents[1]
    repo_lookup = repo_root / "artifacts" / "strategy_cache" / "team_id_lookup.json"
    monkeypatch.setattr("src.data.strategy_cache.CACHE_ROOT", tmp_path)
    monkeypatch.setattr("src.data.strategy_cache.MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr("src.data.strategy_cache.TEAM_LOOKUP_PATH", tmp_path / "team_id_lookup.json")
    (tmp_path / "team_id_lookup.json").write_text(repo_lookup.read_text())

    def fitter(train_years):
        return PoolHyperparameters(blend_alpha=0.5, enabled_modes=(mode_name,))

    common_kwargs = dict(
        years=[2026],
        n_opponents=10,
        n_repeats=2,
        n_model=50,
        opponent_source="pool",
        hparam_fitter=fitter,
        team_identity=True,
    )

    # Pass 1: live compute, populate the tmp cache.
    live_results = run_backtest(**common_kwargs, use_cache=False, write_cache=True)
    assert live_results, f"live run produced no results for {mode_name} 2026"

    # Pass 2: read from the just-populated cache — must match.
    cached_results = run_backtest(**common_kwargs, use_cache=True, write_cache=False)
    assert cached_results, f"cached run produced no results for {mode_name} 2026"

    # Filter to the single-year, single-mode rows we care about. Other
    # legacy modes inside the registry can sneak in via the fitter spec
    # if a mode_name happens to be a legacy alias; we filter precisely.
    def _row_for(rows, year, mode):
        match = [r for r in rows if r["year"] == year and r["mode"] == mode]
        assert len(match) == 1, f"expected 1 row for ({year}, {mode}), got {len(match)}: {match}"
        return match[0]

    live = _row_for(live_results, 2026, mode_name)
    cached = _row_for(cached_results, 2026, mode_name)

    # Bit-exact on the metrics that the budgeted pipeline keys
    # promote / kill decisions on.
    for field in ("best_rank", "mean_rank", "p_first", "p_top5", "p_top25", "best_score", "mean_score", "n_trials"):
        assert live[field] == pytest.approx(cached[field], rel=0, abs=0), (
            f"divergence on {field} for {mode_name} 2026: live={live[field]} cached={cached[field]}"
        )


def test_repo_cache_team_lookup_is_well_formed_if_present() -> None:
    """If the repo's team_id_lookup.json exists, every value is a unique uint16."""
    repo_root = Path(__file__).resolve().parents[1]
    lookup_path = repo_root / "artifacts" / "strategy_cache" / "team_id_lookup.json"
    if not lookup_path.exists():
        pytest.skip("no team_id_lookup populated yet")
    lookup = json.loads(lookup_path.read_text())
    assert isinstance(lookup, dict)
    ids = list(lookup.values())
    assert len(set(ids)) == len(ids), "team_id_lookup contains duplicate IDs"
    assert all(isinstance(v, int) and 0 <= v < 2**16 for v in ids), "team_id_lookup contains non-uint16 IDs"
