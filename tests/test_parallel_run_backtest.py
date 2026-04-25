"""Phase-1 lock test: workers=1 and workers=2 produce identical results.

The integrity gate that justifies trusting --workers > 1 in any budgeted
run. The contract:

  Given identical inputs, run_backtest(workers=1) and
  run_backtest(workers=N>1) MUST produce bit-equal per-(year, mode)
  rows on best_rank / mean_rank / p_first / p_top5 / p_top25 /
  best_score / mean_score / n_trials.

Why this can't drift silently:
  - Phase 4.2 isolated bracket sampling onto bracket_rng = make_mode_rng(
    mode, year), independent of year_rng.
  - year_rng = np.random.default_rng(42 + year) is constructed inside
    _run_one_year, so each worker gets a fresh deterministic stream
    derived purely from its year.
  - Workers don't share state; opponent generation and tournament
    simulation per repeat consume year_rng locally. No cross-year or
    cross-mode coupling remains after the Phase 1.A refactor.

The test exercises one mode × two years, n_repeats=2, to keep wall
time bounded while still going through every code path that consumes
rng (sampler + opponent gen + sim outcomes). Skips cleanly if
numpy/scipy/data prereqs aren't present.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_parallel_matches_sequential_on_torvik_2025_2026() -> None:
    """workers=1 and workers=2 produce bit-equal results for torvik × 2025+2026."""
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    try:
        from scripts.mc_pool_backtest import (
            StrategiesFitter,
            _load_torvik_barthag,
            load_seeds_and_regions,
            load_tournament_results,
            run_backtest,
        )
    except ImportError:
        pytest.skip("project deps not available")

    # Both years must have torvik barthag + tournament results to exercise
    # the full pipeline. Skip otherwise — this test is a parallelism gate,
    # not a data-availability gate.
    test_years = [2025, 2026]
    for yr in test_years:
        try:
            seeds, _ = load_seeds_and_regions(yr)
            if _load_torvik_barthag(yr, seeds) is None:
                pytest.skip(f"torvik barthag for {yr} unavailable")
            if not load_tournament_results(yr):
                pytest.skip(f"{yr} tournament results unavailable")
        except Exception as exc:
            pytest.skip(f"{yr} prereqs unavailable: {exc}")

    common_kwargs = dict(
        years=test_years,
        n_opponents=10,
        n_repeats=2,
        n_model=50,
        opponent_source="pool",
        hparam_fitter=StrategiesFitter(("torvik",)),
        team_identity=True,
    )

    serial_results = run_backtest(**common_kwargs, workers=1)
    parallel_results = run_backtest(**common_kwargs, workers=2)

    assert serial_results, "serial run produced no results"
    assert parallel_results, "parallel run produced no results"

    # Both lists may be in different orders (parallel returns in completion
    # order). Sort by (year, mode) before comparing.
    def _sort_key(r):
        return (r["year"], r["mode"])

    serial_sorted = sorted(serial_results, key=_sort_key)
    parallel_sorted = sorted(parallel_results, key=_sort_key)

    assert len(serial_sorted) == len(parallel_sorted), (
        f"row count differs: serial={len(serial_sorted)} parallel={len(parallel_sorted)}"
    )

    # Bit-exact on every metric the budgeted pipeline keys promote / kill
    # decisions on. Any drift here breaks the parallelism contract.
    metric_fields = (
        "best_rank",
        "mean_rank",
        "p_first",
        "p_top5",
        "p_top25",
        "best_score",
        "mean_score",
        "n_trials",
    )
    for serial, parallel in zip(serial_sorted, parallel_sorted):
        assert serial["year"] == parallel["year"], f"year mismatch: {serial['year']} vs {parallel['year']}"
        assert serial["mode"] == parallel["mode"], (
            f"mode mismatch at year {serial['year']}: {serial['mode']} vs {parallel['mode']}"
        )
        for field in metric_fields:
            assert serial[field] == pytest.approx(parallel[field], rel=0, abs=0), (
                f"divergence on {field} for ({serial['year']}, {serial['mode']}): "
                f"serial={serial[field]} parallel={parallel[field]}"
            )


def test_parallel_matches_sequential_multimode_torvik_2025_2026() -> None:
    """Phase-2 lock: workers=1 vs workers=2 stay bit-equal under multi-mode runs.

    Phase 2 hoists opponent + simulated-outcome generation out of the
    per-mode loop so every mode is scored against the same field per
    repeat (paired comparison). The single-mode test above covers the
    pre-Phase-2 path; this one covers the multi-mode path where the
    hoist actually does something. Bit-equality between workers=1 and
    workers=N must still hold — the year_rng consumption order is
    deterministic per year, independent of completion order.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    try:
        from scripts.mc_pool_backtest import (
            StrategiesFitter,
            _load_torvik_barthag,
            load_seeds_and_regions,
            load_tournament_results,
            run_backtest,
        )
    except ImportError:
        pytest.skip("project deps not available")

    test_years = [2025, 2026]
    for yr in test_years:
        try:
            seeds, _ = load_seeds_and_regions(yr)
            if _load_torvik_barthag(yr, seeds) is None:
                pytest.skip(f"torvik barthag for {yr} unavailable")
            if not load_tournament_results(yr):
                pytest.skip(f"{yr} tournament results unavailable")
        except Exception as exc:
            pytest.skip(f"{yr} prereqs unavailable: {exc}")

    # Two production modes: forces the Phase-2 shared-opponents path
    # (mode_payloads with len>1) through both the workers=1 and
    # workers=2 code paths.
    common_kwargs = dict(
        years=test_years,
        n_opponents=10,
        n_repeats=2,
        n_model=50,
        opponent_source="pool",
        hparam_fitter=StrategiesFitter(("torvik", "f4_first_tv")),
        team_identity=True,
    )

    serial_results = run_backtest(**common_kwargs, workers=1)
    parallel_results = run_backtest(**common_kwargs, workers=2)

    assert serial_results, "serial run produced no results"
    assert parallel_results, "parallel run produced no results"

    def _sort_key(r):
        return (r["year"], r["mode"])

    serial_sorted = sorted(serial_results, key=_sort_key)
    parallel_sorted = sorted(parallel_results, key=_sort_key)

    assert len(serial_sorted) == len(parallel_sorted), (
        f"row count differs: serial={len(serial_sorted)} parallel={len(parallel_sorted)}"
    )
    # Both modes must show up in each year's rows — guards against a
    # silent regression where the multi-mode path collapses to one row.
    serial_modes_per_year = {y: set() for y in test_years}
    for row in serial_sorted:
        serial_modes_per_year[row["year"]].add(row["mode"])
    for yr, modes in serial_modes_per_year.items():
        assert {"torvik", "f4_first_tv"}.issubset(modes), f"year {yr} missing one of torvik/f4_first_tv: got {modes}"

    metric_fields = (
        "best_rank",
        "mean_rank",
        "p_first",
        "p_top5",
        "p_top25",
        "best_score",
        "mean_score",
        "n_trials",
    )
    for serial, parallel in zip(serial_sorted, parallel_sorted):
        assert serial["year"] == parallel["year"]
        assert serial["mode"] == parallel["mode"]
        for field in metric_fields:
            assert serial[field] == pytest.approx(parallel[field], rel=0, abs=0), (
                f"divergence on {field} for ({serial['year']}, {serial['mode']}): "
                f"serial={serial[field]} parallel={parallel[field]}"
            )


def test_per_mode_opponent_strategy_workers_equivalence() -> None:
    """Q3 option A lock: per_mode opponent_strategy stays bit-equal across worker counts.

    opponent_strategy="per_mode" spawns a deterministic
    SeedSequence(42 + year) sub-stream per mode. Spawn order is the
    order modes appear in mode_payloads — derived from
    enabled_modes — so workers=1 and workers=N must produce bit-equal
    per-(year, mode) rows just like the shared path. This test exists
    so a future refactor of the per_mode branch can't silently break
    parallel determinism.

    Sanity check on top of equivalence: per_mode results must NOT be
    bit-equal to shared on a multi-mode run (the two strategies
    consume different RNG bytes), guarding against a no-op branch.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    try:
        from scripts.mc_pool_backtest import (
            StrategiesFitter,
            _load_torvik_barthag,
            load_seeds_and_regions,
            load_tournament_results,
            run_backtest,
        )
    except ImportError:
        pytest.skip("project deps not available")

    test_years = [2025, 2026]
    for yr in test_years:
        try:
            seeds, _ = load_seeds_and_regions(yr)
            if _load_torvik_barthag(yr, seeds) is None:
                pytest.skip(f"torvik barthag for {yr} unavailable")
            if not load_tournament_results(yr):
                pytest.skip(f"{yr} tournament results unavailable")
        except Exception as exc:
            pytest.skip(f"{yr} prereqs unavailable: {exc}")

    common_kwargs = dict(
        years=test_years,
        n_opponents=10,
        n_repeats=2,
        n_model=50,
        opponent_source="pool",
        hparam_fitter=StrategiesFitter(("torvik", "f4_first_tv")),
        team_identity=True,
        opponent_strategy="per_mode",
    )

    serial_results = run_backtest(**common_kwargs, workers=1)
    parallel_results = run_backtest(**common_kwargs, workers=2)

    assert serial_results, "per_mode serial run produced no results"
    assert parallel_results, "per_mode parallel run produced no results"

    def _sort_key(r):
        return (r["year"], r["mode"])

    serial_sorted = sorted(serial_results, key=_sort_key)
    parallel_sorted = sorted(parallel_results, key=_sort_key)

    metric_fields = ("best_rank", "mean_rank", "p_first", "p_top5", "p_top25", "best_score", "mean_score", "n_trials")
    for serial, parallel in zip(serial_sorted, parallel_sorted):
        assert serial["year"] == parallel["year"]
        assert serial["mode"] == parallel["mode"]
        for field in metric_fields:
            assert serial[field] == pytest.approx(parallel[field], rel=0, abs=0), (
                f"per_mode workers divergence on {field} for ({serial['year']}, {serial['mode']}): "
                f"serial={serial[field]} parallel={parallel[field]}"
            )

    # Sanity: per_mode and shared must produce DIFFERENT results on a
    # multi-mode run, otherwise the branch is a no-op.
    shared_results = run_backtest(
        **{**common_kwargs, "opponent_strategy": "shared"},
        workers=1,
    )
    shared_by_key = {(r["year"], r["mode"]): r for r in shared_results}
    differs = False
    for serial in serial_sorted:
        shared = shared_by_key[(serial["year"], serial["mode"])]
        if shared["p_first"] != serial["p_first"] or shared["best_rank"] != serial["best_rank"]:
            differs = True
            break
    assert differs, "per_mode produced identical results to shared — branch may be a no-op"


def test_strategies_fitter_is_picklable() -> None:
    """StrategiesFitter is the picklable closure replacement — protocol contract."""
    pytest.importorskip("numpy")
    try:
        from scripts.mc_pool_backtest import StrategiesFitter, PoolHyperparameters
    except ImportError:
        pytest.skip("project deps not available")

    import pickle

    fitter = StrategiesFitter(("torvik", "f4_first_tv"), blend_alpha=0.5)
    rehydrated = pickle.loads(pickle.dumps(fitter))

    assert rehydrated.strategies == fitter.strategies
    assert rehydrated.blend_alpha == fitter.blend_alpha

    # Identical PoolHyperparameters output regardless of train_years input
    # (the trait that lets parallel workers reconstruct hparams without
    # cross-process coordination).
    a = fitter([2011, 2012])
    b = rehydrated([2011, 2012])
    c = rehydrated([2024])
    assert isinstance(a, PoolHyperparameters)
    assert a.blend_alpha == b.blend_alpha == c.blend_alpha == 0.5
    assert a.enabled_modes == b.enabled_modes == c.enabled_modes == ("torvik", "f4_first_tv")


def test_run_one_year_is_picklable() -> None:
    """_run_one_year must be reachable via dotted path so workers can resolve it."""
    pytest.importorskip("numpy")
    try:
        from scripts.mc_pool_backtest import _run_one_year
    except ImportError:
        pytest.skip("project deps not available")
    import pickle

    # Round-tripping the function reference proves it's at module level
    # (not a closure). The actual function body doesn't run here.
    fn = pickle.loads(pickle.dumps(_run_one_year))
    assert fn is _run_one_year or fn.__qualname__ == "_run_one_year"


def test_workers_arg_accepted_with_default_one() -> None:
    """run_backtest accepts workers kwarg, default 1 = sequential path."""
    pytest.importorskip("numpy")
    try:
        from scripts.mc_pool_backtest import run_backtest
    except ImportError:
        pytest.skip("project deps not available")
    import inspect

    sig = inspect.signature(run_backtest)
    assert "workers" in sig.parameters, "run_backtest is missing the workers kwarg"
    assert sig.parameters["workers"].default == 1, "workers default must be 1 for backward compat"


def test_repo_artifacts_dir_exists() -> None:
    """Sanity: the artifacts/ directory is on disk so write paths don't no-op silently."""
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "artifacts").exists(), "artifacts/ missing — backtest writes would fail"
