"""Drift-guard for the T3 Oracle output.

Locks the per-(year, strategy) ranker-gap, F4-hits, finals-hits, and
champion-hit values from the committed bracket artifacts plus the
strategy-level aggregates. Catches any silent regression in:

- ``oracle_sweep_t3_years()`` math (mis-indexed bracket bits, scoring
  vector drift, etc.)
- The committed ``artifacts/backtest_brackets/backtest_brackets_*.json``
  files (a re-run of T3 should not silently change these numbers)
- Ground-truth ingestion in ``tournament_results_*.json`` (F4 / champ
  derivations)
- The ESPN scoring vector (10/20/40/80/160/320 round-by-round)

Anchor source: ``memory/tournament_oracle.md`` § "System output (ranker
gap per year)" + live oracle output (cross-verified 2026-04-25). The
ledger explicitly flags the *regime → gap* correlation as "open, not
yet locked" (sample size = 4), so the regime check below is a loose
direction-only assertion, not a magnitude lock.

Out of scope:
- Drift from new MC seeds in fresh runs (we read committed artifacts).
- Strategies without committed artifacts (e.g., seed_f4_first) — those
  need to ship their own anchors first.
"""

from pathlib import Path

import pytest

from scripts.run_experiment import oracle_sweep_t3_years

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRACKETS_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"
DATA_ROOT = PROJECT_ROOT / "data"

LEDGER_YEARS = (2023, 2024, 2025, 2026)
LEDGER_STRATEGIES = ("torvik", "f4_first_tv", "e8_first_tv")

# Per-(year, strategy) ranker gap anchors from memory/tournament_oracle.md
# § "System output". These are the canonical values the ledger publishes.
# A future drift in oracle math, bracket artifacts, or ground-truth data
# will tip at least one of these — fail loudly so we notice.
EXPECTED_GAP_BY_YEAR_STRATEGY = {
    2023: {"torvik": 820.0, "f4_first_tv": 200.0, "e8_first_tv": 450.0},
    2024: {"torvik": 0.0, "f4_first_tv": 120.0, "e8_first_tv": 370.0},
    2025: {"torvik": 150.0, "f4_first_tv": 280.0, "e8_first_tv": 440.0},
    2026: {"torvik": 610.0, "f4_first_tv": 830.0, "e8_first_tv": 360.0},
}

# Per-(year, strategy) F4-hits from the best bracket in the portfolio.
EXPECTED_F4_HITS_BY_YEAR_STRATEGY = {
    2023: {"torvik": 1, "f4_first_tv": 1, "e8_first_tv": 1},
    2024: {"torvik": 2, "f4_first_tv": 2, "e8_first_tv": 2},
    2025: {"torvik": 2, "f4_first_tv": 2, "e8_first_tv": 3},
    2026: {"torvik": 3, "f4_first_tv": 4, "e8_first_tv": 2},
}

# Per-(year, strategy) champion-hit (boolean) from the best bracket.
EXPECTED_CHAMP_HIT_BY_YEAR_STRATEGY = {
    2023: {"torvik": True, "f4_first_tv": False, "e8_first_tv": True},
    2024: {"torvik": True, "f4_first_tv": True, "e8_first_tv": True},
    2025: {"torvik": True, "f4_first_tv": True, "e8_first_tv": True},
    2026: {"torvik": True, "f4_first_tv": True, "e8_first_tv": True},
}

# Strategy-level aggregates over the 4-year window.
# Computed as straight means over the per-year values above.
EXPECTED_AGGREGATE = {
    "torvik": {
        "mean_ranker_gap_espn_pts": 395.0,
        "mean_f4_hits": 2.0,
        "champ_hit_rate": 1.0,
    },
    "f4_first_tv": {
        "mean_ranker_gap_espn_pts": 357.5,
        "mean_f4_hits": 2.25,
        "champ_hit_rate": 0.75,
    },
    "e8_first_tv": {
        "mean_ranker_gap_espn_pts": 405.0,
        "mean_f4_hits": 2.0,
        "champ_hit_rate": 1.0,
    },
}


@pytest.fixture(scope="module")
def oracle_sweep_result():
    """Run the oracle sweep once per module — it's not free."""
    return oracle_sweep_t3_years(
        list(LEDGER_YEARS),
        list(LEDGER_STRATEGIES),
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )


@pytest.mark.parametrize(
    "year, strategy, expected_gap",
    [(year, strat, EXPECTED_GAP_BY_YEAR_STRATEGY[year][strat]) for year in LEDGER_YEARS for strat in LEDGER_STRATEGIES],
)
def test_per_year_strategy_ranker_gap_matches_ledger(oracle_sweep_result, year, strategy, expected_gap):
    """Per-(year, strategy) ranker_gap_espn_pts matches the canonical ledger.

    Drift here means either the oracle scoring math changed, the committed
    bracket artifacts changed, or the ground-truth data changed. Any of
    those should be a deliberate, noted update — not a silent regression.
    """
    entry = oracle_sweep_result["per_year"][year][strategy]
    assert "skipped" not in entry, f"{year}/{strategy} unexpectedly skipped: {entry}"
    assert entry["ranker_gap_espn_pts"] == pytest.approx(expected_gap, abs=1.0), (
        f"{year}/{strategy} gap drifted: ledger={expected_gap}, actual={entry['ranker_gap_espn_pts']}"
    )


@pytest.mark.parametrize(
    "year, strategy, expected_hits",
    [
        (year, strat, EXPECTED_F4_HITS_BY_YEAR_STRATEGY[year][strat])
        for year in LEDGER_YEARS
        for strat in LEDGER_STRATEGIES
    ],
)
def test_per_year_strategy_best_f4_hits_match_ledger(oracle_sweep_result, year, strategy, expected_hits):
    """The best bracket's F4-hit count is locked per (year, strategy)."""
    entry = oracle_sweep_result["per_year"][year][strategy]
    assert "skipped" not in entry
    assert entry["best_score_f4_hits"] == expected_hits, (
        f"{year}/{strategy} F4 hits drifted: ledger={expected_hits}, actual={entry['best_score_f4_hits']}"
    )


@pytest.mark.parametrize(
    "year, strategy, expected_hit",
    [
        (year, strat, EXPECTED_CHAMP_HIT_BY_YEAR_STRATEGY[year][strat])
        for year in LEDGER_YEARS
        for strat in LEDGER_STRATEGIES
    ],
)
def test_per_year_strategy_best_champ_hit_matches_ledger(oracle_sweep_result, year, strategy, expected_hit):
    """Whether the portfolio's best bracket picked the actual champion."""
    entry = oracle_sweep_result["per_year"][year][strategy]
    assert "skipped" not in entry
    assert entry["best_score_champ_hit"] is expected_hit, (
        f"{year}/{strategy} champ_hit drifted: ledger={expected_hit}, actual={entry['best_score_champ_hit']}"
    )


@pytest.mark.parametrize("strategy", LEDGER_STRATEGIES)
def test_aggregate_mean_ranker_gap_matches_expected(oracle_sweep_result, strategy):
    """Aggregate mean_ranker_gap_espn_pts equals the straight 4-year mean."""
    agg = oracle_sweep_result["aggregate"][strategy]
    expected = EXPECTED_AGGREGATE[strategy]["mean_ranker_gap_espn_pts"]
    assert agg["mean_ranker_gap_espn_pts"] == pytest.approx(expected, abs=1.0), (
        f"{strategy} aggregate gap drifted: expected={expected}, actual={agg['mean_ranker_gap_espn_pts']}"
    )


@pytest.mark.parametrize("strategy", LEDGER_STRATEGIES)
def test_aggregate_mean_f4_hits_matches_expected(oracle_sweep_result, strategy):
    agg = oracle_sweep_result["aggregate"][strategy]
    expected = EXPECTED_AGGREGATE[strategy]["mean_f4_hits"]
    assert agg["mean_f4_hits"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("strategy", LEDGER_STRATEGIES)
def test_aggregate_champ_hit_rate_matches_expected(oracle_sweep_result, strategy):
    agg = oracle_sweep_result["aggregate"][strategy]
    expected = EXPECTED_AGGREGATE[strategy]["champ_hit_rate"]
    assert agg["champ_hit_rate"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("strategy", LEDGER_STRATEGIES)
def test_aggregate_years_scored_equals_four(oracle_sweep_result, strategy):
    """All 4 ledger years have committed artifacts for all 3 strategies."""
    agg = oracle_sweep_result["aggregate"][strategy]
    assert agg["years_scored"] == 4, f"{strategy}: expected 4 years scored, got {agg['years_scored']}"


def test_chaos_year_2023_has_at_least_one_large_gap_mode(oracle_sweep_result):
    """Direction-only check on the catalog's regime hypothesis.

    Per ``memory/tournament_oracle.md``, chaos years should produce at least
    one mode with a large ranker gap (the canonical example: 2023 torvik
    +820). Locked here as "≥500 in at least one mode for the chaos year"
    — magnitude-loose so a future chaos tournament can shift the actual
    figure without breaking the test, but the *direction* of the regime
    claim is enforced.

    The ledger explicitly notes the regime correlation is "open, not yet
    locked" — this test enforces only the qualitative shape, not a precise
    magnitude.
    """
    gaps = [oracle_sweep_result["per_year"][2023][strat]["ranker_gap_espn_pts"] for strat in LEDGER_STRATEGIES]
    assert max(gaps) >= 500.0, f"Chaos 2023 should have at least one mode with gap >=500, got {gaps}"


def test_chalk_year_2025_no_mode_explodes(oracle_sweep_result):
    """Direction-only chalk check: 2025 (all 4 1-seeds in F4) should not
    produce any mode with the chaos-style 800+ gap. Loose ceiling at 500
    matches the ledger's chalk-year observations (max 280 across modes).
    """
    gaps = [oracle_sweep_result["per_year"][2025][strat]["ranker_gap_espn_pts"] for strat in LEDGER_STRATEGIES]
    assert max(gaps) < 500.0, f"Chalk 2025 should have no mode with gap >=500 (regime hypothesis), got {gaps}"


def test_drift_guard_anchors_match_ledger_documentation():
    """Sanity check on the test's own anchor table — every (year, strategy)
    cell present in EXPECTED_GAP_BY_YEAR_STRATEGY also appears in
    EXPECTED_F4_HITS and EXPECTED_CHAMP_HIT (no missing rows).
    """
    for year in LEDGER_YEARS:
        for strat in LEDGER_STRATEGIES:
            assert strat in EXPECTED_GAP_BY_YEAR_STRATEGY[year]
            assert strat in EXPECTED_F4_HITS_BY_YEAR_STRATEGY[year]
            assert strat in EXPECTED_CHAMP_HIT_BY_YEAR_STRATEGY[year]
        assert (
            EXPECTED_GAP_BY_YEAR_STRATEGY[year].keys()
            == EXPECTED_F4_HITS_BY_YEAR_STRATEGY[year].keys()
            == EXPECTED_CHAMP_HIT_BY_YEAR_STRATEGY[year].keys()
        )
