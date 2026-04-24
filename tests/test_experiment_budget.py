"""Unit tests for the Testing Budget logic in ``scripts.run_experiment``.

These cover the pure helpers (``aggregate_strategy_stats``,
``apply_kill_threshold``, ``promote_top_n``, ``select_tier_years``)
without invoking the backtest. They lock the T1/T2/T3 kill contract
documented in STRATEGY_CATALOG.md § Testing Budget — if a refactor
breaks these the tier pipeline silently under-prunes or over-prunes.
"""

import pytest

from src.evaluation.testing_budget import (
    TIER_CONFIGS,
    aggregate_strategy_stats,
    apply_kill_threshold,
    best_baseline_delta,
    promote_top_n,
    select_tier_years,
    stats_for_artifact,
)


BASELINE = "seed_forward"


def _mk(mode, year, p_first, **overrides):
    """Synthetic backtest result row. Primary/secondary fields default to 0."""
    row = {
        "mode": mode,
        "year": year,
        "p_first": p_first,
        "mean_rank": 0.0,
        "best_rank": 0.0,
        "mean_score": 0.0,
        "best_score": 0.0,
        "p_top5": 0.0,
        "p_top25": 0.0,
    }
    row.update(overrides)
    return row


def test_tier_configs_match_catalog_contract():
    """The tier parameters must match what STRATEGY_CATALOG.md § Testing Budget promises."""
    t1 = TIER_CONFIGS["T1"]
    assert (t1.n_repeats, t1.year_mode, t1.promote_top_n) == (25, "recent8", 10)
    assert t1.kill_on_zero_year is True
    assert t1.kill_min_wins_vs_baseline == 3  # out of 8 screening years

    t2 = TIER_CONFIGS["T2"]
    assert (t2.n_repeats, t2.year_mode, t2.promote_top_n) == (50, "full", 5)
    assert t2.kill_min_wins_vs_baseline == 5

    t3 = TIER_CONFIGS["T3"]
    assert (t3.n_repeats, t3.year_mode) == (100, "full")
    assert t3.kill_min_wins_vs_baseline == 8  # catalog significance gate


def test_select_tier_years_recent8():
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
    t1 = TIER_CONFIGS["T1"]
    picked = select_tier_years(years, t1)
    assert picked == [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
    assert len(picked) == 8


def test_select_tier_years_full():
    years = [2011, 2012, 2013]
    t2 = TIER_CONFIGS["T2"]
    assert select_tier_years(years, t2) == [2011, 2012, 2013]


def test_aggregate_strategy_stats_tracks_primary_metric_triple():
    """Phase-1 metrics rollout: P(1st), BestScore, MeanScore all aggregated
    with mean + 95% CI across years. Secondary p_top5/p_top25 too.
    """
    results = [
        _mk(BASELINE, 2023, 0.05, best_score=900, mean_score=800, p_top5=0.10, p_top25=0.25),
        _mk(BASELINE, 2024, 0.05, best_score=900, mean_score=800, p_top5=0.10, p_top25=0.25),
        _mk("A", 2023, 0.06, best_score=1400, mean_score=1100, p_top5=0.20, p_top25=0.40),
        _mk("A", 2024, 0.08, best_score=1500, mean_score=1200, p_top5=0.25, p_top25=0.45),
    ]
    stats = aggregate_strategy_stats(results, BASELINE)

    # Primary metrics: all three of the catalog's payout-determining columns.
    assert stats["A"]["mean_p_first"] == pytest.approx(0.07, abs=1e-9)
    assert stats["A"]["mean_best_score"] == pytest.approx(1450.0, abs=1e-9)
    assert stats["A"]["mean_mean_score"] == pytest.approx(1150.0, abs=1e-9)
    # Secondary placement diagnostics surfaced (used to be dropped).
    assert stats["A"]["mean_p_top5"] == pytest.approx(0.225, abs=1e-9)
    assert stats["A"]["mean_p_top25"] == pytest.approx(0.425, abs=1e-9)
    # CI95 half-widths present and non-negative; must be 0 for a 1-observation series
    # (but we have 2 years here so non-zero).
    assert stats["A"]["ci95_p_first"] > 0
    assert stats["A"]["ci95_best_score"] > 0


def test_ci95_halfwidth_is_zero_for_single_year_series():
    """Single-observation series can't produce a cross-year SEM — CI half-width = 0."""
    results = [_mk("solo", 2024, 0.06, best_score=1000, mean_score=900)]
    stats = aggregate_strategy_stats(results, "solo")
    assert stats["solo"]["ci95_p_first"] == 0.0
    assert stats["solo"]["ci95_best_score"] == 0.0


def test_aggregate_stats_backward_compatible_with_missing_fields():
    """Rows missing the new p_top5/best_score fields must not crash aggregation.

    Older artifacts (pre-phase-1 metrics rollout) don't have those fields;
    the aggregator must gracefully default to 0.0 so `--oracle <year>` and
    other diagnostic readers can still work against old result dicts.
    """
    legacy_results = [
        {"mode": "legacy", "year": 2023, "p_first": 0.05, "mean_rank": 0, "best_rank": 0, "mean_score": 0},
        {"mode": "legacy", "year": 2024, "p_first": 0.06, "mean_rank": 0, "best_rank": 0, "mean_score": 0},
    ]
    stats = aggregate_strategy_stats(legacy_results, "legacy")
    assert stats["legacy"]["mean_p_first"] == pytest.approx(0.055, abs=1e-9)
    assert stats["legacy"]["mean_best_score"] == 0.0
    assert stats["legacy"]["mean_p_top5"] == 0.0


def test_aggregate_strategy_stats_wins_and_zero_years():
    # Baseline: flat 0.05 every year.
    # A: beats baseline in 3 of 4 years (wins=3).
    # B: hits zero in year 2023 (zero_years=1).
    results = [
        _mk(BASELINE, 2023, 0.05),
        _mk(BASELINE, 2024, 0.05),
        _mk(BASELINE, 2025, 0.05),
        _mk(BASELINE, 2026, 0.05),
        _mk("A", 2023, 0.06),
        _mk("A", 2024, 0.07),
        _mk("A", 2025, 0.04),
        _mk("A", 2026, 0.08),
        _mk("B", 2023, 0.0),
        _mk("B", 2024, 0.06),
        _mk("B", 2025, 0.09),
        _mk("B", 2026, 0.01),
    ]
    stats = aggregate_strategy_stats(results, BASELINE)
    assert stats["A"]["wins_vs_baseline"] == 3
    assert stats["A"]["zero_years"] == 0
    assert stats["A"]["n_years"] == 4
    assert stats["B"]["zero_years"] == 1
    # mean P(1st) sanity
    assert abs(stats["A"]["mean_p_first"] - (0.06 + 0.07 + 0.04 + 0.08) / 4) < 1e-9


def test_apply_kill_threshold_t1_zero_year_kills_strategy():
    t1 = TIER_CONFIGS["T1"]
    stats = {
        BASELINE: {"mean_p_first": 0.04, "wins_vs_baseline": 0, "n_years": 8, "zero_years": 0, "p1_by_year": {}},
        "zero_hitter": {"mean_p_first": 0.10, "wins_vs_baseline": 7, "n_years": 8, "zero_years": 1, "p1_by_year": {}},
        "consistent": {"mean_p_first": 0.05, "wins_vs_baseline": 5, "n_years": 8, "zero_years": 0, "p1_by_year": {}},
    }
    survivors, killed = apply_kill_threshold(stats, t1, BASELINE)
    killed_names = {s for s, _ in killed}
    # High P(1st) but one zero year -> killed under T1's zero-year rule.
    assert "zero_hitter" in killed_names
    assert "consistent" in survivors
    assert BASELINE in survivors  # baseline always survives


def test_apply_kill_threshold_t1_min_wins_kills_weak_strategy():
    t1 = TIER_CONFIGS["T1"]
    stats = {
        BASELINE: {"mean_p_first": 0.04, "wins_vs_baseline": 0, "n_years": 8, "zero_years": 0, "p1_by_year": {}},
        "weak": {
            "mean_p_first": 0.03,
            "wins_vs_baseline": 2,
            "n_years": 8,
            "zero_years": 0,
            "p1_by_year": {},
        },  # < 3 wins
        "ok": {
            "mean_p_first": 0.05,
            "wins_vs_baseline": 3,
            "n_years": 8,
            "zero_years": 0,
            "p1_by_year": {},
        },  # exactly 3
    }
    survivors, killed = apply_kill_threshold(stats, t1, BASELINE)
    killed_names = {s for s, _ in killed}
    assert "weak" in killed_names
    assert "ok" in survivors


def test_apply_kill_threshold_t2_no_zero_year_rule():
    t2 = TIER_CONFIGS["T2"]
    stats = {
        BASELINE: {"mean_p_first": 0.04, "wins_vs_baseline": 0, "n_years": 14, "zero_years": 0, "p1_by_year": {}},
        "has_zero": {"mean_p_first": 0.05, "wins_vs_baseline": 6, "n_years": 14, "zero_years": 1, "p1_by_year": {}},
    }
    survivors, killed = apply_kill_threshold(stats, t2, BASELINE)
    # T2 does not kill on zero-year (only T1 does).
    assert "has_zero" in survivors
    assert killed == []


def test_apply_kill_threshold_t3_significance_gate():
    """T3 kill threshold == catalog significance gate (>= 8 wins out of 14)."""
    t3 = TIER_CONFIGS["T3"]
    stats = {
        BASELINE: {"mean_p_first": 0.04, "wins_vs_baseline": 0, "n_years": 14, "zero_years": 0, "p1_by_year": {}},
        "7_wins": {"mean_p_first": 0.06, "wins_vs_baseline": 7, "n_years": 14, "zero_years": 0, "p1_by_year": {}},
        "8_wins": {"mean_p_first": 0.07, "wins_vs_baseline": 8, "n_years": 14, "zero_years": 0, "p1_by_year": {}},
    }
    survivors, killed = apply_kill_threshold(stats, t3, BASELINE)
    killed_names = {s for s, _ in killed}
    assert "7_wins" in killed_names
    assert "8_wins" in survivors


def test_promote_top_n_ranks_by_mean_p_first_and_keeps_baseline():
    survivors = {
        BASELINE: {"mean_p_first": 0.040},
        "A": {"mean_p_first": 0.080},
        "B": {"mean_p_first": 0.050},
        "C": {"mean_p_first": 0.065},
    }
    promoted = promote_top_n(survivors, BASELINE, n=2)
    # Top-2 by mean P(1st) are A (0.08) and C (0.065); baseline appended.
    assert promoted[:2] == ["A", "C"]
    assert BASELINE in promoted


def test_promote_top_n_includes_baseline_when_baseline_is_top():
    survivors = {
        BASELINE: {"mean_p_first": 0.090},
        "A": {"mean_p_first": 0.020},
    }
    promoted = promote_top_n(survivors, BASELINE, n=1)
    # Baseline is excluded from ranking then appended — we always get it exactly once.
    assert promoted.count(BASELINE) == 1
    assert "A" in promoted


def test_best_baseline_delta_picks_largest_positive():
    stats = {
        BASELINE: {"mean_p_first": 0.040},
        "A": {"mean_p_first": 0.050},
        "B": {"mean_p_first": 0.032},
        "C": {"mean_p_first": 0.070},  # best
    }
    # C is +0.030 over baseline — the largest delta.
    assert abs(best_baseline_delta(stats, BASELINE) - 0.030) < 1e-12


def test_best_baseline_delta_returns_none_when_baseline_missing():
    assert best_baseline_delta({"A": {"mean_p_first": 0.05}}, BASELINE) is None


def test_stats_for_artifact_includes_primary_metric_triple_plus_secondaries():
    """Locks the JSON-artifact contract — every metric surfaced in tier reports
    must also be serialized, so post-run audits don't lose information."""
    results = [
        _mk("A", 2023, 0.06, best_score=1400, mean_score=1100, p_top5=0.20, p_top25=0.40),
        _mk("A", 2024, 0.08, best_score=1500, mean_score=1200, p_top5=0.25, p_top25=0.45),
    ]
    stats = aggregate_strategy_stats(results, "A")
    artifact = stats_for_artifact(stats)["A"]

    # Primary: payout-determining.
    for key in (
        "mean_p_first",
        "ci95_p_first",
        "mean_best_score",
        "ci95_best_score",
        "mean_mean_score",
        "ci95_mean_score",
    ):
        assert key in artifact, f"primary metric {key!r} missing from artifact"

    # Secondary: placement diagnostics.
    for key in (
        "mean_p_top5",
        "ci95_p_top5",
        "mean_p_top25",
        "ci95_p_top25",
        "mean_best_rank",
        "mean_mean_rank",
    ):
        assert key in artifact, f"secondary metric {key!r} missing from artifact"

    # Values round-trip correctly.
    assert artifact["mean_best_score"] == pytest.approx(1450.0)
    assert artifact["mean_p_top5"] == pytest.approx(0.225, abs=1e-9)
