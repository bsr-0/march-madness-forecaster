"""Regression test for COUNCIL_LESSONS §2 O6 — pool-optimizer P(1st) calibration.

Gate: "Historical verification that brackets ranked highest by optimizer actually
won more often."

Evidence artifact: `artifacts/o6_winner_rank_diagnostic.json`, generated from
`artifacts/rank_correlation_diagnostic.json` (the O3 closure data).

## What this test enforces

The O3 closure showed mean Spearman ρ = +0.37 across 14 years between
predicted P(1st) rank and actual pool placement. O6 asks the stronger
calibration question: *do the actual winners tend to be in the top of the
predicted ranking?* Yes — across 14 years (2011-2025 ex 2020):

- mean ρ = 0.372, one-sample one-sided t-test p = 0.0022
- 12/14 years show positive ρ, binomial sign test p = 0.0065
- mean fractional rank of actual winner in predicted ordering = 0.39
  (vs 0.5 expected by chance)
- actual winner lands in top-half of predicted order in 9/14 years
  (vs 7 expected by chance)

This is a modest but statistically significant calibration signal. Any
future change that drops mean ρ below 0.20 or the significance below
p=0.05 would be a regression — the optimizer's ranking would have lost
its directional signal. Don't relax the bounds if this test fires.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest


ARTIFACT = Path("artifacts/o6_winner_rank_diagnostic.json")


@pytest.fixture(scope="module")
def diagnostic() -> dict:
    with ARTIFACT.open() as f:
        return json.load(f)


def test_diagnostic_artifact_present(diagnostic: dict) -> None:
    """The committed evidence must be present and minimally well-formed."""
    assert diagnostic["council_item"] == "O6"
    assert "aggregate" in diagnostic
    assert "per_year" in diagnostic
    assert diagnostic["aggregate"]["n_years"] >= 14


def test_mean_rho_is_significantly_positive(diagnostic: dict) -> None:
    """Aggregate calibration: mean Spearman ρ across all backtest years must
    be positive and statistically significant (one-sided t-test, H0 mean=0).
    Today's value is +0.37 with p=0.0022; the gate at p<0.05 passes by
    25× margin. If this fires, a recent change to the optimizer,
    opponent model, or MC simulator has eroded the ranking signal."""
    agg = diagnostic["aggregate"]
    assert agg["mean_spearman_rho"] > 0.20, (
        f"mean ρ={agg['mean_spearman_rho']} below calibration threshold 0.20. "
        f"Find the regression — do not relax this bound. "
        f"See COUNCIL_LESSONS.md §2 O6."
    )
    assert agg["one_sample_t_test_p_value_one_sided"] < 0.05, (
        f"mean ρ is not significantly > 0 (one-sided t-test p={agg['one_sample_t_test_p_value_one_sided']})"
    )


def test_majority_of_years_show_positive_rho(diagnostic: dict) -> None:
    """Sign test: at least 10 of 14 years must show positive ρ. A binomial
    at 10/14 gives one-sided p=0.090; 11/14 gives p=0.029; 12/14 gives
    p=0.006. Today's 12/14 passes by 2-year margin over the 10/14 floor.
    Below 10/14 the aggregate mean is probably being carried by a single
    high-ρ year and the calibration claim loses credibility."""
    agg = diagnostic["aggregate"]
    min_positive = 10
    assert agg["years_positive_rho"] >= min_positive, (
        f"only {agg['years_positive_rho']}/{agg['n_years']} years show "
        f"positive ρ (need ≥ {min_positive}). Aggregate signal may be "
        f"driven by a single outlier year."
    )


def test_actual_winners_skew_toward_top_of_predicted_ranking(
    diagnostic: dict,
) -> None:
    """Winner-rank calibration: the specific O6 gate. Across all backtest
    years, the ACTUAL winner of the pool must land in the TOP HALF of the
    predicted P(1st) ordering in a majority of years. Today's value is
    9/14 (expected 7 by chance). A value at or below 7 means the
    optimizer's ranking has no winner-specific signal."""
    agg = diagnostic["aggregate"]
    n = agg["n_years"]
    min_top_half = n // 2 + 1  # strict majority
    assert agg["years_winner_in_top_half"] >= min_top_half, (
        f"actual winner in top half only "
        f"{agg['years_winner_in_top_half']}/{n} years (need ≥ "
        f"{min_top_half} for strict majority). Optimizer's winner-rank "
        f"signal is at or below chance."
    )


def test_winner_mean_fractional_rank_better_than_random(
    diagnostic: dict,
) -> None:
    """Mean fractional rank of actual winner must be meaningfully below
    0.5 (chance level). Today's value is 0.39. A threshold of 0.45 leaves
    headroom for noise but catches any regression that erases the signal."""
    agg = diagnostic["aggregate"]
    assert agg["mean_fractional_rank_of_actual_winner"] < 0.45, (
        f"mean fractional rank of actual winner = "
        f"{agg['mean_fractional_rank_of_actual_winner']} ≥ 0.45 "
        f"(close to or above chance at 0.5). Optimizer's P(1st) "
        f"ranking may have lost winner-discrimination power."
    )


def test_per_year_records_are_well_formed(diagnostic: dict) -> None:
    """Every per-year record must carry the inputs used for the aggregate
    stats. A test failure here indicates the evidence artifact was
    regenerated with a different schema and the other tests may be
    reading values that don't mean what they used to."""
    for py in diagnostic["per_year"]:
        assert "year" in py
        assert "n_brackets" in py and py["n_brackets"] > 0
        assert "spearman_rho" in py
        assert 1 <= py["pred_rank_of_actual_winner"] <= py["n_brackets"]
        assert 0 < py["fractional_rank_of_actual_winner"] <= 1.0


def test_aggregate_stats_match_per_year_data(diagnostic: dict) -> None:
    """Cross-check: recomputing the aggregate from per_year must match the
    stored aggregate. Guards against silent drift between the evidence
    artifact's summary and its detail."""
    agg = diagnostic["aggregate"]
    pys = diagnostic["per_year"]
    rhos = [py["spearman_rho"] for py in pys]
    recomputed_mean = statistics.mean(rhos)
    recomputed_pos = sum(1 for r in rhos if r > 0)
    assert abs(recomputed_mean - agg["mean_spearman_rho"]) < 1e-3
    assert recomputed_pos == agg["years_positive_rho"]
