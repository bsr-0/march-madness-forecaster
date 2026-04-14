"""Regression test for COUNCIL_LESSONS §2 O8 — feature/seed collinearity.

The question O8 posed: are the production `SIMPLE_FEATURE_SET` features
just proxies for seed? If every feature has |r| > 0.7 with seed, then the
locked "BSS ≈ 0 is ground truth" claim (MEMORY.md §2 D1) would be
tautological — the features ARE seed.

The gate this test enforces: no production feature reaches a total-tautology
threshold of |r| >= 0.85. The actual measured max as of 2026-04-13 is
|r|(adj_off_eff) ≈ 0.77, comfortably below. If a future data regeneration
or feature redefinition pushes any correlation past 0.85, this test fails
loudly — forcing a council re-litigation of the pivot before the change
ships.

The diagnostic itself lives in scripts/feature_seed_correlation.py; this
test just asserts the measured values are within expected bounds.
"""

from __future__ import annotations

import math

import pytest


# If any single production feature crosses this bound, BSS ≈ 0 as a claim
# needs to be re-evaluated — the feature may be a pure seed proxy and the
# lock in MEMORY.md §2 D1 would be tautological.
TAUTOLOGY_THRESHOLD = 0.85

# Soft warning bound: observed max as of 2026-04-13 is 0.77 for
# adj_off_eff. If we exceed 0.80 we're close enough to the boundary that
# the council should be notified even if the hard gate hasn't tripped.
DRIFT_WARNING_BOUND = 0.80

# Required minimum sample size per feature before the test is meaningful.
MIN_N = 100


def test_compute_production_feature_correlations_runs_offline():
    """The diagnostic must be runnable offline from committed historical
    JSON snapshots. If this fails, the data dependency or loader broke."""
    from scripts.feature_seed_correlation import compute_production_feature_correlations

    results = compute_production_feature_correlations()
    assert isinstance(results, dict)
    assert len(results) >= 5, (
        f"Expected correlations for at least 5 production features, got {len(results)}. "
        f"Check that data/raw/historical/tournament_seeds_*.json and data/raw/torvik_*.json "
        f"are present for the production training years."
    )


def test_no_production_feature_is_total_seed_proxy():
    """O8 gate: no production feature has |r| >= 0.85 with seed.

    This is the tautology guard. If a refactor or data regeneration pushes
    any production feature past this bound, the locked 'BSS ≈ 0 is ground
    truth' claim in MEMORY.md §2 D1 becomes 'BSS ≈ 0 is tautological
    because the features ARE seed' — a fundamentally different situation
    that must be re-litigated by the council.
    """
    from scripts.feature_seed_correlation import compute_production_feature_correlations

    results = compute_production_feature_correlations()
    violators = []
    for prod_name, (r, n) in results.items():
        if not math.isfinite(r):
            continue  # feature absent from offline data; ignore
        if n < MIN_N:
            continue
        if abs(r) >= TAUTOLOGY_THRESHOLD:
            violators.append((prod_name, r, n))

    assert not violators, (
        f"Feature/seed correlation exceeds tautology threshold {TAUTOLOGY_THRESHOLD}: "
        f"{violators}. Re-litigate MEMORY.md §2 D1 before accepting this change. "
        f"See COUNCIL_LESSONS.md §2 O8."
    )


@pytest.mark.parametrize(
    "feature, expected_abs_r_lt",
    [
        # Expected |r| upper bounds from 2026-04-13 run. Loose bounds that
        # catch real drift while tolerating sampling noise year-to-year.
        ("diff_adj_tempo", 0.40),
        ("diff_opp_to_rate", 0.40),
        ("diff_orb_rate", 0.55),
    ],
)
def test_independent_features_remain_independent(feature, expected_abs_r_lt):
    """Features the 2026-04-13 diagnostic labelled 'independent' or
    'partial independence' (|r| < 0.3 to ~0.32) should stay loose. A
    sudden jump in these specifically would indicate the feature
    definition changed — not sampling noise."""
    from scripts.feature_seed_correlation import compute_production_feature_correlations

    results = compute_production_feature_correlations()
    if feature not in results:
        pytest.skip(f"{feature} not present in offline snapshots")
    r, n = results[feature]
    if not math.isfinite(r) or n < MIN_N:
        pytest.skip(f"{feature}: insufficient data (r={r}, n={n})")
    assert abs(r) < expected_abs_r_lt, (
        f"{feature} correlation |r|={abs(r):.3f} exceeds loose bound {expected_abs_r_lt}. "
        f"Check whether the feature definition changed or data regenerated."
    )
