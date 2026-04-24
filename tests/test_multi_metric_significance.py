"""Tests for phase-2 multi-metric significance harness.

Locks that ``_paired_permutation_on_metric`` can be pointed at any
primary metric (P(1st), BestScore, MeanScore) and returns ordered
test tuples, plus ``_run_significance_tests`` now emits three metric
blocks rather than just one.
"""

import io
from contextlib import redirect_stdout

from scripts.run_experiment import _paired_permutation_on_metric, _run_significance_tests


def _mk_row(mode, year, p_first=0.05, best_score=0.0, mean_score=0.0):
    return {
        "mode": mode,
        "year": year,
        "p_first": p_first,
        "best_score": best_score,
        "mean_score": mean_score,
        "best_rank": 0.0,
        "mean_rank": 0.0,
        "p_top5": 0.0,
        "p_top25": 0.0,
        "n_trials": 2500,
    }


def _baseline_plus_winner_on_metric(metric_key: str):
    """8 years of data: 'baseline' flat, 'winner' beats it on `metric_key` every year."""
    rows = []
    for year in range(2018, 2026):
        rows.append(_mk_row("baseline", year, p_first=0.05, best_score=900, mean_score=800))
        # winner is identical on other metrics but wins the target one.
        winner_kwargs = {"p_first": 0.05, "best_score": 900, "mean_score": 800}
        winner_kwargs[metric_key] = {"p_first": 0.09, "best_score": 1300, "mean_score": 1200}[metric_key]
        rows.append(_mk_row("winner", year, **winner_kwargs))
    return rows


def test_paired_permutation_returns_sorted_results_by_p_value():
    results = _baseline_plus_winner_on_metric("p_first")
    out = _paired_permutation_on_metric(results, ["winner"], "baseline", "p_first")
    assert out, "Should return at least one tuple"
    strategy, delta, p, wins, n_shared = out[0]
    assert strategy == "winner"
    assert delta > 0  # winner beats baseline
    assert 0.0 <= p <= 1.0
    assert wins == 8  # 8/8 years


def test_paired_permutation_works_on_best_score():
    """Scoring on a continuous ESPN-points metric returns a valid p-value."""
    results = _baseline_plus_winner_on_metric("best_score")
    out = _paired_permutation_on_metric(results, ["winner"], "baseline", "best_score")
    assert out
    _, delta, p, wins, _ = out[0]
    assert delta > 0  # ESPN points delta positive
    assert wins == 8


def test_paired_permutation_works_on_mean_score():
    results = _baseline_plus_winner_on_metric("mean_score")
    out = _paired_permutation_on_metric(results, ["winner"], "baseline", "mean_score")
    assert out
    _, delta, _, wins, _ = out[0]
    assert delta > 0
    assert wins == 8


def test_paired_permutation_missing_metric_falls_back_to_zero():
    """Rows without the metric key (e.g., legacy rows missing best_score) get 0.0 —
    so differencing zero against zero yields delta=0 without crashing.
    """
    legacy = [
        {"mode": "baseline", "year": 2023, "p_first": 0.05},
        {"mode": "winner", "year": 2023, "p_first": 0.09},
        {"mode": "baseline", "year": 2024, "p_first": 0.05},
        {"mode": "winner", "year": 2024, "p_first": 0.09},
        {"mode": "baseline", "year": 2025, "p_first": 0.05},
        {"mode": "winner", "year": 2025, "p_first": 0.09},
        {"mode": "baseline", "year": 2026, "p_first": 0.05},
        {"mode": "winner", "year": 2026, "p_first": 0.09},
        {"mode": "baseline", "year": 2022, "p_first": 0.05},
        {"mode": "winner", "year": 2022, "p_first": 0.09},
    ]
    # Querying best_score on rows that don't have it → delta ≈ 0, still runs.
    out = _paired_permutation_on_metric(legacy, ["winner"], "baseline", "best_score")
    assert out
    _, delta, _, _, _ = out[0]
    assert delta == 0.0


def test_run_significance_tests_prints_three_metric_blocks():
    """Smoke test: stdout capture shows all three metric blocks (P(1st), BestScore, MeanScore)."""
    # _run_significance_tests hard-codes "seed_forward" as the baseline key.
    results = []
    for y in range(2018, 2026):
        results.append(_mk_row("seed_forward", y, p_first=0.05, best_score=900, mean_score=800))
        results.append(_mk_row("cand", y, p_first=0.08, best_score=1200, mean_score=1100))

    buf = io.StringIO()
    with redirect_stdout(buf):
        _run_significance_tests(results, ["seed_forward", "cand"])

    out = buf.getvalue()
    # Each primary metric appears as a block header.
    assert "[P(1st)]" in out
    assert "[BestScore]" in out
    assert "[MeanScore]" in out
    # The candidate appears in every block.
    assert out.count("cand") >= 3
