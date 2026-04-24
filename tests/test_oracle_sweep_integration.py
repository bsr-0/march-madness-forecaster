"""Integration tests for the phase-3 metrics rollout — T3 Oracle sweep.

Exercises ``oracle_sweep_t3_years`` against the real bracket artifacts
committed to the repo (``artifacts/backtest_brackets/backtest_brackets_{year}.json``
for 2023-2026). No backtest runs — reads existing artifacts only.
"""

import json
from pathlib import Path

import pytest

from scripts.run_experiment import oracle_sweep_t3_years

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRACKETS_DIR = PROJECT_ROOT / "artifacts" / "backtest_brackets"
DATA_ROOT = PROJECT_ROOT / "data"

# Modes we know are present in the committed bracket artifacts.
KNOWN_MODES = ("torvik", "f4_first_tv", "e8_first_tv")
KNOWN_YEARS = (2023, 2024, 2025, 2026)


def test_oracle_sweep_returns_per_year_and_aggregate_shapes():
    result = oracle_sweep_t3_years(
        list(KNOWN_YEARS),
        list(KNOWN_MODES),
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    assert set(result["per_year"].keys()) == set(KNOWN_YEARS)
    assert set(result["aggregate"].keys()) == set(KNOWN_MODES)


def test_oracle_sweep_per_year_entries_are_serializable_or_skipped():
    result = oracle_sweep_t3_years(
        list(KNOWN_YEARS),
        list(KNOWN_MODES),
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    # Every per-(year, strategy) entry round-trips through JSON.
    blob = json.dumps(result["per_year"])
    assert isinstance(blob, str)
    # Each entry is either a scored report or a {"skipped": ...} tombstone.
    for year_dict in result["per_year"].values():
        for strat_entry in year_dict.values():
            assert isinstance(strat_entry, dict)
            assert "skipped" in strat_entry or "ranker_gap_espn_pts" in strat_entry


def test_oracle_sweep_aggregate_has_all_required_fields():
    result = oracle_sweep_t3_years(
        list(KNOWN_YEARS),
        list(KNOWN_MODES),
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    for strat, agg in result["aggregate"].items():
        for key in (
            "mean_ranker_gap_espn_pts",
            "mean_f4_hits",
            "mean_finals_hits",
            "champ_hit_rate",
            "years_scored",
        ):
            assert key in agg, f"{strat} aggregate missing {key}"
        assert agg["years_scored"] == 4  # all 4 years have artifacts


def test_oracle_sweep_torvik_2026_matches_known_portfolio_facts():
    """MEMORY.md §3 anchor: 2026 torvik portfolio's best bracket was 1550 pts,
    the submitted (min mean_rank) bracket was 940 pts → ranker gap ≈ 610 pts.
    Locks the per-year report against the A-investigation numbers.
    """
    result = oracle_sweep_t3_years(
        [2026],
        ["torvik"],
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    torvik_2026 = result["per_year"][2026]["torvik"]
    # Known from earlier Oracle CLI output: torvik 2026 ranker gap ≈ +610.
    assert torvik_2026["ranker_gap_espn_pts"] == pytest.approx(610.0, abs=1.0)
    # Best bracket was Michigan champ, 3/4 F4, both finalists correct.
    assert torvik_2026["best_score_champ_hit"] is True
    assert torvik_2026["best_score_f4_hits"] == 3
    assert torvik_2026["best_score_finals_hits"] == 2


def test_oracle_sweep_handles_missing_year_gracefully():
    """A year without a bracket artifact produces skip tombstones, not a crash."""
    result = oracle_sweep_t3_years(
        [1999],  # no artifact exists
        ["torvik"],
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    entry = result["per_year"][1999]["torvik"]
    assert "skipped" in entry
    # Aggregate: 0 years scored → all-zero metrics, no crash.
    agg = result["aggregate"]["torvik"]
    assert agg["years_scored"] == 0
    assert agg["mean_ranker_gap_espn_pts"] == 0.0


def test_oracle_sweep_handles_strategy_not_in_artifact():
    """If a T3 survivor isn't in the saved artifact (mode-name mismatch),
    that strategy×year cell gets a skip tombstone without crashing."""
    result = oracle_sweep_t3_years(
        [2026],
        ["nonexistent_mode"],
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    entry = result["per_year"][2026]["nonexistent_mode"]
    assert "skipped" in entry
    assert result["aggregate"]["nonexistent_mode"]["years_scored"] == 0


def test_oracle_sweep_aggregate_ranker_gap_matches_per_year_mean():
    """The aggregate mean_ranker_gap_espn_pts must equal the straight mean
    of successfully-scored per-year gaps — catches bugs in the accumulator."""
    result = oracle_sweep_t3_years(
        list(KNOWN_YEARS),
        ["torvik"],
        brackets_dir=BRACKETS_DIR,
        data_root=DATA_ROOT,
    )
    per_year_gaps = [
        result["per_year"][y]["torvik"]["ranker_gap_espn_pts"]
        for y in KNOWN_YEARS
        if "ranker_gap_espn_pts" in result["per_year"][y]["torvik"]
    ]
    agg_mean = result["aggregate"]["torvik"]["mean_ranker_gap_espn_pts"]
    assert agg_mean == pytest.approx(sum(per_year_gaps) / len(per_year_gaps))
