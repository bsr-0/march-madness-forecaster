"""Regression test for COUNCIL_LESSONS §2 O7 — tournament-only vs blended+stacked.

Gate: "LOYO BSS comparison; tournament-only (500-600 games) vs blended (2200).
Flagged 04-01; never run."

The tournament-only LOYO run was executed 2026-04-01 and stored as
`artifacts/baseline_experiment.json`. This test reads the paired
per-year Brier scores from that file and the derived statistics in
`artifacts/o7_regime_comparison_2026-04-13.json` and locks the
headline findings:

1. Tournament-only simple_7_logistic beats seed baseline by paired BSS
   +4.8% across 14 years, with paired t-test p = 0.036 (uncorrected
   at α=0.05) and 11/14 year-wins.
2. fixed_27_logistic, seed_only_logistic, and seed_plus do NOT reach
   significance vs seed baseline when LOYO-trained on tournament-only.
3. None of the tournament-only configs survive Bonferroni correction
   across the 4 configs tested (α=0.0125).

Head-to-head conclusion: the "blended+stacked" arm (MEMORY.md §2 D1)
aggregates BSS = 0 for LightGBM, XGBoost, SpreadRegressor, 27-feature
ensemble + LightGBM stacking. Tournament-only simple_7 is directionally
better (+4.8% vs 0%) but the effect is marginal and underpowered per §1
("14-17-year backtests are underpowered (~9-16% power). Use them for
binary/directional claims — do NOT use them to tune parameter values").
The 2026-04-02 pivot (§3 row 7) is not reopened.

If this test fires, something has materially changed. Either:
 - A new run of baseline_experiment.py produced different numbers
   (regenerate o7_regime_comparison_*.json and update the ledger), or
 - A code change has broken the tournament-only training path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BASELINE_ARTIFACT = Path("artifacts/baseline_experiment.json")
O7_ARTIFACT = Path("artifacts/o7_regime_comparison_2026-04-13.json")


@pytest.fixture(scope="module")
def baseline() -> dict:
    with BASELINE_ARTIFACT.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def o7() -> dict:
    with O7_ARTIFACT.open() as f:
        return json.load(f)


def test_baseline_artifact_is_tournament_only(baseline: dict) -> None:
    assert baseline["experiment"] == "tournament_only_baseline"
    assert len(baseline["years"]) >= 14
    assert baseline["total_tournament_samples"] >= 2000


def test_o7_artifact_present_and_references_baseline(o7: dict) -> None:
    assert o7["council_item"] == "O7"
    assert o7["source_artifact"].endswith("baseline_experiment.json")
    assert o7["regime"] == "tournament-only training"
    assert o7["n_years"] >= 14


def test_simple_7_logistic_beats_seed_baseline(o7: dict) -> None:
    """Headline finding: tournament-only simple_7 has paired BSS > 3%
    and Wilcoxon/paired-t p < 0.05 uncorrected."""
    rec = next(r for r in o7["per_config"] if r["name"] == "simple_7_logistic")
    assert rec["paired_bss"] > 0.03, f"paired BSS {rec['paired_bss']} <= 0.03"
    assert rec["paired_t_p_value_two_sided"] < 0.05, f"paired t-test p={rec['paired_t_p_value_two_sided']} not < 0.05"
    assert rec["wilcoxon_p_value"] < 0.05, f"wilcoxon p={rec['wilcoxon_p_value']} not < 0.05"
    assert rec["model_beats_seed_in_years"] >= 10, (
        f"simple_7 beats seed in only {rec['model_beats_seed_in_years']}/14 years"
    )


def test_no_tournament_only_config_clears_bonferroni(o7: dict) -> None:
    """With 4 configs tested, Bonferroni threshold is α=0.0125. The
    ledger's "underpowered backtests" lesson means a single uncorrected
    p=0.036 is directional evidence, not a license to re-litigate the
    pivot. Locking this: no tournament-only config passes the corrected
    threshold."""
    for rec in o7["per_config"]:
        assert not rec["passes_bonferroni_corrected"], (
            f"{rec['name']} now passes Bonferroni-corrected α=0.0125 "
            f"(p={rec['paired_t_p_value_two_sided']}) — re-evaluate the "
            f"pivot lock in COUNCIL_LESSONS §3 row 7."
        )


def test_fixed_27_does_not_significantly_beat_seed(o7: dict) -> None:
    """Complex-feature model should NOT beat seed at uncorrected p<0.05
    on tournament-only training (consistent with MEMORY.md §2 D1
    "BSS = 0 across all model configurations tested")."""
    rec = next(r for r in o7["per_config"] if r["name"] == "fixed_27_logistic")
    assert rec["paired_t_p_value_two_sided"] > 0.05, (
        f"fixed_27 now significant vs seed (p={rec['paired_t_p_value_two_sided']}) — MEMORY §2 D1 may need revisiting."
    )


def test_per_year_paired_records_match_baseline_brier(baseline: dict) -> None:
    """Sanity: per-year Brier scores in baseline_experiment.json must
    still reflect tournament-only training (n_train < 3000 per year).
    A regression here means someone re-ran the experiment with a
    different training set."""
    simp = baseline["configurations"]["simple_7_logistic"]["per_year"]
    for rec in simp:
        assert rec["n_train"] <= 3000, (
            f"{rec['year']}: n_train={rec['n_train']} — exceeds "
            f"tournament-only ceiling. baseline_experiment.json may have "
            f"been re-run with regular-season games accidentally included."
        )
