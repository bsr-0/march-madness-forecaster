"""Regression test for COUNCIL_LESSONS §2 O9 — round-specific BSS decomposition.

Gate: "Per-round BSS breakdown (R64 → NCG). Does the ceiling hold uniformly
or break in E8/F4?"

Answer (from `scripts/backtest_by_round.py` run over 17 years, 1071 games,
committed as `artifacts/o9_round_decomposition_2026-04-13.json`):

- **Ceiling is NOT uniform.** E8 upset rate is 42.6% vs R64's 24.6% (+18.0pp).
  Later rounds (S16+) average 29.8% upset rate vs early rounds' (R64+R32)
  24.5%, a statistically meaningful +5.3pp gap.
- **But seed-independent signals still don't help anywhere.** |BSS| across
  every round is ≤ 0.033 (the 0.033 is NCG, n=17 — underpowered). Every
  other round has |BSS| ≤ 0.018. The most upset-prone round (E8, 42.6%)
  shows BSS = −0.010 — SI is *slightly worse* than seed there.

Interpretation (from the same script and confirmed by this test): the
non-uniformity does not open an exploitable gap. The 2026-04-02 pivot
(§3 row 7) is consistent with this finding — even in the highest-variance
round, prediction accuracy is not the improvement axis.

If any of these tests fire, the 1071-game round-segmented backtest
produced different numbers on a re-run. Regenerate the JSON artifact
and review the ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT = Path("artifacts/o9_round_decomposition_2026-04-13.json")


@pytest.fixture(scope="module")
def d() -> dict:
    with ARTIFACT.open() as f:
        return json.load(f)


def test_artifact_present_and_references_script(d: dict) -> None:
    assert d["council_item"] == "O9"
    assert d["source_script"].endswith("backtest_by_round.py")
    assert d["n_games_total"] >= 1000
    assert d["n_years"] >= 15


def test_every_round_has_expected_structure(d: dict) -> None:
    expected_rounds = {"R64", "R32", "S16", "E8", "F4", "NCG"}
    rounds_in_artifact = {r["round"] for r in d["per_round"]}
    assert rounds_in_artifact == expected_rounds, f"expected {expected_rounds}, got {rounds_in_artifact}"
    for r in d["per_round"]:
        assert r["n_games"] > 0
        assert 0 <= r["upset_rate"] <= 1.0
        assert r["avg_seed_gap"] >= 0


def test_ceiling_is_non_uniform_across_rounds(d: dict) -> None:
    """The key O9 finding: later rounds have measurably higher upset
    rate than early rounds. If this closes, the council's row-7
    framing ("seeds beat everything uniformly") would need revision.
    Today: E8 42.6% vs R64 24.6%; later-round aggregate 29.8% vs
    early-round 24.5% (+5.3pp)."""
    agg = d["aggregate"]
    early = agg["early_rounds"]["upset_rate"]
    later = agg["later_rounds"]["upset_rate"]
    assert later > early, f"later={later} vs early={early} — non-uniform ceiling closed?"
    assert agg["upset_rate_delta_later_minus_early_pp"] >= 2.0, (
        f"delta={agg['upset_rate_delta_later_minus_early_pp']}pp < 2.0pp; "
        f"ceiling non-uniformity has weakened materially."
    )


def test_e8_is_most_upset_prone_round(d: dict) -> None:
    """Every prior run of this analysis (including the 2026-04-01 data
    feeding the pivot) has found E8 as the highest-upset round. If that
    ever changes, a data refresh has shifted the tournament's historical
    shape and the pool strategy may need a rethink."""
    per_round = {r["round"]: r for r in d["per_round"]}
    e8_rate = per_round["E8"]["upset_rate"]
    for rname, r in per_round.items():
        if rname == "E8":
            continue
        assert r["upset_rate"] < e8_rate, (
            f"{rname} upset rate {r['upset_rate']} >= E8 {e8_rate} — E8 is no longer the highest-upset round."
        )


def test_si_signals_do_not_meaningfully_help_in_any_round(d: dict) -> None:
    """The O9 bottom line: even in the most-upset round (E8), SI signals
    don't add meaningful BSS. Threshold 0.05 is 5× the largest observed
    effect (+0.018 at S16) and catches any future change that would
    re-open the prediction-accuracy axis.

    NCG (n=17) is excluded because its +0.033 BSS is noise at that
    sample size — the regression guard only fires if a well-powered
    round ever breaks."""
    POWERED_MIN_N = 30
    BSS_TAUTOLOGY_BOUND = 0.05
    for r in d["per_round"]:
        if r["n_games"] < POWERED_MIN_N:
            continue
        assert abs(r["bss"]) < BSS_TAUTOLOGY_BOUND, (
            f"Round {r['round']}: |BSS|={abs(r['bss'])} exceeds "
            f"{BSS_TAUTOLOGY_BOUND}. Prediction-accuracy axis may have "
            f"reopened. Re-evaluate §3 row 7 pivot."
        )


def test_aggregate_sample_counts_match_per_round(d: dict) -> None:
    per_round_sum = sum(r["n_games"] for r in d["per_round"])
    early_sum = d["aggregate"]["early_rounds"]["n_games"]
    later_sum = d["aggregate"]["later_rounds"]["n_games"]
    assert per_round_sum == early_sum + later_sum == d["n_games_total"], (
        f"game counts inconsistent: per-round={per_round_sum}, "
        f"early={early_sum}, later={later_sum}, total={d['n_games_total']}"
    )


def test_later_rounds_si_signal_is_flat(d: dict) -> None:
    """Council row 7's "+0.2pp lift even in the most favorable round"
    quote: later-round High-SI vs Low-SI quintile difference must stay
    within ±3pp. A deviation means a new data source or a feature-
    engineering change has made SI signals informative in later rounds,
    which would be a meaningful finding worth re-litigating the pivot."""
    si = d["later_rounds_si_quintile"]
    assert abs(si["difference_pp"]) < 3.0, (
        f"later-round SI-quintile difference = {si['difference_pp']}pp (>3pp). Re-evaluate §3 row 7 and MEMORY §2 D8."
    )
