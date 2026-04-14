"""Closure / lock test for COUNCIL_LESSONS §2 O14 — calibration-to-pool.

Gate: "Measure whether slightly upset-over-estimating calibration
improves pool rank vs true calibration."

Verdict (from artifacts/o14_calibration_sweep_audit_2026-04-14.md):
FALSIFIED. Upset-over-estimation (T>1) monotonically DECREASES the
fraction of sampled brackets that beat chalk. The pool-rank-optimal T
is actually sharper (T=0.5), and honest calibration (T=1) is
near-optimal on every metric.

This test fails if:
  - The audit artifacts disappear
  - Brier-optimal T drifts from 1.0 (would break the honest-cal theorem)
  - The pBeatChalk-vs-T curve flips monotonicity (would reopen the gate)
  - The T≠1 bracket-score effect blows up beyond the documented
    'small-interaction' magnitude
  - The recorded aggregate numbers shift wildly without a new audit

DO NOT relax a bound. A flip in any of these is a real finding worth
a fresh audit, not a reason to weaken this test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_MD = REPO_ROOT / "artifacts" / "o14_calibration_sweep_audit_2026-04-14.md"
AUDIT_JSON = REPO_ROOT / "artifacts" / "o14_calibration_sweep_2026-04-14.json"
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "o14_calibration_sweep.py"


def test_audit_artifacts_present() -> None:
    """Dated audit (md + json) and the sweep script must stay on disk."""
    for p in (AUDIT_MD, AUDIT_JSON, SWEEP_SCRIPT):
        assert p.exists(), f"Missing {p}; restore from git or re-issue."
    assert AUDIT_MD.stat().st_size > 3000, "Audit md suspiciously short."


def test_json_schema_has_expected_fields() -> None:
    """Sweep output must carry the fields the closure depends on."""
    data = json.loads(AUDIT_JSON.read_text())
    for key in (
        "aggregate",
        "brier_optimal_T",
        "score_mean_optimal_T",
        "score_best_optimal_T",
        "beats_chalk_optimal_T",
        "temperatures",
        "backtest_years",
    ):
        assert key in data, f"o14 sweep json missing {key!r}"

    agg = data["aggregate"]
    assert len(agg) >= 7, f"Expected ≥7 temperatures, got {len(agg)}"
    # Every row carries the core metrics.
    for T_str, row in agg.items():
        for m in ("mean_brier", "mean_best_score", "mean_beats_chalk_rate"):
            assert m in row, f"T={T_str} row missing {m!r}"


def test_brier_optimal_stays_at_honest_calibration() -> None:
    """T=1 is the well-calibrated identity. If the Brier curve's
    minimum drifts, either the base model's calibration changed or
    the methodology broke. Either way, re-audit before relocking."""
    data = json.loads(AUDIT_JSON.read_text())
    brier_T = float(data["brier_optimal_T"])
    assert brier_T == 1.0, (
        f"Brier-optimal T = {brier_T}, expected 1.0. The base seed "
        f"probabilities drifted out of calibration, or the sweep "
        f"methodology shifted. Re-audit O14 before locking a new "
        f"temperature."
    )


def test_pbeats_chalk_monotone_decreasing_in_T() -> None:
    """The gate's hypothesis predicts P(beats chalk) peaks at T > 1.
    The audit finds the opposite — P(beats chalk) decreases
    monotonically in T. A flip here is the signal that the gate
    should be reopened.

    Tolerance: allow tiny non-monotone wobbles (< 0.005) from sampling
    noise. Anything larger is a real shift."""
    data = json.loads(AUDIT_JSON.read_text())
    agg = data["aggregate"]

    # Sort by T ascending
    rows = sorted(agg.items(), key=lambda kv: float(kv[0]))
    rates = [row["mean_beats_chalk_rate"] for _, row in rows]
    Ts = [float(k) for k, _ in rows]

    for i in range(1, len(rates)):
        if rates[i] > rates[i - 1] + 0.005:
            pytest.fail(
                f"P(beats chalk) INCREASED from T={Ts[i - 1]:.2f} "
                f"({rates[i - 1]:.4f}) to T={Ts[i]:.2f} ({rates[i]:.4f}). "
                f"This would flip the O14 verdict — upset-over-estimation "
                f"(T>1) would now be helping pool rank. Re-audit."
            )

    # Specifically guard the recorded claim: pBeatChalk at T=0.5 exceeds
    # pBeatChalk at T=1.5 (the gate's hypothetical winning direction).
    def _rate(T):
        return agg[str(T)]["mean_beats_chalk_rate"]

    assert _rate(0.5) > _rate(1.5), (
        f"pBeatChalk at T=0.5 ({_rate(0.5):.4f}) must exceed T=1.5 "
        f"({_rate(1.5):.4f}) to preserve the O14 verdict. If this "
        f"inverts, the gate's 'upset-over-estimation helps pool rank' "
        f"hypothesis becomes plausible again — re-audit."
    )


def test_calibration_to_pool_effect_stays_small() -> None:
    """The best-score gap between T=1 (honest) and the T≠1 optimum
    is the magnitude of the 'interaction.' Recorded: ≤ 5 ESPN points.
    If this grows past 15 pts, the interaction is no longer 'small'
    and the decision to keep T=1 needs revisiting."""
    data = json.loads(AUDIT_JSON.read_text())
    agg = data["aggregate"]
    honest = agg["1.0"]["mean_best_score"]
    best_T_score = max(row["mean_best_score"] for row in agg.values())
    gap = best_T_score - honest
    assert gap < 15, (
        f"Best-score gap between T=1 ({honest}) and T≠1 optimum "
        f"({best_T_score}) is {gap:.1f} ESPN points. Recorded ≤ 5. "
        f"The calibration-to-pool interaction has grown meaningful — "
        f"re-audit before continuing to recommend T=1."
    )


def test_aggregate_numbers_within_tolerance_of_recorded() -> None:
    """Catch re-runs whose outputs drift materially from the recorded
    audit. Small numerical drift is fine (sampler seed vs platform
    differences); flag large shifts so the audit stays trustworthy."""
    data = json.loads(AUDIT_JSON.read_text())
    agg = data["aggregate"]

    # Recorded values for T=1.0 at commit time:
    #   mean_brier ≈ 0.1838, mean_best_score ≈ 909,
    #   mean_beats_chalk_rate ≈ 0.2238
    row = agg["1.0"]
    assert 0.18 <= row["mean_brier"] <= 0.20, (
        f"T=1 mean_brier = {row['mean_brier']:.4f}; recorded 0.1838. Drift > 0.02. Re-audit."
    )
    assert 880 <= row["mean_best_score"] <= 940, (
        f"T=1 mean_best_score = {row['mean_best_score']:.0f}; recorded 909. Drift > 30. Re-audit."
    )
    assert 0.19 <= row["mean_beats_chalk_rate"] <= 0.26, (
        f"T=1 pBeatChalk = {row['mean_beats_chalk_rate']:.4f}; recorded 0.2238. Drift > 0.03. Re-audit."
    )


def test_memory_dead_end_row_present() -> None:
    """MEMORY.md §2 must document the 'upset-over-estimation for pool
    rank' dead-end (D13). Removing it would invite future re-litigation."""
    memory = (REPO_ROOT / "MEMORY.md").read_text()
    assert "upset-over-estimation" in memory.lower() or "upset over-estimation" in memory.lower(), (
        "MEMORY.md §2 no longer documents the O14 dead-end. Future sessions may re-litigate. Restore the D13 row."
    )
