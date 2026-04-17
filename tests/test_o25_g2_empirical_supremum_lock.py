"""Closure / lock test for COUNCIL_LESSONS §2 O25 G2 — empirical-supremum ranker.

The 2026-04-17 closure evaluated the last open sub-gate of O25: swap
the implicit IID-opponents ranker for the empirical supremum
``P(b_i > max(actual pool opponents))`` computed from
``pool_hist_results.json``.  The gate (COUNCIL_LESSONS.md §2 row 229)
required either per-year Spearman ρ(new_ranker, actual_scores) > 0.50
**or** the candidate with the highest actual score ranking top-5
under the new ranker.

Result: **null / dead-end.**  Across 4 years (2023-2026), per-year
Spearman ρ oscillates near zero (+0.37, −0.38, +0.27, −0.38), winner
brackets land at ranks {16, 32, 38, 41}, and the G2b O4-effect check
(upset-heavier brackets should rank higher under the empirical ranker
than under IID) fails at the mean-upset-contrast level.  The structural
prediction var(pool_max) > var(iid_max) holds directionally in 3 of 4
years but magnitudes are noisy.

Recorded as MEMORY.md §2 **D17**: switching from IID-opponent sampling
to empirical-pool-opponent sampling does not reliably improve within-
portfolio bracket ranking at the n=4-year evidence window.  The
ranker's errors live in *tournament-outcome uncertainty* (50-bracket
actual-score rankings are dominated by the realized-upset path), not
in the opponent-model's independence structure.

This test locks that null.  It fails if:
  - The artifact disappears or mutates schema.
  - ``overall_pass`` ever flips to True without a corresponding
    reopening of D17 (because that would mean the dead-end was
    premature — treat it as a finding, not a drift).
  - Per-year ρ values drift far outside the recorded band (suggests
    the methodology silently changed).

DO NOT relax these bounds.  A flipped ``overall_pass`` is a real
result that warrants reopening O25 G2 + D17, not weakening the test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "artifacts" / "o25_g2_empirical_supremum_2026-04-17.json"
SCRIPT = REPO_ROOT / "scripts" / "o25_g2_empirical_supremum.py"
POOL_DATA = REPO_ROOT / "pool_hist_results.json"
EXPECTED_YEARS = {"2023", "2024", "2025", "2026"}


def _load_artifact() -> dict:
    assert ARTIFACT.exists(), f"Missing {ARTIFACT}; re-run scripts/o25_g2_empirical_supremum.py."
    return json.loads(ARTIFACT.read_text())


def test_artifacts_present() -> None:
    assert SCRIPT.exists(), f"Missing {SCRIPT}; O25 G2 closure lost its executable."
    assert ARTIFACT.exists(), f"Missing {ARTIFACT}; re-run the G2 script."
    assert POOL_DATA.exists(), (
        f"Missing {POOL_DATA}; the source pool-brackets JSON is the basis "
        f"for the G2 test. If deleted, re-issue the closure."
    )


def test_schema_has_expected_fields() -> None:
    data = _load_artifact()
    for key in ("generated", "gate", "parameters", "per_year", "pooled", "gate_results"):
        assert key in data, f"G2 artifact missing top-level {key!r}"
    params = data["parameters"]
    for key in (
        "years",
        "n_candidates",
        "n_tourn",
        "noise_std",
        "rng_seed",
        "spearman_gate",
        "winner_top_k",
    ):
        assert key in params, f"G2 parameters missing {key!r}"
    assert set(str(y) for y in params["years"]) == EXPECTED_YEARS, (
        "G2 gate was evaluated on exactly the 4 years of real pool data. "
        "Changing the year set invalidates the D17 closure's power claim."
    )
    assert set(data["per_year"].keys()) == EXPECTED_YEARS, (
        f"G2 per_year keys {sorted(data['per_year'])} != {sorted(EXPECTED_YEARS)}."
    )


def test_parameters_locked() -> None:
    data = _load_artifact()
    params = data["parameters"]
    assert params["n_candidates"] == 50, "G2 locked at N_BRACKETS=50."
    assert params["n_tourn"] == 5000, "G2 locked at N_TOURN=5000 (§2 O5)."
    assert params["rng_seed"] == 42, "G2 reproducibility relies on RNG seed 42."
    assert math.isclose(params["spearman_gate"], 0.50), (
        "G2 council-mandated Spearman gate is 0.50; drifting this rewrites the closure."
    )
    assert params["winner_top_k"] == 5, "G2 council-mandated winner gate is top-5; drifting this rewrites the closure."


def test_overall_gate_failed() -> None:
    """G2 closed as dead-end D17. ``overall_pass`` must remain False.

    If this flips True, it means the empirical-supremum ranker started
    beating the gate — that's a **new finding**, not a test failure.
    Reopen O25 G2 and D17, document the shift, and re-run the audit
    before weakening this assertion.
    """
    data = _load_artifact()
    gate = data["gate_results"]
    assert gate["overall_pass"] is False, (
        "G2 ``overall_pass`` flipped True. This is a real result: the "
        "empirical-supremum ranker now clears the council's gate, which "
        "would reopen D17. Do not relax this test to hide the flip."
    )


def test_no_year_passed_g2a() -> None:
    """At closure, 0/4 years passed G2a. Locking the full-null record.

    If any single year starts passing (ρ>0.50 or winner top-5), that
    year-specific pass is worth investigating — may indicate a specific
    pool-year where empirical-opponent structure actually helps.
    """
    data = _load_artifact()
    passes = {y: row["g2a_pass"] for y, row in data["per_year"].items()}
    assert not any(passes.values()), (
        f"Some year now passes G2a: {[y for y, p in passes.items() if p]}. "
        f"Investigate whether the empirical-supremum ranker has genuinely "
        f"acquired signal in that pool, then reopen D17 if confirmed."
    )


def test_per_year_rho_within_recorded_band() -> None:
    """Recorded per-year ρ(new_ranker, actual_scores):
    2023=+0.371, 2024=−0.381, 2025=+0.274, 2026=−0.382.

    Wide tolerance (±0.30) because the sample (50 candidates × 1 actual
    outcome) is noisy year-to-year.  A drift beyond this band means the
    pool data, first_round derivation, or candidate generation silently
    shifted.
    """
    data = _load_artifact()
    recorded = {"2023": 0.371, "2024": -0.381, "2025": 0.274, "2026": -0.382}
    for year, expected in recorded.items():
        actual = data["per_year"][year]["rho_new_vs_actual"]
        assert abs(actual - expected) < 0.30, (
            f"Year {year}: ρ_new = {actual:+.3f}, recorded {expected:+.3f}. "
            f"Drift > 0.30 — the candidate set, opponents, or scoring changed. "
            f"Re-audit before accepting."
        )


def test_winner_never_reached_top_5() -> None:
    """Recorded winner ranks under new ranker: {16, 32, 38, 41} across
    4 years. Anything landing ≤ 5 is a distinctly different regime.
    """
    data = _load_artifact()
    top5_years = [(y, row["winner_rank_new"]) for y, row in data["per_year"].items() if row["winner_rank_new"] <= 5]
    assert not top5_years, (
        f"G2 winner-top-5 gate began passing in {top5_years}. Not drift — "
        f"a real result that reopens D17. Investigate before relaxing."
    )


def test_variance_ordering_partial() -> None:
    """O4 predicts var(pool_max) > var(iid_max) in every year. At
    closure, the directional check held in 3 of 4 years; 2024 reversed.
    Lock this mixed pattern so a future all-reverse result surfaces."""
    data = _load_artifact()
    wins = 0
    for row in data["per_year"].values():
        if row["var_pool_opp_max"] > row["var_iid_opp_max"]:
            wins += 1
    assert wins >= 2, (
        f"O4 variance prediction now holds in {wins}/4 years (recorded 3/4). "
        f"A collapse to < 2 suggests the pool opponents or IID sampler "
        f"methodology changed. Re-audit."
    )


def test_memory_records_d17() -> None:
    """MEMORY.md §2 must document D17 so future sessions don't retry
    the same null."""
    memory = (REPO_ROOT / "MEMORY.md").read_text()
    assert "D17" in memory, "MEMORY.md missing D17 row for O25 G2 dead-end."
    # Some phrasing indicating the "empirical supremum" closure must
    # appear near the D17 reference.
    assert "empirical supremum" in memory.lower() or "o25 g2" in memory.lower(), (
        "MEMORY.md D17 row doesn't reference the empirical-supremum closure. Restore the row."
    )


def test_council_lessons_records_closure() -> None:
    """COUNCIL_LESSONS §2 O25 row must reflect G2 closure."""
    council = (REPO_ROOT / "COUNCIL_LESSONS.md").read_text()
    assert "O25" in council, "COUNCIL_LESSONS.md missing O25 row."
    # The row should mention G2 being closed; any of the dead-end markers
    # below satisfies that (dead-end | D17 | 2026-04-17).
    assert any(
        marker in council for marker in ("G2 closed", "G2 dead-end", "D17", "2026-04-17 — dead-end", "G2 null")
    ), "COUNCIL_LESSONS.md O25 row doesn't mark G2 as closed. Update §2 row 229."
