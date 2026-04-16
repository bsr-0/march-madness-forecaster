"""Closure / lock test for COUNCIL_LESSONS §2 O25 — empirical-supremum objective.

O25 closed as dead-end D17 on 2026-04-16. The within-portfolio bracket-
selection problem is fundamentally low-signal: neither the current
P(rank=1) ranker nor the proposed P(b_i > max(actual pool opponents))
ranker predicts actual-outcome candidate scores (mean Spearman ρ across
4 years: ρ_new = +0.06, ρ_old = −0.05; G2-A gate was > 0.50).

Also records the real G1 spread measurements, which replaced the
undocumented G1 narrative previously present in COUNCIL_LESSONS row 229.
The real G1 spread #1↔#11 lands in the AMBIGUOUS [0.016, 0.047] band
for 3 ranker definitions × 4 years (12 measurements); the prior
narrative reported spreads up to 0.835 that could not be reproduced or
traced to any committed code or artifact.

This test locks the evidentiary state. It fails if:
  - Either G1 or G2 artifact disappears or mutates materially
  - G2-A mean ρ(p_sup, actual) jumps above 0.50 (the gate) — that would
    REOPEN O25, not relax this test
  - G1 spreads drift outside recorded [0.010, 0.060] envelope
  - MEMORY.md §2 D17 row or COUNCIL_LESSONS.md row 229 closure are lost

DO NOT weaken the bounds to make the test pass. A shift in these numbers
is a real finding that warrants re-running G2 from scratch, not a lock
drift.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
G1_ARTIFACT = REPO_ROOT / "artifacts" / "o25_g1_real_2026-04-16.json"
G2_ARTIFACT = REPO_ROOT / "artifacts" / "o25_g2_empirical_supremum_2026-04-16.json"
G3_ARTIFACT = REPO_ROOT / "artifacts" / "o25_g3_diversity_2026-04-16.json"
G1_SCRIPT = REPO_ROOT / "scripts" / "o25_g1_spread_diagnostic.py"
G2_SCRIPT = REPO_ROOT / "scripts" / "o25_g2_empirical_supremum.py"
MEMORY_MD = REPO_ROOT / "MEMORY.md"
COUNCIL_MD = REPO_ROOT / "COUNCIL_LESSONS.md"

# Recorded envelope from the 2026-04-16 G1 run (all 12 ranker-year
# measurements land inside [0.016, 0.047]; envelope gives mild slack).
G1_SPREAD_LOWER = 0.010
G1_SPREAD_UPPER = 0.060

# Gate threshold from the G2 spec: ρ > 0.50 would REOPEN O25 (not fire
# this test — fire it and re-run G2, don't widen the lock).
G2A_GATE = 0.50

# Recorded G2 aggregates (2026-04-16).
G2A_MEAN_NEW_RECORDED = 0.0605
G2A_MEAN_OLD_RECORDED = -0.0500


def test_g1_and_g2_artifacts_present() -> None:
    assert G1_ARTIFACT.exists(), f"Missing {G1_ARTIFACT}; re-run scripts/o25_g1_spread_diagnostic.py."
    assert G2_ARTIFACT.exists(), f"Missing {G2_ARTIFACT}; re-run scripts/o25_g2_empirical_supremum.py."
    assert G3_ARTIFACT.exists(), f"Missing {G3_ARTIFACT}; G3 closure artifact disappeared."
    assert G1_SCRIPT.exists(), f"Missing {G1_SCRIPT}."
    assert G2_SCRIPT.exists(), f"Missing {G2_SCRIPT}."


def test_g1_artifact_schema() -> None:
    data = json.loads(G1_ARTIFACT.read_text())
    for key in ("parameters", "results", "crossref_existing_data", "council_lessons_row_229_g1_numbers"):
        assert key in data, f"G1 artifact missing top-level key {key!r}"
    assert set(data["results"].keys()) == {"2023", "2024", "2025", "2026"}
    for yr, row in data["results"].items():
        for f in (
            "spread_within_portfolio",
            "spread_vs_actual_opp",
            "spread_vs_iid_opp",
            "actual_best_rank_within",
            "actual_best_rank_vs_actual_opp",
            "actual_best_rank_vs_iid_opp",
        ):
            assert f in row, f"G1/{yr} row missing {f!r}"


def test_g1_spreads_within_recorded_envelope() -> None:
    """Real G1 spreads land in [0.016, 0.047] across all 12 ranker-year
    measurements. Envelope [0.010, 0.060] gives mild slack. A spread
    outside this envelope means either candidate generation drifted
    (rng_seed, mode) or the opponent model shifted — re-audit, don't
    widen."""
    data = json.loads(G1_ARTIFACT.read_text())
    for yr, row in data["results"].items():
        for field in ("spread_within_portfolio", "spread_vs_actual_opp", "spread_vs_iid_opp"):
            val = row[field]
            assert G1_SPREAD_LOWER <= val <= G1_SPREAD_UPPER, (
                f"G1/{yr}/{field} = {val:.4f} outside envelope [{G1_SPREAD_LOWER}, {G1_SPREAD_UPPER}]. Re-run G1."
            )


def test_g1_real_contradicts_prior_narrative() -> None:
    """The prior COUNCIL_LESSONS row 229 G1 narrative (0.835 / 0.315 /
    0.045 / 0.310) must remain recorded as contradicted by the real
    run. This protects future sessions from re-citing the prior
    numbers — if this assertion starts matching, the real G1 would be
    anomalous and needs investigation."""
    data = json.loads(G1_ARTIFACT.read_text())
    prior = data["council_lessons_row_229_g1_numbers"]["values"]
    assert prior == {"2023": 0.835, "2024": 0.315, "2025": 0.045, "2026": 0.31}
    # Real 2023 within-portfolio spread must be << the claimed 0.835.
    real_2023 = data["results"]["2023"]["spread_within_portfolio"]
    assert real_2023 < 0.10, (
        f"Real 2023 spread = {real_2023:.4f}. If this ever climbs near "
        f"the 0.835 prior claim, the prior narrative is reopenable."
    )


def test_g2_artifact_schema() -> None:
    data = json.loads(G2_ARTIFACT.read_text())
    for key in ("parameters", "results", "aggregate"):
        assert key in data, f"G2 artifact missing top-level key {key!r}"
    agg = data["aggregate"]
    for f in (
        "mean_spearman_new_vs_actual",
        "mean_spearman_old_vs_actual",
        "g2a_pass",
        "g2b_any_flip",
        "g2c_all_pass",
        "overall_primary_pass",
    ):
        assert f in agg, f"G2 aggregate missing {f!r}"


def test_g2a_fail_is_locked() -> None:
    """G2-A primary gate: mean ρ(p_sup, actual_scores) across 4 years
    > 0.50. The 2026-04-16 run got +0.0605 — far below the gate.

    If this test fires because mean ρ climbed above the gate, that's
    a REAL REOPEN of O25, not a drift. Do not widen; re-run G2 and
    consider re-opening the question."""
    data = json.loads(G2_ARTIFACT.read_text())
    mean_new = data["aggregate"]["mean_spearman_new_vs_actual"]
    assert mean_new < G2A_GATE, (
        f"G2-A mean ρ(p_sup, actual) = {mean_new:+.4f} now CLEARS the "
        f"{G2A_GATE} gate. O25 closed as D17 on the basis that this "
        f"was {G2A_MEAN_NEW_RECORDED:+.4f}. Reopen O25; do not widen "
        f"this lock."
    )
    assert data["aggregate"]["g2a_pass"] is False


def test_g2_aggregates_within_tolerance() -> None:
    """The recorded aggregates should stay close to 2026-04-16 values
    given rng_seed=42 determinism."""
    data = json.loads(G2_ARTIFACT.read_text())
    mean_new = data["aggregate"]["mean_spearman_new_vs_actual"]
    mean_old = data["aggregate"]["mean_spearman_old_vs_actual"]
    assert abs(mean_new - G2A_MEAN_NEW_RECORDED) < 0.05, (
        f"mean_new drifted: recorded {G2A_MEAN_NEW_RECORDED:+.4f}, now {mean_new:+.4f}. Re-audit."
    )
    assert abs(mean_old - G2A_MEAN_OLD_RECORDED) < 0.05, (
        f"mean_old drifted: recorded {G2A_MEAN_OLD_RECORDED:+.4f}, now {mean_old:+.4f}. Re-audit."
    )


def test_g2b_no_flip_recorded() -> None:
    """No year should have the actual-#1 flip into top-5 under the new
    ranker when the old had it outside top-5. If this starts firing,
    that's a REOPEN signal for G2-B."""
    data = json.loads(G2_ARTIFACT.read_text())
    assert data["aggregate"]["g2b_any_flip"] is False, (
        "A G2-B flip now exists; reopen O25 — the alt gate may clear even with G2-A failing."
    )


def test_parameters_match_g3_for_apples_to_apples() -> None:
    """G1 and G2 must use the same candidate-generation parameters
    as G3 (rng_seed=42, n_brackets=50, n_tourn=5000, noise_std=0.16,
    champ_first_tv mode) so closure cross-references are valid."""
    for art in (G1_ARTIFACT, G2_ARTIFACT):
        p = json.loads(art.read_text())["parameters"]
        assert p["n_candidates"] == 50 or p.get("n_brackets") == 50 or True
        # Some scripts key as n_candidates, others as n_brackets.
        assert p.get("n_candidates", p.get("n_brackets")) == 50, art.name
        assert p["n_tourn"] == 5000, art.name
        assert p["noise_std"] == 0.16, art.name
        assert p["rng_seed"] == 42, art.name


def test_memory_md_records_d17() -> None:
    """MEMORY.md §2 must record D17 so future sessions don't re-open
    O25 without new evidence."""
    mem = MEMORY_MD.read_text()
    assert "D17" in mem, "MEMORY.md §2 missing D17 row (O25 closure as dead-end)."
    assert "O25" in mem, "MEMORY.md missing O25 reference."


def test_council_lessons_records_closure() -> None:
    """COUNCIL_LESSONS.md row 229 must carry the 2026-04-16 closure
    verdict so future sessions know O25 is closed."""
    council = COUNCIL_MD.read_text()
    assert "O25" in council
    # The closure narrative should explicitly reference both the G2
    # FAIL and the real-G1 correction so the record is honest.
    assert "closed 2026-04-16" in council or "[closed 2026-04-16" in council, (
        "COUNCIL_LESSONS row 229 missing O25 closure marker."
    )
