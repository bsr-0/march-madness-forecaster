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
G2_TEAM_ID_ARTIFACT = REPO_ROOT / "artifacts" / "o25_g2_team_identity_2026-04-16.json"
G3_ARTIFACT = REPO_ROOT / "artifacts" / "o25_g3_diversity_2026-04-16.json"
G1_SCRIPT = REPO_ROOT / "scripts" / "o25_g1_spread_diagnostic.py"
G2_SCRIPT = REPO_ROOT / "scripts" / "o25_g2_empirical_supremum.py"
G2_TEAM_ID_SCRIPT = REPO_ROOT / "scripts" / "o25_g2_team_identity.py"
MEMORY_MD = REPO_ROOT / "MEMORY.md"
COUNCIL_MD = REPO_ROOT / "COUNCIL_LESSONS.md"

# Recorded envelope from the 2026-04-16 G1 run (all 12 ranker-year
# measurements land inside [0.016, 0.047]; envelope gives mild slack).
G1_SPREAD_LOWER = 0.010
G1_SPREAD_UPPER = 0.060

# Gate threshold from the G2 spec: ρ > 0.50 would REOPEN O25 (not fire
# this test — fire it and re-run G2, don't widen the lock).
G2A_GATE = 0.50

# Recorded G2 shape-encoded aggregates (2026-04-16).
G2A_MEAN_NEW_SHAPE_RECORDED = 0.0605
G2A_MEAN_OLD_SHAPE_RECORDED = -0.0500

# Recorded G2 team-identity aggregates (2026-04-16). Real ESPN scoring —
# the scoring that actually determines pool payouts.
G2A_MEAN_NEW_TEAM_ID_RECORDED = 0.0086
G2A_MEAN_OLD_TEAM_ID_RECORDED = 0.0660


def test_g1_and_g2_artifacts_present() -> None:
    assert G1_ARTIFACT.exists(), f"Missing {G1_ARTIFACT}; re-run scripts/o25_g1_spread_diagnostic.py."
    assert G2_ARTIFACT.exists(), f"Missing {G2_ARTIFACT}; re-run scripts/o25_g2_empirical_supremum.py."
    assert G2_TEAM_ID_ARTIFACT.exists(), f"Missing {G2_TEAM_ID_ARTIFACT}; re-run scripts/o25_g2_team_identity.py."
    assert G3_ARTIFACT.exists(), f"Missing {G3_ARTIFACT}; G3 closure artifact disappeared."
    assert G1_SCRIPT.exists(), f"Missing {G1_SCRIPT}."
    assert G2_SCRIPT.exists(), f"Missing {G2_SCRIPT}."
    assert G2_TEAM_ID_SCRIPT.exists(), f"Missing {G2_TEAM_ID_SCRIPT}."


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


def test_g2a_fail_is_locked_shape() -> None:
    """G2-A primary gate under shape encoding: mean ρ(p_sup, actual) > 0.50.
    The 2026-04-16 run got +0.0605 — far below the gate.

    If this test fires because mean ρ climbed above the gate, that's a
    REAL REOPEN of O25, not a drift. Do not widen; re-run G2 and
    consider re-opening the question."""
    data = json.loads(G2_ARTIFACT.read_text())
    mean_new = data["aggregate"]["mean_spearman_new_vs_actual"]
    assert mean_new < G2A_GATE, (
        f"G2-A shape mean ρ(p_sup, actual) = {mean_new:+.4f} now CLEARS "
        f"the {G2A_GATE} gate. Reopen O25; do not widen this lock."
    )
    assert data["aggregate"]["g2a_pass"] is False


def test_g2a_fail_is_locked_team_identity() -> None:
    """G2-A primary gate under team-identity (real ESPN) scoring: mean
    ρ(p_sup, actual) > 0.50. The 2026-04-16 run got +0.0086. Team-
    identity is the scoring that actually determines pool payouts, so
    this is the load-bearing FAIL behind D17.

    Under team-identity, the OLD within-portfolio P(rank=1) baseline is
    marginally BETTER (+0.066) than the proposed supremum (+0.009).

    If team-identity mean ρ_new ever climbs above 0.50, that's a real
    REOPEN signal — do not widen."""
    data = json.loads(G2_TEAM_ID_ARTIFACT.read_text())
    mean_new = data["aggregate"]["mean_spearman_new_vs_actual"]
    assert mean_new < G2A_GATE, (
        f"G2-A team-identity mean ρ(p_sup, actual) = {mean_new:+.4f} "
        f"now CLEARS the {G2A_GATE} gate. Reopen O25; do not widen."
    )
    assert data["aggregate"]["g2a_pass"] is False


def test_g2_team_identity_baseline_beats_new() -> None:
    """Under team-identity scoring, the OLD within-portfolio P(rank=1)
    ranker has HIGHER mean ρ than the new empirical-supremum ranker
    (+0.066 vs +0.009). Locks the direction of the comparison — if the
    new ranker ever overtakes the old under team-identity, O25 reopens
    with a "maybe we were wrong" signal."""
    data = json.loads(G2_TEAM_ID_ARTIFACT.read_text())
    mean_new = data["aggregate"]["mean_spearman_new_vs_actual"]
    mean_old = data["aggregate"]["mean_spearman_old_vs_actual"]
    assert mean_old > mean_new, (
        f"Team-identity: p_rank1 baseline ρ = {mean_old:+.4f} is no "
        f"longer > p_sup ρ = {mean_new:+.4f}. D17 closed on the basis "
        f"that the new objective did NOT beat the old. Reopen O25."
    )


def test_g2_aggregates_within_tolerance() -> None:
    """The recorded aggregates should stay close to 2026-04-16 values
    given rng_seed=42 determinism. Locks both encodings."""
    shape = json.loads(G2_ARTIFACT.read_text())["aggregate"]
    team = json.loads(G2_TEAM_ID_ARTIFACT.read_text())["aggregate"]
    assert abs(shape["mean_spearman_new_vs_actual"] - G2A_MEAN_NEW_SHAPE_RECORDED) < 0.05, (
        f"shape mean_new drifted: recorded {G2A_MEAN_NEW_SHAPE_RECORDED:+.4f}, "
        f"now {shape['mean_spearman_new_vs_actual']:+.4f}. Re-audit."
    )
    assert abs(shape["mean_spearman_old_vs_actual"] - G2A_MEAN_OLD_SHAPE_RECORDED) < 0.05, (
        f"shape mean_old drifted: recorded {G2A_MEAN_OLD_SHAPE_RECORDED:+.4f}, "
        f"now {shape['mean_spearman_old_vs_actual']:+.4f}. Re-audit."
    )
    assert abs(team["mean_spearman_new_vs_actual"] - G2A_MEAN_NEW_TEAM_ID_RECORDED) < 0.05, (
        f"team-identity mean_new drifted: recorded "
        f"{G2A_MEAN_NEW_TEAM_ID_RECORDED:+.4f}, now "
        f"{team['mean_spearman_new_vs_actual']:+.4f}. Re-audit."
    )
    assert abs(team["mean_spearman_old_vs_actual"] - G2A_MEAN_OLD_TEAM_ID_RECORDED) < 0.05, (
        f"team-identity mean_old drifted: recorded "
        f"{G2A_MEAN_OLD_TEAM_ID_RECORDED:+.4f}, now "
        f"{team['mean_spearman_old_vs_actual']:+.4f}. Re-audit."
    )


def test_g2b_no_flip_recorded() -> None:
    """No year should have the actual-#1 flip into top-5 under the new
    ranker when the old had it outside top-5, under either encoding.
    If this starts firing, that's a REOPEN signal for G2-B."""
    for label, art in (("shape", G2_ARTIFACT), ("team-identity", G2_TEAM_ID_ARTIFACT)):
        data = json.loads(art.read_text())
        assert data["aggregate"]["g2b_any_flip"] is False, (
            f"{label}: A G2-B flip now exists; reopen O25 — the alt gate may clear even with G2-A failing."
        )


def test_coverage_under_team_identity_holds() -> None:
    """Under team-identity scoring, the 50-bracket portfolio contains a
    bracket that beats the stored pool winner pts in ALL 4 years. This
    is recorded as the reason D17 says 'selection, not generation, is
    the limiting factor'. If this fails, regenerate candidates are no
    longer a superset of pool-winning brackets — that's a generation
    capability regression, not a lock drift."""
    data = json.loads(G2_TEAM_ID_ARTIFACT.read_text())
    assert data["aggregate"]["coverage_all_years"] is True
    for yr, row in data["results"].items():
        assert row["coverage_beats_stored"], (
            f"Year {yr}: cand_best={row['actual_best_score']} no "
            f"longer beats stored_pool_max={row['stored_pool_max_pts']}. "
            f"Generation regression — investigate candidate sampler."
        )


def test_parameters_match_g3_for_apples_to_apples() -> None:
    """G1 and G2 (both encodings) must use the same candidate-generation
    parameters as G3 (rng_seed=42, n_brackets=50, n_tourn=5000,
    noise_std=0.16, champ_first_tv mode) so closure cross-references
    are valid."""
    for art in (G1_ARTIFACT, G2_ARTIFACT, G2_TEAM_ID_ARTIFACT):
        p = json.loads(art.read_text())["parameters"]
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
