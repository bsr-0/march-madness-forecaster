"""Closure / lock test for COUNCIL_LESSONS §2 O26-G2 — per-closure
team-identity audit of O21, O24, and D12/O13.

G2 ran three per-closure team-identity reruns on 2026-04-17. The
verdicts:

  G2-a (O21):  partially reopens. Team-identity recovers signal that
               shape flattened (per-weight mean ρ +0.63 to +0.77 vs
               shape's ~+0.45). Argmax best-weight flips from w=0.00
               (shape) to w=0.25 (team-identity), but the advantage
               over ESPN-only is modest (+0.058 ρ on n=3 yrs).

  G2-b (O24):  encoding-invariant FAIL. Per-year ρ(p_rank1, p_threshold)
               under team-identity = 0.31/0.16/0.34 (2023-2025), all
               well below the 0.95 gate. FAIL harder than shape.

  G2-c (D12):  encoding-invariant, strengthened. Stochastic sweeps
               14/14 years on BestRank under team-identity (shape:
               10/13). BestRank stoch 1.79 vs argmax 7.44.

This test locks each verdict. It fails if:
  - Any of the 3 G2 artifacts disappears or mutates materially.
  - G2-a's team-identity best-weight argmax moves off w=0.25.
  - G2-b's team-identity per-year ρ crosses the 0.95 gate (would
    REOPEN O24, not relax the test).
  - G2-c's team-identity BestRank direction flips (argmax < stochastic)
    or stochastic falls below 12/14 years on BestRank wins.
  - MEMORY.md §2 D12 or COUNCIL_LESSONS rows O21/O24/O26 lose their
    G2 amendments.

DO NOT weaken the bounds to make the test pass. A shift is a real
finding that warrants rerunning G2, not a lock drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
G2A_ARTIFACT = REPO_ROOT / "artifacts" / "o26_g2_o21_team_identity_2026-04-17.json"
G2B_ARTIFACT = REPO_ROOT / "artifacts" / "o26_g2_o24_team_identity_2026-04-17.json"
G2C_ARTIFACT = REPO_ROOT / "artifacts" / "o26_g2_d12_team_identity_2026-04-17.json"
G2A_SCRIPT = REPO_ROOT / "scripts" / "o26_g2_o21_team_identity.py"
G2B_SCRIPT = REPO_ROOT / "scripts" / "o26_g2_o24_team_identity.py"
G2C_SCRIPT = REPO_ROOT / "scripts" / "o26_g2_d12_team_identity.py"
G3_ARTIFACT = REPO_ROOT / "artifacts" / "o26_g3_n31_team_identity_2026-04-17.json"
G3_SCRIPT = REPO_ROOT / "scripts" / "o26_g3_mc_backtest_n31_team_identity.py"
MEMORY_MD = REPO_ROOT / "MEMORY.md"
COUNCIL_MD = REPO_ROOT / "COUNCIL_LESSONS.md"

# O24 reopen threshold — if team-identity ρ crosses 0.95 in any year,
# O24 reopens.
O24_RHO_GATE = 0.95

# D12 reopen criteria — argmax aggregate BestRank must remain ≥ stochastic
# AND stochastic must win at least 12 of 14 years. The team-identity
# result (14/14 wins; 7.44 vs 1.79) has margin against both thresholds.
D12_STOCH_MIN_WINS = 12
D12_ARGMAX_MIN_BESTRANK = 5.0  # argmax BestRank must stay worse (higher) than this
D12_STOCH_MAX_BESTRANK = 3.0  # stochastic BestRank must stay better than this


def _load(path: Path) -> dict:
    assert path.exists(), f"Required G2 artifact missing: {path}"
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------
# G2-a — O21 partial reopen
# ---------------------------------------------------------------------


def test_g2a_artifact_structure():
    d = _load(G2A_ARTIFACT)
    assert d["purpose"].startswith("O26-G2a")
    assert d["config"]["scoring"].startswith("team-identity")
    assert d["config"]["years"] == [2023, 2024, 2025]
    assert d["config"]["weights"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    agg = d["aggregate_team_identity"]
    for w in ("w=0.00", "w=0.25", "w=0.50", "w=0.75", "w=1.00"):
        assert w in agg, f"missing aggregate key {w}"
        assert agg[w]["n_years"] == 3


def test_g2a_argmax_best_weight_is_quarter():
    """Under team-identity, the argmax best blend weight is w=0.25.
    If this moves, O21 either re-locks to ESPN-only (shape-era) or
    opens a production blend-weight change — both material findings."""
    d = _load(G2A_ARTIFACT)
    agg = d["aggregate_team_identity"]
    best_w = max(agg, key=lambda k: agg[k]["mean_rho_team_identity"])
    assert best_w == "w=0.25", (
        f"Expected argmax weight w=0.25, got {best_w}. O21 verdict has moved — reopen §2 O21, do not relax this test."
    )


def test_g2a_team_identity_beats_shape_at_every_weight():
    """Every blend weight's team-identity ρ must exceed the shape baseline.
    The whole point of G2-a is that team-identity recovers signal shape
    masks; if this inverts, re-audit the O21 original artifact."""
    d = _load(G2A_ARTIFACT)
    cmp = d["comparison_to_shape_baseline"]
    assert cmp is not None, "G2-a artifact missing shape-baseline comparison"
    for w, row in cmp.items():
        assert row["delta_rho"] > 0.10, (
            f"{w}: team-identity should exceed shape by > 0.10 ρ, got Δ={row['delta_rho']:+.3f}. Re-audit G2-a."
        )


def test_g2a_no_weight_drops_below_0_5_rho_under_team_identity():
    """All 5 weights currently hold ρ ∈ [0.627, 0.767] under team-identity.
    A drop below 0.5 means the consistent-scoring signal has weakened —
    material finding."""
    d = _load(G2A_ARTIFACT)
    agg = d["aggregate_team_identity"]
    for w, row in agg.items():
        assert row["mean_rho_team_identity"] >= 0.50, (
            f"{w}: team-identity ρ dropped below 0.50 ({row['mean_rho_team_identity']:+.3f}). "
            f"Re-audit the G2-a opponent scorer."
        )


# ---------------------------------------------------------------------
# G2-b — O24 encoding-invariant FAIL
# ---------------------------------------------------------------------


def test_g2b_artifact_structure():
    d = _load(G2B_ARTIFACT)
    assert d["purpose"].startswith("O26-G2b")
    assert d["parameters"]["scoring"] == "team-identity"
    assert d["parameters"]["rho_gate"] == 0.95
    assert len(d["per_year"]) >= 3


def test_g2b_every_year_fails_rho_gate():
    """O24's 0.95 ρ gate must continue to fail under team-identity —
    if it ever passes, that REOPENS O24 (the opponent simulation is NOT
    load-bearing) and invalidates the D17 closure's claim of coverage."""
    d = _load(G2B_ARTIFACT)
    for yr in d["per_year"]:
        assert not yr["gate_rho_pass"], f"{yr['year']}: ρ={yr['spearman_rho']:.3f} >= {O24_RHO_GATE}. REOPEN §2 O24."
        assert yr["spearman_rho"] < 0.60, (
            f"{yr['year']}: ρ={yr['spearman_rho']:.3f} exceeds recorded [0.16, 0.34] envelope. "
            f"G2-b verdict may have shifted."
        )


def test_g2b_overall_fail():
    d = _load(G2B_ARTIFACT)
    assert not d["overall_pass"], "G2-b overall_pass flipped to True — REOPEN §2 O24."


# ---------------------------------------------------------------------
# G2-c — D12 encoding-invariant, strengthened
# ---------------------------------------------------------------------


def test_g2c_artifact_structure():
    d = _load(G2C_ARTIFACT)
    assert d["purpose"].startswith("O26-G2c")
    assert d["config"]["scoring"] == "team-identity"
    assert "champ_first_tv" in d["aggregate"]
    assert "det_champ_tv" in d["aggregate"]


def test_g2c_d12_direction_preserved():
    """Stochastic (`champ_first_tv`) must still dominate argmax on
    aggregate BestRank under team-identity. Direction flip would REOPEN
    D12 and potentially re-enable the `det_*` mode family."""
    d = _load(G2C_ARTIFACT)
    s = d["aggregate"]["champ_first_tv"]["best_rank_mean"]
    a = d["aggregate"]["det_champ_tv"]["best_rank_mean"]
    assert s < a, f"Stochastic BestRank {s:.2f} is not better than argmax's {a:.2f}. REOPEN §2 D12 (MEMORY.md)."
    assert s < D12_STOCH_MAX_BESTRANK, (
        f"Stochastic BestRank {s:.2f} exceeds {D12_STOCH_MAX_BESTRANK}. "
        f"Recorded value is 1.79 — large drift warrants reinvestigation."
    )
    assert a > D12_ARGMAX_MIN_BESTRANK, (
        f"Argmax BestRank {a:.2f} closer to stochastic than recorded 7.44. Re-audit G2-c before relaxing D12."
    )


def test_g2c_stochastic_sweeps_most_years():
    """Stochastic must win ≥ 12 of 14 years on BestRank. Team-identity
    recorded 14/14; shape baseline was 10/13. Drop below 12 means the
    "stochastic wins regardless of regime" narrative no longer holds."""
    d = _load(G2C_ARTIFACT)
    winners = d["best_rank_winners"]
    assert winners["stochastic_wins"] >= D12_STOCH_MIN_WINS, (
        f"Stochastic wins {winners['stochastic_wins']} / {winners['n_years']} years, "
        f"below threshold {D12_STOCH_MIN_WINS}. REOPEN §2 D12."
    )


# ---------------------------------------------------------------------
# Narrative locks — COUNCIL_LESSONS / MEMORY amendments
# ---------------------------------------------------------------------


def test_d12_memory_has_encoding_invariant_amendment():
    text = MEMORY_MD.read_text()
    assert "O26-G2c" in text, "MEMORY.md §2 D12 lost the O26-G2c amendment."
    assert "encoding-invariant" in text.lower(), "D12 amendment text mutated — relock or re-audit."


def test_council_lessons_o21_partially_reopened():
    text = COUNCIL_MD.read_text()
    assert "partially reopened 2026-04-17" in text, (
        "§2 O21 row lost the partial-reopen marker. G2-a verdict no longer recorded."
    )
    assert "O26-G2a" in text, "§2 O21 lost the O26-G2a reference."


def test_council_lessons_o24_encoding_invariant_marker():
    text = COUNCIL_MD.read_text()
    assert "encoding-invariant 2026-04-17" in text, (
        "§2 O24 row lost the encoding-invariant marker. G2-b verdict no longer recorded."
    )


def test_council_lessons_o26_status_advanced_to_g2_complete():
    text = COUNCIL_MD.read_text()
    # Match either the G2-era status ("G1+G2 complete") or any later
    # evolution that preserves the "G1+G2" prefix (e.g. "G1+G2+G3-narrow+
    # G4-companion complete"). What must never be lost: G1 and G2
    # landed and the 2026-04-17 date anchor.
    assert "G1+G2" in text and "2026-04-17" in text, (
        "§2 O26 status does not record G1+G2 completion with 2026-04-17 anchor."
    )


def test_g2_scripts_present():
    for p in (G2A_SCRIPT, G2B_SCRIPT, G2C_SCRIPT):
        assert p.exists(), f"Required G2 script missing: {p}"


# ---------------------------------------------------------------------
# G3 (narrow, N=31) — production MC backtest under team-identity
# ---------------------------------------------------------------------
#
# G3-narrow triggered the escalation gate: BestRank argmax flipped
# (f4_first_tv → torvik) and the top-3 ordering rearranged. User
# deferred N=1000 rerun to §2 O27. These locks hold the N=31 finding
# in place AND enforce that the P(1st) argmax stays preserved — the
# single invariant that keeps the `champ_first_tv` recommendation
# defensible until O27 closes.

# Recorded BestRank envelope from 2026-04-17 G3-narrow run.
# Per-mode aggregate BestRank (team-identity, N=31, 14 yrs, 50×50).
G3_BESTRANK_RECORDED = {
    "seed": 2.30,
    "torvik": 1.42,
    "champ_first_tv": 2.67,
    "champ_first_chalkfade_tv": 1.87,
    "f4_first_tv": 2.06,
    "e8_first_tv": 1.87,
    "det_champ_tv": 7.33,
    "det_f4_tv": 8.46,
    "det_e8_tv": 7.30,
}
G3_BESTRANK_TOL = 0.25  # ± tolerance per mode (tight but not brittle)

# Pre-committed invariant: P(1st) argmax among the 5 headline modes
# (seed/torvik/champ_first_tv/f4_first_tv/e8_first_tv) must remain
# `torvik`. If this flips, §2 O27 MUST be run before any mode
# recommendation moves.
G3_HEADLINE_MODES = ["seed", "torvik", "champ_first_tv", "f4_first_tv", "e8_first_tv"]
G3_PFIRST_ARGMAX_EXPECTED = "torvik"


def test_g3_artifact_structure():
    d = _load(G3_ARTIFACT)
    assert d["purpose"].startswith("O26-G3")
    assert d["config"]["scoring"] == "team-identity"
    assert d["config"]["n_opponents"] == 30
    assert set(d["aggregate"].keys()) >= set(G3_BESTRANK_RECORDED.keys())


def test_g3_bestrank_envelope():
    """Every mode's team-identity BestRank stays within ±0.25 of the
    recorded 2026-04-17 value. A drift wider than that is a methodology
    regression, not stochastic noise (14 yrs × 50×50 is enough to pin
    BestRank to ~0.1)."""
    d = _load(G3_ARTIFACT)
    agg = d["aggregate"]
    for mode, expected in G3_BESTRANK_RECORDED.items():
        assert mode in agg, f"G3 artifact missing mode {mode}"
        actual = agg[mode]["best_rank_mean"]
        assert abs(actual - expected) <= G3_BESTRANK_TOL, (
            f"{mode}: BestRank {actual:.2f} drifted outside envelope "
            f"{expected:.2f} ± {G3_BESTRANK_TOL}. Re-audit before relaxing."
        )


def test_g3_pfirst_argmax_preserved_torvik():
    """Among the 5 headline modes, torvik must remain P(1st) argmax
    under team-identity. If this flips, the WTA `champ_first_tv`
    recommendation in MEMORY §3 is no longer empirically supported —
    run §2 O27 immediately and do NOT weaken this assertion."""
    d = _load(G3_ARTIFACT)
    agg = d["aggregate"]
    headline_pfirst = {m: agg[m]["p_first_mean"] for m in G3_HEADLINE_MODES}
    argmax = max(headline_pfirst, key=lambda m: headline_pfirst[m])
    assert argmax == G3_PFIRST_ARGMAX_EXPECTED, (
        f"P(1st) argmax across 5 headline modes is {argmax}, expected "
        f"{G3_PFIRST_ARGMAX_EXPECTED}. Run §2 O27 before relaxing this test."
    )


def test_g3_bestrank_reorder_recorded():
    """G3-narrow's BestRank reorder is a durable finding. The 2026-04-17
    top-5 ordering (torvik, e8, f4/chalkfade, seed, champ_first) must
    remain recorded in the artifact; dropping it loses the escalation
    evidence that gave rise to O27."""
    d = _load(G3_ARTIFACT)
    assert d["best_argmax_flipped_vs_shape"] is True
    assert d["top3_bestrank_order_preserved"] is False
    # Confirm torvik is top-1 under team-identity
    assert d["team_bestrank_order"][0] == "torvik", (
        f"Team-identity BestRank top-1 is {d['team_bestrank_order'][0]}, "
        f"expected 'torvik'. Either G3-narrow regressed or drift."
    )
    # And f4_first_tv was top-1 under shape
    assert d["shape_bestrank_order"][0] == "f4_first_tv"


def test_g3_noseed_blend_flagged_as_deferred():
    """Document the sandbox-constrained scope: noseed + blend NOT run.
    §2 O27 must cover them when N=1000 rerun happens."""
    d = _load(G3_ARTIFACT)
    deferred = d["config"].get("modes_deferred_noml", [])
    assert "noseed" in deferred
    assert "blend" in deferred


def test_memory_md_has_team_identity_caveat():
    text = MEMORY_MD.read_text()
    assert "Team-identity caveat" in text, (
        "MEMORY.md §3 lost the team-identity caveat block. G3-narrow narrative "
        "no longer flags the N=1000 shape-encoded numbers as scope-limited."
    )
    assert "O26-G3-narrow" in text


def test_council_lessons_o27_exists():
    text = COUNCIL_MD.read_text()
    assert "**O27**" in text, "§2 O27 (G3 escalation to N=1000) is missing — G3 deferral not tracked."
    assert "G3 escalation" in text


def test_council_lessons_o26_status_advanced_to_g3_narrow():
    text = COUNCIL_MD.read_text()
    # Accept either the G3-landing phrasing or a later evolution that
    # still records G3-narrow completion (e.g. G4-companion also landing).
    assert ("G3 narrow (N=31) complete" in text) or ("G3-narrow" in text), (
        "§2 O26 status row does not record G3-narrow completion."
    )


def test_g3_script_present():
    assert G3_SCRIPT.exists(), f"G3 script missing: {G3_SCRIPT}"


# ---------------------------------------------------------------------
# G4 — team-identity companion landed in src/simulation/pool_competition.py
# ---------------------------------------------------------------------
#
# G4 adds a public team-identity scorer + helpers next to the shape
# scorer. Call-site migration of scripts/mc_pool_backtest.py is gated
# behind §2 O27 (N=1000 rerun) — touching it before O27 runs would
# leave MEMORY §3 stale-shape-encoded while the code produced
# team-identity. These tests assert the companion API landed and the
# shape function's docstring now flags the encoding explicitly.


def test_g4_companion_api_present():
    """G4 acceptance: the team-identity companion functions must be
    importable from src.simulation.pool_competition as public names."""
    from src.simulation import pool_competition as pc

    assert hasattr(pc, "score_brackets_team_identity"), (
        "G4: score_brackets_team_identity missing from pool_competition module."
    )
    assert hasattr(pc, "picks_by_round"), "G4: picks_by_round missing."
    assert hasattr(pc, "actual_winners_by_round"), "G4: actual_winners_by_round missing."


def test_g4_shape_docstring_flags_encoding():
    """G4 acceptance: the shape scorer's docstring must reference the
    team-identity companion so future readers don't re-trip the
    mixed-encoding bug class that G2-a found."""
    from src.simulation.pool_competition import score_brackets_against_outcome

    doc = score_brackets_against_outcome.__doc__ or ""
    assert "score_brackets_team_identity" in doc, (
        "G4: shape scorer docstring does not point to the team-identity companion."
    )
    assert "SHAPE ENCODING" in doc or "shape-encoded" in doc.lower(), (
        "G4: shape scorer docstring does not flag encoding explicitly."
    )
    assert "O26" in doc, "G4: shape scorer docstring does not reference §2 O26."


def test_g4_team_identity_scorer_signature():
    """Stable public signature so downstream callers can rely on it."""
    import inspect

    from src.simulation.pool_competition import score_brackets_team_identity

    sig = inspect.signature(score_brackets_team_identity)
    params = list(sig.parameters.keys())
    assert params == [
        "brackets",
        "winners_by_round",
        "first_round_matchups",
        "scoring_system",
    ], f"G4: unexpected signature {params}"


def test_g4_actual_winners_by_round_ncg_to_champ_contract():
    """The scorer expects 'CHAMP' keys; the winners-decoder must map
    the 'NCG' label used in tournament_results.json. Contract check."""
    from src.simulation.pool_competition import actual_winners_by_round

    games = [
        {"round_name": "NCG", "team1_id": "a", "team2_id": "b", "team1_won": True},
    ]
    w = actual_winners_by_round(games)
    assert "CHAMP" in w
    assert "NCG" not in w
