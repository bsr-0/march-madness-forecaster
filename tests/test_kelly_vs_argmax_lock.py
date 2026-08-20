"""Closure / lock test for COUNCIL_LESSONS §2 O13 — Kelly-variance vs argmax.

Decision (from artifacts/o13_kelly_vs_argmax_audit_2026-04-14.md):
stochastic (variance-aware) construction beats deterministic argmax
in winner-take-all pools. Production locks `champ_first_tv` /
`f4_first_tv` / `e8_first_tv` as the recommended modes; the `det_*`
argmax modes stay in the harness for regression-test purposes but
must NOT be promoted to production.

This test fails if:
- The aggregated evidence JSON disappears, mutates, or its invariant
  inverts (stochastic BestRank ≥ argmax BestRank)
- `scripts/mc_pool_backtest.py` drops either the `det_*` or
  `*_first_tv` mode families (we need both present so a future
  rerun reproduces the comparison)
- The deprecated `opt_*` modes reappear in ALL_MODES

DO NOT relax a bound. If production moves to a different mode, author
a new dated audit artifact and re-issue the closure.

SCOPE REDUCED 2026-08-19. Two guards were retired because the documents
they read were deliberately deleted by commit 6393ef0 ("Consolidate
historical docs into FINDINGS.md"), not lost:

  - the ``artifacts/o13_kelly_vs_argmax_audit_2026-04-14.md`` existence
    check, and
  - the MEMORY.md §1 / §2 content checks (MEMORY.md no longer exists,
    and FINDINGS.md does not carry the strings they matched).

Restoring dead documents purely to satisfy assertions would be
cargo-culting a green suite. The substantive evidence — the machine-
readable ``o13_kelly_vs_argmax_2026-04-14.json`` and the code-level
mode-registry guards — is intact and still fully checked below, which
is where this closure's real protection lives.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_JSON = REPO_ROOT / "artifacts" / "o13_kelly_vs_argmax_2026-04-14.json"
MC_BACKTEST = REPO_ROOT / "scripts" / "mc_pool_backtest.py"


def test_audit_artifacts_present() -> None:
    """The machine-readable aggregate JSON must exist and be well-formed.

    The companion human-readable ``o13_..._audit_2026-04-14.md`` was
    deleted on purpose by commit 6393ef0; only the JSON is load-bearing
    for the invariants asserted in this file.
    """
    assert AUDIT_JSON.exists(), f"Missing {AUDIT_JSON}; restore from git."
    data = json.loads(AUDIT_JSON.read_text())
    assert "head_to_head" in data
    assert "modes" in data


def test_aggregated_head_to_head_invariant_holds() -> None:
    """Stochastic arm must have lower BestRank than argmax arm.

    This is the core Kelly-vs-argmax finding: despite argmax's higher
    peak P(1st), stochastic delivers consistent portfolio coverage so
    SOME bracket always lands near the top. The BestRank inequality is
    the durable signal; P(1st) aggregates are dominated by 2 spike
    years (2015, 2019) and swing."""
    data = json.loads(AUDIT_JSON.read_text())
    h2h = data["head_to_head"]
    stoch = h2h["stochastic"]
    argmax = h2h["argmax"]

    assert stoch["best_rank"] < argmax["best_rank"], (
        f"O13 invariant violated: stochastic BestRank "
        f"{stoch['best_rank']:.2f} >= argmax BestRank "
        f"{argmax['best_rank']:.2f}. If this flipped, a reinstatement "
        f"of argmax-as-production needs a fresh audit."
    )
    # And by a comfortable margin — the artifact documents 1.52 vs 9.92.
    # Allow slow drift but catch catastrophic narrowing.
    assert (argmax["best_rank"] - stoch["best_rank"]) > 2.0, (
        f"BestRank gap has narrowed to "
        f"{argmax['best_rank'] - stoch['best_rank']:.2f}. Expected >2.0 "
        f"per the 2026-04-12 MC run. Re-audit if the gap closes further."
    )


def test_mc_backtest_keeps_both_mode_families() -> None:
    """The det_* and *_first_tv families must both remain in
    ALL_MODES so a future audit can reproduce the comparison. Dropping
    either arm would make O13 un-verifiable from the harness."""
    source = MC_BACKTEST.read_text()
    for mode in (
        "det_champ_tv",
        "det_f4_tv",
        "det_e8_tv",
        "champ_first_tv",
        "f4_first_tv",
        "e8_first_tv",
    ):
        assert f'"{mode}"' in source or f"'{mode}'" in source, (
            f"mc_pool_backtest.py no longer references mode {mode!r}. "
            f"Both the det_* and *_first_tv families must stay so that "
            f"a future O13 audit can reproduce the head-to-head."
        )


def test_deprecated_opt_modes_stay_deprecated() -> None:
    """`opt_seed`/`opt_blend`/`opt_torvik` (P(1st)-optimizing Pareto-leverage
    modes) are a documented dead-end: a 13-year N=31 backtest put them at
    BestRank 62-88 against seed's 38 (p<0.05 Bonferroni). O13's verdict is
    that maximizing P(1st) directly is the wrong objective. Re-introducing
    them as production would re-open the same failure mode.

    The companion MEMORY.md §2 D6 prose check was retired with that file
    (commit 6393ef0); this code-level registry guard is the part that
    actually prevents the regression.
    """
    mc_source = MC_BACKTEST.read_text()
    # ALL_MODES should not list opt_seed/opt_blend/opt_torvik.
    all_modes_block_start = mc_source.find("ALL_MODES")
    if all_modes_block_start != -1:
        # Look at the next ~800 chars for the tuple body.
        block = mc_source[all_modes_block_start : all_modes_block_start + 800]
        for dead in ("opt_seed", "opt_blend", "opt_torvik"):
            assert f'"{dead}"' not in block, (
                f"Deprecated mode {dead!r} appears in ALL_MODES. It was "
                f"removed by §2 D6 after a 13-year N=31 backtest showed "
                f"BestRank 62-88 vs seed's 38 (p<0.05 Bonferroni)."
            )


def test_aggregate_p_first_numbers_remain_near_recorded() -> None:
    """Catch large drift in the aggregated P(1st) values. If someone
    regenerates the JSON with a different methodology, we want the
    numbers to stay near 0.04 (stoch) and 0.06 (argmax) — or a new
    audit justifying the shift."""
    data = json.loads(AUDIT_JSON.read_text())
    stoch_p = data["head_to_head"]["stochastic"]["p_first"]
    argmax_p = data["head_to_head"]["argmax"]["p_first"]

    # Recorded: stoch 0.0409, argmax 0.0603. Allow ±0.02 drift before
    # flagging as regeneration-without-audit.
    assert 0.02 <= stoch_p <= 0.07, (
        f"Stochastic aggregate P(1st) = {stoch_p:.4f}; audit recorded "
        f"0.0409 ± 0.02. Regenerate the audit if this legitimately "
        f"shifted."
    )
    assert 0.04 <= argmax_p <= 0.10, (
        f"Argmax aggregate P(1st) = {argmax_p:.4f}; audit recorded "
        f"0.0603 ± 0.02. Regenerate the audit if this legitimately "
        f"shifted."
    )


def test_per_year_stochastic_wins_majority() -> None:
    """Per-year winner breakdown should favor stochastic. Recorded:
    stoch 10 / argmax 3 / ties 0 of 13. A flip to argmax majority
    would invert the closure."""
    data = json.loads(AUDIT_JSON.read_text())
    winners = data["per_year_winner"]
    stoch_wins = winners["stoch_wins"]
    argmax_wins = winners["argmax_wins"]
    total = stoch_wins + argmax_wins + winners["ties"]

    assert total >= 10, f"Only {total} years in the comparison; expected ≥10."
    assert stoch_wins > argmax_wins, (
        f"Argmax is now winning the year-by-year comparison "
        f"({argmax_wins} vs {stoch_wins}). This inverts the O13 "
        f"closure — re-audit before continuing to recommend stochastic."
    )
    # And comfortably — require ≥ 2×.
    assert stoch_wins >= 2 * argmax_wins, (
        f"Per-year stochastic:argmax ratio is {stoch_wins}:{argmax_wins}; "
        f"expected stochastic ≥ 2× argmax. Re-audit if the gap narrows."
    )
