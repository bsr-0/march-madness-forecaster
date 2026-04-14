"""Closure / lock test for COUNCIL_LESSONS §2 O4 — opponent independence.

The 2026-04-13 closure (ANALYSIS_O4_OPPONENT_CORRELATION.md) established:
across 4 years of real pool data (2023-2026, 93 brackets total), inter-
bracket correlation is consistently BELOW what independent draws from
the empirical marginals would produce (Stouffer pooled z = −4.15). The
independence assumption behind the 13-year backtest's opponent model
is safe — if anything, real pools are LESS clustered than the model
assumes.

This test locks that invariant. It fails if:
  - The serialized evidence artifact disappears or mutates
  - Pooled Stouffer z becomes ≥ 0 (brackets would be MORE clustered
    than IID, re-opening the council's validity concern)
  - Any per-year z flips far above 0 (specific year shows clustering)
  - The scripts or source data file moves without a re-audit

DO NOT relax a bound. A positive pooled z is a real methodology
problem that would invalidate every 13-year backtest result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_MD = REPO_ROOT / "ANALYSIS_O4_OPPONENT_CORRELATION.md"
AUDIT_JSON = REPO_ROOT / "artifacts" / "o4_opponent_independence_2026-04-14.json"
LOCK_SCRIPT = REPO_ROOT / "scripts" / "o4_independence_lock.py"
POOL_DATA = REPO_ROOT / "pool_hist_results.json"


def test_audit_artifacts_present() -> None:
    """Both the narrative audit (2026-04-13) and the re-serialized
    machine-readable JSON (2026-04-14) must stay on disk."""
    assert AUDIT_MD.exists(), f"Missing {AUDIT_MD}; O4 closure lost its narrative evidence."
    assert AUDIT_MD.stat().st_size > 3000, "AUDIT_MD suspiciously short."
    assert AUDIT_JSON.exists(), f"Missing {AUDIT_JSON}; re-run scripts/o4_independence_lock.py."
    assert LOCK_SCRIPT.exists(), f"Missing {LOCK_SCRIPT}."
    assert POOL_DATA.exists(), (
        f"Missing {POOL_DATA}; the source pool-brackets JSON is the "
        f"basis for the O4 test. If deleted, re-issue the closure."
    )


def test_json_schema_has_expected_fields() -> None:
    """The re-serialized artifact carries the fields the closure relies on."""
    data = json.loads(AUDIT_JSON.read_text())
    for key in ("per_year", "pooled_stouffer_z", "n_years", "n_sim_per_year"):
        assert key in data, f"o4 json missing {key!r}"

    assert data["n_years"] >= 4, (
        f"O4 closure relied on 4 years of pool history. Current artifact only has {data['n_years']}. Re-audit."
    )
    for yr, row in data["per_year"].items():
        for m in ("observed_mean", "simulated_mean", "excess", "z_score"):
            assert m in row, f"year {yr} row missing {m!r}"


def test_pooled_stouffer_z_is_negative() -> None:
    """The CORE O4 invariant: pooled z < 0 means bracket correlations
    are systematically LESS than IID draws from the marginals.

    If this flips positive, the council's validity concern is
    vindicated — every 13-year backtest rests on an unjustified
    independence assumption. That's a real finding worth a fresh
    audit, not a reason to weaken this test.

    Recorded value (2026-04-14 re-run): z = -4.15.
    """
    data = json.loads(AUDIT_JSON.read_text())
    z = data["pooled_stouffer_z"]
    assert z < 0, (
        f"Pooled Stouffer z = {z:+.2f}. The O4 independence-holds "
        f"verdict requires z < 0. A positive pooled z means brackets "
        f"are MORE correlated than IID from the marginals — a real "
        f"regression of the closure. Re-audit."
    )
    # And comfortably negative — the recorded -4.15 is ~4 sd below 0.
    # Allow substantial drift but catch a slide toward the boundary.
    assert z < -1.0, (
        f"Pooled Stouffer z = {z:+.2f} has weakened toward 0 (recorded -4.15). Re-audit if this climbs above -1.0."
    )


def test_no_year_shows_clustering_beyond_noise() -> None:
    """No individual year should show z > +1.96 (2-sd clustering).

    Per-year z values in 2026-04-14 run: −2.38, −2.18, −1.82, −1.91.
    All ≤ 0. Any single year hitting z > +1.96 is a clustering signal
    and should trigger investigation — the pool in that year might
    have had a demographic or chat-group effect that violated the
    independence assumption."""
    data = json.loads(AUDIT_JSON.read_text())
    offenders = [(yr, row["z_score"]) for yr, row in data["per_year"].items() if row["z_score"] > 1.96]
    assert not offenders, (
        f"Year(s) show 2-sd clustering beyond IID: {offenders}. The O4 "
        f"closure documented a consistent cross-year deficit of "
        f"correlation; a single-year inversion at |z|>1.96 is a "
        f"separate finding that reopens the validity concern for "
        f"that pool-year. Re-audit."
    )


def test_recorded_aggregate_within_tolerance() -> None:
    """Catch re-runs whose numbers drift materially from the 2026-04-14
    reference. Small numerical drift from RNG seed is fine (the script
    uses seed=42); large shifts suggest methodology changed.

    Recorded: z = -4.15 ± some, per-year observed ρ in [0.40, 0.47]."""
    data = json.loads(AUDIT_JSON.read_text())
    z = data["pooled_stouffer_z"]
    assert -6.0 < z < -2.0, (
        f"Pooled z = {z:+.2f}; recorded -4.15. Drift beyond [-6, -2] "
        f"suggests the analysis methodology or source data changed. "
        f"Re-audit."
    )

    # Per-year observed ρ sanity bounds.
    for yr, row in data["per_year"].items():
        obs = row["observed_mean"]
        assert 0.30 < obs < 0.55, (
            f"Year {yr} observed ρ = {obs:.4f}, outside [0.30, 0.55]. "
            f"Either the pool data changed or the correlation metric "
            f"drifted. Re-audit."
        )


def test_stouffer_computed_correctly() -> None:
    """The pooled_stouffer_z field must equal sum(z_i) / sqrt(n)
    computed over the per-year rows. If the definition shifts, the
    invariant above ceases to mean what the O4 closure claimed."""
    data = json.loads(AUDIT_JSON.read_text())
    per_year_zs = [row["z_score"] for row in data["per_year"].values()]
    expected = sum(per_year_zs) / math.sqrt(len(per_year_zs))
    actual = data["pooled_stouffer_z"]
    assert abs(expected - actual) < 0.01, (
        f"pooled_stouffer_z = {actual:+.4f} but sum(z_i)/sqrt(n) = {expected:+.4f}. The definition drifted."
    )


def test_memory_records_o4_closure() -> None:
    """MEMORY.md §1 Pool-strategy should document the O4 closure so
    future sessions don't re-litigate the independence question."""
    memory = (REPO_ROOT / "MEMORY.md").read_text()
    assert "O4" in memory, "MEMORY.md has no O4 reference."
    # Some phrasing indicating the independence-holds conclusion must
    # appear near the O4 reference.
    assert "independence" in memory.lower(), (
        "MEMORY.md doesn't document the independence-holds conclusion. Restore the O4 row in §1 Pool-strategy."
    )
