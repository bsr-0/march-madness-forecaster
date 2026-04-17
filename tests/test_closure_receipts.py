"""Closure-receipt drift guard for COUNCIL_LESSONS.md §2.

Imports the checker from ``scripts/check_closure_receipts.py`` and
asserts that the set of unverified closures does not grow beyond the
recorded baseline. Prevents future ``[closed YYYY-MM-DD]`` rows from
being added without citing at least one file under ``scripts/``,
``artifacts/``, or ``tests/`` that exists on disk.

Backstory: §2 O25's first G1 closure cited spread numbers
(0.835/0.315/0.045/0.310) that had no script, no artifact, and no test
on disk. It was caught only by accident on re-run. This guard makes
that defect class fail in CI instead of silently corrupting the
council's dedup index.

Update protocol:
- Adding to BASELINE_UNVERIFIED to silence a new failure is forbidden.
  Fix the closure to cite real evidence instead.
- When a baselined Ox is fixed (its closure now cites a real path),
  remove it from BASELINE_UNVERIFIED — the second test in this file
  will fail loudly until you do.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_closure_receipts import check_closure_receipts  # noqa: E402

# Pre-existing closures whose evidence lives outside scripts/artifacts/tests/.
# Each entry is grandfathered with an explicit reason — do NOT extend.
BASELINE_UNVERIFIED: set[str] = {
    # O1: closure cites pool_hist_results.json (repo-root data file, not under
    # an audited dir). Evidence is real and verifiable; just lives elsewhere.
    "O1",
    # O15: conditional-blocked closure (no historical Vegas data exists in the
    # repo). The closure is the formalization of the blocker — there is
    # legitimately nothing to serialize.
    "O15",
    # O21: closure cites ANALYSIS_O21_MARGINAL_BLEND.md (repo-root analysis
    # doc, not under an audited dir). Evidence is real and verifiable.
    "O21",
}


def test_no_new_unverified_closures() -> None:
    """Every new ``[closed YYYY-MM-DD]`` row must cite an existing file
    under scripts/, artifacts/, or tests/."""
    rows = check_closure_receipts()
    unverified_now = {r.ox_id for r in rows if not r.is_verified}
    new = unverified_now - BASELINE_UNVERIFIED
    assert not new, (
        f"New COUNCIL_LESSONS §2 closures lack receipts: {sorted(new)}. "
        "Each [closed] row must cite at least one path under scripts/, "
        "artifacts/, or tests/ that exists on disk. Run "
        "`python scripts/check_closure_receipts.py --report-only` for details."
    )


def test_baseline_entries_are_actually_unverified() -> None:
    """If a grandfathered Ox now cites real evidence, remove it from
    BASELINE_UNVERIFIED so the gate continues to ratchet forward."""
    rows = check_closure_receipts()
    unverified_now = {r.ox_id for r in rows if not r.is_verified}
    stale = BASELINE_UNVERIFIED - unverified_now
    assert not stale, (
        f"Baselined entries are no longer unverified: {sorted(stale)}. "
        "Remove them from BASELINE_UNVERIFIED in tests/test_closure_receipts.py."
    )
