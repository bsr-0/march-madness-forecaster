"""Stale-citation drift guard for MEMORY.md and COUNCIL_LESSONS.md.

Imports the checker from ``scripts/check_lesson_citations.py`` and
asserts no NEW broken citations appear beyond the recorded baseline.
Companion to ``tests/test_closure_receipts.py``: receipts prevent
narrative-only NEW closures, citations prevent existing closures from
rotting away when refactors move or delete the cited paths.

Backstory: MEMORY.md sits at the top of the source-of-truth precedence
(``pipeline_freeze.json`` > MEMORY > council > comments) and is dense
with paths. When `src/optimization/leverage.py` is refactored or an
artifact gets cleaned up, every MEMORY row pointing at the old path
silently rots. The next agent that tries to verify a "settled" claim
hits a dead reference and either re-derives the answer or, worse,
re-runs an experiment that's already settled.

Update protocol:
- Adding to BASELINE_BROKEN to silence a new failure is forbidden.
  Either restore the missing file, or update the citation in MEMORY.md
  / COUNCIL_LESSONS.md to point at where the evidence now lives.
- When a baselined citation is fixed (file restored or path updated),
  remove it from BASELINE_BROKEN — the second test in this file will
  fail loudly until you do.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_lesson_citations import check_citations  # noqa: E402

# Pre-existing broken citations grandfathered with explicit reasons.
# Format: cleaned path string (the regex matches one canonical form per
# unique path per source file). Do NOT extend this list to silence new
# breakage — fix the citation or restore the file instead.
BASELINE_BROKEN: set[str] = {
    # MEMORY.md §2 D4 cites the deleted learned-feature-selection module.
    # The dead-end verdict (fixed 7-feature set beat learned selection)
    # still holds; only the source file is gone. Update if the citation
    # is ever rewritten to point at the freeze flag or audit doc.
    "src/data/features/feature_selection.py",
    # MEMORY.md §2 D16 + COUNCIL_LESSONS.md §2 O23 cite a chalkfade
    # backtest log under artifacts/backtest_runs/ — the directory itself
    # no longer exists on disk. The dead-end is still valid (P(1st)
    # collapsed 14×) but the original log is gone. Restore from git
    # history or re-run if you need the raw numbers.
    "artifacts/backtest_runs/mc_pool_backtest_20260415_040656.txt",
}


def test_no_new_broken_citations() -> None:
    """Every NEW path token under scripts/ tests/ src/ artifacts/ data/
    configs/ in MEMORY.md / COUNCIL_LESSONS.md must exist on disk."""
    report = check_citations()
    broken_now = report.broken_paths
    new_breakage = broken_now - BASELINE_BROKEN
    assert not new_breakage, (
        f"New broken citations in MEMORY.md / COUNCIL_LESSONS.md: "
        f"{sorted(new_breakage)}. Fix the citation (point it at where "
        "the evidence actually lives) or restore the missing file. Run "
        "`python scripts/check_lesson_citations.py --report-only` for the "
        "full audit with line numbers."
    )


def test_baseline_entries_are_actually_broken() -> None:
    """If a baselined citation is fixed (file restored or path
    updated), remove it from BASELINE_BROKEN so the gate continues to
    ratchet forward."""
    report = check_citations()
    broken_now = report.broken_paths
    stale = BASELINE_BROKEN - broken_now
    assert not stale, (
        f"Baselined citations are no longer broken: {sorted(stale)}. "
        "Remove them from BASELINE_BROKEN in tests/test_lesson_citations.py."
    )
