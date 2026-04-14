"""Production-invariance gate for Tier 2/3 code cuts.

Context: the 2026-04-14 council verdict's Critical Action #2 flagged
that aggressive module deletion beyond Tier 1 needs an empirical
safety net — "run the frozen backtest pre/post each cut as empirical
verification, the only test that actually proves production is
unaffected."

This module IS that safety net. It:

1. Runs `scripts/backtest_by_round.py` (the 17-year × 1071-game
   round-segmented analysis whose structured output backs the O9
   closure in COUNCIL_LESSONS §2).
2. Computes a SHA-256 hash of the output.
3. Asserts the hash matches `artifacts/backtest_by_round_golden.txt`
   (the committed golden baseline).

If a code change alters the output by even one byte, this test
fires. That is the signal that a proposed module-deletion is NOT
a no-op on production behaviour and must be rolled back.

## When to run

- Before every Tier 2/3 deletion commit: run to establish baseline
  passes.
- Immediately after every Tier 2/3 deletion: run as the gate. Any
  failure → revert.
- In CI on every PR touching `src/` (paid for by the rare runtime
  cost — the script takes ~30-60s over committed historical JSON;
  fully offline).

## When NOT to run

- This test intentionally covers the script-level output, not every
  production invariant. It does NOT catch changes that happen to
  produce identical round-upset-rate tables but break a different
  downstream consumer. Pair with the closure-test suite
  (test_o9_round_decomposition, test_pool_optimizer_calibration,
  etc.) for broader coverage.

## If the golden hash needs to regenerate

If the upstream data files (`data/raw/historical/torvik_*.json`,
`historical_games_*.json`, `tournament_seeds_*.json`) legitimately
change, regenerate the golden with:

    python scripts/backtest_by_round.py > artifacts/backtest_by_round_golden.txt

and commit the new file. Document the data-source change in the
commit message. Do NOT regenerate to "fix" a test failure caused by
a code change — that defeats the purpose.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "artifacts" / "backtest_by_round_golden.txt"
SCRIPT = REPO_ROOT / "scripts" / "backtest_by_round.py"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.slow
def test_backtest_by_round_output_matches_golden() -> None:
    """Run `backtest_by_round.py` fresh and assert its stdout hashes
    match the committed golden artifact byte-for-byte. Tier 2/3 code
    cuts that pass this test have empirically demonstrated production-
    path invariance; cuts that fail it must be rolled back.

    Runtime: ~30-60s (offline — reads committed historical JSON only).
    Skip with `-m 'not slow'` for fast dev loops; run explicitly before
    every deletion commit."""
    assert GOLDEN.exists(), (
        f"Golden artifact missing at {GOLDEN}. Cannot gate code cuts "
        f"without a baseline. Regenerate with: "
        f"python scripts/backtest_by_round.py > artifacts/backtest_by_round_golden.txt"
    )
    assert SCRIPT.exists(), f"Script missing: {SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"backtest_by_round.py failed with code {result.returncode}. stderr tail: {result.stderr.decode()[-500:]}"
    )

    fresh_hash = _sha256(result.stdout)
    golden_hash = _sha256(GOLDEN.read_bytes())

    assert fresh_hash == golden_hash, (
        f"backtest_by_round output diverged from golden.\n"
        f"  golden SHA-256: {golden_hash}\n"
        f"  fresh  SHA-256: {fresh_hash}\n"
        f"\n"
        f"A recent code change altered the backtest output. Either:\n"
        f"  (a) roll back the change — it is NOT a no-op on production, or\n"
        f"  (b) if the change is intentional (feature add, bug fix, data "
        f"refresh), document it in the commit message and regenerate "
        f"the golden: "
        f"python scripts/backtest_by_round.py > {GOLDEN.relative_to(REPO_ROOT)}\n"
        f"\n"
        f"Do NOT blindly regenerate the golden to pass this test."
    )
