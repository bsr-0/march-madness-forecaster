"""The tournament_context_{year}.json files are backtest ground truth AND the
source of the live model's training rows (scripts/build_training_matrix.py).

scripts/audit_tournament_results.py checks four bracket invariants (A-D). It
was a manual tool until the 2026-09 methodology audit found a defect its
invariants A-C could not see: the 2025 file named San Diego State, the First
Four loser, in North Carolina's Round of 64 game. That put SDSU's features into
one of the 1,008 training rows and left the ground-truth walker to recover the
result through its per-team fallback. Invariant D was added for it, and this
test makes the auditor a gate rather than a script somebody remembers to run.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.audit_tournament_results import YEARS, audit_year  # noqa: E402


pytestmark = pytest.mark.data_contract


@pytest.mark.parametrize("year", YEARS)
def test_tournament_results_satisfy_bracket_invariants(year):
    issues = audit_year(year)
    assert not issues, f"{year}: " + "; ".join(issues)
