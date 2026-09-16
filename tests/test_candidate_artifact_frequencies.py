"""The shipped "true" frequencies in the candidate artifact must be counted
over simulated tournaments only.

scripts/experiments/build_candidate_artifact.py appends the constructed and
shipped brackets to the simulated bank before sampling (so a user can filter
to them). Until the 2026-09 audit (Step 2, P2-1) the constraint, Final Four
and per-round frequencies were then counted over that appended list, so ~21
deterministic brackets were tallied as if they were draws from the outcome
model. Every shipped frequency should be k / n_sims for an integer k; with the
appended rows it was k / (n_sims + 21), which this test detects.

Runs a tiny build (300 tournaments) so it is fast; the property does not
depend on the count.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.experiments.build_candidate_artifact import build  # noqa: E402

pytestmark = pytest.mark.backtest_regression

N_SIMS = 300  # split evenly across the three rating sources -> 100 each


@pytest.fixture(scope="module")
def tiny_artifact():
    return build(year=2026, n_sims=N_SIMS, target=40, trials=8, seed=1)


def _is_multiple_of_one_over_n(p, n, places=5):
    # Frequencies are rounded to `places` decimals in the artifact.
    k = round(p * n)
    return abs(p - k / n) < 10 ** (-places) * 1.5


def test_team_round_frequencies_are_over_simulations_only(tiny_artifact):
    art = tiny_artifact
    assert art["meta"]["n_sims"] == N_SIMS
    bad = [
        (t["id"], stage, p)
        for t, row in zip(art["teams"], art["team_round_probabilities"])
        for stage, p in enumerate(row)
        if not _is_multiple_of_one_over_n(p, N_SIMS)
    ]
    assert not bad, f"frequencies not k/{N_SIMS}: {bad[:5]}"


def test_constraint_and_f4_frequencies_are_over_simulations_only(tiny_artifact):
    art = tiny_artifact
    bad = [(k, p) for k, p in art["constraint_probabilities"].items() if not _is_multiple_of_one_over_n(p, N_SIMS)]
    bad += [(k, p) for k, p in art["team_final_four_probabilities"].items() if not _is_multiple_of_one_over_n(p, N_SIMS)]
    assert not bad, f"frequencies not k/{N_SIMS}: {bad[:5]}"


def test_round_frequencies_are_bracket_coherent(tiny_artifact):
    # 32 R64 winners, 16 R32 winners, ... 1 champion per simulated tournament.
    art = tiny_artifact
    for stage, slots in enumerate((32, 16, 8, 4, 2, 1)):
        total = sum(row[stage] for row in art["team_round_probabilities"])
        assert total == pytest.approx(slots, abs=64 * 1e-5)
