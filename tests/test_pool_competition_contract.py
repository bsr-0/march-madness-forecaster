"""simulate_tournament_outcomes is the referee simulator for every P(1st) in
the project. Its contract (2026-09 audit, Step 3, F3-3): it walks exactly 64
teams, it reads a probability for every game in either orientation, and it
never invents 0.5 for a pair it cannot find.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.simulation.pool_competition import simulate_tournament_outcomes  # noqa: E402

pytestmark = pytest.mark.unit


def _teams(n=64):
    return [f"t{i:02d}" for i in range(n)]


def _table(teams, rng, both=True):
    r = {t: float(np.clip(rng.beta(2, 2), 0.05, 0.95)) for t in teams}
    probs = {}
    for a in teams:
        for b in teams:
            if a < b:
                p = r[a] * (1 - r[b]) / (r[a] * (1 - r[b]) + r[b] * (1 - r[a]))
                probs[(a, b)] = p
                if both:
                    probs[(b, a)] = 1 - p
    return probs


def test_raises_on_non_64_bracket():
    rng = np.random.default_rng(0)
    teams = _teams(62)
    with pytest.raises(ValueError):
        simulate_tournament_outcomes(1, teams, _table(teams, rng), {}, 0.0, rng)


def test_raises_on_missing_pair_instead_of_defaulting_to_half():
    rng = np.random.default_rng(1)
    teams = _teams()
    probs = _table(teams, rng)
    del probs[("t00", "t01")], probs[("t01", "t00")]
    with pytest.raises(KeyError):
        simulate_tournament_outcomes(1, teams, probs, {}, 0.0, rng)


def test_reverse_orientation_is_used():
    rng = np.random.default_rng(2)
    teams = _teams()
    one_way = _table(teams, rng, both=False)          # only (a, b) with a < b
    both = {**one_way, **{(b, a): 1 - p for (a, b), p in one_way.items()}}
    o1, _ = simulate_tournament_outcomes(500, teams, one_way, {}, 0.0, np.random.default_rng(9))
    o2, _ = simulate_tournament_outcomes(500, teams, both, {}, 0.0, np.random.default_rng(9))
    assert np.array_equal(o1, o2)


@pytest.mark.backtest_regression
def test_reproduces_analytic_marginals_within_binomial_noise():
    from scipy.stats import binomtest

    rng = np.random.default_rng(3)
    teams = _teams()
    probs = _table(teams, rng)
    # positional recursion on the clipped table (the simulator caps at [0.01, 0.99])
    P = {k: min(0.99, max(0.01, v)) for k, v in probs.items()}
    reach = np.ones(64)
    marg = []
    for r in range(6):
        block, half = 2 ** (r + 1), 2 ** r
        win = np.zeros(64)
        for i, t in enumerate(teams):
            b0 = (i // block) * block
            mine = (i // half) * half
            opp0 = b0 if mine != b0 else b0 + half
            win[i] = reach[i] * sum(reach[j] * P[(t, teams[j])] for j in range(opp0, opp0 + half))
        marg.append(win)
        reach = win
    N = 50_000
    _, rounds = simulate_tournament_outcomes(N, teams, probs, {}, 0.0, np.random.default_rng(4))
    counts = np.zeros((64, 6))
    idx = {t: i for i, t in enumerate(teams)}
    for sr in rounds:
        for r in range(6):
            for w in sr[r]:
                counts[idx[w], r] += 1
    worst_p = 1.0
    for i in range(64):
        for r in range(6):
            p = marg[r][i]
            if 0 < p < 1:
                worst_p = min(worst_p, binomtest(int(counts[i, r]), N, p).pvalue)
    assert worst_p * 384 > 1e-3, f"systematic discrepancy: Bonferroni p={worst_p * 384:.2e}"
