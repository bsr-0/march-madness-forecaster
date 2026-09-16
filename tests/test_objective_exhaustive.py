"""Exhaustive toy-tournament test of the objective pipeline (2026-09 audit, Step 4, items 6, 8, 9, 18).

A 64-team bracket in which every game is decided except K Round-of-64 games at
p = 1/2, so the outcome space has exactly 2**K equally likely tournaments and
EV / P(1st) can be computed by hand. The production quantities must match:

  * expected_scores()      == mean realised score over the enumerated bank;
  * pool_p_first() / score_candidate_p1()  == exact first-place share under the
    canonical tie rule (a tie for first is split among the tied entries);
  * the Monte Carlo bank (simulate_bracket_outcomes, no noise) converges to the
    same numbers.

Opponents are FIXED brackets here (the opponent model is audited separately):
this is a test of scoring, comparison and tie handling, with the answer known.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.experiments.build_candidate_artifact import _encode_rows  # noqa: E402
from scripts.experiments.conditional_bracket_engine import expected_scores, round_marginals  # noqa: E402
from scripts.experiments.objective_diversity_matrix import pool_p_first  # noqa: E402
from scripts.mc_pool_backtest import ESPN_SCORING, score_candidate_p1  # noqa: E402
from src.optimization.payout import first_place_share  # noqa: E402
from src.prediction.pairwise import PairwiseProbabilities, simulate_bracket_outcomes  # noqa: E402
from src.simulation.pool_competition import score_brackets_team_identity  # noqa: E402

pytestmark = pytest.mark.backtest_regression

TEAMS = [f"t{i:02d}" for i in range(64)]
PTS = [10, 20, 40, 80, 160, 320]
UNCERTAIN = (0, 1, 2, 3)  # R64 games 0..3 are coin flips; every other game: first-listed team wins


def _table():
    probs = {}
    for a in TEAMS:
        for b in TEAMS:
            if a != b:
                probs[(a, b)] = 0.5
    # deterministic games: the team with the lower index wins
    for i, a in enumerate(TEAMS):
        for j, b in enumerate(TEAMS):
            if i < j:
                probs[(a, b)], probs[(b, a)] = 1.0, 0.0
    for g in UNCERTAIN:
        a, b = TEAMS[2 * g], TEAMS[2 * g + 1]
        probs[(a, b)] = probs[(b, a)] = 0.5
    return PairwiseProbabilities.from_dict(probs, "toy")


def _walk(vec):
    cur = list(TEAMS)
    gi = 0
    rounds = []
    for _ in range(6):
        nxt = []
        for g in range(0, len(cur), 2):
            nxt.append(cur[g] if vec[gi] else cur[g + 1])
            gi += 1
        rounds.append(nxt)
        cur = nxt
    return rounds


def _outcome_vec(flips):
    """All games first-listed wins, except the uncertain R64 games given by `flips`.
    Later rounds: lower index always wins -- which is 'first-listed' in the walk
    EXCEPT when an R64 upset put a higher-index team in the top slot. Resolve by
    walking with the table rather than assuming."""
    pw = _table()
    vec = np.zeros(63, dtype=bool)
    cur = list(TEAMS)
    gi = 0
    for r in range(6):
        nxt = []
        for g in range(0, len(cur), 2):
            a, b = cur[g], cur[g + 1]
            if r == 0 and g // 2 in UNCERTAIN:
                w = b if flips[UNCERTAIN.index(g // 2)] else a
            else:
                w = a if pw.p(a, b) >= 0.5 else b
            vec[gi] = w == a
            nxt.append(w)
            gi += 1
        cur = nxt
    return vec


def _enumerate():
    outs = [_outcome_vec(f) for f in itertools.product((0, 1), repeat=len(UNCERTAIN))]
    return outs, [_walk(v) for v in outs]


def _score(vec, outcome_vec):
    w = {R: set(r) for R, r in zip(("R64", "R32", "S16", "E8", "F4", "CHAMP"), _walk(outcome_vec))}
    return float(score_brackets_team_identity(vec.reshape(1, 63), w, TEAMS, ESPN_SCORING)[0])


@pytest.fixture(scope="module")
def world():
    outs, rounds = _enumerate()
    rng = np.random.default_rng(0)
    cands = [outs[0].copy(), outs[-1].copy(), outs[5].copy()]           # three "perfect for one outcome" brackets
    cands.append(rng.random(63) < 0.5)                                   # and a random one
    opps = np.stack([outs[3], outs[10], outs[0], (rng.random(63) < 0.5)])  # fixed opponents; one duplicates cand 0
    return outs, rounds, cands, opps


def test_expected_score_equals_mean_realised_score(world):
    outs, rounds, cands, _ = world
    marg = round_marginals(rounds)
    for c in cands:
        ev = float(expected_scores([_walk(c)], marg, ESPN_SCORING)[0])
        exact = np.mean([_score(c, o) for o in outs])
        assert ev == pytest.approx(exact, abs=1e-9)


def _exact_first_share(c, outs, opps):
    tot = 0.0
    for o in outs:
        cs = _score(c, o)
        os_ = np.array([_score(x, o) for x in opps])
        tot += first_place_share(cs, os_)
    return tot / len(outs)


def test_first_place_share_exact_with_ties(world):
    """Candidate 0 IS opponent 2, so whenever it would win it ties -> share 1/2,
    never 1. The old '>=' rule would count those as full wins."""
    outs, _, cands, opps = world
    trials = [(opps, {R: set(r) for R, r in zip(("R64", "R32", "S16", "E8", "F4", "CHAMP"), _walk(o))}) for o in outs]
    rows = np.stack(cands)
    got = pool_p_first(rows, trials, TEAMS)
    for i, c in enumerate(cands):
        exact = _exact_first_share(c, outs, opps)
        assert got[i] == pytest.approx(exact, abs=1e-12), f"cand {i}: pool_p_first {got[i]} vs exact {exact}"
        assert score_candidate_p1(c, trials, TEAMS, ESPN_SCORING) == pytest.approx(exact, abs=1e-12)
    # the tie case is real: candidate 0 can tie but never solely win
    assert 0 < got[0] <= 0.5


def test_monte_carlo_bank_converges_to_exact(world):
    outs, rounds, cands, opps = world
    pw = _table()
    N = 20_000
    _, sim_rounds = simulate_bracket_outcomes(pw, TEAMS, N, np.random.default_rng(1), noise_std=0.0)
    # every simulated tournament must be one of the 16 enumerated ones
    keys = {tuple(tuple(r) for r in w) for w in rounds}
    assert all(tuple(tuple(r) for r in w) in keys for w in sim_rounds[:2000])
    marg = round_marginals(sim_rounds)
    for c in cands:
        ev_mc = float(expected_scores([_walk(c)], marg, ESPN_SCORING)[0])
        exact = np.mean([_score(c, o) for o in outs])
        assert abs(ev_mc - exact) < 8.0, f"EV MC {ev_mc} vs exact {exact}"   # ~4 SE at N=20k for a 320-pt swing
    trials = [(opps, {R: set(r) for R, r in zip(("R64", "R32", "S16", "E8", "F4", "CHAMP"), w)}) for w in sim_rounds[:5000]]
    got = pool_p_first(np.stack(cands), trials, TEAMS)
    for i, c in enumerate(cands):
        exact = _exact_first_share(c, outs, opps)
        assert abs(got[i] - exact) < 0.03, f"cand {i}: MC {got[i]} vs exact {exact}"


def test_objectives_are_not_proxies_for_each_other(world):
    """Item 13: build an artifact-shaped candidate list where EV and first-place
    share disagree, and check the product selector picks by the objective it is
    asked for, not by the other one."""
    from src.product.selection import select

    outs, rounds, cands, opps = world
    marg = round_marginals(rounds)
    trials = [(opps, {R: set(r) for R, r in zip(("R64", "R32", "S16", "E8", "F4", "CHAMP"), _walk(o))}) for o in outs]
    rows = np.stack(cands)
    p1 = pool_p_first(rows, trials, TEAMS)
    ev = expected_scores([_walk(c) for c in cands], marg, ESPN_SCORING)
    assert int(np.argmax(p1)) != int(np.argmax(ev)), "fixture must make the objectives disagree"
    artifact = {
        "teams": [{"id": t, "seed": (i % 16) + 1, "region": ["E", "W", "S", "M"][i // 16]} for i, t in enumerate(TEAMS)],
        "candidates": [{"w": [[TEAMS.index(x) for x in r] for r in _walk(c)], "ev": float(ev[i]), "p1": float(p1[i])} for i, c in enumerate(cands)],
    }
    assert select(artifact, "p1", "none", k=1)[0] == int(np.argmax(p1))
    assert select(artifact, "ev", "none", k=1)[0] == int(np.argmax(ev))
