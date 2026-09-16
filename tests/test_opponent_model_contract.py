"""Opponent-model contract (2026-09 audit, Step 6).

What the opponent machinery must do, pinned with answers that are known
without the production code: every generated opponent is a legal bracket; the
candidate and its opponents are scored on the SAME simulated tournament (a
pool shares one reality); P(1st) against synthetic pools with a known answer
(all-chalk, all-identical, uniform random, mixed) equals an independent
calculation exactly; and P(1st) is non-increasing in the number of opponents
when the extra opponents are added to the same field.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.experiments.objective_diversity_matrix import pool_p_first  # noqa: E402
from src.simulation.pool_competition import (  # noqa: E402
    ROUND_NAMES,
    generate_opponent_brackets,
    score_brackets_team_identity,
    simulate_tournament_outcomes,
)

pytestmark = pytest.mark.unit

TEAMS = [f"t{i:02d}" for i in range(64)]
SEEDS = {t: (i % 16) + 1 for i, t in enumerate(TEAMS)}
PTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


def _table():
    rng = np.random.default_rng(0)
    r = {t: float(np.clip(1.0 - 0.05 * SEEDS[t] + 0.02 * rng.random(), 0.05, 0.98)) for t in TEAMS}
    return {(a, b): r[a] * (1 - r[b]) / (r[a] * (1 - r[b]) + r[b] * (1 - r[a])) for a in TEAMS for b in TEAMS if a != b}


def _walk(vec):
    cur, gi, out = list(TEAMS), 0, []
    for _ in range(6):
        nxt = [cur[g] if vec[gi + g // 2] else cur[g + 1] for g in range(0, len(cur), 2)]
        gi += len(cur) // 2
        out.append(nxt)
        cur = nxt
    return out


def _pick_dist(rng):
    d = {}
    for t in TEAMS:
        base = max(0.02, 1.0 - 0.06 * SEEDS[t])
        d[t] = {R: base ** (i + 1) for i, R in enumerate(ROUND_NAMES)}
    return d


def test_generated_opponents_are_legal_brackets():
    opp = generate_opponent_brackets(500, TEAMS, _table(), _pick_dist(None), SEEDS, np.random.default_rng(1))
    assert opp.shape == (500, 63)
    for i in range(500):
        w = _walk(opp[i])
        assert [len(set(r)) for r in w] == [32, 16, 8, 4, 2, 1]
        assert all(set(w[r + 1]) <= set(w[r]) for r in range(5))


def _chalk():
    cur, v, gi = list(TEAMS), np.zeros(63, dtype=bool), 0
    for _ in range(6):
        nxt = []
        for g in range(0, len(cur), 2):
            a, b = cur[g], cur[g + 1]
            w = a if SEEDS[a] <= SEEDS[b] else b
            v[gi] = w == a
            nxt.append(w)
            gi += 1
        cur = nxt
    return v


def _sims(n, seed=100):
    out = []
    for i in range(n):
        _, br = simulate_tournament_outcomes(1, TEAMS, _table(), SEEDS, 0.0, np.random.default_rng(seed + i))
        out.append({R: set(br[0][r]) for r, R in enumerate(ROUND_NAMES)})
    return out


def _independent_share(cand, opp, sims):
    tot = 0.0
    for w in sims:
        c = score_brackets_team_identity(cand.reshape(1, 63), w, TEAMS, PTS)[0]
        o = score_brackets_team_identity(opp, w, TEAMS, PTS)
        top, k = o.max(), int((o == o.max()).sum())
        tot += 1.0 if c > top else (1.0 / (1 + k) if c == top else 0.0)
    return tot / len(sims)


@pytest.mark.parametrize("pool", ["all_chalk", "identical", "uniform_random", "mixed"])
def test_synthetic_pools_match_an_independent_calculation(pool):
    rng = np.random.default_rng(3)
    chalk = _chalk()
    cand = rng.random(63) < 0.7
    if pool == "all_chalk":
        opp = np.repeat(chalk.reshape(1, 63), 29, axis=0)
    elif pool == "identical":
        opp = np.repeat(cand.reshape(1, 63), 29, axis=0)
    elif pool == "uniform_random":
        opp = rng.random((29, 63)) < 0.5
    else:
        opp = np.vstack([np.repeat(chalk.reshape(1, 63), 10, axis=0), np.repeat(cand.reshape(1, 63), 5, axis=0), rng.random((14, 63)) < 0.5])
    sims = _sims(300)
    prod = pool_p_first(cand.reshape(1, 63), [(opp, w) for w in sims], TEAMS)[0]
    assert prod == pytest.approx(_independent_share(cand, opp, sims), abs=1e-12)
    if pool == "identical":
        assert prod == pytest.approx(1 / 30, abs=1e-12)  # tied with all 29 in every tournament


def test_p_first_is_non_increasing_in_pool_size_on_a_shared_field():
    rng = np.random.default_rng(5)
    field = generate_opponent_brackets(100, TEAMS, _table(), _pick_dist(None), SEEDS, rng)
    sims = _sims(200, seed=900)
    cand = _chalk()
    prev = 2.0
    for n in (1, 5, 10, 20, 50, 100):
        p = pool_p_first(cand.reshape(1, 63), [(field[:n], w) for w in sims], TEAMS)[0]
        assert p <= prev + 1e-12, f"P(1st) rose from {prev} to {p} when opponents were added"
        prev = p


def test_candidate_and_opponents_share_the_tournament_realisation():
    """draw_selection_trials returns (opponents, winners) pairs; the same winners
    score both sides. Scoring them on different realisations is a different
    quantity (measured 0.07 vs 0.20 on the 2026 bracket) and is not what a pool does."""
    from scripts.mc_pool_backtest import draw_selection_trials

    trials = draw_selection_trials(20, n_opponents=5, first_round=TEAMS, pick_dist=_pick_dist(None), matchup_probs=_table(), seeds=SEEDS, rng=np.random.default_rng(8))
    assert len(trials) == 20
    for opp, winners in trials:
        assert opp.shape == (5, 63)
        assert set(winners) == set(ROUND_NAMES) and len(winners["CHAMP"]) == 1
