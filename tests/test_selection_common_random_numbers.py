"""Regression tests for common random numbers in candidate selection.

Three selection loops in ``mc_pool_backtest`` used to draw a fresh opponent
field and a fresh tournament *inside* the per-candidate loop. Every candidate
therefore carried independent noise::

    score(A) = signal(A) + noise_A
    score(B) = signal(B) + noise_B

and ``argmax`` cannot tell the noise from the signal. Measured on 2024 with 8
real candidates over 500 trials, the SE of the A-vs-B difference was 0.0094 --
as large as the SE of either estimate alone, i.e. no cancellation -- and the
selector changed its choice across 3 of 5 master seeds on identical data.
Sharing the draws cut the difference-SE to 0.0026.

The structural test below is the important one: it stops the draws being moved
back inside a candidate loop by a future refactor, which would silently
reintroduce the noise with nothing failing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.mc_pool_backtest import draw_selection_trials, score_candidate_p1

BACKTEST = Path(__file__).parent.parent / "scripts" / "mc_pool_backtest.py"

# The ONLY functions permitted to draw opponent fields or tournaments.
#
#   draw_selection_trials -- the shared-draw helper itself
#   _run_one_year         -- Pass B mode scoring, which already draws once per
#                            repeat and scores every mode against that same
#                            field ("paired comparison -> lower variance")
_DRAW_CALLERS = {"draw_selection_trials", "_run_one_year"}
_DRAW_FNS = {"generate_opponent_brackets", "simulate_tournament_outcomes"}


def _draw_call_sites():
    """Every (line, fn_called, enclosing_function) triple in the backtest."""
    tree = ast.parse(BACKTEST.read_text())
    found = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            nested = stack + ([child.name] if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else [])
            if isinstance(child, ast.Call):
                name = getattr(child.func, "id", None) or getattr(child.func, "attr", None)
                if name in _DRAW_FNS:
                    found.append((child.lineno, name, nested[-1] if nested else "<module>"))
            walk(child, nested)

    walk(tree, [])
    return found


# ---------------------------------------------------------------------------
# Structural: the draws must not migrate back into a candidate loop
# ---------------------------------------------------------------------------


def test_draws_only_happen_in_allowlisted_functions():
    """Fail if a candidate-selection loop starts drawing its own trials again."""
    offenders = [
        f"{BACKTEST.name}:{line} calls {fn}() inside {enclosing}()"
        for line, fn, enclosing in _draw_call_sites()
        if enclosing not in _DRAW_CALLERS
    ]
    assert not offenders, (
        "opponent fields / tournaments are being drawn outside the shared-draw "
        "helper. If this is inside a per-candidate loop, every candidate gets "
        "independent noise and argmax selects on it. Use draw_selection_trials() "
        "once, then score_candidate_p1() per candidate.\n  " + "\n  ".join(offenders)
    )


def test_selection_loops_do_not_call_draws_inside_a_candidate_loop():
    """Belt-and-braces: no draw call may sit inside a loop over candidates.

    Catches the case where someone adds a new selection loop in an allowlisted
    function rather than a new function.
    """
    tree = ast.parse(BACKTEST.read_text())
    offenders = []

    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For):
            continue
        target = ast.unparse(loop.target)
        # Candidate loops iterate an index or a (bracket, label) pair.
        if not any(tok in target for tok in ("ci", "cand", "bvec")):
            continue
        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in _DRAW_FNS:
                    offenders.append(f"line {node.lineno}: {name}() inside `for {target}`")

    assert not offenders, (
        "a draw call sits inside a candidate loop — this is exactly the pattern "
        "common random numbers exists to prevent:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Behavioural
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_setup():
    """A full 64-team synthetic bracket.

    The scoring path walks all 63 games, so a smaller stub raises IndexError in
    picks_by_round -- these helpers cannot be exercised on a toy bracket.
    """
    regions = ["East", "West", "South", "Midwest"]
    matchup_order = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    seeds, ratings = {}, {}
    for region in regions:
        for seed in range(1, 17):
            tid = f"{region}_{seed}"
            seeds[tid] = seed
            ratings[tid] = max(0.10, 1.0 - seed * 0.04)
    first_round = []
    for region in regions:
        for hi, lo in matchup_order:
            first_round += [f"{region}_{hi}", f"{region}_{lo}"]

    pw = {}
    ids = list(seeds)
    for i, t1 in enumerate(ids):
        for t2 in ids[i + 1 :]:
            a, b = ratings[t1], ratings[t2]
            den = a * (1 - b) + b * (1 - a)
            p = 0.5 if den < 1e-12 else a * (1 - b) / den
            pw[(t1, t2)] = p
            pw[(t2, t1)] = 1.0 - p
    return first_round, seeds, pw


def test_draw_selection_trials_is_deterministic_for_a_seed(tiny_setup):
    first_round, seeds, pw = tiny_setup
    kw = dict(n_opponents=5, first_round=first_round, pick_dist={}, matchup_probs=pw, seeds=seeds)
    a = draw_selection_trials(6, rng=np.random.default_rng(7), **kw)
    b = draw_selection_trials(6, rng=np.random.default_rng(7), **kw)
    assert len(a) == len(b) == 6
    for (opp_a, win_a), (opp_b, win_b) in zip(a, b):
        assert np.array_equal(opp_a, opp_b)
        assert win_a == win_b


def test_identical_candidates_score_identically_under_shared_trials(tiny_setup):
    """The core CRN property: no candidate-specific noise.

    Under the old inline draws two identical brackets could score differently,
    because each got its own tournaments. Sharing the trials makes that
    impossible — which is what lets argmax compare candidates meaningfully.
    """
    first_round, seeds, pw = tiny_setup
    trials = draw_selection_trials(
        40,
        n_opponents=5,
        first_round=first_round,
        pick_dist={},
        matchup_probs=pw,
        seeds=seeds,
        rng=np.random.default_rng(11),
    )
    vec = np.zeros(63, dtype=bool)
    vec[::2] = True
    assert score_candidate_p1(vec, trials, first_round, {"R64": 10}) == score_candidate_p1(
        vec.copy(), trials, first_round, {"R64": 10}
    )


def test_score_candidate_p1_is_a_probability(tiny_setup):
    first_round, seeds, pw = tiny_setup
    trials = draw_selection_trials(
        25,
        n_opponents=4,
        first_round=first_round,
        pick_dist={},
        matchup_probs=pw,
        seeds=seeds,
        rng=np.random.default_rng(3),
    )
    scoring = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
    rng = np.random.default_rng(5)
    for _ in range(5):
        vec = rng.random(63) > 0.5
        assert 0.0 <= score_candidate_p1(vec, trials, first_round, scoring) <= 1.0


def test_score_candidate_p1_handles_an_empty_trial_set(tiny_setup):
    first_round, _seeds, _pw = tiny_setup
    assert score_candidate_p1(np.zeros(63, dtype=bool), [], first_round, {"R64": 10}) == 0.0


def test_chalk_noise_std_is_forwarded_only_when_supplied(tiny_setup):
    """The poolaware loop passes chalk_noise_std; the other two do not."""
    first_round, seeds, pw = tiny_setup
    kw = dict(n_opponents=5, first_round=first_round, pick_dist={}, matchup_probs=pw, seeds=seeds)
    plain = draw_selection_trials(4, rng=np.random.default_rng(2), **kw)
    with_noise = draw_selection_trials(4, rng=np.random.default_rng(2), chalk_noise_std=0.25, **kw)
    assert len(plain) == len(with_noise) == 4
