"""One tie convention in the whole pipeline (2026-09 audit, Step 4, F4-1).

There used to be three. The same event -- our bracket tied with one opponent
for first -- was worth 1.0 to the selector (`>=`), 0.0 to the backtest table
(rank 1.5 != 1) and 0.5 to the prize column. Ties are ~7% of the events the
`>=` rule called wins, so the published P(1st) and the reported P(1st) erred
in opposite directions. The canonical quantity is now the expected
first-place SHARE (`payout.first_place_share`): 1 for an outright win,
1/(1+k) tied with k opponents, 0 otherwise. Selection, reporting, the CLI,
the referee audit and the artifact all use it; this file pins that.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.objective_diversity_matrix import pool_p_first
from scripts.mc_pool_backtest import ESPN_SCORING, score_candidate_p1
from src.optimization.payout import (
    TIE_SPLIT,
    first_place_share,
    first_place_share_from_counts,
    first_place_shares,
    payout_shares,
    prize_for_scores,
    probability_any_entry_wins,
)
from src.optimization.pool_objectives import TrialScores, p_first_from_scores

pytestmark = pytest.mark.unit


def test_share_definition():
    assert first_place_share(100, np.array([90, 80])) == 1.0
    assert first_place_share(100, np.array([100, 80])) == 0.5
    assert first_place_share(100, np.array([100, 100, 100])) == 0.25
    assert first_place_share(100, np.array([110, 100])) == 0.0
    assert first_place_share(100, np.array([])) == 1.0
    assert first_place_share_from_counts(0, 0) == 1.0
    assert first_place_share_from_counts(0, 1) == 0.5
    assert first_place_share_from_counts(2, 0) == 0.0
    np.testing.assert_allclose(first_place_shares(np.array([100, 100, 90, 110]), np.array([100, 80])), [0.5, 0.5, 0.0, 1.0])


def test_share_equals_winner_take_all_prize():
    shares = payout_shares("winner_take_all", 3)
    for our, opp in [(100, [90, 80]), (100, [100, 80]), (100, [100, 100]), (90, [100, 80])]:
        assert first_place_share(our, np.array(opp)) == pytest.approx(
            float(prize_for_scores(np.array([our]), np.array(opp, dtype=float), shares, tie_policy=TIE_SPLIT)[0])
        )


def test_selector_and_selection_scorer_agree_on_a_tie():
    # One trial where the candidate ties one opponent for the top score.
    teams = [f"t{i:02d}" for i in range(64)]
    cand = np.ones(63, dtype=bool)
    opp = np.stack([np.ones(63, dtype=bool), np.zeros(63, dtype=bool)])   # opponent 0 identical -> tie
    cur = list(teams)
    winners = {}
    for R in ("R64", "R32", "S16", "E8", "F4", "CHAMP"):
        nxt = [cur[g] for g in range(0, len(cur), 2)]
        winners[R] = set(nxt)
        cur = nxt
    trials = [(opp, winners)]
    assert score_candidate_p1(cand, trials, teams, ESPN_SCORING) == pytest.approx(0.5)
    assert pool_p_first(cand.reshape(1, 63), trials, teams)[0] == pytest.approx(0.5)
    ts = TrialScores(candidate=np.array([[1920.0]]), opponent=[np.array([1920.0, 0.0])])
    assert p_first_from_scores(ts, 0) == pytest.approx(0.5)
    assert probability_any_entry_wins(np.array([[1920.0]]), [np.array([1920.0, 0.0])]) == pytest.approx(0.5)


def test_portfolio_share_splits_among_our_own_entries_too():
    # Two of our entries tie each other for first with no opponent tied: the
    # portfolio collects the whole first place (0.5 + 0.5).
    assert probability_any_entry_wins(np.array([[100.0, 100.0]]), [np.array([90.0])]) == pytest.approx(1.0)
    # ...and only 2/3 of it if one opponent is tied with both.
    assert probability_any_entry_wins(np.array([[100.0, 100.0]]), [np.array([100.0])]) == pytest.approx(2 / 3)
