"""Three tie conventions in one pipeline (found building audit recommendation 13).

Adding a prize column made a pre-existing inconsistency visible. The same
event -- our bracket tied with one opponent for first place -- is worth three
different things depending on which part of the pipeline is asked:

    selecting   `score_candidate_p1`: `c_score >= opp_scores.max()`   -> 1.0
    reporting   `p_first = (all_ranks == 1.0).mean()`, where
                `all_ranks = better + 1 + tied/2` so a tie gives 1.5   -> 0.0
    paying      `_record_prize`, splitting the tied places             -> 0.5

Only the third is what a pool actually pays. The first two are both wrong and
wrong in opposite directions: selection over-rewards brackets that tie, and the
published P(1st) under-reports by discarding ties entirely.

Neither is changed here. `score_candidate_p1` defines the published ~12%
headline and `all_ranks` defines every number in the backtest's output tables;
silently re-defining either would move published figures as a side effect of
adding a feature. These tests pin the discrepancy so it is documented behaviour
rather than a latent surprise, and so anyone who later unifies the conventions
has to do it deliberately and re-run the headline.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.optimization.payout import TIE_SPLIT, TIE_WIN, payout_shares, prize_for_scores  # noqa: E402


def _selection_value(our: float, opp: np.ndarray) -> float:
    """`score_candidate_p1`'s inner test, isolated."""
    return 1.0 if our >= opp.max() else 0.0


def _evaluation_value(better: int, tied: int) -> float:
    """The backtest's `p_first` criterion, isolated."""
    all_ranks = better + 1 + tied / 2.0
    return 1.0 if all_ranks == 1.0 else 0.0


def _prize_value(better: int, tied: int, shares: np.ndarray) -> float:
    from scripts.mc_pool_backtest import _record_prize

    out = np.zeros((1, 1))
    _record_prize(out, 0, 0, better, tied, shares)
    return float(out[0, 0])


def test_the_three_conventions_disagree_on_a_shared_first():
    """The finding, pinned. If this ever passes by agreeing, say so loudly."""
    shares = payout_shares("winner_take_all", 3)
    our, opp = 100.0, np.array([100.0, 50.0])
    better, tied = 0, 1

    assert _selection_value(our, opp) == 1.0, "selection counts a tie as a full win"
    assert _evaluation_value(better, tied) == 0.0, "reporting counts a tie as no win at all"
    assert _prize_value(better, tied, shares) == pytest.approx(0.5), "a real pool splits it"


def test_all_three_agree_on_an_outright_win():
    """The inconsistency is confined to ties, which is why it went unnoticed."""
    shares = payout_shares("winner_take_all", 3)
    our, opp = 100.0, np.array([90.0, 50.0])
    better, tied = 0, 0

    assert _selection_value(our, opp) == 1.0
    assert _evaluation_value(better, tied) == 1.0
    assert _prize_value(better, tied, shares) == pytest.approx(1.0)


def test_all_three_agree_on_an_outright_loss():
    shares = payout_shares("winner_take_all", 3)
    our, opp = 10.0, np.array([90.0, 50.0])
    better, tied = 2, 0

    assert _selection_value(our, opp) == 0.0
    assert _evaluation_value(better, tied) == 0.0
    assert _prize_value(better, tied, shares) == pytest.approx(0.0)


@pytest.mark.parametrize("n_tied_opponents", [1, 2, 3, 5])
def test_prize_splits_evenly_however_many_are_tied(n_tied_opponents):
    """k+1 entrants sharing first each take 1/(k+1) of a winner-take-all pot."""
    shares = payout_shares("winner_take_all", n_tied_opponents + 2)
    value = _prize_value(0, n_tied_opponents, shares)
    assert value == pytest.approx(1.0 / (n_tied_opponents + 1))


def test_prize_split_across_paying_places_uses_their_mean():
    """A tie straddling the pay line splits what those places jointly pay.

    Two entrants tied for 3rd under top_3 share 3rd place's 10% and 4th
    place's nothing -- 5% each, not 10% each.
    """
    shares = payout_shares("top_3", 8)
    value = _prize_value(2, 1, shares)  # 2 opponents above, 1 tied with us
    assert value == pytest.approx((0.10 + 0.0) / 2)


def test_tie_win_and_tie_split_bracket_the_evaluation_convention():
    """Reporting is the most conservative of the three; selection the least.

    Useful framing for anyone reading the headline: the published P(1st)
    is a lower bound on "how often does this bracket share or take first".
    """
    shares = payout_shares("winner_take_all", 4)
    our, opp = 100.0, np.array([100.0, 80.0, 70.0])

    as_win = prize_for_scores(np.array([our]), opp, shares, TIE_WIN)[0]
    as_split = prize_for_scores(np.array([our]), opp, shares, TIE_SPLIT)[0]
    as_reported = _evaluation_value(0, 1)

    assert as_reported <= as_split <= as_win
