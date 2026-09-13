"""Prize-weighted objectives and multi-entry portfolios (audit recommendation 13, parts 3-4).

The load-bearing test here is `test_winner_take_all_reproduces_the_incumbent_p1`:
the published ~12% headline is computed by `score_candidate_p1`, and adding
payout support must not move it by so much as a float. Everything else checks
that the new objectives mean what they claim -- particularly that a portfolio's
entries compete with each other, which is the difference between a multi-entry
model and wishful thinking.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.optimization.payout import (  # noqa: E402
    TIE_SPLIT,
    TIE_WIN,
    VALID_PAYOUT_STRUCTURES,
    expected_portfolio_prize,
    payout_shares,
    prize_for_scores,
    probability_any_entry_wins,
    resolve_pool_size,
)


# ---------------------------------------------------------------------------
# Share vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structure", sorted(VALID_PAYOUT_STRUCTURES))
def test_every_structure_sums_to_one_and_fits_the_pool(structure):
    shares = payout_shares(structure, 30)
    assert shares.shape == (30,)
    assert shares.sum() == pytest.approx(1.0)
    assert np.all(shares >= 0)
    assert shares[0] > 0, "rank 1 must always be paid"


def test_winner_take_all_pays_only_first():
    shares = payout_shares("winner_take_all", 20)
    assert shares[0] == pytest.approx(1.0)
    assert shares[1:].sum() == pytest.approx(0.0)


def test_top_3_pays_three_places_in_descending_order():
    shares = payout_shares("top_3", 30)
    assert np.count_nonzero(shares) == 3
    assert shares[0] > shares[1] > shares[2] > 0


def test_percentile_structures_scale_with_pool_size():
    """top_25pct pays more places in a bigger pool -- the point of the structure."""
    small = payout_shares("top_25pct", 8)
    large = payout_shares("top_25pct", 40)
    assert np.count_nonzero(small) == 2
    assert np.count_nonzero(large) == 10
    # Equal split among the paying places.
    assert small[0] == pytest.approx(small[1])


def test_a_pool_cannot_pay_more_places_than_it_has_entrants():
    with pytest.raises(ValueError, match="cannot pay more places"):
        payout_shares("top_5", 3)


def test_custom_shares_are_renormalised_so_dollar_amounts_work():
    shares = payout_shares("custom", 10, custom_shares=[50, 30, 20])
    assert shares[:3] == pytest.approx([0.5, 0.3, 0.2])
    assert shares.sum() == pytest.approx(1.0)


def test_custom_shares_reject_nonsense():
    with pytest.raises(ValueError, match="negative"):
        payout_shares("custom", 10, custom_shares=[1.0, -0.5])
    with pytest.raises(ValueError, match="sums to zero"):
        payout_shares("custom", 10, custom_shares=[0.0, 0.0])


def test_unknown_structure_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown payout structure"):
        payout_shares("top_1_billion", 30)


def test_resolve_pool_size_counts_our_entries_too():
    """The off-by-one audit finding H6 turned on, written once."""
    assert resolve_pool_size(29, 1) == 30
    assert resolve_pool_size(29, 3) == 32, "3 entries against 29 opponents is a 32-person pool"
    with pytest.raises(ValueError):
        resolve_pool_size(29, 0)


# ---------------------------------------------------------------------------
# Prize attribution
# ---------------------------------------------------------------------------


def test_outright_win_takes_the_whole_pot_under_winner_take_all():
    shares = payout_shares("winner_take_all", 4)
    prize = prize_for_scores(np.array([100.0]), np.array([90.0, 80.0, 70.0]), shares)
    assert prize[0] == pytest.approx(1.0)


def test_a_shared_first_splits_the_prize():
    """What a real pool does, and what the incumbent P(1st) does NOT do."""
    shares = payout_shares("winner_take_all", 4)
    tied = prize_for_scores(np.array([100.0]), np.array([100.0, 80.0, 70.0]), shares, TIE_SPLIT)
    assert tied[0] == pytest.approx(0.5)

    as_win = prize_for_scores(np.array([100.0]), np.array([100.0, 80.0, 70.0]), shares, TIE_WIN)
    assert as_win[0] == pytest.approx(1.0), "TIE_WIN exists to reproduce the incumbent convention"


def test_finishing_out_of_the_money_pays_nothing():
    shares = payout_shares("top_3", 10)
    prize = prize_for_scores(np.array([10.0]), np.array([90.0, 80.0, 70.0, 60.0, 50.0]), shares)
    assert prize[0] == pytest.approx(0.0)


def test_our_own_entries_compete_with_each_other():
    """Two entries beating the field take 1st AND 2nd, not 1st twice.

    Getting this wrong is the difference between a portfolio model and
    double-counting: under top_3 the pair collects 0.60 + 0.30, not 1.20.
    """
    shares = payout_shares("top_3", 5)
    prizes = prize_for_scores(np.array([100.0, 95.0]), np.array([80.0, 70.0, 60.0]), shares)
    assert prizes[0] == pytest.approx(0.60)
    assert prizes[1] == pytest.approx(0.30)
    assert prizes.sum() == pytest.approx(0.90)


def test_tie_between_our_own_two_entries_splits_between_them():
    shares = payout_shares("top_3", 5)
    prizes = prize_for_scores(np.array([100.0, 100.0]), np.array([80.0, 70.0, 60.0]), shares)
    # Both sit in a 2-way tie for places 1-2, so each gets (0.60 + 0.30) / 2.
    assert prizes[0] == pytest.approx(0.45)
    assert prizes[1] == pytest.approx(0.45)
    assert prizes.sum() == pytest.approx(0.90), "the pair still collects exactly 1st + 2nd"


def test_total_payout_never_exceeds_the_pot():
    """Across random fields, our entries plus everyone else's cannot exceed 1."""
    rng = np.random.default_rng(3)
    shares = payout_shares("top_5", 12)
    for _ in range(200):
        field = rng.integers(0, 500, size=12).astype(float)
        ours, opps = field[:3], field[3:]
        total = prize_for_scores(ours, opps, shares).sum()
        assert total <= 1.0 + 1e-9


def test_unknown_tie_policy_raises():
    shares = payout_shares("winner_take_all", 3)
    with pytest.raises(ValueError, match="tie_policy"):
        prize_for_scores(np.array([1.0]), np.array([0.0, 0.0]), shares, "whatever")


# ---------------------------------------------------------------------------
# Portfolio objectives
# ---------------------------------------------------------------------------


def test_a_second_entry_cannot_reduce_expected_prize():
    """Monotonicity: adding an entry can only add prize, never subtract.

    True because our entries displace OPPONENTS, never each other's money --
    the pool size is held fixed by the caller drawing fewer opponents.
    """
    rng = np.random.default_rng(5)
    shares = payout_shares("top_3", 10)
    opp = [rng.integers(0, 400, size=8).astype(float) for _ in range(100)]
    one = np.array([[rng.integers(0, 400)] for _ in range(100)], dtype=float)
    two = np.hstack([one, np.array([[rng.integers(0, 400)] for _ in range(100)], dtype=float)])

    # Same opponent count on both sides, so this isolates the extra entry.
    opp7 = [o[:7] for o in opp]
    e1 = expected_portfolio_prize(one, opp7, shares)
    e2 = expected_portfolio_prize(two, opp7, shares)
    assert e2 >= e1 - 1e-12


def test_duplicate_entries_add_nothing_beyond_the_tie_split():
    """Entering the same bracket twice is not a strategy.

    Two identical entries always tie with each other, so they split whatever
    places they jointly occupy -- the portfolio search must see no gain worth
    having here, which is what makes greedy selection pick DIVERSE brackets.
    """
    shares = payout_shares("winner_take_all", 6)
    scores = np.array([[100.0, 100.0]] * 50)
    opp = [np.array([50.0, 40.0, 30.0, 20.0]) for _ in range(50)]
    doubled = expected_portfolio_prize(scores, opp, shares)

    single = expected_portfolio_prize(np.array([[100.0]] * 50), [o[:4] for o in opp], shares)
    assert doubled == pytest.approx(single), "a duplicated entry wins the same pot, split two ways"


def test_probability_any_entry_wins_is_at_least_the_best_single():
    rng = np.random.default_rng(9)
    scores = rng.integers(0, 400, size=(200, 3)).astype(float)
    opp = [rng.integers(0, 400, size=10).astype(float) for _ in range(200)]

    any_win = probability_any_entry_wins(scores, opp)
    for col in range(3):
        single = probability_any_entry_wins(scores[:, col : col + 1], opp)
        assert any_win >= single - 1e-12


def test_empty_trial_set_returns_zero_rather_than_dividing_by_zero():
    shares = payout_shares("winner_take_all", 4)
    assert expected_portfolio_prize(np.zeros((0, 1)), [], shares) == 0.0
    assert probability_any_entry_wins(np.zeros((0, 1)), []) == 0.0
