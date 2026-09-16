"""Trial-score precomputation and portfolio selection (audit recommendation 13).

`test_precomputed_p1_is_bit_identical_to_score_candidate_p1` is the gate that
matters. The published ~12% headline is produced by `score_candidate_p1`, and
the selection loop now runs on a precomputed score matrix instead -- a pure
performance change that must be provably not a behaviour change. It is checked
on real tournament data rather than synthetic vectors, because the failure mode
would be a subtle disagreement in scoring, not in arithmetic.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.optimization.payout import TIE_SPLIT, TIE_WIN, payout_shares  # noqa: E402
from src.optimization.pool_objectives import (  # noqa: E402
    expected_prize_from_scores,
    objective_for,
    p_first_from_scores,
    precompute_trial_scores,
    select_best_single,
    select_portfolio,
)


@pytest.fixture(scope="module")
def real_trials():
    """A real season's trial set: 2025 seeds, real pool picks, real F4 pairing."""
    pytest.importorskip("scipy")
    try:
        from scripts.mc_pool_backtest import (
            POOL_HIST_PATH,
            build_first_round_matchups,
            build_seed_probabilities,
            derive_f4_region_pairing,
            draw_selection_trials,
            load_seeds_and_regions,
            load_tournament_results,
        )
        from src.simulation.pool_history_opponent_model import (
            build_pool_pick_distribution,
            load_pool_brackets,
        )
    except ImportError:  # pragma: no cover
        pytest.skip("project deps unavailable")

    year = 2025
    try:
        seeds, regions = load_seeds_and_regions(year)
        games = load_tournament_results(year)
        brackets, _ = load_pool_brackets(POOL_HIST_PATH, year)
    except Exception as exc:  # pragma: no cover - data-availability guard
        pytest.skip(f"{year} data unavailable: {exc}")

    # The real F4 pairing, not REGION_ORDER -- see test_independent_referee_check.
    # Play-ins resolved first: the unified builder refuses a contested slot
    # rather than guessing by file order (2026-09 audit, Step 3).
    from scripts.mc_pool_backtest import resolve_first_four

    resolve_first_four(games, seeds, regions)
    first_round = build_first_round_matchups(seeds, regions, region_order=derive_f4_region_pairing(games, regions))
    trials = draw_selection_trials(
        60,
        n_opponents=20,
        first_round=first_round,
        pick_dist=build_pool_pick_distribution(brackets, seeds),
        matchup_probs=build_seed_probabilities(seeds),
        seeds=seeds,
        rng=np.random.default_rng(2025),
    )
    rng = np.random.default_rng(11)
    candidates = [rng.random(63) < 0.5 for _ in range(12)]
    return first_round, trials, candidates


def test_precomputed_p1_is_bit_identical_to_score_candidate_p1(real_trials):
    """The headline must survive the refactor exactly, not approximately."""
    from scripts.mc_pool_backtest import ESPN_SCORING, score_candidate_p1

    first_round, trials, candidates = real_trials

    incumbent = [score_candidate_p1(c, trials, first_round, ESPN_SCORING) for c in candidates]
    scores = precompute_trial_scores(candidates, trials, first_round, ESPN_SCORING)
    refactored = [p_first_from_scores(scores, i) for i in range(len(candidates))]

    assert refactored == incumbent, (
        "the precomputed selection path disagrees with score_candidate_p1. That function "
        "defines the published P(1st); any difference here is a silent change to the headline."
    )


def test_precompute_scores_every_candidate_and_opponent(real_trials):
    first_round, trials, candidates = real_trials
    scores = precompute_trial_scores(candidates, trials, first_round, ESPN := {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320})
    del ESPN
    assert scores.candidate.shape == (len(candidates), len(trials))
    assert scores.n_trials == len(trials)
    assert all(o.shape == (20,) for o in scores.opponent)


def test_winner_take_all_expected_prize_equals_p1_under_the_win_tie_rule(real_trials):
    """The reduction that makes payouts a generalisation rather than a rewrite."""
    from scripts.mc_pool_backtest import ESPN_SCORING

    first_round, trials, candidates = real_trials
    scores = precompute_trial_scores(candidates, trials, first_round, ESPN_SCORING)
    shares = payout_shares("winner_take_all", 21)

    for i in range(len(candidates)):
        assert expected_prize_from_scores(scores, i, shares, TIE_WIN) == pytest.approx(
            p_first_from_scores(scores, i)
        )


def test_tie_split_never_exceeds_tie_win_for_winner_take_all(real_trials):
    """Splitting a shared first can only reduce the expected prize."""
    from scripts.mc_pool_backtest import ESPN_SCORING

    first_round, trials, candidates = real_trials
    scores = precompute_trial_scores(candidates, trials, first_round, ESPN_SCORING)
    shares = payout_shares("winner_take_all", 21)

    for i in range(len(candidates)):
        split = expected_prize_from_scores(scores, i, shares, TIE_SPLIT)
        win = expected_prize_from_scores(scores, i, shares, TIE_WIN)
        assert split <= win + 1e-12


def test_objective_for_keeps_winner_take_all_on_the_incumbent_tie_rule():
    """Default-path protection: only winner_take_all inherits TIE_WIN."""
    _, tie = objective_for("winner_take_all", 30)
    assert tie == TIE_WIN

    _, tie = objective_for("top_3", 30)
    assert tie == TIE_SPLIT, "a real payout splits ties; only the legacy headline does not"

    _, tie = objective_for("winner_take_all", 30, custom_shares=[1.0])
    assert tie == TIE_SPLIT, "an explicit custom structure is a new objective, not the legacy one"


# ---------------------------------------------------------------------------
# Portfolio selection
# ---------------------------------------------------------------------------


def _synthetic_scores(n_cand=6, n_trials=200, n_opp=10, seed=0):
    """Candidates with deliberately complementary strengths.

    Candidate 0 wins one set of trials, candidate 1 a disjoint set. A portfolio
    search that understands complementarity picks both; one that just takes the
    top two by individual score may not.
    """
    from src.optimization.pool_objectives import TrialScores

    rng = np.random.default_rng(seed)
    cand = rng.integers(200, 400, size=(n_cand, n_trials)).astype(float)
    half = n_trials // 2
    cand[0, :half] = 2000.0  # dominates the first half
    cand[1, half:] = 2000.0  # dominates the second half
    opp = [rng.integers(200, 400, size=n_opp).astype(float) for _ in range(n_trials)]
    return TrialScores(candidate=cand, opponent=opp)


def test_portfolio_picks_complementary_entries_not_two_of_the_same():
    scores = _synthetic_scores()
    shares = payout_shares("winner_take_all", 12)

    result = select_portfolio(scores, 2, shares)

    assert set(result.indices) == {0, 1}, (
        f"expected the two complementary candidates, got {result.indices}. Greedy selection "
        "must value covering a trial nobody else covers."
    )
    assert result.p_any_first > 0.9


def test_portfolio_beats_the_best_single_entry():
    scores = _synthetic_scores()
    shares = payout_shares("winner_take_all", 12)

    result = select_portfolio(scores, 2, shares)

    assert result.expected_prize > result.single_best_expected_prize


def test_marginal_gains_are_non_increasing():
    """Diminishing returns -- the honest answer to 'is a 4th entry worth it?'.

    Greedy on a submodular objective produces a non-increasing gain sequence.
    If this ever fails, the objective is not behaving submodularly and the
    greedy justification in select_portfolio's docstring no longer holds.
    """
    scores = _synthetic_scores(n_cand=8)
    shares = payout_shares("top_3", 18)

    result = select_portfolio(scores, 4, shares)

    gains = list(result.marginal_gains)
    assert len(gains) == 4
    for earlier, later in zip(gains, gains[1:]):
        assert later <= earlier + 1e-9, f"marginal gains increased: {gains}"


def test_single_entry_portfolio_matches_best_single_selection():
    scores = _synthetic_scores()
    shares = payout_shares("top_3", 12)

    result = select_portfolio(scores, 1, shares)
    idx, value = select_best_single(scores, shares)

    assert result.indices == (idx,)
    assert result.expected_prize == pytest.approx(value)


def test_asking_for_more_entries_than_candidates_raises():
    scores = _synthetic_scores(n_cand=3)
    shares = payout_shares("winner_take_all", 14)
    with pytest.raises(ValueError, match="only 3 candidates"):
        select_portfolio(scores, 4, shares)
    with pytest.raises(ValueError, match="n_entries must be >= 1"):
        select_portfolio(scores, 0, shares)


def test_portfolio_result_serialises_with_its_lift():
    scores = _synthetic_scores()
    shares = payout_shares("winner_take_all", 12)
    payload = select_portfolio(scores, 2, shares).to_dict()

    assert payload["lift_over_single_entry"] == pytest.approx(
        payload["expected_prize"] - payload["single_best_expected_prize"]
    )
    import json

    json.dumps(payload)


def test_payout_changes_which_single_candidate_is_chosen():
    """The whole point of part 3: a different payout can want a different bracket.

    A high-variance candidate that sometimes wins outright and usually finishes
    far down should beat a steady mid-table one under winner-take-all, and lose
    to it under a structure that pays a quarter of the field.
    """
    from src.optimization.pool_objectives import TrialScores

    # A fixed opponent field, so the two candidates' ranks are exact rather
    # than distributional -- the point being tested is the objective, not the
    # sampling.
    n_trials = 400
    opp_field = np.array([1000.0, 990.0, 980.0, 970.0, 960.0, 950.0, 940.0, 930.0, 920.0, 910.0, 900.0])
    spiky = np.full(n_trials, 10.0)
    spiky[:40] = 5000.0  # rank 1 in 10% of trials, dead last in the other 90%
    steady = np.full(n_trials, 995.0)  # always rank 2: never wins, always paid
    opp = [opp_field for _ in range(n_trials)]
    scores = TrialScores(candidate=np.vstack([spiky, steady]), opponent=opp)

    wta_idx, _ = select_best_single(scores, *objective_for("winner_take_all", 12))
    broad_idx, _ = select_best_single(scores, *objective_for("top_25pct", 12))

    assert wta_idx == 0, "winner-take-all should prefer the candidate that actually wins sometimes"
    assert broad_idx == 1, "a structure paying a quarter of the field should prefer consistency"
