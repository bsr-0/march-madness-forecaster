"""Selection objectives over pre-drawn pool trials (audit recommendation 13, parts 3-4).

WHAT THIS REPLACES
------------------
``scripts.mc_pool_backtest.score_candidate_p1`` scores ONE candidate against a
trial set, and the selection loops call it once per candidate. Each call
re-scores the whole opponent field, so with 25 candidates the same 29 opponents
are scored 25 times per trial to produce a number that does not depend on the
candidate at all. Measured on 2025 data: 12,500 redundant opponent scorings per
season, ~6.4s, against ~0.2s for the same information computed once.

``precompute_trial_scores`` does it once. Everything else here is a cheap
function of that matrix, which is what makes a prize-weighted objective and a
multi-entry portfolio search affordable rather than a rewrite of the harness.

BIT-EXACTNESS
-------------
``p_first_from_scores`` reproduces ``score_candidate_p1`` exactly -- same ``>=``
tie convention, same integer win count, same division -- because the published
~12% headline is computed with it and must not move as a side effect of adding
payout support. ``tests/test_pool_objectives.py`` pins the equivalence on real
tournament data rather than trusting the reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.optimization.payout import (
    TIE_SPLIT,
    TIE_WIN,
    expected_portfolio_prize,
    payout_shares,
    prize_for_scores,
    probability_any_entry_wins,
)
from src.simulation.pool_competition import score_brackets_team_identity


@dataclass(frozen=True)
class TrialScores:
    """Scores for every candidate and every opponent, across a pre-drawn trial set.

    Attributes:
        candidate: ``(n_candidates, n_trials)`` -- our brackets' scores.
        opponent: length ``n_trials``, each ``(n_opponents,)``. A list rather
            than a 2-D array because the opponent count is fixed within a run
            but the structure stays honest if a caller ever varies it.
        n_trials: Convenience.
    """

    candidate: np.ndarray
    opponent: List[np.ndarray]

    @property
    def n_trials(self) -> int:
        return len(self.opponent)

    @property
    def n_candidates(self) -> int:
        return int(self.candidate.shape[0])


def precompute_trial_scores(
    candidate_vecs: Sequence[np.ndarray],
    trials: Sequence[Tuple[np.ndarray, Dict[str, set]]],
    first_round: Sequence[str],
    scoring_system: Dict[str, int],
) -> TrialScores:
    """Score all candidates and all opponents once per trial.

    Args:
        candidate_vecs: Each ``(63,)`` bool. Our bracket(s).
        trials: ``(opponent_matrix, winners_by_round)`` pairs, as produced by
            ``scripts.mc_pool_backtest.draw_selection_trials``. Every candidate
            is scored against the SAME trials -- the common-random-numbers
            design the selection loop already relies on, kept intact.
        first_round: 64-team ordered list. Must carry the season's real Final
            Four region pairing (``derive_f4_region_pairing``); the default
            ``REGION_ORDER`` matches reality in none of the 15 seasons.
        scoring_system: Round name -> points.

    Returns:
        A :class:`TrialScores`.
    """
    if not trials:
        return TrialScores(candidate=np.zeros((len(candidate_vecs), 0), dtype=float), opponent=[])
    cands = np.stack([np.asarray(v, dtype=bool).reshape(63) for v in candidate_vecs])

    cand_scores = np.empty((cands.shape[0], len(trials)), dtype=float)
    opp_scores: List[np.ndarray] = []
    for t, (opp, winners) in enumerate(trials):
        cand_scores[:, t] = score_brackets_team_identity(cands, winners, first_round, scoring_system)
        opp_scores.append(np.asarray(score_brackets_team_identity(opp, winners, first_round, scoring_system), dtype=float))
    return TrialScores(candidate=cand_scores, opponent=opp_scores)


def p_first_from_scores(scores: TrialScores, index: int) -> float:
    """P(1st) for one candidate: expected first-place SHARE over the trials.

    A tie for the top score is split among the tied entries
    (:func:`src.optimization.payout.first_place_share`). This is the one
    definition of P(1st) in the project since the 2026-09 audit (Step 4,
    F4-1); it is also exactly the expected winner-take-all prize.
    """
    from src.optimization.payout import first_place_share

    n = scores.n_trials
    if n == 0:
        return 0.0
    row = scores.candidate[index]
    return float(sum(first_place_share(row[t], scores.opponent[t]) for t in range(n)) / n)


def expected_prize_from_scores(
    scores: TrialScores,
    index: int,
    shares: np.ndarray,
    tie_policy: str = TIE_SPLIT,
) -> float:
    """Expected share of the pot for one candidate under a payout structure."""
    n = scores.n_trials
    if n == 0:
        return 0.0
    row = scores.candidate[index]
    total = 0.0
    for t in range(n):
        total += float(prize_for_scores(row[t : t + 1], scores.opponent[t], shares, tie_policy)[0])
    return total / n


@dataclass(frozen=True)
class PortfolioResult:
    """Outcome of a multi-entry selection.

    Attributes:
        indices: Chosen candidate indices, in the order they were added.
        labels: Their human labels, same order.
        expected_prize: Expected TOTAL share of the pot across all entries.
        p_any_first: P(at least one entry finishes first).
        marginal_gains: Expected-prize gain from each entry as it was added --
            the diminishing-returns curve, which is the honest answer to "is a
            second entry worth it?".
        single_best_expected_prize: What the best SINGLE entry would have
            earned, for comparison.
    """

    indices: Tuple[int, ...]
    labels: Tuple[str, ...]
    expected_prize: float
    p_any_first: float
    marginal_gains: Tuple[float, ...]
    single_best_expected_prize: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "indices": list(self.indices),
            "labels": list(self.labels),
            "expected_prize": self.expected_prize,
            "p_any_first": self.p_any_first,
            "marginal_gains": list(self.marginal_gains),
            "single_best_expected_prize": self.single_best_expected_prize,
            "lift_over_single_entry": self.expected_prize - self.single_best_expected_prize,
        }


def select_portfolio(
    scores: TrialScores,
    n_entries: int,
    shares: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    tie_policy: str = TIE_SPLIT,
) -> PortfolioResult:
    """Choose ``n_entries`` brackets jointly, maximising expected total prize.

    Greedy forward selection: start empty, repeatedly add whichever remaining
    candidate most improves the portfolio's expected prize, evaluated on the
    shared trial set. Greedy rather than exhaustive because the objective is
    submodular in practice (a second entry that duplicates the first adds
    nothing) and the exhaustive search over C(25, 3) with 500 trials is not
    worth its cost for a gain that is, by construction, bounded by the greedy
    gap.

    THIS IS NOT BEST-OF-K. Every chosen entry is scored in every trial and
    contributes its own prize -- an entry that finishes 20th contributes 0 and
    still counted against the pool size. The distinction matters because
    ``scripts/real_pool_placement.py`` exists to prevent exactly the other
    thing: *"nobody submits 50 brackets and keeps only the winner"*. Selection
    here happens before any outcome is known; scoring happens over all entries.

    Args:
        scores: From :func:`precompute_trial_scores`. The opponent fields must
            already have been drawn with ``pool_size - n_entries`` opponents --
            entering k brackets does not shrink the field, it means k of the
            pool's seats are yours.
        n_entries: How many brackets to enter. ``1`` reduces to picking the
            single best candidate by expected prize.
        shares: From ``payout.payout_shares``, length ``pool_size``.
        labels: Candidate labels, parallel to ``scores.candidate`` rows.
        tie_policy: See :func:`src.optimization.payout.prize_for_scores`.

    Returns:
        A :class:`PortfolioResult`.

    Raises:
        ValueError: if ``n_entries`` is not in ``1..n_candidates``.
    """
    n_cand = scores.n_candidates
    if n_entries < 1:
        raise ValueError(f"n_entries must be >= 1, got {n_entries}")
    if n_entries > n_cand:
        raise ValueError(f"asked for {n_entries} entries but only {n_cand} candidates were supplied")
    if scores.n_trials == 0:
        return PortfolioResult((), (), 0.0, 0.0, (), 0.0)

    label_list = list(labels) if labels is not None else [f"candidate_{i}" for i in range(n_cand)]

    chosen: List[int] = []
    gains: List[float] = []
    running = 0.0

    for _ in range(n_entries):
        best_idx, best_value = -1, -np.inf
        for i in range(n_cand):
            if i in chosen:
                continue
            trial_scores = scores.candidate[chosen + [i], :].T  # (n_trials, k)
            value = expected_portfolio_prize(trial_scores, scores.opponent, shares, tie_policy)
            # Strictly-greater keeps the first candidate on a tie, matching the
            # incumbent selector's convention -- so candidate ORDER is
            # load-bearing here too (see poolaware_recipe's warning).
            if value > best_value:
                best_idx, best_value = i, value
        if best_idx < 0:  # pragma: no cover - unreachable while n_entries <= n_cand
            break
        chosen.append(best_idx)
        gains.append(best_value - running)
        running = best_value

    final = scores.candidate[chosen, :].T
    single_best = max(
        expected_prize_from_scores(scores, i, shares, tie_policy) for i in range(n_cand)
    )
    return PortfolioResult(
        indices=tuple(chosen),
        labels=tuple(label_list[i] for i in chosen),
        expected_prize=running,
        p_any_first=probability_any_entry_wins(final, scores.opponent),
        marginal_gains=tuple(gains),
        single_best_expected_prize=single_best,
    )


def select_best_single(
    scores: TrialScores,
    shares: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    tie_policy: str = TIE_SPLIT,
) -> Tuple[int, float]:
    """Index and expected prize of the best single candidate under a payout.

    The payout-aware replacement for the incumbent ``argmax P(1st)`` selector.
    Under ``winner_take_all`` with ``tie_policy=TIE_WIN`` it selects identically
    to it, which is what keeps the default path unchanged.
    """
    del labels  # kept for call-site symmetry with select_portfolio
    best_idx, best_value = 0, -np.inf
    for i in range(scores.n_candidates):
        value = expected_prize_from_scores(scores, i, shares, tie_policy)
        if value > best_value:
            best_idx, best_value = i, value
    return best_idx, best_value


def objective_for(
    structure: str,
    pool_size: int,
    custom_shares: Optional[Sequence[float]] = None,
    tie_policy: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Shares and tie policy for a named structure, with the default that keeps
    winner-take-all bit-identical to the incumbent selector.

    Returns:
        ``(shares, tie_policy)``.
    """
    shares = payout_shares(structure, pool_size, custom_shares)
    if tie_policy is None:
        # Winner-take-all must reproduce the published headline exactly, so it
        # inherits the incumbent tie-as-win convention. Every other structure
        # splits, which is what real pools do.
        tie_policy = TIE_WIN if structure == "winner_take_all" and custom_shares is None else TIE_SPLIT
    return shares, tie_policy
