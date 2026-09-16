"""Prize-weighted objectives for pool selection (audit recommendation 13, part 3).

WHY THIS EXISTS
---------------
The optimiser has one objective: P(1st). That is the right objective for exactly
one pool -- winner-take-all -- and the repo has always assumed it. Five payout
structures already exist in ``src.optimization.leverage.PAYOUT_ADJUSTMENTS``, but
they are *multipliers on a strategy mix*, not an objective: nothing anywhere
computes ``sum_k prize_k * P(rank = k)``. ``scripts/mc_pool_backtest.py`` hardcodes
``payout_structure="winner_take_all"`` and has no ``--payout`` flag, and the
``p_top5`` / ``p_top25`` columns it prints are computed after the fact and
selected on by nothing.

So for a pool that pays its top three, the tool is answering a question that pool
is not asking. Recommendation 13 calls this one of "the features that separate a
recommender from a pool tool".

WHAT A PAYOUT STRUCTURE IS HERE
-------------------------------
A vector of shares of the prize pool, indexed by finishing rank, summing to 1.
Normalising to 1 makes the objective *expected share of the pot*, which is
directly comparable across structures and reduces to P(1st) exactly under
winner-take-all. Nothing here needs to know the size of the pot.

TIES
----
Real pools split a tied prize between the tied entrants, and with ESPN scoring
(sums of 10/20/40/80/160/320) exact ties are common enough to matter. So the
default here is ``TIE_SPLIT``: entrants tied at ranks r..r+m-1 each receive the
mean of shares[r-1 : r-1+m].

This differs from the incumbent ``score_candidate_p1``, which counts any tie for
first as a full win (``c_score >= opp_scores.max()``). That convention is
deliberately reproducible here via ``TIE_WIN``, because the published ~12%
headline is computed under it and must not move as a side effect of adding
payouts. ``tests/test_payout.py`` pins the equivalence.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# Tie conventions.
#
# TIE_SPLIT is what a real pool does and, since the 2026-09 audit (Step 4,
# F4-1), it is THE definition of P(1st) everywhere: selection, reporting and
# payout. Before that the same name meant three things -- the selector counted a
# shared first as a full win (>=), the backtest table counted it as a loss
# (rank 1.5 != 1), and only the prize column split it. Ties are ~7% of the
# events the >= rule called wins, so the two wrong conventions erred in
# opposite directions. TIE_WIN is retained only so the old figures can be
# reproduced on demand; nothing in production uses it.
TIE_SPLIT = "split"
TIE_WIN = "win"


def first_place_share(our_score: float, opp_scores: np.ndarray) -> float:
    """Canonical P(1st) contribution of one pool realisation.

    1 if we outscore every opponent, 1/(1+k) if we tie with k opponents for the
    top score, else 0. Its expectation over realisations is the expected share
    of a winner-take-all pot, which is what "P(1st)" means in this project.
    With no opponents the share is 1.
    """
    opp = np.asarray(opp_scores, dtype=float).ravel()
    if opp.size == 0:
        return 1.0
    top = opp.max()
    if our_score > top:
        return 1.0
    if our_score < top:
        return 0.0
    return 1.0 / (1.0 + float(np.count_nonzero(opp == top)))


def first_place_shares(our_scores: np.ndarray, opp_scores: np.ndarray) -> np.ndarray:
    """Vectorised :func:`first_place_share` for many of our brackets against
    one opponent field (one pool realisation)."""
    ours = np.asarray(our_scores, dtype=float)
    opp = np.asarray(opp_scores, dtype=float).ravel()
    if opp.size == 0:
        return np.ones_like(ours)
    top = opp.max()
    n_top = float(np.count_nonzero(opp == top))
    out = np.zeros_like(ours)
    out[ours > top] = 1.0
    out[ours == top] = 1.0 / (1.0 + n_top)
    return out


def first_place_share_from_counts(better: int, tied: int) -> float:
    """Same quantity from the backtest's (opponents strictly better, opponents
    tied) counts: 1/(1+tied) when nobody is better, else 0."""
    return 0.0 if better > 0 else 1.0 / (1.0 + float(tied))

# Share vectors for the named structures, as fractions of the pot.
#
# Names are kept identical to leverage.PAYOUT_ADJUSTMENTS so the repo has one
# payout vocabulary rather than two. The percentage structures are defined by a
# rule rather than a literal because their length depends on pool size.
#
# The top_3 split (60/30/10) is the most common convention in ESPN-style office
# pools. It is a convention, not a measurement: a pool paying 50/30/20 is a
# different objective and can be passed explicitly via `custom_shares`.
_FIXED_SHARES: Dict[str, Sequence[float]] = {
    "winner_take_all": (1.0,),
    "top_3": (0.60, 0.30, 0.10),
    "top_5": (0.50, 0.25, 0.13, 0.07, 0.05),
}

# Structures that pay an equal share to a fraction of the field.
_PERCENTILE_SHARES: Dict[str, float] = {
    "top_10pct": 0.10,
    "top_25pct": 0.25,
}

# `tiered` in leverage.PAYOUT_ADJUSTMENTS is a strategy-mix neutral point with no
# stated share vector. Rather than invent one, it is aliased to top_5 and the
# alias is declared, so a caller asking for it gets something defined.
_ALIASES: Dict[str, str] = {"tiered": "top_5"}

VALID_PAYOUT_STRUCTURES = frozenset(_FIXED_SHARES) | frozenset(_PERCENTILE_SHARES) | frozenset(_ALIASES)


def payout_shares(
    structure: str,
    pool_size: int,
    custom_shares: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Shares of the pot by finishing rank, as a ``(pool_size,)`` array summing to 1.

    Args:
        structure: One of :data:`VALID_PAYOUT_STRUCTURES`, or ``"custom"`` when
            ``custom_shares`` is given.
        pool_size: Total entries in the pool, including yours. Determines the
            length of the vector and, for percentage structures, how many spots
            pay.
        custom_shares: Explicit shares for ranks 1..n. Renormalised to sum to 1,
            so callers may pass raw dollar amounts.

    Returns:
        ``shares[i]`` is the fraction of the pot paid to rank ``i + 1``.

    Raises:
        ValueError: on an unknown structure, a non-positive pool size, a
            structure that pays more spots than the pool has entries, or
            custom shares that are negative or all zero.
    """
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")

    if custom_shares is not None:
        raw = np.asarray(custom_shares, dtype=float)
        if raw.size == 0:
            raise ValueError("custom_shares is empty")
        if np.any(raw < 0):
            raise ValueError("custom_shares contains a negative share")
        total = raw.sum()
        if total <= 0:
            raise ValueError("custom_shares sums to zero; nothing would ever be paid")
        if raw.size > pool_size:
            raise ValueError(f"custom_shares pays {raw.size} spots but the pool has only {pool_size} entries")
        out = np.zeros(pool_size, dtype=float)
        out[: raw.size] = raw / total
        return out

    name = _ALIASES.get(structure, structure)
    if name in _FIXED_SHARES:
        raw = np.asarray(_FIXED_SHARES[name], dtype=float)
        if raw.size > pool_size:
            raise ValueError(
                f"payout structure {structure!r} pays {raw.size} spots but the pool has only "
                f"{pool_size} entries; a pool cannot pay more places than it has entrants"
            )
        out = np.zeros(pool_size, dtype=float)
        out[: raw.size] = raw / raw.sum()
        return out

    if name in _PERCENTILE_SHARES:
        frac = _PERCENTILE_SHARES[name]
        n_paid = max(1, math.ceil(frac * pool_size))
        out = np.zeros(pool_size, dtype=float)
        out[:n_paid] = 1.0 / n_paid
        return out

    raise ValueError(f"unknown payout structure {structure!r}; known: {sorted(VALID_PAYOUT_STRUCTURES)}")


def prize_for_scores(
    our_scores: np.ndarray,
    opp_scores: np.ndarray,
    shares: np.ndarray,
    tie_policy: str = TIE_SPLIT,
) -> np.ndarray:
    """Prize share earned by each of our entries in one pool realisation.

    Args:
        our_scores: ``(k,)`` scores for our entries. ``k = 1`` is the ordinary
            single-entry case.
        opp_scores: ``(n_opp,)`` scores for everyone else in the pool.
        shares: ``(pool_size,)`` from :func:`payout_shares`. ``pool_size`` should
            equal ``k + n_opp``; a shorter vector is padded with zeros, which is
            what "only the top spots pay" means.
        tie_policy: :data:`TIE_SPLIT` (real pools) or :data:`TIE_WIN` (reproduces
            the incumbent P(1st) convention, where a tie for first counts as a
            full win).

    Returns:
        ``(k,)`` prize shares, one per entry. Sum over entries to get the
        portfolio's total share of the pot.

    Note:
        Our own entries compete with each other, exactly as they would in a real
        pool: with ``k = 2`` and both beating every opponent, they occupy ranks 1
        and 2, and under ``top_3`` collect 0.60 + 0.30 rather than 0.60 twice.
        Getting this wrong is the difference between a portfolio model and
        wishful thinking.
    """
    our = np.atleast_1d(np.asarray(our_scores, dtype=float))
    opp = np.asarray(opp_scores, dtype=float).ravel()
    k = our.size
    pool_size = k + opp.size

    padded = np.zeros(pool_size, dtype=float)
    take = min(shares.size, pool_size)
    padded[:take] = shares[:take]

    field = np.concatenate([our, opp])
    out = np.empty(k, dtype=float)

    for i in range(k):
        s = our[i]
        n_better = int(np.count_nonzero(field > s))
        # Everyone tied with us, including ourselves.
        n_tied = int(np.count_nonzero(field == s))
        rank0 = n_better  # 0-based rank of the first tied position

        if tie_policy == TIE_WIN:
            # Incumbent convention: a tie for a paying place collects that
            # place's full share rather than a split of it.
            out[i] = padded[rank0]
        elif tie_policy == TIE_SPLIT:
            out[i] = float(padded[rank0 : rank0 + n_tied].mean()) if n_tied > 0 else 0.0
        else:
            raise ValueError(f"tie_policy must be {TIE_SPLIT!r} or {TIE_WIN!r}, got {tie_policy!r}")

    return out


def expected_prize(
    our_scores_by_trial: np.ndarray,
    opp_scores_by_trial: Sequence[np.ndarray],
    shares: np.ndarray,
    tie_policy: str = TIE_SPLIT,
) -> float:
    """Mean prize share for a single entry across pre-drawn trials.

    Args:
        our_scores_by_trial: ``(n_trials,)`` our entry's score in each trial.
        opp_scores_by_trial: length ``n_trials``; each ``(n_opp,)``.
        shares: from :func:`payout_shares`.
        tie_policy: see :func:`prize_for_scores`.

    Returns:
        Expected share of the pot. Under ``winner_take_all`` with
        ``tie_policy=TIE_WIN`` this is exactly the incumbent P(1st).
    """
    n_trials = len(opp_scores_by_trial)
    if n_trials == 0:
        return 0.0
    if our_scores_by_trial.shape[0] != n_trials:
        raise ValueError(
            f"our_scores_by_trial has {our_scores_by_trial.shape[0]} trials but "
            f"opp_scores_by_trial has {n_trials}"
        )
    total = 0.0
    for t in range(n_trials):
        total += float(prize_for_scores(our_scores_by_trial[t : t + 1], opp_scores_by_trial[t], shares, tie_policy)[0])
    return total / n_trials


def expected_portfolio_prize(
    entry_scores_by_trial: np.ndarray,
    opp_scores_by_trial: Sequence[np.ndarray],
    shares: np.ndarray,
    tie_policy: str = TIE_SPLIT,
) -> float:
    """Mean TOTAL prize share for a portfolio of k entries (recommendation 13, part 4).

    Args:
        entry_scores_by_trial: ``(n_trials, k)`` scores for our k entries.
        opp_scores_by_trial: length ``n_trials``; each ``(n_opp,)`` with
            ``n_opp = pool_size - k``.
        shares: from :func:`payout_shares`, length ``pool_size``.
        tie_policy: see :func:`prize_for_scores`.

    Returns:
        Expected total share of the pot across all k entries.

    This is the objective a multi-entry player actually faces, and it is NOT
    "the best of our k brackets". Best-of-k evaluated after the fact is the
    hindsight error that ``scripts/real_pool_placement.py`` exists to prevent --
    "nobody submits 50 brackets and keeps only the winner". Here the k entries
    are chosen *before* any outcome is known and every one of them is scored;
    entries that finish out of the money simply contribute 0.
    """
    n_trials = len(opp_scores_by_trial)
    if n_trials == 0:
        return 0.0
    scores = np.atleast_2d(entry_scores_by_trial)
    if scores.shape[0] != n_trials:
        raise ValueError(f"entry_scores_by_trial has {scores.shape[0]} trials but {n_trials} opponent draws")
    total = 0.0
    for t in range(n_trials):
        total += float(prize_for_scores(scores[t], opp_scores_by_trial[t], shares, tie_policy).sum())
    return total / n_trials


def probability_any_entry_wins(
    entry_scores_by_trial: np.ndarray,
    opp_scores_by_trial: Sequence[np.ndarray],
) -> float:
    """Expected first-place share of a portfolio: the sum over our entries of
    their :func:`first_place_share` in each realisation (our own entries tie
    with each other like anyone else), averaged over trials.

    The multi-entry generalisation of P(1st). For one entry it is exactly
    P(1st) under the canonical tie rule.
    """
    n_trials = len(opp_scores_by_trial)
    if n_trials == 0:
        return 0.0
    scores = np.atleast_2d(entry_scores_by_trial)
    total = 0.0
    for t in range(n_trials):
        opp = opp_scores_by_trial[t]
        ours = scores[t]
        field_top = max(ours.max(), opp.max() if opp.size else -np.inf)
        n_top = float(np.count_nonzero(ours == field_top) + (np.count_nonzero(opp == field_top) if opp.size else 0))
        total += float(np.count_nonzero(ours == field_top)) / n_top
    return total / n_trials


def describe(structure: str, pool_size: int, custom_shares: Optional[Sequence[float]] = None) -> str:
    """One-line human description, for run headers and artifact provenance."""
    shares = payout_shares(structure, pool_size, custom_shares)
    paid = int(np.count_nonzero(shares))
    top = ", ".join(f"{s:.0%}" for s in shares[: min(paid, 5)])
    more = "…" if paid > 5 else ""
    return f"{structure} — {paid} of {pool_size} places pay ({top}{more})"


def resolve_pool_size(n_opponents: int, n_entries: int = 1) -> int:
    """Total entries in the pool, given how many opponents and how many of ours.

    Exists so the off-by-one that audit finding H6 turned on is written once.
    ``--n-opponents 29`` means a 30-person pool; entering 2 brackets against 29
    opponents means a 31-person pool, not 30.
    """
    if n_opponents < 0:
        raise ValueError(f"n_opponents must be >= 0, got {n_opponents}")
    if n_entries < 1:
        raise ValueError(f"n_entries must be >= 1, got {n_entries}")
    return n_opponents + n_entries


def shares_summary(shares: np.ndarray) -> Mapping[str, float]:
    """Compact, JSON-safe summary for artifacts."""
    nz = np.nonzero(shares)[0]
    return {
        "n_paying_places": int(nz.size),
        "top_share": float(shares[0]) if shares.size else 0.0,
        "last_paying_share": float(shares[nz[-1]]) if nz.size else 0.0,
        "total": float(shares.sum()),
    }
