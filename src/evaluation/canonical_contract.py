"""The canonical measurement contract, in one place (audit recommendation 15).

WHY THIS EXISTS
---------------
Every P(1st) figure this project publishes is measured under one specific
contract, and the contract is part of the claim rather than a detail: P(1st) is
mechanically pool-size dependent, so the same strategy scores roughly 2.5x worse
in a 1000-entry field than a 30-entry one. Audit finding H6 is the cost of
forgetting that -- ``N_OPPONENTS`` was 999 until 2026-09-10, and every pre-2023
season was silently measured in a pool 33x larger than any that exists here.

Recommendation 15 flagged one instance of the same class: ``run_experiment.py``
defaulted to 30 opponents -- a **31**-person pool -- while the canonical
contract is 29 opponents in a **30**-person pool, so its numbers were not
directly comparable to the headline. It was not one instance. Six modules
declared their own opponent count, all of them 30, none with a recorded reason:

    scripts/run_experiment.py              (4 separate defaults)
    scripts/retrospective_scorer.py        POOL_SIZE = 31, N_OPPONENTS = 30
    scripts/rank_correlation_diagnostic.py N_OPPONENTS = 30, pool_size = 31
    scripts/validate_opponent_model.py     N_OPPONENTS = 30
    scripts/pool_retrospective.py          n_opponents = 30 (twice)
    src/optimization/recency_hparam_fitter.py  n_opponents = 30

The clearest symptom was a variable literally named ``recency_fitter_3yr_pool30``
holding ``n_opponents=30``, i.e. a 31-person pool -- the name asserting the thing
the value contradicts.

THE OFF-BY-ONE, ONCE
--------------------
``n_opponents`` counts the OTHER entries. ``pool_size`` counts everyone,
including ours. They differ by the number of brackets we enter, which is 1 in
every published figure:

    pool_size = n_opponents + n_entries

That identity lives in :func:`src.optimization.payout.resolve_pool_size`; this
module is where the canonical *values* live. Import them rather than writing
``29`` or ``30`` again: a module that re-declares its own number is how the two
drift apart, and the drift is invisible at a glance because 30 is a plausible
value for either quantity.

A module that genuinely needs a different field size should say so out loud and
say why -- ``tests/test_canonical_contract.py`` will make it.
"""

from __future__ import annotations

from typing import Any, Dict

# --- The contract ---------------------------------------------------------
#
# These are the values behind every published P(1st). Changing one invalidates
# the comparison between any figure measured before and after, so a change here
# means re-running and re-publishing, not just editing.

#: Opponents in the canonical pool. 29 others + 1 of ours = 30 entries.
#: Was 999 until 2026-09-10; see audit finding H6 for what that cost.
CANONICAL_N_OPPONENTS: int = 29

#: Total entries in the canonical pool, ours included.
CANONICAL_POOL_SIZE: int = 30

#: Pool realisations per season. Reduces opponent-sampling variance; the
#: estimand does not depend on it.
CANONICAL_N_REPEATS: int = 100

#: MC trials per candidate inside pool-aware selection.
CANONICAL_PA_TRIALS: int = 500

#: Opponent pick distribution: the real pool's own picks where the season has
#: them (2023-2026), ESPN national otherwise.
CANONICAL_OPPONENT_SOURCE: str = "pool"

#: Team-identity scoring -- what the ESPN pool actually pays under.
CANONICAL_TEAM_IDENTITY: bool = True

#: Prize structure. The real pool behind ``pool_hist_results.json`` is
#: winner-take-all; other structures are supported (recommendation 13) but the
#: published figures are this one.
CANONICAL_PAYOUT: str = "winner_take_all"

#: Brackets entered. Multi-entry is supported (recommendation 13); every
#: published figure is a single entry.
CANONICAL_N_ENTRIES: int = 1

# The identity, asserted at import so a future edit that breaks it fails
# immediately and loudly rather than producing quietly incomparable numbers.
assert CANONICAL_POOL_SIZE == CANONICAL_N_OPPONENTS + CANONICAL_N_ENTRIES, (
    f"canonical contract is inconsistent: {CANONICAL_N_OPPONENTS} opponents + "
    f"{CANONICAL_N_ENTRIES} entries != pool size {CANONICAL_POOL_SIZE}"
)


def as_backtest_kwargs(**overrides: Any) -> Dict[str, Any]:
    """The contract as ``run_backtest`` keyword arguments.

    Use this instead of spelling the contract out at each call site, so a
    measurement script cannot drift from the published one by forgetting a
    parameter. Pass ``overrides`` for a deliberate departure -- which then
    reads as a departure at the call site, rather than looking like the
    contract.
    """
    kwargs: Dict[str, Any] = {
        "n_opponents": CANONICAL_N_OPPONENTS,
        "n_repeats": CANONICAL_N_REPEATS,
        "opponent_source": CANONICAL_OPPONENT_SOURCE,
        "team_identity": CANONICAL_TEAM_IDENTITY,
        "pa_trials": CANONICAL_PA_TRIALS,
    }
    kwargs.update(overrides)
    return kwargs


def describe() -> str:
    """One line for run headers and artifact provenance."""
    return (
        f"{CANONICAL_POOL_SIZE}-person pool ({CANONICAL_N_OPPONENTS} opponents), "
        f"{CANONICAL_N_REPEATS} repeats, {CANONICAL_OPPONENT_SOURCE} opponents, "
        f"{'team-identity' if CANONICAL_TEAM_IDENTITY else 'shape'} scoring, "
        f"pa_trials={CANONICAL_PA_TRIALS}, payout={CANONICAL_PAYOUT}"
    )


def is_comparable_to_headline(n_opponents: int, n_entries: int = 1) -> bool:
    """Whether a run at this field size can be quoted against the headline.

    Pool size is part of the claim, not a detail. A run at 30 opponents is a
    31-person pool and its P(1st) is not the published quantity -- close
    enough to look right, which is exactly what makes it dangerous.
    """
    return n_opponents == CANONICAL_N_OPPONENTS and n_entries == CANONICAL_N_ENTRIES
