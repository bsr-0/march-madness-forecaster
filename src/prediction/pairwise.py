"""The pairwise probability contract.

This module defines the *one legal direction* of probability flow in this
codebase:

    pairwise probabilities
        -> tournament simulator
        -> simulation outcomes
        -> marginal advancement probabilities

and forbids the reverse:

    marginal advancement probabilities
        -> "pairwise" probabilities            <-- INVALID, DO NOT DO THIS

Why the reverse is invalid
--------------------------
``round_probs[team][R]`` is the *marginal* probability that ``team`` wins its
round-R game — the joint probability of reaching R **and** winning R. Several
call sites used to reconstruct a head-to-head probability from two marginals as
``p1 / (p1 + p2)``. Decompose ``p_i = r_i * w_i``, where ``r_i`` is the
probability that team *i* reaches the slot and ``w_i`` is its average win
probability there. Then::

    p1 / (p1 + p2)  =  r1*w1 / (r1*w1 + r2*w2)

The reach terms ``r1, r2`` never cancel, so the expression is **not**
``P(t1 beats t2)``. It is ``P(t1 wins this slot | one of {t1,t2} wins it)``,
which is a different quantity as soon as more than two teams can reach the slot.

In R64 the two coincide, because both teams reach with certainty (``r1 = r2 = 1``)
and exactly one of them wins, so ``p1 + p2 = 1``. From R32 onward four or more
teams contest each slot, ``p1 + p2 < 1``, and the approximation degrades. Because
the favorite has both a larger ``r`` and a larger ``w``, its edge gets multiplied
rather than isolated, so the error is one-directional — always toward chalk — and
grows with round depth:

    round   mean |error|   mean signed error
    R64        0.0008           0.0001
    R32        0.0708           0.0708
    S16        0.1118           0.1118
    E8         0.1169           0.1169
    F4         0.1283           0.1278
    CHAMP      0.1358           0.1350

See ``tests/test_pairwise_contract.py::test_marginal_ratio_is_invalid`` which
pins these numbers, and ``ARCHITECTURE_AUDIT_PREFERENCE_BRACKETS.md`` §3.

What marginals ARE valid for
----------------------------
Marginals are the mathematically correct quantity for **expected score**. Under
team-identity scoring you earn ``pts_R`` for each team you named at round R that
actually won a round-R game, so by linearity of expectation::

    E[points] = sum_R pts_R * sum_{t in picked_R} marginal[t][R]

exactly, with no independence assumption. So ``_make_ev_scorer`` and
``_compute_expected_points`` in ``bracket_construction.py`` are correct as
written and are deliberately NOT changed by this contract. The contract governs
one thing only: never manufacture a *pairwise* probability out of *marginals*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ROUND_NAMES",
    "MissingPairwiseSource",
    "PairwiseProbabilities",
    "ProbabilityBase",
    "log5",
    "marginals_from_pairwise",
    "simulate_bracket_outcomes",
]


class MissingPairwiseSource(RuntimeError):
    """Raised when code needs a pairwise probability and only marginals exist.

    This is the enforcement point for the contract. Historically such call
    sites silently fabricated ``p1 / (p1 + p2)`` from marginals; now they fail
    loudly instead. If you hit this, the fix is to supply a genuine pairwise
    source for the probability base, not to re-derive one from round_probs.
    """


ROUND_NAMES: Tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

# Probabilities are stored as plain floats; these bounds only guard against
# log(0) / division blowups, they are not a modelling choice.
_NUMERICAL_FLOOR = 1e-12


def log5(rating_a: float, rating_b: float) -> float:
    """P(A beats B) from two "win rate vs average team" ratings (Bill James).

    This is the canonical definition for the project. ``mc_pool_backtest._log5``
    delegates here so there is exactly one implementation.
    """
    pa, pb = rating_a, rating_b
    num = pa * (1.0 - pb)
    denom = pa * (1.0 - pb) + pb * (1.0 - pa)
    if denom < _NUMERICAL_FLOOR:
        return 0.5
    return num / denom


@dataclass(frozen=True)
class PairwiseProbabilities:
    """A validated table of ``P(t1 beats t2)`` for every ordered pair.

    This is the *primary* probability object in the pipeline. Marginal round
    probabilities are derived from it (via :func:`marginals_from_pairwise`),
    never the other way around.

    ``source`` is a free-text provenance label (e.g. ``"log5(torvik_barthag)"``)
    carried through so that any artifact can say where its probabilities came
    from, and so that a generator/evaluator mismatch is visible rather than
    implicit.
    """

    probs: Mapping[Tuple[str, str], float]
    source: str

    def p(self, t1: str, t2: str, default: float = 0.5) -> float:
        """P(t1 beats t2). Falls back to ``default`` for unknown pairs."""
        v = self.probs.get((t1, t2))
        if v is None:
            rev = self.probs.get((t2, t1))
            if rev is None:
                return default
            return 1.0 - rev
        return v

    def as_dict(self) -> Dict[Tuple[str, str], float]:
        """Plain dict for the existing ``matchup_probs`` call sites."""
        return dict(self.probs)

    @property
    def teams(self) -> List[str]:
        seen: Dict[str, None] = {}
        for a, b in self.probs:
            seen.setdefault(a, None)
            seen.setdefault(b, None)
        return list(seen)

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_ratings(
        cls,
        ratings: Mapping[str, float],
        source: str,
        default_rating: float = 0.5,
    ) -> "PairwiseProbabilities":
        """Build from "win rate vs average" ratings (barthag and friends) via log5.

        Nearly every probability base in this project is a barthag-equivalent
        rating vector, so this is the common path.
        """
        probs: Dict[Tuple[str, str], float] = {}
        team_ids = list(ratings.keys())
        for t1, t2 in combinations(team_ids, 2):
            p = log5(ratings.get(t1, default_rating), ratings.get(t2, default_rating))
            probs[(t1, t2)] = p
            probs[(t2, t1)] = 1.0 - p
        return cls(probs=probs, source=source)

    @classmethod
    def from_dict(
        cls,
        probs: Mapping[Tuple[str, str], float],
        source: str,
        validate: bool = True,
    ) -> "PairwiseProbabilities":
        """Wrap an existing pairwise dict (e.g. ``build_seed_probabilities``)."""
        obj = cls(probs=dict(probs), source=source)
        if validate:
            obj.validate()
        return obj

    # -- invariants -------------------------------------------------------

    def validate(self, tol: float = 1e-9) -> None:
        """Assert the two invariants a pairwise table must satisfy.

        1. Every probability lies in [0, 1].
        2. Antisymmetry: ``p(a,b) + p(b,a) == 1``.

        A table reconstructed from marginals satisfies (2) by construction —
        that is exactly why the bug was invisible for so long — so passing
        validation does NOT prove a table is legitimate. Provenance
        (``source``) is what distinguishes a real pairwise table from a
        fabricated one.
        """
        for (a, b), v in self.probs.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"pairwise probability out of range: p({a},{b})={v}")
            rev = self.probs.get((b, a))
            if rev is not None and abs(v + rev - 1.0) > tol:
                raise ValueError(f"pairwise table is not antisymmetric: p({a},{b})={v}, p({b},{a})={rev}")


class ProbabilityBase(_MappingABC):
    """A probability base: marginal round probs bundled with their pairwise source.

    Behaves as a ``Mapping[team_id, Dict[round_name, float]]`` so every existing
    call site that reads marginals (``base[t]["F4"]``, ``base.keys()``,
    ``for tid in base``, ``dict(base)``) keeps working unchanged. The addition is
    :attr:`pairwise` — the head-to-head table the marginals were derived from.

    The point of the bundle is that a base can no longer be passed around with
    its marginals separated from its pairwise source. Code that samples a game
    winner asks for ``base.pairwise``; code that computes expected points reads
    the marginals directly. Neither can be silently substituted for the other.

    ``pairwise`` may be ``None`` for bases that genuinely have no head-to-head
    model (e.g. ``contrarian``, which is a marginal-space adjustment of another
    base, or ``pool_wisdom``, which is an empirical ownership table). Accessing
    :attr:`pairwise` on such a base raises :class:`MissingPairwiseSource` rather
    than fabricating one.
    """

    __slots__ = ("name", "round_probs", "_pairwise")

    def __init__(
        self,
        name: str,
        round_probs: Mapping[str, Mapping[str, float]],
        pairwise: Optional[PairwiseProbabilities] = None,
    ) -> None:
        self.name = name
        self.round_probs = round_probs
        self._pairwise = pairwise

    # -- Mapping interface over the MARGINALS (the legacy read path) -------

    def __getitem__(self, team_id: str):
        return self.round_probs[team_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self.round_probs)

    def __len__(self) -> int:
        return len(self.round_probs)

    def __repr__(self) -> str:
        src = self._pairwise.source if self._pairwise is not None else "<none>"
        return f"ProbabilityBase(name={self.name!r}, n_teams={len(self)}, pairwise={src!r})"

    # -- the pairwise side -------------------------------------------------

    @property
    def pairwise(self) -> PairwiseProbabilities:
        if self._pairwise is None:
            raise MissingPairwiseSource(
                f"probability base {self.name!r} has no pairwise source, so it cannot be "
                "used to sample game winners. Marginal round probabilities must not be "
                "converted into head-to-head probabilities — see src/prediction/pairwise.py."
            )
        return self._pairwise

    @property
    def has_pairwise(self) -> bool:
        return self._pairwise is not None


def simulate_bracket_outcomes(
    pairwise: PairwiseProbabilities,
    first_round_matchups: Sequence[str],
    n_sims: int,
    rng: np.random.Generator,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, List[List[List[str]]]]:
    """Propagate a bracket ``n_sims`` times from pairwise probabilities.

    This is the sanctioned propagation primitive: it is the only way to turn
    pairwise probabilities into tournament-level quantities.

    Args:
        pairwise: the probability source.
        first_round_matchups: 64 team_ids in bracket order (game g is
            ``[2g], [2g+1]``).
        n_sims: number of tournaments to simulate.
        rng: NumPy generator (caller owns the seed — see the common-random-
            numbers note in the audit).
        noise_std: logit-space per-game noise. Default 0.0 gives pure
            propagation of the stated probabilities, which is what
            marginalization wants. Pool simulation uses 0.16.

    Returns:
        ``(outcomes, outcomes_by_round)`` where ``outcomes`` is an
        ``(n_sims, 63)`` bool array (True = first-listed team won) and
        ``outcomes_by_round[sim][round_idx]`` is the list of that round's
        winner ids. Index 0 is the R64 winners (i.e. the teams that reach
        R32); index 5 is the champion.

    Note this returns the *per-simulation* outcomes rather than aggregating
    them away. Conditional scenario work needs the full matrix; callers that
    only want marginals should use :func:`marginals_from_pairwise`.
    """
    outcomes = np.zeros((n_sims, 63), dtype=bool)
    outcomes_by_round: List[List[List[str]]] = []

    for sim in range(n_sims):
        current = list(first_round_matchups)
        game_idx = 0
        sim_rounds: List[List[str]] = []

        for _round_idx in range(6):
            next_teams: List[str] = []
            round_winners: List[str] = []

            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    next_teams.append(current[g])
                    continue

                t1, t2 = current[g], current[g + 1]
                p = pairwise.p(t1, t2)

                if noise_std > 0.0:
                    safe = min(max(p, 0.001), 0.999)
                    logit = math.log(safe / (1.0 - safe)) + rng.normal(0, noise_std)
                    p = 1.0 / (1.0 + math.exp(-logit))
                    p = min(max(p, 0.01), 0.99)

                if rng.random() < p:
                    outcomes[sim, game_idx] = True
                    winner = t1
                else:
                    outcomes[sim, game_idx] = False
                    winner = t2

                next_teams.append(winner)
                round_winners.append(winner)
                game_idx += 1

            sim_rounds.append(round_winners)
            current = next_teams

        outcomes_by_round.append(sim_rounds)

    return outcomes, outcomes_by_round


def marginals_from_pairwise(
    pairwise: PairwiseProbabilities,
    first_round_matchups: Sequence[str],
    team_ids: Iterable[str],
    n_sims: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    seed: int = 42,
    floor: float = 0.001,
) -> Dict[str, Dict[str, float]]:
    """Derive marginal round-advancement probabilities by simulation.

    THE ONLY legal way to obtain ``round_probs``. ``round_probs[t][R]`` is
    ``P(t wins its round-R game)``.

    Args:
        floor: minimum reported probability, so downstream log/ratio code
            cannot hit zero. Matches the historical behaviour of
            ``build_torvik_round_probabilities``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    team_ids = list(team_ids)
    counts: Dict[str, Dict[str, int]] = {t: {r: 0 for r in ROUND_NAMES} for t in team_ids}

    for _ in range(n_sims):
        current = list(first_round_matchups)
        for round_name in ROUND_NAMES:
            next_teams: List[str] = []
            for g in range(0, len(current), 2):
                if g + 1 >= len(current):
                    next_teams.append(current[g])
                    continue
                t1, t2 = current[g], current[g + 1]
                winner = t1 if rng.random() < pairwise.p(t1, t2) else t2
                if winner in counts:
                    counts[winner][round_name] += 1
                next_teams.append(winner)
            current = next_teams

    return {t: {r: max(floor, counts[t][r] / n_sims) for r in ROUND_NAMES} for t in team_ids}
