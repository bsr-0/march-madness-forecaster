"""The one bracket every layer walks.

A 64-team tournament is a positional tree: game g of the first round is
``first_round[2g]`` vs ``first_round[2g+1]``, winners are appended in order,
and the Final Four pairs the E8 winners of regions at positions 0/1 and 2/3.
Which two regions meet in each semifinal is decided by the NCAA every year and
announced with the bracket; it is NOT fixed.

Until the 2026-09 methodology audit (Step 3, findings F3-1 / F3-2) three
layers disagreed about that pairing:

  * the referee and the ground truth used the real per-season pairing
    (``derive_f4_region_pairing``, from the played F4 games);
  * the marginalizers (``build_torvik_round_probabilities``,
    ``build_bracket_order``, the candidate artifact) hardcoded
    East-West / South-Midwest;
  * construction (``construct_bracket``) hardcoded the same and emitted picks
    keyed ``F4_East_West`` / ``F4_South_Midwest``, which the backtest then
    projected onto the real tree by round-winner SET with a silent ``t2``
    fallback. When both F4 picks fell in the same real semifinal the scored
    bracket carried a champion construction never chose (2015: Kentucky
    scored as Virginia; 2023: Houston as Tennessee).

The real pairing differs from the hardcoded one in 9 of the 15 backtest
seasons, 2026 included. This module is the single source of truth. Every
producer of a bracket order, every marginalizer and every projection takes an
explicit ``region_order`` and there is no silent default in production code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Labels only. Never a production fallback -- see resolve_region_order().
DEFAULT_REGION_ORDER: Tuple[str, str, str, str] = ("East", "West", "South", "Midwest")
SEED_MATCHUP_ORDER: Tuple[Tuple[int, int], ...] = ((1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15))
ROUND_NAMES: Tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "CHAMP")


class TopologyUnavailable(ValueError):
    """No source for the Final Four pairing: no F4 games played and no
    ``f4_pairing`` in the season's seeds block."""


class TopologyMismatch(ValueError):
    """A picks dict does not describe a bracket on the given tree: some game
    has neither or both of its teams picked as that round's winner."""


def derive_f4_region_pairing(games, regions) -> Tuple[str, str, str, str]:
    """Region order whose positional tree reproduces the real F4 games.

    Reads the two ``round_name == "F4"`` games. The first two regions in the
    result met in the first F4 game, the last two in the other. Raises if
    fewer than two F4 games exist, a team's region is unresolved, a game has
    two teams from one region, or the pairs do not cover four distinct regions.
    Only usable once the F4 has been played; before that use the season's
    ``f4_pairing`` field through :func:`resolve_region_order`.
    """
    f4_games = [g for g in games if g.get("round_name") == "F4"]
    if len(f4_games) < 2:
        raise ValueError(f"expected 2 F4 games to derive region pairing, got {len(f4_games)}")
    pairs = []
    for g in f4_games[:2]:
        t1, t2 = g["team1_id"], g["team2_id"]
        r1, r2 = regions.get(t1), regions.get(t2)
        if not r1 or not r2:
            raise ValueError(f"could not resolve regions for F4 game {t1} vs {t2}: {r1!r} vs {r2!r}")
        if r1 == r2:
            raise ValueError(f"F4 game has two teams from the same region ({r1}): {t1} vs {t2}")
        pairs.append((r1, r2))
    if len({r for pair in pairs for r in pair}) != 4:
        raise ValueError(f"F4 pairs do not cover 4 distinct regions: pairs={pairs}")
    return (pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])


def _same_pairing(a: Sequence[str], b: Sequence[str]) -> bool:
    return {frozenset(a[:2]), frozenset(a[2:])} == {frozenset(b[:2]), frozenset(b[2:])}


def resolve_region_order(
    year: int,
    *,
    games: Optional[Iterable[Mapping]] = None,
    regions: Optional[Mapping[str, str]] = None,
    seeds_block: Optional[Mapping] = None,
) -> Tuple[str, str, str, str]:
    """The Final Four pairing for ``year``, from data, never from a default.

    Sources, in order:
      1. played F4 games (``games`` + ``regions``) -> :func:`derive_f4_region_pairing`;
      2. ``seeds_block["f4_pairing"]`` = ``[[A, B], [C, D]]``, the pairing as
         announced on Selection Sunday -- the only source for a prospective
         season.
    If both are present they must agree (as unordered pairs); a disagreement is
    a data error and raises. If neither is present, raises
    :class:`TopologyUnavailable` rather than guessing.
    """
    derived: Optional[Tuple[str, str, str, str]] = None
    if games is not None and regions is not None:
        f4 = [g for g in games if g.get("round_name") == "F4"]
        if len(f4) >= 2:
            derived = derive_f4_region_pairing(games, regions)

    declared: Optional[Tuple[str, str, str, str]] = None
    fp = (seeds_block or {}).get("f4_pairing")
    if fp is not None:
        if (
            len(fp) != 2
            or any(len(p) != 2 for p in fp)
            or len({r for p in fp for r in p}) != 4
        ):
            raise ValueError(f"{year}: seeds.f4_pairing must be two pairs of four distinct regions, got {fp!r}")
        declared = (fp[0][0], fp[0][1], fp[1][0], fp[1][1])

    if derived is not None and declared is not None and not _same_pairing(derived, declared):
        raise ValueError(f"{year}: seeds.f4_pairing {declared} disagrees with the played F4 games {derived}")
    if derived is not None:
        return derived
    if declared is not None:
        return declared
    raise TopologyUnavailable(
        f"{year}: no Final Four games on disk and no `f4_pairing` in the seeds block. "
        f"Add seeds.f4_pairing = [[A, B], [C, D]] from the announced bracket."
    )


def build_bracket_order(seeds: Mapping[str, int], regions: Mapping[str, str], *, region_order: Sequence[str]) -> List[str]:
    """The 64 team_ids in positional bracket order for ``region_order``.

    Raises if any (region, seed) slot holds more than one team -- an
    unresolved play-in game; call ``resolve_first_four`` first. A slot with no
    team is filled with an ``unknown_<region>_<seed>`` placeholder so the list
    is always 64 long.
    """
    if region_order is None or len(region_order) != 4 or len(set(region_order)) != 4:
        raise ValueError(f"region_order must be 4 distinct regions, got {region_order!r}; use resolve_region_order()")
    region_teams: Dict[str, Dict[int, str]] = {r: {} for r in region_order}
    contested: Dict[Tuple[str, int], List[str]] = {}
    for tid, seed in seeds.items():
        r = regions.get(tid, "")
        if r not in region_teams:
            continue
        if seed in region_teams[r]:
            contested.setdefault((r, seed), [region_teams[r][seed]]).append(tid)
        region_teams[r][seed] = tid
    if contested:
        detail = "; ".join(f"{r} {seed}: {sorted(t)}" for (r, seed), t in sorted(contested.items()))
        raise ValueError(
            f"{len(contested)} bracket slot(s) still hold more than one team, so the draw is not "
            f"determined: {detail}. Resolve the play-in games first (resolve_first_four)."
        )
    order: List[str] = []
    for region in region_order:
        rt = region_teams[region]
        for high, low in SEED_MATCHUP_ORDER:
            order.extend([rt.get(high, f"unknown_{region}_{high}"), rt.get(low, f"unknown_{region}_{low}")])
    return order


def f4_game_keys(region_order: Sequence[str]) -> Tuple[str, str]:
    """Picks-dict keys for the two semifinals under ``region_order``."""
    return (f"F4_{region_order[0]}_{region_order[1]}", f"F4_{region_order[2]}_{region_order[3]}")


def picks_to_winners_by_round(picks: Mapping[str, str], first_round: Sequence[str]) -> List[List[str]]:
    """Walk ``first_round`` and read each game's winner out of ``picks``.

    STRICT. At every game exactly one of the two teams must be in that
    round's pick set; otherwise the picks were built on a different tree and
    :class:`TopologyMismatch` is raised. This replaces the old set-membership
    projection whose ``else: t2`` branch silently invented winners.
    """
    if len(first_round) != 64:
        raise ValueError(f"first_round must have 64 teams, got {len(first_round)}")
    by_round: Dict[str, set] = defaultdict(set)
    for key, winner in picks.items():
        by_round[key.split("_", 1)[0]].add(winner)
    current = list(first_round)
    winners_by_round: List[List[str]] = []
    for round_name in ROUND_NAMES:
        picked = by_round[round_name]
        nxt: List[str] = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            a, b = t1 in picked, t2 in picked
            if a == b:
                raise TopologyMismatch(
                    f"{round_name} game {t1} vs {t2}: {'both' if a else 'neither'} picked as a {round_name} winner; "
                    f"the picks were built on a different bracket topology than first_round"
                )
            nxt.append(t1 if a else t2)
        winners_by_round.append(nxt)
        current = nxt
    return winners_by_round


def winners_to_bool_vector(winners_by_round: Sequence[Sequence[str]], first_round: Sequence[str]) -> np.ndarray:
    """(63,) bool vector, True = first-listed team won, in walk order."""
    result = np.zeros(63, dtype=bool)
    current = list(first_round)
    gi = 0
    for r in range(6):
        winners = list(winners_by_round[r])
        nxt: List[str] = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            w = winners[g // 2]
            if w == t1:
                result[gi] = True
            elif w != t2:
                raise TopologyMismatch(f"round {r} game {t1} vs {t2}: winner {w} is neither")
            nxt.append(w)
            gi += 1
        current = nxt
    return result


def picks_to_bool_vector(picks: Mapping[str, str], first_round: Sequence[str]) -> np.ndarray:
    """Strict picks -> (63,) vector; the drop-in for the old projection."""
    return winners_to_bool_vector(picks_to_winners_by_round(picks, first_round), first_round)


def region_order_from_first_round(first_round: Sequence[str], regions: Mapping[str, str]) -> Tuple[str, str, str, str]:
    """Recover the region order a positional 64-team list was built with.

    Positions 0, 16, 32, 48 are the 1-seeds of the four regions in order. Lets
    a sampler that is handed only ``first_round`` build picks on the same tree.
    """
    if len(first_round) != 64:
        raise ValueError(f"first_round must have 64 teams, got {len(first_round)}")
    order = tuple(regions[first_round[i]] for i in (0, 16, 32, 48))
    if len(set(order)) != 4:
        raise ValueError(f"first_round does not have four distinct regions at its quarter boundaries: {order}")
    return order  # type: ignore[return-value]


def winner_sets_to_bool_vector(winner_sets: Sequence[Iterable[str]], first_round: Sequence[str]) -> np.ndarray:
    """Strict projection from per-round winner SETS (unordered) to the (63,) vector.

    At every game exactly one of the two teams must be in that round's set,
    else :class:`TopologyMismatch`. Replaces the set-membership encoders whose
    ``else: t2`` branch silently invented a winner when the sets did not
    describe a bracket on this tree (2026-09 audit, Step 4, F4-7).
    """
    if len(first_round) != 64:
        raise ValueError(f"first_round must have 64 teams, got {len(first_round)}")
    picked = [set(r) for r in winner_sets]
    if len(picked) != 6:
        raise ValueError(f"expected 6 rounds of winners, got {len(picked)}")
    result = np.zeros(63, dtype=bool)
    current = list(first_round)
    gi = 0
    for r in range(6):
        nxt: List[str] = []
        for g in range(0, len(current), 2):
            t1, t2 = current[g], current[g + 1]
            a, b = t1 in picked[r], t2 in picked[r]
            if a == b:
                raise TopologyMismatch(
                    f"round {r} game {t1} vs {t2}: {'both' if a else 'neither'} in the round's winner set"
                )
            result[gi] = a
            nxt.append(t1 if a else t2)
            gi += 1
        current = nxt
    return result
