"""Guards the exact bug found building the audit C2/recommendation-11 check.

`_round_winners_to_bool_vector` reconstructs a (63,) picks vector from
{round_name: [teams advancing]}, which requires the `first_round` list's
bracket-tree topology (which regions meet in the Final Four) to match
whatever topology produced the round-winner sets. Built against the default
`REGION_ORDER` instead of the real year's F4 pairing (`derive_f4_region_pairing`),
2015's saved bracket raised "neither villanova nor wisconsin ... F4 game 2"
-- silently WOULD have mis-scored every year where the real pairing differs
from the default, had the mismatch not raised.
"""

from __future__ import annotations

import pytest

from scripts.independent_referee_check import _round_winners_to_bool_vector


def _walk_to_champion(first_round, winner_of_game) -> dict:
    """Build {round_name: [advancing teams]} from a per-game winner rule.

    `winner_of_game(t1, t2) -> t1 or t2`, applied round by round -- the
    reference implementation this test's fixtures are built from.
    """
    from scripts.mc_pool_backtest import ROUND_NAMES

    picks = {}
    current = list(first_round)
    for round_name in ROUND_NAMES:
        winners = []
        for g in range(0, len(current), 2):
            winners.append(winner_of_game(current[g], current[g + 1]))
        picks[round_name] = winners
        current = winners
    return picks


def _fake_first_round(n=64):
    return [f"t{i}" for i in range(n)]


def test_reconstructs_a_consistent_bracket_exactly():
    """Lower-indexed team always wins -- picks derive from the same rule the
    reconstruction is asked to invert."""
    first_round = _fake_first_round()
    picks = _walk_to_champion(first_round, lambda a, b: a if int(a[1:]) < int(b[1:]) else b)

    result = _round_winners_to_bool_vector(first_round, picks)

    assert result.shape == (63,)
    assert result[0]  # t0 beats t1 in game 0
    assert picks["CHAMP"] == ["t0"], "sanity check on the fixture itself"


def test_wrong_bracket_topology_raises_rather_than_mis_scores():
    """The exact failure mode found for 2015: a first_round whose tree
    topology (which quarter meets which in the Final Four) does not match
    the one that produced the round-winner sets must raise, not silently
    attribute a win to the wrong game."""
    first_round = _fake_first_round()
    picks = _walk_to_champion(first_round, lambda a, b: a if int(a[1:]) < int(b[1:]) else b)

    # Pair the two R64 LOSERS (t1 and t3, each beaten by its lower-indexed
    # teammate) against each other. A whole-list reverse, a whole-pair swap,
    # or re-pairing a loser against a winner all leave at least one genuine
    # round-winner in every synthetic game -- membership passes by
    # coincidence even though the "winner" attributed is meaningless. Two
    # losers paired together is the one case with no legitimate winner in
    # either team, which is what must be caught.
    scrambled = list(first_round)
    scrambled[1], scrambled[2] = scrambled[2], scrambled[1]

    with pytest.raises(ValueError, match="neither .* in"):
        _round_winners_to_bool_vector(scrambled, picks)


def test_champion_is_recoverable_end_to_end():
    """The last game's winner (result[62]) must equal the recorded champion,
    round-tripping the whole 6-round walk, not just an early game."""
    first_round = _fake_first_round()
    picks = _walk_to_champion(first_round, lambda a, b: a if int(a[1:]) % 2 == 0 else b)

    result = _round_winners_to_bool_vector(first_round, picks)

    champ = picks["CHAMP"][0]
    finalists = picks["F4"]
    winning_side = finalists[0] if champ == finalists[0] else finalists[1]
    assert winning_side == champ
    assert result[62] == (champ == finalists[0])
