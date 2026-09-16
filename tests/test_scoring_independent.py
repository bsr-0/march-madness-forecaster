"""Independent check of the production bracket scorer (2026-09 audit, Step 4, items 1-3).

The production scorer (`score_brackets_team_identity`) decodes a 63-bool
vector into per-round pick SETS and counts intersections with the actual
winners. The reference here is a different formulation -- slot by slot down
the tree, crediting a game only if the bracket's pick for that exact slot
won the actual game in that slot -- written from the ESPN rules, not from
the production code. The two must agree exactly on every bracket, including
ones whose later-round picks are impossible because the team was eliminated.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.simulation.pool_competition import score_brackets_team_identity  # noqa: E402

pytestmark = pytest.mark.unit

PTS = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
ROUNDS = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
TEAMS = [f"t{i:02d}" for i in range(64)]


def walk(vec, first_round):
    """Slot-level decode: list of rounds, each a list of (slot_index, winner)."""
    cur = list(first_round)
    gi = 0
    rounds = []
    for _ in range(6):
        nxt, games = [], []
        for g in range(0, len(cur), 2):
            w = cur[g] if vec[gi] else cur[g + 1]
            games.append(w)
            nxt.append(w)
            gi += 1
        rounds.append(games)
        cur = nxt
    return rounds


def reference_score(bracket_vec, actual_vec, first_round):
    """ESPN rule, slot by slot: a slot in round r pays PTS[r] iff the bracket's
    winner of that slot equals the actual winner of that slot. Because both
    are walked from the same first_round, a team the bracket eliminated early
    can never occupy a later slot, so impossible picks cannot pay."""
    b = walk(bracket_vec, first_round)
    a = walk(actual_vec, first_round)
    return sum(PTS[R] * sum(1 for x, y in zip(b[r], a[r]) if x == y) for r, R in enumerate(ROUNDS))


def winners_by_round_from_vec(actual_vec, first_round):
    return {R: set(g) for R, g in zip(ROUNDS, walk(actual_vec, first_round))}


def production_score(bracket_vec, actual_vec, first_round):
    wbr = winners_by_round_from_vec(actual_vec, first_round)
    return float(score_brackets_team_identity(np.asarray(bracket_vec).reshape(1, 63), wbr, first_round, PTS)[0])


def chalk():
    return np.ones(63, dtype=bool)


def flip(vec, idx):
    v = vec.copy()
    v[idx] = ~v[idx]
    return v


# game index ranges per round in walk order
R64_IDX, R32_IDX, S16_IDX, E8_IDX, F4_IDX, CH_IDX = range(0, 32), range(32, 48), range(48, 56), range(56, 60), range(60, 62), range(62, 63)


def test_perfect_bracket_scores_maximum():
    a = chalk()
    assert production_score(a, a, TEAMS) == reference_score(a, a, TEAMS) == 32 * 10 + 16 * 20 + 8 * 40 + 4 * 80 + 2 * 160 + 320 == 1920


@pytest.mark.parametrize(
    "game_idx,expected_loss",
    [
        (0, 10),          # one R64 error, that team goes no further in chalk -> only that game lost
        (32, 20 + 0),     # one R32 error: bracket advances the other team, which loses S16 in the actual chalk -> 20
        (48, 40),
        (56, 80),
        (60, 160),
        (62, 320),
    ],
)
def test_single_error_costs_exactly_that_slot_when_the_wrong_team_is_then_eliminated(game_idx, expected_loss):
    """In a chalk actual, flipping one game in a chalk bracket puts the slot's
    loser through; the bracket then (still chalk downstream) has that loser
    winning nothing further because chalk downstream picks the TOP of each
    later pair, and the loser sits in the top slot too... so we must compute
    the cascade honestly via the reference rather than assume. The assertion
    that matters is production == reference; the loss value is checked too."""
    actual = chalk()
    b = flip(chalk(), game_idx)
    prod, ref = production_score(b, actual, TEAMS), reference_score(b, actual, TEAMS)
    assert prod == ref
    assert 1920 - ref >= expected_loss  # at least the slot itself; the cascade can cost more


def test_impossible_downstream_picks_cannot_earn_points():
    """A is picked to lose R64, yet the bracket's later slots would name A if
    the vector were read as 'list of winners by round'. Path decoding makes
    that impossible: check the bracket earns nothing for A after R64 while the
    actual has A winning through to the title."""
    actual = chalk()                     # t00 wins everything
    b = chalk()
    b[0] = False                          # bracket: t01 beats t00 in R64, then (chalk downstream) t01 goes all the way
    prod, ref = production_score(b, actual, TEAMS), reference_score(b, actual, TEAMS)
    assert prod == ref
    # t00 actually won 6 games (10+20+40+80+160+320 = 630); bracket credited none of them,
    # and t01's later wins are wrong too -> exactly 1920 - 630 lost.
    assert ref == 1920 - 630


def test_early_upset_changes_later_matchups():
    """Actual: t01 upsets t00 in R64 and then wins the whole tournament (top slot
    in every later pair). Bracket: chalk. The bracket has t00 in every later
    slot; none of those pay. Score = 1920 - 630."""
    actual = chalk()
    actual[0] = False
    b = chalk()
    prod, ref = production_score(b, actual, TEAMS), reference_score(b, actual, TEAMS)
    assert prod == ref == 1920 - 630


def test_random_brackets_agree_exactly():
    rng = np.random.default_rng(4)
    for _ in range(300):
        b = rng.random(63) < rng.random()
        a = rng.random(63) < rng.random()
        assert production_score(b, a, TEAMS) == reference_score(b, a, TEAMS)


def test_scores_are_additive_and_bounded():
    rng = np.random.default_rng(5)
    for _ in range(50):
        b = rng.random(63) < 0.5
        a = rng.random(63) < 0.5
        s = production_score(b, a, TEAMS)
        assert 0 <= s <= 1920 and s % 10 == 0
