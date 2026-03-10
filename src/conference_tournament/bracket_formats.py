"""Conference tournament bracket format definitions.

Defines the actual bracket structure for each D1 conference tournament.
Many conferences use more rounds than the mathematical minimum
(ceil(log2(N))), giving top seeds additional byes.  The most notable
example is the ACC (15 teams, 5 rounds instead of 4).

Each ``BracketFormat`` specifies which seeds enter the tournament at
which round.  Seeds entering in round 1 play immediately; seeds
entering in round 2 have a single bye; round 3 = double bye, etc.

Usage::

    from src.conference_tournament.bracket_formats import get_bracket_format

    fmt = get_bracket_format("ACC", 15)
    # fmt.total_rounds == 5
    # fmt.round_entry == {1: [10,11,12,13,14,15], 2: [5,6,7,8,9], 3: [1,2,3,4]}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BracketFormat:
    """Bracket format definition for a conference tournament.

    Attributes:
        num_teams: Number of teams in the tournament.
        total_rounds: Total number of rounds (may exceed ceil(log2(N))).
        round_entry: Maps round number -> list of seeds that enter the
            tournament at that round.  Seeds entering round 1 play
            immediately.  Seeds entering round R have (R-1) byes.
        first_round_matchups: Explicit (seed_a, seed_b) pairings for
            round 1.  If None, pairs are generated as top-vs-bottom
            within the round-1 seed group.
    """

    num_teams: int
    total_rounds: int
    round_entry: Dict[int, Tuple[int, ...]]
    first_round_matchups: Optional[Tuple[Tuple[int, int], ...]] = None

    def seeds_entering_round(self, round_num: int) -> Tuple[int, ...]:
        """Return seeds that enter the tournament at the given round."""
        return self.round_entry.get(round_num, ())

    def num_byes_for_seed(self, seed: int) -> int:
        """Return how many byes a given seed receives."""
        for rnd, seeds in self.round_entry.items():
            if seed in seeds:
                return rnd - 1
        return 0

    def validate(self) -> List[str]:
        """Validate the format definition.  Returns list of errors."""
        errors = []

        # All seeds 1..num_teams must appear exactly once
        all_seeds = []
        for seeds in self.round_entry.values():
            all_seeds.extend(seeds)

        if sorted(all_seeds) != list(range(1, self.num_teams + 1)):
            errors.append(
                f"Seeds mismatch: expected 1..{self.num_teams}, "
                f"got {sorted(all_seeds)}"
            )

        # Round 1 seeds must pair evenly
        r1_seeds = self.round_entry.get(1, ())
        if len(r1_seeds) % 2 != 0:
            errors.append(
                f"Round 1 has {len(r1_seeds)} seeds (must be even)"
            )

        # Check first_round_matchups consistency
        if self.first_round_matchups:
            matchup_seeds = set()
            for a, b in self.first_round_matchups:
                matchup_seeds.add(a)
                matchup_seeds.add(b)
            if matchup_seeds != set(r1_seeds):
                errors.append(
                    f"first_round_matchups seeds {matchup_seeds} != "
                    f"round_entry[1] seeds {set(r1_seeds)}"
                )

        # Each subsequent round must have correct team count
        # Round R teams = new entrants + (round R-1 winners)
        teams_available = len(r1_seeds) // 2  # R1 winners
        for rnd in range(2, self.total_rounds + 1):
            new_seeds = self.round_entry.get(rnd, ())
            total = teams_available + len(new_seeds)
            if total % 2 != 0:
                errors.append(
                    f"Round {rnd}: {total} entrants (must be even). "
                    f"{teams_available} winners + {len(new_seeds)} new."
                )
            teams_available = total // 2

        # Final round should produce 1 winner
        if teams_available != 1:
            errors.append(
                f"Final round produces {teams_available} winners, expected 1"
            )

        # Total games should equal num_teams - 1
        total_games = 0
        current_teams = len(r1_seeds) // 2  # R1 games = R1 seeds / 2
        total_games += len(r1_seeds) // 2
        for rnd in range(2, self.total_rounds + 1):
            new_seeds = self.round_entry.get(rnd, ())
            entrants = current_teams + len(new_seeds)
            games = entrants // 2
            total_games += games
            current_teams = games
        if total_games != self.num_teams - 1:
            errors.append(
                f"Total games = {total_games}, expected {self.num_teams - 1}"
            )

        return errors


def _standard_format(num_teams: int) -> BracketFormat:
    """Generate a standard bracket format using the mathematical minimum.

    Uses ceil(log2(N)) rounds and 2^rounds - N byes for top seeds.
    When there are no byes, first_round_matchups is left as None so the
    bracket builder applies ``bracket_seed_order`` for correct bracket
    positioning (guaranteeing #1 and #2 on opposite halves).

    When there ARE byes, matchups are top-vs-bottom within the R1 group,
    since the proper bracket ordering is established when bye teams merge
    in round 2 via ``bracket_seed_order``.
    """
    total_rounds = math.ceil(math.log2(num_teams))
    full_size = 2 ** total_rounds
    num_byes = full_size - num_teams

    # Top seeds get byes, bottom seeds play in round 1
    bye_seeds = tuple(range(1, num_byes + 1))
    r1_seeds = tuple(range(num_byes + 1, num_teams + 1))

    round_entry: Dict[int, Tuple[int, ...]] = {1: r1_seeds}
    if bye_seeds:
        round_entry[2] = bye_seeds

    # Generate first-round matchups only when there are byes
    # (top-vs-bottom within R1 group).  Without byes, leave matchups
    # as None so the bracket builder uses bracket_seed_order.
    matchups = None
    if num_byes > 0:
        n_games = len(r1_seeds) // 2
        matchups = tuple(
            (r1_seeds[i], r1_seeds[len(r1_seeds) - 1 - i])
            for i in range(n_games)
        )

    return BracketFormat(
        num_teams=num_teams,
        total_rounds=total_rounds,
        round_entry=round_entry,
        first_round_matchups=matchups,
    )


# ──────────────────────────────────────────────────────────────────────
# Conference-specific format overrides
#
# Only conferences whose REAL format differs from the mathematical
# default need explicit entries here.  All others use _standard_format.
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# ACC: 15 teams, 5 rounds
# Seeds 1-4: double bye (enter at QF, round 3)
# Seeds 5-9: single bye (enter at round 2)
# Seeds 10-15: play in round 1
#
#   R1: 10v15, 11v14, 12v13
#   R2: 7vW(10/15), 6vW(11/14), 5vW(12/13), 8v9
#   QF: 1vW(8/9), 4vW(5/12/13), 2vW(7/10/15), 3vW(6/11/14)
#   SF: 2 games
#   F:  1 game
_ACC_15 = BracketFormat(
    num_teams=15,
    total_rounds=5,
    round_entry={
        1: (10, 11, 12, 13, 14, 15),
        2: (5, 6, 7, 8, 9),
        3: (1, 2, 3, 4),
    },
    first_round_matchups=((10, 15), (11, 14), (12, 13)),
)

# ──────────────────────────────────────────────────────────────────────
# Big Ten: 18 teams, 6 rounds (NOT the mathematical 5)
# Seeds 1-4: triple bye (enter at QF, round 4)
# Seeds 5-8: double bye (enter at round 3)
# Seeds 9-14: single bye (enter at round 2)
# Seeds 15-18: play in round 1
#
#   R1: 16v17, 15v18
#   R2: 9vW(16/17), 12v13, 10vW(15/18), 11v14
#   R3: 8vW(9/16/17), 5vW(12/13), 7vW(10/15/18), 6vW(11/14)
#   QF: 1vW(8), 4vW(5), 2vW(7), 3vW(6)
#   SF: 2 games
#   F:  1 game
_B10_18 = BracketFormat(
    num_teams=18,
    total_rounds=6,
    round_entry={
        1: (15, 16, 17, 18),
        2: (9, 10, 11, 12, 13, 14),
        3: (5, 6, 7, 8),
        4: (1, 2, 3, 4),
    },
    first_round_matchups=((16, 17), (15, 18)),
)

# ──────────────────────────────────────────────────────────────────────
# Big 12: 16 teams, 5 rounds (NOT the mathematical 4)
# Seeds 1-4: double bye (enter at QF, round 3)
# Seeds 5-8: single bye (enter at round 2)
# Seeds 9-16: play in round 1
#
#   R1: 12v13, 9v16, 10v15, 11v14
#   R2: 5vW(12/13), 8vW(9/16), 7vW(10/15), 6vW(11/14)
#   QF: 4vW(5), 1vW(8), 2vW(7), 3vW(6)
#   SF: 2 games
#   F:  1 game
_B12_16 = BracketFormat(
    num_teams=16,
    total_rounds=5,
    round_entry={
        1: (9, 10, 11, 12, 13, 14, 15, 16),
        2: (5, 6, 7, 8),
        3: (1, 2, 3, 4),
    },
    first_round_matchups=((12, 13), (9, 16), (10, 15), (11, 14)),
)

# ──────────────────────────────────────────────────────────────────────
# SEC: 16 teams, 5 rounds (NOT the mathematical 4)
# Same stepladder structure as Big 12
# Seeds 1-4: double bye (enter at QF, round 3)
# Seeds 5-8: single bye (enter at round 2)
# Seeds 9-16: play in round 1
#
#   R1: 16v9, 13v12, 15v10, 14v11
#   R2: 8vW(16/9), 5vW(13/12), 7vW(14/11), 6vW(15/10)
#   QF: 1vW(8), 4vW(5), 2vW(7), 3vW(6)
#   SF: 2 games
#   F:  1 game
_SEC_16 = BracketFormat(
    num_teams=16,
    total_rounds=5,
    round_entry={
        1: (9, 10, 11, 12, 13, 14, 15, 16),
        2: (5, 6, 7, 8),
        3: (1, 2, 3, 4),
    },
    first_round_matchups=((16, 9), (13, 12), (15, 10), (14, 11)),
)

# ──────────────────────────────────────────────────────────────────────
# American Athletic: 10 teams, 5 rounds (NOT the mathematical 4)
# Stepladder format:
# Seeds 1-2: triple bye (enter at SF, round 4)
# Seeds 3-4: double bye (enter at QF, round 3)
# Seeds 5-6: single bye (enter at round 2)
# Seeds 7-10: play in round 1
#
#   R1: 8v9, 7v10
#   R2: 5vW(8/9), 6vW(7/10)
#   QF: 4vW(5), 3vW(6)
#   SF: 1vW(4), 2vW(3)
#   F:  1 game
_AMER_10 = BracketFormat(
    num_teams=10,
    total_rounds=5,
    round_entry={
        1: (7, 8, 9, 10),
        2: (5, 6),
        3: (3, 4),
        4: (1, 2),
    },
    first_round_matchups=((8, 9), (7, 10)),
)

# ──────────────────────────────────────────────────────────────────────
# Atlantic 10: 14 teams, 5 rounds (NOT the mathematical 4)
# Seeds 1-4: double bye (enter at QF, round 3)
# Seeds 5-10: single bye (enter at round 2)
# Seeds 11-14: play in round 1
#
#   R1: 12v13, 11v14
#   R2: 8v9, 5vW(12/13), 7v10, 6vW(11/14)
#   QF: 1vW(8/9), 4vW(5), 2vW(7/10), 3vW(6)
#   SF: 2 games
#   F:  1 game
_A10_14 = BracketFormat(
    num_teams=14,
    total_rounds=5,
    round_entry={
        1: (11, 12, 13, 14),
        2: (5, 6, 7, 8, 9, 10),
        3: (1, 2, 3, 4),
    },
    first_round_matchups=((12, 13), (11, 14)),
)

# ──────────────────────────────────────────────────────────────────────
# WCC: 12 teams, 6 rounds (stepladder format)
# Seeds 1-2: enter at round 5 (4 byes)
# Seeds 3-4: enter at round 4 (3 byes)
# Seeds 5-6: enter at round 3 (2 byes)
# Seeds 7-8: enter at round 2 (1 bye)
# Seeds 9-12: play in round 1
#
#   R1: 9v12, 10v11   (2 games → 2 winners)
#   R2: 7vW, 8vW      (2 games → 2 winners)
#   R3: 5vW, 6vW      (2 games → 2 winners)
#   R4: 3vW, 4vW      (2 games → 2 winners)
#   R5: 1vW, 2vW      (2 games → 2 winners)
#   R6: Final          (1 game)
_WCC_12 = BracketFormat(
    num_teams=12,
    total_rounds=6,
    round_entry={
        1: (9, 10, 11, 12),
        2: (7, 8),
        3: (5, 6),
        4: (3, 4),
        5: (1, 2),
    },
    first_round_matchups=((9, 12), (10, 11)),
)

# ──────────────────────────────────────────────────────────────────────
# Conference-specific overrides keyed by (conference_abbrev, num_teams)
_CONF_FORMATS: Dict[Tuple[str, int], BracketFormat] = {
    ("ACC", 15): _ACC_15,
    ("B10", 18): _B10_18,
    ("B12", 16): _B12_16,
    ("SEC", 16): _SEC_16,
    ("Amer", 10): _AMER_10,
    ("A10", 14): _A10_14,
    ("WCC", 12): _WCC_12,
}

# Conferences with standard bracket formats (mathematical default).
# These use ceil(log2(N)) rounds with 2^rounds - N byes.
_STANDARD_SIZES: Dict[str, int] = {
    "BE": 11,
    "MWC": 12,
    "MVC": 11,
    "CAA": 13,
    "AE": 8,
    "ASun": 12,
    "BSky": 10,
    "BSth": 9,
    "BW": 8,
    "CUSA": 10,
    "Horz": 11,
    "Ivy": 4,
    "MAAC": 10,
    "MAC": 8,
    "MEAC": 8,
    "NEC": 8,
    "OVC": 8,
    "Pat": 10,
    "SB": 14,
    "SC": 10,
    "SWAC": 12,
    "Slnd": 8,
    "Sum": 8,
    "WAC": 7,
}


def get_bracket_format(
    conference: str,
    num_teams: int,
) -> BracketFormat:
    """Look up the bracket format for a conference tournament.

    Args:
        conference: Conference abbreviation (e.g. "ACC").
        num_teams: Number of teams in the tournament.

    Returns:
        BracketFormat for the specified conference and team count.
        Falls back to the mathematical standard format if no
        conference-specific override exists.
    """
    # Check conference-specific overrides first
    key = (conference, num_teams)
    if key in _CONF_FORMATS:
        return _CONF_FORMATS[key]

    # Fall back to standard mathematical format
    return _standard_format(num_teams)


def get_all_formats() -> Dict[str, BracketFormat]:
    """Return bracket formats for all known conferences.

    Returns:
        Dict mapping conference abbreviation to its BracketFormat.
    """
    result = {}

    # Add conference-specific overrides
    for (conf, _), fmt in _CONF_FORMATS.items():
        result[conf] = fmt

    # Add standard formats for remaining conferences
    for conf, size in _STANDARD_SIZES.items():
        if conf not in result:
            result[conf] = _standard_format(size)

    return result
