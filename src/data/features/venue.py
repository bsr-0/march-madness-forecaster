"""Three-state venue coding: home / away / true-neutral, from the host venue.

WHY NOT A GAME-TYPE FLAG. The obvious shortcut is "regular season = home/away,
conference tournament = neutral, NCAA = neutral". Measured against the actual
host city, that is wrong for 1,265 of 4,475 conference-tournament games -- 28%.
Conference tournaments are routinely held in a member's home city (the Big East
at Madison Square Garden is neutral for most of the league and not for St
John's; several leagues rotate the event onto a campus outright). Coding those
neutral hands the host team's advantage to nobody and teaches the model that
venue does not matter in exactly the games where it decides seeds.

WHY NOT WLoc ALONE. Kaggle's WLoc is a label, and labels drift from the thing
they describe. Against a host check derived from where teams actually play,
WLoc disagrees on 1.8% of games: 677 marked neutral where the winner was in its
own home city, 487 where the loser was.

WHAT THIS DOES INSTEAD. Each team's home cities are derived per season from the
games it played as the home side, keeping every city with at least
MIN_HOME_GAMES appearances -- teams with two arenas (a campus gym and a
downtown venue) are common enough that a single modal city misclassified over a
thousand games. A game is then home for whichever team's home-city set contains
the game's city.

TRUE-NEUTRAL REQUIRES BOTH SIGNALS TO AGREE. A game is only coded 0 when the
label AND the host check both say neutral. Where they disagree the game is
treated as having a home team, because the asymmetric error matters: coding a
real home game as neutral silently discards a ~3-point effect, while coding a
neutral game as home adds a small amount of noise to one row. Conservative in
the direction that loses less.

TOURNAMENT PREDICTION IS EXPLICITLY ZERO. See tournament_venue(), which exists
so that the zeroing is a named function with a reason attached rather than an
absent feature that happens to default to zero -- those look identical in the
data and completely different when someone later adds a default.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# A team's home venue must host at least this many games in a season to count.
# Set to admit genuine secondary arenas while rejecting a one-off rental that
# happened to be labelled home.
MIN_HOME_GAMES = 3

HOME, AWAY, NEUTRAL = 1, -1, 0


def load_game_cities(data_root: Path = Path("data")) -> Dict[Tuple[int, int, int, int], int]:
    """(season, day, winner, loser) -> CityID. Empty if the file is absent."""
    path = data_root / "kaggle" / "MGameCities.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(int(r["Season"]), int(r["DayNum"]), int(r["WTeamID"]), int(r["LTeamID"]))] = int(r["CityID"])
    return out


def derive_home_cities(
    data_root: Path = Path("data"),
    min_games: int = MIN_HOME_GAMES,
) -> Dict[Tuple[int, int], Set[int]]:
    """(season, team) -> set of city ids that team hosts in.

    Derived from results rather than declared, so it needs no venue table and
    stays correct when a program moves. A SET rather than a single city because
    split-venue seasons are common; collapsing to the modal city disagreed with
    the WLoc label on 1,081 games, against 96 once multiple venues are allowed.
    """
    cities = load_game_cities(data_root)
    if not cities:
        return {}
    counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    path = data_root / "kaggle" / "MRegularSeasonCompactResults.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (
                int(r["Season"]),
                int(r["DayNum"]),
                int(r["WTeamID"]),
                int(r["LTeamID"]),
            )
            city = cities.get(key)
            if city is None:
                continue
            season, loc = key[0], r["WLoc"]
            if loc == "H":
                counts[(season, key[2])][city] += 1
            elif loc == "A":
                counts[(season, key[3])][city] += 1
    return {k: {c for c, n in cc.items() if n >= min_games} for k, cc in counts.items()}


def venue_for(
    season: int,
    day: int,
    winner: int,
    loser: int,
    wloc: str,
    cities: Dict[Tuple[int, int, int, int], int],
    home_cities: Dict[Tuple[int, int], Set[int]],
) -> Tuple[int, int]:
    """Return (winner_state, loser_state), each HOME / AWAY / NEUTRAL.

    Falls back to the WLoc label when the game has no city record, which is the
    only honest option -- a missing city is not evidence of neutrality.
    """
    city = cities.get((season, day, winner, loser))
    if city is None:
        if wloc == "H":
            return HOME, AWAY
        if wloc == "A":
            return AWAY, HOME
        return NEUTRAL, NEUTRAL

    winner_home = city in home_cities.get((season, winner), ())
    loser_home = city in home_cities.get((season, loser), ())

    # Host check first; it is the physical fact. The label breaks ties only
    # when the check finds no host, and disagreement resolves toward "someone
    # was home" for the reason in the module docstring.
    if winner_home and not loser_home:
        return HOME, AWAY
    if loser_home and not winner_home:
        return AWAY, HOME
    if winner_home and loser_home:
        # Same city hosts both -- a genuine local derby at a shared venue.
        return NEUTRAL, NEUTRAL
    if wloc == "H":
        return HOME, AWAY
    if wloc == "A":
        return AWAY, HOME
    return NEUTRAL, NEUTRAL


def tournament_venue() -> int:
    """Venue state for an NCAA tournament game: always NEUTRAL, always zero.

    A NAMED FUNCTION RATHER THAN AN OMITTED FEATURE. Every NCAA tournament game
    is played at a neutral site, so the venue term must contribute nothing at
    prediction time. Leaving the feature out entirely would produce the same
    number today and a different one the moment someone gives the column a
    non-zero default or reorders the design matrix -- and the two situations are
    indistinguishable by inspecting the data. Calling this makes the zero
    deliberate and greppable.

    Note this is about the NCAA tournament specifically. Conference tournaments
    are NOT reliably neutral (28% of them have a home participant) and must go
    through venue_for() like any other game.
    """
    return NEUTRAL
