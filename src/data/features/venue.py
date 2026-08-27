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

# A FOURTH STATE, AND WHY IT IS A STOPGAP.
# A conference tournament played in a team's home CITY is usually a neutral
# arena rather than that team's own gym -- St John's at Madison Square Garden,
# most of the ACC in Greensboro. Measured, the home effect in those games is
# +1.219 (95% CI 0.64-1.80) against +2.916 for a true home game, so coding them
# full HOME overstates by roughly 1.7 points.
#
# That 1.7 is not a rounding error in this context: it is the same size as the
# effects being chased elsewhere, and it is biased in one direction on the
# neutral-est games in the training data -- which is the sample the
# neutral-court residual scale is estimated from. Overstating home advantage
# there inflates the residual scale on exactly the population used to calibrate
# it.
#
# LOG THIS AS A POOLED ESTIMATE. Conference tournaments are not homogeneous:
# the ACC in Greensboro and the Big East at MSG for St John's are functionally
# home games, while most others are genuinely neutral. +1.219 averages over a
# mixed population, so it is an improvement on +2.916 rather than the value of
# a coherent state. The principled split is by host PROXIMITY, not by game
# type -- and at that point this fourth state dissolves back into the three
# above, which is the cleaner end state. It is not done here because Cities.csv
# carries no coordinates.
HOST_CITY_HOME, HOST_CITY_AWAY = 2, -2


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


def load_city_coords(data_root: Path = Path("data")) -> Dict[int, Tuple[float, float]]:
    """CityID -> (lat, lng). Empty if the file carries no coordinates."""
    path = data_root / "kaggle" / "Cities.csv"
    if not path.exists():
        return {}
    out: Dict[int, Tuple[float, float]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["CityID"])] = (float(r["lat"]), float(r["lng"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def team_location(
    season: int,
    team: int,
    home_cities: Dict[Tuple[int, int], Set[int]],
    city_coords: Dict[int, Tuple[float, float]],
    canonical_id: Optional[str] = None,
) -> Optional[Tuple[float, float]]:
    """Where a team is based, as (lat, lng), or None if not derivable.

    COMPOSED RATHER THAN CURATED. travel_distance.TEAM_COORDINATES is 133
    hand-entered campuses covering 107 of the 255 teams that have reached the
    tournament since 2010. Curating the other 148 would be slow and would go
    stale as programmes move. Every team's home city is already derivable from
    where it actually hosts games, and Cities.csv now carries coordinates for
    all 454 city ids the game data references, so composing the two covers 370
    teams with no manual entry.

    Validated against the curated campuses: median error 1.3 miles, 90th
    percentile 4.4. Immaterial against travel distances in the hundreds. The
    curated value is still preferred where it exists because it is a campus
    rather than a city centroid, and the worst derived cases are exactly the
    sprawling-metro ones -- UCLA and UCF at 12 miles, where "Los Angeles" or
    "Orlando" sits well off campus.
    """
    if canonical_id:
        from src.data.features.travel_distance import TEAM_COORDINATES

        curated = TEAM_COORDINATES.get(canonical_id)
        if curated:
            return curated

    cids = home_cities.get((season, team))
    if not cids:
        return None
    pts = [city_coords[c] for c in cids if c in city_coords]
    if not pts:
        return None
    # Mean of the hosting cities. A split-venue team (campus gym plus a
    # downtown arena) is genuinely between them, and both are in the same
    # metro, so the midpoint is closer than either for travel purposes.
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def haversine_miles(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle miles between two (lat, lng) points."""
    import math

    r = 3958.7613
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# Travel differences are reported in THOUSANDS of miles, so a fitted
# coefficient reads directly as "points per 1000 miles of relative travel
# advantage" -- the span from a home game to a cross-country trip is about 3
# units, which keeps the term on a comparable scale to the +/-1 venue states.
TRAVEL_SCALE_MILES = 1000.0


def travel_advantage(
    game_city: Optional[int],
    team_loc: Optional[Tuple[float, float]],
    opp_loc: Optional[Tuple[float, float]],
    city_coords: Dict[int, Tuple[float, float]],
) -> float:
    """How much LESS the team travelled than its opponent, in 1000s of miles.

    WHY THIS IS THE PRINCIPLED VERSION OF THE FOURTH VENUE STATE.
    HOST_CITY_HOME exists because a conference tournament in a team's home city
    is worth about +1.9 points rather than the +3.4 of a true home game. But
    that state is defined by GAME TYPE, and the population it covers is
    heterogeneous: the Big East at Madison Square Garden is a five-mile trip
    for St John's and a 2,400-mile trip for a visitor, and both currently get
    the same coding. Distance separates them continuously and needs no
    game-type test at all.

    It is also the term that belongs on NCAA tournament games. Those are
    neutral-court by rule -- venue_home is unconditionally zero there -- but
    they are NOT travel-neutral: a pod can land in a participant's back yard.
    That is proximity, not home advantage, and this is where it lives.

    Returns 0.0 when either location or the game city is unknown, which is
    honest: an unknown trip is not evidence of an equal one.
    """
    if game_city is None or team_loc is None or opp_loc is None:
        return 0.0
    site = city_coords.get(game_city)
    if site is None:
        return 0.0
    team_miles = haversine_miles(team_loc, site)
    opp_miles = haversine_miles(opp_loc, site)
    return (opp_miles - team_miles) / TRAVEL_SCALE_MILES


def venue_for(
    season: int,
    day: int,
    winner: int,
    loser: int,
    wloc: str,
    cities: Dict[Tuple[int, int, int, int], int],
    home_cities: Dict[Tuple[int, int], Set[int]],
    is_conference_tournament: bool = False,
) -> Tuple[int, int]:
    """Return (winner_state, loser_state).

    States are HOME / AWAY / NEUTRAL, or HOST_CITY_HOME / HOST_CITY_AWAY when
    a conference-tournament game lands in a participant's home city -- a
    materially weaker effect that deserves its own coefficient rather than
    being folded into HOME. See the constants for why that split is pooled and
    provisional.

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
        return (HOST_CITY_HOME, HOST_CITY_AWAY) if is_conference_tournament else (HOME, AWAY)
    if loser_home and not winner_home:
        return (HOST_CITY_AWAY, HOST_CITY_HOME) if is_conference_tournament else (AWAY, HOME)
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


def split_states(state: int) -> Tuple[int, int]:
    """One venue state -> (true_home_term, host_city_term).

    TWO COLUMNS RATHER THAN ONE SCALED COLUMN. A single column would require
    knowing the ratio between the two effects in advance and hard-coding it;
    two indicators let the fit recover +2.9 and +1.2 independently from the
    data. Both are signed, so mirroring a row to (-x, -m) negates them
    correctly -- home for A is away for B in either state.
    """
    if state == HOME:
        return 1, 0
    if state == AWAY:
        return -1, 0
    if state == HOST_CITY_HOME:
        return 0, 1
    if state == HOST_CITY_AWAY:
        return 0, -1
    return 0, 0


def assert_neutral_for_prediction(venue_terms) -> None:
    """Raise unless every venue term is zero. Call this on the tournament path.

    THE FAILURE THIS CATCHES IS AN OMISSION, WHICH IS WHY IT IS AN ASSERTION.
    NCAA tournament games are neutral, so the venue terms must contribute
    nothing at prediction time. The dangerous version of getting this wrong is
    not a wrong value -- it is the feature simply not being passed, which looks
    identical to "correctly zero" in a code review and differs by about three
    points in every prediction.

    In this codebase the browser fit has no intercept (mirrored rows force it
    to zero), so an omitted venue term does not surface as a constant offset.
    It is absorbed into the correlated strength coefficients instead -- venue
    correlates with team strength at r = 0.13 because strong teams buy home
    games -- inflating them by 13% (srs_blend) to 44% (barthag). The resulting
    error is proportional to the quality gap rather than constant, so it
    presents as a scale problem and invites a calibration fix for what is
    actually a specification bug.
    """
    bad = [i for i, v in enumerate(venue_terms) if v != 0]
    if bad:
        raise AssertionError(
            f"tournament prediction carries a non-zero venue term on "
            f"{len(bad)} row(s) (first at index {bad[0]}). NCAA games are "
            f"neutral; see tournament_venue()."
        )
