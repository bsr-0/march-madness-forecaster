"""Team ratings solved as of an arbitrary date within a season.

WHY THIS IS SEPARATE FROM custom_ratings.py. custom_ratings.compute_srs_ratings
takes a cutoff_day but defaults it to 133 (Selection Sunday) and re-reads the
results CSV on every call. That is correct for tournament rows, which are all
predicted from the same boundary, but it is the wrong shape for regular-season
rows in two ways. First, a December game must be predicted from a December
solve; reusing a Selection Sunday solve for it is hindsight wearing a
point-in-time wrapper, and it looks completely healthy in every metric
downstream. Second, sweeping a season week by week re-reads and re-filters the
whole file at each boundary, which turns a cheap study into an expensive one.
This module loads a season once and solves at as many cutoffs as asked.

WHAT THE PIECES ARE FOR
  srs                     the rating itself, solved directly rather than
                          iterated, so a solve is one call and not a loop.
  largest_component_share the feasibility gate. An SRS solve on a
                          disconnected game graph is not a bad rating, it is
                          an undefined one: each component floats on its own
                          additive constant and cross-component comparisons
                          are meaningless. In normal seasons the graph
                          connects almost immediately, so this gate is
                          usually satisfied by late November -- but it fails
                          catastrophically in 2021, which is exactly why the
                          floor is a per-season gate and not a calendar date.
  shrink_to_prior         early-season SRS is noise-dominated and loses to
                          prior-season SRS until roughly week 4-5, while a
                          blend beats both components at every week. The
                          weight is per-team, not per-week, because teams have
                          not played the same number of games by a given date
                          and a per-week constant would shrink a 3-game team
                          and a 12-game team identically.
  road_margin_adjusted    opponent-adjusted road margin, the replacement for
                          a raw road win rate. See its docstring for why the
                          rate is the wrong object.

Point-in-time contract: every function here takes games already filtered to
strictly before the as-of boundary. Nothing in this module reads a result it
was not handed, so a leak has to be introduced by the caller, and
scripts/audit_snapshot_boundary.py check F asserts the caller does not.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Selection Sunday in the Kaggle day-number calendar. Present so callers can
# express "the tournament boundary" without re-deriving the constant; it is
# deliberately not a default on any solve in this module, because defaulting
# to it is the exact mistake this module exists to prevent.
SELECTION_SUNDAY_DAY = 133

# Shrinkage half-weight: the game count at which a team's own record and the
# prior-season rating carry equal weight. Flat across k=4..10 in validation,
# so the exact value is not load-bearing.
DEFAULT_SHRINK_K = 7.0


@dataclass(frozen=True)
class Game:
    """One regular-season result. Scores are as-played; loc is the winner's."""

    day: int
    winner: int
    loser: int
    winner_score: int
    loser_score: int
    winner_loc: str  # "H", "A", or "N"


def load_season_games(season: int, data_root: Path = Path("data")) -> List[Game]:
    """Load one season's regular-season results, sorted by day.

    Loads the whole season including days at and after Selection Sunday.
    Callers slice with games_before(); this function does not apply a cutoff
    of its own, so that a single load can serve many as-of dates.
    """
    for filename in ("MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv"):
        path = data_root / "kaggle" / filename
        if not path.exists():
            continue
        games: List[Game] = []
        with open(path) as f:
            for row in csv.DictReader(f):
                if int(row["Season"]) != season:
                    continue
                games.append(
                    Game(
                        day=int(row["DayNum"]),
                        winner=int(row["WTeamID"]),
                        loser=int(row["LTeamID"]),
                        winner_score=int(row["WScore"]),
                        loser_score=int(row["LScore"]),
                        winner_loc=row.get("WLoc", "N"),
                    )
                )
        if games:
            games.sort(key=lambda g: g.day)
            return games
    return []


def games_before(games: Sequence[Game], cutoff_day: int) -> List[Game]:
    """Games strictly before cutoff_day. The single point-in-time filter."""
    return [g for g in games if g.day < cutoff_day]


def _teams_in(games: Iterable[Game]) -> List[int]:
    seen = set()
    for g in games:
        seen.add(g.winner)
        seen.add(g.loser)
    return sorted(seen)


def game_counts(games: Sequence[Game]) -> Dict[int, int]:
    """Games played per team, the denominator for per-team shrinkage."""
    counts: Dict[int, int] = {}
    for g in games:
        counts[g.winner] = counts.get(g.winner, 0) + 1
        counts[g.loser] = counts.get(g.loser, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Connectivity gate
# ---------------------------------------------------------------------------


def largest_component_share(
    games: Sequence[Game],
    universe: Optional[Sequence[int]] = None,
) -> float:
    """Fraction of the team universe in the largest connected component.

    WHY THE DENOMINATOR IS THE UNIVERSE AND NOT THE TEAMS WHO HAVE PLAYED.
    Measured among teams who have played, this number flatters the early
    season badly: on opening night a single game is a component containing
    100% of the teams who have played. The question the gate has to answer is
    whether ratings are mutually comparable across the field being rated, so
    the denominator is the whole field. Pass `universe` as the season's full
    D1 team list; it defaults to teams appearing in `games`, which is the
    flattering denominator and is only appropriate once the season is complete.

    Returns 0.0 for no games, so an empty slice fails any positive threshold.
    """
    teams = _teams_in(games)
    if not teams:
        return 0.0
    denom = len(set(universe)) if universe is not None else len(teams)
    if denom == 0:
        return 0.0

    idx = {t: i for i, t in enumerate(teams)}
    parent = list(range(len(teams)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in games:
        a, b = find(idx[g.winner]), find(idx[g.loser])
        if a != b:
            parent[a] = b

    sizes: Dict[int, int] = {}
    for i in range(len(teams)):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) / denom


# ---------------------------------------------------------------------------
# SRS
# ---------------------------------------------------------------------------


def srs(games: Sequence[Game], max_iter: int = 100, tol: float = 1e-8) -> Dict[int, float]:
    """Simple Rating System solved directly: rating = avg margin + avg opponent.

    Solved as the linear system (I - N) r = m, where N[i][j] is the share of
    team i's games played against team j and m is team i's average margin,
    with a mean-centering row appended to pin the additive constant. Least
    squares rather than an exact solve because the system is singular whenever
    the game graph is disconnected -- which is the normal state of affairs in
    November, and the reason largest_component_share gates this.

    A disconnected solve returns numbers rather than raising: each component
    is internally valid and the minimum-norm solution is the sensible choice.
    Cross-component comparisons are meaningless, so gate before trusting them.
    """
    if not games:
        return {}

    teams = _teams_in(games)
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    total_margin = np.zeros(n)
    counts = np.zeros(n)
    pair = np.zeros((n, n))

    for g in games:
        wi, li = idx[g.winner], idx[g.loser]
        margin = g.winner_score - g.loser_score
        total_margin[wi] += margin
        total_margin[li] -= margin
        counts[wi] += 1
        counts[li] += 1
        pair[wi][li] += 1
        pair[li][wi] += 1

    safe = np.where(counts > 0, counts, 1.0)
    avg_margin = total_margin / safe
    N = pair / safe[:, None]

    A = np.eye(n) - N
    A_c = np.vstack([A, np.ones((1, n))])
    b_c = np.concatenate([avg_margin, [0.0]])

    ratings, *_ = np.linalg.lstsq(A_c, b_c, rcond=None)
    ratings = ratings - ratings.mean()
    return {teams[i]: float(ratings[i]) for i in range(n)}


def srs_asof(
    games: Sequence[Game],
    cutoff_day: int,
    **kwargs,
) -> Dict[int, float]:
    """SRS from games strictly before cutoff_day. The per-date solve."""
    return srs(games_before(games, cutoff_day), **kwargs)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


def shrink_to_prior(
    current: Dict[int, float],
    prior: Dict[int, float],
    counts: Dict[int, int],
    k: float = DEFAULT_SHRINK_K,
) -> Dict[int, float]:
    """Blend a point-in-time rating toward a prior-season rating, per team.

    weight on prior = k / (k + games_played), so a team with no games sits
    entirely on its prior and the prior fades as evidence accumulates. Teams
    absent from `prior` (new programs, reclassifications) fall back to 0.0,
    which is the mean of a centered SRS and therefore the correct
    no-information value rather than a sentinel.

    Applied per team rather than per week because by any given date teams have
    not played the same number of games, and a per-week constant would shrink
    a 3-game team and a 12-game team identically.
    """
    out: Dict[int, float] = {}
    for team, rating in current.items():
        n = counts.get(team, 0)
        lam = k / (k + n)
        out[team] = (1.0 - lam) * rating + lam * prior.get(team, 0.0)
    return out


# ---------------------------------------------------------------------------
# Opponent-adjusted road margin
# ---------------------------------------------------------------------------


def road_margin_adjusted(
    games: Sequence[Game],
    ratings: Optional[Dict[int, float]] = None,
    k: float = DEFAULT_SHRINK_K,
    include_neutral: bool = False,
) -> Dict[int, float]:
    """Opponent-adjusted average margin in true road games.

    WHY THIS REPLACES A ROAD WIN RATE. A rate throws away margin, which
    commit 735df4a deliberately kept, and it has a pathological denominator:
    early in a season a team with one road game reads as 0.000 or 1.000. It
    also conflates two different things -- how well a team plays away from
    home, and how hard the away games it happened to draw were.

    WHY IT IS NOT ACTUAL-MINUS-EXPECTED. Defining the feature against what
    the model expected would make it a function of the current model rather
    than of the game results. This UI refits live on arbitrary variable
    subsets, so such a feature would change every time the user toggles a
    variable and the training matrix would need recomputing per subset,
    across 2^30 of them. Adjusting by opponent rating instead keeps it a fixed
    function of results, computable once per as-of date.

    Restricting to true road games holds venue constant by construction, so no
    separate venue term is needed. Per-team shrinkage toward 0.0 handles the
    thin denominator: with few road games the value stays near neutral rather
    than swinging to an extreme.

    `ratings` supplies the opponent adjustment and must itself be a
    point-in-time solve from the same slice; it defaults to srs(games).

    include_neutral controls the population, and the choice is not obvious.
    Off (default), this is true road games only, which holds venue constant by
    construction so no venue term is needed. On, neutral sites count too and
    both teams are credited, matching the existing true_road_win_pct column --
    whose stated rationale is that the tournament is played entirely at
    neutral sites, so "not at home" is the relevant population. The two
    populations measure different things (a true road game is harder than a
    neutral one) and mixing them reintroduces the venue variation that the
    road-only form removes. Which one predicts tournament margin better is an
    empirical question; see scripts/evaluate_road_features.py.

    MEASURED, AND NOT ADOPTED AS A FEATURE. On 1,001 tournament games across
    15 seasons, leave-one-year-out, this adds nothing to an SRS differential:
    +0.013 RMSE at best, 95% CI straddling zero, and the same for the win rate
    it was meant to replace. The reason is visible in the standalone numbers:
    alone it scores 11.62 against a 14.74 intercept-only floor, nearly matching
    SRS's own 11.54 -- it is an excellent measure of team quality and therefore
    very nearly a restatement of SRS, not a road-specific signal. Residualising
    against the team's own SRS does not rescue it (identical incremental
    numbers, since the regression absorbs a linear function of SRS).

    So the original argument was right that this is a better MEASUREMENT than a
    win rate, and wrong that the distinction buys anything once strength is in
    the model. Kept here because it is a useful quality estimator and is what
    the evaluation script measures; deliberately not wired into the feature
    pipeline.
    """
    if not games:
        return {}
    if ratings is None:
        ratings = srs(games)

    total: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    def credit(team: int, opponent: int, margin: float) -> None:
        # Adding the opponent's rating credits a result against a strong
        # opponent and discounts one against a weak opponent.
        total[team] = total.get(team, 0.0) + margin + ratings.get(opponent, 0.0)
        counts[team] = counts.get(team, 0) + 1

    for g in games:
        win_margin = g.winner_score - g.loser_score
        if g.winner_loc == "A":
            credit(g.winner, g.loser, win_margin)
        elif g.winner_loc == "H":
            credit(g.loser, g.winner, -win_margin)
        elif include_neutral:
            # Neither team is away, so neither is privileged: both sides count.
            credit(g.winner, g.loser, win_margin)
            credit(g.loser, g.winner, -win_margin)

    out: Dict[int, float] = {}
    for team in _teams_in(games):
        n = counts.get(team, 0)
        raw = total.get(team, 0.0) / n if n else 0.0
        out[team] = (n / (k + n)) * raw
    return out
