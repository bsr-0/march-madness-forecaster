"""Kaggle-derived team features recomputed as of an arbitrary date.

WHY THESE WERE "UNDATED" AND WHY THAT WAS NEVER A DATA LIMITATION.
audit_snapshot_boundary check E reports which UI variables can be
reconstructed point-in-time. Most of the Form and shooting columns showed up
as undated, which reads like the source is missing -- but every Kaggle results
row carries a DayNum. They were undated only because
generate_team_stats_table aggregates a whole season in one pass and never
takes a cutoff. This module takes the cutoff.

THE CORRECTNESS PROPERTY THIS IS BUILT AROUND. Each function here mirrors its
season-final counterpart in generate_team_stats_table formula for formula,
including the rounding and the None conditions. That is deliberate: evaluated
at the last day of a season, every value here must equal the season-final
value already shipping. If it does not, one of the two is wrong, and a silent
disagreement would put a scale discontinuity at the boundary between
regular-season rows and tournament rows -- the same class of failure as the
torvik vintage split. scripts/validate_point_in_time_kaggle.py asserts the
equality rather than trusting the mirroring.

WHAT IS DELIBERATELY NOT HERE.
  sos_avg_opp_barthag and losses_to_weaker_rate need each OPPONENT's rating,
  and the season-final builders use the opponent's FINAL barthag / t_rank.
  Reusing that for a November row would characterise a November game with
  March information -- hindsight wearing a point-in-time wrapper, exactly what
  check F guards against for SRS. Both are provided here but require the
  caller to pass dated opponent ratings; there is no default, so the leaking
  version cannot be written by accident. t_rank in particular has no dated
  snapshot at all, so losses_to_weaker_rate takes a ranking derived from
  whatever dated rating the caller has (barthag is the intended one).

Everything stays in Kaggle team-ID space, like custom_ratings; converting to
canonical ids is the caller's job via ratings_to_canonical.
"""

from __future__ import annotations

import csv
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Must match generate_team_stats_table.CLOSE_GAME_MARGIN. A close game is
# decided by this many points or fewer.
CLOSE_GAME_MARGIN = 6

# Must match generate_team_stats_table.KAGGLE_SEASON_COMPLETE_DAYNUM. Kaggle
# publishes the in-progress season incrementally, so a season whose last
# regular-season day falls short of this has not been fully ingested. As of
# writing, 2026 stops at day 93 against 132 for a complete season -- it is
# missing roughly the last six weeks including the conference tournaments.
# Features built from it would not be wrong so much as truncated: a team would
# read 22 games played when it has played 31, and every rate would silently
# describe a partial season. generate_team_stats_table handles this by falling
# back to the cbbpy log; callers here must check rather than get a quiet
# half-season.
SEASON_COMPLETE_DAYNUM = 125


def season_is_complete(games) -> bool:
    """Whether Kaggle has ingested this season through the end of the year.

    Takes the loaded games rather than a season number so it costs nothing at
    the call site -- the caller has already paid for the load.
    """
    return bool(games) and max(g.day for g in games) >= SEASON_COMPLETE_DAYNUM


@dataclass(frozen=True)
class BoxGame:
    """One regular-season result with the box-score columns the UI needs."""

    day: int
    winner: int
    loser: int
    winner_score: int
    loser_score: int
    winner_loc: str
    # per-side box stats, winner first
    w_fgm: int
    w_fga: int
    w_fgm3: int
    w_fga3: int
    w_fta: int
    w_oreb: int
    w_dreb: int
    w_ast: int
    w_to: int
    w_stl: int
    w_blk: int
    l_fgm: int
    l_fga: int
    l_fgm3: int
    l_fga3: int
    l_fta: int
    l_oreb: int
    l_dreb: int
    l_ast: int
    l_to: int
    l_stl: int
    l_blk: int


def load_detailed_games(season: int, data_root: Path = Path("data")) -> List[BoxGame]:
    """Load one season of MRegularSeasonDetailedResults, sorted by day.

    Detailed results start in 2003; earlier seasons return an empty list rather
    than raising, so a caller sweeping a wide year range degrades to "no box
    features" instead of failing.
    """
    path = data_root / "kaggle" / "MRegularSeasonDetailedResults.csv"
    if not path.exists():
        return []
    out: List[BoxGame] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["Season"]) != season:
                continue
            out.append(
                BoxGame(
                    day=int(r["DayNum"]),
                    winner=int(r["WTeamID"]),
                    loser=int(r["LTeamID"]),
                    winner_score=int(r["WScore"]),
                    loser_score=int(r["LScore"]),
                    winner_loc=r.get("WLoc", "N"),
                    w_fgm=int(r["WFGM"]),
                    w_fga=int(r["WFGA"]),
                    w_fgm3=int(r["WFGM3"]),
                    w_fga3=int(r["WFGA3"]),
                    w_fta=int(r["WFTA"]),
                    w_oreb=int(r["WOR"]),
                    w_dreb=int(r["WDR"]),
                    w_ast=int(r["WAst"]),
                    w_to=int(r["WTO"]),
                    w_stl=int(r["WStl"]),
                    w_blk=int(r["WBlk"]),
                    l_fgm=int(r["LFGM"]),
                    l_fga=int(r["LFGA"]),
                    l_fgm3=int(r["LFGM3"]),
                    l_fga3=int(r["LFGA3"]),
                    l_fta=int(r["LFTA"]),
                    l_oreb=int(r["LOR"]),
                    l_dreb=int(r["LDR"]),
                    l_ast=int(r["LAst"]),
                    l_to=int(r["LTO"]),
                    l_stl=int(r["LStl"]),
                    l_blk=int(r["LBlk"]),
                )
            )
    out.sort(key=lambda g: g.day)
    return out


def detailed_before(games: Sequence[BoxGame], cutoff_day: int) -> List[BoxGame]:
    """Games strictly before cutoff_day. The single point-in-time filter."""
    return [g for g in games if g.day < cutoff_day]


# ---------------------------------------------------------------------------
# Form: margin, volatility, close games
# ---------------------------------------------------------------------------


def form_stats(games) -> Dict[int, Dict[str, Optional[float]]]:
    """Per-team scoring form, mirroring build_regular_season_stats.

    Accepts the Game records from point_in_time_ratings.load_season_games (or
    anything exposing winner/loser/scores), already sliced to the as-of date.

    Mirrors the season-final builder exactly, including that
    reg_season_margin_std uses POPULATION stdev and is None for a single game,
    and that close_game_win_rate is None when a team has played no close games
    -- None meaning "not measurable yet" rather than 0.0, which would read as
    "loses every close game" and is a materially different claim.
    """
    margins: Dict[int, List[int]] = {}
    for g in games:
        m = g.winner_score - g.loser_score
        margins.setdefault(g.winner, []).append(m)
        margins.setdefault(g.loser, []).append(-m)

    out: Dict[int, Dict[str, Optional[float]]] = {}
    for team, vals in margins.items():
        n = len(vals)
        close = [m for m in vals if abs(m) <= CLOSE_GAME_MARGIN]
        out[team] = {
            "games_played": n,
            "reg_season_margin_avg": round(statistics.fmean(vals), 2),
            "reg_season_margin_std": round(statistics.pstdev(vals), 2) if n > 1 else None,
            "close_game_rate": round(len(close) / n, 4),
            "close_game_win_rate": (round(sum(1 for m in close if m > 0) / len(close), 4) if close else None),
        }
    return out


# ---------------------------------------------------------------------------
# Box profile: shooting, ball security, havoc, road record
# ---------------------------------------------------------------------------


def box_profile(games: Sequence[BoxGame]) -> Dict[int, Dict[str, Optional[float]]]:
    """Per-team shooting/disruption profile, mirroring build_kaggle_box_profile.

    true_road_win_pct keeps the existing away-AND-neutral population, since
    that column's stated rationale is tournament readiness and the tournament
    is played at neutral sites. Changing the population here would silently
    redefine a shipping variable; that question belongs to
    scripts/evaluate_road_features.py, which measured it and found no
    meaningful difference either way.
    """
    acc: Dict[int, Dict[str, int]] = {}

    def side(team: int) -> Dict[str, int]:
        return acc.setdefault(
            team,
            {
                "fga": 0,
                "fgm3": 0,
                "fga3": 0,
                "opp_fgm3": 0,
                "opp_fga3": 0,
                "ast": 0,
                "to": 0,
                "stl": 0,
                "blk": 0,
                "games": 0,
                "away_neutral_games": 0,
                "away_neutral_wins": 0,
            },
        )

    for g in games:
        lloc = {"H": "A", "A": "H", "N": "N"}[g.winner_loc]

        a = side(g.winner)
        a["fga"] += g.w_fga
        a["fgm3"] += g.w_fgm3
        a["fga3"] += g.w_fga3
        a["opp_fgm3"] += g.l_fgm3
        a["opp_fga3"] += g.l_fga3
        a["ast"] += g.w_ast
        a["to"] += g.w_to
        a["stl"] += g.w_stl
        a["blk"] += g.w_blk
        a["games"] += 1
        if g.winner_loc in ("A", "N"):
            a["away_neutral_games"] += 1
            a["away_neutral_wins"] += 1

        b = side(g.loser)
        b["fga"] += g.l_fga
        b["fgm3"] += g.l_fgm3
        b["fga3"] += g.l_fga3
        b["opp_fgm3"] += g.w_fgm3
        b["opp_fga3"] += g.w_fga3
        b["ast"] += g.l_ast
        b["to"] += g.l_to
        b["stl"] += g.l_stl
        b["blk"] += g.l_blk
        b["games"] += 1
        if lloc in ("A", "N"):
            b["away_neutral_games"] += 1

    out: Dict[int, Dict[str, Optional[float]]] = {}
    for team, a in acc.items():
        if not a["games"]:
            continue
        out[team] = {
            "three_pt_rate": round(a["fga3"] / a["fga"], 4) if a["fga"] else None,
            "three_pt_pct": round(a["fgm3"] / a["fga3"], 4) if a["fga3"] else None,
            "opp_three_pt_pct": (round(a["opp_fgm3"] / a["opp_fga3"], 4) if a["opp_fga3"] else None),
            "ast_to_ratio": round(a["ast"] / a["to"], 4) if a["to"] else None,
            "havoc_rate": round((a["stl"] + a["blk"]) / a["games"], 4),
            "true_road_win_pct": (
                round(a["away_neutral_wins"] / a["away_neutral_games"], 4) if a["away_neutral_games"] else None
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Opponent-dependent features: dated ratings are REQUIRED, not optional
# ---------------------------------------------------------------------------


def strength_of_schedule(
    games,
    opponent_rating: Dict[int, float],
) -> Dict[int, Optional[float]]:
    """Mean opponent rating over games played so far (sos_avg_opp_barthag).

    `opponent_rating` has no default on purpose. The season-final builder uses
    each opponent's final barthag, which is correct when every row is as-of
    Selection Sunday and is hindsight anywhere else. Pass the dated snapshot
    for this row's own cutoff.

    Opponents absent from the mapping are skipped rather than guessed at, so a
    team with many non-D1 games gets a smaller sample, not a wrong one --
    matching the season-final behaviour.
    """
    seen: Dict[int, List[float]] = {}
    for g in games:
        for team, opp in ((g.winner, g.loser), (g.loser, g.winner)):
            r = opponent_rating.get(opp)
            if r is not None:
                seen.setdefault(team, []).append(r)
    return {t: round(statistics.fmean(v), 4) if v else None for t, v in seen.items()}


def losses_to_weaker_rate(
    games,
    rank_of: Dict[int, int],
) -> Dict[int, float]:
    """Share of games that were losses to a LOWER-ranked (worse) opponent.

    `rank_of` must be a dated ranking. t_rank has no dated snapshot, so the
    intended input is a ranking derived from dated barthag (rank 1 = best).
    Both teams must be ranked for a loss to count, matching the season-final
    rule that a non-D1 opponent never produces a bad loss.

    The denominator is total games played, not total losses -- the same
    definition the shipping column uses.
    """
    played: Dict[int, int] = {}
    bad: Dict[int, int] = {}
    for g in games:
        played[g.winner] = played.get(g.winner, 0) + 1
        played[g.loser] = played.get(g.loser, 0) + 1
        own, opp = rank_of.get(g.loser), rank_of.get(g.winner)
        if own is not None and opp is not None and opp > own:
            bad[g.loser] = bad.get(g.loser, 0) + 1
    return {t: round(bad.get(t, 0) / n, 4) for t, n in played.items() if n}


def rank_from_rating(rating: Dict[int, float]) -> Dict[int, int]:
    """Dense 1-based ranking from a rating where higher is better."""
    ordered = sorted(rating, key=lambda t: -rating[t])
    return {t: i + 1 for i, t in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Opponent adjustment
# ---------------------------------------------------------------------------

# The raw rates that audit_opponent_adjustment found carry conference strength
# at |partial r| 0.15-0.36 regardless of which quality control is used.
ADJUSTABLE_RATES = (
    "effective_fg_pct",
    "three_pt_pct",
    "offensive_reb_rate",
    "turnover_rate",
)


def _side_rates(fgm, fga, fgm3, fga3, fta, oreb, to, opp_dreb) -> Dict[str, Optional[float]]:
    """One team's rates in one game. None where the denominator is empty."""
    poss = fga - oreb + to + 0.475 * fta
    return {
        "effective_fg_pct": (fgm + 0.5 * fgm3) / fga if fga else None,
        "three_pt_pct": fgm3 / fga3 if fga3 else None,
        "offensive_reb_rate": oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else None,
        "turnover_rate": to / poss if poss > 0 else None,
    }


def per_game_rates(games: Sequence[BoxGame]):
    """Yield (team, opponent, rates) for both sides of every game.

    Game-level rather than season-aggregate because the opponent adjustment
    needs to know WHO each performance came against. A season total cannot be
    attributed back to opponents, which is precisely why the aggregate version
    of these columns cannot be adjusted after the fact.
    """
    for g in games:
        yield g.winner, g.loser, _side_rates(
            g.w_fgm, g.w_fga, g.w_fgm3, g.w_fga3, g.w_fta, g.w_oreb, g.w_to, g.l_dreb
        )
        yield g.loser, g.winner, _side_rates(
            g.l_fgm, g.l_fga, g.l_fgm3, g.l_fga3, g.l_fta, g.l_oreb, g.l_to, g.w_dreb
        )


def opponent_adjust(
    observations: List[tuple],
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Dict[int, float]:
    """Two-way fit: rate(i vs j) ~ mu + off_i + def_j. Returns mu + off_i.

    WHAT THIS FIXES. A raw rate is measured against whoever a team happened to
    play, so it encodes the schedule alongside the skill. Splitting each
    observation into an offensive effect for the team and a defensive effect
    for the opponent separates them, and reporting mu + off_i states the rate
    the team would post against an AVERAGE opponent -- same units and roughly
    the same scale as the raw rate, so it stays interpretable and drops into
    the same slot.

    Solved by alternating means rather than one big least-squares system: the
    design matrix is ~11,000 x 720 per season and this reaches the same fixed
    point in a fraction of the memory. Both effects are centred each pass so
    mu keeps its meaning as the league average; without that, off and def can
    drift by equal and opposite constants forever without converging.

    `observations` is (team, opponent, value) with value already non-None.
    """
    if not observations:
        return {}
    mu = sum(v for _, _, v in observations) / len(observations)

    by_team: Dict[int, List[int]] = {}
    by_opp: Dict[int, List[int]] = {}
    for idx, (t, o, _) in enumerate(observations):
        by_team.setdefault(t, []).append(idx)
        by_opp.setdefault(o, []).append(idx)

    off = {t: 0.0 for t in by_team}
    dfn = {o: 0.0 for o in by_opp}

    for _ in range(max_iter):
        moved = 0.0
        for t, idxs in by_team.items():
            new = sum(observations[i][2] - mu - dfn.get(observations[i][1], 0.0) for i in idxs) / len(idxs)
            moved = max(moved, abs(new - off[t]))
            off[t] = new
        c = sum(off.values()) / len(off)
        for t in off:
            off[t] -= c
        for o, idxs in by_opp.items():
            new = sum(observations[i][2] - mu - off.get(observations[i][0], 0.0) for i in idxs) / len(idxs)
            dfn[o] = new
        c = sum(dfn.values()) / len(dfn)
        for o in dfn:
            dfn[o] -= c
        if moved < tol:
            break

    return {t: mu + v for t, v in off.items()}


def adjusted_rates(games: Sequence[BoxGame]) -> Dict[int, Dict[str, float]]:
    """Opponent-adjusted versions of ADJUSTABLE_RATES, keyed by team.

    Takes games already sliced to the as-of date. The adjustment is a function
    of the games handed in and nothing else, so a per-date slice yields a
    per-date adjustment -- the opponent effects are re-solved from the same
    window as the rates they correct. Reusing a season-final adjustment for a
    December row would reintroduce exactly the hindsight this module exists to
    prevent, in a subtler place.
    """
    obs: Dict[str, List[tuple]] = {k: [] for k in ADJUSTABLE_RATES}
    for team, opp, rates in per_game_rates(games):
        for k in ADJUSTABLE_RATES:
            v = rates[k]
            if v is not None:
                obs[k].append((team, opp, v))

    solved = {k: opponent_adjust(obs[k]) for k in ADJUSTABLE_RATES}
    out: Dict[int, Dict[str, float]] = {}
    for k, table in solved.items():
        for team, v in table.items():
            out.setdefault(team, {})[k] = round(v, 6)
    return out
