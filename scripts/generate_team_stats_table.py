"""Generate a multi-year pre-tournament team stats table for the web app.

Produces:
  docs/data/team_stats_by_year.json - one row per tournament-qualified team
    per year, 2010-2026, combining three source families:

  1. Torvik pre-tournament ratings (data/raw/historical/torvik_{year}.json) —
     barthag, adjusted efficiencies, tempo, and the eight four-factor fields.
  2. Regular-season volatility, computed here from Kaggle's regular-season
     results (data/kaggle/MRegularSeasonCompactResults.csv) — scoring margin
     mean/spread, close-game rate and record, and rate of losses to
     lower-ranked opponents. The in-progress season, which Kaggle has not
     ingested yet, falls back to data/raw/historical/historical_games_{year}.json.
  3. Post-hoc tournament outcome (tournament_context_{year}.json) — how far
     the team actually got. This is NOT pre-tournament information; it is
     kept in clearly-labelled `outcome_*` fields and rendered as a visually
     separate block in the UI so it can never be mistaken for a feature that
     was knowable before the tournament tipped off.

Leakage guard: family (2) needs no date filter, because Kaggle keeps NCAA
tournament games in a separate file — verified, 0 of 1,449 tournament games
appear in the regular-season file. Correctness comes from the source's scope
rather than from trusting a per-game timestamp.

This used to read the cbbpy game log and drop anything dated on or after
tournament_start. That filter did keep every tournament game out (checked
game-by-game, 2010-2026, none got through) but it failed the other way: the
logs' per-game dates are largely synthetic, so 600-800 real regular-season
games a season were stamped into April and silently discarded. Median
games_played fell to 22 in 2012 against a true D1 schedule of ~31. The cbbpy
fallback is therefore restricted to seasons whose log stops before the
tournament on its own, and refuses rather than guesses otherwise.

Deliberately excluded: data/raw/torvik_shooting_{year}.json (3PT%/FT%).
docs/data-provenance-and-leakage-audit.md classifies the 2008-2025 shooting
files as post-tournament player stats with "no date filtering possible" —
only 2026 is verified pre-tournament. A column clean in 1 of 16 years and
contaminated in the rest is worse than no column. Shooting splits ARE built
below, from a different, leakage-safe source — see family (4).

  4. Kaggle regular-season box scores (data/kaggle/MRegularSeasonDetailedResults.csv)
     — 3PT rate/pct, opponent 3PT defense, assist-to-turnover ratio, steal+block
     ("havoc") rate, and true road/neutral win rate. This file contains ZERO
     NCAA tournament games (they live in a separate Kaggle file), so unlike the
     Torvik shooting files it is pre-tournament by construction with no date
     filtering needed — the same property `overtime_rate` already relies on.
  5. Kaggle coach history (data/kaggle/MTeamCoaches.csv +
     MNCAATourneyCompactResults.csv) — each coach's cumulative NCAA tournament
     games/wins BEFORE the season in question. Strictly backward-looking: a
     coach's record going into year Y only counts seasons < Y, so a first-time
     tournament coach reads 0 games / 0 wins, never their own upcoming result.
  6. Kaggle conference tournament games (data/kaggle/MConferenceTourneyGames.csv)
     — each team's win count in ITS OWN conference tournament this season.
     Unlike coach history, no backward-looking filter is needed: conference
     tournaments finish before Selection Sunday, so this season's result is
     itself pre-tournament information.

Still not buildable: clutch/late-game splits and blown-lead rate need
play-by-play, and there is no play-by-play anywhere in this repository —
only final scores and box-score totals. Coach experience above is as close
as the data gets to a "big-game readiness" signal.
"""

import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._common import HIST_DIR, load_seeds_and_regions, load_torvik_and_ff, load_tournament_results
from src.data.normalize import load_d1_team_ids
from src.data.kaggle_loader import KaggleDataLoader
from src.data.normalize import resolve_cbbpy_bridge

KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle"
OUT = PROJECT_ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = range(2010, 2027)
VALID_PRETOURNAMENT_TYPES = {"pre_tournament", "pre_tournament_computed"}

CLOSE_GAME_MARGIN = 6  # points; a "close game" is decided by <= this

# Extra cbbpy->canonical aliases, applied on top of src.data.normalize's shared
# bridge. These are the ~10 programs whose cbbpy IDs share no prefix with their
# Torvik ID ("unlv_rebels" vs "nevada_las_vegas"), so longest-prefix matching
# can't reach them; without these they silently lose their game-log stats.
#
# Kept LOCAL rather than added to normalize._CBBPY_EDGE_CASES on purpose: that
# shared dict feeds three production probability bases (Elo A4, roster_adj C2,
# volatile), so widening it would change what those bases see for these teams
# and could move backtest results without going through the acceptance gate.
# This table is display-only, so the blast radius stays here.
_SEEDS_TO_TORVIK_ALIASES = {
    # A handful of seeds-file IDs that Torvik spells differently. Without
    # these the team is dropped from the table entirely (it has no Torvik row
    # to join to), which also punches a hole in that year's bracket.
    "sam_houston": "sam_houston_state",
    "southern_miss": "southern_mississippi",
    "umass": "massachusetts",
}

STAT_FIELDS = (
    "conference",
    "t_rank",
    "barthag",
    "adj_offensive_efficiency",
    "adj_defensive_efficiency",
    "adj_tempo",
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "defensive_reb_rate",
    "opp_free_throw_rate",
)

# Bracket rounds in order. FF (First Four play-in) is excluded on purpose:
# the rest of this project scores a 63-game R64-onward bracket, so a play-in
# win is not a "round won" here.
BRACKET_ROUNDS = ("R64", "R32", "S16", "E8", "F4", "NCG")
FINISH_ON_LOSS = {
    "FF": "First Four",
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "NCG": "Runner-up",
}


def _num(v) -> float:
    """Coerce to a finite float, mapping None/NaN/garbage to 0.0.

    `or 0` is not enough here: NaN is truthy, so a NaN minutes figure would
    sail through and poison the sum. Python's json.dump then writes a bare
    `NaN` literal, which json.load happily reads back but every browser
    rejects as invalid JSON — a failure only visible in the browser.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def build_roster_stats(year: int, canonical_ids):
    """Roster composition: share of minutes from returners and from freshmen.

    `returning_minutes_pct` is the share of this season's minutes played by
    players who were on the SAME team's roster the previous season (matched
    on player_id, which is stable year to year). `freshman_minutes_pct` is
    the share played by `eligibility_year == 1`. `n_returning_players` is the
    raw headcount version of the same match -- how many of this year's roster
    were on last year's roster, unweighted by minutes. Unlike the two ratios
    below, it needs no tournament-contamination caveat: it is pure roster
    membership (was player_id X on both lists), with no minutes or games-played
    term in it at all, so there is no mechanism for a longer tournament run to
    move it. `None` when there is no prior-season roster file to compare
    against, distinct from `0` (compared, and nobody came back).

    Provenance caveat, stated plainly, for the two ratios: every `cbbpy_rosters_*.json` carries
    the same scrape timestamp, so for 2010-2025 the per-player minute
    averages are full-season figures that include that year's tournament
    games (2026's snapshot is genuinely mid-February, pre-tournament).

    That is a milder problem here than it would be for, say, a shooting
    percentage — which is why the shooting files are excluded outright while
    these are kept. Both inputs to these ratios are pre-tournament facts: a
    player's class and whether he was on last year's roster are settled long
    before March. Only the weighting shifts, and because the metric uses
    minutes-PER-GAME rather than season totals, playing four extra games
    barely moves a rotation share. The contamination is second-order on the
    weights, not first-order on the quantity itself.
    """
    path = HIST_DIR / f"cbbpy_rosters_{year}.json"
    prev_path = HIST_DIR / f"cbbpy_rosters_{year - 1}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        teams = json.load(f).get("teams", [])

    prior_ids: dict[str, set] = {}
    if prev_path.exists():
        with open(prev_path) as f:
            for t in json.load(f).get("teams", []):
                prior_ids[t["team_id"]] = {p.get("player_id") for p in t.get("players", [])}

    # Weight the bridge by total rotation minutes so that when two schools'
    # cbbpy IDs collide on one canonical team, the real D1 roster wins. The
    # plain first-write-wins this replaced handed the ID to whichever roster
    # happened to come first in the file.
    minutes = {
        t["team_id"]: sum(_num(p.get("minutes_per_game")) for p in t.get("players", []))
        for t in teams
        if t.get("team_id")
    }
    # BRIDGE ON PLAYER-GAMES, NOT SUMMED MINUTES-PER-GAME. mpg is a per-game
    # AVERAGE, so summing it measures roster size x rotation depth and does not
    # scale with how much of the dataset belongs to the team -- which is what
    # resolve_cbbpy_bridge asks its weight to measure. virginia_lynchburg
    # (non-D1, 8 games, 17 listed players) summed to 326.1 against real
    # Virginia's 220.5 and took the canonical `virginia` slot. On player-games
    # it is 78 against 346, a 4.4x margin the right way.
    player_games = {
        t["team_id"]: sum(_num(p.get("games_played")) for p in t.get("players", []))
        for t in teams
        if t.get("team_id")
    }
    bridge_map = resolve_cbbpy_bridge(player_games, canonical_ids, universe=_d1_universe(year))

    out = {}
    for t in teams:
        raw_id = t["team_id"]
        canonical = bridge_map.get(raw_id)
        if canonical is None:
            continue
        players = t.get("players", [])
        total = minutes[raw_id]
        if total <= 0:
            continue
        returners = prior_ids.get(raw_id, set())
        roster_ids = {p.get("player_id") for p in players if p.get("player_id")}
        # The two minute RATIOS moved to build_roster_minutes, which weights by
        # pre-tournament box-score minutes instead of cbbpy's tournament-
        # inclusive minutes_per_game. Only the headcount stays here: it is pure
        # roster membership with no minutes or games-played term, so it never
        # had the contamination to begin with.
        out[canonical] = {
            "n_returning_players": len(roster_ids & returners) if prev_path.exists() else None,
        }
    return out


def build_roster_minutes(year: int, canonical_ids):
    """returning/freshman minute shares weighted by PRE-TOURNAMENT minutes.

    THE CONTAMINATION THIS REPLACES. d954902 excluded returning_minutes_pct and
    freshman_minutes_pct because cbbpy's per-player minutes_per_game is
    averaged over a game count that INCLUDES that season's tournament run.
    Purdue 2024 carries 39 games against 33 played before the tournament --
    exactly its six-game run to the final -- and extra games on the roster
    correlate with rounds actually won at r = +0.93. The weights straddled the
    prediction point even though every input to them looked settled in October.

    That commit declined to patch it: "the honest reconstruction is minutes
    before tournament_start over team minutes before tournament_start, and
    player_minutes_*.json holds only season aggregates with no per-game
    breakdown. Blocked until game-level box scores land." They have landed.

    WHAT CHANGES AND WHAT DOES NOT. Only the WEIGHTS. Roster membership and
    eligibility_year stay on cbbpy, because those inputs were never
    contaminated -- a player's class and whether he was on last year's roster
    are facts settled long before March. Swapping the weighting for box-score
    minutes is the minimal correct fix rather than a rebuild.

    PRE-TOURNAMENT BY CONSTRUCTION. EspnBoxscoreScraper stops strictly before
    TOURNAMENT_START_DATES[year], so the file cannot contain a tournament game
    to filter out. Verified per season by backfill_pbp_history.
    verify_season_coverage, which is what caught 2023 and 2024 shipping
    truncated.

    BOTH SIDES ARE BRIDGED TO CANONICAL BEFORE JOINING. The two sources use
    different slugs for the same school -- `alabama` in the rosters against
    `alabama_crimson_tide` in the box scores, 115 such teams -- so a direct
    join on raw team id silently drops them. Player ids ARE shared
    (athlete_id == player_id, 100% overlap where teams match), so the join is
    safe once the team side is canonicalised.
    """
    box_path = HIST_DIR / f"boxscores_{year}.json"
    roster_path = HIST_DIR / f"cbbpy_rosters_{year}.json"
    prev_path = HIST_DIR / f"cbbpy_rosters_{year - 1}.json"
    if not box_path.exists() or not roster_path.exists():
        return {}

    with open(box_path) as f:
        games = json.load(f).get("games", [])
    if not games:
        return {}

    raw_minutes: dict[str, dict[str, float]] = {}
    for game in games:
        for team in game.get("teams") or []:
            tid = team["team_id"]
            bucket = raw_minutes.setdefault(tid, {})
            for p in team.get("players") or []:
                mins = p.get("minutes")
                if mins is None:
                    continue  # DNP
                aid = str(p["athlete_id"])
                bucket[aid] = bucket.get(aid, 0.0) + float(mins)

    box_bridge = resolve_cbbpy_bridge(
        {tid: sum(v.values()) for tid, v in raw_minutes.items()},
        canonical_ids,
        universe=_d1_universe(year),
    )
    canon_minutes: dict[str, dict[str, float]] = {}
    for tid, players in raw_minutes.items():
        canonical = box_bridge.get(tid)
        if canonical is None:
            continue
        agg = canon_minutes.setdefault(canonical, {})
        for aid, m in players.items():
            agg[aid] = agg.get(aid, 0.0) + m

    with open(roster_path) as f:
        teams = json.load(f).get("teams", [])
    prior_ids: dict[str, set] = {}
    if prev_path.exists():
        with open(prev_path) as f:
            for t in json.load(f).get("teams", []):
                prior_ids[t["team_id"]] = {str(p.get("player_id")) for p in t.get("players", [])}

    roster_bridge = resolve_cbbpy_bridge(
        # player-games, not summed mpg -- see build_roster_stats for why.
        {
            t["team_id"]: sum(_num(p.get("games_played")) for p in t.get("players", []))
            for t in teams
            if t.get("team_id")
        },
        canonical_ids,
        universe=_d1_universe(year),
    )
    returners_by_canon: dict[str, set] = {}
    frosh_by_canon: dict[str, set] = {}
    for t in teams:
        canonical = roster_bridge.get(t["team_id"])
        if canonical is None:
            continue
        returners_by_canon.setdefault(canonical, set()).update(
            prior_ids.get(t["team_id"], set())
        )
        frosh_by_canon.setdefault(canonical, set()).update(
            str(p.get("player_id"))
            for p in t.get("players", [])
            if p.get("eligibility_year") == 1
        )

    out = {}
    for canonical, players in canon_minutes.items():
        total = sum(players.values())
        if total <= 0:
            continue
        returners = returners_by_canon.get(canonical, set())
        frosh = frosh_by_canon.get(canonical, set())
        ret_min = sum(m for aid, m in players.items() if aid in returners)
        frosh_min = sum(m for aid, m in players.items() if aid in frosh)
        out[canonical] = {
            "returning_minutes_pct": (
                round(ret_min / total, 4) if returners and prev_path.exists() else None
            ),
            "freshman_minutes_pct": round(frosh_min / total, 4) if frosh else None,
        }
    return out


DOUBLE_DIGIT_PPG = 10.0


def _d1_universe(year: int):
    """Full D1 canonical id set for a season, for resolve_cbbpy_bridge.

    WITHOUT THIS THE BRIDGE IS DEMONSTRABLY WRONG, and its own docstring says
    so: bridging against a 68-team target set lets `north_carolina_a_t_aggies`
    prefix-match the canonical `north_carolina`, because `north_carolina_a_t`
    is not in the target set to out-match it. Weighting cannot save it -- the
    impostor often carries more data than the real team.

    Every caller in this file omitted the argument, so 142 of 1,084
    team-seasons resolved a canonical id to the wrong school, including North
    Carolina, Kansas, Texas and Florida. clutch_metrics and pbp_box_scores
    already passed it; this file did not.
    """
    return load_d1_team_ids(year, PROJECT_ROOT / "data")


def build_scoring_depth(year: int, canonical_ids):
    """Count of double-digit (>= 10 ppg) scorers, from raw ESPN box scores.

    Source is `data/raw/historical/boxscores_{year}.json` -- the same
    per-game ESPN box-score cache `player_minutes_{year}.json` is built from,
    one level upstream of that aggregation. `player_minutes_*.json` only kept
    minutes; points are in the same per-game payload (the scraper's own
    docstring: "the full stat line is retained, not just minutes"), just never
    pulled out into a season aggregate before now.

    Pre-tournament by construction, not by date filtering: `EspnBoxscoreScraper`
    stops at `TOURNAMENT_START_DATES[year]` by default (see
    boxscore_player_minutes.py / cbbpy_pbp.py), so this file never contains a
    tournament game to filter out in the first place.

    Only seasons the backfill has reached have this file (2013 on, as of this
    writing, filling in behind the still-running historical backfill); earlier
    seasons return {} and the field reads None, same convention as every other
    partially-backfilled column here.
    """
    path = HIST_DIR / f"boxscores_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        payload = json.load(f)
    games = payload.get("games", [])
    if not games:
        return {}

    points: dict[tuple[str, str], float] = {}
    games_played: dict[tuple[str, str], int] = {}
    for game in games:
        for team in game.get("teams") or []:
            tid = team["team_id"]
            for p in team.get("players") or []:
                minutes = p.get("minutes")
                if minutes is None:
                    continue  # DNP
                pts = _num((p.get("stats") or {}).get("points"))
                key = (tid, p["athlete_id"])
                points[key] = points.get(key, 0.0) + pts
                games_played[key] = games_played.get(key, 0) + 1

    ppg_by_team: dict[str, list[float]] = {}
    for (tid, athlete_id), total_pts in points.items():
        gp = games_played[(tid, athlete_id)]
        if gp > 0:
            ppg_by_team.setdefault(tid, []).append(total_pts / gp)

    # Bridge raw ESPN team ids to canonical ids, weighted by total scoring
    # (same idiom as build_roster_stats's minutes weighting), so a collision
    # resolves to whichever roster is actually the D1 team in question.
    weights = {tid: sum(ppgs) for tid, ppgs in ppg_by_team.items()}
    bridge_map = resolve_cbbpy_bridge(weights, canonical_ids, universe=_d1_universe(year))

    out = {}
    for raw_id, ppgs in ppg_by_team.items():
        canonical = bridge_map.get(raw_id)
        if canonical is None:
            continue
        out[canonical] = {"n_double_digit_scorers": sum(1 for p in ppgs if p >= DOUBLE_DIGIT_PPG)}
    return out


def _kaggle_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load_kaggle_team_map(canonical_ids, year: int | None = None) -> dict[str, str]:
    """Kaggle numeric TeamID (as string) -> canonical team_id, via MTeamSpellings.

    Shared by every Kaggle-sourced builder below so the name-normalization
    logic exists in exactly one place.

    DISAMBIGUATING QUALIFIED NAMES. Some schools appear in Kaggle only with a
    state qualifier that the canonical id lacks: `saint_francis` matches neither
    "saint francis (ny)" nor "saint francis (pa)", so 2025's 16-seed dropped out
    of every Kaggle-sourced column. Passing `year` enables a second pass that
    accepts a qualified name as a prefix match, but ONLY when exactly one such
    school was Division I that season. St Francis NY left D1 after 2023, so 2025
    resolves to PA unambiguously; "miami" still matches both FL and OH in every
    year and is therefore left unmapped rather than guessed at.
    """
    spellings = KAGGLE_DIR / "MTeamSpellings.csv"
    if not spellings.exists():
        return {}
    canonical_by_norm = {_kaggle_norm(c): c for c in canonical_ids}
    kaggle_to_canonical: dict[str, str] = {}
    unmatched: dict[str, list[tuple[str, str]]] = {}

    with open(spellings, encoding="latin-1") as f:
        for r in csv.DictReader(f):
            norm = _kaggle_norm(r["TeamNameSpelling"])
            canonical = canonical_by_norm.get(norm)
            if canonical is not None:
                kaggle_to_canonical[r["TeamID"]] = canonical
            elif year is not None:
                unmatched.setdefault(norm, []).append((r["TeamID"], r["TeamNameSpelling"]))

    if year is None:
        return kaggle_to_canonical

    d1_span = _kaggle_d1_span()
    mapped_ids = set(kaggle_to_canonical)
    for norm_canonical, canonical in canonical_by_norm.items():
        if canonical in kaggle_to_canonical.values():
            continue
        candidates = set()
        for norm, entries in unmatched.items():
            if not norm.startswith(norm_canonical) or norm == norm_canonical:
                continue
            for team_id, _spelling in entries:
                if team_id in mapped_ids:
                    continue
                first, last = d1_span.get(team_id, (None, None))
                if first is not None and first <= year <= last:
                    candidates.add(team_id)
        if len(candidates) == 1:
            kaggle_to_canonical[candidates.pop()] = canonical
    return kaggle_to_canonical


def _kaggle_d1_span() -> dict[str, tuple[int, int]]:
    """TeamID -> (FirstD1Season, LastD1Season), for resolving qualified names."""
    path = KAGGLE_DIR / "MTeams.csv"
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["TeamID"]: (int(r["FirstD1Season"]), int(r["LastD1Season"])) for r in csv.DictReader(f)}


def build_overtime_rate(year: int, canonical_ids):
    """Share of regular-season games that went to overtime.

    A clutch proxy from Kaggle's box-score file. `MRegularSeasonDetailedResults`
    contains no NCAA tournament games at all (those live in a separate file),
    so this is pre-tournament by construction.

    This is as close to "late-game performance" as the repo's data supports:
    genuine clutch splits and blown-lead rate need play-by-play, and there is
    no play-by-play anywhere in this repository — only final scores and box
    score totals. Those columns are not buildable, full stop.
    """
    results = KAGGLE_DIR / "MRegularSeasonDetailedResults.csv"
    kaggle_to_canonical = _load_kaggle_team_map(canonical_ids, year)
    if not results.exists() or not kaggle_to_canonical:
        return {}

    games: dict[str, int] = {}
    ot: dict[str, int] = {}
    with open(results) as f:
        for r in csv.DictReader(f):
            if int(r["Season"]) != year:
                continue
            went_ot = int(r["NumOT"]) > 0
            for side in ("WTeamID", "LTeamID"):
                team = kaggle_to_canonical.get(r[side])
                if team is None:
                    continue
                games[team] = games.get(team, 0) + 1
                if went_ot:
                    ot[team] = ot.get(team, 0) + 1
    return {t: {"overtime_rate": round(ot.get(t, 0) / n, 4)} for t, n in games.items() if n}


def build_kaggle_box_profile(year: int, canonical_ids):
    """Shooting profile and defensive pressure from the regular-season box score.

    All from `MRegularSeasonDetailedResults` — pre-tournament by construction,
    same as `build_overtime_rate` above.

      three_pt_rate       — share of field-goal attempts that were threes
      three_pt_pct        — 3PM / 3PA
      opp_three_pt_pct    — opponents' 3PM / 3PA against this team (defense)
      ast_to_ratio        — assists per turnover (ball security / court vision;
                             distinct from Torvik's turnover_rate, which is
                             turnovers alone with no assist context)
      havoc_rate          — (steals + blocks) per game (defensive disruption;
                             Torvik's four factors have no steal/block signal)
      true_road_win_pct   — win rate in true-road + neutral-site games only,
                             using `WLoc`. Torvik's ratings are already
                             opponent-adjusted, so this isn't strength — it's
                             whether a team's results hold up away from home,
                             a proxy for tournament (all-neutral-site) readiness.
    """
    results = KAGGLE_DIR / "MRegularSeasonDetailedResults.csv"
    kaggle_to_canonical = _load_kaggle_team_map(canonical_ids, year)
    if not results.exists() or not kaggle_to_canonical:
        return {}

    acc: dict[str, dict[str, float]] = {}

    def team_acc(t):
        return acc.setdefault(
            t,
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

    with open(results) as f:
        for r in csv.DictReader(f):
            if int(r["Season"]) != year:
                continue
            wt = kaggle_to_canonical.get(r["WTeamID"])
            lt = kaggle_to_canonical.get(r["LTeamID"])
            wloc = r["WLoc"]  # location of the WINNER: H/A/N
            lloc = {"H": "A", "A": "H", "N": "N"}[wloc]

            if wt is not None:
                a = team_acc(wt)
                a["fga"] += int(r["WFGA"])
                a["fgm3"] += int(r["WFGM3"])
                a["fga3"] += int(r["WFGA3"])
                a["opp_fgm3"] += int(r["LFGM3"])
                a["opp_fga3"] += int(r["LFGA3"])
                a["ast"] += int(r["WAst"])
                a["to"] += int(r["WTO"])
                a["stl"] += int(r["WStl"])
                a["blk"] += int(r["WBlk"])
                a["games"] += 1
                if wloc in ("A", "N"):
                    a["away_neutral_games"] += 1
                    a["away_neutral_wins"] += 1
            if lt is not None:
                a = team_acc(lt)
                a["fga"] += int(r["LFGA"])
                a["fgm3"] += int(r["LFGM3"])
                a["fga3"] += int(r["LFGA3"])
                a["opp_fgm3"] += int(r["WFGM3"])
                a["opp_fga3"] += int(r["WFGA3"])
                a["ast"] += int(r["LAst"])
                a["to"] += int(r["LTO"])
                a["stl"] += int(r["LStl"])
                a["blk"] += int(r["LBlk"])
                a["games"] += 1
                if lloc in ("A", "N"):
                    a["away_neutral_games"] += 1

    out = {}
    for t, a in acc.items():
        if not a["games"]:
            continue
        out[t] = {
            "three_pt_rate": round(a["fga3"] / a["fga"], 4) if a["fga"] else None,
            "three_pt_pct": round(a["fgm3"] / a["fga3"], 4) if a["fga3"] else None,
            "opp_three_pt_pct": round(a["opp_fgm3"] / a["opp_fga3"], 4) if a["opp_fga3"] else None,
            "ast_to_ratio": round(a["ast"] / a["to"], 4) if a["to"] else None,
            "havoc_rate": round((a["stl"] + a["blk"]) / a["games"], 4),
            "true_road_win_pct": (
                round(a["away_neutral_wins"] / a["away_neutral_games"], 4) if a["away_neutral_games"] else None
            ),
        }
    return out


_coach_history_cache = None


def _load_coach_tourney_history():
    """Cache of every coach's per-season NCAA tournament games/wins, 1985-2025.

    Loaded once and memoized — every `build_coach_experience(year, ...)` call
    just slices the same in-memory structure, rather than re-reading two CSVs
    per year across 17 years.

    Returns `(coach_of_record, per_coach_season_games, per_coach_season_wins)`:
      coach_of_record[(season, kaggle_team_id)] -> coach_name — the coach with
        the latest LastDayNum that season, i.e. the one who took the team into
        the postseason (handles in-season coaching changes correctly).
      per_coach_season_{games,wins}[coach_name][season] -> count, from
        MNCAATourneyCompactResults attributed via coach_of_record.
    """
    global _coach_history_cache
    if _coach_history_cache is not None:
        return _coach_history_cache

    coaches_path = KAGGLE_DIR / "MTeamCoaches.csv"
    tourney_path = KAGGLE_DIR / "MNCAATourneyCompactResults.csv"
    if not coaches_path.exists() or not tourney_path.exists():
        _coach_history_cache = ({}, {}, {})
        return _coach_history_cache

    coach_of_record: dict[tuple[int, str], str] = {}
    best_last_day: dict[tuple[int, str], int] = {}
    with open(coaches_path) as f:
        for r in csv.DictReader(f):
            key = (int(r["Season"]), r["TeamID"])
            last_day = int(r["LastDayNum"])
            if last_day >= best_last_day.get(key, -1):
                best_last_day[key] = last_day
                coach_of_record[key] = r["CoachName"]

    per_coach_season_games: dict[str, dict[int, int]] = {}
    per_coach_season_wins: dict[str, dict[int, int]] = {}
    with open(tourney_path) as f:
        for r in csv.DictReader(f):
            season = int(r["Season"])
            wcoach = coach_of_record.get((season, r["WTeamID"]))
            lcoach = coach_of_record.get((season, r["LTeamID"]))
            if wcoach:
                per_coach_season_games.setdefault(wcoach, {})[season] = (
                    per_coach_season_games.setdefault(wcoach, {}).get(season, 0) + 1
                )
                per_coach_season_wins.setdefault(wcoach, {})[season] = (
                    per_coach_season_wins.setdefault(wcoach, {}).get(season, 0) + 1
                )
            if lcoach:
                per_coach_season_games.setdefault(lcoach, {})[season] = (
                    per_coach_season_games.setdefault(lcoach, {}).get(season, 0) + 1
                )

    _coach_history_cache = (coach_of_record, per_coach_season_games, per_coach_season_wins)
    return _coach_history_cache


def build_coach_experience(year: int, canonical_ids):
    """Each team's head coach and that coach's tournament track record BEFORE this year.

    Strictly backward-looking: only seasons < `year` count toward
    `coach_prior_tourney_games`/`_wins`, so a coach's own upcoming result in
    `year` can never leak into their "experience" going into it. A coach with
    zero prior tournament games reads `coach_first_tourney: True` — this is the
    closest the data gets to the "big-game readiness" signal the shooting/PBP
    columns can't provide.
    """
    coach_of_record, per_coach_games, per_coach_wins = _load_coach_tourney_history()
    kaggle_to_canonical = _load_kaggle_team_map(canonical_ids, year)
    if not coach_of_record or not kaggle_to_canonical:
        return {}

    canonical_to_kaggle = {v: k for k, v in kaggle_to_canonical.items()}
    out = {}
    for canonical, kaggle_id in canonical_to_kaggle.items():
        coach = coach_of_record.get((year, kaggle_id))
        if coach is None:
            continue
        prior_games = sum(n for s, n in per_coach_games.get(coach, {}).items() if s < year)
        prior_wins = sum(n for s, n in per_coach_wins.get(coach, {}).items() if s < year)
        out[canonical] = {
            "coach_name": coach,
            "coach_prior_tourney_games": prior_games,
            "coach_prior_tourney_wins": prior_wins,
            "coach_first_tourney": prior_games == 0,
        }
    return out


_conf_tourney_cache = None


def _load_conf_tourney_history():
    """Cache of every (season, kaggle_team_id)'s conference-tournament wins.

    Loaded once from data/kaggle/MConferenceTourneyGames.csv (2001-2025) and
    memoized like `_load_coach_tourney_history`.

    Conference tournaments finish before Selection Sunday, so this season's
    OWN result is pre-tournament information -- unlike coach experience, no
    backward-looking `< year` filter is needed here.

    Returns `(wins_by_season_team, played_by_season_team)`:
      wins_by_season_team[(season, kaggle_team_id)] -> win count.
      played_by_season_team is the set of (season, kaggle_team_id) that
      appear in the file at all (as winner OR loser) -- needed to tell "lost
      its only game" (0 wins, played) apart from "no conference tournament
      row at all" (independents, or a season this file has not caught up to).
    """
    global _conf_tourney_cache
    if _conf_tourney_cache is not None:
        return _conf_tourney_cache

    path = KAGGLE_DIR / "MConferenceTourneyGames.csv"
    if not path.exists():
        _conf_tourney_cache = ({}, set())
        return _conf_tourney_cache

    wins_by_season_team: dict[tuple[int, str], int] = {}
    played_by_season_team: set[tuple[int, str]] = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            season = int(r["Season"])
            wkey = (season, r["WTeamID"])
            lkey = (season, r["LTeamID"])
            wins_by_season_team[wkey] = wins_by_season_team.get(wkey, 0) + 1
            played_by_season_team.add(wkey)
            played_by_season_team.add(lkey)

    _conf_tourney_cache = (wins_by_season_team, played_by_season_team)
    return _conf_tourney_cache


def build_conf_tourney_result(year: int, canonical_ids):
    """Each team's win count in its OWN conference tournament this season.

    A team that lost its conference-tournament opener reads 0; a team that
    won it outright (the auto-bid path) reads however many rounds that
    conference's bracket has (typically 3-5). Teams with no row at all
    (independents, or seasons the Kaggle file has not caught up to yet -- see
    module docstring on the current season's Kaggle lag) get None, handled
    the same way every other Kaggle-sourced column's gap is handled in
    `build_year_rows`.
    """
    wins_by_season_team, played_by_season_team = _load_conf_tourney_history()
    kaggle_to_canonical = _load_kaggle_team_map(canonical_ids, year)
    if not played_by_season_team or not kaggle_to_canonical:
        return {}

    canonical_to_kaggle = {v: k for k, v in kaggle_to_canonical.items()}
    out = {}
    for canonical, kaggle_id in canonical_to_kaggle.items():
        key = (year, kaggle_id)
        if key not in played_by_season_team:
            continue
        out[canonical] = {"conf_tourney_wins": wins_by_season_team.get(key, 0)}
    return out


_massey_loader = None


def build_massey_composite(year: int):
    """Each team's average ordinal rank across every computer ranking system.

    Uses `KaggleDataLoader.load_massey_ordinals`, which already auto-computes
    a safe `max_day` from Selection Sunday and DayZero (data/kaggle/MSeasons.csv)
    and excludes any ranking day after it -- the same leakage guard
    `_compute_max_ranking_day` provides for `ExternalRatingsLoader`. This
    function does no date logic of its own; it only averages what that loader
    already returns.

    A team's average is taken over whichever systems ranked it that season
    (system coverage varies year to year, ~54-70 systems), so a team missing
    from a handful of systems is not penalized -- it is a mean over
    however many systems actually rated it, not over a fixed denominator.
    """
    global _massey_loader
    if _massey_loader is None:
        _massey_loader = KaggleDataLoader(str(KAGGLE_DIR))

    ordinals = _massey_loader.load_massey_ordinals(year)
    if not ordinals:
        return {}

    ranks_by_team: dict[str, list[int]] = {}
    for system_map in ordinals.values():
        for team_id, entry in system_map.items():
            ranks_by_team.setdefault(team_id, []).append(entry.ordinal_rank)

    return {team_id: {"massey_avg_rank": round(statistics.fmean(ranks), 2)} for team_id, ranks in ranks_by_team.items()}


def _torvik_meta(year: int) -> dict:
    path = HIST_DIR / f"torvik_{year}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {"data_type": data.get("data_type"), "tournament_start": data.get("tournament_start")}


# Kaggle's DayNum runs to ~132 (Selection Sunday) in a fully-ingested season.
# Anything materially short of that means the file was pulled mid-season and
# would understate every team's schedule.
KAGGLE_SEASON_COMPLETE_DAYNUM = 125


def _kaggle_season_games(year: int, canonical_ids):
    """Pre-tournament games for one season from Kaggle, or None if incomplete.

    `MRegularSeasonCompactResults` is the right universe by construction: it
    holds regular-season AND conference-tournament games, and NCAA tournament
    games live in a separate file. Verified empirically — 0 of 1,449 tournament
    games appear in it. So there is no date filter here, and nothing to get
    wrong: correctness comes from the file's scope rather than from trusting a
    timestamp.

    Returns a list of `(team, opponent, own_score, opp_score)` rows, both
    orientations, canonical ids. `None` means this season is not usable.
    """
    path = KAGGLE_DIR / "MRegularSeasonCompactResults.csv"
    kaggle_to_canonical = _load_kaggle_team_map(canonical_ids, year)
    if not path.exists() or not kaggle_to_canonical:
        return None

    rows = []
    max_day = -1
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["Season"]) != year:
                continue
            max_day = max(max_day, int(r["DayNum"]))
            w = kaggle_to_canonical.get(r["WTeamID"])
            lo = kaggle_to_canonical.get(r["LTeamID"])
            ws, ls = int(r["WScore"]), int(r["LScore"])
            rows.append((w, lo, ws, ls))
            rows.append((lo, w, ls, ws))

    if max_day < KAGGLE_SEASON_COMPLETE_DAYNUM:
        return None
    return rows


def _cbbpy_season_games(year: int, canonical_ids, tournament_start):
    """Fallback: pre-tournament games from the cbbpy log, date-filtered.

    Only correct when the log itself stops before the tournament, which is why
    the caller restricts this to the in-progress season. The historical logs
    carry unreliable per-game dates -- roughly 600-800 genuine regular-season
    games per season are stamped into April and would be silently discarded --
    so this path must never be used to backfill a completed season.
    """
    path = HIST_DIR / f"historical_games_{year}.json"
    if not path.exists():
        return None
    with open(path) as f:
        games = json.load(f).get("games", [])
    if not games:
        return None

    # Resolve the whole log at once, weighted by schedule length, so a non-D1
    # school that merely starts with a D1 school's name does not fold its
    # blowout losses into that team's volatility. `canonical_ids` here is the
    # full Torvik D1 list, which doubles as the disambiguating universe.
    appearances: dict[str, int] = {}
    for g in games:
        for key in ("team1_id", "team2_id"):
            raw = g.get(key)
            if raw:
                appearances[raw] = appearances.get(raw, 0) + 1
    bridge_map = resolve_cbbpy_bridge(appearances, canonical_ids, universe=_d1_universe(year))

    latest = max((g.get("date") or "") for g in games)
    if tournament_start and latest >= tournament_start:
        # The log runs into the tournament, so the per-game dates would have to
        # be trusted to separate them. They cannot be. Refuse rather than guess.
        return None

    rows = []
    for g in games:
        s1, s2 = g.get("team1_score"), g.get("team2_score")
        if s1 is None or s2 is None:
            continue
        t1 = bridge_map.get(g.get("team1_id", ""))
        t2 = bridge_map.get(g.get("team2_id", ""))
        rows.append((t1, t2, s1, s2))
        rows.append((t2, t1, s2, s1))
    return rows


def build_srs(year: int, canonical_ids):
    """Simple Rating System from this season's regular-season results.

    WHY THIS IS BUILT HERE RATHER THAN BOLTED ONTO THE TRAINING MATRIX. It
    exists to answer one question -- how much does barthag add once an
    opponent-adjusted rating computed from game results alone is present -- and
    that comparison is destroyed by a provenance difference between the two
    columns. Every other column in this table is cut at this season's
    tournament boundary; adding SRS from a separate path with its own cutoff
    semantics would make "barthag's incremental contribution" indistinguishable
    from "the two columns are bounded differently".

    So it runs through the same builder, on the same Kaggle regular-season file
    every other Kaggle-derived column here uses, which by construction contains
    no tournament games. custom_ratings.DEFAULT_CUTOFF_DAY (133, Selection
    Sunday) matches the tournament_start - 1 boundary audit check A enforces
    for the Torvik columns.

    Measured on inter-conference games, SRS is less conference-biased than
    barthag (+0.144 against +0.213, both point-in-time), so this is not a clean
    instrument replacing a dirty one -- both carry the confound, one carries
    less of it.
    """
    from src.data.features.custom_ratings import compute_srs_ratings

    raw = compute_srs_ratings(year, data_root=PROJECT_ROOT / "data")
    if not raw:
        return {}
    # USE THE SAME ID MAP AS EVERY OTHER KAGGLE-DERIVED COLUMN HERE.
    # custom_ratings.ratings_to_canonical is a second, independent mapping
    # (MTeams name -> normalize_team_id) and it dropped 113 of 1,085
    # team-seasons, 4-14 per year. Those rows would have standardised to 0.0,
    # i.e. "league average", which for a test of how much barthag adds OVER srs
    # would have silently handed barthag the credit on 10% of the field. Two id
    # maps in one table is the same provenance split this column exists to
    # avoid.
    kaggle_to_canonical = _load_kaggle_team_map(canonical_ids, year)
    if not kaggle_to_canonical:
        return {}
    out = {}
    for kid, v in raw.items():
        tid = kaggle_to_canonical.get(str(kid))
        if tid is not None and tid in canonical_ids:
            out[tid] = {"srs": round(v, 4)}
    return out


def build_regular_season_stats(year: int, canonical_ids, t_ranks, tournament_start, torvik=None):
    """Per-team pre-tournament form: scoring margin, volatility, close games.

    Returns `{canonical_team_id: {...}}` covering everything in the UI's "Form"
    group except `true_road_win_pct`, which comes from the Kaggle box score.

    SOURCE, AND WHY IT CHANGED. This used to read the cbbpy season log and drop
    games dated on or after `tournament_start`. That filter did keep every NCAA
    tournament game out -- checked game-by-game across 2010-2026, zero got
    through -- so it never leaked. It failed the other way: the logs' per-game
    dates are largely synthetic, and 600-800 real regular-season games a season
    were stamped past the cutoff and thrown away. Median games_played fell to
    22 in 2012 and 23 in 2021 against a true D1 schedule of ~31, and some teams
    had their entire form profile built from 7 games.

    Kaggle avoids the problem rather than patching it: its regular-season file
    simply does not contain tournament games, so no date logic is needed.

    STRENGTH OF SCHEDULE. `sos_avg_opp_barthag` is the mean of every regular-
    season opponent's OWN pre-tournament barthag (Torvik's power rating), i.e.
    "how good were the teams you played" rather than "how good are you." Torvik
    itself publishes a SOS/WAB column, but the scraper never captured it, so
    this is computed from what is already on hand instead of re-scraping.

    THIS COLUMN CORRELATES WITH CONFERENCE STRENGTH ON PURPOSE. An
    opponent-adjustment audit flags it at partial r = +0.29 against conference
    strength, controlling for team quality -- the same test that correctly
    flagged `reg_season_margin_avg` at -0.49 as a confound. Here it is not a
    confound, it is the measurement: a team in a strong conference genuinely
    played stronger opponents, and a schedule-strength column that did NOT move
    with conference strength would be broken. Do not "fix" it, and do not
    opponent-adjust it -- adjusting a measure of opposition for the quality of
    the opposition would leave nothing behind.
    Opponents outside the Torvik universe (non-D1, or D1 schools this season's
    torvik dict lacks) are skipped rather than guessed at, so a team with a lot
    of buy games has a smaller sample, not a wrong one.
    """
    rows = _kaggle_season_games(year, canonical_ids)
    if rows is None:
        # Kaggle has not ingested this season yet -- the in-progress year.
        rows = _cbbpy_season_games(year, canonical_ids, tournament_start)
    if rows is None:
        return {}

    torvik = torvik or {}
    margins: dict[str, list[int]] = {}
    bad_losses: dict[str, int] = {}
    opp_barthags: dict[str, list[float]] = {}
    for team, opp, own_score, opp_score in rows:
        if team is None:
            continue
        margins.setdefault(team, []).append(own_score - opp_score)
        # A "bad loss" needs both sides ranked, so non-D1 opponents (which
        # never bridge) are excluded from the numerator.
        if own_score < opp_score and opp is not None:
            own_rank, opp_rank = t_ranks.get(team), t_ranks.get(opp)
            if own_rank is not None and opp_rank is not None and opp_rank > own_rank:
                bad_losses[team] = bad_losses.get(team, 0) + 1
        opp_barthag = torvik.get(opp, {}).get("barthag") if opp is not None else None
        if opp_barthag is not None:
            opp_barthags.setdefault(team, []).append(opp_barthag)

    out = {}
    for team, vals in margins.items():
        n = len(vals)
        close = [m for m in vals if abs(m) <= CLOSE_GAME_MARGIN]
        sos = opp_barthags.get(team)
        out[team] = {
            "games_played": n,
            "reg_season_margin_avg": round(statistics.fmean(vals), 2),
            "reg_season_margin_std": round(statistics.pstdev(vals), 2) if n > 1 else None,
            "close_game_rate": round(len(close) / n, 4),
            "close_game_win_rate": (round(sum(1 for m in close if m > 0) / len(close), 4) if close else None),
            "losses_to_weaker_rate": round(bad_losses.get(team, 0) / n, 4),
            "sos_avg_opp_barthag": round(statistics.fmean(sos), 4) if sos else None,
        }
    return out


def build_tournament_outcomes(year: int):
    """Post-hoc: how far each team actually got. NOT pre-tournament data.

    Derived from the deepest round a team APPEARS in, not from per-game
    `team1_won` flags. Appearance is the more reliable signal: a bracket is
    single-elimination, so playing in the Sweet 16 proves you won two games
    no matter what the R32 record claims. This matters because the source
    data has at least one transposed game — `tournament_context_2018.json`
    records Cincinnati beating Nevada in the R32 while also showing Nevada
    (correctly) in the Sweet 16. Reality: Nevada won 75-73. A win-counting
    approach silently mislabels both teams there; this one self-corrects and
    prints a warning so the upstream bug stays visible.
    """
    games = load_tournament_results(year)
    if not games:
        return {}

    depth = {rnd: i for i, rnd in enumerate(BRACKET_ROUNDS)}  # R64=0 ... NCG=5
    deepest: dict[str, int] = {}
    losses: dict[str, list[str]] = {}
    ncg_winner = None
    for g in games:
        rnd, t1, t2 = g.get("round_name"), g.get("team1_id"), g.get("team2_id")
        if not t1 or not t2:
            continue
        d = depth.get(rnd, -1)  # FF (play-in) sits below R64 at -1
        for team in (t1, t2):
            deepest[team] = max(deepest.get(team, -1), d)
        if g.get("team1_won") is not None:
            loser = t2 if g["team1_won"] else t1
            losses.setdefault(loser, []).append(rnd)
            if rnd == "NCG":
                ncg_winner = t1 if g["team1_won"] else t2

    # Single-elimination: nobody can lose twice. More than one recorded loss
    # means the source file has a game whose result contradicts the bracket.
    for team, rounds in sorted(losses.items()):
        if len(rounds) > 1:
            print(
                f"    {year}: SOURCE DATA BUG — {team!r} recorded as losing "
                f"{len(rounds)} games ({', '.join(rounds)}); single-elimination "
                f"allows one. Using round-appearance instead."
            )

    out = {}
    for team, d in deepest.items():
        if d < 0:
            out[team] = {"outcome_finish": "First Four", "outcome_rounds_won": 0}
        elif d == depth["NCG"]:
            won = team == ncg_winner
            out[team] = {
                "outcome_finish": "Champion" if won else "Runner-up",
                "outcome_rounds_won": 6 if won else 5,
            }
        else:
            out[team] = {"outcome_finish": FINISH_ON_LOSS[BRACKET_ROUNDS[d]], "outcome_rounds_won": d}
    return out


def build_year_rows(year: int) -> list[dict]:
    seeds, regions = load_seeds_and_regions(year)
    if not seeds:
        return []  # no tournament that year (2020)

    meta = _torvik_meta(year)
    if meta.get("data_type") not in VALID_PRETOURNAMENT_TYPES:
        print(f"  {year}: SKIP — data_type={meta.get('data_type')!r}, not pre-tournament")
        return []

    torvik, _ff = load_torvik_and_ff(year)
    t_ranks = {tid: t.get("t_rank") for tid, t in torvik.items()}
    reg = build_regular_season_stats(year, set(torvik), t_ranks, meta.get("tournament_start"), torvik=torvik)
    outcomes = build_tournament_outcomes(year)
    rosters = build_roster_stats(year, set(torvik))
    overtime = build_overtime_rate(year, set(torvik))
    srs_ratings = build_srs(year, set(torvik))
    roster_minutes = build_roster_minutes(year, set(torvik))
    box_profile = build_kaggle_box_profile(year, set(torvik))
    coach = build_coach_experience(year, set(torvik))
    conf_tourney = build_conf_tourney_result(year, set(torvik))
    massey = build_massey_composite(year)
    scoring_depth = build_scoring_depth(year, set(torvik))

    rows = []
    for team_id, seed in seeds.items():
        # The seeds file and Torvik disagree on a few IDs; alias across so the
        # team keeps its stats instead of vanishing from the bracket.
        tv_id = _SEEDS_TO_TORVIK_ALIASES.get(team_id, team_id)
        t = torvik.get(tv_id)
        if not t:
            print(f"    {year}: no Torvik row for {team_id!r} — dropped")
            continue
        row = {
            "team_id": team_id,
            "team_name": t.get("team_name"),
            "seed": seed,
            "region": regions.get(team_id),
        }
        for field in STAT_FIELDS:
            row[field] = t.get(field)
        row.update(
            reg.get(
                tv_id,
                {
                    "games_played": None,
                    "reg_season_margin_avg": None,
                    "reg_season_margin_std": None,
                    "close_game_rate": None,
                    "close_game_win_rate": None,
                    "losses_to_weaker_rate": None,
                    "sos_avg_opp_barthag": None,
                },
            )
        )
        row.update(
            rosters.get(
                tv_id,
                {"n_returning_players": None},
            )
        )
        row.update(overtime.get(tv_id, {"overtime_rate": None}))
        row.update(srs_ratings.get(tv_id, {"srs": None}))
        row.update(
            roster_minutes.get(
                tv_id,
                {"returning_minutes_pct": None, "freshman_minutes_pct": None},
            )
        )
        row.update(
            box_profile.get(
                tv_id,
                {
                    "three_pt_rate": None,
                    "three_pt_pct": None,
                    "opp_three_pt_pct": None,
                    "ast_to_ratio": None,
                    "havoc_rate": None,
                    "true_road_win_pct": None,
                },
            )
        )
        row.update(
            coach.get(
                tv_id,
                {
                    "coach_name": None,
                    "coach_prior_tourney_games": None,
                    "coach_prior_tourney_wins": None,
                    "coach_first_tourney": None,
                },
            )
        )
        row.update(outcomes.get(team_id, {"outcome_finish": None, "outcome_rounds_won": None}))
        row.update(conf_tourney.get(tv_id, {"conf_tourney_wins": None}))
        row.update(massey.get(tv_id, {"massey_avg_rank": None}))
        row.update(scoring_depth.get(tv_id, {"n_double_digit_scorers": None}))
        rows.append(row)

    rows.sort(key=lambda r: r["t_rank"] if r["t_rank"] is not None else 999)
    return rows


def attach_seed_deltas(stats_by_year: dict) -> dict:
    """Add `outcome_vs_seed_delta` = rounds won minus the average for that seed.

    The baseline is computed across every year in this dataset, so it is a
    descriptive "how did this team do relative to a typical team on its seed
    line" — not a forecast. Like the other `outcome_*` fields it is post-hoc.
    """
    by_seed: dict[int, list[int]] = {}
    for rows in stats_by_year.values():
        for r in rows:
            if r.get("outcome_rounds_won") is not None:
                by_seed.setdefault(r["seed"], []).append(r["outcome_rounds_won"])
    expected = {seed: statistics.fmean(vals) for seed, vals in by_seed.items()}
    for rows in stats_by_year.values():
        for r in rows:
            exp = expected.get(r["seed"])
            if r.get("outcome_rounds_won") is None or exp is None:
                r["outcome_vs_seed_delta"] = None
            else:
                r["outcome_vs_seed_delta"] = round(r["outcome_rounds_won"] - exp, 2)
    return {seed: round(v, 3) for seed, v in sorted(expected.items())}


def attach_historical_residual(stats_by_year: dict) -> None:
    """Add `hist_residual` — a program's tournament over/under-performance to date.

    For team T in year Y: the mean of T's `outcome_vs_seed_delta` across its
    tournament appearances in years STRICTLY BEFORE Y. Because it only ever
    looks backwards, this is genuinely pre-tournament information — unlike
    the `outcome_*` block it sits alongside the other stat columns.

    This is the "does a program's March history tell you anything beyond its
    current seed and strength?" question, expressed as a residual rather than
    a raw count of Final Fours (which would just re-encode program strength).

    `hist_appearances` carries the sample size, which matters a lot here: the
    dataset starts in 2010, so early years have little or no prior history
    and a single appearance makes for a very noisy residual.

    The per-seed baseline inside `outcome_vs_seed_delta` is pooled across all
    years rather than walked forward. It is a structural constant (a 1-seed
    averages ~3.3 wins, a fact established over decades and not by this
    16-year window) and carries no team-specific information, so it leaks
    nothing about the team whose residual is being computed — while a
    walk-forward baseline would make 2011-2013 unusable.
    """
    history: dict[str, list[float]] = {}
    for year in sorted(stats_by_year, key=int):
        rows = stats_by_year[year]
        # Read history BEFORE folding this year in, so a team never sees itself.
        for r in rows:
            past = history.get(r["team_id"], [])
            r["hist_appearances"] = len(past)
            r["hist_residual"] = round(statistics.fmean(past), 2) if past else None
        for r in rows:
            if r.get("outcome_vs_seed_delta") is not None:
                history.setdefault(r["team_id"], []).append(r["outcome_vs_seed_delta"])


def main() -> None:
    stats_by_year = {}
    for year in YEARS:
        rows = build_year_rows(year)
        if rows:
            stats_by_year[str(year)] = rows
            with_reg = sum(1 for r in rows if r["games_played"])
            print(f"  {year}: {len(rows)} teams ({with_reg} with game-log stats)")

    expected_by_seed = attach_seed_deltas(stats_by_year)
    attach_historical_residual(stats_by_year)

    payload = {
        "years": sorted(int(y) for y in stats_by_year),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "expected_rounds_won_by_seed": expected_by_seed,
        "stats_by_year": stats_by_year,
    }

    out_path = OUT / "team_stats_by_year.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)  # NaN/Infinity are not valid JSON
        f.write("\n")
    print(f"\nWrote {out_path} ({len(payload['years'])} years)")


if __name__ == "__main__":
    main()
