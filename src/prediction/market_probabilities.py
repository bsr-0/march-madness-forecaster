"""Market-implied team ratings from betting odds via Bradley-Terry model.

Converts season-long betting odds into per-team strength ratings (barthag-
equivalent) that plug directly into the existing MC bracket simulation.

Data source: data/processed/betting_odds/unified_odds_{year}.json (2008-2026)
Algorithm: Bradley-Terry MLE on implied probabilities from closing lines
Output: Dict[team_id, float] in [0, 1] — same format as _load_torvik_barthag()

See STRATEGY_CATALOG.md bases A3 (odds) and A7 (spread_power).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Tournament cutoff: use only games before this date to prevent look-ahead.
# Selection Sunday is typically mid-March; March 15 is conservative.
DEFAULT_CUTOFF_MMDD = "03-15"

# Minimum games involving tournament teams required for a valid fit.
MIN_GAMES_THRESHOLD = 100

# Bradley-Terry iterations. Convergence is guaranteed for connected graphs;
# 30 iterations is more than sufficient for ~5000 games.
BT_MAX_ITER = 30


def load_market_ratings(
    year: int,
    seeds: Dict[str, int],
    cutoff_date: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Derive barthag-equivalent ratings from season betting odds.

    DEFECTIVE CONSTRUCTION -- kept only to reproduce pre-audit numbers.
    The 2026-09 referee qualification audit found (from the inputs alone,
    before any calibration was seen) that this loader (1) never resolves the
    SBRO-era team spellings, so 27-33 of 68 tournament teams per season in
    2011-2022 fall to the seed proxy, and (2) drops every |spread| > 5 game
    whose spread sign disagrees with the implied probability, a convention
    SBRO rows follow only 29-45% of the time, so most decisive games are
    discarded. Since the methodology audit (Step 7, D7-1) every runtime
    caller uses :func:`load_market_ratings_v2`; call this one only with
    ``allow_defective=True`` semantics in mind, i.e. never for a new number.

    Fits a Bradley-Terry model to all regular-season games with odds data,
    using market-implied win probabilities as observations. Produces a
    per-team strength parameter that can be fed to Log5 for arbitrary
    pairwise matchup probabilities.

    Args:
        year: Season year (e.g., 2025)
        seeds: Tournament teams {team_id: seed}. Only these teams appear
            in the output; non-tournament teams are used in the fit but
            not returned.
        cutoff_date: ISO date string (YYYY-MM-DD). Only games before this
            date are used. Defaults to {year}-03-15.

    Returns:
        Dict[team_id, float] for all teams in seeds, or None if
        insufficient odds data is available.
    """
    from src.data.scrapers.unified_odds import load_unified_odds

    if cutoff_date is None:
        cutoff_date = f"{year}-{DEFAULT_CUTOFF_MMDD}"

    games = load_unified_odds(year)
    if not games:
        logger.warning("No unified odds for %d, returning None", year)
        return None

    # Filter to pre-tournament games only (PIT compliance)
    games = [g for g in games if g.game_date < cutoff_date]

    # Collect all teams that appear in the odds data
    all_teams = set()
    valid_games = []
    for g in games:
        # Skip games with degenerate probabilities
        if g.implied_prob_home <= 0.01 or g.implied_prob_home >= 0.99:
            continue
        all_teams.add(g.home_team_id)
        all_teams.add(g.away_team_id)
        valid_games.append(g)

    # Check if we have enough tournament-team games
    tourney_teams = set(seeds.keys())
    tourney_games = [g for g in valid_games if g.home_team_id in tourney_teams or g.away_team_id in tourney_teams]
    if len(tourney_games) < MIN_GAMES_THRESHOLD:
        logger.warning(
            "Only %d games involving tournament teams for %d (need %d), returning None",
            len(tourney_games),
            year,
            MIN_GAMES_THRESHOLD,
        )
        return None

    # --- Bradley-Terry MLE ---
    # For each game, implied_prob_home gives P(home wins). We treat this
    # as a "soft outcome" — the market's belief, not the binary result.
    # This means we're fitting to market consensus, not actual outcomes,
    # which is what we want (market probabilities are better calibrated
    # than binary W/L for estimating true team strength).

    team_list = sorted(all_teams)
    team_idx = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)

    # Initialize ratings uniformly
    r = [1.0] * n_teams

    for iteration in range(BT_MAX_ITER):
        # Accumulate numerator (wins) and denominator for each team
        wins = [0.0] * n_teams
        denom = [0.0] * n_teams

        for g in valid_games:
            i = team_idx[g.home_team_id]
            j = team_idx[g.away_team_id]
            p_home = g.implied_prob_home

            # Data quality guard: if spread and implied_prob disagree on
            # who is favored, skip the game. Some Covers data has inverted
            # implied probs (see Florida 2025 investigation).
            if abs(g.spread) > 5.0:
                spread_says_home_fav = g.spread < 0
                prob_says_home_fav = p_home > 0.5
                if spread_says_home_fav != prob_says_home_fav:
                    continue  # skip inconsistent game

            # Market-implied "wins": fractional wins based on implied prob
            wins[i] += p_home
            wins[j] += 1.0 - p_home

            # Bradley-Terry denominator: 1 / (r_i + r_j) for each game
            pair_sum = r[i] + r[j]
            if pair_sum > 1e-12:
                inv = 1.0 / pair_sum
                denom[i] += inv
                denom[j] += inv

        # Update ratings
        max_delta = 0.0
        for k in range(n_teams):
            if denom[k] > 1e-12:
                new_r = wins[k] / denom[k]
            else:
                new_r = r[k]
            max_delta = max(max_delta, abs(new_r - r[k]))
            r[k] = new_r

        # Normalize to prevent numerical drift (anchor median to 1.0)
        sorted_r = sorted(r)
        median_r = sorted_r[n_teams // 2]
        if median_r > 1e-12:
            for k in range(n_teams):
                r[k] /= median_r

        if max_delta < 1e-6:
            logger.debug("Bradley-Terry converged in %d iterations", iteration + 1)
            break

    # Convert ratings to barthag (probability of beating a median team)
    # Since we normalized median_r = 1.0, barthag = r / (r + 1)
    barthag = {}
    for tid in tourney_teams:
        if tid in team_idx:
            rating = r[team_idx[tid]]
            barthag[tid] = rating / (rating + 1.0)
        else:
            # Team not in odds data — use seed-based fallback
            seed = seeds[tid]
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)
            logger.debug("No odds for %s (seed %d), using fallback barthag=%.3f", tid, seed, barthag[tid])

    n_fallback = sum(1 for tid in tourney_teams if tid not in team_idx)
    if n_fallback > 0:
        logger.info(
            "%d/%d tournament teams missing from odds, using seed fallback",
            n_fallback,
            len(tourney_teams),
        )

    return barthag


def load_odds_api_market_ratings(
    year: int,
    seeds: Dict[str, int],
    cutoff_date: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Derive barthag-equivalent ratings from the Odds API closing consensus.

    Uses direct multi-book H2H consensus closing probabilities (avg ~13 books)
    instead of the spread-conversion approach in load_market_ratings(). Available
    for seasons 2021-2026 only; returns None for earlier years.

    Args:
        year: Season year (e.g., 2025)
        seeds: Tournament teams {team_id: seed}
        cutoff_date: ISO date string (YYYY-MM-DD). Only games before this
            date are used. Defaults to {year}-03-15.

    Returns:
        Dict[team_id, float] for all teams in seeds, or None if year is outside
        2021-2026 or insufficient data is available.
    """
    import json
    from pathlib import Path

    if year < 2021:
        logger.debug("No Odds API coverage before 2021, returning None for %d", year)
        return None

    if cutoff_date is None:
        cutoff_date = f"{year}-{DEFAULT_CUTOFF_MMDD}"

    artifact_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "odds_api_closing_consensus.json"
    if not artifact_path.exists():
        logger.warning("odds_api_closing_consensus.json not found, returning None for %d", year)
        return None

    with open(artifact_path) as f:
        data = json.load(f)

    raw_games = data.get("games", [])

    # Filter: correct season, not tournament window, before cutoff, enough books
    valid_games = []
    all_teams: set[str] = set()
    for g in raw_games:
        if g.get("season") != year:
            continue
        if g.get("tournament_window", False):
            continue
        if g.get("closing_snapshot", "9999") >= cutoff_date:
            continue
        p_home = g.get("home_win_prob")
        p_away = g.get("away_win_prob")
        if p_home is None or p_away is None:
            continue
        p_home = float(p_home)
        if p_home <= 0.01 or p_home >= 0.99:
            continue
        h_id = g.get("home_team_id")
        a_id = g.get("away_team_id")
        if not h_id or not a_id:
            continue
        all_teams.add(h_id)
        all_teams.add(a_id)
        valid_games.append((h_id, a_id, p_home))

    tourney_teams = set(seeds.keys())
    tourney_games = [(h, a, p) for h, a, p in valid_games if h in tourney_teams or a in tourney_teams]
    if len(tourney_games) < MIN_GAMES_THRESHOLD:
        logger.warning(
            "Only %d tournament-team games from Odds API for %d (need %d), returning None",
            len(tourney_games),
            year,
            MIN_GAMES_THRESHOLD,
        )
        return None

    # Bradley-Terry MLE on direct consensus probabilities
    team_list = sorted(all_teams)
    team_idx = {t: i for i, t in enumerate(team_list)}
    n_teams = len(team_list)
    r = [1.0] * n_teams

    for iteration in range(BT_MAX_ITER):
        wins = [0.0] * n_teams
        denom = [0.0] * n_teams

        for h_id, a_id, p_home in valid_games:
            i = team_idx[h_id]
            j = team_idx[a_id]
            wins[i] += p_home
            wins[j] += 1.0 - p_home
            pair_sum = r[i] + r[j]
            if pair_sum > 1e-12:
                inv = 1.0 / pair_sum
                denom[i] += inv
                denom[j] += inv

        max_delta = 0.0
        for k in range(n_teams):
            new_r = wins[k] / denom[k] if denom[k] > 1e-12 else r[k]
            max_delta = max(max_delta, abs(new_r - r[k]))
            r[k] = new_r

        sorted_r = sorted(r)
        median_r = sorted_r[n_teams // 2]
        if median_r > 1e-12:
            for k in range(n_teams):
                r[k] /= median_r

        if max_delta < 1e-6:
            logger.debug("Odds API BT converged in %d iterations for %d", iteration + 1, year)
            break

    barthag = {}
    for tid in tourney_teams:
        if tid in team_idx:
            rating = r[team_idx[tid]]
            barthag[tid] = rating / (rating + 1.0)
        else:
            seed = seeds[tid]
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)
            logger.debug("No Odds API data for %s (seed %d), using seed fallback", tid, seed)

    n_fallback = sum(1 for tid in tourney_teams if tid not in team_idx)
    if n_fallback > 0:
        logger.info("%d/%d tournament teams missing from Odds API, using seed fallback", n_fallback, len(tourney_teams))

    return barthag


def load_spread_power_ratings(
    year: int,
    seeds: Dict[str, int],
    cutoff_date: Optional[str] = None,
    home_court_adjustment: float = 3.5,
) -> Optional[Dict[str, float]]:
    """Derive team ratings from average closing spread (A7: spread_power).

    Simpler than Bradley-Terry: just averages each team's closing spread
    across the season and converts to a barthag-equivalent via logistic.

    Args:
        year: Season year
        seeds: Tournament teams {team_id: seed}
        cutoff_date: PIT cutoff (default: {year}-03-15)
        home_court_adjustment: Points subtracted from home spread to
            neutralize home court advantage (default 3.5)

    Returns:
        Dict[team_id, float] or None if insufficient data.
    """
    from src.data.scrapers.unified_odds import load_unified_odds

    if cutoff_date is None:
        cutoff_date = f"{year}-{DEFAULT_CUTOFF_MMDD}"

    games = load_unified_odds(year)
    if not games:
        return None

    games = [g for g in games if g.game_date < cutoff_date]

    tourney_teams = set(seeds.keys())

    # Use implied_prob (not raw spread) — spread sign convention is
    # unreliable across sources, but implied_prob is always consistent.
    # Convert implied_prob to a logit "spread equivalent" per game.
    team_logits: Dict[str, list] = {tid: [] for tid in tourney_teams}

    for g in games:
        p_home = g.implied_prob_home
        if p_home <= 0.01 or p_home >= 0.99:
            continue

        # Convert to logit space (positive = home favored)
        home_logit = math.log(p_home / (1.0 - p_home))

        # Adjust for home court advantage in non-neutral games
        # HCA ≈ 3.5 points ≈ 0.875 logit units
        if not g.is_neutral:
            hca_logit = home_court_adjustment / 4.0
            home_logit -= hca_logit

        away_logit = -home_logit

        if g.home_team_id in team_logits:
            team_logits[g.home_team_id].append(home_logit)
        if g.away_team_id in team_logits:
            team_logits[g.away_team_id].append(away_logit)

    # Need at least 5 games per team
    barthag = {}
    for tid in tourney_teams:
        logits = team_logits.get(tid, [])
        if len(logits) >= 5:
            avg_logit = sum(logits) / len(logits)
            # Logistic transform back to probability
            barthag[tid] = 1.0 / (1.0 + math.exp(-avg_logit))
        else:
            seed = seeds[tid]
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)

    if len(barthag) < len(tourney_teams) * 0.8:
        logger.warning("Too few teams with spread data for %d", year)
        return None

    return barthag


# ---------------------------------------------------------------------------
# market_v2: the corrected market referee (referee qualification audit)
# ---------------------------------------------------------------------------
#
# Defined in artifacts/referee_audit/PREREGISTRATION_QUALIFICATION.md BEFORE
# its calibration was measured. `load_market_ratings` above is left exactly as
# it was: it is the harness's `odds` base and the first audit's `market`
# referee, and rewriting it would move numbers that are already published.
#
# What v2 changes, and why each is a construction fix rather than a tuning:
#   * team ids: the unified odds file carries un-normalised SBRO spellings
#     (`ohiostate`, `ohio_state_buckeyes`) so v1 handed ~30 of 68 tournament
#     teams per season the seed fallback. v2 resolves both sides through the
#     curated TeamNameResolver's high-confidence tiers plus a short alias list.
#   * the spread-sign consistency guard is gone: SBRO's sign convention is
#     mixed (55-71% agreement), so the guard discarded most decisive games.
#     Implied probability is the only signal, as `load_spread_power_ratings`
#     already does.
#   * home court: the repo's spread_power convention (3.5 pts / 4 = 0.875
#     logit) is removed from non-neutral games before the fit.

#: SBRO compact spellings the resolver's strict tiers cannot place. Values are
#: the ids the tournament seeds files use. Curated from the unresolved list
#: across 2011-2025, with no outcome data involved.
ODDS_ID_ALIASES: Dict[str, str] = {
    "vcu_rams": "virginia_commonwealth",
    "vacommonwealth": "virginia_commonwealth",
    "st_johns": "st__john_s__ny",
    "longisland": "long_island_university",
    "liu_brooklyn": "long_island_university",
    "liu_brooklyn_blackbirds": "long_island_university",
    "stephenaustin": "stephen_f_austin",
    "bostonu": "boston_university",
    "ncasheville": "unc_asheville",
    "nc_asheville_bulldogs": "unc_asheville",
    "southernmiss": "southern_miss",
    "loyolamaryland": "loyola_md",
    "middletennst": "middle_tennessee",
    "middle_tn": "middle_tennessee",
    "middle_tennessee_st_blue_raiders": "middle_tennessee",
    "n_carolinaa_t": "north_carolina_a_t",
    "n_carolinaat": "north_carolina_a_t",
    "northwesternst": "northwestern_state",
    "centralflorida": "ucf",
    "ucf_knights": "ucf",
    "prairieviewa_m": "prairie_view",
    "prairieviewam": "prairie_view",
    "prairie_view_a_m_panthers": "prairie_view",
    "st_francispa": "saint_francis",
    "miamiflorida": "miami__fl",
    "miamiohio": "miami__oh",
    "calsantabarb": "uc_santa_barbara",
    "calirvine": "uc_irvine",
    "csfullerton": "cal_state_fullerton",
    "csbakersfield": "cal_state_bakersfield",
    "csnorthridge": "cal_state_northridge",
    "calpolyslo": "cal_poly",
    "flagulfcoast": "florida_gulf_coast",
    "ncwilmington": "unc_wilmington",
    "ncgreensboro": "unc_greensboro",
    "nccentral": "north_carolina_central",
    "e_washington": "eastern_washington",
    "loyolachicago": "loyola__il",
    "mdbaltimoreco": "maryland_baltimore_county",
    "collcharleston": "college_of_charleston",
    "etennesseest": "east_tennessee_state",
    "wiscgreenbay": "green_bay",
    "wiscmilwaukee": "milwaukee",
    "st_josephs": "saint_joseph_s",
    "geowashington": "george_washington",
    "ullafayette": "louisiana",
    "arkansaslr": "little_rock",
    "no_colorado": "northern_colorado",
    "texsanantonio": "utsa",
    "texasa_mcorpus": "texas_a_m_corpus_christi",
    "detroitu": "detroit_mercy",
}

# SBRO also writes a trailing "u" for a university ("indianau", "houstonu") and
# a trailing "st" for "state" ("appalachianst"). Each is expanded and accepted
# ONLY on exact compact-spelling equality with a canonical id, so the rule can
# merge a spelling but never guess one.
_SBRO_SUFFIX_EXPANSIONS = (("u", ""), ("st", "state"))

_STRICT_RESOLVER_METHODS = frozenset({"exact_id", "alias", "alias_id", "slug", "prefix_strip"})
HOME_COURT_LOGIT = 3.5 / 4.0  # the spread_power convention, unchanged


def _make_canonical_key():
    """Return ``key(team_id) -> canonical id or None`` using only high-confidence resolution."""
    from src.data.team_name_resolver import TeamNameResolver

    resolver = TeamNameResolver()
    known = set(resolver.known_teams)  # a property on this resolver, not a method
    compact = {k.replace("_", ""): k for k in known}
    cache: Dict[str, Optional[str]] = {}

    def key(team_id: str) -> Optional[str]:
        if team_id in cache:
            return cache[team_id]
        tid = ODDS_ID_ALIASES.get(team_id, team_id)
        out: Optional[str]
        if tid in known:
            out = tid
        elif tid.replace("_", "") in compact:
            out = compact[tid.replace("_", "")]
        elif any(tid.endswith(suf) and (tid[: -len(suf)] + rep).replace("_", "") in compact for suf, rep in _SBRO_SUFFIX_EXPANSIONS):
            suf, rep = next((s, r) for s, r in _SBRO_SUFFIX_EXPANSIONS if tid.endswith(s) and (tid[: -len(s)] + r).replace("_", "") in compact)
            out = compact[(tid[: -len(suf)] + rep).replace("_", "")]
        else:
            m = resolver.resolve(tid)
            out = m.canonical_id if m.method in _STRICT_RESOLVER_METHODS else None
        cache[team_id] = out
        return out

    return key


def load_market_ratings_v2(
    year: int,
    seeds: Dict[str, int],
    cutoff_date: Optional[str] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, float]]:
    """Corrected market referee. See the block comment above and the pre-registration.

    ``diagnostics``, if given, receives ``n_games``, ``n_resolved_teams``,
    ``fallback_teams`` (tournament teams still absent from the odds data,
    which keep the seed fallback so the count is visible, never hidden).
    """
    from src.data.scrapers.unified_odds import load_unified_odds

    if cutoff_date is None:
        cutoff_date = f"{year}-{DEFAULT_CUTOFF_MMDD}"
    games = [g for g in load_unified_odds(year) if g.game_date < cutoff_date]
    if not games:
        return None

    key = _make_canonical_key()
    # Seed ids canonicalised the same way; an id the resolver does not know keeps itself.
    seed_key = {tid: (key(tid) or tid) for tid in seeds}

    fitted = []
    for g in games:
        p = g.implied_prob_home
        if p <= 0.01 or p >= 0.99:
            continue
        h, a = key(g.home_team_id), key(g.away_team_id)
        if h is None or a is None or h == a:
            continue
        if not g.is_neutral:
            logit = math.log(p / (1.0 - p)) - HOME_COURT_LOGIT
            p = 1.0 / (1.0 + math.exp(-logit))
        fitted.append((h, a, p))

    tourney_keys = set(seed_key.values())
    if sum(1 for h, a, _ in fitted if h in tourney_keys or a in tourney_keys) < MIN_GAMES_THRESHOLD:
        return None

    team_list = sorted({t for h, a, _ in fitted for t in (h, a)})
    idx = {t: i for i, t in enumerate(team_list)}
    r = [1.0] * len(team_list)
    for _ in range(BT_MAX_ITER):
        wins = [0.0] * len(team_list)
        denom = [0.0] * len(team_list)
        for h, a, p in fitted:
            i, j = idx[h], idx[a]
            wins[i] += p
            wins[j] += 1.0 - p
            inv = 1.0 / (r[i] + r[j])
            denom[i] += inv
            denom[j] += inv
        max_delta = 0.0
        for k in range(len(team_list)):
            new_r = wins[k] / denom[k] if denom[k] > 1e-12 else r[k]
            max_delta = max(max_delta, abs(new_r - r[k]))
            r[k] = new_r
        median_r = sorted(r)[len(r) // 2]
        if median_r > 1e-12:
            r = [x / median_r for x in r]
        if max_delta < 1e-6:
            break

    barthag: Dict[str, float] = {}
    fallback = []
    for tid, seed in seeds.items():
        k = seed_key[tid]
        if k in idx:
            barthag[tid] = r[idx[k]] / (r[idx[k]] + 1.0)
        else:
            barthag[tid] = max(0.10, 1.0 - seed * 0.04)
            fallback.append(tid)
    if diagnostics is not None:
        diagnostics.update({"n_games": len(fitted), "n_resolved_teams": len(seeds) - len(fallback), "fallback_teams": sorted(fallback)})
    return barthag
