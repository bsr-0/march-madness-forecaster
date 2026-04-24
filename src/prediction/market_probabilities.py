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
    tourney_games = [
        g for g in valid_games
        if g.home_team_id in tourney_teams or g.away_team_id in tourney_teams
    ]
    if len(tourney_games) < MIN_GAMES_THRESHOLD:
        logger.warning(
            "Only %d games involving tournament teams for %d (need %d), returning None",
            len(tourney_games), year, MIN_GAMES_THRESHOLD,
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
            n_fallback, len(tourney_teams),
        )

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
