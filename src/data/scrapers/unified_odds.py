"""Unified betting odds data layer.

Normalizes game-level odds from multiple sources (SBRO, Covers, SBR)
into a consistent schema for pipeline integration.

Sources:
    SBRO (2008-2022): Full season. Spreads (open/close), ML, totals.
    Covers (2023-2026): Full season. Spreads, totals. No moneylines.
    SBR (2025-2026): Tournament only. Spreads, ML, totals.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from ..normalize import normalize_team_id
from .betting_markets import american_to_probability

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed/betting_odds"


@dataclass
class UnifiedGameOdds:
    """Normalized game-level odds record."""

    season: int
    game_date: str
    home_team_id: str
    away_team_id: str
    spread: float = 0.0
    spread_open: float = 0.0
    total: float = 0.0
    total_open: float = 0.0
    moneyline_home: float = 0.0
    moneyline_away: float = 0.0
    implied_prob_home: float = 0.5
    implied_prob_away: float = 0.5
    home_score: int = 0
    away_score: int = 0
    is_neutral: bool = False
    source: str = ""


# ------------------------------------------------------------------
# Normalizers: raw JSON dict → UnifiedGameOdds
# ------------------------------------------------------------------

def normalize_from_sbro(raw: dict) -> UnifiedGameOdds:
    return UnifiedGameOdds(
        season=raw.get("season", 0),
        game_date=raw.get("game_date", ""),
        home_team_id=raw.get("home_team_id", ""),
        away_team_id=raw.get("away_team_id", ""),
        spread=raw.get("spread", 0.0),
        spread_open=raw.get("spread_open", 0.0),
        total=raw.get("total", 0.0),
        total_open=raw.get("total_open", 0.0),
        moneyline_home=raw.get("moneyline_home", 0.0),
        moneyline_away=raw.get("moneyline_away", 0.0),
        implied_prob_home=raw.get("implied_prob_home", 0.5),
        implied_prob_away=raw.get("implied_prob_away", 0.5),
        home_score=raw.get("home_score", 0),
        away_score=raw.get("away_score", 0),
        is_neutral=raw.get("is_neutral", False),
        source="sbro",
    )


def normalize_from_covers(raw: dict) -> UnifiedGameOdds:
    return UnifiedGameOdds(
        season=raw.get("season", 0),
        game_date=raw.get("game_date", ""),
        home_team_id=raw.get("home_team_id", ""),
        away_team_id=raw.get("away_team_id", ""),
        spread=raw.get("spread", 0.0),
        total=raw.get("total", 0.0),
        implied_prob_home=raw.get("implied_prob_home", 0.5),
        implied_prob_away=raw.get("implied_prob_away", 0.5),
        home_score=raw.get("home_score", 0),
        away_score=raw.get("away_score", 0),
        is_neutral=raw.get("is_neutral", False),
        source="covers",
    )


def normalize_from_sbr(raw: dict) -> UnifiedGameOdds:
    # SBR team IDs include mascots — re-normalize from team name
    home_id = normalize_team_id(raw.get("home_team_name", ""))
    away_id = normalize_team_id(raw.get("away_team_name", ""))

    return UnifiedGameOdds(
        season=raw.get("season", 0),
        game_date=raw.get("game_date", ""),
        home_team_id=home_id,
        away_team_id=away_id,
        spread=raw.get("spread", 0.0),
        total=raw.get("total", 0.0),
        moneyline_home=raw.get("moneyline_home", 0.0),
        moneyline_away=raw.get("moneyline_away", 0.0),
        implied_prob_home=raw.get("implied_prob_home", 0.5),
        implied_prob_away=raw.get("implied_prob_away", 0.5),
        source="sbr",
    )


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_unified_odds(season: int, data_dir: str = PROCESSED_DIR) -> List[UnifiedGameOdds]:
    """Load unified odds for a season from processed JSON."""
    path = os.path.join(data_dir, f"unified_odds_{season}.json")
    if not os.path.isfile(path):
        return []

    with open(path) as f:
        data = json.load(f)

    return [UnifiedGameOdds(**g) for g in data.get("games", [])]


def load_unified_odds_by_team(
    season: int, data_dir: str = PROCESSED_DIR
) -> Dict[str, List[UnifiedGameOdds]]:
    """Load unified odds grouped by team, sorted by date."""
    games = load_unified_odds(season, data_dir)
    by_team: Dict[str, List[UnifiedGameOdds]] = defaultdict(list)
    for g in games:
        by_team[g.home_team_id].append(g)
        by_team[g.away_team_id].append(g)
    for tid in by_team:
        by_team[tid].sort(key=lambda g: g.game_date)
    return dict(by_team)


# ------------------------------------------------------------------
# Feature computation (point-in-time safe)
# ------------------------------------------------------------------

def compute_team_market_features(
    team_id: str,
    as_of_date: str,
    team_odds: List[UnifiedGameOdds],
) -> Dict[str, float]:
    """Compute market-implied features using only games before as_of_date.

    Returns dict with:
        market_implied_prob: avg implied win probability for this team
        market_spread: avg closing spread from team's perspective
    """
    if not team_odds:
        return {"market_implied_prob": 0.0, "market_spread": 0.0}

    probs = []
    spreads = []

    for g in team_odds:
        if g.game_date >= as_of_date:
            continue
        if g.spread == 0.0 and g.implied_prob_home == 0.5:
            continue  # no real odds data

        if g.home_team_id == team_id:
            probs.append(g.implied_prob_home)
            spreads.append(g.spread)  # negative = favored
        elif g.away_team_id == team_id:
            probs.append(g.implied_prob_away)
            spreads.append(-g.spread)  # flip to away perspective

    if not probs:
        return {"market_implied_prob": 0.0, "market_spread": 0.0}

    return {
        "market_implied_prob": sum(probs) / len(probs),
        "market_spread": sum(spreads) / len(spreads),
    }
