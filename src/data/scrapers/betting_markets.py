"""
Betting market integration for tournament predictions.

Scrapes sportsbook implied probabilities (FanDuel, DraftKings, etc.)
to provide a "market consensus" signal.  Market odds reflect the
aggregated wisdom of sharp bettors and are often better calibrated
than public pick percentages.

The market consensus can be:
1. Used as a post-hoc calibration blend with model predictions.
2. Compared to public picks to identify "smart money" divergence.
3. Fed into the EV optimizer as an alternative "field model".
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BettingMarketOdds:
    """Odds for a single team from a single sportsbook."""

    team_id: str
    team_name: str
    season: int
    source: str  # "fanduel", "draftkings", "betmgm", "consensus"
    championship_odds: float  # American odds (e.g., +500, -150)
    implied_probability: float  # Derived from odds (0-1)
    timestamp: str = ""
    confidence: float = 1.0  # Freshness weight (0-1); lower for stale data


@dataclass
class MarketConsensus:
    """Aggregated market consensus across multiple sportsbooks."""

    team_probabilities: Dict[str, float]  # team_id -> implied P(championship)
    sources: List[str]  # Contributing sportsbooks
    timestamp: str = ""
    vig_adjusted: bool = False  # Whether probabilities sum to 1.0

    def get_top_teams(self, n: int = 10) -> List[tuple]:
        """Return top-N teams by implied championship probability."""
        sorted_teams = sorted(
            self.team_probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_teams[:n]


# ---------------------------------------------------------------------------
# Odds conversion utilities
# ---------------------------------------------------------------------------

def american_to_probability(odds: float) -> float:
    """Convert American odds to implied probability.

    Args:
        odds: American odds (positive for underdogs, negative for favorites).
              e.g., +500 means bet $100 to win $500.
              e.g., -150 means bet $150 to win $100.

    Returns:
        Implied probability in [0, 1].
    """
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 0.5


def decimal_to_probability(odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        odds: Decimal odds (e.g., 6.0 for +500).

    Returns:
        Implied probability in [0, 1].
    """
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def remove_vig(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Remove bookmaker vig (overround) by normalizing to sum to 1.0.

    Sportsbook implied probabilities typically sum to >1.0 due to the
    bookmaker's margin.  This normalizes them to a proper distribution.

    Args:
        probabilities: team_id -> raw implied probability.

    Returns:
        Vig-adjusted probabilities summing to 1.0.
    """
    total = sum(probabilities.values())
    if total <= 0:
        return probabilities
    return {tid: p / total for tid, p in probabilities.items()}


# ---------------------------------------------------------------------------
# Scraper base class
# ---------------------------------------------------------------------------

class BettingMarketScraper(ABC):
    """Abstract base for sportsbook odds scrapers.

    Follows the same caching pattern as other scrapers in this package:
    JSON file cache with configurable directory.
    """

    def __init__(self, cache_dir: str = "data/raw/betting_odds"):
        self.cache_dir = cache_dir

    @abstractmethod
    def scrape(self, season: int) -> Dict[str, BettingMarketOdds]:
        """Fetch odds from the sportsbook.

        Args:
            season: Tournament year.

        Returns:
            team_id -> BettingMarketOdds.
        """

    def load_from_json(self, filepath: str) -> Dict[str, BettingMarketOdds]:
        """Load odds from a JSON file.

        Expected format:
        {
            "teams": {
                "team_id": {
                    "team_name": "Duke",
                    "championship_odds": 500,
                    "implied_probability": 0.167,
                    "source": "fanduel"
                }
            },
            "source": "fanduel",
            "timestamp": "2026-03-17T12:00:00Z"
        }
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load betting odds from %s: %s", filepath, e)
            return {}

        source = data.get("source", "unknown")
        timestamp = data.get("timestamp", "")
        season = data.get("season", 0)

        odds_map = {}
        teams_data = data.get("teams", {})
        for team_id, team_data in teams_data.items():
            raw_odds = team_data.get("championship_odds", 0)
            imp_prob = team_data.get("implied_probability")
            if imp_prob is None:
                imp_prob = american_to_probability(raw_odds)

            odds_map[team_id] = BettingMarketOdds(
                team_id=team_id,
                team_name=team_data.get("team_name", team_id),
                season=season,
                source=source,
                championship_odds=raw_odds,
                implied_probability=imp_prob,
                timestamp=timestamp,
                confidence=team_data.get("confidence", 1.0),
            )

        return odds_map

    def _load_cached(self, season: int) -> Optional[Dict]:
        """Load from JSON cache."""
        cache_file = os.path.join(self.cache_dir, f"betting_odds_{season}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_cached(self, season: int, data: Dict) -> None:
        """Save to JSON cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"betting_odds_{season}.json")
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)


class FanDuelScraper(BettingMarketScraper):
    """FanDuel sportsbook odds scraper.

    In production, this would hit the FanDuel API or scrape their
    futures page.  Currently supports JSON file loading only.
    """

    def scrape(self, season: int) -> Dict[str, BettingMarketOdds]:
        cached = self._load_cached(season)
        if cached:
            return self.load_from_json(
                os.path.join(self.cache_dir, f"betting_odds_{season}.json")
            )
        logger.info("FanDuel: no cached data for season %d", season)
        return {}


class DraftKingsScraper(BettingMarketScraper):
    """DraftKings sportsbook odds scraper.

    Supports JSON file loading; live scraping would require
    DraftKings API access.
    """

    def scrape(self, season: int) -> Dict[str, BettingMarketOdds]:
        cached = self._load_cached(season)
        if cached:
            return self.load_from_json(
                os.path.join(self.cache_dir, f"betting_odds_{season}.json")
            )
        logger.info("DraftKings: no cached data for season %d", season)
        return {}


# ---------------------------------------------------------------------------
# Market consensus calculator
# ---------------------------------------------------------------------------

def compute_market_consensus(
    odds_by_source: List[Dict[str, BettingMarketOdds]],
    source_weights: Optional[Dict[str, float]] = None,
    adjust_vig: bool = True,
) -> MarketConsensus:
    """Aggregate implied probabilities across multiple sportsbooks.

    Args:
        odds_by_source: List of {team_id -> BettingMarketOdds} from each source.
        source_weights: Optional source -> weight mapping.
        adjust_vig: Whether to normalize probabilities to sum to 1.0.

    Returns:
        MarketConsensus with aggregated probabilities.
    """
    if not odds_by_source:
        return MarketConsensus(team_probabilities={}, sources=[])

    # Collect all team_ids and sources
    all_teams: Dict[str, List[tuple]] = {}  # team_id -> [(prob, weight)]
    sources = []

    for source_odds in odds_by_source:
        for team_id, odds in source_odds.items():
            if team_id not in all_teams:
                all_teams[team_id] = []

            weight = 1.0
            if source_weights and odds.source in source_weights:
                weight = source_weights[odds.source]

            # Apply confidence (freshness) weighting
            weight *= odds.confidence

            all_teams[team_id].append((odds.implied_probability, weight))

            if odds.source not in sources:
                sources.append(odds.source)

    # Weighted average per team
    team_probs = {}
    for team_id, prob_weights in all_teams.items():
        total_weight = sum(w for _, w in prob_weights)
        if total_weight > 0:
            team_probs[team_id] = sum(
                p * w for p, w in prob_weights
            ) / total_weight
        else:
            team_probs[team_id] = np.mean([p for p, _ in prob_weights])

    if adjust_vig:
        team_probs = remove_vig(team_probs)

    return MarketConsensus(
        team_probabilities=team_probs,
        sources=sources,
        vig_adjusted=adjust_vig,
    )


def blend_with_model(
    model_probs: Dict[str, float],
    market_probs: Dict[str, float],
    market_weight: float = 0.20,
) -> Dict[str, float]:
    """Blend model predictions with market consensus.

    Args:
        model_probs: team_id -> model's P(championship).
        market_probs: team_id -> market's P(championship).
        market_weight: Weight on market signal (0-1).

    Returns:
        Blended probabilities.
    """
    model_weight = 1.0 - market_weight
    blended = {}

    all_teams = set(model_probs.keys()) | set(market_probs.keys())
    for team_id in all_teams:
        m_prob = model_probs.get(team_id, 0.0)
        mkt_prob = market_probs.get(team_id, 0.0)
        blended[team_id] = model_weight * m_prob + market_weight * mkt_prob

    return blended
