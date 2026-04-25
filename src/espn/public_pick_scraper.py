"""Public pick data acquisition for ESPN bracket optimization.

Protocol v2, Section 4.1: Public pick rates are the primary input for
leverage computation (model_prob - public_pick_rate).

This module re-exports the scrapers from src/data/scrapers/espn_picks.py
and provides a convenience function for the ESPN optimization pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    # Re-export the full scraper implementations
    from ..data.scrapers.espn_picks import (  # noqa: F401
        ESPNPicksScraper,
        YahooPicksScraper,
        CBSPicksScraper,
        aggregate_consensus,
    )
except ImportError:
    ESPNPicksScraper = None  # type: ignore[assignment,misc]
    YahooPicksScraper = None  # type: ignore[assignment,misc]
    CBSPicksScraper = None  # type: ignore[assignment,misc]
    aggregate_consensus = None  # type: ignore[assignment]


def load_public_picks(
    year: int,
    public_picks_json: Optional[str] = None,
    cache_dir: str = "data/raw",
) -> Dict[str, Dict[str, float]]:
    """Load public pick distribution for ESPN leverage computation.

    Args:
        year: Tournament year.
        public_picks_json: Path to pre-fetched public picks JSON.
        cache_dir: Cache directory for scraped data.

    Returns:
        Dict of team_id -> {round_name: pick_rate}.
        Empty dict if no data available (ESPN optimizer falls back to model probs).
    """
    import json
    import os

    # 1. Try explicit JSON file
    if public_picks_json and os.path.isfile(public_picks_json):
        try:
            with open(public_picks_json) as f:
                data = json.load(f)
            if isinstance(data, dict):
                logger.info("Loaded public picks from %s", public_picks_json)
                return _normalize_picks(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load public picks from %s: %s", public_picks_json, e)

    # 2. Try cached file
    cached_path = os.path.join(cache_dir, f"public_picks_{year}.json")
    if os.path.isfile(cached_path):
        try:
            with open(cached_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                logger.info("Loaded cached public picks from %s", cached_path)
                return _normalize_picks(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load cached public picks: %s", e)

    # 3. Try scraping
    try:
        scraper = ESPNPicksScraper(cache_dir=cache_dir)
        result = scraper.scrape(year)
        if result:
            picks = {}
            for team_pick in result:
                picks[team_pick.team_id] = {
                    rn: getattr(team_pick, rn.lower(), 0.0)
                    for rn in ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
                    if hasattr(team_pick, rn.lower())
                }
            if picks:
                logger.info("Scraped public picks for %d: %d teams", year, len(picks))
                return picks
    except Exception as e:
        logger.warning("Failed to scrape public picks: %s", e)

    logger.warning(
        "No public pick data available for %d. ESPN optimizer will use model probabilities as fallback.",
        year,
    )
    return {}


def _normalize_picks(data: Dict) -> Dict[str, Dict[str, float]]:
    """Normalize pick data to standard format."""
    result: Dict[str, Dict[str, float]] = {}
    round_names = {"R64", "R32", "S16", "E8", "F4", "CHAMP"}

    for team_id, rounds in data.items():
        if isinstance(rounds, dict):
            result[str(team_id)] = {rn: float(rounds.get(rn, 0.0)) for rn in round_names if rn in rounds}

    return result
