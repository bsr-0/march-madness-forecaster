"""
NCAA NET Rankings scraper for women's basketball.

The NET (NCAA Evaluation Tool) is the selection committee's official ranking
system. It combines game results, strength of schedule, game location,
scoring margin, and net offensive/defensive efficiency.

Publicly available at ncaa.com.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class NETRanking:
    """NCAA NET ranking for a team."""

    team_name: str
    team_id: str
    net_ranking: int
    record: str = ""
    conference: str = ""
    net_rating: float = 0.0  # Normalized 0-100 score

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "team_id": self.team_id,
            "net_ranking": self.net_ranking,
            "record": self.record,
            "conference": self.conference,
            "net_rating": self.net_rating,
        }


class WomensNETScraper:
    """Load or generate women's NCAA NET rankings.

    Supports cached JSON files. When no cache is available,
    generates estimates from seed data.
    """

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)

    def load_cached(self, year: int) -> Dict[str, NETRanking]:
        """Load cached NET rankings."""
        cache_path = self.cache_dir / f"womens_net_rankings_{year}.json"
        if not cache_path.exists():
            logger.info("No cached women's NET rankings for %d", year)
            return {}

        with open(cache_path, "r") as f:
            data = json.load(f)

        rankings = {}
        for entry in data:
            r = NETRanking(**{k: v for k, v in entry.items() if k in NETRanking.__dataclass_fields__})
            rankings[r.team_id] = r

        logger.info("Loaded %d women's NET rankings for %d", len(rankings), year)
        return rankings

    def estimate_from_seeds(self, seed_map: Dict[str, int], total_teams: int = 360) -> Dict[str, NETRanking]:
        """Estimate NET rankings from tournament seeds.

        Historical correlation: seed 1 ≈ NET 1-4, seed 2 ≈ NET 5-12, etc.
        """
        # Map seeds to approximate NET ranking ranges
        seed_to_net_center = {
            1: 3,
            2: 8,
            3: 14,
            4: 20,
            5: 28,
            6: 35,
            7: 42,
            8: 52,
            9: 60,
            10: 68,
            11: 55,
            12: 48,
            13: 90,
            14: 120,
            15: 180,
            16: 280,
        }

        rankings = {}
        for team_id, seed in seed_map.items():
            net_rank = seed_to_net_center.get(seed, 150)
            # Normalized rating: higher is better (invert ranking)
            net_rating = max(0.0, 100.0 * (1.0 - net_rank / total_teams))

            rankings[team_id] = NETRanking(
                team_name=team_id,
                team_id=team_id,
                net_ranking=net_rank,
                net_rating=net_rating,
            )

        return rankings

    def save_cache(self, year: int, rankings: Dict[str, NETRanking]) -> None:
        """Save rankings to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"womens_net_rankings_{year}.json"
        data = [r.to_dict() for r in rankings.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
