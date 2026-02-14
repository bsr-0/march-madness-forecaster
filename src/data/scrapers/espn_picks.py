"""
ESPN Tournament Challenge public picks scraper.

Scrapes public pick percentages for game-theory bracket optimization.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PublicPicks:
    """Public pick percentages for a team."""
    
    team_id: str
    team_name: str
    seed: int
    region: str
    
    # Pick percentages by round
    round_of_64_pct: float = 0.0  # % picked to win first game
    round_of_32_pct: float = 0.0  # % picked to reach Sweet 16
    sweet_16_pct: float = 0.0  # % picked to reach Elite 8
    elite_8_pct: float = 0.0  # % picked to reach Final Four
    final_four_pct: float = 0.0  # % picked to reach Championship
    champion_pct: float = 0.0  # % picked to win tournament
    
    @property
    def as_dict(self) -> Dict[str, float]:
        """Return pick percentages as dictionary."""
        return {
            "R64": self.round_of_64_pct,
            "R32": self.round_of_32_pct,
            "S16": self.sweet_16_pct,
            "E8": self.elite_8_pct,
            "F4": self.final_four_pct,
            "CHAMP": self.champion_pct,
        }


@dataclass
class ConsensusData:
    """Aggregated consensus data from multiple sources."""
    
    teams: Dict[str, PublicPicks] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    
    def get_champion_favorites(self, top_n: int = 10) -> List[PublicPicks]:
        """Get top N teams by championship pick percentage."""
        sorted_teams = sorted(
            self.teams.values(),
            key=lambda t: t.champion_pct,
            reverse=True
        )
        return sorted_teams[:top_n]
    
    def get_contrarian_picks(
        self, 
        model_probs: Dict[str, float],
        min_leverage: float = 2.0
    ) -> List[tuple]:
        """
        Find teams where model probability exceeds public percentage.
        
        Args:
            model_probs: Dict of team_id -> model championship probability
            min_leverage: Minimum leverage ratio to include
            
        Returns:
            List of (team_id, model_prob, public_pct, leverage) tuples
        """
        contrarian = []
        
        for team_id, model_prob in model_probs.items():
            if team_id not in self.teams:
                continue
            
            public_pct = self.teams[team_id].champion_pct
            
            if public_pct > 0:
                leverage = model_prob / public_pct
                
                if leverage >= min_leverage:
                    contrarian.append((
                        team_id,
                        model_prob,
                        public_pct,
                        leverage
                    ))
        
        # Sort by leverage
        contrarian.sort(key=lambda x: x[3], reverse=True)
        return contrarian


class ESPNPicksScraper:
    """
    Scraper for ESPN Tournament Challenge pick percentages.
    
    These percentages represent public consensus and are used for
    game-theory optimization to find contrarian value.
    """
    
    BASE_URL = "https://fantasy.espn.com/tournament-challenge-bracket"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """
        Fetch public pick percentages from ESPN.
        
        Note: Data only available after Selection Sunday.
        
        Args:
            year: Tournament year
            
        Returns:
            ConsensusData with pick percentages
        """
        cached = self._load_from_cache(f"espn_picks_{year}.json")
        if cached:
            return self._dict_to_consensus(cached)
        
        # ESPN pick data only available during tournament
        logger.warning(
            "ESPN pick data not available. Data becomes available "
            "after Selection Sunday when brackets are submitted."
        )
        return ConsensusData(sources=["espn"])
    
    def load_from_json(self, filepath: str) -> ConsensusData:
        """
        Load pick data from JSON file.
        
        Expected format:
        {
            "teams": {
                "duke": {
                    "team_name": "Duke",
                    "seed": 1,
                    "region": "East",
                    "round_of_64_pct": 99.2,
                    "round_of_32_pct": 94.5,
                    ...
                }
            },
            "sources": ["espn", "yahoo"],
            "timestamp": "2026-03-17T12:00:00Z"
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_consensus(data)
    
    def _dict_to_consensus(self, data: dict) -> ConsensusData:
        """Convert dictionary to ConsensusData."""
        teams = {}
        
        for team_id, team_data in data.get("teams", {}).items():
            teams[team_id] = PublicPicks(
                team_id=team_id,
                team_name=team_data.get("team_name", ""),
                seed=team_data.get("seed", 0),
                region=team_data.get("region", ""),
                round_of_64_pct=team_data.get("round_of_64_pct", 0.0),
                round_of_32_pct=team_data.get("round_of_32_pct", 0.0),
                sweet_16_pct=team_data.get("sweet_16_pct", 0.0),
                elite_8_pct=team_data.get("elite_8_pct", 0.0),
                final_four_pct=team_data.get("final_four_pct", 0.0),
                champion_pct=team_data.get("champion_pct", 0.0),
            )
        
        return ConsensusData(
            teams=teams,
            sources=data.get("sources", []),
            timestamp=data.get("timestamp"),
        )
    
    def _load_from_cache(self, filename: str) -> Optional[dict]:
        """Load from cache."""
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        return None


class YahooPicksScraper:
    """Scraper for Yahoo bracket game picks."""
    
    BASE_URL = "https://tournament.fantasysports.yahoo.com"
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """Fetch picks from Yahoo."""
        logger.warning("Yahoo pick data not available.")
        return ConsensusData(sources=["yahoo"])


class CBSPicksScraper:
    """Scraper for CBS bracket game picks."""
    
    BASE_URL = "https://www.cbssports.com/college-basketball/bracketology"
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """Fetch picks from CBS."""
        logger.warning("CBS pick data not available.")
        return ConsensusData(sources=["cbs"])


def aggregate_consensus(
    espn: ConsensusData,
    yahoo: ConsensusData,
    cbs: ConsensusData,
    weights: Dict[str, float] = None
) -> ConsensusData:
    """
    Aggregate pick percentages from multiple sources.
    
    Args:
        espn: ESPN consensus data
        yahoo: Yahoo consensus data
        cbs: CBS consensus data
        weights: Source weights (default: equal)
        
    Returns:
        Aggregated ConsensusData
    """
    if weights is None:
        weights = {"espn": 0.5, "yahoo": 0.3, "cbs": 0.2}
    
    # Collect all team IDs
    all_teams = set()
    for source in [espn, yahoo, cbs]:
        all_teams.update(source.teams.keys())
    
    aggregated = {}
    
    for team_id in all_teams:
        total_weight = 0.0
        weighted_picks = {
            "round_of_64_pct": 0.0,
            "round_of_32_pct": 0.0,
            "sweet_16_pct": 0.0,
            "elite_8_pct": 0.0,
            "final_four_pct": 0.0,
            "champion_pct": 0.0,
        }
        
        # Get team info from first available source
        team_info = None
        
        for source, weight in [(espn, weights["espn"]), 
                                (yahoo, weights["yahoo"]), 
                                (cbs, weights["cbs"])]:
            if team_id in source.teams:
                team = source.teams[team_id]
                
                if team_info is None:
                    team_info = team
                
                total_weight += weight
                
                for key in weighted_picks:
                    weighted_picks[key] += weight * getattr(team, key)
        
        if total_weight > 0 and team_info:
            # Normalize by total weight
            for key in weighted_picks:
                weighted_picks[key] /= total_weight
            
            aggregated[team_id] = PublicPicks(
                team_id=team_id,
                team_name=team_info.team_name,
                seed=team_info.seed,
                region=team_info.region,
                **weighted_picks
            )
    
    return ConsensusData(
        teams=aggregated,
        sources=["espn", "yahoo", "cbs"],
    )


def create_synthetic_picks(seeds_by_region: Dict[str, List[str]]) -> ConsensusData:
    """
    Create synthetic public pick data based on seed.
    
    Uses historical averages for pick percentages by seed.
    
    Args:
        seeds_by_region: Dict mapping region -> list of team_ids by seed (1-16)
        
    Returns:
        Synthetic ConsensusData
    """
    # Historical average pick percentages by seed
    # Based on typical ESPN Tournament Challenge distributions
    historical_picks = {
        1: {"R64": 99.0, "R32": 92.0, "S16": 75.0, "E8": 55.0, "F4": 35.0, "CHAMP": 18.0},
        2: {"R64": 94.0, "R32": 80.0, "S16": 55.0, "E8": 32.0, "F4": 18.0, "CHAMP": 8.0},
        3: {"R64": 85.0, "R32": 60.0, "S16": 35.0, "E8": 18.0, "F4": 8.0, "CHAMP": 3.0},
        4: {"R64": 79.0, "R32": 48.0, "S16": 25.0, "E8": 12.0, "F4": 5.0, "CHAMP": 1.5},
        5: {"R64": 65.0, "R32": 35.0, "S16": 15.0, "E8": 6.0, "F4": 2.0, "CHAMP": 0.5},
        6: {"R64": 63.0, "R32": 30.0, "S16": 12.0, "E8": 5.0, "F4": 1.5, "CHAMP": 0.4},
        7: {"R64": 60.0, "R32": 25.0, "S16": 10.0, "E8": 4.0, "F4": 1.2, "CHAMP": 0.3},
        8: {"R64": 49.0, "R32": 18.0, "S16": 6.0, "E8": 2.0, "F4": 0.6, "CHAMP": 0.1},
        9: {"R64": 51.0, "R32": 15.0, "S16": 5.0, "E8": 1.5, "F4": 0.4, "CHAMP": 0.1},
        10: {"R64": 40.0, "R32": 12.0, "S16": 4.0, "E8": 1.2, "F4": 0.3, "CHAMP": 0.05},
        11: {"R64": 37.0, "R32": 10.0, "S16": 3.0, "E8": 1.0, "F4": 0.2, "CHAMP": 0.03},
        12: {"R64": 35.0, "R32": 8.0, "S16": 2.5, "E8": 0.8, "F4": 0.15, "CHAMP": 0.02},
        13: {"R64": 21.0, "R32": 4.0, "S16": 1.0, "E8": 0.3, "F4": 0.05, "CHAMP": 0.01},
        14: {"R64": 15.0, "R32": 2.5, "S16": 0.5, "E8": 0.1, "F4": 0.02, "CHAMP": 0.005},
        15: {"R64": 6.0, "R32": 1.0, "S16": 0.2, "E8": 0.05, "F4": 0.01, "CHAMP": 0.001},
        16: {"R64": 1.0, "R32": 0.2, "S16": 0.05, "E8": 0.01, "F4": 0.002, "CHAMP": 0.0001},
    }
    
    teams = {}
    
    for region, team_ids in seeds_by_region.items():
        for seed_idx, team_id in enumerate(team_ids, start=1):
            picks = historical_picks.get(seed_idx, historical_picks[16])
            
            teams[team_id] = PublicPicks(
                team_id=team_id,
                team_name=team_id.replace("_", " ").title(),
                seed=seed_idx,
                region=region,
                round_of_64_pct=picks["R64"],
                round_of_32_pct=picks["R32"],
                sweet_16_pct=picks["S16"],
                elite_8_pct=picks["E8"],
                final_four_pct=picks["F4"],
                champion_pct=picks["CHAMP"],
            )
    
    return ConsensusData(teams=teams, sources=["synthetic"])
