"""
KenPom data scraper for team efficiency metrics.

Note: KenPom requires a subscription for full data access.
This scraper provides the structure for integration.
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class KenPomTeam:
    """Team data from KenPom."""
    
    team_id: str
    name: str
    conference: str
    
    # Efficiency metrics (points per 100 possessions)
    adj_efficiency_margin: float  # AdjEM
    adj_offensive_efficiency: float  # AdjO
    adj_defensive_efficiency: float  # AdjD
    adj_tempo: float  # AdjT - possessions per 40 minutes
    
    # Rankings
    overall_rank: int
    offensive_rank: int
    defensive_rank: int
    
    # Luck and strength
    luck: float  # How much better/worse record is than expected
    sos_adj_em: float  # Strength of schedule by AdjEM
    sos_opp_o: float  # Average opponent offensive efficiency
    sos_opp_d: float  # Average opponent defensive efficiency
    
    # Non-conference record analysis
    ncsos_adj_em: float  # Non-conference SOS
    
    # Record
    wins: int = 0
    losses: int = 0


class KenPomScraper:
    """
    Scraper for KenPom efficiency ratings.
    
    Usage:
        scraper = KenPomScraper()
        teams = scraper.fetch_current_rankings()
    """
    
    BASE_URL = "https://kenpom.com"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize scraper.
        
        Args:
            cache_dir: Directory to cache scraped data
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_current_rankings(self, year: int = 2026) -> List[KenPomTeam]:
        """
        Fetch current season rankings.
        
        Note: Requires KenPom subscription for full access.
        This method provides structure for when data is available.
        
        Args:
            year: Season year (e.g., 2026 for 2025-26 season)
            
        Returns:
            List of KenPomTeam objects
        """
        # Check cache first
        cached = self._load_from_cache(f"rankings_{year}.json")
        if cached:
            return [self._dict_to_team(t) for t in cached]
        
        # In production, this would scrape KenPom
        # For now, return empty list (data not yet available)
        logger.warning(
            "KenPom data not available. Provide data via load_from_json() "
            "or implement authentication for live scraping."
        )
        return []
    
    def fetch_team_schedule(self, team_id: str, year: int = 2026) -> List[dict]:
        """
        Fetch game-by-game schedule with efficiency data.
        
        Args:
            team_id: KenPom team identifier
            year: Season year
            
        Returns:
            List of game dictionaries with efficiency metrics
        """
        cached = self._load_from_cache(f"schedule_{team_id}_{year}.json")
        if cached:
            return cached
        
        logger.warning(f"Schedule data for {team_id} not available.")
        return []
    
    def fetch_four_factors(self, year: int = 2026) -> Dict[str, dict]:
        """
        Fetch Four Factors data for all teams.
        
        Returns dict mapping team_id to Four Factors:
        - eFG% (Effective FG%)
        - TO% (Turnover Rate)
        - OR% (Offensive Rebound Rate)
        - FTR (Free Throw Rate)
        
        Args:
            year: Season year
            
        Returns:
            Dictionary of team_id -> four factors dict
        """
        cached = self._load_from_cache(f"four_factors_{year}.json")
        if cached:
            return cached
        
        logger.warning("Four Factors data not available.")
        return {}
    
    def load_from_json(self, filepath: str) -> List[KenPomTeam]:
        """
        Load KenPom data from JSON file.
        
        Expected format:
        {
            "teams": [
                {
                    "team_id": "duke",
                    "name": "Duke",
                    "conference": "ACC",
                    "adj_efficiency_margin": 28.5,
                    "adj_offensive_efficiency": 122.3,
                    "adj_defensive_efficiency": 93.8,
                    ...
                }
            ]
        }
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of KenPomTeam objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        teams = []
        for team_data in data.get("teams", []):
            teams.append(self._dict_to_team(team_data))
        
        return teams
    
    def _dict_to_team(self, data: dict) -> KenPomTeam:
        """Convert dictionary to KenPomTeam."""
        return KenPomTeam(
            team_id=data["team_id"],
            name=data["name"],
            conference=data.get("conference", ""),
            adj_efficiency_margin=data.get("adj_efficiency_margin", 0.0),
            adj_offensive_efficiency=data.get("adj_offensive_efficiency", 100.0),
            adj_defensive_efficiency=data.get("adj_defensive_efficiency", 100.0),
            adj_tempo=data.get("adj_tempo", 68.0),
            overall_rank=data.get("overall_rank", 999),
            offensive_rank=data.get("offensive_rank", 999),
            defensive_rank=data.get("defensive_rank", 999),
            luck=data.get("luck", 0.0),
            sos_adj_em=data.get("sos_adj_em", 0.0),
            sos_opp_o=data.get("sos_opp_o", 100.0),
            sos_opp_d=data.get("sos_opp_d", 100.0),
            ncsos_adj_em=data.get("ncsos_adj_em", 0.0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
        )
    
    def _load_from_cache(self, filename: str) -> Optional[dict]:
        """Load data from cache if available."""
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _save_to_cache(self, filename: str, data: dict) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / filename
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)


def create_synthetic_kenpom_data(num_teams: int = 362) -> List[KenPomTeam]:
    """
    Create synthetic KenPom-like data for testing.
    
    Generates realistic efficiency distributions based on historical KenPom data.
    
    Args:
        num_teams: Number of teams to generate
        
    Returns:
        List of synthetic KenPomTeam objects
    """
    import random
    
    conferences = [
        "ACC", "Big 12", "Big East", "Big Ten", "Pac-12", "SEC", 
        "AAC", "A-10", "MWC", "WCC", "MVC", "CAA", "Horizon",
        "MAAC", "Sun Belt", "C-USA", "MAC", "WAC", "Big West",
        "Ivy", "Patriot", "Southern", "SWAC", "MEAC", "NEC",
        "OVC", "Big Sky", "Big South", "Southland", "Summit"
    ]
    
    teams = []
    
    for i in range(num_teams):
        # Generate realistic efficiency margin (ranges from ~+35 to ~-25)
        # Top teams: +25 to +35
        # Good teams: +10 to +25
        # Average: -5 to +10
        # Below average: -15 to -5
        # Worst: -25 to -15
        
        if i < 10:  # Elite teams
            adj_em = random.gauss(28, 3)
        elif i < 50:  # Very good teams
            adj_em = random.gauss(18, 4)
        elif i < 150:  # Good to average
            adj_em = random.gauss(5, 5)
        elif i < 280:  # Below average
            adj_em = random.gauss(-8, 4)
        else:  # Worst teams
            adj_em = random.gauss(-18, 4)
        
        # Offensive/Defensive split
        adj_o = 100 + adj_em/2 + random.gauss(0, 5)
        adj_d = 100 - adj_em/2 + random.gauss(0, 5)
        
        teams.append(KenPomTeam(
            team_id=f"team_{i:03d}",
            name=f"Team {i:03d}",
            conference=random.choice(conferences),
            adj_efficiency_margin=round(adj_em, 2),
            adj_offensive_efficiency=round(adj_o, 1),
            adj_defensive_efficiency=round(adj_d, 1),
            adj_tempo=round(random.gauss(68, 4), 1),
            overall_rank=i + 1,
            offensive_rank=i + 1,  # Simplified
            defensive_rank=i + 1,  # Simplified
            luck=round(random.gauss(0, 0.03), 3),
            sos_adj_em=round(random.gauss(0, 8), 2),
            sos_opp_o=round(random.gauss(105, 5), 1),
            sos_opp_d=round(random.gauss(100, 5), 1),
            ncsos_adj_em=round(random.gauss(-2, 6), 2),
            wins=random.randint(8, 28),
            losses=random.randint(3, 20),
        ))
    
    return teams
