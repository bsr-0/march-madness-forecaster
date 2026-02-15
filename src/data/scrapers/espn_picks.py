"""
ESPN Tournament Challenge public picks scraper.

Scrapes public pick percentages for game-theory bracket optimization.
"""

import json
import logging
import os
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

        source_url = os.getenv("ESPN_PUBLIC_PICKS_URL")
        if source_url:
            try:
                response = self.session.get(source_url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    if self.cache_dir:
                        self._save_to_cache(f"espn_picks_{year}.json", payload)
                    return self._dict_to_consensus(payload)
            except Exception as exc:
                logger.warning("ESPN picks fetch failed: %s", exc)

        logger.warning(
            "ESPN pick data unavailable. Set ESPN_PUBLIC_PICKS_URL or provide --public-picks JSON."
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

    def _save_to_cache(self, filename: str, data: dict) -> None:
        if not self.cache_dir:
            return
        cache_path = self.cache_dir / filename
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)


class YahooPicksScraper:
    """Scraper for Yahoo bracket game picks."""
    
    BASE_URL = "https://tournament.fantasysports.yahoo.com"

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            }
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """Fetch picks from Yahoo."""
        cache_name = f"yahoo_picks_{year}.json"
        if self.cache_dir:
            cached = self._load_cache(cache_name)
            if cached:
                return ESPNPicksScraper()._dict_to_consensus(cached)

        source_url = os.getenv("YAHOO_PUBLIC_PICKS_URL")
        if source_url:
            try:
                response = self.session.get(source_url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if self.cache_dir and isinstance(payload, dict):
                    self._save_cache(cache_name, payload)
                if isinstance(payload, dict):
                    return ESPNPicksScraper()._dict_to_consensus(payload)
            except Exception as exc:
                logger.warning("Yahoo picks fetch failed: %s", exc)

        logger.warning("Yahoo pick data unavailable. Set YAHOO_PUBLIC_PICKS_URL.")
        return ConsensusData(sources=["yahoo"])

    def _load_cache(self, filename: str) -> Optional[dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    def _save_cache(self, filename: str, data: dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)


class CBSPicksScraper:
    """Scraper for CBS bracket game picks."""
    
    BASE_URL = "https://www.cbssports.com/college-basketball/bracketology"

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            }
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_picks(self, year: int = 2026) -> ConsensusData:
        """Fetch picks from CBS."""
        cache_name = f"cbs_picks_{year}.json"
        if self.cache_dir:
            cached = self._load_cache(cache_name)
            if cached:
                return ESPNPicksScraper()._dict_to_consensus(cached)

        source_url = os.getenv("CBS_PUBLIC_PICKS_URL")
        if source_url:
            try:
                response = self.session.get(source_url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if self.cache_dir and isinstance(payload, dict):
                    self._save_cache(cache_name, payload)
                if isinstance(payload, dict):
                    return ESPNPicksScraper()._dict_to_consensus(payload)
            except Exception as exc:
                logger.warning("CBS picks fetch failed: %s", exc)

        logger.warning("CBS pick data unavailable. Set CBS_PUBLIC_PICKS_URL.")
        return ConsensusData(sources=["cbs"])

    def _load_cache(self, filename: str) -> Optional[dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        with open(p, "r") as f:
            return json.load(f)

    def _save_cache(self, filename: str, data: dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)


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
