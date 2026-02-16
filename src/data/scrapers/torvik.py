"""
BartTorvik data scraper for advanced team metrics.

Scrapes T-Rank efficiency ratings, Four Factors, and game-by-game data
for temporal modeling.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class TorVikTeam:
    """Team data from BartTorvik T-Rank."""
    
    team_id: str
    name: str
    conference: str
    
    # T-Rank ratings
    t_rank: int
    barthag: float  # Expected win percentage vs average team on neutral
    
    # Efficiency metrics (per 100 possessions)
    adj_offensive_efficiency: float
    adj_defensive_efficiency: float
    adj_tempo: float  # Possessions per 40 minutes
    
    # Four Factors (Offense)
    effective_fg_pct: float  # eFG%
    turnover_rate: float  # TO%
    offensive_reb_rate: float  # ORB%
    free_throw_rate: float  # FTR (FT/FGA)
    
    # Four Factors (Defense)
    opp_effective_fg_pct: float
    opp_turnover_rate: float  # Forced TO%
    defensive_reb_rate: float  # DRB%
    opp_free_throw_rate: float
    
    # Additional metrics
    two_pt_pct: float = 0.0
    three_pt_pct: float = 0.0
    three_pt_rate: float = 0.0  # % of shots from 3
    ft_pct: float = 0.0
    block_pct: float = 0.0
    steal_pct: float = 0.0
    
    # Opponent-adjusted versions
    opp_two_pt_pct: float = 0.0
    opp_three_pt_pct: float = 0.0
    opp_three_pt_rate: float = 0.0
    
    # WAB metrics
    wab: float = 0.0  # Wins Above Bubble
    
    # Record
    wins: int = 0
    losses: int = 0
    conf_wins: int = 0
    conf_losses: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary — includes ALL scraped fields."""
        return {
            'team_id': self.team_id,
            'name': self.name,
            'conference': self.conference,
            't_rank': self.t_rank,
            'barthag': self.barthag,
            'adj_offensive_efficiency': self.adj_offensive_efficiency,
            'adj_defensive_efficiency': self.adj_defensive_efficiency,
            'adj_tempo': self.adj_tempo,
            'effective_fg_pct': self.effective_fg_pct,
            'turnover_rate': self.turnover_rate,
            'offensive_reb_rate': self.offensive_reb_rate,
            'free_throw_rate': self.free_throw_rate,
            'opp_effective_fg_pct': self.opp_effective_fg_pct,
            'opp_turnover_rate': self.opp_turnover_rate,
            'defensive_reb_rate': self.defensive_reb_rate,
            'opp_free_throw_rate': self.opp_free_throw_rate,
            # Shooting splits & extended metrics
            'two_pt_pct': self.two_pt_pct,
            'three_pt_pct': self.three_pt_pct,
            'three_pt_rate': self.three_pt_rate,
            'ft_pct': self.ft_pct,
            'block_pct': self.block_pct,
            'steal_pct': self.steal_pct,
            'opp_two_pt_pct': self.opp_two_pt_pct,
            'opp_three_pt_pct': self.opp_three_pt_pct,
            'opp_three_pt_rate': self.opp_three_pt_rate,
            # WAB + Record
            'wab': self.wab,
            'wins': self.wins,
            'losses': self.losses,
            'conf_wins': self.conf_wins,
            'conf_losses': self.conf_losses,
        }


@dataclass
class TorVikGame:
    """Single game data from BartTorvik."""
    
    game_id: str
    date: str
    team_id: str
    opponent_id: str
    
    # Location
    is_home: bool
    is_neutral: bool
    
    # Result
    team_score: int
    opponent_score: int
    
    @property
    def margin(self) -> int:
        return self.team_score - self.opponent_score
    
    @property
    def is_win(self) -> bool:
        return self.margin > 0
    
    # Efficiency in this game
    offensive_efficiency: float = 0.0
    defensive_efficiency: float = 0.0
    tempo: float = 0.0
    
    # Four Factors for this game
    effective_fg_pct: float = 0.0
    turnover_rate: float = 0.0
    offensive_reb_rate: float = 0.0
    free_throw_rate: float = 0.0
    
    # Opponent quality
    opponent_rank: int = 0
    opponent_adj_em: float = 0.0
    
    # Game quality metrics
    game_quality: float = 0.0  # How well team played vs expectation


class BartTorvikScraper:
    """
    Scraper for BartTorvik T-Rank data.
    
    BartTorvik provides:
    - T-Rank efficiency ratings
    - Four Factors analysis
    - Game-by-game efficiency data
    - WAB (Wins Above Bubble) calculations
    
    Usage:
        scraper = BartTorvikScraper()
        teams = scraper.fetch_current_rankings()
        games = scraper.fetch_team_games("duke", 2026)
    """
    
    BASE_URL = "https://barttorvik.com"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize scraper.
        
        Args:
            cache_dir: Directory to cache scraped data
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_current_rankings(self, year: int = 2026) -> List[TorVikTeam]:
        """
        Fetch current T-Rank ratings for all teams.
        
        Args:
            year: Season year (e.g., 2026 for 2025-26 season)
            
        Returns:
            List of TorVikTeam objects
        """
        # Check cache
        cached = self._load_from_cache(f"torvik_rankings_{year}.json")
        if cached:
            return [self._dict_to_team(t) for t in cached.get('teams', [])]
        
        try:
            # Attempt to scrape the rankings page
            url = f"{self.BASE_URL}/trank.php?year={year}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            teams = self._parse_rankings_page(response.text)
            
            if teams:
                self._save_to_cache(f"torvik_rankings_{year}.json", {
                    'teams': [t.to_dict() for t in teams],
                    'timestamp': datetime.now().isoformat()
                })
            
            return teams
            
        except Exception as e:
            logger.warning(f"Could not fetch Torvik rankings: {e}")
            logger.info("Use load_from_json() to load data from file.")
            return []
    
    def _parse_rankings_page(self, html: str) -> List[TorVikTeam]:
        """Parse rankings from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        teams = []
        
        # Find the main data table
        table = soup.find('table', {'id': 'data-table'})
        if not table:
            # Try alternate table structure
            table = soup.find('table')
        
        if not table:
            logger.warning("Could not find rankings table")
            return []
        
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 10:
                continue
            
            try:
                team = TorVikTeam(
                    team_id=self._extract_team_id(cells[1]),
                    name=cells[1].get_text(strip=True),
                    conference=cells[2].get_text(strip=True) if len(cells) > 2 else "",
                    t_rank=int(cells[0].get_text(strip=True)),
                    barthag=self._safe_float(cells[6].get_text(strip=True)) if len(cells) > 6 else 0.0,
                    adj_offensive_efficiency=self._safe_float(cells[4].get_text(strip=True)),
                    adj_defensive_efficiency=self._safe_float(cells[5].get_text(strip=True)),
                    adj_tempo=self._safe_float(cells[7].get_text(strip=True)) if len(cells) > 7 else 68.0,
                    # Leave Four Factors as NaN when rankings page does not provide them.
                    effective_fg_pct=math.nan,
                    turnover_rate=math.nan,
                    offensive_reb_rate=math.nan,
                    free_throw_rate=math.nan,
                    opp_effective_fg_pct=math.nan,
                    opp_turnover_rate=math.nan,
                    defensive_reb_rate=math.nan,
                    opp_free_throw_rate=math.nan,
                )
                teams.append(team)
            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
                continue
        
        return teams
    
    def fetch_four_factors(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch Four Factors data for all teams.
        
        Args:
            year: Season year
            
        Returns:
            Dict of team_id -> four factors dict
        """
        cached = self._load_from_cache(f"torvik_four_factors_{year}.json")
        if cached:
            return cached
        
        try:
            # Scrape Four Factors page
            url = f"{self.BASE_URL}/fourfactors.php?year={year}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            four_factors = self._parse_four_factors_page(response.text)
            
            if four_factors:
                self._save_to_cache(f"torvik_four_factors_{year}.json", four_factors)
            
            return four_factors
            
        except Exception as e:
            logger.warning(f"Could not fetch Four Factors: {e}")
            return {}
    
    def _parse_four_factors_page(self, html: str) -> Dict[str, Dict]:
        """Parse Four Factors from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        result = {}
        
        table = soup.find('table')
        if not table:
            return result
        
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 9:
                continue
            
            try:
                team_id = self._extract_team_id(cells[0])
                result[team_id] = {
                    'effective_fg_pct': self._safe_float(cells[2].get_text(strip=True)) / 100,
                    'turnover_rate': self._safe_float(cells[3].get_text(strip=True)) / 100,
                    'offensive_reb_rate': self._safe_float(cells[4].get_text(strip=True)) / 100,
                    'free_throw_rate': self._safe_float(cells[5].get_text(strip=True)) / 100,
                    'opp_effective_fg_pct': self._safe_float(cells[6].get_text(strip=True)) / 100,
                    'opp_turnover_rate': self._safe_float(cells[7].get_text(strip=True)) / 100,
                    'defensive_reb_rate': self._safe_float(cells[8].get_text(strip=True)) / 100,
                    'opp_free_throw_rate': self._safe_float(cells[9].get_text(strip=True)) / 100 if len(cells) > 9 else 0.30,
                }
            except Exception as e:
                logger.debug(f"Error parsing four factors row: {e}")
                continue
        
        return result
    
    def fetch_team_games(self, team_id: str, year: int = 2026) -> List[TorVikGame]:
        """
        Fetch game-by-game data for a team.
        
        Args:
            team_id: Team identifier
            year: Season year
            
        Returns:
            List of TorVikGame objects
        """
        cached = self._load_from_cache(f"torvik_games_{team_id}_{year}.json")
        if cached:
            return [self._dict_to_game(g) for g in cached.get('games', [])]
        
        try:
            url = f"{self.BASE_URL}/team.php?team={team_id}&year={year}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            games = self._parse_team_games(response.text, team_id, year)
            
            if games:
                self._save_to_cache(f"torvik_games_{team_id}_{year}.json", {
                    'games': [self._game_to_dict(g) for g in games],
                    'timestamp': datetime.now().isoformat()
                })
            
            return games
            
        except Exception as e:
            logger.warning(f"Could not fetch games for {team_id}: {e}")
            return []
    
    def _parse_team_games(self, html: str, team_id: str, year: int) -> List[TorVikGame]:
        """Parse team game log from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        games = []
        
        # Find game log table
        tables = soup.find_all('table')
        game_table = None
        
        for table in tables:
            header = table.find('tr')
            if header and 'Date' in header.get_text():
                game_table = table
                break
        
        if not game_table:
            return games
        
        rows = game_table.find_all('tr')[1:]
        
        for idx, row in enumerate(rows):
            cells = row.find_all('td')
            if len(cells) < 5:
                continue
            
            try:
                date_str = cells[0].get_text(strip=True)
                opponent_cell = cells[1]
                opponent_text = opponent_cell.get_text(strip=True)
                
                # Determine home/away/neutral
                is_home = '@' not in opponent_text
                is_neutral = 'N' in opponent_text or '*' in opponent_text
                
                # Extract score
                score_text = cells[2].get_text(strip=True) if len(cells) > 2 else "0-0"
                team_score, opp_score = self._parse_score(score_text)
                
                game = TorVikGame(
                    game_id=f"{team_id}_{year}_{idx}",
                    date=date_str,
                    team_id=team_id,
                    opponent_id=self._extract_team_id(opponent_cell),
                    is_home=is_home,
                    is_neutral=is_neutral,
                    team_score=team_score,
                    opponent_score=opp_score,
                    offensive_efficiency=self._safe_float(cells[3].get_text(strip=True)) if len(cells) > 3 else 0,
                    defensive_efficiency=self._safe_float(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                )
                games.append(game)
            except Exception as e:
                logger.debug(f"Error parsing game row: {e}")
                continue
        
        return games
    
    def load_from_json(self, filepath: str) -> List[TorVikTeam]:
        """
        Load Torvik data from JSON file.
        
        Expected format:
        {
            "teams": [
                {
                    "team_id": "duke",
                    "name": "Duke",
                    "conference": "ACC",
                    "t_rank": 1,
                    "adj_offensive_efficiency": 122.3,
                    "adj_defensive_efficiency": 93.8,
                    ...
                }
            ]
        }
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of TorVikTeam objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [self._dict_to_team(t) for t in data.get('teams', [])]
    
    def _dict_to_team(self, data: dict) -> TorVikTeam:
        """Convert dictionary to TorVikTeam."""
        return TorVikTeam(
            team_id=data.get('team_id', ''),
            name=data.get('name', ''),
            conference=data.get('conference', ''),
            t_rank=data.get('t_rank', 999),
            barthag=data.get('barthag', 0.5),
            adj_offensive_efficiency=data.get('adj_offensive_efficiency', 100.0),
            adj_defensive_efficiency=data.get('adj_defensive_efficiency', 100.0),
            adj_tempo=data.get('adj_tempo', 68.0),
            effective_fg_pct=data.get('effective_fg_pct', 0.5),
            turnover_rate=data.get('turnover_rate', 0.18),
            offensive_reb_rate=data.get('offensive_reb_rate', 0.30),
            free_throw_rate=data.get('free_throw_rate', 0.30),
            opp_effective_fg_pct=data.get('opp_effective_fg_pct', 0.5),
            opp_turnover_rate=data.get('opp_turnover_rate', 0.18),
            defensive_reb_rate=data.get('defensive_reb_rate', 0.70),
            opp_free_throw_rate=data.get('opp_free_throw_rate', 0.30),
            two_pt_pct=data.get('two_pt_pct', 0.0),
            three_pt_pct=data.get('three_pt_pct', 0.0),
            three_pt_rate=data.get('three_pt_rate', 0.0),
            ft_pct=data.get('ft_pct', 0.0),
            block_pct=data.get('block_pct', 0.0),
            steal_pct=data.get('steal_pct', 0.0),
            opp_two_pt_pct=data.get('opp_two_pt_pct', 0.0),
            opp_three_pt_pct=data.get('opp_three_pt_pct', 0.0),
            opp_three_pt_rate=data.get('opp_three_pt_rate', 0.0),
            wab=data.get('wab', 0.0),
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            conf_wins=data.get('conf_wins', 0),
            conf_losses=data.get('conf_losses', 0),
        )
    
    def _dict_to_game(self, data: dict) -> TorVikGame:
        """Convert dictionary to TorVikGame."""
        return TorVikGame(
            game_id=data.get('game_id', ''),
            date=data.get('date', ''),
            team_id=data.get('team_id', ''),
            opponent_id=data.get('opponent_id', ''),
            is_home=data.get('is_home', True),
            is_neutral=data.get('is_neutral', False),
            team_score=data.get('team_score', 0),
            opponent_score=data.get('opponent_score', 0),
            offensive_efficiency=data.get('offensive_efficiency', 0.0),
            defensive_efficiency=data.get('defensive_efficiency', 0.0),
        )
    
    def _game_to_dict(self, game: TorVikGame) -> dict:
        """Convert TorVikGame to dictionary."""
        return {
            'game_id': game.game_id,
            'date': game.date,
            'team_id': game.team_id,
            'opponent_id': game.opponent_id,
            'is_home': game.is_home,
            'is_neutral': game.is_neutral,
            'team_score': game.team_score,
            'opponent_score': game.opponent_score,
            'offensive_efficiency': game.offensive_efficiency,
            'defensive_efficiency': game.defensive_efficiency,
        }
    
    def _extract_team_id(self, cell) -> str:
        """Extract team ID from table cell."""
        link = cell.find('a')
        if link and 'href' in link.attrs:
            href = link['href']
            if 'team=' in href:
                return href.split('team=')[1].split('&')[0]
        return cell.get_text(strip=True).lower().replace(' ', '_')
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value.replace('%', '').strip())
        except (ValueError, AttributeError):
            return 0.0
    
    def _parse_score(self, score_text: str) -> Tuple[int, int]:
        """Parse score from text like '85-72' or 'W 85-72'."""
        # Remove W/L prefix
        score_text = score_text.replace('W', '').replace('L', '').strip()
        
        if '-' in score_text:
            parts = score_text.split('-')
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                pass
        return 0, 0
    
    def _load_from_cache(self, filename: str) -> Optional[dict]:
        """Load data from cache if available."""
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None
    
    def _save_to_cache(self, filename: str, data: dict) -> None:
        """Save data to cache."""
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / filename
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
