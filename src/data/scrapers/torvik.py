"""
BartTorvik data scraper for advanced team metrics.

Scrapes T-Rank efficiency ratings, Four Factors, and game-by-game data
for temporal modeling.

Data acquisition strategy (March 2026):
  Torvik's main HTML pages (trank.php, fourfactors.php) are behind a
  JavaScript browser verification wall that blocks ``requests``.  However,
  two CSV endpoints remain accessible without JS:

  1. ``/{year}_team_results.csv`` — team-level T-Rank ratings, AdjOE/DE,
     Barthag, SOS, WAB, record.  Does NOT include Four Factors.
  2. ``/getadvstats.php?year={year}&csv=1`` — player-level advanced stats
     (eFG%, TO%, ORB%, 3PM/3PA, FTM/FTA, etc.).  Aggregating to team
     level yields offensive Four Factors and shooting splits.

  The ``fetch_four_factors`` and ``fetch_shooting_stats`` methods first
  try the HTML pages (for backward compatibility with saved HTML files),
  then fall back to the CSV player-stats endpoint automatically.
"""

import csv
import io
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
        ratings = scraper.fetch_four_factors(2026)
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

        Attempts three sources in order:
          1. Local cache file.
          2. HTML scrape of ``fourfactors.php`` (fails if JS wall is up).
          3. **CSV fallback**: aggregates player-level stats from
             ``getadvstats.php?year={year}&csv=1`` to compute team-level
             offensive Four Factors (eFG%, TO%, ORB%, FTR) plus shooting
             splits (3P%, FT%).  This endpoint is NOT behind the JS wall.

        Args:
            year: Season year

        Returns:
            Dict of team_id -> four factors dict
        """
        cached = self._load_from_cache(f"torvik_four_factors_{year}.json")
        if cached:
            return cached

        four_factors: Dict[str, Dict] = {}

        # --- Strategy 1: HTML scrape (original approach) ---
        try:
            url = f"{self.BASE_URL}/fourfactors.php?year={year}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            four_factors = self._parse_four_factors_page(response.text)
        except Exception as e:
            logger.debug("HTML Four Factors scrape failed: %s", e)

        # --- Strategy 2: CSV player-stats fallback ---
        if not four_factors:
            logger.info(
                "HTML Four Factors page blocked (JS wall); "
                "falling back to player-stats CSV aggregation."
            )
            four_factors = self._four_factors_from_player_csv(year)

        if four_factors:
            self._save_to_cache(f"torvik_four_factors_{year}.json", four_factors)

        return four_factors
    
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
    
    def fetch_shooting_stats(self, year: int = 2026) -> Dict[str, Dict]:
        """
        Fetch per-team shooting percentages (FT%, 3PT%) for all teams.

        Attempts three sources in order:
          1. Local cache file.
          2. HTML scrape of the extended ``trank.php`` page.
          3. **CSV fallback**: aggregates from ``getadvstats.php`` player
             data (same endpoint used by ``fetch_four_factors`` fallback).

        Returns:
            Dict of team_id -> {'three_pt_pct': float, 'ft_pct': float}
        """
        cached = self._load_from_cache(f"torvik_shooting_{year}.json")
        if cached:
            return cached

        shooting: Dict[str, Dict] = {}

        # --- Strategy 1: HTML scrape (original approach) ---
        try:
            url = f"{self.BASE_URL}/trank.php?year={year}&sort=&hteam=&t=2&q=&dual=show&top=0"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            shooting = self._parse_shooting_page(response.text)
        except Exception as e:
            logger.debug("HTML shooting scrape failed: %s", e)

        # --- Strategy 2: CSV player-stats fallback ---
        if not shooting:
            logger.info(
                "HTML shooting page blocked (JS wall); "
                "falling back to player-stats CSV aggregation."
            )
            shooting = self._shooting_from_player_csv(year)

        if shooting:
            self._save_to_cache(f"torvik_shooting_{year}.json", shooting)
        return shooting

    def _parse_shooting_page(self, html: str) -> Dict[str, Dict]:
        """
        Parse 3PT% and FT% from the extended BartTorvik trank page.

        The extended trank table has columns (0-indexed):
          0: Rank  1: Team  2: Conf  3: Record  4: AdjOE  5: AdjDE  6: Barthag
          7: Tempo  8: eFG%  9: TO%  10: ORB%  11: FTR  12: 3P%  13: 2P%
          14: FT%  ...  (defensive columns follow)

        Column positions may shift; we fall back to scanning for known headers.
        """
        soup = BeautifulSoup(html, 'lxml')
        result: Dict[str, Dict] = {}

        table = soup.find('table')
        if not table:
            return result

        # Detect column positions from header row
        header_row = table.find('tr')
        col_3pt: Optional[int] = None
        col_ft: Optional[int] = None
        if header_row:
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
            for i, h in enumerate(headers):
                if '3p%' in h or '3pt%' in h or 'three' in h:
                    col_3pt = i
                if 'ft%' in h and col_ft is None:
                    col_ft = i

        # Default column positions if header detection fails
        if col_3pt is None:
            col_3pt = 12
        if col_ft is None:
            col_ft = 14

        rows = table.find_all('tr')[1:]
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < max(col_3pt, col_ft) + 1:
                continue
            try:
                team_id = self._extract_team_id(cells[1])
                three_pt = self._safe_float(cells[col_3pt].get_text(strip=True)) / 100
                ft_pct   = self._safe_float(cells[col_ft].get_text(strip=True)) / 100
                if three_pt > 0 or ft_pct > 0:
                    result[team_id] = {'three_pt_pct': three_pt, 'ft_pct': ft_pct}
            except Exception as e:
                logger.debug(f"Error parsing shooting row: {e}")
                continue
        return result

    # ------------------------------------------------------------------
    # CSV player-stats fallback (bypasses JS verification wall)
    # ------------------------------------------------------------------

    def _fetch_player_csv(self, year: int) -> str:
        """Download the raw player-level advanced stats CSV.

        Endpoint: ``/getadvstats.php?year={year}&csv=1``

        This CSV has no header row.  Each row is one player with 68
        columns.  The key columns (0-indexed) are:

          [0]  Player name
          [1]  Team name
          [2]  Conference abbreviation
          [3]  Games played
          [4]  Minutes percentage
          [7]  eFG% (individual)
          [9]  ORB%
          [10] DRB%
          [12] TO%
          [13] FTM
          [14] FTA
          [15] FT%
          [16] 2PM
          [17] 2PA
          [19] 3PM
          [20] 3PA
        """
        url = f"{self.BASE_URL}/getadvstats.php?year={year}&csv=1"
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _normalize_team_name_to_id(name: str) -> str:
        """Convert display team name to a snake_case team_id."""
        tid = name.strip().lower()
        tid = tid.replace("'", "_").replace("&", "_").replace(".", "").replace("-", "_")
        tid = tid.replace("  ", " ").replace(" ", "_")
        return tid

    def _aggregate_player_csv(self, csv_text: str) -> Dict[str, Dict]:
        """Aggregate player-level CSV rows into per-team shooting totals.

        Returns:
            Dict of team_id -> {
                'fgm', 'fga', 'fg2m', 'fg2a', 'fg3m', 'fg3a',
                'ftm', 'fta', 'orb_pct_weighted', 'drb_pct_weighted',
                'to_pct_weighted', 'min_pct_total',
            }
        """
        teams: Dict[str, Dict] = defaultdict(lambda: {
            'fgm': 0.0, 'fga': 0.0,
            'fg2m': 0.0, 'fg2a': 0.0,
            'fg3m': 0.0, 'fg3a': 0.0,
            'ftm': 0.0, 'fta': 0.0,
            'orb_pct_weighted': 0.0,
            'drb_pct_weighted': 0.0,
            'to_pct_weighted': 0.0,
            'min_pct_total': 0.0,
            'conf': '',
            'team_name': '',
        })

        for line in csv_text.strip().split('\n'):
            cols = [c.strip('"') for c in line.split(',')]
            if len(cols) < 22:
                continue
            try:
                team_name = cols[1].strip()
                conf = cols[2].strip()
                min_pct = float(cols[4]) if cols[4] else 0.0
                orb_pct = float(cols[9]) if cols[9] else 0.0
                drb_pct = float(cols[10]) if cols[10] else 0.0
                to_pct = float(cols[12]) if cols[12] else 0.0
                ftm = float(cols[13]) if cols[13] else 0.0
                fta = float(cols[14]) if cols[14] else 0.0
                fg2m = float(cols[16]) if cols[16] else 0.0
                fg2a = float(cols[17]) if cols[17] else 0.0
                fg3m = float(cols[19]) if cols[19] else 0.0
                fg3a = float(cols[20]) if cols[20] else 0.0
            except (ValueError, IndexError):
                continue

            tid = self._normalize_team_name_to_id(team_name)
            t = teams[tid]
            t['team_name'] = team_name
            t['conf'] = conf
            t['fg2m'] += fg2m
            t['fg2a'] += fg2a
            t['fg3m'] += fg3m
            t['fg3a'] += fg3a
            t['fgm'] += fg2m + fg3m
            t['fga'] += fg2a + fg3a
            t['ftm'] += ftm
            t['fta'] += fta
            # Weight percentage stats by minutes share for team-level avg
            t['orb_pct_weighted'] += orb_pct * min_pct
            t['drb_pct_weighted'] += drb_pct * min_pct
            t['to_pct_weighted'] += to_pct * min_pct
            t['min_pct_total'] += min_pct

        return dict(teams)

    def _four_factors_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level Four Factors from the player CSV endpoint.

        Returns dict compatible with ``_parse_four_factors_page`` output::

            {team_id: {
                'effective_fg_pct': float,
                'turnover_rate': float,
                'offensive_reb_rate': float,
                'free_throw_rate': float,
            }}
        """
        try:
            csv_text = self._fetch_player_csv(year)
        except Exception as e:
            logger.warning("Player CSV fetch failed: %s", e)
            return {}

        aggregated = self._aggregate_player_csv(csv_text)
        result: Dict[str, Dict] = {}

        for tid, t in aggregated.items():
            fga = t['fga']
            if fga < 10:  # skip teams with negligible data
                continue

            efg = (t['fgm'] + 0.5 * t['fg3m']) / fga
            ftr = t['fta'] / fga
            min_total = t['min_pct_total'] or 1.0
            orb = (t['orb_pct_weighted'] / min_total) / 100.0
            drb = (t['drb_pct_weighted'] / min_total) / 100.0
            to_rate = (t['to_pct_weighted'] / min_total) / 100.0

            result[tid] = {
                'effective_fg_pct': round(efg, 4),
                'turnover_rate': round(to_rate, 4),
                'offensive_reb_rate': round(orb, 4),
                'free_throw_rate': round(ftr, 4),
                # Defensive Four Factors are not available from offensive
                # player stats alone.  Set to 0 so downstream code knows
                # they are unavailable.
                'opp_effective_fg_pct': 0.0,
                'opp_turnover_rate': 0.0,
                'defensive_reb_rate': round(drb, 4),
                'opp_free_throw_rate': 0.0,
            }

        logger.info(
            "Four Factors from player CSV (%d): computed for %d teams",
            year, len(result),
        )
        return result

    def _shooting_from_player_csv(self, year: int) -> Dict[str, Dict]:
        """Compute team-level shooting splits from the player CSV endpoint.

        Returns dict compatible with ``_parse_shooting_page`` output::

            {team_id: {'three_pt_pct': float, 'ft_pct': float}}
        """
        try:
            csv_text = self._fetch_player_csv(year)
        except Exception as e:
            logger.warning("Player CSV fetch failed: %s", e)
            return {}

        aggregated = self._aggregate_player_csv(csv_text)
        result: Dict[str, Dict] = {}

        for tid, t in aggregated.items():
            fg3a = t['fg3a']
            fta = t['fta']
            if fg3a < 5 and fta < 5:
                continue

            three_pt = (t['fg3m'] / fg3a) if fg3a > 0 else 0.0
            ft_pct = (t['ftm'] / fta) if fta > 0 else 0.0

            result[tid] = {
                'three_pt_pct': round(three_pt, 4),
                'ft_pct': round(ft_pct, 4),
            }

        logger.info(
            "Shooting stats from player CSV (%d): computed for %d teams",
            year, len(result),
        )
        return result

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
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value.replace('%', '').strip())
        except (ValueError, AttributeError):
            return 0.0

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
