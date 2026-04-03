"""
Tournament context scrapers for enrichment data.

Provides three pipelines:
1. Preseason AP rankings (Sports-Reference polls page)
2. Head coach tournament experience (Barttorvik coach tournament table)
3. Conference tournament champions (Sports-Reference season summary)

Each pipeline scrapes a public source, normalizes the data into a standard
dict format, and supports JSON caching for reproducibility.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _make_soup(html_or_text):
    try:
        return BeautifulSoup(html_or_text, "lxml")
    except Exception:
        return BeautifulSoup(html_or_text, "html.parser")


class TournamentContextScraper:
    """
    Scraper for tournament context enrichment data.

    Provides preseason AP rankings, coach tournament experience,
    and conference tournament champions from public sources.
    """

    BASE_URL_SR = "https://www.sports-reference.com/cbb"
    BASE_URL_TORVIK = "https://barttorvik.com"

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0"
                ),
            }
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Preseason AP Rankings
    # ------------------------------------------------------------------

    def fetch_preseason_ap_rankings(self, year: int) -> Dict[str, int]:
        """
        Fetch preseason AP Top 25 rankings for a season.

        Args:
            year: Season end year (e.g., 2026 for 2025-26 season).

        Returns:
            Dict mapping normalized team name -> preseason AP rank (1-25).
            Teams not in preseason Top 25 are omitted.
        """
        cache_name = f"preseason_ap_{year}.json"
        cached = self._load_cache(cache_name)
        if cached and "rankings" in cached:
            return cached["rankings"]

        rankings = self._scrape_preseason_ap(year)
        if rankings:
            try:
                from .schemas import validate_preseason_ap

                rankings = validate_preseason_ap(rankings)
            except Exception as e:
                logger.warning("Preseason AP schema validation failed: %s", e)
            self._save_cache(cache_name, {"rankings": rankings, "year": year})
        return rankings

    def _scrape_preseason_ap(self, year: int) -> Dict[str, int]:
        """Fetch preseason AP rankings, trying Massey Ordinals first.

        Primary: Massey Ordinals (pre-cached JSON from Kaggle) which
        include AP rankings as one of 100+ ranking systems.
        Fallback: Sports-Reference polls page HTML scraping.
        """
        # --- Primary: Massey Ordinals (pre-cached, no scraping) ---
        rankings = self._ap_from_massey_ordinals(year)
        if rankings:
            logger.info(
                "Loaded %d preseason AP rankings from Massey Ordinals for %d",
                len(rankings),
                year,
            )
            return rankings

        # --- Fallback: Sports Reference HTML (fragile) ---
        logger.info(
            "Massey Ordinals AP data unavailable; falling back to SR HTML for %d",
            year,
        )
        url = f"{self.BASE_URL_SR}/seasons/men/{year}-polls.html"
        rankings = {}

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Could not fetch AP polls page for %d: %s", year, e)
            return rankings

        soup = _make_soup(resp.text)

        # Strategy 1: Look for the weekly poll grid table with a "Pre" column
        table = soup.find("table", {"id": "ap-polls"})
        if not table:
            for t in soup.find_all("table"):
                headers = [th.get_text(strip=True) for th in t.find_all("th")]
                if "Pre" in headers or "Preseason" in headers:
                    table = t
                    break

        if table:
            rankings = self._parse_poll_grid_table(table)
            if rankings:
                return rankings

        # Strategy 2: Parse the standalone "Preseason" AP Top 25 table
        for t in soup.find_all("table"):
            caption = t.find("caption")
            if caption and "preseason" in caption.get_text(strip=True).lower():
                rankings = self._parse_simple_top25_table(t)
                if rankings:
                    return rankings

        logger.warning("Could not parse preseason AP rankings for %d", year)
        return rankings

    def _ap_from_massey_ordinals(self, year: int) -> Dict[str, int]:
        """Try to extract preseason AP rankings from cached Massey Ordinals.

        Massey Ordinals (Kaggle dataset) include AP rankings as one of
        100+ systems.  This avoids fragile HTML scraping entirely.
        """
        try:
            from .external_ratings import ExternalRatingsLoader

            cache_dir = str(self.cache_dir) if self.cache_dir else "data/raw"
            loader = ExternalRatingsLoader(cache_dir=cache_dir)

            # Check if cached ratings exist for this year
            cache_path = Path(cache_dir) / f"external_ratings_{year}.json"
            if not cache_path.exists():
                return {}

            with open(cache_path, "r") as f:
                data = json.load(f)

            # Look for AP system in the cached ratings
            systems = data.get("systems", {})
            ap_entries = systems.get("AP") or systems.get("ap") or systems.get("AP Top 25") or {}
            if not ap_entries:
                # Try fuzzy match on system name
                for sys_name, entries in systems.items():
                    if "ap" in sys_name.lower() and "top" in sys_name.lower():
                        ap_entries = entries
                        break

            if not ap_entries:
                return {}

            rankings = {}
            if isinstance(ap_entries, list):
                for entry in ap_entries:
                    team = entry.get("team_id") or entry.get("team_name") or ""
                    rank = entry.get("ranking") or entry.get("rank") or 0
                    if team and isinstance(rank, (int, float)) and 1 <= rank <= 25:
                        rankings[self._normalize_name(team)] = int(rank)
            elif isinstance(ap_entries, dict):
                for team, rank in ap_entries.items():
                    if isinstance(rank, (int, float)) and 1 <= rank <= 25:
                        rankings[self._normalize_name(team)] = int(rank)

            return rankings
        except Exception as exc:
            logger.debug("Massey Ordinals AP extraction failed: %s", exc)
            return {}

    def _parse_poll_grid_table(self, table) -> Dict[str, int]:
        """Parse the weekly poll grid and extract the 'Pre' column."""
        rankings: Dict[str, int] = {}
        headers = []
        header_row = table.find("thead")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

        pre_col = None
        for idx, h in enumerate(headers):
            if h in ("Pre", "Preseason"):
                pre_col = idx
                break

        if pre_col is None:
            return rankings

        rows = table.find("tbody")
        if not rows:
            return rankings

        for row in rows.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) <= pre_col:
                continue

            # Team name is typically in the first or second cell
            team_cell = cells[1] if len(cells) > 1 else cells[0]
            team_name = self._extract_team_name(team_cell)
            if not team_name:
                continue

            rank_text = cells[pre_col].get_text(strip=True)
            rank = self._parse_rank(rank_text)
            if rank and 1 <= rank <= 25:
                rankings[self._normalize_name(team_name)] = rank

        return rankings

    def _parse_simple_top25_table(self, table) -> Dict[str, int]:
        """Parse a simple ranked table (Rank, School, ...)."""
        rankings: Dict[str, int] = {}
        rows = table.find_all("tr")[1:]  # Skip header

        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            rank = self._parse_rank(cells[0].get_text(strip=True))
            team_name = self._extract_team_name(cells[1])
            if rank and team_name and 1 <= rank <= 25:
                rankings[self._normalize_name(team_name)] = rank

        return rankings

    # ------------------------------------------------------------------
    # 2. Coach Tournament Experience
    # ------------------------------------------------------------------

    def fetch_coach_tournament_experience(self, year: int) -> Dict[str, Dict[str, object]]:
        """
        Fetch head coach NCAA tournament appearance counts.

        Args:
            year: Season end year.

        Returns:
            Dict mapping normalized coach name -> {
                "appearances": int,
                "wins": int,
                "losses": int,
                "teams": List[str],  # teams coached in tournament
            }
        """
        cache_name = f"coach_tournament_{year}.json"
        cached = self._load_cache(cache_name)
        if cached and "coaches" in cached:
            return cached["coaches"]

        coaches = self._scrape_coach_tournament_data(year)
        if coaches:
            try:
                from .schemas import validate_coach_tournament_data

                coaches = validate_coach_tournament_data(coaches)
            except Exception as e:
                logger.warning("Coach tournament data schema validation failed: %s", e)
            self._save_cache(cache_name, {"coaches": coaches, "year": year})
        return coaches

    def _scrape_coach_tournament_data(self, year: int) -> Dict[str, Dict[str, object]]:
        """Fetch coach tournament data, trying CSV first, HTML as fallback.

        Primary: Barttorvik team_results CSV (structured, not behind JS wall).
        Fallback: Barttorvik coach tournament HTML page (fragile soup.find("table")).
        """
        # --- Primary: CSV-based extraction (structured, reliable) ---
        coaches = self._coach_data_from_csv(year)
        if coaches:
            logger.info("Loaded %d coaches from Barttorvik CSV for %d", len(coaches), year)
            return coaches

        # --- Fallback: HTML scraping (fragile) ---
        logger.info("CSV coach data unavailable; falling back to HTML scrape for %d", year)
        return self._coach_data_from_html(year)

    def _coach_data_from_csv(self, year: int) -> Dict[str, Dict[str, object]]:
        """Extract coach data from Barttorvik team_results CSV.

        The CSV is not behind Barttorvik's JS verification wall and has
        structured columns including coach name, wins, losses.
        """
        url = f"{self.BASE_URL_TORVIK}/{year}_team_results.csv"
        coaches: Dict[str, Dict[str, object]] = {}

        try:
            resp = self.session.get(url, timeout=45)
            resp.raise_for_status()
        except Exception as e:
            logger.debug("Could not fetch Barttorvik team results CSV for %d: %s", year, e)
            return coaches

        text = resp.text.strip()
        if not text:
            return coaches

        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        if not reader.fieldnames:
            return coaches

        # Find relevant columns (case-insensitive)
        lower_fields = {(f or "").strip().lower(): f for f in reader.fieldnames}
        coach_key = None
        wins_key = None
        losses_key = None
        for alias in ("coach", "head coach", "head_coach"):
            if alias in lower_fields:
                coach_key = lower_fields[alias]
                break
        for alias in ("w", "wins", "rec_w"):
            if alias in lower_fields:
                wins_key = lower_fields[alias]
                break
        for alias in ("l", "losses", "rec_l"):
            if alias in lower_fields:
                losses_key = lower_fields[alias]
                break

        if not coach_key:
            return coaches

        for row in reader:
            coach_name = (row.get(coach_key) or "").strip()
            if not coach_name:
                continue
            wins = self._safe_int(row.get(wins_key, "0")) if wins_key else 0
            losses = self._safe_int(row.get(losses_key, "0")) if losses_key else 0

            normalized = self._normalize_name(coach_name)
            if normalized in coaches:
                # Accumulate across multiple teams coached
                coaches[normalized]["appearances"] = int(coaches[normalized]["appearances"]) + 1
                coaches[normalized]["wins"] = int(coaches[normalized]["wins"]) + wins
                coaches[normalized]["losses"] = int(coaches[normalized]["losses"]) + losses
            else:
                coaches[normalized] = {
                    "name": coach_name,
                    "appearances": 1,
                    "wins": wins,
                    "losses": losses,
                    "teams": [],
                }

        return coaches

    def _coach_data_from_html(self, year: int) -> Dict[str, Dict[str, object]]:
        """Scrape coach tournament data from Barttorvik HTML (legacy fallback)."""
        url = f"{self.BASE_URL_TORVIK}/cgi-bin/ncaat.cgi?type=coach"
        coaches: Dict[str, Dict[str, object]] = {}

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Could not fetch coach tournament data: %s", e)
            return coaches

        soup = _make_soup(resp.text)

        table = soup.find("table")
        if not table:
            logger.warning("No table found on Barttorvik coach tournament page")
            return coaches

        rows = table.find_all("tr")
        if len(rows) < 2:
            return coaches

        header_cells = rows[0].find_all(["th", "td"])
        headers = [c.get_text(strip=True).lower() for c in header_cells]

        col_map: Dict[str, int] = {}
        for idx, h in enumerate(headers):
            if "coach" in h:
                col_map["coach"] = idx
            elif h in ("w", "wins"):
                col_map["wins"] = idx
            elif h in ("l", "losses"):
                col_map["losses"] = idx

        coach_col = col_map.get("coach", 1)
        wins_col = col_map.get("wins", 4)
        losses_col = col_map.get("losses", 5)

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < max(coach_col, wins_col, losses_col) + 1:
                continue

            coach_name = cells[coach_col].get_text(strip=True)
            if not coach_name:
                continue

            wins = self._safe_int(cells[wins_col].get_text(strip=True))
            losses = self._safe_int(cells[losses_col].get_text(strip=True))
            appearances = wins + losses

            teams = []
            for link in cells[coach_col].find_all("a"):
                team = link.get_text(strip=True)
                if team and team != coach_name:
                    teams.append(team)

            normalized = self._normalize_name(coach_name)
            coaches[normalized] = {
                "name": coach_name,
                "appearances": max(1, appearances // 2) if appearances > 0 else 0,
                "wins": wins,
                "losses": losses,
                "teams": teams,
            }

        logger.info("Parsed %d coaches from tournament HTML data", len(coaches))
        return coaches

    def build_team_to_coach_appearances(
        self,
        coach_data: Dict[str, Dict[str, object]],
        team_to_coach_map: Dict[str, str],
    ) -> Dict[str, int]:
        """
        Map team IDs to their coach's tournament appearance count.

        Args:
            coach_data: Output from fetch_coach_tournament_experience().
            team_to_coach_map: Dict of team_id -> coach name.

        Returns:
            Dict of team_id -> coach tournament appearances.
        """
        result: Dict[str, int] = {}
        for team_id, coach_name in team_to_coach_map.items():
            normalized = self._normalize_name(coach_name)
            info = coach_data.get(normalized, {})
            result[team_id] = int(info.get("appearances", 0))

            # Fuzzy match: try last name only
            if not info and " " in coach_name:
                last_name = coach_name.split()[-1].lower()
                for k, v in coach_data.items():
                    if k.endswith(last_name) or last_name in k:
                        result[team_id] = int(v.get("appearances", 0))
                        break

        return result

    def build_team_to_coach_win_rate(
        self,
        coach_data: Dict[str, Dict[str, object]],
        team_to_coach_map: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Map team IDs to their coach's tournament win rate (wins / games).

        Uses coach_data fields: "wins", "losses", "appearances".
        Falls back to 0.0 if no tournament games found.

        Returns:
            Dict of team_id -> coach tournament win rate [0, 1].
        """
        result: Dict[str, float] = {}
        for team_id, coach_name in team_to_coach_map.items():
            normalized = self._normalize_name(coach_name)
            info = coach_data.get(normalized, {})

            # Fuzzy match: try last name only
            if not info and " " in coach_name:
                last_name = coach_name.split()[-1].lower()
                for k, v in coach_data.items():
                    if k.endswith(last_name) or last_name in k:
                        info = v
                        break

            wins = int(info.get("wins", 0))
            losses = int(info.get("losses", 0))
            total = wins + losses
            if total > 0:
                result[team_id] = wins / total
            else:
                result[team_id] = 0.0

        return result

    # ------------------------------------------------------------------
    # 2b. Team-to-Coach Mapping (Barttorvik team ratings page)
    # ------------------------------------------------------------------

    def fetch_team_coaches(self, year: int) -> Dict[str, str]:
        """Fetch a mapping of team_id -> head coach name from Barttorvik.

        Primary: Team results CSV (structured, not behind JS wall).
        Fallback: HTML trank.php page (fragile, behind Cloudflare).

        Args:
            year: Season end year.

        Returns:
            Dict mapping normalized team_id -> coach full name.
        """
        cache_name = f"team_coaches_{year}.json"
        cached = self._load_cache(cache_name)
        if cached and "coaches" in cached:
            return cached["coaches"]

        # Strategy 1: CSV (reliable, not behind JS wall)
        coaches = self._team_coaches_from_csv(year)

        # Strategy 2: HTML fallback (fragile, behind Cloudflare)
        if not coaches:
            logger.info(
                "CSV team-coach mapping returned no results; falling back to HTML scrape for %d.",
                year,
            )
            coaches = self._scrape_team_coaches(year)

        if coaches:
            self._save_cache(cache_name, {"coaches": coaches, "year": year})
        return coaches

    def _scrape_team_coaches(self, year: int) -> Dict[str, str]:
        """Scrape team-to-coach mapping from Barttorvik team ratings page."""
        url = f"{self.BASE_URL_TORVIK}/trank.php?year={year}&conyes=1"
        coaches: Dict[str, str] = {}

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Could not fetch Barttorvik team ratings for %d: %s", year, e)
            return coaches

        soup = _make_soup(resp.text)
        table = soup.find("table")
        if not table:
            logger.warning("No table found on Barttorvik team ratings page")
            return coaches

        rows = table.find_all("tr")
        if len(rows) < 2:
            return coaches

        # Parse header to find team and coach columns
        header_cells = rows[0].find_all(["th", "td"])
        headers = [c.get_text(strip=True).lower() for c in header_cells]

        team_col: Optional[int] = None
        coach_col: Optional[int] = None
        for idx, h in enumerate(headers):
            if h in ("team", "team name") and team_col is None:
                team_col = idx
            elif h in ("coach", "head coach") and coach_col is None:
                coach_col = idx

        if team_col is None or coach_col is None:
            # Fallback: try common positions (team=1, coach=2 on trank page)
            if len(headers) >= 3:
                team_col = team_col if team_col is not None else 1
                coach_col = coach_col if coach_col is not None else 2
            else:
                logger.warning("Could not identify team/coach columns in headers: %s", headers)
                return coaches

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= max(team_col, coach_col):
                continue

            team_name = self._extract_team_name(cells[team_col])
            coach_name = cells[coach_col].get_text(strip=True)

            if team_name and coach_name:
                team_id = self._normalize_name(team_name)
                coaches[team_id] = coach_name

        logger.info("Parsed %d team-coach mappings from Barttorvik for %d", len(coaches), year)
        return coaches

    def _team_coaches_from_csv(self, year: int) -> Dict[str, str]:
        """Extract team-to-coach mapping from Barttorvik's team results CSV.

        The CSV at ``/{year}_team_results.csv`` is not behind the JS
        verification wall.  It typically includes a Coach column alongside
        team ratings data.
        """
        url = f"{self.BASE_URL_TORVIK}/{year}_team_results.csv"
        coaches: Dict[str, str] = {}

        try:
            resp = self.session.get(url, timeout=45)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Could not fetch Barttorvik team results CSV for %d: %s", year, e)
            return coaches

        text = resp.text.strip()
        if not text:
            return coaches

        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        if reader.fieldnames is None:
            logger.warning("Team results CSV has no headers")
            return coaches

        logger.debug(
            "Barttorvik team results CSV columns for %d: %s",
            year,
            reader.fieldnames,
        )

        # Find team and coach columns (case-insensitive)
        lower_fields = {(f or "").strip().lower(): f for f in reader.fieldnames}
        team_key = None
        coach_key = None
        for alias in ("team", "team name", "team_name", "school"):
            match = lower_fields.get(alias)
            if match:
                team_key = match
                break
        for alias in ("coach", "head coach", "head_coach", "coach_name"):
            match = lower_fields.get(alias)
            if match:
                coach_key = match
                break

        if not team_key or not coach_key:
            logger.warning(
                "Team results CSV missing team/coach columns — team_key=%s, coach_key=%s (available headers: %s)",
                team_key,
                coach_key,
                list(reader.fieldnames),
            )
            return coaches

        for row in reader:
            team_name = (row.get(team_key) or "").strip()
            coach_name = (row.get(coach_key) or "").strip()
            if team_name and coach_name:
                team_id = self._normalize_name(team_name)
                coaches[team_id] = coach_name

        logger.info(
            "Parsed %d team-coach mappings from Barttorvik CSV for %d",
            len(coaches),
            year,
        )
        return coaches

    # ------------------------------------------------------------------
    # 3. Conference Tournament Champions
    # ------------------------------------------------------------------

    def fetch_conference_tournament_champions(self, year: int) -> Dict[str, str]:
        """
        Fetch conference tournament champions for a season.

        Args:
            year: Season end year.

        Returns:
            Dict mapping normalized team name -> conference name.
            Only teams that WON their conference tournament are included.
        """
        cache_name = f"conf_tourney_champs_{year}.json"
        cached = self._load_cache(cache_name)
        if cached and "champions" in cached:
            return cached["champions"]

        champions = self._scrape_conf_tourney_champions(year)
        if champions:
            self._save_cache(cache_name, {"champions": champions, "year": year})
        return champions

    def _scrape_conf_tourney_champions(self, year: int) -> Dict[str, str]:
        """Fetch conference tournament champions.

        Primary: ESPN conference seeds API (structured JSON).
        Fallback: Sports-Reference season summary HTML (fragile id="conference-summary").
        """
        # --- Primary: ESPN conference seeds (structured JSON) ---
        champions = self._conf_champions_from_espn(year)
        if champions:
            logger.info(
                "Loaded %d conference tournament champions from ESPN for %d",
                len(champions),
                year,
            )
            return champions

        # --- Fallback: SR HTML (fragile) ---
        logger.info(
            "ESPN conference data unavailable; falling back to SR HTML for %d",
            year,
        )
        return self._conf_champions_from_sr_html(year)

    def _conf_champions_from_espn(self, year: int) -> Dict[str, str]:
        """Try to extract conference tournament champions from ESPN API.

        Uses the conference_seeds module which already implements ESPN
        scoreboard API integration for conference tournament data.
        """
        try:
            from .conference_seeds import ConferenceSeedScraper

            scraper = ConferenceSeedScraper(
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            seeds_by_conf = scraper.fetch_all_conference_seeds(year)
            if not seeds_by_conf:
                return {}

            # The #1 seed in each conference tournament is typically the
            # tournament champion (for completed tournaments) or regular
            # season champion. Extract them.
            champions: Dict[str, str] = {}
            for conf_name, seeds in seeds_by_conf.items():
                if not seeds:
                    continue
                # Find the team with seed 1
                for team in seeds:
                    if isinstance(team, dict) and team.get("seed") == 1:
                        team_name = team.get("team_name") or team.get("team_id") or ""
                        if team_name:
                            champions[self._normalize_name(team_name)] = conf_name
                        break

            return champions
        except Exception as exc:
            logger.debug("ESPN conference champions extraction failed: %s", exc)
            return {}

    def _conf_champions_from_sr_html(self, year: int) -> Dict[str, str]:
        """Scrape conference tournament champions from SR HTML (legacy fallback)."""
        url = f"{self.BASE_URL_SR}/seasons/men/{year}.html"
        champions: Dict[str, str] = {}

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Could not fetch SR season summary for %d: %s", year, e)
            return champions

        soup = _make_soup(resp.text)

        table = soup.find("table", {"id": "conference-summary"})
        if not table:
            for t in soup.find_all("table"):
                headers = [th.get_text(strip=True) for th in t.find_all("th")]
                if any("tournament" in h.lower() and "champ" in h.lower() for h in headers):
                    table = t
                    break

        if not table:
            logger.warning("Conference summary table not found for %d", year)
            return champions

        header_row = table.find("thead")
        if not header_row:
            return champions

        headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
        champ_col = None
        conf_col = None
        for idx, h in enumerate(headers):
            h_lower = h.lower()
            if "tournament" in h_lower and "champ" in h_lower:
                champ_col = idx
            elif h_lower in ("conference", "conf"):
                conf_col = idx
            elif idx == 0 and conf_col is None:
                conf_col = idx

        if champ_col is None:
            logger.warning("Tournament Champ column not found in headers: %s", headers)
            return champions

        body = table.find("tbody")
        rows = body.find_all("tr") if body else table.find_all("tr")[1:]

        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) <= max(champ_col, conf_col or 0):
                continue

            champ_name = self._extract_team_name(cells[champ_col])
            conf_name = cells[conf_col].get_text(strip=True) if conf_col is not None else ""

            if champ_name:
                champions[self._normalize_name(champ_name)] = conf_name

        logger.info("Found %d conference tournament champions from SR HTML for %d", len(champions), year)
        return champions

    # ------------------------------------------------------------------
    # Loading from pre-built JSON files (for SOTA pipeline consumption)
    # ------------------------------------------------------------------

    @staticmethod
    def load_preseason_ap_from_json(filepath: str) -> Dict[str, int]:
        """Load preseason AP rankings from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("rankings", {})

    @staticmethod
    def load_coach_data_from_json(filepath: str) -> Dict[str, Dict]:
        """Load coach tournament data from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("coaches", {})

    @staticmethod
    def load_conf_champions_from_json(filepath: str) -> Dict[str, str]:
        """Load conference tournament champions from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("champions", {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_team_name(cell) -> str:
        """Extract team name from an HTML table cell."""
        link = cell.find("a")
        if link:
            return link.get_text(strip=True)
        return cell.get_text(strip=True)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a team or coach name to a lookup key."""
        return "".join(c.lower() if c.isalnum() else "_" for c in (name or "")).strip("_")

    @staticmethod
    def _parse_rank(text: str) -> Optional[int]:
        """Parse a rank number from text, handling tied ranks like '14T'."""
        match = re.match(r"(\d+)", text.strip())
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _safe_int(text: str) -> int:
        try:
            return int(text.strip())
        except (ValueError, AttributeError):
            return 0

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / filename
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, data: Dict) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
