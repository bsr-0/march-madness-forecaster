"""Scrape NCAA tournament game results.

Primary source: ESPN scoreboard API (structured JSON, stable endpoint).
Fallback: Sports Reference bracket HTML parsing.

Usage::

    python -m src.main scrape-tournament-results --year 2024 \\
        --cache-dir data/raw/cache \\
        --output data/raw/historical/tournament_results_2024.json
"""

from __future__ import annotations

import html as _html
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ..normalize import normalize_team_id

logger = logging.getLogger(__name__)

# ESPN scoreboard API — stable, structured, no auth required.
_ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

# ESPN season type 3 = postseason (NCAA tournament).
_ESPN_SEASON_TYPE = 3

# Map ESPN round slugs / display names to our canonical round labels.
_ESPN_ROUND_MAP = {
    "first four": "R68",
    "first round": "R64",
    "second round": "R32",
    "sweet 16": "S16",
    "elite 8": "E8",
    "elite eight": "E8",
    "final four": "F4",
    "national championship": "NCG",
    "championship": "NCG",
}

# Round labels based on column position in SR bracket layout (fallback)
_ROUND_LABELS = {
    0: "R64",
    1: "R32",
    2: "S16",
    3: "E8",
    4: "F4",
    5: "NCG",
}


class TournamentResultsScraper:
    """Fetch NCAA tournament game results from Sports Reference."""

    BASE_URL = "https://www.sports-reference.com/cbb/postseason/men"

    def __init__(self, cache_dir: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def scrape_year(self, season: int) -> List[Dict]:
        """Fetch tournament game results, trying ESPN API first.

        Args:
            season: The tournament year (e.g., 2024).

        Returns:
            List of game dicts with keys:
                year, round_name, region, team1_id, team1_seed, team1_score,
                team2_id, team2_seed, team2_score, team1_won
        """
        cache_name = f"tournament_results_{season}.json"
        cached = self._load_cache(cache_name)
        if cached and isinstance(cached, dict) and len(cached.get("games", [])) >= 60:
            logger.info("Using cached tournament results for %d", season)
            return cached["games"]

        # --- Primary: ESPN scoreboard API (structured JSON) ---
        games = self._scrape_espn(season)
        if games and len(games) >= 10:
            logger.info("ESPN API returned %d tournament games for %d", len(games), season)
        else:
            # --- Fallback: Sports Reference bracket HTML ---
            logger.info(
                "ESPN returned %d games; falling back to Sports Reference HTML for %d",
                len(games) if games else 0, season,
            )
            games = self._scrape_sports_reference(season)

        # Validate through pydantic schema
        if games:
            try:
                from .schemas import validate_tournament_games
                games = validate_tournament_games(games)
            except Exception as e:
                logger.warning("Tournament games schema validation failed: %s", e)

        if games:
            self._save_cache(cache_name, {"year": season, "games": games})
            logger.info("Scraped %d tournament games for %d", len(games), season)
        else:
            logger.warning("No games parsed for %d", season)

        return games

    # ------------------------------------------------------------------
    # ESPN scoreboard API (primary — structured JSON, no HTML parsing)
    # ------------------------------------------------------------------

    def _scrape_espn(self, season: int) -> List[Dict]:
        """Fetch NCAA tournament results from ESPN scoreboard API.

        ESPN's public API returns structured JSON with team names, scores,
        seeds, and tournament round info.  No authentication required.
        """
        games = []

        # ESPN paginates by date.  Tournament spans ~3 weeks.
        # We fetch day-by-day from mid-March through early April.
        import datetime as _dt

        # Tournament window: March 15 through April 10
        start = _dt.date(season, 3, 15)
        end = _dt.date(season, 4, 10)
        day = start

        while day <= end:
            day_str = day.strftime("%Y%m%d")
            params = {
                "dates": day_str,
                "seasontype": _ESPN_SEASON_TYPE,
                "groups": 100,  # NCAA tournament group
                "limit": 100,
            }
            try:
                resp = self.session.get(
                    _ESPN_SCOREBOARD_URL, params=params, timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.debug("ESPN scoreboard fetch failed for %s: %s", day_str, e)
                day += _dt.timedelta(days=1)
                continue

            for event in data.get("events", []):
                game = self._parse_espn_event(event, season)
                if game:
                    games.append(game)

            day += _dt.timedelta(days=1)
            # Be polite — small delay between requests
            time.sleep(0.3)

        # Deduplicate by (team1_id, team2_id, round_name)
        seen = set()
        unique = []
        for g in games:
            key = (g["team1_id"], g["team2_id"], g["round_name"])
            rev_key = (g["team2_id"], g["team1_id"], g["round_name"])
            if key not in seen and rev_key not in seen:
                seen.add(key)
                unique.append(g)

        return unique

    def _parse_espn_event(self, event: Dict, season: int) -> Optional[Dict]:
        """Parse a single ESPN event into a tournament game dict."""
        competitions = event.get("competitions", [])
        if not competitions:
            return None
        comp = competitions[0]

        # Only include completed games
        status = comp.get("status", {})
        if isinstance(status, dict):
            type_info = status.get("type", {})
            if type_info.get("completed") is not True:
                return None
        else:
            return None

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        # Parse both competitors
        teams = []
        for c in competitors:
            team_info = c.get("team", {})
            name = team_info.get("displayName") or team_info.get("shortDisplayName") or ""
            if not name:
                return None
            try:
                score = int(float(c.get("score") or 0))
            except (TypeError, ValueError):
                return None
            # Seed is in curateRank or seed field
            seed = 0
            if c.get("curatedRank", {}).get("current"):
                try:
                    seed = int(c["curatedRank"]["current"])
                except (TypeError, ValueError):
                    pass
            if seed == 0 and isinstance(c.get("seed"), (int, str)):
                try:
                    seed = int(c["seed"])
                except (TypeError, ValueError):
                    pass
            teams.append({
                "name": name,
                "team_id": normalize_team_id(name),
                "score": score,
                "seed": seed,
                "winner": c.get("winner", False),
            })

        if teams[0]["score"] == 0 and teams[1]["score"] == 0:
            return None

        # Determine round from event notes or season type detail
        round_name = "R64"  # default
        notes = event.get("competitions", [{}])[0].get("notes", [])
        for note in notes:
            headline = (note.get("headline") or "").lower()
            for pattern, label in _ESPN_ROUND_MAP.items():
                if pattern in headline:
                    round_name = label
                    break

        # Region from notes
        region = ""
        for note in notes:
            headline = note.get("headline") or ""
            for r in ("East", "West", "South", "Midwest"):
                if r.lower() in headline.lower():
                    region = r
                    break

        t1, t2 = teams[0], teams[1]
        t1_won = t1.get("winner", t1["score"] > t2["score"])

        return {
            "year": season,
            "round_name": round_name,
            "region": region,
            "team1_id": t1["team_id"],
            "team1_seed": t1["seed"],
            "team1_score": t1["score"],
            "team2_id": t2["team_id"],
            "team2_seed": t2["seed"],
            "team2_score": t2["score"],
            "team1_won": bool(t1_won),
        }

    # ------------------------------------------------------------------
    # Sports Reference HTML fallback
    # ------------------------------------------------------------------

    def _scrape_sports_reference(self, season: int) -> List[Dict]:
        """Fetch tournament results from Sports Reference bracket HTML.

        This is the legacy fallback — fragile because it parses bracket
        visualization HTML with regex patterns.
        """
        url = f"{self.BASE_URL}/{season}-ncaa.html"
        logger.info("Fetching tournament results from %s", url)

        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        return self._parse_bracket(response.text, season)

    def scrape_years(
        self, years: List[int], delay: float = 3.0
    ) -> Dict[int, List[Dict]]:
        """Scrape multiple years with polite delays.

        Args:
            years: List of tournament years.
            delay: Seconds between requests (be polite to SR).

        Returns:
            Dict of year -> list of game dicts.
        """
        results = {}
        for i, year in enumerate(years):
            if i > 0:
                time.sleep(delay)
            try:
                results[year] = self.scrape_year(year)
            except Exception as e:
                logger.error("Failed to scrape year %d: %s", year, e)
                results[year] = []
        return results

    def _parse_bracket(self, html: str, season: int) -> List[Dict]:
        """Parse the bracket HTML to extract game results.

        Sports Reference bracket structure (per region):
        - Each region is a ``<div id="region_name">``
        - Games are in ``<div class="round">`` columns
        - Each game has two team rows with seed + name + score
        - Winner is indicated by ``<strong>`` tag on score

        Final Four and Championship are in ``<div id="national">``
        """
        soup = BeautifulSoup(html, "lxml")
        brackets = soup.find("div", {"id": "brackets"})
        if not brackets:
            logger.warning("No brackets div found")
            return []

        games = []

        # Parse regional brackets
        regions = ("east", "west", "south", "midwest",
                   "southeast", "southwest", "mideast", "northwest")
        for region in regions:
            region_div = brackets.find("div", {"id": region})
            if region_div is None:
                continue
            region_games = self._parse_region(region_div, season, region.title())
            games.extend(region_games)

        # Parse national bracket (Final Four + Championship)
        national = brackets.find("div", {"id": "national"})
        if national:
            national_games = self._parse_region(national, season, "National")
            # Assign F4/NCG round names based on order
            for g in national_games:
                if g["round_name"] == "R64":
                    g["round_name"] = "F4"
                elif g["round_name"] == "R32":
                    g["round_name"] = "NCG"
            games.extend(national_games)

        return games

    def _parse_region(
        self, region_div, season: int, region_name: str
    ) -> List[Dict]:
        """Parse games from a single region bracket div."""
        games = []

        # Find all round divs within this region
        round_divs = region_div.find_all("div", class_="round")

        for round_idx, round_div in enumerate(round_divs):
            round_name = _ROUND_LABELS.get(round_idx, f"R{round_idx}")

            # Each game is a pair of consecutive team entries
            # Look for game containers or parse team rows
            game_entries = self._extract_games_from_round(
                round_div, season, round_name, region_name
            )
            games.extend(game_entries)

        return games

    def _extract_games_from_round(
        self, round_div, season: int, round_name: str, region: str
    ) -> List[Dict]:
        """Extract individual games from a round div.

        SR bracket HTML typically has this structure per game:
        <div>
          <span>SEED</span>
          <a href="/cbb/schools/SLUG/men/YEAR.html">TEAM</a>
          <strong>SCORE</strong>  (winner has <strong>)
        </div>
        """
        games = []

        # Strategy: find all team links within this round and pair them
        # Each game is two consecutive teams
        team_links = round_div.find_all(
            "a", href=re.compile(r"/cbb/schools/.+/men/\d+\.html")
        )

        # Process pairs of teams
        for i in range(0, len(team_links) - 1, 2):
            link1 = team_links[i]
            link2 = team_links[i + 1]

            game = self._parse_game_pair(link1, link2, season, round_name, region)
            if game:
                games.append(game)

        return games

    def _parse_game_pair(
        self, link1, link2, season: int, round_name: str, region: str
    ) -> Optional[Dict]:
        """Parse a pair of team links into a game result.

        For each team, we need: slug (from href), seed (from nearby span),
        score (from nearby text/strong), and whether they won.
        """
        try:
            info1 = self._extract_team_info(link1, season)
            info2 = self._extract_team_info(link2, season)

            if not info1 or not info2:
                return None

            # Determine winner — the team with score in <strong> tag
            # or higher score wins
            t1_won = True
            if info1["score"] is not None and info2["score"] is not None:
                if info1["score"] > info2["score"]:
                    t1_won = True
                elif info2["score"] > info1["score"]:
                    t1_won = False
                else:
                    # Tie — check for <strong> indicator
                    t1_won = info1.get("is_winner", True)

            return {
                "year": season,
                "round_name": round_name,
                "region": region,
                "team1_id": info1["team_id"],
                "team1_seed": info1["seed"],
                "team1_score": info1["score"] or 0,
                "team2_id": info2["team_id"],
                "team2_seed": info2["seed"],
                "team2_score": info2["score"] or 0,
                "team1_won": t1_won,
            }
        except Exception as e:
            logger.debug("Failed to parse game pair: %s", e)
            return None

    def _extract_team_info(self, link_tag, season: int) -> Optional[Dict]:
        """Extract team ID, seed, and score from a team's bracket entry."""
        # Team name and slug from the <a> tag
        href = link_tag.get("href", "")
        slug_match = re.search(r"/cbb/schools/([^/]+)/", href)
        if not slug_match:
            return None

        school_slug = slug_match.group(1)
        team_name = _html.unescape(link_tag.get_text(strip=True))
        team_id = normalize_team_id(team_name)

        # Walk up/around to find seed and score
        parent = link_tag.parent
        if parent is None:
            return None

        # Seed: look for a <span> with just a number near the team link
        seed = 0
        seed_span = parent.find("span")
        if seed_span:
            seed_text = seed_span.get_text(strip=True)
            if seed_text.isdigit():
                seed = int(seed_text)

        # Score: look for text content after the team link
        # SR puts the score as text or in a <strong> for the winner
        score = None
        is_winner = False

        # Check for <strong> (indicates winner)
        strong = parent.find("strong")
        if strong:
            score_text = strong.get_text(strip=True)
            if score_text.isdigit():
                score = int(score_text)
                # Is this strong for OUR team's link?
                # Check if the strong is a sibling/child near our link
                is_winner = True

        # If no strong, try plain text after the link
        if score is None:
            # Look for numeric text in the parent
            all_text = parent.get_text(strip=True)
            # Extract numbers that aren't the seed
            nums = re.findall(r'\b(\d+)\b', all_text)
            if len(nums) >= 2:
                # First number is likely seed, second is score
                for n in nums[1:]:
                    if int(n) != seed:
                        score = int(n)
                        break

        return {
            "team_id": team_id,
            "school_slug": school_slug,
            "team_name": team_name,
            "seed": seed,
            "score": score,
            "is_winner": is_winner,
        }

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / filename
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, filename: str, data: Dict) -> None:
        if not self.cache_dir:
            return
        p = self.cache_dir / filename
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
