"""ESPN BPI (Basketball Power Index) scraper.

Fetches game-specific win probabilities and season-level team ratings
from ESPN's undocumented public API. No authentication required.

Endpoints:
    sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball
        /seasons/{year}/powerindex          — team BPI ratings (paginated)
        /seasons/{year}/teams               — ESPN team ID → name mapping
        /events/{id}/competitions/{id}/predictor  — game-specific predictions

Coverage: 2020-2026+ (confirmed). Data refreshes ~every 15 minutes during season.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.data.normalize import normalize_team_id
from src.data.scrapers._retry import retry_request

logger = logging.getLogger(__name__)

_API_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
_REQUEST_DELAY = 2.0  # seconds between API calls
_PAGE_SIZE = 50
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BPITeamRating:
    """Season-level BPI rating for a single team."""

    team_id: str  # canonical (normalize_team_id output)
    team_name: str  # display name from ESPN
    espn_id: int = 0
    bpi: float = 0.0
    bpi_offense: float = 0.0
    bpi_defense: float = 0.0
    sor: float = 0.0  # strength of record
    sos: float = 0.0  # strength of schedule
    rpi_rank: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class BPIGamePrediction:
    """Game-specific BPI prediction."""

    event_id: str
    game_date: str
    team1_id: str  # canonical
    team2_id: str  # canonical
    team1_espn_id: int = 0
    team2_espn_id: int = 0
    team1_win_pct: float = 0.5
    team2_win_pct: float = 0.5
    team1_pred_margin: float = 0.0
    matchup_quality: float = 0.0
    round_name: str = ""


@dataclass
class BPISeasonData:
    """All BPI data for a single season."""

    season: int
    timestamp: str = ""
    teams: Dict[str, BPITeamRating] = field(default_factory=dict)
    games: List[BPIGamePrediction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class BPIScraper:
    """Fetches BPI data from ESPN's public API."""

    def __init__(self, cache_dir: str = "data/raw/bpi"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT})
        self._espn_team_cache: Dict[int, Dict[str, str]] = {}

    # -- HTTP helpers -------------------------------------------------------

    def _get_json(self, url: str, label: str = "") -> Optional[Dict[str, Any]]:
        """GET a JSON endpoint with retry logic. Returns None on 404."""
        try:
            resp = retry_request(self.session.get, url, timeout=15)
            if resp.status_code == 404:
                logger.debug("404 for %s: %s", label, url)
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            logger.warning("HTTP error fetching %s: %s", label, exc)
            return None
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            logger.warning("Error fetching %s: %s", label, exc)
            return None

    def _get_paginated(self, base_url: str, label: str, limit: int = _PAGE_SIZE) -> List[Dict[str, Any]]:
        """Fetch all pages from a paginated ESPN endpoint."""
        items: List[Dict[str, Any]] = []
        page = 1
        while True:
            url = f"{base_url}?limit={limit}&page={page}"
            data = self._get_json(url, f"{label} page {page}")
            if not data:
                break

            page_items = data.get("items", [])
            items.extend(page_items)

            total = data.get("count", 0)
            if len(items) >= total or not page_items:
                break

            page += 1
            time.sleep(_REQUEST_DELAY)

        logger.info("Fetched %d items from %s", len(items), label)
        return items

    # -- ESPN team ID resolution --------------------------------------------

    _SITE_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=500"

    def _resolve_espn_teams(self, season: int) -> Dict[int, Dict[str, str]]:
        """Map ESPN numeric team IDs to {espn_id: {name, canonical_id}}.

        Uses the site API which returns all ~362 D1 teams inline in one
        request (no pagination or $ref resolution needed).

        Results are cached per season both in memory and on disk.
        """
        if season in self._espn_team_cache:
            return self._espn_team_cache[season]

        cache_path = self.cache_dir / f"espn_teams_{season}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            mapping = {int(k): v for k, v in raw.items()}
            self._espn_team_cache[season] = mapping
            return mapping

        logger.info("Fetching ESPN team list for %d...", season)
        data = self._get_json(self._SITE_TEAMS_URL, "site-teams")

        mapping: Dict[int, Dict[str, str]] = {}
        if data:
            sports = data.get("sports", [{}])
            leagues = sports[0].get("leagues", [{}]) if sports else [{}]
            teams = leagues[0].get("teams", []) if leagues else []

            for entry in teams:
                team = entry.get("team", entry)
                espn_id = int(team.get("id", 0))
                name = team.get("displayName", team.get("name", ""))
                if espn_id and name:
                    mapping[espn_id] = {
                        "name": name,
                        "canonical_id": normalize_team_id(name),
                    }

        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in mapping.items()}, f)

        logger.info("Resolved %d ESPN teams for %d", len(mapping), season)
        self._espn_team_cache[season] = mapping
        return mapping

    def _canonical(self, espn_id: int, season: int) -> str:
        """Convert ESPN team ID to canonical team_id."""
        teams = self._resolve_espn_teams(season)
        entry = teams.get(espn_id)
        if entry:
            return entry["canonical_id"]
        return f"espn_{espn_id}"

    def _espn_name(self, espn_id: int, season: int) -> str:
        """Get display name for an ESPN team ID."""
        teams = self._resolve_espn_teams(season)
        entry = teams.get(espn_id)
        if entry:
            return entry["name"]
        return f"Unknown ({espn_id})"

    # -- Season-level BPI ratings -------------------------------------------

    def fetch_team_ratings(
        self,
        season: int,
        espn_ids: Optional[List[int]] = None,
    ) -> Dict[str, BPITeamRating]:
        """Fetch BPI ratings for teams in a season.

        Args:
            season: Season year (e.g. 2026).
            espn_ids: If provided, fetch only these ESPN team IDs (fast).
                If None, fetches the top-25 from the paginated leaders
                endpoint plus any tournament teams discovered.

        Returns dict of canonical_team_id -> BPITeamRating.
        """
        logger.info("Fetching BPI team ratings for %d...", season)
        ratings: Dict[str, BPITeamRating] = {}

        if espn_ids:
            # Per-team fetch — used when we know which teams we want
            for eid in espn_ids:
                url = f"{_API_BASE}/seasons/{season}/powerindex/{eid}"
                data = self._get_json(url, f"bpi-team-{eid}")
                if data:
                    rating = self._parse_team_rating(data, season)
                    if rating:
                        ratings[rating.team_id] = rating
                time.sleep(_REQUEST_DELAY)
        else:
            # Paginated leaders — returns top 25 by BPI
            url = f"{_API_BASE}/seasons/{season}/powerindex"
            items = self._get_paginated(url, f"bpi-ratings-{season}")

            for item in items:
                team_data = item
                if "$ref" in item and "team" not in item:
                    ref_data = self._get_json(item["$ref"], "bpi-ref")
                    if not ref_data:
                        continue
                    team_data = ref_data
                    time.sleep(0.5)

                rating = self._parse_team_rating(team_data, season)
                if rating:
                    ratings[rating.team_id] = rating

            # Also fetch tournament teams not in top 25
            events = self._get_tournament_events(season)
            tourney_espn_ids = set()
            for evt in events:
                tourney_espn_ids.add(evt["team1_espn_id"])
                tourney_espn_ids.add(evt["team2_espn_id"])

            existing = {r.espn_id for r in ratings.values()}
            missing = tourney_espn_ids - existing - {0}
            if missing:
                logger.info("Fetching BPI for %d additional tournament teams", len(missing))
                for eid in sorted(missing):
                    url = f"{_API_BASE}/seasons/{season}/powerindex/{eid}"
                    data = self._get_json(url, f"bpi-team-{eid}")
                    if data:
                        rating = self._parse_team_rating(data, season)
                        if rating:
                            ratings[rating.team_id] = rating
                    time.sleep(_REQUEST_DELAY)

        # Save raw
        raw_path = self.cache_dir / f"bpi_ratings_{season}.json"
        with open(raw_path, "w") as f:
            json.dump([asdict(r) for r in ratings.values()], f, indent=2)

        logger.info("Got BPI ratings for %d teams in %d", len(ratings), season)
        return ratings

    def _parse_team_rating(self, data: Dict[str, Any], season: int) -> Optional[BPITeamRating]:
        """Parse a single team's BPI rating from API response."""
        # Extract team info
        team_ref = data.get("team", {})
        if isinstance(team_ref, dict) and "$ref" in team_ref:
            # Extract ESPN ID from the $ref URL (e.g. .../teams/150?lang=en)
            ref_url = team_ref["$ref"]
            try:
                m = re.search(r"/teams/(\d+)", ref_url)
                espn_id = int(m.group(1)) if m else 0
            except (ValueError, AttributeError):
                return None
        elif isinstance(team_ref, dict):
            espn_id = int(team_ref.get("id", 0))
        else:
            return None

        if not espn_id:
            return None

        # Parse stats — API uses "stats" (live) or "statistics" (older cache)
        stats: Dict[str, float] = {}
        for stat in data.get("stats", data.get("statistics", [])):
            name = stat.get("name", "")
            val = stat.get("value")
            if name and val is not None:
                try:
                    stats[name] = float(val)
                except (ValueError, TypeError):
                    pass

        return BPITeamRating(
            team_id=self._canonical(espn_id, season),
            team_name=self._espn_name(espn_id, season),
            espn_id=espn_id,
            bpi=stats.get("bpi", stats.get("bpirating", 0.0)),
            bpi_offense=stats.get("bpioffense", stats.get("bpioffrating", 0.0)),
            bpi_defense=stats.get("bpidefense", stats.get("bpidefrating", 0.0)),
            sor=stats.get("sorrank", stats.get("sor", 0.0)),
            sos=stats.get("sospastrank", stats.get("sosrank", stats.get("sos", 0.0))),
            rpi_rank=int(stats.get("rpirank", stats.get("rpi", 0))),
            wins=int(stats.get("wins", 0)),
            losses=int(stats.get("losses", 0)),
        )

    # -- Tournament event discovery -----------------------------------------

    def _get_tournament_events(self, season: int) -> List[Dict[str, Any]]:
        """Get NCAA tournament event/competition IDs for a season.

        Tries postseason events endpoint first, then falls back to
        calendar-based event discovery.
        """
        cache_path = self.cache_dir / f"tournament_events_{season}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        logger.info("Discovering tournament events for %d...", season)

        # Try the groups endpoint (NCAA tournament = group 100, type 3 = postseason)
        events: List[Dict[str, Any]] = []
        for type_id in (3, 4):  # 3 = conference tourneys + NCAA, 4 = NCAA only
            url = f"{_API_BASE}/seasons/{season}/types/{type_id}/events"
            data = self._get_json(url, f"events-type{type_id}-{season}")
            if data and data.get("items"):
                for item in data["items"]:
                    event_data = item
                    if "$ref" in item:
                        event_data = self._get_json(item["$ref"], "event-ref")
                        if not event_data:
                            continue
                        time.sleep(0.5)
                    parsed = self._parse_event(event_data, season)
                    if parsed:
                        events.append(parsed)
                if events:
                    break
            time.sleep(_REQUEST_DELAY)

        if not events:
            # Fallback: try scoreboard for recent/upcoming tournament games
            url = f"{_API_BASE}/scoreboard?dates={season}0301-{season}0410&limit=100"
            data = self._get_json(url, f"scoreboard-{season}")
            if data:
                for event_data in data.get("events", []):
                    parsed = self._parse_event(event_data, season)
                    if parsed:
                        events.append(parsed)

        with open(cache_path, "w") as f:
            json.dump(events, f, indent=2)

        logger.info("Found %d tournament events for %d", len(events), season)
        return events

    def _parse_event(self, data: Dict[str, Any], season: int) -> Optional[Dict[str, Any]]:
        """Parse an event into {event_id, competition_id, teams, date, round}."""
        event_id = str(data.get("id", ""))
        if not event_id:
            return None

        competitions = data.get("competitions", [])
        if not competitions:
            return None

        comp = competitions[0]
        comp_id = str(comp.get("id", event_id))

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        team1 = competitors[0]
        team2 = competitors[1]

        t1_id = int(team1.get("id", team1.get("team", {}).get("id", 0)))
        t2_id = int(team2.get("id", team2.get("team", {}).get("id", 0)))

        game_date = data.get("date", comp.get("date", ""))
        round_name = ""
        notes = data.get("competitions", [{}])[0].get("notes", [])
        if notes:
            round_name = notes[0].get("headline", "")

        return {
            "event_id": event_id,
            "competition_id": comp_id,
            "team1_espn_id": t1_id,
            "team2_espn_id": t2_id,
            "game_date": game_date[:10] if game_date else "",
            "round_name": round_name,
        }

    # -- Game-specific predictions ------------------------------------------

    def fetch_game_predictions(self, season: int) -> List[BPIGamePrediction]:
        """Fetch BPI predictions for all tournament games in a season."""
        events = self._get_tournament_events(season)
        if not events:
            logger.warning("No tournament events found for %d", season)
            return []

        logger.info(
            "Fetching BPI predictions for %d tournament games (%d)...",
            len(events),
            season,
        )
        predictions: List[BPIGamePrediction] = []

        for evt in events:
            eid = evt["event_id"]
            cid = evt["competition_id"]

            url = f"{_API_BASE}/events/{eid}/competitions/{cid}/predictor"
            data = self._get_json(url, f"predictor-{eid}")
            if not data:
                time.sleep(_REQUEST_DELAY)
                continue

            pred = self._parse_prediction(data, evt, season)
            if pred:
                predictions.append(pred)

            time.sleep(_REQUEST_DELAY)

        # Save raw
        raw_path = self.cache_dir / f"bpi_predictions_{season}.json"
        with open(raw_path, "w") as f:
            json.dump([asdict(p) for p in predictions], f, indent=2)

        logger.info("Got %d game predictions for %d", len(predictions), season)
        return predictions

    def _parse_prediction(
        self,
        data: Dict[str, Any],
        event: Dict[str, Any],
        season: int,
    ) -> Optional[BPIGamePrediction]:
        """Parse a predictor response into BPIGamePrediction."""
        # The predictor response has homeTeam/awayTeam or a list of teams
        home = data.get("homeTeam", {})
        away = data.get("awayTeam", {})

        # Some responses use a flat structure
        if not home and not away:
            teams = data.get("teams", data.get("competitors", []))
            if isinstance(teams, list) and len(teams) >= 2:
                home = teams[0]
                away = teams[1]

        if not home or not away:
            return None

        t1_espn = event["team1_espn_id"]
        t2_espn = event["team2_espn_id"]

        # Extract win probability — field names vary across responses
        t1_win = _extract_float(
            home,
            "gameProjection",
            "teampredwinpct",
            "winPercentage",
        )
        t2_win = _extract_float(
            away,
            "gameProjection",
            "teampredwinpct",
            "winPercentage",
        )

        # Normalize to sum to 100 if needed
        if t1_win + t2_win > 0 and abs(t1_win + t2_win - 100.0) > 1.0:
            total = t1_win + t2_win
            t1_win = (t1_win / total) * 100.0
            t2_win = (t2_win / total) * 100.0

        t1_margin = _extract_float(home, "teampredmov", "predictedMargin")
        mq = _extract_float(data, "matchupQuality", "matchupquality")

        return BPIGamePrediction(
            event_id=event["event_id"],
            game_date=event["game_date"],
            team1_id=self._canonical(t1_espn, season),
            team2_id=self._canonical(t2_espn, season),
            team1_espn_id=t1_espn,
            team2_espn_id=t2_espn,
            team1_win_pct=round(t1_win, 2),
            team2_win_pct=round(t2_win, 2),
            team1_pred_margin=round(t1_margin, 2),
            matchup_quality=round(mq, 2),
            round_name=event.get("round_name", ""),
        )

    # -- Orchestration ------------------------------------------------------

    def scrape_season(self, season: int) -> BPISeasonData:
        """Fetch all BPI data for a season (ratings + game predictions)."""
        from datetime import datetime

        ratings = self.fetch_team_ratings(season)
        games = self.fetch_game_predictions(season)

        data = BPISeasonData(
            season=season,
            timestamp=datetime.utcnow().isoformat() + "Z",
            teams=ratings,
            games=games,
        )

        self.save_processed(data)
        return data

    def save_processed(self, data: BPISeasonData) -> Path:
        """Write normalized BPI data to processed directory."""
        out_dir = Path("data/processed/bpi")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"bpi_{data.season}.json"

        payload = {
            "season": data.season,
            "timestamp": data.timestamp,
            "n_teams": len(data.teams),
            "n_games": len(data.games),
            "teams": {k: asdict(v) for k, v in data.teams.items()},
            "games": [asdict(g) for g in data.games],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved processed BPI to %s", path)
        return path

    @staticmethod
    def load_processed(season: int) -> Optional[BPISeasonData]:
        """Load processed BPI data from disk."""
        path = Path(f"data/processed/bpi/bpi_{season}.json")
        if not path.exists():
            return None

        with open(path) as f:
            raw = json.load(f)

        teams = {}
        for tid, tdata in raw.get("teams", {}).items():
            teams[tid] = BPITeamRating(**tdata)

        games = [BPIGamePrediction(**g) for g in raw.get("games", [])]

        return BPISeasonData(
            season=raw["season"],
            timestamp=raw.get("timestamp", ""),
            teams=teams,
            games=games,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_float(data: Dict[str, Any], *keys: str) -> float:
    """Try multiple keys, return the first non-None float value."""
    for key in keys:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ESPN BPI Scraper")
    parser.add_argument("--season", type=int, help="Season year (e.g. 2026)")
    parser.add_argument("--all", action="store_true", help="Scrape all seasons 2020-2026")
    parser.add_argument(
        "--ratings-only",
        action="store_true",
        help="Only fetch team ratings (skip game predictions)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch one page and print, don't save",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scraper = BPIScraper()

    if args.dry_run:
        season = args.season or 2026
        url = f"{_API_BASE}/seasons/{season}/powerindex?limit=3"
        data = scraper._get_json(url, "dry-run")
        print(json.dumps(data, indent=2)[:3000] if data else "No data returned")
        return

    seasons = list(range(2020, 2027)) if args.all else [args.season or 2026]

    for season in seasons:
        logger.info("=== Season %d ===", season)
        if args.ratings_only:
            ratings = scraper.fetch_team_ratings(season)
            logger.info("Got %d team ratings", len(ratings))
        else:
            data = scraper.scrape_season(season)
            logger.info("Done: %d teams, %d games", len(data.teams), len(data.games))


if __name__ == "__main__":
    main()
