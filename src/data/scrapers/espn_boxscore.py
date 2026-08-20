"""Per-player box scores from ESPN's public boxscore pages.

Why this exists
---------------
``pbp_player_minutes.py`` reconstructs on-court intervals from play-by-play
substitution events. That works, but only where ESPN publishes substitutions —
and it does not do so before **2025-02-11**. Measured directly (2026-08-19):

    season   opening-day games w/ subs   March games w/ subs
    2026            60/60                     80/80
    2025             0/60                     80/80     <- cutover mid-season
    2024             0/60                      0/80
    2023             0/60                      0/80

Narrowing the 2025 file to the day: 2025-02-08 → 0/12, 2025-02-10 → 0/12,
2025-02-11 → 1/12, 2025-02-12 → 12/12. Sub-like *text* under any play_type is
also absent before that date, so the data is missing rather than relabelled and
no parser change can recover it.

The boxscore page carries what we actually want, directly and historically:
ESPN's own published per-player stat line, minutes first, with players grouped
into ``starters`` / ``bench``. Probed live 2026-08-19 — a 2022 game, a 2015 game
and a 2009 game all return complete data whose per-team minutes sum to exactly
200 (5 players x 40 minutes of regulation).

This is strictly better than the PBP reconstruction on every axis: it covers
every season rather than post-2025-02-11, it is ESPN's published figure rather
than an inference validated against a budget, and the starter flag is labelled
rather than heuristic. It supersedes the PBP route even for 2025-2026, where
the PBP-derived file covers only 2025-02-12 onward and is therefore a biased
basis for season-long minutes shares.

Provenance
----------
Same access pattern as ``cbbpy_pbp``: a plain ``requests.get`` with a bare
``User-Agent: Mozilla/5.0`` against a public page. No JS execution, no
challenge solving, no fingerprint spoofing. Game discovery reuses that module's
scoreboard scraper rather than duplicating it.

Leakage safety
--------------
The season scrape is bounded to dates strictly before that season's
``TOURNAMENT_START_DATES`` entry, mirroring ``cbbpy_pbp`` and ``torvik``. This
is the point of the exercise: the pre-existing minutes sources all carry
*post-tournament* totals from a single scrape date, which is what made
``returning_minutes_pct`` / ``freshman_minutes_pct`` unusable. Per-game rows
bounded to pre-tournament dates fix that by construction.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ._retry import rate_limited_call

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0"
_BOXSCORE_URL = "https://www.espn.com/mens-college-basketball/boxscore/_/gameId/{game_id}"

# The hydration blob holds two "bxscr" keys: a column-schema config, and the
# data array. Only the latter is followed by a team object.
_BXSCR_DATA_MARKER = '"bxscr":[{"tm"'

_DEFAULT_GAME_DELAY = 2.0  # matches cbbpy_pbp's politeness budget
_DEFAULT_SCOREBOARD_DELAY = 2.0
_DEFAULT_CHECKPOINT_EVERY = 10

# A regulation game is 5 on-court players x 40 minutes; each overtime adds
# 5 x 5. Accept any of those, with slack for ESPN's rounding of partial
# minutes (individual rows are whole numbers and need not sum exactly).
_REGULATION_TEAM_MINUTES = 200
_OVERTIME_TEAM_MINUTES = 25
_MAX_OVERTIMES = 6
_MINUTES_TOLERANCE = 6


def _slugify_team_name(name: str) -> str:
    """ "Cornell Big Red" -> "cornell_big_red".

    Matches ``cbbpy_pbp._slugify_team_name`` so team ids line up with the
    ``athlete_team`` / ``home_team_raw`` values already on disk from the PBP
    scrape. Kept as a local copy rather than an import so this module does not
    depend on the PBP scraper's internals.
    """
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _extract_bxscr(html: str) -> Optional[list]:
    """Pull the boxscore data array out of the page's hydration JSON.

    Bracket-matches from the data marker so a nested ``]`` inside a string
    cannot terminate the scan early.
    """
    i = html.find(_BXSCR_DATA_MARKER)
    if i == -1:
        return None
    start = html.find("[", i)
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(html)):
        c = html[j]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start : j + 1])
                except json.JSONDecodeError:
                    logger.warning("bxscr array failed to parse (len=%d)", j + 1 - start)
                    return None
    return None


def _to_minutes(value) -> Optional[float]:
    """ESPN reports whole minutes as a string; DNPs come through as '' or '--'."""
    if value in (None, "", "--", "-"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_boxscore(html: str, game_id: str, game_date: Optional[str] = None) -> Optional[Dict]:
    """Parse one boxscore page into a per-player payload.

    The full stat line is retained, not just minutes. Scraping is the expensive
    step here (one request per game, ~6000 games per season), so throwing away
    points/rebounds/fouls now would mean re-scraping to get them later.

    Returns None when the page carries no boxscore array (postponed game,
    pre-tip page, or an ESPN layout this parser does not recognise) so callers
    can count misses rather than silently recording an empty game.
    """
    bx = _extract_bxscr(html)
    if not bx:
        return None

    teams: List[Dict] = []
    for team in bx:
        tm = team.get("tm") or {}
        display = tm.get("dspNm") or tm.get("nm") or ""
        players: List[Dict] = []

        for group in team.get("stats") or []:
            gtype = group.get("type")
            if gtype not in ("starters", "bench"):
                continue  # "totals" carries no athlete rows
            keys = group.get("keys") or []
            for row in group.get("athlts") or []:
                athlete = row.get("athlt") or {}
                aid = athlete.get("id")
                if not aid:
                    continue
                values = row.get("stats") or []
                stat_line = dict(zip(keys, values))
                players.append(
                    {
                        "athlete_id": str(aid),
                        "athlete_name": athlete.get("dspNm"),
                        "jersey": athlete.get("jersey"),
                        "position": athlete.get("pos"),
                        "started": gtype == "starters",
                        "minutes": _to_minutes(stat_line.get("minutes")),
                        "stats": stat_line,
                    }
                )

        teams.append(
            {
                "team_id": _slugify_team_name(display),
                "team_display": display,
                "espn_team_id": tm.get("id"),
                "home": bool(tm.get("hm")),
                "players": players,
            }
        )

    if not teams:
        return None

    return {"game_id": str(game_id), "game_date": game_date, "teams": teams}


def validate_boxscore_minutes(game_payload: Dict) -> Dict[str, bool]:
    """Per-team check that minutes sum to a legal team-minutes budget.

    Returns ``{team_id: ok}``. A team passes if its minutes total lands within
    tolerance of ``200 + 25*k`` for some overtime count k. This is the same
    idea as ``pbp_player_minutes.validate_game_minutes``, but here it is a
    genuine integrity check on published data rather than a filter compensating
    for a lossy reconstruction — in practice it should almost always pass, and
    a run with a high failure rate means the parser or the page shape changed.
    """
    budgets = [_REGULATION_TEAM_MINUTES + _OVERTIME_TEAM_MINUTES * k for k in range(_MAX_OVERTIMES + 1)]
    out: Dict[str, bool] = {}
    for team in game_payload.get("teams") or []:
        total = sum(p["minutes"] for p in team.get("players") or [] if p.get("minutes") is not None)
        out[team["team_id"]] = any(abs(total - b) <= _MINUTES_TOLERANCE for b in budgets)
    return out


class EspnBoxscoreScraper:
    """Fetch and persist a season of per-player box scores, bounded pre-tournament."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- io ---------------------------------------------------------------

    def _cache_path(self, year: int) -> Optional[Path]:
        return self.cache_dir / f"boxscores_{year}.json" if self.cache_dir else None

    def _load_cache(self, year: int) -> Optional[Dict]:
        path = self._cache_path(year)
        if not path or not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Discarding unreadable checkpoint %s", path)
            return None

    def _save_cache(self, year: int, payload: Dict) -> None:
        """Atomic checkpoint write (temp + fsync + os.replace).

        A season takes hours and this file is rewritten periodically. A plain
        in-place dump leaves a window where a crash truncates the checkpoint,
        which ``_load_cache`` then discards — restarting the whole season.
        """
        path = self._cache_path(year)
        if not path:
            return
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # -- fetching ---------------------------------------------------------

    def fetch_game(self, game_id: str, *, delay: float = _DEFAULT_GAME_DELAY, timeout: int = 20) -> Optional[Dict]:
        """Fetch and parse one game. Returns None if the page has no boxscore."""
        import requests

        url = _BOXSCORE_URL.format(game_id=game_id)
        resp = rate_limited_call(requests.get, url, delay=delay, headers={"User-Agent": _UA}, timeout=timeout)
        if resp is None or getattr(resp, "status_code", None) != 200:
            logger.debug("boxscore fetch failed for %s", game_id)
            return None
        return parse_boxscore(resp.text, game_id)

    def fetch_season(
        self,
        year: int,
        *,
        include_tournament: bool = False,
        max_games: int = 0,
        game_delay: float = _DEFAULT_GAME_DELAY,
        scoreboard_delay: float = _DEFAULT_SCOREBOARD_DELAY,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> Dict:
        """Scrape one season of box scores, resuming from any checkpoint.

        Mirrors ``CBBpyPbpScraper.fetch_season_pbp``: iterate dates, discover
        game ids from the scoreboard page, fetch each game, checkpoint every
        ``checkpoint_every`` dates, and resume the day after
        ``metadata.last_completed_date``.

        Args:
            include_tournament: when False (default) the scrape stops strictly
                before ``TOURNAMENT_START_DATES[year]``, keeping the payload
                leakage-safe for pre-tournament feature building.
            max_games: stop after this many games (0 = no cap). For smoke runs.
        """
        from .cbbpy_pbp import CBBpyPbpScraper

        cutoff = self._season_cutoff(year, include_tournament)

        payload = self._load_cache(year) or {
            "season": year,
            "source": "espn_boxscore_html",
            "cutoff_date": cutoff.isoformat(),
            "games": [],
            "metadata": {"last_completed_date": None, "games_missing_boxscore": 0},
        }
        seen = {g["game_id"] for g in payload["games"]}
        resume_after = payload["metadata"].get("last_completed_date")

        dates = list(self._season_dates(year, cutoff))
        if resume_after:
            dates = [d for d in dates if d.isoformat() > resume_after]

        for i, day in enumerate(dates, 1):
            day_str = day.isoformat()
            try:
                game_ids = CBBpyPbpScraper._scrape_game_ids_for_date(day_str)
            except Exception as exc:  # network/parse failure for one date
                logger.warning("scoreboard failed for %s: %s", day_str, exc)
                continue

            import time

            time.sleep(scoreboard_delay)

            for gid in game_ids:
                if gid in seen:
                    continue
                game = self.fetch_game(gid, delay=game_delay)
                if game is None:
                    payload["metadata"]["games_missing_boxscore"] += 1
                    continue
                game["game_date"] = day_str
                payload["games"].append(game)
                seen.add(gid)
                if max_games and len(payload["games"]) >= max_games:
                    payload["metadata"]["last_completed_date"] = day_str
                    self._save_cache(year, payload)
                    return payload

            payload["metadata"]["last_completed_date"] = day_str
            if i % checkpoint_every == 0:
                self._save_cache(year, payload)
                logger.info("Season %d: %d games through %s", year, len(payload["games"]), day_str)

        self._save_cache(year, payload)
        return payload

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _season_cutoff(year: int, include_tournament: bool):
        from datetime import date

        if include_tournament:
            return date(year, 4, 30)
        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
        except ImportError as exc:  # pragma: no cover - configuration error
            raise RuntimeError(
                "Cannot import TOURNAMENT_START_DATES from src.pipeline.config — "
                "refusing to scrape without a pre-tournament bound."
            ) from exc
        cutoff = TOURNAMENT_START_DATES.get(year)
        if cutoff is None:
            raise RuntimeError(f"No TOURNAMENT_START_DATES entry for {year}; refusing to scrape unbounded.")
        return cutoff if hasattr(cutoff, "isoformat") else datetime.fromisoformat(str(cutoff)).date()

    @staticmethod
    def _season_dates(year: int, cutoff) -> Iterable:
        """Every date from the season's November start up to (not incl.) cutoff."""
        from datetime import date, timedelta

        current = date(year - 1, 11, 1)
        while current < cutoff:
            yield current
            current += timedelta(days=1)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
