"""Play-by-play scraper that derives per-game clutch/blown-lead source data.

Scrapes ESPN's own public pages directly rather than going through `cbbpy` or
ESPN's JSON API — see "Why this hits ESPN's HTML pages, not cbbpy or the API"
below. Persists raw play events to ``pbp_{season}.json`` so
``src/data/features/clutch_metrics.py`` can build blown-lead / clutch
team-season features from them.

**Why this hits ESPN's HTML pages, not cbbpy or the API**: both
``site.api.espn.com`` (Akamai, confirmed ``403 Access Denied``) and
``www.espn.com`` requested the way ``cbbpy``'s internal scraper does it
(rotating a hardcoded User-Agent/Referer list) sit behind bot-management that
returns an unpassable JS challenge (AWS WAF, confirmed via
``x-amzn-waf-action: challenge`` on a `202` empty-body response) — neither is
fixable with different headers, and this module doesn't try. What *does*
work, confirmed live: a plain ``requests.get()`` with a bare
``User-Agent: Mozilla/5.0`` and no ``Referer`` against the same public pages
a browser loads —
``www.espn.com/mens-college-basketball/scoreboard/_/date/{YYYYMMDD}/seasontype/2/group/50``
for game discovery, and
``www.espn.com/mens-college-basketball/playbyplay/_/gameId/{game_id}`` for
play-by-play. The playbyplay page embeds ESPN's own hydration JSON
(a ``"plays":[...]`` array) directly in the HTML, which is what's parsed
below — no cbbpy dependency, no JSON API dependency, just this page's own
data. This isn't bot-detection bypass (no JS execution, no challenge
solving, no fingerprint spoofing) — it's a plain HTTP GET on a public page
that happens not to be behind the same wall as the other two paths.

**Confirmed play schema** (verified live, game 401714261, 2025-02-10):
each element of the ``"plays"`` array looks like::

    {
        "id": "401714261101806001",
        "period": {"number": 1},
        "text": "Jahnathan Lamothe missed Three Point Jumper.",
        "homeAway": "away",
        "athlete": {..., "team": "North Carolina A&T Aggies"},
        "scoringPlay": true,          # absent on non-scoring plays
        "type": {"categoryId": "1006", "id": "558", "txt": "JumpShot"},
        "awScr": 0, "hmScr": 0,       # ESPN's own running score -- use this,
                                       # don't reconstruct from play types
        "clock": {"value": 1179, "displayValue": "19:39"},  # secs remaining
                                                              # in the period
    }

``clock.value`` is seconds remaining in the current half/period (1179 secs =
19:39 remaining out of a 20-minute half) — exactly the semantics
``clutch_metrics.py`` already expects, so no change needed there.

**Beyond clutch/blown-lead**: the same per-play payload also carries enough
to close two other documented data gaps without a separate scrape --
``scoringPlay``/``pointsAttempted``/``type.txt``/``text`` let a downstream
consumer derive exact per-game FGM/FGA/3PM/3PA/FTM/FTA (closing the
``three_pt_pct``/``ft_pct`` gap left by Torvik's player CSV not honoring
date filters), and ``athlete``/``homeAway`` let it derive per-game player
minutes bounded to the same pre-tournament window (closing the
``cbbpy_rosters`` full-season-contamination caveat on
``returning_minutes_pct``/``freshman_minutes_pct`` for 2010-2025). Those
aggregators don't exist yet -- this module just stops discarding the fields
they'd need, storing them alongside the clutch-relevant ones below.

**Leakage discipline**: unlike ``torvik.py``'s ratings guard (which must
refuse a live scrape entirely, because Torvik's cumulative stats can't be
un-mixed after the fact), PBP events carry their own game date, so date
filtering can safely happen after the fact — *except* that would still let a
caller accidentally build "pre-tournament" features from data that was
actually scraped past the cutoff. To avoid needing to trust every caller to
filter correctly, ``fetch_season_pbp`` bounds the scrape window itself:
by default it only walks dates from Nov 1 of the prior year through the day
before that season's ``TOURNAMENT_START_DATES`` entry, the same way
``torvik.py._get_pre_tournament_date_range`` does. Pass
``include_tournament=True`` to opt into scraping the full season (e.g. for a
retrospective study of tournament-game clutch performance itself) — that is
a deliberate, separate use case, not the default.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ._retry import rate_limited_call

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0"
_SCOREBOARD_URL = "https://www.espn.com/mens-college-basketball/scoreboard/_/date/{date}/seasontype/2/group/50"
_PLAYBYPLAY_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/{game_id}"

_GAME_ID_RE = re.compile(r'gameId[/"=:]+["\']?(\d{6,})')

# Conservative, "respectful scraper" defaults: a full historical backfill is
# tens of thousands of requests against a public, unauthenticated site with
# no API agreement, so these lean slow rather than fast. Override via
# fetch_season_pbp's kwargs for a quick pilot run only.
_DEFAULT_SCOREBOARD_DELAY = 2.0  # between per-day game-ID lookups
_DEFAULT_PBP_DELAY = 2.0  # between per-game PBP fetches

# Dates between checkpoint writes. The season payload reaches ~2 GB, and each
# checkpoint rewrites the whole document, so writing every date makes I/O
# quadratic in season length. 10 bounds crash loss to ~20-30 min of scraping.
_DEFAULT_CHECKPOINT_EVERY = 10


def _dig(row: Dict, *path: str):
    """Nested dict lookup, e.g. _dig(row, 'period', 'number')."""
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slugify_team_name(name: str) -> str:
    """ "North Carolina A&T Aggies" -> "north_carolina_a_t_aggies".

    Matches the underscore-joined, mascot-suffixed convention cbbpy IDs
    already use elsewhere in this repo (see src/data/normalize.py's
    _CBBPY_EDGE_CASES), so resolve_cbbpy_bridge can bridge these the same way.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return re.sub(r"_+", "_", slug)


def _extract_json_array(html: str, key: str) -> Optional[list]:
    """Extract and parse the JSON array value of `"{key}":[...]` embedded in *html*.

    Scans bracket depth by hand (respecting quoted strings) rather than
    slicing a fixed length, since a play-by-play array can run past 300KB.
    Returns None if the key isn't found or the array doesn't parse.
    """
    marker = f'"{key}":'
    pos = html.find(marker)
    if pos == -1:
        return None
    start = html.find("[", pos)
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    i = start
    n = len(html)
    while i < n:
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
        i += 1
    else:
        return None  # never closed -- malformed/truncated

    snippet = html[start : i + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        logger.warning("Failed to parse extracted '%s' JSON array (len=%d)", key, len(snippet))
        return None


class CBBpyPbpScraper:
    """Fetch and persist season play-by-play from ESPN's own pages, bounded pre-tournament."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_season_pbp(
        self,
        year: int,
        *,
        include_tournament: bool = False,
        max_games: int = 0,
        scoreboard_delay: float = _DEFAULT_SCOREBOARD_DELAY,
        pbp_delay: float = _DEFAULT_PBP_DELAY,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> Dict:
        """Fetch and persist one season's play-by-play.

        Checkpointed periodically: the cache file is rewritten every
        ``checkpoint_every`` dates (and always at season end), and a re-run
        picks up the day after ``metadata.last_completed_date`` rather than
        re-scraping from scratch. A full season is tens of thousands of
        requests at a deliberately conservative pace (see
        ``_DEFAULT_*_DELAY``), so surviving an interruption without losing
        everything isn't optional.

        **Why not checkpoint every date.** The payload is a single JSON
        document that grows to ~2 GB per season under the full play schema,
        and each checkpoint rewrites all of it. Writing after every one of
        ~130 dates costs on the order of 130 GB of I/O per season — the cost
        is quadratic in season length. Batching to every
        ``checkpoint_every`` dates cuts that proportionally while bounding
        crash loss to that many dates (~20-30 minutes of scraping).

        Args:
            year: Tournament year (season ending in this spring).
            include_tournament: If True, scrape through end of season instead
                of stopping the day before the tournament starts. Off by
                default — see module docstring.
            max_games: If > 0, stop after this many games (for a cheap pilot
                run/smoke test rather than a full backfill).
            scoreboard_delay: Seconds to sleep after each day's game-ID
                lookup.
            pbp_delay: Seconds to sleep after each per-game PBP fetch
                (passed to rate_limited_call, which also sleeps this long
                before each retry).
        """
        cache_name = f"pbp_{year}.json"
        start, end, cutoff_str = self._date_window(year, include_tournament=include_tournament)

        cached = self._load_cache(cache_name)
        games: List[Dict] = []
        seen_game_ids: set = set()
        resume_from = start

        if cached and isinstance(cached.get("games"), list):
            if cached.get("metadata", {}).get("complete"):
                return cached
            last_completed = cached.get("metadata", {}).get("last_completed_date")
            if last_completed:
                try:
                    resume_from = date.fromisoformat(last_completed) + timedelta(days=1)
                except ValueError:
                    resume_from = start
            games = list(cached["games"])
            seen_game_ids = {g["game_id"] for g in games if g.get("game_id")}
            logger.info(
                "Resuming %d PBP fetch from %s (%d games already cached)",
                year,
                resume_from.isoformat(),
                len(games),
            )

        if resume_from > end:
            return self._finalize(cache_name, year, games, cutoff_str, start, end, include_tournament)

        days_done = 0
        for day in self._iter_dates(resume_from, end):
            if max_games > 0 and len(seen_game_ids) >= max_games:
                break
            try:
                game_ids = self._scrape_game_ids_for_date(day.isoformat())
            except Exception as e:
                logger.debug("Game-ID lookup failed for %s: %s", day.isoformat(), e)
                game_ids = []
            time.sleep(scoreboard_delay)

            for game_id in game_ids:
                gid = str(game_id).strip()
                if not gid or gid in seen_game_ids:
                    continue
                seen_game_ids.add(gid)
                if max_games > 0 and len(seen_game_ids) > max_games:
                    break

                raw_plays = self._fetch_game_pbp(gid, delay=pbp_delay)
                if raw_plays:
                    games.append(self._build_game_payload(gid, day.isoformat(), raw_plays))

            # Periodic checkpoint (see docstring): bounded crash loss without
            # rewriting a multi-GB document on every single date.
            days_done += 1
            if checkpoint_every > 0 and days_done % checkpoint_every == 0:
                self._finalize(
                    cache_name,
                    year,
                    games,
                    cutoff_str,
                    start,
                    end,
                    include_tournament,
                    last_completed_date=day,
                    complete=False,
                )

        return self._finalize(
            cache_name,
            year,
            games,
            cutoff_str,
            start,
            end,
            include_tournament,
            last_completed_date=end,
            complete=True,
        )

    def _finalize(
        self,
        cache_name: str,
        year: int,
        games: List[Dict],
        cutoff_str: Optional[str],
        start: date,
        end: date,
        include_tournament: bool,
        *,
        last_completed_date: Optional[date] = None,
        complete: bool = True,
        write: bool = True,
    ) -> Dict:
        payload = {
            "season": year,
            "source": "espn_playbyplay_html",
            "cutoff_date": cutoff_str,
            "games": games,
        }
        if complete:
            try:
                from .schemas import validate_pbp_payload

                payload = validate_pbp_payload(payload)
            except Exception as e:
                logger.warning("PBP payload schema validation failed: %s", e)

        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload["metadata"] = {
            "raw_game_count": len(games),
            "date_window": [start.isoformat(), end.isoformat()],
            "include_tournament": include_tournament,
            "complete": complete,
            "last_completed_date": (last_completed_date or start).isoformat(),
        }
        if write:
            self._save_cache(cache_name, payload)
        return payload

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _fetch_game_pbp(self, game_id: str, delay: float = _DEFAULT_PBP_DELAY) -> List[Dict]:
        import requests

        url = _PLAYBYPLAY_URL.format(game_id=game_id)
        try:
            resp = rate_limited_call(
                requests.get,
                url,
                headers={"User-Agent": _UA},
                timeout=15,
                delay=delay,
                max_retries=2,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug("PBP fetch failed for game %s: %s", game_id, e)
            return []

        plays = _extract_json_array(resp.text, "plays")
        if not plays:
            logger.debug("No 'plays' array found for game %s", game_id)
            return []
        return [p for p in plays if isinstance(p, dict)]

    def _build_game_payload(self, game_id: str, game_date: str, raw_plays: List[Dict]) -> Dict:
        home_raw = away_raw = None
        for row in raw_plays:
            side = row.get("homeAway")
            team = _dig(row, "athlete", "team") or row.get("team")
            if not team:
                continue
            if side == "home" and home_raw is None:
                home_raw = _slugify_team_name(team)
            elif side == "away" and away_raw is None:
                away_raw = _slugify_team_name(team)
            if home_raw and away_raw:
                break

        normalized_plays = [self._normalize_play_row(game_id, row) for row in raw_plays]
        normalized_plays = [p for p in normalized_plays if p is not None]

        return {
            "game_id": game_id,
            "game_date": game_date,
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "plays": normalized_plays,
        }

    @staticmethod
    def _normalize_play_row(game_id: str, row: Dict) -> Optional[Dict]:
        """Map ESPN's confirmed play schema onto the canonical PbpEventSchema shape.

        Returns None for rows missing a required field (period, clock, or
        either score) rather than guessing — those rows can't contribute to a
        margin trajectory anyway (e.g. header/filler rows with no state).

        Keeps considerably more than clutch_metrics.py currently reads
        (scoringPlay/pointsAttempted/type/text/athlete/win probability) so a
        future box-score-from-PBP or player-minutes-from-PBP aggregator can
        be built without a second scrape — see module docstring.
        """
        period = _dig(row, "period", "number")
        seconds_remaining = _dig(row, "clock", "value")
        home_score = row.get("hmScr")
        away_score = row.get("awScr")
        if None in (period, seconds_remaining, home_score, away_score):
            return None

        try:
            normalized = {
                "game_id": game_id,
                "period": int(period),
                "seconds_remaining": float(seconds_remaining),
                "home_score": int(home_score),
                "away_score": int(away_score),
            }
        except (TypeError, ValueError):
            return None

        normalized["home_away"] = row.get("homeAway")
        normalized["scoring_play"] = bool(row.get("scoringPlay", False))
        normalized["shooting_play"] = bool(row.get("shootingPlay", False))
        normalized["points_attempted"] = row.get("pointsAttempted")
        normalized["play_type"] = _dig(row, "type", "txt")
        normalized["play_type_category_id"] = _dig(row, "type", "categoryId")
        normalized["text"] = row.get("text") or row.get("title")

        athlete_team = _dig(row, "athlete", "team")
        normalized["athlete_id"] = _dig(row, "athlete", "id")
        normalized["athlete_name"] = _dig(row, "athlete", "name")
        normalized["athlete_team"] = _slugify_team_name(athlete_team) if athlete_team else None

        win_prob_raw = _dig(row, "favoredTeam", "winProbability")
        normalized["win_probability"] = _safe_float(win_prob_raw)
        normalized["favored_is_away"] = _dig(row, "favoredTeam", "isAway")

        normalized["coordinate_x"] = _dig(row, "coordinate", "x")
        normalized["coordinate_y"] = _dig(row, "coordinate", "y")

        return normalized

    # ------------------------------------------------------------------
    # Date window (leakage-safe by construction)
    # ------------------------------------------------------------------

    def _date_window(self, year: int, *, include_tournament: bool) -> Tuple[date, date, Optional[str]]:
        start = date(year - 1, 11, 1)
        today = datetime.now(timezone.utc).date()

        if include_tournament:
            end = min(date(year, 5, 1), today)
            return start, end, None

        try:
            from ...pipeline.config import TOURNAMENT_START_DATES
        except ImportError:
            raise ValueError(
                "Cannot import TOURNAMENT_START_DATES from src.pipeline.config — "
                "refusing to guess a pre-tournament cutoff."
            )

        cutoff = TOURNAMENT_START_DATES.get(year)
        if cutoff is None:
            if year <= today.year:
                raise ValueError(
                    f"TOURNAMENT_START_DATES missing entry for {year}; add it to "
                    f"src/pipeline/config.py. Scraping an unbounded window would "
                    f"risk including tournament games in pre-tournament features — "
                    f"the same leakage class torvik.py's guard exists to prevent."
                )
            # Future year, no cutoff yet -- nothing pre-tournament to bound to.
            return start, min(start, today), None

        end = min(cutoff - timedelta(days=1), today)
        return start, end, cutoff.isoformat()

    @staticmethod
    def _iter_dates(start: date, end: date) -> Iterable[date]:
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _scrape_game_ids_for_date(day_str: str, http_timeout: int = 15) -> List[str]:
        """Scrape game IDs from ESPN's public scoreboard HTML page.

        Confirmed live: plain requests.get with a bare 'Mozilla/5.0'
        User-Agent and no Referer returns 200 here, unlike the JSON API
        (Akamai 403) or cbbpy's own request pattern (AWS WAF challenge) --
        see module docstring.
        """
        import requests

        d = day_str.replace("-", "")
        url = _SCOREBOARD_URL.format(date=d)
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=http_timeout)
        resp.raise_for_status()
        return sorted(set(_GAME_ID_RE.findall(resp.text)))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _load_cache(self, filename: str) -> Optional[Dict]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / filename
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, filename: str, payload: Dict) -> None:
        """Write the checkpoint atomically.

        This file is rewritten after every scraped date, and a season takes
        hours. A plain in-place ``json.dump`` leaves a window where the file
        on disk is truncated: a concurrent reader sees invalid JSON, and —
        much worse — a crash or kill mid-write corrupts the checkpoint, which
        ``_load_cache`` then discards, restarting the whole season from
        scratch. Writing to a temp file in the same directory and calling
        ``os.replace`` makes the swap atomic, so readers and restarts always
        see either the previous checkpoint or the new one.
        """
        if not self.cache_dir:
            return
        path = self.cache_dir / filename
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
