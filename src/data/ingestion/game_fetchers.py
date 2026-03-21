"""Clean fetcher classes for historical and incremental game data.

Two public classes:

``HistoricalGameFetcher``
    Batch multi-season fetcher.  Produces ``List[IngestionGameRecord]`` for a
    complete NCAA season (Nov – Apr).

    Provider priority (hard-coded, no configurable overrides):
    1. ESPN scoreboard API — per-day HTTP calls; real dates + box scores
    2. sportsdataverse     — batch season load; real dates + box scores
    3. cbbpy               — legacy fallback; dates may require repair

``IncrementalGameFetcher``
    Single-year since-date fetcher for live-season top-up pulls.  Uses ESPN
    scoreboard (day-by-day from ``since`` to today) so new games have correct
    dates immediately.

Both classes share ESPN parsing helpers and a common deduplication routine
that logs game_id collisions across providers (indicating inconsistency rather
than silently discarding data).
"""

from __future__ import annotations

import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

from ..normalize import normalize_team_id as _normalize
from .schemas import IngestionGameRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sanity floor: a full NCAA regular season has ~5 000+ team-game rows
# (≈ 350 games × ≈ 30 teams / 2).  We require at least this many *game*
# records before trusting a source as complete.
_MIN_GAMES_FOR_COMPLETE_SEASON = 500

# Cache version — bump to force re-fetch when the fetch logic changes
_CACHE_VERSION = "v2_espn_primary"

_ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


# ---------------------------------------------------------------------------
# Season window helpers
# ---------------------------------------------------------------------------

def season_window(season: int):
    """Return (start_date, end_date) for an NCAA season."""
    return date(season - 1, 11, 1), date(season, 4, 15)


def is_in_season_window(date_str: str, season: int) -> bool:
    """Return True if *date_str* (YYYY-MM-DD) falls within the season window."""
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    start, end = season_window(season)
    return start <= d <= end


# ---------------------------------------------------------------------------
# ESPN parsing helpers (shared by both fetchers)
# ---------------------------------------------------------------------------

def _parse_espn_stats(statistics: List[Dict]) -> Dict[str, float]:
    """Parse an ESPN competitor ``statistics`` array to a flat dict.

    Handles both ``"made-attempted"`` display values (FG, 3PT, FT) and plain
    integer values (rebounds, assists, etc.).
    """
    result: Dict[str, float] = {}
    for stat in statistics:
        name = stat.get("name", "")
        abbrev = stat.get("abbreviation", "")
        display = str(stat.get("displayValue") or "").strip()
        if not display:
            continue

        if "-" in display:
            parts = display.split("-", 1)
            try:
                made, attempted = float(parts[0]), float(parts[1])
            except (ValueError, IndexError):
                made = attempted = 0.0
        else:
            try:
                made = float(display)
            except (ValueError, TypeError):
                made = 0.0
            attempted = 0.0

        if name == "fieldGoalsMade-fieldGoalsAttempted" or abbrev == "FG":
            result["fgm"] = made
            result["fga"] = attempted
        elif name == "threePointFieldGoalsMade-threePointFieldGoalsAttempted" or abbrev == "3PT":
            result["fg3m"] = made
            result["fg3a"] = attempted
        elif name == "freeThrowsMade-freeThrowsAttempted" or abbrev == "FT":
            result["ftm"] = made
            result["fta"] = attempted
        elif name == "offensiveRebounds" or abbrev == "OREB":
            result["orb"] = made
        elif name == "defensiveRebounds" or abbrev == "DREB":
            result["drb"] = made
        elif name == "assists" or abbrev == "AST":
            result["ast"] = made
        elif name == "steals" or abbrev == "STL":
            result["stl"] = made
        elif name == "blocks" or abbrev == "BLK":
            result["blk"] = made
        elif name == "turnovers" or abbrev == "TO":
            result["tov"] = made
        elif name == "fouls" or abbrev == "PF":
            result["pf"] = made
    return result


def parse_espn_event(event: Dict, season: int, fallback_date: str) -> Optional[IngestionGameRecord]:
    """Parse a single ESPN scoreboard event into an ``IngestionGameRecord``.

    Returns *None* if the event is malformed or lacks enough data to produce a
    useful record (e.g. future/cancelled game with no score).
    """
    game_id = str(event.get("id", "")).strip()
    if not game_id:
        return None

    competitions = event.get("competitions", [])
    if not competitions:
        return None
    comp = competitions[0]

    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return None

    # Game date: prefer the event-level ISO timestamp, fall back to caller-supplied
    raw_date = str(event.get("date", "")).strip()
    game_date = raw_date[:10] if len(raw_date) >= 10 else fallback_date

    # Identify home / away explicitly; fall back to index order
    home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

    def _team_id(comp_: Dict) -> str:
        t = comp_.get("team", {})
        name = t.get("displayName") or t.get("shortDisplayName") or ""
        return _normalize(name)

    def _team_name(comp_: Dict) -> str:
        t = comp_.get("team", {})
        return t.get("displayName") or t.get("shortDisplayName") or ""

    home_id = _team_id(home_comp)
    away_id = _team_id(away_comp)
    if not home_id or not away_id:
        return None

    try:
        home_score = int(float(home_comp.get("score") or 0))
        away_score = int(float(away_comp.get("score") or 0))
    except (TypeError, ValueError):
        return None

    # Skip games with no score (upcoming / cancelled)
    if home_score == 0 and away_score == 0:
        return None

    home_stats = _parse_espn_stats(home_comp.get("statistics", []))
    away_stats = _parse_espn_stats(away_comp.get("statistics", []))

    neutral = bool(comp.get("neutralSite", False))
    conference = comp.get("conferenceCompetition")
    status_name = (
        comp.get("status", {}).get("type", {}).get("name", "")
        if isinstance(comp.get("status"), dict) else ""
    )
    overtime = "ot" in status_name.lower() if status_name else False

    return IngestionGameRecord(
        game_id=game_id,
        date=game_date,
        season=season,
        home_team_id=home_id,
        home_team_name=_team_name(home_comp),
        away_team_id=away_id,
        away_team_name=_team_name(away_comp),
        home_score=home_score,
        away_score=away_score,
        home_fgm=home_stats.get("fgm", 0.0),
        home_fga=home_stats.get("fga", 0.0),
        home_fg3m=home_stats.get("fg3m", 0.0),
        home_fg3a=home_stats.get("fg3a", 0.0),
        home_ftm=home_stats.get("ftm", 0.0),
        home_fta=home_stats.get("fta", 0.0),
        home_orb=home_stats.get("orb", 0.0),
        home_drb=home_stats.get("drb", 0.0),
        home_ast=home_stats.get("ast", 0.0),
        home_stl=home_stats.get("stl", 0.0),
        home_blk=home_stats.get("blk", 0.0),
        home_tov=home_stats.get("tov", 0.0),
        home_pf=home_stats.get("pf", 0.0),
        away_fgm=away_stats.get("fgm", 0.0),
        away_fga=away_stats.get("fga", 0.0),
        away_fg3m=away_stats.get("fg3m", 0.0),
        away_fg3a=away_stats.get("fg3a", 0.0),
        away_ftm=away_stats.get("ftm", 0.0),
        away_fta=away_stats.get("fta", 0.0),
        away_orb=away_stats.get("orb", 0.0),
        away_drb=away_stats.get("drb", 0.0),
        away_ast=away_stats.get("ast", 0.0),
        away_stl=away_stats.get("stl", 0.0),
        away_blk=away_stats.get("blk", 0.0),
        away_tov=away_stats.get("tov", 0.0),
        away_pf=away_stats.get("pf", 0.0),
        overtime=overtime,
        neutral_site=neutral,
        conference_game=bool(conference) if conference is not None else None,
        provider="espn_scoreboard",
    )


def _team_perspective_rows_to_records(
    rows: List[Dict],
    season: int,
    provider: str,
) -> List[IngestionGameRecord]:
    """Convert a list of team-perspective dicts (one per team per game) to
    ``IngestionGameRecord`` objects.

    Pairs rows by ``game_id``.  Home/away assignment is alphabetical by
    ``team_id`` (arbitrary but deterministic); all such games are treated as
    neutral-site since provider data rarely carries venue information.
    """
    by_game: Dict[str, List[Dict]] = {}
    for row in rows:
        gid = str(row.get("game_id", "")).strip()
        if gid:
            by_game.setdefault(gid, []).append(row)

    records: List[IngestionGameRecord] = []
    for gid, game_rows in by_game.items():
        if len(game_rows) < 2:
            continue
        # Stable alphabetical ordering → deterministic home/away
        game_rows = sorted(game_rows, key=lambda r: str(r.get("team_id", "")))
        t1, t2 = game_rows[0], game_rows[1]

        raw_date = t1.get("date") or t2.get("date") or ""
        try:
            home_score = int(float(t1.get("team_score", 0) or 0))
            away_score = int(float(t2.get("team_score", 0) or 0))
        except (TypeError, ValueError):
            continue
        if home_score == 0 and away_score == 0:
            continue

        def _f(row: Dict, key: str) -> float:
            return float(row.get(key) or 0.0)

        records.append(IngestionGameRecord(
            game_id=gid,
            date=raw_date,
            season=season,
            home_team_id=str(t1.get("team_id", "")),
            home_team_name=str(t1.get("team_name", t1.get("team_id", ""))),
            away_team_id=str(t2.get("team_id", "")),
            away_team_name=str(t2.get("team_name", t2.get("team_id", ""))),
            home_score=home_score,
            away_score=away_score,
            home_fgm=_f(t1, "fgm"),
            home_fga=_f(t1, "fga"),
            home_fg3m=_f(t1, "fg3m"),
            home_fg3a=_f(t1, "fg3a"),
            home_fta=_f(t1, "fta"),
            home_tov=_f(t1, "turnovers") or _f(t1, "tov"),
            home_orb=_f(t1, "orb"),
            home_drb=_f(t1, "drb"),
            away_fgm=_f(t2, "fgm"),
            away_fga=_f(t2, "fga"),
            away_fg3m=_f(t2, "fg3m"),
            away_fg3a=_f(t2, "fg3a"),
            away_fta=_f(t2, "fta"),
            away_tov=_f(t2, "turnovers") or _f(t2, "tov"),
            away_orb=_f(t2, "orb"),
            away_drb=_f(t2, "drb"),
            neutral_site=True,
            provider=provider,
        ))
    return records


def dedup_records(records: List[IngestionGameRecord]) -> List[IngestionGameRecord]:
    """Deduplicate by ``game_id``.

    When multiple providers return the same game, keep the record with the
    most box-score data (highest FGA as a proxy).  Log a WARNING when two
    records for the same game have different dates — this indicates a
    provider inconsistency that should be investigated.
    """
    seen: Dict[str, IngestionGameRecord] = {}
    for rec in records:
        existing = seen.get(rec.game_id)
        if existing is None:
            seen[rec.game_id] = rec
            continue

        if existing.date != rec.date:
            logger.warning(
                "Date inconsistency for game_id=%s: provider '%s' says %s, "
                "provider '%s' says %s — keeping the ESPN date",
                rec.game_id, existing.provider, existing.date,
                rec.provider, rec.date,
            )

        # Prefer the record with more box-score content (higher total FGA)
        if (rec.home_fga + rec.away_fga) > (existing.home_fga + existing.away_fga):
            logger.debug(
                "Dedup game_id=%s: replacing '%s' record with richer '%s' record",
                rec.game_id, existing.provider, rec.provider,
            )
            seen[rec.game_id] = rec
        else:
            logger.debug(
                "Dedup game_id=%s: discarding duplicate from '%s' (kept '%s')",
                rec.game_id, rec.provider, existing.provider,
            )

    result = sorted(seen.values(), key=lambda r: (r.date, r.game_id))
    return result


# ---------------------------------------------------------------------------
# HistoricalGameFetcher
# ---------------------------------------------------------------------------

class HistoricalGameFetcher:
    """Batch multi-season game fetcher.

    Provider cascade:
    1. **ESPN scoreboard API** (primary) — day-by-day HTTP calls, real dates
       and box scores directly from ESPN.  Concurrent (8 workers) to keep
       a full season fetch under ~5 minutes.
    2. **sportsdataverse** (secondary) — single batch load of the PBP dataset
       that aggregates ESPN data.  Falls back to this when ESPN returns fewer
       than ``_MIN_GAMES_FOR_COMPLETE_SEASON`` game records.
    3. **cbbpy** (tertiary) — legacy fallback retained for resilience; dates
       may need repair.

    Results are cached in ``{cache_dir}/espn_historical_{season}.json`` with a
    version tag so stale caches are automatically invalidated when the fetch
    logic changes.
    """

    def __init__(
        self,
        cache_dir: str = "data/raw/cache",
        http_timeout: int = 30,
        max_workers: int = 8,
    ):
        self.cache_dir = Path(cache_dir)
        self.http_timeout = http_timeout
        self.max_workers = max_workers
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_season(self, season: int) -> List[IngestionGameRecord]:
        """Fetch all games for a complete NCAA season.

        Returns a deduplicated list sorted by date then game_id.
        """
        cache_path = self.cache_dir / f"espn_historical_{season}.json"

        cached = self._load_cache(cache_path, season)
        if cached is not None:
            logger.info("Season %d: loaded %d records from ESPN cache", season, len(cached))
            return cached

        # ── Provider 1: ESPN scoreboard ─────────────────────────────────────
        espn_records = self._fetch_via_espn_scoreboard(season)
        logger.info("Season %d: ESPN scoreboard returned %d games", season, len(espn_records))

        if len(espn_records) >= _MIN_GAMES_FOR_COMPLETE_SEASON:
            final = dedup_records(espn_records)
            self._save_cache(cache_path, season, final)
            return final

        # ── Provider 2: sportsdataverse ─────────────────────────────────────
        logger.warning(
            "Season %d: ESPN scoreboard returned only %d games (need %d); "
            "trying sportsdataverse",
            season, len(espn_records), _MIN_GAMES_FOR_COMPLETE_SEASON,
        )
        sdv_records = self._fetch_via_sportsdataverse(season)
        logger.info("Season %d: sportsdataverse returned %d games", season, len(sdv_records))

        merged = dedup_records(espn_records + sdv_records)
        if len(merged) >= _MIN_GAMES_FOR_COMPLETE_SEASON:
            self._save_cache(cache_path, season, merged)
            return merged

        # ── Provider 3: cbbpy ───────────────────────────────────────────────
        logger.warning(
            "Season %d: ESPN + sportsdataverse gave only %d games; trying cbbpy fallback",
            season, len(merged),
        )
        cbbpy_records = self._fetch_via_cbbpy(season)
        logger.info("Season %d: cbbpy returned %d games", season, len(cbbpy_records))

        all_records = dedup_records(merged + cbbpy_records)
        if all_records:
            self._save_cache(cache_path, season, all_records)
        return all_records

    # ── Provider implementations ───────────────────────────────────────────

    def _fetch_via_espn_scoreboard(self, season: int) -> List[IngestionGameRecord]:
        """Day-by-day ESPN scoreboard fetch for *season*."""
        start, end = season_window(season)
        today = datetime.now(timezone.utc).date()
        end = min(end, today)

        days = []
        current = start
        while current <= end:
            days.append(current)
            current += timedelta(days=1)

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })

        all_records: List[IngestionGameRecord] = []
        errors = 0

        def _fetch_day(d: date) -> List[IngestionGameRecord]:
            return self._fetch_day(d, season, session)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch_day, d): d for d in days}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    day_records = future.result()
                    all_records.extend(day_records)
                except Exception as exc:
                    errors += 1
                    logger.debug("ESPN day-fetch error: %s", exc)
                if done % 50 == 0 or done == len(days):
                    logger.info(
                        "Season %d ESPN: %d/%d days processed, %d games so far",
                        season, done, len(days), len(all_records),
                    )

        session.close()
        if errors:
            logger.warning("Season %d ESPN: %d day-fetches failed", season, errors)

        return all_records

    def _fetch_day(
        self,
        day: date,
        season: int,
        session: Optional[requests.Session] = None,
    ) -> List[IngestionGameRecord]:
        """Fetch all games on a single *day* for *season* from ESPN scoreboard."""
        date_str = day.strftime("%Y%m%d")
        fallback_iso = day.isoformat()
        getter = session or requests
        records: List[IngestionGameRecord] = []

        for season_type in (2, 3):  # 2=regular+conf-tourney, 3=postseason/NCAA
            params = {
                "dates": date_str,
                "seasontype": season_type,
                "limit": 200,
                "groups": 50,
            }
            try:
                resp = getter.get(_ESPN_SCOREBOARD_URL, params=params, timeout=self.http_timeout)
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError, KeyError) as exc:
                logger.debug(
                    "ESPN scoreboard %s seasontype=%d failed: %s",
                    fallback_iso, season_type, exc,
                )
                continue

            for event in data.get("events", []):
                rec = parse_espn_event(event, season, fallback_iso)
                if rec is not None:
                    # Validate date is within season window
                    if not is_in_season_window(rec.date, season):
                        logger.debug(
                            "Rejected game_id=%s: date %s outside season %d window",
                            rec.game_id, rec.date, season,
                        )
                        continue
                    records.append(rec)

        # Dedup within this day (same game returned by seasontype 2 and 3)
        seen: Set[str] = set()
        deduped: List[IngestionGameRecord] = []
        for r in records:
            if r.game_id not in seen:
                seen.add(r.game_id)
                deduped.append(r)
        return deduped

    def _fetch_via_sportsdataverse(self, season: int) -> List[IngestionGameRecord]:
        """Load games from sportsdataverse (ESPN PBP aggregation)."""
        try:
            mbb = importlib.import_module("sportsdataverse.mbb")
        except (ImportError, ModuleNotFoundError):
            logger.debug("sportsdataverse not installed; skipping")
            return []

        call_candidates = [
            ("load_mbb_pbp", {"seasons": [season]}),
            ("load_mbb_pbp", {"seasons": season}),
            ("load_mbb_pbp", {"year": season}),
        ]
        for fn_name, kwargs in call_candidates:
            fn = getattr(mbb, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                rows = self._frame_to_records(df)
                if rows:
                    logger.info(
                        "Season %d: sportsdataverse %s returned %d rows",
                        season, fn_name, len(rows),
                    )
                    self._normalize_date_field(rows)
                    return _team_perspective_rows_to_records(rows, season, "sportsdataverse")
            except Exception as exc:
                logger.debug("sportsdataverse %s failed: %s", fn_name, exc)
                continue
        return []

    def _fetch_via_cbbpy(self, season: int) -> List[IngestionGameRecord]:
        """Fetch games via cbbpy (legacy fallback)."""
        try:
            scraper = importlib.import_module("cbbpy.mens_scraper")
        except (ImportError, ModuleNotFoundError):
            logger.debug("cbbpy not installed; skipping")
            return []

        # Try fast path first (full-season bulk call)
        for fn_name in ("get_games_season", "get_games_range"):
            fn = getattr(scraper, fn_name, None)
            if fn is None:
                continue
            try:
                if fn_name == "get_games_season":
                    result = self._run_with_timeout(
                        fn,
                        args=(season,),
                        kwargs={"info": True, "box": True, "pbp": False},
                        timeout=600,
                    )
                else:
                    result = self._run_with_timeout(
                        fn,
                        args=(f"{season - 1}-11-01", f"{season}-04-15"),
                        kwargs={"info": True, "box": True, "pbp": False},
                        timeout=600,
                    )
                rows = self._normalize_cbbpy_result(result, season)
                if rows:
                    return _team_perspective_rows_to_records(rows, season, "cbbpy")
            except Exception as exc:
                logger.debug("cbbpy %s failed: %s", fn_name, exc)
                continue
        return []

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _load_cache(self, path: Path, season: int) -> Optional[List[IngestionGameRecord]]:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("version") != _CACHE_VERSION:
            logger.info("Season %d: cache version mismatch — re-fetching", season)
            return None
        if not data.get("complete"):
            return None
        raw_records = data.get("records", [])
        if not raw_records:
            return None
        try:
            return [IngestionGameRecord(**r) for r in raw_records]
        except (TypeError, KeyError) as exc:
            logger.warning("Season %d: cache deserialization failed: %s", season, exc)
            return None

    def _save_cache(
        self,
        path: Path,
        season: int,
        records: List[IngestionGameRecord],
    ) -> None:
        data = {
            "version": _CACHE_VERSION,
            "season": season,
            "complete": True,
            "record_count": len(records),
            "records": [asdict(r) for r in records],
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                "Season %d: wrote %d records to ESPN cache %s",
                season, len(records), path,
            )
        except OSError as exc:
            logger.warning("Season %d: failed to write cache: %s", season, exc)

    # ── Static / utility helpers ───────────────────────────────────────────

    @staticmethod
    def _frame_to_records(obj) -> List[Dict]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            return [obj]
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                recs = to_dict("records")
                if isinstance(recs, list):
                    return [r for r in recs if isinstance(r, dict)]
            except Exception:
                pass
        return []

    @staticmethod
    def _normalize_date_field(records: List[Dict]) -> None:
        """Normalise ``date`` fields in-place to YYYY-MM-DD."""
        _DATE_KEYS = ("date", "game_date", "start_date", "game_day", "gamedate")
        for rec in records:
            raw = ""
            for key in _DATE_KEYS:
                val = rec.get(key)
                if val and str(val).strip():
                    raw = str(val).strip()
                    break
            if not raw:
                rec["date"] = ""
                continue
            if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
                rec["date"] = raw[:10]
                continue
            try:
                epoch_val = float(raw)
                if epoch_val > 1e12:
                    epoch_val /= 1000.0
                if 9e8 < epoch_val < 3e9:
                    rec["date"] = datetime.utcfromtimestamp(epoch_val).strftime("%Y-%m-%d")
                    continue
            except (ValueError, TypeError, OverflowError, OSError):
                pass
            for fmt in ("%B %d, %Y", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y%m%d"):
                try:
                    rec["date"] = datetime.strptime(raw[:20], fmt).strftime("%Y-%m-%d")
                    break
                except (ValueError, TypeError):
                    continue
            else:
                rec.setdefault("date", "")

    @staticmethod
    def _normalize_cbbpy_result(obj, season: int) -> List[Dict]:
        """Normalize the (info, box, pbp) tuple or plain DataFrame from cbbpy."""
        if not isinstance(obj, tuple):
            rows = HistoricalGameFetcher._frame_to_records(obj)
            HistoricalGameFetcher._normalize_date_field(rows)
            return rows

        info_df = obj[0] if len(obj) > 0 else None
        box_df = obj[1] if len(obj) > 1 else None

        # Build game_id → date mapping from info DataFrame
        game_date_map: Dict[str, str] = {}
        if info_df is not None and hasattr(info_df, "iterrows") and not getattr(info_df, "empty", True):
            for _, row in info_df.iterrows():
                gid = str(row.get("game_id", "")).strip()
                raw_day = str(row.get("game_day", "")).strip()
                if gid and raw_day:
                    try:
                        game_date_map[gid] = datetime.strptime(raw_day, "%B %d, %Y").strftime("%Y-%m-%d")
                    except ValueError:
                        pass

        rows = HistoricalGameFetcher._frame_to_records(box_df)
        HistoricalGameFetcher._normalize_date_field(rows)

        # Aggregate player box rows into team-level rows
        aggregated = HistoricalGameFetcher._aggregate_box_rows(rows)
        for row in aggregated:
            gid = str(row.get("game_id", "")).strip()
            if gid in game_date_map:
                row["date"] = game_date_map[gid]
            row.setdefault("season", season)
        return aggregated

    @staticmethod
    def _aggregate_box_rows(rows: List[Dict]) -> List[Dict]:
        """Aggregate per-player cbbpy box rows into per-team rows."""
        by_game_team: Dict[str, Dict[str, Dict]] = {}
        for row in rows:
            game_id = str(row.get("game_id") or row.get("id") or "").strip()
            team_name = str(row.get("team") or row.get("team_name") or "").strip()
            player_name = str(row.get("player") or "").strip().lower()
            if not game_id or not team_name:
                continue
            if player_name in {"team", "totals", "team totals"}:
                continue
            team_id = _normalize(team_name)
            game_bucket = by_game_team.setdefault(game_id, {})
            stats = game_bucket.setdefault(team_id, {
                "team_id": team_id, "team_name": team_name,
                "team_score": 0.0, "fgm": 0.0, "fga": 0.0,
                "fg3m": 0.0, "fg3a": 0.0, "fta": 0.0,
                "turnovers": 0.0, "orb": 0.0, "drb": 0.0,
                "player_rows": 0, "date": "",
            })
            def _f(key: str) -> float:
                v = row.get(key)
                try:
                    return float(v) if v is not None else 0.0
                except (TypeError, ValueError):
                    return 0.0
            stats["team_score"] += _f("pts")
            stats["fgm"] += _f("fgm")
            stats["fga"] += _f("fga")
            stats["fg3m"] += _f("3pm")
            stats["fg3a"] += _f("3pa")
            stats["fta"] += _f("fta")
            stats["turnovers"] += _f("to")
            stats["orb"] += _f("oreb")
            stats["drb"] += _f("dreb")
            stats["player_rows"] += 1
            if not stats["date"]:
                stats["date"] = str(row.get("date") or row.get("game_day") or "").strip()

        out: List[Dict] = []
        for game_id, teams in by_game_team.items():
            team_list = sorted(teams.values(), key=lambda t: (-t["player_rows"], t["team_name"]))
            if len(team_list) < 2:
                continue
            first, second = team_list[:2]
            for team, opp in ((first, second), (second, first)):
                poss = team["fga"] - team["orb"] + team["turnovers"] + 0.475 * team["fta"]
                out.append({
                    "game_id": game_id,
                    "team_id": team["team_id"],
                    "team_name": team["team_name"],
                    "opponent_id": opp["team_id"],
                    "opponent_name": opp["team_name"],
                    "team_score": int(round(team["team_score"])),
                    "opponent_score": int(round(opp["team_score"])),
                    "possessions": max(float(poss), 0.0),
                    "fgm": team["fgm"], "fga": team["fga"],
                    "fg3m": team["fg3m"], "fg3a": team["fg3a"],
                    "fta": team["fta"], "turnovers": team["turnovers"],
                    "orb": team["orb"], "drb": team["drb"],
                    "date": team.get("date") or "",
                })
        return out

    @staticmethod
    def _run_with_timeout(fn, args=(), kwargs=None, timeout=120):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FT
        kwargs = kwargs or {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FT:
                raise TimeoutError(
                    f"{getattr(fn, '__name__', fn)} timed out after {timeout}s"
                )


# ---------------------------------------------------------------------------
# IncrementalGameFetcher
# ---------------------------------------------------------------------------

class IncrementalGameFetcher:
    """Fetches new games since a given date for live-season top-up pulls.

    Uses the ESPN scoreboard API day-by-day from ``since`` to today.  Falls
    back to sportsdataverse if ESPN returns nothing for a given date.

    Intended usage::

        fetcher = IncrementalGameFetcher()
        new_records = fetcher.fetch_since(year=2026, since="2026-03-08")
        # new_records: List[IngestionGameRecord]
    """

    def __init__(
        self,
        http_timeout: int = 30,
        max_workers: int = 4,
    ):
        self.http_timeout = http_timeout
        self.max_workers = max_workers
        # Share a single HistoricalGameFetcher for parsing helpers
        self._hist = HistoricalGameFetcher(http_timeout=http_timeout, max_workers=max_workers)

    def fetch_since(self, year: int, since: str) -> List[IngestionGameRecord]:
        """Fetch all games from *since* (inclusive) to today for *year*.

        Args:
            year:  NCAA season (e.g. 2026).
            since: ISO date string YYYY-MM-DD.  Fetch games from this date on.

        Returns:
            Deduplicated ``IngestionGameRecord`` list sorted by date.
        """
        try:
            since_date = datetime.strptime(since[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.error("IncrementalGameFetcher: invalid since date '%s'", since)
            return []

        today = datetime.now(timezone.utc).date()
        _, season_end = season_window(year)
        end_date = min(today, season_end)

        if since_date > end_date:
            logger.info("IncrementalGameFetcher: no days to fetch for year %d since %s", year, since)
            return []

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })

        days: List[date] = []
        current = since_date
        while current <= end_date:
            days.append(current)
            current += timedelta(days=1)

        all_records: List[IngestionGameRecord] = []

        def _fetch_day(d: date) -> List[IngestionGameRecord]:
            return self._hist._fetch_day(d, year, session)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch_day, d): d for d in days}
            for future in as_completed(futures):
                try:
                    all_records.extend(future.result())
                except Exception as exc:
                    logger.debug("IncrementalGameFetcher day-fetch error: %s", exc)

        session.close()

        result = dedup_records(all_records)
        logger.info(
            "IncrementalGameFetcher: year=%d since=%s → %d games",
            year, since, len(result),
        )
        return result
