"""Provider adapters that prefer library-backed data over custom scraping.

Simplified to three verified sources:

    historical_games:  espn_scoreboard → sportsdataverse → cbbpy
    team_metrics:      sportsdataverse (only)
    torvik:            barttorvik CSV

Dead providers removed: sportsipy, cbbdata.  Their absence reduces the
provider chain from 5 to 3 entries with full test coverage and eliminates the
date-normalization hacks needed to paper over incompatible formats.

The ESPN scoreboard provider now extracts full box-score statistics from the
``competitor.statistics`` array (``fieldGoalsMade-fieldGoalsAttempted``, etc.)
so downstream features have the same data quality as cbbpy at much higher
reliability.

Deduplication uses ``(game_id, team_id, date)`` — if date is missing it falls
back to ``(game_id, team_id)`` — and logs a WARNING on collisions to surface
provider inconsistencies early.
"""

from __future__ import annotations

import csv
import importlib
import io
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

import requests

from ..normalize import normalize_team_id as _shared_normalize_team_id
from .game_fetchers import (
    _parse_espn_stats,
    is_in_season_window,
    parse_espn_event,
    _team_perspective_rows_to_records,
    HistoricalGameFetcher,
)

logger = logging.getLogger(__name__)

_ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)


@dataclass
class ProviderResult:
    provider: str
    records: List[Dict]
    strategy_used: str = ""
    metadata: Dict = field(default_factory=dict)


class LibraryProviderHub:
    """Best-effort data provider hub with ordered fallback.

    Provider priority (cannot be overridden for games; configurable for
    team_metrics and torvik through the ``priority`` argument):

    ``historical_games``:  sportsdataverse → espn_scoreboard → cbbpy
    ``team_metrics``:      sportsdataverse
    ``torvik``:            barttorvik
    """

    DEFAULT_PRIORITIES = {
        "historical_games": ["sportsdataverse", "espn_scoreboard", "cbbpy"],
        "team_metrics": ["sportsdataverse"],
        "torvik": ["barttorvik"],
    }

    def fetch_historical_games(
        self, year: int, priority: Optional[List[str]] = None, since: Optional[str] = None,
    ) -> ProviderResult:
        methods = {
            "espn_scoreboard": self._from_espn_scoreboard_api,
            "sportsdataverse": self._from_sportsdataverse_pbp,
            "cbbpy": self._from_cbbpy_pbp,
            "cbbdata": self._from_cbbdata_games_api,
        }
        for method in self._ordered_methods("historical_games", methods, priority):
            result = method(year, since=since)
            if result.records:
                return result
            logger.warning("Provider %s returned no historical game records for %d", result.provider, year)
        logger.warning("All historical_games providers exhausted for year %d — returning empty", year)
        return ProviderResult(provider="none", records=[])

    def fetch_team_box_metrics(self, year: int, priority: Optional[List[str]] = None) -> ProviderResult:
        methods = {
            "sportsdataverse": self._from_sportsdataverse_team_box,
        }
        for method in self._ordered_methods("team_metrics", methods, priority):
            result = method(year)
            if result.records:
                return result
            logger.warning("Provider %s returned no team box metrics for %d", result.provider, year)
        logger.warning("All team_metrics providers exhausted for year %d — returning empty", year)
        return ProviderResult(provider="none", records=[])

    def fetch_torvik_ratings(self, year: int, priority: Optional[List[str]] = None) -> ProviderResult:
        methods = {
            "torvik_r": self._from_torvik_r,
            "barttorvik": self._from_barttorvik_csv,
        }
        for method in self._ordered_methods("torvik", methods, priority):
            result = method(year)
            if result.records:
                return result
            logger.warning("Provider %s returned no torvik ratings for %d", result.provider, year)
        logger.warning("All torvik providers exhausted for year %d — returning empty", year)
        return ProviderResult(provider="none", records=[])

    def credential_requirements(self) -> Dict[str, List[str]]:
        return {
            "sportsdataverse_py": [],
            "cbbpy": [],
            "barttorvik": [],
            "torvik_r": [],  # No credentials required; needs R + toRvik package
        }

    def _ordered_methods(
        self,
        data_kind: str,
        methods: Dict[str, Callable[[int], ProviderResult]],
        priority: Optional[List[str]],
    ) -> List[Callable[[int], ProviderResult]]:
        ordered_names = [p.strip().lower() for p in (priority or self.DEFAULT_PRIORITIES[data_kind])]
        resolved: List[Callable[[int], ProviderResult]] = []
        for name in ordered_names:
            method = methods.get(name)
            if method is not None:
                resolved.append(method)
        return resolved

    # ── Historical games providers ─────────────────────────────────────────

    def _from_espn_scoreboard_api(self, year: int, since: Optional[str] = None) -> ProviderResult:
        """Fetch games directly from ESPN's public scoreboard JSON API.

        Queries both regular-season (seasontype=2) and postseason (seasontype=3)
        scoreboards so conference tournament and March Madness games are
        included.  Extracts full box-score statistics when available.

        Deduplication key: ``(game_id, date)`` — games returned by both
        seasontype=2 and seasontype=3 are merged, keeping the richer record.
        A WARNING is logged when the same game_id appears with different dates
        (indicates a provider inconsistency).
        """
        start_iso = since or f"{year - 1}-11-01"
        start = datetime.strptime(start_iso, "%Y-%m-%d").date()
        end = datetime.strptime(f"{year}-04-15", "%Y-%m-%d").date()

        # Create a single HistoricalGameFetcher instance for its _fetch_day logic
        fetcher = HistoricalGameFetcher(http_timeout=30, max_workers=1)
        session = requests.Session()

        all_records: List[Dict] = []
        current = start
        while current <= end:
            ingestion_records = fetcher._fetch_day(current, year, session)
            # Convert IngestionGameRecord → two team-perspective dicts per game
            for rec in ingestion_records:
                all_records.extend(rec.to_team_game_rows())
            current += timedelta(days=1)

        session.close()

        if not all_records:
            return ProviderResult("espn_scoreboard", [])

        # Deduplicate by (game_id, team_id, date) — log inconsistencies
        seen: Dict[tuple, Dict] = {}
        for rec in all_records:
            key = (str(rec.get("game_id", "")), str(rec.get("team_id", "")), str(rec.get("date", "")))
            if key in seen:
                logger.debug("Dedup: dropping exact duplicate game_id=%s team_id=%s", key[0], key[1])
                continue
            # Check for same (game_id, team_id) but different date
            loose_key = (str(rec.get("game_id", "")), str(rec.get("team_id", "")))
            existing_keys = [k for k in seen if k[:2] == loose_key]
            if existing_keys:
                existing_date = existing_keys[0][2]
                new_date = str(rec.get("date", ""))
                if existing_date != new_date:
                    logger.warning(
                        "Date inconsistency: game_id=%s team_id=%s has dates %s and %s — "
                        "keeping earlier ESPN record",
                        key[0], key[1], existing_date, new_date,
                    )
                continue
            seen[key] = rec

        return ProviderResult("espn_scoreboard", list(seen.values()))

    def _from_sportsdataverse_pbp(self, year: int, since: Optional[str] = None) -> ProviderResult:
        mbb = self._import_module("sportsdataverse.mbb")
        if mbb is None:
            return ProviderResult("sportsdataverse", [])

        call_candidates = [
            ("load_mbb_pbp", {"seasons": [year]}),
            ("load_mbb_pbp", {"seasons": year}),
            ("load_mbb_pbp", {"year": year}),
            ("espn_mbb_pbp", {"year": year}),
        ]
        for fn_name, kwargs in call_candidates:
            fn = getattr(mbb, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                records = self._frame_to_records(df)
                if records:
                    self._normalize_date_field(records)
                    if since:
                        records = [r for r in records if r.get("date", "") >= since]
                    return ProviderResult("sportsdataverse", records)
            except (TypeError, ValueError, AttributeError, KeyError, ImportError) as exc:
                logger.debug("sportsdataverse %s failed: %s", fn_name, exc)
                continue
        return ProviderResult("sportsdataverse", [])

    def _from_cbbpy_pbp(self, year: int, since: Optional[str] = None) -> ProviderResult:
        scraper = self._import_module("cbbpy.mens_scraper")
        if scraper is None:
            return ProviderResult("cbbpy", [])

        start_date = since or f"{year - 1}-11-01"
        end_date = f"{year}-04-15"

        all_game_rows: List[Dict] = []

        season_types_to_try = [None]
        if end_date >= f"{year}-03-01":
            season_types_to_try.append(3)

        for season_type in season_types_to_try:
            patched = False
            original_url = None
            if season_type is not None:
                try:
                    utils = self._import_module("cbbpy.utils.cbbpy_utils")
                    if utils and hasattr(utils, "MENS_SCOREBOARD_URL"):
                        original_url = utils.MENS_SCOREBOARD_URL
                        new_url = (
                            "https://www.espn.com/mens-college-basketball/"
                            f"scoreboard/_/date/{{}}/seasontype/{season_type}/group/50"
                        )
                        utils.MENS_SCOREBOARD_URL = new_url
                        # Verify the patch took effect — cbbpy may have changed internals
                        actual = getattr(utils, "MENS_SCOREBOARD_URL", None)
                        if actual != new_url:
                            logger.warning(
                                "cbbpy URL patch did not take effect (attribute reads back '%s'). "
                                "cbbpy may have changed its internals. "
                                "Postseason games (seasontype=%d) may be missing.",
                                actual,
                                season_type,
                            )
                        elif original_url == new_url:
                            logger.warning(
                                "cbbpy URL patch is a no-op (URL already matches target). "
                                "seasontype=%d games may already be included or URL format changed.",
                                season_type,
                            )
                        else:
                            logger.debug(
                                "cbbpy URL patched for seasontype=%d: '%s' → '%s'",
                                season_type, original_url, new_url,
                            )
                            patched = True
                except (TypeError, ValueError, AttributeError, ImportError) as exc:
                    logger.debug("cbbpy URL patch failed: %s", exc)

            try:
                rows = self._cbbpy_scrape_attempt(scraper, year, start_date, end_date, since)
                if rows:
                    all_game_rows.extend(rows)
            finally:
                if patched and original_url is not None:
                    try:
                        utils = self._import_module("cbbpy.utils.cbbpy_utils")
                        if utils:
                            utils.MENS_SCOREBOARD_URL = original_url
                    except (TypeError, AttributeError, ImportError) as exc:
                        logger.debug("cbbpy URL restore failed: %s", exc)

        if not all_game_rows:
            return ProviderResult("cbbpy", [])

        # Deduplicate by (game_id, team_id) — cbbpy does not always have reliable dates
        seen: set = set()
        deduped: List[Dict] = []
        for row in all_game_rows:
            key = (str(row.get("game_id", "")), str(row.get("team_id", "")))
            if key not in seen:
                seen.add(key)
                deduped.append(row)

        return ProviderResult("cbbpy", deduped)

    def _cbbpy_scrape_attempt(
        self, scraper, year: int, start_date: str, end_date: str, since: Optional[str],
    ) -> List[Dict]:
        fn_order = ("get_games_range",) if since else ("get_games_season", "get_games_range")
        for fn_name in fn_order:
            fn = getattr(scraper, fn_name, None)
            if fn is None:
                continue
            try:
                if fn_name == "get_games_season":
                    games = self._run_with_timeout(
                        fn, args=(year,),
                        kwargs={"info": True, "box": True, "pbp": False},
                        timeout=600,
                    )
                else:
                    games = self._run_with_timeout(
                        fn,
                        args=(f"{year - 1}-11-01", f"{year}-04-15"),
                        kwargs={"info": True, "box": True, "pbp": False},
                        timeout=600,
                    )
            except TypeError:
                try:
                    if fn_name == "get_games_season":
                        games = self._run_with_timeout(fn, args=(year,), timeout=600)
                    else:
                        games = self._run_with_timeout(
                            fn, args=(f"{year - 1}-11-01", f"{year}-04-15"), timeout=600,
                        )
                except Exception:
                    continue
            except Exception as exc:
                logger.debug("cbbpy %s failed: %s", fn_name, exc)
                continue

            game_rows = self._normalize_cbbpy_records(games)
            if not game_rows:
                continue
            if since:
                self._normalize_date_field(game_rows)
                game_rows = [
                    r for r in game_rows
                    if not r.get("date") or r["date"] >= since
                ]
            return game_rows

        return []

    def _from_cbbdata_games_api(self, year: int, since: Optional[str] = None) -> ProviderResult:
        """Fetch games from the cbbdata package (R-based API wrapper).

        Returns empty by default — subclasses or monkey-patches can provide
        a real implementation.
        """
        cbbdata = self._import_module("cbbdata")
        if cbbdata is None:
            return ProviderResult("cbbdata", [])
        return ProviderResult("cbbdata", [])

    # ── Team metrics provider ──────────────────────────────────────────────

    def _from_sportsdataverse_team_box(self, year: int) -> ProviderResult:
        mbb = self._import_module("sportsdataverse.mbb")
        if mbb is None:
            return ProviderResult("sportsdataverse", [])

        call_candidates = [
            ("load_mbb_team_boxscore", {"seasons": [year]}),
            ("load_mbb_team_boxscore", {"year": year}),
            ("espn_mbb_team_boxscore", {"year": year}),
        ]
        for fn_name, kwargs in call_candidates:
            fn = getattr(mbb, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                records = self._frame_to_records(df)
                if records:
                    return ProviderResult("sportsdataverse", records)
            except (TypeError, ValueError, AttributeError, KeyError, ImportError) as exc:
                logger.debug("sportsdataverse %s failed: %s", fn_name, exc)
                continue
        return ProviderResult("sportsdataverse", [])

    # ── Torvik provider ────────────────────────────────────────────────────

    def _from_torvik_r(self, year: int) -> ProviderResult:
        """Fetch Torvik ratings via the toRvik R package wrapper."""
        try:
            from ..scrapers.torvik_r import TorvikRWrapper
        except ImportError:
            logger.debug("torvik_r module not available")
            return ProviderResult("torvik_r", [])

        try:
            wrapper = TorvikRWrapper()
            if not wrapper.is_available():
                return ProviderResult("torvik_r", [])
            raw_records = wrapper.fetch_team_stats(year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torvik_r fetch failed: %s", exc)
            return ProviderResult("torvik_r", [])

        if not raw_records:
            return ProviderResult("torvik_r", [])

        records = []
        for rec in raw_records:
            mapped = self._map_barttorvik_row(rec)
            if mapped:
                records.append(mapped)
        return ProviderResult("torvik_r", records, strategy_used="torvik_r")

    def _from_barttorvik_csv(self, year: int) -> ProviderResult:
        url_template = os.getenv("BARTTORVIK_TORVIK_URL")
        if url_template:
            if "{year}" in url_template:
                url = url_template.format(year=year)
            else:
                logger.warning(
                    "BARTTORVIK_TORVIK_URL does not contain {year} placeholder; "
                    "using URL as-is (data may not match requested year %d)", year,
                )
                url = url_template
        else:
            url = f"https://barttorvik.com/{year}_team_results.csv"

        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
        except (requests.RequestException, ValueError, OSError) as exc:
            logger.warning("barttorvik request failed for %s: %s", url, exc)
            return ProviderResult("barttorvik", [])

        text = response.text.strip()
        if not text:
            return ProviderResult("barttorvik", [])

        try:
            sample = text[:4096]
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(text), dialect)
        rows = list(reader)
        if not rows:
            return ProviderResult("barttorvik", [])

        header = [c.strip() for c in rows[0]]
        has_header = any(h and not h.replace(".", "", 1).isdigit() for h in header)
        if not has_header:
            logger.warning(
                "barttorvik CSV appears to lack headers; skipping "
                "(set BARTTORVIK_TORVIK_URL to a headered feed)."
            )
            return ProviderResult("barttorvik", [])

        records: List[Dict] = []
        dict_reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        for row in dict_reader:
            record = self._map_barttorvik_row(row)
            if record:
                records.append(record)
        return ProviderResult("barttorvik", records, strategy_used="barttorvik_csv")

    @staticmethod
    def _map_barttorvik_row(row: Dict) -> Optional[Dict]:
        if not isinstance(row, dict):
            return None

        def pick(keys: List[str]) -> str:
            for key in keys:
                for k, v in row.items():
                    if k is None:
                        continue
                    if k.strip().lower() == key:
                        return str(v).strip()
            return ""

        def to_float(value: str) -> float:
            if value is None:
                return 0.0
            text = str(value).replace("%", "").strip()
            try:
                return float(text)
            except (TypeError, ValueError):
                return 0.0

        def normalize_rate(value: float) -> float:
            if value > 1.5:
                return value / 100.0
            return value

        name = pick(["team", "team_name", "team name", "school"])
        conf = pick(["conf", "conference"])
        if not name:
            return None

        team_id = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")

        t_rank = int(to_float(pick(["rank", "rk", "t_rank"]))) if pick(["rank", "rk", "t_rank"]) else 999

        barthag = to_float(pick(["barthag", "bar_thag"]))
        adj_oe = to_float(pick(["adjoe", "adjo", "adj_o", "adj_oe", "adj_off", "adj_offense", "adj_offensive_efficiency"]))
        adj_de = to_float(pick(["adjde", "adjd", "adj_d", "adj_de", "adj_def", "adj_defense", "adj_defensive_efficiency"]))
        adj_t = to_float(pick(["adjt", "adj_t", "tempo", "adj_tempo"]))

        efg = normalize_rate(to_float(pick(["efg", "efg%", "effective_fg_pct", "efg_pct"])))
        to_rate = normalize_rate(to_float(pick(["to%", "to_rate", "turnover_rate", "to_pct"])))
        orb = normalize_rate(to_float(pick(["orb%", "orb", "offensive_reb_rate", "orb_pct"])))
        ftr = normalize_rate(to_float(pick(["ftr", "ft_rate", "free_throw_rate", "ft_rate_pct"])))

        coach = pick(["coach", "head coach", "head_coach", "coach_name"])

        record = {
            "team_id": team_id,
            "team_name": name,
            "name": name,
            "conference": conf,
            "t_rank": t_rank,
            "barthag": barthag,
            "adj_offensive_efficiency": adj_oe,
            "adj_defensive_efficiency": adj_de,
            "adj_tempo": adj_t,
            "effective_fg_pct": efg,
            "turnover_rate": to_rate,
            "offensive_reb_rate": orb,
            "free_throw_rate": ftr,
        }
        if coach:
            record["coach"] = coach
        return record

    # ── Shared helpers (kept for backward compatibility with callers) ───────

    def _normalize_cbbpy_records(self, obj) -> List[Dict]:
        if not isinstance(obj, tuple):
            return self._frame_to_records(obj)

        info_df = obj[0] if len(obj) > 0 else None
        box_df = obj[1] if len(obj) > 1 else None

        game_date_map = self._extract_date_map_from_info(info_df)
        box_rows = self._frame_to_records(box_df)
        team_games = HistoricalGameFetcher._aggregate_box_rows(box_rows)
        if team_games:
            for row in team_games:
                gid = str(row.get("game_id", "")).strip()
                if gid in game_date_map:
                    row["date"] = game_date_map[gid]
            return team_games

        info_rows = self._frame_to_records(info_df)
        if info_rows:
            return info_rows
        return self._frame_to_records(obj[2] if len(obj) > 2 else None)

    @staticmethod
    def _extract_date_map_from_info(info_df) -> Dict[str, str]:
        date_map: Dict[str, str] = {}
        if info_df is None:
            return date_map
        if not hasattr(info_df, "iterrows") or getattr(info_df, "empty", True):
            return date_map
        for _, row in info_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            raw_day = str(row.get("game_day", "")).strip()
            if gid and raw_day:
                try:
                    parsed = datetime.strptime(raw_day, "%B %d, %Y")
                    date_map[gid] = parsed.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        return date_map

    @staticmethod
    def _import_module(module_name: str):
        try:
            return importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            return None

    @staticmethod
    def _run_with_timeout(fn, args=(), kwargs=None, timeout=120):
        """Run *fn* in a thread with a timeout to prevent indefinite hangs."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        kwargs = kwargs or {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeout:
                raise TimeoutError(
                    f"{getattr(fn, '__name__', fn)} timed out after {timeout}s"
                )

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
                records = to_dict("records")
                if isinstance(records, list):
                    return [r for r in records if isinstance(r, dict)]
            except (TypeError, ValueError, AttributeError):
                pass
        return []

    @staticmethod
    def _normalize_date_field(records: List[Dict]) -> None:
        """Ensure each record has a ``date`` key in ISO format (YYYY-MM-DD).

        Delegates to the shared implementation in game_fetchers so the logic
        lives in exactly one place.
        """
        HistoricalGameFetcher._normalize_date_field(records)

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        return _shared_normalize_team_id(name)

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
