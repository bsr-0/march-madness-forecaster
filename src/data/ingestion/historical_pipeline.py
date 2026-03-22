"""Historical game ingestion pipeline for multi-season NCAA data.

The pipeline fetches complete historical seasons (default 2022–2025) and
writes canonical JSON artifacts consumed by the ML training pipeline.

Key changes from the previous version
--------------------------------------
* **No more cbbpy mandatory** — ``HistoricalGameFetcher`` (ESPN primary,
  sportsdataverse secondary, cbbpy tertiary) replaces the hard dependency on
  cbbpy.  cbbpy is tried last and only when both ESPN and sportsdataverse
  return insufficient games.
* **Real per-game dates** — ESPN scoreboard returns actual game dates;
  monthly-bucket placeholders are gone for newly fetched seasons.
* **Separated concerns** — game fetching is fully delegated to
  ``HistoricalGameFetcher``.  ``HistoricalDataPipeline`` handles only
  orchestration (season loop, validation, writing artifacts, Kaggle data).
* **Clean dedup** — ``game_fetchers.dedup_records`` uses ``game_id`` as the
  dedup key and logs date inconsistencies between providers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..normalize import normalize_team_id
from ..scrapers import SportsReferenceScraper, TournamentSeedScraper
from .game_fetchers import HistoricalGameFetcher, dedup_records, is_in_season_window, season_window
from .providers import LibraryProviderHub
from .validators import validate_games_payload, validate_ratings_payload

logger = logging.getLogger(__name__)


@dataclass
class HistoricalIngestionConfig:
    """Configuration for multi-season historical data pulls."""

    start_season: int = 2022
    end_season: int = 2025
    output_dir: str = "data/raw/historical"
    cache_dir: str = "data/raw/cache"

    include_pbp: bool = False
    strict_validation: bool = True
    retry_attempts: int = 2
    per_game_timeout_seconds: int = 25
    max_games_per_season: Optional[int] = None
    include_tournament_context: bool = True
    include_torvik: bool = True
    team_metrics_provider_priority: Optional[List[str]] = None
    torvik_provider_priority: Optional[List[str]] = None

    # Kaggle competition data directory for supplemental data
    kaggle_dir: Optional[str] = None


class HistoricalDataPipeline:
    """Collects real historical team/game data for model training.

    Delegates game fetching entirely to ``HistoricalGameFetcher`` so that:

    1. The provider cascade (ESPN → sportsdataverse → cbbpy) is centralised.
    2. Cache management and date repair live in one place.
    3. HistoricalDataPipeline focuses on orchestration: season loop, artifact
       writing, validation, torvik/seeds/kaggle enrichment.
    """

    def __init__(self, config: Optional[HistoricalIngestionConfig] = None):
        self.config = config or HistoricalIngestionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.providers = LibraryProviderHub()
        self.sports_reference = SportsReferenceScraper(str(self.cache_dir))
        self.tournament_seed_scraper = TournamentSeedScraper(str(self.cache_dir))
        self.game_fetcher = HistoricalGameFetcher(cache_dir=str(self.cache_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        if self.config.start_season > self.config.end_season:
            raise ValueError("start_season must be <= end_season")

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_season": self.config.start_season,
            "end_season": self.config.end_season,
            "artifacts": {},
            "providers": {},
            "validation_errors": {},
            "season_counts": {},
        }

        for season in range(self.config.start_season, self.config.end_season + 1):
            game_payload, game_provider = self._collect_season_games(season)
            game_errors = validate_games_payload({"games": game_payload["games"]})
            self._assert_valid(f"historical_games_{season}", game_errors)
            games_path = self._write_json(f"historical_games_{season}.json", game_payload)

            all_game_rows = game_payload.get("games", []) + game_payload.get("team_games", [])
            team_payload, team_provider = self._collect_team_metrics(
                season, game_records=all_game_rows,
            )
            team_errors = validate_ratings_payload(team_payload, name_field="team_name")
            self._assert_valid(f"team_metrics_{season}", team_errors)
            teams_path = self._write_json(f"team_metrics_{season}.json", team_payload)

            manifest["artifacts"][str(season)] = {
                "historical_games_json": games_path,
                "team_metrics_json": teams_path,
            }
            manifest["providers"][str(season)] = {
                "historical_games_json": game_provider,
                "team_metrics_json": team_provider,
            }
            manifest["validation_errors"][str(season)] = {
                "historical_games_json": game_errors,
                "team_metrics_json": team_errors,
            }
            manifest["season_counts"][str(season)] = {
                "games": len(game_payload["games"]),
                "team_games": len(game_payload.get("team_games", [])),
                "teams": len(team_payload.get("teams", [])),
            }

            if self.config.include_torvik:
                torvik_payload, torvik_provider = self._collect_torvik(season)
                if torvik_payload.get("teams"):
                    torvik_errors = validate_ratings_payload(torvik_payload)
                    self._assert_valid(f"torvik_{season}", torvik_errors)
                    torvik_path = self._write_json(f"torvik_{season}.json", torvik_payload)
                    manifest["artifacts"][str(season)]["torvik_json"] = torvik_path
                    manifest["providers"][str(season)]["torvik_json"] = torvik_provider
                    manifest["validation_errors"][str(season)]["torvik_json"] = torvik_errors
                    manifest["season_counts"][str(season)]["torvik_teams"] = len(torvik_payload["teams"])

            if self.config.include_tournament_context:
                tournament_payload, tournament_provider = self._collect_tournament_context(season)
                if tournament_payload.get("teams"):
                    seeds_path = self._write_json(f"tournament_seeds_{season}.json", tournament_payload)
                    manifest["artifacts"][str(season)]["tournament_seeds_json"] = seeds_path
                    manifest["providers"][str(season)]["tournament_seeds_json"] = tournament_provider
                    manifest["validation_errors"][str(season)]["tournament_seeds_json"] = []
                    manifest["season_counts"][str(season)]["tournament_seed_teams"] = len(tournament_payload["teams"])

            # Kaggle Massey Ordinals → external rating caches
            if not self.config.kaggle_dir:
                try:
                    from ..kaggle_downloader import ensure_kaggle_data
                    _resolved = ensure_kaggle_data(kaggle_dir=None, auto_download=True)
                    if _resolved:
                        self.config.kaggle_dir = _resolved
                        logger.info("Auto-resolved kaggle_dir: %s", _resolved)
                except (ImportError, OSError, ValueError) as _e:
                    logger.debug("kaggle_downloader.ensure_kaggle_data failed: %s", _e)
            if self.config.kaggle_dir:
                self._collect_kaggle_data(season, manifest)

        manifest_path = self._write_json(
            f"historical_manifest_{self.config.start_season}_{self.config.end_season}.json",
            manifest,
        )
        manifest["manifest_path"] = manifest_path
        return manifest

    # ── Game collection ────────────────────────────────────────────────────

    def _collect_season_games(self, season: int) -> Tuple[Dict, str]:
        """Fetch all games for *season* using HistoricalGameFetcher.

        Returns ``(payload_dict, provider_name)`` where payload_dict has the
        same structure as the legacy cbbpy-based output so all downstream
        consumers (``team_games_to_game_records``, etc.) remain unchanged::

            {
                "season": int,
                "provider": str,
                "games": [...],       # game-level rows (team1/team2)
                "team_games": [...],  # team-perspective rows (one per team per game)
                "failed_game_ids": [],
                "complete": bool,
            }
        """
        ingestion_records = self.game_fetcher.fetch_season(season)

        if self.config.max_games_per_season is not None:
            ingestion_records = ingestion_records[: self.config.max_games_per_season]

        if not ingestion_records:
            raise ValueError(f"No games collected for season {season}")

        # Determine which provider was used (majority vote)
        providers = [r.provider for r in ingestion_records]
        dominant_provider = max(set(providers), key=providers.count) if providers else "unknown"

        # Build games (game-level) and team_games (team-perspective) from records
        games = [r.to_game_row() for r in ingestion_records]
        team_games: List[Dict] = []
        for r in ingestion_records:
            team_games.extend(r.to_team_game_rows())

        # Validate dates — raise on CRITICAL issues instead of just logging
        date_warnings = self._validate_game_dates(games, season)
        critical_warnings = []
        for warning in date_warnings:
            logger.warning("Season %d date check: %s", season, warning)
            if warning.startswith("CRITICAL"):
                critical_warnings.append(warning)
        if critical_warnings:
            from .game_fetchers import IngestionQualityError
            raise IngestionQualityError(
                f"Season {season} date validation failed: "
                + "; ".join(critical_warnings)
            )

        payload = {
            "season": season,
            "provider": dominant_provider,
            "games": games,
            "team_games": team_games,
            "failed_game_ids": [],
            "complete": True,
        }
        return payload, dominant_provider

    # ── Other data collection methods (unchanged from legacy) ──────────────

    def _collect_torvik(self, season: int) -> Tuple[Dict, str]:
        provider = self.providers.fetch_torvik_ratings(
            season,
            priority=self.config.torvik_provider_priority,
        )
        teams = [t for t in provider.records if isinstance(t, dict)]
        return {"teams": teams}, provider.provider

    def _collect_team_metrics(
        self,
        season: int,
        game_records: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, str]:
        provider_result = self.providers.fetch_team_box_metrics(
            season,
            priority=self.config.team_metrics_provider_priority,
        )
        provider = provider_result.provider
        rows = self._ensure_team_ids(provider_result.records)

        if not rows:
            rows = self._ensure_team_ids(
                self.sports_reference.fetch_team_season_stats(
                    season, game_records=game_records,
                )
            )
            provider = "sports_reference_scraper"

        if rows and game_records:
            zero_count = sum(1 for r in rows if (r.get("def_rtg") or 0) <= 0)
            if zero_count > len(rows) * 0.5:
                from ..scrapers.sports_reference import SportsReferenceScraper
                team_paces = {
                    SportsReferenceScraper._normalize_id(r.get("team_name", "")): float(r.get("pace", 0))
                    for r in rows
                    if float(r.get("pace", 0)) > 0
                }
                game_def_rtg = SportsReferenceScraper._compute_def_rtg_from_games(
                    game_records, team_paces=team_paces,
                )
                for row in rows:
                    if (row.get("def_rtg") or 0) <= 0:
                        tid = self._normalize_team_name(row.get("team_name") or "")
                        if tid in game_def_rtg:
                            row["def_rtg"] = game_def_rtg[tid]

        if not rows:
            raise ValueError(f"No team metrics available for season {season}")
        return {"season": season, "teams": rows}, provider

    def _collect_kaggle_data(self, season: int, manifest: Dict) -> None:
        from ..scrapers.external_ratings import ExternalRatingsLoader

        kaggle_dir = self.config.kaggle_dir
        if not kaggle_dir:
            return

        try:
            ratings_loader = ExternalRatingsLoader(cache_dir=str(self.output_dir))
            n_systems = ratings_loader.populate_from_massey_ordinals(kaggle_dir, season)
            if n_systems > 0:
                manifest["artifacts"].setdefault(str(season), {})["massey_ordinals_systems"] = n_systems
                manifest["providers"].setdefault(str(season), {})["massey_ordinals"] = "kaggle_csv"
                manifest["season_counts"].setdefault(str(season), {})["massey_ordinal_systems"] = n_systems
                logger.info("Cached %d Massey Ordinal systems for season %d", n_systems, season)
        except (ValueError, KeyError, OSError, ImportError) as e:
            logger.warning("Massey Ordinals ingestion failed for season %d: %s", season, e)

    def _collect_tournament_context(self, season: int) -> Tuple[Dict, str]:
        try:
            teams = self.tournament_seed_scraper.fetch_tournament_seeds(season)
            return {"season": season, "teams": teams}, "sports_reference_tournament_scraper"
        except (ValueError, AttributeError, RuntimeError, OSError) as exc:
            logger.debug("Tournament seed scraping failed for %d: %s", season, exc)
            return {"season": season, "teams": []}, "none"

    # ── Date validation (retained for downstream callers) ──────────────────

    @staticmethod
    def _validate_game_dates(games: List[Dict], season: int) -> List[str]:
        """Return warnings if game dates look suspicious."""
        if not games:
            return []
        warnings: List[str] = []
        fallback = f"{season - 1}-11-01"
        fallback_count = sum(1 for g in games if g.get("date") == fallback)
        total = len(games)
        if fallback_count > total * 0.5:
            warnings.append(
                f"CRITICAL: {fallback_count}/{total} games have fallback date "
                f"{fallback}. Dates are likely missing from source data."
            )
        empty_count = sum(1 for g in games if not g.get("date"))
        if empty_count > 0:
            warnings.append(
                f"WARNING: {empty_count}/{total} games have empty or missing date field."
            )
        unique_dates = len(set(g.get("date", "") for g in games))
        if unique_dates < 10 and total > 100:
            warnings.append(
                f"CRITICAL: Only {unique_dates} unique dates across {total} games. "
                f"Date diversity is suspiciously low."
            )
        return warnings

    def repair_historical_dates(
        self,
        seasons: Optional[List[int]] = None,
        dry_run: bool = False,
        force_slow: bool = False,
    ) -> Dict[int, Dict]:
        """Re-fetch game dates and patch existing historical JSON files.

        Now delegates to ``HistoricalGameFetcher`` for ESPN-based date fetching.
        The ``force_slow`` argument is accepted for backward compatibility but
        ignored (the fetcher always uses ESP concurrent fetching).

        Returns a dict mapping season → {total, repaired, unique_dates}.
        """
        historical_dir = self.output_dir
        if seasons is None:
            import glob as _glob
            files = sorted(_glob.glob(str(historical_dir / "historical_games_*.json")))
            seasons = []
            for f in files:
                try:
                    yr = int(Path(f).stem.split("_")[-1])
                    seasons.append(yr)
                except ValueError:
                    continue

        results: Dict[int, Dict] = {}
        for season in seasons:
            json_path = historical_dir / f"historical_games_{season}.json"
            if not json_path.exists():
                logger.warning("No historical file for season %d, skipping", season)
                continue

            with open(json_path) as f:
                data = json.load(f)
            games = data.get("games", [])
            team_games = data.get("team_games", [])

            # Build game_id → date mapping via ESPN concurrent fetch
            game_date_map = self._fetch_date_map_for_season(season)

            if not game_date_map:
                logger.warning("Season %d: could not retrieve any dates. Skipping.", season)
                results[season] = {"total": len(games), "repaired": 0, "unique_dates": 0}
                continue

            repaired = 0
            for g in games:
                gid = str(g.get("game_id", "")).strip()
                if gid in game_date_map and g.get("date") != game_date_map[gid]:
                    if not dry_run:
                        g["date"] = game_date_map[gid]
                    repaired += 1

            for tg in team_games:
                gid = str(tg.get("game_id", "")).strip()
                if gid in game_date_map and tg.get("date") != game_date_map[gid]:
                    if not dry_run:
                        tg["date"] = game_date_map[gid]

            unique_dates = len(set(
                game_date_map.get(str(g.get("game_id", "")).strip(), g.get("date", ""))
                for g in games
            ))

            if not dry_run and repaired > 0:
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)

            logger.info(
                "Season %d: %d games, %d dates repaired, %d unique dates%s",
                season, len(games), repaired, unique_dates,
                " (dry run)" if dry_run else "",
            )
            results[season] = {
                "total": len(games),
                "repaired": repaired,
                "unique_dates": unique_dates,
            }

        return results

    def _fetch_date_map_for_season(self, season: int) -> Dict[str, str]:
        """Return {game_id: 'YYYY-MM-DD'} by querying ESPN scoreboard per-day."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import requests as _requests

        game_date_map: Dict[str, str] = {}
        days = list(self._season_dates(season))

        logger.info("Season %d: fetching game IDs for %d days via ESPN API", season, len(days))

        session = _requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })

        def _fetch_day(day):
            day_str = day.isoformat()
            ids = self._scrape_game_ids_for_date(day_str, session=session)
            return day_str, ids

        skipped = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_day, d): d for d in days}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    day_str, ids = future.result()
                except Exception as exc:
                    logger.debug("Failed to fetch IDs for a day: %s", exc)
                    skipped += 1
                    continue
                for gid in ids:
                    gid = str(gid).strip()
                    if gid and gid not in game_date_map:
                        game_date_map[gid] = day_str

        session.close()
        if skipped:
            logger.warning("Season %d: skipped %d days due to errors", season, skipped)
        logger.info("Season %d: extracted %d game-date mappings", season, len(game_date_map))
        return game_date_map

    # ── Utility methods ────────────────────────────────────────────────────

    _NCAA_SUFFIX_RE = re.compile(r"NCAA$", re.IGNORECASE)

    def _ensure_team_ids(self, rows: List[Dict]) -> List[Dict]:
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get("team_name") and row.get("name"):
                row["team_name"] = row["name"]
            if not row.get("name") and row.get("team_name"):
                row["name"] = row["team_name"]
            for key in ("team_name", "name"):
                val = row.get(key)
                if val and self._NCAA_SUFFIX_RE.search(val):
                    row[key] = self._NCAA_SUFFIX_RE.sub("", val).rstrip()
            if not row.get("team_id"):
                row["team_id"] = self._normalize_team_name(str(row.get("team_name") or row.get("name") or ""))
            else:
                tid = row["team_id"]
                if tid.endswith("ncaa"):
                    row["team_id"] = tid[:-4].rstrip("_")
            if row.get("team_id") and row.get("team_name"):
                out.append(row)
        return out

    def _season_dates(self, season: int) -> Iterable[date]:
        start = date(season - 1, 11, 1)
        end = date(season, 5, 1)
        current = start
        today = datetime.now(timezone.utc).date()
        stop = min(end, today)
        while current <= stop:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _scrape_game_ids_for_date(day_str: str, http_timeout: int = 15, session=None) -> List[str]:
        """Lightweight ESPN API call for game IDs on a single date."""
        import requests as _requests
        d = day_str.replace("-", "")
        api_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
            f"mens-college-basketball/scoreboard?dates={d}&groups=50&limit=200"
        )
        getter = session or _requests
        resp = getter.get(api_url, timeout=http_timeout)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("events", [])
        return [str(e["id"]) for e in events if "id" in e]

    def _write_json(self, filename: str, payload: Dict) -> str:
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(path)

    def _assert_valid(self, artifact_name: str, errors: List[str]) -> None:
        if errors and self.config.strict_validation:
            raise ValueError(f"{artifact_name} validation failed: {errors[:5]}")

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        return normalize_team_id(name)

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
