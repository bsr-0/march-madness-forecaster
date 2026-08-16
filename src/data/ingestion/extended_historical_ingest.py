"""Extended historical data ingestion for NCAA basketball (1996-2025).

Orchestrates multi-source data collection across the full available range:
- Tournament results (Sports Reference): 1996-2025
- Team season stats (Sports Reference): 2003-2025
- Game-level data (ESPN/cbbpy): 2003-2025
- Torvik ratings: 2008-2025
- External ratings (Kaggle Massey): 2003-2025

Each source is collected independently with appropriate year ranges,
rate limiting, and skip-if-exists logic.  Produces canonical JSON
artifacts in the same format consumed by the training pipeline.

No leakage risk: this module is purely data collection, not model training.
All temporal gating is handled downstream by the pipeline stages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Source-specific earliest available years
TOURNAMENT_RESULTS_START = 1996
TEAM_STATS_START = 2003
GAME_DATA_START = 2003
TORVIK_START = 2008
EXTERNAL_RATINGS_START = 2003

# Rate limiting for Sports Reference (aggressive bot detection)
SR_REQUEST_DELAY_SECONDS = 3.5


@dataclass
class ExtendedIngestionConfig:
    """Configuration for extended multi-source historical data pulls."""

    start_season: int = 2003
    end_season: int = 2025
    output_dir: str = "data/raw/historical"
    cache_dir: str = "data/raw/cache"

    include_tournament_results: bool = True
    include_team_stats: bool = True
    include_game_data: bool = True
    include_torvik: bool = True
    include_external_ratings: bool = True

    skip_existing: bool = True
    strict_validation: bool = True
    retry_attempts: int = 2
    per_game_timeout_seconds: int = 25

    kaggle_dir: Optional[str] = None

    # Rate limit delay between scraper requests (seconds)
    scraper_delay: float = SR_REQUEST_DELAY_SECONDS


class ExtendedHistoricalIngestor:
    """Collect historical NCAA data across all available sources and years.

    Produces the standard training pipeline artifacts:
    - ``historical_games_{year}.json``
    - ``team_metrics_{year}.json``
    - ``tournament_results_{year}.json``
    - ``torvik_{year}.json``
    """

    def __init__(self, config: Optional[ExtendedIngestionConfig] = None):
        self.config = config or ExtendedIngestionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        """Execute extended historical ingestion across all sources.

        Returns a manifest dict with per-year, per-source artifact paths
        and completeness metrics.
        """
        if self.config.start_season > self.config.end_season:
            raise ValueError("start_season must be <= end_season")

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_season": self.config.start_season,
            "end_season": self.config.end_season,
            "sources": {},
            "skipped": {},
            "errors": {},
            "summary": {},
        }

        if self.config.include_tournament_results:
            self._collect_tournament_results(manifest)

        if self.config.include_game_data:
            self._collect_game_data(manifest)

        if self.config.include_team_stats:
            self._collect_team_stats(manifest)

        if self.config.include_torvik:
            self._collect_torvik_ratings(manifest)

        if self.config.include_external_ratings:
            self._collect_external_ratings(manifest)

        manifest_path = self.output_dir / (
            f"extended_manifest_{self.config.start_season}_{self.config.end_season}.json"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        manifest["manifest_path"] = str(manifest_path)

        self._log_summary(manifest)
        return manifest

    def _effective_start(self, source_start: int) -> int:
        """Return the effective start year for a source."""
        return max(self.config.start_season, source_start)

    def _artifact_exists(self, filename: str) -> bool:
        """Check if an artifact already exists and should be skipped.

        For tournament results files, also validates that the file contains
        actual game data (not just a placeholder with empty games list).
        """
        if not self.config.skip_existing:
            return False
        path = self.output_dir / filename
        if not path.exists() or path.stat().st_size <= 100:
            return False
        if "tournament_results" in filename:
            try:
                data = json.loads(path.read_text())
                return len(data.get("games", [])) > 0
            except (json.JSONDecodeError, OSError):
                return False
        return True

    def _write_json(self, filename: str, data: Dict) -> str:
        """Write JSON artifact and return the path."""
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        return str(path)

    def _context_subkey_exists(self, year: int, sub_key: str, legacy_filename: str) -> bool:
        """Check if `sub_key` data for `year` already exists, either in the
        consolidated `tournament_context_{year}.json` or the old per-type
        file (skip-if-exists logic spanning the migration window)."""
        ctx_path = self.output_dir / f"tournament_context_{year}.json"
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
            if sub_key in ctx:
                if sub_key == "results" and "tournament_results" in legacy_filename:
                    return len(ctx[sub_key].get("games", [])) > 0
                return True
        return self._artifact_exists(legacy_filename)

    def _write_context_subkey(self, year: int, sub_key: str, payload: Dict) -> str:
        """Read-merge-write `payload` into `tournament_context_{year}.json`
        under `sub_key`, preserving any other sub-keys already present."""
        ctx_path = self.output_dir / f"tournament_context_{year}.json"
        ctx: Dict = {}
        if ctx_path.exists():
            with open(ctx_path) as f:
                ctx = json.load(f)
        ctx[sub_key] = payload
        ctx_path.write_text(json.dumps(ctx, indent=2, default=str))
        return str(ctx_path)

    def _write_json_preserving_ff(self, filename: str, payload: Dict) -> str:
        """Write `payload` to `filename`, preserving any existing
        `four_factors`/`four_factors_snapshots` keys already on disk at
        that path (this writer never populates those keys itself)."""
        path = self.output_dir / filename
        existing: Dict = {}
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        merged = dict(payload)
        for k in ("four_factors", "four_factors_snapshots"):
            if k in existing:
                merged[k] = existing[k]
        path.write_text(json.dumps(merged, indent=2, default=str))
        return str(path)

    # ── Tournament Results ─────────────────────────────────────────────

    def _collect_tournament_results(self, manifest: Dict) -> None:
        """Scrape tournament bracket results from Sports Reference."""
        from ..scrapers.tournament_results import TournamentResultsScraper

        scraper = TournamentResultsScraper(cache_dir=str(self.cache_dir))
        start = self._effective_start(TOURNAMENT_RESULTS_START)

        for year in range(start, self.config.end_season + 1):
            if year == 2020:
                continue  # No tournament (COVID)
            filename = f"tournament_results_{year}.json"
            if self._context_subkey_exists(year, "results", filename):
                manifest.setdefault("skipped", {}).setdefault("tournament_results", []).append(year)
                continue

            try:
                results = scraper.scrape_tournament(year)
                if results:
                    self._write_context_subkey(
                        year,
                        "results",
                        {
                            "season": year,
                            "source": "sports_reference",
                            "games": results,
                            "n_games": len(results),
                        },
                    )
                    manifest.setdefault("sources", {}).setdefault("tournament_results", {})[str(year)] = len(results)
                    logger.info("Tournament results %d: %d games", year, len(results))
                else:
                    logger.warning("No tournament results for %d", year)
            except Exception as exc:
                logger.warning("Tournament results %d failed: %s", year, exc)
                manifest.setdefault("errors", {}).setdefault("tournament_results", {})[str(year)] = str(exc)

            time.sleep(self.config.scraper_delay)

    # ── Game Data ──────────────────────────────────────────────────────

    def _collect_game_data(self, manifest: Dict) -> None:
        """Fetch game-level data using the historical game fetcher."""
        from .game_fetchers import HistoricalGameFetcher
        from .validators import validate_games_payload

        fetcher = HistoricalGameFetcher(cache_dir=str(self.cache_dir))
        start = self._effective_start(GAME_DATA_START)

        for year in range(start, self.config.end_season + 1):
            if year == 2020:
                continue
            filename = f"historical_games_{year}.json"
            if self._artifact_exists(filename):
                manifest.setdefault("skipped", {}).setdefault("game_data", []).append(year)
                continue

            try:
                records = fetcher.fetch_season(year)
                if not records:
                    logger.warning("No game data for season %d", year)
                    continue

                games = [r.to_game_row() for r in records]
                team_games = []
                for r in records:
                    team_games.extend(r.to_team_game_rows())

                providers = [r.provider for r in records]
                dominant = max(set(providers), key=providers.count) if providers else "unknown"

                payload = {
                    "season": year,
                    "provider": dominant,
                    "games": games,
                    "team_games": team_games,
                    "failed_game_ids": [],
                    "complete": True,
                }

                errors = validate_games_payload({"games": games})
                if errors and self.config.strict_validation:
                    logger.warning("Game data validation errors for %d: %s", year, errors)

                self._write_json(filename, payload)
                manifest.setdefault("sources", {}).setdefault("game_data", {})[str(year)] = len(games)
                logger.info("Game data %d: %d games (%s)", year, len(games), dominant)

            except Exception as exc:
                logger.warning("Game data %d failed: %s", year, exc)
                manifest.setdefault("errors", {}).setdefault("game_data", {})[str(year)] = str(exc)

    # ── Team Stats ─────────────────────────────────────────────────────

    def _collect_team_stats(self, manifest: Dict) -> None:
        """Fetch team-level advanced stats from Sports Reference."""
        from ..scrapers.sports_reference import SportsReferenceScraper
        from .validators import validate_ratings_payload

        scraper = SportsReferenceScraper(cache_dir=str(self.cache_dir))
        start = self._effective_start(TEAM_STATS_START)

        for year in range(start, self.config.end_season + 1):
            filename = f"team_metrics_{year}.json"
            if self._context_subkey_exists(year, "team_metrics", filename):
                manifest.setdefault("skipped", {}).setdefault("team_stats", []).append(year)
                continue

            try:
                # Load game records if available (for def_rtg fallback)
                game_records = self._load_game_records(year)
                teams = scraper.fetch_team_season_stats(year, game_records=game_records)

                if not teams:
                    logger.warning("No team stats for %d", year)
                    continue

                payload = {"season": year, "teams": teams}
                errors = validate_ratings_payload(payload, name_field="team_name")
                if errors and self.config.strict_validation:
                    logger.warning("Team stats validation for %d: %s", year, errors)

                self._write_context_subkey(year, "team_metrics", payload)
                manifest.setdefault("sources", {}).setdefault("team_stats", {})[str(year)] = len(teams)
                logger.info("Team stats %d: %d teams", year, len(teams))

            except Exception as exc:
                logger.warning("Team stats %d failed: %s", year, exc)
                manifest.setdefault("errors", {}).setdefault("team_stats", {})[str(year)] = str(exc)

            time.sleep(self.config.scraper_delay)

    # ── Torvik Ratings ─────────────────────────────────────────────────

    def _collect_torvik_ratings(self, manifest: Dict) -> None:
        """Fetch Torvik T-Rank ratings (available ~2008+)."""
        from .providers import LibraryProviderHub

        providers = LibraryProviderHub()
        start = self._effective_start(TORVIK_START)

        for year in range(start, self.config.end_season + 1):
            if year == 2020:
                continue
            filename = f"torvik_{year}.json"
            if self._artifact_exists(filename):
                manifest.setdefault("skipped", {}).setdefault("torvik", []).append(year)
                continue

            try:
                result = providers.fetch_torvik_ratings(year)
                teams = [t for t in result.records if isinstance(t, dict)]
                if teams:
                    self._write_json_preserving_ff(filename, {"teams": teams})
                    manifest.setdefault("sources", {}).setdefault("torvik", {})[str(year)] = len(teams)
                    logger.info("Torvik %d: %d teams", year, len(teams))
                else:
                    logger.warning("No Torvik data for %d", year)
            except Exception as exc:
                logger.warning("Torvik %d failed: %s", year, exc)
                manifest.setdefault("errors", {}).setdefault("torvik", {})[str(year)] = str(exc)

    # ── External Ratings ───────────────────────────────────────────────

    def _collect_external_ratings(self, manifest: Dict) -> None:
        """Populate external ratings from Kaggle Massey Ordinals."""
        kaggle_dir = self.config.kaggle_dir
        if not kaggle_dir:
            try:
                from ..kaggle_downloader import ensure_kaggle_data

                kaggle_dir = ensure_kaggle_data(kaggle_dir=None, auto_download=True)
            except (ImportError, OSError, ValueError) as exc:
                logger.debug("Could not auto-resolve kaggle_dir: %s", exc)
                return

        if not kaggle_dir:
            logger.info("No kaggle_dir available, skipping external ratings")
            return

        from ..scrapers.external_ratings import ExternalRatingsLoader

        loader = ExternalRatingsLoader(cache_dir=str(self.output_dir))
        start = self._effective_start(EXTERNAL_RATINGS_START)

        for year in range(start, self.config.end_season + 1):
            if year == 2020:
                continue
            try:
                n_systems = loader.populate_from_massey_ordinals(kaggle_dir, year)
                if n_systems > 0:
                    manifest.setdefault("sources", {}).setdefault("external_ratings", {})[str(year)] = n_systems
                    logger.info("External ratings %d: %d systems", year, n_systems)
            except (ValueError, KeyError, OSError, ImportError) as exc:
                logger.warning("External ratings %d failed: %s", year, exc)
                manifest.setdefault("errors", {}).setdefault("external_ratings", {})[str(year)] = str(exc)

    # ── Helpers ────────────────────────────────────────────────────────

    def _load_game_records(self, year: int) -> Optional[list]:
        """Load previously collected game records for a year (for def_rtg fallback)."""
        games_path = self.output_dir / f"historical_games_{year}.json"
        if not games_path.exists():
            return None
        try:
            data = json.loads(games_path.read_text())
            return data.get("games", []) + data.get("team_games", [])
        except (json.JSONDecodeError, OSError):
            return None

    def _log_summary(self, manifest: Dict) -> None:
        """Log a summary of the ingestion run."""
        sources = manifest.get("sources", {})
        skipped = manifest.get("skipped", {})
        errors = manifest.get("errors", {})

        for source_name, years in sources.items():
            logger.info(
                "Source '%s': %d years collected, %d skipped, %d errors",
                source_name,
                len(years),
                len(skipped.get(source_name, [])),
                len(errors.get(source_name, {})),
            )


def get_data_availability_summary(output_dir: str = "data/raw/historical") -> Dict:
    """Scan the historical data directory and report per-year availability.

    Returns a dict mapping year → {source: bool} for each expected artifact.
    Useful for identifying gaps in historical coverage.
    """
    hist_dir = Path(output_dir)
    if not hist_dir.exists():
        return {}

    summary: Dict[int, Dict[str, bool]] = {}

    for year in range(TOURNAMENT_RESULTS_START, 2026):
        if year == 2020:
            continue
        year_data: Dict[str, bool] = {}
        year_data["tournament_results"] = (hist_dir / f"tournament_results_{year}.json").exists()
        year_data["game_data"] = (hist_dir / f"historical_games_{year}.json").exists()
        year_data["team_metrics"] = (hist_dir / f"team_metrics_{year}.json").exists()
        year_data["torvik"] = (hist_dir / f"torvik_{year}.json").exists()
        summary[year] = year_data

    return summary
