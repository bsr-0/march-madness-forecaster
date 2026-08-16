"""End-to-end real data ingestion orchestrator."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..normalize import normalize_team_id
from ..scrapers import (
    BartTorvikScraper,
    CBSPicksScraper,
    CBBpyRosterScraper,
    ESPNPicksScraper,
    NCAAStatsScraper,
    OpenDataFeedScraper,
    PlayerMetricsScraper,
    SportsReferenceScraper,
    TournamentContextScraper,
    TransferPortalScraper,
    YahooPicksScraper,
    aggregate_consensus,
)
from ..features.public_advanced_metrics import PublicAdvancedMetricsBuilder
from .game_fetchers import IncrementalGameFetcher
from .providers import LibraryProviderHub

logger = logging.getLogger(__name__)

from .validators import (
    validate_four_factors,
    validate_games_payload,
    validate_public_picks_payload,
    validate_ratings_payload,
    validate_odds_payload,
    validate_rosters_payload,
    validate_teams_payload,
    validate_transfer_payload,
)


@dataclass
class IngestionConfig:
    year: int
    output_dir: str = "data/raw"
    cache_dir: str = "data/raw/cache"

    ncaa_teams_url: Optional[str] = None
    ncaa_games_url: Optional[str] = None
    transfer_portal_url: Optional[str] = None
    transfer_portal_format: str = "json"
    roster_url: Optional[str] = None
    roster_format: str = "json"
    odds_url: Optional[str] = None
    odds_format: str = "json"
    polls_url: Optional[str] = None
    torvik_splits_url: Optional[str] = None
    ncaa_team_stats_url: Optional[str] = None
    weather_context_url: Optional[str] = None
    travel_context_url: Optional[str] = None

    # Kaggle competition data directory (contains CSV files like
    # MRegularSeasonDetailedResults.csv, MMasseyOrdinals.csv, etc.)
    kaggle_dir: Optional[str] = None

    # Tournament context enrichment (AP polls, coach history, conf tourney)
    scrape_tournament_context: bool = True
    preseason_ap_json: Optional[str] = None  # Pre-built JSON path override
    coach_tournament_json: Optional[str] = None  # Pre-built JSON path override
    conf_champions_json: Optional[str] = None  # Pre-built JSON path override

    scrape_torvik: bool = True
    scrape_public_picks: bool = True
    scrape_sports_reference: bool = True
    scrape_rosters: bool = True
    scrape_historical_games: bool = True
    historical_games_since: Optional[str] = None  # ISO date, e.g. "2026-03-08"

    historical_games_provider_priority: Optional[List[str]] = None
    team_metrics_provider_priority: Optional[List[str]] = None
    torvik_provider_priority: Optional[List[str]] = None
    strict_validation: bool = True
    min_nonzero_rapm_players_per_team: int = 3


_FF_FIELDS = (
    "effective_fg_pct",
    "turnover_rate",
    "offensive_reb_rate",
    "free_throw_rate",
    "opp_effective_fg_pct",
    "opp_turnover_rate",
    "opp_free_throw_rate",
    "defensive_reb_rate",
)


def _compute_defensive_ff_from_games(torvik_payload: dict, game_rows: list) -> int:
    """Compute defensive four factors from game box score data.

    Used when Torvik's four_factors endpoint is blocked (e.g. leakage guard)
    and opp_* fields are zero in the ratings CSV.  Aggregates opponent stats
    from each team's game records.

    Returns count of teams updated.
    """
    from collections import defaultdict

    teams = torvik_payload.get("teams", [])
    if not teams or not game_rows:
        return 0

    # group game rows by opponent_id: row.fga/fgm/etc are the stats of the
    # team that *opposed* opponent_id, i.e. the defensive exposure data.
    opp_games: dict = defaultdict(list)
    for g in game_rows:
        if not isinstance(g, dict):
            continue
        opp_id = g.get("opponent_id")
        if opp_id:
            opp_games[normalize_team_id(opp_id)].append(g)

    if not opp_games:
        return 0

    updated = 0
    for team in teams:
        tid = normalize_team_id(team.get("team_id", ""))
        if abs(float(team.get("opp_effective_fg_pct", 0) or 0)) > 1e-6:
            continue  # already populated

        games = opp_games.get(tid, [])
        if not games:
            continue

        efg_sum = to_sum = ftr_sum = 0.0
        n_efg = n_to = n_ftr = 0

        for g in games:
            fga = float(g.get("fga", 0) or 0)
            fgm = float(g.get("fgm", 0) or 0)
            fg3m = float(g.get("fg3m", 0) or 0)
            fta = float(g.get("fta", 0) or 0)
            turnovers = float(g.get("turnovers", 0) or 0)
            possessions = float(g.get("possessions", 0) or 0)

            if fga > 0:
                efg_sum += (fgm + 0.5 * fg3m) / fga
                n_efg += 1
                ftr_sum += fta / fga
                n_ftr += 1
            if possessions > 0:
                to_sum += turnovers / possessions
                n_to += 1

        changed = False
        if n_efg > 0:
            team["opp_effective_fg_pct"] = round(efg_sum / n_efg, 4)
            changed = True
        if n_to > 0:
            team["opp_turnover_rate"] = round(to_sum / n_to, 4)
            changed = True
        if n_ftr > 0:
            team["opp_free_throw_rate"] = round(ftr_sum / n_ftr, 4)
            changed = True

        if changed:
            updated += 1

    return updated


def _merge_four_factors_into_torvik(torvik_payload: dict, four_factors: dict) -> int:
    """Merge four_factors entries into torvik team dicts, filling zero fields.

    Uses normalize_team_id to resolve _st/_state and alias mismatches.
    Returns count of teams updated.
    """
    teams = torvik_payload.get("teams", [])
    if not teams:
        return 0

    # Build normalized lookup: normalize_team_id(ff_key) -> ff_entry
    normalized_ff: dict = {}
    for key, entry in four_factors.items():
        if isinstance(entry, dict):
            normalized_ff[normalize_team_id(key)] = entry

    updated = 0
    for team in teams:
        tid = team.get("team_id", "")
        ff = four_factors.get(tid)
        if ff is None:
            ff = normalized_ff.get(normalize_team_id(tid))
        if ff is None or not isinstance(ff, dict):
            continue
        changed = False
        for field in _FF_FIELDS:
            current = float(team.get(field, 0) or 0)
            source = float(ff.get(field, 0) or 0)
            if abs(current) < 1e-6 and abs(source) > 1e-6:
                team[field] = ff[field]
                changed = True
        if changed:
            updated += 1
    return updated


class RealDataCollector:
    """Collects real-world data and writes canonical JSON artifacts."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.cache_dir = Path(config.cache_dir)
        self.providers = LibraryProviderHub()
        self.adv_builder = PublicAdvancedMetricsBuilder()
        self.incremental_fetcher = IncrementalGameFetcher()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        year = self.config.year
        out: Dict[str, str] = {}
        provider_lineage: Dict[str, str] = {}
        validation_errors: Dict[str, List[str]] = {}
        historical_team_rows: List[Dict] = []
        torvik_teams: List[Dict] = []

        if self.config.ncaa_teams_url:
            teams = NCAAStatsScraper(str(self.cache_dir)).fetch_tournament_teams(year, self.config.ncaa_teams_url)
            payload = {"teams": teams}
            validation_errors["teams_json"] = validate_teams_payload(payload)
            self._assert_valid("teams_json", validation_errors["teams_json"])
            out["teams_json"] = self._write_context_subkey(year, "teams", payload)

        torvik_payload: Optional[Dict] = None
        four_factors: Dict = {}
        if self.config.scrape_torvik:
            # Leakage guard: warn if collecting Torvik data after tournament start
            try:
                from ...pipeline.config import TOURNAMENT_START_DATES
                from datetime import date as _date

                _cutoff = TOURNAMENT_START_DATES.get(year)
                if _cutoff and _date.today() >= _cutoff:
                    logger.warning(
                        "Torvik data collection for %d on %s (tournament started %s). "
                        "Efficiency metrics may include tournament games — LEAKAGE RISK.",
                        year,
                        _date.today(),
                        _cutoff,
                    )
            except ImportError:
                pass
            tv_provider = self.providers.fetch_torvik_ratings(year, priority=self.config.torvik_provider_priority)
            if tv_provider.records:
                torvik_teams = tv_provider.records
                torvik_payload = {"teams": tv_provider.records}
                # Validate only the core efficiency fields available from the
                # T-Rank CSV.  Four Factors (eFG%, TO%, ORB%, FTR) are fetched
                # separately via fetch_four_factors() below and enriched into
                # the team data — they should NOT block ratings validation.
                validation_errors["torvik_json"] = validate_ratings_payload(
                    torvik_payload,
                    required_numeric_fields=[
                        "barthag",
                        "adj_offensive_efficiency",
                        "adj_defensive_efficiency",
                        "adj_tempo",
                    ],
                    variance_fields=[
                        "barthag",
                        "adj_offensive_efficiency",
                        "adj_defensive_efficiency",
                    ],
                    stddev_thresholds={
                        "adj_offensive_efficiency": 5.0,
                        "adj_defensive_efficiency": 5.0,
                        "barthag": 0.05,
                    },
                )
                self._assert_valid("torvik_json", validation_errors["torvik_json"])
                out["torvik_json"] = self._write(f"torvik_{year}.json", torvik_payload)
                provider_lineage["torvik_json"] = tv_provider.provider

            # Fetch Four Factors and shooting stats (separate Torvik endpoints)
            torvik_scraper = BartTorvikScraper(str(self.cache_dir))
            try:
                four_factors = torvik_scraper.fetch_four_factors(year)
            except Exception as exc:
                if "LeakageError" in type(exc).__name__:
                    logger.warning("Four factors fetch blocked by leakage guard: %s", exc)
                else:
                    raise
            if four_factors:
                out["torvik_four_factors_json"] = self._write(
                    f"torvik_four_factors_{year}.json",
                    four_factors,
                )
                provider_lineage["torvik_four_factors_json"] = "barttorvik"

                # Cross-source consistency check (advisory only — never blocks)
                if torvik_teams:
                    from src.data.ingestion.validators import cross_validate_torvik_sources

                    cross_warnings = cross_validate_torvik_sources(
                        torvik_teams,
                        [{"team_id": k, **v} for k, v in four_factors.items() if isinstance(v, dict)],
                        thresholds={"effective_fg_pct": 0.03, "turnover_rate": 0.03},
                    )
                    for w in cross_warnings:
                        logger.warning("[cross-validate] %s", w)
            # Merge four_factors back into torvik_payload so the main file
            # has non-zero Four Factors from the start (prevents the
            # all-zeros corruption that required post-hoc repair scripts).
            if four_factors and torvik_payload:
                _merged = _merge_four_factors_into_torvik(torvik_payload, four_factors)
                if _merged:
                    out["torvik_json"] = self._write(f"torvik_{year}.json", torvik_payload)
                    logger.info(
                        "Merged Four Factors into torvik_%d.json for %d teams",
                        year,
                        _merged,
                    )

            try:
                shooting = torvik_scraper.fetch_shooting_stats(year)
            except Exception as exc:
                if "LeakageError" in type(exc).__name__:
                    logger.warning("Shooting stats fetch blocked by leakage guard: %s", exc)
                    shooting = {}
                else:
                    raise
            if shooting:
                out["torvik_shooting_json"] = self._write(
                    f"torvik_shooting_{year}.json",
                    shooting,
                )
                provider_lineage["torvik_shooting_json"] = "barttorvik"

        if self.config.scrape_public_picks:
            espn = ESPNPicksScraper(str(self.cache_dir)).fetch_picks(year)
            yahoo = YahooPicksScraper(str(self.cache_dir)).fetch_picks(year)
            cbs = CBSPicksScraper(str(self.cache_dir)).fetch_picks(year)

            picks = aggregate_consensus(espn, yahoo, cbs)
            source_order = [
                ("espn", espn),
                ("yahoo", yahoo),
                ("cbs", cbs),
            ]
            populated_sources = [name for name, data in source_order if data.teams]
            if not picks.teams:
                # Fallback to whichever source returned data first.
                for name, data in source_order:
                    if data.teams:
                        picks = data
                        populated_sources = [name]
                        break

            if picks.teams:
                payload = {
                    "teams": {
                        team_id: {
                            "team_name": p.team_name,
                            "seed": p.seed,
                            "region": p.region,
                            "round_of_64_pct": p.round_of_64_pct,
                            "round_of_32_pct": p.round_of_32_pct,
                            "sweet_16_pct": p.sweet_16_pct,
                            "elite_8_pct": p.elite_8_pct,
                            "final_four_pct": p.final_four_pct,
                            "champion_pct": p.champion_pct,
                        }
                        for team_id, p in picks.teams.items()
                    },
                    "sources": populated_sources or picks.sources,
                    "timestamp": picks.timestamp or datetime.now(timezone.utc).isoformat(),
                }
                validation_errors["public_picks_json"] = validate_public_picks_payload(payload)
                self._assert_valid("public_picks_json", validation_errors["public_picks_json"])
                out["public_picks_json"] = self._write(f"public_picks_{year}.json", payload)
                provider_lineage["public_picks_json"] = ",".join(populated_sources or picks.sources or ["unknown"])

        if not self.config.scrape_historical_games:
            logger.info("Skipping historical games scrape (--skip-historical-games)")
        elif self.config.ncaa_games_url:
            games = NCAAStatsScraper(str(self.cache_dir)).fetch_historical_games(year, self.config.ncaa_games_url)
            self._ensure_game_dates(games, year)
            payload = {"games": games}
            validation_errors["historical_games_json"] = validate_games_payload(payload)
            self._assert_valid("historical_games_json", validation_errors["historical_games_json"])
            out["historical_games_json"] = self._write(f"historical_games_{year}.json", payload)
            provider_lineage["historical_games_json"] = "ncaa_stats_scraper"
            historical_team_rows = [g for g in games if isinstance(g, dict)]
        else:
            # Incremental path: use IncrementalGameFetcher (ESPN primary) when
            # a since-date is provided, then merge with existing data.
            # Full-season path: fall back to LibraryProviderHub (ESPN → sdv → cbbpy).
            if self.config.historical_games_since:
                ingestion_records = self.incremental_fetcher.fetch_since(
                    year=year,
                    since=self.config.historical_games_since,
                )
                # Convert to team-perspective dicts
                new_records: List[Dict] = []
                for rec in ingestion_records:
                    new_records.extend(rec.to_team_game_rows())
                if new_records:
                    new_records = self._merge_incremental_games(year, new_records)
                    provider_lineage["historical_games_json"] = "espn_scoreboard+incremental"
                game_provider_name = "espn_scoreboard"
            else:
                game_provider = self.providers.fetch_historical_games(
                    year,
                    priority=self.config.historical_games_provider_priority,
                )
                new_records = game_provider.records
                if new_records:
                    provider_lineage["historical_games_json"] = game_provider.provider
                game_provider_name = (
                    game_provider.provider if not self.config.historical_games_since else "espn_scoreboard"
                )

            if new_records:
                self._ensure_game_dates(new_records, year)
                historical_team_rows = [g for g in new_records if isinstance(g, dict)]
                payload = {"games": new_records}
                validation_errors["historical_games_json"] = validate_games_payload(payload)
                self._assert_valid("historical_games_json", validation_errors["historical_games_json"])
                out["historical_games_json"] = self._write(f"historical_games_{year}.json", payload)

                advanced_metrics = self.adv_builder.build(new_records, teams=torvik_teams)
                if "advanced_metrics_json" not in out:
                    validation_errors["advanced_metrics_json"] = validate_ratings_payload(advanced_metrics)
                    self._assert_valid("advanced_metrics_json", validation_errors["advanced_metrics_json"])
                    out["advanced_metrics_json"] = self._write(f"advanced_metrics_{year}.json", advanced_metrics)
                    provider_lineage["advanced_metrics_json"] = game_provider_name

        # Compute defensive four factors from game box scores when Torvik's
        # opp_* fields are zero (e.g. leakage guard blocks four_factors endpoint).
        if torvik_payload and historical_team_rows:
            _needs_dff = any(
                abs(float(t.get("opp_effective_fg_pct", 0) or 0)) < 1e-6 for t in torvik_payload.get("teams", [])
            )
            if _needs_dff:
                _dff_updated = _compute_defensive_ff_from_games(torvik_payload, historical_team_rows)
                if _dff_updated:
                    out["torvik_json"] = self._write(f"torvik_{year}.json", torvik_payload)
                    provider_lineage["torvik_defensive_ff"] = "game_box_scores"
                    logger.info(
                        "Computed defensive four factors from game box scores for %d teams",
                        _dff_updated,
                    )

        if self.config.scrape_rosters:
            roster_payload = CBBpyRosterScraper(str(self.cache_dir)).fetch_rosters(year)
            external_roster_payload = {}
            if self.config.roster_url or os.getenv("PLAYER_METRICS_URL"):
                external_roster_payload = PlayerMetricsScraper(str(self.cache_dir)).fetch_rosters(
                    year,
                    source_url=self.config.roster_url,
                    fmt=self.config.roster_format,
                )
            if roster_payload and external_roster_payload:
                roster_payload = self._merge_roster_payloads(roster_payload, external_roster_payload)
            elif external_roster_payload:
                roster_payload = external_roster_payload
            if roster_payload:
                validation_errors["rosters_json"] = validate_rosters_payload(roster_payload)
                validation_errors["rosters_json"].extend(
                    self._validate_roster_rapm_quality(
                        roster_payload,
                        min_players=self.config.min_nonzero_rapm_players_per_team,
                    )
                )
                self._assert_valid("rosters_json", validation_errors["rosters_json"])
                out["rosters_json"] = self._write(f"rosters_{year}.json", roster_payload)
                provider_lineage["rosters_json"] = str(roster_payload.get("source", "cbbpy_schedule_boxscore"))

        if self.config.scrape_sports_reference:
            sr_provider = self.providers.fetch_team_box_metrics(
                year,
                priority=self.config.team_metrics_provider_priority,
            )
            if sr_provider.records:
                payload = {"teams": self._ensure_ids(sr_provider.records)}
                validation_errors["sports_reference_json"] = validate_ratings_payload(
                    payload,
                    name_field="team_name",
                    required_numeric_fields=["off_rtg", "def_rtg", "pace"],
                    variance_fields=["off_rtg", "def_rtg", "pace"],
                )
                self._assert_valid("sports_reference_json", validation_errors["sports_reference_json"])
                out["sports_reference_json"] = self._write(f"sports_reference_{year}.json", payload)
                provider_lineage["sports_reference_json"] = sr_provider.provider
                sr = []
            else:
                try:
                    sr = SportsReferenceScraper(str(self.cache_dir)).fetch_team_season_stats(
                        year,
                        game_records=historical_team_rows,
                    )
                except (ValueError, Exception) as exc:
                    logger.warning("SportsReference scraper fallback failed: %s", exc)
                    sr = []
            if sr:
                payload = {"teams": self._ensure_ids(sr)}
                validation_errors["sports_reference_json"] = validate_ratings_payload(
                    payload,
                    name_field="team_name",
                    required_numeric_fields=["off_rtg", "def_rtg", "pace"],
                    variance_fields=["off_rtg", "def_rtg", "pace"],
                )
                self._assert_valid("sports_reference_json", validation_errors["sports_reference_json"])
                out["sports_reference_json"] = self._write(f"sports_reference_{year}.json", payload)
                provider_lineage["sports_reference_json"] = "sports_reference_scraper"

        if self.config.transfer_portal_url:
            transfers = TransferPortalScraper(str(self.cache_dir)).fetch_entries(
                year,
                self.config.transfer_portal_url,
                fmt=self.config.transfer_portal_format,
            )
            payload = {"entries": transfers}
            validation_errors["transfer_portal_json"] = validate_transfer_payload(payload)
            self._assert_valid("transfer_portal_json", validation_errors["transfer_portal_json"])
            out["transfer_portal_json"] = self._write(f"transfer_portal_{year}.json", payload)

        if self.config.odds_url:
            odds_rows = OpenDataFeedScraper(str(self.cache_dir)).fetch_records(
                cache_name=f"odds_{year}.json",
                source_url=self.config.odds_url,
                fmt=self.config.odds_format,
                records_key="teams",
            )
            odds_payload = {"teams": odds_rows}
            validation_errors["odds_json"] = validate_odds_payload(odds_payload)
            self._assert_valid("odds_json", validation_errors["odds_json"])
            out["odds_json"] = self._write(f"odds_{year}.json", odds_payload)
            provider_lineage["odds_json"] = "open_feed"

        # --- Tournament context enrichment ---
        if self.config.scrape_tournament_context:
            ctx_scraper = TournamentContextScraper(str(self.cache_dir))
            try:
                ap_rankings = ctx_scraper.fetch_preseason_ap_rankings(year)
                if ap_rankings:
                    ap_payload = {"rankings": ap_rankings, "year": year}
                    out["preseason_ap_json"] = self._write(f"preseason_ap_{year}.json", ap_payload)
                    provider_lineage["preseason_ap_json"] = "sports_reference"
            except Exception as e:
                logger.warning(f"Preseason AP scrape failed: {e}")

            try:
                coach_data = ctx_scraper.fetch_coach_tournament_experience(year)
                if coach_data:
                    coach_payload = {"coaches": coach_data, "year": year}
                    out["coach_tournament_json"] = self._write(f"coach_tournament_{year}.json", coach_payload)
                    provider_lineage["coach_tournament_json"] = "barttorvik"
            except Exception as e:
                logger.warning(f"Coach tournament scrape failed: {e}")

            try:
                conf_champs = ctx_scraper.fetch_conference_tournament_champions(year)
                if conf_champs:
                    champs_payload = {"champions": conf_champs, "year": year}
                    out["conf_champions_json"] = self._write(f"conf_champions_{year}.json", champs_payload)
                    provider_lineage["conf_champions_json"] = "sports_reference"
            except Exception as e:
                logger.warning(f"Conference champions scrape failed: {e}")
        else:
            # Load from pre-built JSON files if provided
            for key, path in [
                ("preseason_ap_json", self.config.preseason_ap_json),
                ("coach_tournament_json", self.config.coach_tournament_json),
                ("conf_champions_json", self.config.conf_champions_json),
            ]:
                if path:
                    out[key] = path

        supplemental_feeds = [
            ("polls_json", self.config.polls_url, f"polls_{year}.json", "polls"),
            ("torvik_splits_json", self.config.torvik_splits_url, f"torvik_splits_{year}.json", "teams"),
            ("ncaa_team_stats_json", self.config.ncaa_team_stats_url, f"ncaa_team_stats_{year}.json", "teams"),
            ("weather_context_json", self.config.weather_context_url, f"weather_context_{year}.json", "records"),
            ("travel_context_json", self.config.travel_context_url, f"travel_context_{year}.json", "records"),
        ]
        feed_scraper = OpenDataFeedScraper(str(self.cache_dir))
        for artifact_key, source_url, filename, records_key in supplemental_feeds:
            if not source_url:
                continue
            records = feed_scraper.fetch_records(
                cache_name=filename,
                source_url=source_url,
                fmt="json",
                records_key=records_key,
            )
            payload = {"records": records}
            out[artifact_key] = self._write(filename, payload)
            provider_lineage[artifact_key] = "open_feed"

        # --- Kaggle competition CSV data integration ---
        # Auto-resolve kaggle_dir if not explicitly set (downloads if needed)
        if not self.config.kaggle_dir:
            try:
                from ..kaggle_downloader import ensure_kaggle_data

                _resolved = ensure_kaggle_data(kaggle_dir=None, auto_download=True)
                if _resolved:
                    self.config.kaggle_dir = _resolved
                    logger.info("Auto-resolved kaggle_dir: %s", _resolved)
            except Exception as _e:
                logger.debug("kaggle_downloader.ensure_kaggle_data failed: %s", _e)
        if self.config.kaggle_dir:
            self._ingest_kaggle_data(year, out, provider_lineage, validation_errors)

        # --- Cross-source team ID consistency check ---
        try:
            from .validators import validate_team_id_consistency

            _ctx_path = self.output_dir / f"tournament_context_{year}.json"
            _seeds_data = None
            if _ctx_path.exists():
                with open(_ctx_path) as _cf:
                    _seeds_data = json.load(_cf).get("seeds")
            if _seeds_data is None:
                _seeds_path = self.output_dir / f"tournament_seeds_{year}.json"
                if _seeds_path.exists():
                    with open(_seeds_path) as _sf:
                        _seeds_data = json.load(_sf)
            if _seeds_data is not None and torvik_payload:
                _tid_errors = validate_team_id_consistency(_seeds_data, torvik_payload, year=year)
                if _tid_errors:
                    validation_errors["team_id_consistency"] = _tid_errors
                    for _e in _tid_errors:
                        logger.warning("Team ID consistency: %s", _e)
        except Exception as _exc:
            logger.debug("Team ID consistency check skipped: %s", _exc)

        manifest = {
            "year": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": out,
            "providers": provider_lineage,
            "provenance": {},
            "credential_requirements": self.providers.credential_requirements(),
            "validation_errors": validation_errors,
        }
        manifest_path = self._write(f"manifest_{year}.json", manifest)
        manifest["manifest_path"] = manifest_path
        return manifest

    def _write(self, name: str, payload: Dict) -> str:
        p = self.output_dir / name
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        return str(p)

    def _write_context_subkey(self, year: int, sub_key: str, payload: Dict) -> str:
        """Read-merge-write `payload` into `tournament_context_{year}.json`
        under `sub_key`, preserving any other sub-keys already present."""
        path = self.output_dir / f"tournament_context_{year}.json"
        ctx: Dict = {}
        if path.exists():
            with open(path) as f:
                ctx = json.load(f)
        ctx[sub_key] = payload
        with open(path, "w") as f:
            json.dump(ctx, f, indent=2)
        return str(path)

    def _merge_incremental_games(self, year: int, new_records: List[Dict]) -> List[Dict]:
        """Load existing historical games file and merge in new records."""
        existing_path = self.output_dir / f"historical_games_{year}.json"
        existing_games: List[Dict] = []
        if existing_path.exists():
            with open(existing_path) as f:
                data = json.load(f)
            existing_games = data.get("games", [])

        # Deduplicate by (game_id, team_id).  Earlier versions included date
        # in the key, which caused mismatches when dates were parsed
        # differently across runs (empty vs populated) — leading to either
        # duplicates or silent data loss.
        seen = set()
        for g in existing_games:
            key = (g.get("game_id", ""), g.get("team_id", ""))
            seen.add(key)

        added = 0
        for g in new_records:
            key = (g.get("game_id", ""), g.get("team_id", ""))
            if key not in seen:
                existing_games.append(g)
                seen.add(key)
                added += 1

        logger.info(
            "Incremental merge: %d existing + %d new records (%d after dedup)",
            len(existing_games) - added,
            len(new_records),
            len(existing_games),
        )
        return existing_games

    def _assert_valid(self, artifact_name: str, errors: List[str]) -> None:
        if errors and self.config.strict_validation:
            raise ValueError(f"{artifact_name} validation failed: {errors[:5]}")

    @staticmethod
    def _ensure_game_dates(records: List[Dict], year: int) -> None:
        """Normalise date fields and log warnings for games missing dates.

        This is a safety net: every provider *should* already supply dates, but
        if any records slip through without one this method ensures the ``date``
        key exists and logs the gap so operators can investigate.
        """
        from .providers import LibraryProviderHub

        LibraryProviderHub._normalize_date_field(records)

        missing = sum(1 for r in records if not r.get("date"))
        if missing:
            logger.warning(
                "Year %d: %d/%d game records have no date after provider "
                "normalisation. Downstream date-dependent features (Elo, "
                "momentum) may be inaccurate for these games.",
                year,
                missing,
                len(records),
            )

    @staticmethod
    def _normalize_team_id(name: str) -> str:
        # Delegate to shared utility (kept for backwards compat)
        return normalize_team_id(name)

    _NCAA_SUFFIX_RE = re.compile(r"NCAA$", re.IGNORECASE)

    def _ensure_ids(self, records: List[Dict]) -> List[Dict]:
        result = []
        for row in records:
            if not isinstance(row, dict):
                continue
            if not row.get("team_name") and row.get("name"):
                row["team_name"] = row["name"]
            if not row.get("name") and row.get("team_name"):
                row["name"] = row["team_name"]
            # Strip "NCAA" suffix from team names (Sports Reference appends it
            # for tournament qualifiers, producing IDs like "akronncaa").
            for key in ("team_name", "name"):
                val = row.get(key)
                if val and self._NCAA_SUFFIX_RE.search(val):
                    row[key] = self._NCAA_SUFFIX_RE.sub("", val).rstrip()
            team_id = row.get("team_id")
            if not team_id:
                team_id = self._normalize_team_id(row.get("team_name") or row.get("name"))
                row["team_id"] = team_id
            elif team_id.endswith("ncaa"):
                row["team_id"] = team_id[:-4].rstrip("_")
            result.append(row)
        return result

    def _merge_roster_payloads(self, base_payload: Dict, overlay_payload: Dict) -> Dict:
        merged = copy.deepcopy(base_payload)
        merged_teams = merged.get("teams", [])
        overlay_teams = overlay_payload.get("teams", [])
        if not isinstance(merged_teams, list) or not isinstance(overlay_teams, list):
            return merged

        team_index = {}
        for idx, team in enumerate(merged_teams):
            if not isinstance(team, dict):
                continue
            key = self._team_key(team)
            if key:
                team_index[key] = idx

        for overlay_team in overlay_teams:
            if not isinstance(overlay_team, dict):
                continue
            key = self._team_key(overlay_team)
            if not key:
                continue

            if key in team_index:
                base_team = merged_teams[team_index[key]]
                self._merge_team_players(base_team, overlay_team)
            else:
                merged_teams.append(copy.deepcopy(overlay_team))
                team_index[key] = len(merged_teams) - 1

        base_source = str(base_payload.get("source", "cbbpy_schedule_boxscore"))
        overlay_source = str(overlay_payload.get("source", "player_metrics"))
        merged["source"] = f"{base_source}+{overlay_source}"
        merged["timestamp"] = datetime.now(timezone.utc).isoformat()
        return merged

    def _merge_team_players(self, base_team: Dict, overlay_team: Dict) -> None:
        if not base_team.get("team_name"):
            base_team["team_name"] = overlay_team.get("team_name")
        if not base_team.get("team_id"):
            base_team["team_id"] = overlay_team.get("team_id")

        base_players = base_team.get("players")
        overlay_players = overlay_team.get("players")
        if not isinstance(base_players, list):
            base_players = []
            base_team["players"] = base_players
        if not isinstance(overlay_players, list):
            return

        base_index = {}
        for idx, player in enumerate(base_players):
            if not isinstance(player, dict):
                continue
            key = self._player_key(player)
            if key:
                base_index[key] = idx

        for overlay_player in overlay_players:
            if not isinstance(overlay_player, dict):
                continue
            key = self._player_key(overlay_player)
            if key and key in base_index:
                base_player = base_players[base_index[key]]
                for field, value in overlay_player.items():
                    if value is None:
                        continue
                    if isinstance(value, str) and not value.strip():
                        continue
                    base_player[field] = value
            else:
                base_players.append(copy.deepcopy(overlay_player))
                if key:
                    base_index[key] = len(base_players) - 1

    def _team_key(self, team: Dict) -> str:
        return self._normalize_team_id(str(team.get("team_id") or team.get("team_name") or team.get("name") or ""))

    def _player_key(self, player: Dict) -> str:
        raw_id = str(player.get("player_id") or "").strip()
        if raw_id:
            return f"id:{raw_id.lower()}"
        name = str(player.get("name") or "").strip().lower()
        return f"name:{name}" if name else ""

    def _ingest_kaggle_data(
        self,
        year: int,
        out: Dict[str, str],
        provider_lineage: Dict[str, str],
        validation_errors: Dict[str, List[str]],
    ) -> None:
        """Load Kaggle competition CSV data and cache external ratings."""
        from ..kaggle_loader import KaggleDataLoader
        from ..scrapers.external_ratings import ExternalRatingsLoader

        kaggle_dir = self.config.kaggle_dir
        if not kaggle_dir:
            return

        loader = KaggleDataLoader(kaggle_dir)

        # 1. Massey Ordinals → external ratings cache
        try:
            ratings_loader = ExternalRatingsLoader(cache_dir=str(self.output_dir))
            n_systems = ratings_loader.populate_from_massey_ordinals(kaggle_dir, year)
            if n_systems > 0:
                out["massey_ordinals_systems"] = str(n_systems)
                provider_lineage["massey_ordinals"] = "kaggle_csv"
                logger.info("Cached %d Massey Ordinal systems from Kaggle", n_systems)
        except Exception as e:
            logger.warning("Massey Ordinals ingestion failed: %s", e)

        # 2. Tournament seeds (authoritative source)
        try:
            seeds = loader.load_tourney_seeds(year)
            if seeds:
                payload = {"teams": seeds, "source": "kaggle_csv"}
                out["kaggle_tourney_seeds_json"] = self._write(
                    f"kaggle_tourney_seeds_{year}.json",
                    payload,
                )
                provider_lineage["kaggle_tourney_seeds_json"] = "kaggle_csv"
        except Exception as e:
            logger.warning("Kaggle tourney seeds load failed: %s", e)

        # 3. Regular season detailed results (official Four Factors)
        try:
            games = loader.load_regular_season_detailed(year)
            if not games:
                games = loader.load_regular_season_compact(year)
            if games:
                payload = {"games": games, "source": "kaggle_csv"}
                out["kaggle_regular_season_json"] = self._write(
                    f"kaggle_regular_season_{year}.json",
                    payload,
                )
                provider_lineage["kaggle_regular_season_json"] = "kaggle_csv"
        except Exception as e:
            logger.warning("Kaggle regular season load failed: %s", e)

        # 4. Tournament results (for backtesting)
        try:
            tourney = loader.load_tourney_detailed(year)
            if not tourney:
                tourney = loader.load_tourney_compact(year)
            if tourney:
                payload = {"games": tourney, "source": "kaggle_csv"}
                out["kaggle_tourney_results_json"] = self._write(
                    f"kaggle_tourney_results_{year}.json",
                    payload,
                )
                provider_lineage["kaggle_tourney_results_json"] = "kaggle_csv"
        except Exception as e:
            logger.warning("Kaggle tourney results load failed: %s", e)

        # 5. Team coaches
        try:
            coaches = loader.load_team_coaches(year)
            if coaches:
                payload = {"coaches": coaches, "source": "kaggle_csv"}
                out["kaggle_coaches_json"] = self._write(
                    f"kaggle_coaches_{year}.json",
                    payload,
                )
                provider_lineage["kaggle_coaches_json"] = "kaggle_csv"
        except Exception as e:
            logger.warning("Kaggle coaches load failed: %s", e)

        # 6. Team conferences
        try:
            confs = loader.load_team_conferences(year)
            if confs:
                payload = {"conferences": confs, "source": "kaggle_csv"}
                out["kaggle_conferences_json"] = self._write(
                    f"kaggle_conferences_{year}.json",
                    payload,
                )
                provider_lineage["kaggle_conferences_json"] = "kaggle_csv"
        except Exception as e:
            logger.warning("Kaggle conferences load failed: %s", e)

    def _validate_roster_rapm_quality(self, payload: Dict, min_players: int) -> List[str]:
        if min_players <= 0:
            return []
        errors: List[str] = []
        for idx, team in enumerate(payload.get("teams", [])):
            if not isinstance(team, dict):
                continue
            players = [p for p in team.get("players", []) if isinstance(p, dict)]
            nonzero = 0
            for p in players:
                rapm_total = p.get("rapm_total")
                rapm_off = p.get("rapm_offensive")
                rapm_def = p.get("rapm_defensive")
                value = None
                try:
                    if rapm_total is not None:
                        value = float(rapm_total)
                    elif rapm_off is not None or rapm_def is not None:
                        value = float(rapm_off or 0.0) + float(rapm_def or 0.0)
                except (TypeError, ValueError):
                    value = None
                if value is not None and abs(value) > 1e-6:
                    nonzero += 1
            if players and nonzero < min_players:
                errors.append(f"teams[{idx}] has only {nonzero} non-zero RAPM players; expected >= {min_players}")
        return errors
