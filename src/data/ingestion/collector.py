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
from .providers import LibraryProviderHub
from ...conference_tournament.data_enrichment import (
    compute_defensive_four_factors_from_games,
)
logger = logging.getLogger(__name__)

from .validators import (
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


class RealDataCollector:
    """Collects real-world data and writes canonical JSON artifacts."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.cache_dir = Path(config.cache_dir)
        self.providers = LibraryProviderHub()
        self.adv_builder = PublicAdvancedMetricsBuilder()
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
            out["teams_json"] = self._write(f"teams_{year}.json", payload)

        torvik_payload: Optional[Dict] = None
        four_factors: Dict = {}
        if self.config.scrape_torvik:
            tv_provider = self.providers.fetch_torvik_ratings(year, priority=self.config.torvik_provider_priority)
            if tv_provider.records:
                torvik_teams = tv_provider.records
                torvik_payload = {"teams": tv_provider.records}
                # Validate only the fields that the T-Rank CSV can provide.
                # Defensive four factors (opp_*) are NOT available from the
                # CSV source — they are enriched from game box scores below.
                validation_errors["torvik_json"] = validate_ratings_payload(
                    torvik_payload,
                    required_numeric_fields=[
                        "barthag",
                        "adj_offensive_efficiency",
                        "adj_defensive_efficiency",
                        "adj_tempo",
                        "effective_fg_pct",
                        "turnover_rate",
                        "offensive_reb_rate",
                        "free_throw_rate",
                    ],
                    variance_fields=[
                        "barthag", "adj_offensive_efficiency", "adj_defensive_efficiency",
                        "effective_fg_pct", "turnover_rate",
                    ],
                )
                self._assert_valid("torvik_json", validation_errors["torvik_json"])
                out["torvik_json"] = self._write(f"torvik_{year}.json", torvik_payload)
                provider_lineage["torvik_json"] = tv_provider.provider

            # Fetch Four Factors and shooting stats (separate Torvik endpoints)
            torvik_scraper = BartTorvikScraper(str(self.cache_dir))
            four_factors = torvik_scraper.fetch_four_factors(year)
            if four_factors:
                out["torvik_four_factors_json"] = self._write(
                    f"torvik_four_factors_{year}.json", four_factors,
                )
                provider_lineage["torvik_four_factors_json"] = "barttorvik"
            shooting = torvik_scraper.fetch_shooting_stats(year)
            if shooting:
                out["torvik_shooting_json"] = self._write(
                    f"torvik_shooting_{year}.json", shooting,
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
            game_provider = self.providers.fetch_historical_games(
                year,
                priority=self.config.historical_games_provider_priority,
                since=self.config.historical_games_since,
            )
            new_records = game_provider.records
            if self.config.historical_games_since and new_records:
                new_records = self._merge_incremental_games(year, new_records)
                provider_lineage["historical_games_json"] = f"{game_provider.provider}+incremental"
            elif new_records:
                provider_lineage["historical_games_json"] = game_provider.provider
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
                    provider_lineage["advanced_metrics_json"] = game_provider.provider

        # --- Enrich Torvik data with defensive Four Factors from game box scores ---
        # The T-Rank CSV and player-stats CSV fallback only provide offensive
        # metrics.  Now that historical games are available, compute defensive
        # Four Factors (opp_effective_fg_pct, opp_turnover_rate,
        # opp_free_throw_rate) from opponent box scores and merge them in.
        if torvik_payload and historical_team_rows:
            self._enrich_defensive_four_factors(
                torvik_payload, four_factors, historical_team_rows,
                year, out, validation_errors, provider_lineage,
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
                sr = SportsReferenceScraper(str(self.cache_dir)).fetch_team_season_stats(
                    year, game_records=historical_team_rows,
                )
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
            len(existing_games) - added, len(new_records), len(existing_games),
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

    def _enrich_defensive_four_factors(
        self,
        torvik_payload: Dict,
        four_factors: Dict,
        game_records: List[Dict],
        year: int,
        out: Dict[str, str],
        validation_errors: Dict[str, List[str]],
        provider_lineage: Dict[str, str],
    ) -> None:
        """Compute defensive Four Factors from game box scores and merge into Torvik data.

        The T-Rank CSV and player-stats CSV only provide offensive metrics.
        This method reconstructs defensive metrics (opp_effective_fg_pct,
        opp_turnover_rate, opp_free_throw_rate) from opponent box scores and
        merges them into both the main torvik payload and the four_factors dict.
        """
        def_ff = compute_defensive_four_factors_from_games(game_records)
        if not def_ff:
            logger.warning(
                "Could not compute defensive Four Factors from game records "
                "(no paired box scores with shooting data)"
            )
            return

        # Build collapsed-ID lookup for fuzzy matching
        collapsed_def_ff: Dict[str, Dict] = {}
        for k, v in def_ff.items():
            collapsed_def_ff[k.replace("_", "")] = v

        torvik_updated = 0
        for team in torvik_payload.get("teams", []):
            team_id = team.get("team_id", "")
            d = def_ff.get(team_id) or collapsed_def_ff.get(team_id.replace("_", ""))
            if d is None:
                continue
            changed = False
            for field in ("opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate"):
                current = team.get(field, 0.0)
                new_val = d.get(field, 0.0)
                if (current is None or abs(float(current)) < 1e-6) and abs(new_val) > 1e-6:
                    team[field] = new_val
                    changed = True
            if changed:
                torvik_updated += 1

        # Also merge into the separate four_factors dict
        ff_updated = 0
        for team_id, d in def_ff.items():
            ff_entry = four_factors.get(team_id)
            if ff_entry is None:
                # Try collapsed lookup
                for key in four_factors:
                    if key.replace("_", "") == team_id.replace("_", ""):
                        ff_entry = four_factors[key]
                        break
            if ff_entry is not None:
                for field in ("opp_effective_fg_pct", "opp_turnover_rate", "opp_free_throw_rate"):
                    current = ff_entry.get(field, 0.0)
                    new_val = d.get(field, 0.0)
                    if abs(float(current or 0)) < 1e-6 and abs(new_val) > 1e-6:
                        ff_entry[field] = new_val
                ff_updated += 1

        logger.info(
            "Defensive Four Factors enrichment: %d/%d torvik teams updated, "
            "%d four_factors entries updated (from %d game-derived teams)",
            torvik_updated, len(torvik_payload.get("teams", [])),
            ff_updated, len(def_ff),
        )

        # Re-write enriched files
        out["torvik_json"] = self._write(f"torvik_{year}.json", torvik_payload)
        if four_factors:
            out["torvik_four_factors_json"] = self._write(
                f"torvik_four_factors_{year}.json", four_factors,
            )

        # Post-enrichment validation: now check that defensive fields are populated
        post_errors = validate_ratings_payload(
            torvik_payload,
            required_numeric_fields=[
                "opp_effective_fg_pct",
                "opp_turnover_rate",
                "opp_free_throw_rate",
            ],
        )
        if post_errors:
            validation_errors.setdefault("torvik_json", []).extend(post_errors)
            logger.warning(
                "Post-enrichment validation: %d defensive FF issues remain",
                len(post_errors),
            )
        provider_lineage["torvik_defensive_ff"] = "game_box_scores"

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

        if not isinstance(base_team.get("stints"), list) and isinstance(overlay_team.get("stints"), list):
            base_team["stints"] = copy.deepcopy(overlay_team["stints"])

    def _team_key(self, team: Dict) -> str:
        return self._normalize_team_id(
            str(team.get("team_id") or team.get("team_name") or team.get("name") or "")
        )

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
                    f"kaggle_tourney_seeds_{year}.json", payload,
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
                    f"kaggle_regular_season_{year}.json", payload,
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
                    f"kaggle_tourney_results_{year}.json", payload,
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
                    f"kaggle_coaches_{year}.json", payload,
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
                    f"kaggle_conferences_{year}.json", payload,
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
                errors.append(
                    f"teams[{idx}] has only {nonzero} non-zero RAPM players; expected >= {min_players}"
                )
        return errors
