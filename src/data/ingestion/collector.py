"""End-to-end real data ingestion orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..scrapers import (
    ESPNPicksScraper,
    KenPomScraper,
    NCAAStatsScraper,
    ShotQualityScraper,
    SportsReferenceScraper,
    TransferPortalScraper,
)
from ..features.public_advanced_metrics import PublicAdvancedMetricsBuilder
from .providers import LibraryProviderHub
from .validators import (
    validate_games_payload,
    validate_public_picks_payload,
    validate_ratings_payload,
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

    scrape_torvik: bool = True
    scrape_kenpom: bool = True
    scrape_shotquality: bool = True
    scrape_public_picks: bool = True
    scrape_sports_reference: bool = True

    historical_games_provider_priority: Optional[List[str]] = None
    team_metrics_provider_priority: Optional[List[str]] = None
    kenpom_provider_priority: Optional[List[str]] = None
    torvik_provider_priority: Optional[List[str]] = None
    strict_validation: bool = True


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

        if self.config.ncaa_teams_url:
            teams = NCAAStatsScraper(str(self.cache_dir)).fetch_tournament_teams(year, self.config.ncaa_teams_url)
            payload = {"teams": teams}
            validation_errors["teams_json"] = validate_teams_payload(payload)
            self._assert_valid("teams_json", validation_errors["teams_json"])
            out["teams_json"] = self._write(f"teams_{year}.json", payload)

        if self.config.scrape_torvik:
            tv_provider = self.providers.fetch_torvik_ratings(year, priority=self.config.torvik_provider_priority)
            if tv_provider.records:
                payload = {"teams": tv_provider.records}
                validation_errors["torvik_json"] = validate_ratings_payload(payload)
                self._assert_valid("torvik_json", validation_errors["torvik_json"])
                out["torvik_json"] = self._write(f"torvik_{year}.json", payload)
                provider_lineage["torvik_json"] = tv_provider.provider

        if self.config.scrape_kenpom:
            kp_provider = self.providers.fetch_kenpom_ratings(
                year,
                priority=self.config.kenpom_provider_priority,
            )
            if kp_provider.records:
                payload = {"teams": kp_provider.records}
                validation_errors["kenpom_json"] = validate_ratings_payload(payload)
                self._assert_valid("kenpom_json", validation_errors["kenpom_json"])
                out["kenpom_json"] = self._write(f"kenpom_{year}.json", payload)
                provider_lineage["kenpom_json"] = kp_provider.provider
            else:
                kp_rows = KenPomScraper(str(self.cache_dir)).fetch_ratings(year)
                if kp_rows:
                    payload = {"teams": [row.to_dict() for row in kp_rows]}
                    validation_errors["kenpom_json"] = validate_ratings_payload(payload)
                    self._assert_valid("kenpom_json", validation_errors["kenpom_json"])
                    out["kenpom_json"] = self._write(f"kenpom_{year}.json", payload)
                    provider_lineage["kenpom_json"] = "kenpom_scraper"

        if self.config.scrape_public_picks:
            picks = ESPNPicksScraper(str(self.cache_dir)).fetch_picks(year)
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
                    "sources": picks.sources,
                    "timestamp": picks.timestamp,
                }
                validation_errors["public_picks_json"] = validate_public_picks_payload(payload)
                self._assert_valid("public_picks_json", validation_errors["public_picks_json"])
                out["public_picks_json"] = self._write(f"public_picks_{year}.json", payload)

        if self.config.ncaa_games_url:
            games = NCAAStatsScraper(str(self.cache_dir)).fetch_historical_games(year, self.config.ncaa_games_url)
            payload = {"games": games}
            validation_errors["historical_games_json"] = validate_games_payload(payload)
            self._assert_valid("historical_games_json", validation_errors["historical_games_json"])
            out["historical_games_json"] = self._write(f"historical_games_{year}.json", payload)
            provider_lineage["historical_games_json"] = "ncaa_stats_scraper"
        else:
            game_provider = self.providers.fetch_historical_games(
                year,
                priority=self.config.historical_games_provider_priority,
            )
            if game_provider.records:
                payload = {"games": game_provider.records}
                validation_errors["historical_games_json"] = validate_games_payload(payload)
                self._assert_valid("historical_games_json", validation_errors["historical_games_json"])
                out["historical_games_json"] = self._write(f"historical_games_{year}.json", payload)
                provider_lineage["historical_games_json"] = game_provider.provider

                advanced_metrics = self.adv_builder.build(game_provider.records)
                if "kenpom_json" not in out:
                    validation_errors["kenpom_json"] = validate_ratings_payload(advanced_metrics)
                    self._assert_valid("kenpom_json", validation_errors["kenpom_json"])
                    out["kenpom_json"] = self._write(f"kenpom_{year}.json", advanced_metrics)
                    provider_lineage["kenpom_json"] = game_provider.provider

        if self.config.scrape_shotquality:
            sq_scraper = ShotQualityScraper(str(self.cache_dir))
            sq_teams = sq_scraper.fetch_team_metrics(year)
            if sq_teams:
                sq_teams_payload = {
                    "teams": [
                        {
                            "team_id": row.team_id,
                            "team_name": row.team_name,
                            "offensive_xp_per_possession": row.offensive_xp_per_possession,
                            "defensive_xp_per_possession": row.defensive_xp_per_possession,
                            "rim_rate": row.rim_rate,
                            "three_rate": row.three_rate,
                            "midrange_rate": row.midrange_rate,
                        }
                        for row in sq_teams
                    ]
                }
                validation_errors["shotquality_teams_json"] = validate_ratings_payload(
                    sq_teams_payload, name_field="team_name"
                )
                self._assert_valid("shotquality_teams_json", validation_errors["shotquality_teams_json"])
                out["shotquality_teams_json"] = self._write(f"shotquality_teams_{year}.json", sq_teams_payload)
                provider_lineage["shotquality_teams_json"] = "shotquality"

            sq_games = sq_scraper.fetch_games(year)
            if sq_games:
                sq_games_payload = {"games": sq_games}
                validation_errors["shotquality_games_json"] = validate_games_payload(sq_games_payload)
                self._assert_valid("shotquality_games_json", validation_errors["shotquality_games_json"])
                out["shotquality_games_json"] = self._write(f"shotquality_games_{year}.json", sq_games_payload)
                provider_lineage["shotquality_games_json"] = "shotquality"

        if self.config.scrape_sports_reference:
            sr_provider = self.providers.fetch_team_box_metrics(
                year,
                priority=self.config.team_metrics_provider_priority,
            )
            if sr_provider.records:
                payload = {"teams": self._ensure_ids(sr_provider.records)}
                validation_errors["sports_reference_json"] = validate_ratings_payload(payload, name_field="team_name")
                self._assert_valid("sports_reference_json", validation_errors["sports_reference_json"])
                out["sports_reference_json"] = self._write(f"sports_reference_{year}.json", payload)
                provider_lineage["sports_reference_json"] = sr_provider.provider
                sr = []
            else:
                sr = SportsReferenceScraper(str(self.cache_dir)).fetch_team_season_stats(year)
            if sr:
                payload = {"teams": self._ensure_ids(sr)}
                validation_errors["sports_reference_json"] = validate_ratings_payload(payload, name_field="team_name")
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

        manifest = {
            "year": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": out,
            "providers": provider_lineage,
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

    def _assert_valid(self, artifact_name: str, errors: List[str]) -> None:
        if errors and self.config.strict_validation:
            raise ValueError(f"{artifact_name} validation failed: {errors[:5]}")

    @staticmethod
    def _normalize_team_id(name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in (name or "")).strip("_")

    def _ensure_ids(self, records: List[Dict]) -> List[Dict]:
        result = []
        for row in records:
            if not isinstance(row, dict):
                continue
            if not row.get("team_name") and row.get("name"):
                row["team_name"] = row["name"]
            if not row.get("name") and row.get("team_name"):
                row["name"] = row["team_name"]
            team_id = row.get("team_id")
            if not team_id:
                team_id = self._normalize_team_id(row.get("team_name") or row.get("name"))
                row["team_id"] = team_id
            result.append(row)
        return result
