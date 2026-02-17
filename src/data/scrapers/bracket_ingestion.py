"""
Selection Sunday bracket ingestion pipeline.

Pulls the official 68-team NCAA tournament bracket from multiple sources
and produces a unified bracket JSON that the prediction pipeline consumes.

Sources (in priority order):
1. bigdance package (Warren Nolan data, available immediately after Selection Sunday)
2. Sports Reference HTML scraper (existing TournamentSeedScraper)
3. Manual JSON file (user-provided fallback)

All team names are resolved through TeamNameResolver for cross-source consistency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..team_name_resolver import TeamNameResolver, MatchResult

logger = logging.getLogger(__name__)

try:
    from bigdance.wn_cbb_scraper import Standings
    from bigdance.cbb_brackets import Team as BDTeam
    from bigdance.bigdance_integration import create_teams_from_standings

    BIGDANCE_AVAILABLE = True
except ImportError:
    BIGDANCE_AVAILABLE = False

try:
    from .tournament_bracket import TournamentSeedScraper

    SPORTS_REF_AVAILABLE = True
except ImportError:
    SPORTS_REF_AVAILABLE = False


@dataclass
class BracketTeam:
    """A team in the tournament bracket."""

    canonical_id: str
    display_name: str
    seed: int
    region: str
    rating: float = 0.0
    conference: str = ""
    source: str = ""
    name_confidence: float = 1.0


@dataclass
class TournamentBracketData:
    """Complete tournament bracket data."""

    season: int
    teams: List[BracketTeam] = field(default_factory=list)
    source: str = ""
    fetched_at: str = ""
    resolution_warnings: List[str] = field(default_factory=list)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    def get_first_round_matchups(self) -> List[Tuple[BracketTeam, BracketTeam]]:
        """Generate first-round matchups from seeds (1v16, 2v15, etc.)."""
        matchups = []
        for region in ("East", "West", "South", "Midwest"):
            region_teams = sorted(
                [t for t in self.teams if t.region == region],
                key=lambda t: t.seed,
            )
            seed_map = {t.seed: t for t in region_teams}
            for high, low in [(1, 16), (2, 15), (3, 14), (4, 13),
                              (5, 12), (6, 11), (7, 10), (8, 9)]:
                t_high = seed_map.get(high)
                t_low = seed_map.get(low)
                if t_high and t_low:
                    matchups.append((t_high, t_low))
        return matchups

    def to_pipeline_json(self) -> Dict:
        """Convert to the JSON format expected by SOTAPipeline."""
        return {
            "season": self.season,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "teams": [
                {
                    "season": self.season,
                    "team_name": t.display_name,
                    "team_id": t.canonical_id,
                    "seed": t.seed,
                    "region": t.region,
                    "rating": t.rating,
                    "conference": t.conference,
                    "school_slug": t.canonical_id.replace("_", "-"),
                }
                for t in self.teams
            ],
        }

    def to_espn_bracket(self) -> Dict:
        """
        Convert bracket to ESPN Tournament Challenge pick format.

        Returns a dict mapping round -> list of picked winners.
        The user fills in predicted winners; this provides the initial
        bracket structure for the prediction pipeline to populate.
        """
        matchups = self.get_first_round_matchups()
        return {
            "season": self.season,
            "format": "espn_tournament_challenge",
            "regions": {
                region: {
                    "teams": [
                        {"seed": t.seed, "name": t.display_name, "id": t.canonical_id}
                        for t in sorted(
                            [t for t in self.teams if t.region == region],
                            key=lambda t: t.seed,
                        )
                    ]
                }
                for region in ("East", "West", "South", "Midwest")
            },
            "first_round_matchups": [
                {
                    "higher_seed": {"seed": m[0].seed, "name": m[0].display_name, "id": m[0].canonical_id},
                    "lower_seed": {"seed": m[1].seed, "name": m[1].display_name, "id": m[1].canonical_id},
                    "region": m[0].region,
                }
                for m in matchups
            ],
        }


class BracketIngestionPipeline:
    """
    Fetches and normalizes the NCAA tournament bracket from available sources.

    Usage:
        pipeline = BracketIngestionPipeline(season=2026)
        bracket = pipeline.fetch()
        bracket_json = bracket.to_pipeline_json()

        # Save for the prediction pipeline
        pipeline.save(bracket, "data/raw/bracket_2026.json")

        # Export for ESPN picks
        espn = bracket.to_espn_bracket()
    """

    def __init__(
        self,
        season: int = 2026,
        cache_dir: str = "data/raw",
        resolver: Optional[TeamNameResolver] = None,
    ):
        self.season = season
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resolver = resolver or TeamNameResolver()

    def fetch(self, source: str = "auto") -> TournamentBracketData:
        """
        Fetch bracket data from the best available source.

        Args:
            source: "auto" (try bigdance first, then sports_reference),
                    "bigdance", "sports_reference", or path to a JSON file.

        Returns:
            TournamentBracketData with resolved team names
        """
        if source == "auto":
            # Try bigdance first (fastest, most reliable post-Selection Sunday)
            if BIGDANCE_AVAILABLE:
                try:
                    return self._fetch_bigdance()
                except Exception as e:
                    logger.warning("bigdance fetch failed: %s", e)

            # Fall back to Sports Reference
            if SPORTS_REF_AVAILABLE:
                try:
                    return self._fetch_sports_reference()
                except Exception as e:
                    logger.warning("Sports Reference fetch failed: %s", e)

            raise RuntimeError(
                "No bracket source available. Install bigdance (`pip install bigdance`) "
                "or provide a manual bracket JSON file."
            )

        elif source == "bigdance":
            return self._fetch_bigdance()
        elif source == "sports_reference":
            return self._fetch_sports_reference()
        else:
            # Assume it's a file path
            return self._load_manual_json(source)

    def _fetch_bigdance(self) -> TournamentBracketData:
        """Fetch bracket via bigdance package (Warren Nolan data)."""
        if not BIGDANCE_AVAILABLE:
            raise ImportError("bigdance package not installed")

        standings = Standings(season=self.season)
        bracket = create_teams_from_standings(standings)

        teams = []
        warnings = []

        for bd_team in bracket.teams:
            result = self.resolver.resolve(bd_team.name)
            if result.confidence < 0.85:
                warnings.append(
                    f"Low confidence ({result.confidence:.2f}) for '{bd_team.name}' "
                    f"-> '{result.canonical_id}' via {result.method}"
                )

            teams.append(
                BracketTeam(
                    canonical_id=result.canonical_id,
                    display_name=result.display_name,
                    seed=bd_team.seed,
                    region=bd_team.region,
                    rating=bd_team.rating,
                    conference=getattr(bd_team, "conference", ""),
                    source="bigdance",
                    name_confidence=result.confidence,
                )
            )

        return TournamentBracketData(
            season=self.season,
            teams=teams,
            source="bigdance",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            resolution_warnings=warnings,
        )

    def _fetch_sports_reference(self) -> TournamentBracketData:
        """Fetch bracket via Sports Reference HTML scraper."""
        if not SPORTS_REF_AVAILABLE:
            raise ImportError("TournamentSeedScraper not available")

        scraper = TournamentSeedScraper(cache_dir=str(self.cache_dir))
        raw_teams = scraper.fetch_tournament_seeds(self.season)

        teams = []
        warnings = []

        for raw in raw_teams:
            name = raw.get("team_name", "")
            slug = raw.get("school_slug", "")

            # Try slug first (more reliable than display name)
            result = self.resolver.resolve(slug)
            if result.confidence < 0.90:
                # Fall back to display name
                result = self.resolver.resolve(name)

            if result.confidence < 0.85:
                warnings.append(
                    f"Low confidence ({result.confidence:.2f}) for '{name}' "
                    f"(slug={slug}) -> '{result.canonical_id}' via {result.method}"
                )

            teams.append(
                BracketTeam(
                    canonical_id=result.canonical_id,
                    display_name=result.display_name,
                    seed=raw.get("seed", 0),
                    region=raw.get("region", ""),
                    source="sports_reference",
                    name_confidence=result.confidence,
                )
            )

        return TournamentBracketData(
            season=self.season,
            teams=teams,
            source="sports_reference",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            resolution_warnings=warnings,
        )

    def _load_manual_json(self, path: str) -> TournamentBracketData:
        """Load bracket from a user-provided JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        raw_teams = data.get("teams", [])
        teams = []
        warnings = []

        for raw in raw_teams:
            name = raw.get("team_name", raw.get("name", ""))
            result = self.resolver.resolve(name)

            if result.confidence < 0.85:
                warnings.append(
                    f"Low confidence ({result.confidence:.2f}) for '{name}' "
                    f"-> '{result.canonical_id}' via {result.method}"
                )

            teams.append(
                BracketTeam(
                    canonical_id=result.canonical_id,
                    display_name=result.display_name,
                    seed=raw.get("seed", 0),
                    region=raw.get("region", ""),
                    rating=raw.get("rating", 0.0),
                    conference=raw.get("conference", ""),
                    source="manual_json",
                    name_confidence=result.confidence,
                )
            )

        return TournamentBracketData(
            season=data.get("season", self.season),
            teams=teams,
            source=f"manual:{path}",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            resolution_warnings=warnings,
        )

    def save(self, bracket: TournamentBracketData, path: Optional[str] = None) -> str:
        """Save bracket to JSON for pipeline consumption."""
        if path is None:
            path = str(self.cache_dir / f"bracket_{self.season}.json")

        output = bracket.to_pipeline_json()
        output["resolution_warnings"] = bracket.resolution_warnings

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        return path
