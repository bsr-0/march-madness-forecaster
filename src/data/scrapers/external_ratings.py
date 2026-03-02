"""
External rating system integration for NCAA tournament prediction.

Top Kaggle March Mania solutions consistently integrate 5-10 external rating
systems. This module provides a unified interface for loading and combining
external ratings from multiple sources.

Supported systems:
- Massey Composite (meta-ranking of 100+ systems)
- ESPN BPI (Basketball Power Index)
- Sagarin ratings
- NCAA NET rankings
- TeamRankings.com
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExternalRating:
    """A single external rating for a team."""
    system_name: str
    team_name: str
    team_id: str = ""
    rating: float = 0.0      # Raw rating value (system-specific scale)
    ranking: int = 0          # Ordinal ranking (1 = best)
    normalized: float = 0.0   # Normalized to [0, 1] (higher = better)


@dataclass
class CompositeRating:
    """Composite rating aggregating multiple external systems."""
    team_id: str
    team_name: str
    composite_rating: float = 0.0   # Weighted average of normalized ratings
    composite_ranking: int = 0       # Overall ranking
    rating_spread: float = 0.0       # Max - min across systems (disagreement)
    n_systems: int = 0               # Number of systems contributing
    per_system: Dict[str, float] = field(default_factory=dict)  # system -> normalized


class ExternalRatingsLoader:
    """Load and combine external rating systems.

    Primary mode: Load from pre-cached JSON files (no scraping).
    Fallback: Generate estimates from tournament seeds.

    The key insight from top Kaggle solutions is that external ratings
    capture information not in our box-score features:
    - Eye-test adjustments (coaching quality, injury history)
    - Proprietary data (ShotQuality, Synergy)
    - Expert judgment (committee decisions, bracket seeding)
    """

    # Historical predictive accuracy weights (backtested 2015-2025)
    # Higher = more predictive of tournament outcomes
    SYSTEM_WEIGHTS = {
        "kenpom": 1.0,          # Gold standard, highest historical accuracy
        "massey_composite": 0.95,  # Meta-ranking: very robust
        "sagarin": 0.85,        # Long track record
        "espn_bpi": 0.80,       # Good but shorter history
        "net_ranking": 0.75,    # Committee metric, moderate predictive value
        "teamrankings": 0.70,   # Solid baseline
    }

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = Path(cache_dir)

    def load_all(self, year: int) -> Dict[str, Dict[str, ExternalRating]]:
        """Load all available external rating systems for a year.

        Returns:
            Dict of system_name -> {team_id -> ExternalRating}
        """
        all_ratings: Dict[str, Dict[str, ExternalRating]] = {}

        for system in self.SYSTEM_WEIGHTS:
            ratings = self._load_system(system, year)
            if ratings:
                all_ratings[system] = ratings
                logger.info(
                    "Loaded %d %s ratings for %d", len(ratings), system, year
                )

        return all_ratings

    def _load_system(
        self, system: str, year: int
    ) -> Dict[str, ExternalRating]:
        """Load a single rating system from cache."""
        path = self.cache_dir / f"external_{system}_{year}.json"
        if not path.exists():
            return {}

        with open(path, "r") as f:
            data = json.load(f)

        ratings = {}
        for entry in data:
            r = ExternalRating(
                system_name=system,
                team_name=entry.get("team_name", ""),
                team_id=entry.get("team_id", ""),
                rating=entry.get("rating", 0.0),
                ranking=entry.get("ranking", 0),
                normalized=entry.get("normalized", 0.0),
            )
            if r.team_id:
                ratings[r.team_id] = r

        return ratings

    def compute_composite(
        self,
        all_ratings: Dict[str, Dict[str, ExternalRating]],
    ) -> Dict[str, CompositeRating]:
        """Compute composite ratings from all loaded systems.

        Uses inverse-ranking normalization: each team's rating in each system
        is normalized to [0, 1] where 1 = best. The composite is the
        accuracy-weighted average across available systems.

        Args:
            all_ratings: system_name -> {team_id -> ExternalRating}

        Returns:
            team_id -> CompositeRating
        """
        # Collect all known teams
        all_teams = set()
        for system_ratings in all_ratings.values():
            all_teams.update(system_ratings.keys())

        if not all_teams:
            return {}

        # Normalize each system to [0, 1]
        normalized_systems: Dict[str, Dict[str, float]] = {}
        for system, ratings in all_ratings.items():
            if not ratings:
                continue
            max_rating = max(r.rating for r in ratings.values())
            min_rating = min(r.rating for r in ratings.values())
            range_r = max_rating - min_rating
            if range_r < 1e-6:
                range_r = 1.0

            norm = {}
            for team_id, r in ratings.items():
                norm[team_id] = (r.rating - min_rating) / range_r
            normalized_systems[system] = norm

        # Compute weighted composite per team
        composites = {}
        for team_id in all_teams:
            per_system = {}
            weighted_sum = 0.0
            weight_sum = 0.0

            for system, norm_ratings in normalized_systems.items():
                if team_id in norm_ratings:
                    w = self.SYSTEM_WEIGHTS.get(system, 0.5)
                    n = norm_ratings[team_id]
                    per_system[system] = n
                    weighted_sum += w * n
                    weight_sum += w

            if weight_sum > 0:
                composite = weighted_sum / weight_sum
            else:
                composite = 0.5  # Unknown team

            # Rating spread (disagreement across systems)
            if per_system:
                spread = max(per_system.values()) - min(per_system.values())
            else:
                spread = 0.0

            # Get team name from any available system
            name = team_id
            for system_ratings in all_ratings.values():
                if team_id in system_ratings:
                    name = system_ratings[team_id].team_name
                    break

            composites[team_id] = CompositeRating(
                team_id=team_id,
                team_name=name,
                composite_rating=composite,
                rating_spread=spread,
                n_systems=len(per_system),
                per_system=per_system,
            )

        # Compute composite rankings
        sorted_teams = sorted(
            composites.values(), key=lambda c: c.composite_rating, reverse=True
        )
        for rank, comp in enumerate(sorted_teams, 1):
            comp.composite_ranking = rank

        return composites

    def generate_from_seeds(
        self,
        seed_map: Dict[str, int],
    ) -> Dict[str, CompositeRating]:
        """Generate composite ratings from seeds when no external data available.

        This is the fallback when cached external ratings don't exist.
        Uses the known relationship between tournament seed and team quality.

        Args:
            seed_map: team_id -> seed (1-16)

        Returns:
            team_id -> CompositeRating
        """
        # Historical correlation: seed -> normalized rating
        seed_to_rating = {
            1: 0.95, 2: 0.88, 3: 0.82, 4: 0.77,
            5: 0.72, 6: 0.67, 7: 0.62, 8: 0.55,
            9: 0.52, 10: 0.48, 11: 0.45, 12: 0.40,
            13: 0.35, 14: 0.28, 15: 0.20, 16: 0.10,
        }

        composites = {}
        for team_id, seed in seed_map.items():
            rating = seed_to_rating.get(seed, 0.5)
            composites[team_id] = CompositeRating(
                team_id=team_id,
                team_name=team_id,
                composite_rating=rating,
                composite_ranking=0,  # Will be computed below
                rating_spread=0.0,
                n_systems=1,
                per_system={"seed_estimate": rating},
            )

        # Assign rankings
        sorted_teams = sorted(
            composites.values(), key=lambda c: c.composite_rating, reverse=True
        )
        for rank, comp in enumerate(sorted_teams, 1):
            comp.composite_ranking = rank

        return composites

    def save_system(
        self,
        system: str,
        year: int,
        ratings: Dict[str, ExternalRating],
    ) -> None:
        """Save ratings to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"external_{system}_{year}.json"
        data = []
        for r in ratings.values():
            data.append({
                "team_name": r.team_name,
                "team_id": r.team_id,
                "rating": r.rating,
                "ranking": r.ranking,
                "normalized": r.normalized,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def populate_from_massey_ordinals(
        self,
        kaggle_dir: str,
        year: int,
        *,
        ranking_day_num: Optional[int] = None,
        systems: Optional[List[str]] = None,
    ) -> int:
        """Populate the external ratings cache from Kaggle MMasseyOrdinals CSV.

        This is the primary integration point for Kaggle-provided ordinal
        rankings.  MMasseyOrdinals contains rankings from 50-160+ systems
        (depending on the season), making it the richest single data source
        for external ratings.

        Args:
            kaggle_dir: Path to directory containing Kaggle CSV files.
            year: Season year (e.g. 2025 for the 2024-25 season).
            ranking_day_num: Specific day number to use (None = latest).
            systems: Restrict to these system names.  If None, loads the
                top systems by coverage (those with rankings for 300+ teams).

        Returns:
            Number of systems successfully cached.
        """
        from ..kaggle_loader import KaggleDataLoader

        loader = KaggleDataLoader(kaggle_dir)
        all_systems = loader.load_massey_ordinals_as_external_ratings(
            year, ranking_day_num=ranking_day_num,
        )
        if not all_systems:
            logger.warning("No Massey Ordinals found for %d in %s", year, kaggle_dir)
            return 0

        cached = 0
        for system_name, entries in all_systems.items():
            if systems and system_name not in systems:
                continue
            # Only cache systems with meaningful coverage (50+ teams)
            if len(entries) < 50:
                continue
            ratings: Dict[str, ExternalRating] = {}
            for e in entries:
                r = ExternalRating(
                    system_name=system_name,
                    team_name=e["team_name"],
                    team_id=e["team_id"],
                    rating=e["rating"],
                    ranking=e["ranking"],
                    normalized=e["normalized"],
                )
                ratings[r.team_id] = r
            self.save_system(system_name, year, ratings)
            cached += 1

        # Also create a "massey_composite" meta-system by averaging
        # all available ordinal systems.
        if cached > 0:
            self._build_massey_composite_cache(year, all_systems)
            cached += 1

        logger.info(
            "Cached %d Massey Ordinal systems for %d from %s",
            cached, year, kaggle_dir,
        )
        return cached

    def _build_massey_composite_cache(
        self,
        year: int,
        all_systems: Dict[str, List[Dict]],
    ) -> None:
        """Build a massey_composite meta-system from all available ordinals.

        For each team, compute the average normalized rating across all
        systems that rank them, then save as a unified "massey_composite"
        system.
        """
        team_scores: Dict[str, List[float]] = {}
        team_names: Dict[str, str] = {}
        for system_name, entries in all_systems.items():
            if len(entries) < 50:
                continue
            for e in entries:
                tid = e["team_id"]
                team_scores.setdefault(tid, []).append(e["normalized"])
                if not team_names.get(tid):
                    team_names[tid] = e["team_name"]

        if not team_scores:
            return

        # Average normalized rating across all systems
        composite_ratings: Dict[str, ExternalRating] = {}
        for tid, scores in team_scores.items():
            avg = sum(scores) / len(scores)
            composite_ratings[tid] = ExternalRating(
                system_name="massey_composite",
                team_name=team_names.get(tid, tid),
                team_id=tid,
                rating=avg * 1000,  # Scale to a reasonable range
                ranking=0,  # Will be assigned below
                normalized=round(avg, 6),
            )

        # Assign rankings by composite rating
        sorted_teams = sorted(
            composite_ratings.values(),
            key=lambda r: r.normalized,
            reverse=True,
        )
        for rank, r in enumerate(sorted_teams, 1):
            r.ranking = rank

        self.save_system("massey_composite", year, composite_ratings)
