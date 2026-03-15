"""Selection Sunday snapshot integrity enforcement.

Validates that a Selection Sunday snapshot contains only information
that would have been available at the moment the bracket was released.

Allowed inputs:
- Regular-season game stats with date < selection_sunday
- Conference tournament results (if conf_tournament_end < selection_sunday)
- Ratings published before bracket release
- Roster information known at the time

Forbidden inputs:
- Tournament game results
- Post-selection injuries/news
- Retrospective metric revisions (end-of-season recalculations)
- Full-season aggregated team_metrics (must use incremental engine)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selection Sunday dates
# ---------------------------------------------------------------------------

# Hardcoded historical Selection Sunday dates (verified from NCAA records).
# Selection Sunday is the day the bracket is announced.
SELECTION_SUNDAY_DATES: Dict[int, date] = {
    2008: date(2008, 3, 16),
    2009: date(2009, 3, 15),
    2010: date(2010, 3, 14),
    2011: date(2011, 3, 13),
    2012: date(2012, 3, 11),
    2013: date(2013, 3, 17),
    2014: date(2014, 3, 16),
    2015: date(2015, 3, 15),
    2016: date(2016, 3, 13),
    2017: date(2017, 3, 12),
    2018: date(2018, 3, 11),
    2019: date(2019, 3, 17),
    2021: date(2021, 3, 14),
    2022: date(2022, 3, 13),
    2023: date(2023, 3, 12),
    2024: date(2024, 3, 17),
    2025: date(2025, 3, 16),
    2026: date(2026, 3, 15),
}


def get_selection_sunday(year: int) -> date:
    """Return the Selection Sunday date for a given year.

    Uses hardcoded dates where available, otherwise derives from
    TOURNAMENT_START_DATES (tournament_start - 5 days).
    """
    if year in SELECTION_SUNDAY_DATES:
        return SELECTION_SUNDAY_DATES[year]

    # Fallback: derive from tournament start dates
    try:
        from ..pipeline.config import TOURNAMENT_START_DATES
        if year in TOURNAMENT_START_DATES:
            return TOURNAMENT_START_DATES[year] - timedelta(days=5)
    except ImportError:
        pass

    raise ValueError(
        f"No Selection Sunday date available for {year}. "
        f"Add it to SELECTION_SUNDAY_DATES or TOURNAMENT_START_DATES."
    )


def selection_sunday_as_datetime(year: int) -> datetime:
    """Return Selection Sunday as a datetime (end of day, conservative)."""
    d = get_selection_sunday(year)
    return datetime(d.year, d.month, d.day, 23, 59, 59)


# ---------------------------------------------------------------------------
# Integrity violation types
# ---------------------------------------------------------------------------

class SnapshotIntegrityError(RuntimeError):
    """Raised when a snapshot contains data that violates Selection Sunday cutoff."""

    def __init__(self, violations: List[SnapshotViolation]):
        self.violations = violations
        msg = f"{len(violations)} snapshot integrity violation(s):\n"
        for v in violations[:10]:
            msg += f"  - {v}\n"
        if len(violations) > 10:
            msg += f"  ... and {len(violations) - 10} more\n"
        super().__init__(msg)


@dataclass(frozen=True)
class SnapshotViolation:
    """A single integrity violation in a snapshot."""

    source: str
    description: str
    timestamp: Optional[datetime] = None
    cutoff: Optional[datetime] = None

    def __str__(self) -> str:
        parts = [f"[{self.source}] {self.description}"]
        if self.timestamp and self.cutoff:
            parts.append(f"(timestamp={self.timestamp.isoformat()}, cutoff={self.cutoff.isoformat()})")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Data source descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSourceRecord:
    """Metadata for a single data source used in a snapshot."""

    name: str
    source_type: str  # "games", "metrics", "ratings", "roster", "bracket", "seeds"
    timestamp: Optional[datetime] = None
    record_count: int = 0
    file_path: str = ""


# ---------------------------------------------------------------------------
# Snapshot metadata
# ---------------------------------------------------------------------------

@dataclass
class SnapshotMetadata:
    """Metadata describing how a snapshot was constructed."""

    year: int
    selection_sunday: date
    cutoff_datetime: datetime
    data_sources: List[DataSourceRecord] = field(default_factory=list)
    games_included: int = 0
    games_excluded: int = 0
    teams_count: int = 0
    feature_hash: str = ""
    features_computed_after_filter: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "selection_sunday": self.selection_sunday.isoformat(),
            "cutoff_datetime": self.cutoff_datetime.isoformat(),
            "data_sources": [
                {
                    "name": ds.name,
                    "source_type": ds.source_type,
                    "timestamp": ds.timestamp.isoformat() if ds.timestamp else None,
                    "record_count": ds.record_count,
                    "file_path": ds.file_path,
                }
                for ds in self.data_sources
            ],
            "games_included": self.games_included,
            "games_excluded": self.games_excluded,
            "teams_count": self.teams_count,
            "feature_hash": self.feature_hash,
            "features_computed_after_filter": self.features_computed_after_filter,
        }


# ---------------------------------------------------------------------------
# Snapshot integrity validator
# ---------------------------------------------------------------------------

# Data source types that are FORBIDDEN in a Selection Sunday snapshot.
FORBIDDEN_SOURCE_TYPES: Set[str] = {
    "tournament_results",
    "post_selection_injury",
    "retrospective_revision",
    "end_of_season_recalculation",
}


class SnapshotIntegrityValidator:
    """Validates that a snapshot contains only Selection-Sunday-available data."""

    def __init__(self, year: int):
        self.year = year
        self.selection_sunday = get_selection_sunday(year)
        self.cutoff = selection_sunday_as_datetime(year)

    def validate(self, metadata: SnapshotMetadata) -> List[SnapshotViolation]:
        """Check a snapshot's metadata for integrity violations.

        Returns:
            List of violations (empty = passed).

        Raises:
            SnapshotIntegrityError if fail_on_violation is True and violations found.
        """
        violations: List[SnapshotViolation] = []

        # Check year matches
        if metadata.year != self.year:
            violations.append(SnapshotViolation(
                source="metadata",
                description=f"Snapshot year {metadata.year} != validator year {self.year}",
            ))

        # Check each data source
        for ds in metadata.data_sources:
            violations.extend(self._check_data_source(ds))

        # Check feature computation order
        if not metadata.features_computed_after_filter:
            violations.append(SnapshotViolation(
                source="feature_computation",
                description=(
                    "Features must be computed AFTER timestamp filtering. "
                    "Set features_computed_after_filter=True after verifying "
                    "IncrementalMetricsEngine received only pre-cutoff games."
                ),
            ))

        return violations

    def validate_or_raise(self, metadata: SnapshotMetadata) -> None:
        """Validate and raise SnapshotIntegrityError if violations found."""
        violations = self.validate(metadata)
        if violations:
            raise SnapshotIntegrityError(violations)

    def _check_data_source(self, ds: DataSourceRecord) -> List[SnapshotViolation]:
        """Check a single data source for violations."""
        violations: List[SnapshotViolation] = []

        # Forbidden source types
        if ds.source_type in FORBIDDEN_SOURCE_TYPES:
            violations.append(SnapshotViolation(
                source=ds.name,
                description=f"Forbidden source type: {ds.source_type}",
            ))

        # Timestamp check: source data must predate Selection Sunday
        if ds.timestamp is not None and ds.timestamp > self.cutoff:
            violations.append(SnapshotViolation(
                source=ds.name,
                description=f"Data source timestamp after Selection Sunday cutoff",
                timestamp=ds.timestamp,
                cutoff=self.cutoff,
            ))

        return violations

    def check_game_dates(
        self,
        game_dates: List[date],
    ) -> List[SnapshotViolation]:
        """Check that all game dates are before Selection Sunday.

        Args:
            game_dates: List of game dates from the snapshot.

        Returns:
            List of violations for any games on or after Selection Sunday.
        """
        violations: List[SnapshotViolation] = []
        for gd in game_dates:
            if gd >= self.selection_sunday:
                violations.append(SnapshotViolation(
                    source="games",
                    description=f"Game on {gd.isoformat()} is on/after Selection Sunday {self.selection_sunday.isoformat()}",
                    timestamp=datetime(gd.year, gd.month, gd.day),
                    cutoff=self.cutoff,
                ))
        return violations

    def check_tournament_game_exclusion(
        self,
        round_names: List[str],
    ) -> List[SnapshotViolation]:
        """Check that no tournament round games are in the snapshot."""
        tournament_rounds = {"R64", "R32", "S16", "E8", "F4", "NCG", "CHAMP",
                             "Round of 64", "Round of 32", "Sweet 16",
                             "Elite 8", "Final Four", "Championship"}
        violations: List[SnapshotViolation] = []
        for rn in round_names:
            if rn in tournament_rounds:
                violations.append(SnapshotViolation(
                    source="games",
                    description=f"Tournament game (round={rn}) found in snapshot",
                ))
        return violations


# ---------------------------------------------------------------------------
# Feature hash computation
# ---------------------------------------------------------------------------

def compute_feature_hash(feature_matrix_bytes: bytes) -> str:
    """Compute a SHA-256 hash of the feature matrix for lineage tracking."""
    return hashlib.sha256(feature_matrix_bytes).hexdigest()[:16]
