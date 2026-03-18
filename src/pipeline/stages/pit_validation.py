"""Point-in-Time (PIT) Feature Validation for LOYO Cross-Validation.

Protocol v2, Section 2: Every feature used in LOYO must be a Point-in-Time
snapshot. This module enforces that constraint.

Tier 1 (Static):     No restriction (seed, conference, preseason).
Tier 2 (Cumulative): Must use only games before tournament start date.
Tier 3 (External):   Must use Selection Sunday snapshot for that year.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Selection Sunday approximate dates by year (used for PIT enforcement)
# These are the dates by which all Tier 2/3 features must be frozen.
SELECTION_SUNDAY_DATES: Dict[int, date] = {
    2016: date(2016, 3, 13),
    2017: date(2017, 3, 12),
    2018: date(2018, 3, 11),
    2019: date(2019, 3, 17),
    # 2020: COVID — no tournament
    2021: date(2021, 3, 14),
    2022: date(2022, 3, 13),
    2023: date(2023, 3, 12),
    2024: date(2024, 3, 17),
    2025: date(2025, 3, 16),
    2026: date(2026, 3, 15),
}


class PITViolationError(Exception):
    """Raised when a feature violates the Point-in-Time protocol."""
    pass


@dataclass
class PITCheckResult:
    """Result of a PIT validation check for a single LOYO fold."""
    year: int
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tier2_features_checked: int = 0
    tier3_features_checked: int = 0


@dataclass
class FeatureTierInfo:
    """Tier classification for a single feature."""
    name: str
    tier: int  # 1, 2, or 3
    description: str = ""
    sparse: bool = False


class PITValidator:
    """Validates Point-in-Time compliance for LOYO cross-validation.

    Loads tier classifications from features/MANIFEST.yaml and checks that:
    - Tier 2 features use only pre-tournament games
    - Tier 3 features use Selection Sunday snapshot (or latest pre-tournament)
    """

    def __init__(self, manifest_path: Optional[str] = None):
        """Initialize PITValidator.

        Args:
            manifest_path: Path to features/MANIFEST.yaml.
                Defaults to features/MANIFEST.yaml relative to repo root.
        """
        if manifest_path is None:
            # Try to find manifest relative to this file's location
            repo_root = Path(__file__).resolve().parents[3]
            manifest_path = str(repo_root / "features" / "MANIFEST.yaml")

        self.manifest_path = manifest_path
        self._tier_map: Dict[str, FeatureTierInfo] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load and parse the feature MANIFEST.yaml."""
        if not os.path.exists(self.manifest_path):
            logger.warning(
                "PIT MANIFEST not found at %s. "
                "PIT enforcement will be limited to structural checks.",
                self.manifest_path,
            )
            return

        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML not installed. PIT enforcement requires PyYAML. "
                "Install with: pip install pyyaml"
            )
            return

        with open(self.manifest_path) as f:
            manifest = yaml.safe_load(f)

        if manifest is None:
            logger.warning("Empty MANIFEST.yaml at %s", self.manifest_path)
            return

        for tier_key, tier_num in [
            ("tier_1_static", 1),
            ("tier_2_cumulative", 2),
            ("tier_3_external", 3),
        ]:
            section = manifest.get(tier_key, {})
            features = section.get("features", [])
            for feat in features:
                name = feat.get("name", "")
                if name:
                    self._tier_map[name] = FeatureTierInfo(
                        name=name,
                        tier=tier_num,
                        description=feat.get("description", ""),
                        sparse=feat.get("sparse", False),
                    )

        logger.info(
            "PIT MANIFEST loaded: %d features "
            "(Tier 1: %d, Tier 2: %d, Tier 3: %d)",
            len(self._tier_map),
            sum(1 for f in self._tier_map.values() if f.tier == 1),
            sum(1 for f in self._tier_map.values() if f.tier == 2),
            sum(1 for f in self._tier_map.values() if f.tier == 3),
        )

    def get_tier(self, feature_name: str) -> Optional[int]:
        """Get the PIT tier for a feature. Returns None if unclassified."""
        info = self._tier_map.get(feature_name)
        return info.tier if info else None

    def get_unclassified_features(self, feature_names: List[str]) -> List[str]:
        """Return features that are not classified in the MANIFEST."""
        return [f for f in feature_names if f not in self._tier_map]

    def get_tier3_features(self) -> List[str]:
        """Return all Tier 3 (external rating) feature names."""
        return [f.name for f in self._tier_map.values() if f.tier == 3]

    def get_tier2_features(self) -> List[str]:
        """Return all Tier 2 (cumulative) feature names."""
        return [f.name for f in self._tier_map.values() if f.tier == 2]

    def validate_fold(
        self,
        year: int,
        feature_names: List[str],
        feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        strict: bool = True,
    ) -> PITCheckResult:
        """Validate PIT compliance for a single LOYO fold year.

        Args:
            year: The held-out year being validated.
            feature_names: Names of all features in the dataset.
            feature_metadata: Optional per-feature metadata dict, e.g.:
                {"adj_off_eff": {"latest_game_date": "2019-03-10", ...}}
                Used for temporal checks on Tier 2/3 features.
            strict: If True, raise PITViolationError on violations.

        Returns:
            PITCheckResult with pass/fail status and any violations.
        """
        result = PITCheckResult(year=year, passed=True)

        # Check for unclassified features
        unclassified = self.get_unclassified_features(feature_names)
        if unclassified:
            msg = (
                f"PIT: {len(unclassified)} features not classified in MANIFEST: "
                f"{unclassified[:5]}{'...' if len(unclassified) > 5 else ''}"
            )
            result.warnings.append(msg)
            logger.warning(msg)

        selection_sunday = SELECTION_SUNDAY_DATES.get(year)
        if selection_sunday is None:
            msg = f"PIT: No Selection Sunday date for year {year}. Cannot validate temporal bounds."
            result.warnings.append(msg)
            logger.warning(msg)
            return result

        if feature_metadata is None:
            # Without metadata, we can only do structural checks
            result.warnings.append(
                f"PIT: No feature metadata provided for year {year}. "
                "Temporal validation limited to structural checks."
            )
            return result

        # Validate Tier 2 features: latest game must be before Selection Sunday
        for fname in feature_names:
            tier = self.get_tier(fname)
            if tier is None:
                continue

            meta = feature_metadata.get(fname, {})

            if tier == 2:
                result.tier2_features_checked += 1
                latest_date_str = meta.get("latest_game_date")
                if latest_date_str:
                    try:
                        latest_date = date.fromisoformat(latest_date_str)
                    except (ValueError, TypeError):
                        continue
                    if latest_date > selection_sunday:
                        violation = (
                            f"PIT VIOLATION: Tier 2 feature '{fname}' has data from "
                            f"{latest_date} which is after Selection Sunday "
                            f"{selection_sunday} for fold year {year}."
                        )
                        result.violations.append(violation)
                        result.passed = False
                        logger.error(violation)

                    # 48-hour warning (Protocol Section 2.3)
                    elif latest_date > selection_sunday - timedelta(days=2):
                        warning = (
                            f"PIT WARNING: Tier 2 feature '{fname}' has data from "
                            f"{latest_date}, within 48h of Selection Sunday "
                            f"{selection_sunday} for year {year}."
                        )
                        result.warnings.append(warning)
                        logger.warning(warning)

            elif tier == 3:
                result.tier3_features_checked += 1
                snapshot_date_str = meta.get("snapshot_date")
                if snapshot_date_str:
                    try:
                        snapshot_date = date.fromisoformat(snapshot_date_str)
                    except (ValueError, TypeError):
                        continue
                    if snapshot_date > selection_sunday:
                        violation = (
                            f"PIT VIOLATION: Tier 3 feature '{fname}' uses snapshot from "
                            f"{snapshot_date} which is after Selection Sunday "
                            f"{selection_sunday} for fold year {year}."
                        )
                        result.violations.append(violation)
                        result.passed = False
                        logger.error(violation)

        if not result.passed and strict:
            raise PITViolationError(
                f"PIT validation failed for year {year}: "
                f"{len(result.violations)} violation(s). "
                f"First: {result.violations[0]}"
            )

        if result.passed:
            logger.info(
                "PIT validation passed for year %d: "
                "%d Tier 2 features, %d Tier 3 features checked.",
                year, result.tier2_features_checked, result.tier3_features_checked,
            )

        return result

    def validate_loyo_folds(
        self,
        years: List[int],
        feature_names: List[str],
        feature_metadata_by_year: Optional[Dict[int, Dict[str, Dict[str, Any]]]] = None,
        strict: bool = True,
    ) -> List[PITCheckResult]:
        """Validate PIT compliance across all LOYO folds.

        Args:
            years: List of held-out years (e.g., [2018, 2019, 2021, ...]).
            feature_names: Feature names used across all folds.
            feature_metadata_by_year: Per-year, per-feature metadata.
            strict: If True, raise on first violation.

        Returns:
            List of PITCheckResult, one per year.
        """
        results = []
        for year in years:
            year_meta = (
                feature_metadata_by_year.get(year)
                if feature_metadata_by_year
                else None
            )
            result = self.validate_fold(
                year=year,
                feature_names=feature_names,
                feature_metadata=year_meta,
                strict=strict,
            )
            results.append(result)

        n_passed = sum(1 for r in results if r.passed)
        logger.info(
            "PIT validation complete: %d/%d folds passed.",
            n_passed, len(results),
        )
        return results
