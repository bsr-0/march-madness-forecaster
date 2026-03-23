"""Multi-system Massey Ordinals feature extraction.

Top Kaggle March Madness competition winners use individual rating system
values as separate features rather than collapsing 160+ systems into a
single composite.  Each system (KenPom, Sagarin, AP Poll, etc.) captures
orthogonal information about team quality that gets destroyed by premature
aggregation.

This module extracts per-system normalized ratings for the top ~10 most
predictive ranking systems, plus derived agreement/disagreement metrics,
yielding 12 new features per team.

Reference: MMasseyOrdinals.csv from the Kaggle March Machine Learning Mania
competition (50-160+ systems per season, 2003-2026).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System selection
# ---------------------------------------------------------------------------
# Selected based on:
#   1. Historical coverage (300+ teams rated in most seasons)
#   2. Known predictive accuracy for tournament outcomes
#   3. Orthogonality to each other (diverse methodologies)
#   4. Availability across the 2003-2025 historical window
#
# System name mapping uses the canonical codes from MMasseyOrdinals.csv.
# ---------------------------------------------------------------------------

MASSEY_TOP_SYSTEMS: List[str] = [
    "POM",   # KenPom — efficiency-based, gold standard
    "SAG",   # Sagarin — Elo/Bayesian hybrid, long track record
    "MOR",   # Massey — mathematical, namesake of the dataset
    "DOL",   # Dolchini — reliability-adjusted ratings
    "COL",   # Colley — bias-free linear algebra method
    "WOL",   # Wolfe — schedule-adjusted power ratings
    "RTH",   # Rothman — Bayesian approach
    "AP",    # AP Poll — human expert consensus (sports writers)
    "USA",   # Coaches Poll — insider perspective
    "RPI",   # RPI — NCAA's historical selection metric
]

# Canonical feature names (used in TeamFeatures.get_feature_names)
SYSTEM_FEATURE_NAMES: List[str] = [
    f"massey_{s.lower()}" for s in MASSEY_TOP_SYSTEMS
]
DERIVED_FEATURE_NAMES: List[str] = [
    "massey_rank_mean",
    "massey_rank_std",
]
ALL_MASSEY_FEATURE_NAMES: List[str] = SYSTEM_FEATURE_NAMES + DERIVED_FEATURE_NAMES

# Total new features added by this module
MASSEY_MULTI_SYSTEM_DIM = len(ALL_MASSEY_FEATURE_NAMES)  # 12


@dataclass
class MasseyMultiSystemFeatures:
    """Per-team features extracted from individual Massey Ordinal systems.

    Attributes:
        system_ratings: Mapping of system code → normalized rating [0, 1].
            Missing systems are absent from the dict (caller should use NaN).
        rank_mean: Mean normalized rating across all available systems.
        rank_std: Std deviation of normalized ratings (inter-system disagreement).
            Higher values indicate systems disagree about team quality — a
            signal correlated with tournament upset likelihood.
        n_systems: Number of systems that rated this team.
    """
    system_ratings: Dict[str, float] = field(default_factory=dict)
    rank_mean: float = float("nan")
    rank_std: float = float("nan")
    n_systems: int = 0


def extract_multi_system_features(
    ordinals: Dict[str, Dict[str, Any]],
    team_id: str,
    n_teams_approx: int = 363,
) -> MasseyMultiSystemFeatures:
    """Extract individual system ratings + derived features for one team.

    Args:
        ordinals: Dict of {system_name: {canonical_team_id: entry}}.
            Each entry must have an ``ordinal_rank`` attribute or key.
        team_id: The canonical team ID to extract features for.
        n_teams_approx: Approximate number of teams rated (used for
            normalization when a system's team count is unavailable).

    Returns:
        MasseyMultiSystemFeatures with individual ratings and derived metrics.
    """
    system_ratings: Dict[str, float] = {}
    raw_normed: List[float] = []

    for system_code in MASSEY_TOP_SYSTEMS:
        system_data = ordinals.get(system_code)
        if system_data is None:
            continue

        entry = system_data.get(team_id)
        if entry is None:
            continue

        # Extract ordinal rank from either an object attribute or dict key
        rank = _get_rank(entry)
        if rank is None or rank <= 0:
            continue

        n_teams = max(len(system_data), n_teams_approx)
        # Normalize: rank 1 → ~1.0, rank N → ~0.0
        normalized = 1.0 - (rank - 1) / max(n_teams - 1, 1)
        normalized = max(0.0, min(1.0, normalized))

        system_ratings[system_code] = normalized
        raw_normed.append(normalized)

    if not raw_normed:
        return MasseyMultiSystemFeatures(
            system_ratings={},
            rank_mean=float("nan"),
            rank_std=float("nan"),
            n_systems=0,
        )

    mean_val = sum(raw_normed) / len(raw_normed)
    if len(raw_normed) >= 2:
        variance = sum((x - mean_val) ** 2 for x in raw_normed) / (len(raw_normed) - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0

    return MasseyMultiSystemFeatures(
        system_ratings=system_ratings,
        rank_mean=mean_val,
        rank_std=std_val,
        n_systems=len(raw_normed),
    )


def extract_all_teams(
    ordinals: Dict[str, Dict[str, Any]],
) -> Dict[str, MasseyMultiSystemFeatures]:
    """Extract multi-system features for all teams found in ordinals.

    Args:
        ordinals: Dict of {system_name: {canonical_team_id: entry}}.

    Returns:
        Dict of {canonical_team_id: MasseyMultiSystemFeatures}.
    """
    # Collect all team IDs across all systems
    all_team_ids: set = set()
    for system_data in ordinals.values():
        all_team_ids.update(system_data.keys())

    if not all_team_ids:
        return {}

    n_teams_approx = max(len(sd) for sd in ordinals.values())

    result: Dict[str, MasseyMultiSystemFeatures] = {}
    for team_id in all_team_ids:
        features = extract_multi_system_features(ordinals, team_id, n_teams_approx)
        if features.n_systems > 0:
            result[team_id] = features

    n_with_5plus = sum(1 for f in result.values() if f.n_systems >= 5)
    logger.info(
        "Massey multi-system: extracted features for %d teams "
        "(%d with 5+ systems) from %d total systems in ordinals",
        len(result),
        n_with_5plus,
        len(ordinals),
    )

    return result


def features_to_vector(features: MasseyMultiSystemFeatures) -> List[float]:
    """Convert MasseyMultiSystemFeatures to a flat list of floats.

    Returns exactly MASSEY_MULTI_SYSTEM_DIM values in the canonical order:
    10 individual system ratings + rank_mean + rank_std.
    Missing systems use NaN (tree models handle natively).
    """
    values: List[float] = []
    for system_code in MASSEY_TOP_SYSTEMS:
        val = features.system_ratings.get(system_code, float("nan"))
        values.append(val)
    values.append(features.rank_mean)
    values.append(features.rank_std)
    assert len(values) == MASSEY_MULTI_SYSTEM_DIM
    return values


def _get_rank(entry: Any) -> Optional[int]:
    """Extract ordinal rank from a MasseyOrdinalEntry or dict."""
    if hasattr(entry, "ordinal_rank"):
        return entry.ordinal_rank
    if isinstance(entry, dict):
        return entry.get("ordinal_rank") or entry.get("rank") or entry.get("ranking")
    return None
