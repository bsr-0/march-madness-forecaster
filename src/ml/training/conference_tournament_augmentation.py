"""Conference tournament game augmentation for small-sample mitigation.

Conference tournaments (~300 games/year across 32 conferences) share the
same single-elimination format as the NCAA tournament:
  - Neutral-ish sites (some home-court advantage for higher seeds)
  - Win-or-go-home pressure
  - Seeded brackets with upsets

This makes them the closest available proxy for NCAA tournament games.
Including them as additional training data increases the effective
tournament-like sample size from ~63 NCAA games/year to ~360 games/year.

Temporal safety
---------------
Conference tournaments occur 1-2 weeks BEFORE the NCAA tournament.
For any given year, conference tournament games are temporally safe as
training data for that year's NCAA tournament predictions because:
  1. They happen before Selection Sunday
  2. Features are computed at point-in-time (PIT) using only pre-game data
  3. Conference tournament outcomes are known before NCAA bracket is set

Weighting
---------
Conference tournament games receive a configurable discount weight
(default 0.60) relative to NCAA tournament games because:
  - Opponent quality is lower (intra-conference, not national)
  - Some conferences have large talent gaps (e.g., 1-seed vs 16-seed)
  - Venue is not fully neutral (proximity advantage for higher seeds)
  - Round structure varies (4-6 rounds vs NCAA's fixed 6 rounds)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default weight for conference tournament games relative to NCAA tournament
# games. 0.60 reflects the structural similarity (single-elimination) minus
# the quality gap (intra-conference vs national matchups).
CONF_TOURNEY_DEFAULT_WEIGHT = 0.60

# Conference tournaments typically start ~2 weeks before NCAA tournament.
# These are approximate start dates by year for filtering.
# Format: {year: (month, day)} — earliest conference tournament game.
CONF_TOURNEY_START_OFFSETS_DAYS = 14  # Days before NCAA tournament start

# Round-weight mapping for conference tournament games.
# Maps conference tournament round depth to an approximate NCAA-equivalent
# weight. Conference championship ≈ NCAA R64 importance (same stakes:
# win = advance to the big dance).
CONF_ROUND_WEIGHTS = {
    1: 0.5,   # Play-in / first round — low stakes, large talent gaps
    2: 0.7,   # Second round — moderate
    3: 1.0,   # Quarterfinals — competitive matchups
    4: 1.5,   # Semifinals — conference elite
    5: 2.0,   # Championship — win-or-go-home, high pressure
    6: 2.0,   # Championship (for 6-round brackets like Big Ten)
}


@dataclass
class ConferenceTournamentSample:
    """A single conference tournament training sample."""
    year: int
    conference: str
    round_num: int
    team1_id: str
    team2_id: str
    feature_vector: np.ndarray
    label: int  # 1 = team1 wins, 0 = team2 wins
    margin: float  # team1_score - team2_score
    weight: float = 1.0


@dataclass
class ConferenceAugmentationResult:
    """Result of conference tournament augmentation."""
    X: np.ndarray
    y: np.ndarray
    margins: Optional[np.ndarray]
    sample_weights: np.ndarray
    n_games: int
    n_conferences: int
    years_included: List[int]
    round_distribution: Dict[int, int] = field(default_factory=dict)


def compute_conf_tourney_weight(
    round_num: int,
    base_weight: float = CONF_TOURNEY_DEFAULT_WEIGHT,
    apply_round_scaling: bool = True,
) -> float:
    """Compute sample weight for a conference tournament game.

    Args:
        round_num: Round number within the conference tournament (1-6).
        base_weight: Base weight relative to NCAA tournament games.
        apply_round_scaling: If True, scale by round importance.

    Returns:
        Combined sample weight.
    """
    if apply_round_scaling:
        round_w = CONF_ROUND_WEIGHTS.get(round_num, 1.0)
        return base_weight * round_w
    return base_weight


def build_conf_tourney_training_data(
    games: List[Dict],
    feature_dim: int,
    base_weight: float = CONF_TOURNEY_DEFAULT_WEIGHT,
    apply_round_scaling: bool = True,
    min_feature_completeness: float = 0.30,
) -> Optional[ConferenceAugmentationResult]:
    """Build training matrices from conference tournament game records.

    Accepts pre-computed game dicts with feature vectors already generated
    at point-in-time. This function does NOT compute features — it expects
    them as input, ensuring PIT safety is handled by the caller.

    Args:
        games: List of game dicts, each containing:
            - "feature_vector": np.ndarray of shape (feature_dim,)
            - "label": int (1 = team1 win, 0 = team2 win)
            - "margin": float (team1_score - team2_score)
            - "year": int
            - "conference": str
            - "round": int (conference tournament round number)
            - "team1_id": str (optional, for logging)
            - "team2_id": str (optional, for logging)
        feature_dim: Expected feature vector dimension.
        base_weight: Base weight for conference tournament samples.
        apply_round_scaling: Scale weights by round importance.
        min_feature_completeness: Minimum fraction of non-zero features.

    Returns:
        ConferenceAugmentationResult or None if no valid games.
    """
    if not games:
        logger.warning("No conference tournament games provided for augmentation")
        return None

    valid_X = []
    valid_y = []
    valid_margins = []
    valid_weights = []
    round_dist: Dict[int, int] = {}
    conferences_seen = set()
    years_seen = set()
    skipped = 0

    for game in games:
        fv = game.get("feature_vector")
        if fv is None:
            skipped += 1
            continue

        fv = np.asarray(fv, dtype=np.float64)

        # Dimension check — pad or truncate to match expected dim
        if fv.shape[0] < feature_dim:
            padded = np.zeros(feature_dim, dtype=np.float64)
            padded[:fv.shape[0]] = fv
            fv = padded
        elif fv.shape[0] > feature_dim:
            fv = fv[:feature_dim]

        # Feature completeness check
        completeness = float(np.mean(np.abs(fv) > 1e-8))
        if completeness < min_feature_completeness:
            skipped += 1
            continue

        # NaN/Inf check
        if np.any(np.isnan(fv)) or np.any(np.isinf(fv)):
            skipped += 1
            continue

        label = int(game.get("label", 0))
        margin = float(game.get("margin", 0.0))
        round_num = int(game.get("round", 3))
        year = int(game.get("year", 0))
        conference = str(game.get("conference", "UNK"))

        weight = compute_conf_tourney_weight(round_num, base_weight, apply_round_scaling)

        valid_X.append(fv)
        valid_y.append(label)
        valid_margins.append(margin)
        valid_weights.append(weight)
        round_dist[round_num] = round_dist.get(round_num, 0) + 1
        conferences_seen.add(conference)
        years_seen.add(year)

    if not valid_X:
        logger.warning(
            "No valid conference tournament games after filtering "
            "(skipped %d)", skipped
        )
        return None

    X = np.array(valid_X, dtype=np.float64)
    y = np.array(valid_y, dtype=np.float64)
    margins = np.array(valid_margins, dtype=np.float64)
    weights = np.array(valid_weights, dtype=np.float64)

    logger.info(
        "Conference tournament augmentation: %d games from %d conferences "
        "across years %s (skipped %d)",
        len(valid_X), len(conferences_seen), sorted(years_seen), skipped,
    )

    return ConferenceAugmentationResult(
        X=X,
        y=y,
        margins=margins,
        sample_weights=weights,
        n_games=len(valid_X),
        n_conferences=len(conferences_seen),
        years_included=sorted(years_seen),
        round_distribution=round_dist,
    )


def merge_with_primary_training_data(
    primary_X: np.ndarray,
    primary_y: np.ndarray,
    primary_weights: Optional[np.ndarray],
    primary_margins: Optional[np.ndarray],
    conf_result: ConferenceAugmentationResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Merge conference tournament data into primary training matrices.

    Concatenates conference tournament samples with the primary training
    data, handling weight and margin arrays correctly.

    Args:
        primary_X: Primary feature matrix [N, D].
        primary_y: Primary labels [N].
        primary_weights: Primary sample weights [N] (None = uniform 1.0).
        primary_margins: Primary margins [N] (None = no margins available).
        conf_result: Conference tournament augmentation result.

    Returns:
        Tuple of (merged_X, merged_y, merged_weights, merged_margins).
        merged_weights is always non-None (defaults to 1.0 for primary).
        merged_margins is None only if both inputs are None.
    """
    n_primary = primary_X.shape[0]
    feature_dim = primary_X.shape[1]

    # Ensure conference data matches feature dimension
    conf_X = conf_result.X
    if conf_X.shape[1] != feature_dim:
        if conf_X.shape[1] < feature_dim:
            padded = np.zeros((conf_X.shape[0], feature_dim), dtype=np.float64)
            padded[:, :conf_X.shape[1]] = conf_X
            conf_X = padded
        else:
            conf_X = conf_X[:, :feature_dim]

    # Weights: default primary to 1.0 if not provided
    if primary_weights is None:
        primary_weights = np.ones(n_primary, dtype=np.float64)

    merged_X = np.vstack([primary_X, conf_X])
    merged_y = np.concatenate([primary_y, conf_result.y])
    merged_weights = np.concatenate([primary_weights, conf_result.sample_weights])

    # Margins
    merged_margins = None
    if primary_margins is not None or conf_result.margins is not None:
        pm = primary_margins if primary_margins is not None else np.zeros(n_primary)
        cm = conf_result.margins if conf_result.margins is not None else np.zeros(conf_result.n_games)
        merged_margins = np.concatenate([pm, cm])

    logger.info(
        "Merged training data: %d primary + %d conference tournament = %d total",
        n_primary, conf_result.n_games, len(merged_y),
    )

    return merged_X, merged_y, merged_weights, merged_margins
