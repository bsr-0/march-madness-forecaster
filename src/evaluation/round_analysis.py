"""Round-aware performance evaluation.

Global metrics (Brier, log loss) can hide weaknesses in specific matchup
zones that matter most for bracket decisions. This module segments
tournament games by round and seed matchup, computing metrics per segment
with bootstrap confidence intervals.

Segments:
- Heavy favorites: 1v16, 2v15, 3v14
- Mid-tier: 4v13, 5v12
- Toss-ups: 6v11, 7v10, 8v9
- Later rounds: Sweet 16, Elite 8, Final Four, Championship
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bootstrap_metrics import (
    BootstrapResult,
    accuracy,
    bootstrap_accuracy,
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_log_loss,
    bootstrap_metric,
    brier_score,
    compute_all_metrics_with_ci,
    expected_calibration_error,
    log_loss,
    upset_recall,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------

# Seed-matchup segments (first round only — identified by seed pair)
SEED_MATCHUP_SEGMENTS: Dict[str, List[Tuple[int, int]]] = {
    "heavy_favorites": [(1, 16), (2, 15), (3, 14)],
    "mid_tier": [(4, 13), (5, 12)],
    "toss_ups": [(6, 11), (7, 10), (8, 9)],
}

# Round-based segments (later rounds — identified by round name)
ROUND_SEGMENTS: Dict[str, List[str]] = {
    "round_of_64": ["R64"],
    "round_of_32": ["R32"],
    "sweet_16": ["S16", "Sweet 16"],
    "elite_8": ["E8", "Elite 8"],
    "final_four": ["F4", "Final Four"],
    "championship": ["NCG", "CHAMP", "Championship"],
}

# Map from round name variants to canonical names
ROUND_NAME_MAP: Dict[str, str] = {
    "R64": "R64",
    "Round of 64": "R64",
    "R32": "R32",
    "Round of 32": "R32",
    "S16": "S16",
    "Sweet 16": "S16",
    "E8": "E8",
    "Elite 8": "E8",
    "F4": "F4",
    "Final Four": "F4",
    "NCG": "NCG",
    "CHAMP": "NCG",
    "Championship": "NCG",
}


# ---------------------------------------------------------------------------
# Game record for evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalGame:
    """A tournament game with prediction and outcome for evaluation."""

    team1_id: str
    team2_id: str
    team1_seed: int
    team2_seed: int
    round_name: str
    prediction: float  # P(team1 wins)
    outcome: float  # 1.0 if team1 won, 0.0 otherwise
    year: int = 0

    @property
    def canonical_round(self) -> str:
        return ROUND_NAME_MAP.get(self.round_name, self.round_name)

    @property
    def seed_pair(self) -> Tuple[int, int]:
        """Return (lower_seed, higher_seed) for matchup categorisation."""
        return (min(self.team1_seed, self.team2_seed), max(self.team1_seed, self.team2_seed))

    @property
    def favorite_prediction(self) -> float:
        """P(favorite wins), oriented so favorite is the lower seed."""
        if self.team1_seed <= self.team2_seed:
            return self.prediction
        return 1.0 - self.prediction

    @property
    def favorite_won(self) -> bool:
        """Whether the lower-seeded (favorite) team won."""
        if self.team1_seed < self.team2_seed:
            return self.outcome == 1.0
        elif self.team1_seed > self.team2_seed:
            return self.outcome == 0.0
        return self.outcome == 1.0  # Equal seeds: team1 is "favorite"

    @property
    def is_upset(self) -> bool:
        return not self.favorite_won and self.team1_seed != self.team2_seed


# ---------------------------------------------------------------------------
# Segment metrics result
# ---------------------------------------------------------------------------


@dataclass
class SegmentMetrics:
    """Metrics for a single evaluation segment."""

    segment_name: str
    n_games: int
    brier: Optional[BootstrapResult] = None
    log_loss_val: Optional[BootstrapResult] = None
    ece: Optional[BootstrapResult] = None
    accuracy_val: Optional[BootstrapResult] = None
    upset_recall_val: Optional[float] = None
    n_upsets: int = 0
    n_upsets_predicted: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "segment_name": self.segment_name,
            "n_games": self.n_games,
            "n_upsets": self.n_upsets,
            "n_upsets_predicted": self.n_upsets_predicted,
        }
        if self.brier:
            result["brier"] = self.brier.to_dict()
        if self.log_loss_val:
            result["log_loss"] = self.log_loss_val.to_dict()
        if self.ece:
            result["ece"] = self.ece.to_dict()
        if self.accuracy_val:
            result["accuracy"] = self.accuracy_val.to_dict()
        if self.upset_recall_val is not None:
            result["upset_recall"] = self.upset_recall_val
        return result


# ---------------------------------------------------------------------------
# Round analysis report
# ---------------------------------------------------------------------------


@dataclass
class RoundAnalysisReport:
    """Complete round-aware evaluation report.

    Contains global metrics plus per-segment breakdowns with bootstrap CIs.
    """

    global_metrics: Dict[str, BootstrapResult] = field(default_factory=dict)
    seed_matchup_segments: Dict[str, SegmentMetrics] = field(default_factory=dict)
    round_segments: Dict[str, SegmentMetrics] = field(default_factory=dict)
    n_total_games: int = 0
    years_evaluated: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total_games": self.n_total_games,
            "years_evaluated": self.years_evaluated,
            "global_metrics": {k: v.to_dict() for k, v in self.global_metrics.items()},
            "seed_matchup_segments": {k: v.to_dict() for k, v in self.seed_matchup_segments.items()},
            "round_segments": {k: v.to_dict() for k, v in self.round_segments.items()},
        }


# ---------------------------------------------------------------------------
# Segmentation logic
# ---------------------------------------------------------------------------


class RoundSegmenter:
    """Categorises tournament games into evaluation segments."""

    def __init__(
        self,
        seed_matchup_segments: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        round_segments: Optional[Dict[str, List[str]]] = None,
    ):
        self.seed_segments = seed_matchup_segments or SEED_MATCHUP_SEGMENTS
        self.round_segments = round_segments or ROUND_SEGMENTS

    def segment_by_seed_matchup(
        self,
        games: List[EvalGame],
    ) -> Dict[str, List[EvalGame]]:
        """Group first-round games by seed matchup category."""
        result: Dict[str, List[EvalGame]] = {name: [] for name in self.seed_segments}

        for game in games:
            if game.canonical_round != "R64":
                continue
            pair = game.seed_pair
            for seg_name, pairs in self.seed_segments.items():
                if pair in pairs:
                    result[seg_name].append(game)
                    break

        return result

    def segment_by_round(
        self,
        games: List[EvalGame],
    ) -> Dict[str, List[EvalGame]]:
        """Group games by tournament round."""
        result: Dict[str, List[EvalGame]] = {name: [] for name in self.round_segments}

        for game in games:
            canonical = game.canonical_round
            for seg_name, round_names in self.round_segments.items():
                canonical_names = [ROUND_NAME_MAP.get(rn, rn) for rn in round_names]
                if canonical in canonical_names:
                    result[seg_name].append(game)
                    break

        return result

    def categorise_game(self, game: EvalGame) -> List[str]:
        """Return all segment names a game belongs to."""
        categories: List[str] = []

        # Round segment
        canonical = game.canonical_round
        for seg_name, round_names in self.round_segments.items():
            canonical_names = [ROUND_NAME_MAP.get(rn, rn) for rn in round_names]
            if canonical in canonical_names:
                categories.append(seg_name)
                break

        # Seed matchup segment (first round only)
        if canonical == "R64":
            pair = game.seed_pair
            for seg_name, pairs in self.seed_segments.items():
                if pair in pairs:
                    categories.append(seg_name)
                    break

        return categories


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_segment_metrics(
    games: List[EvalGame],
    segment_name: str,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> SegmentMetrics:
    """Compute all metrics for a segment of games with bootstrap CIs.

    Args:
        games: Games in this segment.
        segment_name: Name of the segment.
        n_bootstrap: Bootstrap resamples.
        ci_level: Confidence level.
        random_seed: For reproducibility.

    Returns:
        SegmentMetrics with all computed metrics.
    """
    n = len(games)
    if n == 0:
        return SegmentMetrics(segment_name=segment_name, n_games=0)

    predictions = np.array([g.prediction for g in games])
    outcomes = np.array([g.outcome for g in games])

    # Core metrics with bootstrap CIs
    brier_result = bootstrap_brier(
        predictions,
        outcomes,
        n_bootstrap,
        ci_level,
        random_seed,
    )
    ll_result = bootstrap_log_loss(
        predictions,
        outcomes,
        n_bootstrap,
        ci_level,
        random_seed + 1 if random_seed else None,
    )
    acc_result = bootstrap_accuracy(
        predictions,
        outcomes,
        n_bootstrap,
        ci_level,
        random_seed + 2 if random_seed else None,
    )
    ece_result = bootstrap_ece(
        predictions,
        outcomes,
        min(max(n // 3, 2), 10),
        n_bootstrap,
        ci_level,
        random_seed + 3 if random_seed else None,
    )

    # Upset metrics (not bootstrapped — counts)
    n_upsets = sum(1 for g in games if g.is_upset)
    # Predicted upset: model gives < 0.5 to the favorite
    n_upsets_predicted = sum(1 for g in games if g.favorite_prediction < 0.5 and g.team1_seed != g.team2_seed)
    ur = None
    if n_upsets > 0:
        correct_upset_preds = sum(1 for g in games if g.is_upset and g.favorite_prediction < 0.5)
        ur = correct_upset_preds / n_upsets

    return SegmentMetrics(
        segment_name=segment_name,
        n_games=n,
        brier=brier_result,
        log_loss_val=ll_result,
        ece=ece_result,
        accuracy_val=acc_result,
        upset_recall_val=ur,
        n_upsets=n_upsets,
        n_upsets_predicted=n_upsets_predicted,
    )


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def compute_round_analysis(
    games: List[EvalGame],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_seed: Optional[int] = 42,
) -> RoundAnalysisReport:
    """Compute the full round-aware evaluation report.

    Args:
        games: All tournament games with predictions and outcomes.
        n_bootstrap: Bootstrap resamples.
        ci_level: Confidence level.
        random_seed: For reproducibility.

    Returns:
        RoundAnalysisReport with global and per-segment metrics.
    """
    if not games:
        return RoundAnalysisReport()

    segmenter = RoundSegmenter()

    # Global metrics
    predictions = np.array([g.prediction for g in games])
    outcomes = np.array([g.outcome for g in games])
    global_metrics = compute_all_metrics_with_ci(
        predictions,
        outcomes,
        n_bootstrap,
        ci_level,
        random_seed,
    )

    # Seed matchup segments
    seed_segments = segmenter.segment_by_seed_matchup(games)
    seed_segment_metrics: Dict[str, SegmentMetrics] = {}
    for seg_name, seg_games in seed_segments.items():
        seed_segment_metrics[seg_name] = compute_segment_metrics(
            seg_games,
            seg_name,
            n_bootstrap,
            ci_level,
            random_seed + hash(seg_name) % 10000 if random_seed else None,
        )

    # Round segments
    round_segments = segmenter.segment_by_round(games)
    round_segment_metrics: Dict[str, SegmentMetrics] = {}
    for seg_name, seg_games in round_segments.items():
        round_segment_metrics[seg_name] = compute_segment_metrics(
            seg_games,
            seg_name,
            n_bootstrap,
            ci_level,
            random_seed + hash(seg_name) % 10000 if random_seed else None,
        )

    years = sorted(set(g.year for g in games if g.year > 0))

    return RoundAnalysisReport(
        global_metrics=global_metrics,
        seed_matchup_segments=seed_segment_metrics,
        round_segments=round_segment_metrics,
        n_total_games=len(games),
        years_evaluated=years,
    )
