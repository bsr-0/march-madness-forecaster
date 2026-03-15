"""Statistical evaluation and ablation tools for model comparison."""

from .loyo_protocol import (
    FeatureAblator,
    LOYOFoldResult,
    LOYOResult,
    LOYOValidator,
    LOYO_YEARS,
    MINIMUM_BRIER_IMPROVEMENT,
)

from .evaluation_integrity import (
    CANONICAL_DEV_YEARS,
    CANONICAL_HOLDOUT_YEARS,
    BracketPoolResult,
    CanonicalLeaderboard,
    FreezeRequiredError,
    HoldoutContaminationError,
    LeaderboardEntry,
    UpsetMetrics,
    YearSplitPolicy,
    build_leaderboard_entry,
    compute_bracket_pool_score,
    compute_probability_metrics,
    compute_upset_metrics,
    require_freeze_for_season,
)

__all__ = [
    "FeatureAblator",
    "LOYOFoldResult",
    "LOYOResult",
    "LOYOValidator",
    "LOYO_YEARS",
    "MINIMUM_BRIER_IMPROVEMENT",
    # Evaluation integrity (Phase 1)
    "CANONICAL_DEV_YEARS",
    "CANONICAL_HOLDOUT_YEARS",
    "BracketPoolResult",
    "CanonicalLeaderboard",
    "FreezeRequiredError",
    "HoldoutContaminationError",
    "LeaderboardEntry",
    "UpsetMetrics",
    "YearSplitPolicy",
    "build_leaderboard_entry",
    "compute_bracket_pool_score",
    "compute_probability_metrics",
    "compute_upset_metrics",
    "require_freeze_for_season",
]
