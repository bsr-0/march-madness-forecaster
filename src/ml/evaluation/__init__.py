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

# Phase 4: Decision-Moment Validation Protocol
# Lazy import to avoid circular dependencies — use phase4_integration module
# for explicit imports, or access via src.evaluation directly.
try:
    from .phase4_integration import (
        SelectionSundayBacktester,
        CalibrationGate,
        EvaluationReport,
        compute_round_analysis,
    )
    _PHASE4_AVAILABLE = True
except ImportError:
    _PHASE4_AVAILABLE = False

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
    # Phase 4 (if available)
    "SelectionSundayBacktester",
    "CalibrationGate",
    "EvaluationReport",
    "compute_round_analysis",
]
