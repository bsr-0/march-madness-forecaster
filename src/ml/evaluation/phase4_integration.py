"""Phase 4 integration bridge.

Thin re-export layer connecting the new ``src/evaluation/`` package
with the existing ``src/ml/evaluation/`` infrastructure so that
downstream code can import Phase 4 classes from either location.
"""

from __future__ import annotations

# Core backtest engine
from ...evaluation.selection_sunday_backtest import (
    SelectionSundayBacktester,
    SelectionSundayManifest,
    SelectionSundaySnapshot,
    SnapshotBuilder,
    YearBacktestResult,
)

# Snapshot integrity
from ...evaluation.snapshot_integrity import (
    SELECTION_SUNDAY_DATES,
    DataSourceRecord,
    SnapshotIntegrityError,
    SnapshotIntegrityValidator,
    SnapshotMetadata,
    SnapshotViolation,
    get_selection_sunday,
)

# Bracket snapshot
from ...evaluation.bracket_snapshot import (
    BracketSnapshot,
    PlayInMatchup,
    TeamEntry,
    load_bracket_snapshot,
    marginalise_play_in,
)

# Round analysis
from ...evaluation.round_analysis import (
    EvalGame,
    RoundAnalysisReport,
    RoundSegmenter,
    SegmentMetrics,
    compute_round_analysis,
    compute_segment_metrics,
)

# Calibration
from ...evaluation.calibration_gate import (
    CalibrationGate,
    CalibrationGateResult,
    FrozenProbabilities,
)
from ...evaluation.calibration_diagnostics import (
    CalibrationDiagnostics,
    CalibrationReport,
    ReliabilityDiagram,
)
from ...evaluation.seed_gap_calibration import (
    SeedGapCalibrationReport,
    compute_seed_gap_calibration,
)

# Baselines
from ...evaluation.baselines import (
    BaselineComparison,
    compare_to_baseline,
    run_all_baseline_comparisons,
    seed_baseline_probability,
)

# Bootstrap metrics
from ...evaluation.bootstrap_metrics import (
    BootstrapResult,
    bootstrap_brier,
    bootstrap_ece,
    bootstrap_log_loss,
    compute_all_metrics_with_ci,
)

# Evaluation report
from ...evaluation.evaluation_report import (
    AggregateEvaluationReport,
    EvaluationReport,
    ReproducibilityInfo,
)


__all__ = [
    # Backtest
    "SelectionSundayBacktester",
    "SelectionSundayManifest",
    "SelectionSundaySnapshot",
    "SnapshotBuilder",
    "YearBacktestResult",
    # Integrity
    "SELECTION_SUNDAY_DATES",
    "DataSourceRecord",
    "SnapshotIntegrityError",
    "SnapshotIntegrityValidator",
    "SnapshotMetadata",
    "SnapshotViolation",
    "get_selection_sunday",
    # Bracket
    "BracketSnapshot",
    "PlayInMatchup",
    "TeamEntry",
    "load_bracket_snapshot",
    "marginalise_play_in",
    # Round analysis
    "EvalGame",
    "RoundAnalysisReport",
    "RoundSegmenter",
    "SegmentMetrics",
    "compute_round_analysis",
    "compute_segment_metrics",
    # Calibration
    "CalibrationGate",
    "CalibrationGateResult",
    "FrozenProbabilities",
    "CalibrationDiagnostics",
    "CalibrationReport",
    "ReliabilityDiagram",
    "SeedGapCalibrationReport",
    "compute_seed_gap_calibration",
    # Baselines
    "BaselineComparison",
    "compare_to_baseline",
    "run_all_baseline_comparisons",
    "seed_baseline_probability",
    # Bootstrap
    "BootstrapResult",
    "bootstrap_brier",
    "bootstrap_ece",
    "bootstrap_log_loss",
    "compute_all_metrics_with_ci",
    # Report
    "AggregateEvaluationReport",
    "EvaluationReport",
    "ReproducibilityInfo",
]
