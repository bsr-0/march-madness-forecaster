"""Training utilities for March Madness prediction models."""

from .symmetric import (
    symmetric_augment,
    swap_matchup_vector,
    swap_matchup_batch,
    verify_zero_sum_property,
)

from .conference_tournament_augmentation import (  # noqa: F401
    build_conf_tourney_training_data,
    merge_with_primary_training_data,
    ConferenceAugmentationResult,
)

from .two_stage_training import (  # noqa: F401
    two_stage_train_lightgbm,
    two_stage_train_spread,
    two_stage_train_logistic,
    TwoStageConfig,
    TwoStageResult,
)

__all__ = [
    "symmetric_augment",
    "swap_matchup_vector",
    "swap_matchup_batch",
    "verify_zero_sum_property",
    "build_conf_tourney_training_data",
    "merge_with_primary_training_data",
    "ConferenceAugmentationResult",
    "two_stage_train_lightgbm",
    "two_stage_train_spread",
    "two_stage_train_logistic",
    "TwoStageConfig",
    "TwoStageResult",
]
