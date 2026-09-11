"""Training utilities for March Madness prediction models."""

from .symmetric import (
    symmetric_augment,
    swap_matchup_vector,
    swap_matchup_batch,
    verify_zero_sum_property,
)

__all__ = [
    "symmetric_augment",
    "swap_matchup_vector",
    "swap_matchup_batch",
    "verify_zero_sum_property",
]

# conference_tournament_augmentation and two_stage_training were deleted in
# commit 44b048f (2026-04-21) and never restored. Importing them eagerly here
# made this whole package -- and therefore `symmetric_augment`, which the
# training pipeline does use -- unimportable, and took the ML pipeline down at
# its first training step. They are optional now: present if someone restores
# them, absent otherwise, and nothing that still exists depends on them.
try:
    from .conference_tournament_augmentation import (  # noqa: F401
        ConferenceAugmentationResult,
        build_conf_tourney_training_data,
        merge_with_primary_training_data,
    )

    __all__ += ["build_conf_tourney_training_data", "merge_with_primary_training_data", "ConferenceAugmentationResult"]
except ImportError:
    pass

try:
    from .two_stage_training import (  # noqa: F401
        TwoStageConfig,
        TwoStageResult,
        two_stage_train_lightgbm,
        two_stage_train_logistic,
        two_stage_train_spread,
    )

    __all__ += ["two_stage_train_lightgbm", "two_stage_train_spread", "two_stage_train_logistic", "TwoStageConfig", "TwoStageResult"]
except ImportError:
    pass
