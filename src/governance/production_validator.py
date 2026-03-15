"""Hard-fail governance checks for the frozen 2026 production path."""

from __future__ import annotations

from typing import Any, Dict

from ..pipeline.config import SOTAPipelineConfig


REQUIRED_CONFIG_VALUES: Dict[str, Any] = {
    "year": 2026,
    "probability_profile": "production",
    "mode": "calibration",
    "model_complexity": "simple",
    "use_agent_orchestration": False,
    "enable_gnn": False,
    "enable_transformer": False,
    "enable_embedding_projections": False,
    "enable_stacking": False,
    "enable_feature_selection": False,
    "enable_brier_sharpening": False,
    "enable_seed_overrides": False,
    "enable_goto_conversion": False,
    "enable_round_weighted_calibration": False,
    "enable_bayesian_bt": False,
    "enable_multi_year_training": True,
    "enable_multi_year_calibration": True,
    "enable_loyo_cv": True,
    "strict_leakage_mode": True,
    "require_freeze_file": True,
    "calibration_method": "temperature",
    "enable_tournament_adaptation": True,
    "scoring_metric": "brier",
}

EXPECTED_TRAINING_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
EXPECTED_DEV_YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
EXPECTED_HOLDOUT_YEARS = [2025]


class ProductionValidationError(ValueError):
    """Raised when production governance checks fail."""


def validate_2026_production_config(config: SOTAPipelineConfig) -> None:
    """Validate that config exactly matches the frozen 2026 production profile."""
    violations: list[str] = []

    for key, expected in REQUIRED_CONFIG_VALUES.items():
        actual = getattr(config, key)
        if actual != expected:
            violations.append(f"{key}={actual!r} (expected {expected!r})")

    if config.seed_prior_weight > 0:
        violations.append(f"seed_prior_weight={config.seed_prior_weight} (expected <= 0)")
    if config.consistency_bonus_max > 0:
        violations.append(
            f"consistency_bonus_max={config.consistency_bonus_max} (expected <= 0)"
        )

    if config.training_years is None:
        violations.append("training_years is missing")
    elif list(config.training_years) != EXPECTED_TRAINING_YEARS:
        violations.append(
            f"training_years={config.training_years} (expected {EXPECTED_TRAINING_YEARS})"
        )

    if config.dev_years is None:
        violations.append("dev_years is missing")
    elif list(config.dev_years) != EXPECTED_DEV_YEARS:
        violations.append(f"dev_years={config.dev_years} (expected {EXPECTED_DEV_YEARS})")

    if config.holdout_years is None:
        violations.append("holdout_years is missing")
    elif list(config.holdout_years) != EXPECTED_HOLDOUT_YEARS:
        violations.append(
            f"holdout_years={config.holdout_years} (expected {EXPECTED_HOLDOUT_YEARS})"
        )

    if config.dev_years and 2025 in config.dev_years:
        violations.append("2025 appears in dev_years")

    if config.training_years and config.dev_years:
        extra = sorted(set(config.training_years) - set(config.dev_years))
        if extra:
            violations.append(f"training_years has years outside dev_years: {extra}")

    all_years = []
    if config.training_years:
        all_years.extend(config.training_years)
    if config.dev_years:
        all_years.extend(config.dev_years)
    if config.holdout_years:
        all_years.extend(config.holdout_years)
    if 2020 in all_years:
        violations.append("2020 appears in training/dev/holdout years")

    if not config.multi_year_games_dir or config.multi_year_games_dir == "auto":
        violations.append("multi_year_games_dir must be explicit and not 'auto'")
    if not config.kaggle_dir:
        violations.append("kaggle_dir must be explicitly set")

    if violations:
        raise ProductionValidationError(
            "2026 production configuration validation failed: " + "; ".join(violations)
        )
