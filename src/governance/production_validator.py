"""Hard-fail governance checks for the frozen 2026 production path."""

from __future__ import annotations

import os
from typing import Any, Dict

from ..pipeline.config import SOTAPipelineConfig


DIRECTORY_PATH_FIELDS = [
    "multi_year_games_dir",
    "kaggle_dir",
    "external_ratings_dir",
]

FILE_PATH_FIELDS = [
    "teams_json",
    "torvik_json",
    "historical_games_json",
    "roster_json",
    "public_picks_json",
    "scoring_rules_json",
    "mc_calibration_json",
    "freeze_file",
]

ALL_PATH_FIELDS = DIRECTORY_PATH_FIELDS + FILE_PATH_FIELDS


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
    # --- Continuous hyperparameters that affect model output ---
    "random_seed": 2026,
    "num_simulations": 50000,
    "tournament_shrinkage": 0.06,
    "massey_blend_weight": 0.25,
    "massey_sigma": 4.5,
    "ensemble_lgb_weight": 0.15,
    "ensemble_xgb_weight": 0.15,
    "pre_calibration_clip_lo": 0.001,
    "pre_calibration_clip_hi": 0.999,
    "mc_noise_std": 0.12,
    "enable_spread_model": True,
    "enable_recency_weighting": True,
    "enable_symmetric_augmentation": True,
    "scrape_live": False,
    "enable_market_blend": False,
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

    if config.training_years and 2025 in config.training_years:
        violations.append("2025 appears in training_years")

    cal_years = getattr(config, "calibration_years", None)
    if cal_years is not None and list(cal_years) != [2025]:
        violations.append(
            f"calibration_years={cal_years} (must be None or [2025] for production)"
        )

    for field in ALL_PATH_FIELDS:
        value = getattr(config, field, None)
        if not value:
            violations.append(f"{field} must be explicitly set")
        elif isinstance(value, str) and value.strip().lower() == "auto":
            violations.append(f"{field}={value!r} must not be 'auto'")

    if not getattr(config, "external_ratings_dir", None):
        if "external_ratings_dir must be explicitly set" not in violations:
            violations.append("external_ratings_dir must be explicitly set")

    if violations:
        raise ProductionValidationError(
            "2026 production configuration validation failed: " + "; ".join(violations)
        )


def validate_production_2026(
    config: SOTAPipelineConfig,
    *,
    check_paths_on_disk: bool = True,
    base_dir: str | None = None,
) -> None:
    """Canonical production validator entry point.

    Calls the core config validation and, when *check_paths_on_disk* is True,
    also verifies that every declared directory exists on the filesystem and
    every declared file exists on disk.
    """
    validate_2026_production_config(config)

    if not check_paths_on_disk:
        return

    root = base_dir or os.getcwd()
    missing: list[str] = []

    for field in DIRECTORY_PATH_FIELDS:
        value = getattr(config, field, None)
        if value:
            resolved = os.path.join(root, value) if not os.path.isabs(value) else value
            if not os.path.isdir(resolved):
                missing.append(f"{field}={value!r} directory not found at {resolved}")

    for field in FILE_PATH_FIELDS:
        value = getattr(config, field, None)
        if value:
            resolved = os.path.join(root, value) if not os.path.isabs(value) else value
            if not os.path.isfile(resolved):
                missing.append(f"{field}={value!r} file not found at {resolved}")

    if missing:
        raise ProductionValidationError(
            "Production data path verification failed: " + "; ".join(missing)
        )
