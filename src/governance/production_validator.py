"""Validation helpers for production entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from . import ProductionValidationError
from ..pipeline.config import ForecastConfig

_REQUIRED_PRODUCTION_PATHS = (
    "external_ratings_dir",
    "multi_year_games_dir",
    "kaggle_dir",
    "teams_json",
    "torvik_json",
    "historical_games_json",
    "roster_json",
    "public_picks_json",
    "scoring_rules_json",
    "mc_calibration_json",
)


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _validate_paths_exist(config: ForecastConfig, base_dir: Path, fields: Iterable[str]) -> None:
    missing = []
    for field_name in fields:
        value = getattr(config, field_name, None)
        if not value:
            missing.append(f"{field_name}=<missing>")
            continue
        path = _resolve(base_dir, str(value))
        if not path.exists():
            missing.append(f"{field_name}={path}")
    if missing:
        raise ProductionValidationError("Required production paths missing: " + ", ".join(missing))


def validate_production_config(config: ForecastConfig, year: int) -> None:
    """Validate structural production constraints for a target year."""
    if config.year != year:
        raise ProductionValidationError(f"Config year {config.year} does not match requested year {year}")
    if config.probability_profile != "production":
        raise ProductionValidationError(
            f"Production config must use probability_profile='production', got {config.probability_profile!r}"
        )
    if config.pipeline_mode != "production":
        raise ProductionValidationError(
            f"Production config must use pipeline_mode='production', got {config.pipeline_mode!r}"
        )
    if config.mode != "calibration":
        raise ProductionValidationError(f"Production config must use mode='calibration', got {config.mode!r}")
    try:
        config.validate_locked_production_path()
    except ValueError as exc:
        raise ProductionValidationError(str(exc)) from exc


def validate_production_2026(
    config: ForecastConfig,
    check_paths_on_disk: bool = False,
    base_dir: str | None = None,
) -> None:
    """Validate the frozen 2026 production path."""
    validate_production_config(config, 2026)
    if not config.require_freeze_file:
        raise ProductionValidationError("Frozen 2026 production config must require a freeze file")
    if not config.freeze_file:
        raise ProductionValidationError("Frozen 2026 production config must specify freeze_file")
    if check_paths_on_disk:
        root = Path(base_dir or ".").resolve()
        _validate_paths_exist(config, root, _REQUIRED_PRODUCTION_PATHS + ("freeze_file",))
