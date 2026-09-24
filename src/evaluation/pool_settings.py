"""Validated pool-size and scoring presets for comparable evaluations.

The shipped contract remains ``standard_30``.  Alternate pool sizes are
explicit presets so callers cannot accidentally report a score under a
different field size while labelling it as the canonical result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ROUND_KEYS = ("R64", "R32", "S16", "E8", "F4", "CHAMP")
ESPN_STANDARD = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}
SCORING_PRESETS = {"espn_standard": ESPN_STANDARD}
SUPPORTED_POOL_SIZES = (10, 30, 50, 100)


@dataclass(frozen=True)
class PoolSettings:
    pool_size: int = 30
    scoring_id: str = "espn_standard"

    @property
    def n_opponents(self) -> int:
        return self.pool_size - 1

    @property
    def scoring(self) -> dict[str, int]:
        return dict(SCORING_PRESETS[self.scoring_id])

    @property
    def preset_id(self) -> str:
        return f"pool{self.pool_size}_{self.scoring_id}"


def resolve_pool_settings(pool_size: int = 30, scoring_id: str = "espn_standard") -> PoolSettings:
    """Validate user/config input and return an immutable settings object."""
    try:
        size = int(pool_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("pool_size must be an integer") from exc
    if size not in SUPPORTED_POOL_SIZES:
        raise ValueError(f"unsupported pool size {size}; choose one of {SUPPORTED_POOL_SIZES}")
    if scoring_id not in SCORING_PRESETS:
        raise ValueError(f"unsupported scoring preset; choose one of {tuple(SCORING_PRESETS)}")
    return PoolSettings(pool_size=size, scoring_id=scoring_id)


def settings_from_mapping(values: Mapping[str, object] | None) -> PoolSettings:
    values = values or {}
    return resolve_pool_settings(values.get("pool_size", 30), str(values.get("scoring_id", "espn_standard")))
