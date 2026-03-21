"""Feature manifest helpers for leakage-audit hardening."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _stable_feature_hash(values: List[str]) -> str:
    payload = json.dumps(list(values), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class FeatureManifest:
    """Frozen record of the active feature set used by a run."""

    generated_at_utc: str
    target_year: int
    model_complexity: str
    selection_method: str
    original_dim: int
    final_dim: int
    original_features: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    training_years: List[int] = field(default_factory=list)
    dev_years: List[int] = field(default_factory=list)
    holdout_years: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_hash(self) -> str:
        return _stable_feature_hash(self.selected_features)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["manifest_hash"] = self.manifest_hash
        return data


def build_feature_manifest(
    *,
    target_year: int,
    model_complexity: str,
    selection_method: str,
    original_features: List[str],
    selected_features: List[str],
    training_years: Optional[List[int]] = None,
    dev_years: Optional[List[int]] = None,
    holdout_years: Optional[List[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureManifest:
    """Construct a stable feature manifest from pre/post-selection names."""
    original = list(original_features or [])
    selected = list(selected_features or [])
    dropped = [name for name in original if name not in set(selected)]

    return FeatureManifest(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        target_year=int(target_year),
        model_complexity=str(model_complexity),
        selection_method=str(selection_method),
        original_dim=len(original),
        final_dim=len(selected),
        original_features=original,
        selected_features=selected,
        dropped_features=dropped,
        training_years=list(training_years or []),
        dev_years=list(dev_years or []),
        holdout_years=list(holdout_years or []),
        metadata=dict(metadata or {}),
    )


def write_feature_manifest(manifest: Dict[str, Any], path: str) -> str:
    """Persist a feature manifest JSON artifact."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
