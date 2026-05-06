"""Feature manifest helper used by baseline training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List


def _manifest_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class FeatureManifest:
    target_year: int
    model_complexity: str
    selection_method: str
    original_features: List[str]
    selected_features: List[str]
    training_years: List[int]
    dev_years: List[int]
    holdout_years: List[int]
    metadata: Dict[str, Any]
    manifest_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_year": self.target_year,
            "model_complexity": self.model_complexity,
            "selection_method": self.selection_method,
            "original_features": list(self.original_features),
            "selected_features": list(self.selected_features),
            "training_years": list(self.training_years),
            "dev_years": list(self.dev_years),
            "holdout_years": list(self.holdout_years),
            "metadata": dict(self.metadata),
            "manifest_hash": self.manifest_hash,
        }


def build_feature_manifest(
    target_year: int,
    model_complexity: str,
    selection_method: str,
    original_features: List[str],
    selected_features: List[str],
    training_years: List[int],
    dev_years: List[int],
    holdout_years: List[int],
    metadata: Dict[str, Any],
) -> FeatureManifest:
    payload = {
        "target_year": target_year,
        "model_complexity": model_complexity,
        "selection_method": selection_method,
        "original_features": list(original_features),
        "selected_features": list(selected_features),
        "training_years": list(training_years),
        "dev_years": list(dev_years),
        "holdout_years": list(holdout_years),
        "metadata": dict(metadata or {}),
    }
    return FeatureManifest(manifest_hash=_manifest_hash(payload), **payload)
