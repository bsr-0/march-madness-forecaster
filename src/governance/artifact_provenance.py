"""Artifact provenance helpers for leakage-sensitive outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def build_artifact_provenance(
    *,
    pipeline=None,
    artifact_kind: str,
    artifact_role: str = "durable",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized provenance block for an artifact or report."""
    config = getattr(pipeline, "config", None)
    feature_manifest = getattr(pipeline, "_feature_manifest", None) or {}
    year_split_policy = getattr(pipeline, "_year_split_policy", None)

    provenance = {
        "artifact_kind": artifact_kind,
        "artifact_role": artifact_role,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_year": getattr(config, "year", None),
        "training_years": list(getattr(config, "training_years", None) or []),
        "dev_years": list(getattr(config, "dev_years", None) or []),
        "holdout_years": list(getattr(config, "holdout_years", None) or []),
        "calibration_years": (
            list(config.resolve_calibration_years())
            if config is not None and hasattr(config, "resolve_calibration_years")
            else []
        ),
        "strict_leakage_mode": bool(getattr(config, "strict_leakage_mode", False)),
        "feature_manifest_hash": feature_manifest.get("manifest_hash"),
        "year_split_policy": (
            year_split_policy.to_dict()
            if year_split_policy is not None and hasattr(year_split_policy, "to_dict")
            else {}
        ),
    }
    if extra:
        provenance.update(extra)
    return provenance


def classify_artifact_role(path: str) -> str:
    """Best-effort durable vs scratch classification."""
    value = str(Path(path))
    scratch_markers = ("/tmp/", "/cache/", "/scratch/", ".pytest_cache", "__pycache__")
    return "scratch" if any(marker in value for marker in scratch_markers) else "durable"
