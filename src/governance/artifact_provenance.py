"""Minimal artifact provenance helpers.

This module was lost from source control while call sites remained.
The implementation here restores the small interface the pipeline uses:
attach stable, machine-readable provenance to emitted artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_artifact_provenance(
    pipeline: Any,
    artifact_kind: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a small provenance payload for emitted artifacts."""
    config = getattr(pipeline, "config", None)
    provenance = {
        "artifact_kind": artifact_kind,
        "generated_at": _utc_now(),
        "year": getattr(config, "year", None),
        "probability_profile": getattr(config, "probability_profile", None),
        "pipeline_mode": getattr(config, "pipeline_mode", None),
        "random_seed": getattr(config, "random_seed", None),
        "production_model_stack": getattr(config, "production_model_stack", None),
        "production_calibration": getattr(config, "production_calibration", None),
    }
    if extra:
        provenance["extra"] = dict(extra)
    return provenance
