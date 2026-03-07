"""Append-only experiment ledger for tracking pipeline runs.

Records experiment metadata, configuration hashes, and LOYO validation
results in a JSONL file.  Extends the RDoF audit framework's config_hash
and feature_set_hash for cross-run comparison.

Usage:
    python -m src.main log-experiment --result pipeline_result.json
    python -m src.main list-experiments --last 10
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = "data/experiment_ledger.jsonl"


@dataclass
class ExperimentRecord:
    """A single experiment entry in the ledger."""

    experiment_id: str = ""
    timestamp: str = ""
    config_hash: str = ""
    feature_set_hash: str = ""
    dataset_version: str = ""  # e.g. "2016-2025"
    loyo_mean_brier: float = 0.0
    loyo_std_brier: float = 0.0
    loyo_year_briers: Dict[int, float] = field(default_factory=dict)
    holdout_integrity_level: int = 3  # 1=prospective, 2=quasi, 3=retrospective
    model_components: List[str] = field(default_factory=list)
    calibration_method: str = ""
    scoring_metric: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExperimentRecord:
        # Handle int keys in loyo_year_briers from JSON
        if "loyo_year_briers" in d and isinstance(d["loyo_year_briers"], dict):
            d["loyo_year_briers"] = {
                int(k): v for k, v in d["loyo_year_briers"].items()
            }
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


class ExperimentRegistry:
    """Append-only JSONL experiment ledger.

    Each line in the ledger file is a JSON object representing one
    ExperimentRecord.  The file is append-only — records are never
    modified or deleted.
    """

    def __init__(self, ledger_path: str = DEFAULT_LEDGER_PATH):
        self.ledger_path = Path(ledger_path)

    def log(self, record: ExperimentRecord) -> str:
        """Append a record to the ledger.  Returns the experiment_id."""
        if not record.experiment_id:
            record.experiment_id = str(uuid.uuid4())[:8]
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()

        # Check for duplicate config_hash (warn but still log)
        if record.config_hash:
            existing = self._find_by_config_hash(record.config_hash)
            if existing:
                logger.warning(
                    "Experiment with config_hash=%s already exists (id=%s). "
                    "Logging anyway — this may indicate a duplicate run.",
                    record.config_hash,
                    existing.experiment_id,
                )

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(
            "Logged experiment %s (Brier=%.6f, hash=%s)",
            record.experiment_id,
            record.loyo_mean_brier,
            record.config_hash[:8] if record.config_hash else "n/a",
        )
        return record.experiment_id

    def list(self, n: int = 20) -> List[ExperimentRecord]:
        """Return the last N experiments, most recent first."""
        all_records = self._load_all()
        return list(reversed(all_records[-n:]))

    def best(self, metric: str = "loyo_mean_brier") -> Optional[ExperimentRecord]:
        """Return the experiment with the lowest value of the given metric."""
        all_records = self._load_all()
        if not all_records:
            return None
        return min(all_records, key=lambda r: getattr(r, metric, float("inf")))

    def compare(self, id_a: str, id_b: str) -> Dict[str, Any]:
        """Compare two experiments by ID prefix."""
        rec_a = self._find_by_id(id_a)
        rec_b = self._find_by_id(id_b)
        if not rec_a or not rec_b:
            missing = id_a if not rec_a else id_b
            return {"error": f"Experiment {missing} not found"}

        d_a = rec_a.to_dict()
        d_b = rec_b.to_dict()

        diff: Dict[str, Any] = {}
        for key in d_a:
            if d_a[key] != d_b.get(key):
                diff[key] = {"a": d_a[key], "b": d_b.get(key)}

        return {
            "experiment_a": rec_a.experiment_id,
            "experiment_b": rec_b.experiment_id,
            "brier_delta": rec_b.loyo_mean_brier - rec_a.loyo_mean_brier,
            "differences": diff,
        }

    def _load_all(self) -> List[ExperimentRecord]:
        if not self.ledger_path.exists():
            return []
        records = []
        with open(self.ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ExperimentRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skipping malformed ledger line: %s", e)
        return records

    def _find_by_id(self, experiment_id: str) -> Optional[ExperimentRecord]:
        for rec in self._load_all():
            if rec.experiment_id.startswith(experiment_id):
                return rec
        return None

    def _find_by_config_hash(self, config_hash: str) -> Optional[ExperimentRecord]:
        for rec in self._load_all():
            if rec.config_hash == config_hash:
                return rec
        return None

    def summary(self) -> str:
        """Human-readable summary of the ledger."""
        records = self._load_all()
        if not records:
            return "No experiments logged yet."

        best_rec = min(records, key=lambda r: r.loyo_mean_brier)
        lines = [
            f"Experiment Ledger: {len(records)} experiments",
            f"  Best Brier: {best_rec.loyo_mean_brier:.6f} (id={best_rec.experiment_id})",
            f"  Latest: {records[-1].timestamp} (id={records[-1].experiment_id})",
            "",
            "  Last 5 experiments:",
        ]
        for rec in records[-5:]:
            lines.append(
                f"    {rec.experiment_id}  Brier={rec.loyo_mean_brier:.6f}  "
                f"hash={rec.config_hash[:8] if rec.config_hash else 'n/a'}  "
                f"{rec.timestamp[:19]}"
            )
        return "\n".join(lines)
