"""Deterministic hashing of pipeline inputs for reproducibility tracking.

Computes SHA-256 hashes for every data file referenced by a pipeline
configuration, the configuration itself, and a combined reproducibility
hash that uniquely identifies a pipeline run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional


# Data-file field names on SOTAPipelineConfig that point to JSON paths.
_DATA_FILE_FIELDS = (
    "teams_json",
    "torvik_json",
    "historical_games_json",
    "sports_reference_json",
    "public_picks_json",
    "scoring_rules_json",
    "roster_json",
    "transfer_portal_json",
    "preseason_ap_json",
    "coach_tournament_json",
    "conf_champions_json",
    "injury_report_json",
    "venue_locations_json",
    "team_locations_json",
    "bracket_json",
)


class RunHasher:
    """Hash pipeline inputs for reproducibility verification.

    Parameters
    ----------
    config : Any
        A ``SOTAPipelineConfig`` instance (typed as ``Any`` to avoid
        circular imports between the reproducibility and pipeline
        packages).
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hash_all_inputs(self) -> Dict[str, str]:
        """Compute SHA-256 hashes for every data file referenced by config.

        Returns a mapping ``{file_path: hex_digest}``.  Fields whose
        value is ``None`` or whose file does not exist are silently
        skipped.
        """
        hashes: Dict[str, str] = {}
        for field_name in _DATA_FILE_FIELDS:
            path_str: Optional[str] = getattr(self.config, field_name, None)
            if path_str is None:
                continue
            path = Path(path_str)
            if not path.is_file():
                continue
            hashes[str(path)] = self._sha256_file(path)
        return hashes

    def hash_config(self) -> str:
        """Return a deterministic SHA-256 hex digest of the config."""
        config_dict = asdict(self.config)
        canonical = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def compute_reproducibility_hash(self, code_version: str) -> str:
        """Combine code version, data hashes, and config hash into one digest.

        The resulting hash changes whenever the code, any input data
        file, or the pipeline configuration changes.
        """
        data_hashes = self.hash_all_inputs()
        config_hash = self.hash_config()

        parts: list[str] = [code_version]
        for key in sorted(data_hashes):
            parts.append(f"{key}={data_hashes[key]}")
        parts.append(config_hash)

        combined = "|".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def verify_against(self, record: Any) -> bool:
        """Check that current data hashes match those stored in *record*.

        Parameters
        ----------
        record : Any
            An ``ExperimentRecord`` instance (typed as ``Any`` to avoid
            circular imports).  Must expose a ``dataset_hashes`` dict.

        Returns
        -------
        bool
            ``True`` if every hash in *record.dataset_hashes* matches the
            current file hash.
        """
        current_hashes = self.hash_all_inputs()
        stored_hashes: Dict[str, str] = getattr(record, "dataset_hashes", {})
        if not stored_hashes:
            return False
        for path, expected_hash in stored_hashes.items():
            if current_hashes.get(path) != expected_hash:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute the SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
