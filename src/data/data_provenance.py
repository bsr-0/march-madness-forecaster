"""Data-artifact provenance ledger.

Closes COUNCIL_LESSONS.md §1 Data pipeline / "No request logging / response
hashing / artifact provenance tracking" — specifically the *response-hashing*
and *artifact-provenance* halves. URL / fetch-time recording at scrape time
is a separate piece that requires touching every scraper; this module only
covers what can be reconstructed from files already on disk.

Design constraints:
    - Pure read-only over data/raw/ — does NOT modify any data file.
    - No scraper code changes — operates on the cache layer that already
      exists (data/raw/{year}/* + data/raw/manifest_{year}.json).
    - Idempotent — running the audit twice produces identical output.
    - Reuses ``src/data/manifest_generator.py::_sha256_file``  rather than
      re-implementing hashing; if that helper changes (chunk size,
      algorithm), the ledger inherits the change.

The ledger answers the council's literal question — *"which data source was
used for a given model run?"* — by giving every file referenced by the
year-level manifest a content hash + mtime + size. Two runs that read the
same manifest are bit-identical iff the hashes match. Two runs whose
hashes diverge tell you the underlying scrape changed, even if filenames
are unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .manifest_generator import _sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sentinel used when a manifest references a file that does not exist on
# disk. This is NOT a hash — downstream consumers must check for it
# explicitly. Keeping it short so it doesn't look like a truncated hex.
MISSING_SENTINEL = "FILE_NOT_FOUND"


def _resolve_artifact_path(path_str: str, base: Path) -> Path:
    """Resolve a manifest artifact path against the repo root.

    Manifest paths are repo-relative strings like
    ``data/raw/torvik_2026.json``. Some entries may also be absolute or
    pre-resolved; handle both shapes uniformly.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return base / p


def _file_record(path: Path) -> Dict[str, Any]:
    """Return a deterministic record for a single artifact file.

    Includes:
        - sha256: full hex digest of the file contents (or sentinel if absent)
        - sha256_short: first 16 chars (matches pipeline_freeze.json convention)
        - bytes: file size, or 0 if missing
        - mtime_utc: modification timestamp in UTC ISO-8601, or null if missing
        - exists: bool

    The size and mtime are not used as identity; they exist as
    cross-checks that surface "file modified but content unchanged" or
    "file replaced with same hash" anomalies during audit.
    """
    if not path.exists():
        return {
            "exists": False,
            "sha256": MISSING_SENTINEL,
            "sha256_short": MISSING_SENTINEL,
            "bytes": 0,
            "mtime_utc": None,
        }
    digest = _sha256_file(str(path))
    stat = path.stat()
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return {
        "exists": True,
        "sha256": digest,
        "sha256_short": digest[:16] if digest != MISSING_SENTINEL else MISSING_SENTINEL,
        "bytes": stat.st_size,
        "mtime_utc": mtime_utc,
    }


def compute_data_provenance(
    year: int,
    manifest_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute a content-hash provenance ledger for one year.

    Reads ``manifest_{year}.json`` from ``manifest_dir`` (default
    ``data/raw/`` under the repo root) and produces a ledger keyed by the
    manifest's logical artifact name (e.g., ``"torvik_json"``). Each entry
    pairs the provider name (from the manifest's ``providers`` field) with
    the file record (hash + size + mtime).

    The ledger is designed to be merged into the manifest under
    ``provenance.data_artifacts``; see ``scripts/audit_data_provenance.py``.

    Args:
        year: Tournament year whose manifest to audit.
        manifest_dir: Directory containing ``manifest_{year}.json``;
            defaults to ``data/raw`` under the repo root.
        repo_root: Repo root for resolving relative artifact paths.

    Returns:
        Dict with shape::

            {
                "year": int,
                "audited_at_utc": str,
                "manifest_path": str,
                "data_artifacts": {
                    logical_name: {
                        "path": str,
                        "provider": str,
                        "exists": bool,
                        "sha256": str,
                        "sha256_short": str,
                        "bytes": int,
                        "mtime_utc": str | None,
                    },
                    ...
                },
            }

    Raises:
        FileNotFoundError: if ``manifest_{year}.json`` itself is absent.
        ValueError: if the manifest is malformed (missing ``artifacts``).
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    mdir = Path(manifest_dir) if manifest_dir is not None else root / "data" / "raw"
    manifest_path = mdir / f"manifest_{year}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"manifest_{year}.json has no 'artifacts' dict; cannot audit")
    providers = manifest.get("providers", {}) or {}

    data_artifacts: Dict[str, Any] = {}
    for logical_name in sorted(artifacts):
        path_str = artifacts[logical_name]
        path = _resolve_artifact_path(str(path_str), root)
        record = _file_record(path)
        data_artifacts[logical_name] = {
            "path": str(path_str),
            "provider": providers.get(logical_name, "unknown"),
            **record,
        }

    return {
        "year": int(year),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path.relative_to(root)),
        "data_artifacts": data_artifacts,
    }


def write_provenance_into_manifest(
    year: int,
    manifest_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit one year and merge the result into its manifest.

    Reads ``manifest_{year}.json``, computes the data-artifact ledger via
    ``compute_data_provenance``, and writes it back under
    ``provenance.data_artifacts``. The manifest's existing ``provenance``
    fields (e.g., ``data_quality_repair`` blocks injected manually) are
    preserved — only the ``data_artifacts`` key under ``provenance`` is
    overwritten.

    Returns the ledger that was written (same shape as
    ``compute_data_provenance``).

    Idempotent: running twice produces identical hashes; only the
    ``audited_at_utc`` timestamp on the ``data_artifacts`` field changes.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    mdir = Path(manifest_dir) if manifest_dir is not None else root / "data" / "raw"
    manifest_path = mdir / f"manifest_{year}.json"

    ledger = compute_data_provenance(year, manifest_dir=mdir, repo_root=root)

    manifest = json.loads(manifest_path.read_text())
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["data_artifacts"] = {
        "audited_at_utc": ledger["audited_at_utc"],
        "files": ledger["data_artifacts"],
    }
    manifest["provenance"] = provenance
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return ledger
