"""Drift guard for the data-artifact provenance ledger.

Closes COUNCIL_LESSONS.md §1 Data pipeline / "No request logging / response
hashing / artifact provenance tracking" — the response-hashing + artifact-
provenance halves. Production manifests must carry a content-hash ledger
under ``provenance.data_artifacts``; the recorded hashes must still match
the files currently on disk.

If a data file changes after the audit was last run, this test fires —
which is the desired behavior. The fix is to re-run the audit:

    python scripts/audit_data_provenance.py

If a manifest legitimately gains new artifacts, the audit picks them up
automatically (the ledger is keyed by the manifest's ``artifacts`` dict).
The test fails until you re-run it; that's the lock.

URL / fetch-time recording at scrape time is a separate piece tracked as
the deferred follow-up — see COUNCIL_LESSONS §1 Data pipeline + §3 row
for the closure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.data.data_provenance import (
    MISSING_SENTINEL,
    compute_data_provenance,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_DIR = _REPO_ROOT / "data" / "raw"
_MANIFEST_RE = re.compile(r"^manifest_(\d{4})\.json$")

# Years that must carry a populated provenance ledger. These are the
# active production manifests. Adding a new production year here without
# running scripts/audit_data_provenance.py first will fail the audit
# tests, which is the desired loud-fail.
PRODUCTION_AUDIT_YEARS = (2025, 2026)


def _discover_manifest_years() -> list[int]:
    return sorted(int(m.group(1)) for p in _MANIFEST_DIR.glob("manifest_*.json") if (m := _MANIFEST_RE.match(p.name)))


def _load_manifest(year: int) -> dict:
    return json.loads((_MANIFEST_DIR / f"manifest_{year}.json").read_text())


# ---------------------------------------------------------------------------
# Ledger contract — shape + completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("year", PRODUCTION_AUDIT_YEARS)
def test_production_manifest_has_data_artifacts_ledger(year: int) -> None:
    """Every production manifest carries provenance.data_artifacts."""
    manifest = _load_manifest(year)
    provenance = manifest.get("provenance")
    assert isinstance(provenance, dict), (
        f"manifest_{year}.json has no 'provenance' dict. Run: python scripts/audit_data_provenance.py"
    )
    ledger = provenance.get("data_artifacts")
    assert isinstance(ledger, dict), (
        f"manifest_{year}.json provenance has no 'data_artifacts' ledger. Run: python scripts/audit_data_provenance.py"
    )
    assert "files" in ledger and "audited_at_utc" in ledger, (
        f"data_artifacts ledger for {year} is missing required keys (files, audited_at_utc). Got: {sorted(ledger)}"
    )


@pytest.mark.parametrize("year", PRODUCTION_AUDIT_YEARS)
def test_ledger_covers_every_artifact_in_manifest(year: int) -> None:
    """Every entry in manifest['artifacts'] must appear in the ledger.

    Catches the case where a new artifact was added to the manifest but
    the audit script hasn't been re-run.
    """
    manifest = _load_manifest(year)
    artifacts = manifest.get("artifacts", {})
    ledger = manifest["provenance"]["data_artifacts"]["files"]
    missing_from_ledger = sorted(set(artifacts) - set(ledger))
    assert not missing_from_ledger, (
        f"manifest_{year}.json artifacts not represented in the ledger: "
        f"{missing_from_ledger}. Run: python scripts/audit_data_provenance.py"
    )


@pytest.mark.parametrize("year", PRODUCTION_AUDIT_YEARS)
def test_ledger_entries_have_required_fields(year: int) -> None:
    """Every ledger entry has the contract fields."""
    ledger = _load_manifest(year)["provenance"]["data_artifacts"]["files"]
    required = {"path", "provider", "exists", "sha256", "sha256_short", "bytes", "mtime_utc"}
    for name, rec in ledger.items():
        actual = set(rec)
        missing = required - actual
        assert not missing, f"{year}/{name} ledger entry is missing fields {missing}. Got: {sorted(actual)}"


# ---------------------------------------------------------------------------
# Drift detection — recorded hashes still match files on disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("year", PRODUCTION_AUDIT_YEARS)
def test_recorded_hashes_match_current_disk(year: int) -> None:
    """The recorded sha256 must equal a freshly-computed sha256.

    This is the load-bearing drift guard. Any underlying data file edit
    that doesn't get followed by a re-run of audit_data_provenance.py
    fires this test.
    """
    fresh = compute_data_provenance(year, manifest_dir=_MANIFEST_DIR, repo_root=_REPO_ROOT)
    recorded = _load_manifest(year)["provenance"]["data_artifacts"]["files"]
    drift = []
    for name, fresh_rec in fresh["data_artifacts"].items():
        rec = recorded.get(name)
        if rec is None:
            drift.append(f"{name}: not in recorded ledger")
            continue
        if rec["sha256"] != fresh_rec["sha256"]:
            drift.append(f"{name}: recorded={rec['sha256_short']} fresh={fresh_rec['sha256_short']}")
    assert not drift, (
        f"Hash drift in manifest_{year}.json provenance ledger:\n  "
        + "\n  ".join(drift)
        + "\nRun: python scripts/audit_data_provenance.py"
    )


# ---------------------------------------------------------------------------
# Idempotency — re-computing the ledger yields identical hashes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("year", PRODUCTION_AUDIT_YEARS)
def test_compute_is_idempotent(year: int) -> None:
    """Two consecutive computations produce identical sha256 values.

    The audited_at_utc timestamp WILL differ between calls — that's
    expected; only the file content hashes must be stable.
    """
    a = compute_data_provenance(year, manifest_dir=_MANIFEST_DIR, repo_root=_REPO_ROOT)
    b = compute_data_provenance(year, manifest_dir=_MANIFEST_DIR, repo_root=_REPO_ROOT)
    a_hashes = {n: r["sha256"] for n, r in a["data_artifacts"].items()}
    b_hashes = {n: r["sha256"] for n, r in b["data_artifacts"].items()}
    assert a_hashes == b_hashes


# ---------------------------------------------------------------------------
# Sentinel handling — missing files are flagged, not crashed
# ---------------------------------------------------------------------------


def test_missing_file_uses_sentinel_not_crash(tmp_path: Path) -> None:
    """A manifest referencing a non-existent file gets MISSING_SENTINEL,
    not an exception. Production manifests legitimately have missing
    artifacts during partial-rebuild windows; loud-fail there is wrong."""
    manifest_dir = tmp_path / "raw"
    manifest_dir.mkdir()
    (tmp_path / "data").mkdir(exist_ok=True)
    fake_manifest = {
        "year": 1999,
        "artifacts": {"some_file": "data/raw/does_not_exist.json"},
        "providers": {"some_file": "test"},
    }
    (manifest_dir / "manifest_1999.json").write_text(json.dumps(fake_manifest))
    led = compute_data_provenance(1999, manifest_dir=manifest_dir, repo_root=tmp_path)
    rec = led["data_artifacts"]["some_file"]
    assert rec["exists"] is False
    assert rec["sha256"] == MISSING_SENTINEL
    assert rec["bytes"] == 0


def test_malformed_manifest_raises() -> None:
    """A manifest without an 'artifacts' dict is a programmer error,
    not a runtime degradation. Raise loudly."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mdir = Path(td) / "raw"
        mdir.mkdir()
        (mdir / "manifest_1999.json").write_text(json.dumps({"year": 1999}))
        with pytest.raises(ValueError) as excinfo:
            compute_data_provenance(1999, manifest_dir=mdir, repo_root=Path(td))
        assert "no 'artifacts' dict" in str(excinfo.value)


def test_missing_manifest_file_raises() -> None:
    """Auditing a year with no manifest file is a typo, not a no-op."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mdir = Path(td) / "raw"
        mdir.mkdir()
        with pytest.raises(FileNotFoundError):
            compute_data_provenance(1999, manifest_dir=mdir, repo_root=Path(td))


# ---------------------------------------------------------------------------
# Coverage — ensure we didn't drop a year accidentally
# ---------------------------------------------------------------------------


def test_production_audit_years_covers_existing_manifests() -> None:
    """All manifest_*.json files on disk are covered by the audit
    parameter list. Surfaces the case where someone adds a new manifest
    but forgets to enroll it in PRODUCTION_AUDIT_YEARS."""
    on_disk = set(_discover_manifest_years())
    enrolled = set(PRODUCTION_AUDIT_YEARS)
    missing = sorted(on_disk - enrolled)
    assert not missing, (
        f"Manifest years on disk but not enrolled for audit: {missing}. "
        "If these are production manifests, add them to "
        "PRODUCTION_AUDIT_YEARS at the top of this file. If they are "
        "dev/scratch artifacts, move them out of data/raw/."
    )
