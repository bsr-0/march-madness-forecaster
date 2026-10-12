"""Archive and restore the raw inputs consumed by the candidate-artifact builds.

``data/raw/`` is a local working cache and is not tracked, but the candidate
artifacts embed the SHA-256 of every file they consumed. A checksum is only
evidence while the bytes still exist, so the consumed files are kept as
deterministic gzip copies under ``data/archive/build_inputs/`` with a manifest
of each original file's SHA-256 (roadmap decision log, 2026-10-11).

    python -m scripts.restore_build_inputs            # restore missing inputs
    python -m scripts.restore_build_inputs --check    # verify archive and live files
    python -m scripts.restore_build_inputs --write    # (re)archive inputs the artifacts record

Restoring never overwrites a live file whose bytes differ from the archive; it
reports the conflict instead.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "data" / "archive" / "build_inputs"
MANIFEST = ARCHIVE / "MANIFEST.json"
CANDIDATES = REPO / "artifacts" / "candidates"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _records(value, out: dict[str, str]) -> dict[str, str]:
    if isinstance(value, dict):
        if isinstance(value.get("file"), str) and isinstance(value.get("sha256"), str):
            out[value["file"]] = value["sha256"]
        for child in value.values():
            _records(child, out)
    elif isinstance(value, list):
        for child in value:
            _records(child, out)
    return out


def recorded_inputs() -> dict[str, str]:
    """Every input path -> SHA-256 recorded by the pool-30 candidate artifacts."""
    inputs: dict[str, str] = {}
    for path in sorted(CANDIDATES.glob("candidates_*.json")):
        for file, digest in _records(json.loads(path.read_text())["provenance"], {}).items():
            if inputs.setdefault(file, digest) != digest:
                raise SystemExit(f"{file}: artifacts record two different hashes")
    return inputs


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return set(out.split())


def write() -> int:
    tracked = _tracked()
    manifest = {"note": "gzip copies (mtime=0, level 9) of untracked build inputs; sha256 is of the original bytes", "files": {}}
    for file, digest in sorted(recorded_inputs().items()):
        if file in tracked:
            continue
        raw = (REPO / file).read_bytes()
        if _sha256(raw) != digest:
            raise SystemExit(f"{file}: live bytes differ from the hash the artifacts record")
        target = ARCHIVE / (file + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        manifest["files"][file] = {"sha256": digest, "archive": target.relative_to(REPO).as_posix(), "bytes": len(raw)}
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"archived {len(manifest['files'])} files -> {MANIFEST.relative_to(REPO)}")
    return 0


def _archived(entry: dict) -> bytes:
    return gzip.decompress((REPO / entry["archive"]).read_bytes())


def check() -> list[str]:
    problems = []
    for file, entry in json.loads(MANIFEST.read_text())["files"].items():
        raw = _archived(entry)
        if _sha256(raw) != entry["sha256"]:
            problems.append(f"{file}: archive does not decompress to the recorded bytes")
        live = REPO / file
        if live.exists() and _sha256(live.read_bytes()) != entry["sha256"]:
            problems.append(f"{file}: live file differs from the archive")
    return problems


def restore() -> int:
    restored, conflicts = 0, []
    for file, entry in json.loads(MANIFEST.read_text())["files"].items():
        live = REPO / file
        if live.exists():
            if _sha256(live.read_bytes()) != entry["sha256"]:
                conflicts.append(file)
            continue
        raw = _archived(entry)
        if _sha256(raw) != entry["sha256"]:
            raise SystemExit(f"{file}: archive is corrupt")
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_bytes(raw)
        restored += 1
    print(f"restored {restored} file(s)")
    for file in conflicts:
        print(f"  conflict (left untouched): {file}")
    return 1 if conflicts else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        return write()
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else "archive verified")
        return 1 if problems else 0
    return restore()


if __name__ == "__main__":
    sys.exit(main())
