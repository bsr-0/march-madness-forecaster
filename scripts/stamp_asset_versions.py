#!/usr/bin/env python3
"""Stamp docs/index.html's ``?v=`` tokens with each asset's content hash.

WHY NOT A COUNTER. The tokens were hand-incremented integers, and index.html
already carried a comment recording that stale JS had shipped once because
somebody forgot. It happened again on 2026-09-06 -- app.js was edited *after*
its bump to v46, so v46 named two different files, and a browser holding the
first one had no way to learn about the second. That is not a discipline
problem worth solving with more discipline: a version a human types can always
disagree with the file it names.

A content hash cannot. Editing the file changes the token by construction, and
``tests/test_selection_sunday_rehearsal.py`` fails when the stamp and the file
disagree, so a forgotten stamp is caught rather than shipped.

    python scripts/stamp_asset_versions.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX = REPO / "docs" / "index.html"
ASSETS = ("app.js", "app.css", "fit.js")


def asset_token(name: str) -> str:
    """Eight hex characters of the file's sha256 -- plenty for cache busting."""
    return hashlib.sha256((REPO / "docs" / name).read_bytes()).hexdigest()[:8]


def stamp(check: bool = False) -> int:
    html = INDEX.read_text()
    stale = []

    for name in ASSETS:
        want = asset_token(name)
        pattern = re.compile(rf"({re.escape(name)}\?v=)([A-Za-z0-9]+)")
        found = pattern.search(html)
        if not found:
            print(f"{name}: no ?v= token in {INDEX.name}", file=sys.stderr)
            return 1
        if found.group(2) != want:
            stale.append(f"{name}: {found.group(2)} -> {want}")
            html = pattern.sub(rf"\g<1>{want}", html)

    if not stale:
        print("asset versions are current")
        return 0
    if check:
        print("STALE asset versions:\n  " + "\n  ".join(stale), file=sys.stderr)
        print("Run: python scripts/stamp_asset_versions.py", file=sys.stderr)
        return 1

    INDEX.write_text(html)
    print("stamped:\n  " + "\n  ".join(stale))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report staleness without rewriting")
    return stamp(check=ap.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
