#!/usr/bin/env python3
"""Closure-receipt checker for COUNCIL_LESSONS.md §2.

Every Ox row tagged ``[closed YYYY-MM-DD]`` must cite at least one
file path under ``scripts/``, ``artifacts/``, or ``tests/`` that
exists on disk. This catches the failure mode that retracted §2 O25's
first G1 closure: a narrative verdict citing spread numbers
(0.835/0.315/0.045/0.310) that had no script, no artifact, and no lock
test on disk — discovered only by accident on re-run.

Closure marker matched: ``[closed YYYY-MM-DD`` (with or without a
trailing ``— FAIL`` / ``— dead-end DXX`` qualifier). ``[mostly closed]``,
``[withdrawn ...]``, and ``[partially actioned]`` are intentionally NOT
enforced — they signal partial / no work done.

Run standalone for a console report:
    python scripts/check_closure_receipts.py
    python scripts/check_closure_receipts.py --report-only

Or import :func:`check_closure_receipts` from
``tests/test_closure_receipts.py`` for CI gating.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS_FILE = REPO_ROOT / "COUNCIL_LESSONS.md"

CLOSURE_MARKER = re.compile(r"\[closed\s+\d{4}-\d{2}-\d{2}")
OX_ID = re.compile(r"\*\*\s*(O\d+[a-z]?)\s*\*\*")
PATH_RE = re.compile(r"(?:scripts|artifacts|tests)/[\w./\-]+\.(?:py|json|md|txt)")


@dataclass
class ClosureRow:
    ox_id: str
    line_number: int
    cited_paths: List[str] = field(default_factory=list)
    existing_paths: List[str] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return len(self.existing_paths) > 0

    @property
    def reason(self) -> str:
        if self.is_verified:
            return "verified"
        if not self.cited_paths:
            return "no paths cited"
        return "all cited paths missing on disk"


def parse_closures(lessons_text: str, repo_root: Path) -> List[ClosureRow]:
    rows: List[ClosureRow] = []
    seen_ox: Set[str] = set()
    for i, line in enumerate(lessons_text.splitlines(), start=1):
        if not CLOSURE_MARKER.search(line):
            continue
        ox_match = OX_ID.search(line)
        if not ox_match:
            continue
        ox_id = ox_match.group(1)
        if ox_id in seen_ox:
            continue
        seen_ox.add(ox_id)
        cited = sorted({m.group(0) for m in PATH_RE.finditer(line)})
        existing = [p for p in cited if (repo_root / p).exists()]
        rows.append(
            ClosureRow(
                ox_id=ox_id,
                line_number=i,
                cited_paths=cited,
                existing_paths=existing,
            )
        )
    return rows


def check_closure_receipts(
    lessons_path: Path = LESSONS_FILE,
    repo_root: Path = REPO_ROOT,
) -> List[ClosureRow]:
    return parse_closures(lessons_path.read_text(encoding="utf-8"), repo_root)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the report and exit 0 even when unverified closures exist.",
    )
    args = parser.parse_args(argv)

    rows = check_closure_receipts()
    verified = [r for r in rows if r.is_verified]
    unverified = [r for r in rows if not r.is_verified]

    print("COUNCIL_LESSONS.md §2 closure-receipt audit")
    print(f"  closed rows audited : {len(rows)}")
    print(f"  verified            : {len(verified)}")
    print(f"  unverified          : {len(unverified)}")
    if unverified:
        print()
        print("Unverified closures (no cited path under scripts/ artifacts/ tests/ exists):")
        for r in unverified:
            cited = ", ".join(r.cited_paths) if r.cited_paths else "<no paths cited>"
            print(f"  {r.ox_id} (line {r.line_number}, {r.reason}): {cited}")
    if args.report_only:
        return 0
    return 1 if unverified else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
