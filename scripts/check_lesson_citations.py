#!/usr/bin/env python3
"""Stale-citation checker for MEMORY.md and COUNCIL_LESSONS.md.

Both files cite repo paths heavily — `src/optimization/leverage.py:1618`,
`scripts/o25_g2_empirical_supremum.py`, `artifacts/o6_winner_rank_diagnostic.json`,
`tests/test_holdout_enforcement.py`, etc. When a refactor moves or
deletes one of those files, the citation rots silently and the next
agent that tries to verify a "settled" claim hits a dead reference.

This checker scans both files for path tokens under the standard
top-level dirs and verifies each exists on disk. Pairs with the
closure-receipt gate (`tests/test_closure_receipts.py`): receipts
prevent narrative-only NEW closures, citations prevent existing
closures from rotting away.

Closure-receipt gate ratchets forward; this gate ratchets too — any
*new* broken citation fails CI. Pre-existing broken citations are
explicitly allowlisted.

Run standalone for a console report:
    python scripts/check_lesson_citations.py
    python scripts/check_lesson_citations.py --report-only

Or import :func:`check_citations` from
``tests/test_lesson_citations.py`` for CI gating.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

AUDITED_FILES = (
    REPO_ROOT / "MEMORY.md",
    REPO_ROOT / "COUNCIL_LESSONS.md",
)

# Top-level dirs whose contents we treat as "real, citable repo paths".
# Tokens that don't start with one of these are ignored (root-level
# docs like PROJECT_STATUS.md change rarely; if they ever do, the
# allowlist absorbs the breakage).
AUDITED_PREFIXES = ("scripts/", "tests/", "src/", "artifacts/", "data/", "configs/")

# Allowed file extensions. Bare directory references are not enforced.
AUDITED_EXTENSIONS = (".py", ".json", ".md", ".txt", ".yaml", ".yml")

_PREFIX_ALT = "|".join(re.escape(p) for p in AUDITED_PREFIXES)
_EXT_ALT = "|".join(ext.lstrip(".") for ext in AUDITED_EXTENSIONS)
PATH_RE = re.compile(rf"(?P<path>(?:{_PREFIX_ALT})[\w/.\-]+\.(?:{_EXT_ALT}))")


@dataclass
class Citation:
    source_file: Path
    line_number: int
    raw_token: str
    cleaned_path: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.source_file.name, self.cleaned_path)


@dataclass
class CitationReport:
    citations: List[Citation] = field(default_factory=list)
    broken: List[Citation] = field(default_factory=list)

    @property
    def broken_keys(self) -> Set[Tuple[str, str]]:
        return {c.key for c in self.broken}

    @property
    def broken_paths(self) -> Set[str]:
        return {c.cleaned_path for c in self.broken}


def _strip_suffix(token: str) -> str:
    """Strip ``:123`` line-number suffix and ``::TestClass::test_method``
    pytest selectors. Hypothetical-template paths (containing ``{``) are
    returned as-is so the caller can skip them."""
    return token.split("::", 1)[0].split(":", 1)[0]


def _iter_candidate_paths(text: str) -> Iterable[Tuple[int, str, str]]:
    for i, line in enumerate(text.splitlines(), start=1):
        for match in PATH_RE.finditer(line):
            raw = match.group("path")
            cleaned = _strip_suffix(raw)
            if "{" in cleaned or "}" in cleaned:
                continue
            yield i, raw, cleaned


def check_citations(
    audited_files: Iterable[Path] = AUDITED_FILES,
    repo_root: Path = REPO_ROOT,
) -> CitationReport:
    report = CitationReport()
    for source in audited_files:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        seen: Set[str] = set()
        for line_no, raw, cleaned in _iter_candidate_paths(text):
            if cleaned in seen:
                continue
            seen.add(cleaned)
            citation = Citation(
                source_file=source,
                line_number=line_no,
                raw_token=raw,
                cleaned_path=cleaned,
            )
            report.citations.append(citation)
            if not (repo_root / cleaned).exists():
                report.broken.append(citation)
    return report


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the report and exit 0 even when broken citations exist.",
    )
    args = parser.parse_args(argv)

    report = check_citations()
    print("MEMORY.md + COUNCIL_LESSONS.md citation audit")
    print(f"  citations scanned : {len(report.citations)}")
    print(f"  broken            : {len(report.broken)}")
    if report.broken:
        print()
        print("Broken citations (path under audited dir does not exist on disk):")
        for c in report.broken:
            print(f"  {c.source_file.name}:{c.line_number}  {c.cleaned_path}")
    if args.report_only:
        return 0
    return 1 if report.broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
