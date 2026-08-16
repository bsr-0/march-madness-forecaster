"""Round-name taxonomy gate for data/raw/historical/tournament_results_{year}.json.

Closes COUNCIL_LESSONS.md §2 O22.

Guards the canonical labels documented in
``src/data/historical_tournament_results.py`` (FF, R64, R32, S16, E8, F4, NCG).
O22 was triggered by a 2026 ingestion that produced {R68×4, NCG×49, ...} —
the ESPN event-note substring matcher in
``src/data/scrapers/tournament_results.py`` collapsed every round whose
headline contains "championship" but not a more specific round phrase onto
``NCG``. The canonical-label schema had no loud-failure enforcement, so the
defect only surfaced during O21's year-over-year analysis.

Era-aware canonical counts:
    - pre-2011 (2005–2010): 64 games, 1×FF + 32×R64 + 16×R32 + 8×S16 + 4×E8 + 2×F4 + 1×NCG
    - 2011-2026           : 67 games, 4×FF + 32×R64 + 16×R32 + 8×S16 + 4×E8 + 2×F4 + 1×NCG
    - 2027+               : 75 games, 12×FF + 32×R64 + 16×R32 + 8×S16 + 4×E8 + 2×F4 + 1×NCG
      (76-team expansion: 12 play-in games reduce field to 64 for R64)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HIST_DIR = _REPO_ROOT / "data" / "raw" / "historical"

CANONICAL_ROUND_NAMES = {"FF", "R64", "R32", "S16", "E8", "F4", "NCG"}

CANONICAL_COUNTS_POST_2027 = {
    "FF": 12,  # 2027+ expansion: 76 teams, 12 play-in games → 64 in R64.
    "R64": 32,
    "R32": 16,
    "S16": 8,
    "E8": 4,
    "F4": 2,
    "NCG": 1,
}

CANONICAL_COUNTS_POST_2011 = {
    "FF": 4,
    "R64": 32,
    "R32": 16,
    "S16": 8,
    "E8": 4,
    "F4": 2,
    "NCG": 1,
}

CANONICAL_COUNTS_PRE_2011 = {
    "FF": 1,  # Pre-2011 opening round was a single play-in game.
    "R64": 32,
    "R32": 16,
    "S16": 8,
    "E8": 4,
    "F4": 2,
    "NCG": 1,
}

# 2020 NCAA tournament was cancelled (COVID); no file expected.
MISSING_YEARS = {2020}


def _available_years() -> list[int]:
    years: set[int] = set()
    for p in _HIST_DIR.glob("tournament_context_*.json"):
        try:
            year = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        with p.open() as f:
            if "results" in json.load(f):
                years.add(year)
    for p in _HIST_DIR.glob("tournament_results_*.json"):
        try:
            years.add(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(years)


def _load(year: int) -> list[dict]:
    ctx_path = _HIST_DIR / f"tournament_context_{year}.json"
    doc = None
    if ctx_path.exists():
        with ctx_path.open() as f:
            doc = json.load(f).get("results")
    if doc is None:
        with (_HIST_DIR / f"tournament_results_{year}.json").open() as f:
            doc = json.load(f)
    return doc["games"] if isinstance(doc, dict) else doc


def _expected_counts(year: int) -> dict[str, int]:
    if year >= 2027:
        return CANONICAL_COUNTS_POST_2027
    if year >= 2011:
        return CANONICAL_COUNTS_POST_2011
    return CANONICAL_COUNTS_PRE_2011


@pytest.mark.parametrize("year", _available_years())
def test_round_name_taxonomy_canonical(year: int) -> None:
    """Every game uses a canonical round_name label."""
    games = _load(year)
    bad = sorted({g.get("round_name") for g in games} - CANONICAL_ROUND_NAMES)
    assert not bad, (
        f"{year}: non-canonical round_name label(s): {bad}. "
        f"Canonical set is {sorted(CANONICAL_ROUND_NAMES)}. "
        "If the scraper produced 'R68', run "
        "scripts/repair_tournament_results_2026_o22.py (or the equivalent) "
        "and fix the upstream matcher."
    )


@pytest.mark.parametrize("year", _available_years())
def test_round_name_counts_match_era(year: int) -> None:
    """Per-round counts match the canonical tournament structure."""
    games = _load(year)
    counts = dict(Counter(g.get("round_name") for g in games))
    expected = _expected_counts(year)
    assert counts == expected, (
        f"{year}: round_name counts {counts} do not match canonical "
        f"{expected}. This was the O22 2026 failure mode — see "
        "COUNCIL_LESSONS.md §2 O22."
    )


def test_covid_year_file_absent() -> None:
    """2020 was cancelled; tournament results data for 2020 would be suspicious."""
    for year in MISSING_YEARS:
        assert not (_HIST_DIR / f"tournament_results_{year}.json").exists(), (
            f"tournament_results_{year}.json exists but {year} tournament was cancelled."
        )
        ctx_path = _HIST_DIR / f"tournament_context_{year}.json"
        if ctx_path.exists():
            with ctx_path.open() as f:
                ctx = json.load(f)
            assert "results" not in ctx, (
                f"tournament_context_{year}.json has a 'results' key but {year} tournament was cancelled."
            )


def test_2026_has_canonical_counts() -> None:
    """Pinned O22-closure assertion so a regression on 2026 fails under a
    direct name rather than only inside the parametrized year matrix."""
    games = _load(2026)
    assert len(games) == 67
    assert dict(Counter(g["round_name"] for g in games)) == CANONICAL_COUNTS_POST_2011
