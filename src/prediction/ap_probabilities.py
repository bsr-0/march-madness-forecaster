"""AP-strength source (catalog A8).

Final-pre-tournament AP poll → per-team barthag for the test year.
Captures public-perception signal that's distinct from efficiency-metric
sources (torvik, odds, elo, massey). The field anchors on AP rankings
when picking brackets, so this source carries the *anchor* the opponent
field is using even though it ignores efficiency.

Walk-forward safety: each test year reads only its own AP poll rows
(year-keyed). No cross-year contamination.

Algorithm (catalog spec A8):
  - Final pre-tournament poll = max(week) for that year.
  - For each canonical tournament team:
      * If team is in poll with ap_rank ≤ 25: barthag = 1.0 - (rank / 50).
        Top of poll (rank=1) = 0.98; bottom of top-25 (rank=25) = 0.50.
      * If team is in poll with ap_rank > 25 AND ap_votes > 0
        ("receiving votes"): flat barthag = 0.45.
      * Otherwise (unranked, no votes): seed-based fallback
        max(0.10, 1.0 - 0.04 × seed). Same fallback shape as torvik.
  - All values clipped to [0.10, 0.99].

Bridge: canonical tournament IDs (e.g., ``duke_blue_devils``) → Kaggle
``team_no`` integer via ``MTeamSpellings.csv``. Same cascade pattern
as ``coach_adj_probabilities`` (deliberate inline duplication —
revisit shared module if a third Kaggle source ships).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROUND_NAMES = ("R64", "R32", "S16", "E8", "F4", "CHAMP")

# Per catalog spec.
_RECEIVING_VOTES_BARTHAG = 0.45
_BARTHAG_CLIP_MIN = 0.10
_BARTHAG_CLIP_MAX = 0.99

# Same alias table as coach_adj — canonical IDs that don't bridge via
# the cascade alone. Keep this short; every entry should be auditable.
_CANONICAL_TO_KAGGLE_ALIAS: Dict[str, str] = {
    "maryland_baltimore_county": "umbc",
    "st__john_s__ny": "st john's",
}


def _normalize_kaggle_spellings(data_root: Path) -> Dict[str, int]:
    """Lowercase spelling → Kaggle TeamID lookup from MTeamSpellings.csv.

    Inline-duplicated from ``coach_adj_probabilities._normalize_kaggle_spellings``.
    Refactor if/when a third Kaggle-based source ships.
    """
    path = Path(data_root) / "kaggle" / "MTeamSpellings.csv"
    out: Dict[str, int] = {}
    with open(path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            out[row["TeamNameSpelling"].lower()] = int(row["TeamID"])
    return out


def _canonical_to_kaggle_id(
    canonical_id: str,
    spellings_map: Dict[str, int],
) -> Optional[int]:
    """Cascade-bridge canonical → Kaggle TeamID. See coach_adj for full notes."""
    if canonical_id in _CANONICAL_TO_KAGGLE_ALIAS:
        aliased = _CANONICAL_TO_KAGGLE_ALIAS[canonical_id]
        if aliased in spellings_map:
            return spellings_map[aliased]

    candidates: List[str] = []

    naive = canonical_id.replace("__", " ").replace("_", " ").strip().lower()
    candidates.append(naive)

    possessive = canonical_id.replace("_s__", "'s ").replace("__", " ").replace("_", " ").strip().lower()
    candidates.append(possessive)

    if "_a_m" in canonical_id:
        am = canonical_id.replace("_a_m", " a&m").replace("__", " ").replace("_", " ").strip().lower()
        candidates.append(am)

    if canonical_id.startswith("saint_"):
        st_form = (
            canonical_id.replace("saint_", "st ", 1)
            .replace("_s__", "'s ")
            .replace("__", " ")
            .replace("_", " ")
            .strip()
            .lower()
        )
        candidates.append(st_form)

    for c in candidates:
        if c in spellings_map:
            return spellings_map[c]
    return None


def _load_ap_rows(data_root: Path) -> Optional[Dict]:
    """Load the AP poll columns/data structure from the Kaggle file."""
    path = Path(data_root) / "kaggle" / "ap_poll_data.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    if not isinstance(raw, dict) or "columns" not in raw or "data" not in raw:
        return None
    return raw


def _ap_entries_for_year(
    raw: Dict,
    year: int,
    spellings_map: Dict[str, int],
) -> Optional[Dict[int, Dict]]:
    """Return {kaggle_team_id: row_dict} for the final pre-tournament week.

    The "final pre-tournament week" = max(week) for that year. Returns
    None if the year is absent from the file.

    The AP file's own ``team_no`` field is in a different ID space than
    Kaggle's MTeams.csv (e.g., AP team_no=1207 is "Duke" but MTeams
    TeamID=1207 is "Georgetown"), so we ignore it and bridge via the
    AP ``team`` display name → MTeamSpellings → MTeams TeamID. The
    same MTeams TeamID is then used to look up canonical tournament
    teams via ``_canonical_to_kaggle_id``.
    """
    cols: List[str] = raw["columns"]
    rows: List[List] = raw["data"]
    year_idx = cols.index("year")
    week_idx = cols.index("week")

    year_rows = [r for r in rows if r[year_idx] == year]
    if not year_rows:
        return None

    final_week = max(r[week_idx] for r in year_rows)
    final_rows = [r for r in year_rows if r[week_idx] == final_week]

    team_idx = cols.index("team")
    out: Dict[int, Dict] = {}
    for r in final_rows:
        ap_team_name = r[team_idx]
        if not ap_team_name:
            continue
        # Try the AP team-name as-is, then a few normalizations to match
        # MTeamSpellings entries (which are all lowercase, periods-stripped).
        candidates = [
            ap_team_name.lower(),
            ap_team_name.lower().replace(".", ""),
            ap_team_name.lower().replace(".", "").strip(),
        ]
        kag_id: Optional[int] = None
        for c in candidates:
            if c in spellings_map:
                kag_id = spellings_map[c]
                break
        if kag_id is None:
            continue  # AP team name doesn't bridge — skip
        out[kag_id] = dict(zip(cols, r))
    return out


def _seed_fallback_barthag(seed: int) -> float:
    """Same seed-based fallback shape used by torvik/elo/massey."""
    return max(_BARTHAG_CLIP_MIN, 1.0 - 0.04 * float(seed))


def _ap_entry_to_barthag(entry: Dict, seed: int) -> float:
    """Apply the catalog A8 spec to a single AP poll row.

    - ap_rank ≤ 25 → 1.0 - rank/50.
    - ap_rank > 25 with ap_votes > 0 → 0.45 (receiving votes).
    - Otherwise (rare — entry exists but rank=0/None and no votes) →
      fall back to seed-based estimate.
    """
    ap_rank = entry.get("ap_rank")
    ap_votes = entry.get("ap_votes")
    if ap_rank is not None and ap_rank <= 25 and ap_rank >= 1:
        return max(_BARTHAG_CLIP_MIN, min(_BARTHAG_CLIP_MAX, 1.0 - float(ap_rank) / 50.0))
    if ap_votes is not None and ap_votes > 0:
        return _RECEIVING_VOTES_BARTHAG
    return _seed_fallback_barthag(seed)


def load_ap_strength_barthag(
    year: int,
    seeds: Dict[str, int],
    canonical_ids: Iterable[str],
    data_root: Path,
) -> Optional[Dict[str, float]]:
    """Per-team AP-poll-derived barthag for the test year.

    Returns None if the year is missing from ap_poll_data.json (e.g.,
    2020 COVID — explicitly absent). Returns a dict covering every
    requested canonical_id otherwise — teams not in the AP poll fall
    back to the seed-based estimate.

    Args:
        year: Tournament year (must be present in ap_poll_data.json).
        seeds: Per-team seed map (used for the unranked fallback).
        canonical_ids: Tournament team IDs to compute barthag for.
        data_root: Repo ``data/`` root.

    Returns:
        Dict[canonical_id, barthag ∈ [0.10, 0.99]], or None if the year
        has no AP rows.
    """
    raw = _load_ap_rows(data_root)
    if raw is None:
        return None

    spellings = _normalize_kaggle_spellings(data_root)
    final_week_entries = _ap_entries_for_year(raw, year, spellings)
    if final_week_entries is None:
        return None

    result: Dict[str, float] = {}
    canonical_set = list(canonical_ids)
    for tid in canonical_set:
        seed = seeds.get(tid, 16)  # missing seed → conservative fallback
        kag = _canonical_to_kaggle_id(tid, spellings)
        if kag is not None and kag in final_week_entries:
            result[tid] = _ap_entry_to_barthag(final_week_entries[kag], seed)
        else:
            result[tid] = _seed_fallback_barthag(seed)
    return result
