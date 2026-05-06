"""Shared Kaggle ID bridge utilities.

Maps canonical tournament team IDs (e.g., ``duke``, ``michigan_state``) to
Kaggle ``TeamID`` integers via ``MTeamSpellings.csv`` and a small manual
alias table.  Used by any probability source that reads Kaggle CSVs
(coach_adj, ap_strength, and future Kaggle-based sources).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Manual aliases for canonical IDs that don't bridge via name normalization.
# Keep this list short — every entry should be auditable.
CANONICAL_TO_KAGGLE_ALIAS: Dict[str, str] = {
    "maryland_baltimore_county": "umbc",
    "st__john_s__ny": "st john's",  # the cascade misses this; (NY) suffix isn't standalone
}


def normalize_kaggle_spellings(data_root: Path) -> Dict[str, int]:
    """Lowercase spelling -> Kaggle TeamID lookup from MTeamSpellings.csv."""
    path = Path(data_root) / "kaggle" / "MTeamSpellings.csv"
    out: Dict[str, int] = {}
    with open(path, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            out[row["TeamNameSpelling"].lower()] = int(row["TeamID"])
    return out


def canonical_to_kaggle_id(
    canonical_id: str,
    spellings_map: Dict[str, int],
) -> Optional[int]:
    """Try a cascade of name normalizations to bridge canonical -> Kaggle TeamID.

    Order matters — earlier candidates take precedence. A direct underscore
    -to-space match handles the common case (``duke`` -> ``duke``,
    ``michigan_state`` -> ``michigan state``). Special cases:

    - ``_s__`` -> ``'s `` resolves possessives (``saint_mary_s__ca`` -> ``saint mary's ca``).
    - ``_a_m`` -> `` a&m`` resolves ``texas_a_m`` -> ``texas a&m``.
    - ``__`` -> `` `` resolves the double-underscore separator from the
      canonicalizer's ``,``/``.`` substitution (``miami__fl`` -> ``miami fl``).
    - ``saint_`` -> ``st `` resolves the saint/st prefix difference.
    - Manual aliases handle the residual misses (``maryland_baltimore_county`` -> ``umbc``).
    """
    if canonical_id in CANONICAL_TO_KAGGLE_ALIAS:
        aliased = CANONICAL_TO_KAGGLE_ALIAS[canonical_id]
        if aliased in spellings_map:
            return spellings_map[aliased]

    # Build candidate spellings, in order of precedence.
    candidates = []

    naive = canonical_id.replace("__", " ").replace("_", " ").strip().lower()
    candidates.append(naive)

    # Possessive: replace `_s__` (e.g., 'st__john_s__ny') with `'s `.
    possessive = canonical_id.replace("_s__", "'s ").replace("__", " ").replace("_", " ").strip().lower()
    candidates.append(possessive)

    # texas a&m
    if "_a_m" in canonical_id:
        am = canonical_id.replace("_a_m", " a&m").replace("__", " ").replace("_", " ").strip().lower()
        candidates.append(am)

    # saint -> st prefix swap
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


def build_bridge(
    canonical_ids: Iterable[str],
    spellings_map: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map canonical_id <-> Kaggle TeamID for the supplied tournament field.

    Returns:
        (canonical_to_kaggle, kaggle_to_canonical). Teams that don't bridge
        are omitted from both dicts.
    """
    canon_to_kag: Dict[str, int] = {}
    kag_to_canon: Dict[int, str] = {}
    for tid in canonical_ids:
        kag = canonical_to_kaggle_id(tid, spellings_map)
        if kag is not None:
            canon_to_kag[tid] = kag
            kag_to_canon[kag] = tid
    return canon_to_kag, kag_to_canon
