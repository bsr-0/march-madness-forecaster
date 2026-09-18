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
    "mount_st__mary_s": "mount st. mary's",  # 2025 field; Kaggle abbreviates with a period
    "saint_francis": "saint francis (pa)",  # 2025 field; Kaggle keeps the (NY)/(PA) suffix
    "stephen_f_austin": "stephen f. austin",  # 2014-2018 fields; initial takes a period
    "texas_a_m_corpus_christi": "texas a&m-corpus christi",  # 2022-2023 fields
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
    - A trailing ``_s`` -> ``'s`` resolves possessives with no suffix
      (``saint_peter_s`` -> ``saint peter's``).
    - A trailing ``__xx`` -> `` (xx)`` resolves Kaggle's parenthesised
      disambiguators (``loyola__il`` -> ``loyola (il)``).
    - ``_a_t`` -> `` a&t`` resolves ``north_carolina_a_t``.
    - ``-`` for ``_`` resolves Kaggle's hyphenated spellings
      (``nevada_las_vegas`` -> ``nevada-las-vegas``).
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

    # trailing possessive with no suffix: saint_peter_s -> saint peter's
    if canonical_id.endswith("_s"):
        candidates.append(possessive[:-2] + "'s")

    # parenthesised disambiguator: loyola__il -> loyola (il)
    if "__" in canonical_id and not canonical_id.endswith("__"):
        head, _, tail = canonical_id.rpartition("__")
        candidates.append(f"{head.replace('_', ' ')} ({tail.replace('_', ' ')})".strip().lower())

    # north carolina a&t
    if canonical_id.endswith("_a_t"):
        candidates.append(canonical_id[:-4].replace("__", " ").replace("_", " ").strip().lower() + " a&t")

    # hyphenated: nevada_las_vegas -> nevada-las-vegas
    candidates.append(canonical_id.replace("__", "-").replace("_", "-").strip().lower())

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
