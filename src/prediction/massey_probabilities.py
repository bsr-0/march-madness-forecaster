"""Massey source (catalog A5 ``massey_avg``).

Loads the already-aggregated Massey composite rating
(``data/raw/external_massey_composite_{year}.json``) and exposes it as
a barthag-shaped probability source that slots into the same
``base_round_probs`` dispatch used by torvik / odds / elo.

Why the composite (not per-system top-10 averaging):
- The catalog's original spec was "average across top-10 most stable
  systems" but our composite file is *already* an ensemble rating
  across all systems Massey tracks. Same signal, cleaner source.
- The per-system files (``external_POM_{year}.json``, etc.) exist and
  would support the catalog's A6 ``massey_best`` walk-forward
  Brier-selection approach — that's a separate phase of work deferred
  for now (see STRATEGY_CATALOG.md A6).

Team-ID bridge:
- Massey uses its own abbreviated IDs (``abilene_chr``, ``mt_st_mary_s``,
  ``ne_omaha``, ``siue``). ~91% of tournament teams match canonical IDs
  directly (62/68 for 2026); the remaining edge cases are handled with
  an explicit alias dict mirroring the pattern of the cbbpy bridge.

Output shape matches ``_load_torvik_barthag``:
``Dict[canonical_id, barthag ∈ (0, 1)]`` with seed-based fallback for
teams not present in the Massey composite for that year. Returns
``None`` when the composite file is missing for the year (the caller
skips this source for that year — typically 2026, which has not yet
been scraped into the composite).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

# Massey-ID → canonical-ID aliases for teams where the Massey composite uses
# an abbreviation that diverges from the Torvik/seeds canonical form. Mirrors
# the cbbpy bridge pattern in src/data/normalize.py but is Massey-specific
# (different scheme).
_MASSEY_EDGE_CASES: Dict[str, str] = {
    "american_univ": "american",
    "mt_st_mary_s": "mount_st__mary_s",
    "ne_omaha": "omaha",
    "st_francis_pa": "saint_francis",
    "st_mary_s_ca": "saint_mary_s__ca",
    "siue": "siu_edwardsville",
}

# Barthag clip range — matches the catalog spec for massey_avg.
_BARTHAG_MIN = 0.10
_BARTHAG_MAX = 0.99


def _bridge_massey_id(massey_id: str, canonical_ids: frozenset) -> Optional[str]:
    """Map a Massey team ID to a canonical tournament ID.

    Three-tier: exact match → explicit alias → None. No longest-prefix
    fallback here because Massey IDs are heavily abbreviated (``abilene_chr``)
    rather than mascot-suffixed — prefix matching would collide.
    """
    if massey_id in canonical_ids:
        return massey_id
    aliased = _MASSEY_EDGE_CASES.get(massey_id)
    if aliased is not None and aliased in canonical_ids:
        return aliased
    return None


def _seed_fallback_barthag(seed: Optional[int]) -> float:
    """Conservative seed-based barthag fallback when Massey has no team entry."""
    if seed is None:
        return 0.50
    return max(0.10, 1.0 - seed * 0.04)


def load_massey_avg_barthag(
    year: int,
    seeds: Dict[str, int],
    data_root: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """Massey composite rating → barthag per tournament team.

    Args:
        year: Tournament year.
        seeds: Canonical tournament team ID → seed. Used for the output
            keyset and the seed-based fallback.
        data_root: Repo data root. Defaults to ``<project>/data``.

    Returns:
        ``Dict[canonical_id, barthag ∈ (0, 1)]`` for every team in
        ``seeds``. Teams not in Massey get the seed fallback. Returns
        ``None`` if no Massey composite file exists for ``year`` (caller
        skips this source for that year — notably 2026 at 2026-04-24).
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"

    path = Path(data_root) / "raw" / f"external_massey_composite_{year}.json"
    if not path.exists():
        path = Path(data_root) / "raw" / "historical" / f"external_massey_composite_{year}.json"
    if not path.exists():
        return None

    raw = json.load(open(path))
    if not isinstance(raw, list) or not raw:
        return None

    canonical_ids = frozenset(seeds.keys())

    # Build canonical_id → normalized rating via the bridge.
    canonical_rating: Dict[str, float] = {}
    for team in raw:
        massey_id = team.get("team_id")
        norm = team.get("normalized")
        if massey_id is None or norm is None:
            continue
        canonical = _bridge_massey_id(massey_id, canonical_ids)
        if canonical is None:
            continue
        # If multiple Massey IDs map to the same canonical (shouldn't happen
        # but defensive), keep the highest-rated one.
        if canonical not in canonical_rating or norm > canonical_rating[canonical]:
            canonical_rating[canonical] = float(norm)

    # Build output using seed-based fallback for unresolved teams.
    result: Dict[str, float] = {}
    for canonical, seed in seeds.items():
        if canonical in canonical_rating:
            # Clip per catalog spec to avoid numerical edge cases.
            barthag = max(_BARTHAG_MIN, min(_BARTHAG_MAX, canonical_rating[canonical]))
        else:
            barthag = _seed_fallback_barthag(seed)
        result[canonical] = barthag
    return result


def massey_coverage(year: int, canonical_ids: Iterable[str], data_root: Optional[Path] = None) -> Dict[str, int]:
    """Diagnostic: how many canonical IDs bridge to the Massey composite?

    Useful for regression tests — if a future data refresh breaks the
    bridge, this surfaces the delta without touching the barthag math.
    """
    if data_root is None:
        data_root = Path(__file__).resolve().parent.parent.parent / "data"
    path = Path(data_root) / "raw" / f"external_massey_composite_{year}.json"
    if not path.exists():
        return {"covered": 0, "total": len(list(canonical_ids)), "file_exists": 0}
    raw = json.load(open(path))
    canonical_set = frozenset(canonical_ids)
    covered = sum(1 for team in raw if _bridge_massey_id(team.get("team_id", ""), canonical_set) is not None)
    return {"covered": covered, "total": len(canonical_set), "file_exists": 1}
