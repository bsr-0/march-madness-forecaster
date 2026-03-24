"""Shared team ID resolver for mapping game-data IDs to Torvik canonical IDs.

Game data (CBBpy/ESPN) uses mascot-suffixed IDs like ``michigan_wolverines``
while Torvik uses short canonical IDs like ``michigan``.  This module provides
a single, consolidated resolver used by:

- ``scripts/repair_2026_data_quality.py``
- ``src/data/ingestion/collector.py`` (enrichment at ingestion time)
- ``src/pipeline/stages/data_loader.py`` (safety-net repair at pipeline time)
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set

from src.data.normalize import normalize_team_id

# Explicit aliases for game-data names/IDs that cannot be resolved via prefix
# matching or st/state expansion.  Keys are lowercased, underscore-delimited
# game IDs (as produced by the compute_*_from_games functions); values are the
# corresponding Torvik canonical IDs.
#
# Tournament-critical teams (barthag > 0.7) are marked.
_GAME_ID_ALIASES: Dict[str, str] = {
    # --- Tournament-critical (barthag > 0.7) ---
    "uconn_huskies": "connecticut",
    "miami_hurricanes": "miami_fl",
    "nc_state_wolfpack": "n_c__state",
    "mcneese_cowboys": "mcneese_st",
    # --- Other edge cases ---
    "app_state_mountaineers": "appalachian_st",
    "cal_baptist_lancers": "cal_baptist",
    "california_baptist_lancers": "cal_baptist",
    "cal_state_bakersfield_roadrunners": "cal_st__bakersfield",
    "cal_state_fullerton_titans": "cal_st__fullerton",
    "cal_state_northridge_matadors": "cal_st__northridge",
    "florida_international_panthers": "fiu",
    "grambling_tigers": "grambling_st",
    "uic_flames": "illinois_chicago",
    "iu_indianapolis_jaguars": "iu_indy",
    "long_island_university_sharks": "liu",
    "ul_monroe_warhawks": "louisiana_monroe",
    "loyola_maryland_greyhounds": "loyola_md",
    "miami_oh_redhawks": "miami_oh",
    "omaha_mavericks": "nebraska_omaha",
    "nicholls_colonels": "nicholls_st",
    "sam_houston_bearkats": "sam_houston_st",
    "san_jose_state_spartans": "san_jose_st",
    "se_louisiana_lions": "southeastern_louisiana",
    "ut_martin_skyhawks": "tennessee_martin",
    "texas_a_m_corpus_christi_islanders": "texas_a_m_corpus_chris",
    "kansas_city_roos": "umkc",
    "south_carolina_upstate_spartans": "usc_upstate",
    "william_mary_tribe": "william___mary",
    # Penn vs Penn State disambiguation
    "penn_quakers": "penn",
    "penn_state_nittany_lions": "penn_st",
    # Hawaii
    "hawaii_rainbow_warriors": "hawaii",
}

# Display-name aliases (used when the input is a display name like
# "UConn Huskies" rather than a lowercased game ID).
_DISPLAY_NAME_ALIASES: Dict[str, str] = {
    "App State Mountaineers": "appalachian_st",
    "Cal Baptist Lancers": "cal_baptist",
    "California Baptist Lancers": "cal_baptist",
    "Cal State Bakersfield Roadrunners": "cal_st__bakersfield",
    "Cal State Fullerton Titans": "cal_st__fullerton",
    "Cal State Northridge Matadors": "cal_st__northridge",
    "UConn Huskies": "connecticut",
    "Florida International Panthers": "fiu",
    "Grambling Tigers": "grambling_st",
    "UIC Flames": "illinois_chicago",
    "IU Indianapolis Jaguars": "iu_indy",
    "Long Island University Sharks": "liu",
    "UL Monroe Warhawks": "louisiana_monroe",
    "Loyola Maryland Greyhounds": "loyola_md",
    "McNeese Cowboys": "mcneese_st",
    "Miami Hurricanes": "miami_fl",
    "Miami (OH) RedHawks": "miami_oh",
    "Omaha Mavericks": "nebraska_omaha",
    "Nicholls Colonels": "nicholls_st",
    "Sam Houston Bearkats": "sam_houston_st",
    "San Jose State Spartans": "san_jose_st",
    "SE Louisiana Lions": "southeastern_louisiana",
    "UT Martin Skyhawks": "tennessee_martin",
    "Texas A&M-Corpus Christi Islanders": "texas_a_m_corpus_chris",
    "Kansas City Roos": "umkc",
    "South Carolina Upstate Spartans": "usc_upstate",
}


class GameToTorvikResolver:
    """Map game-data team IDs (mascot-suffixed) to Torvik canonical IDs.

    Combines four matching strategies in priority order:

    1. Explicit alias table (handles edge cases like ``fiu``, ``connecticut``)
    2. Exact match against Torvik ID set
    3. Longest-prefix matching with ``_st`` / ``_state`` expansion
    4. Collapsed-underscore matching (fallback)

    Usage::

        resolver = GameToTorvikResolver({"michigan", "michigan_st", "duke"})
        resolver.resolve("michigan_wolverines")    # -> "michigan"
        resolver.resolve("michigan_state_spartans")  # -> "michigan_st"
        resolver.resolve("duke_blue_devils")       # -> "duke"

        # Re-key a dict from game IDs to Torvik IDs
        remapped = resolver.remap_dict({"michigan_wolverines": {...}})
        # -> {"michigan": {...}}
    """

    def __init__(self, torvik_ids: Set[str]) -> None:
        self._torvik_ids = frozenset(torvik_ids)
        self._cache: Dict[str, Optional[str]] = {}

        # Build collapsed lookup: "michiganst" -> "michigan_st"
        self._collapsed_to_torvik: Dict[str, str] = {}
        for tid in torvik_ids:
            self._collapsed_to_torvik[tid.replace("_", "")] = tid

        # Build expanded forms for prefix matching:
        # "alabama_st" -> "alabama_state", "cal_st__fullerton" -> "cal_st_fullerton"
        self._expanded_to_torvik: Dict[str, str] = {}
        for tid in torvik_ids:
            self._expanded_to_torvik[tid] = tid
            # Normalize double underscores
            norm = tid.replace("__", "_")
            if norm != tid:
                self._expanded_to_torvik[norm] = tid
            # Expand _st to _state
            for variant in (tid, norm):
                if "_st_" in variant or variant.endswith("_st"):
                    expanded = variant.replace("_st_", "_state_")
                    if expanded == variant:
                        expanded = variant + "ate"  # _st -> _state
                    self._expanded_to_torvik[expanded] = tid

        # Sort longest-first so longest prefix wins
        self._expanded_sorted = sorted(
            self._expanded_to_torvik.keys(), key=len, reverse=True
        )

    def resolve(self, game_id: str) -> Optional[str]:
        """Resolve a game-data team ID to a Torvik canonical ID.

        Parameters
        ----------
        game_id : str
            Lowercased, underscore-delimited team ID from game data
            (e.g. ``"michigan_wolverines"``).

        Returns
        -------
        str or None
            The matching Torvik ID, or ``None`` if no match found.
        """
        if game_id in self._cache:
            return self._cache[game_id]

        result = self._resolve_uncached(game_id)
        self._cache[game_id] = result
        return result

    def _resolve_uncached(self, game_id: str) -> Optional[str]:
        # 1. Explicit alias
        alias = _GAME_ID_ALIASES.get(game_id)
        if alias and alias in self._torvik_ids:
            return alias

        # 2. Exact match
        if game_id in self._torvik_ids:
            return game_id

        # 3. Longest-prefix matching with st/state expansion
        for expanded_id in self._expanded_sorted:
            if game_id.startswith(expanded_id + "_") or game_id == expanded_id:
                return self._expanded_to_torvik[expanded_id]

        # 4. Collapsed-underscore fallback
        collapsed = game_id.replace("_", "")
        match = self._collapsed_to_torvik.get(collapsed)
        if match:
            return match

        return None

    def resolve_display_name(self, display_name: str) -> Optional[str]:
        """Resolve a display name (e.g. ``"UConn Huskies"``) to a Torvik ID.

        Tries the display-name alias table first, then falls back to
        progressive mascot stripping with ``state`` -> ``st`` normalization.
        """
        # Display-name alias
        alias = _DISPLAY_NAME_ALIASES.get(display_name)
        if alias and alias in self._torvik_ids:
            return alias

        # normalize_team_id handles many known aliases
        normalized = normalize_team_id(display_name)
        if normalized in self._torvik_ids:
            return normalized

        # Progressive mascot stripping: try shorter prefixes
        parts = display_name.split()
        for i in range(len(parts) - 1, 0, -1):
            candidate = "_".join(parts[:i]).lower()
            candidate = re.sub(r"[^a-z0-9_]", "_", candidate)
            candidate = re.sub(r"_+", "_", candidate).strip("_")
            if candidate in self._torvik_ids:
                return candidate
            # state -> st abbreviation (Torvik convention)
            st_version = re.sub(r"_state$", "_st", candidate)
            if st_version in self._torvik_ids:
                return st_version
            # Collapsed form
            collapsed = candidate.replace("_", "")
            match = self._collapsed_to_torvik.get(collapsed)
            if match:
                return match

        return None

    def remap_dict(self, ff_dict: Dict[str, dict]) -> Dict[str, dict]:
        """Re-key a Four Factors dict from game IDs to Torvik canonical IDs.

        Parameters
        ----------
        ff_dict : dict
            Mapping from game-data team IDs to stat dicts.

        Returns
        -------
        dict
            Mapping from Torvik canonical IDs to stat dicts.
            Unresolved keys are kept as-is for downstream fallback matching.
        """
        if not ff_dict:
            return ff_dict
        remapped: Dict[str, dict] = {}
        for game_id, vals in ff_dict.items():
            torvik_id = self.resolve(game_id)
            if torvik_id:
                if torvik_id not in remapped:
                    remapped[torvik_id] = vals
            else:
                # Keep original key for downstream collapsed matching
                remapped[game_id] = vals
        return remapped
