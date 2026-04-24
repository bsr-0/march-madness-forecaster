"""Shared team ID / name normalization used across all pipeline modules.

This module consolidates the duplicated ``_normalize_team_id`` /
``_normalize_team_name`` functions that previously lived independently
in ``collector.py``, ``providers.py``, ``historical_pipeline.py``,
``materialization.py``, and ``tournament_bracket.py``.  Having a single
implementation prevents silent divergence (e.g. Unicode handling of
``San José State`` producing different IDs in different modules).

FIX #2: Added resolver-backed canonical resolution.  The raw string
normalization alone cannot handle cross-source naming mismatches
(CBBpy mascots vs Sports Reference school names vs Kaggle canonical IDs).
The ``_QUICK_ALIAS`` table maps the most common mismatches deterministically
without the overhead of fuzzy matching.
"""

from __future__ import annotations

import html as _html
import json
import logging
from typing import Iterable
import re
import unicodedata
from pathlib import Path

_logger = logging.getLogger(__name__)

# FIX #2: Quick alias table for the most common cross-source mismatches.
# These are the 7 known tournament-team misresolution risks from the audit
# plus additional high-frequency mismatches found in historical data.
# Maps raw normalized form → canonical ID.
#
# Primary source: configs/team_aliases.json (versioned, easy to update).
# Inline fallback below is used when the config file is missing.
_QUICK_ALIAS_INLINE: dict[str, str] = {
    # Audit-identified critical mismatches
    "unc": "north_carolina",
    "byu": "brigham_young",
    "uconn": "connecticut",
    "vcu": "virginia_commonwealth",
    "ole_miss": "mississippi",
    "saint_mary_s": "saint_mary_s__ca",
    "texas_a_amp_m": "texas_a_m",
    # CBBpy mascot-suffixed forms (top 30 tournament programs)
    "duke_blue_devils": "duke",
    "kansas_jayhawks": "kansas",
    "kentucky_wildcats": "kentucky",
    "north_carolina_tar_heels": "north_carolina",
    "gonzaga_bulldogs": "gonzaga",
    "villanova_wildcats": "villanova",
    "michigan_state_spartans": "michigan_state",
    "virginia_cavaliers": "virginia",
    "purdue_boilermakers": "purdue",
    "houston_cougars": "houston",
    "alabama_crimson_tide": "alabama",
    "tennessee_volunteers": "tennessee",
    "auburn_tigers": "auburn",
    "connecticut_huskies": "connecticut",
    "baylor_bears": "baylor",
    "arizona_wildcats": "arizona",
    "creighton_bluejays": "creighton",
    "marquette_golden_eagles": "marquette",
    "iowa_state_cyclones": "iowa_state",
    "texas_longhorns": "texas",
    "florida_gators": "florida",
    "indiana_hoosiers": "indiana",
    "oregon_ducks": "oregon",
    "clemson_tigers": "clemson",
    "san_diego_state_aztecs": "san_diego_state",
    "florida_atlantic_owls": "florida_atlantic",
    "northwestern_wildcats": "northwestern",
    "memphis_tigers": "memphis",
    "drake_bulldogs": "drake",
    "colorado_state_rams": "colorado_state",
    # Sports Reference slug mismatches
    "louisiana_state": "louisiana_state",
    "lsu": "louisiana_state",
    "smu": "southern_methodist",
    "usc": "southern_california",
    "ucf": "ucf",
    "fau": "florida_atlantic",
    "fdu": "fairleigh_dickinson",
    # Common abbreviation mismatches
    "nc_state": "nc_state",
    "n_c_state": "nc_state",
    "pitt": "pittsburgh",
    "cuse": "syracuse",
    "wisc": "wisconsin",
    "mich": "michigan",
    "mich_st": "michigan_state",
    "miss_st": "mississippi_state",
    "ok_state": "oklahoma_state",
    "okla_st": "oklahoma_state",
    # Torvik abbreviated "_st" forms → canonical "_state" forms
    "iowa_st": "iowa_state",
    "michigan_st": "michigan_state",
    "ohio_st": "ohio_state",
    "utah_st": "utah_state",
    "north_dakota_st": "north_dakota_state",
    "south_dakota_st": "south_dakota_state",
    "tennessee_st": "tennessee_state",
    "kennesaw_st": "kennesaw_state",
    "mcneese_st": "mcneese_state",
    "wright_st": "wright_state",
    "penn_st": "penn_state",
    "oregon_st": "oregon_state",
    "colorado_st": "colorado_state",
    "fresno_st": "fresno_state",
    "boise_st": "boise_state",
    "kansas_st": "kansas_state",
    "arizona_st": "arizona_state",
    "mississippi_st": "mississippi_state",
    "wichita_st": "wichita_state",
    "san_diego_st": "san_diego_state",
    "san_jose_st": "san_jose_state",
    "ball_st": "ball_state",
    "florida_st": "florida_state",
    "georgia_st": "georgia_state",
    "cleveland_st": "cleveland_state",
    "murray_st": "murray_state",
    "montana_st": "montana_state",
    "morehead_st": "morehead_state",
    "norfolk_st": "norfolk_state",
    "oklahoma_st": "oklahoma_state",
    "washington_st": "washington_state",
    "jacksonville_st": "jacksonville_state",
    "kent_st": "kent_state",
    "long_beach_st": "long_beach_state",
    "new_mexico_st": "new_mexico_state",
    "grambling_st": "grambling",
    "cal_st__fullerton": "cal_state_fullerton",
    "cal_st_fullerton": "cal_state_fullerton",
    "cal_st__bakersfield": "cal_state_bakersfield",
    "cal_st_bakersfield": "cal_state_bakersfield",
    "cal_st__northridge": "cal_state_northridge",
    "cal_st_northridge": "cal_state_northridge",
    # Additional _st → _state mappings found in historical data
    "illinois_st": "illinois_state",
    "indiana_st": "indiana_state",
    "texas_st": "texas_state",
    "missouri_st": "missouri_state",
    "arkansas_st": "arkansas_state",
    "idaho_st": "idaho_state",
    "appalachian_st": "appalachian_state",
    "portland_st": "portland_state",
    "jackson_st": "jackson_state",
    "weber_st": "weber_state",
    "east_tennessee_st": "east_tennessee_state",
    "sam_houston_st": "sam_houston_state",
    "youngstown_st": "youngstown_state",
    "coppin_st": "coppin_state",
    "delaware_st": "delaware_state",
    "alcorn_st": "alcorn_state",
    "alabama_st": "alabama_state",
    "morgan_st": "morgan_state",
    "sacramento_st": "sacramento_state",
    "south_carolina_st": "south_carolina_state",
    "north_carolina_st": "nc_state",
    "nc_st": "nc_state",
    "bowie_st": "bowie_state",
    "northwestern_st": "northwestern_state",
    # Abbreviation aliases
    "unlv": "nevada_las_vegas",
    "fiu": "florida_international",
    "umkc": "kansas_city",
    "loyola_chicago": "loyola__il",
    "albany": "albany__ny",
    "nebraska_omaha": "omaha",
    # Torvik short forms for specific schools
    "penn": "pennsylvania",
    "umbc": "maryland_baltimore_county",
    "liu": "long_island_university",
    "queens": "queens__nc",
    "prairie_view_a_m": "prairie_view",
    "cal_baptist": "california_baptist",
    "charleston": "college_of_charleston",
    "texas_a_m_corpus_chris": "texas_a_m_corpus_christi",
    # Collapsed double-underscore forms (normalize_team_id collapses __ to _)
    "st_john_s": "st__john_s__ny",
    "st_john_s_ny": "st__john_s__ny",
    "saint_mary_s_ca": "saint_mary_s__ca",
    "queens_nc": "queens__nc",
    # Kaggle team ID forms (Kaggle uses integer IDs mapped to these names)
    "miami_fl": "miami__fl",
    "miami_oh": "miami__oh",
    "saint_john_s": "st__john_s__ny",
    "saint_joseph_s": "saint_joseph_s",
    "saint_peter_s": "saint_peter_s",
    "st_peter_s": "saint_peter_s",
    "texas_a_m": "texas_a_m",
    # Additional cross-source forms for teams with ambiguous short names
    "unc_asheville": "unc_asheville",
    "unc_greensboro": "unc_greensboro",
    "unc_wilmington": "unc_wilmington",
    "north_carolina_asheville": "unc_asheville",
    "north_carolina_greensboro": "unc_greensboro",
    "north_carolina_wilmington": "unc_wilmington",
    # NCAA suffix forms (Sports Reference tournament pages)
    "dukencaa": "duke",
    "kansasncaa": "kansas",
    "kentuckyncaa": "kentucky",
    "north_carolinancaa": "north_carolina",
    "gonzagancaa": "gonzaga",
    "connecticutncaa": "connecticut",
    "purduencaa": "purdue",
    "houstonncaa": "houston",
    "alabamancaa": "alabama",
    "tennesseencaa": "tennessee",
    "auburnncaa": "auburn",
    "floridancaa": "florida",
    "michiganncaa": "michigan",
    "michigan_statencaa": "michigan_state",
    "arizonancaa": "arizona",
    "texasncaa": "texas",
    "iowa_statencaa": "iowa_state",
    # From proprietary_metrics.py inline aliases
    "mcneese": "mcneese_state",
    "american_university": "american",
    # Cross-source aliases found during data audit (2026-03)
    "loyola_il": "loyola__il",
    "ucsb": "uc_santa_barbara",
    "nicholls": "nicholls_state",
    "southeast_missouri": "southeast_missouri_state",
    "st_joseph_s": "saint_joseph_s",
    "etsu": "east_tennessee_state",
    "saint_francis_pa": "saint_francis",
    # Collapsed double-underscore forms that need round-trip stability
    "albany_ny": "albany__ny",
    "mount_st_mary_s": "mount_st__mary_s",
    # Torvik Four Factors ↔ main Torvik team ID mismatches (data audit 2026-03)
    "sam_houston": "sam_houston_state",
    "southern_miss": "southern_mississippi",
    "central_connecticut": "central_connecticut_state",
    "umass_lowell": "massachusetts_lowell",
    "ut_rio_grande_valley": "texas_rio_grande_valley",
    "usc_upstate": "south_carolina_upstate",
}


def _load_aliases() -> dict[str, str]:
    """Load alias map from configs/team_aliases.json, falling back to inline dict."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "team_aliases.json"
    try:
        with open(config_path) as f:
            data = json.load(f)
        aliases = data.get("aliases", {})
        if aliases:
            _logger.debug(
                "Loaded %d team aliases from %s (version %s)", len(aliases), config_path, data.get("version", "?")
            )
            return aliases
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        _logger.debug("Could not load team aliases config (%s), using inline fallback", exc)
    return _QUICK_ALIAS_INLINE


_QUICK_ALIAS: dict[str, str] = _load_aliases()

_NCAA_SUFFIX_RE = re.compile(r"ncaa$", re.IGNORECASE)


def _raw_normalize(name: str) -> str:
    """Raw string normalization without alias resolution."""
    if not name:
        return ""
    s = _html.unescape(str(name))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalize_team_id(name: str) -> str:
    """Convert an arbitrary team name string to a canonical underscore-delimited ID.

    Steps:
    1. Decode HTML entities (``&amp;`` → ``&``)
    2. NFKD-normalize Unicode and strip combining marks (``é`` → ``e``)
    3. Lowercase
    4. Replace non-alphanumeric characters with ``_``
    5. Collapse repeated underscores and strip leading/trailing ``_``
    6. Check quick alias table for known cross-source mismatches
    7. Strip NCAA suffix if present

    Examples::

        >>> normalize_team_id("Texas A&amp;M")
        'texas_a_m'
        >>> normalize_team_id("San José State")
        'san_jose_state'
        >>> normalize_team_id("UConn")
        'connecticut'
        >>> normalize_team_id("BYU")
        'brigham_young'
        >>> normalize_team_id("DukeNCAA")
        'duke'
    """
    raw = _raw_normalize(name)
    if not raw:
        return ""
    # FIX #2: Check quick alias table first (O(1) lookup, no fuzzy matching)
    if raw in _QUICK_ALIAS:
        return _QUICK_ALIAS[raw]
    # Strip NCAA suffix (Sports Reference tournament qualifier)
    stripped = _NCAA_SUFFIX_RE.sub("", raw).rstrip("_") if _NCAA_SUFFIX_RE.search(raw) else raw
    if stripped != raw and stripped in _QUICK_ALIAS:
        return _QUICK_ALIAS[stripped]
    # Generic _st -> _state suffix expansion.  All 62+ Torvik teams ending
    # in "_st" are State schools (verified via audit).  The alias table is
    # checked first so explicit overrides (e.g. grambling_st -> grambling)
    # still take priority.
    target = stripped if stripped != raw else raw
    if target.endswith("_st"):
        return target + "ate"  # _st -> _state
    return target


def normalize_team_name(name: str) -> str:
    """Normalize a team display-name for fuzzy comparison.

    Produces a lowercase, space-separated string with HTML entities decoded,
    Unicode normalized, and common stop-words removed.
    """
    if not name:
        return ""
    s = _html.unescape(str(name))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t not in {"the", "university", "college", "at", "of"}]
    return " ".join(tokens)


def strip_ncaa_suffix(team_id: str) -> str:
    """Strip the ``NCAA`` suffix that Sports Reference appends to tournament qualifiers.

    Example: ``alabamancaa`` → ``alabama``
    """
    if team_id and _NCAA_SUFFIX_RE.search(team_id):
        return _NCAA_SUFFIX_RE.sub("", team_id).rstrip("_")
    return team_id


def strip_ncaa_suffix_name(name: str) -> str:
    """Strip ``NCAA`` suffix from a display name.

    Example: ``AlabamaNCAA`` → ``Alabama``
    """
    if name and re.search(r"NCAA$", name):
        return re.sub(r"NCAA$", "", name).rstrip()
    return name


# cbbpy (scraped game results) uses mascot-suffixed team IDs that the rest of the
# repo doesn't. The bridge below maps `illinois_fighting_illini` → `illinois` and
# the ~12 edge cases where cbbpy uses an abbreviation or disambiguator that
# diverges from Torvik's canonical form (UConn, BYU, SMU, VCU, the two Miamis, etc.).
#
# 82% of the 2026 tournament field (56/68) bridges via simple longest-prefix
# match against known canonical IDs. The remaining 12 need explicit aliases,
# measured against `data/raw/historical/historical_games_{year}.json`.
_CBBPY_EDGE_CASES: dict[str, str] = {
    # cbbpy_id → canonical Torvik/seeds ID
    "byu_cougars": "brigham_young",
    "uconn_huskies": "connecticut",
    "hawai_i_rainbow_warriors": "hawaii",
    "umbc_retrievers": "maryland_baltimore_county",
    "mcneese_cowboys": "mcneese_state",
    "miami_hurricanes": "miami__fl",
    "miami_oh_redhawks": "miami__oh",
    "queens_university_royals": "queens__nc",
    "saint_mary_s_gaels": "saint_mary_s__ca",
    "smu_mustangs": "southern_methodist",
    "st_john_s_red_storm": "st__john_s__ny",
    "vcu_rams": "virginia_commonwealth",
}


def bridge_cbbpy_id(cbbpy_id: str, canonical_ids: Iterable[str]) -> str | None:
    """Map a cbbpy-style mascot-suffixed team ID to a canonical ID.

    Tries three strategies, in order:
      1. **Exact match** — the cbbpy ID is already canonical (happens for teams
         whose names Torvik spells the same way, e.g. ``duke`` and ``duke``).
      2. **Edge-case alias** — explicit mapping in ``_CBBPY_EDGE_CASES`` for
         the 12 teams where cbbpy diverges materially (UConn/BYU/SMU/...).
      3. **Longest-prefix match** — find the longest element of
         ``canonical_ids`` that is a prefix of ``cbbpy_id`` followed by ``_``.
         This catches the bulk of cases: ``illinois_fighting_illini`` →
         ``illinois``; ``james_madison_dukes`` → ``james_madison``.

    Longest-prefix ordering matters: ``north_carolina`` is a prefix of
    ``north_carolina_state``, so without length-sorting a North Carolina State
    cbbpy ID could be misrouted to North Carolina.

    Args:
        cbbpy_id: Snake-cased team ID from ``historical_games_*.json``
            (already lowercase, alphanumeric + underscore).
        canonical_ids: Iterable of canonical IDs to match against — typically
            the tournament seeds set or the Torvik ``team_id`` column.

    Returns:
        The canonical ID if a match was found, else ``None``.
    """
    if not cbbpy_id:
        return None
    canonical_set = frozenset(canonical_ids)
    if cbbpy_id in canonical_set:
        return cbbpy_id
    edge = _CBBPY_EDGE_CASES.get(cbbpy_id)
    if edge is not None and edge in canonical_set:
        return edge
    # Longest-prefix fallback: sort candidates by length descending so the
    # most specific prefix wins (north_carolina_state before north_carolina).
    for canonical in sorted(canonical_set, key=len, reverse=True):
        if cbbpy_id.startswith(canonical + "_"):
            return canonical
    return None
