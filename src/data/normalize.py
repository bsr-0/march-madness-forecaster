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
import re
import unicodedata

# FIX #2: Quick alias table for the most common cross-source mismatches.
# These are the 7 known tournament-team misresolution risks from the audit
# plus additional high-frequency mismatches found in historical data.
# Maps raw normalized form → canonical ID.
_QUICK_ALIAS: dict[str, str] = {
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
    # Torvik short forms for specific schools
    "penn": "pennsylvania",
    "umbc": "maryland_baltimore_county",
    "liu": "long_island_university",
    "queens": "queens__nc",
    "prairie_view_a_m": "prairie_view",
    "cal_baptist": "california_baptist",
    # Kaggle team ID forms (Kaggle uses integer IDs mapped to these names)
    "miami_fl": "miami__fl",
    "miami_oh": "miami__oh",
    "saint_john_s": "st_john_s",
    "saint_joseph_s": "saint_joseph_s",
    "saint_peter_s": "saint_peter_s",
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
}


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
    return stripped if stripped != raw else raw


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
