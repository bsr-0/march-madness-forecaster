"""Shared team ID / name normalization used across all pipeline modules.

This module consolidates the duplicated ``_normalize_team_id`` /
``_normalize_team_name`` functions that previously lived independently
in ``collector.py``, ``providers.py``, ``historical_pipeline.py``,
``materialization.py``, and ``tournament_bracket.py``.  Having a single
implementation prevents silent divergence (e.g. Unicode handling of
``San José State`` producing different IDs in different modules).
"""

from __future__ import annotations

import html as _html
import re
import unicodedata


def normalize_team_id(name: str) -> str:
    """Convert an arbitrary team name string to a canonical underscore-delimited ID.

    Steps:
    1. Decode HTML entities (``&amp;`` → ``&``)
    2. NFKD-normalize Unicode and strip combining marks (``é`` → ``e``)
    3. Lowercase
    4. Replace non-alphanumeric characters with ``_``
    5. Collapse repeated underscores and strip leading/trailing ``_``

    Examples::

        >>> normalize_team_id("Texas A&amp;M")
        'texas_a_m'
        >>> normalize_team_id("San José State")
        'san_jose_state'
        >>> normalize_team_id("DukeNCAA")
        'dukencaa'
    """
    if not name:
        return ""
    s = _html.unescape(str(name))
    # NFKD decomposition + strip combining characters (accents)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


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


_NCAA_SUFFIX_RE = re.compile(r"ncaa$", re.IGNORECASE)


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
