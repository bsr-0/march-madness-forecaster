"""Unit tests for the cbbpy team-ID bridge.

cbbpy's game archives use mascot-suffixed team IDs (``illinois_fighting_illini``,
``james_madison_dukes``, ``new_mexico_state_aggies``) that diverge from the
Torvik / seeds canonical forms (``illinois``, ``james_madison``,
``new_mexico_state``). ``bridge_cbbpy_id`` is what Elo (A4), roster_adj (C2),
and volatile (D1) rely on to bind cbbpy game records to the tournament field.

The coverage test exercises the bridge against the real 2026 cbbpy ID set
and asserts we now cover every tournament team (vs 17/68 before the bridge).
"""

import json
from pathlib import Path

from src.data.normalize import _CBBPY_EDGE_CASES, bridge_cbbpy_id

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _load_2026_tournament_ids() -> set:
    seeds_path = DATA_ROOT / "raw" / "tournament_seeds_2026.json"
    seeds = json.load(open(seeds_path))
    return {s["team_id"] for s in seeds}


def _load_2026_cbbpy_ids() -> set:
    games_path = DATA_ROOT / "raw" / "historical" / "historical_games_2026.json"
    games = json.load(open(games_path))["games"]
    return {t for g in games for t in (g["team1_id"], g["team2_id"])}


def test_bridge_exact_match_passthrough():
    """Teams whose cbbpy ID is already canonical (e.g., ``duke``) pass through."""
    canonical = {"duke", "kansas", "florida"}
    assert bridge_cbbpy_id("duke", canonical) == "duke"
    assert bridge_cbbpy_id("kansas", canonical) == "kansas"


def test_bridge_longest_prefix_match():
    """The 82%-common case: mascot suffix stripped via prefix match."""
    canonical = {"illinois", "james_madison", "new_mexico_state"}
    assert bridge_cbbpy_id("illinois_fighting_illini", canonical) == "illinois"
    assert bridge_cbbpy_id("james_madison_dukes", canonical) == "james_madison"
    assert bridge_cbbpy_id("new_mexico_state_aggies", canonical) == "new_mexico_state"


def test_bridge_longest_prefix_prefers_more_specific():
    """``north_carolina_state`` must bind NC State, not North Carolina."""
    canonical = {"north_carolina", "north_carolina_state"}
    # cbbpy ID for NC State — longest-prefix must win.
    assert bridge_cbbpy_id("north_carolina_state_wolfpack", canonical) == "north_carolina_state"
    # cbbpy ID for UNC — falls through the longer prefix (no trailing _state).
    assert bridge_cbbpy_id("north_carolina_tar_heels", canonical) == "north_carolina"


def test_bridge_edge_case_aliases_applied():
    """The 12 cbbpy divergences (UConn, BYU, SMU, VCU, ...) each map correctly."""
    # Build canonical set from all 12 edge cases' canonical IDs plus a few extras.
    canonical = set(_CBBPY_EDGE_CASES.values()) | {"duke", "kansas"}
    for cbbpy_id, expected in _CBBPY_EDGE_CASES.items():
        assert bridge_cbbpy_id(cbbpy_id, canonical) == expected, f"Edge case {cbbpy_id!r} should map to {expected!r}"


def test_bridge_returns_none_for_unknown():
    """An ID that doesn't match any canonical team returns None, not a silent wrong answer."""
    assert bridge_cbbpy_id("nonexistent_zips", {"duke", "kansas"}) is None


def test_bridge_returns_none_for_empty_input():
    assert bridge_cbbpy_id("", {"duke"}) is None


def test_bridge_covers_all_2026_tournament_teams():
    """The end-to-end gate: every 2026 tournament team has a matching cbbpy ID via the bridge.

    Before the bridge we measured 17/68 coverage (missing 51). After the
    longest-prefix + edge-case bridge we expect 68/68 — every tournament
    team can now be resolved from cbbpy game data, unblocking A4/C2/D1.
    """
    tourney_ids = _load_2026_tournament_ids()
    cbbpy_ids = _load_2026_cbbpy_ids()

    covered_tourney_teams = set()
    for cbbpy_id in cbbpy_ids:
        canonical = bridge_cbbpy_id(cbbpy_id, tourney_ids)
        if canonical is not None:
            covered_tourney_teams.add(canonical)

    missing = tourney_ids - covered_tourney_teams
    assert not missing, f"Expected all 2026 tournament teams to bridge from cbbpy data; missing: {sorted(missing)}"


def test_bridge_does_not_collide_on_shared_prefixes():
    """Teams with overlapping canonical names must each map to their own canonical form.

    This is the substantive test that longest-prefix matching handles common
    prefix structures — e.g. ``miami`` is the conceptual shared prefix but
    there's no canonical team literally named ``miami`` (Torvik uses
    ``miami__fl`` with a disambiguator).
    """
    canonical = {"miami__fl", "miami__oh"}
    # cbbpy IDs for each (from _CBBPY_EDGE_CASES)
    assert bridge_cbbpy_id("miami_hurricanes", canonical) == "miami__fl"
    assert bridge_cbbpy_id("miami_oh_redhawks", canonical) == "miami__oh"
