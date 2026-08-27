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

from src.data.normalize import _CBBPY_EDGE_CASES, bridge_cbbpy_id, resolve_cbbpy_bridge

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


def test_resolve_bridge_drops_lower_division_impostors():
    """A non-D1 school sharing a D1 name prefix must not claim the canonical ID.

    ``virginia_union_panthers`` (Division II) prefix-matches ``virginia``
    exactly the way ``virginia_cavaliers`` does, so per-ID bridging hands
    both of them the same canonical team. ``resolve_cbbpy_bridge`` breaks the
    tie on weight — the real team dominates any dataset built from D1 games
    or rosters — and drops the loser entirely so callers' ``.get()`` misses it.
    """
    canonical = {"virginia", "arkansas"}
    weights = {
        "virginia_cavaliers": 31,
        "virginia_union_panthers": 2,
        "arkansas_razorbacks": 33,
        "arkansas_tech_wonder_boys": 1,
        "montana_state_bobcats": 30,  # bridges nowhere
    }
    resolved = resolve_cbbpy_bridge(weights, canonical)
    assert resolved == {"virginia_cavaliers": "virginia", "arkansas_razorbacks": "arkansas"}
    assert resolved.get("virginia_union_panthers") is None


def test_resolve_bridge_is_deterministic_on_tied_weights():
    """Equal weights must not make the winner depend on dict ordering."""
    canonical = {"virginia"}
    forward = resolve_cbbpy_bridge({"virginia_cavaliers": 5, "virginia_union_panthers": 5}, canonical)
    reverse = resolve_cbbpy_bridge({"virginia_union_panthers": 5, "virginia_cavaliers": 5}, canonical)
    assert forward == reverse


def test_resolve_bridge_keeps_exact_matches():
    """An ID that is already canonical still resolves to itself."""
    resolved = resolve_cbbpy_bridge({"duke": 30, "duke_blue_devils": 1}, {"duke"})
    assert resolved == {"duke": "duke"}


def test_resolve_bridge_uses_universe_to_separate_d1_schools():
    """Another D1 school must not claim a canonical ID just because the target
    set is a subset of D1.

    ``alabama_state_hornets`` prefix-matches ``alabama`` whenever
    ``alabama_state`` is missing from the set being matched against — which is
    exactly what happens when the target set is the 68 tournament teams. It is
    not rescuable by weighting: Alabama State played more games than Alabama in
    2026. Only the full D1 universe fixes it.
    """
    tourney = {"alabama"}
    universe = {"alabama", "alabama_state", "alabama_a_m"}
    weights = {
        "alabama": 31,
        "alabama_state_hornets": 32,  # deliberately outweighs the real team
        "alabama_a_m_bulldogs": 31,
    }
    assert resolve_cbbpy_bridge(weights, tourney, universe=universe) == {"alabama": "alabama"}
    # Without the universe the heavier impostor wins — the bug this guards.
    assert resolve_cbbpy_bridge(weights, tourney) == {"alabama_state_hornets": "alabama"}


def test_edge_cases_cover_teams_the_prefix_fallback_cannot_reach():
    """Teams whose cbbpy name shares no prefix with the canonical ID.

    Each of these was found by sweeping every seeded team in 2011-2026 for ones
    that bridge to nothing. ``ole_miss_rebels`` is the important one: before it
    was listed, ``mississippi`` prefix-matched ``mississippi_state_bulldogs``
    and Ole Miss silently ran on Mississippi State's roster and game log.
    """
    cases = {
        "ole_miss_rebels": "mississippi",
        "lsu_tigers": "louisiana_state",
        "usc_trojans": "southern_california",
        "unlv_rebels": "nevada_las_vegas",
        "ualbany_great_danes": "albany__ny",
        "charleston_cougars": "college_of_charleston",
        "loyola_chicago_ramblers": "loyola__il",
        "loyola_maryland_greyhounds": "loyola_md",
        "app_state_mountaineers": "appalachian_state",
        "mount_st_mary_s_mountaineers": "mount_st__mary_s",
    }
    for cbbpy_id, expected in cases.items():
        assert bridge_cbbpy_id(cbbpy_id, {expected}) == expected, cbbpy_id

    # Ole Miss must win "mississippi" over Mississippi State even when both
    # are present and the state school carries more data.
    resolved = resolve_cbbpy_bridge(
        {"ole_miss_rebels": 31, "mississippi_state_bulldogs": 34},
        {"mississippi"},
        universe={"mississippi", "mississippi_state"},
    )
    assert resolved == {"ole_miss_rebels": "mississippi"}


def test_resolve_bridge_weight_must_measure_volume_not_roster_size():
    """A per-game AVERAGE is the wrong weight, and it loses to a tiny impostor.

    This is the second half of the defect that took 17 of 1,084 team-seasons in
    generate_team_stats_table, and the half the universe argument does NOT fix.
    virginia_lynchburg is not Division I, so no universe can exclude it -- it
    has to lose on weight. It does, on any weight that measures how much of the
    dataset belongs to the team, and it WINS on summed minutes-per-game,
    because mpg is already a per-game average and summing it measures roster
    size times rotation depth instead.

    Real 2024 figures: Virginia 13 players, 34 games, 346 player-games, 220.5
    summed mpg. Virginia-Lynchburg 17 players, 8 games, 78 player-games, 326.1
    summed mpg.
    """
    canonical = {"virginia"}
    universe = {"virginia", "virginia_tech", "virginia_commonwealth"}

    summed_mpg = {"virginia_cavaliers": 220.5, "virginia_lynchburg_dragons": 326.1}
    assert resolve_cbbpy_bridge(summed_mpg, canonical, universe=universe) == {
        "virginia_lynchburg_dragons": "virginia"
    }, "summed minutes-per-game hands the canonical id to a non-D1 school"

    player_games = {"virginia_cavaliers": 346, "virginia_lynchburg_dragons": 78}
    assert resolve_cbbpy_bridge(player_games, canonical, universe=universe) == {
        "virginia_cavaliers": "virginia"
    }, "player-games measures dataset volume and resolves correctly"
