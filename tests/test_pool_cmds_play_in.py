"""Play-in (First Four) resolution in the optimize-pool CLI path.

Before this fix, `_load_seeds`/`_load_regions` returned the raw seeds file
verbatim — up to 68 teams, with both First Four participants sharing one
(region, seed) slot. That duplicate reached
`bracket_construction._build_seed_map`, which correctly raises rather than
guessing, so `optimize-pool --mode seed` crashed on every year with an
unresolved play-in slot (every year on record — see
AUDIT_INDEPENDENT_EVALUATOR_2027.md finding H4). Resolution now happens once,
in `_resolve_play_ins`, shared by both loaders.
"""

import json

import pytest

from src.cli import pool_cmds as pc


def _write_seeds_file(tmp_path, monkeypatch, year, teams):
    hist_dir = tmp_path / "data" / "raw" / "historical"
    hist_dir.mkdir(parents=True)
    (hist_dir / f"tournament_seeds_{year}.json").write_text(
        json.dumps({"season": year, "teams": teams})
    )
    monkeypatch.chdir(tmp_path)


def _write_results_file(tmp_path, year, games):
    hist_dir = tmp_path / "data" / "raw" / "historical"
    hist_dir.mkdir(parents=True, exist_ok=True)
    (hist_dir / f"tournament_results_{year}.json").write_text(json.dumps({"games": games}))


PLAY_IN_TEAMS = [
    {"team_id": "alpha", "seed": 1, "region": "East"},
    {"team_id": "beta_a", "seed": 16, "region": "East"},  # play-in loser
    {"team_id": "beta_b", "seed": 16, "region": "East"},  # play-in winner
    {"team_id": "gamma", "seed": 2, "region": "West"},
]


def test_seeds_with_unresolved_play_in_keep_both_teams_when_no_results(tmp_path, monkeypatch):
    """No results on disk yet: the draw is genuinely undetermined, so both
    play-in teams remain — the caller (bracket_construction) is the right
    place for that to become a clear error, not this loader."""
    _write_seeds_file(tmp_path, monkeypatch, 2099, PLAY_IN_TEAMS)

    seeds = pc._load_seeds(2099)

    assert set(seeds) == {"alpha", "beta_a", "beta_b", "gamma"}


def test_seeds_with_resolved_play_in_drop_the_loser(tmp_path, monkeypatch):
    _write_seeds_file(tmp_path, monkeypatch, 2099, PLAY_IN_TEAMS)
    _write_results_file(
        tmp_path,
        2099,
        [{"round_name": "FF", "team1_id": "beta_a", "team2_id": "beta_b", "team1_won": False}],
    )

    seeds = pc._load_seeds(2099)
    regions = pc._load_regions(2099)

    assert set(seeds) == {"alpha", "beta_b", "gamma"}
    assert "beta_a" not in regions
    assert seeds["beta_b"] == 16


def test_no_duplicate_region_seed_slot_after_resolution(tmp_path, monkeypatch):
    _write_seeds_file(tmp_path, monkeypatch, 2099, PLAY_IN_TEAMS)
    _write_results_file(
        tmp_path,
        2099,
        [{"round_name": "FF", "team1_id": "beta_a", "team2_id": "beta_b", "team1_won": False}],
    )

    seeds = pc._load_seeds(2099)
    regions = pc._load_regions(2099)

    slots = [(regions.get(t), s) for t, s in seeds.items()]
    assert len(slots) == len(set(slots)), f"duplicate (region, seed) slot survived: {slots}"


def test_resolve_play_ins_is_a_noop_without_first_four_games(tmp_path, monkeypatch):
    _write_seeds_file(tmp_path, monkeypatch, 2099, PLAY_IN_TEAMS)
    _write_results_file(
        tmp_path,
        2099,
        [{"round_name": "R64", "team1_id": "alpha", "team2_id": "gamma", "team1_won": True}],
    )

    seeds = pc._load_seeds(2099)

    # No FF game in results -> nothing resolved, both play-in teams remain.
    assert {"beta_a", "beta_b"} <= set(seeds)


@pytest.mark.integration
@pytest.mark.slow
def test_real_2026_seeds_resolve_cleanly():
    """Regression for the actual crash: 2026's real seeds file has a First
    Four collision, and every optimize-pool mode failed on it. Runs against
    real repo data, not a fixture."""
    seeds = pc._load_seeds(2026)
    regions = pc._load_regions(2026)

    assert len(seeds) == 64, f"expected a resolved 64-team field, got {len(seeds)}"
    slots = [(regions.get(t), s) for t, s in seeds.items()]
    assert len(slots) == len(set(slots)), f"duplicate (region, seed) slot: {slots}"
