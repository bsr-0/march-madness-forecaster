"""Roster overlay values must land in the slots their names say.

`compute_roster_feature_overlay` used to return {hardcoded_index: value}
from a 71-wide team-vector layout. TEAM_FEATURE_DIM is 56 now. Two of the
indices (69, 70) raised IndexError; three of the in-range ones were silently
wrong: avg_experience and bench_depth were off by one (bench_depth overwrote
xp_per_poss), and top5_minutes_share (now index 39) was written into
injury_risk (54). A March 2026 commit had renumbered 74/75 -> 69/70 to stop
an earlier IndexError -- treating the symptom.

The overlay is now keyed by feature name and resolved against
TeamFeatures.get_feature_names() at load time. These tests pin that.
"""

import json

import pytest

from src.data.features.feature_engineering import TEAM_FEATURE_DIM, TeamFeatures
from src.pipeline.stages import data_loader as dl

NAMES = TeamFeatures.get_feature_names()


def _players(n=8):
    return [
        {
            "name": f"p{i}",
            "position": "G" if i % 2 else "F",
            "games_played": 30,
            "minutes_per_game": 30.0 - i,
            "warp": 1.0,
            "rapm_offensive": 1.0,
            "rapm_defensive": 0.5,
            "box_plus_minus": 2.0,
            "usage_rate": 20.0,
            "eligibility_year": 2,
            "is_transfer": False,
        }
        for i in range(n)
    ]


def test_overlay_is_keyed_by_feature_name():
    overlay = dl.compute_roster_feature_overlay(_players(), "team")
    assert overlay
    assert all(isinstance(k, str) for k in overlay), "overlay must be name-keyed, not index-keyed"
    assert set(overlay) <= set(NAMES)


def test_index_map_matches_the_canonical_layout():
    index_of = dl.roster_overlay_index_map()
    assert set(index_of) == {
        "total_rapm", "top5_rapm", "bench_rapm", "total_warp", "roster_continuity",
        "avg_experience", "bench_depth", "top5_minutes_share", "backcourt_rapm", "frontcourt_rapm",
    }
    for name, idx in index_of.items():
        assert 0 <= idx < TEAM_FEATURE_DIM
        assert NAMES[idx] == name


def test_the_previously_misplaced_slots_are_now_correct():
    """The old hardcoded map put these in the wrong place."""
    index_of = dl.roster_overlay_index_map()
    assert index_of["avg_experience"] == NAMES.index("avg_experience")      # was 17 (bench_depth's slot)
    assert index_of["bench_depth"] == NAMES.index("bench_depth")            # was 18 (xp_per_poss's slot)
    assert index_of["top5_minutes_share"] == NAMES.index("top5_minutes_share")  # was 54 (injury_risk's slot)
    assert index_of["backcourt_rapm"] < TEAM_FEATURE_DIM                    # was 69: IndexError
    assert index_of["frontcourt_rapm"] < TEAM_FEATURE_DIM                   # was 70: IndexError
    # And the slot the old map clobbered is not an overlay target at all.
    assert NAMES.index("xp_per_poss") not in index_of.values()
    assert NAMES.index("injury_risk") not in index_of.values()


def test_load_roster_overlay_returns_indices_within_the_vector(tmp_path):
    from datetime import timedelta

    year = 2024
    ts = (dl.TOURNAMENT_START_DATES[year] - timedelta(days=30)).isoformat()
    path = tmp_path / f"cbbpy_rosters_{year}.json"
    path.write_text(json.dumps({"year": year, "timestamp": ts, "teams": [{"team_id": "purdue", "players": _players()}]}))
    overlay = dl.load_roster_overlay(str(path), year=year)
    assert "purdue" in overlay
    idxs = overlay["purdue"]
    assert idxs and all(isinstance(i, int) and 0 <= i < TEAM_FEATURE_DIM for i in idxs)


def test_a_renamed_feature_fails_loudly(monkeypatch):
    monkeypatch.setattr(TeamFeatures, "get_feature_names", staticmethod(lambda **_k: [n for n in NAMES if n != "total_warp"] + ["renamed"]))
    with pytest.raises(KeyError) as exc:
        dl.roster_overlay_index_map()
    assert "total_warp" in str(exc.value)


def test_no_caller_hardcodes_overlay_indices():
    """Three copies of a hardcoded {index: value} map is how this bug survived
    two fixes. Any literal integer key at an overlay site is a regression."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in ("src/pipeline/stages/data_loader.py", "src/pipeline/stages/baseline_training/_orchestrator.py"):
        src = (root / rel).read_text()
        hits = re.findall(r"^\s+(?:11|12|13|14|15|17|18|54|69|70|74|75):\s", src, flags=re.M)
        assert not hits, f"{rel} hardcodes overlay indices again: {hits[:4]}"
