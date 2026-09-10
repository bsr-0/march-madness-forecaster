"""Unit tests for scripts/real_pool_placement.py's own logic.

Uses synthetic saved-bracket files rather than the real 2023-2026 data, so
these run fast and pin the two behaviors that matter: (1) a single-pick
mode (meta_region_poolaware) is reported as-is, and (2) a multi-bracket
mode (seed) is reported as the MEAN real score across its saved brackets,
never the best — taking the best would be hindsight-biased, since the
brackets are saved sorted descending by their real-outcome score.
"""

import json

import pytest

from scripts import real_pool_placement as rpp


def _write_saved_modes(tmp_path, monkeypatch, year, modes):
    bracket_dir = tmp_path / "artifacts" / "backtest_brackets"
    bracket_dir.mkdir(parents=True)
    (bracket_dir / f"backtest_brackets_{year}.json").write_text(
        json.dumps({"year": year, "modes": modes})
    )
    monkeypatch.setattr(rpp, "BRACKET_DIR", bracket_dir)


def _bracket(score, champion="team_a"):
    return {"score_team_identity": score, "champion": champion}


def test_single_pick_mode_reported_as_is(tmp_path, monkeypatch):
    _write_saved_modes(
        tmp_path,
        monkeypatch,
        2030,
        [{"mode": "meta_region_poolaware", "brackets": [_bracket(500, "duke")]}],
    )
    monkeypatch.setattr(rpp, "_real_pool_scores", lambda year: [700.0, 600.0, 400.0, 300.0])

    r = rpp.placement_for_year(2030, "meta_region_poolaware")

    assert r["is_single_pick"] is True
    assert r["model_score"] == 500
    assert r["model_champion"] == "duke"
    # Beats 400 and 300 (2 entries), tied by none -> rank = 2 + 0 + 1 = 3
    assert r["rank"] == 3
    assert r["n_pool_entries"] == 4


def test_multi_bracket_mode_uses_mean_not_best(tmp_path, monkeypatch):
    # Brackets pre-sorted descending, as the production save-brackets code
    # writes them (mode_bracket_records.sort(key=lambda x: -x["score_team_identity"])).
    brackets = [_bracket(900, "a"), _bracket(500, "b"), _bracket(100, "c")]
    _write_saved_modes(
        tmp_path,
        monkeypatch,
        2030,
        [{"mode": "seed", "brackets": brackets}],
    )
    monkeypatch.setattr(rpp, "_real_pool_scores", lambda year: [700.0, 400.0])

    r = rpp.placement_for_year(2030, "seed")

    assert r["is_single_pick"] is False
    assert r["n_brackets"] == 3
    # Mean of 900/500/100 = 500, NOT the best (900).
    assert r["model_score"] == pytest.approx(500.0)
    assert r["model_champion"] == "3 different"
    # 500 beats only the 400 entry -> rank = 1 + 0 + 1 = 2
    assert r["rank"] == 2


def test_taking_best_of_n_would_have_given_a_different_rank(tmp_path, monkeypatch):
    """Guards the specific hindsight-bias bug: best-of-3 (900) would beat
    both real entries (rank 1), but the honest mean (500) does not."""
    brackets = [_bracket(900), _bracket(500), _bracket(100)]
    _write_saved_modes(tmp_path, monkeypatch, 2030, [{"mode": "seed", "brackets": brackets}])
    monkeypatch.setattr(rpp, "_real_pool_scores", lambda year: [700.0, 400.0])

    r = rpp.placement_for_year(2030, "seed")

    assert r["model_score"] != 900.0
    assert r["rank"] != 1


def test_missing_mode_returns_none(tmp_path, monkeypatch):
    _write_saved_modes(tmp_path, monkeypatch, 2030, [{"mode": "seed", "brackets": [_bracket(1)]}])
    assert rpp.placement_for_year(2030, "meta_region_poolaware") is None


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(rpp, "BRACKET_DIR", tmp_path / "does_not_exist")
    assert rpp.placement_for_year(2030, "seed") is None


def test_rank_against_pool_ties_are_conservative():
    # A tie is not credited as a win: placed after every tied real entry.
    result = rpp._rank_against_pool(500.0, [500.0, 500.0, 400.0])
    assert result["rank"] == 3  # 0 better + 2 tied + 1
    assert result["n_pool_entries"] == 3
