"""The harness must persist finished folds before starting the next one.

Found 2026-09-11: a nine-season walk-forward run died in fold four on a data
defect ("Region West has 15 teams") and the three completed folds -- forty
minutes -- were gone, because the result JSON was written only at the end.
"""

import json

from src.evaluation.backtest_harness import BacktestHarness


def test_checkpoint_written_with_per_year_keys(tmp_path):
    ckpt = tmp_path / "out" / "result.json.partial"
    h = BacktestHarness(historical_dir=str(tmp_path), years=[2016, 2017], checkpoint_path=str(ckpt), walk_forward=True)
    games = [{"team1": "a", "team2": "b", "seed1": 1, "seed2": 16, "round": "R64", "outcome": 1, "pipeline": 0.9}]
    h._write_checkpoint({2016: 0.2}, {2016: "pipeline"}, {2016: games}, {2016: 0.05}, [2016, 2017])

    data = json.loads(ckpt.read_text())
    assert data["partial"] is True
    assert data["walk_forward"] is True
    assert data["years_planned"] == [2016, 2017]
    assert data["years_completed"] == [2016]
    assert data["per_year_source"] == {"2016": "pipeline"}
    assert data["per_year_games"] == {"2016": games}
    assert data["per_year_brier"] == {"2016": 0.2}
    assert not ckpt.with_suffix(".partial.tmp").exists(), "temp file must be renamed into place"


def test_no_checkpoint_path_is_a_noop(tmp_path):
    h = BacktestHarness(historical_dir=str(tmp_path), years=[2016])
    h._write_checkpoint({}, {}, {}, {}, [2016])
    assert list(tmp_path.iterdir()) == []


def test_compare_script_accepts_a_checkpoint(tmp_path):
    """scripts/compare_pipeline_vs_pit reads the same keys, so an aborted run
    is still comparable on the seasons it finished."""
    from scripts.compare_pipeline_vs_pit import load_rows

    ckpt = tmp_path / "r.json.partial"
    h = BacktestHarness(historical_dir=str(tmp_path), years=[2016], checkpoint_path=str(ckpt), walk_forward=True)
    games = [
        {"team1": "a", "team2": "b", "seed1": 1, "seed2": 16, "round": "R64", "outcome": 1, "pipeline": 0.9},
        {"team1": "c", "team2": "d", "seed1": 16, "seed2": 16, "round": "FF", "outcome": 0, "pipeline": 0.5},
    ]
    h._write_checkpoint({2016: 0.2}, {2016: "pipeline"}, {2016: games}, {2016: 0.0}, [2016])
    rows, dropped = load_rows(ckpt)
    assert [r["team1"] for r in rows] == ["a"]
    assert dropped == {"round=FF": 1}
