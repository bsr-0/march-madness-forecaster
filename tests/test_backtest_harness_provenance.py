"""The LOYO harness must never pass off the seed baseline as the model.

Found 2026-09-10: no ``teams_{year}.json`` has ever existed on disk -- it is a
virtual path the loader redirects into ``tournament_context_{year}.json`` --
but the pre-run validator checked the literal string, so every fold failed
pre-run validation, and the harness then silently substituted the seed
baseline and reported its Brier in the model's column with only a log
warning. ``configs/backtest_baseline.json``'s 2025 entry (0.141963) is that
substitution, and a fresh run reproduces it to four decimals.

Three guards, one per link in that chain.
"""

import json
from pathlib import Path

import pytest

from src.data.loader import DataLoader
from src.evaluation.backtest_harness import BacktestHarness


# --- 1. The validator must honour the virtual teams path -------------------


def test_virtual_teams_path_resolves_to_context_file(tmp_path):
    ctx = tmp_path / "tournament_context_2030.json"
    ctx.write_text(json.dumps({"teams": {"teams": [{"team_id": "a", "seed": 1, "region": "East"}]}}))
    virtual = tmp_path / "teams_2030.json"
    assert not virtual.exists()
    assert DataLoader.resolve_teams_json_path(str(virtual)) == str(ctx)


def test_virtual_teams_path_without_context_is_absent(tmp_path):
    assert DataLoader.resolve_teams_json_path(str(tmp_path / "teams_2030.json")) is None


def test_context_without_teams_key_does_not_count(tmp_path):
    (tmp_path / "tournament_context_2030.json").write_text(json.dumps({"seeds": {}}))
    assert DataLoader.resolve_teams_json_path(str(tmp_path / "teams_2030.json")) is None


def test_real_file_wins_over_redirect(tmp_path):
    real = tmp_path / "teams_2030.json"
    real.write_text("{}")
    assert DataLoader.resolve_teams_json_path(str(real)) == str(real)


def test_pre_run_validator_uses_the_same_resolution():
    """The validator must call the resolver, not os.path.exists, for teams_json."""
    src = (Path(__file__).resolve().parents[1] / "src" / "pipeline" / "pipeline_runner.py").read_text()
    assert "resolve_teams_json_path" in src


# --- 2. The seed fallback is opt-in and recorded ----------------------------


class _Boom:
    def __init__(self, *_a, **_k):
        raise RuntimeError("simulated pipeline failure")


def _harness(tmp_path, **kw):
    hist = tmp_path / "hist"
    hist.mkdir()
    (hist / "historical_games_2030.json").write_text("[]")
    return BacktestHarness(historical_dir=str(hist), years=[2030], n_bootstrap=10, **kw)


_GAMES = [{"team1_id": "a", "team2_id": "b", "team1_seed": 1, "team2_seed": 16, "team1_won": True}]


def test_pipeline_failure_raises_by_default(tmp_path):
    h = _harness(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        h._run_year(2030, "unused", lambda *_: _GAMES, _Boom, dict, RuntimeError)
    assert "seed" in str(exc.value).lower()
    assert "--allow-seed-fallback" in str(exc.value)


def test_pipeline_failure_with_opt_in_is_labelled(tmp_path):
    h = _harness(tmp_path, allow_seed_fallback=True)
    preds, games, source = h._run_year(2030, "unused", lambda *_: _GAMES, _Boom, dict, RuntimeError)
    assert source == "seed_fallback"
    assert preds and games


def test_summary_marks_fallback_years_and_excludes_them_from_the_mean(tmp_path):
    from types import SimpleNamespace

    from src.evaluation.backtest_harness import BacktestResult

    # Only the two members BacktestResult touches; a real AggregateEvaluationReport
    # cannot summarise zero year reports.
    agg = SimpleNamespace(summary="aggregate summary", to_dict=lambda: {})
    r = BacktestResult(
        aggregate_report=agg,
        per_year_brier={2024: 0.18, 2025: 0.14},
        per_year_source={2024: "pipeline", 2025: "seed_fallback"},
    )
    text = r.summary()
    assert "SEED_FALLBACK" in text
    assert "Model years n=1: Mean=0.1800" in text
    assert "1 year(s) fell back" in text
    assert r.to_dict()["per_year_source"] == {"2024": "pipeline", "2025": "seed_fallback"}


# --- 3. Walk-forward trains only on earlier seasons -------------------------


def test_walk_forward_dev_years_exclude_the_future(tmp_path, monkeypatch):
    captured = {}

    class _Capture:
        def __init__(self, config):
            captured["dev_years"] = config.dev_years
            raise RuntimeError("stop after capture")

    class _Cfg:
        def __init__(self, **kw):
            self.dev_years = kw["dev_years"]

    h = _harness(tmp_path, walk_forward=True, allow_seed_fallback=True)
    h.years = [2015, 2016, 2017]
    h._run_year(2016, "unused", lambda *_: _GAMES, _Capture, _Cfg, RuntimeError)
    assert captured["dev_years"], "pipeline was never constructed"
    assert max(captured["dev_years"]) < 2016
    assert 2020 not in captured["dev_years"]


def test_loyo_default_dev_years_include_the_future(tmp_path):
    captured = {}

    class _Capture:
        def __init__(self, config):
            captured["dev_years"] = config.dev_years
            raise RuntimeError("stop after capture")

    class _Cfg:
        def __init__(self, **kw):
            self.dev_years = kw["dev_years"]

    h = _harness(tmp_path, allow_seed_fallback=True)
    h.years = [2015, 2016, 2017]
    h._run_year(2016, "unused", lambda *_: _GAMES, _Capture, _Cfg, RuntimeError)
    assert 2017 in captured["dev_years"], "default LOYO trains on later seasons -- that is the documented protocol"
    assert 2016 not in captured["dev_years"]
