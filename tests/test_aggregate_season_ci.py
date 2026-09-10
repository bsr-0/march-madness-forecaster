"""The backtest's aggregate table must report P(1st) uncertainty over seasons.

Repeats within a season share one seed layout and one pick distribution, so a
CI over repeats understates the real uncertainty several-fold. The unit of
independence is the season, and the printed half-width must be a t-interval
over seasons — pinned here so it cannot quietly regress to a repeat-level CI.
"""

import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def backtest():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("mc_pool_backtest", ROOT / "scripts" / "mc_pool_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POOLAWARE_P1 = [0.01, 0.11, 0.14, 0.05, 0.15, 0.21, 0.07, 0.08, 0.06, 0.09, 0.11, 0.15, 0.14, 0.15]
SEASONS = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]


def _row(mode, year, p_first, n_trials=100):
    return {
        "mode": mode,
        "year": year,
        "best_rank": 10.0,
        "mean_rank": 10.0,
        "p_first": p_first,
        "p_top5": 0.2,
        "p_top25": 0.5,
        "mean_score": 600.0,
        "n_trials": n_trials,
    }


def _render(backtest, rows):
    buf = io.StringIO()
    with redirect_stdout(buf):
        backtest.print_aggregate_block(rows, "TEST")
    return buf.getvalue()


def _parse_row(text, mode):
    line = next(l for l in text.splitlines() if l.strip().startswith(mode + " "))
    cols = line.split()
    return {"p_first": float(cols[3]), "ci": float(cols[4]), "n": int(cols[5])}


def test_ci_is_t_interval_over_seasons(backtest):
    rows = [_row("seed", y, 0.04) for y in SEASONS]
    rows += [_row("poolaw", y, p) for y, p in zip(SEASONS, POOLAWARE_P1)]
    text = _render(backtest, rows)

    assert "±95%" in text
    got = _parse_row(text, "poolaw")

    n = len(POOLAWARE_P1)
    expected = sp_stats.t.ppf(0.975, n - 1) * np.std(POOLAWARE_P1, ddof=1) / np.sqrt(n)
    assert got["n"] == n
    assert got["p_first"] == pytest.approx(np.mean(POOLAWARE_P1), abs=5e-5)
    assert got["ci"] == pytest.approx(expected, abs=5e-5)

    # A repeat-level CI over n_trials = 1400 draws would be ~0.016; the season
    # interval is roughly twice that. Guard the direction, not just the value.
    repeat_level = 1.96 * np.sqrt(got["p_first"] * (1 - got["p_first"]) / (n * 100))
    assert got["ci"] > 1.5 * repeat_level


def test_single_season_has_no_interval(backtest):
    rows = [_row("seed", 2025, 0.04), _row("poolaw", 2025, 0.12)]
    text = _render(backtest, rows)
    got = _parse_row(text, "poolaw")
    assert got["n"] == 1
    assert got["ci"] == 0.0


def test_footer_names_the_unit_of_independence(backtest):
    rows = [_row("seed", y, 0.04) for y in SEASONS] + [_row("poolaw", y, 0.1) for y in SEASONS]
    text = _render(backtest, rows)
    assert re.search(r"over seasons", text)


# --- Contaminated seasons never enter an aggregate or the returned results ---


def _report(backtest, rows, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = backtest.report_backtest_results(rows, **kw)
    return out, buf.getvalue()


def test_contaminated_season_is_excluded_from_aggregate_and_return(backtest):
    assert 2026 in backtest.CONTAMINATED_EVAL_YEARS
    rows = [_row("seed", y, 0.04) for y in SEASONS + [2026]]
    rows += [_row("poolaw", y, p) for y, p in zip(SEASONS, POOLAWARE_P1)]
    rows.append(_row("poolaw", 2026, 0.99))  # would move the mean by ~6pp if it leaked

    out, text = _report(backtest, rows)

    assert {r["year"] for r in out} == set(SEASONS)
    got = _parse_row(text.split("AGGREGATE 2021+")[0], "poolaw")
    assert got["n"] == len(SEASONS)
    assert got["p_first"] == pytest.approx(np.mean(POOLAWARE_P1), abs=5e-5)
    assert "AGGREGATE EVALUATION YEARS (2011–2025, n=14)" in text
    assert "2026 ran as integration season(s) only" in text


def test_clean_run_prints_no_contamination_note(backtest):
    rows = [_row("seed", y, 0.04) for y in SEASONS] + [_row("poolaw", y, 0.1) for y in SEASONS]
    out, text = _report(backtest, rows)
    assert len(out) == len(rows)
    assert "integration season" not in text


def test_only_contaminated_seasons_returns_nothing(backtest):
    rows = [_row("seed", 2026, 0.04), _row("poolaw", 2026, 0.3)]
    out, text = _report(backtest, rows)
    assert out == []
    assert "AGGREGATE" not in text


# --- Pool size must be recorded in the run header ------------------------


def test_pool_size_header_names_the_fallback(backtest):
    """`--opponent pool` only knows the real size for seasons in
    pool_hist_results.json; the rest use this fallback. The header used to
    hide that, so a 1000-person default run and a real 30-person run
    produced identical-looking logs."""
    desc = backtest.describe_pool_size("pool", 29)
    assert "30" in desc
    assert "pool_hist_results.json" in desc


def test_default_fallback_is_a_real_pool_size(backtest):
    """The default was 999 -- a 1000-person field, ~33x any real pool here --
    so every run that omitted --n-opponents silently measured something
    incomparable to the published figure. It is now the canonical 30."""
    assert backtest.N_OPPONENTS == 29
    desc = backtest.describe_pool_size("pool", backtest.N_OPPONENTS)
    assert "else 30" in desc
    assert "pool_factor" not in desc, "the default must not trip the incomparability flag"


def test_header_flags_a_field_large_enough_to_change_construction(backtest):
    """Above 50 entries `pool_factor` engages and brackets are built
    differently, so the result stops being comparable to the canonical 30."""
    desc = backtest.describe_pool_size("pool", 999)
    assert "1000" in desc
    assert "pool_factor" in desc


def test_pool_factor_threshold_matches_bracket_construction(backtest):
    """The threshold is duplicated from _make_ev_scorer; if that moves and
    this does not, the header starts lying about comparability."""
    src = (ROOT / "src" / "optimization" / "bracket_construction.py").read_text()
    assert f"if pool_size > {backtest._POOL_FACTOR_THRESHOLD}:" in src


def test_non_pool_sources_report_a_plain_size(backtest):
    assert backtest.describe_pool_size("espn", 29) == "30"
    assert "pool_factor" not in backtest.describe_pool_size("espn", backtest.N_OPPONENTS)
