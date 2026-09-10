"""Brackets must be built and chosen for the field they are scored in.

Under `--opponent pool` the CLI's `--n-opponents` is only a FALLBACK: a season
present in pool_hist_results.json overrides it with that pool's real group
size. Final scoring honoured that (`year_n_opponents`), but every
`construct_bracket` call and every `draw_selection_trials` call in
`_run_one_year` passed the raw CLI parameter instead. So a bracket was
constructed and selected for one field size, then ranked in another — and at
the 999 default that meant building for a 1000-person pool and scoring in a
19-person one.

The bug was invisible at every pool size anyone actually uses, because
`_make_ev_scorer`'s `pool_factor` is exactly 1.0 at or below 50 entries. That
is precisely why it needs a static guard rather than a behavioural test: a
value-based test at a realistic pool size cannot see the difference.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "scripts" / "mc_pool_backtest.py"
SOURCE = SOURCE_PATH.read_text()
TREE = ast.parse(SOURCE)


def _function(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {SOURCE_PATH.name}")


def _calls_to(func_node, callee_name):
    out = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            fn = node.func
            label = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if label == callee_name:
                out.append(node)
    return out


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


RUN_ONE_YEAR = _function("_run_one_year")


def test_there_are_construct_and_select_calls_to_check():
    """Guard the guard: if these disappear the assertions below go vacuous."""
    assert len(_calls_to(RUN_ONE_YEAR, "construct_bracket")) >= 8
    assert len(_calls_to(RUN_ONE_YEAR, "draw_selection_trials")) >= 3


@pytest.mark.parametrize("callee,kwarg,required", [
    ("construct_bracket", "pool_size", "pool_size"),
    ("draw_selection_trials", "n_opponents", "year_n_opponents"),
])
def test_field_size_comes_from_the_season_not_the_cli(callee, kwarg, required):
    offenders = []
    for call in _calls_to(RUN_ONE_YEAR, callee):
        value = _kwarg(call, kwarg)
        if value is None:
            continue
        if not (isinstance(value, ast.Name) and value.id == required):
            shown = getattr(value, "id", ast.dump(value)[:40])
            offenders.append(f"line {call.lineno}: {kwarg}={shown}")
    assert not offenders, (
        f"{callee}(...) must take {kwarg}={required}, the season's real field size, "
        f"not the raw --n-opponents fallback. Offenders: {offenders}"
    )


def test_pool_size_is_derived_from_the_resolved_opponent_count():
    """pool_size must stay `year_n_opponents + 1` — total entries, not opponents."""
    for node in ast.walk(RUN_ONE_YEAR):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "pool_size"
        ):
            assert isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
            assert getattr(node.value.left, "id", None) == "year_n_opponents"
            assert getattr(node.value.right, "value", None) == 1
            return
    raise AssertionError("no `pool_size = year_n_opponents + 1` assignment in _run_one_year")
