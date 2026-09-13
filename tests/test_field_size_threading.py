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


# Names that carry the season's resolved field size. `year_n_opponents` is the
# value `resolve_opponent_pick_distribution` returns; `entry_n_opponents` is it
# minus our extra entries (multi-entry takes seats rather than growing the
# pool, recommendation 13); `_pa_n_opp` is a local alias of that. The forbidden
# name is the raw CLI parameter, which is only a fallback.
RESOLVED_FIELD_NAMES = {"year_n_opponents", "entry_n_opponents", "_pa_n_opp"}
CLI_FALLBACK_NAME = "n_opponents"


@pytest.mark.parametrize("callee,kwarg", [
    ("construct_bracket", "pool_size"),
    ("draw_selection_trials", "n_opponents"),
])
def test_field_size_comes_from_the_season_not_the_cli(callee, kwarg):
    """The H6 invariant: build and select for the field you will be scored in.

    Checks the invariant rather than one spelling of it. The original pinned
    the literal name `year_n_opponents`, which was right until multi-entry
    introduced `entry_n_opponents` — a value DERIVED from it, guarded by
    `test_the_derived_field_size_still_comes_from_the_season` below. Pinning
    the name would have forced a choice between keeping the guard and making
    the change; pinning the property keeps both.
    """
    offenders = []
    for call in _calls_to(RUN_ONE_YEAR, callee):
        value = _kwarg(call, kwarg)
        if value is None:
            continue
        name = getattr(value, "id", None)
        if name == CLI_FALLBACK_NAME:
            offenders.append(f"line {call.lineno}: {kwarg}=n_opponents (the raw CLI fallback)")
        elif name not in RESOLVED_FIELD_NAMES and name != "pool_size":
            shown = name or ast.dump(value)[:40]
            offenders.append(f"line {call.lineno}: {kwarg}={shown} (not a known resolved-field name)")
    assert not offenders, (
        f"{callee}(...) must take {kwarg} from the season's resolved field size, not the raw "
        f"--n-opponents fallback. At the old 999 default that meant building for a 1000-person "
        f"pool and scoring in a 19-person one. Offenders: {offenders}"
    )


def test_the_derived_field_size_still_comes_from_the_season():
    """`entry_n_opponents` must be computed from `year_n_opponents`.

    This is what licenses accepting it above. If it is ever rebound to the CLI
    parameter, the H6 defect returns through the new name and the parametrized
    test would not notice.
    """
    for node in ast.walk(RUN_ONE_YEAR):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == "entry_n_opponents"
        ):
            names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            assert "year_n_opponents" in names, (
                f"entry_n_opponents is assigned from {sorted(names)} — it must derive from "
                "year_n_opponents, the season's resolved field size"
            )
            assert CLI_FALLBACK_NAME not in names, "entry_n_opponents must not read the CLI fallback"
            return
    raise AssertionError("no `entry_n_opponents = ...` assignment in _run_one_year")


def test_pool_size_counts_our_entries_plus_the_opponents():
    """pool_size is TOTAL entries, and multi-entry must not inflate it.

    Was `year_n_opponents + 1`. It is now `resolve_pool_size(entry_n_opponents,
    n_entries)`, which is the same number when n_entries == 1 and stays the
    season's real pool size as entries grow — because k entries occupy k seats
    rather than adding k people to the pool. Measuring against a pool that grew
    with every entry made a 4th bracket look more valuable than a 2nd, which is
    not how portfolios behave.
    """
    for node in ast.walk(RUN_ONE_YEAR):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == "pool_size"
        ):
            call = node.value
            assert isinstance(call, ast.Call) and getattr(call.func, "id", None) == "resolve_pool_size", (
                f"pool_size must come from resolve_pool_size(), got `{ast.unparse(call)}`"
            )
            args = [getattr(a, "id", None) for a in call.args]
            assert args == ["entry_n_opponents", "n_entries"], f"unexpected resolve_pool_size args: {args}"
            return
    raise AssertionError("no `pool_size = ...` assignment in _run_one_year")


def test_resolve_pool_size_is_still_opponents_plus_entries():
    """The behaviour the AST check above stands in for."""
    from src.optimization.payout import resolve_pool_size

    assert resolve_pool_size(29, 1) == 30
    assert resolve_pool_size(18, 1) == 19
    assert resolve_pool_size(26, 4) == 30, "4 entries against 26 opponents is still a 30-person pool"
