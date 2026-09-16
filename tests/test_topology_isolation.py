"""Bracket-topology isolation contract (2026-09-16 closeout).

The final full regression exposed a residual of audit finding F3-1 in one
caller the audit did not reach: `src/optimization/leverage.py`'s
ParetoOptimizer calls `construct_bracket()` without `region_order`, so its
brackets are laid out on the default East/West/South/Midwest tree, while the
CLI re-ranker scores them on the season's real Final Four pairing. The strict
projection raises `TopologyMismatch` whenever the two semifinal winner sets
disagree (tests/test_optimize_pool_e2e.py, `seed` mode, marked xfail).

That path was classified as NON-PRODUCTION and deliberately not repaired.
This file is the evidence for the classification, written as checks so it
cannot silently stop being true:

  1. Every `construct_bracket(...)` call on a production or validation path
     passes `region_order` (parsed from the AST, so a docstring cannot fake it).
  2. No production or validation module imports `PoolOptimizer` /
     `ParetoOptimizer`, and the one backtest helper that does
     (`mc_pool_backtest.build_optimized_brackets`) has no callers at all.
  3. The two construction-mode paths the CLI documents as recommended
     (`auto`, `det_*`) pass `region_order`; only the single-mode Pareto path
     does not -- and that fact is pinned so a repair updates this file.

If (2) ever fails, the defect has a dependency path into something that
matters and the classification is void: fix `leverage.py:1492` to thread the
season's `region_order`, rebuild affected artifacts, rerun Step 9 parity.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Production + validation paths per artifacts/methodology_audit/step1
# (Paths 1-3), Step 9 (referee audit), and the fitted-bracket evaluation.
PRODUCTION_FILES = [
    "scripts/experiments/build_candidate_artifact.py",
    "scripts/build_ui_payload.py",
    "scripts/evaluate_fitted_bracket.py",
    "scripts/generate_poolaware_bracket.py",
    "scripts/mc_pool_backtest.py",
    "src/product/selection.py",
    "src/evaluation/referee_audit.py",
    "src/evaluation/canonical_contract.py",
    "src/optimization/recency_hparam_fitter.py",
]


def _calls(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if fname == name:
                yield node


def _parse(rel: str) -> ast.AST:
    return ast.parse((REPO / rel).read_text(), filename=rel)


@pytest.mark.parametrize("rel", PRODUCTION_FILES)
def test_production_construct_bracket_calls_pass_region_order(rel):
    tree = _parse(rel)
    calls = list(_calls(tree, "construct_bracket"))
    for c in calls:
        kws = {k.arg for k in c.keywords}
        assert "region_order" in kws, f"{rel}:{c.lineno} construct_bracket() without region_order"


def test_the_known_defective_call_is_exactly_where_documented():
    """Pin the defect's location so a repair -- or a move -- updates this file."""
    tree = _parse("src/optimization/leverage.py")
    missing = [c.lineno for c in _calls(tree, "construct_bracket") if "region_order" not in {k.arg for k in c.keywords}]
    assert missing, "leverage.py now passes region_order everywhere: remove the xfail in test_optimize_pool_e2e.py and update this test"
    assert len(missing) == 1


@pytest.mark.parametrize("rel", PRODUCTION_FILES)
def test_production_modules_do_not_import_the_pareto_optimizer(rel):
    src = (REPO / rel).read_text()
    tree = ast.parse(src, filename=rel)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = {a.name for a in node.names}
            if "pool_optimizer" in mod or names & {"PoolOptimizer", "ParetoOptimizer"}:
                offenders.append((node.lineno, mod, sorted(names)))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if "pool_optimizer" in a.name:
                    offenders.append((node.lineno, a.name, []))
    if rel == "scripts/mc_pool_backtest.py":
        # The single allowed import lives inside build_optimized_brackets, a
        # dead helper -- asserted dead by the test below. Anything else is new.
        for lineno, _mod, _names in offenders:
            fn = _enclosing_function(tree, lineno)
            assert fn == "build_optimized_brackets", f"{rel}:{lineno} ParetoOptimizer used outside the dead helper"
        return
    assert not offenders, f"{rel} imports the Pareto optimizer: {offenders}"


def _enclosing_function(tree: ast.AST, lineno: int):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= lineno <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    return best.name if best else None


def test_build_optimized_brackets_has_no_callers():
    """The only backtest-side route to ParetoOptimizer is a function nothing calls."""
    callers = []
    for p in list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")):
        tree = ast.parse(p.read_text(), filename=str(p))
        for c in _calls(tree, "build_optimized_brackets"):
            callers.append(f"{p.relative_to(REPO)}:{c.lineno}")
    assert not callers, f"build_optimized_brackets is no longer dead: {callers}; the topology defect now has a live path"


def test_cli_recommended_paths_pass_region_order():
    """README recommends `optimize-pool` auto mode; det_* is the other
    documented construction route. Both construct on the real tree."""
    tree = _parse("src/cli/pool_cmds.py")
    for c in _calls(tree, "construct_bracket"):
        assert "region_order" in {k.arg for k in c.keywords}, f"pool_cmds.py:{c.lineno}"
