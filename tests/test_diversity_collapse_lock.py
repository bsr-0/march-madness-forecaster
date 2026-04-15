"""Drift guard for the diversity-collapse closure (COUNCIL_LESSONS §1 Diversity).

The council observation "diversity collapse is underdiagnosed" bundled two
claims with different resolutions:

* **Observation A** — the Pareto frontier produces only 5-14 unique brackets —
  persists in current production. It is the INTENDED design per
  ``src/optimization/leverage.py::ParetoOptimizer.generate_pareto_brackets``
  (the risk sweep honestly reflects the scoring curve; ``include_
  champion_augmentation=True`` is the opt-in remediation). Not a bug.

* **Observation B** — "stochastic-tilt mode made mean rank worse (620 vs 532)"
  — was the ``leveraged`` mode aggregate vs ``seed``, from
  ``scripts/mc_pool_backtest_equalized_results.txt`` (17 yrs; paired
  t=−3.808, p=0.0015). The producing modes (``leveraged``, ``opt_seed``,
  ``opt_blend``, ``opt_torvik``, ``hedge_tv``) were retired from
  ``mc_pool_backtest.ALL_MODES`` per MEMORY.md §2 D6/D7. The stochastic-tilt
  utility ``build_leverage_tilted_round_probs`` and the Pareto-leverage
  constructor ``build_optimized_brackets`` remain in the source file
  (retained for the one-off ``scripts/diagnose_catastrophic_years.py``
  diagnostic) but have zero live production callers.

These tests assert the closure stays true: if any deprecated mode creeps
back into ``ALL_MODES`` or any dead utility acquires a live ``src/`` caller,
the regression re-opens and CI fires.

Running the diagnostic script regenerates the evidence JSON:

    python scripts/diversity_collapse_diagnostic.py
    # → artifacts/diversity_collapse_2026-04-15.json
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MC_BACKTEST_SRC = _REPO_ROOT / "scripts" / "mc_pool_backtest.py"

DEAD_MODES = frozenset(
    {
        "leveraged",
        "opt_seed",
        "opt_blend",
        "opt_torvik",
        "hedge_tv",
    }
)

DEAD_UTILITY_FUNCTIONS = ("build_leverage_tilted_round_probs", "build_optimized_brackets")


def _extract_all_modes() -> tuple[str, ...]:
    tree = ast.parse(_MC_BACKTEST_SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "ALL_MODES" and isinstance(node.value, ast.Tuple):
                return tuple(e.value for e in node.value.elts if isinstance(e, ast.Constant))
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "ALL_MODES":
                    if isinstance(node.value, ast.Tuple):
                        return tuple(e.value for e in node.value.elts if isinstance(e, ast.Constant))
    raise AssertionError("Could not parse ALL_MODES tuple literal")


def test_all_modes_has_expected_members() -> None:
    """Pin the current production mode set so a drift (added or removed
    mode) surfaces in code review rather than silently altering backtest
    semantics."""
    actual = _extract_all_modes()
    expected = (
        "seed",
        "noseed",
        "blend",
        "torvik",
        "champ_first_tv",
        "f4_first_tv",
        "e8_first_tv",
        "det_champ_tv",
        "det_f4_tv",
        "det_e8_tv",
    )
    assert actual == expected, (
        f"ALL_MODES drift: got {actual}, expected {expected}. "
        "If this is intentional, update this test AND add a row to "
        "MEMORY.md §2 Dead-End Ledger (if retiring) or §1 Pool strategy "
        "(if adding a new mode)."
    )


def test_deprecated_modes_not_in_all_modes() -> None:
    """Dead-mode gate. leveraged / opt_* / hedge_tv reappearing means
    Observation B's regression path is live again and the diversity-
    collapse closure (§1 Diversity / §3 row 49) is invalid."""
    active = set(_extract_all_modes())
    intersection = sorted(DEAD_MODES & active)
    assert not intersection, (
        f"Deprecated modes have reappeared in ALL_MODES: {intersection}. "
        "These were retired per MEMORY.md §2 D6 (Pareto-leverage pool optimizer) "
        "and §2 D7 (hedge_tv) because they produced BestRank 62-88 vs seed's "
        "38 with P(1st)≈0 in upset years. Reopening the diversity-collapse "
        "closure requires new empirical evidence, not mode reinstatement."
    )


def _find_call_sites_under(func_name: str, root: Path) -> list[str]:
    """Return repo-relative paths of .py files under ``root`` that invoke
    ``func_name`` (matched as a call, not a bare reference)."""
    pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(")
    hits: list[str] = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(_REPO_ROOT).as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Filter out the definition line in the module that defines the
        # function (mc_pool_backtest.py) — only count actual call sites.
        lines = [
            ln for ln in text.splitlines() if pattern.search(ln) and not ln.lstrip().startswith(f"def {func_name}")
        ]
        if lines:
            hits.append(rel)
    return sorted(hits)


def test_dead_utility_functions_have_no_live_src_callers() -> None:
    """Dead-utility gate. If any ``src/`` module imports or calls these
    functions, they are no longer dead and Observation B's path has a
    live consumer — the closure is invalid until the new consumer is
    audited."""
    src_root = _REPO_ROOT / "src"
    offenders: dict[str, list[str]] = {}
    for fn in DEAD_UTILITY_FUNCTIONS:
        callers = _find_call_sites_under(fn, src_root)
        if callers:
            offenders[fn] = callers
    assert not offenders, (
        f"Dead utility functions gained src/ callers: {offenders}. These "
        f"functions ({', '.join(DEAD_UTILITY_FUNCTIONS)}) are retained in "
        "scripts/mc_pool_backtest.py only for the one-off "
        "scripts/diagnose_catastrophic_years.py diagnostic. Any new live "
        "consumer means the leveraged/opt_* regression path has been "
        "re-introduced. Audit the new caller before removing this test."
    )


def test_build_leverage_tilted_round_probs_has_zero_callers_anywhere() -> None:
    """Tightest gate: ``build_leverage_tilted_round_probs`` must have zero
    call sites *anywhere* in the repo (src/, scripts/, tests/). Even the
    allowlisted diagnostic script doesn't call this one; it was removed
    entirely. Any new caller reactivates the stochastic-tilt path."""
    callers = _find_call_sites_under("build_leverage_tilted_round_probs", _REPO_ROOT)
    # The definition file counts only if it has a call site beyond the
    # def line; _find_call_sites_under already excludes that.
    assert callers == [], (
        f"build_leverage_tilted_round_probs acquired callers: {callers}. "
        "This function was part of the deprecated stochastic-tilt path "
        "(MEMORY.md §2 D6) and has no production consumer. A new caller "
        "means Observation B is being re-litigated without new evidence."
    )


def test_build_optimized_brackets_callers_are_allowlisted() -> None:
    """``build_optimized_brackets`` may be called by the pre-pivot
    ``scripts/diagnose_catastrophic_years.py`` diagnostic (that was its
    original purpose). Any other caller is unauthorized."""
    callers = _find_call_sites_under("build_optimized_brackets", _REPO_ROOT)
    allowed = {"scripts/diagnose_catastrophic_years.py"}
    unauthorized = sorted(set(callers) - allowed)
    assert not unauthorized, (
        f"build_optimized_brackets acquired unauthorized callers: {unauthorized}. "
        f"Allowed caller set: {sorted(allowed)}. New callers reactivate the "
        "Pareto-leverage construction path (MEMORY.md §2 D6) that produced the "
        "diversity-collapse regression."
    )


def test_pareto_frontier_contract_documents_possible_collapse() -> None:
    """Observation A is a documented contract, not a bug. This test pins
    the ParetoOptimizer docstring language so that if someone later
    changes the behavior without removing the docstring caveat, CI will
    surface the drift."""
    leverage_src = (_REPO_ROOT / "src" / "optimization" / "leverage.py").read_text()
    assert "include_champion_augmentation" in leverage_src, (
        "ParetoOptimizer lost its include_champion_augmentation flag — the "
        "opt-in remediation for Observation A. Re-evaluate the diversity-"
        "collapse closure before removing."
    )
    assert "May return fewer" in leverage_src, (
        "ParetoOptimizer.generate_pareto_brackets docstring no longer warns "
        "that fewer than num_brackets may be returned. That caveat is what "
        "distinguishes Observation A from a silent bug — keep it."
    )
