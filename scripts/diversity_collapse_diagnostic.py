#!/usr/bin/env python3
"""Diversity-collapse diagnostic closure.

Closes COUNCIL_LESSONS.md §1 Diversity/optimizer (the "open" bullet pointing
at §3 row 17). The council observation had two parts that were diagnosed
under one label; they have different resolutions and this script makes both
explicit.

Observation A — "Pareto frontier was producing 8–14 unique brackets"
    This is the ``ParetoOptimizer.generate_pareto_brackets`` output used by
    the user-facing forecast report. The behavior persists in current
    production (``pool_report*.json`` show 5–8 Pareto brackets and 10–13
    ``det_*`` unique brackets). It is the INTENDED design per the
    ``ParetoOptimizer.generate_pareto_brackets`` docstring: the risk sweep
    honestly reflects the (probability × differentiation) scoring curve,
    which genuinely has few crossovers when a single team dominates both
    model_prob and leverage (Michigan 2026, Gonzaga 2017). The opt-in
    ``include_champion_augmentation=True`` path exists for callers that
    want forced-champion diversity, and is OFF by default per the user
    preference for methodology-driven alternatives. This is documented
    design, not a bug.

Observation B — "stochastic-tilt mode made mean rank worse (620 vs 532)"
    Aggregate from ``mc_pool_backtest_equalized_results.txt:272,275``:
    ``seed`` MeanRnk = 532.4; ``leveraged`` MeanRnk = 620.2 (17 yrs, paired
    t = −3.808, p = 0.0015, wins seed 3/17). The ``leveraged``,
    ``opt_seed``, ``opt_blend``, ``opt_torvik``, and ``hedge_tv`` modes
    that produced the regression were REMOVED from
    ``mc_pool_backtest.ALL_MODES`` per MEMORY.md §2 D6/D7 (2026-04-12
    council). The stochastic-tilt utility
    ``build_leverage_tilted_round_probs`` and the Pareto-leverage path
    ``build_optimized_brackets`` remain defined in the module but have
    **zero live callers** — they are dead code retained only for the
    one-off ``scripts/diagnose_catastrophic_years.py`` diagnostic that
    investigated the ``opt_torvik`` catastrophic-year problem pre-pivot.

This script emits ``artifacts/diversity_collapse_2026-04-15.json``
containing both findings so the closure is reproducible.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MC_BACKTEST_SRC = _PROJECT_ROOT / "scripts" / "mc_pool_backtest.py"


def _extract_all_modes() -> tuple[str, ...]:
    """Parse ``ALL_MODES`` out of ``scripts/mc_pool_backtest.py`` by AST.

    Avoids importing the module (which pulls scipy + numpy + the whole
    backtest stack). The diagnostic only needs the tuple literal; parsing
    it directly keeps the audit runnable in CI without heavy deps.
    """
    tree = ast.parse(_MC_BACKTEST_SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "ALL_MODES" and isinstance(node.value, ast.Tuple):
                values = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                if all(isinstance(v, str) for v in values):
                    return tuple(values)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "ALL_MODES":
                    if isinstance(node.value, ast.Tuple):
                        values = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                        if all(isinstance(v, str) for v in values):
                            return tuple(values)
    raise RuntimeError(
        "Could not locate ALL_MODES tuple literal in scripts/mc_pool_backtest.py. "
        "The closure assumes this tuple is a direct literal assignment."
    )


# Modes retired per MEMORY.md §2 D6 (Pareto-leverage pool optimizer) +
# D7 (hedge_tv). If any of these reappear in ALL_MODES, Observation B
# is live again and this closure is invalid.
DEAD_MODES = frozenset(
    {
        "leveraged",
        "opt_seed",
        "opt_blend",
        "opt_torvik",
        "hedge_tv",
    }
)

# Dead utility functions — defined in mc_pool_backtest.py but no live
# callers in the active ALL_MODES path. The closure asserts this.
DEAD_UTILITY_FUNCTIONS = (
    "build_leverage_tilted_round_probs",
    "build_optimized_brackets",
)

# Allowlisted call sites for the dead utilities. These are one-off
# diagnostic scripts, not production code paths.
ALLOWED_CALLERS = {
    "build_optimized_brackets": {
        "scripts/diagnose_catastrophic_years.py",  # pre-pivot opt_torvik investigation
    },
    "build_leverage_tilted_round_probs": set(),  # no live callers anywhere
}

# Paths to search for live callers. Excludes scripts/ (diagnostic) and
# tests/ (fixture/data references), but the dead-utility assertion below
# also checks that no tests REFERENCE the functions to silently keep them
# alive without an active production consumer.
LIVE_PATH_ROOTS = ("src/",)


def _find_call_sites(func_name: str) -> list[str]:
    """Return repo-relative paths of files that appear to call ``func_name``."""
    pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(")
    hits: list[str] = []
    for p in _PROJECT_ROOT.rglob("*.py"):
        rel = p.relative_to(_PROJECT_ROOT).as_posix()
        if rel.startswith((".git/", ".claude/")):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            # Exclude the definition itself.
            if f"def {func_name}(" in text and rel == "scripts/mc_pool_backtest.py":
                # The definition file counts only if it has a call, not
                # just the definition. Check for an actual invocation
                # anywhere in the file (regex already matches call shape).
                # Filter out the def line.
                lines = [
                    ln
                    for ln in text.splitlines()
                    if pattern.search(ln) and not ln.lstrip().startswith(f"def {func_name}")
                ]
                if lines:
                    hits.append(rel)
            else:
                hits.append(rel)
    return sorted(hits)


def _collect_pareto_bracket_counts() -> dict[str, dict[str, int]]:
    """Scan pool_report_*.json files for the unique-bracket counts that
    embody Observation A (5–8 Pareto brackets; 10–13 det_* uniques)."""
    out: dict[str, dict[str, int]] = {}
    for p in sorted(_PROJECT_ROOT.glob("pool_report*.json")):
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rec: dict[str, int] = {}
        if isinstance(doc.get("n_pareto_brackets"), int):
            rec["n_pareto_brackets"] = doc["n_pareto_brackets"]
        if isinstance(doc.get("n_unique_brackets"), int):
            rec["n_unique_brackets"] = doc["n_unique_brackets"]
        if rec:
            out[p.name] = rec
    return out


def main() -> int:
    # --- Observation B: dead-mode and dead-utility audit ---
    active_modes = _extract_all_modes()
    dead_modes_still_live = sorted(DEAD_MODES & set(active_modes))

    dead_util_status: dict[str, dict] = {}
    for fn in DEAD_UTILITY_FUNCTIONS:
        call_sites = _find_call_sites(fn)
        # The mc_pool_backtest.py definition file may appear if it has
        # a secondary invocation; filter the definition-only case.
        call_sites = [s for s in call_sites if not (s == "scripts/mc_pool_backtest.py")]
        allowed = ALLOWED_CALLERS[fn]
        unauthorized = sorted(set(call_sites) - allowed)
        live_in_src = [s for s in call_sites if any(s.startswith(r) for r in LIVE_PATH_ROOTS)]
        dead_util_status[fn] = {
            "call_sites": call_sites,
            "live_src_callers": live_in_src,
            "allowed_diagnostic_callers": sorted(allowed),
            "unauthorized_callers": unauthorized,
        }

    # --- Observation A: Pareto bracket counts in production reports ---
    report_counts = _collect_pareto_bracket_counts()
    pareto_values = [v["n_pareto_brackets"] for v in report_counts.values() if "n_pareto_brackets" in v]
    unique_values = [v["n_unique_brackets"] for v in report_counts.values() if "n_unique_brackets" in v]
    pareto_stats = {
        "files_with_n_pareto_brackets": len(pareto_values),
        "min": min(pareto_values) if pareto_values else None,
        "max": max(pareto_values) if pareto_values else None,
        "distribution": dict(Counter(pareto_values)),
    }
    unique_stats = {
        "files_with_n_unique_brackets": len(unique_values),
        "min": min(unique_values) if unique_values else None,
        "max": max(unique_values) if unique_values else None,
        "distribution": dict(Counter(unique_values)),
    }

    payload = {
        "observation_a": {
            "description": (
                "ParetoOptimizer.generate_pareto_brackets returns 5-8 unique "
                "brackets in production forecast reports; det_* construction "
                "paths return 10-13. This matches the 8-14 observation from "
                "COUNCIL_LESSONS §1."
            ),
            "status": "intentional_design",
            "evidence": {
                "pool_report_counts": report_counts,
                "pareto_bracket_stats": pareto_stats,
                "det_unique_bracket_stats": unique_stats,
            },
            "opt_in_remediation": (
                "ParetoOptimizer.generate_pareto_brackets("
                "include_champion_augmentation=True) forces champion "
                "diversity via per-champion forced-winner brackets. Default "
                "False per user preference for methodology-driven "
                "alternatives over post-hoc variants."
            ),
            "source_file": "src/optimization/leverage.py:1143",
        },
        "observation_b": {
            "description": (
                "The 'stochastic-tilt made mean rank worse (620 vs 532)' "
                "regression was produced by the `leveraged` mode "
                "(aggregate MeanRnk 620.2) vs `seed` (532.4) in the 17-year "
                "equalized backtest (paired t=-3.808, p=0.0015, wins "
                "3/17)."
            ),
            "status": "dead_code_resolved",
            "active_modes": list(active_modes),
            "dead_modes_expected_absent": sorted(DEAD_MODES),
            "dead_modes_still_in_all_modes": dead_modes_still_live,
            "dead_utility_functions": dead_util_status,
            "source_files": {
                "evidence": "scripts/mc_pool_backtest_equalized_results.txt:272-293",
                "deprecation": "MEMORY.md §2 D6/D7",
                "current_all_modes": "scripts/mc_pool_backtest.py:83-94",
            },
        },
        "observation_c": {
            "description": (
                "COUNCIL_LESSONS §1 also says 'Mean rank vs P(top 5%) tell "
                "different stories' — a real design principle. The 2026-04-02 "
                "strategic pivot (MEMORY.md §1 Strategic pivot) moved the "
                "production target from prediction accuracy and mean rank to "
                "pool EV + P(1st) + P(top 5%); `champ_first_tv` is the "
                "recommended mode per MEMORY.md §1 Pool strategy. This part "
                "of the council observation is already locked forward."
            ),
            "status": "locked_by_pivot",
            "source_files": {
                "pivot": "MEMORY.md §1 Strategic pivot (2026-04-02)",
                "recommendation": "MEMORY.md §1 Pool strategy (champ_first_tv)",
            },
        },
    }

    # Write artifact.
    artifact_path = _PROJECT_ROOT / "artifacts" / "diversity_collapse_2026-04-15.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n")

    # Human summary.
    print("=" * 72)
    print("Diversity-collapse diagnostic closure (COUNCIL_LESSONS §1 Diversity)")
    print("=" * 72)
    print()
    print("Observation B — dead-code audit")
    print(f"  ALL_MODES (current, {len(active_modes)}): {list(active_modes)}")
    print(f"  Dead modes expected absent: {sorted(DEAD_MODES)}")
    if dead_modes_still_live:
        print(f"  FAIL — dead modes still in ALL_MODES: {dead_modes_still_live}")
    else:
        print("  PASS — none of the deprecated modes are in ALL_MODES.")
    for fn, rec in dead_util_status.items():
        print(f"  {fn}:")
        print(f"    live src/ callers: {rec['live_src_callers']}")
        print(f"    unauthorized callers: {rec['unauthorized_callers']}")
    print()
    print("Observation A — Pareto bracket counts from pool_report*.json")
    if pareto_values:
        print(
            f"  n_pareto_brackets: range [{pareto_stats['min']}..{pareto_stats['max']}] "
            f"across {pareto_stats['files_with_n_pareto_brackets']} reports"
        )
    if unique_values:
        print(
            f"  n_unique_brackets (det_*): range [{unique_stats['min']}..{unique_stats['max']}] "
            f"across {unique_stats['files_with_n_unique_brackets']} reports"
        )
    print(
        "  Status: intentional design per ParetoOptimizer docstring; "
        "include_champion_augmentation=True is the opt-in remediation."
    )
    print()
    print(f"Artifact written: {artifact_path.relative_to(_PROJECT_ROOT)}")

    # Exit nonzero only if a dead mode is still active OR an unauthorized
    # caller exists — both would invalidate the closure.
    fatal = bool(dead_modes_still_live)
    for rec in dead_util_status.values():
        if rec["unauthorized_callers"]:
            fatal = True
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
