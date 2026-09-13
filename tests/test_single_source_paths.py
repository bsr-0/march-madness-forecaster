"""One pool-history file, and a CI gate that actually gates.

Two defects found while sweeping for open items after the 2026-09 audit pass,
neither of which was in the audit:

1. `pool_hist_results.json` existed TWICE -- at the repo root and under
   `data/pool_history/` -- byte-identical, with the callers split between them
   and no rule about which was authoritative. The split ran through production:
   `scripts/mc_pool_backtest.py` read the root copy while `src/cli/pool_cmds.py`
   read the other, so the backtest and the CLI were one uncoordinated edit away
   from modelling different pools. Identical contents made it invisible; it
   would have surfaced the first time the 2027 pool was scraped into one path
   and not the other, as numbers that disagreed for no apparent reason.

2. `ci.yml` did not trigger on pushes to `main`, while branch protection on
   `main` required a status check named "CI" that only `ci.yml` produces. The
   rule was therefore unsatisfiable for a direct push and had to be bypassed
   every time, and CI Pipeline had never run on `main` in the repo's history.

Both are the same failure class as audit finding H6 and the deploy workflow's
renamed-stylesheet path filter: a guard that silently does nothing. Neither
raises, neither logs, and both look fine in a diff.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# One pool-history file
# ---------------------------------------------------------------------------


def test_pool_history_exists_exactly_once():
    """The root duplicate must not come back."""
    from src.simulation.pool_history_opponent_model import POOL_HISTORY_PATH

    assert POOL_HISTORY_PATH.exists(), f"the canonical pool history is missing: {POOL_HISTORY_PATH}"

    # `.claude/worktrees/` holds throwaway git worktrees from agent runs, each
    # a full checkout; they are not part of the repo's own layout.
    ignored = {".git", ".claude", "node_modules", ".venv"}
    found = sorted(
        p.relative_to(ROOT)
        for p in ROOT.rglob("pool_hist_results.json")
        if not ignored & set(p.parts)
    )
    assert len(found) == 1, (
        f"pool_hist_results.json exists in {len(found)} places: {found}. One file, one location -- "
        "duplicate copies stay identical right up until the season they do not, and the "
        "callers were previously split between them with production on both sides."
    )
    assert found[0] == POOL_HISTORY_PATH.relative_to(ROOT)


def test_no_module_rebuilds_the_pool_history_path_itself():
    """Rebuilding the path from PROJECT_ROOT is how the second copy happened.

    Only path CONSTRUCTION counts. Prose that names the file -- docstrings,
    argparse help, comments explaining where the data came from -- is exactly
    what you want in a repo like this, and a rule that forbade it would be
    obeyed by deleting the explanations.
    """
    import re

    # A path being built: `/ "pool_hist_results.json"`, `Path("…/pool_hist_results.json")`,
    # or the full relative path written out as a literal.
    construction = re.compile(
        r"""(/\s*["']pool_hist_results\.json["'])"""
        r"""|(["'][^"']*data/pool_history/pool_hist_results\.json["'])"""
        r"""|(Path\(\s*["'][^"']*pool_hist_results\.json)"""
    )

    offenders = []
    for pattern in ("scripts/**/*.py", "src/**/*.py"):
        for path in ROOT.glob(pattern):
            rel = str(path.relative_to(ROOT))
            if rel == "src/simulation/pool_history_opponent_model.py":
                continue  # the owner; it defines the constant
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if line.lstrip().startswith("#"):
                    continue
                if construction.search(line):
                    offenders.append(f"{rel}:{lineno} {line.strip()[:90]}")

    assert not offenders, (
        "pool_hist_results.json's path is rebuilt instead of importing POOL_HISTORY_PATH from "
        f"src.simulation.pool_history_opponent_model. Offenders: {offenders}"
    )


def test_every_caller_resolves_to_the_same_file():
    """Behavioural version of the scan above -- imports, not greps."""
    import importlib

    from src.simulation.pool_history_opponent_model import POOL_HISTORY_PATH

    callers = [
        ("scripts.mc_pool_backtest", "POOL_HIST_PATH"),
        ("scripts.real_pool_placement", "POOL_HIST_PATH"),
        ("src.prediction.contrarian_probabilities", "POOL_HIST_PATH"),
        ("scripts.analyze_pool_vs_espn", "POOL_HIST_PATH"),
        ("scripts.analyze_pool_bias", "POOL_DATA"),
        ("scripts.noise_floor_ceiling", "POOL_PATH"),
        ("scripts.analyze_pool_history", "POOL_HIST"),
    ]
    for module_name, attr in callers:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - dependency guard
            pytest.skip(f"{module_name} unavailable: {exc}")
        assert Path(getattr(module, attr)) == POOL_HISTORY_PATH, (
            f"{module_name}.{attr} points somewhere else; every caller must read one file"
        )


# ---------------------------------------------------------------------------
# A CI gate that can actually fire
# ---------------------------------------------------------------------------


def _ci_workflow() -> str:
    path = ROOT / ".github" / "workflows" / "ci.yml"
    if not path.exists():  # pragma: no cover - repo layout guard
        pytest.skip("ci.yml not present")
    return path.read_text()


def test_ci_runs_on_pushes_to_main():
    """Branch protection requires a check only this workflow can produce.

    Without `main` in the push triggers, a direct push to main produces no CI
    run at all, the required check never appears, and the push can only land by
    bypassing the rule. That was the state until 2026-09-13, and CI Pipeline had
    never once run on main.
    """
    source = _ci_workflow()
    on_block = source.split("concurrency:")[0]
    assert "'main'" in on_block or '"main"' in on_block or "[main" in on_block, (
        "ci.yml does not trigger on pushes to main. The branch-protection rule on main "
        "requires a status check named 'CI', which only this workflow emits, so without this "
        "trigger the rule is unsatisfiable and every push to main must bypass it."
    )


def test_the_required_check_name_still_exists():
    """The protected check is a job NAME, so renaming the job silently unguards main."""
    source = _ci_workflow()
    assert 'name: "CI"' in source or "name: CI\n" in source, (
        "no job named exactly 'CI' in ci.yml. Branch protection on main requires a check by "
        "that name; renaming the aggregating job makes the requirement unsatisfiable without "
        "any error appearing anywhere."
    )
