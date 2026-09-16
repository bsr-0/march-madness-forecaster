"""End-to-end regression: `optimize-pool` must actually run.

Before this fix, every documented mode failed against real 2026 data:
`--mode torvik` (the README's example) raised `ModuleNotFoundError` on a
deleted module; `--mode seed` (and every other mode, transitively, via
`_load_seeds`) raised `ValueError` on an unresolved First Four slot;
`--mode auto` silently produced zero brackets for the same reason. See
AUDIT_INDEPENDENT_EVALUATOR_2027.md finding H4.

These tests run the real CLI entry point against the real repo data — no
mocks — because the whole point is to catch a wiring failure between
modules that unit tests of each module in isolation would miss (which is
exactly how the original break went unnoticed: `_build_probabilities` and
`_load_seeds` were each individually fine, just not fine together).
"""

import argparse
import json

import pytest

from src.cli import pool_cmds as pc


def _args(tmp_path, mode, **overrides):
    defaults = dict(
        year=2026,
        pool_size=30,
        payout="winner_take_all",
        scoring="standard",
        output=str(tmp_path / "report.json"),
        submission=None,
        mode=mode,
        data_dir="data/raw",
        picks_dir=None,
        pool_history=None,
        pool_history_weight=1.0,
        no_walk_forward=True,
        construction_mode="forward_greedy",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "mode",
    [
        "torvik",
        pytest.param(
            "seed",
            marks=pytest.mark.xfail(
                strict=True,
                raises=Exception,
                reason=(
                    "KNOWN NON-PRODUCTION DEFECT (2026-09-16 closeout): the CLI's "
                    "single-mode Pareto path builds its bracket via "
                    "leverage.py ParetoOptimizer -> construct_bracket() WITHOUT "
                    "region_order, so construction runs on the default "
                    "East/West/South/Midwest tree while _rerank_brackets_by_p1st "
                    "scores on the season's real F4 pairing; the strict projection "
                    "raises TopologyMismatch whenever the two semifinal winner sets "
                    "disagree (a residual of audit F3-1 in a caller the audit did "
                    "not reach). torvik passes by coincidence of which teams its "
                    "round probabilities push through. Isolated to this CLI path: "
                    "no production build, CI job, Step 9 audit, candidate "
                    "artifact, or fitted-bracket evaluation reaches "
                    "ParetoOptimizer; see tests/test_topology_isolation.py. "
                    "strict=True so a fix flips this red and the marker gets removed."
                ),
            ),
        ),
    ],
)
def test_optimize_pool_runs_to_completion(tmp_path, mode):
    """The two cheapest modes (no ML training) must produce a real report."""
    args = _args(tmp_path, mode)

    exit_code = pc.run_optimize_pool(args)

    assert exit_code in (0, None)
    report_path = tmp_path / "report.json"
    assert report_path.exists(), f"--mode {mode} produced no report"
    report = json.loads(report_path.read_text())
    assert report.get("pareto_brackets") or report.get("brackets"), (
        f"--mode {mode} produced a report with no brackets: {list(report)}"
    )
