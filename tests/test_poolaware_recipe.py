"""The poolaware candidate recipe has exactly one definition.

`meta_region_poolaware` is generated in two places: the backtest that
measures it, and the script that builds a bracket for a live year. Those
copies drifted — the backtest swept five probability bases, the live script
swept two — so the strategy that was measured at ~11% P(1st) and the
strategy that would ship were different strategies. Over the 15-season
reference run the backtest selected a base the live script could not build
in 11 of 15 seasons.

These tests pin the recipe's contents and ordering, and assert that neither
caller has quietly gone back to building its own base list.
"""

import ast
from pathlib import Path

import pytest

from src.optimization.poolaware_recipe import (
    POOLAWARE_EXHAUSTIVE_RISKS,
    POOLAWARE_RISK_LEVELS,
    build_poolaware_prob_bases,
    build_tv_mass80,
    poolaware_base_names,
)

ROOT = Path(__file__).resolve().parents[1]

TV = {"a": {"R64": 0.90, "CHAMP": 0.20}, "b": {"R64": 0.10, "CHAMP": 0.01}}
MA = {"a": {"R64": 0.70, "CHAMP": 0.10}, "b": {"R64": 0.30, "CHAMP": 0.05}}
MB = {"a": {"R64": 0.60, "CHAMP": 0.15}, "b": {"R64": 0.40, "CHAMP": 0.02}}
BL = {"a": {"R64": 0.80, "CHAMP": 0.12}, "b": {"R64": 0.20, "CHAMP": 0.03}}


def test_full_sweep_order_is_pinned():
    """Order is load-bearing: the selector keeps the FIRST candidate with the
    best P(1st) (`p1 > best_p1`), so reordering can change which bracket
    ships without changing any probability."""
    bases = build_poolaware_prob_bases(TV, massey_avg=MA, massey_best=MB, blend=BL)
    assert poolaware_base_names(bases) == ["tv", "mass_avg", "mass_best", "blend", "tv_mass80"]


def test_missing_sources_are_omitted_not_substituted():
    assert poolaware_base_names(build_poolaware_prob_bases(TV)) == ["tv"]
    assert poolaware_base_names(build_poolaware_prob_bases(TV, blend=BL)) == ["tv", "blend"]
    # tv_mass80 requires massey_avg, so it disappears with it.
    assert "tv_mass80" not in poolaware_base_names(build_poolaware_prob_bases(TV, massey_best=MB))


def test_tv_mass80_is_an_80_20_mix():
    mixed = build_tv_mass80(TV, MA)
    assert mixed["a"]["R64"] == pytest.approx(0.8 * 0.90 + 0.2 * 0.70)
    assert mixed["b"]["CHAMP"] == pytest.approx(0.8 * 0.01 + 0.2 * 0.05)


def test_tv_mass80_falls_back_to_torvik_for_cells_massey_lacks():
    sparse = {"a": {"R64": 0.70}}  # no 'b', no CHAMP for 'a'
    mixed = build_tv_mass80(TV, sparse)
    assert mixed["a"]["CHAMP"] == pytest.approx(TV["a"]["CHAMP"])
    assert mixed["b"]["R64"] == pytest.approx(TV["b"]["R64"])


def test_risk_grids_are_the_backtested_ones():
    assert POOLAWARE_RISK_LEVELS == (0.1, 0.3, 0.5, 0.7, 0.9)
    assert POOLAWARE_EXHAUSTIVE_RISKS == (0.3, 0.5, 0.7)


# --- Neither caller may rebuild the recipe locally ------------------------

CALLERS = [
    ROOT / "scripts" / "mc_pool_backtest.py",
    ROOT / "scripts" / "generate_poolaware_bracket.py",
]


@pytest.mark.parametrize("path", CALLERS, ids=lambda p: p.name)
def test_caller_imports_the_shared_recipe(path):
    tree = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "src.optimization.poolaware_recipe"
        for alias in node.names
    }
    assert "build_poolaware_prob_bases" in imported, (
        f"{path.name} must build its candidate bases via the shared recipe, "
        "not a local list — that divergence is what made the shipped strategy "
        "differ from the measured one"
    )


@pytest.mark.parametrize("path", CALLERS, ids=lambda p: p.name)
def test_caller_does_not_hardcode_a_base_list(path):
    """A local `[("tv", ...)]` seed list is the exact shape of the drift."""
    src = path.read_text()
    assert '[("tv"' not in src.replace(" ", ""), (
        f"{path.name} appears to build its own probability-base list again"
    )
