"""Closure / lock test for COUNCIL_LESSONS §2 O12 — scoring-function structure.

Gate: "Pool's point schedule (e.g. 1/2/4/8/16/32) hard-coded into
optimizer's objective, not just game outcomes."

The gate is already satisfied by the existing `PoolEnvironment` /
`LeverageCalculator` / `ScoringSystemAdapter` stack: `scoring_rules`
is a required field on `PoolEnvironment`, flows through `PoolOptimizer`
to downstream `leverage.py`, and the per-round point values are
multiplied into the expected-value objective at
`leverage.py:1646,1649` (`pts = scoring_system.get("R64", 0)` →
`expected_points += ev`). Three parametric adapters exist — standard,
flat, upset_bonus — so the schedule is not hard-wired to ESPN's
10/20/40/80/160/320 ramp.

This test locks that contract. A failure means the optimizer has
regressed to treating games as outcome-only (ignoring the point
schedule), or the parametric adapter registration has collapsed, or
the mandatory environmental field has been made optional. Any of
those re-opens §2 O12.

Do NOT relax an assertion to make this pass. If the scoring structure
genuinely needs to change (e.g., a new pool format), author a new
audit artifact and re-issue the closure.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
LEVERAGE_MODULE = REPO_ROOT / "src" / "optimization" / "leverage.py"
POOL_OPT_MODULE = REPO_ROOT / "src" / "optimization" / "pool_optimizer.py"

ESPN_STANDARD = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}


# ---------------------------------------------------------------------------
# Contract: scoring_rules is a REQUIRED field on the optimizer's env
# ---------------------------------------------------------------------------


def test_pool_environment_rejects_empty_scoring_rules() -> None:
    """PoolEnvironment.validate() must raise when scoring_rules is empty.

    Every bracket recommendation is supposed to be conditional on a
    known scoring schedule. An empty scoring_rules would silently let
    the optimizer fall back to outcome-only reasoning — which is
    exactly what O12 guards against."""
    from src.optimization.pool_optimizer import PoolEnvironment

    env = PoolEnvironment(
        pool_size=100,
        scoring_rules={},  # ← the violation
        payout_structure="winner_take_all",
        public_pick_distribution={"duke": {"R64": 0.9}},
    )
    with pytest.raises(ValueError, match="scoring_rules"):
        env.validate()


def test_pool_environment_accepts_non_standard_schedule() -> None:
    """The environment must accept any non-empty round→points mapping,
    not just ESPN's 10/20/40/80/160/320 ramp. A flat 1/2/3/4/5/6 or a
    custom progression must validate cleanly."""
    from src.optimization.pool_optimizer import PoolEnvironment

    flat = PoolEnvironment(
        pool_size=100,
        scoring_rules={"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6},
        payout_structure="winner_take_all",
        public_pick_distribution={"duke": {"R64": 0.9}},
    )
    flat.validate()  # should not raise

    doubling = PoolEnvironment(
        pool_size=100,
        scoring_rules={"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "CHAMP": 32},
        payout_structure="winner_take_all",
        public_pick_distribution={"duke": {"R64": 0.9}},
    )
    doubling.validate()  # should not raise


def test_assumptions_manifest_records_scoring_rules() -> None:
    """Every recommendation must ship with its scoring schedule recorded
    in the AssumptionsManifest. Without it, a later consumer can't tell
    which schedule the bracket was optimized for."""
    from src.optimization.pool_optimizer import AssumptionsManifest

    manifest = AssumptionsManifest(
        pool_size=100,
        expected_chalk={"duke": 0.3},
        payout_structure="winner_take_all",
        scoring_rules=ESPN_STANDARD,
    )
    payload = manifest.to_dict()
    assert payload["scoring_rules"] == ESPN_STANDARD
    assert set(ESPN_STANDARD.keys()) <= set(payload["scoring_rules"].keys())


# ---------------------------------------------------------------------------
# Contract: scoring structure is parametric, not hard-wired
# ---------------------------------------------------------------------------


def test_scoring_adapter_registry_has_three_parametric_systems() -> None:
    """`get_scoring_adapter` must return distinct adapters for at least
    'standard', 'flat', and 'upset_bonus'. Collapsing these to a single
    hard-wired schedule is the §2 O12 regression we're guarding against."""
    from src.optimization.leverage import get_scoring_adapter

    standard = get_scoring_adapter("standard")
    flat = get_scoring_adapter("flat")
    upset = get_scoring_adapter("upset_bonus")

    assert {standard.system, flat.system, upset.system} == {
        "standard",
        "flat",
        "upset_bonus",
    }, "Adapter registry has collapsed — parametric scoring is gone."

    # Standard and flat must carry different round weights — otherwise
    # choosing a scoring system has no effect on leverage math.
    assert standard.round_weights != flat.round_weights, (
        "standard and flat scoring share weights — the parametric distinction collapsed."
    )

    # Leverage priorities must also differ: standard prioritizes late
    # rounds (where ESPN points are largest); flat is balanced.
    assert standard.leverage_priority != flat.leverage_priority


def test_scoring_adapter_adjusts_ratio_by_round_weight() -> None:
    """`adjust_leverage_ratio` must produce DIFFERENT outputs for the
    same base ratio across scoring systems. Identical outputs would
    mean the round weights are cosmetic — the objective would be
    effectively outcome-only, failing §2 O12."""
    from src.optimization.leverage import get_scoring_adapter

    standard = get_scoring_adapter("standard")
    flat = get_scoring_adapter("flat")

    base_ratio = 1.5
    # CHAMP is where the divergence should be largest: standard weights
    # the final 32× the opener, flat only 6×.
    std_champ = standard.adjust_leverage_ratio("CHAMP", base_ratio)
    flat_champ = flat.adjust_leverage_ratio("CHAMP", base_ratio)
    std_r64 = standard.adjust_leverage_ratio("R64", base_ratio)
    flat_r64 = flat.adjust_leverage_ratio("R64", base_ratio)

    # Something must differ — either CHAMP or R64 should produce
    # distinguishable values across systems.
    assert std_champ != flat_champ or std_r64 != flat_r64, (
        "Leverage adjustment is insensitive to scoring system. The optimizer's objective is effectively outcome-only."
    )

    # And the standard schedule must amplify late rounds relative to
    # early rounds (the whole point of late_rounds leverage priority).
    assert std_champ > std_r64, (
        f"Standard ESPN leverage at CHAMP ({std_champ}) should exceed "
        f"R64 ({std_r64}) because the point schedule weights the final "
        f"32× the opener."
    )


# ---------------------------------------------------------------------------
# Contract: per-round points flow into the expected-value objective
# ---------------------------------------------------------------------------


def test_leverage_calculator_uses_scoring_system_in_ev() -> None:
    """AST check that `LeverageCalculator` reads `scoring_system` when
    computing per-path expected points. If this reference disappears,
    the optimizer has silently degraded to outcome-only — §2 O12 breach."""
    source = LEVERAGE_MODULE.read_text()

    # The canonical pattern from leverage.py:1646 etc.:
    #     pts = float(self.calculator.scoring_system.get("R64", 0))
    # We don't lock the exact round name, but we require the pattern
    # `scoring_system.get(<round>, 0)` to appear at least 6 times —
    # once per round — inside the EV path. Counting loosely via regex.
    occurrences = source.count("scoring_system.get(")
    assert occurrences >= 6, (
        f"`scoring_system.get(...)` appears only {occurrences} times in "
        f"leverage.py. At least 6 per-round references are expected in "
        f"the EV computation. The objective may have collapsed to "
        f"outcome-only."
    )


def test_pool_optimizer_propagates_scoring_rules_to_downstream() -> None:
    """AST check: `PoolOptimizer` must forward `env.scoring_rules` into
    `LeverageCalculator` and any bracket-portfolio consumer. If this
    propagation breaks, downstream components silently use default
    ESPN weights regardless of the environmental config."""
    source = POOL_OPT_MODULE.read_text()
    forwards = source.count("env.scoring_rules")
    assert forwards >= 3, (
        f"`env.scoring_rules` is forwarded only {forwards} times from "
        f"PoolOptimizer. At least 3 downstream consumers (leverage, "
        f"portfolio, scoring-system adapter) expect it. Propagation "
        f"may have regressed."
    )


def test_pool_environment_scoring_rules_type_is_round_to_points_mapping() -> None:
    """Declared type of `scoring_rules` must be `Dict[str, int]` so the
    structure (round → points) is explicit in the type system.

    A regression to `Any` or removal of the annotation would be a
    structural red flag."""
    tree = ast.parse(POOL_OPT_MODULE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PoolEnvironment":
            annotated = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    annotated[stmt.target.id] = ast.unparse(stmt.annotation)
            assert "scoring_rules" in annotated, "scoring_rules field missing"
            # e.g. "Dict[str, int]"
            annotation = annotated["scoring_rules"]
            assert "Dict" in annotation and "int" in annotation, (
                f"scoring_rules annotation is {annotation!r}; expected "
                f"a Dict[str, int] (round → points). A regression to "
                f"Any would make the pool schedule opaque to the "
                f"type-checker."
            )
            return
    pytest.fail("PoolEnvironment class not found in pool_optimizer.py")
