# O12 — Scoring-Function Structure in Optimizer Objective

**Date:** 2026-04-14
**Branch:** `claude/simplify-repo-structure-keQXE`
**Status:** closure evidence for `COUNCIL_LESSONS.md §2 O12`

## Gate (from §2)

> Pool's point schedule (e.g. 1/2/4/8/16/32) hard-coded into optimizer's
> objective, not just game outcomes.

## TL;DR

**Already satisfied.** The optimizer's objective multiplies per-round
win probabilities by per-round point values from `PoolEnvironment
.scoring_rules`. The schedule is parametric (three adapters — standard,
flat, upset_bonus) and required on every recommendation. No code change
needed; closure is verification + lock test.

## The existing objective path

### 1. Mandatory environmental field

`PoolEnvironment` (`src/optimization/pool_optimizer.py:30-52`) declares
`scoring_rules: Dict[str, int]` as one of **four mandatory parameters**.
`PoolEnvironment.validate()` raises on empty — there is no silent
fallback to a hard-wired ESPN schedule.

```python
@dataclass
class PoolEnvironment:
    pool_size: int
    scoring_rules: Dict[str, int]   # Round -> points mapping
    payout_structure: str
    public_pick_distribution: Dict[str, Dict[str, float]]

    def validate(self) -> None:
        if not self.scoring_rules:
            raise ValueError("scoring_rules must be a non-empty dict")
```

### 2. Propagation from optimizer to downstream consumers

`PoolOptimizer` forwards `env.scoring_rules` to three downstream paths:

- `leverage.py` — `LeverageCalculator(scoring_system=env.scoring_rules)`
  (`pool_optimizer.py:255,294,307`)
- `bracket_portfolio.py` — via `scoring_system=env.scoring_rules`
  (`pool_optimizer.py:430`)

Every branch of the optimization graph reads the same schedule; there
is no hard-wired shadow copy.

### 3. Per-round points enter the EV objective

`LeverageCalculator._path_ev_var` (`leverage.py:1618+`) multiplies the
survival-conditioned win probability by `scoring_system.get(<round>,
0)` for each round in the path:

```python
pts = float(self.calculator.scoring_system.get("R64", 0))
ev, var = self._path_ev_var(winner, "R64", survival)
expected_points += ev
```

Each of R64, R32, S16, E8, F4, CHAMP appears in this pattern — the
`scoring_system.get(` string occurs **≥ 6 times** inside the EV path.
The test below locks this.

### 4. Parametric scoring adapters

`get_scoring_adapter()` at `leverage.py:618-644` registers three
distinct scoring systems, each with its own round-weight vector and
leverage priority:

| System | Round weights | Leverage priority |
|---|---|---|
| `standard` | `{R64:10, R32:20, S16:40, E8:80, F4:160, CHAMP:320}` | `late_rounds` |
| `flat` | `{R64:1, R32:2, S16:3, E8:4, F4:5, CHAMP:6}` | `balanced` |
| `upset_bonus` | `{R64:10, …, CHAMP:320}` + seed-diff multiplier | `balanced` |

`ScoringSystemAdapter.adjust_leverage_ratio(round, base_ratio, seed_diff)`
(`leverage.py:592-615`) applies the round weight as a `sqrt(weight)`
dampened multiplier for `standard`; the flat adapter returns
`base_ratio * (weight / 3.5)`. The three systems produce distinguishable
leverage outputs (verified by test).

### 5. Recommendations carry the schedule forward

`AssumptionsManifest` (`pool_optimizer.py:55-78`) — attached to every
bracket recommendation — includes `scoring_rules` in its `to_dict()`
output. A bracket and the schedule it was optimized for travel
together; no recommendation can be consumed without the schedule being
visible to the consumer.

## What could break (regression matrix)

| Breakage | Symptom | Lock test |
|---|---|---|
| `scoring_rules` made optional / defaults to ESPN | Silent reversion to ESPN weights on non-ESPN pools | `test_pool_environment_rejects_empty_scoring_rules` |
| Type annotation relaxed to `Any` | Schedule becomes opaque to type-checker | `test_pool_environment_scoring_rules_type_is_round_to_points_mapping` |
| `get_scoring_adapter` collapses to one system | Parametric registry gone | `test_scoring_adapter_registry_has_three_parametric_systems` |
| Round weights become cosmetic (not used in leverage math) | Objective reverts to outcome-only | `test_scoring_adapter_adjusts_ratio_by_round_weight` |
| `scoring_system.get(round, 0)` removed from EV path | Per-round points stop entering EV | `test_leverage_calculator_uses_scoring_system_in_ev` |
| `env.scoring_rules` not forwarded downstream | Leverage/portfolio use defaults regardless of env | `test_pool_optimizer_propagates_scoring_rules_to_downstream` |
| `AssumptionsManifest` stops recording schedule | Recommendations become unprovenanced | `test_assumptions_manifest_records_scoring_rules` |

## Residual notes

- The three registered adapters cover ESPN + flat + upset_bonus pools.
  Custom schedules (Yahoo, CBS, or a user-provided mapping) are
  accepted by `PoolEnvironment` but don't get a pre-configured leverage
  adapter — the caller must register one via `ScoringSystemAdapter`
  directly. This is documented here; not a gate failure.
- The flat adapter's `round_weights` use a 1–6 linear schedule. A pool
  with genuinely flat scoring (all rounds worth 1 point) would be an
  even more degenerate case and is not currently tested. If a real
  pool requires this, register a fourth adapter.

## Closure record

`COUNCIL_LESSONS.md §2 O12` → `[closed 2026-04-14]`. Crumb:

> Scoring-schedule is a required field on `PoolEnvironment`, forwarded
> through `PoolOptimizer` to `LeverageCalculator.scoring_system`, and
> multiplied into per-round EV at `leverage.py:1618+`. Three parametric
> adapters (standard / flat / upset_bonus) registered via
> `get_scoring_adapter`. Audit committed at
> `artifacts/o12_scoring_structure_audit_2026-04-14.md`; drift guard =
> `tests/test_pool_scoring_structure.py` (8 tests, ~0.1 s).
