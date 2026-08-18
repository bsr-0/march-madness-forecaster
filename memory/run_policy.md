# Run Policy — DO NOT launch `--tier budget` without explicit human approval

**Read this before running anything that executes the pool backtest or the experiment loop.**

## The rule

Strategy-addition phases are **no-run phases** by default. Adding a source, adjustment, construction mode, or adjustment chain to the catalog is pre-work — it expands the evaluable set but does **not** give Claude permission to launch a budgeted run. Runs require an explicit message from the human operator ("run the budget", "kick off Phase 3", "let's go on the run", etc.).

This rule exists because:
1. A budgeted run consumes real compute and writes artifacts to `artifacts/experiments/`.
2. Running prematurely (before the strategy set is complete) wastes the run — survivors don't tell us as much if competitive candidates weren't in the pool.
3. Every run commits a JSON artifact to the branch; those are expensive to reason about later if they proliferate.
4. The user (Claude Code operator) has explicitly stated: "We should not be running any of these configs, this is just to ensure we are evaluating all options."

## What counts as a run

Any of:
- `python -m scripts.run_experiment --tier budget`
- `python -m scripts.run_experiment --tier 1 | 2 | 3 | all`
- `python -m scripts.run_experiment --permutations`
- `python -m scripts.run_experiment --strategies ...`
- `python -m scripts.run_experiment --chaos-index` **does** execute, but only does pure data reads + regressions — it's classified as diagnostic, not a backtest run. OK without approval.
- `python -m scripts.run_experiment --oracle <year>` also diagnostic (reads saved artifacts). OK.

## What does NOT count as a run

Fine to do without explicit approval:
- `pytest` and `ruff` verification.
- `python3 -c "from src.prediction.strategy_pipeline import generate_all_permutations; print(len(...))"` — count permutations, inspect data.
- `git` operations on the feature branch.
- Reading existing artifacts under `artifacts/experiments/`.
- Writing new strategy code, new adjustments, new construction modes, new tests.

## Workflow when adding strategies

1. Ship the strategy (module + tests + registry + catalog update).
2. Run **unit tests only** (`pytest tests/...`) to verify the addition works.
3. Update CLAUDE.md's strategy comparison table to reflect the addition.
4. **Stop.** Report permutation count delta. Wait for human approval before any `--tier budget` invocation.

## When a run IS authorized

Human signals like:
- "run the budget" / "kick off phase 3" / "let's go on the run" / "run it"
- Explicit name of a tier: "run tier 1", "run T1", "run the budget pipeline"
- "ready to run" / "we're ready to run"

If the signal is ambiguous ("what's next?" / "keep going" / "sure"), assume it is **not** authorization to run. Ask explicitly: "Should I launch `--tier budget`, or continue adding strategies?"

## Why this is in `memory/` and not just inline

Settled policy lives here as a persistent source of truth, separate from CLAUDE.md. Inline CLAUDE.md notes can be trimmed during auto-compaction; this file survives and is discoverable.

If this policy is ever reversed or amended, append a dated row to the table below — do not edit existing rows.

| Date | Policy | Authorized by | Source |
|---|---|---|---|
| 2026-04-24 | No `--tier budget` runs without explicit human approval. Strategy-addition phases are no-run by default. | User directive during post-Phase-3 catalog expansion | This file |
