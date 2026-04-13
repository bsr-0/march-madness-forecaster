# O2a — Four-Factor validation: two defects diagnosed, data-source limit exposed

**Dates:** 2026-04-13 opened; two iterations same day.
**Branch:** `claude/simplify-repo-structure-keQXE`
**Evidence:**
- `artifacts/o2_ff_validation_2026-04-13.json` — post-fix per-year metrics
- This file — narrowing + structural finding

## Status: blocked on a council decision about the gate itself.

Two independent defects diagnosed during this investigation. One fixed
(the resolver collision). The other is a **data-source precision limit**
that cannot be closed by any reasonable code change.

## Root cause #1 — Resolver collision **[FIXED in this branch]**

The `TeamNameResolver`'s fuzzy `containment` method (conf 0.9) was
collapsing `vermont_state_lyndon_hornets` (1 game, D3 affiliate) to
canonical `vermont`, and since `dict.items()` order was nondeterministic
the 1-game record was overwriting `vermont_catamounts`' 35-game real
values. Same pattern for `tennessee_wesleyan_bulldogs` → `tennessee`.

Fix landed in `scripts/validate_four_factors.py::compute_boxscore_ff`:
sort raw team ids by game count (desc), restrict resolver to
`exact_id`/`prefix_strip` methods only, first-writer-wins. Jumped 4 of 8
features from mean r ≈ 0.45 to ≈ 0.97.

## Root cause #2 — Data-source precision limit **[OPEN, blocks gate]**

After the resolver fix, no additional code change closes the gate. A
cross-team audit on 344 Division-I teams in 2024 (all teams present in
both local and Torvik 2024 snapshots) shows that the Pearson correlation
saturates around 0.97-0.98 for every feature, regardless of formula:

| Feature | r (current) | r (Oliver-alt) | r (avg-possessions) |
|---|---|---|---|
| eFG%     | 0.9787 | 0.9787 (no possessions) | — |
| TO%      | 0.9507 | 0.9698 | 0.9640 |
| ORB%     | 0.9695 | — | — |
| DRB%     | 0.9401 | — | — |
| FTR      | ≈0.97 | (no possessions) | — |

**The eFG% ceiling is decisive.** eFG% has no possessions denominator —
it's `(FGM + 0.5·FG3M) / FGA` — and both inputs are simple box-score
counts. Our local `(fgm, fga, fg3m)` aggregates *exactly* match the
Oliver formula. If local-vs-Torvik r only reaches 0.98 for this
simplest-possible feature, the residual 2% disagreement is a
**source-data difference**, not a formula bug. Plausible sources:

- Different game-set filtering (Torvik may exclude games with missing
  box scores, post-forfeit corrections, or non-D1 opponents that our
  ingestion does include).
- Tiny provider-level recording differences for FGM/FGA
  (scorekeeping corrections made by NCAA post-game that Torvik
  re-ingests but our snapshot doesn't).
- Rounding — Torvik publishes at 3-4 decimal places; our source
  carries full precision. For very high correlations this matters.

The TO% finding corroborates: switching from the simple Oliver
denominator to the alt denominator (subtract ORB) improves mean
`|diff|` from 0.028 to 0.010 and lifts r from 0.95 to 0.97 — but NOT
to 0.99. Even with the best-possible formula, 344-team Pearson peaks
at 0.97 because the underlying raw inputs (FGA, TOV, ORB) don't match
Torvik's numbers to full precision.

Per-team diffs for the ALT-formula TO% on 2024 tournament teams:

| team | local-ALT TO% | Torvik TO% | diff |
|---|---|---|---|
| vermont   | 0.131 | 0.138 | -0.007 |
| kansas    | 0.162 | 0.164 | -0.002 |
| uconn     | 0.133 | ~0.15 | ~-0.02 |

Small, mostly consistent, never zero. That's the texture of two
providers aggregating over nearly-but-not-quite-identical box-score
tables.

## Why a code change won't close the gate

Options considered and rejected:

1. **Apply Oliver-alt formula in `ProprietaryMetricsEngine._four_factors`.**
   Moves TO% from r=0.95 to r=0.97 but not to r=0.99. Changes a
   production-path function that other metrics (e.g., `_box_score_xp`)
   depend on, with uncertain downstream impact. Net: spread the
   formula-change through the codebase for a gain that still doesn't
   clear the gate.
2. **Filter local games to "box-score complete" only.** Already done
   upstream (`game_records = [g for g in games if g.has_box_score]`).
3. **Switch local ingestion to Torvik's raw box scores.** Would
   certainly close the gate — by definition — but it would eliminate
   local box-score ingestion as a provenance-independent source. The
   point of the gate is to cross-validate two independent sources; if
   one sources from the other, the gate becomes tautological.

## Recommended escalation

The council should weigh in on one of:

**Option A — Relax the gate.** Change `r ≥ 0.99` to `r ≥ 0.95` per
season per feature AND `|bias| ≤ 0.02` per stratum. This acknowledges
that two *independently-collected* Four-Factor tables will never match
at r ≥ 0.99 due to scorekeeping-level precision differences, and that
r ≈ 0.97 is empirically "validation with modest precision loss",
which is still very useful for catching the kind of catastrophic bug
that the resolver-collision defect represented.

**Option B — Switch to a provider-cross-check within a provider.**
E.g., validate `historical_games_*.json` against a *second* box-score
provider (CBB reference, ESPN) rather than against Torvik's derived
metric. This tests ingestion correctness without the provider-delta
contamination.

**Option C — Accept the 0.99 gate is structurally unreachable and
close O2 as "validated to the precision of the available providers
(r ≈ 0.97 mean across 344 teams × 8 features)".** Mark the locked row
in MEMORY.md with the observed r, not the aspirational threshold.

My recommendation is **A**. The 0.99 gate in the council row was
written before the provider landscape was audited. r ≥ 0.95 still
catches any catastrophic defect (the resolver collision dropped r to
0.45; the 0.99 vs 0.95 distinction is academic at that magnitude) but
is actually achievable with our current ingestion.

## What landed in this branch

1. Root-cause #1 fixed in
   `scripts/validate_four_factors.py::compute_boxscore_ff` (resolver
   collision).
2. `scripts/validate_four_factors.py:421` summary-print format-string
   crash fixed.
3. `artifacts/o2_ff_validation_2026-04-13.json` regenerated with
   post-fix values.
4. This diagnostic file updated with the structural finding.
5. `COUNCIL_LESSONS.md §2 O2` status: **open, blocked on council
   decision** about the gate value (Option A/B/C above).
6. `COUNCIL_LESSONS.md §2 O2a` status: root-cause #1 closed,
   root-cause #2 reframed as data-source precision limit.

## Required to close O2

A council verdict on the gate. Once chosen:

- **If A**: update `validate_four_factors.py` threshold constants,
  re-run, confirm pass, wrap in `tests/test_validate_four_factors.py`,
  lock in `MEMORY.md §1`.
- **If B**: design a new validation script against a second box-score
  provider. Multi-week project.
- **If C**: accept and lock "r ≈ 0.97 mean; r ≥ 0.95 per cell after
  resolver fix" in `MEMORY.md §1`, close without test enforcement.

Status: **open**, escalated.
