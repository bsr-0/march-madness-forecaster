# Deterministic 2027 Experiment Memo

Use this as the 2027 deterministic decision rule.

## Scope

Treat the next phase as a 4-way bakeoff, not a 7-strategy program:

1. `meta_gbm v2` baseline
2. `meta_gbm_champ_top1`
3. `meta_gbm_champ_leverage` after the public-pick bug fix
4. `meta_gbm_champ_first`

Optional:

5. `S2` feature-expanded meta-selector only as one orthogonal hedge

This matches the repo's current diagnosis that the champion pick is the binding constraint in `COUNCIL_LESSONS.md`, with the champion-first machinery already present in `src/prediction/meta_selector.py` and `scripts/mc_pool_backtest.py`.

## Run Now

Run only these comparisons:

- `meta_gbm` vs `meta_gbm_champ_top1`
- `meta_gbm` vs `meta_gbm_champ_leverage`
- `meta_gbm` vs `meta_gbm_champ_first`
- optional: `meta_gbm` vs S2 model

Then compare the best deterministic winner against the incumbent stochastic shipping baseline `f4_first_tv` from `MEMORY.md`.

## Primary Success Criteria

A candidate is interesting only if all of these are true:

- Beats `meta_gbm v2` on aggregate `P(1st)`.
- Wins at least `8/14` backtest years on the repo's own gate from `CLAUDE.md`.
- Improves champion correctness over the current `6/14` in `artifacts/meta_gbm_v2_champion_diagnostic.json`.
- Does not rely on contrarian weighting or chalk suppression as an objective, per the hard rule in `CLAUDE.md`.

## Secondary Success Criteria

These do not justify promotion on their own, but they can break ties:

- Better performance on real-pool years `2023-2026`
- Better mean score in later rounds
- Better champion calibration without becoming "always 1-seed"

## Kill Criteria

Kill a candidate immediately if any of these happen:

- Aggregate `P(1st)` is below `meta_gbm v2`
- Year wins are `< 8/14`
- Champion accuracy stays flat or worse than `6/14`
- It improves MeanRank but not `P(1st)`; that already failed conceptually in the augmentation path documented in `COUNCIL_LESSONS.md`
- It depends on explicit anti-chalk logic; that conflicts with the repo's strongest lesson

## Promotion Rule

Promote a deterministic successor only if it:

- clears the `meta_gbm v2` gates above, and
- is competitive with or clearly better than `f4_first_tv` on the winner-take-all objective

If it only beats seed, that is not enough.

## What Would Actually Change the 2027 Plan

These outcomes would matter:

- `champ_top1` wins: ship the simplest champion-specialist version.
- `champ_leverage` wins after the bug fix: ship it, but only if the win survives the corrected public-pick lookup in `scripts/mc_pool_backtest.py`.
- `champ_first` wins: accept the two-stage design, but only because it beat the simpler champion-lock variants.
- S2 wins: then the lesson is "missing tournament context mattered more than decomposition."

These outcomes would not matter enough:

- better MeanRank only
- one or two flashy years
- tuning-only wins without structural improvement
- any no-chalk variant win that contradicts the repo's own contrarian findings

## Recommended Order

1. `champ_top1`
2. `champ_leverage`
3. `champ_first`
4. S2 only if you want one non-S1 test

## Prior

Most likely winner if anything works: `champ_top1` or `champ_leverage`, not the more complex two-phase selector. The reason is simple: the diagnosis is about the champion decision, and simpler fixes are less likely to reintroduce selection noise.

## Repo Doc Audit

This section audits the repo's goals and strategy docs against the current evidence.

### Executive Summary

The repo is not pointless, but several docs still overstate what the project can realistically accomplish.

The strongest version of the repo's purpose is:

- help choose one strong bracket for a single-entry, winner-take-all pool
- avoid previously falsified ideas
- make the champion decision more disciplined

The weakest version of the repo's purpose is:

- discover a durable general predictive edge over the market, seed baseline, or public
- engineer a repeatable "solution" to March Madness

The docs are strongest when they frame the project as a pool-decision system. They are weakest when they imply broad predictive or optimization control over an intrinsically noisy problem.

### What Is Well Aligned

These parts of the documentation are coherent and supported by the repo's own evidence:

- `CLAUDE.md` correctly centers `P(1st)` as the north-star metric and explicitly de-prioritizes MeanRank, P(top25%), and MeanScore for winner-take-all pools.
- `CLAUDE.md` also captures the most important structural lesson: ESPN scoring has no direct contrarian bonus, so anti-chalk logic should not be the training objective.
- `MEMORY.md` is the strongest document in the repo. It clearly locks decisions, preserves dead ends, and makes it hard to accidentally re-litigate known failures.
- `COUNCIL_LESSONS.md` and the champion diagnostic support the current deterministic pivot: the champion pick is a bottleneck, and broad prediction-accuracy chasing has low expected return.

### What Is Stale or Internally Contradictory

These docs should be treated as partially stale or in need of reframing:

- `README.md` still presents the system as a general "prediction system that generates calibrated win probabilities and optimizes bracket picks." That is technically true, but strategically too broad for what the evidence supports.
- `README.md` still says "The real edge is in pool strategy" and recommends contrarian pool optimization language that reads more confidently than the actual backtest signal justifies.
- `README.md` describes the pipeline in a way that still foregrounds model training and simulation as if those are the main frontier. The repo's own evidence says the broad model-accuracy frontier is near exhausted.
- `POOL_STRATEGY_RECOMMENDATION.md` is internally solid for the stochastic regime, but it can mislead if read as the universal strategy source of truth after the deterministic pivot. It solves a different problem: ranking among portfolios of stochastic brackets.
- `CLAUDE.md` contains a tension that should be made explicit: it says stochastic generation is retired as the primary path, while `MEMORY.md` still locks `f4_first_tv` as the recommended mode. That is not a contradiction if interpreted as "current best shipping baseline is stochastic, current research frontier is deterministic," but the docs should say exactly that.

### Core Strategic Reality

The repo's evidence points to a narrower, saner mission:

- The project has not shown a durable general predictive edge.
- The project has shown value in falsifying bad ideas.
- The project may still improve one-shot pool decisions at the margin.
- The most leverage is probably in the champion decision, not in global feature/model complexity.

That means the repo should describe itself less like a machine that can solve March Madness and more like a disciplined experimental system for a noisy, low-signal decision problem.

### Recommended Goal Rewrite

The top-level goal should effectively become:

"Maximize the probability of submitting one strong bracket in a winner-take-all pool, using disciplined backtesting to eliminate bad ideas and improve the few decisions that matter most."

What should be removed or softened from the repo narrative:

- any implication that a strong general forecasting edge is likely
- any implication that more model complexity is the obvious path forward
- any implication that contrarianism itself is a source of score

What should be emphasized instead:

- one-shot decision quality
- champion-pick discipline
- dead-end avoidance
- shipping a bracket next year, not endlessly expanding the research surface

### Strategy-Doc Positioning

The docs should distinguish three layers clearly:

- Shipping baseline: currently stochastic `f4_first_tv` per `MEMORY.md`
- Research frontier: deterministic champion-focused successors to `meta_gbm v2`
- Archived ideas: contrarian weighting, anti-chalk objectives, broad tuning/augmentation variants unless they clearly move `P(1st)`

Right now that distinction exists across multiple files, but not cleanly in one place.

### Suggested Documentation Changes Later

No code changes are recommended here. For later doc cleanup, the highest-value edits would be:

1. Update `README.md` to frame the repo as a pool decision-support system, not a broad prediction engine.
2. Add one sentence to `CLAUDE.md` clarifying that the stochastic path remains the best locked submission baseline while deterministic meta-selector work is the active research path.
3. Add a short "project mission" paragraph to `MEMORY.md` or a dedicated strategy doc that says the repo's real job is to improve one submitted bracket, not to prove a universal model edge.
4. Add a short "evidence ceiling" note in strategy docs: this is a tiny-sample, high-noise domain, so the system should optimize for disciplined decisions, not certainty.

### Bottom Line

The repo is worth keeping if it is honest about its mission.

It is not a credible path to "solve" March Madness in the general sense.
It is still a credible path to:

- preserve accumulated lessons
- avoid repeating expensive dead ends
- test a small number of focused bracket-decision ideas
- help produce a better 2027 submission than casual intuition alone
