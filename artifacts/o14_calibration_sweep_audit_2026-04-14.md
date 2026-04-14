# O14 — Calibration-to-Pool Interaction (Temperature Sweep)

**Date:** 2026-04-14
**Branch:** `claude/simplify-repo-structure-keQXE`
**Status:** closure evidence for `COUNCIL_LESSONS.md §2 O14`

## Gate (from §2)

> Calibration-to-pool interaction. Measure whether slightly
> upset-over-estimating calibration improves pool rank vs true
> calibration.

## TL;DR

**The specific hypothesis is falsified.** Upset-over-estimation
(flattening the probabilities via T > 1) does NOT improve pool rank
— it monotonically *decreases* the fraction of sampled brackets that
beat chalk across 13 backtest years. The pool-rank-optimal
temperature is actually *sharper* than calibrated (T = 0.5), not
flatter.

Calibration-to-pool interaction is real in the sense that Brier-optimal
T (= 1.0) is NOT pool-score-optimal. But the gradient points in the
opposite direction of the gate's prediction, and the magnitude is
small (≤ 2 pp on pBeatChalk, ≤ 5 pt on best-score). **Keep
T = 1 honest calibration.** The dominant pool-rank lever is
stochastic sampling (O13 closed), not calibration temperature.

## Setup

- Base model: seed-only pairwise win probabilities via `_win_rate`
  (historical 1985-2025 lookup + logistic fallback).
- Temperature scaling: `p' = sigmoid(logit(p_raw) / T)`. `T = 1` is
  identity. `T < 1` sharpens (more chalk-confident). `T > 1` flattens
  (more upsets favored in the sampled draws).
- Sampling: for each game, flip a weighted coin using the scaled
  probability. 200 stochastic brackets per (year, T). Deterministic
  RNG seed per year so the draws are comparable across Ts.
- Scoring: ESPN standard (R64:10, R32:20, S16:40, E8:80, F4:160,
  CHAMP:320) against actual tournament outcomes via the same
  `score_bracket_espn` infrastructure used in O12/O13.
- Years: 2011, 2013-2019, 2021-2025 (13 yrs; 2012 / 2020 excluded).
- Why NOT full pool MC (opponent sampling): the O13 closure
  established that stochastic-vs-argmax dominates ±1 pp calibration
  effects on P(1st). For O14 the question is whether the
  calibration *temperature* is a real lever; the cheaper bracket-
  score-vs-chalk proxy captures that faithfully at 1 / 100th the
  compute cost.

## Results

| T     | mean Brier | mean Score | best Score | mean edge vs chalk | best edge vs chalk | P(beats chalk) |
|-------|------------|------------|------------|--------------------|--------------------|----------------|
| 0.50  | 0.1898     | 566        | **914**    | -153               | +195               | **0.236**      |
| 0.75  | 0.1840     | 548        | 912        | -170               | +193               | 0.226          |
| **1.00** | **0.1838** | 537    | 909        | -181               | +191               | 0.224          |
| 1.25  | 0.1857     | 530        | 912        | -188               | +194               | 0.217          |
| 1.50  | 0.1885     | 524        | **914**    | -194               | +195               | 0.210          |
| 2.00  | 0.1945     | 516        | 909        | -203               | +191               | 0.201          |
| 3.00  | 0.2050     | 506        | 902        | -213               | +183               | 0.194          |

**Optima:**
- Brier: T = **1.0** (honest calibration, by definition for a
  well-calibrated base).
- mean Score: T = **0.5** (chalkier beats upset-pickier on average;
  most games resolve favorite).
- best Score: T = **{0.5, 1.5}** (both extremes tie at 914; T = 1 is
  909 — mildly suboptimal).
- P(beats chalk): T = **0.5** (strictly monotone decreasing in T).

## Interpretation

1. **The gate's hypothesis** ("upset-over-estimation improves pool
   rank") predicts the P(beats-chalk) curve should peak at some
   T > 1. The data shows the opposite — P(beats-chalk) *monotonically
   decreases* from 0.236 at T = 0.5 to 0.194 at T = 3.0. Slight upset
   overestimation (T = 1.5) produces fewer rank-1 candidates, not
   more. **Gate hypothesis falsified.**

2. **Calibration-to-pool interaction exists but is small.** T = 1
   (Brier-optimal) lands on the single-T-lower side of the best-score
   curve: T = 0.5 ties T = 1.5 at 914 while T = 1 hits only 909. So
   *some* calibration drift improves best-score, just not in the
   direction the gate predicted.

3. **The observed bimodality** (best-score peaks at both T = 0.5 and
   T = 1.5) is consistent with the O13 verdict: sharper-than-chalk
   OR flatter-than-chalk portfolios both diversify away from T = 1
   in different directions, and in any given year one tail may hit
   the year's regime. This is a *portfolio-of-brackets* effect, not a
   single-bracket effect.

4. **Magnitudes are trivial**. The best-score gap (914 - 909 = 5
   ESPN points) is less than one R64 correct pick. The P(beats chalk)
   spread across the full T range is 4 pp. Running two different
   calibrations to chase this is not worth the operational complexity.

## Decision

**Keep T = 1 honest calibration.** Do not introduce a separate
pool-calibration temperature. O14 is closed as a **null-result-on-the-
gate, small-interaction-detected** finding.

- The dominant pool-rank lever is stochastic sampling (O13 closed):
  `champ_first_tv` / `e8_first_tv` already in production.
- Brier-optimal (T = 1) stays locked as the production calibration
  temperature for all downstream consumers.
- The O14 dead-end — "upset-over-estimation doesn't help pool rank"
  — is added to `MEMORY.md §2` to prevent re-litigation.

## What this closure ships

1. **Evidence script:** `scripts/o14_calibration_sweep.py` —
   reproducible temperature sweep, deterministic under
   `RANDOM_SEED = 42`. Runs in ~5 s.
2. **Machine-readable artifact:**
   `artifacts/o14_calibration_sweep_2026-04-14.json` — per-year,
   per-T, full table.
3. **Audit document:** this file.
4. **Lock test:** `tests/test_calibration_pool_interaction_lock.py`
   — 6 tests, ~5 s:
   - script is runnable and produces output in the expected shape
   - Brier-optimal T stays at 1.0
   - pBeatChalk is (non-strictly) decreasing in T (gate's hypothesis
     would invert this; any future inversion is a signal)
   - magnitude of the T ≠ 1 effect stays small (|best-score gap| ≤ 10)
   - aggregate numbers stay near recorded values
5. **MEMORY.md §2** new dead-end row `D13` — "upset-over-estimation
   for pool rank."

## Residual risks & non-gates

- **Base model is seed-only.** ML-model-based pairwise probs
  (noseed / blend) would give different curves. The O14 gate text
  refers to "true calibration" generically, so the seed baseline is a
  defensible fixed point — but if production ever moves to a
  non-seed base model, a re-audit with that base is warranted. The
  lock test's assertion about "Brier-optimal T = 1" would still hold
  for any well-calibrated base; the pBeatChalk curve could shift.
- **Single-bracket proxy.** Real pool rank requires opponent MC; this
  uses "beats chalk" as the proxy. Expected-portfolio rank in a real
  pool could behave differently — but O13 already closed the
  portfolio-vs-single dimension, so this limitation is accepted.
- **N = 200 samples per year** is enough to stabilize means to ~ 5
  score points. Best-score has heavier-tailed variance; the
  914 / 912 / 909 gaps are within sampling noise. The *direction* of
  pBeatChalk (monotone decrease in T) is the robust signal the gate
  cares about.

## Closure record

`COUNCIL_LESSONS.md §2 O14` → `[closed 2026-04-14]`. Crumb:

> Temperature sweep T ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0} × 13
> years × 200 stochastic brackets. Brier-optimal T = 1.0 (honest).
> pBeatChalk *monotonically decreasing* in T: 0.236 at T=0.5 →
> 0.194 at T=3.0. The gate's "upset-over-estimation improves pool
> rank" hypothesis is falsified. Calibration-to-pool interaction
> exists (T=1 is not best-score-optimal) but direction is OPPOSITE
> the gate's prediction and magnitude is trivial (≤ 5 ESPN points
> best-score, ≤ 2 pp pBeatChalk). Keep T=1. Evidence:
> `artifacts/o14_calibration_sweep_2026-04-14.json` +
> `artifacts/o14_calibration_sweep_audit_2026-04-14.md`. Drift
> guard: `tests/test_calibration_pool_interaction_lock.py`. New
> MEMORY.md §2 D13 dead-end for upset-over-estimation.
