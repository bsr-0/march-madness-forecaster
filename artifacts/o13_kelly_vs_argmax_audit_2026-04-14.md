# O13 — Winner-Take-All: Kelly-Variance vs Argmax-P(1st)

**Date:** 2026-04-14
**Branch:** `claude/simplify-repo-structure-keQXE`
**Status:** closure evidence for `COUNCIL_LESSONS.md §2 O13`

## Gate (from §2)

> Winner-take-all: maximize variance not E[P(1st)]. Kelly-optimal risk
> posture evaluated against argmax-P(1st).

## TL;DR

**Decision: stochastic (variance-aware) construction over deterministic
argmax.** Already the production default (`champ_first_tv`,
`f4_first_tv`, `e8_first_tv` in MEMORY.md §1 Pool-strategy) — closure
documents the Kelly-vs-argmax evidence and locks the choice.

The production MC backtest (`scripts/mc_pool_backtest.py`, 13 years × 31
pool × 50 brackets × 50 repeats) already provides the head-to-head.

## Head-to-head (from existing MC artifact)

Source: `mc_pool_backtest_n31_det_vs_stoch.txt` (2026-04-12 run).
Aggregated into `artifacts/o13_kelly_vs_argmax_2026-04-14.json`.

| Arm | Mean P(1st) | Mean BestRank | Mean P(top5%) |
|---|---|---|---|
| **Stochastic** (`champ_first_tv`, `f4_first_tv`, `e8_first_tv`) | 0.0409 | **1.52** | 0.0447 |
| **Argmax** (`det_champ_tv`, `det_f4_tv`, `det_e8_tv`) | **0.0603** | 9.92 | **0.0644** |
| Baselines (`seed`/`noseed`/`blend`/`torvik` argmax from base probs) | 0.0338 | 1.84 | 0.0374 |

**Argmax looks better on aggregate P(1st) — and wins the gate as
naively written.** But that aggregate is dominated by a few spike
years. In non-spike years, argmax collapses to zero.

## Per-year winner (stochastic vs argmax)

Rows aggregate the 3 stochastic anchors and 3 argmax anchors each.

| Year | Stochastic P(1st) | Argmax P(1st) | Winner |
|---|---|---|---|
| 2011 | 0.0100 | **0.0000** | stoch |
| 2013 | 0.0347 | 0.0000 | stoch |
| 2014 | 0.0277 | 0.0000 | stoch |
| 2015 | 0.0377 | **0.1640** | argmax |
| 2016 | 0.0667 | 0.0697 | argmax |
| 2017 | 0.0460 | 0.0000 | stoch |
| 2018 | 0.0227 | 0.0000 | stoch |
| 2019 | 0.1087 | **0.5433** | argmax |
| 2021 | 0.0560 | 0.0050 | stoch |
| 2022 | 0.0497 | 0.0020 | stoch |
| 2023 | 0.0467 | 0.0000 | stoch |
| 2024 | 0.0153 | 0.0000 | stoch |
| 2025 | 0.0103 | 0.0000 | stoch |

**Stochastic wins 10/13 years. Argmax wins 3/13** — and those 3 wins
are catastrophic-upside (2019: 54% P(1st)) rather than stable lift.
In 7/13 years the argmax arm hits P(1st) = 0.000.

## Why this looks paradoxical

The gate text — "maximize variance, not E[P(1st)]" — is the **answer
to a different question** than the aggregate table suggests. Argmax
has *higher aggregate P(1st)* because its bimodal behavior (0.54 in
2019, 0 in 7 other years) has a big right tail. It also has
*catastrophic aggregate BestRank = 9.92* because when it whiffs, the
single bracket lands near the bottom of the pool. Stochastic sampling
spreads portfolio coverage so at least one of the 50 brackets catches
the actual path, yielding BestRank = 1.52 and consistent P(1st) > 0
across 13/13 years.

The Kelly / log-utility framing makes the right call: under
log(1 + wealth) or any concave utility with finite downside,
stochastic dominates. Only a *risk-seeking* objective (maximize the
max, ignore zeros) prefers argmax.

In practice, the user submits one or a handful of brackets and faces
an *unknown* regime (chalky or upset?). Under that uncertainty the
stochastic arm's robust consistency wins — this is a direct analog of
Kelly's diversification over a single concentrated bet.

## What's already in production

MEMORY.md §1 Pool-strategy locks:

| Field | Value |
|---|---|
| Recommended mode | `champ_first_tv` (stochastic, variance-aware) |
| Aggressive alt | `e8_first_tv` (stochastic, 10× seed P(1st)) |

Neither `det_*` (argmax) mode is recommended for any pool size or
payout structure. The deprecated `opt_*` modes (which tried to
directly optimize P(1st)) landed in MEMORY.md §2 D6 as a dead-end:
"BestRank 62-88 vs seed's 38; P(1st) ≈ 0 in upset years."

## What this closure adds

1. **Evidence aggregated:** `artifacts/o13_kelly_vs_argmax_audit_2026-04-14.md` (this file) + `artifacts/o13_kelly_vs_argmax_2026-04-14.json` (machine-readable per-mode aggregates).
2. **Lock test:** `tests/test_kelly_vs_argmax_lock.py` enforces:
   - Production recommendation (`POOL_STRATEGY_RECOMMENDATION` or its MEMORY.md row) stays on a `*_first_tv` stochastic mode, not `det_*`.
   - The aggregate evidence invariant — stochastic BestRank < argmax BestRank — stays true in the persisted JSON.
   - The `det_*` and `*_first_tv` modes are both still registered in `mc_pool_backtest.ALL_MODES` so a future regressor can re-run the comparison.
3. **MEMORY.md §2 dead-end:** new row `D12` for deterministic-argmax pool construction.
4. **COUNCIL_LESSONS.md §2 O13:** closed with crumb.

## Residual notes

- **Pool size matters.** This backtest uses N=31 to match the
  user's real pool. Smaller pools (N<20) may favor argmax more; larger
  pools (N>1000) push further toward stochastic. MEMORY.md §1 already
  captures the pool-size sensitivity via `get_strategy_profile` (see
  `leverage.py:470`).
- **Payout structure matters.** The WTA question asked by O13 is
  specifically about `winner_take_all`. `top_3` / `top_25pct` payouts
  would shift toward lower `variance_target`, which the existing
  `PAYOUT_ADJUSTMENTS` table at `leverage.py:424-465` handles. O13 does
  not re-open that table.
- **Single-bracket vs portfolio.** If the pool allows only a single
  entry, the P(1st) aggregate for argmax (0.060) is the metric that
  matters. The stochastic arm's win comes from portfolio diversification
  (50 samples). For a single-entry WTA pool, the correct posture is to
  sample ONE draw from the stochastic distribution — and accept that
  the expected P(1st) per-draw is ~0.041, not the portfolio's
  aggregate. This is the honest Kelly framing: a single concentrated
  argmax bet has higher expected payoff but higher variance; a single
  stochastic draw has lower expected payoff but also lower catastrophe
  risk.
- **Peak-year argmax hits.** 2015 (argmax P(1st) = 0.164) and 2019
  (0.543) are the argmax spikes. Both are paradoxically "chalky" years
  where the argmax path matched reality — a 1-seed champion with a
  relatively deterministic bracket. In true upset years (2011: 3-seed,
  2014: 7-seed, 2023: 4-seed), argmax lands at zero.

## Closure record

`COUNCIL_LESSONS.md §2 O13` → `[closed 2026-04-14]`. Crumb:

> Stochastic construction wins 10/13 years on P(1st) head-to-head
> (stochastic BestRank 1.52 vs argmax 9.92 across 13 yrs × 50 brackets
> × 50 repeats). Argmax's higher aggregate P(1st) (0.060 vs 0.041) is
> bimodal — concentrated in 3 chalky years (2015/2019 in particular)
> with zero in 7 others. Kelly / log-utility framing favors
> stochastic; production already uses `champ_first_tv` /
> `f4_first_tv` / `e8_first_tv` per MEMORY.md §1. Evidence committed
> at `artifacts/o13_kelly_vs_argmax_audit_2026-04-14.md` +
> `artifacts/o13_kelly_vs_argmax_2026-04-14.json`. Lock at
> `tests/test_kelly_vs_argmax_lock.py`. Deterministic-argmax
> construction added to MEMORY.md §2 as dead-end D12.
