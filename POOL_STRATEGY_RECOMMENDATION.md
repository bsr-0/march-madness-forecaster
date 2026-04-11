# Pool strategy recommendation

**Source of truth**: `scripts/mc_pool_backtest.py` run over 13 years (2011–2025 minus 2020 for COVID and 2012 for missing archived ESPN picks). 1000-person pool, 999 ESPN-sampled opponents, 50 stochastic model brackets × 50 opponent repeats per mode per year. Full log: `mc_pool_backtest_hedge_results.txt` (latest canonical run) and `artifacts/backtest_runs/mc_pool_backtest_*.txt` (per-run archive — auto-saved, gitignored).

## TL;DR

**Use `torvik`** (or plain `seed` — they're statistically tied). **Avoid everything with `opt_` in the name.**

## Aggregate results

| Mode | BestRnk | MeanRnk | P(1st) | P(top5%) | P(top25%) | MeanScr |
|---|---:|---:|---:|---:|---:|---:|
| **torvik** | **31.5** | 546.0 | 0.0002 | **5.06%** | 18.93% | 1080 |
| seed | 38.1 | 527.4 | 0.0001 | 4.90% | 23.81% | 1094 |
| noseed | 39.5 | 575.3 | 0.0003 | 2.87% | 17.99% | 1051 |
| blend | 43.2 | 556.6 | **0.0008** | 3.10% | 18.93% | 1069 |
| hedge_tv | 80.0 | 570.2 | 0.0000 | 2.83% | 14.36% | 1055 |
| opt_seed | 176.4 | **471.6** | 0.0000 | 4.16% | **31.44%** | **1117** |
| opt_torvik | 322.3 | 587.7 | 0.0000 | 2.54% | 12.20% | 1036 |
| opt_blend | 342.7 | 556.2 | 0.0000 | 1.74% | 22.22% | 1052 |

Lower is better for BestRnk/MeanRnk, higher is better for the rest.

## What the paired t-tests actually say

With 7 comparisons and Bonferroni correction (α=0.0071):

### BestRank (the "best of your 50 stochastic brackets" metric — the pool objective)

| vs seed | Δ pos | seed wins | raw p | p_adj | Significant |
|---|---:|---:|---:|---:|:-:|
| noseed | −1.4 | 5/13 | 0.9254 | 1.0000 | — |
| blend | −5.1 | 3/13 | 0.6585 | 1.0000 | — |
| torvik | +6.6 | 5/13 | 0.7435 | 1.0000 | — |
| hedge_tv | −41.9 | 3/13 | 0.1197 | 0.8380 | — |
| **opt_seed** | **−138.3** | **0/13** | 0.0038 | **0.0268** | ★ opt_seed WORSE |
| **opt_torvik** | **−284.2** | **1/13** | 0.0004 | **0.0030** | ★ opt_torvik WORSE |
| **opt_blend** | **−304.6** | **0/13** | 0.0007 | **0.0046** | ★ opt_blend WORSE |

**Every `opt_*` mode is significantly worse than seed on BestRank.** Seed wins 0 or 1 of 13 years against each optimizer. This is the strongest finding in the backtest.

### MeanRank

Nothing is significantly different from seed after Bonferroni. `noseed` trends worse (p_adj=0.065) but doesn't clear the threshold. `opt_seed` has the best raw MeanRank (471.6) but the second-worst BestRank.

## The failure mode of the optimized modes

`opt_seed` has the best MeanRnk (471.6) and best MeanScr (1117) of any mode — yet one of the worst BestRnks (176.4) and zero P(1st) events across 32,500 bracket-trials. What's happening:

**The Pareto-leverage optimizer tightens the bracket distribution.** It finds brackets with high expected value against the public and low variance. Higher average score, no outliers.

**In a 1000-person pool, only the tail matters.** To win, you need a bracket that's in the top 0.1% — that means you need high variance, not high mean. The optimizer is doing exactly the wrong thing for the pool objective: it's hedging away the upside you need to beat 999 other people.

This is measurable, not theoretical. The t-tests at α<0.01 say the optimized modes leave you ~140–300 ranks further from first place than plain stochastic seed draws.

## Why torvik is the recommendation (even though it's not statistically distinguishable)

Among the four statistically-tied base modes (seed, noseed, blend, torvik), torvik is the right pick:

1. **Numerically best BestRnk (31.5)** and **numerically best P(top5%) (5.06%)** — the tie is real, but inside the tie, torvik has the tiniest edge on both pool-relevant metrics. When two options are statistically indistinguishable, pick the one that's numerically better on the metric you care about.
2. **Simplest defensible model**: barthag + Log5 + MC. No ML ensemble, no leverage optimizer, no calibration stage. Fewer moving parts = fewer failure modes.
3. **Data-source aligned**: uses Torvik four-factors directly, the same pre-tournament source that is validated as direct-scrape with no look-ahead provenance risk (see `AUDIT_DATA_LEAKAGE.md`).
4. **Benign failure mode**: when torvik is wrong, it's wrong by landing at rank ~100 instead of rank ~30. It never produces catastrophic single-year outcomes the way `opt_torvik` does (worst year BestRnk: 749 vs torvik worst year: 51.9 in the old 17-year data — the 13-year run shows the same qualitative pattern).

**`seed` is the equally-defensible fallback.** Same statistical tier, slightly worse numerically, simpler still. If you'd rather run zero-dependency seed probabilities and call it a day, the data does not give us grounds to reject that.

**What not to use**:
- Any `opt_*` mode — significantly worse on BestRank at p<0.01
- `noseed` and `blend` — numerically worse on P(top5%) than torvik or seed, no compensating benefit
- `hedge_tv` — 80.0 BestRnk is OK but worse than every base mode; no P(1st) events; adds complexity with no measured upside

## On P(1st) specifically

Every mode has aggregate P(1st) ≤ 0.0008. With 32,500 bracket-trials per mode (13 years × 50 brackets × 50 repeats), that's ~26 first-place finishes for blend (the highest) down to ~0 for all `opt_*` modes. These counts are tiny and confidence intervals overlap substantially. **You cannot reliably pick a mode on P(1st) alone with 13 years of data.** The signal that IS reliable is BestRnk and P(top5%), where torvik has a nominal edge that just happens to not clear the statistical threshold.

## What this means for 2026

For your 2026 pool entry:
- Use the `torvik` mode (barthag + Log5 + MC, no ML, no leverage optimizer)
- Currently this is available through `scripts/mc_pool_backtest.py --years 2026` — but it's not exposed through the `src/cli/pool_cmds.py` path, which only wires up seed/noseed/blend
- To get a single submittable 2026 bracket out of torvik mode you'll need either (a) a small CLI addition to `pool_cmds.py` or (b) running `mc_pool_backtest.py` in a single-year mode and extracting the top-ranked stochastic bracket from its output

## Caveats

- **13-year sample is small.** Even the Bonferroni-significant `opt_*` findings should be held with humility — they're unlikely to be noise, but the effect size we can measure is constrained by the short history.
- **BestRnk is an oracle metric.** It's the rank of the best-of-50 stochastic brackets each year. In a real single-entry pool you pick ONE bracket, and it may not be the oracle pick. Real-world performance will be somewhere between MeanRnk and BestRnk.
- **2012 is excluded** because archived ESPN pick data is missing for that year. **2020 is excluded** because there was no tournament (COVID). That gives 13 years of usable data out of a 15-year window.
- **2026 is the first year with the fresh symmetric-augmented noseed model** (commit `086ab93`). The backtest does not exactly exercise the 2026 production pipeline.
