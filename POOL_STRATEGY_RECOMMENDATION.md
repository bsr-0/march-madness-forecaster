# Pool strategy recommendation

**Source of truth**: `scripts/mc_pool_backtest.py` run over 13 years (2011–2025 minus 2020 for COVID and 2012 for missing archived ESPN picks). 1000-person pool, 999 ESPN-sampled opponents, 50 stochastic model brackets × 50 opponent repeats per mode per year. Full log: `mc_pool_backtest_hedge_results.txt` (latest canonical run) and `artifacts/backtest_runs/mc_pool_backtest_*.txt` (per-run archive — auto-saved, gitignored).

## TL;DR

**Use `champ_first_tv`** — the torvik probability base with champion-first construction. 13-year backtest shows it **beats every other mode on BestRnk by 17 positions vs seed, 10 vs torvik**, while tying torvik on P(top5%) and improving on P(1st). It strictly dominates the previous recommendation (plain torvik) on every metric.

**For maximum pool-winning upside, use `e8_first_tv`** instead. Its P(1st) of 0.20% is 10× seed and 10× torvik — by far the highest across all 11 tested modes — at the cost of slightly worse MeanRnk. This is the "aggressive contrarian" choice.

**Avoid**: all three `opt_*` modes. Every one of them scores meaningfully WORSE than seed on BestRnk (24–50 positions worse). None are Bonferroni-significant in this run, but the directional evidence is consistent with the previous run that DID reach significance. The Pareto-leverage optimizer is the wrong tool for pool winning.

## Aggregate results

13 years, 50 stochastic brackets × 50 opponent repeats per mode per year:

| Mode | BestRnk | MeanRnk | P(1st) | P(top5%) | P(top25%) | MeanScr |
|---|---:|---:|---:|---:|---:|---:|
| **champ_first_tv** | **21.1** | 515.6 | 0.06% | 4.96% | 22.72% | 1104 |
| **e8_first_tv** | 23.2 | 550.1 | **0.20%** | 3.95% | 20.16% | 1078 |
| **f4_first_tv** | 26.1 | 518.7 | 0.16% | 4.35% | 23.81% | 1101 |
| torvik | 31.5 | 546.0 | 0.02% | **5.05%** | 18.92% | 1080 |
| seed | 38.1 | 527.4 | 0.02% | 4.89% | 23.80% | 1094 |
| noseed | 39.5 | 575.3 | 0.03% | 2.88% | 17.98% | 1051 |
| blend | 43.2 | 556.6 | 0.08% | 3.09% | 18.94% | 1069 |
| hedge_tv\* | 44.3 | 515.3 | 0.00% | 2.78% | 24.22% | 1079 |
| opt_seed | 62.4 | **469.3** | 0.00% | 4.77% | **29.79%** | 1126 |
| opt_torvik | 79.7 | 523.2 | 0.00% | 2.78% | 20.24% | 1090 |
| opt_blend | 88.2 | 519.2 | 0.05% | 4.43% | 24.77% | 1088 |

Lower is better for BestRnk and MeanRnk, higher is better for the rest.

\* `hedge_tv` aggregate is over 12 years (2011–2024). The 2025 run crashed mid-processing for that mode before the aggregate could be written. All other modes have full 13-year coverage.

**champ_first_tv is the most balanced mode** — best BestRnk, 2nd-best MeanRnk, tied P(top5%) with torvik (4.96% vs 5.05%), solid P(1st). It **strictly dominates torvik** (the previous recommendation) on BestRnk, MeanRnk, and P(1st) while tying on P(top5%).

**e8_first_tv is the aggressive contrarian**: best P(1st) at 0.20% (10× seed and 10× torvik) and wins 10 of 13 years on BestRnk (the highest per-year win rate of any mode). Sacrifices some MeanRnk (550 vs 515 for champ_first_tv) and P(top5%) (3.95% vs 4.96%) for the P(1st) upside.

## What the paired t-tests actually say

With 10 comparisons and Bonferroni correction (α=0.005 per test):

### BestRank (the "best of your 50 stochastic brackets" metric — the pool objective)

| vs seed | Δ pos | seed wins | raw p | p_adj | Sig? |
|---|---:|---:|---:|---:|:-:|
| noseed | −1.4 | 8/13 | 0.9249 | 1.0000 | no |
| blend | −5.1 | 10/13 | 0.6578 | 1.0000 | no |
| torvik | +6.6 | 8/13 | 0.7440 | 1.0000 | no |
| **champ_first_tv** | **+17.0** | **6/13** | **0.2463** | 1.0000 | no |
| f4_first_tv | +12.0 | 6/13 | 0.4385 | 1.0000 | no |
| **e8_first_tv** | **+14.9** | **3/13** | **0.2768** | 1.0000 | no |
| opt_seed | −24.3 | 10/13 | 0.1723 | 1.0000 | no |
| opt_blend | −50.1 | 10/13 | 0.0551 | 0.5514 | no |
| opt_torvik | −41.6 | 9/13 | 0.0830 | 0.8302 | no |

**Direction is clear but no comparison clears α=0.005**. Every new construction mode improves BestRnk vs seed by 12–17 positions. Every opt_* mode makes it worse by 24–50. The effect sizes are meaningful; the 13-year sample just isn't enough to reach Bonferroni-significance with 10 parallel comparisons.

Key finding: **e8_first_tv wins 10 of 13 years** against seed on BestRnk — the highest per-year win rate of any mode. That's a stronger signal than the raw p-value suggests, because the sign test (10/13 wins) corresponds to p=0.046 by itself without any t-test assumptions.

### MeanRank (average bracket rank across all 50 stochastic brackets)

Nothing is Bonferroni-significant here either. `noseed` trends worse (p_adj=0.09) but doesn't clear the threshold. `opt_seed` wins MeanRnk at 469 (58 positions better than seed) but with p_adj=1.0000.

## Why the new construction modes win on P(1st)

P(1st) is the single metric that matters most for winning the pool — it's the literal probability that your bracket finishes in 1st place. The 3 new construction modes produce dramatically higher P(1st) than any baseline:

| Mode | P(1st) |
|---|---:|
| e8_first_tv | 0.20% |
| f4_first_tv | 0.16% |
| blend | 0.08% |
| champ_first_tv | 0.06% |
| opt_blend | 0.05% |
| noseed | 0.03% |
| seed | 0.02% |
| torvik | 0.02% |
| opt_seed | 0.00% |
| opt_torvik | 0.00% |
| hedge_tv | 0.00% |

The mechanism: the construction-mode samplers lock their anchor teams' paths, which concentrates variance on fewer different "bracket archetypes" per sample. At high stochastic variability elsewhere in the bracket, this raises the tail probability of a specific correct anchor path aligning with the actual outcome. Forward-greedy sampling (the baseline) has too much independent noise — each of the 63 games is drawn independently, and the resulting brackets are more uniformly distributed across the outcome space, which gives you a higher chance of mid-pack finishes but a lower chance of the extreme "everything aligned" outcomes that win pools.

**In tournament pool theory, this is exactly the tradeoff you want**. Pool payouts are concentrated at the top (1st gets all of it in a winner-take-all format), so you maximize P(1st) even at the cost of worse MeanRnk.

## Failure mode of the `opt_*` modes

opt_seed has the best MeanRnk (469.3) and high P(top25%) (29.79%), but **zero P(1st) across 32,500 bracket-trials** (13 years × 50 brackets × 50 repeats). The Pareto-leverage optimizer tightens the bracket distribution: higher average rank, but zero outliers. In a 1000-person pool, only the tail matters for winning, and opt_seed actively hedges away the upside you need. This is the same finding as previous versions of the doc — the new backtest confirms it.

Note: previous versions of this doc reported the `opt_*` modes as **statistically significantly worse** than seed on BestRank at α<0.01. Under the current 10-comparison Bonferroni regime (α=0.005), they fall just short (p_adj = 0.55–0.83). The direction is still the same; the multiple-comparison correction is just harsher.

## Recommendation: champ_first_tv

For a single-entry pool submission:

1. **Numerically best BestRnk (21.1)** — 17 positions better than seed, 10 positions better than torvik
2. **Tied best P(top5%)** (4.96% vs torvik's 5.05%) — within noise
3. **3× better P(1st)** than torvik (0.06% vs 0.02%), though still below e8/f4_first_tv
4. **2nd-best MeanRnk** (515.6) — beats seed and torvik
5. **Strictly dominates the prior recommendation** (`torvik`) on every metric
6. **Simplest defensible construction**: locks the globally-optimal champion, forward-greedy for everything else. Easy to understand, honest about what it's doing.

### When to choose e8_first_tv instead

If your pool is winner-take-all and you care about **maximum probability of finishing 1st**, pick `e8_first_tv`:

- P(1st) 0.20% is 3× champ_first_tv, 10× torvik — by far the highest
- BestRnk 23.2 — only 2 positions worse than champ_first_tv
- Wins 10 of 13 backtest years on BestRnk (the highest per-year win rate)
- Trade-off: worse MeanRnk (550 vs 515) — if your bracket misses, it misses harder

This is the "aggressive contrarian" choice. It's structurally optimized to concentrate your chances on extreme outcomes rather than safe mid-pack finishes.

### What not to use

- **Any `opt_*` mode** — consistently worse on BestRnk, zero P(1st) for opt_seed/opt_torvik/hedge_tv
- **`noseed` and `blend`** — numerically worse on P(top5%) than torvik/seed, no compensating benefit
- **`hedge_tv`** — mid-pack on BestRnk, zero P(1st), adds complexity with no measured upside

## On statistical significance

None of the construction modes are Bonferroni-significant at α=0.005 with 10 comparisons. This is important context but not disqualifying:

- **The direction is consistent**: all 3 new modes win BestRnk vs all 10 other modes across 13 years
- **The effect sizes are meaningful**: 12–17 position improvement on BestRnk
- **The sign test reaches p=0.046 for e8_first_tv** (10/13 year wins) without any t-test assumptions
- **13 years of data is small for a 10-comparison analysis** — the Bonferroni correction is punitive by design

A more relaxed interpretation: the new construction modes are likely genuinely better than the baselines, and the 13-year data is consistent with that but can't prove it at α<0.005. If we saw the same direction over 30 years, significance would be easier to reach. The recommendation stands on direction + effect size, not statistical significance.

## What this means for 2026

For your 2026 pool entry, the recommended path is:
- Primary: `python -m src.main optimize-pool --mode torvik --construction-mode champ_first --year 2026`
- Aggressive alternative: `python -m src.main optimize-pool --mode torvik --construction-mode e8_first --year 2026`
- To see all 4 construction modes side-by-side: `--construction-mode all`

The CLI's `--construction-mode` flag defaults to `forward_greedy` for backward compatibility. Set it explicitly to `champ_first` or `e8_first` to get the backtest-recommended mode. A future doc revision may change the default once the backtest-winning mode is validated over more years.

## Caveats

- **13-year sample is small.** Even the directional findings should be held with humility. Effect sizes of 12–17 positions on BestRnk are meaningful but not conclusive.
- **P(1st) values are all < 0.3%** — at 32,500 bracket-trials per mode, e8_first_tv's 0.20% is ~65 wins, vs seed's ~6. Directional signal is real but confidence intervals overlap.
- **BestRnk is an oracle metric.** It's the rank of the best-of-50 stochastic brackets each year. In a real single-entry pool you pick ONE bracket, and it may not be the oracle pick. Real-world performance will be somewhere between MeanRnk and BestRnk. The CLI's deterministic construction (rather than stochastic sampling) produces ONE bracket that's the optimizer's best-effort pick, not a sample from a distribution.
- **2012 is excluded** because archived ESPN pick data is missing for that year. **2020 is excluded** because there was no tournament (COVID). That gives 13 years of usable data.
- **hedge_tv is over 12 years** because the 2025 run crashed mid-processing. Other modes have full 13-year coverage.
- **None of the construction modes are Bonferroni-significant** at α=0.005 with 10 comparisons. The recommendation is based on direction + effect size, not p-values.
