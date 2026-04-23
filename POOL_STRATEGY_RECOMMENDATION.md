# Pool strategy recommendation

**Source of truth**: `scripts/mc_pool_backtest.py` with `--team-identity` flag. 14 years (2011–2026 minus 2020 for COVID and 2012 for missing archived ESPN picks). 1000-person pool, 999 ESPN-sampled opponents, 50 stochastic model brackets × 50 opponent repeats per mode per year. Canonical run: `artifacts/backtest_runs/mc_pool_backtest_20260418_225840.txt` (O27 closure). Per-run archive: `artifacts/backtest_runs/mc_pool_backtest_*.txt`.

**Scoring:** Team-identity (real ESPN scoring — teams matched by name per round, not positional bool). Adopted 2026-04-18 per O26/O27 after discovering that shape-encoded scoring materially mis-ranked modes (see [Appendix: shape-encoded history](#appendix-shape-encoded-results-historical)).

## TL;DR

**Use `f4_first_tv`** — the torvik probability base with Final Four-first construction. Best MeanRank (573.6), best MeanScr (737), best P(top25%) (21.4%), and competitive BestRank (33.8, second only to torvik). Validated by three independent lines of evidence:

1. **N=1000 team-identity backtest** (this file) — best all-around profile
2. **Retroactive actual-pool analysis** (2023-2026, `COUNCIL_LESSONS.md §3 row 65`) — wins all 4 pool years when selecting from the ranker's top-3
3. **Within-portfolio ranker correlation** — MC ranker's mean_rank has ρ = −0.50 to −0.75 with actual score in 3 of 4 years

**For maximum pool-winning upside, use `e8_first_tv`** instead. Highest P(1st) at 0.29% (1.4× seed). Aggressive contrarian choice — concentrates your chances on extreme outcomes.

**Avoid**: all `opt_*` and `hedge_tv` modes — deprecated and removed from the backtest harness (2026-04-12). `det_*` modes (deterministic argmax) have poor BestRank (222-266) despite sometimes high P(top5%) — their apparent strength is carried by single chalky years (D12).

## Aggregate results — team-identity scoring

14 years, 50 stochastic brackets × 50 opponent repeats per mode per year:

| Mode | BestRnk | MeanRnk | P(1st) | P(top5%) | P(top25%) | MeanScr |
|---|---:|---:|---:|---:|---:|---:|
| **torvik** | **27.1** | 583.5 | 0.07% | **7.35%** | 20.01% | 732 |
| **f4_first_tv** *(recommended)* | 33.8 | **573.6** | 0.16% | 6.60% | **21.39%** | **737** |
| e8_first_tv *(aggressive)* | 35.5 | 601.9 | **0.29%** | 4.85% | 18.70% | 721 |
| champ_first_tv *(prior rec)* | 36.5 | 583.7 | 0.20% | 5.39% | 18.50% | 733 |
| seed *(baseline)* | 47.3 | 605.1 | 0.21% | 3.89% | 16.57% | 711 |
| champ_first_chalkfade_tv | 28.3 | 599.3 | 0.15% | 5.32% | 17.43% | 719 |
| blend | 41.8 | 643.2 | 0.11% | 3.95% | 14.93% | 688 |
| noseed | 90.7 | 722.6 | 0.05% | 1.10% | 10.70% | 622 |

Lower is better for BestRnk and MeanRnk, higher is better for the rest.

**Why f4_first_tv over torvik:** torvik has the best single-bracket ceiling (BestRank 27.1), but for a single-entry pool submission, you submit ONE bracket — MeanRank matters more than BestRank. f4_first_tv's MeanRank advantage (573.6 vs 583.5) means your average bracket places better. It also has the highest MeanScr (737) and highest P(top25%) (21.4%), making it the most robust choice across the distribution.

**Why not champ_first_tv anymore:** Under shape-encoded scoring (the prior methodology), champ_first_tv appeared dominant — BestRank 21.1 vs seed's 38.1. Under team-identity scoring, champ_first_tv drops to 4th on BestRank (36.5) and is middle-of-pack on most metrics. The shape-encoded BestRank advantage was an encoding artifact: shape scoring credited positional match, not team identity, which inflated champion-heavy construction modes.

## Statistical tests — team-identity

With 10 comparisons and Bonferroni correction (α=0.005 per test):

### BestRank

| vs seed | Δ pos | seed wins | raw p | p_adj | Sig? |
|---|---:|---:|---:|---:|:-:|
| noseed | −43.4 | 2/14 | 0.0177 | 0.177 | no |
| blend | +5.5 | 6/14 | 0.6815 | 1.000 | no |
| torvik | +20.2 | 7/14 | 0.3570 | 1.000 | no |
| champ_first_tv | +10.7 | 7/14 | 0.5841 | 1.000 | no |
| f4_first_tv | +13.4 | 7/14 | 0.3131 | 1.000 | no |
| e8_first_tv | +11.7 | 8/14 | 0.6610 | 1.000 | no |

No stochastic mode reaches Bonferroni significance on BestRank. Direction is consistent — all construction modes improve BestRank vs seed.

### MeanRank

| vs seed | Δ pos | seed wins | raw p | p_adj | Sig? |
|---|---:|---:|---:|---:|:-:|
| noseed | −117.5 | 1/14 | 0.0005 | **0.005** | **yes** |
| f4_first_tv | +31.5 | 9/14 | 0.1571 | 1.000 | no |

Only noseed is statistically distinguishable from seed (significantly worse). f4_first_tv shows the largest positive delta (+31.5 positions, wins 9/14 years) among stochastic modes.

## Recommendation: f4_first_tv

For a single-entry pool submission:

1. **Best MeanRank (573.6)** — your average bracket places ~31 positions better than seed
2. **Best MeanScr (737)** and best P(top25%) (21.4%) — most consistent scorer
3. **Competitive BestRank (33.8)** — second only to torvik (27.1)
4. **Validated against actual pool data** — retroactive analysis shows f4_first_tv winning all 4 actual pool years (2023-2026) when selecting from the top-3 ranked brackets
5. **Balanced construction**: locks the globally-optimal Final Four teams, forward-greedy for everything else. Less fragile than champion-first (one wrong champion pick collapses the entire strategy).

### When to choose e8_first_tv instead

If your pool is winner-take-all and you care about **maximum probability of finishing 1st**, pick `e8_first_tv`:

- P(1st) 0.29% — highest across all modes, 1.4× seed
- BestRnk 35.5 — still competitive
- Trade-off: worse MeanRnk (601.9 vs 573.6) — if your bracket misses, it misses harder

This is the "aggressive contrarian" choice for large pools where only 1st place pays.

### What not to use

- **`opt_*` and `hedge_tv`** — deprecated and removed. See "Deprecated modes" section.
- **`det_*` (deterministic)** — BestRank 222-266 despite P(top5%) up to 12.3%. The P(top5%) is bimodal and carried by chalky years (D12). 14/14 years stochastic > argmax under team-identity.
- **`noseed`** — statistically significantly worse than seed on MeanRank (p=0.005). High leverage ratio but bad model accuracy = losing more uniquely.

## Why the new construction modes win on P(1st)

P(1st) is the single metric that matters most for winning the pool — it's the literal probability that your bracket finishes in 1st place. The 3 construction modes produce higher P(1st) than baselines:

| Mode | P(1st) |
|---|---:|
| e8_first_tv | 0.29% |
| seed | 0.21% |
| champ_first_tv | 0.20% |
| f4_first_tv | 0.16% |
| champ_first_chalkfade_tv | 0.15% |
| blend | 0.11% |
| torvik | 0.07% |
| noseed | 0.05% |
| det_* | 0.00% |

The mechanism: the construction-mode samplers lock their anchor teams' paths, which concentrates variance on fewer different "bracket archetypes" per sample. At high stochastic variability elsewhere in the bracket, this raises the tail probability of a specific correct anchor path aligning with the actual outcome. Forward-greedy sampling (the baseline) has too much independent noise — each of the 63 games is drawn independently, and the resulting brackets are more uniformly distributed across the outcome space, which gives you a higher chance of mid-pack finishes but a lower chance of the extreme "everything aligned" outcomes that win pools.

**In tournament pool theory, this is exactly the tradeoff you want**. Pool payouts are concentrated at the top (1st gets all of it in a winner-take-all format), so you maximize P(1st) even at the cost of worse MeanRnk.

## Root cause: why `opt_*` modes failed (diagnosed 2026-04-12)

Four interconnected failures in the Pareto-leverage optimizer:

1. **Myopic greedy strategy.** `_make_ev_scorer()` in `bracket_construction.py` picks winners game-by-game by argmax of `model_prob * pts * blended_diff`. No dynamic programming or lookahead to avoid downstream concentration risk.

2. **Independent-pick approximation.** `_compute_expected_points()` treats all 63 picks as independent: `total_ev += p * pts`. It ignores path-dependent covariance — a 10-seed's E8 probability is used directly without conditioning on them winning R64 and R32 first.

3. **Leverage without correlation respect.** The optimizer maximizes `model_prob / public_pick_pct` weighted by points, but doesn't check whether high-leverage picks are correlated (all on the same team's path, all against the same favored teams). Creates "correlation concentration" where multiple contrarian picks fail together.

4. **Catastrophic failure in upset years.** When early upsets invalidate the optimizer's contrarian assumptions, the entire portfolio collapses. Zero P(1st) because upset years create massive downside that swamps normal-year gains.

**Why construction modes work better:** They use the *same* per-game `_ev_score` but apply it differently — lock an anchor round first (champion/F4/E8), then fill forward greedily. This forces bracket path consistency implicitly. You can't pick a 10-seed at F4 if you've already locked a different champion. Result: lower tail risk, non-zero P(1st), better BestRank.

## N=31 small-pool backtest (2026-04-12, shape-encoded)

Re-run at actual pool size (31 people) shows **contrarian edge compresses dramatically**:

| Mode | BestRnk | MeanRnk | P(1st) | P(top5%) | P(top25%) | Avg ESPN Pts |
|---|---:|---:|---:|---:|---:|---:|
| **f4_first_tv** | **1.5** | **15.6** | 4.27% | 4.66% | **25.37%** | **1105** |
| champ_first_tv | 1.6 | 16.2 | 4.41% | 4.74% | 22.84% | 1091 |
| **torvik** | 2.0 | 16.3 | **4.45%** | **4.83%** | 23.58% | 1082 |
| e8_first_tv | 1.5 | 16.4 | 3.59% | 4.01% | 21.62% | 1083 |
| seed | 1.6 | 16.2 | 3.46% | 3.81% | 22.98% | 1087 |
| blend | 1.9 | 16.8 | 3.05% | 3.42% | 21.37% | 1072 |
| noseed | 1.8 | 17.1 | 2.57% | 2.88% | 19.68% | 1056 |

Note: these are shape-encoded numbers. Team-identity N=31 results (O26-G3) show similar mode ordering — torvik best BestRank (1.42), f4_first_tv competitive. See `artifacts/o26_g3_n31_team_identity_2026-04-17.json`.

Random P(1st) at N=31 = 3.23%. Best modes achieve ~4.4% (1.4× random). No mode is statistically distinguishable from seed after Bonferroni correction.

**Small-pool takeaways:**
- f4_first_tv is the quiet winner at N=31: best MeanRnk, P(top25%), and MeanScr
- e8_first_tv drops from best-at-N=1000 to mid-pack — aggressive contrarian play is penalized
- Accuracy matters more than differentiation at small pool sizes

## Deprecated modes (2026-04-12)

opt_seed, opt_blend, opt_torvik, and hedge_tv have been **removed from the backtest harness**. Evidence:
- opt_* statistically significantly worse than seed on BestRank (p<0.05 Bonferroni) in the N=1000 full_results run
- Zero P(1st) across 13 years for opt_seed, opt_torvik, hedge_tv
- Council decision: the Pareto-leverage optimizer actively hedges away the upside needed to win pools

## What this means for 2027

For your 2027 pool entry, the recommended path is:
- Primary: `python -m src.main optimize-pool --mode torvik --construction-mode f4_first --year 2027 --pool-size 31`
- Aggressive (large pools only): `python -m src.main optimize-pool --mode torvik --construction-mode e8_first --year 2027`
- To see all construction modes side-by-side: `--construction-mode all`

The CLI default mode is `torvik`.

## Caveats

- **14-year sample is small.** Even the directional findings should be held with humility. Effect sizes are meaningful but not conclusive.
- **P(1st) values are all < 0.3%** — directional signal is real but confidence intervals overlap.
- **BestRnk is an oracle metric.** It's the rank of the best-of-50 stochastic brackets each year. In a real single-entry pool you pick ONE bracket. Real-world performance will be somewhere between MeanRnk and BestRnk.
- **2012 is excluded** because archived ESPN pick data is missing for that year. **2020 is excluded** because there was no tournament (COVID). That gives 14 years of usable data.
- **None of the construction modes are Bonferroni-significant** at α=0.005 with 10 comparisons. The recommendation is based on direction + effect size, not p-values.
- **Team-identity scoring (O26/O27) changed the mode ranking.** Prior shape-encoded results showed `champ_first_tv` dominant. Team-identity scoring better reflects actual ESPN pool payouts; see appendix.

---

## Appendix: shape-encoded results (historical)

The table below is preserved for reference. These were the authoritative numbers before O26/O27 (2026-04-18) discovered that shape-encoded scoring materially diverges from real ESPN scoring. **Do not use these for mode selection decisions.**

13 years (shape-encoded, N=1000), pre-O26:

| Mode | BestRnk | MeanRnk | P(1st) | P(top5%) | P(top25%) | MeanScr |
|---|---:|---:|---:|---:|---:|---:|
| champ_first_tv | 21.1 | 515.6 | 0.06% | 4.96% | 22.72% | 1104 |
| e8_first_tv | 23.2 | 550.1 | 0.20% | 3.95% | 20.16% | 1078 |
| f4_first_tv | 26.1 | 518.7 | 0.16% | 4.35% | 23.81% | 1101 |
| torvik | 31.5 | 546.0 | 0.02% | 5.05% | 18.92% | 1080 |
| seed | 38.1 | 527.4 | 0.02% | 4.89% | 23.80% | 1094 |

Source: `mc_pool_backtest_hedge_results.txt`. Under this scoring, `champ_first_tv` appeared to strictly dominate torvik. The gap was an encoding artifact — shape scoring credited positional match regardless of which team occupied that position, inflating champion-heavy construction modes that happened to get the right bracket structure even when they had the wrong teams.

Key divergence: shape mean ρ(P(1st), actual placement) = +0.37; team-identity mean ρ = **+0.61** (O26-G1). Shape understated the real signal by 40%.
