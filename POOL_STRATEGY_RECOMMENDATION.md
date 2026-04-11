Before committing to a strategy, let me be honest about a nuance I glossed over. "Win the pool" can mean two different things in this data, and they point to different modes.

## The P(1st) nuance

The mode with the highest measured P(1st) is actually **blend** (0.0008), not torvik. Here's why I didn't lead with it:

**Year-by-year P(1st) breakdown** (only non-zero cells):

| Year | Mode | P(1st) |
|---|---|---:|
| 2015 | blend | **0.010** |
| 2016 | opt_torvik | 0.001 |
| 2018 | torvik | 0.003 |
| 2021 | seed | 0.002 |
| 2023 | noseed | 0.003 |

Blend's aggregate P(1st) of 0.0008 is **dominated by 2015** — one single year carried the whole number. That's not a reliable signal; it's one lucky bracket draw.

With 13 years × 50 brackets × 50 repeats = 32,500 bracket-trials per mode, the absolute first-place counts are:
- blend: ~26 wins (but ~20 of them from 2015)
- noseed: ~10
- torvik: ~6
- seed: ~3
- all opt_*: ~0

These are small counts. Confidence intervals on P(1st) overlap substantially. **You cannot reliably pick a winner on P(1st) alone with 13 years of data.**

## The stable signal

The metrics that *are* statistically robust across 13 years are BestRnk and P(top5%):

| Mode | BestRnk | P(top5%) | Interpretation |
|---|---:|---:|---|
| **torvik** | **31.5** | **5.06%** | Best-case and top-5% both #1 |
| seed | 38.1 | 4.90% | Close #2 |
| noseed | 39.5 | 2.87% | Worse top-5% |
| blend | 43.2 | 3.10% | Worse top-5% |

Torvik lands its best bracket at rank ~32 out of 1000 on average (top 3.2%). That's the most consistent top-tier performance in the dataset, and it's a real signal, not one-year luck.

## Honest reading of "win the pool"

If "win the pool" means **literal P(1st)** → the data is too noisy to distinguish. Blend's apparent edge is one year.

If "win the pool" means **maximize chance of being in contention for 1st** → torvik is the defensible winner. Best BestRnk, best P(top5%), simplest model.

**Going with torvik.** Here's why it's the right call for your goal:

1. **Statistical reliability**: torvik's win on BestRnk and P(top5%) holds across years, not just one lucky season
2. **Data source**: uses the same Torvik four-factors we already verified as direct-scrape, through a simpler Log5 + MC path with no ML overfitting risk
3. **Failure mode is benign**: if torvik is wrong in any given year, it's wrong by landing at rank ~100 instead of rank ~30, not catastrophically
4. **It matches the input data's quality**: Torvik barthag is the most reliable pre-tournament rating available; using it directly preserves that signal

## What this means for 2026

For your 2026 pool entry, the recommended path is:
- Use the `torvik` mode (barthag + Log5 + MC, no ML ensemble, no leverage optimizer)
- This is what `scripts/mc_pool_backtest.py` runs when you pick `torvik` mode
- It's not currently exposed through the CLI `pool_cmds.py` path — that only supports seed/noseed/blend

## Next steps — your call

1. **Wire up `torvik` mode in `src/cli/pool_cmds.py`** so `optimize-pool --mode torvik` works end-to-end and produces a single submittable bracket for 2026
2. **Commit the Phase 1 + Phase 2 changes** currently in the working tree (symmetric aug + mc_pool_backtest fixes) before doing anything else
3. **Both** — commit first, then add the CLI wiring in a separate commit

Which do you want?
