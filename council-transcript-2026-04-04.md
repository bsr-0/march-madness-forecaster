# LLM Council Transcript — Session 6
**Date:** April 4, 2026
**Question:** Validate new public picks data and propose next steps to clean, fix, or fully integrate

---

## Framed Question

We're building a March Madness bracket forecasting system. Five previous council sessions have guided development. The most recent (v5) identified that our opponent model (SEED_PICK_RATES) is broken — it's a stale proxy for actual human pick distributions, and this is the structural defect explaining why our ML model's real Brier score advantage doesn't translate to pool performance.

Council v5's #1 recommendation: "Swap SEED_PICK_RATES for real ESPN aggregate pick percentages and rerun the backtest."

We just committed 19 years (2008-2026) of real ESPN "Who Picked Whom" data from two Kaggle sources. Validation shows:
- All 19 files pass schema validation: consistent structure, values in [0,1], 64 teams each, snake_case names
- The data loader (`historical_picks.py`) already tries real data first, falls back to SEED_PICK_RATES
- The opponent model already blends ESPN picks at 0.6 weight when available
- BUT: `data/kaggle/public_picks.json` (columnar format, 0-100 scale) is an orphaned intermediate — no code consumes it
- Team names in picks files use snake_case (`brigham_young`, `saint_mary_s__ca`) which may not match team IDs used elsewhere in the codebase

The questions:
1. Is this data valid and trustworthy enough to replace SEED_PICK_RATES as the primary opponent model?
2. What data quality issues or gaps need to be addressed first?
3. What's the concrete next step to wire this data into the backtest and test the council v5 hypothesis?
4. Should the orphaned `data/kaggle/public_picks.json` be deleted, converted, or kept?
5. Are there any team-name matching risks between the picks data and the rest of the codebase?

---

## Advisor Responses

### The Contrarian

The "validation" you're celebrating is cosmetic — you confirmed structure and ranges, not accuracy. Cross-reference against ESPN's original source for spot checks. The snake_case matching is worse than acknowledged — `saint_mary_s__ca` has double underscores that are normalization artifacts. Have you actually run a join and counted orphaned rows? The 0.6 blend weight is made up. Concrete next step isn't "run the backtest" — it's: manually verify 5 games from 2023 against ESPN published data, fix the team-name join, count matched vs fallback rows. Then run the backtest.

### The First Principles Thinker

The real question isn't "is this data valid?" It's: does our opponent model accurately predict how humans in our specific pool will pick? ESPN aggregate data is population-level signal; your pool is a sample. ESPN picks are categorically better than SEED_PICK_RATES but the 0.6 blend weight is arbitrary with no empirical basis. Run the backtest with ESPN at 1.0, then at 0.6, and compare. Delete the orphaned file — it's technical debt. On team-name matching: this isn't a risk to assess, it's a guarantee of silent failures. Fix the matching layer before running any backtest or your results are meaningless.

### The Expansionist

The 19-year ESPN dataset isn't just a bug fix — it's a competitive moat. Real pick distributions from millions of humans encode systematic biases (recency bias, brand recognition, regional favoritism) that SEED_PICK_RATES completely flattens. The team name matching problem is actually an opportunity to build a robust normalization layer for any future data source. Don't delete the orphaned public_picks.json — keep it as a validation/fallback signal. The 0.6 blend weight is a guess that can now be optimized empirically across a grid (0.5-1.0). The concrete next step is: swap, normalize names, optimize blend weight, and rerun.

### The Outsider

Schema validation checks structure, not accuracy. A file with two teams swapped would pass your validator. The orphaned `public_picks.json` being generated but unconsumed suggests the pipeline was never completed. Before deleting, check if its data agrees with yearly files — disagreement means a provenance problem. The team-name matching is the actual danger. "May not match" hides bugs. Concrete next step: build a two-column diagnostic — picks file team names vs codebase team IDs, unmatched rows highlighted. Everything else waits on that.

### The Executor

The data is committed. The loader handles it. Stop analyzing and start running. Monday morning: run the backtest and compare against baseline. Run pytest first, then the backtest, then diff metrics. The team-name mismatch is real but don't let it block you — run first, check logs for fallback behavior. Delete the orphaned file today. Don't audit 19 years of data before running. The backtest will surface data problems faster than manual review.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Expansionist
- Response B = The First Principles Thinker
- Response C = The Executor
- Response D = The Contrarian
- Response E = The Outsider

### Review 1
**Strongest: B** — Correctly frames ESPN as population vs pool signal, proposes concrete experiment. **Biggest blind spot: C** — "run first, check logs" is reckless; silent fallbacks corrupt metrics without errors. **All missed:** Temporal validity — 19 years spans eras where picking behavior shifted dramatically (smartphones, analytics culture, COVID bubble). Should recent years be weighted more heavily?

### Review 2
**Strongest: B** — Combines rigor with specificity. **Biggest blind spot: C** — Silent fallback means backtest appears to complete with wrong data while looking plausible. **All missed:** March Madness expanded 64→68 in 2011. ESPN methodology and sample sizes changed. Naive concatenation assumes stationarity that doesn't hold.

### Review 3
**Strongest: B** — Only response with unambiguous action ordering. **Biggest blind spot: C** — Silent fallbacks corrupt results. **All missed:** Coverage drift over time. Match rates in early years may be terrible. Per-year matched team count is a prerequisite check.

### Review 4
**Strongest: B** — Distinguishes right question from work queue. **Biggest blind spot: C** — Appears to complete with corrupted inputs. **All missed:** ESPN picks from 2006-2019 collected in different information environment. Time-weighted or recency-biased blend warranted.

### Review 5
**Strongest: B** — Correctly frames name matching as guarantee not risk. **Biggest blind spot: C** — Mistakes "won't crash" for "works." **All missed:** Temporal validity — 2010 pick distribution may not generalize to 2024. Test for distributional drift.

---

## Chairman Synthesis

### Where the Council Agrees

**Team-name matching is not a risk to assess — it is a confirmed bug to fix.** Every advisor touched this, and the peer reviews hammered it further. `saint_mary_s__ca` with double underscores is a normalization artifact, not an edge case. Silent fallback to SEED_PICK_RATES means the backtest will appear to complete successfully while producing meaningless results. You won't get an error. You'll get a number that looks plausible and is wrong. This is the consensus position: fix matching before running anything.

**The 0.6 blend weight is made up.** No advisor defended it. It was chosen without empirical basis and should be treated as a parameter to optimize, not a constant to trust.

**Schema validation proved structure, not accuracy.** Values in [0,1], 64 teams, snake_case keys — none of that confirms the data represents what ESPN actually published. The validation passed the easy test.

### Where the Council Clashes

**Run now vs. fix first.** The Executor says run the backtest Monday, let logs surface data problems. Every other advisor, and all five peer reviews, say this is reckless. The disagreement is real but the resolution is clear: the Executor is conflating "won't crash" with "works." A silent fallback that corrupts 40% of your joins produces a metric, not a result. The Executor wins on velocity; everyone else wins on correctness.

**Keep or delete `public_picks.json`.** The Expansionist says keep it as a validation dataset. The Executor says delete it today. The Outsider says check if it agrees with the yearly files before deciding. This is a genuine call. But the Outsider's framing is best: if the columnar file and the yearly files disagree on the same games, you have a provenance problem that invalidates everything. If they agree, you have a redundant file. Check first, then decide.

### Blind Spots the Council Caught

**Temporal validity — every peer review flagged this, no advisor raised it independently.** 19 years of data is not a homogeneous dataset. ESPN's methodology changed. The tournament expanded from 64 to 68 teams in 2011. Smartphone adoption, analytics culture, and COVID all shifted how casual fans pick. A naive concatenation that treats 2008 data as equivalent to 2024 data assumes stationarity that doesn't hold. You need per-year match rates and should consider whether early years should be down-weighted or excluded entirely.

**Coverage drift is a prerequisite check, not a backtest finding.** If 2008-2012 data matches only 30% of your team IDs, those years are effectively SEED_PICK_RATES anyway — you just don't know it. The match rate by year needs to be a diagnostic output before any backtest runs.

### The Recommendation

Fix the team-name matching layer first. Build a two-column diagnostic: picks file team names on the left, codebase team IDs on the right, unmatched rows flagged. Run it across all 19 years and compute per-year match rates. Any year below 80% match rate should be excluded from the initial backtest or flagged with an asterisk. Once matching is verified, check whether `public_picks.json` agrees with the yearly files — if it does, delete it; if it disagrees, stop and investigate provenance. Then run the backtest with ESPN weight at 1.0 and again at 0.6, compare the metrics, and let the data choose the blend weight. Do not treat pre-2011 data as equivalent to post-2011 data without first confirming the distributions are compatible.

The data is not "valid enough to run." It is valid enough to be worth fixing properly before running. The difference matters because a backtest with silent fallbacks gives you false confidence, which is worse than no backtest.

### The One Thing to Do First

Write a diagnostic script that joins every team name in the picks files against the codebase team IDs, prints unmatched names with their year and file of origin, and outputs a per-year match rate. Run it. That single output will tell you whether you have a minor cleanup task or a structural data problem — and nothing else on the roadmap is meaningful until you know which one it is.
