# LLM Council Transcript — Session #10
**Date:** April 4, 2026
**Topic:** Backtest Validity Review — Optimizer Integration Results

---

## Original Question

> Council this: review the notes and results of the most recent test. Are there any bugs or missing steps to increase the validity of this test? Are there data sources from the kaggle zip file that can be included to help rigorously test this? What does the council recommend given these results?

---

## Framed Question

**Decision: Is the optimized mode's mean_rank=396 a real signal or a measurement artifact? What bugs need fixing, what data should be added, and what's the next step?**

The PoolOptimizer was wired into a 17-year historical backtest (2008-2025, ESPN opponents, 1000-person pool). Two new modes were added:
- "optimized": 5 Pareto frontier brackets from leverage analysis (deterministic)
- "leveraged": 50 stochastic brackets sampled from tilted probabilities

Results (mean rank out of 1000, lower=better):
  seed=532, noseed=539, blend=546, leveraged=620, optimized=396
  Optimized beats seed 13/16 years (p=0.013), P(top5%)=22.4% vs seed P(top5%)=5.1%

Bugs found in audit:
1. Unfair sample size: optimized=5 brackets vs others=50
2. Optimized uses blend_rp — may not maximize leverage signal
3. Opponent brackets use seed_pw for game resolution even with ESPN picks
4. 2011 skipped due to incomplete TeamMetadata
5. Leveraged mode tilt_strength=1.0 may be too aggressive

Unused data in repo: 182 Massey rating systems, KenPom, 538, Evan Miya, AP Poll, Heat Check, Torvik shooting, roster data. No betting lines.

---

## Advisor Responses

### The Contrarian

These results are garbage until Bug #1 is fixed, full stop.

You're comparing the best bracket out of 5 against the best bracket out of 50. That's not a methodology difference — that's a sampling artifact masquerading as signal. "Optimized" wins by having fewer competitors in its own cohort. Fix the sample size to 50 vs 50 before drawing any conclusions. The p=0.013 and the P(top5%) numbers are meaningless as stated.

Bug #3 is quietly poisonous. If ESPN opponent picks drive the bracket but seed-based probabilities resolve the games, you've built a chimera. The counterfactual performance you're measuring doesn't correspond to any real-world scenario. The entire backtest is measuring something that doesn't exist.

The 2011 gap isn't a minor footnote — it's a missing data point in a 16-year sample. That's a 6% hole in your validation set, and you don't know what you're missing. If 2011 was an upset-heavy year (it was — VCU Final Four), your optimizer may look artificially better than it deserves.

The Kaggle data situation reveals a deeper problem: you're optimizing bracket selection without validating whether your underlying win probabilities are any good. KenPom, Torvik, and betting lines exist precisely because raw seed-based priors are weak predictors. You're polishing bracket selection on top of a shaky probability foundation.

What to do next: Fix Bug #1 first. Run 50 optimized brackets. If the signal collapses, you've learned something important and saved yourself weeks of work on a false premise. Don't add more data sources until you know whether the core result survives basic methodological scrutiny.

### The First Principles Thinker

What are we actually trying to solve? Not "did the optimizer beat seed?" That's the wrong question. We're trying to answer: is rank 396 a real signal or a measurement artifact?

Strip away everything and the answer is: we don't know yet, because bug #1 invalidates the headline result.

Comparing best-of-5 against best-of-50 is not a backtest — it's a selection bias demonstration. Fix this first, before anything else. Run optimized at n=50 and leveraged at n=5. If optimized at n=50 still beats seed at n=50, then you have a signal. Until then, rank 396 is noise wearing a lab coat.

Bug #3 is the second critical flaw. You're resolving games with seed-based probabilities while populating opponent brackets from ESPN pick distributions. These are from different probability universes. The simulation is internally inconsistent — your opponents are making picks based on a different reality than the games being played.

Fix those two bugs before touching any new data source. Adding KenPom or Evan Miya on top of a broken simulator just adds precision to an invalid measurement.

On the 182 Massey systems: the real value isn't averaging them — it's finding which individual systems have historically predicted upset brackets better than the field. That's a solvable retrieval problem.

The target of median rank < 300 is achievable only if the optimizer is actually exploiting real field mistakes. You can't know if it is until the comparison is apples-to-apples.

Next step is exactly one thing: equalize sample sizes, re-run the backtest, report the corrected numbers. Everything else is premature.

### The Expansionist

The signal is real and you're underselling it.

Mean rank 396 with a broken sample size comparison that disadvantages optimized mode. Fix the sample size parity and that number almost certainly drops further. You have a p=0.013 result that's conservative. The actual effect is stronger than what you measured.

Here's what nobody is saying loudly enough: a 4.4x improvement in P(top5%) — from 5.1% to 22.4% — is not a marginal optimization. That's a category change in how this pool gets played.

The adjacent opportunity hiding in plain sight: You're sitting on 182 Massey rating systems and not using them. The real leverage signal isn't in any single predictor — it's in disagreement between systems. Where KenPom says chalk and Torvik says upset is where the field is uniformly wrong. That's free leverage the optimizer can't see yet because you haven't fed it.

What to do next, ranked by impact:

1. Fix sample size parity immediately
2. Wire in system disagreement as an explicit leverage signal — compute variance across the 182 Massey systems per matchup
3. Add betting line implied probabilities as a consensus proxy
4. Tune tilt_strength with a grid search across 0.3-1.5 rather than defaulting to 1.0

The threshold is 300 and you're at 396 with a broken experiment. The ceiling here is genuinely unknown. Run the clean version before concluding anything about limits.

### The Outsider

The optimized mode's 396 rank is almost certainly an artifact, not a discovery.

Start with the most obvious problem: you're comparing best-of-5 to best-of-50. If you run 50 random brackets and keep the best finisher, you'll outperform a baseline just through selection pressure. The "optimized" mode isn't demonstrating superior strategy — it's demonstrating that fewer brackets means less noise in how you measure it. These aren't comparable.

The p=0.013 result is meaningless until you fix this. Statistical significance on a flawed comparison is still a flawed comparison.

The 2011 skip is a silent validity hole. You can't claim "beats baseline 13/16 years" if the denominator is actually 16 but you only have 15 data points. That's either 13/15 or you're counting a missing year as a win. Either way, the number is wrong.

The model mismatch — public behavior modeled from ESPN data, actual outcomes modeled from seeds — means the optimizer is solving the wrong game. It's finding picks the public undervalues according to ESPN, then evaluating those picks against a seed-based reality. If the seed model and the ESPN model disagree on upset probability, the optimizer is exploiting a spreadsheet gap, not a real edge.

What to do next: Fix the bracket count first. Run optimized at 50 brackets. If the edge survives, it's real. If it collapses, you've saved yourself from publishing a bug as a finding.

The unused rating systems are irrelevant until the mechanics are sound. Don't add complexity to a broken foundation.

### The Executor

The 96-position gap is real but the test is broken. Fix the test before chasing new data.

Monday morning, in order:

Hour 1 — Fix sample size (Bug #1). Change optimized brackets from 5 to 50. This single change could move the mean_rank number significantly.

Hour 2 — Fix model input (Bug #2). Swap blend_rp for seed_rp in the optimized path. The diagnosis showed seed_rp has higher leverage values.

Hour 3 — Re-run the backtest. Get the new mean_rank with both fixes applied. This is your actual baseline.

Hour 4 — Evaluate the gap. If fixes close you to ~340-350, you have a clear path: add KenPom or 538 ratings. If you're still above 380, the model itself is the problem, not the data.

Don't touch Bug #3 or #4 yet. Opponent inconsistency and 2011 skip are real issues but they're not blocking. Fix the comparison integrity first.

Data upgrades come last. KenPom and 538 are one-day integrations. Adding them before you have a clean test just adds noise. Get the clean number first, then layer in features.

Estimated time to a valid baseline: 6 hours.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Review 1
**Strongest: E** — Only response that translates diagnosis into a concrete, time-boxed execution plan. **Blind spot: C** — Building on unvalidated result, speculation presented as analysis. **All missed:** Overfitting risk — optimizer may have implicitly fit to 17 years of ESPN data. Need held-out validation set (most recent 3-4 years).

### Review 2
**Strongest: E** — Prescribes, not just diagnoses. **Blind spot: C** — Treats 396 as validated signal, starts optimizing on top of potentially corrupt data. **All missed:** Pool structure sensitivity — optimal strategy for 50-person pool differs from 1000-person pool. Target pool context matters.

### Review 3
**Strongest: E.** **Blind spot: C** — Premature optimization. **All missed:** 5 deterministic Pareto brackets vs 50 stochastic aren't exchangeable even at equal counts. Variance structure matters — what's the distribution of optimized bracket scores?

### Review 4
**Strongest: E.** **Blind spot: C.** **All missed:** Pareto selection may collapse bracket variance and structurally cap upside. If scoring rewards tail events, deterministic brackets may systematically underperform stochastic ones regardless of count.

### Review 5
**Strongest: E.** **Blind spot: C.** **All missed:** Five Pareto brackets are correlated (same frontier), not independent samples. Comparing correlated vs independent samples isn't just a count problem — it's a variance problem.

---

## Chairman Synthesis

### Where the Council Agrees

The headline result is invalid. Every advisor — including the Expansionist — acknowledges that comparing 5 optimized Pareto brackets against 50 stochastically generated brackets makes the rank-396 finding uninterpretable. The peer reviews reinforce this from a different angle: the 5 brackets are not independent samples, they are correlated outputs from the same Pareto frontier. This is not a minor methodological footnote. It is the load-bearing flaw. All five advisors converge on the Executor's plan as the most actionable response.

### Where the Council Clashes

The Expansionist claims the signal is real and likely to improve after fixing sample parity. The Contrarian, First Principles Thinker, and Outsider treat the signal as entirely artifact-driven until proven otherwise. This is not a philosophical dispute — it is a testable hypothesis. The backtest re-run at equal sample sizes will resolve it. The Expansionist's enthusiasm for wiring in Massey disagreement, betting lines, and grid-search tuning is premature and potentially counterproductive if the corrected backtest shows the baseline is mediocre. Do not optimize a broken instrument.

### Blind Spots the Council Caught

Four critical gaps the advisors collectively surfaced:

1. **Overfitting to the historical window.** The optimizer has implicitly seen all 17 years of ESPN pool data. There is no held-out validation set. The most recent 3-4 years should be withheld before any further tuning or data additions.

2. **Pareto bracket variance collapse.** Deterministic Pareto selection may structurally suppress bracket diversity, capping upside in scoring systems that reward tail outcomes. This is a design flaw, not just a count problem — equal counts won't fully fix it.

3. **Pool size sensitivity.** The optimal bracket strategy for a 50-person pool differs from a 1000-person pool. The backtest has no explicit target pool context, which means the optimizer may be solving for the wrong objective.

4. **The 2011 gap is not cosmetic.** Skipping a year silently changes the denominator. "13/16 years" framed as "14/15 years" is a quiet validity error.

### The Recommendation

Execute the Executor's plan with one addition: after re-running at equal sample sizes, immediately carve out the last 3 years as a held-out test set before touching any data sources or tuning parameters. Do not add KenPom, betting lines, or Massey disagreement signals until you have a validated baseline on unseen years. Expansion before validation is how you build a system that backtests beautifully and performs randomly live.

Fix Bug #2 (sample size) and Bug #4 (2011 skip) this week. Defer Bug #3 (ESPN picks / seed-based resolution mismatch) until the baseline is valid — it matters, but it does not block the corrected headline number.

### The One Thing to Do First

Equalize the bracket sample sizes: 50 stochastic vs 50 optimized, same generation procedure except for the optimization step. Re-run the backtest. Report the corrected rank. Everything else is speculation until that number exists.
