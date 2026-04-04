# LLM Council Transcript — Session #11
**Date:** April 4, 2026
**Topic:** Path Selection — Model Improvement vs Diversity Fix vs Hybrid

---

## Original Question

> Ask council which path to pursue given the latest results — Path A, Path B or another path entirely?

---

## Framed Question

**Decision: After equalizing bracket sample sizes, the optimizer's mean_rank improvement is not statistically significant (p=0.38), but its best-bracket improvement is highly significant (p=0.002). Which path should we pursue next: (A) improve the underlying probability model with better data sources, (B) fix the bracket diversity collapse, or (C) hybrid — and in what order?**

Context:
- March Madness pool optimizer backtest, 17 years (2008-2025), 1000-person ESPN pool
- Current model uses only seed-based win probabilities — the weakest possible predictor
- Optimizer generates Pareto brackets across a risk spectrum (chalk to contrarian)
- 50 Pareto risk levels collapse to ~8-14 unique brackets due to deterministic pick decisions
- Unused data in repo: 182 Massey rating systems, KenPom, 538, Evan Miya, Torvik, AP Poll — none wired in
- No betting line data available yet
- Leveraged mode (stochastic tilt) performed WORSE than baseline (mean_rank=620 vs 532)
- The Executor's prior decision tree: "If still above 380, the model is the problem" — both opt modes are above 380
- Council #10 recommended held-out validation (last 3 years) before any tuning

Results after equalization:
| Mode | MeanRnk | BestRnk | P(top5%) | Unique brackets |
|------|---------|---------|----------|----------------|
| seed | 532 | 32.5 | 5.1% | 50 stochastic |
| opt_seed | 491 | 218 | 4.5% | ~10 unique |
| opt_blend | 460 | 159 | 8.3% | ~12 unique |

What's at stake: Whether to invest engineering time in data integration (days of work) or algorithmic diversity (hours of work), and whether the current results justify continued investment at all.

---

## Advisor Responses

### The Contrarian

The p=0.002 on best-bracket is a trap. You have 17 years of data and you're running a test where 1000-person pool outcomes have enormous variance year-to-year. Best-bracket across 17 trials is your single best outlier run, not a distribution. A result that good on a small-N extreme statistic should make you suspicious, not excited.

Here's what's actually happening: opt_blend is producing ~12 unique brackets. You're effectively comparing 12 carefully constructed brackets against 50 random ones. Of course the best of 12 curated picks beats the best of 50 random picks — you've introduced selection bias by collapsing diversity. The "highly significant" best-bracket result may be an artifact of how the metric interacts with bracket count, not evidence of model skill.

The mean_rank result tells the real story. p=0.38. Not significant. You cannot explain away the primary performance metric with "but look at the best case." Mean rank is what wins pool seasons over time. The Executor's prior threshold was correct: both modes are above 380, the model is the problem.

The held-out validation Council #10 recommended hasn't happened yet. You're about to make an infrastructure investment decision — days of engineering work — on in-sample results with a non-significant primary metric and a suspicious secondary metric.

The correct order: run the held-out validation on the last 3 years first, before touching anything. If opt_blend's mean_rank holds below seed on out-of-sample data, then you have evidence of something real. If it doesn't, you've saved yourself from building on sand. Diversity fixes are hours of work — do those after validation, not before. Data integration comes last.

### The First Principles Thinker

The question assumes "which path" is the right frame. It isn't.

Strip back to what actually matters: you are trying to win a 1000-person pool. Not improve mean rank. Not achieve statistical significance on mean rank. Win, or finish top 5%.

Look at the data from that lens only:

- opt_blend hits top-5% at 8.3% vs seed's 5.1%. That's a 63% lift on the only metric that pays out.
- opt_blend best rank is 159 vs 218 for opt_seed. The optimizer is doing real work on tail outcomes.
- Mean rank improvement being insignificant is irrelevant. You don't get paid for mean rank. A pool optimizer that sacrifices median performance for elite-tail performance is doing exactly what it should.

The Executor's "above 380 means model is the problem" heuristic is built on mean rank intuition. It's the wrong intuition for winner-take-most pools.

The real question you should be asking: Is the optimizer successfully manufacturing asymmetric upside? The answer appears to be yes — 8.3% top-5% rate is not noise.

Which means the actual next question is: does better probability data amplify the asymmetry, or just shift the mean? If it only improves mean rank, it barely matters. You want data sources that sharpen your ability to identify upset-prone games — the specific nodes where contrarian picks pay off.

So the order is:
1. First, validate that 8.3% figure on held-out years (Council #10 was right).
2. Then, wire in upset-predictive features — not just accuracy-maximizing ones.

The diversity collapse is a symptom, not the disease. Fix the probability model with the right objective.

### The Expansionist

The best-bracket signal is a treasure map and everyone's staring at the compass instead.

p=0.002 on best-bracket improvement is not a noise artifact — it's proof the optimizer is finding genuinely superior bracket structures in the tail. The mean_rank flatness is a diversity problem masking a model that's already doing real work. You're sitting on a system that demonstrably knows how to build winning brackets. You just can't generate enough different ones to hit that ceiling consistently.

Here's what everyone's undervaluing: opt_blend already beats seed on P(top5%) — 8.3% vs 5.1% — with zero of the 182 Massey systems wired in. That's the buried lede. You have an optimizer that's outperforming on the metric that actually wins pools (landing in the money), and it's running on the weakest possible inputs.

The upside scenario if this works: wire in KenPom + Torvik (two days of work), fix diversity collapse so you're getting 40+ meaningfully different brackets instead of 12, and that 8.3% P(top5%) could plausibly double. You're not tweaking a marginal system — you're turbocharging something that's already directionally correct.

Recommended order: diversity fix first (hours, not days), then data integration. Diversity is the multiplier on everything downstream. Better probabilities fed into a collapsed optimizer still give you 12 brackets. Better probabilities fed into a diverse optimizer gives you 40+ brackets sampling the full Pareto frontier.

The held-out validation caution is reasonable housekeeping, but don't let it slow the core build. The signal is real. The upside is large. Move.

### The Outsider

The table is telling me something that I think you might be explaining away.

The optimizer's mean_rank is statistically noise. Its best_rank improvement is real. Those are two different things happening simultaneously, and they're in conflict with each other.

Here's what I see as an outsider: you built a machine that finds one spectacular bracket at the cost of making the other brackets worse. The mean went from 532 to 491 — that's a modest gain. But the best went from 32.5 to 218 — that's catastrophically worse. You are trading average performance for... also worse peak performance? The optimizer is not beating the baseline on either metric that matters for a pool.

Wait. I'm re-reading. Is lower rank better or worse here? If you're ranked 32 out of 1000, that's great. If 218 is worse than 32, then the optimizer makes your best case four times worse while only modestly improving your average. That's a broken tool, not a promising one.

Before investing days on data or hours on diversity, I'd want answered: what does winning a 1000-person pool actually require? If the answer is "one bracket needs to finish top 5," then your baseline seed model already does that at 5.1% and the optimizer drops it to 4.5%.

The optimizer appears to be solving a problem that doesn't exist while worsening the metric you actually care about.

The honest question before any next step: what is this optimizer for?

### The Executor

The decision tree already gave us the answer: both opt modes are above 380, so the model is the problem. Stop re-litigating that.

But there's a sequencing trap here. If you wire in 182 rating systems before fixing diversity collapse, you'll get 8-14 unique brackets generated from a slightly-better model. The diversity problem will still eat your alpha. You'll have done days of work and won't know if it helped because the brackets are still basically identical.

Here's the actual Monday morning order:

Step 1 (today, 2-3 hours): Fix diversity collapse. The stochastic tilt already exists, it just performed worse — which means the tilt implementation is wrong, not the concept. Debug why leveraged mode hurt mean_rank. If you can't generate 30+ meaningfully different brackets from the same probabilities, better probabilities won't matter.

Step 2 (this week): Wire in KenPom or Torvik — one source, not all 182. Pick the one with the longest historical coverage. Validate on held-out 2023-2025 before touching the optimizer.

Step 3: Re-run the backtest with better probs AND working diversity. Now you can actually read the signal.

The p=0.002 on best-bracket is real but narrow — it means the optimizer finds good solutions occasionally, not reliably. That's a diversity problem disguised as a model problem.

Don't run the council again until Step 1 is done. The diversity fix is 2-3 hours of work. If it doesn't improve bracket count meaningfully, the optimizer architecture is broken and that changes everything downstream.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Review 1
**Strongest: E.** Concrete, sequenced action plan with time estimates and testable hypothesis (stochastic tilt is implemented wrong, not conceptually wrong), clear stopping condition. **Blind spot: D.** Misreads the table, treats lower rank as worse, confusion derails analysis. **All missed:** The leveraged/stochastic mode performing worse (mean_rank 620 vs 532) is the most diagnostic result in the dataset and every response either ignored it or mentioned it in passing. That result is a direct falsification of the diversity hypothesis — adding randomness made things worse.

### Review 2
**Strongest: E.** Correctly identifies sequencing trap, diagnoses tilt failure as informative debugging clue. **Blind spot: C.** Calls p=0.002 "proof" and waves away held-out validation as "reasonable housekeeping." **All missed:** None asked whether 8-14 unique brackets is actually a problem given pool structure. You typically submit one bracket. If the optimizer is a single-bracket picker, "diversity collapse" is irrelevant.

### Review 3
**Strongest: B.** Correctly reframes objective — pool winners care about P(top5%), not mean rank. The 63% lift is real signal. **Blind spot: C.** Treats p=0.002 as validation when it's likely selection bias. **All missed:** The leveraged mode regression was never explained or resolved. Before adding data or fixing diversity, diagnose why adding variance hurt.

### Review 4
**Strongest: B.** Correctly identifies decision-relevant metric (P(top5%) pays out; mean rank doesn't), maintains epistemic discipline by insisting on held-out validation. **Blind spot: C.** Overstatements on 17 data points, waves away validation. **All missed:** Leveraged mode failure is undiagnosed and nobody flags it as a blocker. If tilt logic is broken, fixing diversity or adding data both build on a flawed mechanism.

### Review 5
**Strongest: B.** The 63% lift on P(top5%) is the actual signal. **Blind spot: C.** Treats p=0.002 as validation, wants to move fast on unvalidated results. **All missed:** The leveraged mode regression was never explained. If stochastic tilt actively hurts performance, that's fundamental — not a footnote.

---

## Chairman Synthesis

### Where the Council Agrees

The held-out validation must happen. A, B, C, D, and the peer reviews all point back to Council #10's recommendation. You are making an infrastructure investment decision on in-sample results. Every advisor who called for moving fast still acknowledged this as "reasonable housekeeping." It isn't housekeeping — it's the foundation. Before you wire in KenPom or fix diversity, you need to know whether the optimizer's gains survive contact with data it hasn't seen.

The diversity collapse is a real constraint. ~12 unique brackets from 50 Pareto levels is broken machinery. Whether you're submitting one bracket or many, a Pareto frontier that collapses to 12 deterministic outputs means your risk spectrum isn't actually spanning the space it claims to. The Executor, the Expansionist, and the peer reviews all agree on this.

Better probability inputs are the ceiling, not the floor. The seed-only model is the weakest possible predictor. Every advisor agrees you're leaving signal on the table with 182 Massey systems sitting unused. The disagreement is only about sequencing.

### Where the Council Clashes

**Is the 8.3% P(top5%) a real signal or a sample artifact?**

The First Principles Thinker and the Expansionist say yes — 63% lift on the metric that actually pays out, statistically distinguishable from noise. The Contrarian says the sample is too small (17 years) to trust a tail metric, and the best-bracket stat may interact badly with bracket count differences. The peer reviews lean Contrarian here: all three that addressed it flagged the p=0.002 as potentially selection-biased by the diversity collapse itself. This is an unresolved tension. The council cannot give you a clean answer because the data is genuinely ambiguous at N=17.

**Mean rank vs. P(top5%) as the decision metric.**

The Contrarian says mean rank is what wins over time. The First Principles Thinker says P(top5%) is what actually pays out and the optimizer is explicitly designed for tail performance. Both positions are coherent. This is a genuine disagreement about pool strategy, not a resolvable empirical question — it depends on whether you're entering one bracket or many, and whether the pool pays top-5% or winner-takes-all.

### Blind Spots the Council Caught

1. **The leveraged mode regression is undiagnosed and nobody treated it as a blocker.** Mean rank 620 vs. 532 when you add stochastic tilt. That's not a footnote — that is direct empirical evidence that adding randomness to the current system makes it worse, not better. Every advisor who argued "fix diversity collapse" was arguing for more randomness in bracket generation. But the one experiment you have on adding variance to this optimizer produced a significant negative result. That falsifies the diversity hypothesis in its naive form. You cannot just "fix the tilt implementation" and wave this away — you need to understand the mechanism before building on it.

2. **The single-bracket submission question was never asked.** Peer Review 2 raised this. If you submit one bracket per year, "12 unique brackets" is irrelevant — you pick the best one and move on. The diversity collapse only matters if you're submitting multiple brackets or using the Pareto spread to inform a single pick.

3. **The table inversion (Outsider's confusion) contains a real signal.** The optimizer's best rank is 218, vs. seed's 32. In the years where the baseline got lucky and placed in the top 35, the optimizer was not in those years. Worth checking: is the optimizer improving the floor and ceiling simultaneously, or trading one for the other year-by-year?

### The Recommendation

**Path C, but not hybrid in the sense of doing two things at once. Hybrid in sequence, gated by diagnosis.**

The sequencing is: diagnose the leveraged mode failure first, validate on held-out data second, then and only then decide whether to fix diversity or wire in data.

The leveraged mode result is a controlled experiment embedded in your backtest. You ran a version of the optimizer with more stochastic variance, and it got worse by a meaningful margin. Before you invest hours fixing diversity collapse or days wiring in KenPom, you need a theory for why that happened that survives scrutiny. If the answer is "the tilt was implemented incorrectly," prove it by finding the bug — don't assume it.

Once you have that diagnosis, run held-out validation on 2023-2025. If opt_blend's P(top5%) advantage holds on out-of-sample years, you have real signal worth investing in. If it evaporates, the Contrarian was right and you've saved yourself weeks.

If held-out validation passes, the order is diversity fix before data integration — the Executor's sequencing is correct. Better probabilities fed into a collapsed optimizer is wasted work. But do not touch diversity until you understand why adding variance hurt.

Do not run all 182 Massey systems. One source — KenPom or Torvik, whichever has the deepest historical coverage — and validate it in isolation before blending.

### The One Thing to Do First

Pull the leveraged mode logs and write down, in one paragraph, exactly why mean_rank went from 532 to 620 when you added stochastic tilt. Not a hypothesis — an explanation traceable to specific bracket decisions in specific years. If you cannot write that paragraph, you do not yet understand what your optimizer is doing, and every path forward is speculation.
