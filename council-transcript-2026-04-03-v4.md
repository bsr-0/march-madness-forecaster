# LLM Council Transcript — Session 4
**Date:** April 3, 2026
**Question:** Next steps after MC pool backtest results

---

## Framed Question

We have a March Madness bracket forecasting system with three modes: `seed` (historical seed win rates), `noseed` (ML model trained on team features), and `blend` (weighted combination). Prior council sessions established that P(rank=1) — probability of finishing first in a bracket pool — is the correct metric, not Brier score or EV-edge. We built and ran the MC pool simulator. Here are the results:

**Brier Score (prediction accuracy):** Noseed is significantly better (p=0.0006). BSS=+0.066 across all rounds, with largest advantages at F4 (+0.138) and E8 (+0.081).

**P(rank=1) via MC simulation (1000-person pool, 17 years, 50 repeats):**
- Seed: mean rank 211, P(1st)=0.054, P(top5%)=0.41
- Noseed: mean rank 283, P(1st)=0.054, P(top5%)=0.22
- Blend: mean rank 240, P(1st)=0.057, P(top5%)=0.36
- No statistical significance: paired t-test p=0.41, noseed beats seed only 5/17 years
- Noseed wins BIG in some years (2008: rank 9 vs 495, 2017: rank 45 vs 566) but loses catastrophically in others (2012: rank 699 vs 15, 2023: rank 676 vs 18)

**The paradox:** Noseed has significantly better prediction accuracy (Brier), but WORSE pool performance (mean rank). The council predicted Brier and P(rank=1) would diverge — and they did, dramatically.

**Current infrastructure:** The opponent bracket sampler uses `SEED_PICK_RATES` (seed-based pick frequencies with chalk bias). Validated with 10 statistical tests. The backtest uses deterministic model brackets (argmax of round probabilities).

**What should we do next?**

---

## Advisor Responses

### The Contrarian

The backtest is broken and the options list doesn't fix it.

Deterministic brackets are the core defect. When you argmax every round, you produce one bracket that never upsets anyone. Real pool entrants pick upsets probabilistically. Your "noseed" model is being evaluated as if it always picks the highest-probability outcome, which means it clusters with the crowd on favorites and gets massacred when underdogs win. The 2012 and 2023 disasters aren't evidence that noseed fails at pool play — they're evidence that argmax brackets are a terrible representation of how anyone actually enters a pool.

Before debating options 1 through 4, you need to answer: what does a stochastic noseed bracket actually score? Option 2 isn't a direction to consider, it's a prerequisite for any valid comparison. Every result in this summary is potentially garbage until that's done.

The second problem: 17 years is not enough data to distinguish signal from noise at P(rank=1) resolution, especially when two catastrophic years (2012, 2023) dominate the variance. Your p=0.41 means nothing except "we can't tell." The confidence intervals on those P(1st) numbers are enormous.

Option 3 (pool-aware optimizer) is the right long-term direction but you're trying to optimize a signal you haven't cleanly isolated yet.

Fix the deterministic bracket problem first. Run stochastic noseed brackets through the same simulator. If noseed still underperforms after that fix, then the conversation about conditional switching or opponent modeling becomes meaningful. Right now you're debating what to do next when you don't know what you actually measured.

### The First Principles Thinker

The surface question is "which mode wins pools?" But the real question is: **why does better prediction accuracy *hurt* pool performance?**

This isn't a paradox — it's a measurement problem. Your backtest uses deterministic brackets (argmax). That means noseed always picks the same bracket. You're not simulating a forecasting agent competing in a pool; you're simulating a static document competing in a pool. The stochasticity of opponent behavior is modeled, but your own strategy has zero variance.

In a 1000-person pool, the winner isn't the most accurate predictor — it's whoever best exploits the gap between their picks and the crowd's picks. Deterministic argmax brackets collapse noseed's probabilistic edge into a single path that may be slightly contrarian but not *optimally* contrarian given crowd pick distributions.

Here's the deeper issue: you're measuring the wrong thing about noseed. Noseed's value isn't "wins more often" — it's "has calibrated probabilities that let you identify high-leverage upsets." The current infrastructure discards that calibration the moment you take argmax.

The question you should be asking: **what is the optimal bracket given noseed's probability estimates AND the SEED_PICK_RATES crowd model?** That's a portfolio optimization problem, not a comparison between three fixed strategies.

Options 1-4 are all downstream of this misframing. Building a pool-aware optimizer (option 3) is correct, but only if it takes stochastic bracket samples (option 2) as input AND uses noseed probabilities as the generative model.

The ML model's value hasn't been measured yet. You've only measured argmax-noseed vs argmax-seed. That's not the test.

### The Expansionist

The data is screaming something nobody is saying: noseed doesn't have a pool performance problem, it has a **variance harvesting problem**. Those "catastrophic" years (2012, 2023) aren't failures — they're the cost of a strategy that produced rank 9 and rank 45 in other years. The question isn't whether to fix noseed, it's whether you're playing the wrong game with it.

Here's the upside everyone is missing: **noseed is a different instrument, not a worse one**. In a 1000-person pool, P(rank=1) = 0.1% for a random player. Noseed achieves 5.4% — 54x baseline. Same as seed. But noseed's variance profile suggests it's doing *something structurally different* in certain tournament types. That's exploitable.

The real opportunity is option 3, but framed more aggressively: don't just optimize P(rank=1) generically — **build a pre-tournament classifier that identifies which regime you're in**. High-chalk tournament incoming? Go seed-heavy. Chaos-prone field (lots of parity, weak top seeds)? Noseed dominates. You have 17 years of labeled data. Train on it.

Even bigger: the opponent model using `SEED_PICK_RATES` means everyone is modeled as chalk-biased. If noseed picks upsets that the field underweights, those wins are *high-leverage by construction*. The infrastructure already exists to measure this — you just haven't extracted the upset-leverage signal from noseed's picks specifically.

Option 2 (stochastic brackets) unlocks this immediately. Deterministic argmax brackets are leaving contrarian leverage on the table. Switch to sampled brackets, measure the pick-frequency divergence from the field, and you'll see exactly where noseed is undervalued.

The ML model isn't marginally useful. It's a regime-specific weapon you haven't aimed yet.

### The Outsider

The catastrophic loss pattern is the most important signal here, and I'm not sure the framing is treating it seriously enough.

Noseed rank 699 in a 1000-person pool. Rank 676. These aren't "bad years" — these are near-worst-case outcomes. If I handed someone a strategy and said "this has a 1-in-3 chance of landing you in the bottom quarter of your pool," most people would not use it, regardless of what the Brier score says.

The question "does noseed have practical value" is being answered with P(rank=1) = 0.054 vs 0.054. That's identical. But the experience of using noseed is wildly different from seed — high variance, spectacular wins in 5 years, near-last-place finishes in others. That's not a bracket strategy. That's a lottery ticket.

Before choosing between options 1-4, I'd ask: what does the user actually want? If they're in a pool every year for the next decade, catastrophic losses compound socially and financially. If they're making a single high-stakes bet and need a moonshot, noseed's 2008 performance is compelling.

The "conditional switching strategy" (option 4) sounds sophisticated but it requires predicting which tournament structure will occur before the tournament happens. That's predicting the unpredictable.

The simplest honest answer from the data: the ML model does not improve pool outcomes on average and adds catastrophic downside risk. The burden of proof is on noseed to demonstrate value, and it hasn't met it. Options 1-3 are ways to rescue a model that may not need rescuing — they need rescuing from.

### The Executor

Option 3. Build the pool-aware optimizer. Everything else is yak-shaving.

Here's why the other options lose:

Option 1 (better opponent model) is research with no guaranteed payoff. You already have a validated opponent sampler. Marginal realism improvements won't flip a system that's losing catastrophically in some years.

Option 2 (stochastic brackets) is a tweak, not a fix. Stochastic sampling adds variance to your submissions but doesn't change what you're optimizing for.

Option 4 (conditional switching) is post-hoc rationalization. "Use ML when it works, don't when it doesn't" — you won't know which year you're in until it's over.

Monday morning task: Replace argmax bracket generation with a direct P(rank=1) optimizer. The MC simulator already exists and runs. The feedback signal already exists. You just need to close the loop.

Concretely:
1. Generate N candidate brackets (stochastic draws from your model probabilities)
2. Score each candidate bracket against your MC pool simulator
3. Keep the bracket with highest P(rank=1), not highest Brier score

This is hill-climbing. You have all the pieces. The MC simulator is your objective function — use it directly instead of using Brier as a proxy that demonstrably fails.

The noseed model's big wins (2008, 2017) happened when the ML diverged from chalk and was right. The optimizer will find those divergences systematically instead of accidentally. The catastrophic losses happened when ML diverged from chalk and was wrong — the optimizer will avoid those too, because the pool simulation will penalize them.

First step: write a function that takes a set of candidate brackets and returns P(rank=1) for each. You already have 90% of that code.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Executor
- Response B = The Outsider
- Response C = The First Principles Thinker
- Response D = The Contrarian
- Response E = The Expansionist

### Review 1
**Strongest: C** — correctly identifies the core measurement problem as portfolio optimization. D makes the same structural point but less precisely.
**Biggest blind spot: E** — regime classifier on 17 data points is nearly impossible to validate. Motivated reasoning dressed as analysis.
**All missed:** The opponent model's construction needs scrutiny. If simulated opponents are themselves chalk-heavy, the entire backtest is measuring two biased representations against each other.

### Review 2
**Strongest: C** — names what noseed's value actually is. Goes deeper than D by identifying portfolio optimization framing.
**Biggest blind spot: B** — recommends abandoning approach without acknowledging backtest may be invalid.
**All missed:** The opponent model is static and may not reflect the actual pool the user competes in. Garbage opponent model = garbage P(rank=1).

### Review 3
**Strongest: C** — diagnoses measurement failure, not model failure. D reaches similar conclusion but less precisely.
**Biggest blind spot: E** — regime classification on 17 examples is p-hacking dressed as a pipeline. Statistically indefensible.
**All missed:** Opponent model validation against ground truth. If miscalibrated, everything is garbage in, garbage out.

### Review 4
**Strongest: C** — root cause as measurement, not model. Portfolio optimization insight is analytically precise.
**Biggest blind spot: E** — 17 data points is not a dataset, it's anecdotes. Confidence is unfounded.
**All missed:** Opponent model is load-bearing assumption nobody questioned. Validate against actual bracket data first.

### Review 5
**Strongest: D** — identifies backtest as potentially invalid before prescribing solutions. Intellectually honest position.
**Biggest blind spot: E** — statistical sand. 17 examples is anecdotes, not training data.
**All missed:** Opponent model crowd calibration is the deeper assumption.

---

## Chairman Synthesis

### Where the Council Agrees
**The backtest is invalid until argmax is replaced.** Every advisor who engaged with methodology converged: deterministic argmax brackets are the core defect. Argmax collapses probabilistic models into crowd-following behavior. The current comparison is not between forecasting systems — it's between two ways of picking the most popular bracket.

**Noseed's value is in its probabilities, not its modal bracket.** Noseed wins by knowing which upsets are underpriced relative to the crowd. Argmax throws that signal away. The 17-year backtest has not yet measured what noseed actually offers.

### Where the Council Clashes
**Regime classification vs. single-mode optimization.** The Expansionist wants a pre-tournament classifier; the Executor wants a direct optimizer. Peer reviews rejected regime classification as statistically indefensible on 17 data points.

**Is noseed salvageable?** The Outsider argues catastrophic losses are disqualifying regardless of backtest mechanics. First Principles counters that this judgment is premature until the measurement problem is fixed. Resolution: fix the backtest first, then revisit.

### Blind Spots the Council Caught
**The opponent model is unvalidated and load-bearing.** Every peer review flagged this independently. No advisor mentioned it initially. If the crowd model doesn't match real bracket patterns, every P(rank=1) estimate is wrong for both modes.

**Regime classification on 17 points is p-hacking.** 4/5 peer reviews rejected it.

### The Recommendation
Fix the backtest, then validate the opponent model — in that order. Replace argmax with stochastic bracket generation, re-run the 17-year backtest. Concurrently audit the opponent model against historical bracket data if available. Do not build a regime classifier. Do not write off noseed from invalid data.

### The One Thing to Do First
Replace argmax bracket generation with probability-weighted stochastic sampling and re-run the full 17-year backtest. Everything else depends on having numbers you can trust. You don't have those yet. Get them.
