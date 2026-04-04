# LLM Council Transcript — Session #9
**Date:** April 4, 2026
**Topic:** Architectural Problems — Why Backtests Show Random Performance

---

## Original Question

> Council this: i want to pursue architectural problems, what should i look into? Maybe data leakage, missing variables, other model types?

---

## Framed Question

**Decision: What architectural problems explain random backtest performance, and what should be fixed?**

This March Madness bracket pool optimizer pivoted from ML prediction (BSS=0, zero value) to game theory pool optimization. After fixing a critical ESPN team-name matching bug, stochastic backtests were run with clean data against ESPN opponents. Results: ALL modes (seed, noseed, blend) show median rank ~530-546/1000 — effectively random. The pre-committed success threshold was median rank <300/1000. The system failed badly.

**Key codebase discovery:** The `mc_pool_backtest.py` script NEVER calls `PoolOptimizer.optimize()`. It uses `sample_model_brackets()` which does pure probabilistic sampling from round advancement probabilities. The entire optimization layer — leverage analysis, Pareto frontier, contrarian strategy — is completely bypassed in backtests.

Additional findings:
- Two different blend logics exist: backtest uses alpha=0.5 seed-noseed; the actual opponent model uses 60/30/10 ESPN/ratings/seed
- Exploitable chalk bias signal exists (F4: public picks 42% vs 21% true advancement for 1-seeds; CHAMP: 18% vs 12%) but is unused
- PoolOptimizer has leverage analysis, Pareto brackets, sensitivity analysis — all well-designed but never called
- Leverage formula: EV-edge = (model_prob - public_pct) × round_points

The user wants to understand WHY results are random and what architectural fixes to pursue. Candidate directions: data leakage, missing variables, other model types, or fixing the optimizer integration.

---

## Advisor Responses

### The Contrarian

Everyone's about to say "wire the optimizer into the backtest." They're right, but they're skipping the real question: does your optimizer ACTUALLY produce different brackets than random sampling?

Before wiring anything, do a single-year sanity check. Take 2023. Run PoolOptimizer.optimize() manually. Look at the brackets it produces. Are they materially different from sample_model_brackets() output? If leverage analysis says "pick a 5-seed over a 4-seed in the Sweet 16 because public underweights them" — does that actually show up in the generated brackets?

The optimizer could be well-designed but produce near-identical output to random sampling if the model probabilities and public pick rates are too similar. That's the real risk nobody wants to examine. If model_prob ≈ public_pct for most games, leverage is near-zero everywhere, and optimization degenerates to random selection.

Also: the backtest uses alpha=0.5 seed-noseed blend for the MODEL but 60/30/10 for opponents. You're comparing apples to oranges. The model's probability estimates come from one blend; the "what the public thinks" comes from another. If these converge, there's no edge to exploit.

Validate the optimizer produces meaningfully different brackets on ONE year before wiring it into 17 years of backtesting.

### The First Principles Thinker

Strip away everything. A pool optimizer needs exactly one thing: a gap between what you believe and what the crowd believes. Call it "information edge" or "leverage" — same thing.

Your system has three potential sources of edge:
1. Better game probability estimates (56 rating systems vs ESPN consensus)
2. Better opponent modeling (knowing what brackets others will submit)
3. Better portfolio construction (choosing brackets that are anti-correlated with the crowd)

Source 1 was already proven dead — BSS=0, your models don't predict better than the public. Source 2 exists (ESPN pick data) but is only used for opponent simulation, never for strategic differentiation. Source 3 exists in code (PoolOptimizer) but is never called.

The architectural problem is simple: you built a gun, loaded it, aimed it, and never pulled the trigger. The PoolOptimizer IS the strategy. sample_model_brackets() is just drawing from the same probability distribution the public uses. Of course it's random.

But here's the uncomfortable follow-up: even after wiring the optimizer, Source 1 being dead means your model_prob is essentially the same as public_pct. Leverage = (same_number - same_number) × points = 0. You may need to artificially inject contrarian bias (overweight underdogs where public systematically overestimates favorites) rather than relying on model superiority that doesn't exist.

### The Expansionist

The optimizer bypass is the obvious fix, but the REAL opportunity is what you do after you wire it in.

You're sitting on 19 years of ESPN "Who Picked Whom" data with clean team matching. Nobody else has this dataset in a structured, backtestable format. The chalk bias signal — public picking 1-seeds to the Final Four at 42% when true rate is 21% — is MASSIVE. That's a 2:1 mispricing in the most consequential rounds.

Here's what you build:
1. Wire optimizer into backtest (table stakes, do this first)
2. Build a "contrarian signal layer" that identifies systematic public mispricings per seed per round across your 19-year dataset
3. Use that signal to TILT model probabilities away from public consensus — not because your model is better, but because you're deliberately exploiting crowd overconfidence on favorites
4. Optimize bracket construction for P(rank=1) not EV — Clair & Letscher (2007) showed this requires conscious differentiation

The chalk bias alone could be worth 5-10 percentile points in pool rank. Combined with anti-correlated bracket construction from the optimizer, you might actually break through the threshold.

Stop trying to predict better. Start trying to be strategically different.

### The Outsider

I'm reading this from scratch and I see something nobody's talking about.

You ran a backtest. It showed random performance. Everyone diagnosed "the optimizer isn't wired in." But the backtest wasn't DESIGNED to test the optimizer — it was designed to test whether stochastic bracket sampling from different probability models (seed vs noseed vs blend) produces different pool outcomes.

And the answer is: no, it doesn't. All three models produce the same pool performance. That's actually a valid and important finding. It means your probability models don't differentiate enough to matter even at the sampling level.

Now, will wiring in the optimizer change this? Maybe. The optimizer does something fundamentally different — it doesn't just sample from probabilities, it strategically SELECTS picks that are anti-correlated with the crowd. But if the underlying probabilities are identical to public consensus, the optimizer has nothing to work with.

The question isn't "why is the backtest random?" The question is "do we have ANY information that differs from what ESPN's crowd already knows?" If the answer is no, then no amount of optimization architecture will help. You'd need to inject external signal — like the chalk bias correction the Expansionist mentioned — because your models don't have it natively.

Wire the optimizer in. But prepare for the possibility that it doesn't help because the input probabilities are the problem, not the architecture.

### The Executor

Here's the concrete plan. Three steps, strict order:

**Step 1 (2-3 hours): Wire PoolOptimizer into mc_pool_backtest.py**
- Add a 4th mode: "optimized" alongside seed/noseed/blend
- For the optimized mode: instantiate PoolOptimizer with model round probs + ESPN public pick distribution
- Call optimize() to get leverage-weighted brackets instead of random samples
- Score and rank alongside the existing modes
- This is the single highest-leverage change. Every other question becomes answerable once you can compare optimized vs unoptimized brackets in the same backtest.

**Step 2 (1 hour): Single-year validation**
- Before running the full 17-year backtest, run 2023 only
- Print the actual bracket picks from optimized mode vs seed mode
- Verify they're materially different (if they're not, the optimizer isn't generating useful contrast)
- Check leverage values: are any picks showing meaningful EV-edge?

**Step 3 (30 min): Full backtest comparison**
- Run the 17-year backtest with all 4 modes
- Compare optimized mode against the threshold (median rank <300)
- If optimized mode still fails: the input probabilities are the bottleneck, not the architecture
- If optimized mode succeeds: you've found the missing piece

Do NOT pursue data leakage, missing variables, or other model types until you've completed these three steps. Those are all downstream of knowing whether the optimizer makes a difference.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Review 1
**Strongest: B (First Principles)** — correctly identifies that BSS=0 means model_prob ≈ public_pct, so leverage may be near-zero even with the optimizer wired in. This is the deepest insight. **Blind spot: C (Expansionist)** — jumps to building a contrarian signal layer before verifying the optimizer works at all. **All missed:** The backtest's OPPONENT model also matters. If opponents are modeled using ESPN pick rates AND the model uses similar probabilities, both sides of the competition are effectively identical. The optimizer can only differentiate if there's asymmetry in the inputs.

### Review 2
**Strongest: E (Executor)** — only response with time-boxed, sequenced steps and clear branch points. **Blind spot: D (Outsider)** — says "the backtest wasn't designed to test the optimizer" as if that's acceptable. The whole point of a backtest is to test the strategy you'd actually deploy. Not testing the optimizer is a bug, not a feature. **All missed:** Nobody discussed the TWO different blend logics (alpha=0.5 vs 60/30/10). This inconsistency means the backtest model and the production opponent model are working with different probability distributions. This needs reconciliation.

### Review 3
**Strongest: A (Contrarian)** — the single-year sanity check before full backtest is the smartest sequencing suggestion. If leverage values are near-zero on inspection, you save hours. **Blind spot: C (Expansionist)** — "2:1 mispricing" on Final Four picks sounds dramatic but hasn't been verified to translate into pool advantage. **All missed:** The sample size problem. Even with the optimizer wired in, 17 years × 50 brackets = limited statistical power. P(rank=1) in a 1000-person pool from 50 brackets may not converge.

### Review 4
**Strongest: E (Executor)** — clear, actionable, time-boxed. **B's insight about leverage=0 is the most important analytical point** but it's more diagnostic than actionable. **Blind spot: A (Contrarian)** — "validate on one year" is good but could become an excuse to delay wiring the optimizer if the one year doesn't look promising. **All missed:** Integration testing. After wiring the optimizer, verify that output brackets are structurally valid before scoring them — all 63 games filled, format compatible with the scorer. A silent format mismatch will produce garbage results with no error message.

### Review 5
**Strongest: E (Executor)** — correct sequence with fallback plan. **Blind spot: B (First Principles)** — too pessimistic. Even if model_prob ≈ public_pct on average, there may be SPECIFIC games/rounds where divergence is large enough to exploit. Leverage doesn't need to be positive everywhere, just in high-value spots. **All missed:** The chalk bias signal is already KNOWN and QUANTIFIED (F4 42% vs 21%). This isn't speculative — it's measured data. The question is whether the optimizer's leverage calculation already captures this or whether it needs to be explicitly injected.

---

## Chairman Synthesis

### Where the Council Agrees

The optimizer bypass is not a mystery — it is a bug. `mc_pool_backtest.py` calls `sample_model_brackets()` instead of `PoolOptimizer.optimize()`, which means 17 years of backtests measured "random sampling vs random sampling." The random result is correct and expected given what was actually tested. Every advisor agrees: wire the optimizer in before pursuing any other architectural question.

The council also agrees the chalk bias signal is real and already measured — F4 public picks 42% vs 21% true advancement rate is not speculation, it is data. The system is sitting on exploitable signal it has never used.

### Where the Council Clashes

The central clash is whether wiring the optimizer will actually help. The First Principles Thinker and the Outsider argue it may not: BSS=0 means model probabilities are essentially identical to public consensus, so leverage = (same number - same number) × points = 0, and the optimizer has nothing to work with. The Executor and Expansionist argue it will help, especially combined with the chalk bias signal. The Contrarian wisely calls for a single-year inspection before committing to the full backtest.

This is a real dispute with stakes. Do not paper over it.

### Blind Spots the Council Caught

Three things nobody said clearly enough:

1. **The two-blend inconsistency.** The backtest model uses alpha=0.5 seed-noseed blend. The production opponent model uses 60/30/10 ESPN/ratings/seed. These are different probability distributions. You cannot validate optimizer performance when the model and opponent simulator are using different inputs. Reconcile these before any backtest means anything.

2. **The leverage-zero failure mode.** Even after wiring the optimizer, if inspection of the 2023 leverage values shows near-zero EV-edge on most picks, the problem is not architecture — it is that the input probabilities have no information the crowd doesn't already have. The optimizer cannot manufacture edge from flat inputs.

3. **Integration validity.** After wiring the optimizer, verify that output brackets are structurally valid before scoring them — all 63 games filled, format compatible with the scorer. A silent format mismatch will produce garbage results with no error message.

### The Recommendation

Wire the optimizer into the backtest. Then immediately inspect leverage values on a single year before running 17 years. If leverage is near-zero across the board, stop and inject the chalk bias correction directly — tilt model probabilities away from public consensus on known mispriced rounds (primarily F4 and Championship picks on 1-seeds). Do not build new model types or chase data leakage until you know whether the optimizer produces meaningfully differentiated brackets.

Reconcile the two blend logics as part of the wiring step, not after. One probability distribution, used consistently in both model and opponent simulator.

Do not pursue data leakage analysis. The backtest failure was not from data leakage — it was from testing the wrong thing entirely. That question is now closed.

### The One Thing to Do First

Before writing a single line of code, run this manually for 2023: instantiate `PoolOptimizer`, call `optimize()`, and print the top leverage picks with their EV-edge values. If the numbers are near-zero, you have a probability input problem and the Contrarian and First Principles Thinker are right — no architecture fix will save you without injecting the chalk bias signal. If the numbers show real divergence, the Executor's plan works and you proceed to full integration.

Ten minutes of inspection saves hours of wiring work that may prove irrelevant. Look before you build.
