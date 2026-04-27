# Council Report — 2026-04-26 post-augmentation-critical-actions

## Question

What are the 1-3 most critical action items right now — fixing bugs, pursuing new deterministic strategies (S1-S7), or something else entirely?

## Metadata

- **Bucket:** E (genuinely novel — post-pivot deterministic regime + augmentation experiment results are new context)
- **Panel:** 5 advisors (Contrarian, First Principles, Statistician, Outsider, Executor)
- **Peer review:** ran (3 reviewers)
- **Timestamp:** 2026-04-26T22:15Z

## Suggested Actions

> 1. **Diagnose meta_gbm v2's champion picks before building anything new.** How many of 14 backtest years does v2 get the champion right? If it's rarely correct on champion, that's the binding constraint — not features or construction modes. This diagnostic takes 30 minutes and determines whether S1 (champion-first) is the right next move or a distraction.
> 2. **Build S1 (champion-first construction) if the diagnostic confirms champion prediction is the bottleneck, OR S2 (box score features) if champion picks are already reasonable.** Pre-register whichever you pick plus at most one other strategy. Do not run all 7.
> 3. **Set a hard deadline: system submittable by January 2027.** You missed 2026. The model works. Ship it.

## Per-Advisor Bottom Line

- **The Contrarian:** The backtest may be lying — only 4 of 14 years use real pool opponents. Stress-test simulated vs real opponent years before building anything new. S1-S7 all assume the evaluation harness is trustworthy and nobody has verified that.
- **The First Principles Thinker:** The question "which S1-S7?" is wrong. The champion pick is 17% of total points and 100% of pool winners got it right. Decompose: champion prediction is THE problem. Stop treating 63 games equally.
- **The Statistician:** Fix multiple comparisons before running anything. meta_gbm v2's p=0.041 inflates past 0.05 with more comparisons. Pre-register max 2-3 strategies with kill criteria. The 11/14 win-rate is more robust than the 2.71% P(1st) aggregate.
- **The Outsider:** The system is at 2.71% vs 3.3% random and has never been used. Submit in 2027 — nothing else matters. Champion pick is one decision. Calculate whether the engineering ROI is worth it.
- **The Executor:** Monday morning: (1) wire S2 box score features (data already on disk), (2) build S1 champion-first, (3) set January 2027 deadline. You missed 2026 because there was no deadline.

## Where Advisors Agree Most

- **Champion pick dominance.** 4/5 advisors (First Principles, Outsider, Executor, plus peer reviewers) converge on the champion pick as the highest-leverage single decision. 320/1920 points, 100% of actual winners got it right. The current model has zero champion-specific logic.
- **Ship deadline.** 3/5 advisors (Outsider, Executor, and implicitly First Principles via "if no, nothing else matters") agree that actually submitting in 2027 is a prerequisite for any of this to matter. The system has never been tested live.
- **Do not run all 7 strategies.** 3/5 advisors (Contrarian, Statistician, Executor) agree that running S1-S7 sequentially is either premature (validate first) or statistically dangerous (multiple comparisons).

## Where Advisors Clash

- **Build now vs validate first.** Executor says "wire features Monday, ship it." Contrarian and Statistician say "stop — the harness might be lying and your significance erodes with each test." This is the core tension: 11 months of runway argues for building, but building on a flawed foundation wastes all of it.
- **Is 2.71% meaningful?** Outsider says it's below random (3.3%) so the system doesn't work. Others note this comparison is misleading — 3.2% is 1/N for N=31, while 2.71% is measured against a seed baseline (1.76%) using a different opponent model. The Statistician notes P(1st) is poorly estimated regardless.
- **Champion prediction: ML problem or not?** First Principles says build a champion predictor. Outsider says "this is one decision, not ML." Neither engaged with whether the existing model already gets champions right sometimes.

## Blind Spots from Peer Review

- **Nobody diagnosed v2's actual champion picks.** All 3 reviewers flagged this. The model is trained with `weight=round_pts` which already upweights the championship game at 320. Does v2 get the champion right in backtest? How often? Which failures were predictable? This diagnostic should precede both S1 and S2 — it determines which is higher leverage.
- **Pool size trajectory.** Pool grew from 18 (2023) to 32 (2025). If it hits 40+ by 2027, P(1st) baselines shift and current estimates are miscalibrated. The harness should parameterize pool size.
- **Opponent model fidelity upstream of training.** Contrarian flagged simulated vs real opponents for evaluation, but one reviewer noted the deeper issue: meta_gbm was *trained* on features computed against simulated-opponent contexts. Opponent model fidelity is upstream of everything, not just evaluation.

## Load-Bearing Assumptions

- **Scoring encoding**: team-identity (real ESPN)
- **Opponent field**: pool history (2023-2026, N=18-32); ESPN simulated (2011-2022)
- **RNG / sample count**: N/A (deterministic — 1 bracket per year, 50 opponent repeats)
- **Year scope**: 2011-2026 ex 2020 (14 years)
- **Baseline anchor**: vs seed stochastic (1.76% P(1st))
- **Data sources**: `scripts/mc_pool_backtest.py`, `src/prediction/meta_selector.py`, `data/pool_history/pool_hist_results.json`, `artifacts/backtest_runs/mc_pool_backtest_20260426_214856.txt`

## Framed Question + Prior Art

**Framed question:** Given meta_gbm v2 (2.71% P(1st), 11/14, p=0.041), the augmentation experiment (smoothing hurts P(1st)), and 7 proposed strategies (S1-S7), what are the 1-3 most critical action items? Options: fix bugs, build new strategies, or something else.

**Prior art injected:**
- §1 locked: "Stop optimizing prediction accuracy. Optimize bracket-pool EV against opponents." (unanimous)
- §1 locked: BSS ≈ 0 vs seed baseline across all models.
- §1 locked: "Stochastic bracket generation is non-negotiable" — pre-pivot; meta_gbm v2 deterministic has since beaten stochastic on P(1st).
- §2 O13: stochastic wins BestRank, argmax wins P(1st) bimodally. meta_gbm v2 avoids bimodality.
- §2 O25: selection from 50-bracket portfolio is near-irreducible noise. Deterministic avoids this.
- §2 O4/O10/O21: all opponent model investigations closed as dead ends or null results.

## Advisor Responses

### The Contrarian

The fatal flaw nobody's naming: your backtest might be lying to you. 14 years LOOY with real opponents for only 4 years. The other 10 use simulated opponents. Run meta_gbm v2 with ONLY the 4 real-opponent years. If there's a large gap, your 2.71% is partly a simulator artifact. Fix the measurement before optimizing what you're measuring. S1-S7 all assume the evaluation harness is trustworthy. Nobody's verified that.

### The First Principles Thinker

The question "which S1-S7" is wrong. Decompose the problem: champion prediction (320 pts, 17% of total, 100% of winners got it right) is a separate problem from the other 62 games. Stop treating all 63 games equally. The one action item: can you predict the champion better than seed baselines? If yes, that dominates. If no, nothing else matters.

### The Statistician

Fix multiple comparisons before running anything. meta_gbm v2 was p=0.041; adding 3 augmentation variants pushed to p=0.0552. Seven more strategies = 11+ comparisons, original result no longer significant. Pre-register max 2-3 strategies with kill criteria before execution. P(1st) is poorly estimated (single bracket per year, 50 opponent repeats, pool-size confounded). The 11/14 win-rate is more robust than the 2.71% aggregate.

### The Outsider

The system is below random chance (2.71% vs 3.3%) and has never been used. Action 1: Submit in 2027, nothing else matters. Action 2: Get the champion right — this is one decision, not an ML problem. Action 3: Calculate the dollar value of moving from 3% to 6%. Is 11 months of engineering worth it?

### The Executor

Three actions for Monday morning: (1) Wire box score features into meta_selector (S2) — data on disk, just plumbing, one backtest run to know. (2) Build champion-first construction (S1) — champion probs already computed, highest-leverage single decision. (3) Set calendar deadline: system submittable by January 2027. You missed 2026 because there was no deadline.

## Peer Review

### Reviewer 1

**Strongest: A.** Identifies the one thing that, if wrong, invalidates everything else. Measurement integrity before optimization. **Biggest blind spot: D.** Conflates random baseline (3.3%) with the measured 2.71%, drawing a false conclusion. "Just submit" without validation is worthless. **All missed:** Pool-size trajectory — grew from 18 to 32 in 4 years. P(1st) is mechanically a function of field size. If pool hits 40+ by 2027, current estimates miscalibrate.

### Reviewer 2

**Strongest: B.** Correctly identifies structural asymmetry. Champion is 320 points, zero champion-specific logic in meta_selector. **Biggest blind spot: D.** Conflates baselines (2.71% vs seed 1.76%, not random 3.3%). **All missed:** The interaction between champion pick and meta_gbm v2. The GBM already upweights championship via round_pts=320. Nobody asked how often v2 gets champion right in backtest. That diagnostic determines whether the ceiling is in champion prediction or earlier rounds.

### Reviewer 3

**Strongest: B.** Challenges problem decomposition rather than tweaking. **Biggest blind spot: D.** Misleading baseline comparison. **All missed:** Same as Reviewer 2 — nobody diagnosed v2's actual champion picks. "Which champion picks did v2 get wrong, and were any predictable?" should precede all new strategy work.

---

*Council session 2026-04-26T22:15Z — post-augmentation critical action items.*
