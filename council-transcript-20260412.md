# LLM Council Transcript — Pool Optimizer Next Actions
**Date:** April 12, 2026
**Question:** Given the latest results from mc_pool_optimizer.py, what are the most critical actions to take next?

---

## Context Provided to All Advisors

- 13-year backtest (1000-person pool, 50 stochastic brackets x 50 repeats, 11 modes)
- 2026 pool report (100-person, winner-take-all, contrarian strategy, 2.51x leverage)
- Construction modes outperform (champ_first_tv BestRnk=21.1, e8_first_tv P(1st)=0.20%)
- opt_* modes statistically significantly worse (p<0.05 Bonferroni), zero P(1st)
- Core model BSS ~ 0 vs seed baseline
- Sensitivity analysis: NOT_EVALUATED
- Missing aggregate section for latest backtest run
- Pool size mismatch: backtest at 1000, deploy at 100

---

## Advisor 1: The Contrarian

opt_* modes aren't just bad — they're catastrophically broken and you're burying the lede. BestRank of 176-342 in a 1000-person pool means these modes are performing WORSE than random bracket selection. That's not a tuning problem. That's a sign your optimization objective is fundamentally misaligned with pool scoring. Stop treating this as "opt modes need work" and start treating it as "something in the optimization pipeline is inverting value." Find the bug before you trust anything downstream of it.

The construction modes look good precisely because they're simpler — and that should worry you. champ_first_tv working well means your value is coming almost entirely from champion selection plus Torvik base rates. Everything you've built on top (hedging, optimization, blending) is destroying that signal. You've over-engineered a system where the dumbest strategy wins.

Your 2026 pool report is running on unvalidated infrastructure. Walk-forward harness is new, sensitivity analysis is NOT_EVALUATED, and you're about to make real picks. You have a 2.51x leverage ratio with zero evidence that your leverage calculations are calibrated. If your contrarian signal is even slightly miscalibrated, you're not fading chalk — you're fading reality.

**Critical actions:**
1. Diagnose opt_* catastrophic failure (not optional — if optimization is inverting, your entire objective function is suspect)
2. Run sensitivity analysis for 2026 NOW
3. Validate leverage ratio calculation against backtest years with ground truth
4. Stop adding features — your simplest mode wins

---

## Advisor 2: The First Principles Thinker

**You're optimizing the wrong layer.**

The backtest proves it: your game-level model has zero predictive edge over seeds (BSS ~ 0). Every model you tested — LR, LightGBM, XGBoost, ensembles — hits the same ceiling. This isn't a feature engineering problem. This is the market being efficient at game-level prediction. Seeds already encode the information your features capture.

Stop trying to beat seeds on game probabilities. That war is lost.

The actual edge is in the bracket construction layer. Champ-first and E8-first construction modes crush base modes (BestRnk 21 vs 38). The optimizer modes are catastrophically bad (176, 342). Why? Because construction topology — how you correlate picks across rounds — is where the pool leverage actually lives. Not in P(team X beats team Y), but in "which 63-pick portfolio maximizes expected rank against the field?"

**Critical actions:**
1. Kill the model improvement track entirely — BSS ~ 0 is terminal
2. Diagnose why opt_* modes are catastrophic (likely over-concentrating on leverage without respecting correlation structure)
3. Double down on construction mode research — explore more topologies
4. Validate the public pick distribution accuracy — garbage in poisons every downstream recommendation

**Uncomfortable question:** Is the 0.20% P(1st) from the best mode actually real? Random brackets in a 1000-person pool get ~0.10%. You've doubled random over 13 years.

---

## Advisor 3: The Statistician

**The N=13 problem dominates everything else.** With 13 years of data and per-year variance spanning 2.1 to 171.7 on BestRank, confidence intervals are enormous. The fact that no mode significantly beats seed on MeanRank with Bonferroni correction is not surprising — you almost certainly lack the statistical power to detect real differences of plausible effect size. A power analysis would likely show you need 30-50+ years.

**Critical actions:**
1. Fix the missing aggregate section immediately — blocking data quality issue
2. Run sensitivity analysis on the 2026 pool — NOT_EVALUATED is unacceptable before deployment
3. Validate the walk-forward harness against known results
4. Stop optimizing on BestRank alone — opt_* being worse on BestRank while MeanRank shows no difference suggests high-variance portfolios that bomb catastrophically
5. Pool size mismatch matters — 1000→100 changes optimal strategy fundamentally. Re-run at N=100.

**Bottom line:** System cannot statistically prove it beats the naive seed baseline. Either a power problem or the system genuinely adds little value. Determine which.

---

## Advisor 4: The Outsider

**Your optimizer is overfitting to noise and you're about to bet real money on it.**

The single most damning finding: the modes specifically designed to win pools perform significantly worse than naive seed-based brackets. When your "smart" thing loses to the dumb thing, the smart thing is broken. Full stop.

**Three things before touching anything else:**
1. Kill opt_* modes or explain why they fail — they're still in the pipeline feeding the 2026 recommendation
2. The 2026 contrarian play is unvalidated — model has BSS ~ 0, so leverage ratio is really just (seed_baseline / public_pick) with extra steps
3. Pool size mismatch is a real problem — backtested at 1000, deploying to 100. Optimal strategy is fundamentally different.

The construction mode finding is actually the most interesting result. The shape of how you build brackets matters more than the probabilities feeding them. Double down there.

---

## Advisor 5: The Executor

**Execution order with dependencies:**

1. **Deprecate opt_* modes** — 15-minute edit, zero risk, no dependencies
2. **Rerun backtest at N=100** — parameter change, compute-bound (hours). Start first.
3. **Fix aggregate summary** — reporting bug in backtest harness. Queue after step 2.
4. **Run sensitivity analysis** — the NOT_EVALUATED flag is unacceptable. Dependency: steps 2-3.
5. **Lock mode shortlist** — champ_first_tv + e8_first_tv + seed. Dependency: steps 2-4.
6. **Skip 2012 ESPN gap** — 12/13 years is sufficient.

Critical path: deprecate opt_* -> rerun at N=100 -> sensitivity sweep -> lock modes -> submit.

---

## Chairman's Synthesis

### Unanimous (5/5)
- Kill/deprecate opt_* modes
- Run sensitivity analysis for 2026

### Strong agreement (4/5)
- Diagnose WHY opt_* fail, not just deprecate
- Construction modes are the real alpha

### Key disagreement
- Executor: just deprecate and move on
- Contrarian/First Principles: must understand root cause because leverage.py (93k lines) powers both opt_* AND the contrarian recommendations

### Blind spot identified
leverage.py powers both the catastrophically broken opt_* modes and the pool report's 2.51x leverage ratio. If the objective function inverts value for optimization, is it sound for leverage calculation?

### Final 3 Actions
1. **Fix backtest aggregate + rerun at N=100** (BLOCKING)
2. **Run 2026 sensitivity analysis** (CRITICAL — minutes to execute)
3. **Deprecate opt_* + lock mode shortlist** (contingent on #1)
