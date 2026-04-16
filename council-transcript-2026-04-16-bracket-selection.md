# Council Transcript — 2026-04-16

## Question
How should we build a bracket selection optimizer?

## §4 Bucket Classification
**E — Genuinely Novel.** No §2 item covers bracket selection optimization. The 2026 ranking failure is flagged in MEMORY.md as a known open diagnostic but has no O-item, no gate, and no prior council verdict. Full 5-advisor panel warranted.

## Panel Size
5 advisors (full panel). Peer review ran — 2+ advisors materially diverged (First Principles proposed new objective function; Outsider proposed feature classifier; Contrarian/Statistician/Executor converged on "diagnose first").

## Framed Question
The pool optimizer generates a portfolio of 50 stochastic brackets per mode (champ_first_tv, e8_first_tv, blend) and selects one to submit using P(rank=1) from a Monte Carlo simulation against simulated opponents. This selection has Spearman ρ=0.37 vs actual pool outcomes — real but weak, explaining only 14% of variance. The portfolio consistently contains a pool-winning bracket across all 4 years (2023–2026); generation is not the bottleneck. In 2026 the winning bracket was ranked #11 out of 50. In 2025, the average bracket placed ~40th percentile of the actual pool. Task: design a bracket selection optimizer. Constraints: O4 (opponents less correlated than IID, z=−4.15), O21 (opponent marginals don't change rankings), O24 FAIL (score threshold failed; joint simulation structure is load-bearing), D6 (Pareto leverage failed), D16 (chalk-fade failed). Pool ~30 people, ESPN scoring 10/20/40/80/160/320, winner-take-all.

Prior art injected: O4, O6, O21, O24, D6, D16; MEMORY §1 locked pool strategy; pool_hist_results.json (2023–2026 actual brackets available).

## Advisor Responses

### The First Principles Thinker
The question assumes the problem is "better selection from the portfolio." It isn't.

The real problem: each candidate bracket defines a probability distribution over final scores. The pool is won by whoever draws the highest score. You need the bracket whose score distribution has the fattest right tail relative to the opponent score distributions. It's the expected supremum problem: which bracket maximizes P(b_i > max(opponents))?

Start here: simulate max(31 opponent scores) across your 5000 tournaments. That's your target distribution. Rank candidate brackets by P(b_i > that max). That's the correct objective function. The joint structure (O4 + O24) means opponents are near-independent, so max(opponents) has a known distributional form.

### The Contrarian
P(rank=1) with ρ=0.37 accounts for 14% of variance — 86% noise. Before designing a new framework, measure the spread of P(rank=1) across the 50-bracket portfolio. If top bracket has P(rank=1)=8% and #11 has P(rank=1)=7.5%, your selector is irrelevant — choosing between near-equivalent options. The selection problem only matters if the spread is actionable. Measure that first or you're solving a problem that doesn't exist.

### The Executor
Backtest the current ranker on pool_hist_results.json. The failure mode: the ranker uses expected P(rank=1) averaged over tournament simulations, but the actual pool is one realized tournament. Fix: conditional clustering — cluster MC simulations by chalk/upset profile, rank brackets within each cluster, submit the bracket that dominates across cluster types. One file, ~50 lines.

### The Statistician
Before building new machinery, quantify how often the rank ordering of your 50-bracket portfolio is correct — run retrospective simulations on years with known outcomes. SE on P(rank=1) near 0.02-0.05 at N=5000 is ~0.003-0.005 — tight enough that the estimator isn't the problem. If rank-order accuracy is already near ceiling, you're playing against irreducible noise, not a fixable methodology flaw.

### The Outsider
You have 200 labeled examples (50 brackets × 4 years). Train a simple classifier on bracket-level features — upset count, entropy of picks, champion popularity — against the binary outcome "did this bracket win the pool?" Start there before building more simulation infrastructure.

## Peer Review

### What all five responses missed (composite from 5 reviewers)
1. **The 2026 post-mortem** (flagged by 4/5 reviewers): bracket #11 won — examine what P(rank=1) the ranker assigned to #11 vs #1-#10 and what distinguishes its pick pattern. That single case is 25% of all labeled data.
2. **O4's sign propagates into the supremum** (flagged by reviewer 3): if opponents are *less* correlated than IID, max(opponents) has *higher* variance, which systematically favors high-upside upset brackets in the supremum calculation. None of the advisors connected O4 back to the objective function shape.
3. **Submission timing / portfolio refresh** (flagged by reviewer 1): late-breaking information (injury reports, updated priors) available before submission could shift which bracket ranks #1.
4. **Opponent model specificity** (flagged by reviewer 5): the actual 30-person pool has a specific human pick distribution. P(rank=1) against a generic prior is optimizing against the wrong adversary.

### Strongest response (4/5 reviewers): First Principles (A)
Correctly identifies that P(b_i > max(opponents)) is the right objective, with O4 enabling tractability of the max(opponents) distribution.

### Biggest blind spot (5/5 reviewers): Outsider (E)
Only 4 positive examples (one pool winner per year). Training a classifier on 200 samples with ~2% positive rate is not statistically viable.

### C flagged as potential D6/D16 re-litigation (reviewer 4)
Conditional clustering by chalk/upset regime risks re-opening dead-ends from D6 (leverage by regime) and D16 (chalk-fade). Needs explicit differentiation from those failed approaches.

## Chairman's Verdict

### Where the Council Agrees
The mathematical objective is correct: select the bracket that maximizes P(b_i > max(opponents)), not P(rank=1) in isolation. Response E is unanimously rejected — 4 positive examples across 200 observations is not a training set. The current ρ=0.37 selector is real but weak, and the weakness is structural rather than a tuning problem.

### Where the Council Clashes
**A vs B on sequencing.** A says build the supremum estimator now. B says measure portfolio spread first — if 50 brackets are bunched within 0.5%, selection optimization is irrelevant regardless of objective function. B is right on the gate: a 5-minute diagnostic should precede any new machinery.

**C vs D on diagnosis vs design.** C wants conditional clustering. D says rank-order accuracy may be near ceiling against irreducible noise. These are not mutually exclusive — D's retrospective analysis tells you whether C's intervention has headroom.

**O4's sign matters.** Review 3 caught something the advisors missed: O4's anti-correlation (z=−4.15) means max(opponents) has *higher* variance than the IID case. This systematically favors high-upside upset brackets. A's framework is correct in structure but incomplete — it doesn't propagate O4 into the distributional shape of max(opponents).

### Blind Spots the Council Caught
**The 2026 post-mortem is missing.** Bracket #11 won — that single case is 25% of all labeled data. Inspect what P(rank=1) the ranker assigned to #11 vs #1-#10, and what distinguishes #11's pick pattern. If #11 had an upset chain the ranker systematically discounted, that's a concrete signal. If it was pure noise, that's also informative.

**The opponent model is the other missing piece.** P(rank=1) is computed against a simulated opponent pool, not the 31 actual humans submitting. pool_hist_results.json exists. O21 says opponent marginals don't change bracket rankings, but joint structure is load-bearing (O24) — which means the actual correlation structure of these 30 humans matters. If the MC runs against a generic prior rather than the actual pool bracket distribution, the adversarial structure is wrong.

### The Chairman's Take
The 2026 case is the Rosetta Stone everyone overlooked. The supremum framework (A) is correct. But two diagnostics are required before implementation: (1) what is the spread of P(rank=1) across the current 50-bracket portfolio, and (2) what did the 2026 ranker actually do to bracket #11.

If spread is narrow (< 1% between rank 1 and rank 11), selection optimization is noise trading and the real problem is portfolio generation diversity. If spread is wide and #11 was legitimately ranked low, the ranker has a systematic blind spot — the O4 effect: the ranker underweights high-variance upset brackets because it's modeling max(opponents) as if opponents were IID, when anti-correlation actually fattens that maximum distribution and makes chalk-busting brackets relatively more valuable.

The fix is not conditional clustering (risks re-litigating D6/D16) and not a classifier (E). It's computing max(opponents) empirically from pool_hist_results.json across the 5000 MC runs, propagating O4's anti-correlation into that distribution, and ranking candidates by P(b_i > that empirical maximum).

### Critical Actions
1. **Run the spread diagnostic and 2026 post-mortem.** For each of the 50 brackets in 2026, extract the P(rank=1) assigned by the current ranker and the actual rank outcome; compute the spread between #1 and #11 and check whether #11 had a structurally distinct upset profile. Gate: a table showing P(rank=1) vs actual rank for 2026, confirming whether the selection gap between #1 and #11 was < 1% (noise) or > 3% (fixable systematic error).

2. **Replace the selection objective with the empirical supremum.** Using pool_hist_results.json, compute max(opponent scores) across 5000 MC runs using the *actual* pool bracket corpus; rank the 50 candidates by P(b_i > that empirical max), and verify O4's anti-correlation is reflected by checking the ranking favors higher-variance brackets relative to the current ranker. Gate: Spearman ρ between new ranker and actual 4-year outcomes exceeds 0.50, or the 2026 winning bracket ranks in the top 5 under the new objective.

3. **If spread was narrow (gate 1 revealed noise), shift focus to portfolio diversity.** Measure pairwise score correlation across the 50-bracket portfolio and add a diversity penalty to generation that explicitly targets the tail of the score distribution. Gate: average pairwise correlation drops below 0.6 and the portfolio contains at least one bracket in the top decile of upset-count distribution.
