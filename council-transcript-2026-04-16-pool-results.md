# Council Transcript — 2026-04-16

## Question
Do I actually need pool_hist_results.json (full pick-level opponent data) or is it sufficient to know ~1300 pts is close to first place every year?

## §4 Bucket Classification
**E — Genuinely novel.** O21 closed the pool-history marginals question (null result), but the score-threshold alternative (eliminating opponent simulation entirely) has never been proposed or tested.

## Panel Size
5 advisors (full panel). Peer review skipped — 4/5 converged on same recommendation (run the Spearman comparison test). First Principles diverged on the prior but not on the action.

## Framed Question
The pool optimizer maximizes P(finishing 1st in a ~30-person ESPN bracket pool). It runs Monte Carlo: simulate tournament → score each bracket → count wins. The opponent model uses 60% ESPN-national + 30% Massey + 10% seed fallback — pool_hist_results.json is NOT in the production path.

Prior work (O21): changing opponent model marginals doesn't change which brackets rank highest — only absolute P(1st) moves.

Proposal: replace opponent simulation with a score threshold. Instead of simulating 30 opponent brackets and counting wins, count how often your bracket scores ≥ ~1300 pts.

QUESTION: Is the full opponent simulation load-bearing for bracket selection, or does "maximize P(score ≥ 1300)" produce the same bracket ranking as "maximize P(rank = 1)"?

Prior art injected: O21 null result, O4 independence lock, O6 calibration signal, production 60/30/10 weights.

## Advisor Responses

### The Contrarian
O21's null result is being stretched beyond its scope. It tested opponent-model *weights* 0–0.5, not elimination of the simulation. Fixed 1300 threshold is a population statistic — your specific pool composition shifts the actual winning score materially. No equivalent O6-style validation exists for P(score ≥ 1300). Needs Spearman comparison before the simplification is justified.

### The First Principles Thinker
P(rank=1) = P(score ≥ max(S₁...S₃₀)). If opponent marginals don't change rankings (O21), the opponent score distribution is a constant backdrop. Threshold = opponent simulation with the theater stripped out. The real issue is 1300 is retrospective — use dynamic E[max] instead of a hardcoded number.

### The Statistician
O21 tested marginals; eliminating simulation entirely is a different question. Fixed threshold conflates your score distribution with the winning-score distribution, which shifts 100+ pts between chalk and chaos years. Required test: Spearman ρ between P(score ≥ 1300) and P(rank=1) rankings on historical simulation runs.

### The Outsider
Score threshold ignores joint probability — opponents face the same tournament realization as you. A bracket scoring 1400 in a chalk year can still lose to everyone who also picked chalk. Threshold sees 1400 ≥ 1300 and calls it a win; simulation correctly penalizes it. Test needed.

### The Executor
Run the test: pull P(rank=1) rankings for 2023-2026, compute P(score ≥ 1300) rankings, compute Spearman ρ. Two days of work. Static threshold needs to become dynamic (97th percentile of simulated scores). Speedup is real but only applies to opponent-scoring step, not tournament simulation itself.

## Peer Review
Skipped — convergence (4/5 same action recommendation, no fatal-flaw objection to the majority).

## Chairman's Verdict

### Where the Council Agrees
All five agree the equivalence is empirically testable and unverified. Test = Spearman ρ between P(rank=1) and P(score ≥ θ) rankings on historical sims. All five agree static 1300 threshold is the weak link.

### Latent Disagreement
Council converged on action but split on prior: First Principles expects high ρ (threshold ≈ simulation structurally); Contrarian/Statistician expect lower ρ for a specific reason — bimodal chalk-heavy brackets that are binary outcome generators cannot be distinguished by a fixed threshold but can by the joint simulation. This split determines what ρ = 0.90 means. First Principles would ship; Contrarian would not.

### Blind Spots the Chairman Flags
1. O21's scope was never interrogated — weight range 0–0.5 never reached the degenerate case of no simulation at all. Cited as proof; it's evidence.
2. Nobody asked which brackets differ between approaches. Overall ρ is the wrong metric — top-decile rank concordance is what matters, because that's the decision space.
3. O4 (real brackets less correlated than IID, z = −4.15) is weak evidence IN FAVOR of the threshold — low correlation means winner's score approximates max of IID draws more closely. Council missed this.
4. The correct replacement threshold isn't 1300 — it's the 97th percentile of simulated first-place scores (max across the simulated field), computed dynamically from the tournament simulation already running.
5. Field-size effect: P(rank=1) naturally penalizes median-safe brackets because with 30 players, 60th-percentile scoring still loses. Threshold approach doesn't know there are 30 opponents; dynamic threshold (97th percentile) implicitly encodes this.

### The Chairman's Take
First Principles is directionally correct but overconfident. Statistician wins the theoretical argument. But ρ ≥ 0.95 is too conservative — if ρ = 0.92 AND top-decile concordance ≥ 0.97, ship it.

The convergence is genuine (test is clearly the right next action) but stopped too early — advisors agreed on the experiment without agreeing on the evaluation criterion or the decision tree at each outcome.

The framing accepts "threshold vs simulation" as binary. It isn't. The real question is whether the joint structure of the opponent simulation — opponents scoring from the same tournament realization as you — is load-bearing. It is, but only for brackets whose win probability is correlated with whether the tournament is chalk or chaos. Testable by looking at the pairwise win matrix.

**On pool_hist_results.json specifically:** it's already NOT in the production path (O21 closed that). It remains valuable for O6-style validation and for calibrating the dynamic threshold. It's not load-bearing for bracket selection; it is load-bearing for system calibration and test coverage.

### Critical Actions
1. **Run the rank-concordance test.** Pull historical P(rank=1) simulation logs for 2023-2026. Rank all candidate brackets by P(rank=1) and by P(score ≥ 1300). Compute Spearman ρ on full ranking AND top-decile concordance. Gate: full ρ ≥ 0.95 AND top-decile concordance ≥ 0.97 in all four years → proceed to Action 2. Either fails in any year → simulation is load-bearing, stop.
2. **If Action 1 passes: replace static 1300 with the 97th percentile of simulated first-place scores,** computed fresh each run from the tournament simulation already running (no pool data needed). Gate: dynamic threshold reproduces ρ ≥ 0.95 on the same historical data.
3. **Profile the opponent-scoring pass before committing to simplification as a performance optimization.** If opponent-scoring runtime is < 15% of total, simplicity is the only remaining justification — which is sufficient only if Action 1's gate passes.
