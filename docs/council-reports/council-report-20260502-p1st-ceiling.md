# Council Report: #1 Critical Step to Improve P(1st) Above 8.0%

**Date:** 2026-05-02
**Panel:** 5 advisors (bucket E — genuinely novel)
**Convergence:** NO (2-2-1 split)
**Peer review:** Skipped (divergence is clear, positions well-defined)

---

## Question

What is the single most critical next step this March Madness bracket optimization repo should take to improve P(1st) above the current 8.0% ceiling?

## Context Provided

- Pool: ~30 person winner-take-all ESPN scoring (10/20/40/80/160/320)
- Current best: meta_region = 8.0% P(1st), 14-year LOYO, p<0.001 vs seed (3.1%)
- Champion pick ~random among 1-seeds, proven irrelevant to P(1st)
- GBM learned model (4.6%) worse than raw probability-based construction
- 14 Kaggle/academic techniques tested, none beat region construction
- R64 accuracy 76.3%, R32 64.7%; 46 contrarian-correct picks drive differentiation
- Real opponent bracket data available for 2023-2026 (105 brackets)
- N=14 LOYO years is fixed

---

## Advisor Responses

### 1. The Contrarian

You're probably already at or near the ceiling, and the "single most critical next step" framing is a trap.

N=14 is statistically meaningless for detecting improvements. Going from 8.0% to 9.5% would need ~200 LOYO years to confirm. You've already exhausted the construction frontier — region and exhaustive converging at 7.7-8.0% isn't coincidence, it's the ceiling making itself visible. The R64-R32 "leak" is a mirage: 8v9 games are 50/50 by definition, 6v11 at 45% means the market already knows.

In a 30-person pool with 2/14 champion accuracy, the theoretical maximum is probably 10-12%. You're capturing 70-80% of available edge already.

**Bottom line:** Accept diminishing returns. Focus on execution reliability for 2027, not chasing phantom percentage points.

### 2. The First Principles Thinker

You're optimizing the wrong variable. P(1st) is dominated by opponent bracket variance, not your bracket's absolute quality. That's a joint distribution over (tournament outcomes) x (opponent bracket distribution). You're pouring effort into the first term while treating the second as noise.

Given BSS=0 ceiling and ~random champion pick, the lever that remains is **opponent-conditional construction**. You have real opponent bracket data from 2023-2026. The optimal bracket isn't the one most likely to score well absolutely — it's the one most likely to score highest relative to these specific 29 opponents.

Stop asking "which picks are most likely correct?" Start asking "which picks, conditional on the tournament outcome distribution, maximize the probability of beating THIS field?"

**Bottom line:** Build opponent-aware bracket construction that maximizes E[P(my score > max(opponent scores))] using pool history behavioral model as the opponent prior.

### 3. The Statistician

The 8.0% number is almost certainly inflated. You tested 20+ strategies against 14 years. The strategies share the same probability backbone, opponent simulation, and 14 years — correlated multiple comparisons on a tiny sample.

Year-level variance: 0% to 18% P(1st). Standard error ~1.5-2 percentage points. 95% CI likely spans 4-12%. The difference between meta_region (8.0%) and meta_exhaustive (7.7%) is pure noise. The "p<0.001 vs seed baseline" is real but irrelevant to "which variant is best?" — the data cannot answer that at N=14.

**Bottom line:** Run a permutation test: shuffle year labels 10,000 times across all 20+ strategies, measure how often the best hits 8.0% by chance. Establish what true P(1st) is after correcting for strategy search before chasing further improvements.

### 4. The Outsider

Why are you optimizing your bracket at all? The ML model was WORSE than raw ratings. 14 techniques couldn't beat the simple approach. Every time you add complexity, it does nothing or makes things worse.

Have you tried modeling the other 29 people instead of modeling basketball? You're not predicting games — you're trying to finish first in a 30-person contest. Those are completely different problems.

You know the pool's tendencies, biases, where they cluster. The question isn't "which team wins game X" — it's "where are the 29 other people making the same mistake, and can I be different there while still being roughly correct?"

**Bottom line:** Stop trying to be more accurate. Start trying to be more usefully unique. The 8% ceiling might not be a prediction ceiling — it might be a strategy ceiling.

### 5. The Executor

33/67 on upset picks means 49% hit rate on games where you actively deviate from chalk. Every additional correct upset pick is worth 10 points AND pool separation. Getting to 36/67 or 38/67 is the concrete lever.

Monday morning: build a dedicated upset classifier for 8v9/7v10/6v11/5v12 matchups. Not a general predictor — a specialist. ~56 instances per seed pairing across 14 years. Features: conference tournament momentum, defensive efficiency trends, travel distance, close-game experience. Simple logistic model. Override beam search when specialist confidence exceeds 55%.

**Bottom line:** One Python file, one integration point in bracket construction, testable in a day.

---

## Agreement (3 themes all 5 share)

1. **Probability improvement is dead.** BSS=0 ceiling, 14 techniques tested, GBM worse than raw probabilities. No advisor suggests better game prediction.

2. **The opponent field is under-modeled.** First Principles, Outsider, and Executor all point to this gap. Contrarian and Statistician don't disagree — they question whether there's enough data to exploit it.

3. **N=14 is a hard constraint.** Any improvement must be large enough to survive the noise floor of 14-year LOYO.

## Clashes

### "Stop vs Go" (Contrarian+Statistician vs First Principles+Outsider+Executor)

- **Stop camp:** 8% may be the ceiling (or inflated). Further optimization risks overfitting to 14 points.
- **Go camp:** Opponent modeling is a fundamentally different axis. The project has the data (105 real brackets) but has never used opponent brackets as INPUT to construction. This isn't another prediction technique — it's a different objective function.

### Resolution question

Is opponent-conditional construction a new optimization axis (where the E[points] ceiling doesn't apply)? Or is it another technique that tests well on 14 years and fails to generalize?

## Suggested Actions (ordered by risk/effort)

1. **Permutation test** (Statistician) — 1 hour compute. Validate that 8.0% is real after multiple comparison correction. If yes, proceed. If no, recalibrate.

2. **Opponent-aware construction** (First Principles + Outsider) — High-leverage pivot. Use pool history data (105 real brackets) as input to construction. Optimize E[P(score > max(opponents))] instead of E[score]. Genuinely untested.

3. **Upset specialist** (Executor) — Low-risk targeted intervention. Dedicated classifier for coin-flip R64 matchups. Override beam search selectively. Testable in a day.
