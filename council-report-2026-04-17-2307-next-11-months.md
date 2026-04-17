# Council Report — 2026-04-17 Next 11-Month Direction

## Question

With 11 months until the 2027 tournament, should I continue the recent focused work (opponent-model calibration + scoring-encoding audit from the O25/O26 cycle), or is there a more valuable direction to pivot to?

## Metadata

- **Bucket:** E (genuinely novel strategic-direction question)
- **Panel:** 5 advisors (Contrarian, First Principles, Statistician, Outsider, Executor)
- **Peer review:** ran — 4 of 5 advisors flagged a fatal flaw in the Executor's "continue the roadmap" plan
- **Timestamp:** 2026-04-17 23:07 UTC

## Critical Actions

> 1. **Pull 2026 realized pool standings** and compute where each of the 50 portfolio brackets would have finished in that actual 30-person pool. Gate: if the submitted bracket won or placed top-3, the "selection failure" premise dies and the 11-month calibration plan needs re-justification from scratch.
>
> 2. **Run O27 tonight as planned, in parallel with action 1.** Gate: log the result to MEMORY as a directional data point only — do not let it reorder priorities regardless of outcome.
>
> 3. **Compute the irreducible noise floor** (D's proposal): given perfect opponent knowledge and n=3-4 pool-years, what is the best achievable BestRank distribution? Gate: if the ceiling gap between "current system" and "perfect opponent model" is <3 rank positions, kill the opponent-calibration track and pivot to pool-rules verification and entry-count leverage.

## Chairman's Verdict

### Where the Council Agrees

Four of five advisors (Contrarian, Outsider, First Principles, Statistician) converge that **the 2026 "failure" is a selection/portfolio problem, not a calibration problem** — the system generated the winning bracket, it just couldn't rank it. Three of five reviewers named the Outsider or First Principles as strongest.

Three advisors (Contrarian, First Principles, Statistician) converge on the **power problem**: n=3-4 pool-years and ~9-16% power cannot resolve the effect sizes §3 row 58 claims. The Statistician formalizes this: row 58 is a prior dressed as a finding, confirmed by the year that motivated it.

Four of five reviewers identify the **Executor as the weakest voice** — not because O27 is wrong, but because "has a runnable script" was treated as equivalent to "is the right question."

Agreement is trustworthy because it emerged across different frames (contrarian, first-principles, statistical, outsider) via different routes, then was independently reinforced by non-coordinating reviewers.

### Where the Council Clashes

**Executor vs. everyone else on whether O27 should run tonight.** The Executor says run it — it's the only gated item with a command. Four reviewers disagree.

**I side with the Executor on running O27, but with the others on what it means.** O27 is 2-4 hours of compute and answers a concrete pre-registered question. Blocking it behind philosophical reframing is the kind of paralysis that kills solo projects. But the Executor is wrong that O27's result reorders priorities. Even a clean team-identity signal at effective n=14 is, as the Statistician correctly notes, astrology-adjacent for ranking 11 modes. **Run O27 as bookkeeping, not as a decision input.**

**Contrarian/Outsider vs. First-Principles on opponent intel.** Outsider wants two months scraping 29 humans. First-Principles wants the noise-floor computation first. Reviewer 2 correctly flags that the Outsider hand-waves the legal/availability question. **First-Principles is right on sequencing.** Compute the noise floor in a week before committing two months to opponent scraping that may not be legally or practically available.

### Blind Spots the Council Caught

1. **The 2026 narrative is underspecified.** Reviewer 5's observation is the sharpest in the entire transcript: a 4/4 F4, 1440-point bracket *entered into a 30-person pool* likely won or placed. Nobody — not the user, not any advisor — stated the realized 2026 pool finish. If the submitted bracket won the pool, there is no failure to diagnose.

2. **Pool composition churn.** Reviewer 2 flagged this and it's fatal to pool-scoped calibration. If 2027 has >30% roster turnover, every pool-scoped correlation argument collapses to noise.

3. **Is winning this pool worth 11 months?** Reviewer 1 raised this and it was ignored. The opportunity cost of 11 months of solo-developer time against one 30-person pool is load-bearing.

### The Chairman's Take

The council collectively missed one thing: **the 2026 realized pool result is a free measurement the user already has.** Before O27, before noise-floor analysis, before opponent scraping — pull the actual 2026 ESPN pool standings and compare where the submitted bracket finished vs. where the other 49 portfolio brackets *would have* finished. This is an afternoon of work against already-collected data and it either dissolves the problem (the bracket won) or localizes it (the ranker's #11 was systematically beaten by internal #12-#20).

This precedes Reviewer 4's "expected rank distribution under current system" because it uses realized, not simulated, outcomes.

Where I disagree with consensus: the Outsider's entry-count leverage observation (25 ESPN brackets) is oversold by Reviewers 4 and 5. Multi-entry in a 30-person pool with declared opponents is often against pool rules or social norms. Verify pool rules before treating it as 25× leverage.

The Executor's O27 is cheap enough to just run. The real question is whether the user spends April–June on calibration (current path) or on the three free measurements above. **Free measurements first.** Always.

## Load-Bearing Assumptions

- **Scoring encoding:** team-identity vs. seed-based changes row 58's direction. Invalidated if O27 flips the sign.
- **Opponent field:** 2027 pool resembles 2026 pool. Invalidated if >30% roster churn or platform change.
- **RNG / sample count:** N=1000 MC is sufficient resolution. Invalidated if Monte Carlo SE exceeds the effect sizes being ranked.
- **Year scope:** 4 pool-years carry signal. Invalidated by Statistician's CI — already effectively invalidated; treat all pool-scoped claims as directional.
- **Baseline anchor:** 2026 #11 ranking is a real failure. Invalidated by Action 1 if realized pool finish was top-3.
- **Data sources:** ESPN pool standings and opponent picks are legally/practically scrapable. Invalidated if pool is private or ToS-blocked.

## Agreement / Disagreement Summary

| Advisor | Bottom line | Alignment |
|---|---|---|
| Contrarian | Kill O27 until power analysis justifies it; row 58 rests on n=1. | Dissent from Executor |
| First Principles | Measure the irreducible noise floor before building anything. | Dissent from Executor |
| Statistician | Prioritize power (H2) before H1/H3/H4; pre-register O27 with Bonferroni α=0.0045. | Dissent from Executor |
| Outsider | Two months on opponent intelligence + ESPN multi-entry leverage before pipeline work. | Dissent from Executor |
| Executor | O27 tonight; G4 week 1-2; O21 bootstrap walk-forward week 3-6. | **Lone endorser of status quo** |

## Framed Question + Prior Art

Full prior-art block injected into advisor prompts covered: strategic pivot locked 2026-04-02 (optimize pool EV, not accuracy); 2026 #11-ranking narrative; §2 O27 (only confirmed-open item, 2-4 hr compute); §2 O21 partial-reopen; §2 O26-G4 (blocked on O27); §3 row 58 priority claim ("opponent-model > selection"); and the full dead-end ledger (D1, D6, D8, D12, D13, D14, D15, D16, D17). Power context: n=14 backtest years ≈ 9-16% power; n=4 pool-scoped years.

## Advisor Responses

### The Contrarian
Your #1 failure mode in 2026 wasn't opponent-model mis-calibration. It was that your system ranked a winning bracket 11th out of 50 of its own brackets. That's a selection/portfolio failure, and you just closed D17 and are about to deprioritize the O25 conditional-reactivate path to chase opponent-model calibration. Read that again.

Three fatal flaws: (1) n=14 is not enough to distinguish signal from noise for ANY of this. O26-G1b shows pool-scoped ρ=+0.42 on n=3 years with a 2026 sign flip to −0.60. The §3 row 58 claim rests on one year of evidence (2026). One data point is not a strategic pivot — it's a story. (2) Pool-scoped recalibration overfits to YOUR pool; 2027's pool composition will differ. (3) You haven't asked the real question: can you actually *detect* a 5% EV lift with n=14 and MC noise? Run that power analysis BEFORE O27. Kill O27 until you've justified it can resolve anything.

### The First Principles Thinker
You're asking the wrong question. "Opponent-model calibration vs. portfolio construction vs. pivot" is a menu. Menus are for people who already know what they're optimizing. You don't.

Strip it down. The actual objective function is not "pool EV" — that's a proxy. You want to win a specific pool, once, in March 2027. That's n=1. You're using 15 years of historical data to optimize a one-shot decision against ~50 opponents whose behavior you've measured for 3 years.

The 2026 failure reveals this: you generated the winning bracket. The system found it. It just ranked it #11. That is not a calibration problem — it's a selection-from-portfolio problem masquerading as one. You already have the signal; you can't rank it.

Real question: is #1-ranking a winning-quality bracket among your 50 even *possible* with n=3 pool-scope opponent data? Power analysis says no. Before O27, before O21 reopen, before any of it — compute the irreducible noise floor. Given perfect opponent knowledge, what's the max achievable BestRank? Given n=3 samples, what's the realistic ceiling? If the gap is tiny, 11 months of calibration buys you nothing. Measure the ceiling first.

### The Statistician
Your §3 row 58 claim — "opponent-model calibration upgrades outrank selection-objective search" — rests on n=4 pool-years with 2026 as the motivating datapoint. That's circular: you cannot use the observation that generated a hypothesis as confirmatory evidence for it. Row 58 is a prior, not a finding.

O26-G1b (mean ρ=+0.42, 75% positive, n=4): the 95% CI on that proportion is roughly [19%, 99%]. Coin flip. O26-G3-narrow (N=31 brackets, BestRank reordering): N=31 brackets is not 31 independent observations — they share years, opponent samples, and the same 13 tournaments. Effective n ≈ 13. A "12-17 position BestRank effect" at 9-16% power means false reversals of this magnitude are expected under the null. O21 partial-reopen: +0.058 ρ on n=3 years with variable K is noise with a decimal.

Pre-registered O27 should look like: (1) lock hypothesis, metric, and decision threshold *before* running MC; (2) Bonferroni across 11 modes (α=0.05/11=0.0045, not 0.05); (3) block-bootstrap CIs by year, not by bracket; (4) pre-specify what result kills the opponent-model track vs. extends it.

Honest verdict: your n's support binary directional claims only. Ranking 11 modes by BestRank at this power is astrology. Prioritize **power** (Hypothesis 2) before prioritizing between 1, 3, or 4.

### The Outsider
Walking in cold: you are building a Formula 1 car to win a go-kart race. A 30-person pool, once a year, 4 years of history. That's 4 data points. Your system has 5,900 unit tests.

The "ranked its own winner #11" story is the whole game and everyone around it seems to be missing it. You already have the bracket-generation capability; the bug is in selection, not generation. "1 out of 50, ranked 11th" on a sample size of one year is not a statistical signal — it's an anecdote. You cannot tune a ranker on one observation without overfitting to it. Running a 2-4 hour backtest at N=1000 to chase a correlation moving from 0.37 to 0.61 is bookkeeping dressed as science.

Obvious things a pool player would do that aren't mentioned: (1) Who are the 29 other humans? Names, biases, favorite teams, past picks. That's your actual opponent distribution, not a public-pick proxy. (2) Can you enter multiple brackets? ESPN usually allows up to 25 per account. If yes, portfolio construction against *known* opponents dominates everything else. (3) What did the 10 brackets ranked above #11 look like? Were they correlated? That tells you if your ranker has a systemic blind spot or just got unlucky.

Eleven months is a lot. Spend two of them on opponent intelligence and entry-count leverage before touching the pipeline again.

### The Executor
Monday morning is already decided. O27 runs tonight. It's the only item with a command, a compute budget, and a binary gate. Everything else is gated on its output. Stop deliberating and kick it off.

Ranking by Monday-morning clarity: H1 (continue opponent-model) clearest — step 1 is `scripts/mc_pool_backtest.py` with team-identity scoring, 2-4 hrs. H4 (G4 migration) clear but blocked — queue position 2. H2 (data-scale) half actionable, half fantasy — synthetic bootstrap over 2023-2026 *is* actionable, literally O21's extension. H3 (portfolio tail-diversity) hand-wavy — no metric, no first commit, dead on arrival until "tail-diversity" is defined as a testable function. H5 (pivot to different pool / live re-optimization) — no code, no scope, no first action; 11 months is not enough runway to start from zero when you have a locked strategic direction from 2026-04-02.

80/20 sequence (next 6 weeks): (1) Tonight: O27 run. (2) Tomorrow: read gate result, amend MEMORY, decide on `champ_first_tv`. (3) Week 1-2: O26-G4 migration (unblocked by #1). (4) Week 3-6: O21 bootstrap walk-forward 2023-2026 — the real opponent-calibration payoff. Everything past week 6 is speculation; plan it in week 6.

## Peer Review

### Reviewer 1
**Strongest: D (First Principles).** Reframes correctly as selection problem, proposes the single most leveraged experiment (irreducible noise floor). Decidable gate either justifies 11 months or kills it. E has sharpest statistics but stops at "measure power"; D asks whether objective is achievable.
**Biggest blind spot: A (Executor).** Treats "Monday-morning clarity" as ranking criterion — rewards whichever hypothesis has a pre-written script, not which is correct. Operationally crisp, strategically blind.
**All missed:** Does winning this specific pool actually matter enough to justify 11 months? Define what BestRank distribution you'd accept as "solved."

### Reviewer 2
**Strongest: D.** Reframes before optimizing; irreducible-noise-floor is the only proposal that can cheaply invalidate the entire 11-month plan.
**Biggest blind spot: C (Outsider).** Hand-waves past the fact that scraping 29 humans' ESPN histories isn't a two-month project — it's a legal/availability question C never asks. Also misses that row 58 and O26/O27 are governance artifacts the dev must engage with, not bypass.
**All missed:** Will the 2026 pool even exist in 2027 (same 30 people? same scoring? same platform?). If pool composition churns >30%, every pool-scoped calibration argument collapses.

### Reviewer 3
**Strongest: C (Outsider).** Only response that reframes in terms of actual objective (winning a specific 30-person pool once). Opponent intelligence + multi-entry leverage dominate any calibration gain. D reaches similar conclusions but stays abstract.
**Biggest blind spot: A.** Treats decision as settled and ranks hypotheses by code-readiness rather than expected value.
**All missed:** Was 2026 outcome even diagnostic? Did 10 brackets in the portfolio score higher than 1440 in that specific pool? If chalk won the pool, no calibration fix recovers it. Pull 2026 portfolio scores against actual pool results before choosing any direction.

### Reviewer 4
**Strongest: C.** Only response that questions whether the whole optimization framing is load-bearing. Entry-count leverage (ESPN allows 25 brackets) is the single highest-EV observation in the entire council — potentially 25× the portfolio problem overnight. D/E right that power is insufficient; C right that power doesn't matter if you're optimizing the wrong object.
**Biggest blind spot: A.** Runs O27 tonight without engaging B/D/E power critique or C's entry-count leverage.
**All missed:** Is the 2026 #11 ranking a real signal or single-year artifact? Before power analysis, opponent intel, or O27 — what's the expected rank distribution of a winning bracket under the *current* system? That calibration is free and comes first.

### Reviewer 5
**Strongest: C.** Only response that questions the frame itself and names the actual game: 30 humans, once a year, ESPN allows 25 entries. Opponent intelligence and entry-count leverage strictly dominate any calibration work.
**Biggest blind spot: A.** Pure project-management throat-clearing; optimizes execution order of experiments E proved are astrology at current n.
**All missed:** Whether 2026 result is actually a failure. A bracket that goes 4/4 F4 and scores 1440 ranking #11/50 internally, entered into a 30-person pool, likely *wins or places*. Before diagnosing "selection failure," check: did the submitted bracket win the 2026 pool? Measure realized pool finish vs. internal rank correlation across 4 years before rebuilding anything.

---

_Council report generated 2026-04-17. What was counciled: next 11-month direction — continue opponent-model track (O27 → O21) or pivot._
