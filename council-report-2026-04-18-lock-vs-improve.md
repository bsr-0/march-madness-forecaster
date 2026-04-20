# Council Report — 2026-04-18 Lock vs Improve for 2027

## Question

With 11 months until the 2027 tournament, should the developer lock the current validated f4_first_tv system and focus on operational readiness, or invest development time in ML/pool optimization improvements?

## Metadata

- **Bucket:** E (genuinely novel — prior council row 64 asked the same directional question but the evidence base has materially changed: rows 65-66 closed the free measurements, O27 closed, all §2 items closed)
- **Panel:** 5 advisors (Contrarian, First Principles, Statistician, Outsider, Executor)
- **Peer review:** skipped (convergence — 4/5 advisors aligned on "lock + focused ops"; no fatal flaw flagged)
- **Timestamp:** 2026-04-18

## Suggested Actions

> 1. **Resolve the BestRank vs MeanRank conflict in 1-2 weeks.** Torvik has best BestRank (27.1); f4_first_tv has best MeanRank (573.6). For winner-take-all, decide which metric is the correct objective and whether a hybrid dominates both. This is the one genuine open thread — not an ML improvement, a portfolio-selection decision.
>
> 2. **Build a single-command submission workflow and do a dry run by January 2027.** The 2026 miss was operational, not analytical. Write `make submit`, document ESPN steps, set calendar reminders for February. Ingest 2026-27 season data monthly starting July.
>
> 3. **Lock the system after action 1 resolves. No new features after resolution.** 17 dead-ends and BSS ≈ 0 prove the improvement space is exhausted. Further development is overfitting on 4 data points.

## Per-Advisor Bottom Line

- **Contrarian:** Option A is premature because the BestRank/MeanRank contradiction is unresolved — torvik dominates on the metric that matters for winner-take-all. Resolve that conflict (1 week), then lock. Don't experiment; decide.
- **First Principles:** The system's job is already done. The remaining variance is tournament randomness, which no code touches. The torvik+f4 hybrid is the one genuine thread worth 2 weeks. Everything else is optimizing against BSS = 0.
- **Statistician:** "Validated" overstates what 4 pool years support. The 4/4 retroactive win rate is inflated by mode selection (17 alternatives tested). The mechanism (portfolio construction, not prediction) is legitimate, but confidence should be "directional lean, not probabilistic guarantee."
- **Outsider:** The product is done. 17 failed improvements proves improvements aren't coming. The 2026 miss is the actual failure — 11 months of model work on 4 years of data is "procrastination wearing a lab coat." Ship it.
- **Executor:** Reframes Option B as an ops calendar, not improvements. First Monday: change default mode to f4_first_tv. Monthly data ingestion July-January. Freeze February 1. Submit March 16. The system has zero production reps — that's the risk.

## Where Advisors Agree Most

- **The ML improvement path is exhausted.** BSS ≈ 0 + 17 dead-ends = no viable prediction-accuracy improvements. All 5 advisors agree further model development has no ROI.
- **Operational readiness is the binding constraint.** The 2026 non-submission is the only real failure. 4/5 advisors cite it as the primary risk for 2027.
- **The torvik vs f4_first_tv question deserves 1-2 weeks, not months.** 3/5 advisors (Contrarian, First Principles, Executor) flag the BestRank/MeanRank divergence as the one legitimate open thread, but all agree it's a focused investigation, not a research program.

## Where Advisors Clash

- **Contrarian vs Outsider on whether to investigate the hybrid at all.** Outsider says ship now, the product is done. Contrarian says you can't lock without resolving which metric you're optimizing for. First Principles sides with Contrarian — the hybrid question is "real edge left on the table."
- **Statistician vs everyone on confidence level.** Statistician warns the 4/4 retroactive result is inflated by mode selection and 4 years is a case series, not validation. Other advisors treat the 4/4 as strong evidence. The Statistician is technically correct but the practical implication is the same: no amount of additional development changes the n=4 constraint.

_Convergence: 4/5 advisors aligned on "lock after 1-2 week focused investigation"; peer review skipped — calibrate confidence accordingly._

## Load-Bearing Assumptions

- **Scoring encoding:** team-identity (real ESPN) — validated by O26/O27
- **Opponent field:** ESPN-national 60/30/10 for backtests; actual ~30-person pool for retroactive analysis
- **RNG / sample count:** rng_seed=42, N=1000 opponents, 50 brackets × 50 repeats
- **Year scope:** 2011-2026 ex. 2020 (14 years for backtest); 2023-2026 (4 years for pool retroactive)
- **Baseline anchor:** f4_first_tv vs all other modes (torvik, champ_first_tv, e8_first_tv, seed, etc.)
- **Data sources:** `pool_hist_results.json`, `artifacts/backtest_brackets/backtest_brackets_{2023-2026}.json`, `artifacts/backtest_runs/mc_pool_backtest_20260418_225840.txt`

## Framed Question + Prior Art

**Core decision:** Lock or improve for 2027, given that f4_first_tv is validated by three independent lines, all investigation items are closed, and the system wasn't ready for 2026.

**Prior art injected:**
- §3 row 7: Strategic pivot (stop optimizing accuracy; optimize pool EV) — unanimous, locked
- §3 row 64: Prior council recommended "free measurements first" — measurements now done (rows 65-66)
- §3 row 65: Retroactive pool analysis — f4_first_tv best bracket wins all 4 years; ranker ρ = −0.50 to −0.75
- §3 row 66: O27 closed — N=1000 team-identity confirms mode reordering
- §1 Pool strategy: BSS ≈ 0 ceiling locked; mode recommendation amended to f4_first_tv
- Dead-ends D1-D17 constrain the space of viable improvements

## Advisor Responses

### The Contrarian
The "Torvik+f4 hybrid" buried in the evidence summary is the thing nobody wants to say out loud: you have two metrics pointing in opposite directions and you don't know why. BestRank vs MeanRank aren't just different metrics — they represent fundamentally different pool strategies. BestRank optimizes for "can this ever win." MeanRank optimizes for "does this usually finish well." For a winner-take-all pool, BestRank is arguably the correct objective. So why is f4_first_tv the recommendation when Torvik dominates on the metric that actually matters for your payout structure? Nobody has answered that. The four-year win rate is also a sample size problem disguised as validation. Retroactive scoring on N=4 years, where you're the bracket author, is not independent. The ρ = −0.50 to −0.75 ranker correlation is real signal — but three out of four years still means one miss ends the story. My verdict: Option A is premature lock-in on unresolved contradictions. Option B is correct only if it starts by resolving the BestRank/MeanRank conflict — not by experimenting, but by deciding which objective function you're actually optimizing for a winner-take-all pool. That decision should take one week, not three months.

### The First Principles Thinker
The question being asked is: "which option should we choose?" But the actual question is: "what would a win in this pool actually require?" Strip everything back. You have a ~30-person winner-take-all pool. The winner needs roughly 3-5σ of luck on top of any systematic edge. Your system's edge is real — four years of retroactive validation is meaningful signal — but it's optimization edge, not prediction edge. The ML ceiling (BSS ≈ 0) is permanent, not a gap to close. Here's the first-principles reframe: your system's job is already done. It selects the right bracket from a portfolio. The remaining variance is tournament randomness, which no development investment touches. The "Torvik best BestRank vs. f4_first_tv best MeanRank" hybrid question is the one genuine open thread — because it's not about improving ML accuracy, it's about portfolio selection strategy. That's worth exactly one focused investigation: run the hybrid against the same 14-year backtest and see if it dominates on both metrics. That's two weeks, not three months. Everything else on the table is the developer solving problems that don't exist because the real constraint — tournament variance — is immune to code. The right answer is D: two weeks to close the hybrid question, then lock and wait.

### The Statistician
The "validated" claim requires much heavier qualification than presented. Four pool years is not validation — it's a case series. With n=4 binary outcomes (win/lose), the exact binomial probability of going 4-for-4 by chance given any non-trivial win probability is non-negligible. At a 15% base win rate (mid-range of the 2-11% portfolio win rates, being generous), P(4/4) ≈ 0.05%. That looks impressive, but you've selected the *best* mode from a menu — f4_first_tv wasn't the only candidate tested. With 17 documented dead-ends, you've run enough comparisons that a 4/4 survivor is unsurprising under the null. The ranker correlation data is stronger but uneven. The 2023 ρ = −0.20 is noise. Three significant correlations from three years, each with n=19-33 brackets, is suggestive but not robust against multiple-year selection effects. The N=1000 team-identity backtest spanning 14 years is the most credible evidence — but BSS ≈ 0 against seed baseline means your ML layer is contributing nothing to game prediction. You're winning through portfolio construction and diversification strategy, not forecasting edge. That's a legitimate mechanism, but the system's validity is entirely load-bearing on the ranker's structural properties, not prediction accuracy. Actual confidence for 2027: directional lean, not probabilistic guarantee.

### The Outsider
The system didn't fail. The process did. Four years of data showing wins. Ranking system finds the winner in top 3 every single time. Seventeen failed improvement attempts. That last number is the loudest signal in this whole description: you've already proven you can't make it better. The improvements aren't coming. But you missed the submission deadline in 2026. That's the actual problem. From where I sit, with no attachment to any of this work, the choice is obvious: Option A, but take the full week seriously. Not as a consolation prize — as the actual job. The "system" is not the ML model. The system is: bracket gets generated, bracket gets submitted, money gets won. Right now that system has a critical failure mode that already cost you one year. Eleven months of model improvements on 4 years of data is not science. It's procrastination wearing a lab coat. Write down the exact steps to submit. Automate what can be automated. Set three calendar reminders in February. Do a dry run in January. That's it. The product is done. Ship it.

### The Executor
Option B, but don't call it "improvements" — call it what it actually is: finishing the job you didn't finish before 2026. The system has never been submitted to a live pool. Zero production reps. That's the only fact that matters. Here's the 11-month calendar: **Now through June (6 weeks):** Fix the one known production gap — f4_first_tv as default mode. Write a `make submit` target that outputs bracket_2027_submission.json from a single command. Document the exact ESPN submission steps. Do a dry run against a fake bracket. **July-January (6 months):** Ingest 2026-27 season data as it comes in. Run the pipeline monthly. Catch data failures early, not on Selection Sunday. **February-March 15:** Freeze the model. No new features after February 1. Run the backtest one final time to confirm f4_first_tv still dominates. Generate the Pareto brackets. **March 16:** Submit. Done. Option C is how you build something that's never ready. Option A assumes ops is already solid — it isn't. The first Monday morning action: change the default mode in the CLI, run the backtest, confirm output.

## Peer Review

_Peer review skipped (convergence — 4/5 advisors aligned on lock + focused ops)._

---

_Council report generated 2026-04-18. What was counciled: lock vs improve system for 2027 tournament — given f4_first_tv validated, all items closed, 2026 not submitted._
