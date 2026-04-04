# LLM Council Transcript — Session #7
**Date:** April 4, 2026
**Topic:** Review EV Pool Optimization Status — Can We Proceed?

---

## Original Question

> Based on the last few council transcripts, we needed public picks data to test strategies for EV pool optimization. Now that we have the public picks data, can we continue with testing and improving this repo? Were there other critical issues to address first? Provide an update on the status of this repo.

---

## Framed Question

**Core Decision:** The March Madness Forecaster repo has pivoted from prediction accuracy to bracket pool EV optimization. We now have 19 years of real ESPN public picks data (2008-2026). Can we proceed with testing EV pool optimization strategies, or must we fix critical issues first?

**Context:**
- 6 prior council sessions established: ML predictions add zero value over seed-based lookups (BSS=0 across 17 years). The repo's mission is now game theory: optimize bracket selection against pool opponents.
- The opponent model blends ESPN picks (60%), ratings-derived (30%), seed-based (10%) — weights arbitrary, never empirically validated.
- Team-name matching between ESPN picks (snake_case: `brigham_young`, `saint_mary_s__ca`) and codebase team IDs is a **confirmed bug**. Silent fallback to SEED_PICK_RATES when matching fails. Per-year match rates unknown.
- Stochastic backtest showed all modes near-random (rank ~490/1000) — but this was BEFORE real ESPN data was wired in.
- Orphaned `data/kaggle/public_picks.json` exists with no consumers.
- Some files are enormous (139K, 81K, 69K lines) making them unauditable.
- Stacking weight contamination was identified in Session 1 but never fixed.
- 203 test files exist but team-name matching has no validation tests.
- Tournament structure changed (64→68 in 2011), COVID gap in 2020, ESPN methodology evolved.

**What's at stake:** If we run backtests with broken team-name matching, we get plausible-looking numbers that are silently wrong. We'd optimize strategies against garbage opponent models and never know. But if we over-engineer the fix, we lose momentum on the actual goal.

---

## Advisor Responses

### The Contrarian

The team-name matching bug is not a "known issue to fix later." It is the load-bearing wall of the entire project, and it is cracked.

Here is the specific failure mode nobody is confronting: when team names fail to match, the system silently falls back to SEED_PICK_RATES. That means for every failed match, your "opponent model" is not using ESPN data at all — it is using the same generic seed-based baseline you already proved has zero predictive edge. You do not know how many teams fail to match per year. Could be 5%. Could be 40%. You have no telemetry. You are flying blind and calling it a strategy.

The 60/30/10 weighting is a distraction argument. Yes, those weights are arbitrary. But arbitrary weights on real data are recoverable. Arbitrary weights where the "real data" channel is silently dropping to zero for an unknown fraction of teams? That is not a modeling choice. That is a data pipeline bug wearing a lab coat.

The "let's not over-engineer" framing is a trap. Nobody is asking for over-engineering. The ask is: run your team-name matcher against all 19 years of ESPN data, log the match rate per year, and report the numbers. That is a single diagnostic script. If match rates are above 90%, fine, proceed with backtests and flag the misses. If they are below 70% for pre-2015 data, your backtests for those years are fiction.

Do not run a single backtest until you have match-rate diagnostics. Every "plausible-looking number" generated before that diagnostic exists is a liability, not an asset. You will anchor on those numbers. You will build strategy on those numbers. And when you finally discover the match rates were garbage, you will not throw away the conclusions — you will rationalize keeping them. That is how bad decisions calcify.

One script. One afternoon. No excuses.

### The First Principles Thinker

The question isn't "fix bugs or test strategies." The question is: **do you have an opponent model at all?**

Strip everything away. The entire value proposition of EV optimization is: "I know what field opponents will pick, so I can pick differently where I have edge." If your opponent model is wrong, you don't have a contrarian strategy — you have a random one. And random strategies in bracket pools have negative EV after accounting for chalk bias.

Here's what's actually true right now:

1. **You don't know your match rate.** Team-name matching silently falls back to seed-based rates. You could be running 90% on seed fallback and wouldn't know. That means your "opponent model" might literally be the seed-based model with extra steps.

2. **The weights are made up.** 60/30/10 blend with no empirical basis. Even if matching worked perfectly, you're blending a real signal (ESPN picks) with two synthetic signals at arbitrary ratios. Why?

3. **You can't validate without knowing ground truth.** What does a "correct" opponent model look like? You need a metric: does the opponent model predict actual pool opponent behavior better than seed-based alone? Without that metric, you're optimizing against a phantom.

**The real first principle:** Before testing any strategy, you need exactly one thing — proof that your opponent model is more accurate than the trivial baseline (seed lookup). If it isn't, everything downstream is theater.

The fix is surgical, not over-engineered:
- Log match rates per year. One afternoon of work.
- Compare opponent model predictions against actual tournament pick distributions (ESPN publishes these).
- If match rates are below 80%, the model is seed-based with noise. Stop and fix matching.

**You're not choosing between fixing and testing. You can't test what doesn't exist yet.**

### The Expansionist

The broken team-name matching isn't a blocker — it's the single highest-leverage unlock in this entire project. Here's why everyone is framing this wrong.

You have **19 years of real ESPN public pick data**. That is an extraordinary dataset. Nobody else in the bracket pool optimization space has calibrated opponent models against nearly two decades of actual revealed preferences from millions of ESPN users. The moment that data flows cleanly into your opponent model, you have something genuinely differentiated.

Fix the name matching — yes — but don't treat it as a chore. Treat it as **the feature**. Once match rates hit 95%+, you can:

1. **Validate the 60/30/10 blend empirically.** You have ground truth now. Run the backtest with ESPN-only opponent models (100/0/0) versus the blend. I bet ESPN-only wins, and that simplification eliminates an entire class of contamination bugs.

2. **Discover year-over-year drift in public pick behavior.** Has chalk bias increased since 2008? Do mid-majors get more love post-social-media? This is publishable-quality analysis that falls out for free once the pipe is clean.

3. **Build contrarian signals that actually compound.** The whole EV optimization thesis is "go where the field isn't." With clean ESPN data across 19 years, you can identify which seed matchups the public systematically misprices — not theoretically, but historically. That's your alpha.

The stochastic backtest showing near-random results? That's the **bullish** signal. It means the system was running on garbage opponent data and producing noise. Clean data flowing through the same pipeline should produce actual signal separation between strategies.

**My recommendation:** Spend 2-3 focused hours on name matching with per-year match-rate logging. Then run ESPN-only opponent models in the backtest. Skip the blend entirely for now. The upside of clean public pick data dwarfs everything else on the roadmap.

### The Outsider

The team-name matching bug is the only thing that matters right now. Everything else is noise.

Here's what I see as someone with no history here: you have a system where the entire value proposition depends on modeling what opponents will do. Your opponent model is fed by ESPN picks data. The pipe between ESPN data and your model is broken — names don't match, it silently falls back to generic seed rates, and nobody knows how often this happens.

You could be running 100% on fallback data right now and you'd never know. That's not a minor issue. That's "the feature doesn't exist yet" masquerading as "the feature has a small bug."

Three things jump out to me as a complete outsider:

**You don't know what you don't know.** Before fixing anything, instrument the match rate. Log how many teams match and how many fall back, per year, for all 19 years. This is a few hours of work. If match rates are 95%+, the bug matters less. If they're 40%, you've been running on fiction. You literally cannot make any other decision without this number.

**The "momentum" argument is a trap.** You said you're worried about losing momentum if you over-engineer the fix. But running backtests on broken data isn't momentum — it's motion. You'll generate charts, tune parameters, draw conclusions, and then have to throw all of it away when you discover the inputs were wrong.

**The 203 tests with zero name-matching coverage tells me something.** The test suite is testing the parts the team finds interesting, not the parts that are fragile. That's a very human pattern, but it means your safety net has a hole exactly where you're most likely to fall.

Get the match rate numbers. That's step one. Everything else is premature.

### The Executor

Fix the team-name matching. That's your Monday morning task. Everything else is noise until that's done.

Here's why this is non-negotiable: your entire EV optimization pipeline flows through the opponent model. The opponent model is 60% ESPN picks. If team-name matching is broken, ESPN picks silently fall back to seed-based rates, which means your opponent model is actually ~70% seed-based instead of the 60/30/10 blend you think you have. You'd be optimizing against a phantom opponent distribution. Every backtest result downstream is untrustable.

**The concrete path:**

1. **Audit match rates first.** Write a script that, for each of the 19 years, attempts to match ESPN team names to your codebase IDs and prints the match rate. This takes 2-3 hours max. You'll know immediately how bad the damage is. If match rates are 95%+, you got lucky and can move fast. If they're 60%, you have a serious problem.

2. **Build a name-mapping dictionary.** Don't write clever fuzzy matching. Make a literal lookup table: `brigham_young` → `BYU`, `saint_mary_s__ca` → `Saint Mary's`. There are ~350 D1 teams. The ones that appear in the tournament are maybe 120 unique names across 19 years. This is an afternoon of manual work, not an engineering project.

3. **Add validation tests for the mapping.** One test per year confirming all 64/68 teams match. This is your regression guard.

4. **Re-run the stochastic backtest.** Now your numbers mean something.

Skip the orphaned files, skip the enormous file cleanup, skip the stacking weight issue. Those are real problems but they don't block your critical path. The team-name matching does. You cannot interpret any EV result until this is fixed.

Don't parallelize this with strategy testing. Serial execution. Fix the pipe, then run water through it.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Expansionist
- Response B = The Outsider
- Response C = The Contrarian
- Response D = The Executor
- Response E = The First Principles Thinker

### Review 1
**Strongest: Response C (Contrarian)** — Names the exact failure mode and correctly frames the diagnostic as trivial (one afternoon, one script). Also identifies the anchoring risk: plausible-looking backtest numbers from broken data will be harder to discard later than no numbers at all. That's the most dangerous outcome and only C flags it explicitly.

**Biggest blind spot: Response A (Expansionist)** — Romanticizes the bug as "the single highest-leverage unlock" and calls near-random backtest results a "bullish signal." That's motivated reasoning. Near-random results are equally consistent with the hypothesis that the entire EV optimization approach has no edge even with clean data.

**All missed:** Nobody questions whether the 64-to-68 tournament expansion in 2011 invalidates pre-2011 data for backtesting. Play-in games change seed-based fallback distributions, bracket structure, and team counts. If the code assumes 64 teams for years that had 68, the simulation is structurally wrong for 15 of 19 years — independent of name matching.

### Review 2
**Strongest: Response C (Contrarian)** — The "liability you'll anchor on" framing is the most operationally important insight. D is a close second for giving a concrete execution plan, but C nails the *why* behind the urgency.

**Biggest blind spot: Response A (Expansionist)** — Assumes clean matching will produce signal. That is wishful thinking. Fixing the pipe does not guarantee the water is valuable. E rightly questions whether an opponent model even exists in meaningful form.

**All missed:** None asked what the silent fallback actually does. If fallback returns seed-based picks, the backtest isn't random — it's testing a seed-based opponent model, which might be a reasonable baseline anyway. Understanding the fallback behavior determines whether existing backtest results are garbage or merely noisy.

### Review 3
**Strongest: Response C (Contrarian).**

**Biggest blind spot: Response A (Expansionist)** — Calling broken matching "the feature" reframes a bug as a deliverable without evidence.

**All missed:** Nobody asked whether the silent fallback produces systematically biased results rather than random noise. The diagnostic script needs to measure not just match rates but WHERE failures cluster — by round, by seed, by year. Uniform 20% fallback and 20% fallback concentrated in first-round upsets are completely different failure modes.

### Review 4
**Strongest: Response C (Contrarian).**

**Biggest blind spot: Response A (Expansionist)** — "Near-random backtest is bullish" is backwards reasoning. You cannot infer signal quality from noisy output without first measuring how noisy the input actually is.

**All missed:** None asked what the silent fallback actually returns. If fallback produces zeros, the model ignores public picks entirely. If fallback produces default/average values, you're diluting signal toward the mean. The shape of the failure determines whether existing results are "noisy but directional" or "completely fictional."

### Review 5
**Strongest: Response C (Contrarian).**

**Biggest blind spot: Response A (Expansionist)** — Assumes clean data will produce signal separation, which is pure speculation. Wants to skip the blend based on zero evidence ESPN-only is sufficient. Replacing one unvalidated assumption with another.

**All missed:** Nobody asked whether the 19 years of ESPN data has consistent format across all years. ESPN has redesigned its bracket game multiple times. Naming convention shifts (UConn vs Connecticut), scraping artifacts — diagnostic should validate source data integrity, not just match rates.

---

## Chairman Synthesis

### Where the Council Agrees

**Unanimous: Do not run backtests until you know your match rate.** All five advisors, despite wildly different temperaments, converged on the same conclusion: the silent fallback to SEED_PICK_RATES is not a minor bug, it is a potentially total pipeline failure. The Contrarian called it "a pipeline bug wearing a lab coat." The First Principles Thinker went further: "you can't test what doesn't exist yet." The Outsider raised the possibility that match rates could be 0% — meaning the ESPN data feature literally does not function. Nobody dissented.

**Unanimous: This is a small fix, not a large one.** Every advisor estimated hours, not days. The Executor laid out a concrete 2-3 hour audit. The Contrarian said "one script, one afternoon." Nobody argued this requires architectural rework. The fix is diagnostic first, then a lookup table — not fuzzy matching, not ML, not refactoring.

**Unanimous: The "momentum" argument is a trap.** Running backtests on a broken pipeline produces numbers that feel like progress but are actively harmful. You will anchor on those numbers. You will rationalize keeping them. The Contrarian's "liability you'll anchor on" was cited as the single most important insight by four of five peer reviewers.

### Where the Council Clashes

**What does clean data actually buy you?** The Expansionist believes near-random stochastic results are bullish — garbage in, noise out implies clean data will produce signal. Four peer reviewers flagged this as wishful thinking. The First Principles Thinker is more cautious: clean data might reveal that EV optimization against real opponent distributions still has no meaningful edge. The honest answer is unknown, and that is precisely why you need the diagnostic first — to find out whether you have a product at all.

**Skip the blend or validate it?** The Expansionist says ditch the 60/30/10 blend entirely and go ESPN-only. The First Principles Thinker wants to validate the blend empirically. The Executor says ignore blend weights for now and just fix the pipe. The Expansionist's instinct is probably right directionally — ESPN data is your strongest signal — but peer reviewers correctly noted that jumping to ESPN-only without validation is replacing one unvalidated assumption with another. Validate first, then simplify.

### Blind Spots the Council Caught

**The shape of failure matters, not just the rate.** No advisor initially asked WHERE matches fail. Peer Review 3 caught this: 20% uniform fallback across all rounds is a very different problem than 80% fallback concentrated in early rounds (where ESPN pick rates matter most for pool strategy). The diagnostic must log failures by year, round, and seed — not just a single aggregate number.

**The silent fallback might be accidentally reasonable.** Peer Reviews 2 and 4 both flagged something nobody asked: what does SEED_PICK_RATES actually return? If it returns seed-based pick distributions, and if real ESPN picks roughly track seeds for most matchups, the fallback might produce a "noisy but directional" opponent model rather than a completely fictional one. This changes the urgency calculus. It could mean existing results are degraded but not worthless — or it could mean the feature adds literally nothing. You need to check.

**Source data integrity across 19 years.** Peer Review 5 caught that ESPN naming conventions have likely shifted over 19 years (UConn vs Connecticut, scraping artifacts, format changes). The diagnostic should validate the source data itself, not just the matching logic. A perfect matcher against corrupted input data is still broken.

**The 64-to-68 expansion in 2011.** Peer Review 1 flagged that pre-2011 tournament data has a structurally different bracket. This may silently break assumptions in the backtest framework independent of name matching. The diagnostic should flag any structural anomalies in pre-2011 data.

### The Recommendation

**Fix the pipe before you run water through it.** This is not a close call. The council is 5-0 with converging peer reviews.

Spend one focused session (2-4 hours) on diagnostics and the fix, in this order:

1. **Measure the damage.** Write a diagnostic script that loads ESPN data for each year (2008-2026), attempts the current matching logic, and logs: match rate per year, failures by round and seed, what the fallback actually returns, and any source data anomalies (name format shifts, missing data, pre-2011 structural differences).

2. **Build a lookup table.** Not fuzzy matching. A literal dictionary mapping ESPN names to your internal names. There are roughly 120 unique tournament teams across 19 years. This is a manual-but-reliable afternoon task.

3. **Add per-year validation tests.** Each year's ESPN data should have a test confirming 100% team matching. This is the test coverage the 203-test suite is missing — testing what's fragile, not what's interesting.

4. **Re-run the stochastic backtest.** Compare results to the previous near-random output. If signal separation appears, you have a product. If results remain near-random with clean data, the problem is architectural, not data quality — and you have saved yourself from optimizing against noise.

Do not touch blend weights. Do not clean up file sizes. Do not address stacking contamination. Serial execution on the critical path only.

### The One Thing to Do First

**Write the diagnostic script that prints match rates per year, with failures broken out by round and seed.** Not the fix — the measurement. You need to see the number before you decide anything else. If match rates are 95%, the urgency drops and you can be surgical. If match rates are 30%, the ESPN feature does not exist and never has. The number determines everything downstream. One script. Run it. Read the output. Then decide.
