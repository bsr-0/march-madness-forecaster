# LLM Council Transcript
**Date:** April 2, 2026
**Question:** "Besides the backtest harness, what is the first thing to prioritize in order to improve this repo?"

---

## Original Question

> Council this: besides the backtest harness, what is the first thing to prioritize in order to improve this repo?

## Framed Question

"In the march-madness-forecaster repo, besides building a backtest harness, what should be the #1 priority to improve the project?"

**Context provided to advisors:**
- NCAA Tournament prediction system with frozen 2026 production pipeline
- Current model: regularized logistic regression on 7 features, trained on ~17,600 regular-season games (2016-2024)
- Calibration: temperature scaling on ~530 tournament games (2016-2025) — only ~60-70 per LOYO fold
- Simulation: 50k Monte Carlo bracket simulations
- Research modules exist but are disabled: GNN, transformer, agent orchestration, seed overrides, Brier sharpening
- Production config enables: stacking, market blend, Bayesian Bradley-Terry, round-weighted calibration, recency weighting, symmetric augmentation, multi-year training, LOYO cross-validation
- 535 Python files in the repo (~122K lines)
- Prior councils identified: (1) stacking weight contamination, (2) calibration on 60-70 tournament games is statistically fatal, (3) README contradicts production config, (4) no baseline comparison against trivially simple models, (5) all complexity additions (GNN/transformers) unanimously rejected by peer reviewers
- The user has already identified "backtest harness" as a known priority

**Stakes:** With 535 Python files and growing complexity, the risk is investing effort in the wrong area — adding features that don't move prediction quality, or ignoring structural issues that undermine everything downstream.

---

## Advisor Responses

### The Contrarian

The other advisors will tell you to fix calibration sample size, add baselines, or clean up dead code. Those are all real problems. But they're not the number one priority. Here's what is:

**You need to delete code, not add it.**

You have 298 Python files in src/ alone. You have 38 directories. You have GNN, transformer, quant, conference tournament, womens pipeline, research loop, agent orchestration, deployment governance, monitoring, reproducibility, optimization, forecasting, evaluation, validation — most of which are disabled or unused in production. This is a codebase for a logistic regression on 7 features that has metastasized into something nobody can reason about.

The fatal flaw everyone is dancing around: **you cannot evaluate whether your pipeline is correct because the codebase is too complex to audit.** The stacking leakage bug proves this — it existed in production and required a council to find it. How many more are hiding in 535 files? You literally cannot know.

The backtest harness everyone agrees on? It will be **wrong** if built on top of this foundation. You'll wire it into the wrong `predict_probability` method (there are at least four: `predict_probability`, `predict_probability_production`, `predict_probability_pool`, `predict_probability_experimental`). You'll test the wrong code path and declare victory.

Priority one: **ruthlessly delete everything not on the production path.** Archive the GNN, transformer, research loop, womens pipeline, and agent orchestration into a separate branch. Get the production pipeline down to something a single person can read top-to-bottom in an afternoon. Then — and only then — can you trust any evaluation you build on top of it.

A model you can't audit is a model you can't trust. No amount of clever features fixes that.

---

### The First Principles Thinker

The question "what should be the #1 priority besides backtest harness" is the wrong question. Here's why.

**You don't have a prediction system. You have 122,000 lines of code wrapping a logistic regression.** That's the actual problem. Not a missing feature, not a missing harness — the system's complexity has outrun its epistemic foundation.

Strip it to first principles: What does this project actually know? It knows ~17,600 regular-season game outcomes and ~530 tournament games. That's it. That's the entire information budget. Every line of code beyond what's needed to extract signal from that data is either (a) extracting real signal or (b) overfitting noise while looking impressive.

You have 535 Python files, 40+ enable flags, Bayesian Bradley-Terry, market blending, bracket portfolio optimization, dual submission strategies, leverage calculations — all sitting on top of a dataset where your calibration folds have 60-70 games each.

**The #1 priority is a ruthless deletion pass.** Not adding anything. Subtracting.

Specifically: establish your bare-minimum baseline (seed-based log-loss, plain logistic regression with 7 features, no bells). Then re-enable each component ONE AT A TIME and measure its marginal Brier improvement with proper held-out evaluation. Anything that doesn't clear a meaningful threshold gets deleted — not disabled, deleted. The 40 enable flags are a code smell that says "we weren't confident enough to commit either way."

122K lines for a tournament prediction system means the complexity itself is the bug. Every additional module is a place where data leakage can hide, where stale assumptions persist, where the next developer (including future-you) loses the thread. The prior councils already found stacking weight contamination — that's not a one-off, that's a symptom. In a 535-file codebase with this little training data, there are almost certainly more leakage paths nobody has traced yet.

**The real question isn't "what should we add?" It's "how much can we remove before prediction quality drops?"** The answer will be sobering and informative.

---

### The Expansionist

**The #1 priority is building a real-time market integration and public pick exploitation layer — and it's not even close.**

Everyone else on this council will tell you to fix leakage, simplify the codebase, or add baselines. Those are defensive moves. Here's the offensive one nobody's talking about: **you already have the scaffolding for market blend and public pick data, but you're barely using it.** The `betting_markets.py` scraper exists. The `competitor_archetypes.py` module models how the public picks. The `market_blend_weight` is hardcoded at 0.20. This is a gold mine sitting behind a locked door.

**Why this is the highest-upside move:**

1. **Market prices are the single strongest predictor of tournament outcomes** — stronger than any feature engineering you can do with box scores. Your 26-feature logistic regression is fighting for the last 2% of signal. Markets hand you the first 80% for free.

2. **The contrarian EV optimizer only works if your crowd model is accurate.** You have an entire `pool_optimizer.py` and `competition_simulation.py` system designed to find +EV contrarian picks — but it's only as good as the public pick distribution feeding it. Right now that data pipeline is a stub. A properly calibrated crowd model turns your whole simulation stack from "interesting exercise" into "actual edge."

3. **Market data solves your calibration crisis.** You have 60-70 tournament games per LOYO fold — statistically insufficient for calibration. But sportsbooks have calibrated against millions of dollars. Blending with market-implied probabilities gives you calibration for free, no sample size required.

The move: build a robust, multi-source market data pipeline with historical backfill, replace the hardcoded 0.20 blend weight with a learned-on-holdout weight, and use the divergence between your model and markets as a feature itself. That divergence signal — "where does my model disagree with sharp money?" — is where the real alpha lives.

---

### The Outsider

I'm going to say something that might sound rude: I have no idea what this project actually does when I look at it, and that's the problem.

You have 535 Python files. You have a README that lies about what the system does. You have research modules that are disabled. You have production config that enables features the documentation doesn't mention. You have prior reviewers who unanimously rejected your complexity additions, and yet those 535 files still exist.

From the outside, this looks like a project that has lost track of what it is.

My #1 priority: **Delete everything that isn't production.** Not disable. Delete.

Here's why. You said the GNN, transformer, agent orchestration, and seed overrides were all rejected by peer reviewers. They're still in the repo. That means every person who opens this codebase — including future you — has to mentally filter "is this real or is this dead?" across 535 files. That's not a minor annoyance. That's a structural problem that makes every other improvement harder.

You can't fix calibration sample size if you can't clearly see what the calibration code actually touches. You can't verify stacking weight contamination if you're not sure which stacking implementation is the live one. You can't compare against a simple baseline if you can't even describe what the current model is (your README proves this — it doesn't know either).

The backtest harness you already identified? It'll be half as hard to build if the codebase is half the size.

Strip it to what runs in production. Make the README match reality. Get to a point where someone with zero context (me) can open the repo and understand the pipeline in 15 minutes. Then improve it.

Complexity you can't explain is complexity you can't trust.

---

### The Executor

**The #1 priority is building and running the seed-only baseline comparison end-to-end, and killing anything that doesn't beat it.**

Here's why this is the Monday morning task, not the backtest harness:

The repo already has the pieces. `src/evaluation/baselines.py` defines `seed_baseline_probability()` with historical seed win rates. `src/ml/evaluation/loyo_protocol.py` has the LOYO infrastructure. But there's no evidence these are wired together into a single command that says: "Here's what seed-only gets on each LOYO fold, here's what the full pipeline gets on each fold, and here's whether the difference is statistically significant."

The LOYO protocol file itself confesses the problem on lines 9-23: 58 tuned constants against ~440 evaluation games gives a DoF/sample ratio of 0.13 (target is <0.01). The standard error across folds is ~0.009. This means **you literally cannot tell if your full pipeline beats seed-only** without a disciplined head-to-head comparison using the same folds and the same statistical test.

The backtest harness is infrastructure. The baseline comparison is the diagnostic that tells you whether 297 source files of complexity are doing anything useful. You could discover Monday afternoon that seed-only ties your full pipeline on honest LOYO Brier. That would change every subsequent decision about where to invest effort.

Concrete steps:
1. Wire `seed_baseline_probability` into the LOYO loop as a comparison model
2. Run both through identical folds, collect per-fold Brier scores
3. Apply `compute_ablation_threshold` to the paired differences
4. If the full pipeline doesn't beat seed-only at p<0.05, everything else is noise and you need to strip back before adding anything

This takes a day, not a week. And the answer determines whether you're building on rock or sand.

---

## Peer Review Round

**Anonymization mapping:**
- Response A = The Executor
- Response B = The Outsider
- Response C = The First Principles Thinker
- Response D = The Contrarian
- Response E = The Expansionist

### Reviewer 1

**Strongest: Response A (Executor).** It identifies the same fundamental problem as B/C/D but prescribes a concrete, scoped, one-day action with a statistical test. It converts the philosophical concern into a measurable gate. If seed-only wins, that result justifies the deletion passes — with evidence instead of intuition.

**Biggest blind spot: Response E (Expansionist).** It proposes adding complexity to a codebase that already has proven leakage and 535 files of unverified complexity. It also assumes market data is freely available with historical backfill — tournament betting lines are notoriously hard to source historically.

**All missed:** Nobody addressed the test suite. With 535 files and known leakage bugs, how much automated test coverage exists? Deletion without tests risks breaking the production path silently. The project needs test infrastructure before or alongside any of these priorities.

### Reviewer 2

**Strongest: Response A (Executor).** Most actionable, highest-leverage step. Cites specific files, quantifies overfitting risk (DoF/sample ratio 0.13), scopes to a day. Everything else is premature without this evidence.

**Biggest blind spot: Response E (Expansionist).** Completely ignores known problems. Adding market data to a leaky, overfit pipeline doesn't fix it — it masks it. Market data availability for historical backtesting is limited.

**All missed:** The calibration sample size problem is structural and unsolvable by code changes. ~60-70 tournament games per fold is a hard constraint of NCAA tournament format. The project should consider adopting wide confidence intervals and communicating irreducible uncertainty rather than chasing precision the data cannot support.

### Reviewer 3

**Strongest: Response A (Executor).** Only response with a falsifiable action with concrete outcome. One day of work, clear binary result.

**Biggest blind spot: Response E (Expansionist).** Adds complexity at the worst possible time. Building on a foundation you can't trust makes the new feature unverifiable.

**All missed:** Power analysis. With ~60-70 tournament games per fold, any calibration or evaluation metric has enormous variance. The project should know whether improvements can even be detected statistically before building anything.

### Reviewer 4

**Strongest: Response A (Executor).** Only one that proposes a falsifiable test with acceptance criterion (p<0.05). Demands proof first. Cheapest to execute.

**Biggest blind spot: Response E (Expansionist).** Ignores known problems entirely. Highest-effort proposal with least diagnostic value.

**All missed:** The overfitting math. 58 tuned constants calibrated on ~60-70 games per fold is not a modeling choice — it's a statistical impossibility. The effective degrees of freedom likely exceed the sample size. The project may need fewer parameters, not fewer files.

### Reviewer 5

**Strongest: Response A (Executor).** Measurement vs activity. Produces an actionable binary outcome. Every other response is activity; this one is measurement.

**Biggest blind spot: Response E (Expansionist).** Sequentially wrong. Directionally interesting but impossible to validate on current foundation.

**All missed:** Calibration sample size. Confidence intervals on reliability diagrams at N=60 are wider than the bins themselves. No response confronts the power analysis question.

---

## Chairman Synthesis

### Where the Council Agrees

**The Executor wins unanimously.** All five peer reviewers independently ranked the Executor's proposal as the strongest. The convergence point: before you improve anything, prove the current system beats a trivial baseline with statistical significance.

**The Expansionist is sequentially wrong.** Every reviewer flagged the market integration proposal as the worst idea right now. Adding complexity to an unverifiable codebase is building on quicksand.

**Deletion is necessary but secondary.** The Contrarian, First Principles Thinker, and Outsider all converge on ruthless deletion. The reviewers agree this matters but rank it behind measurement. You cannot know what to delete until you know what actually contributes to prediction quality.

**The codebase is untrusted.** Stacking leakage was already found. The README contradicts reality. There are four different predict_probability methods. No advisor defended the current state of the codebase as auditable or trustworthy.

### Where the Council Clashes

**Delete first vs. measure first.** The Contrarian and First Principles Thinker say: clean the codebase so the backtest harness can be built correctly. The Executor says: wire the baseline comparison now, because deletion without measurement is just aesthetic preference. Both are right about the problem. The disagreement is about sequencing. The Executor's path is faster and produces a decision gate: if the full pipeline doesn't beat seed-only, the deletion conversation becomes "delete almost everything" rather than "delete the inactive modules."

**How much of the codebase is salvageable.** The Outsider and First Principles Thinker imply very little. The Executor implicitly assumes the core LOYO and baseline machinery works well enough to produce a valid comparison. If the leakage bug or wiring issues contaminate the evaluation loop itself, the Executor's one-day estimate becomes optimistic.

### Blind Spots the Council Caught

**The overfitting math is damning.** 58 tuned constants calibrated against ~60-70 tournament games per LOYO fold is a statistical impossibility. You cannot fit 58 parameters to 65 observations and claim generalization. The project may need fewer parameters, not fewer files. Realistic capacity: probably 5-8 parameters.

**Power analysis is missing.** If your evaluation set is 60-70 games per fold, confidence intervals on Brier score differences are wide. You might build the perfect baseline comparison and get an inconclusive result.

**Calibration sample size is a hard constraint.** No code change fixes the fundamental problem that NCAA tournament calibration data is tiny. The project may need to adopt wide confidence intervals as a permanent feature of its outputs rather than pretending at precision it cannot achieve.

**No test suite.** Deletion without tests risks breaking production silently. The codebase has no safety net for refactoring.

### The Recommendation

**Wire the seed-only baseline into the LOYO evaluation loop and run it.** This is the single highest-leverage action because it produces a binary decision gate:

- **If the full pipeline does not beat seed-only at p < 0.05:** Most of the 122K lines of code, 58 tuned constants, and 40 enable flags are noise. The correct next step becomes radical simplification.
- **If the full pipeline does beat seed-only significantly:** You have proof of value and can proceed to the deletion pass with confidence about what the production path actually is.

Go in with eyes open about the overfitting problem. If you have 58 tuned constants and 65 evaluation games, the baseline comparison may tell you the pipeline "wins" because it is overfit to the evaluation set. Consider running the comparison with a held-out year that was never used during any tuning.

### The One Thing to Do First

Wire `seed_baseline_probability` into the existing LOYO loop in `loyo_protocol.py`, collect per-fold Brier scores for both the seed baseline and the full pipeline, and run a paired statistical test. Do not tune anything. Do not clean anything. Do not delete anything. Just measure. The answer determines every decision that follows.

---

*Council #5 for march-madness-forecaster — April 2, 2026*
