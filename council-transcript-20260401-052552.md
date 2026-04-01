# LLM Council Transcript — Most Critical Repo Limitation

**Date:** April 1, 2026
**Methodology:** Karpathy LLM Council (5 independent advisors → anonymous peer review → chairman synthesis)

---

## Original Question

"Council this: identify the most critical limitation of the repo"

## Framed Question

**What is the most critical limitation of the march-madness-forecaster repository?**

**Context:** This is a Python-based NCAA Tournament prediction system with a locked production pipeline for 2026. It features: an 86-dim feature vector → LightGBM/XGBoost/LogisticRegression ensemble with stacking → temperature calibration → 50k Monte Carlo simulations → bracket optimization. Training uses LOYO cross-validation on 2016-2024 data, holdout is 2025.

**Known issues from existing audits:**
- No end-to-end optimization loop — production is single-pass; ExperimentWorkflow is a plan doc only, never implemented. Research infra (ExperimentScheduler, HypothesisRegistry, AblationStudy) exists as disconnected components.
- Objective function misalignment — models train on log-loss, Kaggle scores on Brier, bracket pools score on ESPN points (exponentially weighted by round). No component bridges these. Stacking optimizes Brier while base models optimize log-loss.
- Non-nested calibration — temperature scaling fit and evaluated on same LOYO predictions.
- Fragile scrapers — hardcoded HTML selectors, non-atomic file writes, API keys in query params, hardcoded event IDs that expire yearly.
- Massive file sizes — baseline_training.py ~139k chars, data_loader.py ~81k chars, sota.py ~2600 lines. Hard to maintain/reason about.
- Test suite can't run (numpy not installed in current env), code coverage threshold only 20%, mypy permissive (no untyped defs), ruff ignores unused imports/variables.
- Dead research modules — GNN, transformer, agent orchestration all disabled but still in codebase adding complexity.
- Small training set — ~400 tournament games across 8 years for a 91-feature model.
- No bracket-EV feedback — simulation results never influence training or calibration.

**What's at stake:** This system is used to generate tournament predictions and Kaggle submissions. Its credibility depends on prediction accuracy, and its maintainability determines whether it can improve year over year.

---

## Advisor Responses

### 1. The Contrarian

The small training set isn't a known issue — it's a death sentence dressed up in conference clothes.

~400 games across 8 years feeding a 91-feature model. That's less than 5 observations per feature at the absolute ceiling, assuming zero correlation between features. In practice, correlated features mean your effective degrees of freedom are far lower. You're not doing machine learning. You're doing aggressive overfitting with extra steps.

And here's what makes it worse: the LOYO cross-validation is doing the work of convincing everyone the model generalizes. It doesn't. LOYO on 8 years gives you 8 test folds. Eight. That's not validation — that's a sample size so small that variance in model performance is completely indistinguishable from variance in tournament randomness. A coin flip model could "validate well" on 8 LOYO folds.

The stacking layer compounds this catastrophically. You're training a meta-learner on 8 data points of out-of-fold predictions. That meta-learner's weights are noise, full stop.

The objective misalignment and non-nested calibration issues everyone has catalogued? Those matter *if your model has signal*. If it doesn't, those are rounding errors on zero.

The existential question this system can't answer: is the model learning basketball, or is it memorizing the 2016-2024 era's specific distribution of upsets, seeding accuracy, and conference strength — which shifts every 3-5 years as the sport evolves?

You can't know. Not on 400 games. Not ever.

Everything else in this system is scaffolding built on a foundation that cannot be verified. That's the critical limitation.

### 2. The First Principles Thinker

The question being asked is "what is the *most critical* limitation?" But that framing assumes the limitations are independent. They're not. Strip it back further: what is this system actually trying to do?

It is trying to maximize bracket scoring outcomes (ESPN points, Kaggle Brier) by learning from historical tournament data.

Now rebuild from scratch: what does that require?

1. A training signal that matches the deployment objective
2. Enough data to learn a generalizable signal
3. A feedback loop connecting predictions to outcomes to improvement

Every listed limitation collapses into one root failure: **the system has no coherent objective function that connects training to deployment.**

Log-loss training → Brier scoring → ESPN exponential points. These are three different optimization surfaces. The model is being trained to minimize one thing, evaluated on a second, and deployed for a third. No amount of engineering refinement — better scrapers, more tests, refactored files — fixes this. You cannot tune your way out of optimizing the wrong thing.

The small dataset (~400 games) is a real constraint, but it's downstream of this same problem. Even with more data, if the gradient signal doesn't point toward bracket EV, more data just means more confidently wrong.

The actual question the team should be asking isn't "what's the most critical limitation?" It's: **"What are we optimizing for, exactly, and does every component point at that target?"**

Until that question has a single, honest answer — and the loss function, calibration target, and simulation objective all align with it — everything else is rearranging deck chairs.

### 3. The Expansionist

The most undervalued opportunity here isn't fixing what's broken — it's recognizing what this system is *almost* capable of but isn't being pushed toward.

The 50k Monte Carlo simulation engine is the crown jewel nobody is exploiting. Right now it terminates at bracket generation. That's like owning a Formula 1 car and only using it to commute. Those simulations contain a full probability distribution over *every possible tournament outcome* — that's a live, queryable oracle for expected value across any scoring format simultaneously.

The critical limitation isn't the small training set or the log-loss/Brier mismatch — those are tractable engineering problems. The limitation is that the simulation layer is disconnected from the objective. You have a machine that can answer "what's my EV under ESPN bracket scoring if I pick this upset?" and it's not being asked that question. Ever.

Wire the Monte Carlo output directly into a bracket-EV optimizer that knows the actual point structure. Suddenly the system isn't just "predicting games" — it's solving a portfolio optimization problem under uncertainty. That's a fundamentally different and far more powerful product.

Beyond that: the multi-format angle is being completely ignored. The same simulation engine could simultaneously optimize for Kaggle Brier, ESPN bracket pools, and custom office pools with different structures. One simulation run, multiple optimized outputs. That's leverage most competitors will never think to build.

The dead GNN and transformer modules aren't liabilities — they're optionality waiting for more data. As tournament game counts grow (expanded fields, more years), those architectures become viable. Don't delete them. Document them and treat them as version-gated features.

The upside here is a system that doesn't just predict — it *prescribes* strategy per scoring context. Nobody else is building that.

### 4. The Outsider

The thing that jumps out immediately, before any of the technical details: **you have roughly 400 data points and 91 features**.

I don't need to know anything about machine learning to find that alarming. That's a ratio any competent statistician would reject on sight. You're fitting a complex ensemble — multiple models, stacking, calibration — to data where the features outnumber the training examples by roughly 4:1 in the base models, and likely worse in the stacked layer.

Everything else on the known issues list is downstream of this. The objective function misalignment matters more when your model is already overfit to noise. The calibration problem is worse when there's not enough data to calibrate on. The Monte Carlo simulations produce confident-looking output from an uncertain foundation.

Here's what stands out to an outsider: the system *looks* like it handles uncertainty rigorously — 50k simulations, temperature scaling, LOYO cross-validation. But those are all process credibility signals layered on top of a fundamentally data-starved problem. The LOYO validation leaves out one year at a time from an already small dataset, so you're training on ~350 games and calling that robust.

The honest question is: does this model actually outperform simpler baselines? A logistic regression on 5 features trained on the same data might perform equivalently. If you can't answer that clearly, you don't know if the complexity is buying you anything.

The credibility risk isn't the scrapers or the file sizes. It's that the system might be producing highly-decorated noise, and the infrastructure makes that hard to see.

### 5. The Executor

The overfitting risk from the training set size is the blocker that makes everything else irrelevant.

~400 tournament games, 91 features, ensemble stacking. The math doesn't work. You're fitting a stacked ensemble on a dataset where the feature count approaches the sample count. LightGBM with LOYO on 8 folds of ~45 games each isn't cross-validation — it's noise amplification with extra steps. Whatever accuracy numbers you're seeing on 2025 holdout are one lucky bracket draw away from being statistical artifacts.

Here's what Monday looks like if this is actually the critical limitation:

1. Run a feature importance audit on the LightGBM model. If the top 10 features explain >80% of gain, you have 81 dead features bloating the model and masking the real signal. Cut them.

2. Pull in regular-season game data to expand the training set. Tournament outcomes are downstream of regular-season performance — that data exists and can be used without leaking.

3. Enforce a hard feature cap: with ~400 samples, you have no business running more than 15-20 features in a stacked ensemble. Regularize hard or drop to a simpler model.

The objective misalignment and scraper fragility are real problems, but they're fixable in a week. Overfitting on 400 samples with 91 features means you may be shipping a noise machine that looks calibrated. That's not a maintenance problem — that's a credibility problem that invalidates the whole pipeline.

Everything else on that issues list is downstream. Fix the data starvation first or the rest of the work is wasted on a broken foundation.

---

## Peer Reviews

### Anonymization Map

| Letter | Advisor |
|--------|---------|
| A | The Contrarian |
| B | The First Principles Thinker |
| C | The Expansionist |
| D | The Outsider |
| E | The Executor |

### Review 1

**Strongest: B** — Correctly identifies that the limitations aren't independent — they share a root cause. Framing the problem as a missing coherent objective function is the sharpest diagnostic. It also correctly notes that more data doesn't fix a misaligned gradient.

**Biggest blind spot: C** — Reframes limitations as opportunities without engaging with whether the Monte Carlo simulations produce reliable probabilities. Wiring a bracket-EV optimizer to overfit samples doesn't produce strategy — it launders noise into recommendations.

**All missed:** No baseline comparison. Not one response asked whether this system demonstrably outperforms seed-based heuristics or simple Elo ratings on holdout years. Without that benchmark, the entire debate is academic.

### Review 2

**Strongest: A** — Names the specific numeric crisis (5 observations per feature, 8 meta-learner training points) and correctly identifies that LOYO is evaluation, not validation. D makes the same core point but less precisely.

**Biggest blind spot: C** — Treats Monte Carlo as a crown jewel and pivots to feature advocacy. Entirely sidesteps whether the underlying win probabilities are trustworthy.

**All missed:** Absence of a baseline comparison. You cannot know if 91 features and stacking add value until you measure them against a 2-feature null model on the same LOYO folds.

### Review 3

**Strongest: A** — Most rigorous. It traces the exact failure chain: 8 test folds → 8 stacking training points. Specific, structural, precise.

**Biggest blind spot: C** — Completely sidesteps whether win probabilities fed into Monte Carlo are reliable. A simulation built on miscalibrated probabilities produces confident, wrong bracket recommendations.

**All missed: Leakage.** No response questioned whether features are clean of temporal leakage. Tournament prediction is acutely vulnerable — if any feature encodes information available only after Selection Sunday, LOYO CV will not catch it, and the entire validation story is false.

### Review 4

**Strongest: B** — Identifies the root cause that makes every other fix potentially worthless. Optimizing against the wrong objective means better models produce worse outcomes.

**Biggest blind spot: C** — Never questions whether the win-probability inputs feeding Monte Carlo are valid. A crown jewel that runs on miscalibrated probabilities is a sophisticated noise amplifier.

**All missed:** Non-nested calibration inflates apparent performance. Calibrating on the same data used for stacking introduces leakage that corrupts the CV signal. Makes it impossible to know if identified problems are real or masked by inflated numbers.

### Review 5

**Strongest: B** — Identifies the root cause making all other work potentially meaningless. Optimizing the wrong objective means the model gets *better at being wrong*.

**Biggest blind spot: C** — Treats Monte Carlo as an asset without questioning whether probabilities feeding it are trustworthy. Garbage-in crown jewel produces confidently wrong bracket strategies.

**All missed: Temporal validity.** NCAA basketball has changed substantially — tempo, three-point volume, transfer portal dynamics, seeding committee behavior. An 8-year LOYO window straddles structural shifts in the sport. Features that predicted 2016 outcomes may be anti-predictive in 2024.

---

## Chairman Synthesis

### Where the Council Agrees

The training set size is a genuine crisis, not a minor limitation. Every advisor who engaged with the numbers arrived at the same place: ~400 games and ~91 features is a ratio that produces a model incapable of distinguishing signal from era-specific noise. The LOYO cross-validation with 8 folds gives 8 test points — this is not validation, it is anecdote. The stacking meta-learner trains on those 8 points, compounding the problem. Three advisors independently flagged this as the foundational issue, and the peer reviews rated the numeric specificity of this critique as the strongest argument in the council. There is no dissent on the core arithmetic.

The second point of convergence: no one defended the current system as provably better than a seed-based heuristic. That silence is itself a verdict.

### Where the Council Clashes

**Data starvation vs. objective misalignment as the primary blocker.**

The Contrarian and the Executor say data starvation is the existential issue — without enough observations per feature, the objective function is irrelevant because you cannot learn anything meaningful regardless of what you are optimizing for.

The First Principles Thinker says objective misalignment is the root cause — even with more data, training on log-loss and deploying for ESPN points means you are ascending the wrong hill more confidently as data accumulates.

Both are right in isolation. The genuine tension is sequencing: which do you fix first? The Contrarian's position is stronger in the near term — you cannot diagnose objective misalignment on a model that may be fitting noise. But the First Principles argument is stronger for the rebuild — if you expand the training set without first defining the correct objective, you will produce a larger, more confident version of the same wrong system. These are not mutually exclusive problems. They are sequential problems with a defined order.

**The Monte Carlo engine: crown jewel or noise amplifier.**

The Expansionist sees the 50k simulation engine as the most valuable asset and argues for wiring it into bracket-EV optimization. Every peer review disagreed, and did so precisely: garbage-in, garbage-out. If the base model probabilities are unreliable — which the data starvation problem suggests they are — the simulation engine is laundering noise through elegant machinery. 50,000 samples of a bad distribution is a more confident bad distribution. The Expansionist is not wrong about the architecture's potential. The peer reviews are not wrong that the potential is currently unrealized and possibly misleading.

### Blind Spots the Council Caught

**No baseline comparison.** Two independent peer reviews flagged this, and no advisor had raised it. The system has no documented comparison against trivially simple models — seeds alone, KenPom ratings alone, a 5-feature logistic regression. Without this, there is no way to know whether the complexity is buying anything. This is not a theoretical concern. It is a reproducibility and credibility gap that makes every reported performance number uninterpretable.

**Temporal leakage and structural drift.** One reviewer raised feature-level temporal leakage — if any feature encodes information available only after Selection Sunday, LOYO validation will not catch it, and the entire validation story is compromised. A separate reviewer raised structural drift — NCAA basketball in 2016 (pre-transfer portal, different pace, different 3-point volume) is a materially different game than 2024. Features calibrated to 2016-2019 may be anti-predictive in 2024-2025. Neither issue was raised by any primary advisor, and both are potential silent killers of the validation methodology.

**Non-nested calibration inflating apparent performance.** The temperature calibration is applied after training, on data that partially overlaps with validation. This makes the model appear better-calibrated than it is, which makes the core reliability question harder to answer honestly.

### The Recommendation

The most critical limitation is the combination of data starvation and the complete absence of a baseline comparison — and these are inseparable. Data starvation means you cannot trust any of the model's learned coefficients. No baseline means you cannot quantify how much you are losing to that starvation. You are operating a complex system with no ground truth reference point and no way to know if it works.

The objective misalignment and calibration issues are real, but they are second-order. You cannot fix the objective until you know what the model is actually learning. You cannot trust the calibration until you have enough data to validate it. The Monte Carlo engine is architecturally sound but currently produces precisely-computed uncertain answers.

Fix the data problem first, and fix it by comparison before you fix it by collection. Establish what the model's actual performance ceiling looks like relative to the simplest possible alternative. Everything else — objective alignment, feature reduction, calibration nesting, EV optimization — is premature until that baseline exists.

### The One Thing to Do First

Run a seed-only logistic regression on the same 2016-2024 LOYO folds using exactly two features — seed number and round number — and compare its Brier score and ESPN-equivalent accuracy against the current ensemble. If the 91-feature stacked model with temperature calibration does not meaningfully beat two features, every other improvement is theater. This takes one afternoon and tells you more about the system's actual value than any architectural enhancement you could make in the next quarter.
