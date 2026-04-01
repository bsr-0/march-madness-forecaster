# LLM Council Transcript

**Date:** April 1, 2026
**Question:** What is the single most critical limitation of this March Madness forecasting repository, and what should be done about it?

---

## Original Question

> Council this: identify the most critical limitation of this repository

## Framed Question

**Core decision:** What is the single most critical limitation of this March Madness forecasting repository, and what should be done about it?

**Context:**
- Production NCAA March Madness tournament prediction system generating calibrated win probabilities for the 68-team bracket
- Trains on ~17,600 regular-season games (8 years, 2016-2024 excluding 2020), deploys on ~67 tournament games/year
- After symmetric augmentation + year-decay weighting: ~2,200 effective training samples
- Calibration uses only 60-70 holdout tournament games (temperature scaling)
- README claims "simple 7-feature logistic regression" but `production_2026.json` enables stacking, market blend, Bayesian Bradley-Terry, spread model, feature selection
- Stacking meta-learner acknowledged in code comments to overfit on ~400 samples, but re-enabled as `required=true` in production config
- 86-dimensional feature space engineered but only 7 features used in production
- GNN and transformer modules exist in source but are disabled at runtime
- Strong governance framework: freeze artifacts, hash verification, audit trails, production validators
- 198+ test files including temporal leakage detection, data contracts, PIT validation
- 9 CI/CD workflows including nightly governance checks
- ~40% of D1 teams appear 0-1 times in tournament history (cold-start)
- Distribution mismatch: regular season vs tournament (neutral sites, single elimination, seeded brackets, high stakes)

**Stakes:** This system ships real predictions. Understanding its deepest limitation determines whether incremental improvements will help or whether a fundamental rethink is needed.

---

## Advisor Responses

### 1. The Contrarian

The fatal flaw is right there in the numbers and everyone's walking past it: **you're calibrating on 60-70 games and then shipping those calibrated probabilities as if they mean something.**

Temperature scaling with 60-70 samples doesn't give you calibration. It gives you a random number generator with a veneer of statistical legitimacy. Your confidence intervals on the temperature parameter itself are so wide that your "calibrated" probabilities could be off by 15-20 percentage points and you'd never know. You can't validate calibration on 60 games. You can't even reliably *detect miscalibration* on 60 games. A reliability diagram with 60 points binned into deciles gives you 6 games per bin. That's not a signal, that's noise you're choosing to trust.

And it gets worse. The stacking meta-learner is re-enabled in production despite the code comments literally admitting it overfits on ~400 samples. So now you're stacking models trained on thin data, then calibrating the stacked output on even thinner data. You've built a pipeline that launders uncertainty at every stage.

**What actually breaks:** In any given year, your probability estimates for 12-5, 11-6, and 10-7 matchups — the ones that actually matter for bracket differentiation — are essentially made up. You don't have enough calibration data in those ranges to distinguish your model from a seed-lookup table.

**What to do:** Stop pretending you can calibrate in-house. Use the one source of ground truth that actually has enough samples: Vegas lines. Calibrate against closing spreads, not against your own holdout set. Your market blend feature already acknowledges the market knows more than you do — commit to that fully for calibration instead of running a statistical ritual on 60 games and calling it rigorous.

### 2. The First Principles Thinker

The question isn't "what's the single most critical limitation." The question is: **can this system actually learn what it claims to learn?**

Strip away the governance, the CI/CD, the 198 test files, the GNN modules. What's left? A system trying to predict tournament outcomes by training on regular season games. That's not a limitation to fix. That's a foundational category error.

Tournament basketball is a different distribution than regular season basketball. Different pressure, different preparation depth, different game theory (single elimination vs. long season), different officiating patterns, different venue effects. You're training a model on "how do teams perform across 30+ games against mixed competition" and deploying it on "how does this team perform in a single must-win game against an opponent that spent a week preparing specifically for them."

**The 60-70 calibration samples aren't a tuning problem. They're telling you the truth: you don't have enough tournament signal to learn tournament-specific dynamics.** Temperature scaling on 60 samples isn't calibration—it's a ritual.

The stacking meta-learner being "re-enabled as required=true" despite acknowledged overfitting on ~400 samples is the system lying to itself. More complexity on insufficient signal doesn't produce better predictions. It produces more confident wrong predictions.

What should actually be done: **Stop pretending this is a prediction problem you can solve with more features or fancier models.** The honest path is:

1. Accept that logistic regression on 7 features is your ceiling given available signal
2. Widen uncertainty intervals dramatically—your calibration data says you don't know what you don't know
3. Invest entirely in the one thing that could actually help: finding proxy data that mimics tournament conditions (conference tournaments, rivalry games, postseason play)

The governance framework is excellent. The question it should be governing is whether the predictions deserve the confidence they carry.

### 3. The Expansionist

The single most critical limitation is that this system is **massively underexploiting its own infrastructure**. You have 86 engineered features, GNN and transformer modules, stacking ensembles, Bayesian Bradley-Terry — and you're shipping 7-feature logistic regression. That's like owning a Formula 1 car and driving it in first gear because you're nervous about the curves.

The real question isn't "what's the biggest risk" — it's **what's the ceiling if you actually turn everything on?**

Here's what everyone else will miss: the 2,200 effective samples problem isn't a wall, it's a **data engineering problem with known solutions**. Transfer learning from the full 17,600 regular-season games as a pretraining step, then fine-tuning on tournament data, would let you responsibly use the GNN and transformer modules that are already built and sitting idle. The distribution mismatch between regular season and tournament is real but it's a feature, not a bug — that's exactly what domain adaptation techniques handle. You have the neutral-site flag. You have seeding information. You can model the shift explicitly.

The 60-70 calibration holdout games? **Synthetic tournament simulation from historical regular-season matchups** between tournament-caliber teams on neutral courts could 10x that calibration set tomorrow.

What should be done: **Build a staged activation pipeline.** Phase 1: Enable the 86-feature space with proper regularization. Phase 2: Activate the stacking ensemble with leave-one-season-out cross-validation instead of in-sample fitting. Phase 3: Pretrain the GNN on regular-season conference tournament graphs, fine-tune on March Madness. Phase 4: Blend with market data using the infrastructure you already have.

The governance framework, the 198 test files, the CI/CD — that's your safety net. **Use it.** You built the runway for a jet and you're taxiing a Cessna. The limitation isn't technical fragility. It's ambition.

### 4. The Outsider

The number that jumps off the page to someone with no investment in this project: **60-70 holdout tournament games for calibration.**

Everything else — the governance framework, the 198 test files, the 9 CI/CD workflows — is elaborate scaffolding built around a system that calibrates its confidence on roughly the same number of data points as a classroom homework assignment.

Here's what an outsider sees: you've built production-grade infrastructure around a fundamentally sample-starved calibration step. Temperature scaling on 60-70 games means your "calibrated probabilities" have wide confidence intervals that nobody is reporting. A user consuming these predictions has no idea that the difference between "72% win probability" and "65% win probability" is essentially noise given the calibration sample size.

Worse, you've re-enabled a stacking meta-learner that your own code comments admit overfits on ~400 samples. That's the builder ignoring their own warning label because the number looked better on a backtest. Classic curse of knowledge — you understand the caveat so you think the caveat is handled. It isn't.

**What to do about it:**

1. **Stop calling these calibrated probabilities.** They're scores. Report them with honest uncertainty bands derived from the calibration sample size. If you can't put a meaningful confidence interval on your calibration, say so.

2. **Pool calibration data across years aggressively.** If tournament dynamics are stable enough to predict, they're stable enough to calibrate across. You need 500+ games in that holdout, not 60.

3. **Disable the stacking meta-learner until you have the sample size to support it.** Your own comments told you this. Listen to past-you.

The governance and testing are impressive, but they're guarding the wrong door. The system's credibility bottleneck is statistical, not operational.

### 5. The Executor

The single most critical limitation is right there in the numbers: **you're calibrating on 60-70 games and training a stacking meta-learner on ~400 samples, then shipping it as `required=true` in production.**

Everything else is noise compared to this. Here's what I mean concretely:

**The calibration problem is fatal.** Temperature scaling on 60-70 games gives you confidence intervals so wide they're meaningless. You're tuning a parameter on a dataset where adding or removing 3-4 upsets shifts your calibration curve dramatically. You cannot know if your probabilities are calibrated. You're guessing and calling it science.

**The stacking re-enablement is malpractice.** The code comments literally say it overfits on ~400 samples. Someone turned it back on anyway. A meta-learner with insufficient training data doesn't add signal — it memorizes noise and launders it as ensemble wisdom. You're adding complexity that makes your predictions worse while making them look more sophisticated.

**What to do Monday morning:**

1. **Kill the stacking meta-learner.** Set `required=false`. Today. Simple weighted average of base models. You lose nothing real and stop actively injecting overfit noise.

2. **Switch calibration to Platt scaling with strong priors** using your regular-season data as a prior distribution, updating with tournament games. This gives you a defensible calibration with your tiny tournament sample.

3. **Pool 3-5 years of tournament games for calibration** instead of single-year holdout. You go from 60 games to 250-300. Still small, but crosses the threshold where temperature scaling starts to mean something.

4. **Ship the 7-feature logistic regression as your primary model.** It's honest about what the data can support. The 86-feature space and disabled GNN/transformer modules are resume decorations, not production tools.

Stop building upward on a foundation that can't hold weight. Shrink to what the data actually supports.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Executor
- Response B = The Outsider
- Response C = The Contrarian
- Response D = The Expansionist
- Response E = The First Principles Thinker

### Review 1

**Strongest: Response C (The Contrarian).** It offers the most pragmatic fix — use Vegas lines instead of in-house calibration. The market blend already exists, making this immediately actionable. **Biggest blind spot: Response D (The Expansionist).** Transfer learning doesn't solve distribution mismatch — it encodes it. "Synthetic tournament simulation" is circular: you need tournament signal to simulate tournaments. **All missed:** Evaluation methodology — a proper backtesting harness with rolling multi-year out-of-sample evaluation. Without that, every recommendation is guesswork.

### Review 2

**Strongest: Response E (First Principles Thinker).** It identifies the root cause — distribution mismatch — rather than just the symptom. **Biggest blind spot: Response D (The Expansionist).** Expanding features and complexity with 2,200 samples guarantees overfitting. D mistakes having tools for needing to use them. **All missed:** The evaluation framework problem — you cannot reliably measure whether any intervention helped with 60-70 games per year. Multi-year pooled evaluation with temporal splits is prerequisite for all recommendations.

### Review 3

**Strongest: Response B (The Outsider).** The "they're scores" reframe is the key insight — output semantics are wrong, not just the method. **Biggest blind spot: Response D (The Expansionist).** More model complexity is the opposite of what's needed. **All missed:** The feedback loop problem. Pooled years span rule changes, conference realignment, transfer portal, COVID. Data is not stationary. Also, the 198+ tests likely validate mechanics, not prediction quality.

### Review 4

**Strongest: Response A (The Executor).** Concrete, conservative fixes matched to the actual constraint. **Biggest blind spot: Response D (The Expansionist).** More rope to hang yourself with. **All missed:** Governance contradiction. Stacking is acknowledged to overfit but re-enabled as `required=true`. This is a process failure, not just technical. Why does the system permit configuration states that contradict its own documented findings?

### Review 5

**Strongest: Response B (The Outsider).** Covers both the statistical and communication failure. **Biggest blind spot: Response D (The Expansionist).** Enabling GNN/transformers on ~2,200 samples with 40% cold-start teams is guaranteed overfit. **All missed:** Why was stacking re-enabled? Is there a Kaggle/competition scoring incentive where aggressive calibration wins on log-loss even if epistemically wrong? Without understanding the team's tradeoff, every recommendation is advice into a vacuum.

---

## Chairman Synthesis

### Where the Council Agrees

**The calibration problem is fatal.** All five advisors independently identified that calibrating on 60-70 tournament games is the central weakness. Temperature scaling on this sample size doesn't produce calibrated probabilities — it produces noise with decimal places. The Outsider's reframe landed hardest: these aren't probabilities, they're scores.

**Stacking must be disabled.** Four of five advisors called out the stacking meta-learner as indefensible. The code comments admit it overfits on ~400 samples. It was re-enabled as `required=true` anyway.

**The Expansionist is wrong.** All five peer reviewers identified adding model complexity as the biggest blind spot. More model complexity on 2,200 samples with 40% cold-start teams is accelerant on an existing fire.

**Seven-feature logistic regression is the correct ceiling** given the data constraints.

### Where the Council Clashes

**What to calibrate against.** The Contrarian says use Vegas lines — the market is already better calibrated. The Executor says pool 3-5 years of tournament data and use Platt scaling with strong priors from regular-season data. The First Principles Thinker says widen uncertainty intervals and stop pretending calibration is achievable at all. The Contrarian's fix is fastest and probably most accurate but makes the system dependent on external market data. The Executor's fix preserves independence but may still be sample-starved at 250-300 games.

**Whether the distribution mismatch is solvable.** The First Principles Thinker says training on regular-season games for tournament prediction is a category error. The Executor disagrees by proposing improvements that assume the approach is viable. The truth likely sits between: the mismatch is real and places a hard ceiling, but useful predictions are possible with honest uncertainty.

### Blind Spots the Council Caught

**Evaluation methodology.** Multiple peer reviews flagged that there is no proper backtesting harness. Without rolling multi-year out-of-sample evaluation, you cannot measure whether any intervention actually helped.

**Non-stationarity of pooled data.** Pooling years for calibration spans rule changes, conference realignment, the transfer portal, and COVID. Pooling helps sample size but introduces distribution drift.

**Governance process failure.** Stacking was acknowledged to overfit then re-enabled as required. A technical fix without a governance fix means stacking gets re-enabled next March.

**The incentive question.** If competitive scoring rewards aggressive calibration, the team may be rationally choosing to ship overfit probabilities.

### The Recommendation

**Shrink the system to what the data supports and be honest about uncertainty.**

1. Disable stacking (`required=false`). Ship 7-feature logistic regression as the primary model.
2. Stop labeling outputs as calibrated probabilities. They are model scores. Report with uncertainty bands.
3. Use closing Vegas spreads as the calibration anchor where available, falling back to Platt scaling with informative priors.
4. Build a rolling multi-year out-of-sample evaluation harness before enabling any new complexity.
5. Add a governance gate: configurations contradicting documented findings require explicit override with written justification.

### The One Thing to Do First

Set `stacking.required = false` in the production config and commit it today. The system's own evidence already proves it's wrong, no new analysis is needed, the fix is a single line, and it immediately stops shipping predictions the codebase itself documents as overfit.
