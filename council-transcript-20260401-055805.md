# LLM Council Transcript — Training Data Strategy for Data Starvation

**Date:** April 1, 2026
**Methodology:** Karpathy LLM Council (5 independent advisors → anonymous peer review → chairman synthesis)

---

## Original Question

"Council this: should this repo train on regular season games in addition to tournament games? How can I best address this critical issue of data starvation?"

## Prior Council Correction

The previous council (20260401-052552) stated the system trains on "~400 tournament games." This was factually incorrect. Investigation of the actual codebase revealed:

- **Current year:** ~300-400 regular season games → 600-800 after symmetric augmentation
- **Historical:** 8 years × ~350 games/year (regular season + tournament with round weights) = ~2800 samples
- **Total effective training samples:** ~2000-2200 (after year-decay weighting)
- **Symmetric augmentation** doubles samples by adding reverse perspectives
- **Tournament games** already included with round weights (R64=1 → NCG=32)

The data starvation manifests specifically in:
1. LOYO validation: 8 folds (one per year), high variance
2. Stacking meta-learner: disabled by default ("overfits to ~400 samples") but re-enabled in production_2026.json
3. Calibration: temperature scaling on ~60-70 holdout tournament games only
4. Distribution mismatch: trains on regular season, deploys on tournament

## Framed Question

**Should this repo change how it handles training data to address data starvation, and if so, how?**

Given that the system already trains on ~2000-2200 regular season + tournament samples across 8 years with symmetric augmentation, year-decay weighting, round-weighted tournament games, per-fold preprocessing, and constrained tree models — where does data starvation actually bite, and what's the best intervention?

---

## Advisor Responses

### 1. The Contrarian

The framing here is a trap. "Data starvation" implies more data solves the problem. It doesn't — not for this specific failure mode.

The real issue is **distribution mismatch**, and throwing more regular season samples at it makes the problem worse, not better. The model already trains on ~2000 regular season games. Adding more regular season data deepens its fluency in a game type that doesn't exist in March. Seeds don't exist in November. Neutral sites are rare. No team is playing for survival. The model is being trained on the wrong distribution and then asked to generalize to a foreign one.

The round-weighting (R64=1 → NCG=32) is the only honest signal here, and it's fighting a losing battle against 2000x more regular season samples. That ratio is the problem, not the absolute count.

The stacking meta-learner "disabled by default because it overfits to ~400 samples" is a confession, not a mitigation. Enabling it in production_2026.json anyway is a known liability that someone decided to ship.

The calibration situation is genuinely alarming: temperature scaling on 60-70 tournament games is not calibration, it's noise fitting. One unusual tournament run can shift that entirely.

What's missing from the proposed solutions: any honest answer to "what does this model actually learn that's tournament-specific?" If the answer is "not much," then the correct fix isn't data augmentation — it's a fundamentally different architecture that treats tournament games as a separate inference problem, not just upweighted training samples.

More data from the wrong distribution is worse than less data from the right one.

### 2. The First Principles Thinker

**What Are We Actually Solving?**

The question is framed as "data starvation." That's a symptom label, not a diagnosis. Strip it back.

**What problem are we actually trying to solve?** Distribution mismatch. The model trains on one population (regular season games) and deploys on a different population (tournament games). Data quantity is a secondary concern. You could have 1 million regular season games and the mismatch would still corrupt predictions.

This reframe changes the intervention space entirely.

**The real constraints:** Tournament games are rare by definition — 67 per year, finite historical record. You cannot manufacture that signal. Any attempt to synthetically generate "tournament-like" games is just encoding your own assumptions about what makes tournament games different, which is circular.

**What actually helps:**

1. **Feature engineering over data augmentation.** Seeds, bracket position, rest days, travel distance, neutral site indicator — these are the distribution gap. Encode them explicitly. The model learns what it has access to.

2. **The calibration problem is the most tractable.** 60-70 holdout games is genuinely thin. Platt scaling or isotonic regression on all available historical tournament games (not just holdout) with proper cross-validation is more defensible than temperature scaling on a sliver.

3. **Stacking on 400 samples is a red flag you've already identified.** Disable it or replace it with a simpler ensemble (weighted average by validation performance). The meta-learner is learning noise.

4. **LOYO variance isn't a bug.** Eight years of tournament results *should* vary. High fold variance is honest signal that the problem is hard, not a flaw to engineer away.

The question to keep asking: are we reducing real uncertainty or just hiding it?

### 3. The Expansionist

The framing here is too small. Everyone's treating this as a data poverty problem. It's actually a signal poverty problem — and those have completely different solutions.

The tournament corpus is deliberately tiny: 63 games per year, 8 years, ~500 games total. You cannot fix that by squeezing harder. But you're sitting on enormous untapped signal in adjacent data that nobody's touching:

**NIT and conference tournament games** are elimination-format, neutral-site, single-elimination — structurally identical to March Madness. There are roughly 200+ NIT games per year. That's a 3x increase in tournament-context data, right now, free.

**Historical seeds can be reconstructed** for regular season games using KenPom or NET rankings. The "seeds not available" problem during regular season is solvable retroactively for training purposes.

**The stacking meta-learner's 9-feature bottleneck is the real vulnerability.** With only 8 years of OOF tournament predictions, you have ~500 samples for that second layer. But if you expand to NIT/conference tournaments, that meta-learner trains on calibrated out-of-sample predictions from games that actually match tournament conditions.

**Calibration on 60-70 games is genuinely dangerous.** Pull in NIT data and you calibrate on 3-4x as many elimination games.

The upside: this repo could own *elimination-format basketball prediction* broadly. The model improvements from NIT integration pay dividends on the primary target too.

Don't tune the existing pipeline. Expand the universe of valid training games first.

### 4. The Outsider

The numbers here don't add up to a confidence problem — they add up to an honesty problem.

You have 8 years of data and you're building a system with 75-80 features, a stacking layer, and probability calibration. That's a lot of machinery for very little fuel. The mitigations listed — 32x weighting, symmetric augmentation, shallow trees — are patches on a fundamental mismatch. You're not solving data starvation, you're decorating it.

The stacking layer being "disabled by default for overfitting but re-enabled in production" is the clearest signal. That's not a configuration choice. That's someone knowing a thing is broken and shipping it anyway.

The domain gap between regular season and tournament is the real issue nobody seems to want to name directly. If the domains genuinely differ — neutral venue, single elimination, different stakes — then training on regular season data isn't giving you a head start. It might be giving you confident wrong answers.

What I'd actually ask: what happens if you throw out most of the regular season data and only train on tournament games, even with far fewer samples? A simple model trained on 500 relevant examples often beats a complex model trained on 2000 irrelevant ones.

The honest answer to "should we change how we handle training data" is: yes, but the first change is admitting that more data infrastructure won't fix a domain mismatch. That requires either getting more tournament data (historical records go back decades) or building a simpler model that's honest about what it doesn't know.

### 5. The Executor

**Monday morning moves, ranked by ROI:**

**Do immediately:**

1. **Extend training years to 2008-2024.** Historical data exists back to 2005. You're leaving 8 years of signal on the table for no stated reason. That's ~2000 more samples before augmentation. One config change.

2. **Drop `late_season_training_cutoff_days` to 0 or 20.** You're throwing away ~60% of regular season games with a 45-day cutoff. Early-season data is noisy but you have augmentation and tree depth constraints already protecting you.

3. **Fix the calibration data starvation directly.** 60-70 holdout games is not a calibration dataset. Use OOF predictions from cross-validation as calibration data instead. ~400 tournament samples across years via CV-based calibration.

**Don't touch yet:** Stacking and round weights — measure cheap wins first.

**The actual constraint:** LOYO variance at 8 folds is a math problem. More years (point 1) directly adds folds. That's the only real fix.

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

**Strongest: D** — Asks the sharpest diagnostic question: what happens if you train tournament-only? Cuts through incremental noise and tests the core assumption.

**Biggest blind spot: E** — Operationally specific but intellectually shallow. Extending years and dropping cutoff adds more of the same problematic distribution. Could make the model confidently wrong faster.

**All missed:** The stacking re-enablement is a diagnostic signal — someone judged accuracy gain outweighed overfitting risk. None asked why or what that reveals about base model quality without stacking.

### Review 2

**Strongest: D** — Correct frame: remove data, not engineer around it. "500 relevant > 2000 irrelevant" is the crux.

**Biggest blind spot: C** — NIT/conference tourneys are NOT structurally identical. Seeding logic, pressure, field quality differ materially. "Structurally identical" is the most overconfident assertion in the set.

**All missed:** Within-tournament distribution shift. R64 and Elite 8 are different games. CV should split by round, not just year.

### Review 3

**Strongest: D** — Testable hypothesis, clean experimental path. The simplicity recommendation guards against overfitting.

**Biggest blind spot: C** — Most overconfident. "Structurally identical" is wrong.

**All missed:** Target variable is path-dependent. If model feeds predictions forward in bracket simulation, errors compound. Is this classification or simulation? Nobody asked.

### Review 4

**Strongest: D** — Honest, names the real problem. 500 relevant > 2000 irrelevant is the correct prescription.

**Biggest blind spot: C** — NIT adds volume while diluting signal quality. Mistakes structural similarity for statistical similarity.

**All missed:** Temporal leakage in season-aggregate features. LOYO partially helps but doesn't eliminate it. If features incorporate end-of-season ratings to predict February games, the model is contaminated before the tournament question even matters.

### Review 5

**Strongest: D** — Most epistemically honest.

**Biggest blind spot: C** — NIT/conference tourneys introduce systematic bias while appearing to solve sample size.

**All missed:** Cold-start problem. ~40% of D1 programs appear zero or one times in tournament history. If features encode team identity rather than team-agnostic metrics, you have a memorization problem. If features are team-agnostic, the domain gap may be smaller than everyone assumed.

---

## Chairman Synthesis

### Where the Council Agrees

The core problem is distribution mismatch, not data starvation. Adding more regular season samples does not fix a tournament-inference problem — it potentially worsens it by burying the relevant signal under 2000x more noise. Every advisor, including the Executor, implicitly accepts this framing even when their recommendations diverge. The calibration on 60-70 holdout tournament games is universally recognized as insufficient — it's noise-fitting, not calibration. The stacking re-enablement is a red flag everyone noticed: something known to overfit was shipped anyway, which tells you the base models aren't good enough on their own.

### Where the Council Clashes

The central clash is between the Expansionist (add NIT/conference tournament data to multiply tournament-context samples) and everyone else. Four out of five peer reviewers rejected this outright. "Structurally identical" is wrong: NIT field quality, seeding logic, stakes, and pressure differ systematically from March Madness. Adding that data introduces a new distribution problem while pretending to solve the old one.

The second clash is Executor vs. Contrarian/First Principles Thinker on extending historical years. The Executor calls it a cheap win; the Contrarian calls it more of the same bad data. The resolution: extending to 2008+ tournament games specifically (not regular season) is defensible. Extending regular season data is not.

The third clash is implicit: whether to treat this as a classification problem with better features or a distribution-aware inference problem with a simpler model trained only on tournament games. This is unresolved and consequential.

### Blind Spots the Council Caught

1. **Within-tournament distribution shift.** R64 and Elite 8 are not the same game. A model calibrated on aggregate tournament games is averaging across structurally different rounds. CV should split by round, not just by year.

2. **Target variable path-dependence.** If predictions feed forward in a bracket simulation, errors compound. Whether this is a classification task or a simulation determines the entire architecture. Nobody stated which it is.

3. **Cold-start problem.** Roughly 40% of D1 programs appear zero or once in tournament history. If any features encode team identity rather than team-agnostic performance metrics, the model is partially memorizing rather than generalizing. LOYO doesn't fix this.

4. **Stacking re-enablement as diagnostic.** Someone judged that the accuracy gain outweighed overfitting risk, which implies the base models' standalone accuracy was low enough to justify a known-broken fix. That's the real problem to solve.

### The Recommendation

Do not add NIT or conference tournament data. Do not expand regular season samples. Do three things:

**1. Fix calibration properly.** Replace the single-year holdout with out-of-fold tournament predictions across all available years using LOYO-CV. This turns 60-70 games into the full historical tournament sample for calibration. This is the highest-ROI fix in the repo and costs almost nothing structurally.

**2. Run the tournament-only baseline experiment.** Train on tournament games only — approximately 500-600 games from 2005 to 2024 if you go back far enough — with a simpler model (logistic regression or a shallow tree). Compare accuracy and calibration against the current blended model. This is a diagnostic, not a deployment decision. If the simple tournament-only model matches or beats the current model, the entire architecture is wrong and you need to know that now before adding more complexity.

**3. Extend historical years to 2005-2024 for tournament games specifically.** The data is on disk. Tournament games only — not regular season. This gets calibration from 60 games to roughly 300-350, at which point Platt scaling or isotonic regression becomes meaningful rather than noise-fitting.

Disable stacking until base model quality justifies it. If you need stacking to be competitive, your features aren't doing enough work.

Do not do the Executor's `late_season_training_cutoff_days` change until after the tournament-only experiment. Cutting late-season data before you know whether regular season data helps or hurts is premature.

### The One Thing to Do First

Run the tournament-only model experiment. Everything else is optimization on an architecture that may be fundamentally wrong. If 500 tournament games with a logistic regression outperforms 2200 blended games with a stacked ensemble, you have your answer — and it means the current training data strategy is actively harming the model. That experiment takes one afternoon. It resolves the Contrarian vs. Executor debate with data instead of argument. Do that first.
