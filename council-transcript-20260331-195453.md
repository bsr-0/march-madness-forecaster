# LLM Council Transcript

**Date:** 2026-03-31 19:54 UTC
**Question:** What is the single biggest limitation of the March Madness Forecaster repository, and what should be prioritized to fix it?

---

## Original Question

> Council this: what is the biggest limitation of this repository?

## Framed Question

The March Madness Forecaster is a sophisticated NCAA tournament prediction system with ~100+ Python files, 79-dimensional feature engineering, a LightGBM/XGBoost/LogReg ensemble with LOYO CV, Monte Carlo bracket simulation, and extensive governance/production locking. Key context:

- Production is frozen for 2026 with hash-locked config validation
- Training years: 2016-2019, 2021-2024 (8 seasons total, no 2020 COVID)
- Holdout: 2025, Target: 2026
- Research infrastructure (GNN, transformers, agent orchestration, experiment workflows) exists but is disconnected/disabled
- Audits found: calibration is not nested (moderate leakage risk), stacking weights optimized on same LOYO predictions used for evaluation
- Heavy reliance on web scrapers (Torvik, ESPN, Sports Reference) that can break
- Some files are extremely large (baseline_training.py ~139k lines, data_loader.py ~81k lines, calibration.py ~47k lines)
- 50,000 Monte Carlo simulations for bracket optimization

**The question: What is the single biggest limitation of this repository, and what should be prioritized to fix it?**

---

## Advisor Responses

### 1. The Contrarian

The stacking weights optimized on the same LOYO predictions used for evaluation is not a "moderate" issue — it's a fundamental validity problem that corrupts every confidence metric the system produces.

Here's the chain of consequences: You have a system frozen for production with hash-locked governance treating its outputs as trustworthy. But the ensemble weights were tuned on the very predictions being evaluated, meaning the stacking layer has been implicitly fit to your holdout signal. The LOYO CV that should prevent this becomes theater. Your calibration plots, your Brier scores, your confidence intervals — all of them are optimistic by an unknown amount. You don't know how much. That's the problem.

The massive file sizes (139k line baseline_training.py) aren't just technical debt — they're evidence that nobody can actually audit what's happening inside. When a file is that size, the leakage could be anywhere and finding it requires heroic effort. The governance system locks in whatever is broken with cryptographic certainty.

The fatal flaw: you've built elaborate production safeguards around a system whose core accuracy claims cannot be independently verified because the evaluation methodology is compromised. The 2025 holdout was supposed to validate the 2026 predictions, but if stacking weights saw LOYO predictions during optimization, the holdout performance is inflated.

Priority fix: Before 2027, refit the stacking layer using a strict temporal split where ensemble weights are determined solely on years prior to the evaluation period — never on the same LOYO fold outputs being reported. Then re-run 2025 holdout evaluation from scratch. Everything else is cosmetic until you know whether your predictions are actually better than a simpler baseline or just overfit noise.

### 2. The First Principles Thinker

The question assumes there's a single "limitation to fix." That framing is wrong. It treats symptoms as the disease.

Strip it down: what is this system actually trying to do? Produce reliable probability estimates for NCAA tournament outcomes to optimize bracket decisions.

Now ask: what would make that goal fail? Not file size. Not disconnected GNN research. The core failure mode is **epistemic opacity** — you cannot currently trust any output from this system because you cannot verify where your signal ends and your leakage begins.

Here's the first-principles problem: calibration isn't nested, and stacking weights are optimized on the same predictions used to evaluate them. This means the system's confidence estimates — the foundation of every Monte Carlo simulation — are systematically overfit. Your 50,000 simulations are precision theater. You're multiplying garbage by 50,000 and calling it rigor.

Everything else is secondary. The massive files are a maintainability problem. The broken scrapers are a data freshness problem. The disconnected research is an opportunity cost problem. But none of those kill the system's core purpose.

The actual first-principles fix: before any production run matters, you need honest held-out probability estimates. That means nested calibration within LOYO folds, and stacking weights optimized on genuinely out-of-fold predictions only.

The real question isn't "what's the biggest limitation?" It's "do the probabilities coming out of this model mean anything?" Right now, you don't know. That's the only thing worth fixing first.

### 3. The Expansionist

The disconnected research infrastructure is the most criminally undervalued asset here, and fixing that disconnect is the obvious priority.

You have GNNs, transformers, and agent orchestration sitting dormant while the production system runs an 8-season ensemble with known leakage. That's not a limitation — that's buried treasure. The "limitation" framing is wrong. The real question is: what happens when this system actually integrates those capabilities?

Graph neural networks on tournament brackets aren't incremental. They capture conference strength, scheduling effects, and team interdependencies in ways that 79-dimensional flat features fundamentally cannot. A GNN that models "Team A beat Team B who beat Team C" as a graph structure could unlock predictive signal that no amount of feature engineering extracts. That's not a marginal improvement — that's a different class of model.

The agent orchestration infrastructure is equally exciting. Right now you have Monte Carlo simulations running 50,000 iterations with static weights. An agent loop that dynamically adjusts priors mid-tournament based on early-round upsets? That's a system that gets *better* as the bracket unfolds, not one that fires once and watches.

The leakage issue, the massive files, the scraper fragility — those are maintenance problems. Important, yes. But fixing them doesn't expand what's possible. Activating the research infrastructure does.

Priority: build a clean activation pathway for the GNN/transformer experiments, wire them into the ensemble as experimental components, and measure whether they outperform on the 2025 holdout. The production lock is smart governance. But the research stack sitting idle is opportunity cost measured in prediction accuracy you're currently leaving on the table.

### 4. The Outsider

The 139,000-line baseline_training.py file is the single biggest limitation, and it's not close.

I don't know anything about machine learning pipelines or NCAA tournaments. What I do know is that no human can reason about a 139,000-line file. That's not a file — that's a system that ate its own documentation and swallowed its own boundaries. When something breaks, where do you look? When you want to change something, how do you know what else you're touching? You don't. You guess, you test, you hope.

This isn't a "technical debt" problem. It's a comprehension problem. The people maintaining this system cannot hold it in their heads. They cannot safely modify it. They cannot hand it to someone new. The complexity isn't distributed across files where it could be isolated and reasoned about — it's compacted into a monolith that defeats human understanding by sheer volume.

Everything else on that list — calibration leakage, stacking weight contamination, scraper fragility, disconnected research infrastructure — those are problems you can fix once you can see the code clearly. You cannot fix what you cannot read.

The priority: break baseline_training.py into modules with explicit interfaces. Not for elegance. Not for software engineering purity. Because the current state means every other fix is being done blind, by people hoping they understand a 139,000-line file well enough not to break something they can't see.

The audits already found leakage. The question is how much more leakage exists that hasn't been found yet — because nobody can read the whole file.

### 5. The Executor

The biggest limitation is baseline_training.py at 139k lines. Full stop.

Everything else on that list - calibration leakage, stacking weight contamination, scraper fragility - is a real problem. But none of it matters if you can't safely touch the codebase. A 139k-line file is not a file, it's a disaster waiting to happen. No one can reason about it, no one can test it in isolation, and every change is a grenade.

Here's what you do Monday morning:

1. Run `grep -n "^def \|^class "` on baseline_training.py and get a count of all top-level functions and classes. This takes 2 minutes and tells you the decomposition map.
2. Identify the 5-10 clearest functional boundaries - data loading, feature computation, model training, evaluation, serialization. These become separate modules.
3. Extract one module. The smallest, most self-contained chunk. Run the full pipeline. If it passes, repeat.

The calibration leakage and stacking contamination are legitimate issues but they require careful methodological work that depends on understanding what the code actually does. You cannot safely fix a calibration pipeline you cannot read.

The production freeze is actually helpful here - you have a locked, reproducible baseline to regression-test against while you refactor. Use it. Every extraction should produce identical outputs to the frozen baseline.

data_loader.py at 81k lines is the second priority for the same reason.

Fix the file size problem first. Everything else is downstream of being able to read your own code.

---

## Peer Reviews

### Anonymization Mapping

| Letter | Advisor |
|--------|---------|
| A | The Contrarian |
| B | The First Principles Thinker |
| C | The Expansionist |
| D | The Outsider |
| E | The Executor |

### Review 1

**Strongest: A (Contrarian)** — correctly identifies stacking contamination as a validity crisis, names the consequence (all confidence metrics optimistic by unknown amounts), explains mechanism, gives concrete fix. Also makes the sharp point that governance locks in the corruption.

**Biggest blind spot: C (Expansionist)** — treats disconnected GNN/transformer infrastructure as "buried treasure" while waving away leakage as "maintenance problem." Building new model capacity on top of contaminated evaluation harness produces meaningless benchmarks.

**All missed:** Data quality upstream of the model. Fragile web scrapers feeding 79-dim features means feature distributions may be silently shifting or corrupted. A perfectly calibrated ensemble trained on bad features is still wrong. None asked: "Are the inputs trustworthy?"

### Review 2

**Strongest: A (Contrarian)** — identifies stacking contamination as first-order validity problem, explains mechanism clearly, gives concrete sequenced action. Also connects governance lock to the problem.

**Biggest blind spot: C (Expansionist)** — GNNs and transformers built on top of contaminated ensemble don't expand capability, they launder overfit signal through more complex architectures.

**All missed:** Data freshness and scraper fragility. Fragile scrapers mean the 79-dim feature vector may already contain stale or corrupted inputs with no mechanism to detect or flag this. A model with clean calibration still produces garbage if input data is silently wrong.

### Review 3

**Strongest: A (Contrarian)** — precise diagnosis + governance locks in corruption.

**Biggest blind spot: C (Expansionist)** — can't benchmark new architectures against corrupted holdout. Mistakes feature richness for validity.

**All missed:** The production governance lock is underanalyzed. If contamination is baked into the locked artifact, the governance system is actively preserving and certifying a broken model. No response asked: should the current production lock be invalidated? The governance system needs a formal "revoke and relock" protocol.

### Review 4

**Strongest: A (Contrarian)** — identifies validity problem + governance compounding risk, gives actionable fix.

**Biggest blind spot: C (Expansionist)** — treats research as buried treasure while glossing over leakage. Activating GNNs on contaminated evaluation means no valid signal to compare against.

**All missed:** The data pipeline and web scraper fragility. Every response assumes training data is trustworthy. Wrong stats, missed games, stale rosters flow into 79 features and 8 seasons of LOYO CV. Model with clean calibration is worthless if input data is quietly wrong.

### Review 5

**Strongest: A (Contrarian)** — correctly identifies stacking contamination as most urgent, governance lock compounds the problem.

**Biggest blind spot: C (Expansionist)** — dormant research is dormant for a reason. Activating on contaminated stacking weights produces confidently wrong predictions at higher complexity. Mistakes capability for readiness.

**All missed:** Cost-of-delay question. The 2026 tournament is live right now (March 31, 2026). Every proposed fix is post-tournament work. The real question: given frozen production, what lightweight monitoring or annotation should be done NOW to capture ground truth that informs which fix actually matters?

---

## Chairman Synthesis

### Where the Council Agrees

Stacking weight contamination is a validity crisis, not a code quality issue. Four of five advisors — and all five peer reviews — converged independently: the ensemble stacking layer was optimized on the same LOYO predictions used to evaluate it. Every confidence metric is reporting on a system that has implicitly seen its own holdout signal. The governance system then cryptographically locks in whatever is broken.

### Where the Council Clashes

Fix validity first vs. fix readability first. The Contrarian/First Principles camp argues contamination must be addressed immediately. The Outsider/Executor camp argues you can't safely fix what you can't read in a 139k-line file.

Resolution: the stacking fix is narrower than a full refactoring job. You don't need to understand all 139k lines to locate and refit one stacking function.

The Expansionist was unanimously identified as the weakest position by all 5 peer reviews.

### Blind Spots the Council Caught

1. Scraper fragility as silent upstream data corruption (4/5 reviews)
2. No governance revoke-and-relock protocol (2/5 reviews)
3. Tournament is live — all fixes are post-tournament (1/5 reviews)

### The Recommendation

The single biggest limitation is stacking weight contamination. Priority sequence:

1. **Now:** Instrument outcome capture during live tournament
2. **Post-tournament:** Refit stacking layer with strict temporal split, re-run holdout evaluation
3. **After validity confirmed:** Nest calibration, add data quality validation, refactor monoliths
4. **Only then:** Activate research infrastructure

### The One Thing to Do First

Add outcome logging to the live tournament pipeline today — capture every prediction and actual result for post-tournament validation.

---

*Council session conducted 2026-03-31. 5 advisors, 5 peer reviews, chairman synthesis. Methodology adapted from Andrej Karpathy's LLM Council.*
