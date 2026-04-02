# Council Transcript — Repo Limitations
**Date:** April 2, 2026 20:52 UTC

## Original Question
"What are the biggest limitations of this repo?"

## Framed Question
**What are the biggest limitations of the March Madness Forecaster repository?**

Context: ~123K LOC Python NCAA tournament prediction system with 537 source files and 202 test files. Production model is a single logistic regression on 7 features that achieves BSS ≈ 0 vs seed baseline. Massive disabled infrastructure (GNN, transformers, ensemble stacking). 20% test coverage threshold. Many lint rules suppressed. Training on ~17,600 regular-season games, calibrating on ~530 tournament games across 9 years.

---

## Advisor Responses

### 1. The Contrarian

The fatal flaw is BSS ≈ 0 versus seed baseline — this entire system produces predictions no better than looking up seed matchup win rates. The "simple model is the ceiling" claim is self-serving nonsense — that conclusion was reached using LOYO on 530 tournament games, roughly 59 games per fold. You cannot draw reliable conclusions about model superiority with 59-game test sets. The error bars on BSS at that sample size swallow any signal. The project declared defeat prematurely, then built 123K lines around that declaration.

The codebase is a monument to sunk cost — GNN, transformer, ensemble all disabled. The quality controls are performative — 20% coverage is meaningless, suppressing F401/F841/F821 means the linter ignores bugs, mypy checks nothing.

The real limitation: tournament basketball has roughly 4 bits of entropy per game that seeds don't already capture. The honest next step is either find genuinely novel data sources or accept the seed baseline and focus on bracket optimization strategy.

### 2. The First Principles Thinker

The question isn't "what are the limitations." The question is: why does 123K lines of code exist to deliver a model that can't beat picking by seed? BSS ≈ 0 means the system produces no value beyond a lookup table.

The actual problem is information-theoretic — March Madness outcomes contain very little predictable signal beyond seed strength. Seeds already encode committee judgments aggregating hundreds of data points. You're trying to extract residual alpha from ~530 tournament games.

The real limitations: 1) Wrong success metric — BSS against seed baseline is the hardest benchmark. The system should measure value-add in bracket pool contexts, where small edges compound across 63 games. 2) Wrong unit of prediction — game-level probabilities are commoditized. The scarce value is in identifying correlated upsets and bracket strategy. 3) No feedback loop — nine years of predictions with no systematic analysis of where the model fails.

### 3. The Expansionist

The massive untapped upside: you built the infrastructure for a much more powerful system and then gave up at the logistic regression. BSS ≈ 0 is not a ceiling — it's a signal that 7 features from logistic regression can't extract the edge that exists. The architecture doc declaring "the simple model is the ceiling on this data" is the single most dangerous sentence in the codebase. It confuses "we haven't found the signal yet" with "no signal exists." Vegas lines beat seed baselines.

The biggest limitation is that 86 features, GNNs, transformers, and ensemble stacking exist but are disabled. Feature selection going from 86 to 7 likely discarded interaction effects. The scraping infrastructure is an unrealized asset — injury reports, transfer portal data, player-level metrics create edge in March specifically.

The codebase is 80% of the way to a competitive prediction system. What's missing is the willingness to let complex models fail expensively until they don't.

### 4. The Outsider

You built a 123K-line monument to a conclusion you already reached. The 20% coverage threshold is a confession — 79K lines of tests but only 20% coverage means large sections are untestable. The lint suppressions for undefined names and unused variables reinforce this.

BSS of approximately zero is the only number that matters, and you're burying it. The scraping infrastructure concerns me most — how many scrapers actually connect to production? The governance system is solving the wrong problem — bank-vault security around an empty safe.

This looks like a research project that kept adding complexity hoping something would stick, then locked down the one thing that worked (barely). The priority isn't improving the model — it's deciding whether 90% of this code should exist at all.

### 5. The Executor

Actionable limitations ranked: 1) Model can't beat seeds — run feature importance on all 86 features against tournament outcomes specifically. 2) 530 tournament games is the real bottleneck — add data back to 2002, add NIT/CBI as auxiliary signal. 3) 123K LOC is active liability — archive non-production code, get under 20K LOC. 4) 20% test coverage means you can't safely change anything — get to 80% on the 7-feature pipeline. 5) Audit which scrapers feed production, kill the rest. Stop building infrastructure. Start beating seeds.

---

## Peer Reviews

**Anonymization mapping:** A = Expansionist, B = Executor, C = Outsider, D = Contrarian, E = First Principles Thinker

### Review 1
- **Strongest:** D (Contrarian) — identifies LOYO sample size invalidates model comparison
- **Biggest blind spot:** A (Expansionist) — gambling advice, not engineering
- **All missed:** Bracket optimization as the actual product where BSS ≈ 0 can still generate pool edge

### Review 2
- **Strongest:** D (Contrarian) — attacks statistical validity of evaluation
- **Biggest blind spot:** A (Expansionist) — ignores Vegas beats seeds through last-minute info, not historical features
- **All missed:** Whether 7 features are collinear proxies for seed strength; undefined product scope; no deployment target

### Review 3
- **Strongest:** D (Contrarian) — quantifiable attack on LOYO methodology
- **Biggest blind spot:** A (Expansionist) — optimism dressed as strategy
- **All missed:** Bracket optimization as the actual product; correlation-aware bracket construction

### Review 4
- **Strongest:** D (Contrarian) — invalidates the "ceiling" conclusion
- **Biggest blind spot:** A (Expansionist) — doesn't acknowledge sample size problem
- **All missed:** Whether the pipeline itself has bugs (20% coverage + suppressed lint = silent data errors possible)

### Review 5
- **Strongest:** D (Contrarian) — concrete statistical argument
- **Biggest blind spot:** A (Expansionist) — expensive exploration without evidence
- **All missed:** Feature collinearity with seed; calibration in specific matchup bands; possible silent data bugs

---

## Chairman's Verdict

### Where the Council Agrees
- BSS ≈ 0 is the defining fact — the system matches seed baseline
- The codebase is massively oversized for what it delivers
- Quality controls are theater (20% coverage, suppressed F821, neutered mypy)
- Disabled infrastructure is dead weight, not latent value

### Where the Council Clashes
- Is there signal beyond seeds? Contrarian/First Principles say information-theoretic wall; Expansionist says Vegas proves otherwise. Peer review: Vegas uses real-time data unavailable to historical models.
- Should disabled infrastructure be revived or deleted? Peer reviews unanimously sided against revival without new evidence.

### Blind Spots the Council Caught
- Silent data bugs possible (20% coverage + F821 suppression)
- 7 features may be collinear proxies for seed — BSS ≈ 0 could be tautological
- Bracket optimization is the actual product, not game-level BSS
- LOYO evaluation is statistically underpowered (59 games per fold)

### The Recommendation
Hard reset on scope, not more model complexity. Reframe success metric around bracket pool value. Audit for data bugs before concluding anything about ceilings. Cut to <15K LOC. Check feature independence from seed.

### The One Thing to Do First
Run a correlation matrix of the 7 production features against seed number. If r > 0.7 for all features, BSS ≈ 0 is tautological. If features are independent and BSS is still ≈ 0, pivot to bracket optimization. Either answer changes everything. It takes 20 minutes.
