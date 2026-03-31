# LLM Council Transcript — March 31, 2026

## Original Question
> What are this repo's most critical limitations?

## Framed Question
This is a March Madness forecasting system (~124K lines of Python across 294 source files, 203 test files). It scrapes data from multiple sources (T-Rank/Barttorvik, ESPN, sports-reference, KenPom, betting markets, Kaggle), engineers features, trains ML models (LightGBM, XGBoost, logistic regression, GNN, transformer, Bayesian Bradley-Terry), calibrates probabilities, runs Monte Carlo simulations (50K), and optimizes bracket picks for ESPN pool competitions. Production config disables GNN/transformer/agent orchestration. The codebase has heavy dependencies (PyTorch, torch-geometric, scipy, scikit-learn, LightGBM, XGBoost), 20+ scrapers tied to specific website structures, a 3500-line CLI entry point, multiple layers of abstraction (pipeline → runner → predictor → engine), and extensive configuration. Training data spans 2016-2024 with 2025 as holdout. What are this repo's most critical limitations?

---

## Advisor Responses

### 1. The Contrarian

The fatal flaw is that this system is a brittle house of cards dressed up as engineering.

**20+ scrapers are a liability, not a feature.** Website structures change without notice. One ESPN layout refresh, one KenPom paywall change, one sports-reference rate-limit policy update kills your data pipeline days before the tournament. You have no control over any of these dependencies.

**The 3500-line CLI entry point is disqualifying.** That's not an entry point, that's where complexity goes to die. It signals that the abstraction layers (pipeline → runner → predictor → engine) failed to contain complexity and it all leaked into one file. This thing is untestable and unmaintainable under tournament-week pressure.

**Training on 2016-2024 with 2025 as holdout is a single holdout year.** March Madness has enormous variance. One year's results prove nothing. You need rolling holdouts or you're just measuring luck.

**Production disabling GNN/transformer/agent orchestration is a confession.** You built it, it doesn't work reliably enough to run in production, but it's still in the codebase adding weight and cognitive overhead.

**50K Monte Carlo simulations sounds rigorous but masks the real problem:** garbage inputs produce confident garbage outputs. If your scrapers pulled stale or malformed data quietly, your calibrated probabilities are precise lies.

**The deepest flaw:** this system was built for a competition where you get one shot per year. There's almost no way to know if it actually works versus got lucky. 124K lines and 8 years of data might be elaborate overfitting to historical tournament noise.

The question to ask is not "what does this system do?" It's "what would prove this system doesn't work?" If you can't answer that, you don't have a forecasting system. You have a belief engine.

### 2. The First Principles Thinker

The question being asked is "what are the limitations?" But the real question is: **what problem is this system actually solving, and is this the right way to solve it?**

Strip it down. March Madness bracket optimization for ESPN pools has one core objective: finish in the top percentile of a specific pool. That is a **social game theory problem**, not a forecasting accuracy problem. The optimal bracket is not the most accurate bracket — it's the bracket that maximizes your expected rank given what other pool participants will pick.

This system has inverted the problem. It built a 124K-line forecasting engine when the actual bottleneck is modeling the **pick distribution of your competition**. If everyone in your pool picks Duke, picking Duke has near-zero expected value even if Duke is the correct favorite.

The second foundational error: **the system treats historical data as stable signal**. Tournament outcomes from 2016-2024 reflect rosters, coaching staffs, and tournament formats that no longer exist. The model is pattern-matching on noise dressed as signal.

Third: the 50K Monte Carlo simulations assume the probability estimates feeding them are well-calibrated. They are not — they cannot be. You have ~500 tournament games per training year, massive class imbalance by seed matchup, and distributional shift every season. Sophisticated simulation infrastructure on top of unreliable priors is precision theater.

The actual limiting factor isn't technical complexity — it's that the system is solving for forecast accuracy when it should be solving for **contrarian value relative to pool behavior**. Everything else is downstream of that misalignment.

### 3. The Expansionist

The system is being used as a bracket optimizer when it should be positioned as a **prediction market edge finder**. That's the buried opportunity.

With calibrated probabilities from 6+ model ensembles, Monte Carlo at 50K runs, and betting market data already ingested, this system generates something extremely valuable: divergence signals between its own probability estimates and market-implied probabilities. Right now that signal feeds bracket picks. It should feed **a Kelly-criterion position sizing engine for prop bets and futures markets**.

The 2016-2024 training window is actually an asset hiding as a limitation. That's 8 full tournaments of round-by-round outcome data. Most public bettors work from narrative and recency bias. An 8-year calibration curve on upset rates by seed matchup, combined with live line ingestion, creates systematic edges the market hasn't priced.

The GNN and transformer are disabled in production — almost certainly because of fragility or compute cost. But graph neural networks on team-player-game relationships are exactly what sportsbooks **don't** have modeled well. Re-enabling them selectively for first-round upset detection (highest variance, most mispriced games) could produce outsized edge.

The biggest upside: the entire architecture generalizes. The scraper infrastructure, feature engineering, calibration pipeline, and Monte Carlo engine aren't March Madness specific — they're a **sports prediction platform**. NFL playoffs, NBA playoffs, NCAAW — the marginal cost of extending to a new tournament is a fraction of the original build.

The system is already 80% of the way to something far larger. The limitation isn't technical — it's that the use case has been artificially capped at "win my office pool."

### 4. The Outsider

The 3500-line CLI entry point is a fire alarm that nobody's treating as a fire. That single number tells me the system has never been seriously questioned about what it actually needs to do. Real tools get broken apart. This one grew.

"Pipeline → runner → predictor → engine" is four words for the same thing. That's not architecture, that's a system that kept adding layers every time something broke instead of fixing the underlying design. Each layer is a place where a bug can hide and a future maintainer can get lost.

20+ scrapers tied to specific website structures means this entire system can be destroyed overnight by a terms-of-service change or a website redesign at ESPN or KenPom. That's not a technical limitation, that's a single point of failure that has nothing to do with your code quality.

The most honest limitation: this system cannot explain why it made a pick. You have six different model types, Monte Carlo on top, bracket optimization on top of that. If the bracket is wrong, there is no audit trail a human can actually follow. That's not a forecasting system, that's a black box with a confidence score stapled to it.

Training on 2016-2024 with 2025 as holdout sounds rigorous. It isn't. March Madness has roughly 67 games per year. That's around 600 games total across eight years. You're training complex neural architectures on what amounts to a small dataset with high variance outcomes, then validating on one tournament that could be an outlier in either direction.

The system is impressively large. Large is not the same as reliable.

### 5. The Executor

The bottlenecks that will actually bite you when you go to run this thing:

**The 3500-line CLI entry point is a deployment disaster.** When something breaks at 11pm before Selection Sunday, you will not find it fast. That file is your biggest operational risk.

**20+ scrapers against live websites means the system fails silently and randomly.** Sites change HTML structure without warning. One dead scraper poisons your feature pipeline and you won't know until you're looking at garbage predictions. There's no Monday morning fix for this except continuous monitoring you probably don't have.

**Production disabling GNN and transformer means you trained models you can't use.** Those model weights are sunk cost. Your actual production ensemble is a subset of what you built, which raises the question: does the calibration and Monte Carlo logic account for the *actual* production model mix, or the full set?

**2025 holdout with 2026 as the current year.** Your holdout is last year. You have no validated out-of-sample performance on the current tournament. The system is effectively blind to whether 2025 distribution shifts carried into 2026.

**50K Monte Carlo simulations with heavy PyTorch dependencies means cold-start time is real.** How long does a full run take? If it's hours, you can't rerun after late injury news drops.

The single most actionable thing: instrument every scraper with timestamped health checks that alert on failure before they silently return stale or empty data. Everything else is architectural debt. That one is a same-day fire that kills your bracket without warning.

---

## Peer Review Round

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Reviewer 1
**Strongest: B** — Correctly reframes as game theory problem. Most structurally important insight.
**Biggest blind spot: C** — Pivots to different product with different legal/operational requirements. Mistakes scope expansion for limitation analysis.
**All missed:** Temporal leakage risk — if features use data not available at bracket lock, calibration is corrupted.

### Reviewer 2
**Strongest: D** — Most grounded, names concrete failure modes, goes beyond symptoms to statistical constraints (~600 games too small for neural architectures).
**Biggest blind spot: C** — Recommends expansion without engaging sample-size constraint. Kelly-criterion advice ignores known calibration defects documented in AUDIT_DATA_LEAKAGE.md.
**All missed:** The repo has an existing audited data leakage problem — retrospective model selection pressure means even training metrics may be inflated by accumulated selection bias over 2016-2024.

### Reviewer 3
**Strongest: A** — Every claim maps to real code (confirmed 3565-line main.py, GNN disabled by default, LOYO at ~63 games/fold).
**Biggest blind spot: C** — Criticizes system for not doing things it already does. Kelly criterion exists in leverage.py. Pool optimization exists.
**All missed:** Confirmed in-sample calibration fitting path (p_fit = p_eval = p_arr) documented in AUDIT_DATA_LEAKAGE.md — the flaw that inflates reported model quality on tournament data where calibration matters most.

### Reviewer 4
**Strongest: A** — All claims verifiable. GNN disabled, LOYO validated at ~63 games/fold with SE ~0.009.
**Biggest blind spot: B** — Asserts system ignores pool competition behavior. Factually wrong: pool_optimizer.py, leverage.py, bracket_portfolio.py all model contrarian value.
**All missed:** LOYO protocol explicitly documents that no improvement <1% Brier is statistically distinguishable from noise. Hyperparameter tuning and model selection rest on validation that can't reject null. 124K lines overfitting to evaluation protocol, not data.

### Reviewer 5
**Strongest: A** — Claims verifiable (3565-line main.py, 92 scraper files, LOYO SE ~0.009).
**Biggest blind spot: B** — Asserts system ignores pool competition. Factually wrong.
**All missed:** The system's own LOYO documentation states SE ~0.009 per fold. The backtest architecture acknowledges its own unreliability in a comment, yet numeric promotion gates assume it is reliable. Self-defeating validation loop.

---

## Chairman Synthesis

### Where the Council Agrees

Three structural failures are independently confirmed:

1. **Scraper dependency is a single point of failure.** All five advisors named it. 20+ scrapers tied to specific website structures means any TOS change, HTML restructure, or IP block silently corrupts the data pipeline.

2. **The 3500-line CLI is a real architectural failure.** Confirmed at 3,565 lines. Debugging under tournament-week time pressure is nearly impossible.

3. **The validation regime cannot support the architecture's complexity.** LOYO folds contain ~63 games each (SE ~0.009 on Brier). No improvement smaller than ~1% Brier is statistically distinguishable from noise. The model selection process rests on a protocol that cannot reject the null hypothesis.

### Where the Council Clashes

**Forecasting vs. game theory:** First Principles argues the system solves the wrong problem. Peer review rebuts: pool_optimizer.py, leverage.py, bracket_portfolio.py already model contrarian value. The weighting question remains genuinely contested.

**Over-engineered vs. appropriately ambitious:** The Contrarian treats 124K lines as failure. The Expansionist treats the architecture as an underutilized asset. The real question — what complexity is defensible given ~63 games per validation fold — was not fully resolved.

### Blind Spots the Council Caught

1. **Confirmed data leakage:** AUDIT_DATA_LEAKAGE.md documents in-sample calibration fitting (p_fit = p_eval = p_arr). Calibration metrics may be optimistic.
2. **Accumulated selection bias:** 8 years of retrospective model selection inflate apparent performance.
3. **2025 holdout consumed:** No validated out-of-sample performance remains.
4. **Temporal leakage risk:** Unverified whether features use data not available at bracket lock time.

### The Recommendation

**Stop adding models and start measuring what you have honestly.**

The fundamental problem is that the evaluation infrastructure cannot tell you whether the system works. Fix the data leakage, re-run calibration cleanly, and accept that you likely cannot distinguish model quality at this sample size. Then either acquire more data or simplify the architecture to match what the data can support.

### The One Thing to Do First

Fix the data leakage in AUDIT_DATA_LEAKAGE.md — specifically the p_fit = p_eval = p_arr calibration path — refit calibration from scratch, and re-run your validation suite before touching anything else.
