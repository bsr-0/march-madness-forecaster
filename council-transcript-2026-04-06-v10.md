# LLM Council Transcript — Session #15
**Date:** 2026-04-06
**Question:** Are locally-computed four factors valid for backtesting?

---

## Original Question
Does the council agree that the four factors (eFG%, TO%, ORB%, FTR for offense and defense) are valid if computed locally from box scores, rather than fetched from Torvik?

## Framed Question
Are locally-computed four factors (eFG%, TO%, ORB%, FTR for offense and defense) valid for backtesting and prediction, or do they diverge from Torvik's published values in ways that matter?

**Context:**
- Locally-computed barthag was just shown to diverge at r=0.73 with catastrophic outliers (0.4+ barthag error for tournament teams). Fixed by using real Torvik data.
- Four factors are still computed locally from ESPN/cbbpy box-score data because Torvik's CSV endpoint is blocked by Cloudflare (even with curl_cffi).
- The claim: four factors are "direct counting stats" — same formula regardless of who computes them.
- The noseed model uses four factors as 8 of 12 feature dimensions.
- No validation has ever been done comparing local four factors to Torvik's published values.
- The live pipeline uses real Torvik data; the backtest uses locally-computed values.

---

## Advisor Responses

### 1. The Contrarian
The "direct counting stats" argument is seductive and almost certainly wrong in practice. Four factors are *rates*, not raw counts — rates require denominators. TO% uses possessions, ORB% uses rebound opportunities. Torvik derives possessions from his own formula. ESPN uses different possession estimates. Even 1-2 possession errors per game compound across 30+ games into meaningfully different TO% and ORB%.

Specific failure modes: (1) Possession estimation divergence — Torvik's formula likely differs from ESPN's. (2) Game coverage gaps — different game populations. (3) Opponent adjustment — Torvik's published four factors may be opponent-adjusted. (4) No validation data because Cloudflare blocks the endpoint.

The claim "same formula, so same result" has already been disproven once in this project. Assume divergence until proven otherwise.

### 2. The First Principles Thinker
The question being asked is "are the formulas correct?" The question that needs answering is "are we measuring the same things?"

Four factors are ratios with denominators that vary by source — especially possession estimates. Torvik uses his own possession formula, likely adjusted for pace and tempo. ESPN/cbbpy uses raw box-score possession proxies. If the possession denominators diverge, every rate stat diverges systematically.

The barthag catastrophe was not caused by a wrong formula. It was caused by different game samples, different data coverage, different edge-case handling. The exact same failure mode applies here.

The real question is not "is the formula valid?" It is: are the inputs identical? Same games, same possessions, same inclusion/exclusion rules. The Cloudflare block is a solvable operational problem. Running an unvalidated backtest because the endpoint is blocked is not a solution — it's a rationalization.

### 3. The Expansionist
The "direct counting stats" claim is almost certainly true for the numerators — FGM, FG3M, FGA are raw counts. But the divergence point is possessions, which affects ORB% and TO%.

Here's what nobody is saying: if you validate and the correlation is r=0.97+, you've confirmed a clean backtesting surface. That's a competitive advantage — a validated locally-computed pipeline means you can extend backward beyond Torvik's data availability, experiment with custom possession formulas, and generate live in-season features.

Barthag failed because it's a derived composite where errors compound through the Pythagorean formula. Four factors are one level simpler. The validation is one afternoon. Don't assume the worst — validate it and potentially discover you already have a working pipeline.

### 4. The Outsider
The exact same trap, twice in a row. You fixed barthag because "locally computed diverges from Torvik." Now the argument is "four factors are just counting stats, so they're fine." That's motivated reasoning — wanting the four factors to be okay.

No validation data exists. None. You're asserting safety based on a formula, not evidence. The live pipeline uses real Torvik data while the backtest uses locally-computed — that's train/test mismatch by definition.

The validation is simple: pull a sample season, compute correlations, look at outliers. This takes an afternoon. The cost of being wrong is 8 of 12 feature dimensions are corrupted. Check before you ship.

### 5. The Executor
The barthag lesson is sitting right there. Don't ask whether to learn it twice.

Step 1: Pick one season, compute correlation and MAE for each of the eight four-factor features between the two sources. Two hours. You either find r≥0.99 with tiny MAE (fine, move on) or you find r=0.73-style divergence (same problem).

Step 2: If divergence is found, you need a fix path — different endpoint, Torvik raw data, or retrain the noseed model on what you can reliably source.

Every hour spent theorizing is an hour you could have spent knowing.

---

## Peer Reviews

### Anonymization Mapping
- Response A = Contrarian
- Response B = First Principles Thinker
- Response C = Expansionist
- Response D = Outsider
- Response E = Executor

### Review 1
- **Strongest: D** — Names motivated reasoning precisely, connects to barthag precedent.
- **Blind spot: C** — Frames validation as upside, buries the risk. Assumes "simpler" means safer without evidence.
- **All missed:** Residual analysis by subgroup. High correlation doesn't mean safe if errors are systematic (biased by opponent strength, tempo, conference tier).

### Review 2
- **Strongest: E** — Exact action, cuts through theorizing.
- **Blind spot: C** — Motivated reasoning disguised as optimism.
- **All missed:** Asymmetry of fix path. If divergence found, can you fix it? If Torvik opponent-adjusts, there's no easy local substitute.

### Review 3
- **Strongest: E** — Measurement over theory.
- **Blind spot: B** — Dismisses barthag divergence too easily.
- **All missed:** Asymmetry of error. Downside of assuming validity and being wrong is catastrophic. Set r<0.99 as hard gate, no renegotiating.

### Review 4
- **Strongest: E** — Direct, actionable.
- **Blind spot: A** — Lists concerns without prioritizing.
- **All missed:** Directionality — single-season check won't catch year-over-year instability. Need per-season correlation checked for stability.

### Review 5
- **Strongest: E** — Cuts through debate.
- **Blind spot: C** — Motivated reasoning as pragmatism.
- **All missed:** Validation target may be wrong. If Torvik opponent-adjusts four factors, correlating against raw outputs is still broken.

---

## Chairman's Verdict

### Where the Council Agrees
Every advisor agrees: stop theorizing and measure. The debate about "counting stats" vs "derived rates" is unanswerable without data. Validation is a two-hour task with no legitimate reason it hasn't been done. 8 of 12 features being potentially corrupted is not a tolerable unknown. The barthag r=0.73 is a live precedent. The live pipeline using Torvik while the backtest uses local is a train/test mismatch by definition.

### Where the Council Clashes
The genuine disagreement is about prior probability. The Contrarian and Outsider argue the possession denominator problem is severe and the prior should be pessimistic. The Expansionist argues four factors are structurally simpler than barthag and the prior should be cautious but not catastrophic. Both are reasonable — the disagreement resolves through measurement. But the optimistic framing is motivated reasoning: "wanting four factors to be okay" is not the same as "four factors are okay."

### Blind Spots the Council Caught
1. **High correlation does not mean safe.** Systematic errors biased toward specific team types won't show in aggregate r-values. Need residual analysis by subgroup.
2. **Single-season validation doesn't catch year-over-year instability.** Data coverage changes across seasons. Need per-season correlations.
3. **If Torvik opponent-adjusts four factors, there is no easy fix.** Confirm whether Torvik's published values are raw or adjusted before running any validation — this determines what you're even measuring.

### The Recommendation
Do not use locally-computed four factors in the backtest until validated. Set the gate at r≥0.99 with no systematic residual bias, all eight features, all available seasons. Set the threshold before seeing results. If any feature fails, replace with Torvik's column or drop from the model. Running corrupted features is worse than fewer features.

### The One Thing to Do First
Before running any correlation, check Torvik's documentation to confirm whether his published four factors are raw or opponent-adjusted. This takes five minutes and determines whether the validation you're about to run is even asking the right question.
