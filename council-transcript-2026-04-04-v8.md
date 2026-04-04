# LLM Council Transcript — Session #13
**Date:** April 4, 2026
**Topic:** Data Provenance & Leakage Audit — What Concerns Should Be Investigated?

---

## Original Question

> Council this: what concerns, if any, should be investigated based on this data provenance and leakage audit?

## Framed Question

Given this March Madness bracket pool optimizer project, the team has completed a comprehensive data provenance and leakage audit (documented in `docs/data-provenance-and-leakage-audit.md`). The backtest uses exactly 4 data sources: tournament seeds (Selection Sunday, no risk), tournament results (ground truth, no risk), locally-computed Torvik barthag (date-filtered, low risk), and locally-computed Torvik four factors (date-filtered, low risk). All 47 contaminated files have been purged, all scraper paths now raise LeakageError post-tournament with strict_leakage=True by default, and providers.py bypass paths have been patched.

**Known gaps remain:**
1. `noseed_model._load_team_stats()` loads Torvik files without `_validate_pretournament()` — files are clean but no runtime guard
2. `team_metrics_{year}.json` contains post-tournament W/L records (e.g., UConn 2024 shows 37-3 instead of pre-tournament 31-3) — no metadata or guards, but not used by backtest
3. `historical_games_{year}.json` includes tournament games — by design, compute scripts filter them, but downstream code using these directly must apply its own date filter
4. Torvik barthag/four-factors are locally computed from ESPN box scores, not from barttorvik.com directly — potential accuracy difference vs the real Torvik methodology
5. The Pythagorean exponent (11.5) and 15-iteration opponent-strength adjustment are approximations of Torvik's proprietary algorithm

**What concerns, if any, should be investigated further based on this audit?**
Stakes: If leakage exists, backtest results are inflated and the pool optimizer gives false confidence. If the locally-computed metrics are significantly less accurate than real Torvik, the model's predictive power is degraded.

---

## Advisor Responses

### The Contrarian

Gap 1 is the one I'd bet money on biting you. "Files are clean" is a statement about the present. Runtime guards are statements about the future. Someone adds a new Torvik file, forgets the date-filter convention, and `_load_team_stats()` silently ingests it. You've documented this gap, which means you know the guard is missing, which means you're one bad file away from silent leakage. "Low risk" without a runtime check is not low risk - it's undetected risk.

Gap 3 is worse than it sounds. "Downstream code must apply its own date filter" is a design philosophy, not a constraint. That's the same logic as "developers should write secure code." It will fail. Someone writes a new feature, grabs `historical_games_{year}.json`, doesn't read the convention, doesn't filter. You've built a data minefield and called it a feature.

The Pythagorean exponent and iteration count concern me for a different reason. You're not just approximating Torvik - you're approximating an approximation of team strength and then validating the whole system against outcomes that the real Torvik methodology might predict differently. If your locally-computed metrics diverge significantly from actual Torvik in specific game contexts (blowouts, pace extremes), your backtest accuracy doesn't translate to forward performance. You need a direct correlation check: run your computed metrics against real Torvik numbers on the same teams and years. If R-squared isn't above 0.95, you're not computing Torvik - you're computing something else with a borrowed name.

The audit is thorough. The gaps are documented. Documentation of unguarded gaps is not the same as closing them.

### The First Principles Thinker

The audit is answering "do we have leakage?" when the real question is "are our backtest results meaningful?"

These are different questions, and gap #4 is the one that exposes it.

You've built a system that approximates Torvik. But you're validating it against tournament outcomes as if it *is* Torvik. The entire predictive value of barthag rests on Torvik's methodology — opponent strength adjustments, tempo normalization, the specific game-weighting choices. Your approximation may correlate with Torvik but diverge precisely in the cases that matter: mid-major teams, late-season momentum, teams with unusual tempo profiles. Tournament upsets cluster in exactly these edge cases.

The first-principles question: **what are you actually backtesting?** If your locally-computed barthag diverges from real Torvik by even 15-20% on certain team profiles, your model isn't a Torvik-based model — it's an unknown model wearing Torvik's label. You have no baseline comparison to know.

Gap #5 compounds this. The Pythagorean exponent and opponent-strength iterations aren't implementation details — they're load-bearing assumptions. Torvik's proprietary choices in these parameters are likely tuned against decades of outcomes. Your choices are guesses.

The leakage audit is clean. That's necessary but not sufficient. The harder investigation is: run your locally-computed barthag against published Torvik ratings for 2019-2023 and compute the correlation. If it's below 0.95, you have an accuracy problem that no amount of leakage prevention fixes. If you can't access published Torvik ratings to do this comparison, that itself is a finding worth stating explicitly before trusting any backtest result.

You're not forecasting brackets. You're forecasting how well your approximation of a proprietary metric predicts outcomes. Those are different problems with different confidence intervals.

### The Expansionist

The audit found four data sources. That's a constraint masquerading as a fact.

The real question buried in gap #4 and #5: you've reverse-engineered Torvik's methodology using ESPN box scores and approximated his Pythagorean exponent and opponent-adjustment iterations. This isn't a leakage risk — it's an **opportunity signal**. If your locally-computed metrics diverge from real Torvik, that divergence is information. You might have accidentally discovered a *better* version.

Run the comparison. Pull real Torvik ratings for historical years and correlate them against your locally-computed barthag. If your r-squared is 0.97+, the approximation is fine and you can stop worrying. But if there are systematic divergences for specific team profiles — high-tempo outliers, weak-schedule teams — those gaps tell you where real Torvik is *also* uncertain, which means your model and real Torvik both have blind spots in the same places. That's a competitive edge to exploit, not a bug to patch.

Gap #3 is similarly undervalued. historical_games_{year}.json includes tournament games and downstream code must apply its own date filter. That means you have tournament game data sitting there, structured and ready. That's a training signal for tournament-specific performance patterns — home court irrelevance, pace normalization under pressure, upset resilience — that regular-season data cannot capture. The audit treated this as a contamination risk. It's actually an untapped feature set for a separate tournament-context model.

The leakage audit was the right defensive move. Now flip it: every "clean" constraint you identified is a place where better data could expand the model's ceiling. The backtest isn't the end product — it's the floor.

### The Outsider

The audit describes a system I know nothing about, and that's exactly why I can flag what jumps out.

Gap 2 is the one I'd actually investigate: `team_metrics_{year}.json` contains post-tournament records but "is not used by backtest." That phrase is doing enormous work. How do you know? Was this confirmed by tracing every code path that reads that file, or by reading the description of what the backtest is supposed to do? Those are very different levels of confidence. Data files that "aren't used" have a way of getting used — by a helper function, by a fallback, by a future change nobody remembered to audit. The fact that this file exists in the project with contaminated data and no guards is an active liability, not a closed finding.

Gap 3 has the same problem stated more honestly: "downstream code using these directly must apply its own date filter." That's not a gap, that's a known hole with future trust placed in developers who haven't written that code yet. Who is responsible for ensuring that filter exists? Right now, nobody is.

Gap 4 is the one the team seems least troubled by, but I'd want to know: has anyone actually compared the locally-computed Torvik approximation against real Torvik numbers for even one year? "Potential accuracy difference" is not an answer. A 10% deviation in a key metric might not matter. A systematic bias — where the approximation is consistently optimistic about certain team types — could corrupt the model's predictions in patterned ways that backtest results would never reveal.

The team has done real work here. But "files are clean" and "not used by backtest" need to be proven, not asserted.

### The Executor

Gap 1 is the only one worth fixing Monday morning. Everything else is either by design, documented, or theoretical.

Here's the action: open `noseed_model.py`, find `_load_team_stats()`, add `_validate_pretournament()` before the file read. One function call, one line of code. Done in 20 minutes. Without it, you have a runtime gap that could silently load a wrong file if someone runs the backtest against a future year's unfiltered data. The audit says files are currently clean - fine, but "currently clean" isn't a guard. The guard is the guard.

Gap 2 is not a backtest problem. You said it's not used by backtest. Stop thinking about it.

Gap 3 is handled by design. Document it in a comment in the compute script if you're worried, but don't build more machinery around something already working.

Gaps 4 and 5 are research questions, not engineering tasks. You can't fix an approximation with code. If you want to validate accuracy, run your locally-computed barthag against known Torvik values for 5 historical seasons and compute correlation. That's a half-day analysis, not a refactor. If correlation is above 0.95, ship it. If it's not, you have a different conversation.

Priority order:
1. Fix Gap 1 today - it's a 20-minute code change
2. Run the barthag correlation check once to put Gap 4/5 to rest permanently
3. Everything else is noise

The leakage risk is contained. The only open execution item is the missing runtime guard in noseed_model. Everything else in this audit is already resolved or is inherently unresolvable with code.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Review 1
**Strongest: D (Outsider)** — The only response that challenges the verification method itself. "Not used by backtest" is an assertion, not a proof. D correctly identifies that tracing claimed non-usage requires code path analysis, not documentation review.

**Biggest blind spot: E (Executor)** — Treats Gap 2 as solved because the file "isn't used." That's precisely the unverified assertion D flags. Also prescribes a 0.95 correlation threshold with no justification for why that number is sufficient.

**All missed:** Training/test split methodology. Even with clean data and accurate Torvik approximations, if the backtest evaluates on years whose outcomes influenced feature engineering, hyperparameter choices, or model selection decisions, the backtest is still optimistic. The 47-file purge addresses data leakage, not researcher degrees of freedom leakage.

### Review 2
**Strongest: D (Outsider)** — Challenges the audit's own epistemology. Code-path tracing and documentation are different things.

**Biggest blind spot: E (Executor)** — Dismisses without evidence. Frames the barthag correlation check as a one-time pass/fail gate (0.95 = ship), ignoring that systematic bias on specific team profiles matters more than aggregate correlation.

**All missed:** The backtest evaluation metric itself. If the scoring function doesn't match real pool dynamics, clean data and accurate metrics can still fail. No one questioned what "backtest results are inflated" actually means in the context of bracket pool scoring.

### Review 3
**Strongest: D (Outsider)** — Most rigorous. Demands code-path tracing rather than assumption.

**Biggest blind spot: E (Executor)** — Overconfident dismissal without verification.

**All missed:** Were the 47 purged files also removed from evaluation, not just training? If contaminated files defined the test set, purging from training doesn't eliminate leakage.

### Review 4
**Strongest: D (Outsider)** — Interrogates the audit's epistemics.

**Biggest blind spot: E (Executor)** — No justification for dismissing Gap 2.

**All missed:** If contaminated data was previously ingested during any model training run, purging files doesn't un-train the model. Were model weights or learned parameters fit on contaminated data before the audit?

### Review 5
**Strongest: D (Outsider).**

**Biggest blind spot: E (Executor).**

**All missed:** Backtest evaluation methodology — even with zero leakage, backtest could be invalid if it uses a biased scoring metric, non-representative season sample, or overfits hyperparameters to historical tournament variance.

---

## Chairman Verdict

### Where the Council Agrees

**The Torvik correlation check is non-negotiable.** Every advisor touched this, and the peer reviews amplified it. You are backtesting a model that may not be Torvik — it may be an unknown model wearing Torvik's label. The guessed Pythagorean exponent and iteration count are not cosmetic choices. They are the calibration constants Torvik tuned against decades of outcomes. If your locally-computed barthag diverges meaningfully from real published Torvik ratings, every insight you've drawn from the backtest is an insight about a model nobody has validated.

**Gap 1 needs a runtime guard, not documentation.** The noseed_model loading without a runtime check is universally flagged. Documented gaps are not mitigated gaps. The Contrarian named this precisely: "low risk without a runtime check is undetected risk." The Executor agreed it's a 20-minute fix. There is no disagreement here.

**"Not used by backtest" is an assertion, not a proof.** The Outsider said it most clearly, and all five peer reviews validated it as the strongest individual contribution to the council. team_metrics contains post-tournament records. If that status was established by reading intent rather than tracing code paths, it is not established.

### Where the Council Clashes

**Gap 2 (team_metrics) — active liability vs. non-issue.** The Outsider treats it as an uninvestigated active risk. The Executor dismisses it entirely. The disagreement hinges on a verification question: has anyone actually traced every code path to confirm team_metrics is unreachable from the backtest? This is a factual question with a deterministic answer.

**Gap 3 (tournament games in historical_games) — contamination vs. feature.** The Expansionist sees tournament games as untapped signal. The Contrarian sees reliance on downstream discipline as doomed to fail. Both are correct: tournament games can be a valid training signal if accessed through an explicit, guarded interface — not through a convention developers are trusted to follow.

**The 0.95 R-squared threshold.** The Executor stated it confidently. Two peer reviews called it unjustified. Aggregate R-squared can hide systematic divergence on mid-major and unusual-tempo profiles — exactly where tournament upsets cluster. Profile-stratified residuals matter more.

### Blind Spots the Council Caught

**Researcher degrees of freedom leakage.** Clean files do not reset researcher memory. If hyperparameters or feature choices were tuned while observing backtest results on those same years, the backtest is optimistic regardless of data cleanliness.

**Whether contaminated data already trained the model.** Purging files prevents future contamination. If model weights were fit on contaminated data before the audit, purging does not un-train them.

**Backtest scoring validity.** Even with zero leakage and accurate metrics, if the backtest scoring function doesn't model real pool dynamics — field size, chalk distribution, tiebreakers — clean data still fails to predict pool performance.

### The Recommendation

Run the Torvik correlation check before trusting any backtest result, but measure it correctly. Aggregate R-squared against published Torvik ratings for 2019-2023 is the starting point, not the finish line. Stratify the residuals by team profile: mid-majors vs. power conferences, high-tempo vs. low-tempo, top-25 seeds vs. mid-seeds. If you see systematic divergence in any stratum, you have a biased model, not just an approximate one.

Simultaneously, trace team_metrics access paths in code, not in documentation. Add the noseed_model runtime guard (20-minute fix). Confirm whether any model artifacts were trained on pre-purge contaminated data — if yes, rebuild from scratch.

### The One Thing to Do First

Pull published Torvik ratings for 2019-2023 and run your locally-computed barthag against them — stratified by seed range and conference tier, not just in aggregate. Everything else in this audit depends on the answer to one question: are you backtesting Torvik, or are you backtesting something else?
