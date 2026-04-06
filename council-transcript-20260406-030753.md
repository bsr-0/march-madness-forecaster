# LLM Council Transcript
**Date:** April 6, 2026
**Topic:** Should cbbdata.com API remain primary source for four factors, or promote trank.php CSV?

---

## Framed Question

Should cbbdata.com API remain the primary source (Strategy 1) for four factors data in a March Madness forecasting pipeline, or should trank.php CSV be promoted to primary?

**Context:**
- cbbdata API: Clean JSON, no date filtering, protected by `_check_tournament_date_guard()` (hard LeakageError)
- trank.php CSV: Server-side begin/end date params + same guard, but requires browser-like headers for Cloudflare
- Player CSV fallback: No date filtering, approximate stats, missing defensive factors
- Historical years: Recomputed from local box scores via `compute_pretournament_four_factors.py`
- Guard uses `TOURNAMENT_START_DATES` — hardcoded dict in `pipeline/config.py`

---

## Advisor Responses

### The Contrarian
The tournament date guard is a single point of failure protecting you from catastrophic leakage, and you're treating it like a solved problem.

Here's what keeps me up at night: `_check_tournament_date_guard()` raises a `LeakageError` based on what exactly? A hardcoded date? A config value? If that date is wrong by even one day — a timezone issue, a config typo, someone bumping the tournament start date in a non-obvious place — the API silently returns contaminated data with zero indication anything went wrong. The model trains, backtests look great, and you ship predictions built on tournament results. You won't know until your bracket is garbage.

trank.php's server-side date filtering is architecturally superior for one reason: the contamination protection lives closer to the data source and doesn't depend on your application code being correct. Your guard can have a bug. The trank.php server filtering the data before it even arrives cannot be circumvented by your bugs.

The Cloudflare fragility argument cuts both ways. Yes, trank.php can break. But when it breaks, it breaks loudly — requests fail, you know immediately. The API's leakage failure mode is silent corruption.

The real question nobody is asking: have you actually audited what the cbbdata.com API returns for a season when scraped on, say, Selection Sunday? Do you have a test that verifies the guard fires correctly across timezone boundaries? Do you know what happens if `CBD_API_KEY` expires mid-season and someone patches around it without reading the guard logic?

Single-layer leakage protection on your most critical features isn't a design — it's a bet.

### The First Principles Thinker
The question being asked is "which source is better?" But the real question is: what is the actual leakage risk surface, and does the source choice materially change it?

Strip away the framing. The cbbdata API doesn't support date filtering — but this is irrelevant if `_check_tournament_date_guard()` makes the entire scrape path unreachable post-tournament-start. The guard isn't a soft warning; it's a hard error. So the API's lack of date filtering is a non-issue in production. You never reach the point where contaminated data could be returned.

For historical years, neither source is scraped at all — box scores are recomputed locally. This is the right architecture. The source comparison is therefore only relevant for one specific scenario: scraping current-year pre-tournament data during the brief window between season-end and tournament-start.

In that window, both sources work. trank.php's server-side filtering is redundant given the guard already exists. Its "belt-and-suspenders" framing treats a structural control as if it were weak. It isn't.

The real risk isn't the data source. It's cache contamination — stale cache from a legitimate pre-tournament scrape persisting into post-tournament inference. The question doesn't mention whether cache invalidation is enforced at prediction time, only that "cache timestamp validation catches it." That validation logic deserves more scrutiny than the source selection debate.

The wrong question: cbbdata vs trank.php. The right question: Is cache invalidation airtight at inference time, and does the system have an end-to-end integration test that deliberately simulates a post-tournament prediction attempt and confirms it fails loudly?

Source choice is secondary architecture. Cache integrity is primary.

### The Expansionist
The framing of this decision undersells what's actually on the table.

trank.php isn't just "more robust leakage protection" — it's a richer data pipeline waiting to be fully exploited. Server-side date filtering means you can trivially generate rolling snapshots: Four Factors at week 4, week 8, week 12, conference tournament start. That's trajectory data. How a team's eFG% defense *trends* into March is a fundamentally different signal than their season aggregate. The current cbbdata.com architecture makes this impossible without significant gymnastics.

Here's what nobody's talking about: if trank.php supports arbitrary begin/end date windows, you can construct conference-schedule-only splits, exclude non-conference cupcakes, or build momentum features over the final 10 games. That's feature engineering that directly addresses one of the hardest problems in March Madness prediction — late-season form versus full-season resume.

The Cloudflare concern is real but solvable. Browser-like headers is table stakes. If the bigger unlock is time-windowed features, the engineering cost to maintain a reliable trank.php scraper is trivially justified.

The hidden upside: once you normalize on a source that supports date parameterization, you can backfill historical snapshots the same way. Consistent feature construction across training and inference isn't just cleaner — it can meaningfully improve model calibration because you're eliminating a systematic asymmetry between how historical features were computed versus live features.

Promote trank.php. Then immediately ask what other features on that site support date windowing. That's where the real leverage is.

### The Outsider
The description tells me the cbbdata.com API has no date filtering and relies entirely on a guard function that throws an error if you try to use it after tournament start. That's one lock on the door.

trank.php has two locks: server-side date filtering built into the query, plus the same guard function.

I don't need to understand machine learning or March Madness to see the obvious thing here: when the cost of failure is "your model is secretly cheating and you won't know it," you want more locks, not fewer.

The cbbdata.com argument seems to be "it returns cleaner JSON." That's a formatting preference. You're trading structural safety for convenience of parsing.

The Cloudflare/User-Agent concern with trank.php is real, but it's a reliability problem, not a correctness problem. If trank.php breaks, you know it broke — the scrape fails loudly. If cbbdata.com silently includes tournament games because someone ran it one day late, or the guard had an off-by-one on the tournament start date, or a timezone edge case — you might never know. The model just quietly becomes contaminated.

From where I'm sitting: a system that fails loudly when fragile is safer than a system that succeeds silently when wrong.

One question the experts might be too close to ask: how confident are you that the tournament start date used by the guard is always accurate? If that date is hardcoded or pulled from a config, it's a single point of failure. trank.php's server-side filtering doesn't share that dependency.

Promote trank.php. Fix the fragility separately.

### The Executor
The leakage guard is the critical dependency here, and it's already built. So the "which source" question collapses into: which one fails more gracefully when the guard isn't enough?

Keep cbbdata.com as primary. Here's why this is a practical decision, not a theoretical one:

The `_check_tournament_date_guard()` is a hard error. It blocks the scrape entirely. That means the risk surface for cbbdata.com is: someone disables the guard, or the guard has a bug in the tournament start date logic. Those are auditable. You grep for guard calls, you write one test that confirms the error fires on the right date, and you're done. Monday's task: `pytest test_leakage_guard.py`. Thirty minutes.

trank.php's server-side filtering sounds like belt-and-suspenders but it's actually a different failure mode: Cloudflare blocks you silently, User-Agent rotation breaks, the endpoint changes its date param behavior quietly, and now you're either getting no data or wrong data with no error raised. You don't find out until model accuracy tanks.

The fragility of trank.php isn't theoretical — it requires browser-like headers TODAY. That's already a maintenance debt. Promoting it to primary means you're one Cloudflare rule update away from a silent failure during the exact three-week window this system matters.

Concrete next step: Write a single test that fires `_check_tournament_date_guard()` with a mocked date of tournament start + 1 day and confirms it raises `LeakageError`. That test is your actual protection. The source debate is secondary to that.

Keep cbbdata primary. Fix the guard test. Ship it.

---

## Peer Reviews

### Review 1
**Strongest: A (Contrarian)** — identifies most dangerous failure mode and demands evidence of testing rather than asserting the guard works.
**Biggest blind spot: C (Expansionist)** — treats Cloudflare as solved; building a rolling-snapshot feature pipeline on browser-like headers is building on sand.
**All missed:** Nobody asked how the tournament date gets into the system and who is responsible for keeping it current.

### Review 2
**Strongest: D (Outsider)** — clearest structural argument separating correctness from reliability.
**Biggest blind spot: E (Executor)** — treats "auditable" as equivalent to "reliable." A test passing in CI doesn't protect against runtime drift.
**All missed:** Where does the guard's cutoff date come from — hardcoded, config-driven, or dynamic? That single detail determines the entire debate.

### Review 3
**Strongest: A (Contrarian)** — identifies failure mode asymmetry with precision.
**Biggest blind spot: E (Executor)** — treats cbbdata's stability as given while dismissing Cloudflare. Both are external dependencies.
**All missed:** The decision conflates production inference with feature engineering. These can use different primary sources. A hybrid architecture wasn't considered.

### Review 4
**Strongest: A (Contrarian)** — asks the right adversarial questions about the guard's implementation.
**Biggest blind spot: E (Executor)** — treats a single test as structural defense. A test passing says nothing about whether the date is actually correct.
**All missed:** The guard date implementation (hardcoded vs dynamic) is the crux. No advisor examined it.

### Review 5
**Strongest: A (Contrarian)** — identifies failure asymmetry with actionable diagnostics.
**Biggest blind spot: C (Expansionist)** — mixing safety with feature roadmap muddies the decision. Promotion should stand on safety alone.
**All missed:** Observability. Without request logging, response hashing, or artifact provenance tracking, you can't audit what data the model actually saw.

---

## Chairman Verdict

### Where the Council Agrees
The guard is load-bearing and under-tested. Every advisor acknowledges it's the real protection mechanism. The failure mode asymmetry is real — loud failure (trank.php/Cloudflare) is safer than silent corruption (cbbdata with misconfigured date). 4 of 5 peer reviews confirmed this.

### Where the Council Clashes
The Executor vs. everyone else on Cloudflare fragility. Both sides are right about different failure modes. The real disagreement: which failure mode you trust yourself to catch — infrastructure failures (observable) vs. config drift (invisible until audit). The "write a test" argument is the sharpest split — 3 peer reviews explicitly reject it.

### Blind Spots the Council Caught
1. **Tournament date provenance** (4 reviewers): `TOURNAMENT_START_DATES` is a hardcoded dict requiring manual yearly update. The guard fails open if the entry is missing or wrong. No test catches this.
2. **Observability** (1 reviewer): Can you audit which source was used and what data the model saw? Without provenance tracking, you can't diagnose or reproduce.
3. **Split-use architecture** (1 reviewer): Production inference and feature engineering can use different sources. The binary framing is artificial.

### The Recommendation
**Promote trank.php to primary.** Treat Cloudflare fragility as a first-class reliability problem to solve immediately — not later. The guard's single point of failure is a hardcoded config dict. trank.php with server-side date filtering provides a second independent correctness mechanism. Choose the loud failure mode over the silent one. Do not promote because of the feature roadmap — promote because the risk surface is smaller.

### The One Thing to Do First
Before touching source priority: audit `TOURNAMENT_START_DATES`. Confirm the current year's entry is correct. Add a runtime check that raises `MissingTournamentDateError` if the pipeline year has no entry. Write one test confirming a missing entry is caught before any data fetch. That converts the silent failure mode into a loud one.

---

*Council convened April 6, 2026 | 5 advisors, 5 peer reviews, chairman synthesis*
