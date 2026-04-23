# COUNCIL_LESSONS.md

Consolidated digest of 25 LLM-council sessions run 2026-03-31 → 2026-04-12.
Raw transcripts and HTML reports were deleted 2026-04-13 after the 2026
tournament; this file is the surviving record. New sessions will append below.

**Source-of-truth precedence** (per MEMORY.md): `pipeline_freeze.json` > this
file > audit `.md` files > code comments. Where a settled decision from these
sessions is indexed in MEMORY.md, MEMORY wins.

---

## 1. Lessons Learned (cross-cutting)

Patterns that recurred across multiple sessions. Keep reading these before
proposing new work — most "new ideas" have been tried or ruled out.

### Modeling
- **BSS ≈ 0 vs seed baseline is ground truth, not a bug.** 17 yrs × every
  tested model (logistic, LightGBM, XGBoost, 27-feature ensemble, stacking,
  GNN, transformer) all BSS ≈ 0. Seeds already encode the signal available in
  tournament-history features. `[locked; MEMORY.md §2 D1–D4, D11]`
- **7-feature logistic is the ceiling given data constraints.** GNN,
  transformer, ensemble stacking, learned feature-selection are all
  overambitious for 60-ish holdout games. `[locked; MEMORY.md §1]`
- **Training on regular season + tournament is a distribution mismatch, not a
  sample-size fix.** Adding NIT / conference tournaments was considered and
  rejected: structurally similar ≠ statistically similar; 2000× more
  irrelevant samples bury tournament signal.
- **Tournament-only model vs blended stacked model** was never directly
  measured head-to-head. Flagged as the diagnostic experiment but not run.
  `[open → §2 O7]`
- **Calibration on 60–70 holdout games is fatal.** Temperature scaling on that
  sample produces "noise with decimal places." Vegas lines proposed as a
  better-calibrated external source.

### Pool strategy (post-pivot, locked 2026-04-02)
- **Stop optimizing prediction accuracy. Optimize bracket-pool EV against
  opponents.** Unanimous 5-advisor agreement. `[locked; MEMORY.md §1]`
- **Optimize P(1st), not EV.** EV ranking failed on 2026: the system produced
  a 1440-pt winning bracket (4/4 F4, det_champ96) but ranked it #11 in its
  own portfolio. Ranking failure, not prediction failure.
- **Recommended mode: `f4_first_tv`.** Aggressive alt: `e8_first_tv`.
  Prior rec was `champ_first_tv`; superseded by O26/O27 team-identity scoring.
  `[locked; MEMORY.md §1]`
- **Chalk bias is measured and exploitable** (F4: public picks 1-seeds 42%
  vs true ~21%). Multiple sessions flagged this as directly actionable
  without better ML; no implementation yet. `[partially actioned]`
- **Stochastic bracket generation is non-negotiable.** Argmax collapses
  probabilistic models into crowd-following; the contrarian signal lives in
  the probabilities, not the modal bracket.
- **Opponent model is the load-bearing wall.** Three separate 2026-04-12
  sessions independently converged: fix opponent model before anything else.
  If independence assumption is wrong, the whole 13-yr backtest measures
  noise. `[governing open item → §2 O1, O4, O10]`
- **Opponent weights 60 % ESPN picks / 30 % Massey / 10 % seed fallback.**
  `[locked 2026-04-12; MEMORY.md §1]`
- **F4 cap at 2, not 1.5.** Tuning the cap from a 14-yr historical average is
  small-sample overfitting.

### Validation / statistics
- **14–17-year backtests are underpowered (~9–16 % power).** Use them for
  binary/directional claims (e.g. "P(1st) beats EV") — do NOT use them to
  tune parameter values (F4 cap, MC sim count, blend weights).
- **Validate before shipping. Formula-correctness is not validation.** The
  barthag r = 0.73 divergence (reverse-engineered from ESPN box scores vs.
  real Torvik) is the precedent to remember. Local four factors still need
  the same validation: gate ≥ 0.99 per-season, no systematic residual bias.
  `[open → §2 O2]`
- **Silent failures are worse than crashes.** "Won't crash" ≠ "works".
  Team-name matching bug (snake_case double-underscores → silent fallback to
  seed rates) was load-bearing before it was caught. Runtime guards beat
  documentation.
- **Stacking contamination was the validity crisis before the pivot.** Meta-
  learner trained on the LOYO predictions used to evaluate it. The pivot made
  it moot (stacking disabled) but the governance lesson stands: no
  revoke-and-relock protocol existed.
- **Researcher degrees-of-freedom leakage** (hyperparameters tuned while
  observing backtest on same years) was raised but never audited.
  `[open → §2 O17]`
- **Define kill-criteria BEFORE running tests.** Without a pre-committed
  threshold you rationalize whatever number comes back.
- **Held-out last-3-years validation is recommended but not enforced.**
  Multiple sessions asked for it; no gating mechanism exists.
  `[open → §2 O20]`

### Data pipeline / infrastructure
- **Trank.php promoted to primary data source over cbbdata.com** (decided
  2026-04-06). Rationale: server-side date filtering is a second independent
  leakage guard. Chooses loudly-failing Cloudflare fragility over silently-
  corrupting hardcoded `TOURNAMENT_START_DATES` dict.
- **`TOURNAMENT_START_DATES` dict is a single point of failure** — hardcoded,
  requires yearly update, no test catches a missing entry. `[open → §2 O16]`
- **`_validate_pretournament()` runtime guard added to
  `noseed_model._load_team_stats()`** with 10 regression tests (2026-04-05).
  Still need to trace the full call graph to confirm coverage on all load
  sites. `[mostly closed]`
- **No request logging / response hashing / artifact provenance tracking.**
  Cannot audit which data source was used for a given model run. `[open]`
- **Pre-2011 vs post-2011 tournaments aren't interchangeable.** Field expanded
  64 → 68, analytics culture shifted. Naive concatenation of 19 yrs of picks
  ignores non-stationarity. `[open → §2 O19]`

### Diversity / optimizer
- **Diversity collapse is underdiagnosed.** Pareto frontier was producing 8–14
  unique brackets; stochastic-tilt mode (which should help) made mean rank
  *worse* (620 vs 532). Root cause never traced. `[open; see §3 row 17]`
- **Leveraged mode regression** (adding variance hurt performance) falsified
  the naive "more diversity = better" hypothesis. Diagnostic gold, never
  followed up.
- **Mean rank vs P(top 5 %) tell different stories.** Pools pay out top
  finishers, not averages. P(top 5 %) 8.3 % vs 5.1 % (a 63 % lift) was real
  signal; mean rank 532 wasn't.

---

## 2. Open Questions / Unresolved for 2027

Stable-ID table so the council summoner (§4) can detect duplicates and check
current gaps before spawning another session. IDs are permanent — do not
renumber. When an item closes, move it to §1 and leave a `[closed <date>]`
crumb here.

**Status key:** `open` = nothing done • `partial` = scoped or started •
`mostly closed` = shipped but needs confirmation • `blocked:<Ox>` = waiting
on another item.

### Blocking — do first (prerequisites for 2027 predictions)

| ID | Item | Gate (how we know it's done) | Raised in §3 | Status |
|---|------|------------------------------|---|---|
| **O1** | Collect all 31 brackets from the actual 2026 pool | Structured dataset of 31 complete brackets (all 63 picks each), saved to repo. Urgent — pool sites archive. | 25 | **open, urgent** |
| **O2** | Validate eight local four-factor features against Torvik | Per-season, per-feature r ≥ 0.99, no systematic residual bias by team type / conference / tempo. Prereq: confirm Torvik docs state whether values are raw or opponent-adjusted. | 21 | open |
| **O3** | Rank-correlation diagnostic on base model | Spearman ρ between predicted-P(1st) rank and actual historical pool placement, top ~10 brackets. If ρ ≈ 0, 2026 #11-ranking failure cannot be attributed between opponent model and base model. | 23, 24 | open |

### High-priority validation gaps

| ID | Item | Gate | Raised in §3 | Status |
|---|------|------|---|---|
| **O4** | Empirical opponent correlation from 31 real brackets | Measured correlation matrix across games; compared against independence assumption. | 25 | `blocked:O1` |
| **O5** | MC sim count for stable rankings | Run optimizer 3× with identical inputs, verify top-20 rank-order identical. (Predicted jump 500 → 5000 sims.) | 24 | open |
| **O6** | Calibration check: simulated P(1st) vs actual placement | Historical verification that brackets ranked highest by optimizer actually won more often. | 23, 24 | open |
| **O7** | Tournament-only vs blended+stacked head-to-head | LOYO BSS comparison; tournament-only (500–600 games) vs blended (2200). Flagged 04-01; never run. | 3 | open |
| **O8** | Feature collinearity with seed | Correlation matrix of 7 production features vs seed number. If any |r| > 0.7, BSS ≈ 0 is tautological. | 6 | open |
| **O9** | Round-specific decomposition of "seeds beat everything" | Per-round BSS breakdown (R64 → NCG). Does the ceiling hold uniformly or break in E8/F4? | 7 | open |

### Design questions still open

| ID | Item | Gate | Raised in §3 | Status |
|---|------|------|---|---|
| **O10** | Opponent correlation: empirical from 1 pool-year vs theoretical copula | Decision + implementation. 31 brackets measures structure but generalization across 13 backtest years is unclear. | 25 | `blocked:O4` |
| **O11** | Is BSS ≈ 0 fatal to game-theory optimization? | Philosophical position (council 25): no — seed baseline is not zero-information. Codebase reflects this assumption but has no explicit invariant/test asserting it. | 25 | partial (settled in transcript, not code) |
| **O12** | Scoring-function structure modeled explicitly | Pool's point schedule (e.g. 1/2/4/8/16/32) hard-coded into optimizer's objective, not just game outcomes. | 23 | open |
| **O13** | Winner-take-all: maximize variance not E[P(1st)] | Kelly-optimal risk posture evaluated against argmax-P(1st). | 24 | open |
| **O14** | Calibration-to-pool interaction | Measure whether slightly upset-over-estimating calibration improves pool rank vs true calibration. | 23 | open |
| **O15** | Vegas-line calibration vs in-house temperature scaling | Drop-in comparison on BSS + pool backtest metrics. | 4 | open |
| **O16** | `TOURNAMENT_START_DATES` hardcoded-dict SPOF | Test fails loudly when a year is missing; owner assigned for yearly update. | 22 | open |
| **O17** | Researcher-DoF leakage audit | Documented trail of which years were observable when which hyperparameters were tuned. | 19 | open |
| **O18** | Approximate-Torvik systematic bias on mid-majors / tempo profiles | Per-subgroup residual analysis (not just aggregate r² ≈ 0.95). | 19 | `blocked:O2` (subset) |
| **O19** | Pre-2011 ESPN picks data — exclude, down-weight, or equivalent | Decision + implementation. Field expanded 64 → 68 in 2011; analytics-culture shift. | 12 | open |
| **O20** | Held-out-last-3-years enforcement mechanism | Code-level gate that refuses to fit on the held-out window. | 14 | open |

---

## 3. Session Index (chronological)

Terse index so you can grep for "what did the council say about X".
Format: `date — question → verdict`. Follow-up or critical items linked by
number to §2.

| # | Date | Question | Verdict |
|---|------|----------|---------|
| 1 | 2026-03-31 | Biggest repo limitation? | Stacking weight contamination; no revoke-and-relock protocol. *(Moot post-pivot.)* |
| 2 | 2026-04-01 05:25 | Most critical limitation? | Training-data starvation; no baseline comparison against seed-only. |
| 3 | 2026-04-01 05:58 | Train on reg-season + tournament? | No — distribution mismatch, not sample-size fix. Don't add NIT. Tournament-only diagnostic flagged *(open → §2 O7)*. |
| 4 | 2026-04-01 21:21 | Single most critical limitation? | Calibration on 60–70 games is fatal; use Vegas lines. 7-feature logistic is the ceiling. |
| 5 | 2026-04-02 05:20 | First priority after backtest harness? | Wire seed-only baseline into LOYO eval loop. If pipeline doesn't beat seed @ p<0.05, most of 122 K LOC is noise. |
| 6 | 2026-04-02 20:52 | Biggest limitations? | BSS ≈ 0 vs seed baseline. Feature collinearity with seed possibly tautological *(open → §2 O8)*. |
| 7 | **2026-04-02 22:03** | Where to focus next? | **STRATEGIC PIVOT: stop optimizing accuracy; optimize pool EV.** 5-advisor unanimous. *(MEMORY.md §1)* |
| 8 | 2026-04-03 v2 | Next step for pool optimizer? | Build MC simulation to measure P(rank=1). Brier ≠ pool performance. |
| 9 | 2026-04-03 v3 | Skip per-round Brier gate check? | Yes, unanimous — Brier doesn't proxy P(rank=1). Build MC directly. |
| 10 | 2026-04-03 v4 | MC results: noseed = seed at P(1st) ≈ 0.054. What now? | Fix backtest (argmax → stochastic), then validate opponent model. |
| 11 | 2026-04-03 v5 | Stochastic backtest near-random. Why? | Opponent model broken (stale SEED_PICK_RATES proxy). Swap in real ESPN aggregate picks. |
| 12 | 2026-04-04 | 19-yr ESPN data ready? | No — team-name matching bug (silent fallback). Build diagnostic, ≥ 80 % match rate gate before backtest. Pre-2011 ≠ post-2011 *(open → §2 O19)*. |
| 13 | 2026-04-04 v2 | Can we run backtests now? | No, unanimous — fix team-name matching first. Literal lookup table for ~120 tournament teams. Silent fallback = flying blind. |
| 14 | 2026-04-04 v3 | Post-fix: next step? | Re-run stochastic backtest; pre-commit kill criterion (e.g. median rank < 400 / 1000). |
| 15 | 2026-04-04 v4 | Why are backtests random? | Optimizer bypassed in backtest harness; inject chalk-bias signal directly if leverage ≈ 0. |
| 16 | 2026-04-04 v5 | Is opt_mode `mean_rank=396` real? | No — selection bias (5 optimized vs 50 seed brackets). Fix parity. |
| 17 | 2026-04-04 v6 | Path A/B/C? | Path C, but diagnose leveraged-mode regression first *(open — see §1 Diversity)*. P(top 5%) is the real signal, not mean rank. |
| 18 | 2026-04-04 v7 | opt_torvik p=0.027 significant? | No after Bonferroni (6 modes, threshold p<0.0083). Confirm Torvik provenance before anything else. |
| 19 | 2026-04-04 v8 | Leakage / provenance audit? | Add `_validate_pretournament()` runtime guard. Don't trust aggregate barthag r² ≈ 0.95 — check per-subgroup *(open → §2 O18)*. Researcher-DoF leakage flagged *(open → §2 O17)*. |
| 20 | 2026-04-05 v9 | Most critical unresolved gap? | Runtime guard + 10 regression tests landed. Guard must live at data-loading layer, not caller-dependent. |
| 21 | 2026-04-06 v10 | Local four factors valid for backtest? | No — validate against Torvik, gate r ≥ 0.99 *(open → §2 O2)*. Barthag r=0.73 is the precedent. |
| 22 | 2026-04-06 03:07 | trank.php vs cbbdata.com as primary? | **Promote trank.php** for server-side date filtering (redundant leakage guard). `TOURNAMENT_START_DATES` is SPOF *(open → §2 O16)*. |
| 23 | 2026-04-12 | Critical steps for 2027? | Optimize P(1st) not EV; enforce F4 historical base rates; validate opponent model. 2026 system ranked its own winning bracket #11. |
| 24 | 2026-04-12 b | Next critical steps? | Fix opponent correlation (1), bump MC sims 500 → 5000 (5). Do NOT tune F4 cap 2 → 1.5 (overfitting). |
| 25 | **2026-04-12 c** | Single most critical action for 2027? | **Collect all 31 brackets from actual 2026 pool this week.** Prerequisite to every opponent-model fix. Independence assumption is a fundamental validity threat. *(MEMORY.md §3 governing; §2 O1)* |

---

## 4. Pre-Council Duplicate-Check Protocol

**Contract for the `llm-council` skill:** before spawning 5 advisor sub-
agents, the skill MUST run the checks below. Most aborts are cheap
("already answered"); one class is genuinely useful ("identified but
unaddressed — do you want to execute or re-debate?").

### Step 1 — Load this file's §1, §2, §3
Read the whole file. It is < 500 lines; cost is negligible vs. spawning 11
sub-agents.

### Step 2 — Classify the user's question
Map it to one of these five buckets:

| Bucket | How to detect | Action |
|---|---|---|
| **A. Already answered (§3 duplicate)** | Question semantically matches a row in §3 (e.g. "what's the biggest repo limitation?" was asked 6× — rows 1, 2, 4, 5, 6, 20). | **Do not spawn advisors.** Tell the user: "Council answered this on `<date>`: `<verdict>`. Re-run only if you have new evidence that invalidates the prior verdict." Ask one clarifying question: new evidence, or different framing? |
| **B. Already-identified open item (§2 hit)** | Question matches an `Ox` item the council has already named — user is asking "what should we do about X" where X is in §2. | **Do not spawn advisors.** Surface the specific row: "§2 O`<n>` already identifies this. Gate: `<gate>`. Status: `<status>`. The work is execution, not debate." Ask: do you want help executing, or is the gate itself wrong? |
| **C. Blocked-on prerequisite** | Question touches an `Ox` that has `blocked:O<m>` status. | **Do not spawn advisors.** Tell the user: "O`<n>` is blocked on O`<m>`. Current status of O`<m>`: `<status>`. Unblock that first." |
| **D. Locked decision (MEMORY.md §1)** | Question proposes changing something locked in MEMORY.md §1 ("should we switch from `champ_first_tv`?", "enable stacking?"). | **Do not spawn advisors yet.** Cite the MEMORY row + source. Ask: is there new evidence that invalidates the lock? If yes → proceed to Step 3. If no → stop. |
| **E. Genuinely novel** | No §1 / §2 / §3 / MEMORY row covers the question. | **Proceed to Step 3** (normal council flow). |

### Step 3 — Pre-framing context injection
For bucket E only: when framing the question for the 5 advisors (existing
step 1 of the skill), append a "Prior art" section listing:
- Any §2 `Ox` items that are tangentially related (so advisors don't
  re-raise them as "new").
- Any §1 locked lessons that bound the space (so advisors don't propose
  solutions that contradict locks).
- The 2-3 most-related §3 rows by keyword match.

This prevents advisors from re-deriving settled ground and focuses their
work on the novel part.

### Step 4 — Post-verdict update
If a new session produced a settled verdict:
- Append a row to §3 (new # continues the sequence).
- If it closes an `Ox`: move to §1 and leave `[closed <date>]` in §2.
- If it opens a new unresolved item: add a new `Ox` row with the next free
  ID (never reuse; never renumber).
- If it supersedes a §1 lesson: append a superseding bullet; mark the old
  one `[SUPERSEDED <date>]`. Do not rewrite.

### What NOT to skip the council for
- Bucket E (genuinely novel) → run the council.
- Bucket A with explicit new evidence from the user → run the council.
- Bucket D with explicit new evidence from the user → run the council.

### What TO skip the council for
- Bucket A without new evidence → cite §3 row and stop.
- Bucket B without challenge to the gate → cite §2 O`<n>` and offer to help
  execute.
- Bucket C → cite the blocker and stop.
- Bucket D without new evidence → cite MEMORY row and stop.

---

## Update rule

When a new council session produces a settled verdict, append a row to §3
with date + one-line verdict. If it also closes or supersedes an item in §2,
move that item to §1 (lessons) or mark `[closed <date>]`. Do not edit old
entries — append, don't rewrite. Same norms as MEMORY.md.
