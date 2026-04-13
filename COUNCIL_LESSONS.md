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
  `[open]`
- **Calibration on 60–70 holdout games is fatal.** Temperature scaling on that
  sample produces "noise with decimal places." Vegas lines proposed as a
  better-calibrated external source.

### Pool strategy (post-pivot, locked 2026-04-02)
- **Stop optimizing prediction accuracy. Optimize bracket-pool EV against
  opponents.** Unanimous 5-advisor agreement. `[locked; MEMORY.md §1]`
- **Optimize P(1st), not EV.** EV ranking failed on 2026: the system produced
  a 1440-pt winning bracket (4/4 F4, det_champ96) but ranked it #11 in its
  own portfolio. Ranking failure, not prediction failure.
- **Recommended mode: `champ_first_tv`.** Aggressive alt: `e8_first_tv`.
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
  noise. `[governing open item]`
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
  `[open]`
- **Silent failures are worse than crashes.** "Won't crash" ≠ "works".
  Team-name matching bug (snake_case double-underscores → silent fallback to
  seed rates) was load-bearing before it was caught. Runtime guards beat
  documentation.
- **Stacking contamination was the validity crisis before the pivot.** Meta-
  learner trained on the LOYO predictions used to evaluate it. The pivot made
  it moot (stacking disabled) but the governance lesson stands: no
  revoke-and-relock protocol existed.
- **Researcher degrees-of-freedom leakage** (hyperparameters tuned while
  observing backtest on same years) was raised but never audited. `[open]`
- **Define kill-criteria BEFORE running tests.** Without a pre-committed
  threshold you rationalize whatever number comes back.
- **Held-out last-3-years validation is recommended but not enforced.**
  Multiple sessions asked for it; no gating mechanism exists. `[open]`

### Data pipeline / infrastructure
- **Trank.php promoted to primary data source over cbbdata.com** (decided
  2026-04-06). Rationale: server-side date filtering is a second independent
  leakage guard. Chooses loudly-failing Cloudflare fragility over silently-
  corrupting hardcoded `TOURNAMENT_START_DATES` dict.
- **`TOURNAMENT_START_DATES` dict is a single point of failure** — hardcoded,
  requires yearly update, no test catches a missing entry. `[open]`
- **`_validate_pretournament()` runtime guard added to
  `noseed_model._load_team_stats()`** with 10 regression tests (2026-04-05).
  Still need to trace the full call graph to confirm coverage on all load
  sites. `[mostly closed]`
- **No request logging / response hashing / artifact provenance tracking.**
  Cannot audit which data source was used for a given model run. `[open]`
- **Pre-2011 vs post-2011 tournaments aren't interchangeable.** Field expanded
  64 → 68, analytics culture shifted. Naive concatenation of 19 yrs of picks
  ignores non-stationarity. `[open — how to weight/exclude]`

### Diversity / optimizer
- **Diversity collapse is underdiagnosed.** Pareto frontier was producing 8–14
  unique brackets; stochastic-tilt mode (which should help) made mean rank
  *worse* (620 vs 532). Root cause never traced. `[open]`
- **Leveraged mode regression** (adding variance hurt performance) falsified
  the naive "more diversity = better" hypothesis. Diagnostic gold, never
  followed up.
- **Mean rank vs P(top 5 %) tell different stories.** Pools pay out top
  finishers, not averages. P(top 5 %) 8.3 % vs 5.1 % (a 63 % lift) was real
  signal; mean rank 532 wasn't.

---

## 2. Open Questions / Unresolved for 2027

Priority order. Blocking items prevent shipping other improvements; the rest
are validation and diagnostic gaps.

### Blocking (do first)
1. **Collect all 31 brackets from the actual 2026 pool**, urgently — data
   stales as pools archive picks. Gate: structured dataset of 31 complete
   brackets. Four-advisor consensus 2026-04-12c: the prerequisite to every
   other opponent-model fix. Without this, opponent correlation fixes are
   theorizing.
2. **Validate the eight local four-factor features against Torvik.** Gate:
   r ≥ 0.99 per-season per-feature, no systematic residual bias by team
   type/conference/tempo. Prerequisite: confirm Torvik docs say whether
   published values are raw or opponent-adjusted (determines the target).
   Until this is done, backtest and live pipeline use different feature
   distributions.
3. **Rank-correlation diagnostic on base model.** Spearman ρ between
   predicted P(1st) rank and actual historical pool placement for top ~10
   brackets. If ρ ≈ 0, the entire optimizer is directionally unreliable and
   the 2026 #11 ranking failure cannot be attributed to either opponent
   model or base model.

### High-priority validation gaps
4. **Measure empirical opponent correlation from the 31 real brackets** once
   collected. The independence assumption is a "fundamental validity threat
   to the entire 13-year backtest" (council 20260412c).
5. **MC sim count for stable rankings.** At 500 sims top-5 ranking is
   unstable between runs (~1.4 % SE); at 5000 drops to ~0.4 %. Gate: run
   optimizer 3× with identical inputs, verify top-20 rank-order identical.
6. **Calibration check** — does simulated P(1st) rank match actual placement
   historically? No one has verified whether brackets ranked highest actually
   won more often.
7. **Tournament-only vs blended model head-to-head.** Does 500–600-game
   tournament-only model outperform 2200-sample blended+stacked? Flagged
   2026-04-01; never run.
8. **Feature collinearity with seed.** If any of the 7 production features
   has |r| > 0.7 with seed number, BSS ≈ 0 is tautological. Quick
   correlation matrix would answer it.
9. **Round-specific decomposition of "seeds beat everything".** Does the
   ceiling hold uniformly across R64→NCG or break down in E8/F4? Raised
   2026-04-02-220323, never run.

### Design questions still open
10. **Opponent correlation: empirical parameterization from one pool-year,
    or theoretical copula?** 31 brackets from one year measures correlation
    structure but may not generalize across 13 backtest years.
11. **Is BSS ≈ 0 fatal to game-theory optimization?** Chairman position
    (council 20260412c): no — seed baseline is not zero-information;
    optimizer exploits pool structure, not prediction alpha. Philosophically
    settled in transcript; technically unsettled in codebase.
12. **Scoring function structure.** Championship-round points may dominate;
    optimal strategy depends on point structure, not just game outcomes.
    Never modeled explicitly.
13. **Winner-take-all pools may want variance, not expected P(1st).** Kelly-
    optimal risk posture differs from argmax-P(1st). Flagged, unresolved.
14. **Calibration-to-pool interaction.** Slightly miscalibrated model that
    over-estimates upsets could be a feature, not a bug, in a winner-take-all
    pool. Never measured.
15. **Vegas-line calibration vs in-house temperature scaling.** Market is
    better-calibrated and has sufficient samples. Never evaluated as a
    drop-in replacement.
16. **`TOURNAMENT_START_DATES` hardcoded dict** — who owns yearly update?
    Needs test that fails loudly when a year is missing.
17. **Researcher-DoF leakage audit.** Were hyperparameters tuned while the
    researcher could observe backtest results on the same years? Unanswered.
18. **Approximate-Torvik systematic bias on mid-majors and tempo profiles.**
    Aggregate r ≈ 0.95 hides per-subgroup divergence exactly where tournament
    upsets cluster.
19. **Pre-2011 ESPN picks data** — exclude, down-weight, or treat as
    equivalent? Tournament expansion 64 → 68, analytics culture pre-/post-
    smartphone all matter.
20. **Held-out-last-3-years enforcement mechanism.** Nothing gates tuning
    decisions on this today.

---

## 3. Session Index (chronological)

Terse index so you can grep for "what did the council say about X".
Format: `date — question → verdict`. Follow-up or critical items linked by
number to §2.

| # | Date | Question | Verdict |
|---|------|----------|---------|
| 1 | 2026-03-31 | Biggest repo limitation? | Stacking weight contamination; no revoke-and-relock protocol. *(Moot post-pivot.)* |
| 2 | 2026-04-01 05:25 | Most critical limitation? | Training-data starvation; no baseline comparison against seed-only. |
| 3 | 2026-04-01 05:58 | Train on reg-season + tournament? | No — distribution mismatch, not sample-size fix. Don't add NIT. Tournament-only diagnostic flagged *(open → §2 item 7)*. |
| 4 | 2026-04-01 21:21 | Single most critical limitation? | Calibration on 60–70 games is fatal; use Vegas lines. 7-feature logistic is the ceiling. |
| 5 | 2026-04-02 05:20 | First priority after backtest harness? | Wire seed-only baseline into LOYO eval loop. If pipeline doesn't beat seed @ p<0.05, most of 122 K LOC is noise. |
| 6 | 2026-04-02 20:52 | Biggest limitations? | BSS ≈ 0 vs seed baseline. Feature collinearity with seed possibly tautological *(open → §2 item 8)*. |
| 7 | **2026-04-02 22:03** | Where to focus next? | **STRATEGIC PIVOT: stop optimizing accuracy; optimize pool EV.** 5-advisor unanimous. *(MEMORY.md §1)* |
| 8 | 2026-04-03 v2 | Next step for pool optimizer? | Build MC simulation to measure P(rank=1). Brier ≠ pool performance. |
| 9 | 2026-04-03 v3 | Skip per-round Brier gate check? | Yes, unanimous — Brier doesn't proxy P(rank=1). Build MC directly. |
| 10 | 2026-04-03 v4 | MC results: noseed = seed at P(1st) ≈ 0.054. What now? | Fix backtest (argmax → stochastic), then validate opponent model. |
| 11 | 2026-04-03 v5 | Stochastic backtest near-random. Why? | Opponent model broken (stale SEED_PICK_RATES proxy). Swap in real ESPN aggregate picks. |
| 12 | 2026-04-04 | 19-yr ESPN data ready? | No — team-name matching bug (silent fallback). Build diagnostic, ≥ 80 % match rate gate before backtest. Pre-2011 ≠ post-2011 *(open → §2 item 19)*. |
| 13 | 2026-04-04 v2 | Can we run backtests now? | No, unanimous — fix team-name matching first. Literal lookup table for ~120 tournament teams. Silent fallback = flying blind. |
| 14 | 2026-04-04 v3 | Post-fix: next step? | Re-run stochastic backtest; pre-commit kill criterion (e.g. median rank < 400 / 1000). |
| 15 | 2026-04-04 v4 | Why are backtests random? | Optimizer bypassed in backtest harness; inject chalk-bias signal directly if leverage ≈ 0. |
| 16 | 2026-04-04 v5 | Is opt_mode `mean_rank=396` real? | No — selection bias (5 optimized vs 50 seed brackets). Fix parity. |
| 17 | 2026-04-04 v6 | Path A/B/C? | Path C, but diagnose leveraged-mode regression first *(open → §2 diversity collapse)*. P(top 5%) is the real signal, not mean rank. |
| 18 | 2026-04-04 v7 | opt_torvik p=0.027 significant? | No after Bonferroni (6 modes, threshold p<0.0083). Confirm Torvik provenance before anything else. |
| 19 | 2026-04-04 v8 | Leakage / provenance audit? | Add `_validate_pretournament()` runtime guard. Don't trust aggregate barthag r² ≈ 0.95 — check per-subgroup *(open → §2 item 18)*. Researcher-DoF leakage flagged *(open → §2 item 17)*. |
| 20 | 2026-04-05 v9 | Most critical unresolved gap? | Runtime guard + 10 regression tests landed. Guard must live at data-loading layer, not caller-dependent. |
| 21 | 2026-04-06 v10 | Local four factors valid for backtest? | No — validate against Torvik, gate r ≥ 0.99 *(open → §2 item 2)*. Barthag r=0.73 is the precedent. |
| 22 | 2026-04-06 03:07 | trank.php vs cbbdata.com as primary? | **Promote trank.php** for server-side date filtering (redundant leakage guard). `TOURNAMENT_START_DATES` is SPOF *(open → §2 item 16)*. |
| 23 | 2026-04-12 | Critical steps for 2027? | Optimize P(1st) not EV; enforce F4 historical base rates; validate opponent model. 2026 system ranked its own winning bracket #11. |
| 24 | 2026-04-12 b | Next critical steps? | Fix opponent correlation (1), bump MC sims 500 → 5000 (5). Do NOT tune F4 cap 2 → 1.5 (overfitting). |
| 25 | **2026-04-12 c** | Single most critical action for 2027? | **Collect all 31 brackets from actual 2026 pool this week.** Prerequisite to every opponent-model fix. Independence assumption is a fundamental validity threat. *(MEMORY.md §3 governing; §2 item 1)* |

---

## Update rule

When a new council session produces a settled verdict, append a row to §3
with date + one-line verdict. If it also closes or supersedes an item in §2,
move that item to §1 (lessons) or mark `[closed <date>]`. Do not edit old
entries — append, don't rewrite. Same norms as MEMORY.md.
