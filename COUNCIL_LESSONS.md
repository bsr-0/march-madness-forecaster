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
- **Recommended mode: `champ_first_tv`.** Aggressive alt: `e8_first_tv`.
  `[locked; MEMORY.md §1]`
- **Chalk bias is measured and exploitable** (F4: public picks 1-seeds 42%
  vs true ~21%). Multiple sessions flagged this as directly actionable
  without better ML; no implementation yet. `[partially actioned]`
- **Stochastic bracket generation is non-negotiable.** Argmax collapses
  probabilistic models into crowd-following; the contrarian signal lives in
  the probabilities, not the modal bracket.
- **Opponent model is NOT the load-bearing wall (reversed 2026-04-13).**
  Three 2026-04-12 sessions flagged it as such. 3 rounds of investigation
  closed the whole branch: O1 (data already existed), O3 (ranking has
  signal, mean ρ = +0.37), O4 (independence holds, pooled z = −4.15),
  O10 (no copula needed), **O21 (pool-history blending does not change
  bracket rankings — null result across 2024-2025)**. The opportunity,
  if any remains, is in the **base model** (game-outcome probabilities),
  not opponent modeling. `[see §2 O1/O3/O4/O10/O21; ANALYSIS_O4 and
  ANALYSIS_O21 for evidence]`
- **Opponent-model marginals affect absolute P(1st) but not bracket
  *ranking*.** Spearman ρ between predicted P(1st) and actual score is
  flat across opponent-model-weight settings w ∈ {0, 0.25, 0.5}, and
  degrades slightly at w ≥ 0.75. Implication: the choice of opponent
  model is irrelevant to "which bracket should I submit?" — it only
  affects "how likely is my bracket to win?" Future opponent-model
  tuning should be framed around calibration, not ranking. `[added
  2026-04-13; ANALYSIS_O21_MARGINAL_BLEND.md]`
- **Check opponent-model marginals before correlations.** The 2026-04-12c
  council diagnosed opponent-model validity as a correlation problem
  without first measuring correlation or marginals. 4-year analysis
  (`ANALYSIS_O4_OPPONENT_CORRELATION.md`) found correlation is fine;
  marginals are off by 5pp (mean absolute) vs ESPN national, up to 18pp
  on individual teams. But see O21: the marginal divergence doesn't
  translate to ranking improvement. Future "opponent model is wrong"
  diagnoses should start with marginals — cheaper and was the actual
  issue, even though it didn't move the needle here. `[added 2026-04-13]`
- **Opponent weights 60 % ESPN picks / 30 % Massey / 10 % seed fallback.**
  `[locked 2026-04-12; MEMORY.md §1]`
- **F4 cap at 2, not 1.5.** Tuning the cap from a 14-yr historical average is
  small-sample overfitting.
- **Pool size N ≠ usable opponent count K.** `N` is the raw pool size
  (e.g. 31 for 2026). The user's own bracket is correctly excluded from
  the opponent model to avoid self-reference, so `K = N − 1` (30 for
  2026). Verified across all pool-history years: 2023 = 18/19,
  2025 = 32/33, 2026 = 30/31 (2024 = 25/25 is the outlier — pre-dated the
  user's entry). This is documented here because the 2026-04-12c council
  flagged "collect 31 brackets" as the top priority without knowing the
  30 already existed on disk, scraped earlier that same day. Future
  "collect real pool brackets" framings should specify `K` explicitly.
  `[added 2026-04-13]`

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
| **O1** | Collect all 31 brackets from the actual 2026 pool | Structured dataset of 31 complete brackets (all 63 picks each), saved to repo. | 25 | **`[closed 2026-04-13]`** — `pool_hist_results.json` has 30 brackets as of 2026-04-12T22:47Z; the 31st is the user's own (correctly excluded to avoid a self-referential opponent model). Pattern verified across 2023 (18/19), 2025 (32/33), 2026 (30/31). See §1 Pool strategy: *pool size N ≠ usable opponent count K*. |
| **O2** | Validate eight local four-factor features against Torvik | Per-season, per-feature r ≥ 0.99, no systematic residual bias by team type / conference / tempo. Prereq: confirm Torvik docs state whether values are raw or opponent-adjusted. | 21 | **`[closed 2026-04-13]`** — resolved via the **barthag precedent from §3 row 22**, not by relaxing the gate. Investigation revealed the council's recommended path was already implemented in production at 5 call sites: `TorVikFFLookup.overlay_metrics` overlays Torvik's trank.php FF values onto local `ProprietaryTeamMetrics`, with local box-score computation as a fallback. The r ≥ 0.99 gate was measuring the wrong artifact (pre-overlay fallback values). Correct gates now locked in `tests/test_validate_four_factors.py`: (1) monthly Torvik FF snapshots present for every production year; (2) pre-tournament snapshot covers all 64-68 tournament teams (currently 68/68 for 2024); (3) local fallback tripwire `mean r ≥ 0.85` to catch catastrophic regressions like the O2a resolver collision. MEMORY.md §1 Validation/training split locks the Torvik-overlay pattern. Prereq "raw vs adjusted" was confirmed by code inspection + empirical match of Oliver formulas. |
| **O2a** | Resolver collision + data-source precision limit in the four-factor validator | Root cause #1 (fuzzy-containment `TeamNameResolver` collapsing `vermont_state_lyndon_hornets` → `vermont` and overwriting real Vermont Catamounts stats) + root cause #2 (provider-level precision limit making r ≥ 0.99 unreachable). | surfaced by O2 fresh validation 2026-04-13; both root causes addressed same day | **`[closed 2026-04-13]`** — root cause #1 fixed in `scripts/validate_four_factors.py` (commit `df8edce`), jumped 4/8 features from mean r ≈ 0.45 to ≈ 0.97. Root cause #2 resolved by applying the council's barthag precedent: Torvik is the production FF source via `TorVikFFLookup.overlay_metrics`; local is a fallback with a mean-r ≥ 0.85 tripwire for catastrophic-bug detection. See `tests/test_validate_four_factors.py` and MEMORY.md §1. |
| **O3** | Rank-correlation diagnostic on base model | Spearman ρ between predicted-P(1st) rank and actual historical pool placement, top ~10 brackets. If ρ ≈ 0, 2026 #11-ranking failure cannot be attributed between opponent model and base model. | 23, 24 | **`[closed 2026-04-13]`** — executed in `artifacts/rank_correlation_diagnostic.json`. Mean Spearman ρ = +0.37 across 14 years, 12/14 positive, median ρ = +0.46. Optimizer's P(1st) ranking has real directional signal. 2023 = −0.64 is the outlier reversal. |
| **O21** | Replace ESPN-national marginals with pool-specific marginals in opponent model | Opponent model rebuilt weighting ESPN-national + pool-history marginals; retrospective pool EV backtested against 2023-2026 to verify switch changes bracket rankings. | surfaced by O4 analysis 2026-04-13 | **`[closed 2026-04-13]`** — **null result.** Backtested 2024-2025 (2023 underpowered, 2026 skipped per O22). Spearman ρ between predicted P(1st) and actual score is **unchanged** at w=0/0.25/0.5 and *slightly decreases* at w≥0.75. Opponent-model marginals affect *absolute* P(1st) but not the *ranking* of pool brackets. Keep locked 60/30/10 ESPN/Massey/seed. See `ANALYSIS_O21_MARGINAL_BLEND.md`. |
| **O22** | Fix malformed `data/raw/historical/tournament_results_2026.json` | 67 games with correct `round_name` taxonomy: 32×R64, 16×R32, 8×S16, 4×E8, 2×F4, 1×NCG, 4×FF. Current file has 49×NCG and 0×R64, blocking any year-over-year analysis using 2026. | surfaced by O21 analysis 2026-04-13 | **open, data-ops** — blocks reuse of 2026 in historical backtests until re-ingested from upstream. |

### High-priority validation gaps

| ID | Item | Gate | Raised in §3 | Status |
|---|------|------|---|---|
| **O4** | Empirical opponent correlation from the 30 real brackets | Measured correlation matrix across games; compared against independence assumption. | 25 | **`[closed 2026-04-13]`** — independence assumption holds. 4-year pooled z = −4.15; brackets are *less* correlated than IID draws from the empirical marginals, not more. See `ANALYSIS_O4_OPPONENT_CORRELATION.md`. Follow-up → O21. |
| **O5** | MC sim count for stable rankings | Run optimizer 3× with identical inputs, verify top-20 rank-order identical. (Predicted jump 500 → 5000 sims.) | 24 | **`[closed 2026-04-13]`** — `n_tournaments=5000` locked in `MEMORY.md §1` Pool-strategy table; class/helper defaults normalized in `src/simulation/pool_competition.py:93,949` and `src/evaluation/evaluation_suite.py:371`; rank stability proven by `tests/test_pool_competition.py::TestRankStability::test_top20_rank_order_identical_across_runs` (3 runs at fixed seed produce identical top-20). |
| **O6** | Calibration check: simulated P(1st) vs actual placement | Historical verification that brackets ranked highest by optimizer actually won more often. | 23, 24 | open |
| **O7** | Tournament-only vs blended+stacked head-to-head | LOYO BSS comparison; tournament-only (500–600 games) vs blended (2200). Flagged 04-01; never run. | 3 | open |
| **O8** | Feature collinearity with seed | Correlation matrix of 7 production features vs seed number. If any |r| > 0.7, BSS ≈ 0 is tautological. | 6 | **`[closed 2026-04-13]`** — **mixed result.** Diagnostic via `scripts/feature_seed_correlation.py::compute_production_feature_correlations()` over 612 team-year rows (2016–2025 ex 2020): max \|r\|=0.77 (`adj_off_eff`), median 0.32, with `diff_adj_tempo` (\|r\|=0.16) and `diff_opp_to_rate` (\|r\|=0.02) essentially independent of seed. Three features are seed proxies, two are independent — features carry signal beyond seed; BSS≈0 is **NOT** tautological (would need \|r\|≥0.85 across all features). Anti-tautology gate locked at `tests/test_feature_seed_collinearity.py::test_no_production_feature_is_total_seed_proxy` (fails if any feature breaches \|r\|=0.85). MEMORY.md §2 D1 updated with this evidence. |
| **O9** | Round-specific decomposition of "seeds beat everything" | Per-round BSS breakdown (R64 → NCG). Does the ceiling hold uniformly or break in E8/F4? | 7 | open |

### Design questions still open

| ID | Item | Gate | Raised in §3 | Status |
|---|------|------|---|---|
| **O10** | Opponent correlation: empirical from 1 pool-year vs theoretical copula | Decision + implementation. 30 brackets measures structure but generalization across 13 backtest years is unclear. | 25 | **`[mostly moot 2026-04-13]`** — independence holds per O4; no copula needed for this pool. Reactivate only if a larger / demographically-different pool shows clustering. |
| **O11** | Is BSS ≈ 0 fatal to game-theory optimization? | Philosophical position (council 25): no — seed baseline is not zero-information. Codebase reflects this assumption but has no explicit invariant/test asserting it. | 25 | partial (settled in transcript, not code) |
| **O12** | Scoring-function structure modeled explicitly | Pool's point schedule (e.g. 1/2/4/8/16/32) hard-coded into optimizer's objective, not just game outcomes. | 23 | open |
| **O13** | Winner-take-all: maximize variance not E[P(1st)] | Kelly-optimal risk posture evaluated against argmax-P(1st). | 24 | open |
| **O14** | Calibration-to-pool interaction | Measure whether slightly upset-over-estimating calibration improves pool rank vs true calibration. | 23 | open |
| **O15** | Vegas-line calibration vs in-house temperature scaling | Drop-in comparison on BSS + pool backtest metrics. | 4 | open |
| **O16** | `TOURNAMENT_START_DATES` hardcoded-dict SPOF | Test fails loudly when a year is missing; owner assigned for yearly update. | 22 | **`[closed 2026-04-13]`** — root cause was actually the silent fallback at `src/data/scrapers/torvik.py:400` (returned `(None, None)` on missing year, causing trank.php to serve unfiltered post-tournament data). Now raises `ValueError` for historical misses. Coverage test at `tests/data_integrity/test_tournament_start_dates.py` enforces `TRAIN_YEARS ∪ diagnostic ∪ holdout ⊆ TOURNAMENT_START_DATES`; owner placeholder added at `src/pipeline/config.py:47`; two stale script duplicates deleted. Follow-up → O16b. |
| **O16b** | Silent skips at `snapshot_integrity.py:70-72` and `test_selection_snapshot.py:66` | Same defect class as O16 but in a check (not a filter): replace silent skip with raise. Separate from O16 because the fix doubles blast radius without improving the gate. | surfaced by O16 fix 2026-04-13 | **`[withdrawn 2026-04-13]`** — re-read on 2026-04-13 confirmed both flagged call sites are actually correct. `snapshot_integrity.py:65-79` already raises `ValueError` at line 76 when a year is absent from both `SELECTION_SUNDAY_DATES` and `TOURNAMENT_START_DATES`; the `if year in` at line 71 is a nested fast-path, not a silent skip. `test_selection_snapshot.py:64-74` iterates over `SELECTION_SUNDAY_DATES` and conditionally asserts consistency *only for years also in* `TOURNAMENT_START_DATES` — a scope filter, not a leakage skip. No defect. Item retracted. |
| **O17** | Researcher-DoF leakage audit | Documented trail of which years were observable when which hyperparameters were tuned. | 19 | open |
| **O18** | Approximate-Torvik systematic bias on mid-majors / tempo profiles | Per-subgroup residual analysis (not just aggregate r² ≈ 0.95). | 19 | `blocked:O2` (subset) |
| **O19** | Pre-2011 ESPN picks data — exclude, down-weight, or equivalent | Decision + implementation. Field expanded 64 → 68 in 2011; analytics-culture shift. | 12 | open |
| **O20** | Held-out-last-3-years enforcement mechanism | Code-level gate that refuses to fit on the held-out window. | 14 | **`[closed 2026-04-13]`** — gate exists as a three-layer composition (verified end-to-end at `tests/test_holdout_enforcement.py`): (1) `validate_locked_production_path()` at `src/pipeline/config.py:901` hard-fails configs whose `holdout_years` drift from `[2025]` or whose `dev_years` overlap with holdout; (2) `_filter_years(include_holdout=False)` at `src/pipeline/pipeline_runner.py:496` strips holdout entries from any year list passed to training callers; (3) `YearSplitPolicy` at `src/ml/evaluation/evaluation_integrity.py:75` is a frozen dataclass that validates no overlap at construction and exposes `assert_dev_only` / `assert_dev_artifact_years` raising `HoldoutContaminationError` — wired at the three ensemble fit entry points + every Optuna tuner call. Note: council §3 row 14 phrased the gate as "last-3-years", but `MEMORY.md §1` locks holdout to the single year [2025]; the gate enforces the locked semantic. Locked in `MEMORY.md §1` Validation/training split table. |

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
| 26 | 2026-04-13 | Close O16: TOURNAMENT_START_DATES SPOF? | **Closed.** Real defect was the silent fallback at `torvik.py:400` (returned `(None, None)` and let trank.php serve unfiltered post-tournament data). Now raises `ValueError`. Coverage test enforces `TRAIN_YEARS ⊆ dict`. Two stale script duplicates removed. Spawned **O16b** for the same defect class in `snapshot_integrity.py`. *(§2 O16 closed; §2 O16b open)* |
| 27 | 2026-04-13 | Close O5: bump MC sims for stable rankings? | **Closed.** `n_tournaments=5000` locked in `MEMORY.md §1` Pool-strategy table. Class/helper/eval-suite defaults normalized (were drifting at 1000 even though the three CLI call sites had bumped to 5000 on 2026-04-12). Rank-stability test (`tests/test_pool_competition.py::TestRankStability`) proves 3 runs at fixed seed produce identical top-20 over 20 brackets. *(§2 O5 closed)* |
| 28 | 2026-04-13 | Close O8: are production features just seed proxies? | **Closed — mixed result, BSS≈0 NOT tautological.** `scripts/feature_seed_correlation.py::compute_production_feature_correlations()` over 612 team-year rows: max \|r\|=0.77 (`adj_off_eff`), median 0.32; `diff_adj_tempo` (\|r\|=0.16) and `diff_opp_to_rate` (\|r\|=0.02) carry information independent of seed. Anti-tautology gate (`tests/test_feature_seed_collinearity.py`) fails if any feature ever crosses \|r\|=0.85; MEMORY.md §2 D1 updated with the evidence. *(§2 O8 closed)* |
| 29 | 2026-04-13 | Close O20: code-level holdout-window enforcement? | **Closed.** Gate is a three-layer composition: (1) `validate_locked_production_path()` config gate, (2) `_filter_years(include_holdout=False)` data filter, (3) frozen `YearSplitPolicy` raising `HoldoutContaminationError`. Wired at all three ensemble fit entry points + every Optuna tuner call. End-to-end closure test at `tests/test_holdout_enforcement.py`; locked in `MEMORY.md §1` Validation/training split. The "last-3-years" phrasing from council row 14 is honored as MEMORY's locked single-year holdout `[2025]`. *(§2 O20 closed)* |
| 30 | 2026-04-13 | Withdraw spawned O16b? | **Withdrawn — analysis error.** Re-read of `src/evaluation/snapshot_integrity.py:65-79` and `tests/evaluation/test_selection_snapshot.py:64-74` confirmed both call sites are correct. The function raises `ValueError` at line 76 for missing years; the test's `if year in TOURNAMENT_START_DATES` is a scope filter for a dual-membership consistency check, not a silent leakage skip. No defect. *(§2 O16b withdrawn)* |
| 31 | 2026-04-13 | Close O2: Four-Factor validation? | **NOT CLOSED — escalated to a council decision.** Two independent defects found: **#1 (resolver collision, FIXED):** `compute_boxscore_ff` was letting TeamNameResolver's fuzzy `containment` method collapse `vermont_state_lyndon_hornets` (1 game) onto canonical `vermont`, overwriting real Vermont Catamounts stats. Fix applied. **#2 (data-source precision limit, ESCALATED):** audit across 344 D1 teams in 2024 shows that even with the best-possible formula, local-vs-Torvik Pearson saturates at r ≈ 0.97 — including for eFG% which has no possessions denominator. This is a scorekeeping-level provider difference, not a code defect. Three council options offered in `artifacts/o2a_diagnostic_2026-04-13.md`: (A) relax gate to r ≥ 0.95, (B) switch to a second box-score provider, (C) accept "r ≈ 0.97" as validated. Recommendation: A. *(§2 O2 blocked on council; §2 O2a root-cause-1 closed, root-cause-2 escalated to O2)* |
| 32 | 2026-04-13 | Close O2 via council-precedent review? | **CLOSED — council's own §3 row 22 (barthag) precedent applied.** Re-reading §1 lines 96–98 ("Validate before shipping. Formula-correctness is not validation. The barthag r = 0.73 divergence is the precedent to remember") + §1 line 112 ("Define kill-criteria BEFORE running tests. Without a pre-committed threshold you rationalize whatever number comes back") rejected row-31's Option A (relax gate → rationalize). Instead applied Option D implied by §3 row 22: **promote Torvik as the production FF source**, demote local to fallback. Discovery: this was **already implemented** in production at 5 call sites via `TorVikFFLookup.overlay_metrics`. The r ≥ 0.99 gate had been pointed at the wrong artifact (pre-overlay fallback values that production never uses as-is for covered teams). Correct gates locked in `tests/test_validate_four_factors.py` (19 tests passing): monthly-snapshot coverage per year, tournament-team coverage in pre-tournament snapshot, local-fallback mean-r ≥ 0.85 tripwire. MEMORY.md §1 Validation/training split updated with the locked pattern. *(§2 O2 closed; §2 O2a closed)* |

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
