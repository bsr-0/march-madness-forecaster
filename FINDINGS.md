# Findings & Lessons (Consolidated 2026-08-18)

This file consolidates ~85 historical documentation files (council reports,
audits, session summaries, dead-end logs) into one summary of findings still
relevant to future work. The source files were deleted after this
consolidation — everything is recoverable from git history if you need the
full original text. **Current production state, the north star metric, and
active priorities live in `CLAUDE.md` and the auto-memory system — this file
does not repeat that, it captures the *reasoning and dead ends* behind it.**

---

## 1. The core architectural insight: construction/selection beats prediction accuracy

This is the single most load-bearing conclusion in the project's history, and
it took ~6 weeks of testing to establish.

- **Every model family tested converges to BSS≈0 vs the seed baseline**:
  logistic regression, LightGBM, XGBoost, a 27-feature ensemble + stacking,
  GNN, transformer — across 17 years of LOYO testing. Production is a 7-feature
  logistic regression (`diff_elo_rating`, `diff_total_warp`, `diff_orb_rate`,
  `diff_momentum`, `diff_adj_tempo`, `diff_sos_adj_em`, `diff_opp_to_rate`)
  because the complex ensembles added nothing over it.
- **Champion pick is genuinely unpredictable from pre-tournament data**: the
  highest-barthag 1-seed is the actual champion only 2/14 years; barthag rank
  among 1-seed champions is nearly uniform (#1=3/11, #2=4/11, #3=3/11, #4=1/11).
  Region/exhaustive construction still hit 8% P(1st) despite only 2/14
  champion accuracy — **construction quality drives the win, not champion
  accuracy.**
- **The 14-technique bakeoff (2026-05-01) is the origin of the current
  construction-first architecture.** A critical bug was found and fixed that
  session: `mc_pool_backtest.py` only passed 4 of 10 `CONTEXT_KEYS` at
  inference time while training used all 10 (features 29-34 were real during
  training, zero at inference) — **all meta_gbm numbers measured before this
  fix were distorted**; corrected baseline is 4.6%, not the older 2.86% figure.
- **GBM-predicted probabilities fed into construction hurt P(1st)** —
  `meta_region_gbm`, `meta_exhaustive_gbm`, `meta_exhaustive_margin` all
  performed worse than pure construction on raw torvik probabilities. GBM
  probabilities are less calibrated than raw torvik for construction purposes,
  even though the GBM itself has comparable Brier.
- Most GBM feature additions (multi-seed ensemble, Vegas R1, backward
  elimination) produced bracket-identical results to the base meta_gbm — a
  shallow GBM (depth=3, 50 trees) converges to the same decisions regardless.

## 2. How the selection-problem was diagnosed (history behind meta_region_poolaware)

- **2026 near-miss diagnosis**: the system's own 50-bracket portfolio
  contained a real winning bracket (`det_champ96`, 1440 pts, 4/4 F4) but the
  internal ranker only ranked it **#11**. This was diagnosed as a
  selection/portfolio problem, not a prediction problem, and is the seed of
  the entire construction-first pivot. (Caveat: this bracket was never
  actually submitted to the pool — it's a retroactive diagnostic, not a
  realized loss.)
- **A scoring-loop bug produced a false "ranker always wins" signal**
  (ρ=−1.000 artifact, fixed 2026-04-19): the backtest had been scoring model
  brackets against the *known actual outcome* instead of simulated
  pre-tournament outcomes. After the fix, the ranker's #1 pick beat the actual
  pool winner in only 1 of 12 mode×year combinations (~random) — this
  retracted an earlier, wrong claim that the ranker picks winners reliably.
- **Shape vs team-identity scoring divergence (O26) — a major methodology
  fix.** The pipeline-wide "shape encoding" (positional bool match) used to
  score brackets diverges materially from real ESPN "team-identity" scoring:
  mean ρ_shape=+0.049 vs ρ_team-identity=+0.491 in one spot check, with sign
  inversions in upset-heavy years (2023: −0.25 vs +0.93). This invalidated and
  required re-running several prior closures; most verdicts held directionally
  but magnitudes shifted substantially. **All current production numbers use
  team-identity scoring.**
- **Failed selection-objective alternatives** (useful if anyone proposes these
  again — they've been tried):
  - Empirical-supremum objective (max P(beat best opponent) instead of raw
    P(rank=1)) — failed under both scoring schemes; the existing ranker was
    marginally *better*.
  - Score-threshold replacement for opponent Monte Carlo simulation — failed
    (ρ ≈ 0.16-0.56, far below the 0.95 safety gate); opponent simulation's
    joint-realization structure is load-bearing.
  - Forced champion diversity / leverage-concentration selection — failed (0-1
    pool wins out of 12 tested combos); ~100% of actual pool winners in
    2024/2026 picked the model's natural champion pick, so spreading brackets
    across champions dilutes when the natural pick is already correct.
  - Deterministic argmax construction — strictly dominated by stochastic
    construction on Kelly/log-utility grounds (argmax BestRank ~9.9 vs
    stochastic ~1.5; argmax's aggregate P(1st) edge is a bimodal artifact
    carried by 1-2 chalky years, zero P(1st) most other years).
- **Irreducible selection noise finding (council 71, 2026-04-19)**: 4
  selection criteria tested across 12 mode×year combos; the MC ranker won
  10/12 on correlation, but every criterion produced 0-1 actual pool wins out
  of 12. Mean gap between the ranker's pick and the oracle-best-of-50 was 8.08
  rank positions — didn't trigger the kill-gate (<3) but showed no tested
  method fully converts opponent-model improvements into rank gains. **This
  ceiling measurement is specific to the old stochastic-50-then-select regime
  and does not apply to the current deterministic meta-selector regime** (see
  CLAUDE.md's "Regime Distinction" section — don't cite this number as a bound
  on future deterministic-strategy headroom).
- **The 200-MC-trial-budget insight**: repeated experiments (2026-05-04
  onward) found that adding more candidates to the ~25-candidate poolaware
  pool produces null-or-regression results, because the pool is calibrated
  against a fixed 200-trial inner MC budget — "a candidate the selector never
  picks adds pure noise." This is why `pa_trials` was later raised 200→500 (the
  one accepted change in that family, per CLAUDE.md's git log) rather than
  adding more candidates.

## 3. Consolidated dead-end ledger (things proven not to work — don't re-try without new evidence)

- **Chalk-fade / champion-weight fading**: P(1st) collapsed 14× vs baseline.
  Static 2.00× chalk-multiplier anchor was wrong — true empirical over-pick
  ratio is ~1.26×, not 2.00×. Trades tail upside for average position, wrong
  direction for winner-take-all.
- **Upset detector / upset specialist as a poolaware candidate** (killed
  twice, independently, in different forms): both times selected in only
  ~1/15 years and hurt P(1st) by ~2pp when added. Extra upset-flavored
  candidates add selection noise without producing structurally different
  brackets.
- **Real pool-history opponent model via seed-walk transfer of actual
  2023-2026 brackets**: −5.6pp regression. A LOYO bug leaked 2023-2026 pool
  brackets as opponents into ALL backtest years including 2011-2022; even
  fixed, seed-walk translation onto different years' teams is too lossy.
  **Explicit do-not-retry: ESPN national pick distribution is a better
  opponent model than translated real brackets.**
- **Regime-adaptive candidate generation** (skewing candidate risk-level by a
  validated field-volatility signal, r=-0.668 with F4 seed-mean, p=0.006):
  still failed the P(1st) gate (10.80% vs 11.20%) — the MC selector already
  implicitly adapts; skewing removes options rather than adding value.
- **Pool-specific opponent modeling / behavioral blend** (blend weights
  0.0-0.7 of a cross-year pool model): all degrade P(1st) monotonically. For
  the years with real pool ground truth, no synthetic model beats using the
  actual year-specific brackets directly. (Infrastructure — `--opponent
  pool_calibrated` — was kept, not deleted.)
- **More candidate diversity in general** (more blend ratios, denser risk
  grids, per-region independent construction): consistently null across
  multiple independent attempts (2026-05-18 session tried 3 variants, all
  killed). New candidates were selected 0/15 or near-0/15 years.
- **Deterministic-argmax construction modes** (`det_champ`/`det_f4`/`det_e8`):
  P(1st)=0.00% in backtest — no diversity, single point of failure.
- **Pareto-leverage / EV optimizer** (`opt_*` modes): catastrophic in upset
  years. Root causes: myopic greedy per-game argmax with no lookahead;
  treats 63 picks as independent EV terms, ignoring path-dependent covariance;
  maximizes leverage without checking whether high-leverage picks are
  correlated (same team's bracket path) — creates "correlation concentration"
  where multiple contrarian picks fail together.
- **Hedge mode**: significantly worse than seed (Bonferroni p<0.05).
- **GNN / Transformer prediction models**: no BSS improvement over plain LR —
  removed from the active prediction path (but the underlying data structures,
  `ScheduleGraph`/`GameEmbedding`/`SeasonSequence`, are still load-bearing for
  SOS features — see §7).
- **Backward construction mode (M5)**: mathematically a strategic duplicate of
  `champ_first` under the marginal-independence sampler; would require
  plumbing `barthag` through the construction dispatcher (an architectural
  lift) to differentiate — not pursued.
- **Training on regular season + tournament games together** (including
  NIT/conference tournaments): rejected — distribution mismatch, not fixable
  by more samples; ~2000× more irrelevant samples would bury tournament
  signal.
- **Kaggle-side, exhausted internal-only calibration mechanisms** (do not
  reopen without new external data): `torvik_isotonic` (nonlinear correction,
  overfit — great on 2025, disastrous on 2026), `odds_api`/`market_movement`/
  `closing_market_blend` features (existing `odds`/`market_prob` signal already
  captures the information), year-level chalk-adaptive correction scaling
  (real signal, r=-0.697 p=0.025, but effect size 3× short of the admission
  gate — irreducible because 2025 is strongly chalk and 2026 only modestly
  upset, no single scaling parameter isolates one without hurting the other),
  massey/ap as correction features (negative training residual correlation
  despite looking individually strong standalone).

## 4. Data provenance & leakage — gotchas worth remembering

- **Torvik barthag is NOT scraped** — it's locally computed from
  `historical_games_{year}.json` via `scripts/compute_pretournament_barthag.py`
  (15-iteration opponent-strength adjustment + Pythagorean formula). Guarded
  by `_validate_pretournament()`.
- **Four Factors consolidation (2026-08-16)**: `torvik_four_factors_{year}.json`
  was merged into `torvik_{year}.json` under a `four_factors` key (plus
  `four_factors_snapshots` for the momentum/coach-adjustment features).
  `data/raw/torvik_four_factors_{year}.json` (non-historical tier) is a
  separate, unconsolidated mirror.
- **Team-name resolver collision bug (found & fixed)**: fuzzy `containment`
  matching (confidence 0.9) collapsed distinct teams — e.g. a 1-game D3
  affiliate (`vermont_state_lyndon_hornets`) overwrote real Vermont's 35-game
  record, due to nondeterministic dict iteration order. Fixed by restricting
  the resolver to `exact_id`/`prefix_strip` and sorting by game count
  descending, first-writer-wins.
- **AP poll ID space collision**: the AP poll's own `team_no` field is in a
  *different ID space* than Kaggle's `MTeams.csv` (AP team_no=1207 is Duke,
  MTeams TeamID 1207 is Georgetown). Any AP-based work must join on display
  name, not team_no.
- **`TOURNAMENT_START_DATES` SPOF (found & fixed)**: root cause was a silent
  fallback in `torvik.py` returning `(None, None)` for a missing year,
  letting trank.php serve unfiltered post-tournament data. Now raises
  `ValueError` instead.
- **cbbpy team-ID bridge defect**: corrected 70% of `volatile` and 15% of
  `roster_adj` feature values across 15 backtest years once fixed. Does not
  affect `meta_region_poolaware` (doesn't consume those sources).
- **Massey/external-ratings date guard has no direct unit test** — filtered by
  `RankingDayNum <= max_day` from a hardcoded per-season table; spot-checked
  clean (correlation between Massey-vs-Torvik disagreement and actual outcomes
  ≈ 0, ρ=−0.22 to +0.03) but the guard itself isn't test-covered.
- **Calibration leakage bugs (found & fixed, 2026-03-31)**: a fallback path
  used to fit-and-evaluate on identical data when samples were insufficient
  (silently inflated metrics); `IsotonicCalibrator.fit_calibrate()` returned
  in-sample predictions (fixed via leave-one-out CV). Neither hit the primary
  production path (temperature scaling, 450+ tournament games).
- **Pre-2011 ESPN picks excluded from aggregated calibration** — genuine
  regime shift, not noise: pre-2011 champions are 3/3 one-seeds vs post-2011's
  1.86 average seed; public over-picked seed-1 by up to +14.5pp pre-2011.
  Driven by the 2011 field expansion (64→68) and an analytics-culture shift.
  Enforced via `MIN_PICKS_CALIBRATION_YEAR = 2011`.
- **Torvik player-CSV date filtering does NOT work** — `getadvstats.php`
  silently ignores `begin`/`end` params (verified byte-identical output with
  and without them, 2026-08-18). Torvik cannot retroactively supply
  pre-tournament player data for any past season; the only clean path is
  scraping *before* that year's tournament starts, going forward only. See
  `memory/next_steps_pretournament_player_data.md` for the live plan (cbbpy
  box-score re-aggregation is the working route).
- **Four-factors local-vs-Torvik validation ceiling**: after fixing the name
  resolver bug (which alone raised 4/8 features from r≈0.45 to r≈0.97),
  correlation saturates at r≈0.97-0.98 regardless of formula variant — this is
  a genuine data-source precision ceiling (Torvik publishes higher-precision
  decimals), not a fixable bug. Resolved by treating Torvik as the production
  source with local computation as a fallback (r≥0.85 tripwire).
- **Tournament ground-truth defects found & partly fixed**: 8 transposed
  games across 6 backtest years in `tournament_context_*.json` (see
  `memory/tournament_results_ground_truth_defect.md`); separately, 6 game
  corrections in 2023 and 5 in 2024 tournament_results (wrong winners/scores,
  one entirely mislabeled game, a missing 12-seed upset). Re-measurement after
  the transposition fix barely moved headline P(1st) (11.33%→11.20%) — a
  provenance fix, not a performance correction.

## 5. Kaggle secondary objective — plateau & forward policy

- **Optimization target is held-out Brier score, not BSS.** BSS is a
  diagnostic/guardrail only (confirms skill over seed baseline).
- **Recency weighting policy**: treat 2023-2025 as the priority evaluation
  window; the most recent 5 tournaments collectively weigh as much as all
  older tournaments combined (`recent_year_weight = n_older / n_recent`).
  Older years still matter for shrinkage/regularization — this is a modeling
  prior, not a theorem. **Scope boundary: this policy applies only to the
  Kaggle path, never to override the primary pool P(1st) objective.**
- **Current ceiling: BSS +0.133 (Brier=0.1305)**, judged an architecture
  plateau. Root cause: the correction model's +0.318 intercept is learned
  from 2009-2024, all of which had positive mean residuals (torvik
  under-predicted team1). 2026 broke the pattern (mean_residual=−0.012, an
  upset-heavy year where torvik over-predicted team1) — only 60% of 2026's
  corrections were in the right direction.
- **Verdict from an external-reviewer-style audit**: "You do not have a model
  architecture problem anymore. You have a signal acquisition problem." Future
  gains require new external data, not more internal blending/tuning.
- **Priority order for new external signal** (if resumed): (1) historical
  market odds — highest confidence, Phase 1 vendor-evaluation checklist
  written but **not yet executed** (`docs/kaggle-market-odds-phase1-checklist.md`
  content is now folded in below); (2) injury/availability data; (3)
  torvik+market correction; (4) better pool-specific opponent data; (5)
  roster continuity/transfer stability. More blend complexity is explicitly
  judged lowest-value.
- **Market-odds vendor acceptance rubric** (if this phase is ever run):
  tournament coverage ≥95% overall (no year <90%), reliable pregame-only
  timestamps (hard PIT gate — reject outright on failure, not weighted), book
  depth for consensus, closing moneyline minimum, deterministic team-name
  mapping. Vendor priority: The Odds API → SportsDataIO → Sportradar.
  Recommended first model if odds land: a small bounded monotone/logit
  correction on `logit(torvik_prob)` + `logit(market_prob)` + `seed_gap` —
  explicitly not a large ensemble.

## 5b. O6: is P(1st) actually calibrated against real outcomes?

A recurring diagnostic question — does a bracket's estimated P(1st) actually
correlate with its real placement rank? Answered three ways, and the answer
depends heavily on scoring/opponent scope (a good example of why §6's
methodology notes matter):

- **Shape-encoded, ESPN-national opponents (original, 2026-04-13)**: mean
  Spearman ρ = +0.37 across 14 years, p=0.002.
- **Team-identity, ESPN-national opponents** (the corrected scoring): mean ρ =
  **+0.61** across 15 years, 100% of years positive, t=+8.53 p<1e-6. The
  shape-encoded baseline materially understated the real signal.
- **Team-identity, actual 30-person pool** (the scope that actually matters):
  mean ρ = **+0.42** across only 4 years with real pool data, not significant
  (p=0.34). Per-year: 2023=+0.97, 2024=+0.36, 2025=+0.97, **2026=−0.60** — in
  2026, brackets with higher estimated P(1st) placed *worse* against the real
  pool than lower-P(1st) brackets (consistent with the "system ranked its own
  winning bracket #11" story in §2). ESPN-national calibration is stronger and
  more consistent than actual-pool calibration — opponent-model
  mis-specification (ESPN-national ≠ your specific pool) has real, if noisy,
  directional impact on P(1st) reliability in edge years.

## 5c. A few more items from the dead-end ledger not covered above

- **11 algebraically-redundant engineered features removed** (old 79-dim
  vector era): `adj_efficiency_margin` (exact linear combo of components
  already in the vector), `barthag` (monotonic transform of an existing
  feature), `true_shooting_pct`/`opp_true_shooting_pct` (r=0.92 with
  eFG%+FT-rate), `close_game_record` (pure noise, stability≈0.1 — computed
  from a ~5-game sample).
- **Brier-sharpening is explicitly prohibited for Kaggle submissions** — it
  would overfit Brier at the cost of realism (protocol-level rule, not just a
  measured null result).
- **Increasing the training window past ~9 seasons doesn't help** — caps out
  around 17.6k regular-season + ~63 tournament games/year; no BSS lift from
  going further back.
- **Kaggle admission mechanics, concretely**: the current baseline
  (`torvik_corrected_recent5_conservative`) passed the strict historical gate
  at final-holdout mean Brier 0.1301 vs incumbent ensemble's 0.1376 (BSS
  +0.1358 vs +0.0861), with the 2026 single-year regression (+0.0019) safely
  inside the 0.003 cap. Submission path:
  `scripts/kaggle_torvik_submission.py --mode torvik_corrected_recent5_conservative`.

## 6. Statistical methodology lessons (apply to any future strategy work)

- **14-17 year backtests have ~9-16% statistical power.** Use them only for
  binary/directional claims (e.g., "does P(1st) beat EV"), never to
  fine-tune parameter values (F4 cap, MC sim count, blend weights) — those
  need historically-averaged or theory-driven defaults instead.
- **Multiple-comparison correction matters.** The production 11.9% baseline
  was validated via a 10,000-draw permutation test (shuffle year labels
  across all 20+ tested strategies) — p=0.0076 after correction. Any new
  claimed improvement should clear the same bar, not just beat the mean once.
- **Check rank stability before trusting any ranker.** Run the ranker twice
  with independent RNG seeds on the same bracket set; if top-5 picks don't
  overlap, the ranker is selecting from noise and no objective-function
  tweak will fix it (the problem is simulation variance, not the objective).
- **Team-identity scoring, not shape scoring, is the correct evaluation
  metric** (§3 above) — shape scoring understated real signal by ~40-90% in
  spot checks and inverted signs in upset-heavy years. If you ever see a
  result computed under "shape" scoring, treat it as provisional.
- **A "validated N/N win rate" is inflated by mode-selection multiple
  comparisons** if many alternative modes were screened first — treat as
  directional lean, not a probabilistic guarantee, unless permutation-tested.
- **Real pool-opponent data covers only 4 of the historical backtest years**;
  the other years use simulated ESPN-national opponents. Any aggregate P(1st)
  claim should ideally be checked on the 4 real-opponent years alone to rule
  out simulator artifacts (this stress test was proposed but its completion
  status is unclear from the reviewed docs — worth re-verifying if precision
  matters).

## 7. Load-bearing code that looks dead but isn't — do not delete

- `src/ml/gnn/schedule_graph.py` — powers live SOS features despite
  `enable_gnn=false`.
- `src/ml/transformer/game_sequence.py` — provides `GameEmbedding`/
  `SeasonSequence` passthrough data structures used even with transformer
  inference off.
- `src/ml/ensemble/stacking_weights.py` + `src/forecaster/stacking.py` —
  unconditional module-level imports from several files despite
  `enable_stacking=false`; removal needs a call-site refactor first.

## 8. Known open technical debt (not yet fixed, roughly prioritized)

- **Scraper security** (P0): API keys passed in query params instead of
  `Authorization: Bearer` headers (`torvik.py`); API key cached via
  `os.environ`, visible via `ps aux`/crash logs.
- **Scraper robustness** (P1): no atomic file writes (temp+`os.replace()`) in
  `collector.py`/`dag.py`/`sports_reference.py` — risk of JSON corruption on
  crash/concurrent writes; advisory leakage warnings should be hard errors in
  strict mode.
- **Feature engineering, 7-feature model**: NaN→0.0 replacement is
  semantically misleading where 0 has real meaning (e.g. `elo_rating` default
  1500, `win_pct` default 0.5) — tree models could use native missing-value
  handling instead. `wab` and `wab_poisson` may be redundant (r>0.85
  unconfirmed).
- **2027 format expansion**: NCAA tournament expands 68→76 teams (12 play-in
  games producing duplicate seeds in the seeds file) — pipeline currently
  assumes 64+4 (First Four) and needs structural changes. `RUNBOOK_2027.md`'s
  troubleshooting content (`len(first_round) != 64` → 76-12=64) is now only in
  git history if needed.
- **Market-odds Phase 1 (vendor feasibility check) is written but not
  executed** — see §5's acceptance rubric; no vendor review or coverage
  artifacts exist yet.
- **Researcher-degrees-of-freedom leakage never fully audited**:
  hyperparameters tuned while observing backtest results on the same years.
  Current mitigation is narrow Optuna search bounds (15 trials, expanding-
  window 5-fold CV) plus a runtime `HoldoutContaminationError` guard, but no
  multiple-testing correction is applied across the ~150 Brier evaluations per
  fit.
- **No unit test directly exercises the Massey day-cutoff exclusion**, and the
  raw `MMasseyOrdinals.csv` is gitignored and uninspectable.
- **Blown-lead / clutch play-by-play features are not built** — no PBP is
  currently stored (only fetchable, via an unused `CBBPY_ROSTER_ENABLE_PBP`
  flag in `cbbpy_rosters.py`). See `memory/next_steps_pretournament_player_data.md`
  for the concrete next steps; per CLAUDE.md's north star, this is Kaggle-path
  work, not pool-P(1st)-path work, unless someone deliberately wants to
  re-test that assumption.

## 9. Data source & interface notes

- **Provider cascade for team ratings**: `trank.php` (primary, has
  server-side date filtering as an independent leakage guard) → ESPN →
  sportsdataverse → cbbpy.
- **`torvik_map` context fields** (`preseason_ap_rank`, coach-tournament
  fields, `conf_tourney_champion`) are injected by `enrich_tournament_context()`
  in `data_loader.py`, *not* by the Torvik scraper itself — `feature_engineering.py`
  reads them as if native to torvik_data. Worth knowing when tracing where a
  feature value actually originates.
- **Four Factors fields may be `math.nan`** when unavailable (handled via
  seed-conditional priors); shooting/extended fields use `0.0` to mean
  "not available" instead — an inconsistency worth remembering when debugging
  missing-value handling.
- Rankings JSON files redundantly carry both `team_name` and `name` keys —
  some consumers read one, some the other.

---

*Sources folded into this file (deleted after consolidation, recoverable via
`git log --all --full-history -- <path>` or `git show <sha>:<path>`):
root-level project docs (AGENTS.md, ARCHITECTURE.md, the ANALYSIS_*/AUDIT_*
docs, COUNCIL_LESSONS.md, DETERMINISTIC_2027_EXPERIMENT_MEMO.md,
FEATURE_ENGINEERING_AUDIT.md, root MEMORY.md, POOL_STRATEGY_RECOMMENDATION.md,
PROJECT_STATUS.md, RUNBOOK_2027.md, STRATEGY_CATALOG.md, WORKFLOW.md, the v2
protocol doc, 6 council reports/transcripts); `docs/` (16 files: council
lessons, provenance/leakage audit, gap-analysis roadmap, Kaggle roadmap/
checklist/policy, statistician review, Torvik dependency/interface specs,
chalk-upset plan, p1st-ceiling council report, 5 session summaries);
`artifacts/` (8 audit docs + 2 log files + the `backtest_runs/` directory of
raw timestamped run logs); and ~20 loose raw backtest-result `.txt` logs at
root and in `scripts/`.*
