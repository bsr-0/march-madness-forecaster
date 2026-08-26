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

- **Target cleaning and game-context features for the per-date matrix**
  (2026-08-26, measured against the 41,321-row regular-season + conference-
  tournament matrix, LOYO, scale fit on the training fold only; baseline log
  loss 0.53184):
  - **Winsorizing margin** at ±30/25/20/15/12/10: null. ±20 costs +0.00002 log
    loss; every cap sits within 0.0003. The reason is in the distribution —
    margin kurtosis is **3.45 against 3.0 for a Gaussian**, so there are no fat
    tails to clip. 14.1% of games exceed ±20 but the tail is not heavy.
    **Careful with the inference here** (corrected 2026-08-26). It is tempting
    to read this as evidence the Student-t link is absorbing the tails. The
    simpler explanation covers both facts: margins are close to Gaussian, so
    neither clipping nor a fat-tailed link has much to work with. Measured
    directly, the t-link is NOT distinguishable from the normal on mean log
    loss (paired bootstrap 95% CI [-0.0032, +0.0159] on 630 games). It does
    earn its parameter on SATURATION rather than on average score: with
    nu = 2-3 no prediction comes within 4.9e-3 of the probability clip, while
    nu = Infinity produces 1.3e-4 and pins against it.
  - **Possession-adjusted margin** (margin / possessions × 100, possessions
    estimated as FGA − OR + TO + 0.475·FTA): null, −0.00000 log loss and
    marginally worse RMSE. The `mp` field is retained on each row so this is
    cheap to re-test; the columns were not.
  - **Rest days** (days since previous game, differenced): −0.0348 ± 0.0456
    points per day. CI straddles zero, and the point estimate has the *wrong
    sign* (more rest → slightly worse), which reads as noise.
  - **Bid-secured proxy** (top-30 dated barthag in the closing three weeks,
    differenced): −0.0101 ± 0.6669 on 1,158 rows. A CI 66× the point estimate
    measures nothing.

  **Scope caveat — do not cite the last two as settled.** Rest and bid-security
  were hypothesised as *tournament-context* effects, and this row set contains
  **zero NCAA tournament games** (they are held out by construction; see
  `scripts/assert_prediction_invariants.py` check 3). Conference-tournament
  games are ~10.8% of rows. A rest effect that is real in March and absent in
  January would read as null here. The correct statement is "null on the
  current row set", not "no rest effect". Winsorization and possession
  adjustment are different: those are target refinements tested on exactly the
  population they apply to, so null means null.

  The `rest_diff` and `bid_secured_diff` columns were REMOVED rather than kept
  as documented nulls. A near-zero column is worse than an inert one because it
  is visible: a reader scanning the feature list infers the model accounts for
  rest and bid security, and it does not.

  **Contrast with what did work.** Venue coding moved LOYO RMSE by −0.446 on
  the same row set. The distinction is that venue fixed a *specification*
  error — an omitted variable correlated with the features (strong teams buy
  home games, so strength correlates with home-ness at r = 0.13, and the home
  effect was being absorbed into the strength coefficients, inflating srs_blend
  13% and barthag 44%). The step-5 items are target/context refinements and
  address no specification error, which is the likely reason all four are null.
  Remaining wins are more likely in omitted variables correlated with existing
  features than in further target cleaning.

- **Ridge lambda retuning for the browser model** (2026-08-26): null, and the
  first two readings of it were both wrong in instructive ways.

  `FIT.LAMBDA` is fixed at 1.0 and never fold-tuned, and a sweep under the OLD
  self-graded calibration showed log loss improving monotonically toward
  lambda = 0 (0.45047 vs 0.45127) — which looked like ridge over-penalising.
  Two errors sat inside that reading:

  1. **`a` is not comparable across the sweep.** nu shifts 3 -> 2 as lambda
     drops, and a fatter-tailed t needs a larger `a` for the same central
     sharpness, so the two parameters trade off. The hypothesis that `a > 1`
     was "undoing ridge shrinkage" is REFUTED: `a` does not fall toward 1 as
     lambda drops, it RISES to 1.70. Log loss is the only comparable column.
  2. **`a` was fit globally at each lambda**, i.e. self-graded one level up.
     Refitting it walk-forward moves the apparent optimum from lambda = 0 to
     lambda ~ 0.25 and flattens the curve — lambda and `a` are partially
     redundant knobs, so pairing a lambda with an `a` fitted elsewhere finds
     the wrong lambda.

  With `a` refit walk-forward inside each lambda, **paired bootstrap on the 630
  warm-year predictions**:
      lambda=1 vs lambda=0     +0.00065  95% CI [-0.00399, +0.00335]
      lambda=1 vs lambda=0.25  +0.00070  95% CI [-0.00285, +0.00278]
  Both straddle zero. Monotone-improvement-toward-an-unregularised-boundary is
  a common shape for noise, and that is what this was. **lambda stays at 1.0.**

  Not pursued as a result: nested per-fold lambda selection (nothing to select
  — everything in [0, 1] is equivalent within noise) and the per-fold
  coefficient-variance check that would have been required before shipping
  lambda -> 0 with 30 collinear rating columns.

  **Method note worth keeping.** RMSE and log loss disagreeing across a sweep
  is not a tie to break by preference: RMSE scores the mean model, log loss
  scores mean and scale together, so less shrinkage buys sharper probabilities
  at the cost of point accuracy. When they diverge, the knob is trading between
  them rather than improving anything, which is what the bootstrap confirmed.

- **Pooling regular-season games into the tournament model** (2026-08-26).
  Closed negative in both standardisation regimes. The expanded matrix is
  41,321 rows against 1,008 and is leakage-clean, but it does not help.

      D1-field standardised   expanded vs tournament-only  -0.00763
                              95% CI [-0.00346, +0.01936]  NOT a finding
      top-100 filtered        expanded vs tournament-only  +0.00498
                              95% CI [-0.01078, +0.02127]  NOT a finding

  **Two structural costs, both larger than any gain**, measured by decomposing
  the distance from the 0.45296 baseline:
      -t_rank (10 keys, same standardisation)   +0.0823
      D1-field standardisation on top of that   +0.0420
      expanded training population              -0.0076 (not significant)

  `t_rank` cannot come along: it has no dated snapshot at ANY boundary, and
  deriving it would ship a guess (ordering by barthag misplaces 324 of 364
  teams — see rescrape_pretournament_torvik). It is the single most valuable
  feature in the model, so the per-date pipeline is structurally barred from
  the thing that matters most.

  Standardisation compresses the signal: the tournament field sits in a narrow
  band at the top of D1, so z-scoring against all ~350 teams shrinks barthag
  differentials from sd 0.949 to 0.731, a 1.30x loss of discrimination on
  exactly the games being predicted. But it is not optional — it is what makes
  regular-season and tournament rows commensurable at all. The two are in
  direct tension: what lets you USE 41,321 rows is what degrades the 63 you
  care about.

  **Filtering instead of rescaling was tried and does not rescue it.** Training
  only on games between top-100 teams (8,232 rows) makes the population
  commensurable by construction rather than by transformation. It does raise
  the differential sd to 1.136 — the 68-team tournament field is actually WIDER
  than the top 100, because it contains ~30 automatic qualifiers including
  weak 16 seeds. But the expanded-population effect remains insignificant, and
  the elite eval numbers are NOT comparable to the D1 ones: filtering removes
  the most predictable games (1-vs-16 style mismatches), so log loss rises for
  reasons that have nothing to do with the model.

  **The per-date matrix is not wasted.** It is the only surface on which the
  point-in-time, venue and confound machinery could be measured, and venue's
  -0.446 came from there. Research instrument, not a replacement pipeline.

### The stopping rule (adopted 2026-08-26)

**If a change is smaller than its bootstrap CI, it is not a finding.** Applied
retroactively this session it dissolved three results that had already been
written up as wins: the lambda sweep (0.0008, CI straddling zero), the
year-to-year instability in the calibration constant `a` (11 of 12 per-year CIs
contain the global value, trend -0.0034 +/- 0.0927), and the expanded-training-
population gain above (-0.00763, CI [-0.00346, +0.01936]) — that last one
reported as a win one message before the bootstrap was run.

This is not incidental sloppiness; it is what the regime demands. Measured
effects this session ran 0.00001-0.00004 (target cleaning), 0.0008 (lambda),
0.0018 (calibration bias) against a test set of 63 games/year. Most differences
in that range are smaller than the sampling noise. College basketball games
carry ~10.5 points of residual SD and no feature engineering changes that.

**The one substantial win all session was venue, -0.446 RMSE, and it was a
specification error** — an omitted variable correlated with the features
(strong teams buy home games, so venue correlates with strength at r = 0.13,
and the home effect was being absorbed into the strength coefficients).
Everything framed as "more data" or "a better target" returned approximately
nothing. Hunt specification errors, not accuracy.

## 4. Data provenance & leakage — gotchas worth remembering

### Bracket Lab feature matrix — scope of the leakage audit (2026-08-22)

State this precisely rather than as a blanket clearance, because the earlier
blanket version is exactly what let a contaminated variable survive:

> The current 24-feature matrix has been audited for known temporal leakage,
> with `returning_minutes_pct` and `freshman_minutes_pct` explicitly excluded
> after empirical confirmation that their season aggregates incorporated
> post-Selection-Sunday games. The remaining features have passed the
> applicable game-by-game or construction-level causal checks.

What each check actually was, so the claim can be re-verified rather than
trusted:

| Family | Check | Result |
|---|---|---|
| Torvik ratings, four factors | per-season `cutoff_date` = day before `tournament_start` | construction-level, clean |
| Kaggle box score, `overtime_rate` | is any tournament game present in `MRegularSeasonDetailedResults`? | 0 of 1,449, clean |
| Form (margin, close games, bad losses) | is any tournament game present in `MRegularSeasonCompactResults`? | 0 of 1,449, clean |
| Coach history | seasons summed strictly `< year` | construction-level, clean |
| Roster minute shares | correlation of extra roster games with rounds actually won | **r = +0.71 to +0.96 — excluded** |

**The roster finding.** Every `cbbpy_rosters_*.json` was scraped 2026-02-21, so
for 2010–2025 a player's minutes-per-game is averaged over a game count that
includes the team's tournament run. Purdue 2024 carries 39 games against 33
played before the tournament — exactly its six-game run to the final. 2026 is
the control: its snapshot is genuinely mid-February and the correlation drops to
−0.13.

**Performance did not depend on it.** 26 → 24 variables left walk-forward
accuracy at 78.2% and marginally improved error (RMSE 10.48 → 10.46, R²
0.536 → 0.538). That is the useful half of the result: the model never needed
the contaminated information.

**Guards** (`tests/test_training_matrix.py`, both mutation-tested): one fails if
either key returns to the training matrix; one fails if the matrix and the UI
menu drift apart. They are built by separate scripts, which is a quiet
maintenance failure mode on its own.

**Not fixed, and deliberately so.** The honest reconstruction is
`minutes before tournament_start / team minutes before tournament_start`.
`player_minutes_*.json` holds only season-level aggregates, so that quantity
cannot be recovered — inferring it from the aggregate would substitute a new
approximation for a known contamination. Blocked until game-level box scores
land.

### Bracket Lab — known limitation, not a data-integrity problem

The shipped 24-variable ridge model is tuned for accuracy, and its coefficients
are **not interpretable**. With everything enabled it prints large offsetting
weights on near-substitutes (`−39 × rating + 36 × national rank`) and more than
half the coefficients change sign between walk-forward folds.

| ridge per 1k rows | accuracy | max abs coefficient | sign-flipping vars |
|---|---|---|---|
| 1 (shipped) | 78.2% | 40.7 | 14 of 26 |
| 20 | 73.7% | 15.3 | 9 |
| 400 | 73.9% | 3.0 | 4 |

Interpretability costs roughly 4 accuracy points. The decision is to keep the
accuracy-first model as the forecaster and treat coefficient instability as a
documented limitation — the equation marks unstable terms rather than presenting
them as effects. If a readable coefficient vector is wanted, the right move is a
separate constrained model built for that purpose, not detuning the predictor.

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

## 6b. The pairwise probability contract, and the corrected stochastic baseline (2026-08-19)

Full detail in `ARCHITECTURE_AUDIT_PREFERENCE_BRACKETS.md`. The short version:

**The bug.** `round_probs[team][R]` is the *marginal* P(team wins its round-R game). Seven
call sites reconstructed a head-to-head probability from two marginals as `p1 / (p1 + p2)`.
Decomposing `p_i = r_i * w_i` (reach × win), the reach terms never cancel, so the expression
is not `P(t1 beats t2)`. It is exact in R64 only (both teams reach with certainty, so
`p1 + p2 = 1`) and degrades from R32 onward, one-directionally toward the favorite:

| round | R64 | R32 | S16 | E8 | F4 | CHAMP |
|-------|----:|----:|----:|---:|---:|------:|
| mean abs error | 0.0008 | 0.0708 | 0.1118 | 0.1169 | 0.1283 | 0.1358 |

**The contract.** `src/prediction/pairwise.py`. Legal direction is
`pairwise -> simulator -> outcomes -> marginals`, never the reverse. A probability base is
now a `ProbabilityBase` — a Mapping over its marginals that also carries the pairwise table
they were derived from. Bases with no head-to-head model (`contrarian`, `pool_wisdom`,
adjusted pipelines) carry `pairwise=None` and raise `MissingPairwiseSource` rather than
fabricate one. Enforced by `tests/test_pairwise_contract.py`, including a static AST scan
of `src/` and `scripts/`.

**Marginals are NOT wrong for expected score.** `_make_ev_scorer` and
`_compute_expected_points` are correct as written and were deliberately left alone:
`E[points] = sum_R pts_R * sum_{t in picked_R} marginal[t][R]` exactly, by linearity of
expectation. The contract governs one thing only — never manufacture a *pairwise*
probability out of *marginals*.

**Production was unaffected.** `torvik` round_probs and the brackets from `region_top_n`,
`exhaustive_champion`, `champ_first`, `f4_first`, `e8_first` and `forward_greedy` are
bit-identical before and after (verified 2023-2026 by hashing picks/champion/F4/EV against a
clean HEAD worktree). **`meta_region_poolaware` and the 11.2% P(1st) figure are unchanged
and required no re-validation.**

**The corrected stochastic baseline.** Canonical contract
(`--team-identity --opponent pool --n-opponents 30 --n-repeats 100`, 2011-2026 excl. 2020,
n=15). These are the numbers to trust going forward; the "before" column is retained only so
nobody re-derives it and thinks the suite regressed.

| Mode | P(1st) pre-fix (invalid) | **P(1st) corrected** | MeanScore pre-fix | **corrected** |
|------|------------------------:|---------------------:|------------------:|--------------:|
| seed | 4.92% | **4.05%** | 695 | **612** |
| noseed | 3.93% | **2.79%** | 637 | **602** |
| blend | 4.78% | **3.43%** | 661 | **610** |
| torvik | 4.23% | **3.40%** | 728 | **662** |

The `noseed` and `blend` figures above are the **2026-08-20 re-run**, after the
train/serve skew fix (§6c). The intermediate values measured on 2026-08-19
(noseed 0.58% / MeanScore 338, blend 1.95% / 464) measured a coin-flip model and
should not be cited. `seed` and `torvik` reproduced to the digit across both runs,
which is what validates the comparison.
| meta_sa | 1.33% | **0.73%** | 199 | **215** |
| meta_sa_chalk | 1.00% | **1.67%** | 195 | **316** |
| meta_sa_vol | 1.00% | **0.93%** | 253 | **281** |

The pre-fix run reproduced the then-documented baselines exactly (seed 4.9%, SA 1-2%), which
validates the harness. SE is ~0.3-0.5pp: noseed/blend are ~6-7 SE and unambiguous; the
meta_sa_chalk gain is ~2.5 SE across correlated years and is not a real improvement.

**The invalid transform was load-bearing — do not let that tempt a revert.** It sharpened
later rounds toward favorites, favorites usually win, and P(1st) in a 31-entry
winner-take-all pool tracks expected score. The historical figures for these modes were
flattered by a defect; the corrected figures are the honest ones. Do not quote the torvik
delta as "the price of upset variation" — the correction moved several things at once and
does not isolate that axis.

**Two open items this exposed** (see §8):
1. `noseed`'s collapse turned out to be a **train/serve feature skew bug**, diagnosed
   2026-08-19 — see §6c. The noseed model is not weak; it is being served a feature vector
   whose four most predictive dimensions are identically zero.
2. A temperature `tau` on the pairwise logit is worth testing as an independent modelling
   experiment — but `tau` must NOT be fit to reproduce the pre-fix numbers, which would be
   reverse-engineering a bug. The question is whether controlled sharpening helps
   winner-take-all performance without destroying calibration; `tau = 1` winning is a fine
   answer.

**Simulated annealing: re-validated, conclusion unchanged.** SA's `predict_fn` was
round-agnostic (`max(probs.values())` is always the R64 value, so F4 games were scored with
first-round win rates), which produced pathological brackets — the 2025 Final Four contained
a 16-seed. With a correct pairwise `predict_fn` the pathologies vanish and MeanScore improves,
but P(1st) is 0.73-1.67% against a 4.05% seed baseline. **SA is ineffective on the merits,
not because of the broken predict_fn. Consider this closed.**

**Known remaining conversions, deliberately unfixed** (both documented in-place and
allowlisted in the contract test): `scripts/_bracket_export_common.py` (feeds the web UI's
displayed `win_prob`; fixing it moves user-visible numbers) and
`src/prediction/meta_selector.py::_pairwise_prob` (consumed as a GBM *feature* computed
identically across bases; changing it requires retraining every `meta_gbm*` mode).

## 6c. `noseed` train/serve feature skew — the B1 base has never actually been tested (2026-08-19)

Diagnosed while investigating why `noseed` fell to 0.58% P(1st) / MeanScore 338 under the
corrected sampler (§6b). **It is not a calibration weakness. It is a bug, and it means the
no-seed model's own matchup predictions have never contributed to any backtest result.**

**Root cause: two different functions named `_load_team_stats`.**

| | reads | returns |
|---|---|---|
| `src/prediction/noseed_model.py::_load_team_stats` (used by **training**) | `torvik_{year}.json` → `data["teams"]`, merged with `four_factors` | full stats: `adj_offensive_efficiency`, `adj_defensive_efficiency`, `adj_tempo`, `barthag`, + 8 four-factors |
| `scripts/mc_pool_backtest.py::_load_team_stats` (used by **inference**, line ~2197) | `_load_torvik_ff(year)` | the `four_factors` sub-dict **only** — 8 keys |

`_build_feature_vector` needs 12 dimensions. Served the four-factors-only dict, four of them
fall through to their hardcoded defaults on *both* sides of the differential, so they are
identically `0.0` for every matchup — including the single most predictive feature,
`barthag`.

UConn (1) vs Stetson (16), 2024:

```
TRAINED features: [ 18.37, -20.02, -2.10, 0.037, -0.009, 0.090, 0.016, -0.069, 0.027, 0.044, 0.066, 0.612 ]
SERVED  features: [  0.00,   0.00,  0.00, 0.037, -0.009, 0.090, 0.016, -0.069, 0.027, 0.044, 0.066, 0.000 ]
                    ^adjO   ^adjD  ^tempo                                                        ^barthag
```

The consequences are exactly what you would predict from that vector:

| source | mean \|p-0.5\| | frac in [0.4,0.6] | agrees with seed favorite (R64) | 1-vs-16 games |
|---|---:|---:|---:|---|
| noseed pairwise (served) | 0.041 | 93.8% | **17/32 — chance** | 0.474 – 0.540 |
| noseed marginal ratio (old path) | 0.211 | 50.0% | 30/32 | ~0.983 |
| torvik log5 (reference) | 0.270 | 28.1% | 28/32 | ~0.980 |

It ranks a 16-seed over North Carolina (0.474). Its R64 favorite agreement is 17/32, which
is a coin flip.

**Why this stayed hidden for so long.** The old sampler never called
`build_noseed_probabilities`. It used `build_noseed_round_probabilities`, which starts from
*seed-based* historical advancement rates and applies a no-seed adjustment — and that
adjustment is computed from the same zeroed features, so it is ~0. The `noseed` base was
therefore **the seed base with a near-null adjustment**, which is why its pre-fix P(1st)
(3.93%) sat just under seed's (4.92%) and why it agreed with the seed favorite 30/32.
Correcting the pairwise contract simply routed sampling to the real model for the first
time and exposed that the model was being fed a crippled vector.

**Implications for the historical record:** any past conclusion of the form "the no-seed
model doesn't beat seed" is unsupported — the B1 base had never been evaluated on its own
matchup probabilities. `blend` (α·seed + (1−α)·noseed) is contaminated by the same bug.

**Scan for the same defect elsewhere (2026-08-20).** The bug had three sites, not one:

| site | status |
|---|---|
| `scripts/mc_pool_backtest.py::_load_team_stats` | **was broken** — fixed |
| `src/optimization/recency_hparam_fitter.py` | **was broken** — imports the loader above, so it was fitting `blend_alpha` walk-forward against a coin-flip model. Fixed transitively. |
| `scripts/divergence_diagnostic.py::_load_team_stats` | **was broken** — an independent copy returning `_load_torvik_ff`. Every divergence figure it produced before today is unreliable. Fixed. |
| `src/cli/pool_cmds.py`, `scripts/unified_mode_evaluation.py`, `scripts/backtest_ev_vs_kaggle.py` | always correct — they imported `noseed_model._load_team_stats` |

Verified empirically rather than by reading: every loader that feeds noseed now returns
362 teams with all 12 required keys present.

Two adjacent findings from the same scan:

- **`torvik_correction` (the production Kaggle model) is structurally immune.** Its
  `_feature_vector` takes explicit positional arguments, so omitting one raises `TypeError`
  rather than silently defaulting, and its optional market/elo sources fall back
  identically in `fit` and `predict` — consistent degradation, not skew. Worth copying as
  a pattern.
- **Seven falsy-`or` defaults** in `src/pipeline/stages/data_loader.py` (`two_pt_pct`,
  `three_pt_pct`, `three_pt_rate`, `ft_pct`, and the `opp_` variants) sat inconsistently
  among neighbours using the correct `.get(key, default)` form. No current value changes —
  a season-aggregate shooting percentage is never exactly zero — but they were the same
  trap, and are now consistent.

**Residual risk not closed:** `feature_engineering.extract_team_features` reads 64 keys and
its call site passes `torvik_map.get(team_id, {})` / `proprietary_map.get(team_id, {})`,
i.e. the same silent-default shape. It sits on the tournament-pipeline path, which does run
`_strict_torvik` validation, so it was not verified broken — but it has no per-key coverage
check equivalent to `validate_stats_payload`, and would fail the same way if a source
went missing.

**FIXED 2026-08-20.** Three changes:

1. `mc_pool_backtest._load_team_stats` now delegates to `noseed_model._load_team_stats`,
   supplying all 12 features instead of the four-factors sub-dict.
2. `noseed_model.validate_stats_payload` + `FeatureSkewError` — checks *per-key coverage
   across the payload*, so one team missing one stat still passes (that is what the
   per-key default is for) while a key missing for everyone raises. Wired into
   `train_noseed_model`, `build_noseed_probabilities` and
   `build_noseed_round_probabilities`, so training and both serving paths are guarded.
3. `_get_stat` no longer uses `stats.get(k) or enriched.get(k)` — the falsy-coalescing form
   treats a legitimate `0.0` as missing. Four-factor rates are never exactly zero so no
   current value changes, but the trap is gone.

Measured effect on the model's own output (2024 R64, versus the recorded pre-fix figures):

| | before | after | torvik reference |
|---|---:|---:|---:|
| mean \|p-0.5\| | 0.041 | **0.241** | 0.270 |
| games in [0.4, 0.6] | 93.8% | **25.0%** | 28.1% |
| agrees with seed favourite | 17/32 (chance) | **27/32** | 28/32 |
| 1-seed over 16-seed | 0.474-0.540 | **0.919-0.941** | ~0.980 |

The model now discriminates about as sharply as torvik. Regression tests in
`tests/test_noseed_feature_skew.py` pin both the defect signature and the guard.

**Re-baselined 2026-08-20** (canonical contract, `seed`/`torvik` held as controls and
reproduced exactly, which validates the run):

| mode | P(1st) skewed | fixed | MeanScore skewed | fixed |
|------|-------------:|------:|-----------------:|------:|
| noseed | 0.58% | **2.79%** | 338 | **602** |
| blend | 1.95% | **3.43%** | 464 | **610** |

The fix restored the model's scoring power almost entirely: MeanScore 338 -> 602,
against seed's 612. It is no longer producing near-random brackets.

**But it still loses to seed, and now that is a real result rather than an artifact.**
noseed 2.79% vs seed 4.05% (MeanRank t=-5.71, p_adj=0.0002; BestRank p_adj=0.0130).
blend 3.43% vs seed 4.05% (MeanRank p_adj=0.0003, though BestRank p_adj=0.9484 — not
separable on that view).

So the question the bug had left open is now answered: the B1 no-seed base *has* been
evaluated on its own matchup probabilities, and it does not beat seed for P(1st). Note
the pre-contract figure of 3.93% was the seed model wearing a noseed label — the real
model scores 2.79%, i.e. worse than the thing it was accidentally imitating.

**One observation worth chasing.** MeanScore is now nearly identical to seed (602 vs
612) while P(1st) is materially lower (2.79% vs 4.05%). Similar expected points, worse
win rate, means the gap is structural rather than a matter of raw strength — the
brackets are positioned differently against the field, not simply weaker. That is
consistent with this project's core finding that construction and selection dominate
prediction accuracy, and it is the kind of gap a preference/portfolio layer could
plausibly close. Not investigated.

**Caveat:** this run used the fixed `blend_alpha=0.5`, not a refitted value. Now that
noseed is a real model the optimal alpha may differ, so `blend` in particular deserves a
refit before its 3.43% is treated as final.

## 6d. ESPN publishes no substitution events before 2025-02-11 (2026-08-19)

**Hard external boundary. Expensive to rediscover — measured, not inferred.**

`pbp_player_minutes.py` reconstructs on-court intervals from play-by-play
substitution events. Those events do not exist in ESPN's feed before **2025-02-11**:

| season | opening-day games with subs | March games with subs |
|--------|---------------------------:|----------------------:|
| 2026 | 60/60 | 80/80 |
| 2025 | 0/60 | 80/80 (cutover mid-season) |
| 2024 | 0/60 | 0/80 |
| 2023 | 0/60 | 0/80 |

Narrowing the 2025 file to the day: `2025-02-08` → 0/12, `2025-02-10` → 0/12,
`2025-02-11` → **1/12**, `2025-02-12` → 12/12.

Play *text* was also searched for sub-like wording (`subbing`, `substitut`,
`enters the game`) under any `play_type`: zero matches pre-cutover. **The data is
absent, not relabelled** — no parser change can recover it. Play schemas are
otherwise identical across eras (same 19 keys, `athlete_id`/`name`/`team` all
populated), so this is not a scraping or parsing defect.

Consequences: `player_minutes_2026` = 9,581 players; `player_minutes_2025` = 4,386
(only 2025-02-12 → 2025-03-17, ~26% of that season, so a biased basis for
season-long minutes shares); 2024 = zero, file never written; 2023 = 26 players
(noise). Every season older than 2025 will produce nothing. `clutch_features` and
`shooting_features` derive from score/clock and are unaffected — they are the
remaining justification for the PBP backfill.

**Use the boxscore route instead** — `src/data/scrapers/espn_boxscore.py`. ESPN's
published per-player stat line, minutes first, players grouped `starters`/`bench`.
Probed live: 2022, 2015 and 2009 games all return complete data whose per-team
minutes sum to exactly 200 (5 x 40 regulation). It is better than the PBP route on
every axis — full historical coverage, published rather than reconstructed,
starters labelled rather than inferred — and supersedes it even for 2025-2026.
`build_season_minutes_features` in `src/data/features/boxscore_player_minutes.py`
emits the identical schema, so `player_minutes_{year}.json` consumers need no change.

The boxscore page is a *separate endpoint*: the playbyplay page carries only the
boxscore tab's column config plus a nav link, so extending the PBP scraper cannot
pick this up.

**Silent-failure guard added.** The 2026-08-19 backfill logged
`player_minutes produced nothing` for 2024 and wrote a 26-player file for 2023, and
the run continued past both — anyone reading the log tail would have concluded it
was healthy. `backfill_pbp_history.py` and `build_pbp_derived_features.py` now carry
per-builder `min_expected` thresholds, escalate empty/thin results to `logger.error`,
and print an end-of-run coverage summary flagging both empty and THIN seasons.
`boxscore_player_minutes` raises `MinutesCoverageError` rather than writing a
misleadingly small artifact.

## 6e. Common random numbers in candidate selection (2026-08-20)

Three selection loops drew a fresh opponent field and a fresh tournament *inside*
the per-candidate loop, giving every candidate independent noise that `argmax`
could not distinguish from signal. Fixed by hoisting the draws into
`draw_selection_trials()` and scoring via `score_candidate_p1()`. Scoring logic
is byte-identical; only the provenance of the trials changed.

Acceptance results (2024 fixture, production candidate recipe, 500 trials, 5 seeds):

| criterion | result |
|---|---|
| per-candidate estimates stay consistent | **PASS** — max abs mean diff 0.0044, min p=0.42 |
| candidate-difference SE >= 2x better | **FAIL** — 0.0102 -> 0.0088 = **1.16x** |
| selection stability improves | **PASS** — modal winner 2/5 -> 3/5, distinct winners 3 -> 2 |
| controls reproduce / production unchanged | **PARTIAL** — see below |

Also 3.3x faster: fields and tournaments were being regenerated once per
*candidate-trial* rather than once per *trial*.

**The >=2x prediction was wrong, and the error is instructive.** It came from a
single candidate pair rather than an average over all pairs. CRN cancels only
*shared* noise, so the benefit scales with candidate correlation: similar pairs
(Hamming <= median) improved 1.85x, distant pairs 0.94x — none at all. A diverse
candidate set gains less than a similar one, which is the opposite of the
intuition that variance reduction helps most when choices are hard.

**The measurement's real value was a different finding.** Production generates a
mean of 25.5 candidates per year, but on the torvik base alone the recipe dedups
to 5 distinct brackets with a mean pairwise Hamming distance of **6 games out of
63**. The selector is choosing among near-identical brackets, so the true P(1st)
differences between them are tiny and no amount of variance reduction makes that
choice reliable. **Candidate diversity, not selection precision, is the binding
constraint** — directly relevant to the scenario-bank design, whose candidates
come from simulated tournaments and are diverse by construction.

**Full backtest (canonical contract, n=15).** Controls reproduced exactly — seed
4.05%/612 and torvik 3.40%/662, to the digit. `meta_region_poolaware` came in at
**10.47%** against the documented 11.2%.

That difference is **-0.73pp = -0.92 SE** (SE ≈ 0.79pp at this rate over 1500
year-repeats), i.e. statistically indistinguishable from unchanged — but it
cannot be claimed as a clean pass either, and the comparison is confounded: the
11.2% reference predates *both* the CRN change and the noseed train/serve fix,
and poolaware's candidate sweep includes a `blend` base that the noseed fix
altered. Isolating CRN's own contribution needs a run at the preceding commit
(~3h), not yet done. The 2026-08-18 reference log is no longer on disk (removed
in the doc-consolidation commit), so no paired per-year comparison is possible.



- `src/ml/gnn/schedule_graph.py` — powers live SOS features despite
  `enable_gnn=false`.
- `src/ml/transformer/game_sequence.py` — provides `GameEmbedding`/
  `SeasonSequence` passthrough data structures used even with transformer
  inference off.
- `src/ml/ensemble/stacking_weights.py` + `src/forecaster/stacking.py` —
  unconditional module-level imports from several files despite
  `enable_stacking=false`; removal needs a call-site refactor first.

## 8. Known open technical debt (not yet fixed, roughly prioritized)

- **`blend_alpha` should be refitted** (P2, opened 2026-08-20): `recency_hparam_fitter`
  was fitting walk-forward alpha against the coin-flip noseed model. The canonical
  backtest uses a fixed `blend_alpha=0.5` so its numbers are unaffected, but any
  `--hparam-fitter` run before 2026-08-20 is void, and `blend`'s re-baselined 3.43%
  (§6c) may move once alpha is refitted against the working model.
- **Why does noseed match seed on MeanScore but not P(1st)?** (P2, opened 2026-08-20):
  602 vs 612 expected points, but 2.79% vs 4.05% win rate. Similar scoring power, worse
  pool position — a structural difference rather than a strength one. Possibly reachable
  by a construction/portfolio layer. See §6c.
- **`extract_team_features` has no per-key coverage check** (P2, opened 2026-08-20): reads
  64 keys, and its call site passes `torvik_map.get(team_id, {})`, the same silent-default
  shape that hid the noseed skew for months. Not verified broken — it sits behind
  `_strict_torvik` validation on the pipeline path — but it has no equivalent of
  `validate_stats_payload`. See §6c.
- **Temperature `tau` on the pairwise logit is untested** (opened 2026-08-19): whether
  controlled sharpening (`p' = sigmoid(logit(p)/tau)`) improves winner-take-all
  performance without wrecking calibration. Must be fit walk-forward against P(1st) /
  MeanScore — explicitly NOT tuned to reproduce the pre-correction numbers. See §6b.
- **Two marginal→pairwise conversions remain by choice** (opened 2026-08-19):
  `scripts/_bracket_export_common.py` (UI-visible `win_prob`) and
  `src/prediction/meta_selector.py::_pairwise_prob` (GBM feature). Both are documented
  in-place and allowlisted in `tests/test_pairwise_contract.py`. Fixing either has
  downstream cost — moved user-facing numbers, or a full meta-selector retrain. See §6b.
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
