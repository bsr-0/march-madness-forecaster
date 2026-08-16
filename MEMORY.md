# MEMORY.md

Living index of settled decisions, dead-ends, and current baselines.
**Read this before proposing model changes, new features, or new experiments.**
If a claim here contradicts your instinct, cite MEMORY.md and ask — do not re-litigate.
For a blunt external-review-style summary of the repo's current strengths, weaknesses, and roadmap across both objectives, see [docs/skeptical-statistician-review.md](/Users/benrosen/march-madness-forecaster/docs/skeptical-statistician-review.md).
For the concrete external-data acquisition plan for the Kaggle Brier/BSS track, see [docs/kaggle-external-data-roadmap.md](/Users/benrosen/march-madness-forecaster/docs/kaggle-external-data-roadmap.md).

- **v1 scope:** three sections only (Locked Decisions, Dead-End Ledger, Baseline Registry).
- **Update rule:** when a council session or backtest produces a new settled verdict, add one row here with date + source file. Do not edit old rows; append a superseding row and mark the old one `[SUPERSEDED YYYY-MM-DD]`.
- **Source of truth precedence:** `pipeline_freeze.json` > council transcripts > audit `.md` files > code comments.

---

## 1. Locked Decisions

Settled by evidence, council, or freeze. **Do not propose changing these without new evidence.**

### Core configuration
| Decision | Value | Why locked | Source |
|---|---|---|---|
| Random seed | `2026` | Rule 1: locked seed (state machine) | `pipeline_freeze.json:175`, `src/forecaster/state_machine.py:136` |
| Calibration method | `temperature` | Production-chosen; isotonic/Platt not in freeze | `pipeline_freeze.json:22` |
| Feature set | `SIMPLE_FEATURE_SET` (fixed) | 7-feature logistic matches or beats 27-feature ensemble with stacking (BSS ≈ 0 either way) | `src/pipeline/config.py:168`, `README.md:99`, `FEATURE_ENGINEERING_AUDIT.md` § Redundancy Audit |
| `enable_feature_selection` | `false` | Fixed set beat learned selection | `pipeline_freeze.json:57` |
| `enable_stacking` | `false` | Zero BSS gain over plain logistic | `pipeline_freeze.json:75` |
| `enable_gnn` | `false` | Gated out; no runtime path | `pipeline_freeze.json:58` |
| `enable_transformer` | `false` | Gated out; no runtime path | `pipeline_freeze.json:78` |
| `enable_brier_sharpening` | `false` | Prohibited for Kaggle; overfits Brier at cost of realism | `pipeline_freeze.json:49`, `march madness pipeline v2 protocol.md:96` |
| `model_complexity` | `"simple"` | Consolidates the above | `pipeline_freeze.json:153` |
| `num_simulations` | `50000` | Sound per Phase-1 ML workflow audit (2026); value locked in freeze | `pipeline_freeze.json:156` |

### Validation / training split
| Decision | Value | Source |
|---|---|---|
| Dev years | 2016–2019, 2021–2024 (COVID-2020 excluded) | `pipeline_freeze.json:32-40` |
| Holdout | 2025 | `pipeline_freeze.json:113-115` |
| LOYO CV | enabled, rolling window, 5 splits | `pipeline_freeze.json:62,122,194` |
| Holdout enforcement | `YearSplitPolicy.assert_dev_only` raises `HoldoutContaminationError`; wired at the three ensemble fit entry points + every Optuna tuner call. Closes §2 O20. | `src/ml/evaluation/evaluation_integrity.py:75-160`; `src/pipeline/stages/baseline_training/_ensemble.py:530,770`; `src/ml/optimization/hyperparameter_tuning.py:60`; `tests/test_holdout_enforcement.py` |
| Researcher-DoF (tuning provenance) | Frozen inventory of every Optuna search-space bound, tuning budget (`optuna_n_trials=15`, `enable_stacking=False`, `temporal_cv_splits=5`), and temporal-CV policy. Code-level containment = O20 gate (`YearSplitPolicy.assert_dev_artifact_years`) invoked from every `tune()` call. Drift guard = lock test. Pre-2026-04-07 history not reconstructible (big-bang import commit); documented as residual risk. Closes §2 O17. | `artifacts/o17_researcher_dof_audit_2026-04-14.md`; `tests/test_researcher_dof_audit.py`; `src/ml/optimization/hyperparameter_tuning.py:51-63` (enforcement gate); `src/pipeline/config.py:469,486` (OOS-FIX reductions) |
| Production Four Factors source | Torvik trank.php monthly snapshots overlay onto local `ProprietaryTeamMetrics` via `TorVikFFLookup.overlay_metrics`. Local box-score `_four_factors` is a fallback only; tripwire `mean r ≥ 0.85` vs Torvik catches catastrophic regressions (e.g., the 2026-04-13 resolver-collision at mean r 0.45). Same precedent as barthag per §3 row 22; closes §2 O2 / O2a. | `src/data/features/torvik_ff_lookup.py`; overlay call sites `src/pipeline/stages/baseline_training/_orchestrator.py:469`, `sample_loading.py:453`, `src/evaluation/seed_baseline_loyo.py:286`, `src/ml/evaluation/rdof_audit.py:1517,1661`; tests `tests/test_validate_four_factors.py` |

### Kaggle secondary path
| Decision | Value | Source |
|---|---|---|
| Kaggle optimization target | Held-out tournament `Brier` is the optimization metric; `BSS` is a guardrail, not the primary ranking target. | `docs/kaggle-objective-policy.md`; `scripts/admit_kaggle_candidate.py` |
| Default priority-1 Kaggle test window for 2027-facing work | `2023-2025` should be the first slice inspected when judging whether a candidate looks relevant for 2027 predictions. This is a reporting/evaluation priority window, not a replacement for the stricter admission split. | `docs/kaggle-objective-policy.md`; `RUNBOOK_2027.md` |
| Kaggle recency policy | Most recent 5 observed tournaments count about as much as all older observed tournaments combined (`recent_year_count=5`, `recent_total_ratio=1.0`). Older years remain in-fit for shrinkage/stability. | `docs/kaggle-objective-policy.md`; `src/prediction/kaggle_recency.py` |
| Current Kaggle baseline | `torvik_corrected_recent5_conservative` is the current secondary-path baseline. It is the admitted recent-5 conservative torvik-correction candidate: `ridge=5.0`, `max_correction=0.06`, recent-block weighting as above. Submission command: `python scripts/kaggle_torvik_submission.py --year 2026 --mode torvik_corrected_recent5_conservative --sample-submission data/kaggle/SampleSubmissionStage2.csv --output /private/tmp/kaggle_torvik_corrected_recent5_2026.csv` | `artifacts/kaggle_admission_recent5_conservative.json`; `artifacts/kaggle_baseline_recent5_conservative.json`; `scripts/kaggle_torvik_submission.py` |
| Kaggle admission status of current baseline | **PASS** under the strict historical gate. Final holdout mean Brier: incumbent ensemble `0.1376`, admitted candidate `0.1301`. Final-holdout mean BSS: incumbent `+0.0861`, admitted candidate `+0.1358`. `2026` regression is only `+0.0019`, inside the `0.003` cap. | `artifacts/kaggle_admission_recent5_conservative.json`; `scripts/admit_kaggle_candidate.py` |
| Final internal-only nonlinear candidate status | `torvik_isotonic` was tested against `torvik_corrected_recent5_conservative` under the same recent-5 admission gate and **failed**. Shadow mean Brier improved (`0.1686`), but final holdout regressed: incumbent `0.1308`, isotonic candidate `0.1351`, mean improvement `-0.0043`. Treat the internal-data Brier/BSS track as closed pending new external signal. | `configs/kaggle_admission_isotonic_recent5.json`; `artifacts/kaggle_admission_isotonic_recent5.json`; `scripts/admit_kaggle_candidate.py` |
| First market-odds pilot status | The first tournament-closing-line reopen attempt did **not** admit a new baseline. `torvik_closing` improved final holdout mean Brier only trivially (`0.13070` vs incumbent `0.13078`, gain `0.00008`) and failed the `0.002` improvement threshold. A post-hoc `closing_market_blend` search selected `alpha=0.0`; its best final holdout mean Brier was `0.13047`, but that gain (`0.00030`) came from the stronger-shrinkage torvik correction, not from using the market signal directly. The tournament-only market artifact covers `302/315` main-bracket games across `2021-2025` (`96.8%`, `95.2%`, `100%`, `93.7%`, `93.7%` by year). | `configs/kaggle_admission_torvik_closing_recent5.json`; `configs/kaggle_admission_closing_blend_recent5.json`; `artifacts/kaggle_admission_torvik_closing.json`; `artifacts/kaggle_admission_closing_blend.json`; `artifacts/ncaa_tournament_market.json` |

#### Current Brier/BSS Plan

- Optimize the Kaggle path on held-out tournament `Brier`.
- Keep `BSS` as a guardrail to confirm continued edge over the seed baseline.
- Treat `2023-2025` as the default priority-1 test window when asking whether a candidate is relevant for 2027.
- Use strong recent-block weighting: the most recent 5 observed tournaments should count about as much as all older observed tournaments combined.
- Report Kaggle experiment summaries in this order: weighted recent-aware held-out `Brier`, final-holdout `Brier`, `2023-2025` `Brier`, then `BSS` / stability guardrails.
- Treat `torvik_corrected_recent5_conservative` as the current admitted baseline until a new candidate clears the same admission gate.
- Highest-priority future upgrade if work resumes: historical market odds, then a torvik-plus-market correction model.
- The last serious repo-only fallback (`torvik_isotonic`) has already been tried and failed against the admitted baseline on final holdout. Do not reopen the internal-only calibration track without a materially different idea.
- Do not reopen broad blend-complexity work by default.

### Pool strategy
| Decision | Value | Source |
|---|---|---|
| Recommended mode | `f4_first_tv` (supersedes `champ_first_tv` per O26/O27 team-identity scoring) | `COUNCIL_LESSONS.md §1 Pool strategy`; `artifacts/backtest_runs/mc_pool_backtest_20260418_225840.txt` |
| Aggressive alt | `e8_first_tv` (highest P(1st) at 0.29% under team-identity) | `COUNCIL_LESSONS.md §1 Pool strategy` |
| Opponent pool size | N=31 | `mc_pool_backtest_n31_results.txt` |
| Opponent model weights (2026) | 60% ESPN picks / 30% Massey / 10% seed fallback | `POOL_STRATEGY_RECOMMENDATION.md`; `COUNCIL_LESSONS.md` §3 row 25 (2026-04-12c) |
| Pool-MC `n_tournaments` | `5000` (rank-stable at fixed seed; closes §2 O5) | `src/simulation/pool_competition.py:93`; `tests/test_pool_competition.py::TestRankStability`; `COUNCIL_LESSONS.md` §2 O5 |
| P(1st) ranking is calibrated vs actual outcomes | **Scope note:** O6 and its amendments measure ρ between a portfolio bracket's P(1st) estimate and its actual *placement rank* within a simulated opponent field. The opponent field's identity matters, so this row records both scopes. **Original (2026-04-13, shape-encoded, ESPN-national opponents):** mean Spearman ρ = +0.37 across 14 years (2011-2025 ex 2020); t-test p=0.002; 12/14 years positive. **Amendment A (2026-04-17 per §2 O26-G1 rerun, team-identity scoring, same ESPN-national opponent field):** mean ρ = **+0.61** across 15 years (2011-2026 ex 2020), median +0.69, **100% of years positive** (15/15), t=+8.53 p<1e-6 (shape companion t=+3.55 p=0.003, reproduces original baseline within 0.01). The shape-encoded baseline was internally consistent but materially UNDER-stated the real signal under the scoring that actually pays the pool. **Amendment B (2026-04-17 per §2 O26-G1b pool-scoped rerun, team-identity scoring, ACTUAL 30-person pool from pool_hist_results.json):** mean ρ = **+0.42** across 4 years (2023-2026), median +0.66, 75% positive (3/4), not significant (t=+1.14 p=0.338). Per-year: 2023=+0.97, 2024=+0.36, 2025=+0.97, **2026=−0.60** (high-P(1st) brackets placed WORSE than low-P(1st) brackets against the actual pool — consistent with the "system ranked its own winning bracket #11" narrative from 2026). The ESPN-national-opponent calibration is stronger and more consistent than the actual-pool calibration, indicating that opponent-model mis-specification (ESPN-national ≠ your pool) has real directional impact on P(1st) reliability in edge years. Closes §2 O6 (locked under both scopes); partially supersedes the "14% variance / 86% noise" framing cited in §3 row 53. Drift guard (`tests/test_pool_optimizer_calibration.py`) still checks the shape-encoded artifact as a regression anchor. | `artifacts/o6_winner_rank_diagnostic.json` (shape, original); `artifacts/o26_g1_full_14yr_rerun_2026-04-17.json` (team-identity, ESPN-opp scope); `artifacts/o26_g1b_pool_scoped_2026-04-17.json` (team-identity, pool-opp scope); `tests/test_pool_optimizer_calibration.py`; `COUNCIL_LESSONS.md` §2 O6 + §2 O26 |
| Scoring-schedule structure | Per-round points `Dict[str, int]` is a mandatory `PoolEnvironment` field; propagated to `LeverageCalculator.scoring_system` and multiplied into per-round EV (`expected_points += ev` with `pts = scoring_system.get(round, 0)`). Three parametric adapters: `standard` (ESPN 10/20/40/80/160/320, `late_rounds` priority), `flat` (1..6, `balanced`), `upset_bonus` (ESPN + seed-diff multiplier). `AssumptionsManifest` records the schedule with every recommendation. Closes §2 O12. | `src/optimization/pool_optimizer.py:30-78`; `src/optimization/leverage.py:577-644,1618+`; `artifacts/o12_scoring_structure_audit_2026-04-14.md`; `tests/test_pool_scoring_structure.py` |
| Pre-2011 ESPN picks boundary | Public-picks calibration window starts at **2011** (`MIN_PICKS_CALIBRATION_YEAR = 2011`). Pre-2011 archives (2008-2010) are **kept on disk** for single-year historical simulation but **excluded** from any aggregated calibration. Empirical: pre-2011 champions are 3/3 one-seeds (mean 1.00) vs post-2011 1.86; public over-picks seed-1 at S16 by +14.5 points — genuine regime shift, not noise. `SEED_PICK_RATES` already calibrated 2015-2024; `scripts/mc_pool_backtest.py` already starts at 2011. Enforcement: `strict_post_2011` kwarg on `load_historical_public_picks` raises on pre-2011. Closes §2 O19. | `src/data/historical_picks.py:36-58`; `artifacts/o19_pre_2011_picks_audit_2026-04-14.md`; `tests/test_pre_2011_picks_gate.py` |
| Opponent-independence invariant | Real pool brackets are **LESS correlated than IID draws** from the empirical marginals: Stouffer pooled z = **−4.15** across 4 years × 93 brackets (2023-2026 real `pool_hist_results.json` data). Per-year z ∈ [−2.4, −1.8], all non-positive. The 13-year backtest's independence assumption is safe; the council's validity concern is not supported by data. The opponent-model error lives in the *marginals* (5.0 pp mean abs divergence from ESPN-national), not the *correlation structure* — tracked separately as §2 O21 (closed). Closes §2 O4. | `pool_hist_results.json`; `ANALYSIS_O4_OPPONENT_CORRELATION.md`; `scripts/o4_independence_lock.py`; `artifacts/o4_opponent_independence_2026-04-14.json`; `tests/test_opponent_independence_lock.py` |

### Constants (Tier 1, locked with citations)
| Constant | Value | Source |
|---|---|---|
| Four Factors weights | `[0.40, 0.25, 0.20, 0.15]` (Oliver 2004 / Kubatko 2007) | `constant_registry` |
| HCA | `3.75` pts | `constant_registry` |
| VIF threshold | `10.0` (Belsley 1980) | `constant_registry` |
| Pre-calibration clips | `[0.03, 0.97]` | `constant_registry` |
| `mc_noise_std` | `0.16` (Lopez & Matthews 2015) | `constant_registry:498` |
| `mc_regional_correlation` | `0.0` (reduced from 0.25 during OOS fix) | `constant_registry:510` |
| `seed_prior_weight` / `_slope` | `0.10` / `0.175` | `pipeline_freeze.json:461,184` |

### Strategic pivot (locked 2026-04-02)
**Stop optimizing prediction accuracy. Optimize bracket-pool EV against opponents.**
BSS ≈ 0 across every tested model. Further Brier improvement is a dead end. All five council agents agreed.
Source: `COUNCIL_LESSONS.md` §3 row 7 (2026-04-02 22:03); `PROJECT_STATUS.md:6-11`.

### Pipeline freeze fingerprint (v1)
- `config_hash`: `9ccdfeb313ef6f4f`
- `feature_set_hash`: `1097d739360c2cd4`
- `git_commit`: `77a85dcb2f5fc984e1395cf976d32ef401729821`
- Frozen: 2026-03-18T17:21:54 (`pipeline_freeze.json:214,961,963,966`)

---

## 2. Dead-End Ledger

Tried, measured, rejected. **Do not re-propose without new data that invalidates the rejection.**

| # | Idea | Tried | Verdict | Evidence |
|---|------|-------|---------|----------|
| D1 | Complex ML architectures to lift prediction accuracy | LightGBM, XGBoost, SpreadRegressor, 27-feature ensemble + LightGBM stacking | **BSS = 0** vs seed baseline across 17 yrs (2008–2025 ex. 2020). Not tautological: 2026-04-13 O8 analysis found max \|r\| with seed across production features is 0.77 (`adj_off_eff`), median 0.32. `diff_adj_tempo` (0.16) and `diff_opp_to_rate` (0.02) are essentially independent of seed. Features carry signal beyond seed; the issue is sample-size-limited calibration, not feature redundancy. | `PROJECT_STATUS.md:6-11`; `scripts/feature_seed_correlation.py`; `tests/test_feature_seed_collinearity.py`; `COUNCIL_LESSONS.md §2 O8` |
| D2 | GNN model | Built, gated behind flag | No lift; disabled | `pipeline_freeze.json:58`; `PROJECT_STATUS.md` |
| D3 | Transformer model | Built, gated behind flag | No lift; disabled | `pipeline_freeze.json:78`; `PROJECT_STATUS.md` |
| D4 | Learned feature selection (VIF / correlation / importance pruning) | `src/data/features/feature_selection.py` | Fixed 7-feature set beat it | `pipeline_freeze.json:57` |
| D5 | 11 redundant engineered features (`adj_efficiency_margin`, `consistency`, `close_game_record`, …) | Included in 91-dim vector | Algebraically redundant or pure noise (stability ≈ 0.1); removed from `to_vector()` | `FEATURE_ENGINEERING_AUDIT.md` § Redundancy Audit; `feature_engineering.py:38-65` |
| D6 | Pareto-leverage pool optimizer (`opt_seed`, `opt_blend`, `opt_torvik`) | 13-year N=31 backtest | BestRank 62–88 vs seed's 38; P(1st) ≈ 0 in upset years. 4 root causes: myopic greedy, independence approximation, leverage without correlation, catastrophic upset-year failure | `POOL_STRATEGY_RECOMMENDATION.md:11,27-29`; `COUNCIL_LESSONS.md` §3 row 25 (2026-04-12c) |
| D7 | `hedge_tv` mode | 13-yr backtest | Statistically worse than seed on BestRank (p<0.05 Bonferroni); removed from harness 2026-04-12 | `POOL_STRATEGY_RECOMMENDATION.md:11` |
| D8 | Pool-value contrarian strategy (pick top quintile by pool-value score) | Historical backtest | Upset hit rate 19.4% < 23.2% base rate; chalk beat it by 7.1% | `PROJECT_STATUS.md:12-14` |
| D9 | Brier-optimal sharpening in Kaggle submissions | Prototype | Prohibited by protocol; overfits Brier at cost of realism | `march madness pipeline v2 protocol.md:96`; `pipeline_freeze.json:49` |
| D10 | FanDuel / DraftKings scrapers | Built | Replaced by `TheOddsAPIScraper`; deleted | `PROJECT_STATUS.md:76` |
| D11 | Increasing training window past 9 seasons | Data exploration | ~17.6k regular-season + ~63 tourney games/yr caps signal; no BSS lift | `COUNCIL_LESSONS.md` §3 row 6 (2026-04-02 20:52) |
| D12 | Deterministic-argmax bracket construction (`det_champ_tv`, `det_f4_tv`, `det_e8_tv`) for WTA pools | 13-year N=31 MC backtest; team-identity rerun added 2026-04-17 (O26-G2c) | BestRank 9.92 vs stochastic's 1.52; wins only 3/13 years (all chalky spikes 2015/2019 + one tie); P(1st)=0.000 in 7 of 13 years. Aggregate P(1st) 0.060 looks higher than stochastic 0.041 but is bimodal — requires pre-knowing chalky-vs-upset regime. Kelly framing (concave utility) favors stochastic consistency. Closes §2 O13. **Encoding-invariant (2026-04-17 per O26-G2c):** narrow-scope rerun of `champ_first_tv` vs `det_champ_tv` under team-identity scoring across 14 yrs (2011-2025 ex 2020) strengthens the verdict — stochastic BestRank 1.79 vs argmax 7.44; stochastic wins 14/14 years under team-identity vs shape's 10/13. Argmax P(1st) still marginally higher on aggregate (0.035 vs 0.034) but bimodality pattern unchanged (2019 carries argmax at P(1st)=0.39 vs stoch 0.08; 9 of 14 years argmax P(1st)=0). | `artifacts/o13_kelly_vs_argmax_audit_2026-04-14.md`; `artifacts/o13_kelly_vs_argmax_2026-04-14.json`; `tests/test_kelly_vs_argmax_lock.py`; `mc_pool_backtest_n31_det_vs_stoch.txt`; `artifacts/o26_g2_d12_team_identity_2026-04-17.json`; `scripts/o26_g2_d12_team_identity.py`; `tests/test_o26_g2_lock.py` |
| D13 | Upset-over-estimation (T>1 temperature scaling) for pool-rank lift | 13-year sweep T∈{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0} × 200 stochastic brackets | **Falsified.** P(beats chalk) monotonically DECREASES in T: 0.236 at T=0.5 → 0.224 at T=1 (honest) → 0.194 at T=3.0. Pool-rank-optimal T is sharper (T=0.5), not flatter. Brier-optimal T=1.0 as expected. Calibration-to-pool interaction exists (T=1 is not best-score-optimal) but the gradient points OPPOSITE the gate's prediction and magnitude is trivial (≤5 ESPN points best-score, ≤2pp pBeatChalk). Keep T=1 honest. Closes §2 O14. | `scripts/o14_calibration_sweep.py`; `artifacts/o14_calibration_sweep_audit_2026-04-14.md`; `artifacts/o14_calibration_sweep_2026-04-14.json`; `tests/test_calibration_pool_interaction_lock.py` |
| D14 | Theoretical-copula opponent-dependence model (Gaussian / Clayton / Frank copula fit to observed bracket correlations) | — (not attempted; superseded) | **Not needed.** O4 data shows pool brackets are *less* correlated than IID from the marginals (pooled z = −4.15). A copula solves the opposite problem — modelling *excess* dependence the marginals miss. Applying one here would be regularization against a non-existent clustering signal. **Conditional dead-end:** if the O4 lock test ever fires (pooled z ≥ 0, or any single year z ≥ +1.96), reopen this row in tandem with O4. Until then, committed opponent-model shape is "independence + empirical marginals." Closes §2 O10. | `COUNCIL_LESSONS.md §2 O10`; `ANALYSIS_O4_OPPONENT_CORRELATION.md`; `tests/test_opponent_independence_lock.py` (the conditional-reactivate trigger) |
| D15 | Vegas-line drop-in for calibration (swap temperature scaling → market-implied probabilities) | — (blocked; data not in repo, scraper deleted per D10) | **Blocked + low-ROI.** No historical Vegas lines anywhere in `data/raw/` (verified 2026-04-14); free re-acquisition paths exhausted (TheOddsAPI historical paywalled, Covers/SBRO archives bot-blocked, Kaggle ships outcomes-only). `market_validation.py` consumer stub is present and would accept input if any existed. ROI argument: even with data, literature consensus puts Vegas ≈ best models at ±0.5-2% BSS on NCAA tournament — second-order relative to §3 row 7 pivot (accuracy → pool EV), O7 closure (simple_7 = +4.8% BSS), and O14 closure (calibration-to-pool interaction <5 ESPN pts / <2 pp pBeatChalk). **Conditional dead-end:** reopen if (1) `data/raw/vegas_lines_{year}.json` (or equivalent) lands, OR (2) a future §2 item reopens where the accuracy ceiling becomes the binding pool-EV constraint (would flip §3 row 7). Until either fires, in-house temperature scaling (`TemperatureScaling` at `src/ml/calibration/calibration.py:384`) is the committed calibration method. Closes §2 O15. | `COUNCIL_LESSONS.md §2 O15`; `src/governance/market_validation.py` (awaiting consumer); `src/ml/calibration/calibration.py:384` (committed alternative) |
| D16 | Empirical chalk-bias champion fading (`champ_first_chalkfade_tv`) — O23 | 13-yr N=1000 backtest (2026-04-15) | **P(1st) collapses 14×** vs production baseline: 0.04% vs 0.56% for `champ_first_tv`. BestRnk worse (19.5 vs 18.8). MeanRnk slightly better (490.9 vs 505.2) but MeanRnk is the wrong metric for winner-take-all pools. Root cause: dividing champion draw weights by the empirical chalk ratio (seed-1/CHAMP ≈ 1.22×) modestly depresses selection of the actual champion — trading tail upside (P(1st)) for average-bracket diversity. **Secondary finding**: the static `_chalk_multiplier` anchor (2.00×) overestimates seed-1/F4 chalk by ~60%; true per-team empirical ratio is 1.26×. G1 artifact (`artifacts/chalk_bias_table_2026-04-15.json`) and `load_chalk_bias_table()` remain correct and reusable. Do not re-attempt champion-weight fading without a mechanism to improve P(1st) simultaneously. Closes §2 O23. | `COUNCIL_LESSONS.md §2 O23`; `artifacts/backtest_runs/mc_pool_backtest_20260415_040656.txt`; `artifacts/chalk_bias_table_2026-04-15.json` |
| D17 | Empirical-supremum objective `P(b_i > max(actual pool opponents))` for within-portfolio bracket selection — O25 | 4-yr (2023-2026) × 50 `champ_first_tv` torvik candidates × 5000 MC × actual pool corpus, run under **both** shape-encoded and team-identity (real ESPN) scoring | **Both rankers are noise against actual-outcome scores, under both scorings.** Shape-encoded: mean Spearman ρ across 4 years new=+0.0605, old=−0.0500. **Team-identity (the scoring that actually determines pool payouts): new=+0.0086, old=+0.0660** — the old within-portfolio P(rank=1) is marginally BETTER than the proposed supremum under team-identity. G2-A gate required ρ > 0.50; both encodings fail by an order of magnitude. G2-B (actual-#1 flips into top-5): 0 of 4 years under either encoding. G2-C (O4 propagation sanity, shape only): fails 2 of 4 years. **Coverage is not the bottleneck** — under team-identity scoring, the 50-candidate portfolio contains a bracket that beats the stored pool winner pts in ALL 4 years (2023 1220>720, 2024 1400>1380, 2025 1660>1600, 2026 1520>1350). **Secondary finding (2026-04-16):** the original G1 preflight narrative in COUNCIL_LESSONS row 229 (spreads 0.835/0.315/0.045/0.310) was undocumented and unreproducible — no script, no artifact, inconsistent with the pre-existing `rank_correlation_diagnostic.json`. Real G1 re-run lands all 12 spread measurements (3 rankers × 4 years) in [0.016, 0.047], mostly AMBIGUOUS (1%-3% zone). G2 was triggered on a preflight that didn't measurably support the trigger. The question was tested fresh under both scorings and closes as dead-end. Closes §2 O25. Conditional reactivate: a materially different candidate-generation strategy (e.g., tail-diversity-biased rather than champ_first_tv) produces ρ that clears 0.50 under team-identity scoring. | `artifacts/o25_g1_real_2026-04-16.json`; `artifacts/o25_g2_empirical_supremum_2026-04-16.json` (shape); `artifacts/o25_g2_team_identity_2026-04-16.json`; `artifacts/o25_g3_diversity_2026-04-16.json`; `scripts/o25_g1_spread_diagnostic.py`; `scripts/o25_g2_empirical_supremum.py`; `scripts/o25_g2_team_identity.py`; `tests/test_o25_g1_g2_lock.py` |
| D18 | Three `meta_region_poolaware` variants proposed after the 2026 UI investigation found (a) forced-champion candidates are 1-seed-only and (b) the real 2026 Final Four (Michigan/Arizona/UConn/Illinois) was unreachable by any tested strategy: `meta_region_poolaware_wideseed` (force seeds 1-4 as candidate champions, not just 1-seeds), `meta_ensemble_agree` (pool-sim selection between the production `meta_region` and `meta_exhaustive` brackets), `meta_region_poolaware_minimax` (poolaware's candidate pool, scored by worst-case P(1st) across 3 opponent-field assumptions instead of one) | 15-yr LOYO (2011-2026 ex 2020), team-identity scoring, N=29 real pool history, run alongside `meta_region_poolaware` as a paired baseline (same opponent draws per repeat) | **All three fail the production acceptance gate (P(1st) improvement in ≥8/14-15 years).** `wideseed`: 0/15 — selected the *identical* candidate to baseline in every single year (verified via per-year `selected=` labels, not just aggregate stats); the 2-4 seed forced-champion candidates never won selection once. `meta_ensemble_agree`: 5/15 wins (aggregate P(1st) 7.60% vs baseline 11.33%) — capped by its own inputs, since it can only choose between `meta_region`/`meta_exhaustive`, which individually run 7-8% P(1st) alone. `meta_region_poolaware_minimax`: 5/15 wins, 4 ties (aggregate P(1st) 9.60% vs baseline 11.33%) — the one with a real tradeoff: better MeanRank (10.5 vs 11.1) and more ties/near-misses than the other two, but MeanRank isn't the metric that pays out (see this file's own **Do NOT optimize MeanRank** framing) and it still fails the win-count gate. None promoted; none merged into the UI. Code removed 2026-08-13 (kept in git history on `claude/bracket-strategy-models-h1g2hf` if any variant is worth revisiting later). | `artifacts/backtest_runs/mc_pool_backtest_20260813_033903.txt` |
| D19 | `meta_region_poolaware_roster` / `meta_region_poolaware_coach` — swap the production construction's primary probability base from plain torvik to roster_adj- / coach_adj-adjusted torvik round_probs (same candidate generation + pool-aware selection as `meta_region_poolaware`, only the "tv" slot and forced-champion candidates change). Follow-up to the informational-only Team Factors UI panel (2026-08-14): tests whether these small per-team nudges help when fed into the *validated* construction, rather than a naive one. | 15-yr LOYO (2011-2026 ex 2020), team-identity scoring, N=29-30 real pool history, run alongside `meta_region_poolaware` and `seed` as a paired baseline (same opponent draws per repeat) | **Both fail the production acceptance gate — 0/15 wins each.** `roster`: 0 wins, 11 ties, **4 losses** (aggregate P(1st) 10.40% vs baseline 11.20%) — the larger, more volatile WARP-based deltas occasionally flip which candidate the pool-aware selector picks, and the flip is negative more often than positive. `coach`: 0 wins, 14 ties, 1 loss (aggregate P(1st) 11.07% vs baseline 11.20%) — essentially statistical noise; the ±0-3% one-sided coach bump is too small to change candidate selection in all but one year. Consistent with the precedent this row's sibling (Architectural Direction §, GBM-probability swap) already established: swapping a *less accurate or barely-different* probability base into the winning construction does not help, and can hurt if the swap is large enough to perturb candidate ranking. Confirms the earlier `torvik+roster_adj`/`torvik+coach_adj` composable-pipeline result (statistically flat, 4.13–4.38% vs 4.23%, `artifacts/experiments/experiment_custom_20260813_222650.json`) generalizes to the production construction too — this was not a construction-quality artifact of the simpler `forward` mode. Not merged into the UI or promoted. Code removed 2026-08-14 (kept in git history on `claude/bracket-strategy-models-h1g2hf` if worth revisiting with a materially different roster/coach signal). | `artifacts/backtest_runs/mc_pool_backtest_20260814_181433.txt` |

---

## 3. Baseline Registry

Current numbers. If you're about to claim an improvement, it has to clear these.

### ML prediction (frozen — do not chase)
| Metric | Value | Scope | Source |
|---|---|---|---|
| Brier Skill Score | **0** | 17 yrs (2008–2025 ex. 2020); all tested blended+stacked models (production regime) | `PROJECT_STATUS.md:6-11`; `MEMORY.md §2 D1` |
| BSS, tournament-only LOYO, simple_7 | **+4.8%** (paired t p=0.036 uncorrected; fails Bonferroni) | 14 yrs (2010–2024 ex. 2020); feature set = `SIMPLE_FEATURE_SET` | `artifacts/baseline_experiment.json` (2026-04-01); `artifacts/o7_regime_comparison_2026-04-13.json`; `tests/test_o7_regime_comparison.py`; closes §2 O7 |
| LogLoss gate | < 0.56 | Training objective | `pipeline_freeze.json:121` |
| Brier gate | 0.19 | Admission threshold | `pipeline_freeze.json:14` |

### Pool backtest — production strategy validation (15 yrs 2011–2026 ex. 2020, N=31 pool: 30 opponents + model bracket)

**Canonical, reproducible baseline — supersedes the ambiguous 11.9%/14-yr figure previously in CLAUDE.md.**
Run with the "non-negotiable" contract from `STRATEGY_CATALOG.md` §Evaluation Contract:
`--team-identity --opponent pool --n-opponents 30 --n-repeats 100`.

| Mode | P(1st) | BestRank | MeanRank | MeanScr | Wins vs seed |
|---|---|---|---|---|---|
| **meta_region_poolaware** (production) | **11.3%** | 11.6 | 11.6 | 849 | **15/15** |
| meta_region | 6.3% | 12.9 | 12.9 | 903 | 8/15 |
| meta_exhaustive | 6.2% | 13.1 | 13.1 | 889 | 8/15 |
| seed (baseline) | 4.9% | 11.8 | 17.1 | 698 | — |

MeanRank seed vs meta_region_poolaware: t=9.219, p<0.0001 (Bonferroni-corrected across 3 comparisons).
Source: `artifacts/backtest_runs/mc_pool_backtest_20260816_012851.txt` (2026-08-16).

**Reconciles/supersedes:**
- CLAUDE.md's "11.9% P(1st), 14/14 years, p=0.008" — that figure predates 2026 being added as a backtest
  year and used a different `--modes` set; 11.3%/15-15 is the direct, reproducible update.
- §2 D18/D19's "11.33%"/"11.20%" paired-baseline mentions — those numbers are corroborated by this run
  (11.3%, same ballpark) but were logged as side context for other experiments, not as a canonical citation.
- CLAUDE.md's "seed baseline 3.1%" — unsourced in this repo; the paired, same-methodology measurement here
  is 4.9%. Treat 4.9% as current until a source for 3.1% turns up.
- CLAUDE.md's "meta_region 8.0%, 11/14 years" and "meta_exhaustive 7.7%, 11/14 years" — under the
  corrected `--n-opponents 30` contract (an earlier attempt at this rerun omitted that flag and silently
  fell back to a 999-opponent pool for the 11 years without real pool-history data, producing a bogus
  ~4% aggregate — see artifact `mc_pool_backtest_20260815_222620.txt` for the failed attempt), both land
  at 6.2-6.3% and 8/15 — right at the `>=8/N` acceptance-gate boundary, down from the previously-cited
  11/14 (79%) win rate. Real erosion, not a measurement artifact: consistent with 2026 being a hard upset
  year (D18) landing in the sample for the first time.
- `scripts/mc_pool_backtest.py`'s `main()` had a bug where `sys.exit(main())` received the raw results
  list from `run_backtest()`, which Python's `sys.exit()` treats as an error message + exit code 1 even on
  a fully successful run. Fixed 2026-08-16 (`main()` no longer returns the list) — prior "failed" exit
  codes on this script are not evidence of an actual failure; check the printed aggregate table instead.

### Leakage confidence for the 15-year production validation (2026-08-16)

Prompted by a direct question ("are we 100% sure there's no leakage") after the 15-year rerun above.
Audited every probability base that actually appears in `meta_region_poolaware`'s selected-candidate log
across the 15 years (`tv`/Torvik, `blend`, `mass_avg`, `mass_best`, `tv_mass80`):

- **Torvik** — strongest evidence. Every historical file carries `data_type`/`cutoff_date` metadata,
  enforced by `_validate_pretournament()`; a dedicated corrective rescrape (`rescrape_pretournament_torvik.py`)
  exists specifically because an earlier acquisition method was found to leak end-of-season data.
- **Elo, market odds** — explicit same-year date-cutoff filtering in code (`elo_probabilities.py`,
  `market_probabilities.py`).
- **Cross-year model selection** (noseed model training, `massey_best`'s walk-forward system pick) —
  `assert all(y < year for y in train_years)` in code, not just documentation.
- **Massey composite/per-system** (`mass_avg`, `mass_best`, `tv_mass80` — together the majority of the raw
  candidate pool by count, and the actually-selected bracket in 4/15 years: 2013, 2017, 2019, 2021) —
  traced to Kaggle's `MMasseyOrdinals.csv` (gitignored, not missing — too large to commit), ingested via
  `kaggle_loader.py`'s `_compute_max_ranking_day()` Selection-Sunday cutoff guard, wired into the real
  `historical_pipeline.py` ingestion (not a one-off script). `docs/data-provenance-and-leakage-audit.md`
  had stale "Backtest impact: None" rows for both — true when last touched 2026-04-04, false since
  `massey_avg`/`massey_best` went live in the candidate pool 2026-04-24/25. Fixed 2026-08-16.
- **Empirical cross-check**: if Massey's composite were contaminated with tournament results, teams it
  rates more favorably than Torvik should systematically be the teams that actually won more games that
  year. Tested across 6 years (all 4 Massey-selected years + 2 controls): correlation between
  (Massey-rank-favor-over-Torvik) and (actual rounds won) is statistically indistinguishable from zero
  (ρ = −0.22 to +0.03, none significant at p<0.05) — methodology validated by the same test on Torvik
  alone, which correctly shows strong significant correlation (ρ = 0.52–0.79, p<0.0001) in every year.

**Verdict: high confidence, not proof.** Genuinely open gaps: no unit test directly exercises the Massey
day-cutoff exclusion (code-verified by reading, not test-verified); the raw `MMasseyOrdinals.csv` itself
can't be inspected (gitignored); the opponent-field data sources (ESPN picks, pool history) weren't put
through the same scrutiny (a different risk — simulated-competition realism, not model foreknowledge);
the pool-aware MC selection code wasn't line-by-line audited beyond the date-cutoff question. Don't cite
this as "leakage-free" — cite it as "the specific mechanisms checked came back clean, here's what wasn't
checked."

### Pool backtest — team-identity scoring (14 yrs 2011–2026 ex. 2012/2020, N=1000, 50 brackets × 50 repeats)

| Mode | BestRank ↓ | MeanRank ↓ | P(1st) ↑ | P(top 5%) ↑ |
|---|---|---|---|---|
| **torvik** | **27.1** | 583.5 | 0.07% | **7.35%** |
| **f4_first_tv** *(recommended)* | 33.8 | **573.6** | 0.16% | 6.60% |
| e8_first_tv *(aggressive)* | 35.5 | 601.9 | **0.29%** | 4.85% |
| champ_first_tv *(prior rec)* | 36.5 | 583.7 | 0.20% | 5.39% |
| seed *(baseline)* | 47.3 | 605.1 | 0.21% | 3.89% |

Source: `artifacts/backtest_runs/mc_pool_backtest_20260418_225840.txt` (O27 closure, 2026-04-18). Team-identity scoring per O26.
**Statistical power:** N=14 yrs, ~9–16% power. No stochastic mode reaches Bonferroni-significance vs seed on BestRank; direction is consistent.

**Why f4_first_tv over torvik:** torvik has best BestRank but f4_first_tv has best MeanRank, highest P(top25%) (21.4%), and highest MeanScr (737). Retroactive actual-pool analysis across 2023-2026 (§3 row 65) independently confirms: f4_first_tv wins all 4 actual pool years when selecting from the ranker's top-3. Three independent lines of evidence converge on f4_first_tv: (1) retroactive pool scoring, (2) within-portfolio ranker correlation, (3) N=1000 team-identity backtest.

**Superseded shape-encoded table:** Prior numbers (champ_first_tv BestRank 21.1, e8 23.2, f4 26.1) were scored under shape encoding which diverges from real ESPN scoring. See `ANALYSIS_O4_OPPONENT_CORRELATION.md` and O26 for the encoding gap. Shape-encoded numbers preserved in `POOL_STRATEGY_RECOMMENDATION.md` for historical reference.

### 2026 tournament result
- System produced a winning-quality bracket (1440 pts, 4/4 Final Four) but **ranked it #11** in its own portfolio.
- Diagnosis: ranking failure, not prediction failure. Opponent-model independence assumption under scrutiny.
- Source: `COUNCIL_LESSONS.md` §3 row 25 (2026-04-12c).

### Data / test coverage
| Metric | Value | Source |
|---|---|---|
| Training years | 8 (2016–2019, 2021–2024) | `pipeline_freeze.json:32-40` |
| Backtest years | 13 (2011–2025 ex. 2012) | `POOL_STRATEGY_RECOMMENDATION.md:3` |
| Test functions | ~5,931 across ~211 files | `tests/` grep |
| Coverage threshold | 20% | `COUNCIL_LESSONS.md` §3 row 6 (2026-04-02 20:52) |

### Kaggle checkpoint (secondary path)
| Metric | Value | Scope | Source |
|---|---|---|---|
| Current admitted Kaggle mode | `torvik_corrected_recent5_conservative` | CLI preset for secondary-path submissions | `scripts/kaggle_torvik_submission.py` |
| Final-holdout mean Brier | **0.1301** | Strict gate final years vs ensemble incumbent `0.1376` | `artifacts/kaggle_admission_recent5_conservative.json` |
| Final-holdout mean BSS | **+0.1358** | Strict gate final years vs ensemble incumbent `+0.0861` | `artifacts/kaggle_admission_recent5_conservative.json` |
| Holdout mean Brier | **0.1562** | Shadow + final holdout block vs ensemble incumbent `0.1623` | `artifacts/kaggle_admission_recent5_conservative.json` |
| Holdout mean BSS | **+0.1052** | Shadow + final holdout block vs ensemble incumbent `+0.0701` | `artifacts/kaggle_admission_recent5_conservative.json` |
| Historical walk-forward mean Brier | **0.1572** | 18-year torvik-correction backtest under recent-5 conservative preset | `scripts/kaggle_torvik_submission.py --multi-year-backtest`; validated 2026-05-10 |
| Historical walk-forward BSS | **+0.1680** | Same 18-year preset backtest | `scripts/kaggle_torvik_submission.py --multi-year-backtest`; validated 2026-05-10 |

### Known open diagnostic (for context — not a TODO)
- Independence assumption in opponent model has been empirically tested (2026-04-13, 4 years × 93 brackets). **Independence holds** — pooled z = −4.15; brackets are *less* correlated than IID draws from the empirical marginals. The council's "validity threat" framing was misdiagnosed: error is in the opponent-model marginals (using ESPN-national instead of pool-specific; 5pp mean absolute divergence, up to 18pp on individual teams), not in correlation. **Pool-history marginal blending tested and rejected** — O21 (2026-04-13) found blending pool-specific marginals into the opponent model does NOT change bracket rankings (Spearman ρ invariant across blend weights 0–1.0 on n=2 usable years; Δρ=0 for 2024, Δρ=−0.07 for 2025). O26-G2a (2026-04-17) retested under team-identity scoring and found modest signal (argmax-best weight=0.25, Δρ=+0.058 vs ESPN-only) but insufficient to justify a production change on n=3 years. Keep locked weights at 60/30/10 ESPN/Massey/seed. Sources: `ANALYSIS_O4_OPPONENT_CORRELATION.md`, `ANALYSIS_O21_MARGINAL_BLEND.md`, `COUNCIL_LESSONS.md §2 O4 [closed] / O21 [closed — negative verdict]`.
- Council 64 action #3 (irreducible noise floor) closed 2026-04-25 — opposite-side companion to council 71's selection-noise finding. Across 4 years (2023-2026) × 3 modes (torvik / f4_first_tv / e8_first_tv), the gap between the MC ranker's pick and the oracle best-of-50 (perfect outcome knowledge) is **mean 8.08 rank positions** when inserted into the actual pool field. Council 64's `<3 → kill opponent-calibration track` gate does NOT trigger, but the n=4-year 95% CI lower bound is ~0.4 (gate boundary), and council 71's D18/D19 already showed every practical selection criterion tested captures ~none of this 8-position theoretical ceiling. Net: the formal kill does not fire, but no method has demonstrated ability to convert opponent-model improvements into rank gains — council 67's "lock + 1-2 week focused investigation, then wait" remains operative. Sources: `scripts/noise_floor_ceiling.py`, `artifacts/noise_floor_ceiling_2026-04-25.json`, `COUNCIL_LESSONS.md §3 row 72`. **Regime applicability (added 2026-04-27):** The 8.08-rank gap is a stochastic-regime measurement — it bounds the gap between an MC ranker's pick and the best of its own 50-bracket portfolio. After the meta-learner pivot (CLAUDE.md §"Architectural Direction"), the primary path emits ONE deterministic bracket per model with no within-strategy selection step. This ceiling does NOT apply to the deterministic meta-learner regime; the deterministic-regime equivalent is unmeasured. Do not cite this row to bound expected effect size of a successor to meta_gbm v2.

### Future exploration topics (only if work resumes)
- More conservative Kaggle correction sweeps are still fair game, but only if they beat the current admitted baseline on the same strict admission gate. Default search neighborhood: stronger shrinkage and smaller `max_correction`, not more complexity.
- A simpler torvik-correction family is still worth testing if resumed later: fewer features, same recent-5 policy, same Brier-first gate.
- Women’s Kaggle path likely has more headroom than the men’s path. If secondary-path work resumes, women’s-specific admission gating is a high-upside direction.
- Do not reopen broad blend-complexity work by default. Global alpha tuning and multi-bucket blending already looked close to exhausted.

---

## Index of source material

- Council lessons + open questions: `COUNCIL_LESSONS.md` (consolidated 2026-04-13; raw transcripts deleted). New sessions append to §3.
- Audits: `AUDIT_DATA_LEAKAGE.md`, `AUDIT_DATA_SCRAPERS.md`, `FEATURE_ENGINEERING_AUDIT.md` (Phase-1 ML workflow audit archived 2026-04-13: explicitly marked HISTORICAL post-pivot; `num_simulations=50000` rationale now lives in freeze + this file only)
- Status: `PROJECT_STATUS.md`, `WORKFLOW.md`
- Data sources (ingested, stubs, gaps vs. successful bracket models): `WORKFLOW.md` Appendix A. Consult before proposing a new data source — EvanMiya / Haslametrics / ESPN BPI / Hoop-Math / returning-production / KenPom-direct / ShotQuality / tournament-futures already scoped there.
- Strategy: `POOL_STRATEGY_RECOMMENDATION.md`
- Freeze: `pipeline_freeze.json`
- Backtest artifacts: `mc_pool_backtest_*.txt`, `pool_report*.json`

---

## Recent Commit Note

Recent commits:

- `5f81e61` added the Odds API archive audit and closing-consensus artifact.
- `486eb15` added the market-odds feasibility checklist.
- `30c78ed` closed the internal Kaggle Brier track and added the external-data roadmap.
- `991b7a2` added the women's pipeline LOYO Brier backtest baseline.
- `6ae5ac3` closed the isotonic calibrator exploration as a non-improvement.

Outstanding next steps:

- Decide whether the market-odds artifacts stay as standalone research outputs or get wired into the Kaggle admission flow.
- If Kaggle secondary-path work resumes, test only materially different candidates against the current recent-5 conservative baseline.
- Keep the broad blend-complexity path closed unless new external signal changes the tradeoff.
