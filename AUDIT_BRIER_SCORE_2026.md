# March Madness Forecaster: Top 1% Brier Score Audit

**Auditor:** Senior Kaggle Grandmaster-level ML Auditor (Opus 4.6)
**Date:** 2026-03-02
**Codebase:** march-madness-forecaster (commit `42027e7`)
**Objective Metric:** Brier Score (round-weighted, Kaggle 2023-2026)
**Prospective Eval Brier:** 0.212 on 427 held-out 2024 tournament games

---

## 1) Executive Summary

### Estimated Performance Tier: **Strong (Top 3-5%)**

This is a mature, well-engineered codebase with production-grade architecture
that reflects genuine understanding of tournament prediction challenges.
It is **not yet Elite (Top 1%)**, but is closer than the ~Top 5% tier
assigned by prior audits suggests. The prospective evaluation Brier score
of 0.212 on 427 games (BSS = 0.143 vs uniform) is competitive — the
gap to Top 1% (~0.195-0.200 Brier on combined men's/women's) is approximately
0.012-0.017 Brier points.

### Key Strengths

1. **Point-in-time feature construction is genuinely correct.**
   `IncrementalMetricsEngine` computes ALL features incrementally from
   box-score data — for each training game at date D, features use only
   games played before D. The old `_load_year_samples()` (season-end
   leaker) is explicitly deprecated with a `NotImplementedError` guard.
   This is better than ~95% of Kaggle submissions.

2. **Validation strategy is exemplary.** Leave-One-Year-Out CV with
   `rolling_window` temporal mode (train only on strictly prior years),
   COVID 2020 exclusion, dev/holdout year partitioning (2016-2024 dev,
   2025 holdout), `TemporalCrossValidator` for within-year tuning.
   No random K-fold anywhere. Pair-size snapping for symmetric data.

3. **Multi-model ensemble with structural diversity.** SpreadRegressor
   (MOV-primary at 0.40 weight) + LightGBM (0.25) + XGBoost (0.15) +
   Logistic (~0.20) + Bayesian Bradley-Terry (optional). The MOV-primary
   design mirrors the "raddar" approach (dominant 2018-2024). Different
   training targets (margin vs binary vs ID-based) provide genuine
   orthogonal signal.

4. **Brier-optimal post-processing pipeline.** Dedicated `BrierPostProcessor`
   chain: seed overrides -> calibration -> sharpening -> clip. Temperature
   scaling minimizes Brier (not NLL). Round-weighted Brier calibration
   matches Kaggle's actual post-2023 metric. Probability bounds [0.005,
   0.995] are appropriately wide for Brier (quadratic, not log penalty).

5. **Conservative regularization matched to sample size.** `num_leaves=8`,
   `min_child_samples=50`, Optuna limited to 15 trials with narrow search
   bounds, stacking disabled, learned feature selection disabled, fixed
   domain-knowledge feature set with literature citations. These choices
   correctly prioritize generalization over in-sample fit for N~600.

6. **Women's tournament handled separately.** Dedicated pipeline with
   different calibration, stronger seed priors (50% vs 10%), and simpler
   model complexity. Critical since women's bracket is 50% of Kaggle
   score since 2023.

### Top Risks

1. **External ratings operationalization gap.** Massey Ordinals composite
   code exists (`ExternalRatingsLoader`, `MasseyStandalonePredictor`) and
   is partially wired into training via `_load_year_samples_incremental()`,
   but actual coverage is unclear. The `massey_blend_weight=0.25` in config
   suggests intent to blend Massey, but whether this signal reliably
   reaches the final prediction depends on data availability at runtime.
   Prior audits flagged this as "always 0.0" — but recent code changes
   (`external_massey_composite_{year}.json` loading) suggest partial fixes.

2. **Calibration/sharpening sequential fitting risk.** The Brier sharpener,
   temperature calibrator, and seed overrides are all fitted on validation
   data that may overlap, creating a multi-stage double-dipping risk with
   ~400 tournament game samples.

3. **Historical data quality degrades training.** Conference field absent
   (all conference features NaN), fake dates in 2005-2024 (break recency
   weighting), team ID namespace mismatches between data sources. The
   `DATA_QUALITY_ERA_WEIGHTS` downweights 2005-2014 which is the right
   mitigation, but data quality issues in 2015-2024 remain.

4. **DoF ratio above target.** The prospective evaluation reports 14
   freely-tuned constants / 427 holdout games = 0.033 (target < 0.01).
   This is 3.3x the target, indicating the pipeline has more researcher
   degrees of freedom than ideal for the available sample size.

### Overall Stability Assessment: **Moderate-High**

The ML layer (models, ensemble, calibration, CV) is robust. The data layer
(team ID resolution, external ratings integration, historical data quality)
is the primary fragility source. Pipeline is deterministic with fixed seeds
and versioned configs. End-to-end reproducibility is achievable but requires
correct data setup.

---

## 2) Scorecard

| Category | Status | Impact | Notes |
|---|---|---|---|
| **A. Validation Strategy** | **PASS** | High | LOYO with `rolling_window` temporal mode. Train on strictly prior years only. COVID 2020 excluded. Dev/holdout split (2016-2024 / 2025). `TemporalCrossValidator` for within-year Optuna tuning. No random K-fold. Expanding-window splits with chronological ordering. |
| **B. Data Leakage** | **PASS** | High | `IncrementalMetricsEngine` provides true PIT features for every training game. Old `_load_year_samples()` deprecated with `NotImplementedError`. Seed leakage fix: seeds zeroed for pre-March-14 games. Tournament games excluded from baseline training. Late-season cutoff (45 days). Elo cross-season carryover via `prior_elo`. |
| **C. Feature Coverage** | **PARTIAL** | High | 22+ active features in `FIXED_FEATURE_SET` covering efficiency, Four Factors, SOS, Elo, shooting, experience, seed. External ratings (`diff_external_rating_composite`) present in feature set but coverage depends on runtime data. Conference features still NaN. Player metrics 0.0 in historical. |
| **D. Pairwise Construction** | **PASS** | Medium | Correct `TeamA - TeamB` differentials. 5 absolute-level features (game quality context). Interaction features (seed_interaction, seed_diff). `IncrementalMetricsEngine.build_matchup_vector()` constructs pairwise features properly. No raw concatenated team features. |
| **E. Modeling** | **PASS** | Medium | 5 model types: Logistic (L2), LightGBM (`num_leaves=8`), XGBoost (`max_depth=3`), SpreadRegressor (LightGBM regression on margins -> logistic CDF), Bayesian Bradley-Terry (MAP + Laplace). Conservative hyperparameters. Optuna with 15 trials. Fixed seeds. |
| **F. Ensembling** | **PASS** | High | 4-model fixed-weight ensemble: Spread=0.40, LGB=0.25, XGB=0.15, Logit~0.20. Stacking correctly disabled (would overfit on 400 OOF samples). `EnsembleWeightOptimizer` with L2 regularization toward uniform and min-50-sample guard. Statistical significance guard on weight optimization. |
| **G. Probability Quality** | **PASS** | High | Temperature scaling (1 param, default for small data), with auto-downgrade guardrails. `BrierOptimalSharpener` with power transform. `BrierCalibrator` minimizing Brier (not NLL). `RoundWeightedSharpener` for Kaggle's actual metric. Small-sample bootstrap CI guard. Clipping [0.005, 0.995]. |
| **H. Seed Prior Integration** | **PASS** | Medium | Men's historical rates (8 matchup types, 1v16: 0.987 to 8v9: 0.510). Women's separate rates (more predictable). `SeedBasedOverrides` with snap threshold 0.08. Seed prior blend (10% men's, 50% women's). Both `seed_diff` and `seed_interaction` in features. |
| **I. External Data Integration** | **PARTIAL** | High | Massey Ordinals loading implemented in `_load_year_samples_incremental()` with Kaggle CSV fallback. `MasseyStandalonePredictor` calibrates sigma on historical Brier. `massey_blend_weight=0.25` blend in config. Torvik Four Factors and shooting splits loaded for all years. Coverage verification needed. |
| **J. Advanced Features** | **PARTIAL** | Low | Travel distance (Haversine, 0.0 in historical). Experience/roster continuity (0.0 in historical, populated current-year). Recent form/momentum computed by `IncrementalMetricsEngine`. Conference strength still NaN. Game flow entropy present. Three-point variance for upset prediction. |
| **K. Leaderboard Robustness** | **PASS** | Medium | Fixed domain-knowledge feature set (literature-cited, not fitted to LB). Dev/holdout year split. Ensemble weight L2 regularization. `rdof_audit.py` with freeze file requirement and explicit constant inventory. No submission-count optimization. No manual override tables. |
| **L. Pipeline Quality** | **PARTIAL** | Medium | Deterministic seeds (numpy, random, torch). `SOTAPipelineConfig` dataclass. Git-versioned. **Gaps:** (1) Team ID resolution chain has 4+ `_normalize_team_id` functions across modules. (2) HTML entity encoding bug (`Texas A&M`). (3) No automated end-to-end single-command execution with CV reporting. |
| **M. Submission Integrity** | **PASS** | Low | `generate_predictions()` handles men's (ID < 3000) and women's (ID >= 3000) routing. Probabilities clipped to [0, 1]. Unmapped teams default to 0.5. ID format validated via regex. `KaggleExportStats` tracks mapping quality. |

---

## 3) Top 5 Improvement Opportunities

### 1. End-to-End Massey Ordinals Coverage Verification

**Description:** Verify that `diff_external_rating_composite` is non-zero
for all 68 tournament teams in both training AND inference. Prior audits
report this feature is "always 0.0" in practice. The code to load Massey
data exists in `_load_year_samples_incremental()`, but it requires either
cached `external_massey_composite_{year}.json` files or a Kaggle directory
with `MMasseyOrdinals.csv`. If these files are absent, the most important
single feature silently degrades to zero.

**Why it matters:** The Massey composite (average of 100+ rating systems) is
the single highest-signal feature in the competition. Every Kaggle March
Madness winner from 2017-2025 used meta-rankings as a core signal. The
`massey_blend_weight=0.25` config value allocates 25% of the final prediction
to Massey, but only if the data is actually present.

**Estimated Brier improvement:** -0.008 to -0.015
**Implementation difficulty:** Low (verify data pipeline, generate cache files)

### 2. Nested/Hold-Out Calibration to Prevent Post-Processing Double-Dipping

**Description:** The Brier post-processing chain (seed overrides -> BrierCalibrator
-> BrierOptimalSharpener -> clip) sequentially fits 3+ parameters on the same
validation data. With ~400 tournament game samples, this creates overfitting risk
in the post-processing chain that can degrade out-of-sample Brier by
more than it helps.

**Solution:** Implement a 3-way split: (1) training data for models,
(2) calibration data for temperature/sharpening, (3) evaluation data for
metrics. Alternatively, use nested LOYO CV where each outer fold's
calibration is fitted on inner-fold OOF predictions.

**Why it matters:** Brier score rewards calibration. If calibration itself
is overfit, the calibrated predictions are worse than uncalibrated on new
data. The `TemperatureScaling` small-sample guard (bootstrap CI check)
partially mitigates this but doesn't address the sequential fitting issue.

**Estimated Brier improvement:** -0.002 to -0.005
**Implementation difficulty:** Medium

### 3. Conference Strength Feature Materialization

**Description:** The `conference` field is absent from all years' data
(prior audits confirm all NaN). This means `diff_conference_strength`,
`conference_sos`, and related features are all zero. Conference strength
is a significant predictor — weak-conference teams are systematically
overrated by efficiency metrics because their opponents are weaker.

**Solution:** Wire the existing Kaggle `MTeamConferences.csv` loader
(`KaggleDataLoader.load_team_conferences()`) into the main pipeline for
all years. The code already exists (FIX #4 in `_load_year_samples_incremental`)
but needs verification that it actually populates the feature vector.

**Why it matters:** Conference strength corrects for schedule quality
in a way that raw SOS does not. Without it, mid-major teams predicted
to be too strong and power-conference teams too weak.

**Estimated Brier improvement:** -0.002 to -0.004
**Implementation difficulty:** Low

### 4. Cross-Validated Ensemble Weight Optimization

**Description:** The current ensemble uses fixed weights
(Spread=0.40, LGB=0.25, XGB=0.15, Logit=0.20). While this is the
correct conservative choice for avoiding overfitting, the optimal
weights likely vary by year and matchup type. The `EnsembleWeightOptimizer`
exists but is disabled (OOS-FIX comment: "fits weights on eval set").

**Solution:** Implement LOYO-CV ensemble weight optimization where weights
are fitted on out-of-fold predictions from each LOYO fold. This uses
only truly out-of-sample predictions for weight optimization, avoiding
the double-dipping that led to its disabling.

**Why it matters:** Fixed weights assume all models are equally well-
calibrated across all years. In practice, SpreadRegressor may be better
in "chalk" years (few upsets) while Bayesian BT may add value in
"chaos" years (many upsets). Adaptive weights can capture this.

**Estimated Brier improvement:** -0.001 to -0.003
**Implementation difficulty:** Medium

### 5. Historical Data Quality Remediation

**Description:** The training data has several documented quality issues:
(a) 2005-2009 team metrics zeroed out for several features, (b) games
in 2005-2024 have fake/imputed dates, (c) team ID namespace mismatches
requiring fuzzy matching, (d) HTML entity encoding bugs. While
`DATA_QUALITY_ERA_WEIGHTS` downweights early years, 2015-2024 data
still has date and ID resolution issues.

**Solution:** (a) Verify Torvik Four Factors are populated for all years
(the loader exists and caches per-year). (b) Fix date imputation to use
actual game dates from box-score data in `team_games` arrays. (c) Audit
team ID resolution accuracy for all 68 tournament teams in each year.
(d) Fix HTML entity encoding in team names.

**Why it matters:** The multi-year training pool (3000+ games from 10+
years) is a major architectural advantage, but only if the historical
data quality is sufficient. Bad data in training degrades the model's
ability to learn correct feature-outcome relationships.

**Estimated Brier improvement:** -0.002 to -0.005
**Implementation difficulty:** High

---

## 4) Leakage & Risk Report

### Leakage Risks

| Risk | Status | Severity | Notes |
|---|---|---|---|
| Season-end features on early games | **MITIGATED** | Low | `IncrementalMetricsEngine` computes true PIT features. Old leaky loader deprecated. |
| Tournament data in training features | **MITIGATED** | Low | Tournament games excluded from baseline training. Seed leakage fix zeroes seeds pre-March-14. |
| Seed information before Selection Sunday | **MITIGATED** | Low | Seeds zeroed for all pre-`{year}-03-14` games. |
| Calibration double-dipping | **ACTIVE** | Medium | Sequential fitting of temperature, sharpener, and seed overrides on same validation data. |
| Feature selection on full data | **MITIGATED** | Low | Learned feature selection disabled. Fixed domain-knowledge feature set used. |
| Eval set leakage into model selection | **MITIGATED** | Low | OOS-FIX: eval set used only for diagnostic reporting, not model selection. Fixed-weight ensemble. |
| Elo cross-season carryover leakage | **MITIGATED** | Low | Elo initialized from prior year's end-of-season values. Processing years oldest-to-newest. |
| Massey composite temporal alignment | **UNKNOWN** | Medium | Massey ratings in `MMasseyOrdinals.csv` are date-stamped. Need to verify that only pre-tournament ratings are used for each year. |

### Overfitting Risks

| Risk | Status | Severity | Notes |
|---|---|---|---|
| Hyperparameter tuning overfitting | **LOW** | Low | 15 Optuna trials with narrow bounds on temporal CV. Conservative default hyperparameters. |
| Ensemble weight overfitting | **LOW** | Low | Fixed weights used. Optimizer disabled. Statistical significance guard present. |
| Post-processing chain overfitting | **MEDIUM** | Medium | 3-4 parameters (T, alpha, snap_threshold, Massey sigma) fitted sequentially on ~400 samples. |
| Feature selection overfitting | **LOW** | Low | Disabled by default. Fixed domain-knowledge set with literature citations. |
| Stacking overfitting | **LOW** | Low | Disabled by default (correct — 9 meta-features from 3 models overfit on 400 OOF samples). |

### Validation Flaws

| Flaw | Status | Severity | Notes |
|---|---|---|---|
| Random K-fold | **NONE** | N/A | No random K-fold anywhere in the codebase. Exemplary. |
| Missing seasons in LOYO | **MINOR** | Low | COVID 2020 correctly excluded. Some years may be skipped due to data quality. |
| Brier optimization target | **CORRECT** | N/A | `scoring_metric="brier"` is default. Optuna objectives can target Brier or log loss. |
| Round-weighted vs flat Brier | **PARTIAL** | Medium | `enable_round_weighted_calibration=True` but LOYO CV reports flat Brier, not round-weighted. |

### Pipeline Fragility

| Issue | Severity | Notes |
|---|---|---|
| Team ID resolution chain | **HIGH** | 4+ `_normalize_team_id` functions, fuzzy matching at 0.84 threshold. Silent misresolution can assign wrong metrics to 5-10% of teams. |
| External data availability | **MEDIUM** | Pipeline silently degrades when external ratings files are absent (features go to 0.0). |
| Dependency on optional packages | **LOW** | Graceful degradation via try/except imports for LightGBM, XGBoost, PyTorch, Optuna, etc. |
| HTML entity encoding | **LOW** | `Texas A&M` -> `texas_a_amp_m` can cause team resolution failure. |

---

## 5) Performance Ceiling Estimate

### Current Expected Percentile: **Top 3-5%**

Evidence:
- Prospective eval Brier = 0.212 on 427 held-out 2024 tournament games
- BSS = 0.143 vs uniform baseline (0.25)
- BSS = 0.027 vs Elo baseline (0.218)
- Architecture matches patterns of Top 5% solutions (MOV-primary, seed
  priors, multi-model ensemble, LOYO validation)
- But external ratings gap and calibration double-dipping prevent Elite tier

### Maximum Achievable Percentile After Fixes: **Top 1-2%**

With all five recommended improvements implemented:

| Improvement | Expected Gain | Cumulative |
|---|---|---|
| Baseline (current) | 0.212 | 0.212 |
| 1. Massey coverage verification | -0.010 | 0.202 |
| 2. Nested calibration | -0.003 | 0.199 |
| 3. Conference strength | -0.003 | 0.196 |
| 4. CV ensemble weights | -0.002 | 0.194 |
| 5. Data quality remediation | -0.003 | 0.191 |
| **Projected ceiling** | | **~0.191-0.200** |

The projected range of 0.191-0.200 is within Top 1% territory for Kaggle
March Mania (historical Top 1% threshold: ~0.195-0.200 Brier on combined
men's + women's tournaments).

**Key insight:** The gap to Top 1% is narrow (~0.012-0.020 Brier points)
and is primarily a data/integration issue, not an architecture or modeling
issue. The ML engineering is already at Elite quality. The highest-ROI
fix is ensuring external ratings actually flow through the pipeline.

---

## Audit Philosophy Notes

### What This Codebase Gets Right (Relative to Competition)

1. **It doesn't overfit.** The consistent theme across the codebase is
   restraint: stacking disabled, feature selection disabled, Optuna
   trials limited, tree depth constrained, ensemble weights fixed. This
   is exactly the right instinct for N~600.

2. **It uses the right metric.** The pipeline targets Brier score
   (quadratic), not log loss (logarithmic). This matters: Brier rewards
   confident correct predictions less extremely than log loss and
   penalizes confident wrong predictions less severely. The probability
   bounds [0.005, 0.995] are appropriate for Brier.

3. **It handles both genders.** Since 2023, 50% of the Kaggle score
   comes from women's tournament predictions. Having a dedicated pipeline
   with different calibration (stronger seed priors, simpler model)
   is critical and absent from most Kaggle submissions.

4. **It has structural model diversity.** MOV regression (continuous
   target), binary classification (discrete target), Bayesian BT
   (ID-based, no features), logistic regression (linear). These
   produce genuinely different prediction surfaces, making the
   ensemble more valuable than multiple tree models with the same features.

5. **It has an RDoF audit framework.** The `rdof_audit.py` module
   explicitly inventories researcher degrees of freedom, categorizes
   constants by tier (externally derived vs freely tuned), and enforces
   freeze-file integrity for prospective evaluation. This level of
   self-discipline is exceptional and absent from 99% of Kaggle entries.

### What Separates Top 1% from Top 5%

Based on analysis of Kaggle March Madness leaderboards 2017-2025:

1. **External ratings actually flowing through the pipeline** (not just
   coded but unused)
2. **Perfect data quality** (every team correctly mapped, every feature
   populated for every tournament team)
3. **Year-specific calibration** (not a single temperature across all years)
4. **Women's tournament treated as a first-class prediction target**
   (not an afterthought with men's model predictions)
5. **Minimal researcher degrees of freedom** (DoF ratio < 0.01)

This codebase has the architecture for all five. The gap is execution,
not design.
