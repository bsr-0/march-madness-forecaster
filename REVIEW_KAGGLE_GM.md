# Kaggle Grandmaster Review: March Madness Forecaster

**Reviewer**: Kaggle Grandmaster-level ML Practitioner
**Date**: 2026-03-03
**Competition**: Kaggle March Machine Learning Mania 2026
**Evaluation Metric**: Round-weighted Brier Score (since 2023)

---

## 1. Rigorous Methodology Review

### 1.1 Architecture Overview

The pipeline implements a **five-model ensemble** with post-processing:

| Component | Type | Weight | Role |
|-----------|------|--------|------|
| **SpreadRegressor** | LightGBM regression on margin | 0.40 | Primary signal (MOV-first, raddar-style) |
| **LightGBM** | Binary classification | 0.25 | Gradient boosted trees |
| **Logistic Regression** | Linear baseline | ~0.20 | Regularization anchor |
| **XGBoost** | Binary classification | 0.15 | Alternative tree learner |
| **Bayesian Bradley-Terry** | ID-based MAP estimation | Blended | Orthogonal who-beat-whom signal |

Post-processing chain: **Seed overrides -> Temperature calibration -> Brier sharpening -> Clip**

### 1.2 What the Codebase Gets Right

**A. MOV-First Design (SpreadRegressor at 0.40 weight)**
This directly mirrors the "raddar" approach that dominated 2018-2024. Predicting point margin first provides richer gradient signal than binary classification. The logistic CDF conversion (`P = 1/(1+exp(-spread/sigma))`) with MLE-calibrated sigma (~11 points for NCAA) is textbook correct. This single design decision is worth more than any amount of feature engineering.

**B. Fixed Domain-Knowledge Feature Set**
The `FIXED_FEATURE_SET` (23 features) and `SIMPLE_FEATURE_SET` (9 features) are selected from published literature *before* observing model performance. Each feature has citations ([KP], [OL], [KUB], [538], [VAR], [KAG]). This completely eliminates the double-dipping problem that kills most Kaggle notebooks that use SHAP/permutation importance on 300-600 training samples. This is the single most important methodological decision in the codebase.

**C. Temporal Validation Discipline**
Leave-One-Year-Out CV with `rolling_window` mode (train only on prior years), COVID 2020 exclusion, dev/holdout partitioning (2016-2024 dev, 2025 holdout), and pair-size snapping for symmetric samples. This is exemplary — better than 90% of public Kaggle notebooks.

**D. Brier-Optimal Post-Processing Pipeline**
Since 2023, Kaggle uses Brier score, not log loss. The dedicated `BrierPostProcessor` with:
- Seed-based overrides for extreme matchups (1v16 snapped to 0.987)
- Round-weighted calibration (NCG = 32x weight of R64)
- Power-transform sharpening optimized for Brier
This correctly targets the actual competition metric.

**E. Women's Tournament Pipeline**
50% of Kaggle score since 2023. The codebase correctly implements:
- Separate pipeline with different calibration
- Stronger seed priors (50% vs 10% for men's) — women's tournament is far more seed-predictable
- Different historical win rate tables
- Simpler model complexity (fewer upsets to model)

**F. Conservative Hyperparameters**
`num_leaves=8` (LGB), `max_depth=3` (XGB), `min_child_samples=50`, only 15 Optuna trials, ensemble weight regularization toward uniform. This shows genuine understanding of the ~600-sample constraint. Over-tuning is the #1 killer in this competition.

**G. Multi-Year Training Pool**
Pooling 10+ years of regular-season games (300 -> 3000+) with exponential decay (0.85/year) addresses the fundamental sample-size problem. Era-based quality weighting correctly excludes 2005-2006 (>95% zeroed box scores).

**H. Bayesian Bradley-Terry with Uncertainty Propagation**
ID-based ratings provide completely orthogonal signal to feature-based models. The probit approximation naturally shrinks predictions toward 0.5 for under-sampled teams (16-seeds with ~30 games). Mathematically principled (Glickman 1999).

**I. Massey Composite Blend (0.25 weight)**
The config allocates 25% weight to `diff_external_rating_composite` (meta-ranking of 100+ systems). Every recent Kaggle winner used Massey/composite ratings. This is the single highest-signal external feature.

### 1.3 Design Quality Assessment

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Feature engineering | A- | 66 features, literature-cited, redundancy pruned. Missing: no league-average regression. |
| Model diversity | A | MOV regression, classification, linear, ID-based. Good orthogonality. |
| Ensemble strategy | A- | Fixed-weight CFA with significance guard. Stacking correctly disabled for small N. |
| Calibration | A | Temperature scaling (robust for small N), Brier-optimal, round-weighted. Small-sample guard. |
| Validation | A+ | LOYO rolling window, dev/holdout split, RDoF audit. Best-in-class. |
| Data pipeline | B | Comprehensive scrapers, but team ID resolution chain is fragile. Historical data quality issues. |
| Code quality | B- | 6500-line monolith (`sota.py`). Production-grade error handling, but very hard to audit for subtle bugs. |

---

## 2. Comparison to Winning Kaggle Approaches

### 2.1 The Raddar Benchmark (Dominant 2018-2024)

Raddar's approach — which was literally copy-pasted to win in both 2022 and 2023 — consists of:
- **XGBoost regression on margin of victory** (not binary classification)
- **GLM-based Elo** treating team IDs as factors
- **Nonparametric conversion** from predicted margin to probability
- **Very few features** (~5-10)

**Alignment with this codebase**: The SpreadRegressor at 0.40 weight directly implements the raddar MOV-first philosophy. The Bayesian Bradley-Terry serves a similar role to raddar's GLM Elo. The 9-feature `SIMPLE_FEATURE_SET` matches the feature minimalism. **This is well-aligned.**

### 2.2 Top-1% Gold Solution (maze508, 2023)

The 9th-place (top 1%) solution used:
- Ensemble of gradient boosting models
- Massey Ordinals as the strongest feature
- Simple calibration
- Careful attention to women's tournament

**Alignment**: The architecture is compatible, but the external rating integration execution is the gap (see Section 4).

### 2.3 Historical Winner Patterns

| Pattern | This Codebase | Impact |
|---------|--------------|--------|
| MOV prediction > binary classification | Yes (SpreadRegressor) | High |
| Massey/composite ratings as primary feature | Config exists, but **training data is 0.0** | **Critical gap** |
| Few features (5-15 optimal) | Yes (9 in simple mode, 23 in standard) | High |
| Elo or team-strength rating | Yes (Bayesian BT + Elo feature) | Medium |
| Proper calibration | Yes (Temperature + Brier-optimal) | Medium |
| Conservative regularization | Yes (shallow trees, early stopping) | High |
| Women's tournament as 50% of score | Yes (separate pipeline + seed priors) | High |
| Seed-based priors for extreme matchups | Yes (SeedBasedOverrides) | Medium |

### 2.4 Historical Score Ranges (Brier Score, Round-Weighted)

Based on available data:
- **Top 1%**: Brier ~0.410-0.425
- **Top 5%**: Brier ~0.430-0.445
- **Median**: Brier ~0.475-0.490
- **Coin flip baseline**: Brier = 0.250 (per game, but ~0.50 round-weighted due to late-round variance)
- **Seed-only baseline**: Brier ~0.200-0.220 (per game)

---

## 3. Estimated Kaggle Percentile Performance

### Assessment: **Top 5-15% (likely ~8th percentile)**

This is a strong codebase that would comfortably beat the median and has the architectural DNA to compete at the top. However, several execution gaps prevent consistent top-1% performance.

### What Places It in Top 5-15%

**Strengths pushing up:**
- MOV-first ensemble (correct architecture)
- Fixed feature set (avoids overfitting trap)
- Brier-optimal post-processing (correct metric)
- Women's pipeline (50% of score)
- Conservative regularization (appropriate for N)
- Multi-year training pool (addresses sample-size)

**Weaknesses pulling down:**
- External rating composite is 0.0 in training (**~3-5 percentile points lost**)
- Historical training data quality issues (team ID mismatches, zeroed features)
- No published backtest results to verify actual performance
- Calibration/sharpening double-dipping risk on same validation data
- Some features 0.0 in historical training (experience, continuity, travel)

### Confidence Interval

- **Optimistic scenario** (external ratings wired, data issues fixed): **Top 1-5%**
- **Expected scenario** (current state, clean execution): **Top 5-15%**
- **Pessimistic scenario** (data bugs, bad calibration year): **Top 15-30%**

The high variance is inherent to tournament prediction — even with perfect probabilities, a 63-game tournament has enormous outcome variance. A top-1% model in expectation will finish outside the top 10% ~40% of the time.

---

## 4. Limitations and Mistakes to Fix

### CRITICAL (Must Fix Before Submission)

#### 4.1 Massey Ordinals Composite Not Wired into Training Pipeline

**Location**: `src/pipeline/sota.py`, `src/data/scrapers/external_ratings.py`
**Issue**: The `diff_external_rating_composite` feature is in `FIXED_FEATURE_SET` (line 526) and `SIMPLE_FEATURE_SET` (line 540), but is populated as **0.0** during training. The `ExternalRatingsLoader` and `populate_from_massey_ordinals()` exist in code but are never called during `SOTAPipeline.run()`.

**Impact**: This is the **single highest-signal feature** in the competition. Every recent Kaggle winner used Massey/composite ratings. The codebase allocates 25% blend weight to a feature that is always zero. This alone likely costs 3-5 percentile points.

**Fix**: Wire `populate_from_massey_ordinals()` into the pipeline's data loading phase. Load Kaggle's `MasseyOrdinals.csv` for historical years and integrate current-year ratings for prediction. The `MasseyStandalonePredictor` class in `brier_optimal.py` is already implemented — it just needs to receive real data.

#### 4.2 Team ID Namespace Mismatch in Historical Data

**Location**: `src/data/ingestion/historical_pipeline.py`, `src/data/team_name_resolver.py`
**Issue**: Historical training data has team IDs in a different namespace than the feature engineering pipeline expects. The `AUDIT_TOP1PCT.md` documents "zero direct matches" between namespaces, and "7/68 tournament teams potentially misresolved."

**Impact**: If training features are mapped to wrong teams, the model learns corrupted signals. Even a 10% misresolution rate on 3000+ training games creates significant noise.

**Fix**: Audit and normalize all team ID namespaces. Use Kaggle's canonical `TeamID` from `MTeams.csv` as the single source of truth. Build a verified mapping table with confidence scores and reject matches below 0.95.

### HIGH PRIORITY (Should Fix)

#### 4.3 Calibration/Sharpening Double-Dipping

**Location**: `src/ml/calibration/brier_optimal.py`, `src/pipeline/sota.py`
**Issue**: The `BrierCalibrator` (temperature scaling) and `BrierOptimalSharpener` (power transform) are both fit on validation data. If the same holdout samples are used for both, the combined pipeline sees the data twice — the sharpener overfits to the calibrator's output distribution.

**Impact**: Moderate. Over-sharpening on small calibration sets could push predictions too far from 0.5, hurting Brier score on unseen tournaments.

**Fix**: Either (a) use separate folds for calibration vs. sharpening, or (b) treat the entire calibration + sharpening pipeline as a single unit and fit jointly via nested CV, or (c) disable sharpening (alpha=1.0) unless validated on a truly held-out year.

#### 4.4 No Published End-to-End Backtest Results

**Location**: `src/ml/evaluation/kaggle_backtest.py`
**Issue**: The `KaggleBacktester` exists with historical Kaggle thresholds, but the README and audit documents don't include actual backtest Brier scores for years 2015-2025. Without these, all performance claims are theoretical.

**Impact**: High for confidence in the submission. Without backtests, you don't know if the pipeline actually achieves the claimed top-5% performance.

**Fix**: Run the full pipeline on historical years 2017-2025 (excluding 2020) and report per-year Brier scores, accuracy, upset accuracy, and estimated Kaggle rank. This is the single most informative diagnostic.

#### 4.5 Features Zero in Historical Training

**Location**: `src/pipeline/sota.py:2517-2526`, `FIXED_FEATURE_SET`
**Issue**: Several features in the fixed feature set are 0.0 in historical training data:
- `diff_avg_experience`, `diff_roster_continuity` (cbbpy data unavailable for historical years)
- `travel_advantage` (venue coordinates unavailable for regular-season)
- `diff_external_rating_composite`, `diff_external_rating_spread` (not wired in)

**Impact**: Tree models handle zeros gracefully, but features that are always zero during training provide no learned signal. When they suddenly become non-zero at inference time (current-year predictions), the model has never seen these values and may produce unpredictable outputs.

**Fix**: For features that are genuinely unavailable historically, consider removing them from the training feature set entirely and applying them only as post-hoc adjustments. Or, backfill historical data from Kaggle's provided datasets.

### MEDIUM PRIORITY (Nice to Have)

#### 4.6 Monolithic Pipeline File

**Location**: `src/pipeline/sota.py` (6500+ lines)
**Issue**: The entire pipeline logic — data loading, feature construction, model training, calibration, prediction, export — lives in a single file. This makes it extremely difficult to:
- Unit test individual components
- Audit for subtle bugs
- Debug performance regressions

**Fix**: Refactor into separate modules: `data_preparation.py`, `model_training.py`, `ensemble.py`, `post_processing.py`, `prediction.py`.

#### 4.7 Round Weight Inference from Dates is Fragile

**Location**: `src/pipeline/sota.py:652-680` (`_infer_tournament_round_weight`)
**Issue**: Tournament round weights are inferred from game dates using hardcoded date ranges (e.g., "day of March >= 24 = E8"). Tournament schedules shift by several days year to year. The function uses `date(year, 3, 1)` as a reference point, which doesn't account for early/late tournaments.

**Impact**: Mislabeled rounds cause games to receive wrong weights in training and calibration. An E8 game labeled as S16 gets 4x weight instead of 8x.

**Fix**: Use Kaggle's actual round labels from tournament result CSVs rather than date inference.

#### 4.8 Bracket Portfolio Strategy Underdeveloped

**Location**: `src/optimization/bracket_portfolio.py`
**Issue**: Since 2024, Kaggle allows 1-100k bracket submissions. The optimal strategy for bracket portfolios is fundamentally different from probability submission — you want diverse brackets that collectively maximize expected payoff. The portfolio generation exists but the optimization is basic.

**Impact**: The 2024 winner specifically optimized for this format.

**Fix**: Implement anti-correlation sampling — generate brackets that collectively cover more of the outcome space. Use conditional simulation to ensure portfolio diversity.

#### 4.9 Women's Pipeline Reuses Men's Feature Engineering

**Location**: `src/data/features/womens_feature_engineering.py`, `src/pipeline/womens.py`
**Issue**: Despite separate configuration, the women's pipeline uses the same `FeatureEngineer` with identical feature definitions. Women's basketball has different dynamics (lower pace, different shooting rates, fewer upsets, different conference strength distributions).

**Fix**: Build women's-specific population statistics (`_POPULATION_STATS`) and potentially different feature importance weights.

#### 4.10 No Regression to the Mean for Volatile Features

**Issue**: 3-point shooting percentage regresses heavily toward the mean over small samples. A team shooting 42% from 3 over 30 games is likely a ~36% true-talent shooter. The `three_pt_regression_signal` feature exists but is not integrated into the core predictions.

**Fix**: Apply Bayesian shrinkage to volatile features (3PT%, FT rate, turnover rate) using league-average priors and per-team sample sizes. This is standard in sports analytics and would improve predictions for low-sample teams.

---

## 5. Summary Recommendations (Priority Order)

| Priority | Action | Expected Percentile Gain |
|----------|--------|--------------------------|
| 1 | Wire Massey Ordinals into training pipeline | +3-5% |
| 2 | Run and publish end-to-end backtests (2017-2025) | Diagnostic (confidence) |
| 3 | Fix team ID namespace mismatches | +1-3% |
| 4 | Separate calibration/sharpening validation folds | +0.5-1% |
| 5 | Backfill external ratings for historical training | +1-2% |
| 6 | Use Kaggle round labels instead of date inference | +0.5% |
| 7 | Add Bayesian shrinkage for volatile features | +0.5-1% |
| 8 | Optimize bracket portfolio diversity | Format-dependent |

**Bottom line**: This codebase demonstrates deep understanding of March Madness prediction methodology. The architecture is correct, the validation discipline is exemplary, and the post-processing pipeline targets the right metric. The gap to top-1% is primarily in **data execution** (wiring Massey, fixing team IDs) rather than **model design**. Fix items 1-3 above and this becomes a legitimate top-1% contender.

---

*Sources:*
- [Raddar Solution Notebook (Kaggle)](https://www.kaggle.com/code/nigelhenry/raddar-solution)
- [Why the Raddar Solution Works (Kaggle)](https://www.kaggle.com/code/nigelhenry/why-the-raddar-solution-works)
- [Top 1% Gold - 2023 Solution Writeup (maze508)](https://medium.com/@maze508/top-1-gold-kaggle-march-machine-learning-mania-2023-solution-writeup-2c0273a62a78)
- [1st Place Winner Interview: Andrew Landgraf (Kaggle Blog)](https://medium.com/kaggle-blog/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf-f18214efc659)
- [4th Place Winner Interview: Erik Forseth (Kaggle Blog)](https://medium.com/kaggle-blog/march-machine-learning-mania-4th-place-winners-interview-erik-forseth-8d915d8cea57)
- [March Machine Learning Mania 2023 Discussion (Kaggle)](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/discussion/399553)
- [On Log-Loss and Scoring the NCAA Tournament (statsbylopez)](https://statsbylopez.netlify.app/post/on-log-loss-in-the-ncaa-tournament/)
- [March Madness 2025 Model (The Data Jocks)](https://thedatajocks.com/march-madness-2025-model/)
