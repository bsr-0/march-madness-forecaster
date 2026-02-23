# Senior ML Audit: March Madness Modeling Stage

**Date:** 2026-02-22
**Auditor:** Principal ML Review (automated)
**Repo:** march-madness-forecaster

---

## Executive Summary

This repo implements a **serious, production-grade** March Madness forecasting system. It meets or exceeds the rubric standards in most areas, with a few residual concerns around out-of-sample validation depth and the effective utilization of its GNN/Transformer components. The modeling layer is substantially more sophisticated than typical Kaggle submissions.

**Overall Grade: B+ / A-**

The system would be competitive in Kaggle March Madness competitions. The main risks are: (1) complexity vs. sample size tension (66 team features, ~400 tournament training samples per year), and (2) whether the GNN/Transformer components add genuine OOS lift vs. the strong LightGBM+XGBoost+Logistic baseline.

---

## 1. Core Game Prediction Model

### 1.1 Model Formulation

| Criterion | Rubric Requirement | Implementation | Verdict |
|---|---|---|---|
| Binary probabilistic classifier | Required | LightGBM (`binary` objective) + XGBoost (`binary:logistic`) + Logistic Regression ensemble | **PASS** |
| Tier 1 models | Logistic, GBM, or Elo | LightGBM, XGBoost, Logistic Regression, Elo (MOV-adjusted), GNN, Transformer | **PASS** |
| Differential features | `X = team1_stats - team2_stats` | `diff = v1 - v2` in `FeatureEngineer.create_matchup_features()` (`feature_engineering.py:1083-1085`) | **PASS** |
| Interaction features | Tempo, style matchup | Tempo interaction, style mismatch, seed interaction (`feature_engineering.py:1096-1112`) | **PASS** |
| Absolute-level features | Avg of both teams for context | Implemented via `ABSOLUTE_LEVEL_INDICES` — avg of `adj_off_eff`, `adj_def_eff`, `sos_adj_em`, `elo_rating`, `win_pct` (`feature_engineering.py:1088-1093`) | **PASS** |

**Strengths:**
- Three-model ensemble (LightGBM + XGBoost + Logistic) is the proven Kaggle-winning architecture
- Differential features are the primary representation (correct)
- Interaction features capture matchup-specific dynamics (tempo, style, seed)
- Absolute-level features added to preserve game-quality context that pure diffs lose

**Concerns:**
- 66 team features before selection is high for ~400 annual tournament samples. The feature selection pipeline (`feature_selection.py`) addresses this with VIF pruning, correlation pruning, importance ranking, and bootstrap stability filtering, but the effective dimensionality after selection should be monitored.

### 1.2 Feature Engineering Quality

| Feature Category | Count | Quality Assessment |
|---|---|---|
| Core efficiency (KenPom-style) | 3 | `adj_off_eff`, `adj_def_eff`, `adj_tempo` — gold standard |
| Four Factors (offense + defense) | 8 | Complete Oliver Four Factors implementation |
| Player-level (RAPM, WARP) | 6 | Regularized APM via ridge regression (`feature_engineering.py:1249-1310`) |
| Volatility/entropy | 4 | Game flow-based lead volatility, entropy, comeback factor |
| Shot quality (xP) | 2 | Expected points per possession decomposition |
| Schedule strength | 4→2 | SOS, Elite SOS (cluster pre-selection reduces from 4) |
| Elo rating | 1 | MOV-adjusted Elo with FiveThirtyEight methodology |
| Shooting splits | 2 | 3PT%, 3PT rate (two_pt_pct correctly removed as redundant) |
| Context features | 7 | Rest days, coach experience, conference tourney champion, etc. |
| Momentum/variance | 3 | Momentum, 3PT variance, pace-adjusted variance |
| Quality metrics | 5 | Win%, Q1 win%, WAB, foul rate, 3PT regression signal |

**Redundancy cleanup (FIX #1):** 10 features correctly identified and removed as algebraically or near-perfectly redundant:
- `adj_efficiency_margin` (exact linear: `off - def`)
- `barthag` (monotonic transform of off/def ratio)
- `efficiency_ratio` (~r=0.95 with off/def components)
- `seed_efficiency_residual` (exact linear combo)
- `consistency` (near-inverse of `pace_adjusted_variance`)
- `momentum_5g` (~r=0.85 with `momentum`)
- `true_shooting_pct` / `opp_true_shooting_pct` (~r=0.92 with eFG% + FT rate)
- `two_pt_pct` (~r=0.88 with eFG%)
- `continuity_learning_rate` (deterministic function of `roster_continuity`)
- `close_game_record` (pure noise — binomial draw on 5-10 games)

This is well-documented and methodologically sound (`feature_engineering.py:38-65`).

**Feature selection pipeline** (`feature_selection.py`):
1. Cluster pre-selection (known correlated groups collapsed)
2. Near-zero variance pruning
3. VIF pruning (threshold=10, enabled by default)
4. Correlation pruning (threshold=0.85, target-correlation tie-breaking)
5. SHAP + permutation importance ranking
6. Bootstrap stability filtering (80% threshold)
7. Post-selection multicollinearity validation

**Verdict: STRONG PASS.** Feature engineering is thorough, well-documented, and addresses multicollinearity systematically.

### 1.3 Model Regularization

| Model | Regularization | Assessment |
|---|---|---|
| LightGBM | `num_leaves=8`, `min_child_samples=50`, `lambda_l1=1.0`, `lambda_l2=1.0`, `feature_fraction=0.7`, `bagging_fraction=0.7` | **Conservative and appropriate** for ~400 samples. `num_leaves=8` forces shallow trees. |
| XGBoost | `max_depth=3`, `min_child_weight=10`, `gamma=0.5`, `reg_alpha=1.0`, `reg_lambda=1.0` | **Conservative.** max_depth=3 is the standard shallow-tree choice. |
| Logistic | `C=1.0`, StandardScaler applied | **Appropriate.** C=1.0 is moderate regularization. |
| Optuna search | `num_leaves=[4,16]`, `min_child_samples=[30,100]`, `max_depth=[2,4]` | **Heavily constrained search space.** Prevents Optuna from finding overfit configs. |

**Verdict: PASS.** The OOS-FIX comments throughout the code indicate awareness of the small-sample problem and deliberate conservative regularization.

---

## 2. Ensemble Architecture

### 2.1 Combinatorial Fusion Analysis (CFA)

The ensemble (`cfa.py`) combines three model families:
- **LightGBM** (gradient boosting)
- **XGBoost** (gradient boosting, different implementation)
- **Logistic Regression** (linear baseline)

Plus optional GNN and Transformer components.

| Criterion | Implementation | Verdict |
|---|---|---|
| Weighted combination | `P = w_lgb * P_lgb + w_xgb * P_xgb + w_log * P_log` | **PASS** |
| Weight optimization | Bootstrap-aggregated grid search (`EnsembleWeightOptimizer`, `hyperparameter_tuning.py:735-897`) | **PASS** |
| Diversity metrics | Spearman rank correlation, prediction spread, per-model deviation (`cfa.py:137-214`) | **PASS** |
| Brier-score optimization | Inverse-Brier softmax with significance guard (`cfa.py:235-320`) | **PASS** |
| Minimum weight floor | 5% per model (`cfa.py:307`) | **PASS** — prevents zero-ing out models |

**Strengths:**
- Bootstrap-aggregated weight optimization prevents overfitting on small validation sets
- Statistical significance guard on Brier differences prevents concentrating weight based on noise
- L2 regularization toward uniform weights in optimizer
- Minimum sample guard (returns uniform weights below 50 samples)
- Diversity bonus correctly REMOVED (was rewarding noise with only 3 models)

**Verdict: STRONG PASS.** The ensemble is well-designed for the small-sample regime.

### 2.2 GNN and Transformer Components

| Component | Implementation | Concern |
|---|---|---|
| Schedule GNN | `ScheduleGCN` using PyTorch Geometric (GCNConv/SAGEConv/GATConv) | Produces 32-dim team embeddings from schedule graph |
| Game Flow Transformer | `GameFlowTransformer` with positional encoding | Produces 64-dim temporal embeddings from game sequences |

**Critical concern:** These components add 96 embedding dimensions to the feature space. With ~400 tournament training samples, the sample-to-feature ratio becomes problematic. The ablation framework (`ablation.py`) exists to measure their OOS contribution, which is the right approach. However, the audit cannot confirm whether these components provide statistically significant OOS lift without seeing actual ablation results.

**Recommendation:** Run full ablation study and require p < 0.05 on Brier improvement before including in production brackets.

---

## 3. Tournament Simulation

### 3.1 Monte Carlo Engine

| Criterion | Rubric Requirement | Implementation | Verdict |
|---|---|---|---|
| Stochastic simulation | >=10,000 sims | 50,000 simulations (`SimulationConfig.num_simulations`) | **PASS** |
| Noise injection | Logit-space noise | `noise_std=0.12` in logit space (`monte_carlo.py:26`) | **PASS** |
| Correct bracket structure | 1v16, 8v9, etc. | Standard seed order implemented (`monte_carlo.py:522`) | **PASS** |
| Parallel execution | Required for 50K sims | `ProcessPoolExecutor` with batch processing (`monte_carlo.py:384-421`) | **PASS** |
| Regional correlation | Intra-region upset clustering | Round-dependent correlation decay [1.0, 0.6, 0.3, 0.15, 0.0, 0.0] (`monte_carlo.py:182`) | **PASS** |
| Injury modeling | Per-team injury shift | 2% per-game probability, severity 0.05-0.25 in logit space (`monte_carlo.py:209-214`) | **PASS** |
| Confidence intervals | Standard errors on odds | Wilson score intervals for all round-level probabilities (`monte_carlo.py:463-486`) | **PASS** |

**Simulation design strengths:**
- **Round-dependent correlation decay** models the empirical observation that upset clustering is strongest in early rounds
- **Variance multiplication** (not signed logit shift) for regional effects — avoids systematic bias toward either team in a matchup
- **Unified noise model** prevents double-counting regional and game-level noise
- **Pre-computed matchup cache** avoids serialize/deserialize issues with predict functions across process boundaries
- **Lognormal cross-region variance** for Final Four games

**Concerns:**
- `noise_std=0.12` was changed multiple times (0.04→0.035→0.02→0.12). The final value is cited as based on Lopez & Matthews (JQAS 2015) suggesting 0.15-0.25 in logit space. 0.12 is on the conservative side of the academic range — this is acceptable but worth sensitivity testing.
- `regional_correlation=0.10` (reduced from 0.25 during OOS fix). This is also in the constant registry and flagged as Tier 3 (freely tuned).

**Upset distribution:** The system models upsets through three mechanisms:
1. Base probability from matchup model (calibrated)
2. Logit-space noise (±0.12 std)
3. Regional correlation (variance multiplier)

**Verdict: STRONG PASS.** The Monte Carlo simulation is among the most sophisticated I've seen in open-source MM forecasters. The round-dependent correlation decay and variance multiplication approach are research-grade.

---

## 4. Calibration

### 4.1 Calibration Methods

Four calibration methods implemented (`calibration.py`):

| Method | Implementation | Appropriateness |
|---|---|---|
| **Temperature Scaling** | Brent's method optimization, small-sample guard with bootstrap CI (`calibration.py:359-541`) | **Best choice** for small samples (1 parameter) |
| **Isotonic Regression** | sklearn IsotonicRegression | Flexible but high variance with small N |
| **Platt Scaling** | Custom gradient descent with best-tracking, adaptive LR, early stopping (`calibration.py:247-356`) | 2 parameters, reasonable for moderate N |
| **None** (identity) | Pass-through | Baseline comparison |

**Key safeguard:** Temperature scaling includes a small-sample guard (`calibration.py:412-482`):
- If N < 30, fits T via bootstrap and checks whether CI includes T=1.0
- If CI includes identity, falls back to T=1.0 (no calibration)
- This prevents harmful distortion from noisy small-sample calibration

### 4.2 Calibration Metrics

Comprehensive metrics (`CalibrationMetrics` dataclass):
- Brier Score + decomposition (reliability, resolution, uncertainty)
- Log Loss
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- ROC-AUC
- Bootstrap CI on Brier score (2000 resamples)
- Per-bin calibration analysis (overconfident/underconfident diagnosis)
- Brier Skill Score (vs. base rate)

**Verdict: STRONG PASS.** Calibration is research-grade with appropriate small-sample safeguards.

---

## 5. Validation & Leakage Prevention

### 5.1 Cross-Validation Strategy

| Method | Implementation | Purpose |
|---|---|---|
| **Temporal CV** | Expanding-window splits, chronological ordering (`hyperparameter_tuning.py:83-169`) | Hyperparameter tuning |
| **Leave-One-Year-Out (LOYO)** | Rolling window (train on past only) or full leave-one-out, COVID year excluded (`hyperparameter_tuning.py:597-732`) | Ensemble weight optimization, holdout evaluation |
| **Holdout evaluation** | Full pipeline retrain per holdout year, AdjEM-logistic baseline (`rdof_audit.py:404-1091`) | Final OOS assessment |

**Critical design decisions:**
- LOYO `rolling_window` mode (default) trains on strictly earlier years only — **honest causal evaluation**
- 15% of training data held out within each fold for early stopping — **prevents test-set leakage into training**
- Early stopping only applied when a real validation set is provided — **not on training loss alone** (`cfa.py:406`)
- Pair-size awareness in temporal CV to keep symmetric augmented samples together (though symmetric augmentation was later removed)

### 5.2 Leakage Prevention

| Leakage Vector | Mitigation | Evidence |
|---|---|---|
| Future game data in training | Temporal CV, LOYO with rolling window | `hyperparameter_tuning.py:652-653` |
| Tournament results in calibration | Calibration fit on training preds only | `rdof_audit.py:929-940` |
| Feature scaling on full dataset | StandardScaler fit on training only | `rdof_audit.py:845` |
| Validation boundary | `_compute_train_val_boundary` separates ~80% train / 20% val | `test_leakage_fixes.py:64-100` |
| Tournament games excluded from training | `_is_tournament_game` filter | `rdof_audit.py:514-528` |
| COVID year excluded | Year 2020 filtered out | `rdof_audit.py:435` |
| Feature selection on test data | `FeatureSelector.fit()` called with training data only | `feature_selection.py:922` |
| Missing-data indicators removed | Flags for scraper availability were encoding data artifacts | `feature_engineering.py:741-745` |

**Dedicated leakage test suite** (`test_leakage_fixes.py`) validates:
- Train/val boundary at expected date
- Boundary None when too few games
- Tournament games excluded from boundary calculation

**Verdict: STRONG PASS.** Leakage prevention is thorough with both architectural safeguards and dedicated tests.

### 5.3 Researcher Degrees of Freedom (RDoF) Audit

The most impressive component: a full RDoF audit framework (`rdof_audit.py`, 1576 lines).

**Constant registry** classifies every hand-tuned constant:
- **Tier 1 (6 constants):** Externally derived (published papers, public data). Examples: Four Factors weights (Oliver 2004), HCA points (meta-analysis), seed prior slope (40-year logistic fit).
- **Tier 2 (10 constants):** Structurally constrained (bounded, monotone). Examples: training year decay (0.85), recency half-life (0.3), LightGBM/XGBoost regularization params.
- **Tier 3 (8 constants):** Freely tuned — **must be cross-validated**. Examples: ensemble weights, tournament shrinkage, MC noise std.

**Sensitivity analysis** (`SensitivityAnalyzer`):
- Grid search each Tier 3 constant via LOYO on dev years
- Train once per year, sweep post-training constants cheaply
- Detects whether current values are near-optimal or potentially overfit
- Reports "flat" constants (insensitive to value) vs. sensitive ones

**Holdout evaluation protocol:**
- Complete pipeline retrain per holdout year
- AdjEM-logistic baseline (stronger than seed-only)
- Bootstrap CIs on Brier scores
- PASS/WARN/FAIL verdict based on improvement over baseline
- Config hashing for audit trail

**DoF ratio:** 8 Tier 3 constants / ~63 holdout tournament games per year = 0.127. Target is < 0.01. This ratio is **too high** for any single holdout year, but pooling across multiple years brings it closer to acceptable range.

**Verdict: EXCEPTIONAL.** This level of self-awareness about researcher degrees of freedom is rare even in published academic work. The framework is well-designed.

---

## 6. Advanced Components

### 6.1 Feature Stability Scoring

Point-in-time degradation model (`sota.py:124-150`) assigns stability scores to features:
- Tempo (0.9), FT% (0.8), eFG% (0.8) — high stability
- Bench RAPM (0.5), transfer impact (0.5) — moderate stability
- 3PT variance, momentum — lower stability

This allows the pipeline to appropriately weight early-season vs. late-season feature reliability.

### 6.2 Leverage Optimization

Pool-aware bracket optimization (`leverage.py`):
- Leverage ratio = model probability / public pick percentage
- Expected value differential accounting for pool size
- Three strategies: balanced, chalk, contrarian

### 6.3 Statistical Significance Testing

Formal model comparison (`statistical_tests.py`):
- Paired t-test on per-game squared errors
- Permutation test (10,000 permutations)
- Cohen's d effect size

### 6.4 Ablation Framework

Systematic component evaluation (`ablation.py`):
- Ablates: GNN, Transformer, travel distance, injury model, tournament adaptation, stacking
- Save/restore pattern for pipeline state
- Paired significance test for each ablation

---

## 7. Identified Weaknesses

### 7.1 High Complexity vs. Sample Size

The system has 66 team features (before selection), 3+ model families, GNN/Transformer embeddings, multiple post-processing steps, and 8 freely-tuned constants. With ~63 tournament games per year and ~400 regular-season training games, this creates genuine overfitting risk. The feature selection pipeline and conservative regularization mitigate this, but the **effective model complexity** should be formally quantified.

**Recommendation:** Report the effective number of parameters (after regularization) relative to training sample size. Target: effective parameters < 10% of training samples.

### 7.2 GNN/Transformer Contribution Unverified

The GNN (32-dim) and Transformer (64-dim) embeddings are architecturally present but their OOS contribution is unverified in the artifacts I can inspect. The ablation framework exists but results are not cached.

**Recommendation:** Run full ablation study. If neither GNN nor Transformer shows p < 0.05 improvement, remove them to reduce complexity.

### 7.3 Monte Carlo Constant Sensitivity

The MC noise parameters (`noise_std=0.12`, `regional_correlation=0.10`) were changed multiple times across fix rounds. They are correctly classified as Tier 3 but are noted as "bracket-level, not game-level" and skipped by the sensitivity analyzer. Since bracket accuracy is the ultimate output, these should be validated against historical bracket scoring.

**Recommendation:** Backtest MC parameters against historical tournament outcomes (2017-2024) using actual bracket scoring.

### 7.4 No Explicit Elo Model as Standalone Baseline

While Elo is computed as a feature, there is no standalone Elo-based prediction model used as a formal baseline comparison. The RDoF audit uses an "AdjEM-logistic" baseline which is stronger than seed-only but weaker than a well-tuned Elo system.

**Recommendation:** Add a standalone Elo model (K-factor optimized) as a second baseline tier.

### 7.5 Calibration Train/Test Separation

Temperature scaling is fit on training ensemble predictions (out-of-bag), which is appropriate. However, for the isotonic and Platt methods, there is no explicit cross-validation of the calibration step itself — these higher-parameter calibrators could overfit.

**Recommendation:** If using isotonic/Platt, use nested CV for calibration or stick with temperature scaling (1 parameter).

---

## 8. Scoring Against Rubric

### Per-Game Prediction Requirements

| Requirement | Score | Notes |
|---|---|---|
| P(Team A wins) output | 10/10 | All models output probabilities |
| Well-calibrated probabilities | 9/10 | Multiple calibration methods with small-sample guards |
| Stable across seasons | 8/10 | LOYO evaluation, temporal CV, but multi-year backtest depth unclear |
| No information leakage | 9/10 | Comprehensive leakage prevention with tests |
| Differential features | 10/10 | Primary representation |
| Regularized models | 10/10 | Conservative hyperparameters throughout |
| Proper cross-validation | 9/10 | Temporal CV + LOYO, but GNN/Transformer CV is separate concern |

### Tournament Simulation Requirements

| Requirement | Score | Notes |
|---|---|---|
| Valid stochastic simulation | 10/10 | 50K sims with proper bracket structure |
| Correct upset distribution | 9/10 | Multi-mechanism upset modeling, but MC params need backtest |
| Realistic championship probabilities | 9/10 | Regional correlation, injury modeling, Wilson CIs |
| Leverage optimization | 10/10 | Pool-aware EV maximization |

### Overall Architecture

| Requirement | Score | Notes |
|---|---|---|
| RDoF audit trail | 10/10 | Exceptional — constant registry, config hashing, sensitivity analysis |
| Feature selection rigor | 9/10 | VIF + correlation + importance + bootstrap stability |
| Code documentation | 9/10 | Extensive FIX comments documenting every design decision |
| Test coverage | 8/10 | Dedicated test files for most components |

### **Overall Score: 88/100 — Strong competitive system with research-grade validation framework**

---

## 9. Comparison to Rubric Tiers

### Would this system produce competitive bracket accuracy?

**Yes, with caveats.**

- **vs. Seed baseline:** The system includes formal holdout evaluation against an AdjEM-logistic baseline. The architecture is capable of beating seed-only predictions.
- **vs. Top Kaggle submissions:** The LightGBM + XGBoost ensemble with Four Factors + Elo + player metrics is the proven Kaggle architecture. Feature engineering is comparable to top-10 submissions.
- **vs. Published academic models:** The RDoF audit framework and Monte Carlo simulation with round-dependent correlation are at or above academic publication standards.

**The primary risk is over-engineering:** with 8 Tier 3 constants and complex multi-model architecture, the system may not outperform a simpler LightGBM-only model with 10 carefully selected features on a true holdout. The audit framework to detect this exists — it needs to be run and the results reported honestly.
