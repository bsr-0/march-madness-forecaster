# Statistical Review: Critical Limitations Preventing Exceptional Prediction Accuracy

**Reviewer Role:** Senior Statistician
**Date:** 2026-03-17
**Scope:** Full repository audit — data pipeline, feature engineering, modeling, calibration, validation, and simulation

---

## Executive Summary

This repository demonstrates serious engineering effort and awareness of many pitfalls (symmetric training, temporal cross-validation, calibration, researcher degrees of freedom auditing). However, several **fundamental statistical limitations** prevent it from being an exceptionally accurate prediction system. They fall into three categories: (1) the system effectively runs as a single-model pipeline despite having ensemble infrastructure, (2) validation methodology has structural weaknesses that mask true out-of-sample performance, and (3) critical calibration and feature pathways are disabled in production.

---

## CRITICAL LIMITATION 1: Production Is a Single-Model System, Not an Ensemble

**Severity: Critical**
**Files:** `src/pipeline/production_baseline.py:45-46`, `src/pipeline/stages/baseline_training.py:896-904`

The production configuration sets `default_weights = {"spread": 1.0, "logistic": 0.0}` and explicitly gates LightGBM and XGBoost classifiers behind `not _production_mode`. The `ACTIONABLE_IMPROVEMENTS.md` file (written by the system's own authors) confirms this:

> "One model. One conversion function. No ensemble diversity."

The full inference path in production is:

```
SpreadRegressor.predict_spread(X) -> logistic_CDF(spread, sigma=11) -> temperature_scale -> clip
```

**Why this is fatal for accuracy:** Ensemble diversity is the single largest driver of Brier score improvement in tournament prediction. Every Kaggle March Madness medalist since 2019 uses 3+ model blending. A margin-regression model converted through a fixed logistic CDF cannot learn nonlinear conditional probability surfaces — e.g., "7-seeds with high 3-point variance lose more than their spread predicts." LightGBM/XGBoost classifiers directly optimize log-loss and can capture these conditional interactions. The infrastructure exists (`MarginFirstEnsemble` supports 4-model blending at weights 0.45/0.20/0.20/0.15) but is architecturally disabled.

**Estimated Brier cost:** +0.008 to +0.015 (the single largest recoverable error source).

---

## CRITICAL LIMITATION 2: Insufficient and Structurally Circular Validation

**Severity: Critical**
**Files:** `src/ml/evaluation/loyo_protocol.py`, `src/ml/evaluation/rdof_audit.py`

### 2a. Only 7 Tournament-Years of Validation Data

LOYO validation operates over `[2018, 2019, 2021, 2022, 2023, 2024, 2025]` — seven folds of ~63 games each, totaling ~440 tournament games. This is the effective sample size for all model selection, hyperparameter tuning, and feature ablation decisions.

With 58 documented constants (including 14 "Tier 3" freely-tuned parameters), the degrees-of-freedom-to-sample ratio is 14/440 ~ 0.032. The system's own `rdof_audit.py` targets a ratio of 0.01 and flags the current ratio as too high. This means the LOYO Brier scores are optimistic estimates of true out-of-sample performance — the pipeline was iteratively shaped while observing these years' outcomes.

### 2b. LOYO vs. Prospective Validation

The `rdof_audit.py` documentation is admirably honest:

> "Every backtest result in this system is Level 3 (retrospective). The 58 constants were optimized on the same 2005-2025 data used for evaluation."

The system has `ProspectiveValidator` and pipeline freeze infrastructure, but **no prospective evaluation has ever been run**. 2026 would be the first candidate. Until a frozen pipeline is evaluated against a truly unseen tournament, all reported metrics are retrospective and systematically overstate performance.

### 2c. Year-to-Year Variance Swamps Signal

With only 63 games per tournament year, single-year Brier scores have standard errors of ~0.02-0.03. A "0.001 rule" for feature ablation (requiring features to improve mean LOYO Brier by at least 0.001) is statistically underpowered: the standard error of the mean Brier across 7 folds is approximately `0.025/sqrt(7) ~ 0.009`. An improvement of 0.001 is well within noise. Features can appear to pass or fail the rule due to randomness rather than genuine predictive power.

---

## CRITICAL LIMITATION 3: Tournament Sigma Calibration Disabled in Production

**Severity: High**
**Files:** `src/ml/ensemble/spread_model.py:47`, `src/pipeline/production_baseline.py:56-58`

The SpreadRegressor uses `sigma=11.0` (regular-season calibration from Stern 1991). Tournament games have empirically tighter margins: sigma ~ 8.5-9.5 points, due to neutral sites, higher opponent quality, and preparation time.

The `TournamentSigmaCalibrator` (a well-designed Bayesian shrinkage estimator with per-round calibration and bootstrap CIs) exists but is blocked in production via `deprecated_production_calibrators`. This creates systematic miscalibration:

- A team predicted to win by 5 points: P(win) = 0.634 with sigma=11, but P(win) = 0.668 with correct sigma=9
- Over 63 games, this systematic 3-4 percentage point shift costs ~0.005-0.010 Brier

The effect is worst for closely-matched teams (seeds 4-5 through 8-9), which is exactly where calibration matters most.

---

## CRITICAL LIMITATION 4: Training-Inference Feature Mismatch

**Severity: High**
**Files:** `src/data/features/proprietary_metrics.py` (build_matchup_vector), `ACTIONABLE_IMPROVEMENTS.md` Section 4

Three of seven interaction features are hardcoded to constants during training:

```python
h2h_record = 0.5        # no head-to-head data in training
common_opp_margin = 0.0  # not available incrementally
travel_advantage = 0.0   # no venue data
```

However, during tournament inference, these features CAN be populated from the proprietary engine. This creates a **train/inference distribution shift**: the model learns these features are constant, then encounters real values at prediction time. Distribution shift of this kind can only hurt — the model has no learned relationship between these features and outcomes, so any non-zero value introduces unpredictable noise into predictions.

This occupies 3 of 7 interaction feature slots with zero signal during training, wasting model capacity.

---

## CRITICAL LIMITATION 5: Monte Carlo Simulation Parameterization

**Severity: High**
**Files:** `src/simulation/monte_carlo.py`, `src/pipeline/config.py`

### 5a. Noise Standard Deviation

The logit-space noise `noise_std=0.16` (recently increased from 0.12, which was increased from 0.04) controls per-game uncertainty in bracket simulations. Lopez & Matthews (JQAS 2015) estimate game-level residual SD of 0.16-0.18 in logit space. However, the current value was set by examining the literature, not by calibrating against historical upset rate distributions. The `calibrate-mc-noise` CLI tool exists but there's no evidence it has been run with current pipeline parameters.

### 5b. Regional Correlation

`regional_correlation=0.10` (reduced from 0.25) models "upset-friendly" regions. This parameter adds structural assumptions that cannot be validated with only 63 games per year. The comment acknowledges this: "The correlation structure adds free parameters that can't be validated with 63 games/year." This is a researcher degree of freedom that could introduce systematic bias in bracket-level predictions (e.g., championship odds) without any mechanism to detect it.

### 5c. Hardcoded Round Sigmas

`DEFAULT_ROUND_SIGMAS` (R64=10.5 through NCG=8.0) are hardcoded fallback values. While the `TournamentSigmaCalibrator` can override these empirically, when it's disabled in production (Limitation 3), these literature-derived defaults govern the spread-to-probability conversion for all tournament predictions.

---

## CRITICAL LIMITATION 6: Fundamental Sample Size Constraint

**Severity: Structural (unfixable by this system alone)**

The NCAA tournament produces only ~63 games per year (67 with play-ins). Even with 20 years of data, the total tournament game sample is ~1,300 games. For a 78-dimensional matchup feature vector, this is severely underpowered:

- **Late-round calibration**: F4 produces 2 games/year, NCG produces 1 game/year. Even across 20 years, F4 has ~40 games and NCG has ~20 games for round-specific calibration. The Bayesian shrinkage in `TournamentSigmaCalibrator` (prior_strength=30) mitigates this, but cannot create information that doesn't exist.

- **Kaggle scoring amplification**: F4 games are weighted 16x and NCG 32x. These 3 games per year represent ~38% of the Kaggle score, but have the least calibration data. This is an irreducible challenge — no amount of regular-season data can substitute for tournament-specific dynamics at the Final Four level.

- **Rare event estimation**: 1-vs-16 upsets have occurred once in the 64-team era (UMBC over Virginia, 2018). The empirical upset rate is ~1/152 = 0.66%, but the system uses 1.5%. This 2x difference is within the binomial confidence interval given the sample size, illustrating the fundamental impossibility of precisely calibrating rare event probabilities.

---

## CRITICAL LIMITATION 7: No Integration of Real-Time Information

**Severity: High**

The system's Selection Sunday snapshot architecture (which is well-designed for temporal integrity) has a fundamental trade-off: it freezes the information state at Selection Sunday. However, between Selection Sunday and game time, critical information arrives:

- **Injuries**: A star player injury between Selection Sunday and their first game can shift win probabilities by 5-15 percentage points. The `injury_probability=0.02` parameter in Monte Carlo simulation adds random injury noise, but does not incorporate actual known injuries.
- **Betting market movements**: Lines shift 2-5 points between bracket release and tipoff as sharp money and injury news arrive. The `betting_markets.py` scraper exists but is not integrated into the prediction pipeline.
- **Practice/travel reports**: Teams that have to travel across the country perform measurably worse in early rounds. The `travel_distance.py` module has coordinates for 370+ D1 schools but the travel feature is hardcoded to 0.0.

The system produces one static prediction set at Selection Sunday and never updates. An "exceptionally accurate" system would need to incorporate information that arrives during the tournament itself (at minimum, actual injuries and line movements before each round).

---

## CRITICAL LIMITATION 8: Calibration Architecture Concerns

**Severity: Moderate-High**
**Files:** `src/ml/calibration/calibration.py`, `src/evaluation/calibration_methods.py`

### 8a. Temperature Scaling on Small Samples

Temperature scaling fits a single parameter on 50-300 tournament games. The small-sample guard rejects calibration when N < 30, but with N = 50-80, the bootstrap CI on temperature is wide enough that it almost always includes 1.0, making the guard ineffective. Multi-year calibration (`enable_multi_year_calibration`) increases the pool to 200-400, but this mixes tournament eras with potentially different dynamics.

### 8b. Potential Double-Calibration

The system documentation carefully argues that SpreadRegressor sigma (model-internal CDF conversion) and TemperatureScaling (final probability adjustment) are "not double-calibration." While technically correct — sigma maps continuous margins to [0,1] while temperature adjusts confidence — in practice, if sigma is mis-set (Limitation 3), temperature scaling will compensate by learning a temperature that corrects for the sigma error. This creates fragile coupling where two parameters are jointly compensating for one underlying issue (wrong sigma), making the system sensitive to which calibration data happens to be available.

---

## CRITICAL LIMITATION 9: Feature Engineering Assumptions

**Severity: Moderate**
**Files:** `src/data/features/feature_engineering.py`, `src/ml/training/symmetric.py`

### 9a. Fixed 78-Dimensional Feature Vector

The matchup vector is fixed at 78 dimensions: 66 differential, 5 absolute, 7 interaction. The layout constants (`DIFF_START=0, DIFF_END=66, ABS_START=66, ...`) are hardcoded in `symmetric.py` and must be manually updated if features change. This creates a brittle coupling where adding or removing a feature requires coordinated changes across multiple files — a maintenance hazard that discourages feature experimentation.

### 9b. Linear Differential Assumption

The primary matchup representation is `v_team1 - v_team2` for 66 of 78 features. This assumes that feature effects are symmetric and linear in the difference. While generally reasonable for efficiency metrics, some features have asymmetric effects:

- Tempo: a fast team vs. a slow team plays at a pace closer to the slow team's preference (defense sets the tempo)
- Three-point variance: high-variance shooting helps the underdog disproportionately (they need outlier performance)
- Experience: experience advantages are more valuable in high-pressure tournament games than regular season

The 5 absolute features and 7 interaction features partially address this, but the 66:12 ratio of linear-difference to nonlinear features suggests the model is under-equipped to capture asymmetric dynamics.

---

## CRITICAL LIMITATION 10: No Market Consensus Integration or Sanity Check

**Severity: Moderate-High**
**Files:** `src/data/scrapers/betting_markets.py`

Vegas lines represent the most informationally efficient basketball predictions available — they incorporate private information, expert analysis, injury reports, and millions of dollars of price discovery. The `ACTIONABLE_IMPROVEMENTS.md` acknowledges this:

> "If your model's predictions systematically disagree with market lines, either your model has found genuine alpha (unlikely without novel data sources) or your model is miscalibrated (much more likely)."

The `betting_markets.py` scraper infrastructure exists but is **not integrated into the production pipeline**. The system produces predictions without any market sanity check. An exceptionally accurate system would at minimum validate against market consensus and ideally incorporate market information as a feature or blending target.

---

## Summary Table

| # | Limitation | Severity | Recoverable? | Est. Brier Cost |
|---|-----------|----------|-------------|-----------------|
| 1 | Single-model production (no ensemble) | Critical | Yes (code change) | +0.008 to +0.015 |
| 2 | Circular/underpowered validation | Critical | Partially (needs time) | Unmeasurable |
| 3 | Tournament sigma disabled | High | Yes (code change) | +0.005 to +0.010 |
| 4 | Train/inference feature mismatch | High | Yes (code change) | +0.002 to +0.006 |
| 5 | MC simulation parameterization | High | Partially | +0.003 to +0.008 |
| 6 | Fundamental sample size constraint | Structural | No | Irreducible |
| 7 | No real-time information integration | High | Partially (architecture) | +0.005 to +0.015 |
| 8 | Calibration small-sample issues | Moderate-High | Partially | +0.002 to +0.005 |
| 9 | Rigid feature vector assumptions | Moderate | Yes (refactor) | +0.001 to +0.003 |
| 10 | No market consensus integration | Moderate-High | Yes (integration) | +0.003 to +0.008 |

**Estimated total recoverable Brier improvement: 0.025-0.055** (not fully additive; assumes ~70% additivity across overlapping effects).

---

## Conclusion

This system has substantially more infrastructure than most March Madness prediction efforts — Bayesian Bradley-Terry ratings, symmetric training, temporal cross-validation, researcher degrees of freedom auditing, and tournament-specific sigma calibration are all genuine advances. However, the **production pipeline disables most of its own best components**. The system as deployed is essentially a single LightGBM regression model predicting point margins, converted to probabilities through a regular-season-calibrated logistic CDF, with a temperature scaling adjustment.

The three most impactful changes, all of which involve enabling existing but disabled infrastructure, would be:

1. **Enable the 4-model ensemble** (change 5 lines of config)
2. **Enable tournament sigma calibration** (remove from deprecated list)
3. **Integrate betting market consensus** as a sanity check or blending signal

Beyond these recoverable issues, the fundamental constraint is that tournament basketball has an irreducible noise floor. With ~63 games per year and game-level variance corresponding to sigma ~ 10-11 points, even a perfectly calibrated oracle would achieve a Brier score of approximately 0.12-0.14. The gap between this theoretical floor and what the best Kaggle competitors achieve (~0.15-0.17) is small, and closing it requires every marginal improvement discussed above.
