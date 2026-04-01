# Feature Engineering & Selection Evaluation

> **STATUS (2026-04-01): HISTORICAL.** Production now uses 7 fixed features (`SIMPLE_FEATURE_SET`), not learned feature selection. A baseline experiment confirmed additional features add no value on tournament data (BSS ≈ 0). See README.md.

Senior statistician's review of the feature reduction, engineering, and selection
pipeline. Written March 2026 for reference by future agents and reviewers.

---

## 1. Feature Engineering (79-dim vector)

### Strengths

- **Domain-grounded features**: Four Factors, RAPM, Elo, SOR, WAB all have
  published statistical justification in sports analytics literature.
- **Point-in-time safety** enforced via tournament cutoff dates per season
  (`materialization.py`, `proprietary_metrics.py`). `LeakageError` raised on
  violations. Tested with synthetic canary injection.
- **Missing-data indicator flags** for sparse features (`preseason_ap_rank`,
  coach metrics) let the model discount imputed defaults rather than treating
  them as signal (`feature_engineering.py:89-95`). Note: these were later
  removed from the final matchup vector (lines 850-854) because they encoded
  scraper availability artifacts, not basketball signal — a correct decision.
- **Explicit redundancy removal** with documented correlation coefficients
  (`feature_engineering.py:38-65`). Eleven features removed with clear
  justification (e.g., `barthag` is a monotonic transform of `adj_off/adj_def`,
  `true_shooting_pct` has r=0.92 with `efg_pct + ft_rate` combo).

### Concerns

- **Effective degrees of freedom**: 79 features for ~440 tournament games
  (7 LOYO folds x 63 games) gives ~0.18 features/sample. Harrell's
  rule-of-thumb targets 10-20 events per predictor — with ~220 "wins" in
  440 games, the data can support ~11-22 predictors, not 79. The codebase
  acknowledges this (`loyo_protocol.py:9-29` discloses DoF/sample =
  58/440 ~ 0.13 for tuned constants), but the raw feature count exceeds
  it further.
- **Redundancy audit was partly manual**: The `REMOVED_REDUNDANCIES` list
  documents 11 removed features with correlation rationale. Question: was
  this exhaustive? Were all pairwise correlations >0.85 checked
  systematically, or only known algebraic ones? The automated
  `CorrelationPruner` (threshold=0.85) handles this downstream, but the
  manual list predates it.
- **Absolute-level features** (`ABSOLUTE_LEVEL_FEATURE_NAMES`, line 81):
  averaging both teams' features for "game quality context" adds 5 more
  dimensions. Marginal Brier improvement with bootstrap CI should be
  documented.

### Feature Categories (79 total)

| Category | Count | Examples |
|----------|-------|---------|
| Core Efficiency | 3 | `adj_off_eff`, `adj_def_eff`, `adj_tempo` |
| Four Factors (off + def) | 8 | `efg_pct`, `to_rate`, `orb_rate`, `ft_rate` + defensive mirrors |
| Player Metrics | 6 | `total_rapm`, `top5_rapm`, `bench_rapm`, `total_warp`, `roster_continuity`, `transfer_impact` |
| Experience/Depth | 3 | `avg_experience`, `bench_depth`, `injury_risk` |
| Volatility/Entropy | 4 | `lead_volatility`, `entropy`, `lead_sustainability`, `comeback_factor` |
| Shot Quality | 2 | `xp_per_poss`, `shot_distribution` |
| Schedule Strength | 4 | `sos_adj_em`, `sos_opp_o`, `sos_opp_d`, `ncsos_adj_em` |
| Resume Metrics | 4 | `wab`, `sor`, `wab_poisson`, `luck` |
| Momentum | 1 | `momentum` (last-10-game rolling AdjEM delta) |
| Variance/Upset Risk | 2 | `three_pt_variance`, `pace_adj_variance` |
| Ratings | 1 | `elo_rating` |
| Shooting | 4 | `free_throw_pct`, `three_pt_pct`, `three_pt_rate`, `three_pt_regression` |
| Ball Movement | 2 | `assist_to_turnover`, `assist_rate` |
| Defensive Disruption | 2 | `steal_rate`, `block_rate` |
| Opp Shot Selection | 2 | `opp_two_pt_pct`, `opp_three_pt_attempt_rate` |
| Conference | 1 | `conference_adj_em` |
| Win/Resume | 3 | `win_pct`, `elite_sos`, `q1_win_pct` |
| Style/Risk | 2 | `foul_rate`, `defensive_xp_per_poss` |
| Context | 5 | `rest_days`, `top5_minutes_share`, `preseason_ap_rank`, `coach_tournament_exp`, `coach_tournament_win_rate` |
| Graph SOS | 2 | `pagerank_sos`, `multi_hop_sos` |
| Win Quality | 3 | `best_win_percentile`, `paper_tiger_score`, `dominance_ratio` |
| Per-Stage Coaching | 3 | `coach_f4_appearances`, `coach_e8_appearances`, `coach_s16_appearances` |
| Other | 7+ | `seed_strength`, `conf_tourney_champ`, `neutral_site_win_pct`, etc. |

### Matchup-Level Feature Construction

The final model input is a matchup vector (~91 dims), not a team vector:

```
[diff_features (79) | absolute_features (~5) | interactions (7)]
```

- **Differential**: Team1 - Team2 for all 79 features
- **Absolute-level (FIX #4)**: `(Team1 + Team2) / 2` for `adj_off_eff`,
  `adj_def_eff`, `sos_adj_em`, `elo_rating`, `win_pct`
- **Interactions (7)**: `tempo_interaction`, `style_mismatch`,
  `seed_em_residual`, `sos_seed_interaction`,
  `three_pt_var_seed_interaction`, `seed_interaction`, `seed_diff`

---

## 2. Feature Selection Pipeline

**Location**: `src/data/features/feature_selection.py`

The pipeline applies six stages in order:

### Stage -2: Cluster Pre-selection (FIX 2.2)

Known-redundant feature groups are pruned before automated selection:

- **SOS Cluster**: Keep `sos_adj_em` + `elite_sos`, drop `sos_opp_o`,
  `sos_opp_d`, `ncsos_adj_em`
- **RAPM Cluster**: Keep `top5_rapm` + `bench_rapm`, drop `total_rapm`,
  `backcourt_rapm`, `frontcourt_rapm`
- Reduces ~8 features before automated stages.

### Stage -1: Near-Zero Variance Pruning

- **Class**: `NearZeroVariancePruner`, threshold=1e-7
- Removes constant/near-constant features
- Must run before VIF to avoid `VIF=inf` on constant features

### Stage 0: VIF Pruning

- **Class**: `VIFPruner`, threshold=10.0, max_drops=10
- Iteratively drops feature with highest VIF until all VIF <= threshold
- Catches 3+ feature linear dependencies that pairwise correlation misses
- VIF computed via OLS: `VIF_j = 1 / (1 - R^2_j)`

### Stage 1: Correlation Pruning

- **Class**: `CorrelationPruner`, threshold=0.85
- **Tie-breaking (FIX #5)**: Changed from variance-based to
  target-correlation-based. When two features are highly correlated, keeps
  the one with higher `|corr(feature, y)|`. Falls back to variance when
  y not provided.

### Stage 2: Importance Ranking

- **Class**: `ImportanceCalculator`
- **Methods (priority order)**:
  1. SHAP TreeExplainer via LightGBM (weight=2.0) — out-of-fold, 3-fold
  2. Permutation importance via LightGBM (weight=1.0) — neg_brier_score metric
  3. Absolute correlation with target (weight=1.0, **suppressed when SHAP succeeds** per FIX #7)
- Features below importance_threshold=0.05 are dropped (min_features=20 enforced)

### Stage 3: Bootstrap Stability Filter (FIX #6)

- 10 bootstrap iterations, resample with replacement
- Feature must be selected in >= 80% of bootstrap runs to survive
- Addresses feature selection instability (Meinshausen & Buhlmann, 2010)
- Safeguard: won't drop below min_features

### Post-Selection Validation

- **Condition number**: SVD-based, thresholds <30 (OK), 30-100 (monitor), >100 (severe)
- **Residual VIF**: Recomputed on final feature set, flags if still >10
- Results stored in `FeatureSelectionResult.multicollinearity_warning`

### Pipeline Defaults

```python
correlation_threshold = 0.85
min_features = 20
max_features = 50
importance_threshold = 0.05
enable_vif_pruning = True
vif_threshold = 10.0
enable_stability_filter = True
stability_threshold = 0.80
n_bootstrap = 10
```

---

## 3. Feature Reduction / Dimensionality

### What exists

- **Distribution shift detection** (`detect_distribution_shift`) using three
  complementary tests per feature:
  - PSI (Population Stability Index): <0.10 OK, 0.10-0.25 moderate, >0.25 significant
  - KS test (Kolmogorov-Smirnov): non-parametric CDF divergence
  - Standardized mean shift: location change in units of train std
  - Feature flagged if ANY test exceeds threshold
  - This is excellent practice borrowed from credit scoring — genuinely unusual
    in sports analytics.

### What is notably absent

- **No formal dimensionality reduction** (PCA, sparse PCA, factor analysis)
  in production. PCA is imported (`feature_selection.py:29`) but not used in
  the main pipeline. With 79 features and ~440 evaluation samples, PCA or
  partial least squares projecting into 15-25 dimensions would bring the
  DoF/sample ratio into an acceptable range.
- **No mutual information or conditional independence testing**: The pipeline
  uses correlation (linear) and SHAP (model-based), but no non-parametric
  measure of feature-target association independent of the model class.
  Mutual information would catch non-linear relationships that both miss.
- **Distribution shift features are flagged but not removed**: The
  `detect_distribution_shift` function logs warnings but does not
  automatically drop shifted features. A statistician would ask whether
  this should feed into the selection pipeline.

---

## 4. Evaluation Infrastructure

### Cross-Validation: LOYO (Leave-One-Year-Out)

- **Location**: `src/ml/evaluation/loyo_protocol.py`
- **Folds**: 7 years (2018, 2019, 2021, 2022, 2023, 2024, 2025). 2020 excluded (COVID).
- **Total eval games**: ~440 (7 x 63)
- **Honest evidence classification**: Level 3 (Retrospective Diagnostic) —
  all 58 tuned constants were optimized on the same data evaluated by LOYO,
  creating circular validation. Explicitly disclosed.
- **Powered ablation threshold**: ~0.018 Brier (computed from SE ~ 0.009,
  t_0.05 ~ 1.943 with 7 folds). Most individual feature additions/removals
  are statistically undetectable.

### Feature Importance Evaluation

- **LOYO Feature Importance** (`feature_explainability.py`): Importance
  averaged across LOYO folds with stability scores (fraction of folds
  where feature ranks in top-K).
- **Per-Round Feature Importance**: Stratified by tournament round to answer
  "which features matter in R64 vs Final Four?"
- **SHAP explanations per matchup**: `MatchupExplainer` provides per-game
  feature attribution for interpretability.

### Statistical Testing for Feature/Model Comparison

- **Paired Brier t-test** (`statistical_tests.py`): Paired t-test on
  per-game squared errors with Cohen's d.
- **Permutation test**: 10k permutations, non-parametric null distribution.
- **Bootstrap model comparison**: 95% CI on Brier difference (A - B).
  If CI excludes zero, significant.
- **Brier decomposition**: Reliability (calibration) - Resolution
  (discrimination) + Uncertainty (inherent).
- **Brier Skill Score vs seeds**: BSS = 1 - Brier(model) / Brier(seed baseline).
  Red flag if BSS < 0.05.

### RDoF Audit

- **Location**: `src/ml/evaluation/rdof_audit.py` (3,361 lines)
- **58 tracked constants** in three tiers:
  - Tier 1 (8): Externally derived (HCA = 3.75 pts, seed prior slope, etc.)
  - Tier 2 (11): Structurally constrained (decay bounds, monotonicity)
  - Tier 3 (39): Freely tuned via LOYO; tuning/eval circularity disclosed
- **Sensitivity analysis**: Perturb each constant, recompute metrics.
  "Keep default" if within 0.005 of best.

---

## 5. The Fundamental Statistical Tension

### The core problem

The codebase is in a regime where:

1. **79 raw features** (reduced to ~50 after selection) for **~440 eval games**
2. **58 tuned constants** optimized on the same data evaluated by LOYO
3. **DoF/sample ratio**: 58/440 ~ 0.13 (target: <0.01 per Harrell's rule)
4. **Powered ablation threshold**: ~0.018 Brier — most individual changes
   are undetectable

### What the codebase gets right

- **Radical honesty**: Level 3 evidence classification, explicit circular
  validation disclosure, DoF/sample ratio computed and documented.
- **Bootstrap stability filtering**: Directly addresses selection instability.
- **Multi-stage pruning**: VIF + correlation + importance is proper
  multi-stage, catching different redundancy types.
- **Distribution shift monitoring**: PSI + KS + mean shift is production-grade.
- **Post-selection collinearity validation**: Condition number + residual VIF
  after all pruning stages.

### Recommendations for improvement

1. **Dimensionality reduction**: Apply PCA or factor analysis to compress to
   ~20 dimensions where DoF/sample ratio is manageable. Or use Lasso/elastic
   net with nested CV as a single-step alternative.
2. **Mutual information screening**: Add non-parametric feature-target
   association measure to catch non-linear relationships.
3. **Close the shift-detection loop**: Feed `detect_distribution_shift`
   results into the selection pipeline (auto-drop or down-weight shifted
   features).
4. **Match SHAP importance hyperparameters to production**: The
   `ImportanceCalculator` uses fixed LightGBM config (31 leaves, 0.05 LR,
   200 rounds) which may rank features differently than the production model's
   hyperparameters.
5. **Document marginal Brier improvement** for absolute-level features and
   interaction features with bootstrap CIs.

---

## 6. Key File Reference

| Purpose | File | Key Lines |
|---------|------|-----------|
| Feature vector definition | `src/data/features/feature_engineering.py` | 75, 413-671, 674-776 |
| Redundancy documentation | `src/data/features/feature_engineering.py` | 38-65 |
| Feature dimension constant | `src/data/features/feature_engineering.py` | 75 (`TEAM_FEATURE_DIM = 79`) |
| Feature selection pipeline | `src/data/features/feature_selection.py` | 866-1032 |
| VIF pruning | `src/data/features/feature_selection.py` | 86-162 |
| Correlation pruning | `src/data/features/feature_selection.py` | 435-533 |
| Importance calculator | `src/data/features/feature_selection.py` | 536-810 |
| Bootstrap stability filter | `src/data/features/feature_selection.py` | 970-1105 |
| Distribution shift detection | `src/data/features/feature_selection.py` | 276-432 |
| Post-selection validation | `src/data/features/feature_selection.py` | 224-273 |
| Cluster pre-selection | `src/data/features/feature_selection.py` | 813-863 |
| Matchup features | `src/data/features/feature_engineering.py` | 797-870 |
| LOYO protocol | `src/ml/evaluation/loyo_protocol.py` | 43-56, 58-100 |
| RDoF audit | `src/ml/evaluation/rdof_audit.py` | 84-250 |
| Statistical tests | `src/ml/evaluation/statistical_tests.py` | 21-285 |
| Feature explainability | `src/ml/evaluation/feature_explainability.py` | 138-704 |
| Leakage enforcement | `src/data/features/materialization.py` | throughout |
| Production config locks | `src/governance/production_validator.py` | 31-75 |
