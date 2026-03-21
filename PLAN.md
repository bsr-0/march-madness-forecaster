# Feature Engineering & Evaluation Refactor — 15 Solutions

Based on the senior statistician's evaluation report (`artifacts/FEATURE_ENGINEERING_EVALUATION.md`)
and the statistical audit already implemented (`src/data/features/statistical_audit.py`), these 15
solutions address every identified gap. Ordered by impact on prediction accuracy.

---

## Solution 1: Mutual Information Feature Screening
**Report gap**: "No mutual information or conditional independence testing" (§3)
**File**: `src/data/features/feature_selection.py`

Add a `MutualInformationScreener` stage between correlation pruning (Stage 1) and importance
ranking (Stage 2). Uses `sklearn.feature_selection.mutual_info_classif` with k-nearest-neighbor
density estimation to detect non-linear feature-target associations that both pairwise correlation
and SHAP miss. Features with MI below a threshold (e.g., 0.01 nats) are dropped. This catches
features that have zero linear correlation with the outcome but carry non-linear signal (e.g.,
tempo interaction effects). Integrate into `FeatureSelector.fit()` as Step 1.5.

**Concrete changes**:
- New class `MutualInformationScreener` with `screen(X, y, feature_names, threshold=0.01)` method
- Add `enable_mi_screening: bool = True` and `mi_threshold: float = 0.01` to `FeatureSelector.__init__`
- Insert between correlation pruning and importance calculation in `fit()`
- Store MI scores in `FeatureSelectionResult` for diagnostics
- Tests: synthetic features with known non-linear relationships

---

## Solution 2: Distribution Shift Auto-Gating in Selection Pipeline
**Report gap**: "Distribution shift features are flagged but not removed" (§3)
**File**: `src/data/features/feature_selection.py`

Close the shift-detection loop. Currently `detect_distribution_shift()` logs warnings but takes
no action. Add a `ShiftGatedSelector` wrapper that runs shift detection between training and
validation sets, then either (a) drops features with PSI > 0.25 AND KS p < 0.05 (both tests
must agree), or (b) down-weights shifted features' importance scores by a decay factor
proportional to PSI severity. This prevents the model from relying on features whose
distribution has fundamentally changed between training and the target year.

**Concrete changes**:
- New method `_apply_shift_gating(self, X_train, X_val, feature_names, importance_scores)` on `FeatureSelector`
- Add `enable_shift_gating: bool = False` (opt-in), `shift_drop_mode: str = "downweight"` (or "drop")
- Call after importance calculation when validation data is available
- Store `shifted_features` list and `shift_actions` dict in `FeatureSelectionResult`
- Update `SOTAPipelineConfig` with corresponding flags
- Tests: inject synthetic distribution-shifted features, verify they're gated

---

## Solution 3: LOYO-Derived Ensemble Weights via Stacking
**Report gap**: "Ensemble weights hardcoded" (spread=0.45, logistic=0.20, lgb=0.20, xgb=0.15)
**File**: `src/ml/ensemble/cfa.py`, `src/pipeline/stages/baseline_training.py`

Replace fixed ensemble weights with a meta-learner trained on LOYO out-of-fold predictions.
For each LOYO fold, collect the 4 component model predictions as features and the actual
outcomes as targets. Train a constrained logistic regression (positive coefficients, sum-to-one
via simplex projection) on these meta-features. This learns optimal model weights from cross-
validated performance rather than human judgment.

**Concrete changes**:
- New class `StackingWeightOptimizer` in `src/ml/ensemble/stacking_weights.py`
  - `fit(fold_predictions: Dict[str, np.ndarray], outcomes: np.ndarray)` → weights dict
  - Constraints: weights >= 0, sum(weights) == 1 (via scipy.optimize.minimize with bounds)
  - Regularization: L2 penalty toward uniform weights to prevent overfit
- Integrate into LOYO loop in `baseline_training.py`: collect per-fold model predictions
- Fall back to current fixed weights if stacking degrades Brier (safety net)
- Store derived weights in `EvaluationReport` for audit trail
- Tests: verify weights sum to 1, verify non-negative, verify improvement over fixed

---

## Solution 4: Nested Cross-Validation for Calibration
**Report gap**: "No cross-validation of calibration" / "calibration happens on single holdout fold"
**File**: `src/ml/calibration/calibration.py`

Temperature scaling currently fits on a single calibration split. With ~440 total tournament games,
this means the calibration parameter T is estimated from ~60-120 games — high variance. Implement
nested CV: within each LOYO fold, further split the calibration data into inner folds, fit T on
each, and average. This reduces variance of the temperature estimate and prevents calibration
overfitting.

**Concrete changes**:
- New method `fit_nested(predictions, outcomes, n_inner_folds=3)` on `TemperatureScaling`
- Inner loop: 3-fold split of calibration data, fit T on each, aggregate via median
- Add `CalibrationStability` dataclass: T_mean, T_std, T_per_fold
- Warn if T_std > 0.3 (unstable calibration)
- Integrate into `CalibrationPipeline` as optional mode (`nested_cv: bool = True`)
- Tests: verify T_std reported, verify nested matches single-fold within tolerance

---

## Solution 5: Automated Tier-3 Constant Sensitivity Analysis
**Report gap**: "Sensitivity analysis not automated" for 58 RDoF constants
**File**: `src/ml/evaluation/rdof_audit.py`

The RDoF audit tracks 58 constants with `valid_range` but never programmatically tests
sensitivity. Implement `SensitivityAnalyzer` that grid-searches each Tier-3 constant (17 total)
over its valid range, re-runs the LOYO evaluation, and records the Brier score surface. Identifies
which constants the model is sensitive to (Brier change > powered ablation threshold ~0.018) and
which are inert. Constants the model is sensitive to get flagged for nested CV tuning.

**Concrete changes**:
- New class `SensitivityAnalyzer` in `src/ml/evaluation/sensitivity.py`
  - `analyze(constant_name, values, pipeline_fn)` → `SensitivityResult` (values, brier_scores, gradient)
  - `analyze_all_tier3()` → batch analysis of all 17 Tier-3 constants
- `SensitivityResult` dataclass: constant_name, tested_values, brier_scores, is_sensitive, optimal_value
- Integration: callable from CLI (`march-madness sensitivity-audit`)
- Tests: mock pipeline_fn, verify grid search logic, verify sensitivity detection

---

## Solution 6: Per-Round Feature Importance with Bootstrap CIs
**Report gap**: "Per-round importance has no confidence intervals"
**File**: `src/ml/evaluation/feature_explainability.py`

Currently `PerRoundImportance` computes mean importance per round with no uncertainty estimates.
With only ~16 games per round per year, single-point estimates are noisy. Add bootstrap CIs
(200 resamples) for per-round SHAP importance. This reveals which features are reliably important
in late rounds vs. just noise — critical for understanding whether expensive features (e.g.,
`three_pt_variance`) actually contribute where Kaggle weights are highest (F4: 16x, NCG: 32x).

**Concrete changes**:
- Modify `PerRoundImportance.compute()` to accept `n_bootstrap=200`
- For each round, resample games with replacement, recompute SHAP importance
- Return `RoundFeatureImportance` dataclass with: mean, ci_lower, ci_upper per feature per round
- Add `is_significant(feature, round)` method: True if CI excludes zero
- Tests: synthetic round data, verify CI coverage

---

## Solution 7: Brier-Optimal Calibration Method Selector
**Report gap**: "No adaptive calibration method selection" — only temperature scaling used
**File**: `src/ml/calibration/calibration.py`

Production locks `calibration_method: "temperature"`, but the research pipeline should
automatically select the best calibration method. Implement `CalibrationMethodSelector` that
fits temperature scaling, Platt scaling, and isotonic regression on a calibration fold, evaluates
each on a validation fold (Brier + ECE), and selects the best. Uses inner CV to prevent selection
bias.

**Concrete changes**:
- New class `CalibrationMethodSelector` in `src/ml/calibration/method_selector.py`
  - `select(predictions, outcomes, methods=["temperature", "platt", "isotonic"])` → best method + metrics
  - Inner 3-fold CV: fit on 2/3, evaluate on 1/3, average Brier across folds
  - Returns `CalibrationSelectionResult`: method_name, brier_scores, ece_scores, selected_method
- Integrate into SOTA pipeline when `calibration_method: "auto"` (new config option)
- Production path unchanged (stays `"temperature"`)
- Tests: verify method with best Brier is selected, verify production path unaffected

---

## Solution 8: Interaction Feature Validation with Bootstrap Marginal Brier
**Report gap**: "Document marginal Brier improvement for interaction features with bootstrap CIs" (§5 rec 5)
**File**: `src/data/features/statistical_audit.py`

The absolute feature validation (SA-2) exists but interaction features (7 dims: tempo_interaction,
style_mismatch, etc.) have no marginal validation. Add `validate_interaction_features()` that
computes bootstrap CI on Brier improvement from adding the 7 interaction features. If CI includes
zero, the interactions are not earning their dimensionality cost and should be flagged.

**Concrete changes**:
- New function `validate_interaction_features(X_base, X_interactions, y, interaction_names, n_bootstrap=200)`
  in `statistical_audit.py`
- Returns `InteractionValidationResult`: marginal_brier_improvement, ci_lower, ci_upper, per_feature_mi
- Also compute MI of each interaction with target to identify which interactions carry signal
- Integrate as SA-3 step in `FeatureSelector.fit()` (diagnostic, no auto-drop)
- Tests: synthetic interactions (known signal vs. noise), verify CI coverage

---

## Solution 9: SHAP Importance Hyperparameter Alignment
**Report gap**: "ImportanceCalculator uses fixed LightGBM config which may rank features differently than production" (§5 rec 4)
**File**: `src/data/features/feature_selection.py`

The `ImportanceCalculator` uses hardcoded LightGBM params (31 leaves, 0.05 LR, 200 rounds) while
production uses different params (8 leaves, 0.05 LR, 500 rounds). Feature importance rankings
depend heavily on tree structure — a 31-leaf model finds very different splits than an 8-leaf
model. Align the importance calculator's model config with production hyperparameters.

**Concrete changes**:
- Add `lgb_params: Optional[Dict]` parameter to `ImportanceCalculator.__init__`
- When provided, use these params for SHAP and permutation importance computation
- Default: load from production config or use current hardcoded values as fallback
- Add `FeatureSelector.__init__` parameter `importance_lgb_params: Optional[Dict] = None`
- Pass through to `ImportanceCalculator`
- Pipeline integration: `baseline_training.py` passes production LightGBM params
- Tests: verify different params produce different importance rankings

---

## Solution 10: Multiple Comparison Correction for Model/Feature Tests
**Report gap**: "No correction for multiple comparisons"
**File**: `src/ml/evaluation/statistical_tests.py`

When comparing K models or testing K feature additions, K(K-1)/2 pairwise tests inflate false
positive rate. Add Holm-Bonferroni correction to all multi-test scenarios. Also add
Diebold-Mariano test for probabilistic forecast comparison (more appropriate than paired t-test
for Brier scores).

**Concrete changes**:
- New function `holm_bonferroni_correction(p_values: List[float], alpha=0.05)` → adjusted p-values
- New function `diebold_mariano_test(losses_a, losses_b, horizon=1)` → DM statistic, p-value
- New class `MultiModelComparison` that wraps pairwise tests with Holm correction
  - `compare_all(model_predictions: Dict[str, np.ndarray], outcomes)` → comparison matrix
- Integrate into LOYO evaluation: when comparing ensemble vs. individual models
- Tests: verify Holm correction controls FWER, verify DM test against known results

---

## Solution 11: Adaptive Feature Selection Threshold via Elbow Detection
**Report gap**: Importance threshold 0.05 is hardcoded with no data-driven justification
**File**: `src/data/features/feature_selection.py`

The `importance_threshold=0.05` cutoff is arbitrary. Implement elbow detection on the sorted
importance curve to automatically find the natural break point where importance drops off.
Uses the Kneedle algorithm (Satopaa et al., 2011) — finds the point of maximum curvature
in the sorted importance scores.

**Concrete changes**:
- New function `detect_importance_elbow(importance_scores: List[float], sensitivity=1.0)` → threshold
- Uses second-derivative (discrete curvature) on sorted importance values
- Falls back to hardcoded 0.05 if no clear elbow detected (monotonically decreasing)
- Add `adaptive_threshold: bool = True` to `FeatureSelector.__init__`
- When enabled, overrides `importance_threshold` with elbow-detected value
- Log detected threshold for audit
- Tests: synthetic importance curves with known elbows

---

## Solution 12: Feature Stability Across LOYO Folds
**Report gap**: "No stability metrics across folds" for feature importance
**File**: `src/ml/evaluation/feature_explainability.py`, `src/data/features/feature_selection.py`

Bootstrap stability (FIX #6) tests within a single training set. But feature selection should
also be stable across LOYO folds — if a feature is selected in fold 2018 but dropped in fold
2022, it's unreliable. Implement cross-fold stability: run feature selection independently on
each LOYO fold, compute overlap coefficient (Jaccard index) between selected sets, and flag
features that appear in < 5/7 folds.

**Concrete changes**:
- New function `compute_cross_fold_stability(fold_selected_features: List[List[str]])` in `feature_selection.py`
  - Returns: per-feature fold-presence count, Jaccard indices between all fold pairs, mean overlap
- New dataclass `CrossFoldStabilityResult`: feature_fold_counts, mean_jaccard, unstable_features
- Integrate into LOYO loop in `baseline_training.py`: collect selected features per fold
- Report in `EvaluationReport` as `feature_stability` field
- Flag if mean_jaccard < 0.7 (warning: feature selection is unstable)
- Tests: synthetic fold selections with known overlap

---

## Solution 13: Conditional Feature Importance (SHAP Interactions)
**Report gap**: "No partial dependence plots" / "No ALE" — limited global explanation methods
**File**: `src/ml/evaluation/feature_explainability.py`

SHAP main effects miss interaction effects. Add SHAP interaction value computation for top-K
features (K=10). This reveals whether feature pairs have synergistic effects (e.g., does
`three_pt_variance` only matter when `seed_diff` is large?). Uses `shap.TreeExplainer` with
`feature_perturbation="interventional"` for causal interaction estimates.

**Concrete changes**:
- New method `compute_interaction_effects(model, X, feature_names, top_k=10)` on `MatchupExplainer`
- Returns `InteractionEffects` dataclass: interaction_matrix (k×k), top_interactions list
- Each interaction: (feature_a, feature_b, interaction_strength, direction)
- Add `enable_interaction_shap: bool = False` to config (expensive: O(n × k²))
- Integrate into explainability report generation
- Tests: synthetic features with known interaction, verify detection

---

## Solution 14: Ensemble Diversity Metrics
**Report gap**: "No explicit diversity metrics beyond effective_model_count"
**File**: `src/ml/ensemble/cfa.py`

The ensemble combines 4 models but never measures whether they're making diverse errors.
If all models make the same mistakes, the ensemble provides no benefit. Add diversity metrics:
disagreement measure (fraction of games where models disagree), correlation of errors, and
Kuncheva's diversity index. Low diversity → the ensemble is wasting compute on redundant models.

**Concrete changes**:
- New class `EnsembleDiversity` in `src/ml/ensemble/diversity.py`
  - `compute(model_predictions: Dict[str, np.ndarray], outcomes: np.ndarray)` → `DiversityResult`
  - Metrics: disagreement_rate, error_correlation_matrix, kuncheva_index, q_statistic
  - `is_diverse()` → True if kuncheva_index > 0.3
- Integrate into LOYO evaluation loop: compute diversity after collecting fold predictions
- Report in `EvaluationReport`
- If not diverse, log warning recommending model architecture changes
- Tests: identical predictions → zero diversity, orthogonal errors → max diversity

---

## Solution 15: Kaggle Round Weight Validation & Round-Weighted Brier
**Report gap**: "Round weights not validated" — hardcoded without runtime verification
**File**: `src/pipeline/config.py`, `src/ml/evaluation/loyo_protocol.py`

Kaggle round weights (R64:1, R32:2, S16:4, E8:8, F4:16, NCG:32) are hardcoded but never validated
against actual Kaggle rules. Also, LOYO evaluation computes flat Brier but never round-weighted
Brier — yet Kaggle scores with round weights. Add round-weighted Brier as a first-class metric
alongside flat Brier, and validate round weights against a canonical source.

**Concrete changes**:
- New function `compute_round_weighted_brier(predictions, outcomes, rounds, weights)` in `loyo_protocol.py`
- Returns `RoundWeightedBrier` dataclass: weighted_brier, per_round_brier, per_round_n
- Add to LOYO evaluation: compute both flat and round-weighted Brier per fold
- Add `KAGGLE_ROUND_WEIGHTS_2026` constant with source annotation and year tag
- Validation: warn if weights don't sum to expected total given bracket structure
- Store round-weighted Brier in `EvaluationReport.global_metrics`
- Tests: verify weighted formula, verify edge cases (missing rounds)

---

## Implementation Priority

| Priority | Solutions | Rationale |
|----------|-----------|-----------|
| P0 (Critical) | 1, 2, 3, 4 | Directly fix the DoF/sample tension & ensemble opacity |
| P1 (High) | 5, 6, 7, 8 | Close evaluation gaps identified in report |
| P2 (Medium) | 9, 10, 11, 12 | Improve rigor of selection & testing |
| P3 (Nice-to-have) | 13, 14, 15 | Enhanced diagnostics & Kaggle alignment |

## Impact Estimate

Combined, these 15 solutions address:
- **3/3 dimensionality concerns** from §3 (MI screening, shift gating, interaction validation)
- **5/5 recommendations** from §5 (PCA→MI, shift loop, SHAP alignment, marginal Brier, diversity)
- **All 7 critical gaps** from the ML infrastructure analysis
- **DoF budget compliance**: MI + shift gating + elbow detection → fewer, better features
- **Evaluation rigor**: nested CV, multiple comparison correction, round-weighted Brier
- **Transparency**: every solution produces auditable diagnostics stored in EvaluationReport
