# Plan: Fix Statistical and Runtime Issues for Expert Approval

## Audit Summary

Three audits identified **8 remaining issues** after the initial fix commit.
Organized by severity:

---

## BLOCKER: predict_proba_batch() Does Not Exist

**File:** `src/ml/ensemble/calibration_first.py` lines 101, 125, 133
**Problem:** CalibrationFirstPipeline calls `model.predict_proba_batch(X)` but
LightGBMRanker only exposes `.predict(X)`. Runtime crash guaranteed.
**Fix:** Replace all three `predict_proba_batch()` calls with `predict()`.

---

## HIGH: Calibration Fold Too Small (Minimum Samples Guard)

**File:** `src/pipeline/stages/baseline_training.py` line ~1079
**Problem:** `valid_samples >= 30` allows only 15 calibration samples.
ECE uses 10 bins → ~1.5 samples/bin. Temperature scaling on 15 points
is statistically meaningless.
**Fix:** Raise guard to `valid_samples >= 80` (40 cal + 40 eval minimum).

---

## HIGH: BrierLightGBMTuner Needed (Objective Alignment)

**File:** `src/ml/optimization/hyperparameter_tuning.py`
**Problem:** LightGBMTuner optimizes log loss during Optuna CV, but
BrierLightGBMRanker trains with Brier objective. Optimal regularization
(num_leaves, lambda_l1, lambda_l2) differs between loss surfaces.
`CVFoldResult.brier_score` is already computed — just not used for
optimization when Brier is the training objective.

**Fix:** Add `BrierLightGBMTuner` subclass that:
1. Uses `brier_objective` callable as the LightGBM objective during CV trials
2. Returns `mean([r.brier_score for r in cv_results])` as the Optuna metric
3. Strips `objective`/`metric` from returned best_params (BrierLightGBMRanker
   force-sets these anyway)

In `baseline_training.py`, select `BrierLightGBMTuner` when
`pipeline.config.use_brier_objective` is True.

---

## MEDIUM: Temperature Grid Excludes Upper Bound

**File:** `src/ml/ensemble/calibration_first.py` line 222
**Problem:** `np.arange(0.5, 3.0, 0.01)` excludes 3.0 (arange semantics).
**Fix:** Replace with `np.linspace(0.5, 3.0, 251)` to include both boundaries.

---

## MEDIUM: Magic Number 5.0 in Calibration Weights

**File:** `src/ml/ensemble/calibration_first.py` line 277
**Problem:** `boost = 1.0 + (1.0 - alpha) * cal_error * 5.0` has unexplained
constant. Maximum boost with alpha=0.7, cal_error=0.5 is 1.75x.
**Fix:** Extract to named constant `CAL_WEIGHT_SCALE = 5.0` with docstring
explaining the scaling rationale: maps ECE range [0, ~0.2] to meaningful
weight range [1.0, ~1.3] given typical alpha=0.7.

---

## MEDIUM: Equal-Width ECE Bins Are Suboptimal

**File:** `src/ml/ensemble/calibration_first.py` lines 176-198
**Problem:** Equal-width bins give unstable estimates in sparse tail regions.
**Fix:** Switch to equal-frequency (quantile) binning. Replace
`np.linspace(0, 1, n_bins + 1)` with `np.quantile(predictions, ...)`.
Fall back to equal-width when N < n_bins * 5 (too few for quantile).

---

## LOW-MEDIUM: Fallback Compares Pass 4 vs Pass 2 (Not vs Baseline)

**File:** `src/ml/ensemble/calibration_first.py` lines 148-160
**Problem:** Brier regression check compares final (Pass 4) against
intermediate (Pass 2) temperature-scaled predictions. If both are worse
than Pass 1 baseline, the pipeline "falls back" to model_p1 with Pass 2
temperature — which may itself be worse than raw Pass 1.
**Fix:** Compare `brier_final` against `brier_p1` (the true baseline).
On fallback, return model_p1 with temperature=1.0 (identity scaling).

---

## Implementation Order

1. **predict_proba_batch → predict** (blocker, 3 line changes)
2. **Raise calibration fold minimum to 80** (1 line change)
3. **Temperature grid → linspace** (1 line change)
4. **Equal-frequency ECE bins** (~15 lines)
5. **Fallback: compare vs Pass 1 baseline** (~5 lines)
6. **Extract CAL_WEIGHT_SCALE constant** (2 lines)
7. **BrierLightGBMTuner subclass** (~60 lines in hyperparameter_tuning.py)
8. **Wire BrierLightGBMTuner in baseline_training.py** (~10 lines)
