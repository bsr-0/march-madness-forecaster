# Data Leakage Audit Report

**Date:** 2026-03-24
**Scope:** Full codebase audit of temporal leakage, target leakage, and data contamination vectors
**Auditor:** Automated (Claude)

---

## Executive Summary

The March Madness Forecaster codebase implements **robust, multi-layered leakage controls**. The production pipeline (2026) is well-protected with `strict_leakage_mode=true`, mandatory cutoff dates, point-in-time feature engineering, and frozen configuration governance. Two actionable issues were found, along with several areas warranting ongoing vigilance.

**Critical issues found:** 0
**Moderate issues found:** 2
**Low-risk observations:** 6

---

## Issue #1 (MODERATE): Calibration Fallback Fits and Evaluates on Same Data

**File:** `src/pipeline/stages/calibration.py`, lines 384-392

**Description:** When calibration has insufficient samples for a proper train/test split, the code falls back to fitting and evaluating on identical data:

```python
else:
    # Too few samples for a meaningful split; fit on all
    p_fit, p_eval = p_arr, p_arr   # <-- same data for fit and eval
    y_fit, y_eval = y_arr, y_arr
    use_oos_eval = False
```

**Impact:** Reports artificially inflated post-calibration Brier/ECE improvements in the low-sample regime. Downstream decisions (e.g., calibration method selection) based on these metrics would be unreliable.

**Recommendation:** When sample count is too low for a meaningful split, either skip calibration evaluation metrics entirely or use leave-one-out cross-validation within the limited set.

---

## Issue #2 (MODERATE): `IsotonicCalibrator.fit_calibrate()` Returns In-Sample Predictions

**File:** `src/ml/calibration/calibration.py`, lines 233-249

**Description:** The `fit_calibrate()` method fits and then returns predictions on the same data:

```python
def fit_calibrate(self, predictions, outcomes):
    self.fit(predictions, outcomes)
    return self.calibrate(predictions)  # same data used for fit
```

**Impact:** Any caller using the returned predictions for evaluation would get overfitting-inflated metrics. Currently this method does not appear to be used in the production path, but it is a latent hazard.

**Recommendation:** Add a deprecation warning or docstring caveat. Consider removing the method or renaming it to `fit_transform_insample()` to make the semantics explicit.

---

## Verified Protections (Working Correctly)

### Feature Engineering

| Protection | Location | Status |
|---|---|---|
| `shift(1)` on all rolling/expanding aggregates | `src/data/features/materialization.py:740-770` | PASS |
| Mandatory `cutoff_date` in ProprietaryMetricsEngine | `src/data/features/proprietary_metrics.py:336-381` | PASS |
| Point-in-time `compute_as_of()` in IncrementalMetricsEngine | `src/data/features/proprietary_metrics.py:2295-2671` | PASS |
| First-game rows validated as NaN for priors | `src/data/features/materialization.py:1033-1037` | PASS |
| Synthetic-date features NaN'd (rest_days, back_to_back) | `src/data/features/materialization.py:709-713` | PASS |
| M3 fallback removed (was leaking end-of-season stats) | `src/data/features/materialization.py:781-787` | PASS (fixed) |
| COVID 2020 season excluded | `src/data/features/materialization.py:163-167` | PASS |

### Training Pipeline

| Protection | Location | Status |
|---|---|---|
| Train/val split BEFORE feature selection & scaling | `src/pipeline/stages/baseline_training.py:400-470` | PASS |
| StandardScaler fit on train_X only | `src/pipeline/stages/baseline_training.py:961-1000` | PASS |
| FeatureSelector fit on train_X only | `src/pipeline/stages/baseline_training.py:874-889` | PASS |
| LOYO per-fold refitting of scaler + selector | `src/pipeline/stages/baseline_training.py:2375-2420` | PASS |
| `leave_one_out` temporal mode permanently blocked | `src/pipeline/stages/baseline_training.py` | PASS |
| Nested calibration (fit on historical, eval on current year) | `src/pipeline/stages/calibration.py:352-380` | PASS |
| Sharpener double-dip guard (alpha=1.0 fallback) | `src/pipeline/stages/calibration.py:641-656` | PASS |

### Data Ingestion & Temporal Gating

| Protection | Location | Status |
|---|---|---|
| Tournament start dates for all years (2016-2026) | `src/pipeline/config.py:44-78` | PASS |
| Regular-season mode uses `game_date < tournament_cutoff` (strict `<`) | `src/pipeline/stages/sample_loading.py:341-348` | PASS |
| Seeds zeroed before tournament cutoff | `src/pipeline/stages/sample_loading.py:416-421` | PASS |
| Massey ordinals NaN'd before tournament cutoff | `src/pipeline/stages/sample_loading.py:423-433` | PASS |
| Massey ordinals capped at Selection Sunday (133-day fallback) | `src/data/kaggle_loader.py` | PASS |
| Coach data leakage guard (career aggregates) | `src/pipeline/stages/data_loader.py:593-604` | PASS |
| Torvik scraper raises `LeakageError` in strict mode | `src/data/scrapers/torvik.py:407-431` | PASS |

### Governance & Production Locks

| Protection | Location | Status |
|---|---|---|
| `strict_leakage_mode: true` frozen in production config | `configs/production_2026.json` | PASS |
| `require_freeze_file: true` for 2026+ | `src/governance/production_validator.py` | PASS |
| Training years 2016-2024 (no 2020, no 2025) enforced | `src/governance/production_validator.py` | PASS |
| Holdout year 2025 isolated from training | `src/governance/production_validator.py` | PASS |
| `scrape_live: false` in production config | `configs/production_2026.json` | PASS |

---

## Testing Infrastructure Assessment

The codebase has ~3,000 lines of leakage-specific tests across multiple files:

| Test Suite | File | Coverage |
|---|---|---|
| Four-rule leakage framework (structural + synthetic evidence) | `tests/data_integrity/test_leakage_rules.py` | Same-game, future-opponent, post-game aggregates, tournament info |
| Point-in-time feature contracts | `tests/data_integrity/test_point_in_time_contracts.py` | Schema validation for 74+ features |
| Leakage guards (Massey, Torvik, LOYO) | `tests/test_leakage_guards.py` | Selection Sunday cap, tournament date guard, blocked modes |
| Canary tests (meta-verification) | `tests/test_leakage_canary.py` | Perfect-correlation injection, shift(1) detection |
| Temporal integrity (Chronos Protocol) | `tests/test_temporal_integrity.py` | Feature timestamps, train/test boundary, global stats |
| Leakage fix verification | `tests/test_leakage_fixes.py` | M3 fallback removal, LOYO refitting |

---

## Low-Risk Observations

### 1. Future Years (2027+) Have No Tournament Date Guards
Tournament date validation in `TOURNAMENT_START_DATES` only covers 2016-2026. New years must be manually added. The Torvik scraper silently skips the guard for unknown years.

### 2. Public Picks Scraper Has No Timestamp Validation
ESPN/Yahoo/CBS pick scrapers do not verify when picks were last updated. If picks are scraped during the tournament, they may reflect completed game results. Mitigated by `scrape_live=false` in production.

### 3. Betting Market Scraper Has No Game Status Filter
Sportsbook odds scrapers do not filter by game status (pre-game vs. live vs. final). Live odds reflect in-game information. Not enabled in production config.

### 4. Contract Tests Validate Schema, Not Feature Values
`test_point_in_time_contracts.py` validates that feature contracts exist and have correct metadata fields, but does not compute actual feature values to verify temporal correctness. Structural and synthetic tests in `test_leakage_rules.py` partially cover this gap.

### 5. No Automated Post-Hoc Holdout Contamination Audit
Tests verify that 2025 is not in `training_years` structurally, but no runtime check confirms that holdout-year games never appear in the actual training data arrays.

### 6. Circular Validation of Tier 3 Constants
The LOYO protocol acknowledges that 58 tuned constants were optimized on the same 2005-2025 data used for validation. This is inherent circularity mitigated by the Level 1 (prospective) design for 2026.

---

## Conclusion

The production pipeline has **strong leakage defenses** at every layer: data ingestion, feature engineering, training, calibration, and governance. The two moderate issues (calibration fallback and `fit_calibrate()`) are edge cases that do not affect the primary production path under normal sample sizes. The codebase demonstrates careful attention to point-in-time safety with explicit `FIX-LEAKAGE-*` comments documenting past corrections.

**Production 2026 risk assessment: LOW** — all critical temporal boundaries are enforced, strict leakage mode is enabled, and live scraping is disabled.
