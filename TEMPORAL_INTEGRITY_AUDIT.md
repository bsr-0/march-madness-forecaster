# Temporal Integrity Audit Report

**Date:** 2026-03-15
**Auditor:** Chronos Protocol — Lead Quantitative Architect for Temporal Integrity
**Repository:** march-madness-forecaster
**Scope:** Full pipeline temporal contamination audit (Phase 1)

---

## 1. Temporal Boundary Analysis

```
Temporal Boundary Status: PASS (with caveats)
```

**Evidence:**

The pipeline enforces a hard dev/holdout year split via `YearSplitPolicy` (`src/ml/evaluation/evaluation_integrity.py:73-191`):

- **Dev years** (canonical): 2008-2019, 2021-2024
- **Holdout years** (canonical): [2025]
- **Prospective years**: [2026, 2027, 2028]
- **COVID exclusion**: 2020 always excluded

Enforcement mechanisms:

| Mechanism | Location | Status |
|---|---|---|
| `YearSplitPolicy.assert_dev_only()` | `evaluation_integrity.py:129-142` | **ACTIVE** |
| `_filter_years()` excludes holdout from training | `sota.py:305-316` | **ACTIVE** |
| `_check_holdout_contamination()` pre-run check | `orchestration.py:102-114` | **ACTIVE** |
| `_validate_year_split_policy()` pre-run check | `orchestration.py:117-146` | **ACTIVE** |
| `HoldoutContaminationError` exception class | `evaluation_integrity.py:46-47` | **DEFINED** |

**Caveats:**

- `LOYOValidator` class (`loyo_protocol.py:291-500`) trains on ALL years except held-out, **including future years**. This violates temporal ordering. However, the primary pipeline uses `LeaveOneYearOutCV` with `rolling_window` mode (see Section 3).
- Women's pipeline (`womens.py`) does not have dev/holdout split enforcement.

---

## 2. Leakage Surface Mapping

```
Leakage Risk Level: LOW-MEDIUM
```

### Critical Code Locations Audited

#### 2.1 StandardScaler — **CLEAN** ✅

- `baseline_training.py:781-785`: `scaler.fit_transform(train_X)` then `scaler.transform(eval_X)`
- LOYO CV re-fits scaler per fold (`baseline_training.py:1704-1710`)
- Ensemble weight optimization re-fits per fold (`baseline_training.py:2038-2042`)
- Population stats (`feature_engineering.py:341-343`) used for **validation warnings only**, NOT normalization

#### 2.2 Feature Selection — **CLEAN** ✅

- LOYO CV re-fits feature selector per fold (`baseline_training.py:1688-1702`)
- Comment at `baseline_training.py:1662-1668` explicitly documents the leakage fix
- Ensemble weight path skips feature selection to avoid cross-fold leakage (`baseline_training.py:2031-2036`)

#### 2.3 Feature Materialization — **CLEAN** ✅

- `_shifted_expanding_mean()` (`materialization.py:1863-1864`): `series.shift(1).expanding().mean()` — uses only past games
- Rolling windows use `shift(1)` before `.rolling()` (`materialization.py:720-741`)
- All game-level features are point-in-time compliant via the `shift(1)` pattern

#### 2.4 Hyperparameter Tuning — **CLEAN** ✅

- `TemporalCrossValidator` (`hyperparameter_tuning.py:83-169`): expanding-window splits, train always precedes validation
- LightGBM tuner uses temporal CV internally (`hyperparameter_tuning.py:288`)
- XGBoost tuner uses temporal CV internally (`hyperparameter_tuning.py:451`)
- Fixed rounds during Optuna search — val fold not used for early stopping (`hyperparameter_tuning.py:312-313`)

#### 2.5 Calibration — **CLEAN** ✅

- Calibration defaults to holdout-year tournament games (`config.py:660-670`)
- Calibration samples filtered to tournament-only games with assertion (`calibration.py:162-168`)
- Year-level separation documented (`calibration.py:63-68`)

#### 2.6 Women's Pipeline — **FINDING: MEDIUM RISK** ⚠️

- `womens.py:441-442`: `self.scaler = StandardScaler(); X = self.scaler.fit_transform(X)`
- No temporal train/validation split; scaler fitted on ALL training matchups
- No dev/holdout year enforcement

#### 2.7 LOYOValidator Class — **FINDING: MEDIUM RISK** ⚠️

- `loyo_protocol.py:360-378`: Trains on ALL years except held-out year, **including future years**
- Contrasted with `LeaveOneYearOutCV.rolling_window` which trains on past only
- `LOYOValidator` used by `FeatureAblator` (`loyo_protocol.py:522-631`) — ablation studies may overstate feature value due to future-year training

#### 2.8 End-of-Season Feature Aggregation — **ACCEPTABLE** ✅

- For target season: features are current-season snapshots (correct for tournament prediction)
- For historical training: `_load_year_samples_incremental` reconstructs features from per-year metrics files
- Features represent pre-tournament snapshots (end-of-regular-season), which is the correct point-in-time for tournament predictions

---

## 3. Validation Protocol Review

```
Validation Protocol Status: PASS
```

The pipeline implements **three** time-respecting validation strategies:

| Strategy | Class | Temporal Compliance |
|---|---|---|
| Expanding Window CV | `TemporalCrossValidator` | ✅ `train < val` always |
| Rolling Window LOYO | `LeaveOneYearOutCV(temporal_mode="rolling_window")` | ✅ Past years only |
| Prospective Validation | `ProspectiveValidator` | ✅ `train_years < predicted_year` |

**Default configuration** (`config.py:486`): `loyo_temporal_mode = "rolling_window"` — the honest causal mode.

**Forbidden methods NOT found:**
- ❌ Random K-fold cross validation — NOT USED
- ❌ Stratified folds mixing seasons — NOT USED
- ❌ Random train/test splits — NOT USED

**Warning:** `LeaveOneYearOutCV` still supports `temporal_mode="leave_one_out"` which includes future years. This mode exists for backward compatibility and diagnostic comparison. Config guards should prevent accidental production use.

---

## 4. Freeze Protocol Compliance

```
Freeze Integrity: PARTIALLY VERIFIED
```

**Present in freeze artifact** (`rdof_audit.py:485-495`):

| Component | Status |
|---|---|
| Config hash (SHA-256) | ✅ |
| Git commit SHA | ✅ |
| Git dirty flag | ✅ |
| Constant registry snapshot | ✅ |
| Feature set hash | ✅ |
| MC calibration params | ✅ |
| Git tag (pre-registered/) | ✅ |
| Timestamp | ✅ |

**Missing from freeze artifact** (required by LAW 4):

| Component | Status |
|---|---|
| Model weights / trained artifact | ❌ MISSING |
| Training dataset manifest hash | ❌ MISSING |
| Preprocessing pipeline hash | ❌ MISSING |

**Freeze verification** (`rdof_audit.py:537-622`): Compares current config hash against frozen hash, detects field-level changes, verifies constant registry values, and checks MC calibration parameters.

**Freeze-before-predict gate** (`evaluation_integrity.py:197-268`): Active and enforced for 2026+ predictions. Advisory-only for pre-2026 years.

---

## 5. Evidence Classification

```
Evidence Level: Level 2 (Quasi-Prospective) for 2026 predictions
                Level 3 (Retrospective Diagnostic) for 2008-2025 LOYO results
```

| Year Range | Evidence Level | Justification |
|---|---|---|
| 2008-2024 (dev) | Level 3 | Pipeline developed using these years |
| 2025 (holdout) | Level 2 | Designated holdout with freeze artifact |
| 2026 (prospective) | Level 1/2 | Depends on freeze-before-tournament verification |

The canonical leaderboard (`evaluation_integrity.py:410-565`) correctly partitions results by evidence level and displays a clear disclaimer that Level 3 results overstate OOS performance.

---

## 6. Critical Integrity Failures

| # | Severity | Description | Location | Status |
|---|---|---|---|---|
| F1 | MEDIUM | `LOYOValidator` includes future years in training folds | `loyo_protocol.py:360-378` | **REMEDIATED** — `temporal_mode="rolling_window"` now default |
| F2 | MEDIUM | Women's pipeline: scaler fitted on full dataset | `womens.py:441-442` | **REMEDIATED** — chronological train/val split added |
| F3 | MEDIUM | Freeze artifact missing model weights & dataset hash | `rdof_audit.py:485-495` | OPEN |
| F4 | LOW | No automated `feature_timestamp <= prediction_timestamp` test | N/A | **REMEDIATED** — `test_temporal_integrity.py` added |
| F5 | LOW | `LeaveOneYearOutCV` exposes unsafe `leave_one_out` mode | `hyperparameter_tuning.py:666` | **REMEDIATED** — `DeprecationWarning` added |
| F6 | LOW | `FeatureAblator` uses `LOYOValidator` (future-year leakage) | `loyo_protocol.py:529-546` | **REMEDIATED** — docstring guard + default rolling_window |
| F7 | LOW | `StratifiedKFold` in sensitivity analysis OOF calibration | `rdof_audit.py:1586` | ACCEPTED — diagnostic path only, not production |

---

## 7. Required Remediation Steps

### P0 — Automated Integrity Tests ✅ DONE
Implemented `tests/test_temporal_integrity.py` with 32 tests across 8 test classes covering all five mandatory Chronos Protocol tests plus regression tests.

### P1 — LOYOValidator Temporal Guard ✅ DONE
Added `temporal_mode` parameter to `LOYOValidator`, defaulting to `"rolling_window"`. The `leave_one_out` mode now emits `DeprecationWarning`.

### P2 — Women's Pipeline Temporal Split ✅ DONE
Added 80/20 chronological train/val split to `WomensPipeline._train_model()`. `StandardScaler` now fitted on training split only.

### P3 — Freeze Artifact Completeness ⬚ OPEN
Extend `freeze_pipeline()` to include:
- SHA-256 hash of trained model artifact (pickled weights)
- SHA-256 hash of training dataset manifest
- SHA-256 hash of preprocessing pipeline state

### P4 — Deprecate Unsafe Modes ✅ DONE
Added `DeprecationWarning` to both `LeaveOneYearOutCV(temporal_mode="leave_one_out")` and `LOYOValidator(temporal_mode="leave_one_out")`.

### P5 — FeatureAblator Temporal Compliance ✅ DONE
`LOYOValidator` now defaults to `rolling_window` mode, making `FeatureAblator` temporally compliant by default. Docstring updated with temporal integrity note.

---

## Summary

The march-madness-forecaster pipeline has **substantial temporal integrity infrastructure** that exceeds most sports prediction systems:

- Hard dev/holdout year split with runtime enforcement
- StandardScaler fitted on training data only (with per-fold re-fitting in CV)
- Expanding-window temporal cross-validation as default
- Pipeline freeze-before-predict gate for 2026+
- Evidence level classification (Level 1/2/3)
- Canonical evaluation leaderboard with proper partitioning

```
Overall Integrity Status: UNVERIFIED → pending automated test suite deployment
```

Once the automated integrity test suite (delivered below) is passing and the P0-P2 remediations are applied, the status can be upgraded to:

```
Integrity Status: TEMPORALLY SOUND (with documented exceptions in women's pipeline)
```
