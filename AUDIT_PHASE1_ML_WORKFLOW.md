# Phase 1 Audit: Existing ML Workflow

> **STATUS (2026-04-01): HISTORICAL.** This audit predates the simplification to `model_complexity="simple"` (7-feature logistic regression, no ensemble). The LightGBM/XGBoost/stacking ensemble described below is no longer the production default. See README.md for current architecture.

**Date:** 2026-03-27
**Scope:** Full parse of repo + EXPERIMENT_WORKFLOW_PLAN.md
**Verdict:** Substantial infrastructure exists but has critical gaps in objective alignment, loop integration, and leakage surface area.

---

## 1. Existing Loops Inventory

### 1.1 Training Loops (Production Path)

| Component | File:Line | Loss/Objective | CV Method | Hyperparameter Selection |
|-----------|-----------|---------------|-----------|------------------------|
| LightGBM | `baseline_training.py:1100-1200` | `binary_logloss` | LOYO (8 folds) | Optuna (minimizes mean log-loss) |
| XGBoost | `baseline_training.py:1280-1340` | `binary:logistic` | LOYO (8 folds) | Optuna (minimizes mean log-loss) |
| Logistic Regression | `baseline_training.py:1341-1410` | Cross-entropy | LOYO (8 folds) | Small grid (C values) |
| BMA Ensemble | `bma.py:74-210` | Log-likelihood (EM) | On LOYO predictions | EM convergence |
| Stacking Ensemble | `stacking_weights.py:71-147` | **Brier score** | On LOYO predictions | SLSQP optimization |
| Temperature Calibration | `forecaster/calibration.py:79-96` | Log-loss | On LOYO predictions | `minimize_scalar` |

### 1.2 Research/Experiment Loops

| Component | File | Status |
|-----------|------|--------|
| `ExperimentWorkflow` (5-phase) | `EXPERIMENT_WORKFLOW_PLAN.md` | **PLANNED ONLY** - `experiment_workflow.py` does not exist |
| `ResearchLoop` | `src/ml/research/research_loop.py` | Single-param sweeps, placeholder evaluator |
| `ExperimentScheduler` | `src/research/experiment_scheduler.py` | Variant generation + queue, no execution engine |
| `HypothesisRegistry` | `src/ml/research/hypothesis_registry.py` | Tracking only, no execution |
| `TrainingWindowOptimizer` | `src/ml/evaluation/training_window_optimizer.py` | Functional but not wired to production |
| `AblationStudy` | `src/ml/evaluation/ablation.py` | Functional but not wired to production |
| Hyperparameter Tuning (Optuna) | `src/ml/optimization/hyperparameter_tuning.py` | Functional, called from `baseline_training.py` |

### 1.3 Loop Integration Assessment

```
ExperimentWorkflow (planned) ──────── NOT IMPLEMENTED
         │
ResearchLoop ──────────────────────── PLACEHOLDER evaluator, single-param only
ExperimentScheduler ───────────────── Generates variants, NO execution engine
HypothesisRegistry ────────────────── Tracks hypotheses, NO auto-execution
         │
         ▼
TrainingWindowOptimizer ───────────── Standalone, not called from pipeline
AblationStudy ─────────────────────── Standalone, not called from pipeline
         │
         ▼
Production Pipeline (sota.py) ─────── Single-pass: train → calibrate → ensemble → simulate
         │
Governance Gates ──────────────────── Config validation + hash freeze ONLY
                                      (Admission/Promotion/Baseline gates NOT wired in)
```

**Key finding:** There is no end-to-end optimization loop. The production pipeline is single-pass. Research infrastructure exists as disconnected components that have never been orchestrated together.

---

## 2. Leakage Risks

### 2.1 CONFIRMED SAFE

- **Point-in-time feature engineering:** Features use tournament cutoff dates per season (`src/data/features/`). Season stats are computed only from games before the tournament.
- **Temporal CV:** LOYO cross-validation (train on N-1 years, test on held-out year). No random splits.
- **2025 holdout isolation:** 2025 is excluded from training years `[2016-2019, 2021-2024]` and only used for advisory validation.
- **No 2020 contamination:** COVID-cancelled season excluded from training.
- **Leakage detection enforcement:** `LeakageError` exception exists and is raised on violations.

### 2.2 MODERATE RISKS

| Risk | Location | Severity | Details |
|------|----------|----------|---------|
| **Calibration on LOYO predictions** | `calibration.py:79-96` | MODERATE | Temperature scaling is fit on the same LOYO out-of-fold predictions used for model selection. This is not nested CV — calibration parameters are optimized on the evaluation set. |
| **Feature selection timing** | `EXPERIMENT_WORKFLOW_PLAN.md:56` | LOW (mitigated) | Plan explicitly states "Feature selection is NOT searched" and uses fixed domain-knowledge features. But `enable_feature_selection` exists as a config flag — if enabled, it could leak. |
| **Stacking on LOYO predictions** | `stacking_weights.py:71-147` | MODERATE | Stacking weights optimized on same LOYO predictions used for evaluation. Should use nested LOYO or a separate inner fold. |
| **Optuna trial count** | `hyperparameter_tuning.py:313-415` | LOW | Optuna runs many trials on LOYO folds. Selection bias from many trials is partially mitigated by conservative estimate (LOYO mean + 1 SE) but not formally corrected. |

### 2.3 STRUCTURAL CONCERN

**Calibration is not nested.** The pipeline:
1. Trains models via LOYO → gets out-of-fold predictions
2. Fits temperature scaling on those same predictions
3. Evaluates calibrated predictions on those same predictions

This means the reported calibration quality is optimistic. A proper nested CV would hold out a separate fold for calibration fitting. The `CalibrationMethodSelector` with nested CV exists (`src/ml/calibration/method_selector.py`) but is **not used in the production path**.

---

## 3. Objective Function Misalignment

### 3.1 The Core Problem

```
TRAINING          →  Log-loss (all 3 models)
HYPERPARAMETER    →  Log-loss (Optuna metric)
CALIBRATION       →  Log-loss (temperature scaling)
ENSEMBLE (BMA)    →  Log-likelihood (consistent with log-loss) ✓
ENSEMBLE (Stack)  →  Brier score ✗ INCONSISTENT
EVALUATION GATE   →  Brier < 0.190 AND Log-loss < 0.560
KAGGLE SCORING    →  Brier score (mean squared error on probabilities)
BRACKET EV        →  ESPN points (10/20/40/80/160/320 per round)
```

### 3.2 Specific Misalignments

#### A. Train on Log-loss, Evaluate on Brier
- **All three base models** minimize log-loss during training (`hyperparameter_tuning.py:321,358`)
- **Kaggle's actual metric** is Brier score (mean squared probability error)
- Log-loss and Brier are correlated but not identical: log-loss penalizes confident wrong predictions much more harshly. A model optimized for log-loss will be more conservative (closer to 0.5) than one optimized for Brier.
- **Impact:** Probably small (<0.002 Brier) but systematic. The Brier-optimal objective exists (`brier_objective.py:29-56`) but is **disabled in production** (`use_brier_objective` not set in config).

#### B. Stacking Weights (Brier) vs. Model Training (Log-loss)
- `stacking_weights.py:159`: Stacking minimizes Brier score
- But the models feeding into stacking were trained on log-loss
- This creates a subtle mismatch where the ensemble weighting optimizes a different surface than the individual models

#### C. No Bracket-EV-Aware Training
- ESPN scoring: R64=10, R32=20, S16=40, E8=80, F4=160, CHAMP=320
- This means getting the championship game right is worth 32x a first-round game
- But training treats all 63 tournament games equally (uniform log-loss)
- The simulation (`pool_competition.py:479-758`) computes bracket EV **post-hoc** but never feeds back into training
- **Impact:** Large for bracket pool competitions. The model has no incentive to be more accurate on late-round games where point values are highest.

#### D. Mislabeled Metrics
- `baseline_training.py:1178,1322,1375`: Reports `"best_brier"` but the values are actually log-loss from Optuna
- This is a bug that could mislead analysis

### 3.3 Alignment Summary

| Pair | Aligned? | Gap Size |
|------|----------|----------|
| Training ↔ BMA Ensemble | YES | — |
| Training ↔ Stacking Ensemble | NO | Small (both are proper scoring rules) |
| Training ↔ Calibration | YES (both log-loss) | — |
| Training ↔ Kaggle Metric (Brier) | NO | Small-medium (systematic conservatism) |
| Training ↔ Bracket EV | NO | **LARGE** (uniform weighting vs. exponential round values) |
| Calibration ↔ Bracket EV | NO | Medium (log-loss calibration ≠ bracket-optimal) |

---

## 4. What Must Be Replaced vs. Reused

### 4.1 REUSE (Solid foundations)

| Component | Why Keep |
|-----------|----------|
| LOYO cross-validation framework | Gold standard for small-N temporal data |
| Point-in-time feature engineering | Correctly prevents future data leakage |
| LightGBM/XGBoost/LogisticRegression base models | Well-suited for N≈400 tabular data |
| BMA ensemble weighting | Principled, consistent with log-loss training |
| Monte Carlo bracket simulation | Sound methodology (50k runs) |
| ExperimentRegistry (JSONL ledger) | Good audit trail, Directive V7 compliant |
| Multi-metric gate (Brier + LL + divergence) | Proper dual-metric quality control |
| Production governance (hash freeze, config validation) | Strong production safety |
| Pool competition scoring | Correct ESPN scoring implementation |
| HypothesisRegistry | Good research tracking framework |
| AblationStudy | Sound component significance testing |

### 4.2 REPLACE / FIX

| Component | Issue | Action |
|-----------|-------|--------|
| **ResearchLoop** | Placeholder evaluator, single-param only | Replace with `ExperimentWorkflow` (implement the plan) |
| **ExperimentScheduler** | No execution engine | Wire into ExperimentWorkflow or replace |
| **Stacking weights objective** | Brier when models train log-loss | Switch to log-loss OR switch model training to Brier |
| **"best_brier" mislabeling** | Reports log-loss as Brier | Fix labels in `baseline_training.py:1178,1322,1375` |
| **Non-nested calibration** | Calibration fit on evaluation set | Use `CalibrationMethodSelector` with nested CV, or add inner fold |

### 4.3 ADD (Missing capabilities)

| Capability | Why Needed | Priority |
|------------|-----------|----------|
| **Brier-objective training** | Align training with Kaggle metric | HIGH |
| **Nested calibration CV** | Prevent calibration leakage | HIGH |
| **ExperimentWorkflow implementation** | Orchestrate the 5-phase search | HIGH |
| **Bracket-EV loss function** | Align training with bracket pool goal (if EV mode matters) | MEDIUM |
| **Round-weighted evaluation** | Evaluate accuracy on late rounds separately | MEDIUM |
| **Proper experiment execution engine** | Connect scheduler → evaluator → registry | MEDIUM |
| **Selection bias correction** | Formal adjustment beyond "mean + 1 SE" | LOW |
| **Governance gate integration** | Wire Admission/Promotion gates into pipeline | LOW |

---

## 5. Gap Summary (Hard List)

### Critical Gaps
1. **No optimization loop exists.** The 5-phase `ExperimentWorkflow` is a plan document only — never implemented. Production is single-pass.
2. **Objective misalignment:** Training minimizes log-loss; Kaggle scores on Brier; bracket pools score on ESPN points. No component bridges these gaps.
3. **Calibration leakage:** Temperature scaling is fit and evaluated on the same LOYO predictions (not nested).

### Significant Gaps
4. **Stacking/training metric inconsistency:** Stacking optimizes Brier while models optimize log-loss.
5. **Mislabeled metrics:** `"best_brier"` fields contain log-loss values.
6. **Disconnected research infrastructure:** ExperimentRegistry, HypothesisRegistry, ExperimentScheduler, Admission/Promotion gates all exist but none are wired into the training pipeline.
7. **No bracket-EV feedback:** Simulation results never influence model training or calibration.

### Minor Gaps
8. **Optuna selection bias:** Many trials tested but no formal multiple-comparisons correction on hyperparameter search.
9. **Governance gates not enforced at runtime:** Admission gate, promotion gate, baseline guard exist but only as CLI utilities.
10. **Brier-optimal objective disabled:** `brier_objective.py` exists and works but `use_brier_objective` is not set in production config.

---

## 6. Recommendations for Phase 2+

1. **Phase 2 (Implement ExperimentWorkflow):** Build the 5-phase orchestrator from `EXPERIMENT_WORKFLOW_PLAN.md`. This is the highest-impact missing piece.
2. **Phase 3 (Fix Objective Alignment):** Enable `use_brier_objective` for LightGBM; evaluate impact. Consider Brier-optimal calibration.
3. **Phase 4 (Nest Calibration):** Add inner CV fold for calibration parameter selection.
4. **Phase 5 (Bracket-EV Mode):** If bracket pool EV matters, add round-weighted loss or bracket-level optimization.
5. **Phase 6 (Wire Governance):** Connect admission/promotion gates to the experiment workflow.
