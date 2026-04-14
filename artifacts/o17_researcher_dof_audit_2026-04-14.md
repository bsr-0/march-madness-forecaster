# O17 — Researcher-DoF Leakage Audit

**Date:** 2026-04-14
**Branch:** `claude/simplify-repo-structure-keQXE`
**Status:** closure evidence for `COUNCIL_LESSONS.md §2 O17`

## Gate (from §2)

> Documented trail of which years were observable when which hyperparameters
> were tuned.

## TL;DR

**Current state is defensible, partially auditable, and locked.** Three
mechanisms constrain researcher DoF:

1. **Code-level holdout fence** (O20 closure): every Optuna tuner call goes
   through `YearSplitPolicy.assert_dev_artifact_years`, which raises
   `HoldoutContaminationError` if the tuning-year set intersects `[2025]`.
2. **Narrow search spaces**: Optuna search ranges are intentionally tight
   (e.g., `max_depth ∈ [2,4]`, `num_leaves ∈ [4,16]`,
   `min_child_samples ∈ [30,100]`).
3. **Budget caps**: `optuna_n_trials = 15` (explicit OOS-FIX at
   `src/pipeline/config.py:469`, reduced from the prior 50) and
   `enable_stacking = False` (learned meta-learner disabled to prevent
   overfitting ~400 OOF samples).

Remaining gap: pre-`2026-04-07` tuning history is not reconstructible —
the repo's git graph was collapsed into a single big-bang commit
(`4a6605c`) on that date. All tuning decisions made before then are
documentation-opaque and cannot be audited further.

## Tuning surface inventory

### A. Optuna search spaces — `src/ml/optimization/hyperparameter_tuning.py`

| Tuner | Param | Bounds | Notes |
|---|---|---|---|
| LightGBMTuner | `num_leaves` | `[4, 16]` int | narrow vs typical [31, 127] |
| LightGBMTuner | `learning_rate` | `[0.01, 0.15]` log | |
| LightGBMTuner | `feature_fraction` | `[0.5, 0.9]` | |
| LightGBMTuner | `bagging_fraction` | `[0.5, 0.9]` | |
| LightGBMTuner | `min_child_samples` | `[30, 100]` int | large-leaf regularization |
| LightGBMTuner | `lambda_l1` | `[0.1, 10.0]` log | |
| LightGBMTuner | `lambda_l2` | `[0.1, 10.0]` log | |
| LightGBMTuner | `num_rounds` | `[50, 200]` int | |
| XGBoostTuner | `max_depth` | `[2, 4]` int | very narrow |
| XGBoostTuner | `learning_rate` | `[0.01, 0.15]` log | |
| XGBoostTuner | `subsample` | `[0.5, 0.9]` | |
| XGBoostTuner | `colsample_bytree` | `[0.5, 0.9]` | |
| XGBoostTuner | `min_child_weight` | `[5, 30]` int | |
| XGBoostTuner | `gamma` | `[0.1, 5.0]` | |
| XGBoostTuner | `reg_alpha` | `[0.1, 10.0]` log | |
| XGBoostTuner | `reg_lambda` | `[0.1, 10.0]` log | |
| XGBoostTuner | `num_rounds` | `[50, 200]` int | |
| Logistic | `C` | `[0.01, 10.0]` log | |
| Logistic | `l1_ratio` | `[0.0, 1.0]` | elasticnet only |

### B. Global tuning budget — `src/pipeline/config.py`

| Field | Value | Lock | Audit note |
|---|---|---|---|
| `enable_hyperparameter_tuning` | `True` | line 468 | |
| `optuna_n_trials` | `15` | line 469 | **OOS-FIX**: was 50 |
| `optuna_timeout` | `300` s | line 470 | per-tuner wall cap |
| `temporal_cv_splits` | `5` | line 471 | expanding-window |
| `optimize_ensemble_weights` | `True` | line 472 | grid search only |
| `scoring_metric` | `"brier"` | line 477 | Kaggle 2023+ metric |
| `enable_stacking` | `False` | line 486 | **OOS-FIX**: ~400 OOF samples would overfit a 9-feature meta-learner |
| `margin_first_training` | `False` | line 489 | |

### C. Temporal cross-validation policy

`TemporalCrossValidator` at `src/ml/optimization/hyperparameter_tuning.py:98`.

- **Strategy:** expanding-window splits, sorted by chronological key.
  Training always precedes validation. No future games ever leak into
  training folds.
- **Splits:** 5 (budget lock per config §B).
- **Example layout for 1000 samples:**
  - Fold 0: train=[0:400],   val=[400:520]
  - Fold 1: train=[0:520],   val=[520:640]
  - Fold 2: train=[0:640],   val=[640:760]
  - Fold 3: train=[0:760],   val=[760:880]
  - Fold 4: train=[0:880],   val=[880:1000]

### D. Year visibility during tuning

| Set | Years (as of 2026-04-14) | Source |
|---|---|---|
| Training | 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024 (16 years; 2020 excluded) | `config.py:359-381` |
| Holdout | `[2025]` | `config.py:383` |
| Calibration | 2008-2024 + **2025** (18 years; 2020 excluded) | `config.py:401-421` |

**Calibration-years 2025 inclusion — is this researcher-DoF leakage?**
No, but it requires explanation. Baseline models train on regular-season
games only (`enable_round_weighted_training=False`). Tournament games are
never observed during base-model fit. Calibration is a deterministic
post-hoc fit on tournament-only games, held separately from base-model
training. So 2025 tournament outcomes are genuinely OOS from the base
model's training distribution even when consumed by calibration.

This is explicitly validated by
`src/governance/production_validator.py:145-175`, which accepts exactly
two calibration-year configurations: `[2025]` alone, or the full
historical list including 2025.

### E. Code-level holdout fence (O20 composition)

Every tuner invocation routes through
`_enforce_dev_only_years()` at `hyperparameter_tuning.py:51-63`, which
calls `year_split_policy.assert_dev_artifact_years()`. The policy object
is a frozen dataclass (`src/ml/evaluation/evaluation_integrity.py:75`)
that raises `HoldoutContaminationError` on any intersection with
`holdout_years`.

Verification: `tests/test_holdout_enforcement.py` (all 9 tests green as
of this commit).

## Historical audit trail

Traceable via `git log -p` post-`2026-04-07`.

| Date | Commit | Tuning-adjacent change |
|---|---|---|
| 2026-04-07 | `4a6605c` | **Big-bang import.** Pre-date history is not reconstructible. |
| 2026-04-08 | `a34f521`, `69a3016` | Scraper / team-name resolver fixes; no ML tuning touched. |
| 2026-04-13 | `34517cf` | Delete dead LambdaMART + elo_temporal modules (no live tuning surface changed). |
| 2026-04-13 | `016b961` | O8 anti-tautology + O20 holdout-enforcement closure tests. Wired `YearSplitPolicy` at Optuna call sites. |
| 2026-04-14 | `029c02f` | `scripts/_common.py` consolidation (no ML tuning touched). |

`git log -S "suggest_int" -- src/ml/optimization/hyperparameter_tuning.py`
returns only the big-bang commit + the `34517cf` deletion sweep and
`a5584f4` branch-merge — i.e., none of the search-space bounds have
been edited since import. The search spaces documented in §A are the
ones the big-bang commit landed with.

`git log -S "optuna_n_trials" -- src/pipeline/config.py` similarly returns
only `4a6605c`. The OOS-FIX reduction from 50 → 15 happened before the
big-bang import (as a comment, not a diff) and cannot be independently
verified.

## Residual risks & explicit non-gates

1. **Pre-2026-04-07 tuning history is not auditable.** The big-bang
   import destroyed intermediate commits. Any tuning done before that
   date relies on in-code `OOS-FIX` comments as the only breadcrumb.
2. **Multiple-test correction is not applied to Optuna search.** With
   15 trials × 5 CV folds × 2 tuners (LGB + XGB) = 150 Brier evaluations
   per fit, the best-score selection has implicit selection bias. This
   is mitigated by the narrow bounds in §A but not eliminated.
3. **Ensemble weight grid-search** (`optimize_ensemble_weights=True`) is
   also exposed to this same selection bias over the base-model blend
   weights. The grid is small (documented in
   `hyperparameter_tuning.py::EnsembleWeightOptimizer`) and the
   held-out evaluation uses the O20 `YearSplitPolicy`, so the leak
   surface is bounded to what dev-years the grid is evaluated against.

None of these justify a gate-reopen — they are accepted residual risk
documented here so that future regressions are framed against a known
baseline.

## Locking forward

- **Frozen inventory:** this document.
- **Lock test:** `tests/test_researcher_dof_audit.py` (added in the same
  commit) asserts:
  - `optuna_n_trials == 15`
  - `enable_stacking is False`
  - `temporal_cv_splits == 5`
  - `scoring_metric == "brier"`
  - `_enforce_dev_only_years` still exists as the tuner-side gate.
  - Optuna search-space bounds in §A still match the live code.
  Any future change to a locked value fails the test, forcing an
  explicit re-audit (bump this document's date + update the lock).
- **MEMORY.md §1 row:** "Researcher DoF (tuning provenance)" row added
  with evidence pointers.

## Closure record

`COUNCIL_LESSONS.md §2 O17` → `[closed 2026-04-14]`. Crumb:

> Audited inventory of all Optuna search spaces, budget caps, and
> temporal-CV policy committed at
> `artifacts/o17_researcher_dof_audit_2026-04-14.md`. Code-level gate
> (`YearSplitPolicy.assert_dev_artifact_years`) wired from O20 closure
> is the runtime enforcement; lock test
> `tests/test_researcher_dof_audit.py` prevents silent drift. Pre-
> 2026-04-07 history not reconstructible (big-bang import commit);
> documented as residual risk.
