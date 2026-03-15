# Phase 2: Production Model Simplification

**Date:** 2026-03-15  
**Status:** Implemented  
**Version:** `spread_logistic_temp_v1`

## Summary

Phase 2 simplifies the production model stack to reduce overfitting risk and make gains easier to trust. The core change is a strict separation between **production** (minimal, auditable) and **experimental** (full research) pipeline modes.

## Production Stack

| Component | Production | Experimental |
|-----------|-----------|-------------|
| SpreadRegressor | **Active** (weight=1.0) | Active (weight=0.55) |
| Logistic Regression | **Active** (weight=0.0, earns via gate) | Active (weight=0.15) |
| LightGBM Classifier | Blocked | Active (weight=0.15) |
| XGBoost Classifier | Blocked | Active (weight=0.15) |
| TemperatureScaling | **Sole calibrator** | Available |
| TournamentSigmaCalibrator | Blocked | Available |
| RoundSpecificCalibrator | Blocked | Available |
| Stacking | Blocked | Available |

## Key Changes

### 1. Pipeline Mode Separation (`config.py`)
- New `pipeline_mode` field: `"production"` (default) or `"experimental"`.
- Production mode enforces the sanctioned stack; experimental preserves all legacy models.
- `experimental_enable_lgb_classifier` and `experimental_enable_xgb_classifier` toggles added (default `False`).
- Locked production path validates pipeline_mode and experimental classifier flags.

### 2. Production Baseline Spec (`production_baseline.py`)
- Frozen dataclass `ProductionBaselineSpec` defines the single source of truth for:
  - Sanctioned models, calibration layer, ensemble weights
  - Admission gate thresholds
  - Deprecated model/calibrator lists
- Version: `spread_logistic_temp_v1`

### 3. Model Stack Simplification (`baseline_training.py`)
- LightGBM/XGBoost classifier training gated behind `pipeline_mode == "experimental"`.
- Logistic Regression always trained (removed fallback-only guard).
- Stacking disabled in production mode.
- TournamentSigmaCalibrator fitting disabled in production mode.
- Fixed-weight ensemble uses production baseline weights in production mode.

### 4. Ensemble Layer (`margin_first_ensemble.py`)
- `MarginFirstEnsemble` gains `production_mode` parameter (default `True`).
- Production mode defaults to spread-only weights (`spread=1.0, logistic=0.0`).
- `set_models()` raises `ValueError` if LGB/XGB models are passed in production mode.
- Docstrings updated with Phase 2 calibration semantics (sigma vs TemperatureScaling).

### 5. Model Registry (`model_registry.py`)
- `ModelFamilySpec` gains `production_active` and `experimental_only` flags.
- LightGBM and XGBoost marked `experimental_only=True`.
- New helpers: `get_production_models()`, `get_experimental_only_models()`.

### 6. Feature Family Ablation (`loyo_protocol.py`)
- New `FeatureFamily` dataclass with `family_type`, `masking_policy`, `neutral_values`.
- 7 default families: seed_priors, elo_ratings, four_factors, roster_continuity, massey_ordinals, recency_form, public_picks.
- `FeatureAblator.ablate_families()` method for family-level masked ablation.
- `validate_family_coverage()` checks all features are assigned to families.
- Retrain-based ablation deferred to Phase 3.

### 7. Hard Admission Gate (`admission_gate.py`)
- Multi-condition gate for production promotion:
  1. Mean OOS Brier improvement > threshold (default 0.0)
  2. Fold improvement rate >= threshold (default 0.60)
  3. Calibration degradation <= threshold (default 0.01)
- Emits JSON report artifacts for auditability.
- Integrated with `AuthorityMatrix` (`admission_gate_pass` condition).

## Files Changed

| File | Change |
|------|--------|
| `src/pipeline/production_baseline.py` | **New** — Production baseline spec |
| `src/pipeline/config.py` | Added pipeline_mode, production settings, admission thresholds |
| `src/pipeline/stages/baseline_training.py` | Pipeline mode guards on deprecated models/calibrators |
| `src/ml/ensemble/margin_first_ensemble.py` | Production mode, new weights, model guards |
| `src/ml/ensemble/model_registry.py` | Production/experimental flags, new helpers |
| `src/ml/evaluation/loyo_protocol.py` | FeatureFamily, ablate_families(), validate_family_coverage() |
| `src/ml/evaluation/admission_gate.py` | **New** — Hard admission gate |
| `src/governance/authority_matrix.py` | Added admission_gate_pass to promote_model rule |
| `tests/test_production_baseline.py` | **New** — 25 tests for baseline, config, ensemble, registry |
| `tests/test_admission_gate.py` | **New** — 13 tests for admission gate |
| `tests/test_loyo_protocol.py` | Extended with 15 family ablation tests |
| `tests/test_ev_mode_config.py` | Fixed: enforce_production_path=False for EV mode tests |
| `tests/test_mode_gating.py` | Fixed: enforce_production_path=False for EV mode tests |
| `tests/test_directive_v7_improvements.py` | Fixed: enforce_production_path=False for overlap test |

## Migration Notes

- **No breaking changes for default production runs.** Default `pipeline_mode="production"` activates automatically.
- To use deprecated classifiers, set `pipeline_mode="experimental"` and the corresponding `experimental_enable_*` flags.
- Existing `model_complexity` setting still controls feature set size; `pipeline_mode` controls which models are trained.
- The admission gate does not retroactively apply to existing production weights — it gates future promotions only.

## Forbidden Actions (per spec)

- Re-optimizing blend weights beyond dev evidence
- Adding new exotic models without admission gate approval
- Restoring deprecated models to production without gate passage
- Soft/audit-only admission gates (must be hard programmatic block)
