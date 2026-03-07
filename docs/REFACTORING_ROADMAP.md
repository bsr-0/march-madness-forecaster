# Refactoring Roadmap: sota.py Decomposition

## Current State

`src/pipeline/sota.py` is a ~3000-line hub module that handles data loading,
feature engineering, model training, prediction generation, and reporting.
While functional, its size makes it difficult to test individual stages,
reason about data flow, and onboard new contributors.

## Proposed Decomposition

### Phase 1: Extract Data Loading (~Lines 100-800)

**Target:** `src/pipeline/data_loading.py`

- Move `_load_and_merge_data()`, `_load_torvik()`, `_load_kenpom()`,
  `_load_barttorvik_game_stats()`, and related helpers
- Define a `PipelineData` dataclass as the return type
- Keep the same file-discovery and fallback logic

**Risk:** Low — pure data loading with no model dependencies.

**Dependencies:** `src/data/ingestion/`, `src/data/normalize.py`

### Phase 2: Extract Model Training (~Lines 1800-2600)

**Target:** `src/pipeline/model_training.py`

- Move `_train_ensemble()`, `_train_spread_model()`, `_train_calibration()`
- Extract `ModelArtifacts` dataclass holding trained models + metadata
- Parameterize hyperparameters via config dict (already partially done)

**Risk:** Medium — tight coupling with feature columns and calibration flow.
Test with frozen dataset to verify identical Brier scores before/after.

**Dependencies:** `src/ml/ensemble/`, `src/ml/calibration/`

### Phase 3: Extract Prediction Generation (~Lines 2600-2800)

**Target:** `src/pipeline/prediction.py`

- Move `_generate_predictions()`, `_apply_calibration()`, `_format_submission()`
- Input: `ModelArtifacts` + tournament matchup features
- Output: submission DataFrame + prediction metadata

**Risk:** Low — mostly matrix operations and formatting.

**Dependencies:** Phase 2 artifacts

### Phase 4: Extract Reporting (~Lines 2800-3000)

**Target:** `src/pipeline/reporting.py`

- Move `_generate_report()`, `_save_artifacts()`, diagnostic summaries
- Include experiment registry logging (from C4)

**Risk:** Low — output-only, no upstream dependencies.

### Phase 5: Slim Orchestrator

**Target:** `src/pipeline/sota.py` reduced to ~500 lines

- Keep CLI argument parsing and the main `run()` orchestration
- Import and call the four extracted modules in sequence
- `run()` becomes: load → train → predict → report

## Dependency Graph

```
sota.py (orchestrator)
├── data_loading.py
│   ├── src/data/ingestion/
│   └── src/data/normalize.py
├── model_training.py
│   ├── src/ml/ensemble/
│   └── src/ml/calibration/
├── prediction.py
│   └── model_training.ModelArtifacts
└── reporting.py
    └── src/ml/evaluation/experiment_registry.py
```

## Execution Order

1. **Phase 1** (data loading) — safest, fewest dependencies
2. **Phase 4** (reporting) — output-only, easy to verify
3. **Phase 3** (prediction) — small scope, clear boundaries
4. **Phase 2** (model training) — most complex, do last
5. **Phase 5** (slim orchestrator) — final cleanup

## Verification Strategy

- Before each phase: capture LOYO Brier scores on frozen test data
- After each phase: re-run LOYO and assert identical Brier scores
- Use `tests/test_walk_forward_replay.py` determinism checks
- Run full test suite after each extraction

## Constraints

- No functional changes during refactoring (behavior-preserving only)
- Each phase should be a single PR with before/after Brier score comparison
- Maintain backwards compatibility of `python -m src.pipeline.sota` entry point
