# Experiment Workflow Plan: Pre-Production Optimization

## Context

The user wants a systematic experiment workflow that exhaustively explores configuration options before freezing production settings. Currently, the codebase has several *individual* pieces of experiment infrastructure:

- **ResearchLoop** (`src/ml/research/research_loop.py`): Single-parameter sweeps with placeholder evaluator, limited to 9 continuous params
- **TrainingWindowOptimizer** (`src/ml/evaluation/training_window_optimizer.py`): Window size evaluation via LOYO
- **AblationStudy** (`src/ml/evaluation/ablation.py`): Component contribution measurement
- **CalibrationMethodSelector** (`src/ml/calibration/method_selector.py`): Nested CV calibration selection
- **ExperimentRegistry** (`src/ml/evaluation/experiment_registry.py`): JSONL experiment ledger
- **HypothesisRegistry** (`src/ml/research/hypothesis_registry.py`): Hypothesis tracking

**What's missing:** A unified, multi-phase workflow that ties all of these together into a principled search strategy a senior sports statistician would approve of. The existing `ResearchLoop` only sweeps continuous params one-at-a-time and uses a placeholder evaluator. There's no orchestrated search over structural choices (model complexity, feature sets, ensemble methods), no budget-aware scheduling, and no final recommendation report.

**Protocol v2 alignment:** The workflow must respect the multi-metric gate (Brier < 0.190, Log Loss < 0.560, Brier-Log Divergence < 0.015), use BMA ensemble, unweighted Brier for selection, and PIT enforcement.

## Statistical Rigor: Addressing the Multiple Comparisons Problem

With ~440 total evaluation games across 8 LOYO folds, testing many configurations creates severe multiple comparisons risk. The "best" config from a large search is guaranteed to appear better than it truly is (selection bias / researcher degrees of freedom).

**Mitigations applied throughout:**
1. **Holm-Bonferroni correction** on all pairwise comparisons within each phase
2. **Minimum effect size** (Cohen's d > 0.2) required alongside p-value
3. **Fold consistency requirement**: improvement must hold in >= 5 of 8 LOYO folds
4. **Conservative final estimate**: report the LOYO mean + 1 SE as the "honest" Brier estimate (accounts for optimization bias)
5. **Brier decomposition** (Murphy 1973: reliability + resolution + uncertainty) used to distinguish calibration quality from discrimination — avoids selecting configs that game one component
6. **Holdout 2025 is advisory only**: with N=63 games, SE ≈ 0.05 on Brier, so we report it with bootstrap 95% CI but do NOT use it for selection

## Approach: 5-Phase Experiment Workflow

### New file: `src/ml/evaluation/experiment_workflow.py`

A single orchestrator class `ExperimentWorkflow` that runs 5 phases sequentially, each building on the prior phase's results. All results logged to `ExperimentRegistry`.

### Phase 1: Structural Search (~40 configs, ~2-4 hours)

Coarse grid over categorical/structural choices. Each config runs full LOYO (8 folds). Aggressively pruned by domain knowledge — a sports statistician already knows what works.

**Search dimensions:**
| Dimension | Options | Rationale |
|-----------|---------|-----------|
| model_complexity | "simple", "standard" | "full" disabled per protocol (GNN/transformer overfit on N=400) |
| calibration_method | "temperature", "platt" | Skip "isotonic" (requires N>500 for reliable fit); skip "none" (always worse) |
| enable_stacking | True, False | Stacking on 3 base models with N=400 is risky; test both |
| enable_bayesian_bt | True, False | Orthogonal signal source; worth testing |
| enable_spread_model | True, False | Margin-based model provides diversity |
| enable_goto_conversion | True, False | FLB correction — strong empirical track record on Kaggle |

**Domain-knowledge pruning rules:**
- "simple" forces: stacking=False, bayesian_bt=False, spread_model=False (only LR available)
- "simple" + goto_conversion + {temperature, platt} = 4 configs
- "standard" + all binary combos, but skip spread_model=False AND bayesian_bt=False (degenerates to LGB+XGB only, known weak)
- **Result: ~36 configs**

Feature selection is NOT searched: fixed domain-knowledge features (FIXED_FEATURE_SET) are always used. Learned selection on N=400 is double-dipping (confirmed by existing ablation framework).

**Metrics per config:**
- LOYO mean Brier (primary selection metric)
- LOYO per-fold Briers + std (stability)
- Log Loss, Accuracy, ECE
- Brier Skill Score vs seed-only baseline
- Brier decomposition: reliability, resolution, uncertainty
- Multi-metric gate pass/fail (Brier < 0.190, LL < 0.560, divergence < 0.015)

**Output:** Top-3 structural configs, ranked by LOYO mean Brier with Holm-Bonferroni adjusted pairwise tests

### Phase 2: Continuous Parameter Optimization (~80 configs, ~4-6 hours)

Take the best structural config from Phase 1. Search continuous parameters in two stages.

**Stage 2a: One-at-a-time sweeps** (~35 runs) to identify which parameters matter:
| Parameter | Grid | Default | Rationale |
|-----------|------|---------|-----------|
| tournament_shrinkage | [0.0, 0.03, 0.06, 0.09, 0.12] | 0.06 | Controls overconfidence correction; literature suggests 0.03-0.10 |
| seed_prior_weight | [0.0, 0.05, 0.10, 0.15, 0.20] | 0.10 | Seed is strongest single predictor; regularization strength |
| massey_blend_weight | [0.10, 0.20, 0.25, 0.30, 0.35] | 0.25 | External rating signal; most Kaggle winners use 0.15-0.30 |
| massey_sigma | [3.0, 4.0, 4.5, 5.0, 6.0] | 4.5 | Logistic CDF spread; calibrated via historical |
| training_year_decay | [0.70, 0.80, 0.85, 0.90, 1.0] | 0.85 | Recency vs sample size tradeoff |
| goto_conversion_margin | [0.0, 0.03, 0.05, 0.08, 0.12] | 0.05 | FLB correction strength |
| mc_noise_std | [0.12, 0.14, 0.16, 0.18, 0.20] | 0.16 | Game-level variance for MC simulation |

For each sweep: paired t-test across 8 folds vs default. Rank params by absolute effect size.

**Stage 2b: Joint optimization** (~45 runs) of top-3 most impactful parameters:
- Use Latin Hypercube Sampling (LHS) with 45 points in the 3D subspace
- This captures interactions that one-at-a-time misses
- Select best by LOYO mean Brier with Holm-Bonferroni correction vs Stage 2a baseline

**Adoption gate (per existing ImprovementGate pattern):**
- Brier improvement > 0.001 (practical significance)
- Paired t-test p < 0.10 after Holm-Bonferroni correction
- >= 5 of 8 LOYO folds improve
- Cohen's d > 0.15

### Phase 3: Training Window + Hyperparameter Tuning (~3-5 hours)

**3a: Training Window** (delegates to existing `TrainingWindowOptimizer`):
- Windows: [5, 7, 8(all), 10] years per model type
- Includes regime-aligned windows (post-shot-clock-2016, post-3pt-2020)
- ~18 evaluations total

**3b: Model Hyperparameters** — small fixed grid (NOT Optuna random search, which introduces more selection bias on small data):
- LightGBM: {num_leaves: [6,8,12], min_child_samples: [40,50,70], learning_rate: [0.03,0.05]} = 18 combos
- XGBoost: {max_depth: [3,4,5], min_child_weight: [5,8], learning_rate: [0.03,0.05]} = 12 combos
- SpreadRegressor: {spread_sigma: [9,11,13]} = 3 combos
- Logistic: {C: [0.1, 1.0, 10.0]} = 3 combos
- Each evaluated via LOYO. Select by mean Brier.
- **Total: ~36 configs**, well within domain expert's "sensible range"

### Phase 4: Ablation Validation (~1-2 hours)

Run full ablation study on best config from Phases 1-3 using existing `AblationStudy`:
- Test each enabled component: stacking, bayesian_bt, spread_model, tournament_adaptation, goto_conversion, recency_weighting, round_weighted_calibration
- Paired t-test per component: must pass p < 0.05 to justify inclusion
- **Embedding gate**: GNN/transformer require p < 0.01 (stricter due to overfitting risk, but these are disabled anyway)
- Remove any component that doesn't significantly help
- Re-evaluate stripped config to confirm no regression

### Phase 5: Final Report + Holdout Check (~30 min)

**5a: Recommended config generation:**
- Export best config as JSON (compatible with `production_2026.json` format)
- Include all parameter values with provenance (which phase selected them)

**5b: Holdout 2025 validation (advisory, NOT used for selection):**
- Run best config on 2025 holdout
- Compute with bootstrap 95% CI (1000 resamples):
  - Brier score + CI
  - Log Loss + CI
  - Accuracy + CI
  - ECE
  - Brier decomposition (reliability, resolution, uncertainty per Murphy 1973)
  - BSS vs seed baseline
  - Reliability diagram data (10 bins)
- Gate check: Brier < 0.190, Log Loss < 0.560, Brier-Log Divergence < 0.015
- **Flag if holdout Brier > LOYO mean + 2*LOYO_std** (signals overfitting to LOYO folds)

**5c: Comparison table:**
- Production default vs recommended config
- Per-phase improvement attribution
- Statistical test results (paired t-test, effect size, fold breakdown)

**5d: Honest caveats section:**
- Total configs tested and implied selection bias
- Conservative Brier estimate: LOYO mean + 1 SE
- Holdout N=63 limitation and CI width
- Any components removed vs Protocol v2 expectations

**Total budget: ~170-210 configs, estimated 10-18 hours**

## Files to Create/Modify

### New Files

1. **`src/ml/evaluation/experiment_workflow.py`** (~800 lines)
   - `ExperimentWorkflow` class: main orchestrator
   - `Phase1StructuralSearch`: grid over categorical options
   - `Phase2ParameterOptimization`: continuous param search
   - `Phase3WindowAndHyperparams`: training window + Optuna
   - `Phase4AblationValidation`: component significance testing
   - `Phase5HoldoutReport`: final validation + recommendation
   - `WorkflowReport`: dataclass for final output
   - Reuses: `ExperimentRegistry`, `TrainingWindowOptimizer`, `AblationStudy`, `CalibrationMethodSelector`
   - Evaluation function: creates lightweight LOYO evaluator that trains/predicts with given config

2. **`src/run_experiments.py`** (~80 lines)
   - Standalone runner script: `python src/run_experiments.py`
   - Accepts `--phases` to run specific phases (e.g., `--phases 1,2`)
   - Accepts `--budget-hours` for time constraint (default: 20)
   - Accepts `--output` for report path (default: `artifacts/experiment_report.json`)
   - Loads data, constructs pipeline, calls `ExperimentWorkflow.run()`

3. **`tests/test_experiment_workflow.py`** (~150 lines)
   - Unit tests for each phase with mock evaluator
   - Test search space generation, pruning, statistical gating
   - Test report generation

### Modified Files

4. **`src/main.py`**: Add `run-experiments` CLI command (~20 lines)
   - Delegates to `ExperimentWorkflow` with CLI args

## Key Design Decisions

1. **LOYO as the single evaluation protocol** - Every config is evaluated by full LOYO (train on 7 years, test on held-out year, repeat 8 times). This is the gold standard for sports prediction with limited data. Holdout 2025 is NEVER used during search.

2. **Statistical gating** (ImprovementGate pattern) - No change adopted without:
   - Practical significance: > 0.001 Brier improvement
   - Statistical significance: paired t-test p < 0.10 across LOYO folds
   - Fold consistency: >= 60% of folds improve

3. **Metrics a sports statistician would approve:**
   - **Primary:** LOYO mean Brier (Kaggle's actual metric)
   - **Secondary:** Log Loss (information-theoretic), Accuracy (interpretable)
   - **Calibration:** ECE, reliability diagram, Brier decomposition
   - **Stability:** LOYO std, worst-fold Brier, fold improvement rate
   - **Skill:** BSS vs seed baseline (are we beating naive seeding?)
   - **Gate:** Protocol v2 multi-metric gate (Brier + LogLoss + divergence)

4. **Computational efficiency:**
   - Phase 1 pruning eliminates ~70% of combinatorial space via domain constraints
   - Phase 2 uses one-at-a-time sweeps before joint search
   - Phase 3 reuses existing Optuna/window optimizer infrastructure
   - Early stopping: if no Phase 2 param improves > 0.001, skip joint search

5. **Integration with existing infrastructure:**
   - All results logged to `ExperimentRegistry` (JSONL ledger)
   - Reuses `TrainingWindowOptimizer`, `AblationStudy`, `ImprovementGate`
   - Output config compatible with `production_2026.json` format

## Verification Plan

1. **Unit tests:** `pytest tests/test_experiment_workflow.py -x`
2. **Dry run:** `python src/run_experiments.py --phases 1 --budget-hours 1` (runs Phase 1 only with reduced grid)
3. **Lint:** `ruff check src/ml/evaluation/experiment_workflow.py src/run_experiments.py`
4. **Full run (integration):** `python src/run_experiments.py --output artifacts/experiment_report.json`
5. **Verify report:** Check `artifacts/experiment_report.json` contains all phases, recommended config, statistical justification
