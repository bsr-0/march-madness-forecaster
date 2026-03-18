# Skills — March Madness Forecaster

---

## Sports Statistics Best Practices

These principles govern all code in this repository. Enforce them during code review, feature development, and refactoring.

### Temporal Integrity (Point-in-Time Safety)

The single most important rule in sports prediction: **never use future information to predict past events.**

- Every feature must be computable using ONLY data available before the prediction moment
- Features are classified into three tiers (see `src/pipeline/stages/pit_validation.py`):
  - **Tier 1 (Static):** No restriction — seed, conference, preseason rankings
  - **Tier 2 (Cumulative):** Must use only games played before tournament start date
  - **Tier 3 (External):** Must use the Selection Sunday snapshot for that year
- Tournament seeds are assigned on Selection Sunday — never use seeds as training features for regular-season games (`SEED LEAKAGE FIX` in `baseline_training.py`)
- The `IncrementalMetricsEngine` in `src/data/features/proprietary_metrics.py` computes temporal-leakage-free features with mandatory `cutoff_date` parameters
- Violations raise `LeakageError` (hard stop) or `PITViolationError` — never downgrade these to warnings
- Run `pytest tests/ -m "leakage"` after any feature or data change

### Calibration Rigor

Predicted probabilities must be *calibrated* — a 70% prediction should win ~70% of the time.

- **Temperature scaling** is the production calibration method (simple, robust, one parameter)
- **ECE (Expected Calibration Error)** is the primary calibration metric, using equal-frequency (quantile) bins — not equal-width bins, which are unstable in sparse tails
- **Brier score** is the primary accuracy metric (and Kaggle's scoring metric since 2023)
- Calibration requires minimum sample sizes: at least 80 samples per fold (40 calibration + 40 evaluation) — small samples give statistically meaningless temperature estimates
- Calibration must be performed on **tournament games only**, not regular season (different dynamics)
- Always compare calibrated output against the uncalibrated baseline (Pass 1) — if calibration makes Brier worse, fall back to identity scaling (temperature=1.0)
- Wilson score confidence intervals are used for simulation-derived probabilities

### Validation Methodology

- **Leave-One-Year-Out (LOYO)** is the primary cross-validation strategy — it respects temporal ordering and prevents information leakage across seasons
- Never use random k-fold on time-series sports data — games within a season are correlated
- Training years: 2016–2024 (excluding 2020 COVID cancellation). Holdout: 2025. Target: 2026
- When evaluating model changes, report per-year Brier scores, not just the aggregate — a model that improves 2023 but destroys 2018 is suspect

### Overfitting Prevention

This codebase has been hardened against overfitting through multiple `OOS-FIX` corrections:

- **Fixed feature set:** Domain-knowledge features (`FIXED_FEATURE_SET`) are preferred over learned feature selection — with ~400 tournament games total, automated selection overfits
- **Constrained hyperparameters:** `num_leaves<=8`, `max_depth<=4`, heavy L1/L2 regularization — small tournament datasets demand simple models
- **Reduced Optuna trials:** 15 trials (down from 50) on narrow search spaces to prevent selection bias
- **Stacking disabled by default:** Meta-learners overfit on small tournament samples
- **GNN/transformer confidence reduced:** Complex architectures get lower ensemble weight
- **Missing-data indicators removed:** Binary "is_missing" features leaked the data-collection process, not game quality

### Domain Knowledge Constants

Sports-specific values that should not be changed without strong justification:

- `TEAM_FEATURE_DIM = 79` — Fixed feature vector dimensionality
- `TOURNAMENT_START_DATES` — Per-year tournament start dates for PIT enforcement
- `SELECTION_SUNDAY_DATES` — Per-year Selection Sunday dates (PIT tier enforcement)
- `KAGGLE_ROUND_WEIGHTS` — Round-wise weighting reflecting Kaggle scoring
- Monte Carlo simulation correlation reduced to 0.10 (from 0.25) — original value overstated inter-game correlation

---

## Rigorous Code Refactoring

Follow these practices when refactoring any code in this repository.

### Before You Touch Anything

1. **Run the full test suite** and record the baseline: `pytest tests/ -x --tb=short`
2. **Run leakage tests** specifically: `pytest tests/ -m "leakage" -x`
3. **Run calibration tests**: `pytest tests/ -m "calibration" -x`
4. **Identify the blast radius** — which pipeline stages, features, and downstream consumers are affected?
5. **Read the `OOS-FIX` and `S5 FIX` comments** near the code you're changing — these document hard-won corrections and the reasoning behind them. Do not undo them without understanding why they exist.

### During the Refactor

1. **Preserve statistical semantics** — renaming `brier_score` to `accuracy` changes meaning. Keep variable names that encode statistical concepts.
2. **Never change magic numbers without analysis** — constants like `CAL_WEIGHT_SCALE = 5.0` have documented rationale (maps ECE range [0, ~0.2] to weight range [1.0, ~1.3]). Understand the math before modifying.
3. **Keep leakage guards intact** — `cutoff_date` parameters, `LeakageError` raises, temporal filters. These are safety-critical, not cleanup targets.
4. **Respect minimum sample size checks** — guards like `valid_samples >= 80` exist because small-sample statistics are unreliable. Do not lower thresholds.
5. **Maintain the audit trail** — `OOS-FIX:` comments document why a change was made. When refactoring, update the comment to reflect the new location, don't delete it.
6. **One concern at a time** — don't mix refactoring with behavior changes. A refactor commit should produce identical outputs.

### After the Refactor

1. **Run the full test suite again** and compare against the baseline
2. **Run leakage and calibration tests**: `pytest tests/ -m "leakage or calibration" -x`
3. **Verify production path**: `python src/run_production_2026.py --dry-run`
4. **If the refactor touches features or model code**, run LOYO validation to confirm Brier scores haven't changed: `march-madness loyo-validate`
5. **Verify reproducibility**: `march-madness verify-freeze` (if a freeze artifact exists)

### What NOT to Refactor

- **`configs/production_2026.json` locked fields** — these are frozen for the 2026 tournament
- **Custom exception hierarchy** (`LeakageError`, `PITViolationError`, etc.) — downstream code depends on catching specific types
- **Selection Sunday / tournament start date mappings** — these are historical facts
- **The production entrypoint** (`run_production_2026.py`) — governance-locked

---

## Skill: Run Tests

Run the test suite before and after making changes.

```bash
# Fast unit tests
pytest tests/ -m "unit" -x --tb=short

# Full suite with coverage
pytest tests/ --cov=src --cov-report=term -x --tb=short

# Specific test file
pytest tests/test_<name>.py -v

# By marker
pytest tests/ -m "leakage"       # Data leakage tests
pytest tests/ -m "calibration"   # Calibration tests
pytest tests/ -m "production"    # Production path tests
```

Always run `pytest tests/ -m "unit" -x --tb=short` after code changes to catch regressions.

## Skill: Lint

```bash
ruff check src/ tests/
ruff check src/ tests/ --fix  # Auto-fix safe issues
```

Line length is 120. Target Python 3.9.

## Skill: Add a New Feature to the Pipeline

1. Add feature computation in `src/data/features/` — must be point-in-time safe (no future data leakage)
2. Update `TEAM_FEATURE_DIM` if the feature vector size changes
3. Register the feature in `FIXED_FEATURE_SET` or `SIMPLE_FEATURE_SET` as appropriate
4. Add a leakage test in `tests/data_integrity/`
5. Run `pytest tests/ -m "leakage" -x` to verify no temporal leakage
6. Run `pytest tests/ -m "unit" -x` for regression check

## Skill: Add a New Scraper

1. Create the scraper in `src/data/scrapers/`
2. Add ingestion integration in `src/data/ingestion/`
3. Register the CLI command in `src/main.py` if needed
4. Add rate limiting and error handling for external requests
5. Write tests with mocked HTTP responses — never hit live endpoints in tests

## Skill: Modify the ML Ensemble

1. Edit `src/ml/ensemble/` for model changes
2. If adding a new model type, integrate it into the ensemble voting/stacking logic
3. Update calibration in `src/ml/calibration/` if probability outputs change
4. Run `pytest tests/ -m "calibration" -x` to verify calibration still holds
5. **Never** change production-locked settings in `configs/production_2026.json`

## Skill: Run Production Pipeline

```bash
# Dry run (validation only, no predictions)
python src/run_production_2026.py --dry-run

# Full production run
python src/run_production_2026.py
```

**Do not** use `march-madness sota` for production — it is blocked from acting as production.
**Do not** modify locked fields in `configs/production_2026.json`.

## Skill: Data Ingestion

```bash
# Historical data (one-time or refresh)
march-madness ingest-historical --start-season 2005 --end-season 2025

# Roster data
march-madness scrape-rosters --start-year 2005 --end-year 2026
march-madness enrich-rosters --start-year 2005 --end-year 2026

# Current year
march-madness ingest --year 2026
```

## Skill: Generate Kaggle Submission

```bash
march-madness kaggle-export
march-madness backtest-kaggle  # Validate against historical results
```

Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Skill: Validate Reproducibility

```bash
march-madness freeze-pipeline    # Create reproducibility artifact
march-madness verify-freeze      # Verify against frozen artifact
march-madness audit-rdof         # Audit researcher degrees of freedom
```

## Skill: Add a New Test

1. Create `tests/test_<name>.py`
2. Markers are auto-assigned by `tests/conftest.py` based on file path — no need to manually decorate
3. Use fixtures from `conftest.py` (team data, predictions, outcomes, configs)
4. For data integrity tests, place in `tests/data_integrity/`
5. Run your new test: `pytest tests/test_<name>.py -v`
