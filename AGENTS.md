# AGENTS.md — Subagent Guide for March Madness Forecaster

This file provides role-specific guidance for Claude subagents working on this codebase. Read this before taking any action.

## Repo at a Glance

- **What:** NCAA March Madness prediction system (LightGBM/XGBoost/LR ensemble, 86-dim features, temperature calibration, 50k Monte Carlo sims)
- **Language:** Python 3.9+
- **Linter:** `ruff check src/ tests/` (line length 120, rules E/F/W)
- **Tests:** `pytest tests/ -x --tb=short` (67+ test files, auto-markers via `conftest.py`)
- **Entry points:** `src/main.py` (CLI), `src/run_production_2026.py` (production-locked)
- **Config:** `configs/production_2026.json` (locked fields — never modify)

## Critical Constraints (All Agents)

1. **Never modify** locked fields in `configs/production_2026.json`
2. **Never modify** `src/run_production_2026.py` without explicit permission
3. **Never downgrade** `LeakageError` or `PITViolationError` to warnings
4. **Never delete** `OOS-FIX` or `S5 FIX` comments — they document hard-won corrections
5. **Never use random k-fold** on time-series data — use LOYO cross-validation
6. **Preserve** the custom exception hierarchy in `src/exceptions.py`
7. **Run tests** after any code change: `pytest tests/ -m "unit" -x --tb=short`

## Role: Code Search / Exploration Agent

**When looking for code:**

| To find... | Look in... |
|------------|-----------|
| CLI commands | `src/main.py` (Click-based, ~148k lines — search for `@cli.command`) |
| Pipeline logic | `src/pipeline/sota.py` (~2600 lines) and `src/pipeline/stages/` |
| Feature engineering | `src/data/features/feature_engineering.py`, `proprietary_metrics.py` |
| ML models | `src/ml/ensemble/` (CFA ensemble), `src/ml/calibration/` |
| Data scrapers | `src/data/scrapers/` (Torvik, ESPN, rosters, sports reference) |
| Simulation | `src/simulation/monte_carlo.py` |
| Exceptions | `src/exceptions.py` |
| Test fixtures | `tests/conftest.py`, `tests/fixtures/` |
| Production config | `configs/production_2026.json` |
| Team name aliases | `configs/team_aliases.json` |

**Important constants:**
- `TEAM_FEATURE_DIM = 86` — fixed feature vector size
- `TOURNAMENT_START_DATES` — per-year dates for PIT enforcement
- `SELECTION_SUNDAY_DATES` — per-year Selection Sunday dates
- `KAGGLE_ROUND_WEIGHTS` — round-wise scoring weights

## Role: Code Editing / Feature Development Agent

**Before editing:**
1. Read the file and surrounding `OOS-FIX` / `S5 FIX` comments
2. Run `pytest tests/ -m "unit" -x --tb=short` to establish baseline
3. Identify blast radius — what pipeline stages and consumers are affected?

**After editing:**
1. Run `ruff check src/ tests/` to verify lint
2. Run `pytest tests/ -m "unit" -x --tb=short` for regression check
3. If touching features or ML code: `pytest tests/ -m "leakage" -x`
4. If touching calibration: `pytest tests/ -m "calibration" -x`
5. If touching production path: `python src/run_production_2026.py --dry-run`

**Adding features:**
1. Implement in `src/data/features/` — must be point-in-time safe
2. Update `TEAM_FEATURE_DIM` if vector size changes
3. Register in `FIXED_FEATURE_SET` or `SIMPLE_FEATURE_SET`
4. Add leakage test in `tests/data_integrity/`

**Adding scrapers:**
1. Create in `src/data/scrapers/`
2. Integrate in `src/data/ingestion/`
3. Add rate limiting and error handling
4. Test with mocked HTTP — never hit live endpoints in tests

## Role: Test Runner / Validation Agent

```bash
# Fast unit tests (run after every change)
pytest tests/ -m "unit" -x --tb=short

# Full suite with coverage
pytest tests/ --cov=src --cov-report=term -x --tb=short

# Specific markers
pytest tests/ -m "leakage"       # Data leakage tests
pytest tests/ -m "calibration"   # Calibration tests
pytest tests/ -m "production"    # Production path tests
pytest tests/ -m "freeze"        # Reproducibility tests

# Lint
ruff check src/ tests/

# Production dry-run
python src/run_production_2026.py --dry-run
```

**Coverage minimum:** 20%. Markers are auto-assigned by `tests/conftest.py` based on file path.

## Role: CI / DevOps Agent

**Workflows** in `.github/workflows/`:
- `ci.yml` — Lint + test on push/PR
- `run-production-pipeline.yml` — Production pipeline
- `run-pipeline.yml` — SOTA pipeline
- `data-ingestion.yml` — Data ingestion
- `nightly-validation.yml` — Nightly checks
- `deploy-pages.yml` — GitHub Pages dashboard
- `generate-web-data.yml` — Web data generation
- `repair-dates.yml` — Date repair

**Shared action:** `.github/actions/setup-python-env/action.yml` (Python 3.10, pip caching)

## Role: Review / Audit Agent

**What to check in PRs:**
1. No changes to locked production config fields
2. No leakage — temporal features use `cutoff_date` parameters
3. No downgraded exception types (LeakageError must stay RuntimeError subclass)
4. `OOS-FIX` comments preserved or updated (not deleted)
5. Magic numbers have documented rationale
6. Sample size guards maintained (e.g., `valid_samples >= 80`)
7. Tests pass: unit, leakage, calibration markers as appropriate

**Data integrity signals:**
- Feature tiers: Tier 1 (Static), Tier 2 (Cumulative w/ cutoff), Tier 3 (External/Selection Sunday)
- Training: 2016-2024 (no 2020). Holdout: 2025. Target: 2026
- Calibration: tournament games only, temperature scaling

## Module Map

```
src/
├── main.py                    # CLI (~148k lines, Click commands)
├── run_production_2026.py     # Governance-locked production entry
├── exceptions.py              # LeakageError, GovernanceApprovalRequired, etc.
├── pipeline/sota.py           # Core pipeline (~2600 lines)
├── pipeline/stages/           # baseline_training, calibration, data_loader,
│                              #   simulation, pit_validation, orchestration,
│                              #   ev_analysis, ev_mode, game_utils, inference
├── data/features/             # feature_engineering, proprietary_metrics,
│                              #   feature_selection, materialization,
│                              #   tournament_features, travel_distance,
│                              #   public_advanced_metrics, massey_systems
├── data/ingestion/            # DAG-based data collection
├── data/scrapers/             # Torvik, ESPN, rosters, sports reference
├── ml/ensemble/               # CFA: LightGBM + XGBoost + LogisticRegression
├── ml/calibration/            # Temperature, isotonic, Platt scaling
├── ml/evaluation/             # RDoF audit, experimentation registry
├── ml/optimization/           # Optuna hyperparameter search
├── ml/gnn/                    # Graph neural network (DISABLED in prod)
├── ml/transformer/            # Transformer model (DISABLED in prod)
├── simulation/                # Monte Carlo bracket sim (50k runs)
├── optimization/              # Bracket strategy optimization
├── governance/                # Production validators, audit trails
├── exports/                   # Kaggle submission generation
├── reproducibility/           # Freeze/verify framework
└── agents/                    # Multi-agent coordination (DISABLED in prod)
```

## Exception Hierarchy

All exceptions are in `src/exceptions.py` and inherit from `RuntimeError`:

| Exception | When raised |
|-----------|------------|
| `LeakageError` | Temporal or data leakage detected (hard stop) |
| `DataFreshnessError` | Required data sources are stale or missing |
| `PreRunValidationError` | Pre-run validation checks fail |
| `ComputeBudgetExceeded` | Pipeline exceeds compute budget (strict mode) |
| `DataRequirementError` | Required data artifact missing/invalid |
| `IntegrityError` | Model calibration or math integrity failure |
| `GovernanceApprovalRequired` | Action requires human approval (`request_id` attr) |
