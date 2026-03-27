# CLAUDE.md — March Madness Forecaster

## Project Overview

NCAA March Madness tournament prediction system with a locked production path for calibrated probabilities and bracket simulation. Built on LightGBM/XGBoost/Logistic Regression ensemble with 86-dimensional team feature vectors, temperature-scaling calibration, and Monte Carlo bracket simulation (50k runs).

## Quick Reference

```bash
# Install
pip install -e .

# Run tests
pytest tests/ -x --tb=short
pytest tests/ -m "unit"                    # Unit tests only
pytest tests/ --cov=src --cov-report=term  # With coverage

# Lint
ruff check src/ tests/
ruff check src/ tests/ --fix               # Auto-fix safe issues

# Run production pipeline (2026)
python src/run_production_2026.py
python src/run_production_2026.py --dry-run  # Validation only

# Run SOTA pipeline (research/development)
march-madness sota --year 2026 --scrape-live --simulations 50000

# Data ingestion
march-madness ingest-historical --start-season 2005 --end-season 2025
march-madness ingest --year 2026
```

## Architecture

```
src/
├── main.py                    # CLI entry point (Click-based, ~148k lines)
├── run_production_2026.py     # Frozen 2026 production entrypoint (governance-locked)
├── exceptions.py              # Custom exceptions (LeakageError, GovernanceApprovalRequired, etc.)
├── pipeline/
│   ├── sota.py                # Main SOTA pipeline (~2600 lines)
│   └── stages/                # Pipeline stages: data loading, training, calibration,
│                              #   simulation, PIT validation, orchestration, EV analysis
├── data/
│   ├── features/              # 86-dim feature engineering (point-in-time safe)
│   │   ├── feature_engineering.py      # Core team feature vector
│   │   ├── feature_selection.py        # Learned feature importance pruning
│   │   ├── proprietary_metrics.py      # IncrementalMetricsEngine (temporal-safe)
│   │   ├── public_advanced_metrics.py  # Public advanced metrics
│   │   ├── materialization.py          # Feature table materialization
│   │   ├── tournament_features.py      # Tournament-specific features
│   │   └── travel_distance.py          # Travel distance features
│   ├── ingestion/             # DAG-based data collection & validation
│   ├── scrapers/              # Torvik, ESPN, rosters, sports reference
│   └── models/                # Player, Roster, GameFlow data models
├── ml/
│   ├── ensemble/              # LightGBM + XGBoost + LogisticRegression (CFA)
│   ├── calibration/           # Temperature scaling, isotonic, Platt
│   ├── evaluation/            # RDoF audit, experimentation registry
│   ├── optimization/          # Hyperparameter optimization (Optuna)
│   ├── ranking/               # Ranking models
│   ├── training/              # Training utilities
│   ├── research/              # Research/experimental models
│   ├── time_series/           # Time series models
│   ├── gnn/                   # Graph neural network (disabled in production)
│   ├── transformer/           # Transformer model (disabled in production)
│   └── meta_learning.py       # Meta-learner for stacking
├── simulation/                # Monte Carlo bracket simulation (50k runs)
├── optimization/              # Bracket optimization & contrarian strategy
├── governance/                # Production validators, audit trails
├── exports/                   # Kaggle submission generation
├── agents/                    # Multi-agent pipeline coordination (disabled in prod)
├── evaluation/                # Model evaluation utilities
├── forecaster/                # Forecasting utilities
├── forecasting/               # Forecasting pipeline
├── espn/                      # ESPN data integration
├── conference_tournament/     # Conference tournament logic
├── models/                    # Shared data models
├── monitoring/                # Pipeline monitoring
├── quant/                     # Quantitative analysis
├── reproducibility/           # Reproducibility framework (freeze/verify)
├── research/                  # Research modules
├── validation/                # Validation utilities
├── deployment/                # Deployment utilities
└── workflows/                 # Workflow orchestration
```

## Code Conventions

- **Linter:** ruff (E, F, W rules). Line length: 120. Target: Python 3.9
- **Naming:** snake_case for functions/modules, UPPER_CASE for constants
- **Imports:** Conditional imports OK; re-exports use `# noqa: F401`
- **Type hints:** Minimal (not strictly enforced)
- **Custom exceptions** (in `src/exceptions.py`):
  - `LeakageError` — temporal/data leakage detected (hard stop)
  - `DataFreshnessError` — required data sources stale or missing
  - `PreRunValidationError` — pre-run checks failed
  - `ComputeBudgetExceeded` — compute budget limits exceeded
  - `DataRequirementError` — required data artifact missing/invalid
  - `IntegrityError` — model calibration or math integrity failure
  - `GovernanceApprovalRequired` — action requires human approval (has `request_id`)

## Testing

- **Framework:** pytest with auto-marker assignment via `tests/conftest.py`
- **Markers:** `unit`, `integration`, `data_contract`, `leakage`, `freeze`, `production`, `calibration`, `live_protocol`
- **Coverage minimum:** 20%
- **Test naming:** `test_*.py` files in `tests/` directory (67+ test files)
- **Subdirectories:** `tests/data_integrity/`, `tests/evaluation/`, `tests/fixtures/`
- **Run specific markers:** `pytest tests/ -m "unit"` or `pytest tests/ -m "leakage"`

## Production Path Rules

**CRITICAL — do not violate these constraints:**

1. Production runs MUST use `python src/run_production_2026.py` or `march-madness run-production-2026`
2. Generic commands (`sota`, `sota-from-manifest`) are blocked from acting as production
3. Production config is locked in `configs/production_2026.json` — do not modify locked fields:
   - `model_complexity: "standard"`, `probability_profile: "production"`, `mode: "calibration"`
   - `calibration_method: "temperature"`, `use_agent_orchestration: false`
   - `enable_gnn: false`, `enable_transformer: false`
4. Disabled modules: `enable_gnn`, `enable_transformer`, `enable_seed_overrides`, `enable_brier_sharpening`, `enable_embedding_projections`
   - Enabled modules: `enable_stacking`, `enable_feature_selection`, `enable_goto_conversion`, `enable_round_weighted_calibration`, `enable_bayesian_bt`
5. Training years: 2016–2024 (no 2020, no 2025). Holdout: 2025. Target: 2026

## Data Integrity

- **Leakage detection** is enforced — `LeakageError` raised on violations
- **Point-in-time features** use tournament cutoff dates per season
- **Temporal cross-validation** via Leave-One-Year-Out (LOYO)
- Feature dimension is fixed: `TEAM_FEATURE_DIM = 86`
- Feature tiers: Tier 1 (Static), Tier 2 (Cumulative with cutoff), Tier 3 (External/Selection Sunday snapshot)

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point (all commands) |
| `src/pipeline/sota.py` | Core prediction pipeline |
| `src/run_production_2026.py` | Frozen production entrypoint |
| `src/exceptions.py` | Custom exception hierarchy |
| `configs/production_2026.json` | Blessed production config (do not change locked fields) |
| `configs/team_aliases.json` | Team name mappings |
| `tests/conftest.py` | Test fixtures and auto-marker logic |
| `pyproject.toml` | Ruff + pytest configuration |
| `.claude/skills.md` | Agent skills and best practices |

## Environment

- Kaggle API credentials: set `KAGGLE_USERNAME` and `KAGGLE_KEY` (see `.env.example`)
- Python dependencies: `requirements.txt` (38 packages)
- Production lock: `requirements-production-lock.txt`
- Python version: 3.9+ (target 3.9 for ruff)

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `ci.yml` — Continuous integration (lint + test)
- `run-production-pipeline.yml` — Production pipeline execution
- `run-pipeline.yml` — SOTA pipeline execution
- `data-ingestion.yml` — Data ingestion pipeline
- `nightly-validation.yml` — Nightly validation runs
- `deploy-pages.yml` — GitHub Pages deployment (dashboard)
- `generate-web-data.yml` — Web data generation
- `repair-dates.yml` — Date repair automation
- `secrets-qaqc.yml` — Secret scanning QA/QC
