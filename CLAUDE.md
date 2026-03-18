# CLAUDE.md — March Madness Forecaster

## Project Overview

NCAA March Madness tournament prediction system with a locked production path for calibrated probabilities and bracket simulation. Built on LightGBM/XGBoost/Logistic Regression ensemble with 79-dimensional team feature vectors, temperature-scaling calibration, and Monte Carlo bracket simulation (50k runs).

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
├── main.py                    # CLI entry point (Click-based)
├── run_production_2026.py     # Frozen 2026 production entrypoint
├── pipeline/sota.py           # Main SOTA pipeline (~2600 lines)
├── pipeline/stages/           # Pipeline stages (data, training, calibration, simulation)
├── data/
│   ├── features/              # 79-dim feature engineering (point-in-time safe)
│   ├── ingestion/             # DAG-based data collection
│   ├── scrapers/              # Torvik, ESPN, rosters, sports reference
│   └── models/                # Player, Roster, GameFlow data models
├── ml/
│   ├── ensemble/              # LightGBM + XGBoost + LogisticRegression
│   ├── calibration/           # Temperature scaling, isotonic, Platt
│   └── evaluation/            # RDoF audit, experimentation registry
├── simulation/                # Monte Carlo bracket simulation
├── governance/                # Production validators, audit trails
└── exports/                   # Kaggle submission generation
```

## Code Conventions

- **Linter:** ruff (E, F, W rules). Line length: 120. Target: Python 3.9
- **Naming:** snake_case for functions/modules, UPPER_CASE for constants
- **Imports:** Conditional imports OK; re-exports use `# noqa: F401`
- **Type hints:** Minimal (not strictly enforced)
- **Custom exceptions:** `LeakageError`, `DataFreshnessError`, `PreRunValidationError`, `GovernanceApprovalRequired`

## Testing

- **Framework:** pytest with auto-marker assignment via `tests/conftest.py`
- **Markers:** `unit`, `integration`, `data_contract`, `leakage`, `freeze`, `production`, `calibration`, `live_protocol`
- **Coverage minimum:** 20%
- **Test naming:** `test_*.py` files in `tests/` directory
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
- Feature dimension is fixed: `TEAM_FEATURE_DIM = 79`

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point (all commands) |
| `src/pipeline/sota.py` | Core prediction pipeline |
| `src/run_production_2026.py` | Frozen production entrypoint |
| `configs/production_2026.json` | Blessed production config (do not change locked fields) |
| `tests/conftest.py` | Test fixtures and auto-marker logic |
| `pyproject.toml` | Ruff + pytest configuration |

## Environment

- Kaggle API credentials: set `KAGGLE_USERNAME` and `KAGGLE_KEY` (see `.env.example`)
- Python dependencies: `requirements.txt` (38 packages)
- Production lock: `requirements-production-lock.txt`
