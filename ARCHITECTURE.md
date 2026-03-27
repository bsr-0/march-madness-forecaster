# ARCHITECTURE.md — March Madness Forecaster

Deep architectural reference for agents and developers working on this codebase. For quick reference, see `CLAUDE.md`. For role-specific agent guidance, see `AGENTS.md`.

## Pipeline Flow

The production prediction pipeline executes in this order:

```
1. Data Loading          → src/pipeline/stages/data_loader.py
2. Feature Engineering   → src/data/features/
3. PIT Validation        → src/pipeline/stages/pit_validation.py
4. Training              → src/pipeline/stages/baseline_training.py
5. Calibration           → src/pipeline/stages/calibration.py
6. Simulation            → src/pipeline/stages/simulation.py
7. Bracket Optimization  → src/optimization/
8. Export                → src/exports/
```

The pipeline is orchestrated by `src/pipeline/sota.py` (~2600 lines) for research runs, or `src/run_production_2026.py` for production (governance-locked).

## Module Deep Dive

### `src/pipeline/` — Pipeline Core

| File | Lines | Purpose |
|------|-------|---------|
| `sota.py` | ~2600 | Main pipeline orchestrator. Builds features, trains, calibrates, simulates |
| `stages/data_loader.py` | ~81k | Loads raw data, resolves team names, builds training samples |
| `stages/baseline_training.py` | ~139k | Model training: LOYO CV, ensemble fitting, feature selection |
| `stages/calibration.py` | ~47k | Temperature/isotonic/Platt calibration on tournament games |
| `stages/simulation.py` | ~33k | Monte Carlo bracket simulation stage |
| `stages/pit_validation.py` | ~13k | Point-in-time validation (Tier 1/2/3 feature checks) |
| `stages/orchestration.py` | ~52k | Pipeline stage coordination and dependency management |
| `stages/ev_analysis.py` | ~41k | Expected value analysis for bracket optimization |
| `stages/game_utils.py` | ~8k | Game-level utility functions |
| `stages/inference.py` | ~3k | Model inference wrapper |
| `stages/context.py` | ~2k | Pipeline context object (shared state between stages) |

### `src/data/features/` — Feature Engineering (86 dimensions)

| File | Purpose |
|------|---------|
| `feature_engineering.py` | Core 86-dim team feature vector builder |
| `proprietary_metrics.py` | `IncrementalMetricsEngine` — temporal-safe metric computation with mandatory `cutoff_date` |
| `feature_selection.py` | Learned feature importance pruning with train/eval drift checks |
| `materialization.py` | Feature table materialization (game-pair → feature matrix) |
| `tournament_features.py` | Tournament-specific features (seed, matchup history, venue) |
| `travel_distance.py` | Travel distance computation for venue effects |
| `public_advanced_metrics.py` | Public advanced stats (KenPom-style, Torvik) |
| `massey_systems.py` | Massey Ordinals integration |
| `statistical_audit.py` | Statistical audit of feature distributions |

**Feature tiers:**
- **Tier 1 (Static):** Seed, conference, preseason rankings — no temporal restriction
- **Tier 2 (Cumulative):** Regular-season stats — must use only games before tournament start date
- **Tier 3 (External):** Torvik, KenPom — must use Selection Sunday snapshot for that year

### `src/ml/` — Machine Learning

| Subdirectory | Purpose |
|-------------|---------|
| `ensemble/` | CFA ensemble: LightGBM + XGBoost + Logistic Regression with LOYO-optimized weights |
| `calibration/` | Probability calibration: temperature scaling (production), isotonic, Platt |
| `evaluation/` | RDoF (Researcher Degrees of Freedom) audit, experimentation registry |
| `optimization/` | Optuna hyperparameter search (15 trials, narrow bounds) |
| `ranking/` | Ranking models (Elo, BT) |
| `training/` | Training utilities |
| `research/` | Experimental models (not used in production) |
| `time_series/` | Time series models |
| `gnn/` | Graph neural network (DISABLED in production) |
| `transformer/` | Transformer model (DISABLED in production) |
| `meta_learning.py` | Meta-learner for ensemble stacking |

**Production ensemble:** LightGBM + XGBoost + LogisticRegression with stacking meta-learner, LOYO-optimized weights, and Bayesian Bradley-Terry. GNN and transformer are disabled.

**Key constraints:**
- `num_leaves <= 8`, `max_depth <= 4` for tree models (overfitting prevention)
- 15 Optuna trials on narrow search spaces
- Temperature scaling on tournament-only games from holdout year

### `src/simulation/` — Monte Carlo Simulation

50,000 Monte Carlo bracket simulations. Each simulation:
1. Draws win probabilities with noise injection
2. Simulates all tournament rounds
3. Tracks advancement frequencies per team
4. Computes Wilson score confidence intervals

Correlation parameter: 0.10 (reduced from 0.25 to avoid overstating inter-game correlation).

### `src/governance/` — Production Governance

Production validators enforce:
- Config hash matches expected locked fields
- Disabled modules are verified `false` at runtime
- Enabled modules are verified `true` at runtime
- Freeze artifact exists and is consistent
- All data paths are explicit (no auto-resolution)

Artifacts generated per production run:
- `artifacts/production_manifest_2026.json` — config + source + data hashes
- `artifacts/production_freeze_2026.json` — git commit, year partitions, freeze hashes
- `artifacts/production_governance_report_2026.json` — human-readable summary

### `src/data/ingestion/` — Data Collection

DAG-based ingestion with validation. Providers:
- `cbbpy` / `sportsipy` / `sportsdataverse` / `cbbdata` — game data
- Torvik scraper — advanced team metrics
- ESPN — public bracket picks
- Sports Reference — historical tournament results
- Kaggle — Massey Ordinals, competition seeds

### `src/exports/` — Output Generation

- Kaggle submission CSV generation
- Backtest validation against historical results

### `src/optimization/` — Bracket Strategy

Contrarian bracket optimization:
- Identifies high-EV upsets vs. public pick frequencies
- Generates bracket portfolios for pool play
- Supports configurable pool sizes

## Data Flow

```
data/raw/              → Raw scraped/ingested data (JSON)
data/raw/historical/   → Multi-year historical games, metrics, seeds
data/kaggle/           → Kaggle competition CSVs (Massey, seeds, results)
features/snapshots/    → Materialized feature tables
artifacts/             → Production manifests, freeze files, governance reports
configs/               → Pipeline configuration (production_2026.json, research_extended.json)
```

## Configuration Files

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `configs/production_2026.json` | Production config | **NO** (locked fields) |
| `configs/research_extended.json` | Research/dev config | Yes |
| `configs/team_aliases.json` | Team name normalization | Yes (append only) |
| `pyproject.toml` | Ruff + pytest config | Yes |
| `pytest.ini` | Test markers + options | Yes |
| `.env.example` | Environment variable template | Yes |

## Testing Architecture

Tests are in `tests/` with auto-marker assignment via `tests/conftest.py`:

| Directory/Pattern | Auto-marker |
|-------------------|------------|
| `tests/data_integrity/` | `data_contract` |
| `tests/evaluation/` | `integration` |
| `test_*leakage*` | `leakage` |
| `test_*calibration*` | `calibration` |
| `test_*production*` | `production` |
| `test_*freeze*` | `freeze` |
| Everything else | `unit` |

Key fixtures in `conftest.py`: team data, predictions, outcomes, production config, mock scrapers.
