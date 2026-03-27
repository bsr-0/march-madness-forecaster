# March Madness Forecaster

NCAA Tournament prediction system using a locked, explicit production path for calibrated probabilities and bracket simulation.

## Production Predictor (2026)

The **shipped 2026 tournament predictor** is a single frozen, deterministic path. All production runs must use the dedicated entrypoint — generic commands (`sota`, `sota-from-manifest`) are blocked from acting as production.

**Dedicated production entrypoint:**

```bash
python src/run_production_2026.py            # full production run
python src/run_production_2026.py --dry-run   # validate config + freeze artifact only
march-madness run-production-2026             # CLI alias
```

**Production constraints (hard-fail on violation):**

- **Model complexity:** `standard` — ensemble with stacking, learned feature selection, and optimized weights via LOYO cross-validation
- **Probability profile:** `production` — raw → temperature calibration → tournament shrinkage → clip
- **Mode:** `calibration`
- **Calibration method:** `temperature` (holdout-year tournament games only)
- **Training years:** 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024 (no 2020, no 2025)
- **Dev years:** 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024
- **Holdout year:** 2025
- **Target year:** 2026
- **Stacking:** enabled (`enable_stacking = true`) — meta-learner ensemble stacking
- **Feature selection:** enabled (`enable_feature_selection = true`) — automated feature importance pruning
- **Goto conversion:** enabled (`enable_goto_conversion = true`) — favourite-longshot bias correction
- **Round-weighted calibration:** enabled (`enable_round_weighted_calibration = true`) — per-round calibrator weighting
- **Bayesian Bradley-Terry:** enabled (`enable_bayesian_bt = true`) — Bayesian pairwise comparison model
- **GNN:** disabled (`enable_gnn = false`)
- **Transformer:** disabled (`enable_transformer = false`)
- **Agent orchestration:** disabled (`use_agent_orchestration = false`)
- **Embedding projections:** disabled (`enable_embedding_projections = false`)
- **Seed overrides:** disabled (`enable_seed_overrides = false`)
- **Brier sharpening:** disabled (`enable_brier_sharpening = false`)
- **Strict leakage mode:** enabled
- **Freeze file required:** enabled — freeze manifest must exist and be consistent with config (hash, years, source file coverage)
- **No "auto" paths:** every data path must be explicit; no runtime path resolution
- **No CLI overrides:** the production entrypoint exposes no flags for model complexity, calibration method, training years, or experimental modules

**Production artifacts generated per run:**

- `artifacts/production_manifest_2026.json` — machine-verifiable manifest with config hash, source hashes, data hashes, runtime inference call counts, and production flags verified at runtime
- `artifacts/production_freeze_2026.json` — freeze manifest with git commit, hashes, year partitions
- `artifacts/production_governance_report_2026.json` — human-readable governance summary

## Research Modules Not Used in Production

The following modules exist in the codebase for research purposes and are **explicitly disabled and blocked** in the shipped 2026 production predictor:

- **GNN** (`enable_gnn`) — graph neural network team embeddings
- **Transformer** (`enable_transformer`) — attention-based team representations
- **Agent orchestration** (`use_agent_orchestration`) — multi-agent pipeline coordination
- **Seed overrides** (`enable_seed_overrides`) — manual seed-matchup probability adjustments
- **Brier sharpening** (`enable_brier_sharpening`) — power-transform probability sharpening
- **Embedding projections** (`enable_embedding_projections`) — dimensionality-reduced team embeddings

Each of these is validated to be `false` by the production validator. Enabling any one of them causes an immediate `ProductionValidationError` and run termination.

## Production Modules Enabled by Default

The following modules are **required to be enabled** in the production validator and are part of the shipped 2026 predictor:

- **Stacking** (`enable_stacking`) — meta-learner ensemble stacking
- **Learned feature selection** (`enable_feature_selection`) — automated feature importance pruning
- **Goto conversion** (`enable_goto_conversion`) — favourite-longshot bias correction
- **Round-weighted calibration** (`enable_round_weighted_calibration`) — per-round calibrator weighting
- **Bayesian Bradley-Terry** (`enable_bayesian_bt`) — Bayesian pairwise comparison model
- **Market blend** (`enable_market_blend`) — Vegas cross-reference integration
- **Spread model** (`enable_spread_model`) — point spread modeling
- **Recency weighting** (`enable_recency_weighting`) — recent game weighting
- **Symmetric augmentation** (`enable_symmetric_augmentation`) — matchup symmetry data augmentation
- **Multi-year training** (`enable_multi_year_training`) — cross-season training data
- **LOYO cross-validation** (`enable_loyo_cv`) — Leave-One-Year-Out validation
- **Optimized ensemble weights** (`optimize_ensemble_weights`) — LOYO-derived weight optimization

Setting any of these to `false` causes a `ProductionValidationError`.

## Architecture

```
Data Ingestion        Feature Engineering       ML Ensemble              Probability Path
──────────────        ───────────────────       ───────────              ───────────────
cbbpy/sportsipy  ──►  tabular features with  ─► standard ensemble with ─► raw → calibrate → shrink → clip
Torvik scraper        learned feature sel.      stacking + LOYO weights    (production profile)
ESPN public picks     PIT-safe aggregates       Bayesian Bradley-Terry     goto + round-weighted cal.
Kaggle Massey         domain checks + guards    no GNN/transformer         holdout-year calibration
```


**Key design principles:**
- **Locked production path**: strict config validation hard-fails drift from the shipped setup (standard mode, production profile, calibration mode, no GNN/transformer/agent orchestration).
- **Explicit year partition**: dev years are 2016-2024; holdout year is 2025; calibration years default to holdout years only.
- **Calibration integrity**: calibrator fitting uses tournament-only rows from calibration years (season-level OOS by default).
- **Distribution-shift mitigation**: learned feature selection with train/eval drift checks prunes volatile features automatically.
- **Point-in-time features**: every training sample uses only data available before game date, with per-year tournament cutoff dates.


**What the production path does not use:** agent orchestration, GNN/transformer embeddings, seed overrides, Brier sharpening, and embedding projections are all disabled in locked production mode.

## Installation

**Requirements:** Python 3.9+

```bash
# Option A: install as package (recommended)
pip install -e .

# Option B: install dependencies only
pip install -r requirements.txt
```

After `pip install -e .`, use the `march-madness` command. Otherwise use `python -m src`.

Both forms are shown below; pick whichever you installed.

## Quick Start

### 1. Ingest historical data (multi-year training)

```bash
# Scrapes game-level data via cbbpy/sportsipy (defaults: 2022-2025)
march-madness ingest-historical

# Or specify a wider range for full training pool:
march-madness ingest-historical --start-season 2005 --end-season 2025

# Or with python -m:
python -m src ingest-historical --start-season 2005 --end-season 2025
```

Output: `historical_games_{year}.json`, `team_metrics_{year}.json`, `tournament_seeds_{year}.json` in `data/raw/historical/`.

Options:
- `--kaggle-dir data/kaggle` — load Massey Ordinals per season
- `--skip-torvik` — skip Torvik historical backfill
- `--include-pbp` — include play-by-play events (larger files)
- `--max-games-per-season 50` — cap for smoke tests

### 2. Scrape rosters

```bash
march-madness scrape-rosters --start-year 2005 --end-year 2026
march-madness enrich-rosters --start-year 2005 --end-year 2026
```

Builds `cbbpy_rosters_{year}.json` with player-level box score data, then cross-references across years for transfer/eligibility tracking.

### 3. Ingest current-year data

```bash
march-madness ingest --year 2026
```

Collects teams, rosters, Torvik, public picks, etc. into `data/raw/` and writes a manifest (`manifest_2026.json`).

Options:
- `--skip-torvik`, `--skip-public-picks`, `--skip-sports-reference`, `--skip-rosters`
- `--kaggle-dir data/kaggle` — load Kaggle competition data
- `--allow-invalid-payloads` — don't fail on schema errors
- `--historical-games-provider-priority sportsdataverse,cbbpy,sportsipy,cbbdata` — provider ordering

### 4. Run the full pipeline

```bash
march-madness sota --year 2026 --scrape-live
```

Runs: feature engineering, model training, calibration, Monte Carlo simulation, bracket optimization. Output: `sota_report.json`.

Key flags:
- `--simulations 50000` — Monte Carlo simulations (default)
- `--pool-size 100` — bracket pool size for strategy optimization
- `--kaggle-dir data/kaggle` — for Massey Ordinals and seeds
- `--enable-bracket-portfolio` — generate diverse bracket set for Kaggle
- `--model-complexity simple|standard|full` — default is `standard` in production
- `--calibration temperature|isotonic|platt|none` — default is `temperature`
- `--probability-profile production|experimental` — default is `production`
- `--mode calibration|ev` — default is `calibration`
- `--dev-years 2016,...,2024` and `--holdout-years 2025` — locked production split
- `--calibration-years 2025` — optional override (defaults to holdout years)

### 5. Run from ingest manifest (combines steps 3+4)

```bash
march-madness sota-from-manifest \
  --manifest data/raw/manifest_2026.json \
  --output sota_report.json \
  --simulations 50000
```

## All Commands

| Command | Description |
|---------|-------------|
| `sota` | Run full prediction pipeline |
| `run-production-2026` | Run frozen 2026 production-only path with governance artifacts |
| `sota-from-manifest` | Run pipeline using an ingestion manifest |
| `ingest` | Collect current-year data sources |
| `ingest-historical` | Scrape historical seasons for training |
| `materialize-features` | Build leakage-safe feature tables |
| `scrape-rosters` | Scrape cbbpy box scores for roster data |
| `enrich-rosters` | Cross-reference rosters for transfers/eligibility |
| `audit-rdof` | Run researcher degrees of freedom audit |
| `freeze-pipeline` | Freeze pipeline constants for reproducibility |
| `verify-freeze` | Verify a freeze artifact matches current code |
| `prospective-eval` | Evaluate on a holdout year |
| `calibrate-mc` | Calibrate Monte Carlo noise parameter |
| `download-kaggle` | Download Kaggle competition CSVs |
| `kaggle-export` | Generate Kaggle submission CSV |
| `loyo-validate` | Run Leave-One-Year-Out validation across historical years |
| `backtest-kaggle` | Evaluate predictions against historical Kaggle results |
| `backtest-unified` | Run unified backtest (Kaggle calibration + ESPN bracket pool) |
| `validate-metrics` | Validate proprietary metrics against public data |
| `scrape-tournament-results` | Scrape historical tournament results from Sports Reference |
| `repair-dates` | Re-fetch and repair game dates in historical JSON files |
| `audit-metrics-coverage` | Audit coverage gaps in historical data |

## Evaluation & Auditing

```bash
# RDoF audit with holdout evaluation
march-madness audit-rdof --holdout-years 2025

# With sensitivity analysis on Tier 3 constants
march-madness audit-rdof --holdout-years 2025 --sensitivity

# Prospective evaluation on a specific year (requires a freeze artifact)
march-madness prospective-eval --freeze-file pipeline_freeze.json --year 2024

# Freeze pipeline constants before tournament
march-madness freeze-pipeline
march-madness verify-freeze
```

## Kaggle Export

```bash
# Download competition data
march-madness download-kaggle --output-dir data/kaggle

# Generate submission
march-madness kaggle-export \
  --manifest data/raw/manifest_2026.json \
  --sample-submission data/kaggle/SampleSubmissionStage1.csv \
  --kaggle-teams data/kaggle/MTeams.csv \
  --output kaggle_submission.csv
```

## Project Structure

```
march-madness-forecaster/
├── src/
│   ├── main.py                    # CLI entry point
│   ├── pipeline/
│   │   └── sota.py                # Main prediction pipeline
│   ├── data/
│   │   ├── features/
│   │   │   ├── feature_engineering.py   # 86-dim team feature vector
│   │   │   └── proprietary_metrics.py   # Incremental PIT engine
│   │   ├── ingestion/             # Data collection & validation
│   │   └── scrapers/              # Torvik, ESPN, rosters, etc.
│   ├── ml/
│   │   ├── ensemble/cfa.py        # LightGBM/XGBoost/Logistic ensemble
│   │   ├── calibration/           # Temperature scaling calibration
│   │   └── evaluation/rdof_audit.py  # RDoF audit framework
│   ├── simulation/
│   │   └── monte_carlo.py         # MC bracket simulation
│   ├── optimization/
│   │   └── leverage.py            # Contrarian bracket optimization
│   └── exports/kaggle.py          # Kaggle submission generation
├── tests/                         # 67 test files
├── data/                          # Historical & current-year data
├── setup.py
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/
pytest tests/ --cov=src
```

## Technical Details

- **Production features:** 86-dim team feature vector with learned feature selection (automated importance pruning)
- **Production ensemble:** standard ensemble with stacking, LOYO-optimized weights, Bayesian Bradley-Terry, no GNN, no transformer
- **Calibration:** temperature scaling on tournament-only calibration years (default holdout year 2025)
- **Monte Carlo:** 50k simulations with configurable noise injection
- **Training partition:** dev years 2016-2024, holdout year 2025 (production-locked defaults)
- **Elo:** K=20, cross-season carryover (0.75 * prior + 0.25 * 1500)

## License

MIT License
