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

- **Model complexity:** `simple` — single regularized logistic regression on 7 domain-knowledge features (no ensemble, no tree models)
- **Probability profile:** `production` — raw → temperature calibration → tournament shrinkage → clip
- **Mode:** `calibration`
- **Calibration method:** `temperature` (tournament games from ALL dev+holdout years 2016-2025)
- **Training data:** regular-season games only from 2016-2024 (~2,200/year, ~17,600 total). Tournament games excluded from training for clean domain separation.
- **Calibration data:** tournament games from 2016-2025 (~530 samples). Genuinely out-of-sample since model trains only on regular season.
- **Dev years:** 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024
- **Holdout year:** 2025
- **Target year:** 2026
- **Features (7):** `diff_adj_off_eff`, `diff_adj_def_eff`, `diff_sos_adj_em`, `diff_elo_rating`, `diff_win_pct`, `diff_free_throw_pct`, `diff_momentum` (SIMPLE_FEATURE_SET)
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

- **Goto conversion** (`enable_goto_conversion`) — favourite-longshot bias correction
- **Round-weighted calibration** (`enable_round_weighted_calibration`) — per-round calibrator weighting
- **Bayesian Bradley-Terry** (`enable_bayesian_bt`) — Bayesian pairwise comparison model
- **Market blend** (`enable_market_blend`) — Vegas cross-reference integration
- **Recency weighting** (`enable_recency_weighting`) — recent game weighting
- **Symmetric augmentation** (`enable_symmetric_augmentation`) — matchup symmetry data augmentation
- **Multi-year training** (`enable_multi_year_training`) — cross-season training data
- **LOYO cross-validation** (`enable_loyo_cv`) — Leave-One-Year-Out validation

Setting any of these to `false` causes a `ProductionValidationError`.

**Modules disabled by default (simple mode):**
- **Stacking** — not needed with single model
- **Learned feature selection** — fixed 7-feature set, no pruning needed
- **Spread model** — tree-based, skipped in simple mode
- **Optimized ensemble weights** — no ensemble to weight

## Architecture

```
Data Ingestion        Feature Engineering       Model                    Probability Path
──────────────        ───────────────────       ─────                    ───────────────
cbbpy/sportsipy  ──►  7 domain features     ─► logistic regression   ─► raw → calibrate → shrink → clip
Torvik scraper        (SIMPLE_FEATURE_SET)      (regular season only)    (production profile)
ESPN public picks     PIT-safe aggregates       + BT/Massey blends       multi-year tournament cal.
Kaggle Massey         domain checks + guards    no ensemble/trees        goto + round-weighted cal.
```


**Key design principles:**
- **Locked production path**: strict config validation hard-fails drift from the shipped setup (simple mode, production profile, calibration mode, no GNN/transformer/agent orchestration).
- **Two-stage domain adaptation**: train on regular-season games (large N, ~17,600), calibrate on tournament games (domain-matched, ~530 across 9 years). Tournament games are genuinely OOS for calibration.
- **Calibration integrity**: temperature scaling fitted on tournament-only games from ALL dev+holdout years (2016-2025). Much more data than single-year calibration.
- **Simplicity over complexity**: baseline experiment proved 7-feature logistic regression matches or beats the 27-feature ensemble with stacking. Additional features and tree models add no value on tournament data (BSS ≈ 0).
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
- `--model-complexity simple|standard|full` — default is `simple` (7-feature logistic regression)
- `--calibration temperature|isotonic|platt|none` — default is `temperature`
- `--probability-profile production|experimental` — default is `production`
- `--mode calibration|ev` — default is `calibration`
- `--dev-years 2016,...,2024` and `--holdout-years 2025` — locked production split
- `--calibration-years 2016,...,2025` — defaults to all dev+holdout years

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
| `baseline-experiment` | Tournament-only baseline experiment (LOYO-CV, compares feature sets vs seed baseline) |
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
│   │   ├── ensemble/cfa.py        # Model infrastructure (logistic regression in simple mode)
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

- **Production features:** 7 domain-knowledge features (SIMPLE_FEATURE_SET) — fixed, no learned selection
- **Production model:** single regularized logistic regression (no ensemble, no tree models). Baseline experiment confirmed adding features/models beyond this adds no value (BSS ≈ 0 vs seed baseline).
- **Training:** regular-season games only (~17,600 samples across 8 years). Tournament games excluded for clean domain separation.
- **Calibration:** temperature scaling on tournament games from ALL dev+holdout years (2016-2025, ~530 samples)
- **Monte Carlo:** 50k simulations with configurable noise injection
- **Training partition:** dev years 2016-2024, holdout year 2025 (production-locked defaults)
- **Elo:** K=20, cross-season carryover (0.75 * prior + 0.25 * 1500)

## License

MIT License
