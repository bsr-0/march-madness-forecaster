# March Madness Forecaster

NCAA Tournament prediction system using a locked, explicit production path for calibrated probabilities and bracket simulation.

## Architecture

```
Data Ingestion        Feature Engineering       ML Ensemble            Probability Path
──────────────        ───────────────────       ───────────            ───────────────
cbbpy/sportsipy  ──►  fixed tabular features ─► simple fixed-weight ─► raw → calibrate → shrink → clip
Torvik scraper        PIT-safe aggregates       ensemble (no stacking)   (production profile)
ESPN public picks     no learned feature sel.   no GNN/transformer       calibrated probabilities only
Kaggle Massey         domain checks + guards    no agent orchestration   holdout-year calibration
```


**Key design principles:**
- **Locked production path**: strict config validation hard-fails drift from the shipped setup (simple mode, production profile, calibration mode, no GNN/transformer/stacking/agent/market blending).
- **Explicit year partition**: dev years are 2016-2024; holdout year is 2025; calibration years default to holdout years only.
- **Calibration integrity**: calibrator fitting uses tournament-only rows from calibration years (season-level OOS by default).
- **Distribution-shift mitigation**: production simple feature set excludes tournament-only and coverage-volatile terms (seed/travel/external blend) and applies fixed-feature selection with train/eval drift checks.
- **Point-in-time features**: every training sample uses only data available before game date, with per-year tournament cutoff dates.


**What the production path does not use:** experimental post-processing, agent orchestration, GNN/transformer embeddings, stacking, and market blending are all disabled in locked production mode.

## Installation

**Requirements:** Python 3.8+

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
- `--model-complexity simple|standard|full` — default is `simple` in production
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
│   │   │   ├── feature_engineering.py   # 79-dim team feature vector
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

- **Production simple features:** 8 fixed, stable features (efficiency/SOS/Elo/win%/FT%/momentum)
- **Production ensemble:** simple fixed-weight tabular ensemble with no stacking, no GNN, no transformer
- **Calibration:** temperature scaling on tournament-only calibration years (default holdout year 2025)
- **Monte Carlo:** 50k simulations with configurable noise injection
- **Training partition:** dev years 2016-2024, holdout year 2025 (production-locked defaults)
- **Elo:** K=20, cross-season carryover (0.75 * prior + 0.25 * 1500)

## License

MIT License
