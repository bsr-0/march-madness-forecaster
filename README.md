# March Madness Forecaster

NCAA Tournament prediction system using ensemble ML, Monte Carlo simulation, and game-theoretic bracket optimization.

## Architecture

```
Data Ingestion        Feature Engineering       ML Ensemble          Optimization
──────────────        ───────────────────       ───────────          ────────────
cbbpy/sportsipy  ──►  64-dim team vector   ──►  LightGBM (0.45)  ──►  Monte Carlo sim
Torvik scraper        75-dim matchup features   XGBoost  (0.35)      Isotonic calibration
ESPN public picks     Incremental PIT metrics   Logistic (0.20)      Leverage/contrarian
Kaggle Massey         Elo, SOS, Four Factors    CFA fusion            Bracket portfolio
```

**Key design principles:**
- Zero temporal leakage — every training sample uses only data available before game date
- Multi-year training pool (2005-2025) with exponential decay weighting
- RDoF audit framework with 58+ tracked constants across 3 tiers
- Brier score optimization (Kaggle metric since 2023)

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
# Scrapes game-level data for 2005-2025 via cbbpy/sportsipy
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
- `--historical-games-provider-priority cbbpy,sportsipy` — provider ordering

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
- `--model-complexity simple|standard|full` — feature count (8/22/all)
- `--calibration isotonic|platt|none` — calibration method

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
| `audit-metrics-coverage` | Audit coverage gaps in historical data |

## Evaluation & Auditing

```bash
# RDoF audit with holdout evaluation
march-madness audit-rdof --holdout-years 2025

# With sensitivity analysis on Tier 3 constants
march-madness audit-rdof --holdout-years 2025 --sensitivity

# Prospective evaluation on a specific year
march-madness prospective-eval --year 2024

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
│   │   │   ├── feature_engineering.py   # 64-dim team feature vector
│   │   │   └── proprietary_metrics.py   # Incremental PIT engine
│   │   ├── ingestion/             # Data collection & validation
│   │   └── scrapers/              # Torvik, ESPN, rosters, etc.
│   ├── ml/
│   │   ├── ensemble/cfa.py        # CFA model fusion
│   │   ├── calibration/           # Isotonic/Platt calibration
│   │   └── evaluation/rdof_audit.py  # RDoF audit framework
│   ├── simulation/
│   │   └── monte_carlo.py         # MC bracket simulation
│   ├── optimization/
│   │   └── leverage.py            # Contrarian bracket optimization
│   └── exports/kaggle.py          # Kaggle submission generation
├── tests/                         # 46 test files
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

- **Feature vector:** 64 team features, 75-dim matchup (64 diff + 5 absolute + 6 interaction)
- **Ensemble:** LightGBM (0.45) / XGBoost (0.35) / Logistic (0.20), CFA fusion
- **Calibration:** Isotonic regression, 70/30 chronological OOS split
- **Monte Carlo:** 50k simulations with configurable noise injection
- **Training pool:** 2005-2025, exponential decay 0.85/yr, floor 0.15
- **Elo:** K=38, cross-season carryover (0.75 * prior + 0.25 * 1500)

## License

MIT License
