# March Madness Forecaster

NCAA Tournament prediction system that generates calibrated win probabilities and optimizes bracket picks.

## How it works

Six-phase pipeline:

1. **Data ingestion** — pulls historical game data, Torvik ratings, ESPN public picks, and Kaggle Massey Ordinals
2. **Feature engineering** — builds point-in-time team features (efficiency margins, Elo, SOS, momentum)
3. **Model training** — regularized logistic regression on regular-season games (2016–2024)
4. **Calibration** — temperature scaling on tournament games to correct probability distortion
5. **Simulation** — 50k Monte Carlo bracket simulations
6. **Optimization** — contrarian pick selection to maximize expected pool score

The production model is intentionally simple: 7 domain features, single logistic regression, no ensemble or neural components. Baseline experiments showed this matches or beats more complex approaches on tournament data.

## Usage

```bash
pip install -e .

# Full production run (2026)
march-madness run-production-2026

# Or step-by-step
march-madness ingest --year 2026
march-madness sota --year 2026
```

## Pool optimization

The real edge is in pool strategy, not raw prediction accuracy. After generating probabilities, run the pool optimizer to pick contrarian brackets that maximize expected finish.

```bash
# Recommended: auto mode sweeps all probability models and construction strategies,
# deduplicates brackets, and ranks by P(1st)
march-madness optimize-pool --year 2026 --pool-size 30

# Specify your pool's payout structure
march-madness optimize-pool --year 2026 --pool-size 30 --payout winner_take_all
march-madness optimize-pool --year 2026 --pool-size 100 --payout top_3

# Production strategy: pool-aware selection over ~25 diverse candidate brackets
# (current baseline 11.2% P(1st), 15-year backtest — see CLAUDE.md for the full table)
march-madness optimize-pool --year 2026 --pool-size 30 \
  --mode meta_region_poolaware

# If you have your pool's prior-year brackets, use them instead of ESPN aggregate
# (calibrates opponent model to your actual pool's tendencies)
march-madness optimize-pool --year 2026 --pool-size 30 \
  --pool-history data/pool_hist_results.json

# Run the full MC pool backtest (15 years, all modes — takes ~30 min)
python scripts/mc_pool_backtest.py
```

See `CLAUDE.md`'s "Architectural Direction" section for the current strategy comparison table and why `meta_region_poolaware` (pool-aware selection over diverse candidates) beats simpler construction modes like `champ_first`/`e8_first`/`f4_first`.

## Maintenance

### Annual data refresh

```bash
# Scrape new season data
march-madness ingest-historical --start-season 2025 --end-season 2026
march-madness scrape-tournament-results
march-madness ingest --year 2026

# Rebuild leakage-safe feature tables after new data arrives
march-madness materialize-features
```

### Pre-tournament checklist

```bash
# Runs readiness checks: data freshness, feature drift, config validation
march-madness pre-tournament-check

# Validate model probabilities against betting market odds
march-madness validate-vs-market --model-report artifacts/sota_report.json

# Freeze the pipeline before first-round games (creates governance artifacts)
march-madness freeze-pipeline
march-madness verify-freeze
```

### Backtesting & validation

```bash
# LOYO backtest with regression gate (runs in CI nightly)
march-madness backtest-harness
march-madness backtest-harness --years "2023,2024,2025" --baseline configs/backtest_baseline.json

# Save current results as new baseline
march-madness backtest-harness --save-baseline configs/backtest_baseline.json

# Leave-one-year-out validation
march-madness loyo-validate

# RDoF audit (researcher degrees of freedom)
march-madness audit-rdof --holdout-years 2025
```

### Monitoring & snapshots

```bash
# Check data freshness and feature drift
march-madness monitor

# Snapshot / restore the data directory
march-madness snapshot
march-madness list-snapshots
march-madness restore-snapshot --name <snapshot-name>
```

## Production Path (Frozen 2026)

The 2026 tournament predictor is a frozen, governance-locked pipeline. It validates config hashes, source tree hashes, dependency versions, and freeze artifacts before execution. No code or config changes are permitted after the freeze date.

```bash
march-madness run-production-2026
```

For future years, use the year-parameterized runner:

```bash
march-madness run-production --year 2027
```

### Research Modules Not Used in Production

GNN, transformer, embedding projections, and stacking modules exist in the codebase for research purposes but are disabled in production (`enable_gnn=False`, `enable_transformer=False`, etc.). The production path hard-fails if any experimental module is enabled.

## Development

```bash
pytest           # run tests
ruff check src/  # lint
```

See `CLAUDE.md` for the current pool strategy baseline and `FINDINGS.md` for the project's dead-end ledger and architectural history.
