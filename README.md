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

# Production strategy (meta_region_poolaware): pool-aware selection over ~25 diverse
# candidate brackets. It runs through the backtest script, not the CLI:
python scripts/mc_pool_backtest.py --team-identity --opponent pool --n-repeats 100 \
  --modes seed meta_region_poolaware

# If you have your pool's prior-year brackets, use them instead of ESPN aggregate
# (calibrates opponent model to your actual pool's tendencies)
march-madness optimize-pool --year 2026 --pool-size 30 \
  --pool-history data/pool_hist_results.json

# Run the full MC pool backtest (15 years, all modes — takes ~30 min)
python scripts/mc_pool_backtest.py
```

### What the backtest number means

`meta_region_poolaware` finishes first in **about 11% of simulated pools (95% CI 8–14%,
n=14 seasons, 2011–2025 excluding 2020)**, against 4% for a seed-only bracket in the same
harness. Read the qualifiers before quoting it:

- **Simulated tournaments, not history.** Under the canonical `--team-identity` contract
  each trial draws a tournament from the seed-model referee and scores the model bracket
  and its opponents against *that*, not the realised result. Real outcomes enter only the
  MeanScore column. The number says how often the bracket would win a pool in a plausible
  tournament, not how often it won past pools.
- **Simulated opponents.** 30 independent entries drawn from ESPN national pick rates
  (real pool brackets exist for 2023–2026 only). It assumes a winner-take-all, ESPN-scored,
  30-entry pool; it is not a universal probability of winning any pool.
- **The CI is over seasons.** Per-year P(1st) ranges 0.01–0.21, so the season-level
  standard error is ~1.4pp. Do not quote a digit after the decimal.
- **2026 is excluded** from the aggregate: it is an in-sample integration season under
  `PROSPECTIVE_2027_v2.md` and cannot be evidence of out-of-sample performance.
- The strategy was selected on this same window, so the figure is in-sample for strategy
  choice. 2027 is the first prospective season.

**The one number measured against reality, not a model of reality.** For 2023–2026 — the
only seasons a real 30-person pool exists — the production bracket that
`meta_region_poolaware` actually selected can be scored against the real tournament result
and ranked against the real pool's real scores:

| Year | Real score | Rank | Pool size | Champion picked |
|---|---:|---:|---:|---|
| 2023 | 470 | 18th | 18 | Purdue (lost R64 as a 1-seed) |
| 2024 | 1090 | 4th | 25 | Purdue |
| 2025 | 1310 | 10th | 32 | Houston |
| 2026 | 990 | 10th | 30 | Illinois |

**0 of 4 finished 1st; 0 of 4 finished top 3.** n=4 is not a rate — one different outcome
moves this by 25 points — and it neither confirms nor refutes the simulated ~11% figure
above. It beats the same strategy's own mean-of-50 `seed` baseline scored the same honest
way in every year (reproduce with `python -m scripts.real_pool_placement`; full table in
`artifacts/real_pool_placement/placement_2023_2026.txt`), but it has not yet actually won a
real pool.

Source: `artifacts/backtest_runs/mc_pool_backtest_20260829_095910.txt`. Full critique in
`AUDIT_INDEPENDENT_EVALUATOR_2027.md`; history and dead ends in `FINDINGS.md`.

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

See "What the backtest number means" above for the current pool strategy baseline, `AUDIT_INDEPENDENT_EVALUATOR_2027.md` for the independent review, and `FINDINGS.md` for the project's dead-end ledger and architectural history.
