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
# --n-opponents 29 is required: it is the fallback for seasons with no recorded
# pool, and omitting it silently measures a 1000-person field instead of a 30-person one.
python scripts/mc_pool_backtest.py --team-identity --opponent pool \
  --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware

# If you have your pool's prior-year brackets, use them instead of ESPN aggregate
# (calibrates opponent model to your actual pool's tendencies)
march-madness optimize-pool --year 2026 --pool-size 30 \
  --pool-history data/pool_hist_results.json

# Run the full MC pool backtest (15 years, all modes — takes ~30 min)
python scripts/mc_pool_backtest.py
```

### What the backtest number means

`meta_region_poolaware` finishes first in **about 12% of simulated pools (95% CI 9–15%,
n=14 seasons, 2011–2025 excluding 2020)**, against 4.0% for a seed-only bracket in the same
harness. Per-season P(1st) ranges from 2% to 21%. Read the qualifiers before quoting it:

- **Simulated tournaments, not history.** Under the canonical `--team-identity` contract
  each trial draws a tournament from the seed-model referee and scores the model bracket
  and its opponents against *that*, not the realised result. Real outcomes enter only the
  MeanScore column. The number says how often the bracket would win a pool in a plausible
  tournament, not how often it won past pools.
- **Simulated opponents, and the pool size is an assumption.** Opponents are independent
  draws from a pick distribution: the real pool's own picks for 2023–2025 (at that pool's
  real size — 18, 25 and 32 entries), and ESPN national pick rates at an assumed 30-entry
  pool for every earlier season. It assumes winner-take-all with ESPN scoring; it is not a
  universal probability of winning any pool. **P(1st) is mechanically pool-size dependent**
  — the same strategy scores roughly 2.5x worse in a 1000-entry field than a 30-entry one —
  so the pool size is part of the claim, not a detail. Reproduce with
  `--team-identity --opponent pool --n-opponents 29 --n-repeats 100`; omitting
  `--n-opponents` silently measures a 1000-person field.
- **The CI is over seasons.** The season-level standard error is 1.5pp. Do not quote a
  digit after the decimal — the difference between "11.2%" and "12.0%" is half of one
  standard error.
- **2026 is excluded** from the aggregate: it is an in-sample integration season under
  `PROSPECTIVE_2027_v2.md` and cannot be evidence of out-of-sample performance.
- The strategy was selected on this same window, so the figure is in-sample for strategy
  choice. 2027 is the first prospective season.
- **It does not describe any bracket this site currently displays.** The number measures
  `meta_region_poolaware` as the backtest builds it. The candidate bank the site serves is
  built by `scripts/experiments/build_candidate_artifact.py`, which uses different rating
  sources, a different risk grid, and no exhaustive- or forced-champion candidates. Quote
  this figure for the strategy, not for a bracket on the page.

**The one number measured against reality, not a model of reality.** For 2023–2026 — the
only seasons with a recorded real pool — the bracket `meta_region_poolaware` actually
selected can be scored against the real tournament result and ranked against the real
pool's real scores:

| Year | Real score | Rank | Pool size | Champion picked |
|---|---:|---:|---:|---|
| 2023 | 460 | 18th | 18 | Purdue (lost in the R64 as a 1-seed) |
| 2024 | 1120 | 4th | 25 | Purdue |
| 2025 | 1330 | 10th | 32 | Houston |
| 2026 | 840 | 12th | 30 | Florida |

**0 of 4 finished 1st; 0 of 4 finished top 3.** n=4 is not a rate — one different outcome
moves this by 25 points — and it neither confirms nor refutes the simulated figure above.
It beats the same strategy's own mean-of-50 `seed` baseline scored the same honest way in
every year (reproduce with `python -m scripts.real_pool_placement`; full table in
`artifacts/real_pool_placement/placement_2023_2026.txt`), but it has not yet actually won a
real pool.

**Why this number moved, and why it is not a cherry-pick.** It was published as "11.2%",
then "about 11%", and is measured here at 12.0%. The figure did not improve because
anything was tuned: `b73d351` (2026-09-06, *"every season shipped a wrong R64"*) fixed the
play-in resolution, which changed the field in **every** season. Every P(1st) published
before that date — 11.2%, 11.33%, 10.47%, 11.87% — was computed on brackets containing
teams that never played the Round of 64. Those figures are void rather than superseded, and
the spread among them is itself the argument for quoting a CI instead of a digit: they all
sit inside a single standard error of each other.

Source: `artifacts/headline_measurement/canonical_2011_2025_n14.txt`, produced by the
command above on the current code. Full critique in `AUDIT_INDEPENDENT_EVALUATOR_2027.md`;
history and dead ends in `FINDINGS.md`.

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
