# Project Status & Direction

**Last updated:** 2026-04-02

## Key Finding: ML Prediction Adds No Value Over Seeds

Two independent backtests across 17 tournament years (2008-2025, excluding 2020) established
that the ML prediction engine adds no measurable value over a seed-based lookup table:

- **Brier Skill Score (BSS) = 0** across all model configurations tested (7-feature logistic
  regression, 27-feature ensemble with stacking, LightGBM, XGBoost, SpreadRegressor)
- **Pool value backtest** (853 matchups): the top quintile by pool-value score had a 19.4% upset
  hit rate — *below* the 23.2% base rate. A contrarian strategy lost 7.1% vs chalk.
- **Round segmentation** (1,071 games): seed dominance varies by round (Elite 8 has 42.6% upset
  rate with avg seed gap 3.3), but seed-independent signals (tempo, rebounding, turnovers)
  showed only +0.2pp lift even in the most favorable round — statistically negligible.

These results were reviewed by an LLM council (5 advisors, unanimous agreement) which
recommended pivoting away from prediction accuracy.

### What This Means

The fundamental problem isn't the model — it's the signal ceiling. With ~63 games per tournament
and high variance, there simply isn't enough exploitable structure in team-level season averages
to beat the information already encoded in seedings. This is consistent with academic literature
on NCAA tournament prediction.

## New Direction: Bracket Pool Optimization

The pivot is from "predict who wins better" to "which bracket maximizes expected payout given
what everyone else picks." This is a game theory problem against pool opponents, not a
prediction accuracy problem.

**Core insight:** You don't need to predict better than seeds to win a bracket pool. You need
to predict *differently* from the crowd in spots where the crowd is wrong. The value comes
from opponent modeling and portfolio construction, not from a better probability estimate.

### What the Pool Optimizer Needs

1. **Seed-based win probabilities** (already have — the production logistic regression)
2. **Public pick distributions** (already have — ESPN/CBS/Yahoo scraping)
3. **Portfolio construction** (partially built — `src/optimization/`, `BracketPortfolioGenerator`)
4. **Opponent modeling** (not yet built — how does the average pool entrant fill a bracket?)
5. **Pool-EV scoring** (not yet built — expected payout given your bracket vs N opponents)

### What's NOT Needed

- Better Brier scores
- More ML model families
- Feature engineering for prediction accuracy
- LOYO cross-validation for model selection (keep only for seed baseline integrity)
- Research loop automation (was optimizing Brier, which is the wrong metric)

## Dead Code Removed (~8,050 LOC)

The following code was removed as part of the pivot cleanup. All removals were verified
with grep to confirm zero downstream dependencies before deletion.

### Phase 1: Unused Pipeline Stages (-2,544 LOC)

Six pipeline stages that were never imported by the pipeline runner:

| File | LOC | Why Removed |
|------|-----|-------------|
| `src/pipeline/stages/hyperparameter_optimization.py` | 584 | Never imported by pipeline runner |
| `src/pipeline/stages/simulation_loop.py` | 410 | Never imported by pipeline runner |
| `src/pipeline/stages/probability_calibration.py` | 352 | Never imported by pipeline runner |
| `src/pipeline/stages/robustness_stage.py` | 178 | Never imported by pipeline runner |
| `src/pipeline/stages/model_trainer.py` | 155 | Never imported by pipeline runner |
| `src/pipeline/stages/reporter.py` | 69 | Never imported by pipeline runner |
| `tests/test_phases_7_to_10.py` | 575 | Exclusively tested deleted stages |
| `tests/test_robustness_stage.py` | 121 | Exclusively tested deleted stage |

### Phase 2: Deprecated Scrapers (-125 LOC)

| File | Change | Why |
|------|--------|-----|
| `src/data/scrapers/betting_markets.py` | Removed `FanDuelScraper`, `DraftKingsScraper` | Explicitly marked deprecated, replaced by `TheOddsAPIScraper` |
| `src/data/scrapers/espn_picks.py` | Removed `_try_html_page()` | Always returned None |
| `src/data/scrapers/__init__.py` | Removed re-exports | Cleaned up barrel file |

### Phase 3: Test-Only ML Modules (-1,160 LOC)

Modules with tests but never imported by the pipeline or CLI:

| File | LOC | Why Removed |
|------|-----|-------------|
| `src/ml/time_series/elo_temporal.py` + `__init__.py` | 270 | Never used outside tests |
| `src/ml/ranking/lambdamart.py` + `__init__.py` | 246 | Never used outside tests |
| `tests/test_elo_temporal.py` | 141 | Exclusively tested deleted module |
| `tests/test_lambdamart.py` | 123 | Exclusively tested deleted module |

Also cleaned up references in `model_registry.py`, `hyperparameter_tuning.py`,
`test_pipeline_ml_coverage.py`, and `conftest.py`.

### Phase 4: Orphaned Dataclasses (-88 LOC)

Removed from `src/pipeline/stages/__init__.py`:
- `OptimizedHyperparams` — contract for deleted Phase 7
- `CalibratedModelPredictions` — contract for deleted Phase 8
- `EnsemblePredictions` — contract for deleted Phase 9
- `SimulationLoopOutput` — contract for deleted Phase 10

### Phase 5: Dead Evaluation Modules (-4,221 LOC)

| File | LOC | Why Removed |
|------|-----|-------------|
| `src/ml/evaluation/model_comparison.py` | 390 | No imports anywhere |
| `src/ml/evaluation/robustness_suite.py` | 313 | No imports anywhere |
| `src/ml/evaluation/dual_loop_evaluator.py` | 613 | No imports anywhere |
| `src/ml/evaluation/explainability_plots.py` | 433 | No imports anywhere |
| `src/ml/evaluation/feature_explainability.py` | 994 | Only imported by explainability_plots (also deleted) |
| `src/ml/evaluation/model_card.py` | 434 | No imports anywhere |
| `tests/test_robustness_and_comparison.py` | 378 | Exclusively tested deleted modules |
| `tests/test_model_card.py` | 114 | Exclusively tested deleted module |
| `tests/test_feature_explainability.py` | 552 | Exclusively tested deleted module |

## What Remains and Why

### Active Infrastructure (keep)

- **Production prediction pipeline** (`src/pipeline/`) — still provides seed-based probabilities
  that feed the pool optimizer
- **Feature engineering** (`src/data/features/`) — computes team metrics used by production model
- **Monte Carlo simulation** (`src/simulation/`) — bracket simulation, needed for pool EV
- **Bracket optimization** (`src/optimization/`) — portfolio construction, the core of the pivot
- **Data scrapers** (`src/data/scrapers/`) — ESPN picks, Torvik stats, betting odds
- **Calibration** (`src/ml/calibration/`) — probability calibration for both paradigms
- **Ensemble models** (`src/ml/ensemble/`) — bayesian_bt, spread_model, cfa all actively used
- **LOYO protocol** (`src/ml/evaluation/loyo_protocol.py`) — validates seed baseline integrity
- **Unified backtest** (`src/ml/evaluation/unified_backtest.py`) — reusable for pool validation

### Deprioritized but Wired In (disable at runtime, don't delete)

- **Research loop** (`src/ml/research/`) — ~15K LOC actively wired into pipeline. Optimizes
  Brier (wrong metric). Should be disabled in production, kept for audit trail.
- **GNN/Transformer** (`src/ml/gnn/`, `src/ml/transformer/`) — behind `enable_gnn=false` and
  `enable_transformer=false` config flags. Properly gated, no runtime cost.

## Backtest Artifacts

The following backtest scripts and results informed this pivot:

- `scripts/backtest_pool_value.py` — 17-year pool value backtest (result: negative)
- `scripts/backtest_by_round.py` — round-segmented analysis (result: SI signals flat)
- Council decisions and open items: `COUNCIL_LESSONS.md` (25 transcripts consolidated 2026-04-13; raw transcripts deleted — see §3 for session index).

## Pool Optimization Pipeline (Completed)

The game theory pivot is now wired end-to-end. New modules:

| Module | Purpose |
|--------|---------|
| `src/prediction/seed_probabilities.py` | Seed-based pairwise win probabilities (replaces ML model) |
| `src/simulation/ratings_opponent_model.py` | Converts 56+ external rating systems into opponent pick distributions |
| `src/cli/pool_cmds.py` | `optimize-pool` CLI entry point |
| `scripts/backtest_pool_strategy.py` | Historical backtest against 2018-2025 data |

### How It Works

1. **Seed probabilities as truth**: Historical win rates (1985-2025) provide pairwise matchup
   probabilities — equivalent to ML model output but simpler and proven equal (BSS=0).
2. **External ratings as opponent model**: Each of 56+ rating systems (Massey Ordinals) represents
   a "type of informed picker." Their consensus approximates the field's bracket behavior.
3. **ESPN public picks blended in**: When available (2018-2025), real ESPN data gets 60% weight,
   ratings-derived picks get 30%, seed-based fallback gets 10%.
4. **Pool optimizer**: `PoolOptimizer` finds leverage picks (model > public), fade picks
   (public > model), and recommends strategy (chalk/balanced/contrarian/aggressive/targeted).
5. **Sensitivity analysis**: Tests stability under ±5% pick distribution shifts.

### CLI Usage

```bash
python -m src optimize-pool --year 2026 --pool-size 100 --payout winner_take_all
python -m src optimize-pool --year 2024 --pool-size 500 --payout top_3 --scoring flat
```

### Tests

- `tests/test_seed_probabilities.py` — 7 tests (symmetry, format, matchup correctness)
- `tests/test_ratings_opponent_model.py` — 7 tests (blending, fallback, format)
- `tests/test_pool_optimization_e2e.py` — 4 integration tests (full pipeline flow)

## Next Steps

1. **Run historical backtest** (`scripts/backtest_pool_strategy.py`) — validate whether
   contrarian optimization actually outperforms the field across 2018-2025
2. **Tune blend weights** — the 60/30/10 ESPN/ratings/seed split is a starting point;
   backtest results may suggest different weights
3. **Add bracket portfolio generation** — wire `BracketPortfolioGenerator` into the CLI
   to output complete printable brackets, not just leverage picks
4. **Live tournament mode** — connect to real-time data feeds during March Madness
