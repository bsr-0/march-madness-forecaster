---
name: pool-optimizer-backtest
description: Use when modifying pool optimizer logic, bracket construction, leverage analysis, or running MC pool backtests — auto-trigger when touching src/optimization/, src/evaluation/, or scripts/mc_pool_backtest.py
---

# Pool Optimizer Backtest — Development Companion

## Quick Reference

### Verification Commands (run after every change)
```bash
pytest                    # Full suite, exits on first failure (-x --tb=short)
ruff check src/           # Lint
pytest tests/test_pool_optimization_e2e.py -v   # Pool optimizer E2E
pytest tests/test_mc_pool_backtest_walk_forward.py -v  # Walk-forward contracts
pytest tests/test_mc_pool_backtest_ground_truth.py -v  # Ground-truth decoding
pytest tests/test_bracket_construction.py -v    # Bracket build modes
pytest tests/test_pool_strategy.py -v           # Strategy validation
```

### Run the Full MC Pool Backtest
```bash
python scripts/mc_pool_backtest.py              # All modes, 2011-2025
python scripts/mc_pool_backtest.py --modes seed noseed blend  # Specific modes
python scripts/mc_pool_backtest.py --save-brackets --years 2023 2024 2025 2026  # Save pick-level brackets
```
Results auto-log to `artifacts/backtest_runs/mc_pool_backtest_<ts>.txt`.
`--save-brackets` writes per-year JSON to `artifacts/backtest_brackets/backtest_brackets_{year}.json`
with full pick-level data, team-identity scores, and ranker mean_rank for each of 50 brackets per mode.

### Useful Pytest Markers
```bash
pytest -m backtest_regression     # LOYO regression gate tests
pytest -m calibration             # Calibration method tests
pytest -m production              # Production path governance
pytest -m leakage                 # Temporal leakage detection
pytest -m unit                    # Fast isolated tests only
pytest -m "not slow"              # Skip long-running tests
```

---

## Architecture Map

### Pipeline Phases (from WORKFLOW.md)
```
Data Foundation → Feature Engineering → Model Selection → Calibration → Simulation → Optimization
```

The pool optimizer lives in **Phase 6 (Optimization)**. It consumes
read-only probabilities from Phase 4 (Calibration) and simulation
advancement rates from Phase 5 (Simulation). It NEVER modifies upstream
probabilities.

### Key Source Files

| File | Purpose |
|------|---------|
| `src/optimization/pool_optimizer.py` | Orchestrator: PoolEnvironment, AssumptionsManifest, SensitivityReport, PoolResult |
| `src/optimization/leverage.py` | EV-edge metric: `(model_prob - public_pct) * round_points` |
| `src/optimization/bracket_construction.py` | 4 build modes: forward_greedy, champ_first, f4_first, e8_first |
| `src/optimization/bracket_search.py` | Search strategies for optimal brackets |
| `src/optimization/bracket_portfolio.py` | Multi-bracket portfolio generation (Kaggle) |
| `src/optimization/path_protection.py` | Protect high-value tournament paths |
| `src/optimization/matchup_vulnerability.py` | Identify vulnerable matchups |
| `src/optimization/e8_matchup_scorer.py` | Elite Eight matchup scoring |
| `src/optimization/portfolio_diversification.py` | Portfolio variance optimization |
| `src/optimization/live_refresh.py` | Live probability refresh during tournament |
| `src/optimization/dual_submission.py` | Multi-bracket submission strategy |
| `src/prediction/meta_selector.py` | **Meta-selector: 26-feature GBM + leverage baseline (v2 primary path)** |
| `scripts/mc_pool_backtest.py` | Pool backtest harness — stochastic (v1) + meta (v2) modes |
| `src/evaluation/backtest_harness.py` | Unified LOYO backtest orchestrator |
| `src/evaluation/selection_sunday_backtest.py` | Point-in-time Selection Sunday reconstruction |
| `configs/production_2026.json` | Production config (frozen, no CLI overrides) |
| `configs/backtest_baseline.json` | Baseline metrics for regression gates |

### Key Test Files

| File | Tests |
|------|-------|
| `tests/test_pool_optimization_e2e.py` | Full pipeline: seeds -> probs -> optimizer -> result |
| `tests/test_mc_pool_backtest_walk_forward.py` | Walk-forward window contracts |
| `tests/test_mc_pool_backtest_ground_truth.py` | Bracket ground-truth decoding (First Four, F4 pairing) |
| `tests/test_bracket_construction.py` | All 4 bracket construction modes |
| `tests/test_pool_strategy.py` | Strategy selection and validation |
| `tests/test_pool_competition.py` | Competitive pool simulation |
| `tests/test_pool_probability_profile.py` | Probability profile integrity |
| `tests/test_unified_backtest.py` | Full LOYO harness |
| `tests/test_kaggle_backtest.py` | Kaggle-specific backtesting |

---

## Invariants You Must Not Break

### 1. Probability Immutability
The optimizer deep-copies the probability dict at construction. It NEVER
mutates game-level probabilities. All adjustments (leverage, contrarian,
portfolio) happen in strategy space only.

```python
# In pool_optimizer.py __init__:
self._probabilities = copy.deepcopy(probabilities)  # READ-ONLY from here
```

If you need to test with modified probabilities, create a new optimizer
instance. Never patch the internal `_probabilities` dict.

### 2. Walk-Forward Discipline
Every training window must contain ONLY years strictly less than the test
year. This applies to both the prediction model AND the pool hyperparameter
fitter.

```python
# The single source of truth:
def walk_forward_train_years(test_year: int) -> Tuple[int, ...]:
    return tuple(y for y in TRAIN_YEARS if y < test_year)
```

**Leakage traps:**
- Fitting pool hyperparameters on data that includes the test year
- Using public pick distributions from the test year to tune strategies
- Any `PoolHyperparameters` field that depends on test-year data
- LOYO-style tuning that lets future crowd behavior shape past brackets

### 3. AssumptionsManifest Required
Every bracket recommendation must include an `AssumptionsManifest` recording
pool_size, expected_chalk, payout_structure, scoring_rules, and
sensitivity_flag. No recommendation is valid without environmental context.

### 4. First Four Resolution
The seeds file includes all 68 teams. R64 games use First Four winners.
You MUST call `resolve_first_four()` before building any bracket to swap
FF losers for winners — otherwise R64 lookups fail on FF-loser team IDs
and the error cascades through the entire bracket.

### 5. Region Order Is Per-Year
The F4 region pairing is NOT fixed. Use `derive_f4_region_pairing()` to
get the actual per-year region order from F4 game data. The hardcoded
`REGION_ORDER` is only a fallback default; using it directly for ground-truth
decoding produces wrong champions (e.g., Duke instead of Florida for 2025).

### 6. Bracket Construction: Meta-Selector (primary) vs Stochastic (baseline)
**Primary path (v2):** The meta-selector produces ONE deterministic bracket
per model per year. It uses multiple probability bases as input features to
a learned model that makes per-game pick decisions. No coin flips.

**Baseline comparator (v1):** The MC backtest samples stochastic brackets
(path-consistent random draws) as a baseline. This approach hit a noise
ceiling at ~5% P(1st) and is no longer the primary development path.

**Dead (D12):** Naive deterministic argmax (always pick the favorite) is
still killed. The meta-selector is NOT argmax — it's a trained model that
learns which upsets to pick based on multi-signal features.

### 7. Sensitivity Stability
If shifting public sentiment by +/-5% changes the optimal champion or >=2
Final Four picks, the recommendation gets flagged as
`HIGH_STRATEGY_UNCERTAINTY`. This flag must propagate to any output that
consumers see.

---

## Common Patterns

### Creating Test Fixtures for Pool Optimizer
```python
ESPN_SCORING = {"R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "CHAMP": 320}

env = PoolEnvironment(
    pool_size=100,
    scoring_rules=ESPN_SCORING,
    payout_structure="winner_take_all",
    public_pick_distribution=synthetic_opponent_picks,
)

optimizer = PoolOptimizer(pairwise_probs, env, model_round_probs=round_probs)
result = optimizer.optimize()
```

### Validating a PoolResult
```python
assert result.manifest is not None
assert result.manifest.pool_size == expected_pool_size
assert result.recommended_strategy in ("chalk", "balanced", "contrarian", "aggressive")
assert result.manifest.sensitivity_flag in ("STABLE", "HIGH_STRATEGY_UNCERTAINTY", "NOT_EVALUATED")
```

### Walk-Forward Test Pattern
```python
@pytest.mark.parametrize("test_year", [2011, 2015, 2019, 2023, 2025])
def test_walk_forward_contract(self, test_year):
    window = walk_forward_train_years(test_year)
    assert all(y < test_year for y in window), "walk-forward violation"
```

### Bracket Construction Mode Testing
All four modes (forward_greedy, champ_first, f4_first, e8_first) share:
- Same scoring function `_ev_score`
- Same round probabilities, seed/region inputs
- Same output type: `(picks, champion, final_four, expected_points, variance)`

Only the anchor round differs. When testing a new mode, always compare
its output structure against the existing modes.

---

## Debugging Checklist

### Backtest Produces Wrong Champion
1. Check `resolve_first_four()` was called before `build_actual_outcome()`
2. Check `derive_f4_region_pairing()` was used instead of hardcoded REGION_ORDER
3. Verify against `KNOWN_CHAMPIONS` dict in test_mc_pool_backtest_ground_truth.py
4. Decode the 63-bit bracket vector manually with `_decode_champion()`

### Strategy Recommendations Seem Wrong
1. Verify EV-edge calculation: `(model_prob - public_pct) * round_points`
2. Check that public_pick_distribution has the right format: `{team_id: {round: pick_pct}}`
3. Run sensitivity analysis — flag may be HIGH_STRATEGY_UNCERTAINTY
4. Check if pool_size is realistic (affects optimal risk level)
5. Verify payout_structure matches scoring_rules

### Walk-Forward Leakage Suspicion
1. Check that `train_noseed_model(max_year=test_year)` uses `<`, not `<=`
2. Verify `PoolHyperparameters` fields don't depend on test-year data
3. Run `pytest -m leakage -v` for automated leakage detection
4. Check `strict_leakage_mode: true` in production config

### Bracket Construction Issues
1. Verify region assignments are correct for the year
2. Check seed matchup order: `[(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]`
3. Ensure 63 games total (32 + 16 + 8 + 4 + 2 + 1)
4. Verify path consistency: a team's path from R64 to CHAMP must be contiguous

### Calibration Drift
1. Compare Brier score against seed baseline (0.230)
2. Check temperature scaling on tournament games (2016-2025, ~530 games)
3. Verify shrinkage toward 0.5 (tournament_shrinkage: 0.06)
4. Run `pytest -m calibration -v` for calibration diagnostics

---

## Production Constants

| Constant | Value | Source |
|----------|-------|--------|
| BACKTEST_YEARS | 2011-2025 (excl. 2020) | mc_pool_backtest.py |
| N_OPPONENTS | 999 (1000-person pool) | mc_pool_backtest.py |
| N_REPEATS | 50 | mc_pool_backtest.py |
| N_MODEL_BRACKETS | 50 per mode per repeat | mc_pool_backtest.py |
| SEED_BASELINE_BRIER | 0.230 | backtest_2025.py |
| BRIER_GATE_THRESHOLD | 0.190 | production_2026.json |
| NUM_SIMULATIONS | 10,000 | production_2026.json |
| TOURNAMENT_SHRINKAGE | 0.06 | production_2026.json |
| MC_NOISE_STD | 0.16 | production_2026.json |
| ALL_MODES | seed, noseed, blend, torvik, champ_first_tv, f4_first_tv, e8_first_tv, **meta_leverage, meta_gbm** | mc_pool_backtest.py |
| META_FEATURES | 26 per game (12 base probs + seeds + ESPN picks + derived + context) | meta_selector.py |
| META_GBM_DEPTH | 3 (max 8 leaves) | meta_selector.py |
| META_GBM_TREES | 50 | meta_selector.py |

---

## Anti-Patterns to Avoid

1. **Testing mock behavior instead of real code.** Pool optimizer tests
   should use synthetic but structurally valid data (real seeds, real
   scoring rules), not mocked optimizer internals.

2. **Optimizing prediction when the edge is in strategy.** Seeds explain
   ~87% of tournament outcomes. 63 games/year is insufficient for ML to
   beat seed baseline. The real edge is pool optimization, not prediction.

3. **Adding features to the model without LOYO evidence.** Any new feature
   must show BSS > 0 vs seed baseline across 16 LOYO folds. If it doesn't,
   it's noise.

4. **Fitting pool hyperparameters on all years.** Pool metagame drifts.
   Always use walk-forward: fit on years < test_year only.

5. **Confusing naive argmax with learned selection.** D12 killed naive
   `argmax(probability)` — that produces chalk. The meta-selector is a
   trained model that can pick upsets when features support it. Don't
   reject all deterministic approaches because D12 killed the dumbest one.

6. **Ignoring sensitivity flags.** If sensitivity_flag is
   HIGH_STRATEGY_UNCERTAINTY, the recommendation is fragile. Surface this
   to the user, don't silently suppress it.

7. **Mutating probabilities inside the optimizer.** All probability dicts
   are deep-copied at construction. If you find yourself wanting to modify
   them, you're in the wrong layer — adjustments belong in strategy space.
