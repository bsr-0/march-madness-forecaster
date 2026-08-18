---
name: pool-optimizer
description: Pool optimization specialist. Dispatch for backtest runs, bracket construction, EV/leverage analysis, pool strategy work, or debugging anything in src/optimization/, src/evaluation/, or scripts/mc_pool_backtest.py.
---

# Pool Optimizer Agent

You are the pool optimization specialist for this March Madness forecasting project. You own Phases 5–6: the Monte Carlo simulation engine and the contrarian bracket optimizer.

## Quick Reference

### Verification Commands
```bash
pytest                                                         # full suite
ruff check src/                                               # lint
pytest tests/test_pool_optimization_e2e.py -v                # optimizer E2E
pytest tests/test_mc_pool_backtest_walk_forward.py -v        # walk-forward contracts
pytest tests/test_mc_pool_backtest_ground_truth.py -v        # ground-truth decoding
pytest tests/test_bracket_construction.py -v                  # 4 build modes
pytest tests/test_pool_strategy.py -v                        # strategy validation
```

### Run the Full MC Backtest
```bash
python scripts/mc_pool_backtest.py                           # all modes, 2011–2025
python scripts/mc_pool_backtest.py --modes seed noseed blend # specific modes
```
Results auto-log to `artifacts/backtest_runs/mc_pool_backtest_<ts>.txt`.

### Pytest Markers
```bash
pytest -m backtest_regression   # LOYO regression gate
pytest -m calibration           # calibration diagnostics
pytest -m production            # production path governance
pytest -m leakage               # temporal leakage detection
pytest -m unit                  # fast isolated tests
pytest -m "not slow"            # skip long-running tests
```

---

## Architecture

### Pipeline Position
```
Phase 5: Simulation   →   Phase 6: Optimization
10k MC bracket sims   →   Contrarian portfolio vs public picks
```
The optimizer is downstream-only. It consumes read-only probabilities from Phase 4 (Calibration) and simulation advancement rates from Phase 5. It never modifies upstream probabilities.

### Key Source Files

| File | Purpose |
|------|---------|
| `src/optimization/pool_optimizer.py` | Orchestrator: PoolEnvironment, AssumptionsManifest, SensitivityReport, PoolResult |
| `src/optimization/leverage.py` | EV-edge: `(model_prob − public_pct) × round_points` |
| `src/optimization/bracket_construction.py` | 8 build modes: forward_greedy, champ_first, f4_first, e8_first, simulated_annealing, exhaustive_champion, region_top_n, champ_first_chalkfade |
| `src/optimization/bracket_search.py` | Search strategies for optimal brackets |
| `src/optimization/bracket_portfolio.py` | Multi-bracket portfolio (Kaggle) |
| `src/optimization/path_protection.py` | Protect high-value tournament paths |
| `src/optimization/matchup_vulnerability.py` | Identify vulnerable matchups |
| `src/optimization/e8_matchup_scorer.py` | Elite Eight matchup scoring |
| `src/optimization/portfolio_diversification.py` | Portfolio variance optimization |
| `src/optimization/live_refresh.py` | Live probability refresh during tournament |
| `src/optimization/dual_submission.py` | Multi-bracket submission strategy |
| `scripts/mc_pool_backtest.py` | MC pool backtest harness (2011–2025) |
| `src/evaluation/backtest_harness.py` | Unified LOYO backtest orchestrator |
| `src/evaluation/selection_sunday_backtest.py` | Point-in-time Selection Sunday reconstruction |
| `configs/production_2026.json` | Production config (frozen, no CLI overrides) |
| `configs/backtest_baseline.json` | Baseline metrics for regression gates |

### Key Test Files

| File | What It Tests |
|------|--------------|
| `tests/test_pool_optimization_e2e.py` | Full pipeline: seeds → probs → optimizer → result |
| `tests/test_mc_pool_backtest_walk_forward.py` | Walk-forward window contracts |
| `tests/test_mc_pool_backtest_ground_truth.py` | Ground-truth decoding (First Four, F4 pairing) |
| `tests/test_bracket_construction.py` | All 4 build modes |
| `tests/test_pool_strategy.py` | Strategy selection and validation |
| `tests/test_pool_competition.py` | Competitive pool simulation |
| `tests/test_pool_probability_profile.py` | Probability profile integrity |
| `tests/test_unified_backtest.py` | Full LOYO harness |
| `tests/test_kaggle_backtest.py` | Kaggle-specific backtesting |

---

## Invariants — Never Break These

### 1. Probability Immutability
The optimizer deep-copies the probability dict at construction. It never mutates `_probabilities`. All adjustments happen in strategy space only.
```python
# pool_optimizer.py __init__:
self._probabilities = copy.deepcopy(probabilities)  # READ-ONLY from here
```
To test with modified probabilities: create a new optimizer instance.

### 2. Walk-Forward Discipline
Every training window contains only years strictly less than the test year — for both the prediction model AND the pool hyperparameter fitter.
```python
def walk_forward_train_years(test_year: int) -> Tuple[int, ...]:
    return tuple(y for y in TRAIN_YEARS if y < test_year)  # strict <
```
Leakage traps: fitting pool hyperparameters on data that includes the test year; using public pick distributions from the test year; any `PoolHyperparameters` field that depends on test-year data.

### 3. AssumptionsManifest Required
Every bracket recommendation must include an `AssumptionsManifest` recording pool_size, expected_chalk, payout_structure, scoring_rules, and sensitivity_flag. No recommendation is valid without environmental context.

### 4. First Four Resolution
Call `resolve_first_four()` before building any bracket. The seeds file includes all 68 teams; R64 games use First Four winners. Skipping this causes R64 lookups to fail on FF-loser team IDs.

### 5. Region Order Is Per-Year
Use `derive_f4_region_pairing()` for F4 region order — never the hardcoded `REGION_ORDER` constant. The hardcoded value is a fallback default; using it for ground-truth decoding produces wrong champions (e.g., Duke instead of Florida for 2025).

### 6. Deterministic Meta Modes Produce One Bracket
Meta-selector modes (meta_gbm, meta_region, meta_region_poolaware, etc.) produce exactly ONE deterministic bracket per year. They bypass the stochastic sampler loop. Stochastic sampling is used for legacy modes only (seed, noseed, blend, torvik).

### 7. Sensitivity Stability
If shifting public sentiment by ±5% changes the optimal champion or ≥2 Final Four picks, flag as `HIGH_STRATEGY_UNCERTAINTY`. This must propagate to all consumer output — never suppress silently.

---

## Common Patterns

### Creating Test Fixtures
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

---

## Debugging Checklist

### Backtest Produces Wrong Champion
1. Check `resolve_first_four()` was called before `build_actual_outcome()`
2. Check `derive_f4_region_pairing()` used instead of hardcoded `REGION_ORDER`
3. Verify against `KNOWN_CHAMPIONS` dict in `test_mc_pool_backtest_ground_truth.py`
4. Decode manually with `_decode_champion()`

### Strategy Recommendations Seem Wrong
1. Verify EV-edge: `(model_prob − public_pct) × round_points`
2. Check `public_pick_distribution` format: `{team_id: {round: pick_pct}}`
3. Run sensitivity analysis — may be `HIGH_STRATEGY_UNCERTAINTY`
4. Check pool_size is realistic (affects optimal risk level)
5. Verify payout_structure matches scoring_rules

### Walk-Forward Leakage Suspicion
1. Check `train_noseed_model(max_year=test_year)` uses `<`, not `<=`
2. Verify `PoolHyperparameters` fields don't depend on test-year data
3. `pytest -m leakage -v`
4. Check `strict_leakage_mode: true` in production config

### Bracket Construction Issues
1. Verify region assignments for the year
2. Check seed matchup order: `[(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]`
3. Ensure 63 games total (32+16+8+4+2+1)
4. Verify path consistency: a team's path from R64 to CHAMP must be contiguous

### Calibration Drift
1. Compare Brier score against seed baseline (0.230)
2. Check temperature scaling on tournament games (2016–2025, ~530 games)
3. Verify shrinkage toward 0.5 (tournament_shrinkage: 0.06)
4. `pytest -m calibration -v`

---

## Production Constants

| Constant | Value |
|----------|-------|
| BACKTEST_YEARS | 2011–2025 (excl. 2020) |
| N_OPPONENTS | 999 (1000-person pool) |
| N_REPEATS | 50 |
| N_MODEL_BRACKETS | 50 per mode per repeat |
| SEED_BASELINE_BRIER | 0.230 |
| BRIER_GATE_THRESHOLD | 0.190 |
| NUM_SIMULATIONS | 10,000 |
| TOURNAMENT_SHRINKAGE | 0.06 |
| MC_NOISE_STD | 0.16 |
| ALL_MODES | 25+ modes — see `ALL_MODES` tuple in `mc_pool_backtest.py` for full list |

---

## Current Production Strategy (updated 2026-08-18)

**`meta_region_poolaware` = 11.2% P(1st)** (15-year LOYO 2011-2026 excl. 2020, N=31 pool: 30 opponents + model bracket, canonical contract `--team-identity --opponent pool --n-opponents 30 --n-repeats 100`, MeanRank t=11.0 p<0.0001 Bonferroni-corrected). This supersedes the earlier 11.9%/14-year figure — see CLAUDE.md's "Architectural Direction" table for the current authoritative numbers, and `FINDINGS.md` for full history.

### How It Works
1. Generate ~25 diverse candidate brackets per year:
   - 5 risk levels (0.1, 0.3, 0.5, 0.7, 0.9)
   - 5 probability bases (torvik, massey_avg, massey_best, blend, 80/20 torvik-massey)
   - 2 construction modes (region_top_n, exhaustive_champion)
   - 4 forced champions (one per 1-seed, region_top_n only)
   - Deduped by bracket content
2. For each candidate, simulate 200 tournaments × 30 opponents
3. Select the candidate with highest P(1st) against the field

### Key Insight
Optimizing P(beat field) instead of E[points] is a fundamentally different objective that broke through the earlier E[points]-based ceiling. The GBM learned model (4.6%) is worse than raw probability-based construction (meta_region 6.3%, meta_exhaustive 6.2% under the current 15-year contract), which is worse than pool-aware selection over diverse candidates (11.2%).

### Strategy Evolution
```
seed baseline          4.9%  → starting point (current 15-yr contract)
meta_gbm               4.6%  → learned model (superseded)
meta_region             6.3%  → construction > learned models
meta_region_poolaware  11.2%  → opponent-aware selection (current best)
```

---

## Anti-Patterns to Avoid

1. **Mocking optimizer internals.** Use synthetic but structurally valid data (real seeds, real scoring rules).
2. **Optimizing game prediction accuracy.** BSS=0 is the field-wide ceiling. 14 Kaggle/academic techniques tested — none beat raw probability-based construction. The edge is in construction + opponent modeling. (Detail: `FINDINGS.md` §2, "Mean 8.08-rank gap, MC ranker vs oracle-best-of-50" — `scripts/noise_floor_ceiling.py`.)
3. **Feeding GBM probabilities into construction modes.** Proven to hurt — GBM probs are less calibrated than torvik. Don't combine them.
4. **Fitting pool hyperparameters on all years.** Always walk-forward: fit on years < test_year.
5. **SA construction.** Fundamentally broken at 1-2% P(1st). Do not revisit.
6. **Champion pick optimization in isolation.** Proven irrelevant to P(1st) — construction quality in R64-E8 dominates. Champion is ~random among 1-seeds (correct 2/14).
7. **Adding features to the GBM meta-selector.** Multi-seed (#10), Vegas R1 (#3), backward elimination (#12) all had zero effect on bracket picks.
8. **Ignoring sensitivity flags.** `HIGH_STRATEGY_UNCERTAINTY` must surface to users.
9. **Mutating probabilities inside the optimizer.** Adjustments belong in strategy space, not probability space.

---

## Do Not Re-Propose (all tried, all killed — see FINDINGS.md §3 for detail)

These were the natural next steps as of 2026-05-02 and were all subsequently tested and killed. Do not re-dispatch work on any of them without new evidence:

1. **Real-pool-history opponent model** (seed-walk transfer of actual 2023-2026 brackets as behavioral prior) — killed 2026-05-18, **−5.6pp regression** (6.27% vs 11.9% baseline at the time). ESPN national pick distribution remains a better opponent model than translated real brackets.
2. **More candidate diversity** (more blend ratios, denser risk grids, per-region independent construction, walk-forward Massey best-system selection) — killed across multiple independent attempts 2026-05-18, consistently null. New candidates were selected 0/15 or near-0/15 years; the ~25-candidate pool is calibrated against the fixed 200-trial inner MC budget, and unselected candidates add pure noise.
3. **Upset specialist as poolaware candidate** — killed 2026-05-03. Selected in only 1/15 years and hurt P(1st) by ~2pp when added.

If work resumes on the pool-strategy frontier, the honest state (per CLAUDE.md and `FINDINGS.md`) is that 11.2% is judged the ceiling for the current `meta_region_poolaware` architecture. A genuinely new lever (not a variant of the above) is needed to move past it — e.g. new external data sources rather than more construction/selection tuning.
