# Session Summary: Gap Analysis Roadmap Execution

**Date:** 2026-05-04
**Duration:** Full session (~4 hours)
**Focus:** Execute all 5 phases of `docs/gap-analysis-and-roadmap.md`

---

## Results Summary

| Phase | Goal | Result | P(1st) |
|-------|------|--------|--------|
| **Phase 0** | Foundation cleanup | **DONE** | 11.2% (no change) |
| **Phase 1** | Pool-specific opponent model | **REJECTED** | 4.5-9.6% (worse) |
| **Phase 2** | Regime-adaptive candidates | **REJECTED** | 10.8% (worse) |
| **Phase 3** | Champion pick specialization | **REJECTED** | 11.33% (neutral) |
| **Phase 4** | Candidate diversity expansion | **REJECTED** | 10.1% (worse) |

**Conclusion:** 11.2% P(1st) from `meta_region_poolaware` is the production ceiling for the current architecture. All 4 improvement axes failed the acceptance gate (≥8/15 years improved).

---

## Phase 0: Foundation (DONE)

### Tournament Data Fixes (11 games across 2 years)

**2023 (6 fixes):**
- S16: Alabama/SDSU — wrong winner (SDSU won 71-64)
- S16: Creighton/Princeton — wrong winner (Creighton won 86-75)
- S16: Houston/Miami — wrong winner (Miami won 89-75)
- S16: Texas/Xavier — wrong score (71, not 77)
- S16: Gonzaga/UCLA — wrong winner (Gonzaga won 79-76)
- E8: Houston/Miami → wrong teams entirely (actual: Miami 88, Texas 81)

**2024 (5 fixes):**
- R64: Saint Mary's/Grand Canyon — GCU won 75-66 (12-seed upset)
- R32: Alabama/Clemson → wrong game (actual: Alabama 72, GCU 61)
- R32: Grand Canyon/Saint Mary's → wrong game (actual: Clemson 72, Baylor 64)
- S16: Clemson/Arizona — wrong winner (Clemson won 77-72)
- E8: Alabama/Clemson — scores swapped (Alabama won 89-82)

### Other Phase 0 Items
- Added `2027: date(2027, 3, 16)` to `TOURNAMENT_START_DATES`
- Updated test gate in `test_pipeline_stages.py`
- O2 (four-factors) and O5 (MC stability) confirmed already closed

---

## Phase 1: Pool-Specific Opponent Model (REJECTED)

### Bug Fix: `build_pool_behavioral_model()` compounding error
The behavioral model computed **per-game conditional win rates** instead of **round advancement probabilities**. This produced near-uniform distributions in late rounds (chalk rate 0.33 for CHAMP vs actual 0.98). Fixed by directly counting per-seed advancement rates from pool data.

**Before fix:** CHAMP MAE 0.41 (40x worse than ESPN)
**After fix:** CHAMP MAE 0.0107 (within 4% of ESPN), chalk rate 0.976 vs actual 0.978

### Blend Weight Sweep Results
| Weight | P(1st) | vs Baseline |
|--------|--------|-------------|
| 0.0 (pure ESPN) | 9.60% | -1.6pp |
| 0.1 | 8.67% | -2.5pp |
| 0.2 | 8.13% | -3.1pp |
| 0.3 | 7.20% | -4.0pp |
| 0.7 | 4.53% | -6.7pp |

**Root cause:** Baseline uses actual pool brackets for 2023-2026 — no synthetic model can beat ground truth.

---

## Phase 2: Regime-Adaptive Candidates (REJECTED)

Used `compute_field_volatility_signal()` (r=-0.668 with F4 seed-mean) to skew risk levels. Signal correctly classifies years (3 chalk, 7 chaos, 5 mixed). P(1st) = 10.80% vs 11.20%.

**Root cause:** MC selector already implicitly adapts by choosing the best candidate against simulated outcomes.

---

## Phase 3: Champion Pick Specialization (REJECTED)

Diagnostic confirmed champion among 1-seeds is random (barthag rank distribution: 3/4/3/1 across 11 years). Added 12 forced-champion variants (4 × 3 risk levels) + 4 massey-based. P(1st) = 11.33% (neutral, 2/15 improved).

**Root cause:** MC selector already evaluates exhaustive_champion mode. More variants don't help because the binding constraint is information, not coverage.

---

## Phase 4: Candidate Diversity Expansion (REJECTED)

Added elo, odds, 70/30 torvik/massey blend to probability base sweep. Candidate count rose from ~25-28 to ~36-50. P(1st) = 10.13%.

**Root cause:** Selection noise. With 200 MC trials, the selector degrades when evaluating 40+ candidates. Same mechanism that killed the upset detector.

---

## Key Architectural Insights Discovered

1. **The 200 MC trial budget is the binding constraint.** It's sufficient for ~25 candidates but degrades with more. The architecture's ceiling isn't candidate quality — it's selection precision.

2. **The existing opponent model is near-optimal.** Year-specific pool brackets + ESPN picks are already the best available opponent data. Cross-year behavioral models can't compete.

3. **Champion pick is genuinely random among 1-seeds.** No pre-tournament rating distinguishes the winner. Improvement requires new data sources (markets, injuries, momentum).

4. **The poolaware selector is already regime-adaptive by construction.** Diverse candidates + MC selection naturally adapts to chalk/chaos years without explicit conditioning.

---

## Files Modified

| File | Change |
|------|--------|
| `data/raw/historical/tournament_results_2023.json` | 6 game fixes |
| `data/raw/historical/tournament_results_2024.json` | 5 game fixes |
| `src/pipeline/config.py` | Added 2027 tournament start date |
| `tests/test_pipeline_stages.py` | Added 2027 to coverage assertion |
| `src/simulation/pool_history_opponent_model.py` | Fixed behavioral model bug + added `build_blended_pool_opponent_model()` |
| `scripts/mc_pool_backtest.py` | Added `pool_calibrated` opponent source, `_try_load_espn` helper, `--pool-blend-weight` CLI arg |
| `docs/gap-analysis-and-roadmap.md` | Updated all phases with results |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/analyze_pool_vs_espn.py` | Diagnostic: pool behavioral model vs ESPN comparison |
| `tests/test_pool_behavioral_blend.py` | 8 unit tests for blend function |
| `artifacts/pool_vs_espn_divergence.json` | Diagnostic output artifact |

## Files NOT Modified (reverted)

- Phase 2 regime-adaptive risk levels (reverted)
- Phase 3 champion diversity expansion (reverted)
- Phase 4 elo/odds/blend additions (reverted)

---

## Test Results

- **645 tests pass** (637 tracked + 8 new blend tests)
- **1 pre-existing failure** (`test_massey_source.py` — unrelated)
- **Ruff clean** across all files
- **Zero `build_actual_outcome` warnings** across 15 backtest years

---

## Decisions Made

1. **Phase 1 blend approach over replacement:** Blended pool behavioral + ESPN rather than full replacement. Still failed.
2. **Behavioral model bug fix kept:** Even though Phase 1 was rejected, the compounding bug fix is correct and stays.
3. **Infrastructure preserved:** `pool_calibrated` mode, blend function, diagnostic script all remain for future use.
4. **No ML classifier for champion:** 14 data points is too few; confirmed no signal exists anyway.
5. **All phases reverted to preserve baseline:** Production code unchanged from pre-session state for bracket construction.

---

## Open Items / Follow-ups

1. **Production baseline is 11.2% P(1st)** — this appears to be a local optimum for the current architecture.
2. **Next improvement vector:** Increase `n_pa_trials` beyond 200 to reduce selection noise (allows more candidates without degradation). This is a runtime-cost tradeoff, not a strategy change.
3. **The 2 unresolved abbreviations** (COFC=Charleston, SDSD/ISI) could be added to `ABBREV_TO_TEAM_ID` for completeness.
4. **2027 tournament format change:** NCAA expands to 76 teams. The pipeline assumes 64+4 (First Four). Will need structural changes for 2027 production.
5. **Massey data gap:** 2026 massey_avg not yet scraped (per exploration notes).
