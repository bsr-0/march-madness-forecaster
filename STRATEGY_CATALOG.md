# Strategy Catalog: Composable Bracket Strategies

Every strategy = **source(s)** + **adjustment(s)** + **construction mode**.

```
[weight*]source[+source...][+adjustment...][_construction]
```

## Architecture (implemented 2026-04-23)

Three composable layers:
1. **SOURCES** produce team ratings (barthag-equivalent, 0-1 scale)
2. **ADJUSTMENTS** modify ratings (contrarian ownership tilt, coach bonus, etc.)
3. **CONSTRUCTION MODES** build brackets from round advancement probabilities

Sources can be blended: `0.7*torvik+0.3*odds`
Adjustments can be chained: `odds+coach_adj+contrarian`
Everything is cross-producted with construction modes: `..._f4_first`

### Pipeline examples
| Spec | Meaning |
|------|---------|
| `torvik` | Torvik ratings, forward construction |
| `odds_f4_first` | Market ratings, F4-first construction |
| `odds+contrarian` | Market ratings with ownership tilt, forward |
| `0.7*torvik+0.3*odds` | 70/30 blend, forward |
| `0.5*pool_wisdom+0.5*torvik+contrarian_e8_first` | Pool+torvik blend, contrarian, E8-first |

### Running experiments
```bash
# Budgeted pipeline (recommended): T1 screen → kill → T2 rank → kill → T3 validate
# Uses the tier parameters + kill rules from § Testing Budget automatically.
python -m scripts.run_experiment --tier budget

# Individual tiers (advanced — usually you want --tier budget instead)
python -m scripts.run_experiment --tier 3      # top 5 at full rigor (N=100, 14 yrs)
python -m scripts.run_experiment --tier 1      # screen all bases (legacy, N=100)
python -m scripts.run_experiment --tier 2      # top 5 × all modes (legacy, N=100)

# Auto-generate all permutations (currently 120 strategies)
python -m scripts.run_experiment --permutations

# Specific pipeline combinations
python -m scripts.run_experiment --strategies "odds+contrarian_f4_first" "0.5*torvik+0.5*pool_wisdom_e8_first"

# Legacy full sweep
python -m scripts.run_experiment --tier all
```

### Implementation files
- Pipeline engine: `src/prediction/strategy_pipeline.py`
- Testing Budget helpers (tier config, kill rules, promotion): `src/evaluation/testing_budget.py`
- Experiment loop: `scripts/run_experiment.py` (`run_budget()` is the canonical entrypoint)
- Budget lock test: `tests/test_experiment_budget.py`
- Backtest harness: `scripts/mc_pool_backtest.py` (base×mode registry)

### Catalog maintenance

Keeping this file honest is a project-level invariant. When a PR changes any of the following, update the catalog in the **same commit** or the reader will be lied to:

| Change | Update |
|--------|--------|
| New source / adjustment / construction added | `## Implementation status` table, `## Build Order` phase row |
| Strategy implemented or deleted | Move the row in `## Implementation status`; if deleted, add to `## Deprecated Strategies` |
| Testing Budget parameters changed (`TIER_CONFIGS`) | `## Testing Budget` table + `tests/test_experiment_budget.py::test_tier_configs_match_catalog_contract` |
| Build Order phase completed | Flip `TODO` → `**DONE**` in `## Build Order` |
| Tier kill rules changed | `## Testing Budget` per-tier table, lock test, `run_budget()` cut-losses threshold |
| New tournament completed | `memory/tournament_oracle.md` (new ledger row + regime label), run `--oracle <year>` and paste the ranker-gap row |

The `test_tier_configs_match_catalog_contract` test is the drift guard — if the budget parameters in code diverge from the catalog's table, it fails CI.

---

## Sources (probability bases)

---

### Category A: Single-Source Ratings

#### A1: `seed`
- **Status:** IMPLEMENTED
- **Data:** Historical NCAA tournament seed win rates (1985-2025)
- **Algorithm:** Lookup table: seed → empirical barthag. 1-seed ≈ 0.96, 8-seed ≈ 0.50, 16-seed ≈ 0.04.
- **Years:** All (no external data needed)
- **File:** `src/prediction/seed_probabilities.py`

#### A2: `torvik`
- **Status:** IMPLEMENTED
- **Data:** `data/raw/historical/torvik_{year}.json` — Bart Torvik barthag ratings
- **Algorithm:** Direct load of pre-tournament barthag (expected win% vs average team). Fallback: `max(0.10, 1.0 - seed * 0.04)` for missing teams.
- **Years:** 2005-2026
- **File:** `scripts/mc_pool_backtest.py:_load_torvik_barthag()`

#### A3: `odds`
- **Status:** IMPLEMENTED
- **Data:** `data/processed/betting_odds/unified_odds_{year}.json` — ~5700 games/season with implied probabilities
- **Algorithm:** Bradley-Terry model fit on all regular-season games (pre-tournament cutoff). For each game with odds, the implied probability provides the observation. Iterative MLE:
  1. Initialize `r_i = 1.0` for all teams
  2. For each team i: `r_i = (wins_i) / sum(1/(r_i + r_j) for all opponents j)`
  3. Repeat 20 iterations (guaranteed convergence for connected graph)
  4. Convert to barthag: `barthag_i = r_i / (r_i + r_median)`
  - Uses market-implied wins (not actual wins): `wins_i = sum(implied_prob for games where i is listed)`
- **Fallback:** If <100 games with odds for tournament teams → use torvik
- **Years:** 2008-2026
- **File:** `src/prediction/market_probabilities.py` (NEW)

#### A4: `elo`
- **Status:** NEW (wrapper around existing computation)
- **Data:** `data/raw/historical/historical_games_{year}.json` — full season game results
- **Algorithm:** Use existing Elo from `proprietary_metrics.py`:
  1. Load `IncrementalMetricsEngine` for the year
  2. Call `compute_as_of(tournament_cutoff_date)`
  3. Extract `elo_rating` per team
  4. Normalize to barthag scale: `barthag_i = 1 / (1 + 10^((1500 - elo_i) / 400))`
  - Elo params: K=38, MOV-adjusted, cross-season carryover (0.75*prior + 0.25*1500)
- **Years:** 2005-2026
- **File:** `src/prediction/elo_probabilities.py` (NEW)

#### A5: `massey_avg`
- **Status:** NEW
- **Data:** `data/kaggle/MMasseyOrdinals.csv` — 36+ ranking systems, daily updates
- **Algorithm:**
  1. Load ordinal ranks for all systems on the last pre-tournament day
  2. For each tournament team, compute average rank across top-10 most stable systems (POM, SAG, MOR, DOL, COL, WOL, RTH, DUN, MAS, SEL — selected by historical availability)
  3. Convert rank to barthag: `barthag = 1.0 - (avg_rank / 360)` (360 ≈ D-I teams)
  4. Clip to [0.10, 0.99]
- **Fallback:** Teams not in Massey → seed-based estimate
- **Years:** 2003-2026
- **File:** `src/prediction/massey_probabilities.py` (NEW)

#### A6: `massey_best`
- **Status:** NEW
- **Data:** Same as A5
- **Algorithm:**
  1. Walk-forward: for each test year, evaluate each Massey system's historical tournament prediction accuracy (Brier score on prior years)
  2. Select the single best system
  3. Use that system's ranks → barthag conversion
  - Prevents cherry-picking: system selection uses only data from years < test_year
- **Years:** 2008-2026 (needs 5+ prior years for system selection)
- **File:** `src/prediction/massey_probabilities.py` (NEW, same file as A5)

#### A7: `spread_power`
- **Status:** IMPLEMENTED (data quality issues — Covers implied_prob is inverted for some games; data quality guard added)
- **Data:** `data/processed/betting_odds/unified_odds_{year}.json`
- **Algorithm:**
  1. Load all regular-season games with spreads for tournament teams
  2. For each team, compute average point spread (positive = favored)
  3. Convert to barthag: `barthag = 1 / (1 + 10^(-avg_spread / 10))` (logistic transform, ~10 pts per 0.85 barthag)
  4. Handle neutral vs home games: subtract 3.5 from home spread before averaging
- **Fallback:** Teams with <5 games with spreads → torvik
- **Years:** 2008-2026
- **File:** `src/prediction/market_probabilities.py` (NEW, same file as A3)

#### A8: `ap_strength`
- **Status:** NEW
- **Data:** `data/kaggle/ap_poll_data.json` — weekly AP rankings
- **Algorithm:**
  1. Load final pre-tournament AP poll
  2. Ranked teams: `barthag = 1.0 - (rank / 50)` (rank 1 = 0.98, rank 25 = 0.50)
  3. Unranked but receiving votes: barthag = 0.45
  4. Unranked, no votes: barthag from seed-based estimate
  - Simple but captures public perception, which matters for opponent modeling
- **Years:** 2008-2026
- **File:** `src/prediction/ap_probabilities.py` (NEW)

---

### Category B: Composite/Blended Ratings

#### B1: `noseed`
- **Status:** IMPLEMENTED
- **Data:** Torvik four-factors (12-dim differential vector)
- **Algorithm:** Logistic Regression (C=0.1) + GradientBoostingRegressor (200 trees), 0.5/0.5 blend. Symmetric augmentation. Walk-forward training on years < test_year.
- **Years:** 2008-2026
- **File:** `src/prediction/noseed_model.py`

#### B2: `blend`
- **Status:** IMPLEMENTED
- **Data:** Combination of seed (A1) + noseed (B1)
- **Algorithm:** `blend_rp = alpha * seed_rp + (1-alpha) * noseed_rp`, walk-forward alpha
- **Years:** 2008-2026
- **File:** `src/prediction/seed_probabilities.py`

#### B3: `market_torvik`
- **Status:** NEW
- **Data:** odds (A3) + torvik (A2)
- **Algorithm:** Simple average: `barthag = 0.5 * market_barthag + 0.5 * torvik_barthag`
  - If market_barthag unavailable for a team, use torvik alone
  - Rationale: market data captures information torvik doesn't (injuries, motivation, matchup-specific betting action)
- **Years:** 2008-2026
- **File:** `src/prediction/composite_probabilities.py` (NEW)

#### B4: `consensus`
- **Status:** NEW
- **Data:** odds (A3) + torvik (A2) + massey_avg (A5)
- **Algorithm:** Equal-weight average: `barthag = (market + torvik + massey) / 3`
  - Drop any missing component and re-weight
  - Rationale: three independent views of team strength reduce variance
- **Years:** 2008-2026
- **File:** `src/prediction/composite_probabilities.py` (NEW)

#### B6: `contrarian`
- **Status:** IMPLEMENTED
- **Data:** Any base's round_probs + public pick distribution (ESPN or pool)
- **Algorithm:** For each (team, round): `contrarian_rp = model_rp × (model_rp / public_pick)^strength`
  - Teams the public undervalues get inflated advancement probabilities
  - Teams the public overvalues get deflated
  - Re-normalized per round so probabilities sum correctly
  - `strength` parameter controls aggressiveness (1.0 = standard)
  - Currently uses torvik as the model base and opponent pick_dist as the public picks
- **Years:** 2011-2026 (wherever ESPN or pool picks are available)
- **File:** `src/prediction/contrarian_probabilities.py`

#### B7: `pool_wisdom`
- **Status:** IMPLEMENTED
- **Data:** `pool_hist_results.json` (2023-2026 actual pool picks) + ESPN national picks
- **Algorithm:**
  - **Years with pool data (2023-2026):** Uses pool's actual aggregate pick distribution directly as round_probs. Your 30 pool members' collective wisdom as a probability base.
  - **Years without pool data (2011-2022):** Extrapolates using ESPN national picks adjusted by the pool's "bias signature" — per-seed, per-round ratios of (pool_pick_pct / espn_pick_pct) averaged across the 4 pool years. E.g., if the pool historically over-picks 1-seeds as champion by 1.3× vs ESPN, that multiplier is applied.
  - This ensures multi-year coverage for the full backtest window.
- **Years:** 2011-2026 (direct for 2023-2026, extrapolated for earlier)
- **File:** `src/prediction/contrarian_probabilities.py`

#### B5: `stacked`
- **Status:** NEW
- **Data:** All Category A bases as features
- **Algorithm:**
  1. For each test year, collect barthag predictions from all Category A bases for prior tournament years
  2. Fit Ridge Regression (alpha=1.0) predicting actual game outcomes from the base predictions
  3. Apply fitted weights to test year's base predictions
  - Walk-forward: only uses years < test_year for weight fitting
  - Needs ≥3 prior years with all bases available
- **Years:** 2011-2026
- **File:** `src/prediction/composite_probabilities.py` (NEW)

---

### Category C: Feature-Enriched Ratings

#### C1: `coach_adj`
- **Status:** NEW
- **Data:** `data/kaggle/MTeamCoaches.csv` + tournament results history
- **Algorithm:**
  1. Start with torvik barthag
  2. For each team's coach, compute tournament experience: count of prior NCAA tournament appearances as head coach
  3. Apply multiplicative adjustment: `barthag *= (1 + 0.01 * min(log(1 + appearances), 3))`
  - Caps at ~3% bonus for veteran coaches (10+ appearances)
  - First-time coaches get no adjustment (0 appearances → 0 bonus)
  - Walk-forward: only count appearances in years < test_year
- **Years:** 2008-2026
- **File:** `src/prediction/enriched_probabilities.py` (NEW)

#### C2: `roster_adj`
- **Status:** NEW
- **Data:** `data/raw/historical/cbbpy_rosters_{year}.json`
- **Algorithm:**
  1. Start with torvik barthag
  2. For each team, compute top-5 player aggregate: `team_talent = mean(top-5 WARP)`
  3. Compute league-average top-5 WARP across all tournament teams
  4. Apply adjustment: `barthag *= (1 + 0.02 * (team_talent - league_avg) / league_std)`
  - Caps at ±4% adjustment
  - Captures star-player effect that team-level efficiency metrics miss
- **Fallback:** Missing roster → torvik unchanged
- **Years:** 2008-2026
- **File:** `src/prediction/enriched_probabilities.py` (NEW)

#### C3: `momentum`
- **Status:** NEW
- **Data:** `data/raw/historical/torvik_four_factors_{year}_*.json` (monthly snapshots)
- **Algorithm:**
  1. Start with torvik barthag (final pre-tournament)
  2. Load January and March four-factor snapshots
  3. Compute trajectory: `delta_eff = (march_adjEM - january_adjEM) / january_adjEM`
  4. Apply adjustment: `barthag *= (1 + 0.03 * tanh(delta_eff * 5))`
  - Teams improving → slight boost (max +3%)
  - Teams declining → slight penalty (max -3%)
  - tanh prevents outliers from dominating
- **Fallback:** Missing snapshots → torvik unchanged
- **Years:** 2008-2026
- **File:** `src/prediction/enriched_probabilities.py` (NEW)

---

### Category D: Upset-Aware Ratings

#### D1: `volatile`
- **Status:** NEW
- **Data:** `data/raw/historical/historical_games_{year}.json`
- **Algorithm:**
  1. Start with torvik barthag
  2. For each team, compute season game-by-game win probability variance
  3. Instead of using barthag directly in Log5, use `barthag ± noise` where noise ~ N(0, team_volatility)
  4. In the MC simulation, each game samples from this distribution, so volatile teams sometimes look much stronger and sometimes much weaker
  - Effect: high-variance teams get more upset wins AND more upset losses, matching real tournament behavior
  - Low-variance teams (consistent) maintain their expected advancement rates
- **Years:** 2008-2026
- **File:** `src/prediction/upset_probabilities.py` (NEW)

#### D2: `upset_tuned`
- **Status:** NEW
- **Data:** Historical tournament game results + `data/kaggle/upset_seed_info.json`
- **Algorithm:**
  1. Compute actual upset rates by seed matchup from all historical data (1985-present)
  2. Compare to Log5-predicted upset rates using torvik barthag
  3. Apply per-matchup calibration: if Log5 says 1v16 upset is 1% but history says 1.5%, scale pairwise probability by 1.5
  4. This calibration is applied INSIDE the MC simulation at the pairwise level
  - Walk-forward: only use upset data from years < test_year
  - Addresses known Log5 bias: tends to underestimate extreme upsets and overestimate mild ones
- **Years:** 2008-2026
- **File:** `src/prediction/upset_probabilities.py` (NEW)

---

## Construction Modes (6 total)

### M1: `forward`
- **Status:** EXISTS (as `sample_model_brackets`)
- **Algorithm:** For each of 63 games in bracket order (R64 → CHAMP):
  1. Look up both teams' round advancement probabilities
  2. Compute pairwise: `P(t1 wins) = rp[t1][round] / (rp[t1][round] + rp[t2][round])`
  3. Draw winner stochastically from this probability
  - No anchoring. Each game is independent given round_probs.
- **File:** `scripts/mc_pool_backtest.py:sample_model_brackets()`

### M2: `champ_first`
- **Status:** EXISTS
- **Algorithm:**
  1. Draw champion from CHAMP probability distribution (categorical weighted sample)
  2. Lock that champion to win every game on their path (R64 → CHAMP)
  3. Sample all remaining 62 games stochastically from round_probs
- **File:** `scripts/mc_pool_backtest.py:sample_champ_first_brackets()`

### M3: `f4_first`
- **Status:** EXISTS
- **Algorithm:**
  1. For each of 4 regions, draw an F4 team from regional F4 probability distribution
  2. Lock those 4 teams to win their regional paths (R64 → E8)
  3. Sample F4 semifinals and championship from round_probs
- **File:** `scripts/mc_pool_backtest.py:sample_f4_first_brackets()`

### M4: `e8_first`
- **Status:** EXISTS
- **Algorithm:**
  1. For each of 8 quadrants (4 regions × top/bottom), draw an S16 winner
  2. Lock those 8 teams to win their quadrant paths (R64 → S16)
  3. Sample E8, F4, CHAMP from round_probs
- **File:** `scripts/mc_pool_backtest.py:sample_e8_first_brackets()`

### M5: `backward`
- **Status:** NEW
- **Algorithm:**
  1. Draw champion from CHAMP probability distribution
  2. Draw F4 from `P(team makes F4 | this champion won)` — conditioned probabilities
  3. Draw E8 from `P(team makes E8 | this F4)` — conditioned
  4. Continue backward: S16, R32, R64
  5. Each round's picks are conditioned on the downstream results already decided
  - **Key difference from champ_first:** champ_first locks the champion then fills R64 FORWARD independently. Backward makes every pick conditioned on the final result. If you pick a 7-seed champion, your R64 picks near that 7-seed reflect that the 7-seed is good enough to win it all.
  - **Computing conditioned probabilities:** `P(team makes E8 | F4 team X) = P(team makes E8 AND team X makes F4) / P(team X makes F4)`. These are computed from the same MC simulation that produces round_probs, by tracking joint advancement counts.
- **File:** `scripts/mc_pool_backtest.py:sample_backward_brackets()` (NEW)

### M6: `confidence`
- **Status:** NEW
- **Algorithm:** Per-game decision based on prediction confidence:
  1. For each game, compute `confidence = |P(fav) - 0.5|`
  2. **High confidence** (P > 0.85): always pick favorite (lock chalk). Wastes no randomness on 1v16 games.
  3. **Medium confidence** (0.60 < P < 0.85): sample from model probability (standard stochastic)
  4. **Low confidence** (P < 0.60): sample with BOOSTED variance — inflate upset probability by 1.5× to explore differentiation opportunities
  - Effect: concentrates bracket diversity on the games that actually matter for pool differentiation (5v12, 6v11, 7v10) while locking in the games everyone agrees on
  - No anchoring — works game-by-game, not by locking teams through paths
- **File:** `scripts/mc_pool_backtest.py:sample_confidence_brackets()` (NEW)

---

## Composable Pipeline (implemented 2026-04-23)

With the pipeline engine, strategies are no longer a flat cross-product. Any source can be blended with any other source, then chained with any adjustment(s), then built with any construction mode.

### Current auto-generated permutations: 120 strategies
From 5 implemented sources × 1 implemented adjustment × 4 implemented construction modes, with pairwise blends:
- 5 single sources × 4 modes = 20
- 5 single sources × 1 adj × 4 modes = 20
- 10 source pairs × 4 modes = 40
- 10 source pairs × 1 adj × 4 modes = 40

As more sources (elo, massey, AP, coach, roster, momentum) and adjustments (volatile, upset_tuned) are implemented, the permutation count grows combinatorially.

### Implementation status

| Component | Implemented | Not Yet |
|-----------|:-----------:|:-------:|
| **Sources** | seed, torvik, odds, spread_power, pool_wisdom (5) | elo, massey_avg, massey_best, ap_strength (4) |
| **Adjustments** | contrarian (1) | coach_adj, roster_adj, momentum, volatile, upset_tuned (5) |
| **Constructions** | forward, champ_first, f4_first, e8_first (4) | backward, confidence (2) |
| **Blending** | Equal-weight and custom-weight blends of any 2+ sources | Stacked meta-learner (B5) |
| **Testing Budget** | `run_budget()` enforces T1/T2/T3 parameters + kill rules, cut-losses gate at T2 | Round-probs caching, multi-proc parallelism, convergence-based repeat stopping |
| **Tournament Oracle** | `--oracle <year>` reports F4/finals/champ hits + ranker_gap_espn_pts; ledger in `memory/tournament_oracle.md` | Auto-run inside `run_budget()` after T3 (currently a separate CLI call) |
| **Chaos Index** | `--chaos-index` computes pre-tournament regime prediction from Torvik features (measured r=−0.668 for `mean_top8_barthag`, walk-forward MAE 0.89 vs baseline 1.13) | Strategy-selection gating (currently informational only) |

_Last verified: 2026-04-24 — see § Catalog maintenance. Three-primary metrics + oracle ledger + chaos index added 2026-04-24._

---

## Strategy Cross-Product (58 strategies)

### Tier 1: Screen all bases (18 strategies)
Every base × M1 (forward) — identifies which probability sources produce the best round advancement estimates.

| # | Strategy Name | Base | Mode |
|---|--------------|------|------|
| 1 | `seed_forward` | A1 seed | M1 forward |
| 2 | `torvik_forward` | A2 torvik | M1 forward |
| 3 | `odds_forward` | A3 odds | M1 forward |
| 4 | `elo_forward` | A4 elo | M1 forward |
| 5 | `massey_avg_forward` | A5 massey_avg | M1 forward |
| 6 | `massey_best_forward` | A6 massey_best | M1 forward |
| 7 | `spread_power_forward` | A7 spread_power | M1 forward |
| 8 | `ap_strength_forward` | A8 ap_strength | M1 forward |
| 9 | `noseed_forward` | B1 noseed | M1 forward |
| 10 | `blend_forward` | B2 blend | M1 forward |
| 11 | `market_torvik_forward` | B3 market_torvik | M1 forward |
| 12 | `consensus_forward` | B4 consensus | M1 forward |
| 13 | `stacked_forward` | B5 stacked | M1 forward |
| 14 | `coach_adj_forward` | C1 coach_adj | M1 forward |
| 15 | `roster_adj_forward` | C2 roster_adj | M1 forward |
| 16 | `momentum_forward` | C3 momentum | M1 forward |
| 17 | `volatile_forward` | D1 volatile | M1 forward |
| 18 | `upset_tuned_forward` | D2 upset_tuned | M1 forward |

### Tier 2: Top 5 bases × all modes (30 strategies)
After Tier 1, take 5 highest-P(1st) bases and cross with all 6 construction modes.

Example if top 5 = {odds, consensus, momentum, volatile, massey_avg}:

| # | Strategy Name | Base | Mode |
|---|--------------|------|------|
| 19 | `odds_forward` | A3 | M1 | (already in Tier 1)
| 20 | `odds_champ_first` | A3 | M2 |
| 21 | `odds_f4_first` | A3 | M3 |
| 22 | `odds_e8_first` | A3 | M4 |
| 23 | `odds_backward` | A3 | M5 |
| 24 | `odds_confidence` | A3 | M6 |
| 25-30 | `consensus_*` | B4 | M1-M6 |
| 31-36 | `momentum_*` | C3 | M1-M6 |
| 37-42 | `volatile_*` | D1 | M1-M6 |
| 43-48 | `massey_avg_*` | A5 | M1-M6 |

### Tier 3: Parameter sweeps + training window (~15 strategies)
Top 3 (base, mode) pairs from Tier 2 with parameter and training window variations:

**Parameter variants:**

| # | Strategy | Variation |
|---|----------|-----------|
| 49 | `consensus_backward_blend60` | consensus with 60/20/20 weights instead of equal |
| 50 | `consensus_backward_blend80` | consensus with 80% market weight |
| 51 | `volatile_confidence_low` | volatility noise × 0.5 |
| 52 | `volatile_confidence_high` | volatility noise × 2.0 |
| 53 | `odds_confidence_tight` | confidence thresholds: 0.90 / 0.70 |
| 54 | `odds_confidence_wide` | confidence thresholds: 0.80 / 0.55 |
| 55 | `momentum_f4_first_strong` | momentum adjustment ±5% instead of ±3% |
| 56 | `stacked_backward` | meta-learned weights + backward construction |
| 57 | `market_torvik_e8_first_70` | 70% market / 30% torvik blend |
| 58 | `upset_tuned_confidence` | calibrated upset rates + confidence routing |

**Training window variants (for ML-based bases: noseed, stacked, massey_best):**

The walk-forward training window currently uses ALL prior years. But more years ≠ better — the game evolves (3-point revolution, tempo changes, transfer portal).

| # | Strategy | Window |
|---|----------|--------|
| 59 | `noseed_forward_5yr` | Train on most recent 5 years only |
| 60 | `noseed_forward_8yr` | Train on most recent 8 years only |
| 61 | `noseed_forward_all` | Train on all prior years (current default) |
| 62 | `stacked_forward_5yr` | Meta-learner trained on 5-year window |
| 63 | `stacked_forward_8yr` | Meta-learner trained on 8-year window |
| 64 | `massey_best_forward_5yr` | System selection from 5-year lookback |

These test whether recency matters — does a model trained on 2019-2024 outperform one trained on 2008-2024?
Implementation: add `--train-window` CLI arg to cap `walk_forward_train_years()` to most recent N years.

---

## Evaluation Contract

### Multi-Year Requirement
Every strategy MUST produce results for ≥12 of 14 backtest years (2011-2026, excl 2020).
Strategies that can't cover ≥12 years are excluded from significance testing.

### Backtest Parameters (non-negotiable)
```
--team-identity          (real ESPN scoring)
--n-opponents 30         (actual pool size)
--n-repeats 100          (convergence-tested minimum)
--opponent pool           (actual pool brackets when available, ESPN fallback)
```

### Metrics Reported (per strategy, per year)

**Primary — the pool pays these directly:**
- **P(1st)** — fraction of pool runs your submitted bracket finishes first. Only metric that pays in a winner-take-all pool.
- **BestScore** — ESPN points of the best bracket in your portfolio for the year. This is the raw score that beats (or loses to) the field. P(1st) is derived from `BestScore > max(field_score)`, so calibration failures show up here first.
- **MeanScore** — ESPN points averaged across the portfolio. Diagnostic for "did this strategy produce a high-scoring fleet, or did P(1st) come from one lucky outlier?"

**Secondary — diagnostic:**
- MeanRank, BestRank — placement diagnostics; don't pay out but surface portfolio consistency.
- **Tournament oracle** — `python -m scripts.run_experiment --oracle <year>` scores the saved portfolio against actual tournament outcomes (F4 hits / finals hits / champion / ranker gap in ESPN points). Per-year ledger with regime labels in `memory/tournament_oracle.md`. The `ranker_gap_espn_pts` field — points left on the table because the ranker didn't promote the best bracket — is the direct KPI for the selection/ranking problem (North Star lever #2).

All three primaries go in `_print_summary` and `_save_budget_summary` outputs; ESPN scoring is the team-identity score under `--team-identity` (locked per MEMORY.md §1 O26/O27).

### Significance Gate
```
Test:   Paired permutation test (10,000 draws) on P(1st) per year
H0:     P(1st)_new = P(1st)_seed_forward
H1:     P(1st)_new > P(1st)_seed_forward
Gate:   p < 0.10 (one-tailed)
Also:   P(1st)_new > P(1st)_seed_forward in ≥ 8/14 years
```
P(1st) is the gate because it's the only metric that converts to money. BestScore is a companion diagnostic — a strategy that improves BestScore but not P(1st) is producing high-ceiling brackets that lose to the field anyway (usually because of opponent-model mis-specification, not scoring shape).

### Dead-End Criteria
A strategy is killed if:
- P(1st) = 0.000 in any year (catastrophic failure)
- P(1st) < seed_forward in ≥ 10/14 years (consistently worse)
- Bonferroni-corrected p > 0.50 (no directional signal at all)

---

## Testing Budget: Comprehensive but Time-Bounded

The tier structure above scales rigor with signal strength. Full-rigor backtests (N=100 repeats × 14 years × ~120 strategies) take days — most of it wasted on strategies that were dead on arrival. Rule: **spend cheap compute to prune, expensive compute only on survivors.** The goal is two overnight runs end-to-end, not a week of babysitting.

### Per-tier budget
| Tier | Strategies | Repeats | Years | Wall-time target | Kill threshold |
|------|-----------:|--------:|-------|------------------|----------------|
| T1 screen   | all (~120)     |  25 | recent 8 (excl 2020)  | ≤1 hr   | P(1st) < `seed_forward` in ≥6/8 years **or** P(1st)=0 in any year |
| T2 rank     | top 10 from T1 |  50 | full 14 (2011-2026 excl 2020) | ≤3 hr   | P(1st) < `seed_forward` in ≥10/14 years |
| T3 validate | top 5 from T2  | 100 | full 14               | ≤6 hr   | significance gate (p<0.10, ≥8/14 years) |
| T3b sweep   | top 3 ± params | 100 | full 14               | overnight | same as T3 |

T1+T2 fit one overnight. T3+T3b fit the next. Everything else is optional.

### Cost-reduction levers (priority order)
1. **Cache `round_probs` per (source, year).** Every strategy sharing a base hits the same probability grid — compute once, reuse across modes and adjustments. This alone cuts the pipeline cross-product cost by ~4-6×.
2. **Early-stop within a strategy.** If after 3 years P(1st) ≤ 1% (below the 3.23% random baseline), abort the remaining years for that strategy.
3. **Parallelize across strategies, not years.** Strategies are embarrassingly parallel; years share no state. Use `multiprocessing.Pool(n_cores)` over the strategy list. Avoid nesting parallelism — it thrashes.
4. **Convergence monitoring, not fixed repeats.** Track running P(1st) CI width; stop when 95% CI < 0.5 pp. Losers stabilize fast (n≈20); survivors need the full 100.
5. **Subsample years in T1 only.** T1 ranks; it does not certify. 8 recent years is enough to separate signal from noise for screening — never cite T1 numbers as the final result.

### Anti-patterns (burn time, yield mediocre results)
- Running T3 params (N=100, 14 years) on every strategy "just to have the data". Don't. Use T1 first.
- Adding a strategy mid-run. Batch it into the next tier run; never lengthen an in-flight job.
- Re-running identical configs hoping the average smooths out. If results look off, fix the **input** — more repeats do not buy you significance the design can't deliver.
- Tuning a losing base. If the base failed T1, no parameter sweep saves it. Move on.
- Re-simulating opponents when only the user bracket changed. Lock the opponent field at a fixed RNG seed across a tier so variance between strategies is purely user-side.

### When to cut losses
If after T2 the best non-baseline strategy improves P(1st) by <0.3 pp over `seed_forward`, stop. The effect size will not clear the significance gate even with more compute — redirect effort to a new source/adjustment, not more repeats. This is the single most time-saving rule in the catalog.

### What "comprehensive" means here
Comprehensive ≠ every permutation at full rigor. It means: **every source/adjustment/mode gets a fair shot at T1 with identical conditions**, and **every survivor gets validated at full rigor**. Strategies killed at T1 are documented in `memory/project_strategies_tested.md` with their T1 numbers — that's the audit trail, not a re-run.

---

## Build Order

| Phase | What | Status | Strategies Enabled |
|-------|------|:------:|-------------------|
| 1a | Base×mode registry + pipeline engine | **DONE** | Infrastructure for all |
| 1b | Market bases (A3 odds, A7 spread_power) | **DONE** | odds, spread_power, blends with torvik |
| 1b2 | Contrarian adjustment (B6) + pool_wisdom (B7) | **DONE** | contrarian chains, pool_wisdom, blends |
| 1b3 | Experiment loop + permutation generator | **DONE** | 120 auto-generated strategies testable |
| 1b4 | Testing Budget (tier configs, kill rules, cut-losses gate) | **DONE** | `--tier budget` runs T1→T2→T3 with automatic pruning |
| 1b5 | Tournament Oracle Ledger (per-year F4/finals/champ + ranker gap KPI) | **DONE** | `--oracle <year>` + `memory/tournament_oracle.md` — 2023/2026 chaos gap +820/+830, 2024/2025 chalk gap +0/+280 |
| 1b6 | Chaos Index (pre-tournament regime prediction from Torvik top-of-field) | **DONE** | `--chaos-index` — `mean_top8_barthag` r=−0.668 p=0.006; walk-forward MAE 0.89 beats 1.13 baseline; informational only (no strategy gating yet) |
| 1c | Elo base (A4) | TODO | elo, elo+contrarian, blends |
| 1d | Massey bases (A5, A6) | TODO | massey_avg, massey_best, blends |
| 1e | AP base (A8) | TODO | ap_strength |
| 1f | Composite (B3 market_torvik, B4 consensus, B5 stacked) | TODO | handled by pipeline blending for B3/B4; B5 needs Ridge |
| 1g | Enriched bases (C1 coach, C2 roster, C3 momentum) | TODO | adjustment chains |
| 1h | Upset bases (D1 volatile, D2 upset_tuned) | TODO | upset-aware MC simulation |
| 2a | Backward construction (M5) | TODO | *_backward |
| 2b | Confidence construction (M6) | TODO | *_confidence |
| 3 | Full permutation evaluation | TODO | Run per §Testing Budget — T1 screen (≤1hr) → T2 rank (≤3hr) |
| 4 | Significance testing + dead-end pruning | TODO | T3 validate — Gate: p<0.10, ≥8/14 years; cut losses if <0.3 pp over seed_forward |
| 5 | Parameter sweeps on top strategies | TODO | T3b — Training window, blend weights, only for T3 survivors |

---

## Deprecated Strategies (removed, do not re-implement)

| Strategy | Why Removed | Closure |
|----------|------------|---------|
| Deterministic argmax (det_champ/f4/e8) | P(1st)=0.00% — no diversity | D12 |
| Pareto-leverage optimizer (opt_seed/blend/torvik) | Myopic greedy, catastrophic in upset years | D6 |
| Hedge mode | Significantly worse than seed (Bonferroni p<0.05) | D7 |
| Chalk-fade champion | P(1st) collapsed 14× vs baseline | D16 |
| GNN/Transformer prediction models | No BSS improvement over LR | Removed |
| Pool marginal blend | Null result — doesn't change rankings | O21 |
| 50-bracket random sampling | Convergence test proved it measures noise (2026-04-23) | To be replaced |
