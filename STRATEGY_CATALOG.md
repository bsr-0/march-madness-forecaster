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

> **⚠️ NO-RUN POLICY — Read `memory/run_policy.md` before invoking any command in this block.**
> Strategy-addition phases are no-run by default. A `--tier budget` invocation requires **explicit human approval** (phrases like "run the budget", "kick off the run", "run tier 1"). Adding new strategies, adjustments, or construction modes does **not** authorize a run. If in doubt, ask.

```bash
# Budgeted pipeline (recommended): T1 screen → kill → T2 rank → kill → T3 validate
# Uses the tier parameters + kill rules from § Testing Budget automatically.
# GATED: requires explicit human approval per memory/run_policy.md.
python -m scripts.run_experiment --tier budget

# Individual tiers (advanced — usually you want --tier budget instead)
python -m scripts.run_experiment --tier 3      # top 5 at full rigor (N=100, 14 yrs)
python -m scripts.run_experiment --tier 1      # screen all bases (legacy, N=100)
python -m scripts.run_experiment --tier 2      # top 5 × all modes (legacy, N=100)

# Auto-generate all permutations (currently 7,920 strategies — 8 sources × (1 + 6 adj + pair chains) × 10 constructions × blends)
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
- **Status:** IMPLEMENTED (2026-04-24) — self-contained Elo rather than the production `IncrementalMetricsEngine` (simpler, no hparam-tuning provenance needed for a strategy source). Uses the cbbpy team-ID bridge (phase 1b7) to map game records to canonical tournament IDs.
- **Data:** `data/raw/historical/historical_games_{year}.json`, filtered to games strictly before Torvik's `tournament_start` (walk-forward safe).
- **Algorithm:**
  1. All teams start at Elo 1500.
  2. Process regular-season games chronologically. After each game:
     - `expected = 1 / (1 + 10^((opp_elo − self_elo) / 400))`
     - `adj = K × (actual − expected)`  (K=38 per catalog)
     - Symmetric: winner gains `adj`, loser loses `adj`
  3. Bridge each cbbpy team ID to its canonical tournament ID via `bridge_cbbpy_id()`. Teams with fewer than 5 games get the seed-based fallback `max(0.10, 1.0 − seed × 0.04)`.
  4. Convert final Elo to barthag: `barthag_i = 1 / (1 + 10^((1500 − elo_i) / 400))` — the implied win probability vs a 1500-rated opponent. Same shape as torvik/odds/spread_power barthag, so it slots into `build_torvik_round_probabilities` with zero pipeline work.
  - **Deviations from catalog spec:** no MOV multiplier (keeps implementation simple; raw Elo already captures the signal); no cross-season carryover (each season is independent — college basketball's annual roster turnover makes prior-year regression noisy).
- **Years:** 2011-2026 (wherever `historical_games_{year}.json` + Torvik `tournament_start` are both ingested)
- **File:** `src/prediction/elo_probabilities.py`; lock test: `tests/test_elo_source.py`

#### A5: `massey_avg`
- **Status:** IMPLEMENTED (2026-04-24) — uses the pre-aggregated Massey composite file rather than per-system top-10 averaging (same signal, cleaner source; see "Deviations" below).
- **Data:** `data/raw/external_massey_composite_{year}.json` (list of `{team_id, rating, ranking, normalized}` per team). Available 2008-2025; 2026 not yet scraped — loader returns None for that year and the backtest skips this source for 2026.
- **Algorithm:**
  1. Load composite JSON (already ensembled across ~150 ranking systems).
  2. For each canonical tournament team, bridge the Massey team_id via exact-match-then-alias (6 explicit edge cases for heavily abbreviated Massey names: `american_univ → american`, `mt_st_mary_s → mount_st__mary_s`, `ne_omaha → omaha`, `st_francis_pa → saint_francis`, `st_mary_s_ca → saint_mary_s__ca`, `siue → siu_edwardsville`).
  3. Use the composite's `normalized` field (already ∈ (0, 1); top teams ~0.99, bottom ~0.01) directly as barthag.
  4. Clip to [0.10, 0.99] per catalog.
- **Fallback:** Teams not in Massey → seed-based estimate `max(0.10, 1 − 0.04 × seed)` (same pattern as Elo/torvik fallbacks).
- **Deviations from original catalog spec:**
  - Uses pre-aggregated `data/raw/external_massey_composite_*.json` instead of `data/kaggle/MMasseyOrdinals.csv` (the latter doesn't exist in this repo; the former is the already-ensembled equivalent).
  - No per-system top-10 averaging — the composite is already the ensemble.
- **Years:** 2008-2025 (2026 pending composite scrape).
- **Coverage:** 68/68 tournament teams bridged for 2025 (exact-match 62 + 6 aliases).
- **File:** `src/prediction/massey_probabilities.py`; lock test: `tests/test_massey_source.py`.

#### A6: `massey_best`
- **Status:** IMPLEMENTED (2026-04-25) — shipped the per-system walk-forward Brier-selection harness. Distinct from A5 `massey_avg`: A5 uses the pre-aggregated composite ensemble; A6 picks one best-historical-Brier system per test year. Only source in the catalog where the data selects the ranker.
- **Data:** `data/raw/historical/external_{SYSTEM}_{year}.json` — 56+ per-system rankings with 22 years of coverage (2005-2026). Each file is a list of `{team_id, rating, ranking, normalized}`. `normalized ∈ [0, 1]` used directly as barthag-equivalent. Team IDs are canonical — no bridge needed. Plus `tournament_results_{year}.json` for Brier scoring (same file `upset_tuned` uses).
- **Algorithm:**
  1. For each candidate system S and each prior year y < test_year (excluding 2020 COVID):
     - Load S's ratings for year y and the year's tournament games.
     - For each game (t1 vs t2) in R64-CHAMP rounds: compute P(t1 wins) via Log5 on `normalized(t1)` / `normalized(t2)`. Brier = (P − y_actual)². Skip games where either team isn't in S's ratings.
  2. Cumulate `(brier_sum, n_games)` per system across all prior years.
  3. Eligibility filter: system must have (a) ≥500 cumulative scored games (≈ 8+ tournaments) and (b) a ratings file for test_year itself.
  4. Select `argmin(brier_sum / n_games)` among eligible systems.
  5. Load the selected system's test-year `normalized` field → treat as barthag → run the same MC round-probs builder as torvik/massey_avg.
- **Walk-forward contract:** at test_year Y, only tournament outcomes from years strictly < Y contribute to Brier scoring. 2020 excluded. Each test year's selection computed in isolation.
- **Selection sanity:**
  - Test_year 2026 → **TRK** (548 games, mean Brier 0.172).
  - Test_year 2025 → **STY** (earlier years' competition resolves differently).
  - Test_year 2021 → **ARG** (first post-COVID year).
  - Top-5 for 2026 cluster tightly at 0.172-0.181 mean Brier — small margins among well-established rankers. 47 systems pass the min_games=500 threshold.
- **Min-games threshold rationale:** at `min_games=100`, 2026 selection is "DP" with only 113 games (≈2 prior tournaments) — overfit risk. At `min_games=500` (≈8 tournaments), selection is a long-standing data-defensible pick. The threshold is a runtime arg (`select_best_system(..., min_games=500)` by default; operator can tighten/loosen).
- **Relation to MEMORY.md §2 D1 (BSS=0 ceiling):** A6 does **not** claim to beat BSS=0. Individual systems all sit at BSS≈0. A6 picks the best-historical-Brier one among them — the orthogonal question is whether seed-level calibration differs between systems enough to shift P(1st). Different framing from D1's accuracy-lift kill; selection is not re-litigation.
- **Fallback:** Missing teams in the selected system → seed-based barthag fallback (`max(0.10, 1 − 0.04 × seed)`, same as torvik). No qualifying system for a test year → loader returns None → the source is unavailable for that year and downstream strategies skip (same pattern as odds / elo missing-data years).
- **Years:** 2011-2026 (needs ≥ 8 prior tournaments' coverage to clear min_games threshold).
- **File:** `src/prediction/massey_best_probabilities.py`; lock test: `tests/test_massey_best_source.py` (14 tests).

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
- **Status:** IMPLEMENTED (2026-04-25) — shipped as an ADJUSTMENT (not a base) for composability with every source × construction. Builds a per-coach prior-tournament-appearance count via Kaggle `MTeamCoaches.csv` × `MNCAATourneySeeds.csv`, joining to the canonical tournament ID via a `MTeamSpellings.csv`-driven bridge.
- **Data:**
  - `data/kaggle/MTeamCoaches.csv` — per-(season, kaggle_team_id) head coach. Mid-season swaps resolved by max tenure (`LastDayNum − FirstDayNum`).
  - `data/kaggle/MNCAATourneySeeds.csv` — set of (season, kaggle_team_id) pairs that made the tournament. Used to count appearances.
  - `data/kaggle/MTeamSpellings.csv` — canonical → Kaggle TeamID bridge via a name-normalization cascade (`__` → ` `, `_s__` → `'s `, `_a_m` → ` a&m`, `saint_` → `st `) plus a 2-entry manual alias table (`maryland_baltimore_county` → `umbc`, `st__john_s__ny` → `st john's`). Coverage: **68/68 on the 2026 tournament field**.
- **Algorithm:**
  1. For each test year Y, build cumulative coach appearances: for every Season < Y where (season, team) is in `MNCAATourneySeeds`, increment that season's primary head coach's count.
  2. For each team in Y, look up Y's primary head coach via MTeamCoaches and attach the cumulative count.
  3. `bump = min(log(1 + apps), 3.0)`; `factor = 1.0 + 0.01 × bump` ∈ [1.00, 1.03].
  4. `adjusted[r] = base[r] × factor`. Re-normalize per round to bracket team-counts {64, 32, 16, 8, 4, 2}.
  - **Asymmetric (one-sided per spec):** zero-experience coaches get factor=1.0 (no boost). The cap binds at ~19 prior appearances (`log(20) ≈ 3`); even a 50-app legend stays at +3%.
  - **Deliberate deviation from catalog spec:** the spec called for adjustment at the barthag level. Applied at the round_probs level (same pattern as `volatile` / `upset_tuned` / `roster_adj`) for composability — every source × construction gets a `_+coach_adj_*` variant for free.
- **Walk-forward contract:** appearances for year Y are cumulated only over Seasons strictly < Y. The current-year coach lookup is taken from Y itself, but no statistics from Y enter the count.
- **Fallback:** Team unbridged → 0 apps. Team's Y-coach not in MTeamCoaches → 0 apps. Both produce factor=1.0 (informationally neutral).
- **2026 sanity (top-10 by prior apps):** Tennessee 29 (Barnes), Michigan State 27 (Izzo), Kansas 26 (Self), Gonzaga 25 (Few), Arkansas 24 (Calipari), Houston 21 (Sampson), Purdue 17 (Painter), TCU 15 (Dixon), UCLA 15 (Cronin), Texas 13.
- **Years:** 2011-2026 (Kaggle data covers 1985-2025).
- **File:** `src/prediction/coach_adj_probabilities.py`; lock test: `tests/test_coach_adj_adjustment.py` (13 tests).

#### C2: `roster_adj`
- **Status:** IMPLEMENTED (2026-04-25) — shipped as an ADJUSTMENT (not a base) for composability with every source × construction. Uses the cbbpy team-ID bridge (phase 1b7) to map per-player roster records to canonical tournament IDs.
- **Data:** `data/raw/historical/cbbpy_rosters_{year}.json` — every player has a `warp` field directly from cbbpy (no proxy needed). Coverage: 65+/68 of 2026 tournament field bridges cleanly; the rest fall back to neutral z=0.
- **Algorithm:**
  1. For each tournament team, take the top-5 players by `warp` (filtered to those with ≥5 games played to drop garbage-time-only entries).
  2. Compute team talent = mean of those top-5 WARPs.
  3. Compute league mean and stddev across the per-team top-5 values (only over teams with rosters; missing teams don't drag the mean).
  4. For each team t and round r: `z_t = (talent_t − league_mean) / league_std`, `factor = 1 + clip(0.02 × z_t, ±0.04)`, `adjusted[r] = base[r] × factor`. Re-normalize per round to bracket team-counts {64, 32, 16, 8, 4, 2}.
  - **Effect:** strong-roster teams get +1 to +4% advancement boost; weak-roster teams get matching penalty. Cap survives even pathological z=±100 outliers (factor stays in [0.96, 1.04]).
  - **Deliberate deviation from catalog spec:** the spec called for adjustment at the barthag level. We apply at the round_probs level (same pattern as `volatile` / `upset_tuned`) for composability — every source × construction gets a `_+roster_adj_*` variant for free.
- **Fallback:** Missing team → neutral z=0 (no adjustment, then renorm).
- **Years:** 2003-2026 (cbbpy_rosters files exist for the full historical window).
- **File:** `src/prediction/roster_adj_probabilities.py`; lock test: `tests/test_roster_adj_adjustment.py` (11 tests).

#### C3: `momentum`
- **Status:** IMPLEMENTED (2026-04-25) — shipped as an ADJUSTMENT (not a base) for composability with every source × construction. 108 snapshot files cover 2008-2026. Team IDs in the four-factor snapshots match canonical tournament IDs directly — no bridge needed.
- **Data:** `data/raw/historical/torvik_four_factors_{year}_{yyyymmdd}.json` — ~5 snapshots per year (Nov/Dec/Jan/Feb/Mar). Uses the January (~YYYY-01-31) and March (~YYYY-03-xx pre-tournament) snapshots for the trajectory delta.
- **Schema note:** the snapshot files carry only the 8 raw four-factor stats (`effective_fg_pct` + opp, `turnover_rate` + opp, `offensive/defensive_reb_rate`, `free_throw_rate` + opp) — **not** `adj_efficiency_margin`. We synthesize a Dean-Oliver-weighted margin proxy.
- **Algorithm:**
  1. For each team t, compute ``four_factor_margin(t)`` at January and March snapshots:
     ```
     margin = (eFG% - opp_eFG%)
            + 0.3 * (off_reb_rate + def_reb_rate - 1.0)
            + 0.2 * (opp_turnover_rate - turnover_rate)
            + 0.1 * (free_throw_rate - opp_free_throw_rate)
     ```
     Dean-Oliver canonical weights (40/25/20/15 for shooting / turnovers / rebounding / FTs).
  2. `delta_t = march_margin(t) - january_margin(t)` (absolute, not relative — see deviation note).
  3. `factor_t = 1 + 0.03 * tanh(delta_t × 10)` ∈ [0.97, 1.03].
  4. `adjusted[r] = base[r] × factor_t`. Re-normalize per round to bracket team-counts.
  - **Two-sided:** improving teams boosted, declining teams penalized. `tanh` saturation caps both ends — even a pathological delta of 0.5 (huge) stays within ±3%.
  - **Deliberate deviations from catalog spec:**
    - Spec called for `delta_eff = (march - january) / january` using Torvik's `adj_efficiency_margin`. But (a) `adj_efficiency_margin` isn't in the snapshot files — only 8 raw four-factor stats — so we use a Dean-Oliver-weighted proxy instead; (b) division by the January value blows up when it's near zero (mean of the distribution), so we switched to absolute delta with a calibrated `tanh_scale=10` multiplier.
    - Applied at round_probs level, not barthag (same pattern as the other adjustments).
- **Walk-forward contract:** only reads `torvik_four_factors_{year}_*.json` for `{year}` — each test year's momentum is computed in isolation from its own Jan and Mar snapshots. No cross-year contamination.
- **Fallback:** Missing January or March snapshot for a year → empty dict for that year → downstream resolver treats every team as delta=0 (no adjustment). Teams present in only one of the two snapshots → absent from result (same neutral fallback).
- **2026 sanity:** 68/68 coverage. Top improvers: Howard (+0.064), Prairie View (+0.059), UMBC (+0.045) — small low-majors often have schedule-strength artifacts in the delta. Top decliners: Iowa (−0.052), NC State (−0.049), BYU (−0.047), Texas A&M (−0.047), Saint Louis (−0.045). `tanh` cap keeps extreme signals bounded either way.
- **Years:** 2008-2026.
- **File:** `src/prediction/momentum_probabilities.py`; lock test: `tests/test_momentum_adjustment.py` (14 tests).

---

### Category D: Upset-Aware Ratings

#### D1: `volatile`
- **Status:** IMPLEMENTED (2026-04-24) — shipped as an ADJUSTMENT rather than a base, composable with every source and construction. Uses the cbbpy team-ID bridge (phase 1b7) to map historical game records to tournament team IDs.
- **Data:** `data/raw/historical/historical_games_{year}.json` (bridged via `src.data.normalize.bridge_cbbpy_id`), cut off at Torvik's `tournament_start` date for walk-forward safety.
- **Algorithm:**
  1. For each tournament team, collect point margins across every pre-tournament regular-season game.
  2. Volatility = standard deviation of those margins (proxy for game-to-game inconsistency).
  3. Normalize per year: `v ∈ [0, 1]` where 1.0 = noisiest team in the field, 0.0 = most consistent.
  4. For each team `t` and round `r`: blend the base round_prob toward the per-round uniform rate (`teams_per_round[r] / 64`):
     `adjusted[t][r] = (1 - v * strength) * base[t][r] + v * strength * uniform[r]` (default `strength=0.5`)
  5. Re-normalize per round to the bracket's team-count targets.
  - **Effect matches catalog intent:** strong volatile teams lose mass (more upset losses toward round mean); weak volatile teams gain mass (more upset wins). Low-vol teams keep their baseline.
  - **Deliberate deviation from catalog spec:** spec called for pairwise Log5 noise at MC sampling time; we apply at the round-probs level for composability (same signal, every source × construction gets a `_+volatile_` variant).
  - Teams with fewer than 5 games get the neutral `v=0.5` fallback — no bias either way.
- **Years:** 2011-2026 (whenever `historical_games_{year}.json` + Torvik cutoff are both ingested)
- **File:** `src/prediction/volatile_probabilities.py`; lock test: `tests/test_volatile_adjustment.py`

#### D2: `upset_tuned`
- **Status:** IMPLEMENTED (2026-04-24) — shipped as an ADJUSTMENT rather than a base. Composable with every source / construction; tuples produce e.g. `torvik+upset_tuned_confidence`, `odds+contrarian+upset_tuned_f4_first`.
- **Data:** `data/raw/historical/tournament_results_{year}.json` (2005-2026; already team-ID-normalized to the Torvik / seeds ID scheme — was the first adjustment shipped before the cbbpy bridge landed)
- **Algorithm:**
  1. Walk-forward: compute empirical seed-by-round reach rates from all tournaments in `[2005, test_year)`
  2. For each team t with seed s(t) in the test year, compute model's mean round-r probability across all s(t)-seeds
  3. Calibration factor = `historical_reach_rate[s(t)][r] / model_mean_rate_for_seed_s(t)[r]`, clipped to [0.5, 2.0]
  4. Adjusted round_prob = `model_rp[t][r] * factor`, then re-normalized per round to {64, 32, 16, 8, 4, 2} teams
  - **Deliberate deviation from the original catalog spec:** the spec called for pairwise calibration inside the MC simulation; this implementation operates on round_probs before sampling so it remains composable with every source and construction. Captures the same signal (historical over/under-performance by seed) at the seed-aggregate level.
  - **Addresses known Log5 bias:** underestimates upset seeds that historically over-perform (11, 12, 15) and over-estimates chalk seeds that have quietly drifted.
- **Years:** 2011-2026 (needs ≥3 prior tournaments; 2008-2010 also workable but backtest window starts at 2011)
- **File:** `src/prediction/upset_tuned_probabilities.py`; lock test: `tests/test_upset_tuned_adjustment.py`

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

### M3a: `f4_chalk` (anchor-restricted F4-first)
- **Status:** IMPLEMENTED (2026-04-24, post-Phase-3)
- **Algorithm:** Identical to `f4_first` (M3) except the per-region F4-anchor draw is restricted to teams seeded 1, 2, or 3. Falls back to the full region pool if no top-3 seed is in `round_probs` for a region.
- **Why:** Phase 3 found `seed_f4_first` is the Bonferroni-significant winner. This variant tests whether the edge comes from locking *strong* anchors specifically. Historically ~75% of F4 spots go to seeds 1-3, so this concentrates the portfolio on the modal F4 outcome.
- **File:** `scripts/mc_pool_backtest.py:sample_f4_chalk_brackets()`; lock test: `tests/test_f4_anchor_constructions.py`

### M3b: `f4_diverse` (anchor-restricted F4-first, no 1-seeds)
- **Status:** IMPLEMENTED (2026-04-24, post-Phase-3)
- **Algorithm:** Identical to `f4_first` except 1-seeds are excluded from the F4-anchor pool — every locked anchor is a 2-15 seed. Falls back to the full region pool if no eligible team exists.
- **Why:** Counterpart to `f4_chalk`. Bets against the modal "1-seed in F4" outcome — useful in chaos years per the Chaos Index hypothesis (predicted regime → which mode to use).
- **File:** `scripts/mc_pool_backtest.py:sample_f4_diverse_brackets()`; same lock test as M3a.

### M3c: `f4_top4` (anchor-restricted F4-first, seeds 1-4)
- **Status:** IMPLEMENTED (2026-04-24)
- **Algorithm:** Identical to `f4_first` except the per-region F4-anchor draw is restricted to seeds 1-4. Middle ground between `f4_chalk` (seeds 1-3) and `f4_first` (all seeds). Historically ~85% of F4 spots go to seeds 1-4.
- **Why:** Closes the anchor-restriction parameter space. If `f4_chalk` (1-3) and `f4_first` (any) bracket the Phase 3 winner's behavior, `f4_top4` is the sweet-spot candidate — chalky enough to match the modal F4 distribution, loose enough to allow the occasional 4-seed regional winner.
- **File:** `scripts/mc_pool_backtest.py:sample_f4_top4_brackets()`; lock test: `tests/test_anchor_construction_variants.py`

### M4a: `e8_chalk` (anchor-restricted E8-first)
- **Status:** IMPLEMENTED (2026-04-24)
- **Algorithm:** Identical to `e8_first` except each of the 8 per-quadrant S16-anchor draws is restricted to seeds 1-6. Top quadrant draws from {1, 4, 5}; bottom from {2, 3, 6}. Falls back to the full quadrant pool if no top-6 seed is in the pool for that quadrant.
- **Why:** Phase 3 found `seed_f4_first` beats `seed_e8_first`. Two possible causes: (a) E8 locks too many anchors, (b) E8 locks the *wrong* anchors (seeds 7-16 that historically don't reach S16). `e8_chalk` isolates (b) — if it closes the gap against F4-first, the E8 problem was anchor-selection not constraint-count.
- **File:** `scripts/mc_pool_backtest.py:sample_e8_chalk_brackets()`; same lock test file.

### M4b: `e8_diverse` (anchor-restricted E8-first, no 1-seeds)
- **Status:** IMPLEMENTED (2026-04-24)
- **Algorithm:** Identical to `e8_first` except 1-seeds are excluded from the anchor pool. Top-quadrant top-6 eligibility drops from {1, 4, 5} to {4, 5}; bottom quadrant is unaffected (2-seeds still eligible).
- **Why:** Counterpart to `e8_chalk`. Forces at least one non-1-seed S16 anchor per region. Tests whether aggressive anchor diversity helps E8-first bracket performance in chaos years (matches the Chaos Index hypothesis).
- **File:** `scripts/mc_pool_backtest.py:sample_e8_diverse_brackets()`; same lock test file.

### M5: `backward`
- **Status:** **KILLED 2026-04-25** — spec-faithful implementation produces a strategic duplicate of `champ_first` under the existing sampler infrastructure. See § Deprecated Strategies for full rationale.
- **Original spec (preserved for reference):**
  1. Draw champion from CHAMP probability distribution
  2. Draw F4 from `P(team makes F4 | this champion won)` — conditioned probabilities
  3. Continue backward: E8, S16, R32, R64
  - **Why it dies:** every existing sampler (`sample_model_brackets`, `_sample_with_locks`, etc.) draws each game using pairwise-normalized round_probs *marginals* under a path-consistency constraint. The "joint" the backward conditional would slice into is therefore the marginal-independence joint, and Bayesian conditioning on it is mathematically equivalent to direct sampling — i.e., `champ_first`. The "different anchor philosophy" framing is normative, not distributional.
  - **What would resurrect it:** plumbing `barthag` (or cached MC samples) through the construction-mode dispatcher so backward could sample from the true Log5+barthag joint instead of the marginal-independence joint. That's a separate architectural lift, not a strategy addition.
- **File:** Not implemented. Stub at `scripts/mc_pool_backtest.py:1954` left commented out.

### M6: `confidence`
- **Status:** IMPLEMENTED (2026-04-24)
- **Algorithm:** Per-game decision based on prediction confidence:
  1. For each game, compute `confidence = |P(fav) - 0.5|`
  2. **High confidence** (P > 0.85): always pick favorite (lock chalk). Wastes no randomness on 1v16 games.
  3. **Medium confidence** (0.60 < P < 0.85): sample from model probability (standard stochastic)
  4. **Low confidence** (P < 0.60): sample with BOOSTED variance — inflate upset probability by 1.5× to explore differentiation opportunities
  - Effect: concentrates bracket diversity on the games that actually matter for pool differentiation (5v12, 6v11, 7v10) while locking in the games everyone agrees on
  - No anchoring — works game-by-game, not by locking teams through paths
- **File:** `scripts/mc_pool_backtest.py:sample_confidence_brackets()`; lock test: `tests/test_confidence_construction.py`

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
| **Sources** | seed, torvik, odds, spread_power, pool_wisdom, elo, massey_avg, massey_best (8) | ap_strength (1) |
| **Adjustments** | contrarian, upset_tuned, volatile, roster_adj, coach_adj, momentum (6) | _(none)_ |
| **Constructions** | forward, champ_first, f4_first, e8_first, confidence, f4_chalk, f4_diverse, f4_top4, e8_chalk, e8_diverse (10) | _(none — M5 backward killed 2026-04-25 as spec-faithful duplicate of champ_first; resurrection requires barthag plumbing)_ |
| **Blending** | Equal-weight and custom-weight blends of any 2+ sources | Stacked meta-learner (B5) |
| **Testing Budget** | `run_budget()` enforces T1/T2/T3 parameters + kill rules, cut-losses gate at T2 | Round-probs caching, multi-proc parallelism, convergence-based repeat stopping |
| **Tournament Oracle** | `--oracle <year>` + auto-run inside `run_budget()` T3 (2026-04-24, phase-3 metrics rollout); F4/finals/champ hits + ranker_gap_espn_pts per (year, strategy); ledger in `memory/tournament_oracle.md` | Drift-guard test hooking oracle output into T3 gate |
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
| ~~23~~ | ~~`odds_backward`~~ | ~~A3~~ | ~~M5~~ — KILLED 2026-04-25 |
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
| ~~49~~ | ~~`consensus_backward_blend60`~~ | ~~consensus with 60/20/20 weights instead of equal~~ — depends on M5 (KILLED 2026-04-25) |
| ~~50~~ | ~~`consensus_backward_blend80`~~ | ~~consensus with 80% market weight~~ — depends on M5 (KILLED 2026-04-25) |
| 51 | `volatile_confidence_low` | volatility noise × 0.5 |
| 52 | `volatile_confidence_high` | volatility noise × 2.0 |
| 53 | `odds_confidence_tight` | confidence thresholds: 0.90 / 0.70 |
| 54 | `odds_confidence_wide` | confidence thresholds: 0.80 / 0.55 |
| 55 | `momentum_f4_first_strong` | momentum adjustment ±5% instead of ±3% |
| ~~56~~ | ~~`stacked_backward`~~ | ~~meta-learned weights + backward construction~~ — depends on M5 (KILLED 2026-04-25) |
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

**Secondary — placement diagnostics:**
- **p_top5%** — fraction of pool runs the bracket finishes in the top 5% (i.e., money-ish in a 20-person pool; noise in a WTA). Produced by the backtest since 2026-04-23 but only surfaced in tier reports as of phase-1 metrics rollout (2026-04-24).
- **p_top25%** — fraction in the top quartile. Consistency diagnostic; a strategy with high p_top25 but low P(1st) is "frequently good but rarely first" — useful for guided-random pool formats, less useful for WTA.
- **MeanRank, BestRank** — placement diagnostics; don't pay out but surface portfolio consistency.
- **Tournament oracle** — `python -m scripts.run_experiment --oracle <year>` scores the saved portfolio against actual tournament outcomes (F4 hits / finals hits / champion / ranker gap in ESPN points). Per-year ledger with regime labels in `memory/tournament_oracle.md`. The `ranker_gap_espn_pts` field — points left on the table because the ranker didn't promote the best bracket — is the direct KPI for the selection/ranking problem (North Star lever #2).

**Cross-year confidence intervals (phase-1 metrics rollout, 2026-04-24):**
- Every primary and secondary mean above ships with a **95% CI half-width** (`ci95_p_first`, `ci95_best_score`, `ci95_mean_score`, `ci95_p_top5`, `ci95_p_top25`). The CI is a normal-approx on the cross-year SEM — it quantifies "how stable is this strategy's mean across the 14-year window?", not per-year binomial uncertainty (that's phase-2 work).
- A strategy with high mean P(1st) and a wide CI needs more years of observation before promotion; a strategy with narrower CI at the same mean is a safer bet.

All metrics go in `_print_summary` (human-readable ranked table) and `stats_for_artifact` (JSON dump). ESPN scoring is the team-identity score under `--team-identity` (locked per MEMORY.md §1 O26/O27).

### Significance Gate
```
Canonical gate (P(1st) — the payout metric):
  Test:   Paired permutation test (10,000 draws) on P(1st) per year
  H0:     P(1st)_new = P(1st)_seed_forward
  H1:     P(1st)_new > P(1st)_seed_forward
  Gate:   p < 0.10 (one-tailed)
  Also:   P(1st)_new > P(1st)_seed_forward in ≥ 8/14 years

Companion diagnostic blocks (phase-2 metrics rollout, 2026-04-24):
  Same paired-permutation harness run on BestScore and MeanScore
  with Bonferroni α = 0.10/N_tests per block. Informational — surfaces
  strategies that win on ESPN points without winning on P(1st) (usually
  opponent-model mis-specification, not scoring shape).
```

**Per-year binomial CI on P(1st)** — phase-2 metrics rollout also added Wilson 95% score intervals per year: `ci = wilson_ci95(p_first, n_trials)` where `n_trials = n_model × n_repeats`. Stored in `stats["strategy"]["p1_wilson_ci_by_year"]` as `{year: (center, halfwidth)}`. Distinct from the cross-year CI from phase-1 — answers "given this year's N_trials, how tight is the P(1st) estimate for this single year?" Small N_trials (e.g., T1's 25 repeats × 50 brackets = 1250) produces noticeable per-year CI; T3's 5000 trials/year narrows it substantially.

P(1st) remains the canonical gate because it's the only metric that converts to money. BestScore and MeanScore blocks catch money-framing sanity checks — a strategy that improves BestScore but not P(1st) is producing high-ceiling brackets that lose to the field anyway.

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
| 1b7 | cbbpy team-ID bridge (unblocks A4 Elo, C2 roster_adj, D1 volatile) | **DONE** | `bridge_cbbpy_id()` in `src/data/normalize.py` — longest-prefix match + 12 explicit edge-case aliases; coverage 17/68 → 68/68 on 2026 tournament field (2026-04-24) |
| 1b8 | Metrics rollout — phase 1 (surface dropped metrics + cross-year CI95) | **DONE** | `aggregate_strategy_stats` now tracks primary triple (P(1st), BestScore, MeanScore) + secondaries (p_top5, p_top25) with 95% CI half-widths; `_print_summary` + `stats_for_artifact` surface them; +16 pass locks the contract (2026-04-24) |
| 1b9 | Metrics rollout — phase 2 (Wilson CI per year + multi-metric significance) | **DONE** | `wilson_ci95()` attaches per-year binomial CI to `stats.p1_wilson_ci_by_year`; `_run_significance_tests` now runs paired permutation on P(1st), BestScore, MeanScore (3 blocks) each with its own Bonferroni α; `run_backtest` emits `n_trials` per result row (2026-04-24) |
| 1b10 | Metrics rollout — phase 3 (auto-Oracle inside T3) | **DONE** | T3 tier passes `save_brackets=True`; new `oracle_sweep_t3_years()` helper scores each (year, strategy) portfolio against ground truth and emits per-year detail + strategy aggregate (`mean_ranker_gap_espn_pts`, `mean_f4_hits`, `mean_finals_hits`, `champ_hit_rate`) into `summary["tiers"]["T3"]["oracle"]`; 7 integration tests lock the sweep against committed 2023-2026 artifacts (2026-04-24) |
| 1c | Elo base (A4) | **DONE** | Self-contained K=38 Elo from historical_games via cbbpy bridge; 68/68 coverage; +420 `elo_*` permutations (2026-04-24) |
| 1d | Massey bases (A5, A6) | **DONE** | A5 massey_avg DONE 2026-04-24 (pre-aggregated composite + 6-alias ID bridge; 68/68 2025 coverage). A6 massey_best DONE 2026-04-25 (walk-forward Brier selection over 56+ per-system rankers; min_games=500 eligibility; 14 lock tests). |
| 1d-2 | A6 massey_best (per-system walk-forward Brier selection) | **DONE** | Shipped 2026-04-25 in phase 1d. `src/prediction/massey_best_probabilities.py`; top-5 for 2026 cluster at mean Brier 0.172-0.181; 47 eligible systems. |
| 1e | AP base (A8) | TODO | ap_strength |
| 1f | Composite (B3 market_torvik, B4 consensus, B5 stacked) | TODO | handled by pipeline blending for B3/B4; B5 needs Ridge |
| 1g | Enriched bases (C1 coach, C2 roster, C3 momentum) | **DONE** | C1 coach_adj DONE 2026-04-25 (one-sided log-experience × +3% cap, MTeamCoaches × MNCAATourneySeeds, 68/68 bridge; 13 lock tests). C2 roster_adj DONE 2026-04-25 (top-5 WARP × ±4% multiplicative, cbbpy-bridged; 11 lock tests). C3 momentum DONE 2026-04-25 (Jan→Mar Dean-Oliver four-factor-margin delta × tanh(x·10) × ±3% cap; 68/68 snapshot coverage; 14 lock tests). |
| 1h | Upset bases (D1 volatile, D2 upset_tuned) | **DONE** | D2 upset_tuned (2026-04-24, walk-forward seed-by-round calibration) + D1 volatile (2026-04-24, per-team margin variance from cbbpy bridge). Both shipped as adjustments rather than bases for composability. |
| 2a | Backward construction (M5) | **KILLED 2026-04-25** | Spec-faithful impl is a strategic duplicate of `champ_first` under the marginal-independence sampler. Resurrection requires plumbing `barthag` through the construction-mode dispatcher (architectural lift, separate phase). See § Deprecated Strategies. |
| 2b | Confidence construction (M6) | **DONE** | *_confidence — lock chalk / sample medium / boost upsets per-game (2026-04-24) |
| 2c | Anchor-restricted F4 modes (M3a f4_chalk, M3b f4_diverse) | **DONE** | Post-Phase-3 expansion — tests whether `seed_f4_first`'s edge comes from chalk anchors (M3a) or survives diverse anchors (M3b) (2026-04-24) |
| 2d | Remaining anchor-restricted modes (M3c f4_top4, M4a e8_chalk, M4b e8_diverse) | **DONE** | Closes the anchor-restriction parameter space around the Phase 3 winner. +180 evaluable permutations; no-run phase per `memory/run_policy.md` (2026-04-24) |
| 3 | Full permutation evaluation | **DONE** | 2026-04-24 T1→T3 run on 301 candidates in 3.6 min wall-time; 81 killed at T1, 5 at T2; log: `artifacts/experiments/experiment_budget_20260424_155838.json` |
| 4 | Significance testing + dead-end pruning | **DONE** | 2026-04-24: `seed_f4_first` cleared Bonferroni α=0.02 (ΔP(1st)=+0.53pp, p=0.008, 11/14 yrs); `seed` cleared gate but not Bonferroni (p=0.058). All torvik / pool_wisdom / contrarian / upset_tuned / confidence variants failed. |
| 5 | Parameter sweeps on top strategies | TODO | T3b — sweep F4-first anchor thresholds, confidence bands around `seed_f4_first`; test against submission ranker |

---

## Phase 3 Results (2026-04-24)

First full tier-budget run on 301 strategies. Wall-time **3.6 minutes** (target ≤4hr). Kill rules pruned 293 strategies before T3 — exactly the behavior the Testing Budget was designed to produce.

Artifact: `artifacts/experiments/experiment_budget_20260424_155838.json`
Log: `artifacts/experiments/logs/budget_run_20260424_155835.log`

### T3 significance gate results (paired permutation, 10K draws, full 14-yr window, N=100)

| Strategy | ΔP(1st) vs `seed_forward` | p-value | wins/14 | Gate |
|---|---:|---:|---:|---|
| **`seed_f4_first`** | **+0.0053** | **0.008** | **11/14** | **PASS ★ Bonferroni α=0.02** |
| `seed` | +0.0055 | 0.058 | 9/14 | PASS gate, fails Bonferroni |
| `seed+contrarian_f4_first` | +0.0005 | 0.436 | 8/14 | fail |
| `0.5*pool_wisdom+0.5*seed+contrarian_e8_first` | −0.0027 | 0.817 | 3/14 | fail |
| `0.5*torvik+0.5*seed` | −0.0036 | 0.936 | 4/14 | fail |

### Headline

**`seed_f4_first` is the Phase 3 winner.** Bare seed table × F4-first construction beats the current `seed_forward` baseline by +0.53 pp P(1st) in 11 of 14 years (Bonferroni-significant). This is the first strategy in the composable-pipeline architecture to clear the full significance gate against the baseline.

### What got killed

Every sophisticated candidate failed to survive:

- **`confidence` construction:** killed at T1 (most `*_confidence` variants had zero-year P(1st) or wins<3/8).
- **`upset_tuned` adjustment:** `seed+contrarian+upset_tuned_f4_first` and `seed+upset_tuned_e8_first` passed T1 but failed T2's wins-vs-baseline threshold. All other `upset_tuned` variants died in T1.
- **`contrarian` chains:** `seed+contrarian_f4_first` reached T3 but posted ΔP(1st)=+0.0005 (p=0.436, 8/14 wins) — statistically indistinguishable from baseline.
- **`torvik` and `pool_wisdom` blends:** all made it to T2 but dropped below baseline when evaluated on the full 14-year window. The contrarian-blend variants in particular posted negative deltas.

### Interpretation

1. **Simple and correct beat elaborate and speculative.** The bare seed probability table, paired with F4-first construction, outperforms every torvik/market/upset-tuned variant. This is consistent with MEMORY.md §1's "BSS = 0 ceiling" — fancier prediction sources don't help.
2. **Construction mode is the binding lever** (North Star priority #3). `seed` with forward construction passed the gate but failed Bonferroni; swapping in F4-first lifted it to Bonferroni-significant. Same source, different construction, materially different result.
3. **The Testing Budget worked.** 301 strategies screened, dead ends pruned in minutes, the winner surfaced with a rigorous p-value. Total compute: 3.6 minutes. No multi-day babysitting.

### Next steps (Phase 5 candidates)

- Sweep F4-first anchor thresholds and confidence bands around `seed_f4_first` to see if a param variant pushes ΔP(1st) higher without breaking Bonferroni.
- Test `seed_f4_first` against the submission ranker — MEMORY.md §3 flags the ranker as the binding constraint (North Star lever #2); confirm whether this strategy keeps its edge once picked.
- Compare `seed_f4_first` against the currently-locked `f4_first_tv` (= `torvik_f4_first`) in MEMORY.md §1 Pool strategy. If the seed-based variant consistently beats the torvik-based variant in a fresh paired run, that's a candidate pool-strategy update.

---

## Outstanding Work

Items below are the open queue as of 2026-04-24 (post-Phase-3). Each is listed with what blocks it, why it's worth doing, and a rough scope. Adding a new strategy or construction below does **not** authorize a `--tier budget` run — see § Running experiments and `memory/run_policy.md`.

### Sources still to ship

| ID | Phase | Blocker / Prereq | Why it's worth it | Scope |
|----|-------|------------------|-------------------|-------|
| ~~A6 `massey_best`~~ | ~~1d-2~~ | ~~Per-system Brier-selection harness (none today)~~ | ~~All 150+ Massey systems are already on disk.~~ — **DONE 2026-04-25** (`src/prediction/massey_best_probabilities.py`, 14 lock tests, walk-forward Brier selection over 56+ systems with min_games=500 eligibility). |
| A8 `ap_strength` | 1e | Verify `data/kaggle/ap_poll_data.json` ingestion is current | Captures *public-perception* signal that's distinct from efficiency metrics — useful for opponent modeling (the field anchors on rankings, not barthag). Adds another orthogonal source for B4 `consensus`. | 1 file (`src/prediction/ap_probabilities.py`) + lock test. Algorithm is straightforward (rank → barthag table). |
| B5 `stacked` | 1f | Needs ≥3 prior years with all Category-A bases available — already true 2014+ | **The only learned-weight source in a sea of hand-weighted blends.** Every existing blend in the catalog is a hand-picked convex combination (`0.5*x+0.5*y`); `stacked` lets a Ridge regression pick the weights from historical accuracy. Different *kind* of strategy, not just another permutation. | 1 file (`src/prediction/composite_probabilities.py`) + walk-forward harness + lock test. Ridge fit is cheap; the work is wiring features and the year-by-year loop. |
| ~~C1 `coach_adj`~~ | ~~1g~~ | ~~Needs `data/kaggle/MTeamCoaches.csv`~~ | ~~Tournament-experience adjustment.~~ — **DONE 2026-04-25** (`src/prediction/coach_adj_probabilities.py`, 13 lock tests, 68/68 bridge coverage). |
| ~~C2 `roster_adj`~~ | ~~1g~~ | ~~UNBLOCKED by cbbpy bridge (1b7)~~ | ~~Top-5 player WARP adjustment.~~ — **DONE 2026-04-25** (`src/prediction/roster_adj_probabilities.py`, 11 lock tests). |
| ~~C3 `momentum`~~ | ~~1g~~ | ~~Needs Torvik four-factor monthly snapshots ingested for 2008-2026~~ | ~~January→March efficiency-trend adjustment.~~ — **DONE 2026-04-25** (`src/prediction/momentum_probabilities.py`, 14 lock tests, 68/68 snapshot coverage, Dean-Oliver proxy in lieu of missing `adj_efficiency_margin` field). |

### Construction modes still to ship

_(empty — M5 `backward` killed 2026-04-25 as a spec-faithful duplicate of `champ_first`. The construction-mode design space is exhausted at the marginal-independence sampling level. Resurrecting `backward` requires plumbing `barthag` through the dispatcher so it can sample from the Log5+barthag joint — see "Architectural lifts" below.)_

### Architectural lifts (not strategy additions, but unblock new strategy classes)

| Item | Why | Scope | Unlocks |
|------|-----|-------|---------|
| Plumb `barthag` through the construction-mode dispatcher | All current samplers (`sample_model_brackets`, `_sample_with_locks`, etc.) sample each game from pairwise-normalized round_probs *marginals*. The true Log5+barthag joint that produces those marginals is invisible at sample time. Plumbing barthag through would let new modes sample from the joint directly. | Modify `_make_sampler` in `mc_pool_backtest.py` to accept `barthag` (optional kwarg for modes that need it); modes that don't, ignore it. ~3 files: dispatcher, one-shot test, catalog. | Resurrects M5 `backward` (joint sampling); enables future modes that need joint access (e.g., "draw the whole bracket from one Log5 MC sample"). |

### Validation & guardrails

| Item | Phase | Why | Scope |
|------|-------|-----|-------|
| Drift-guard test for Oracle output | 1b10 follow-up | `oracle_sweep_t3_years()` ships its data into `summary["tiers"]["T3"]["oracle"]` but nothing currently fails CI if `ranker_gap_espn_pts` regresses. A drift-guard would lock the *expected* gap ranges per regime (chaos vs chalk, per `memory/tournament_oracle.md`). | 1 test file; reads the most-recent T3 artifact and asserts gaps stay within the historical envelope. |
| Phase 5 parameter sweeps on `seed_f4_first` | 5 | The Phase 3 winner is the bare seed table × F4-first. Open question: does anchor threshold (top-3 vs top-4 vs top-6 vs unrestricted) or seed-table sharpness lift ΔP(1st) above the +0.0053 baseline without breaking Bonferroni? | Sweep config + T3-rigor run on ~10-15 variants. **Requires explicit run authorization** per `memory/run_policy.md`. |
| Submission-ranker test for `seed_f4_first` | 5 | MEMORY.md §3 flags the ranker as the binding constraint (North Star lever #2). Phase 3 measured P(1st) over the *portfolio*; the actual submitted bracket is the one the ranker picks. Open question: does `seed_f4_first` keep its edge once filtered through the ranker, or does the ranker's `min mean_rank` rule drop the high-P(1st) brackets? | Re-run Phase 3 winner with `--team-identity --rank-mode mean_rank` and compare `BestScore` (oracle's best) vs the ranker's submitted bracket score. |
| `seed_f4_first` vs locked `f4_first_tv` (`torvik_f4_first`) | 5 | MEMORY.md §1 currently locks `f4_first_tv` as the production pool strategy. If `seed_f4_first` consistently beats it in a fresh paired run, that's a candidate pool-strategy update. Don't swap silently — needs a side-by-side Bonferroni-clearing comparison. | Paired permutation harness on the two strategies, full 14-year window, T3 rigor. **Run-gated.** |

### Run authorization queue

| What | Status |
|------|--------|
| Next `--tier budget` run on the now-7,920-strategy catalog (Phase 3 ran on 301; +7,619 added by 1c Elo, 1d massey_avg, 2c/2d anchor variants, D1 volatile, M6 confidence, C2 roster_adj, C1 coach_adj, C3 momentum, A6 massey_best) | **Authorized 2026-04-25** — operator will run on their machine as an "initial result set". Expected wall-time ~100-115 min at T1+T2+T3 full rigor. Do not trigger from this session. |
| Phase 5 sweeps + submission-ranker test + `seed_f4_first` vs `f4_first_tv` head-to-head | **Not authorized.** Same gate. |

### Why these matter (orthogonality, not just count)

The catalog is at ~7,920 evaluable permutations and growing — pure count is no longer the bottleneck. The remaining items expand the *qualitative* reach of the search:

- **`stacked` (B5)** is the only source where the data picks the blend weights instead of a human. Every other "blend" in the catalog is hand-weighted.
- ~~**`backward` (M5)** is the only construction mode that enforces internal bracket consistency.~~ **KILLED 2026-04-25** — a faithful implementation under the marginal-independence sampler is a strategic duplicate of `champ_first` (see § Deprecated Strategies). The "consistency-by-construction" framing was distributionally incorrect; resurrecting it requires the barthag-plumbing architectural lift described above.
- ~~**`massey_best` (A6)** is the only source that *selects* a single best ranker per year via Brier; everything else either uses one fixed model or ensembles them statically.~~ **DONE 2026-04-25** — shipped as the first (and still only) ranker-selection source in the catalog.
- **C1/C2/C3 enriched bases** introduce roster-, coach-, and trajectory-level signal that no efficiency metric on its own captures.

Each of these closes a *kind of strategy the current catalog literally cannot express* — that's the orthogonality argument for prioritizing them over yet another permutation of the existing surface.

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
| M5 `backward` construction | Spec-faithful impl produces a strategic duplicate of `champ_first` under the existing marginal-independence sampler. The catalog spec assumes "downstream conditioning" produces a different distribution from forward-fill, but Bayesian conditioning on the marginal-independence joint is mathematically equivalent to direct sampling from that joint — i.e., the same distribution `champ_first` already produces. Resurrection requires plumbing `barthag` through the dispatcher so backward can sample from the true Log5+barthag joint instead — listed as an "architectural lift" in § Outstanding Work, not a strategy addition. (2026-04-25) | KILLED — strategic duplicate |
