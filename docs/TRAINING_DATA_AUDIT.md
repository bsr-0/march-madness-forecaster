# Training Data Pipeline Audit Report

**Date**: 2026-02-22
**Scope**: Data ingestion, cleaning, training data construction, feature consistency, team mapping, train/test leakage
**Auditor perspective**: Senior ML Engineer, adversarial/skeptical stance

---

## Executive Summary

The pipeline is more sophisticated than most March Madness projects, with genuine leakage controls and temporal safeguards. However, I identified **17 issues** ranging from silent data corruption risks to structural train/test distribution mismatches that could meaningfully degrade out-of-sample prediction quality.

**Critical (4)**: Issues that likely cause measurable accuracy degradation
**Serious (6)**: Issues that introduce noise or silent failures
**Moderate (7)**: Design concerns that create maintenance/correctness risk

---

## CRITICAL ISSUES

### C1. Massive Feature Sparsity Mismatch Between Current-Year and Historical Training Data

**Location**: `src/pipeline/sota.py:3047-3735` (`_load_year_samples`)
**Severity**: CRITICAL

The historical training vectors produced by `_load_year_samples()` populate only **~20 of 77 matchup dimensions**. The remaining ~57 features are hardcoded to zero. Meanwhile, the current-year inference path (`extract_team_features` + `create_matchup_features`) populates all 77 dimensions from live Torvik/KenPom/roster/game-flow data.

**Populated in historical data** (positions in matchup vector):
- `[0-2]`: adj_off_eff, adj_def_eff, adj_tempo (from Sports Reference)
- `[3-8]`: Four Factors off/def (from BartTorvik cache, when available)
- `[26]`: sos_adj_em
- `[30-32]`: luck, wab, momentum (derived from game results)
- `[33]`: three_pt_variance proxy (margin_std)
- `[35]`: elo_rating (recomputed)
- `[36]`: free_throw_pct (from BartTorvik)
- `[44]`: three_pt_pct (from BartTorvik)
- `[47]`: win_pct
- `[66-70]`: absolute-level features
- `[76]`: seed_interaction

**Always ZERO in historical data** (~57 dimensions):
- `[9-10]`: drb_rate, opp_ft_rate (defensive Four Factors — not scraped)
- `[11-16]`: All RAPM/roster features
- `[17-19]`: Experience, bench depth, injury risk
- `[20-25]`: All volatility/entropy and shot quality metrics
- `[27-29]`: sos_opp_o, sos_opp_d, ncsos_adj_em
- `[34]`: pace_adj_variance
- `[37-43]`: assist metrics, steal/block rates, opp shot selection, conf_adj_em
- `[45-46]`: three_pt_rate, def_xp_per_poss
- `[48-65]`: elite_sos, q1_win_pct, foul_rate, 3pt_regression, rest_days, AP rank, coach metrics, pace_variance, conf_tourney_champ, neutral site, home dependence, transition metrics, positional RAPM, seed_strength
- `[71-75]`: interaction features (tempo, style, h2h, common_opp, travel)

**Impact**: The model trained on multi-year historical data learns that ~57 features are always zero. At inference time, these features have non-zero values. Tree-based models (LightGBM/XGBoost) will effectively ignore any split on these features learned from historical data, or worse, learn spurious splits based on the zero vs non-zero distribution. The year-decay weighting (0.85^years_ago) mitigates this partially but does not fix it — any current-year sample where these features matter will have its signal diluted by a large mass of zero-feature historical games.

**Recommendation**: Either (a) compute the missing features for historical years from the available box-score data (at minimum, drb_rate and opp_ft_rate can be derived from team_games), or (b) explicitly mask these features during multi-year training by zeroing them in current-year samples too, or (c) train a separate model on the reduced feature set for historical years and blend predictions.

---

### C2. Defensive Four Factors Partially Missing in Historical Data

**Location**: `src/pipeline/sota.py:3639-3642` and `3676-3687`

Within the Four Factors that _are_ scraped from BartTorvik for historical years, only offensive and two defensive metrics are populated:
- `[3]` efg_pct, `[4]` to_rate, `[5]` orb_rate, `[6]` ft_rate ✓
- `[7]` opp_efg_pct ✓, `[8]` opp_to_rate ✓
- `[9]` **drb_rate** — NOT SET (stays zero)
- `[10]` **opp_ft_rate** — NOT SET (stays zero)

The `fetch_four_factors()` scraper returns `opp_effective_fg_pct` and `opp_turnover_rate` but the code never assigns `defensive_reb_rate` or `opp_free_throw_rate` to positions `[9]` and `[10]`.

**Impact**: Two of the 8 Four Factors are systematically zero in all multi-year training data. Since DRB% and opponent FTR are moderately predictive features (DRB% especially), the model learns to ignore them from historical data even though they carry signal at inference time.

**Recommendation**: Extend the BartTorvik Four Factors scraper to also return DRB% and opponent FTR, or compute them from box-score data in `team_games` (which has `orb`, `drb`, `fga`, `fta`).

---

### C3. All 2022-2024 Historical Games Have a Single Date — Chronological Ordering Is Synthetic

**Location**: `src/pipeline/sota.py:3352-3372`

2022, 2023, and 2024 historical game files all have exactly **1 unique date** (the season-start date, e.g. "2021-11-01" for all 5,964 games in 2022). Only the 2025 file has real per-game dates.

The pipeline detects this and infers synthetic chronological dates from `game_id` numeric ordering:

```python
if len(unique_dates) <= 1 and len(games) > 50:
    # Infer chronological dates from game_id ordering
    season_start = date(year - 1, 11, 1)
    season_end   = date(year, 3, 13)
    frac = rank / max(len(id_ordered) - 1, 1)
    inferred = season_start + timedelta(days=int(frac * total_days))
```

**Problems**:
1. **ESPN game_id ordering is NOT perfectly chronological**. Game IDs are assigned at schedule creation time, not game time. Non-conference, conference, and tournament games have different ID ranges. The linear interpolation between Nov 1 and Mar 13 will misplace many games by weeks.
2. **Elo ratings**, luck, WAB, and momentum are computed on this synthetic chronological ordering. Elo updates on mis-ordered games produce different ratings than the true chronological sequence. The Elo difference between two teams can be meaningfully wrong.
3. **Tournament game filtering** (`_is_tournament_game(game_date_str)`) uses dates. Since all 2022-2024 games are synthetically dated, tournament games may be incorrectly classified — either leaking tournament outcomes into training, or incorrectly filtering regular-season games.

**Impact**: Rolling metrics (Elo, momentum, luck) are computed on a shuffled game sequence for 2022-2024. This introduces systematic noise in features that are presented to the model as if they were computed on the real chronological order.

**Note**: This problem extends beyond 2022-2024. Inspection of the cbbpy cache reveals that seasons **2010 through 2024** (15 of 16 historical backfill years) all have collapsed single-date data. Only 2005-2009 and 2025 have real per-game dates. All time-dependent features (rest_days, back_to_back, games_in_last_7_days, recency-weighted rolling stats) in the materialized parquet files are therefore meaningless for 2010-2024.

**Recommendation**: Use the actual game dates from the `team_games` box-score data (which should have them from cbbpy), or scrape game dates separately and merge. Alternatively, acknowledge this limitation and reduce the weight on Elo/momentum features for years without real dates.

---

### C4. `sports_reference_2026.json` Cache Is All-Zeros Skeleton

**Location**: `data/raw/cache/sports_reference_2026.json`
**Severity**: CRITICAL

This cache file exists with 365 teams but every field is zero (`pace: 0.0`, `off_rtg: 0.0`, `def_rtg: 0.0`). The schema is also degraded — only 4 fields (`team_name`, `pace`, `off_rtg`, `def_rtg`) vs the standard 8 fields (`team_name`, `pace`, `off_rtg`, `def_rtg`, `wins`, `losses`, `srs`, `sos`).

The `_has_critical_zeros()` check in the SR scraper should reject the all-zero `def_rtg`, forcing a re-fetch. However, if the re-fetch fails (network error, rate limit), the missing `wins`, `losses`, `srs`, and `sos` fields will silently produce zeros downstream, poisoning current-season team metrics.

**Impact**: If this cached file is ever loaded without re-validation, every team's efficiency metrics become zero, making all predictions meaningless.

**Recommendation**: Delete the corrupted cache file. Add a schema-completeness check that rejects files missing required fields (not just zero-value checks).

---

## SERIOUS ISSUES

### S1. Team ID Format Mismatch — Zero Direct Matches Between Games and Metrics

**Location**: Data files in `data/raw/historical/`

Game files use mascot-suffixed IDs (`gonzaga_bulldogs`, `duke_blue_devils`), while metric files use canonical short IDs (`gonzaga`, `duke`). Tournament seed files use yet another format (`uconn` vs metrics' `connecticut`; `byu` vs metrics' `brigham_young`; seeds' `saint_mary_s` vs metrics' `saint_mary_s__ca`).

There are **zero** direct matches between the 700 unique game IDs and 364 metric IDs for any year.

The pipeline has extensive resolution logic (CBBpy CSV lookup, prefix matching, fuzzy matching) that resolves most of these. The code at `sota.py:3322-3350` uses 4-pass resolution: direct match → CBBpy CSV → prefix match → TeamNameResolver fuzzy fallback.

**Problems**:
1. **Prefix matching is greedy and ordered by key length** (`metric_keys = sorted(team_metrics.keys(), key=len, reverse=True)`). This means `new_mexico_state_aggies` might match `new_mexico_state` correctly, but the ordering could cause `new_mexico_lobos` to match `new_mexico_state` before `new_mexico` depending on sort order.
2. **`_LOCATION_TO_METRIC` alias table** (`sota.py:3245-3283`) is incomplete. Missing aliases include: "Pitt" → "pittsburgh", "UNC" → "north_carolina", "SFA" → "stephen_f_austin", potentially causing resolution failures for some teams.
3. The resolution rate is logged but **not gated** below 80% (only a warning). If resolution drops to 60% for an older year, the model silently trains on a biased subset of games (likely biasing toward major-conference teams whose names are simpler to resolve).

**Impact**: Unresolvable teams are silently dropped from training data, creating a systematic bias toward well-known programs. The pipeline logs the resolution rate but doesn't fail or alert on poor rates.

**Recommendation**: Add a hard gate (e.g., fail if resolution rate < 75%). Extend the alias table for known problem cases. Consider building a comprehensive team_id mapping file that maps game IDs to metric IDs once, rather than resolving dynamically.

---

### S2. `def_rtg` Precision Inconsistency Across Years

**Location**: `data/raw/historical/team_metrics_*.json`

- 2022-2024: `def_rtg` has 14-16 decimal digits (e.g., `93.95765723999382`)
- 2025: `def_rtg` has exactly 4 decimal places (e.g., `100.4464`)
- `off_rtg` is always 1 decimal place across all years

This suggests **different data sources** were used across years:
- 2022-2024: The high-precision `def_rtg` is likely computed internally (e.g., `off_rtg - margin` or similar derivation), while `off_rtg` comes directly from Sports Reference
- 2025: Both values come from a consistent source with uniform precision

**Impact**: While floating-point precision differences don't directly cause model errors (the actual values are correct), this is a signal that the data provenance differs across years, which could mean subtle semantic differences in what `def_rtg` represents (e.g., opponent-adjusted vs raw, per-100-possessions vs per-game).

---

### S3. Point-in-Time (PIT) Feature Adjustment Uses Coarse Proxy

**Location**: `src/pipeline/sota.py:1862-1917`

The PIT adjustment for current-year training data blends end-of-season features with noise and mean regression proportionally based on "season progress" — but only for the current year's games. Historical years use full-season-end metrics with no PIT adjustment at all.

**Problems**:
1. Historical games (2005-2024) use **season-end** efficiency metrics to train on games that occurred in November. A team that improved dramatically mid-season has its February quality leaked into its November games.
2. The PIT adjustment for the current year uses a `progress` variable derived from sort-key ordering, but this only applies to current-year samples (lines 1876-1917). Multi-year historical samples get none of this adjustment.
3. The comment says "V2 replaces pure noise injection with actual point-in-time metric snapshots" but the PIT snapshots are only populated where available. For many features, the fallback is still `season_remaining * noise`.

**Impact**: The model sees artificially stable features for early-season games in historical data (because season-end metrics are used), then sees noisy/adjusted features for early-season current-year games. This distribution mismatch degrades the model's ability to generalize across seasons.

---

### S4. BartTorvik Four Factors Lookup Uses Bidirectional Prefix Matching

**Location**: `src/pipeline/sota.py:3621-3629`

```python
def _get_ff(team_id: str, field: str, default: float = 0.0) -> float:
    ff = four_factors.get(team_id)
    if ff is None:
        for bk in four_factors:
            if team_id.startswith(bk) or bk.startswith(team_id):
                ff = four_factors[bk]
                break
    return float(ff.get(field, default)) if ff else default
```

The bidirectional prefix check `team_id.startswith(bk) or bk.startswith(team_id)` is dangerous:
- `miami` matches `miami__fl` ✓ but also `miami__oh` — first match wins
- `texas` matches `texas_a_m`, `texas_arlington`, `texas_state` etc.
- `north_carolina` matches `north_carolina_a_t`, `north_carolina_central`, `north_carolina_state`

Since `four_factors` is a dict and iteration order depends on insertion, the wrong team may be matched silently.

**Impact**: Some teams' Four Factors values may be borrowed from wrong teams. For conference rivals with similar names (e.g., NC State getting UNC's stats), this could introduce significant errors in 8+ training features.

---

### S5. Label Determination From `lead_history` Is Fragile

**Location**: `src/pipeline/sota.py:1758` and `sota.py:4063`

```python
1 if (game.lead_history and game.lead_history[-1] > 0) else 0
```

The training label is derived from whether the last element of `lead_history` is positive. This is fragile because:
1. If `lead_history` is empty (which happens for many historical games), the label is always 0 (team1 lost), regardless of the actual outcome.
2. The last element of `lead_history` represents the final score margin, but only if the game flow was constructed correctly. For games loaded from box scores (no play-by-play), `lead_history` may contain only `[final_margin]` or be empty.
3. For historical years loaded via `_load_year_samples`, labels come from `s1 > s2` (correct). But for current-year games fed through `_train_baseline_model`, they rely on `lead_history[-1] > 0` which requires complete game flow construction.

**Impact**: If current-year game flows have incomplete `lead_history`, some labels will be systematically wrong (all labeled as losses), creating label noise in the training data.

---

## MODERATE ISSUES

### M1. 2025 Game Data Contains Exhibition/Non-D1 Games

The 2025 historical games file contains games against non-D1 opponents:
- "Elms College Blazers", "Vermont State - Lyndon Hornets", "Piedmont Lions", "Alice Lloyd College Eagles", "Lincoln University (CA) Oaklanders"
- Scores like 150-142, 141-58, 136-64 suggest exhibition or NAIA opponents
- 34 games have scores exceeding 120 points

These non-D1 opponents will fail metric resolution (no entry in `team_metrics`), so the games are dropped. However, they inflate the `_total_team_refs` count and lower the reported resolution rate, potentially masking real resolution failures.

**Impact**: Low severity but adds noise to resolution rate metrics and makes it harder to detect actual mapping failures.

---

### M2. Tournament Seed ID Inconsistencies

Tournament seed files use inconsistent team_id formats that don't match metrics:
- Seeds: `uconn` → Metrics: `connecticut`
- Seeds: `byu` → Metrics: `brigham_young`
- Seeds: `saint_mary_s` → Metrics: `saint_mary_s__ca`

The materializer (`materialization.py:63-67`) runs `_align_source_team_ids()` on tournament seeds, which uses fuzzy matching. However, these known mismatches rely on the fuzzy matcher correctly resolving them every time.

**Impact**: If resolution fails for a seeded team, its seed_interaction feature and tournament matchup features will be missing. This is especially problematic because tournament seeds are among the most predictive features.

---

### M3. Year-Based Sort Keys for Historical Data Are Coarse

**Location**: `src/pipeline/sota.py:2107`

```python
year_sort_keys = np.full(len(hy), yr * 10000, dtype=float)
```

All games from a historical year get the same sort key (`yr * 10000`). This means:
1. The chronological train/val split boundary may cut through the middle of a historical year's games, but all games from that year have the same sort key, so they'll all end up on the same side.
2. When combined with current-year sort keys (which use actual date-based values), the boundary between historical and current data is sharp rather than gradual.

**Impact**: Minor — the train/val split is already computed before historical data is prepended, so this primarily affects the combined sample ordering.

---

### M4. COVID Year (2020) Exclusion Is Inconsistent

The code explicitly skips 2020 in multiple places:
- `materialization.py:163`: "Skipping season 2020 (COVID — no tournament)"
- `sota.py:2063`: `if yr != self.config.year and yr != 2020`
- `sota.py:4084`: `y for y in range(2015, self.config.year) if y != 2020`

However, `historical_games_2020.json` exists in the data directory with actual games. The 2020 season played 90%+ of its regular season before cancellation. The exclusion is correct for tournament prediction training, but the regular-season games could be used for Elo computation or team metric estimation if the exclusion were more nuanced.

---

### M5. `_get_ff` and `_get_sh` Functions Have Identical Fragile Pattern

Both `_get_ff` (line 3621) and `_get_sh` (line 3645) use the same dangerous bidirectional prefix matching pattern. This is a code smell that suggests the pattern was copy-pasted without addressing the underlying team resolution problem. A single robust resolver function should be used for all lookups.

---

### S6. Duplicate Normalization Functions With Divergent Behavior

**Location**: `src/data/scrapers/sports_reference.py` (`_normalize_id`) vs `src/data/normalize.py` (`normalize_team_id`)

The Sports Reference scraper has its own `_normalize_id` that diverges from the shared `normalize_team_id`:

| Input | SR `_normalize_id` | Shared `normalize_team_id` |
|-------|--------------------|-----------------------------|
| `"AkronNCAA"` | `"akron"` (strips NCAA first) | `"akronncaa"` (NCAA not stripped) |
| `"Loyola (IL)NCAA"` | `"loyola__il"` (double underscore) | `"loyola_il_ncaa"` (single underscore) |
| `"Saint Mary's (CA)NCAA"` | `"saint_mary_s__ca"` | `"saint_mary_s_ca_ncaa"` |

The SR scraper strips the NCAA suffix *before* ID normalization, while the shared function does not. The SR scraper also preserves double underscores, while the shared function collapses them. This causes silent join failures when downstream code expects one format but receives the other.

**Impact**: Teams with punctuation in their names (Saint Mary's, Loyola IL, St. Peter's) may fail to join between data sources that use different normalizers.

**Recommendation**: Consolidate to a single normalization function. The SR scraper's NCAA-stripping behavior should be moved to the shared normalize module.

---

### M6. 100% Null Columns in Processed Feature Tables

**Location**: `data/processed/matchup_features_2022_2025.parquet`

24 columns are 100% null across all 24,719 rows:
- All `is_home`/`is_away`/`is_neutral_site` columns (6)
- All `prior_prop_*` columns (18 — proprietary metrics not available historically)
- `prior_season_*` columns are ~79% null, `prior_conference_*` ~78% null

The materialization manifest declares these as "covered" in the coverage report, which is misleading.

**Impact**: These empty features add dimensionality without signal. While the model should learn to ignore them, they waste computation and inflate feature counts.

---

### M7. Tournament Matchup Features Only Contain 2025 Data (36 rows)

**Location**: `data/processed/tournament_matchup_features_2022_2025.parquet`

Despite the filename spanning 2022-2025, this file contains only 36 rows, all from season 2025. No tournament matchups from 2022, 2023, or 2024 are materialized.

**Impact**: The tournament matchup feature table is not usable for multi-year tournament outcome training. Tournament-specific features (seed interactions, region context) can only be learned from 36 games — too few for reliable learning.

**Recommendation**: Investigate why the materializer fails to produce tournament matchups for 2022-2024. The most likely cause is that tournament games in those years are filtered by the collapsed-date issue (C3) — if all games are dated before the tournament window, none pass the `_is_tournament_game` filter.

---

## ADDITIONAL OBSERVATIONS

### O1. Feature Coverage Is Honest and Well-Documented

The `_load_year_samples` docstring (lines 3059-3109) explicitly documents which features are populated and which stay zero. This is good practice — the sparsity is acknowledged rather than hidden.

### O2. Symmetric Augmentation Was Correctly Removed

Multiple comments note that symmetric augmentation (flipping team1/team2 for each game) was removed because it doubled sample count without adding information. This is the correct decision — tree models with bagging would overfit to the correlated duplicates.

### O3. The Leakage Controls Are Genuine

The pipeline has real temporal leakage controls:
- Tournament games excluded from baseline training
- Chronological train/val split
- GNN training restricted to training-era graph edges
- Deferred SOS refinement (GNN outputs not fed back into training features)
- Pre-optimization CFA weights used for calibration

### O4. Multi-Year Calibration Pool Is a Strength

Using historical tournament games (FIX 8.1) to augment the calibration pool is a solid approach — tournament games are the target domain, and historical tournament games are genuinely out-of-sample (different team-year combinations).

---

## SUMMARY TABLE

| ID | Severity | Issue | Status | Feature Impact |
|----|----------|-------|--------|---------------|
| C1 | CRITICAL | 57/77 features zero in historical training data | Open | Distribution mismatch between train/inference |
| C2 | CRITICAL | drb_rate and opp_ft_rate missing from historical Four Factors | **FIXED** | 2 predictive features now populated from BartTorvik |
| C3 | CRITICAL | 2010-2024 games all share single date; synthetic chronology | **FIXED** | Piecewise density model replaces linear interpolation |
| C4 | CRITICAL | sports_reference_2026.json cache is all-zeros skeleton | **FIXED** | Schema validation + corrupted cache deleted |
| S1 | SERIOUS | Zero direct team ID matches between games and metrics | **MITIGATED** | Alias table expanded from 33→65+ entries; 60% gate added |
| S2 | SERIOUS | def_rtg precision varies across years (different sources) | Open | Possible semantic inconsistency |
| S3 | SERIOUS | PIT adjustment only for current year, not historical | Open | Train/inference distribution mismatch |
| S4 | SERIOUS | Bidirectional prefix matching in Four Factors lookup | **FIXED** | Safe resolve cache with boundary-aware matching |
| S5 | SERIOUS | Label from lead_history[-1] fragile for incomplete games | **FIXED** | Score-based labels as primary; all 6 call sites updated |
| S6 | SERIOUS | Duplicate normalization functions with divergent behavior | **FIXED** | SR scraper delegates to shared normalize module |
| M1 | MODERATE | 2025 data includes non-D1 games | Open | Inflated game count, resolution rate noise |
| M2 | MODERATE | Tournament seed IDs inconsistent with metric IDs | **MITIGATED** | Covered by expanded alias table (S1) |
| M3 | MODERATE | Coarse sort keys for historical years | Open | Minor ordering artifact |
| M4 | MODERATE | 2020 exclusion is all-or-nothing | Open | Wastes regular-season games |
| M5 | MODERATE | Duplicated fragile prefix matching pattern | **FIXED** | Shared safe resolve cache (S4) |
| M6 | MODERATE | 24 columns 100% null in processed matchup features | Open | Wasted dimensionality |
| M7 | MODERATE | Tournament matchup features only contain 2025 (36 rows) | **Likely fixed** | C3 fix should unblock tournament date filtering |

---

## REMAINING WORK

Issues still open (lower priority or requiring architectural changes):

1. **C1**: Derive remaining ~55 features from box-score data for historical years (large effort — requires computing RAPM, roster metrics, shot quality from play-by-play data that may not be available)
2. **S2**: Audit def_rtg provenance across years (investigation task)
3. **S3**: PIT adjustment for historical years (requires per-game metrics snapshots — significant data collection effort)
4. **M1**: Filter non-D1 games from 2025 data
5. **M3/M4/M6**: Low-priority cleanup items
