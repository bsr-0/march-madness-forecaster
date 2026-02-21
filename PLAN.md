# Plan: Multi-Year Historical Training Data

## Goal
Train the ML pipeline on ALL available historical years (2005-2025), not just the current season's ~300 games. This addresses the #1 weakness identified in the audit: small sample size.

## Key Discovery
The cbbpy cache files (`data/raw/cache/cbbpy_historical_games_{year}.json`) contain full per-team-per-game box-score data for all 21 years: **fgm, fga, fg3m, fg3a, fta, turnovers, orb, drb, possessions, date, scores**. This is sufficient to compute most of the FIXED_FEATURE_SET features for historical years.

## What We Can Compute From Historical Box Scores

From the cached `team_games` data per team (aggregated across their season):
- **eFG% = (fgm + 0.5*fg3m) / fga** ✅
- **TO rate = turnovers / possessions** ✅
- **ORB rate** = orb / (orb + opp_drb) — requires pairing games ✅
- **FT rate = fta / fga** ✅
- **Opp eFG%** — from opponent's stats in the same game ✅
- **Opp TO rate** — from opponent's stats ✅
- **DRB rate** = drb / (drb + opp_orb) ✅
- **3PT% = fg3m / fg3a**, **3PT rate = fg3a / fga** ✅
- **FT%** = ftm / fta, where ftm = pts - 2*(fgm-fg3m) - 3*fg3m ✅
- **3PT variance** — game-to-game std of 3PT% ✅
- **Win%** — from scores ✅
- **Elo** — from game-by-game scores ✅
- **Off/Def rating, Pace, SOS** — from team_metrics files ✅
- **Seed** — from tournament_seeds files (2007-2025) ✅

## What We CANNOT Compute (not in box-score data)
- Player-level metrics (RAPM, experience, roster continuity, transfer impact)
- Assist rate, steal rate, block rate (no AST/STL/BLK in team_games)
- Lead volatility, entropy, comeback factor (no play-by-play granularity)
- Coach tournament data, preseason AP rank, conference tourney champ
- Travel distance (no venue info for historical games)

## Implementation Plan

### Step 1: New module `src/data/features/historical_features.py`

A self-contained module that:
1. Reads a cbbpy cache file for a given year
2. Aggregates team_games into season-level team stats (four factors, shooting, etc.)
3. Computes Elo ratings from game-by-game results
4. Loads tournament seeds when available
5. Builds `TeamFeatures`-compatible feature vectors with the computable fields populated and unavailable fields at sensible defaults
6. Constructs matchup differential vectors for all regular-season games
7. Returns `(X, y, sample_weights, game_dates)` ready for training

Key design decisions:
- Compute features from the same box-score data that produced the game outcomes (season-level aggregates, same as current-year pipeline)
- Use `team_metrics_{year}.json` for off_rtg/def_rtg/pace/SOS as authoritative values (consistent with current pipeline)
- Elo computed fresh per year with season-start regression toward 1500
- Features placed at the SAME indices as `TeamFeatures.get_feature_names()` for compatibility

### Step 2: Enrich `_load_year_samples()` in `sota.py`

Replace the current sparse 5-feature historical vector with the rich feature vectors from Step 1. The method already handles:
- Loading games/metrics per year
- Tournament game filtering
- Team ID resolution

Changes:
- Call `historical_features.compute_year_features()` to get rich team feature dicts
- Build matchup vectors using the same differential layout as current-year data
- Populate ~18 of 22 FIXED_FEATURE_SET features (vs. current 5)
- Remove symmetric augmentation (align with current-year: 1 sample per game)

### Step 3: Add `_build_multi_year_training_samples()` to `sota.py`

New method called from `_train_baseline_model()` that:
1. Iterates over historical years (configurable, default 2010-2025 excluding 2020)
2. Calls enriched `_load_year_samples()` for each year
3. Returns combined `(X_hist, y_hist, weights_hist, years_hist)`

### Step 4: Modify `_train_baseline_model()` to incorporate historical data

Current flow:
```
current-season games → samples → train/val split → train models
```

New flow:
```
current-season games → current_samples
historical years → historical_samples (via Step 3)
combine(historical + current) → all_samples
chronological sort → train/val split → train models
```

Key considerations:
- **Validation set**: Still comes from ONLY the current season (latest ~20% of current-year games). Historical data is training-only.
- **Year-based sample weighting**: Historical years are down-weighted relative to the current year using exponential decay. Current year weight = 1.0, each prior year decays by a configurable factor (e.g., 0.85 per year → 5 years ago = 0.44x). This combines multiplicatively with the existing within-season recency weighting.
- **PIT adjustment**: Only applied to current-season samples (historical features are already end-of-season aggregates matching their own labels).
- **Late-season cutoff**: Only applied to current-season games. Historical games use all regular-season games (their features are already season-level aggregates).
- **Feature selection**: Same FIXED_FEATURE_SET applied to all samples. Features unavailable in historical data get zeros — the model learns these features have zero variance in historical data and relies on them only for current-year discrimination.

### Step 5: Config additions to `SOTAPipelineConfig`

```python
# --- Multi-year training ---
enable_multi_year_training: bool = True
multi_year_training_start: int = 2010  # Earliest year to include
multi_year_training_exclude: List[int] = [2020]  # COVID year
multi_year_weight_decay: float = 0.85  # Per-year decay (0.85^5 = 0.44)
multi_year_cache_dir: Optional[str] = "data/raw/cache"  # Where cbbpy caches live
```

### Step 6: Update tests

- Unit tests for `historical_features.py` (feature computation from sample data)
- Integration test verifying multi-year training produces more samples
- Regression test ensuring single-year mode still works when `enable_multi_year_training=False`

## Expected Impact

- **Training samples**: ~300 → ~30,000+ (100x increase)
- **Feature coverage**: 5/22 → ~18/22 FIXED_FEATURE_SET features populated for historical data
- **Statistical power**: Brier score estimates, ensemble weights, and calibration all become much more reliable
- **Generalization**: Model learns from 15+ years of diverse team matchups instead of one season's worth

## Risk Mitigation

1. **Distribution shift across eras**: Year-based decay weighting + the model can learn which features are era-stable
2. **Feature mismatch**: ~4 features unavailable in historical data (player metrics, experience, continuity, coach data). These get zeros, which the gradient boosters can handle via split logic. The tree simply won't split on these features for historical samples.
3. **Backward compatibility**: `enable_multi_year_training=False` preserves current behavior exactly
4. **No scraping needed**: All data already cached locally
