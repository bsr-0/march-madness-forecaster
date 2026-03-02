# Plan: Top 5% → Top 1% — Closing the 7 Gaps

**Author**: Senior ML Engineer
**Date**: 2026-03-02
**Goal**: Close all 7 identified gaps to move from top 5-15% to top 1%
**Constraint**: ~600 tournament training samples; complexity must be justified

---

## Priority Order (by expected Brier improvement per engineering hour)

| Priority | Gap | Expected Δ Brier | Effort | Risk |
|----------|-----|-----------------|--------|------|
| **P0** | #1 Massey Ordinals composite | −0.008 to −0.015 | Low | Low |
| **P0** | #7 Round-weighted Brier scoring | −0.005 to −0.010 | Low | Low |
| **P1** | #2 Margin-of-victory primary path | −0.005 to −0.012 | Medium | Low |
| **P1** | #3 Model simplification | −0.003 to −0.008 | Medium | Medium |
| **P1** | #6 Data quality cleanup | −0.002 to −0.005 | Medium | Low |
| **P2** | #4 Women's bracket optimization | +0% on men's, critical for overall rank | Medium | Low |
| **P2** | #5 Bracket portfolio optimization | +0% on Brier, critical for bracket contests | High | Medium |

---

## Gap #1: Massey Ordinals Composite (P0) — PARTIALLY DONE

### Status After Previous Commit
- `KaggleDataLoader` exists and reads MMasseyOrdinals.csv
- `ExternalRatingsLoader.populate_from_massey_ordinals()` caches ordinal systems
- `massey_composite` meta-system is built (average normalized rank across all systems)
- `diff_external_rating_composite` and `diff_external_rating_spread` are in FIXED_FEATURE_SET
- **BUT**: The pipeline never actually calls `populate_from_massey_ordinals()` during
  `SOTAPipeline.run()`, and the external ratings are never populated on `TeamFeatures`

### Remaining Work

#### 1.1 Wire ExternalRatingsLoader into SOTAPipeline.run()
**File**: `src/pipeline/sota.py` — after `_load_team_stat_sources()` (line ~884)

```
Add new method: _load_external_ratings(teams) -> Dict[str, CompositeRating]
  1. If config.kaggle_dir is set:
     - Call ExternalRatingsLoader.populate_from_massey_ordinals(kaggle_dir, year)
  2. If config.external_ratings_dir is set:
     - Call ExternalRatingsLoader(cache_dir=external_ratings_dir).load_all(year)
     - Call .compute_composite()
  3. Fallback: generate_from_seeds(seed_map) for each team
  4. Return composite ratings dict
```

#### 1.2 Populate TeamFeatures.external_rating_composite/spread
**File**: `src/pipeline/sota.py` — in the feature extraction loop (line ~903)

```
Before extract_team_features(), look up composite rating for team_id:
  features.external_rating_composite = composite.composite_rating
  features.external_rating_spread = composite.rating_spread
```

This requires adding `external_ratings` parameter to `extract_team_features()` or
setting it post-extraction on the TeamFeatures object.

#### 1.3 Historical training with Massey Ordinals
**File**: `src/pipeline/sota.py` — in `_load_year_samples_incremental()`

For historical years (2005-2025), load Massey Ordinals per season if kaggle_dir
is available. This is critical: if the feature is 0.0 in training but nonzero at
inference, the model learns to ignore it.

**Alternative (simpler, recommended)**: Use massey_composite as a **post-hoc blend**
rather than a training feature. Train the main model on box-score features only,
then blend:

```
p_final = 0.7 * p_model + 0.3 * p_massey_composite
```

Where `p_massey_composite` = logistic(composite_diff / sigma), sigma calibrated on
historical tournament results. This avoids the train/inference mismatch entirely.

#### 1.4 Add kaggle_dir to SOTAPipelineConfig
**File**: `src/pipeline/sota.py` — SOTAPipelineConfig dataclass

```python
kaggle_dir: Optional[str] = None  # Path to Kaggle CSV directory
```

And wire it through main.py CLI args for `sota` and `sota-from-manifest` commands.

### Validation
- A/B test: Run LOYO CV with and without massey_composite blend
- Expected: 0.008-0.015 Brier improvement (based on published Kaggle solutions)
- Sanity check: massey_composite rank should correlate r > 0.85 with seed

---

## Gap #7: Round-Weighted Brier Scoring (P0)

### Current State
- Brier scoring at `src/ml/calibration/brier_optimal.py` uses **uniform game weights**
- Kaggle's actual metric since 2023: round-weighted, with finals worth much more
- The calibration pipeline optimizes for flat Brier, not the actual competition metric

### Implementation

#### 7.1 Define Kaggle's round-weight schedule
**File**: `src/ml/calibration/brier_optimal.py` — new constant

```python
# Kaggle March Mania 2024-2026 round weights
# Each game contributes differently to the total Brier score
# Later rounds are weighted more heavily
KAGGLE_ROUND_WEIGHTS = {
    "R64": 1.0,    # 32 games × 1.0  = 32
    "R32": 2.0,    # 16 games × 2.0  = 32
    "S16": 4.0,    #  8 games × 4.0  = 32
    "E8":  8.0,    #  4 games × 8.0  = 32
    "F4": 16.0,    #  2 games × 16.0 = 32
    "NCG": 32.0,   #  1 game  × 32.0 = 32
}
# Result: all rounds contribute equally to total score, but individual
# late-round games are worth 32× an R64 game
```

#### 7.2 Round-aware calibration target
**File**: `src/ml/calibration/brier_optimal.py` — modify `BrierOptimalSharpener.fit()`

Currently optimizes: `min_α Σ(p_sharp - y)²`

Change to: `min_α Σ w_round × (p_sharp - y)²`

where `w_round` comes from the weight schedule. This requires passing round
information alongside predictions/outcomes.

**Key insight**: Late-round games tend to be between good teams (seeds 1-4).
Round-weighting means the model should be **more carefully calibrated** in the
0.4-0.6 probability range (close matchups between top teams) rather than in the
0.85-0.99 range (1-seed vs 16-seed blowouts).

#### 7.3 Round-weighted training loss
**File**: `src/pipeline/sota.py` — in `_train_baseline_model()`

When computing sample_weight for tournament games used in training/validation,
multiply by the round weight. This focuses the gradient on the games that matter
most for the competition score.

For regular-season games, keep weight = 1.0 (they're not scored by Kaggle but
provide training signal).

#### 7.4 Round-weighted LOYO evaluation
**File**: `src/pipeline/sota.py` — in LOYO CV loop

Report both flat Brier and round-weighted Brier per year. The round-weighted
metric is the one that matters for competition ranking.

### Validation
- Compare flat vs round-weighted Brier on historical data
- If sharpening alpha differs significantly between flat and weighted, this gap
  was costing us competition points

---

## Gap #2: Margin-of-Victory Primary Path (P1)

### Current State
- `SpreadRegressor` already exists at `src/ml/ensemble/spread_model.py`
- It's trained on actual margins via LightGBM regression
- Converts to P(win) via logistic CDF with calibrated σ
- Currently weight ≈ 0.20-0.30 in the ensemble (behind LGB classifier and XGB)

### Problem
The spread model is a **secondary** component, not the primary path. The "raddar"
benchmark (dominant 2018-2024) makes margin prediction THE primary model and derives
probabilities from it. This is better because:
1. Richer gradient: continuous target vs binary label
2. Margin encodes calibration information: a 20-point win vs 1-point win
3. The logistic CDF conversion is a natural calibrator

### Implementation

#### 2.1 Promote SpreadRegressor to primary model
**File**: `src/pipeline/sota.py` — SOTAPipelineConfig

```python
# Change default ensemble weights:
ensemble_lgb_weight: float = 0.25  # was 0.45
ensemble_xgb_weight: float = 0.15  # was 0.35
ensemble_spread_weight: float = 0.45  # was ~0.20 (implicit)
# Logistic: 0.15 (residual)
```

The spread model should be the highest-weight component.

#### 2.2 Spread σ calibration on tournament data
**File**: `src/ml/ensemble/spread_model.py`

Currently σ is initialized at `spread_sigma_init` (default 11.0). This should be
**calibrated specifically on tournament margins**, which are systematically different
from regular-season margins (neutral site, higher variance, more upsets).

Add: tournament-specific σ calibration in the calibration pipeline.

```python
def calibrate_tournament_sigma(predicted_margins, actual_margins, actual_outcomes):
    """Find σ that minimizes Brier score on tournament games."""
    def brier(sigma):
        p = logistic_cdf(predicted_margins / sigma)
        return np.mean((p - actual_outcomes) ** 2)
    result = minimize_scalar(brier, bounds=(5.0, 20.0), method='bounded')
    return result.x
```

#### 2.3 Dual-target training
**File**: `src/pipeline/sota.py` — `_train_baseline_model()`

Instead of training LGB/XGB on binary labels and spread model on margins separately,
train all three on **margins** (regression) and convert to probabilities at prediction
time. This unifies the training objective.

Steps:
1. Train LightGBM regression (not classification) on margins
2. Train XGBoost regression on margins
3. Train SpreadRegressor on margins (already does this)
4. For each model, fit a separate logistic σ for margin → P(win) conversion
5. Ensemble the probabilities

### Validation
- LOYO CV comparing current (classification primary) vs proposed (margin primary)
- Expected: 0.005-0.012 Brier improvement
- Watch for: σ instability with small tournament samples

---

## Gap #3: Model Simplification (P1)

### Current State
- 3-model ensemble: LightGBM + XGBoost + Logistic (+ optional Spread, BT, GNN, Transformer)
- 22 features in FIXED_FEATURE_SET (reduced from 66 raw features)
- GNN and Transformer disabled by default (correct decision)
- ~600 tournament samples for training

### Problem
Even the reduced pipeline is overfit-prone at 600 samples. The 2024 winner used Monte
Carlo simulation in R. The 2017 winner used logistic regression on 5 features.

### Implementation

#### 3.1 Add "simple mode" config option
**File**: `src/pipeline/sota.py` — SOTAPipelineConfig

```python
model_complexity: str = "simple"  # "simple", "standard", "full"
# simple:   Logistic regression + SpreadRegressor only, 8-10 features
# standard: LGB + XGB + Logistic + Spread (current)
# full:     All models including GNN, transformer, BT
```

#### 3.2 Define the "killer 8" feature set
Based on published tournament prediction research and Kaggle leaderboard analysis:

```python
SIMPLE_FEATURE_SET = [
    "diff_adj_off_eff",             # [KP] Core efficiency
    "diff_adj_def_eff",             # [KP] Core defense
    "diff_sos_adj_em",              # [KAG] Schedule strength
    "diff_external_rating_composite", # Massey composite (THE key feature)
    "diff_elo_rating",              # [538] Season trajectory
    "diff_win_pct",                 # Simplest, strongest signal
    "diff_free_throw_pct",          # Most stable shooting metric
    "seed_interaction",             # Nonlinear upset dynamics
]
```

This captures >90% of the predictive signal with 8 features (well within the
sample size budget at 600/8 = 75 samples per feature).

#### 3.3 Regularization increase for standard mode
**File**: `src/pipeline/sota.py` — LightGBM/XGBoost hyperparameters

Current defaults may underregularize for 600 samples:
- Increase `min_child_weight` / `min_data_in_leaf` to 20+ (from likely ~10)
- Reduce `num_leaves` to 15-20 (from likely 31)
- Increase `lambda_l2` to 1.0+
- Reduce `num_rounds` to 100-150 with early stopping

#### 3.4 Cross-validation with simplicity bias
When LOYO CV shows simple ≈ standard within confidence interval, prefer simple.

### Validation
- LOYO CV: simple vs standard vs full
- Expected: simple ≈ standard (within CI), simple > full (reduced overfitting)
- Metric: round-weighted Brier (gap #7)

---

## Gap #6: Data Quality Cleanup (P1)

### Current State
- 2005-2009 data has quality issues: many zero fields, fake dates, team ID mismatches
- `materialization.py` skips COVID year (2020) but doesn't address early-year degradation
- Validator catches all-zero ratings but not partial zeros
- Historical pipeline defaults to 2022-2025 (avoiding the problem)

### Implementation

#### 6.1 Year-based quality weighting (already partially exists)
**File**: `src/pipeline/sota.py` — `training_year_decay: 0.85`

Current exponential decay already downweights old data. But it treats 2005
(bad data) the same as 2015 (good data, just older).

Change to: quality-aware decay.

```python
# Data quality multiplier per era
DATA_QUALITY_WEIGHTS = {
    range(2005, 2010): 0.3,   # Incomplete box scores, ID mismatches
    range(2010, 2015): 0.7,   # Better data, still some gaps
    range(2015, 2020): 1.0,   # High-quality era
    # 2020 skipped (COVID)
    range(2021, 2026): 1.0,   # Current era
}
# Final weight = quality_weight × temporal_decay_weight
```

#### 6.2 Feature completeness validation per season
**File**: `src/data/features/materialization.py`

Before including a historical season in training, check:
- What % of features are non-zero for each team?
- If < 50% of teams have non-zero adj_off_eff, skip the season
- Log warnings for partially complete seasons

```python
def _validate_season_quality(self, season: int, team_features: Dict) -> float:
    """Return quality score [0, 1] for the season's feature completeness."""
    n_teams = len(team_features)
    if n_teams == 0:
        return 0.0

    core_features = ["adj_off_eff", "adj_def_eff", "sos_adj_em", "win_pct"]
    complete = sum(
        1 for f in team_features.values()
        if all(getattr(f, feat, 0.0) != 0.0 for feat in core_features)
    )
    return complete / n_teams
```

#### 6.3 Team ID reconciliation with Kaggle IDs
**File**: `src/data/kaggle_loader.py`

When Kaggle data is available, cross-reference our normalized team IDs against
Kaggle's authoritative TeamID mapping. Log and fix mismatches.

```python
def reconcile_team_ids(self, pipeline_teams: Dict[str, str]) -> Dict[str, str]:
    """Return mapping: pipeline_team_id -> kaggle_canonical_id for mismatches."""
    kaggle_teams = self.load_teams()
    # Build fuzzy matcher using team names
    # Return corrections for teams that don't match
```

#### 6.4 Use Kaggle game results as authoritative backfill
**File**: `src/data/ingestion/historical_pipeline.py`

When `kaggle_dir` is set and the season data from cbbpy has quality issues,
fall back to Kaggle's MRegularSeasonDetailedResults.csv as the authoritative
source. The Kaggle data is cleaner for pre-2010 seasons.

### Validation
- Compare feature distributions 2005-2009 vs 2015-2019
- After cleanup, LOYO CV should show reduced variance in early-year Brier scores

---

## Gap #4: Women's Bracket Optimization (P2)

### Current State
- `enable_womens_pipeline: bool = True` in config
- `womens_seed_only_mode: bool = False` (can force seed-only)
- `SeedBasedOverrides` in brier_optimal.py has separate men's/women's tables
- Women's scrapers exist (Her Hoop Stats, NET rankings)
- **BUT**: No dedicated women's model — uses same pipeline with different data

### Problem
Since 2023, women's bracket is 50% of the evaluation. Women's basketball has
different dynamics:
- Fewer upsets (1-seeds win ~55% of championships vs ~45% for men)
- More concentrated talent (top 4-5 teams much stronger)
- Different pace/style distributions
- Much less public data for calibration

### Implementation

#### 4.1 Women's-specific seed-based historical rates
**File**: `src/ml/calibration/brier_optimal.py`

Already has `is_womens` parameter. Verify the seed override table uses women's
historical rates (should differ from men's: e.g., 1-vs-16 upset rate ~0.3% for
women vs ~1.5% for men).

#### 4.2 Dedicated women's feature weights
**File**: `src/pipeline/sota.py` — new section

Women's model should use a **simpler feature set** because:
- Less training data (women's bracket in Kaggle since 2023 only)
- More predictable (simpler dynamics → fewer features needed)

```python
WOMENS_FEATURE_SET = [
    "diff_adj_off_eff",
    "diff_adj_def_eff",
    "diff_sos_adj_em",
    "diff_win_pct",
    "diff_external_rating_composite",
    "seed_interaction",
]
```

#### 4.3 Women's calibration with stronger seed priors
The women's model should blend more heavily with seed-based priors because:
- Less data → more uncertainty → stronger priors
- Women's seeds are more predictive (fewer upsets)

```python
# Women's blend: 60% model + 40% seed prior (vs 80/20 for men's)
womens_seed_prior_weight: float = 0.40
mens_seed_prior_weight: float = 0.20
```

#### 4.4 Separate women's Massey composite
Load women's Massey Ordinals (WMasseyOrdinals.csv from Kaggle) and build a
separate composite for women's teams.

### Validation
- Historical women's tournament results (2010-2025) for backtesting
- Compare seed-only vs model+seed blend on women's bracket

---

## Gap #5: Bracket Portfolio Optimization (P2)

### Current State
- `ParetoOptimizer` generates 5 brackets (chalk → contrarian)
- `BracketPortfolioGenerator` at `src/optimization/bracket_portfolio.py` generates
  1000 brackets across 4 strategies
- Monte Carlo simulation with 50k iterations
- **BUT**: `enable_bracket_portfolio: False` by default — not wired into pipeline

### Problem
Since 2024, Kaggle allows submitting up to 100k brackets. This changes optimal
strategy from "best single prediction" to "portfolio that maximizes P(at least
one bracket finishes top-N)".

### Implementation

#### 5.1 Wire bracket portfolio into pipeline.run()
**File**: `src/pipeline/sota.py` — after Monte Carlo simulation

```python
if self.config.enable_bracket_portfolio:
    portfolio = self._generate_bracket_portfolio(bracket_sim, public_picks)
    report["bracket_portfolio"] = portfolio
```

#### 5.2 Champion diversity constraint
**File**: `src/optimization/bracket_portfolio.py`

Ensure the portfolio covers all viable champions (model P(champion) > 1%):

```python
def ensure_champion_coverage(brackets, viable_champions, min_coverage=0.02):
    """Ensure each viable champion appears in ≥ min_coverage fraction of brackets."""
    for champion in viable_champions:
        current_frac = sum(1 for b in brackets if b.champion == champion) / len(brackets)
        if current_frac < min_coverage:
            # Replace some low-EV brackets with champion-forced variants
            ...
```

#### 5.3 Anti-correlation scoring
Brackets in the portfolio should be **diverse**, not just individually optimal.
Measure portfolio diversity via:

```python
def portfolio_diversity(brackets):
    """Fraction of games where brackets disagree."""
    n = len(brackets)
    agreement = sum(
        sum(1 for b2 in brackets if b1.pick(game) == b2.pick(game))
        for b1 in brackets
        for game in all_games
    ) / (n * n * 63)
    return 1.0 - agreement  # Higher = more diverse
```

#### 5.4 Pool-size-adaptive strategy mix
**File**: `src/optimization/bracket_portfolio.py`

```python
POOL_SIZE_STRATEGIES = {
    "small":  {"chalk": 0.40, "balanced": 0.40, "contrarian": 0.15, "targeted": 0.05},
    "medium": {"chalk": 0.20, "balanced": 0.35, "contrarian": 0.30, "targeted": 0.15},
    "large":  {"chalk": 0.10, "balanced": 0.25, "contrarian": 0.35, "targeted": 0.30},
    "massive":{"chalk": 0.05, "balanced": 0.15, "contrarian": 0.40, "targeted": 0.40},
}
```

### Validation
- Backsim: Run portfolio strategy on 2018-2025 tournaments
- Metric: "best bracket in portfolio" rank in simulated ESPN pool

---

## Implementation Sequence

### Phase 1: Quick Wins (1-2 days) — Gaps #1, #7
1. Wire Massey composite into SOTAPipeline.run() as post-hoc blend
2. Add round-weighted Brier to calibration and evaluation
3. Add kaggle_dir to SOTAPipelineConfig and CLI
4. Validate on LOYO CV

### Phase 2: Primary Model Improvement (2-3 days) — Gaps #2, #3
5. Promote SpreadRegressor to primary model weight
6. Add tournament-specific σ calibration
7. Implement "simple mode" with killer-8 features
8. A/B test simple vs standard on LOYO CV

### Phase 3: Data & Women's (2-3 days) — Gaps #6, #4
9. Add data quality weights per era
10. Feature completeness validation
11. Kaggle data as authoritative backfill for pre-2010
12. Women's dedicated feature set and calibration

### Phase 4: Portfolio (2-3 days) — Gap #5
13. Wire bracket portfolio into pipeline
14. Champion diversity constraint
15. Anti-correlation scoring
16. Pool-size-adaptive strategy

### Phase 5: Integration Testing (1-2 days)
17. End-to-end LOYO CV with all changes
18. Compare flat Brier, round-weighted Brier, and bracket portfolio score
19. Final hyperparameter tuning (simple model should need minimal tuning)
20. Pre-registration freeze for 2026 submission

---

## Key Principle: The Paradox of Simplicity

Every decision should be evaluated against:
> "Would logistic regression on 5 features plus Massey composite beat this?"

If the answer is "probably yes" or "roughly equal", prefer the simpler approach.
The complexity ceiling hits at ~600 tournament samples. A well-calibrated
logistic regression with the right features is within 0.002 Brier of the
theoretical optimum for this sample size.

The path to top 1% is not more models — it's:
1. **Better features** (Massey composite = free lunch)
2. **Better calibration** (round-weighted, tournament-specific σ)
3. **Better portfolio strategy** (for bracket competitions)
4. **Better women's predictions** (50% of score since 2023)
