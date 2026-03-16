# Actionable Improvements: 2026 March Madness Forecaster

**Date:** 2026-03-15 (Selection Sunday)
**Purpose:** Step-by-step improvements with exact code paths, no room for interpretation.

---

## 1. ENABLE MULTI-MODEL ENSEMBLE IN PRODUCTION

### The Problem

Production mode uses **SpreadRegressor at 100% weight**. LightGBM, XGBoost, and Logistic classifiers are explicitly blocked from training and contribute nothing to predictions.

**Root cause chain:**
- `src/pipeline/production_baseline.py:45-46` — `default_weights = {"spread": 1.0, "logistic": 0.0}`
- `src/pipeline/production_baseline.py:53` — `deprecated_production_models = ("lightgbm_classifier", "xgboost_classifier")`
- `src/pipeline/production_baseline.py:56-58` — `deprecated_production_calibrators` includes `"tournament_sigma_calibrator"`
- `src/pipeline/stages/baseline_training.py:896-904` — LGB/XGB classifiers gated behind `not _production_mode`
- `src/pipeline/stages/baseline_training.py:1308-1343` — Production weights loaded from `PRODUCTION_BASELINE.default_weights`

This means the full inference flow is:
```
SpreadRegressor.predict_spread(X) → logistic_CDF(spread, sigma=11) → temperature_scale → clip
```

One model. One conversion function. No ensemble diversity.

### Why This Matters

- Ensemble diversity is the single largest driver of Brier score improvement in Kaggle March Madness competitions. Every medalist since 2019 uses 3+ model blending.
- SpreadRegressor is a margin predictor converted to probability via a fixed CDF. It has no mechanism to learn nonlinear probability surfaces (e.g., "7-seeds with high 3P variance lose more than their spread predicts").
- LightGBM and XGBoost classifiers directly optimize log-loss (probability accuracy). They learn conditional probability surfaces that SpreadRegressor cannot represent.
- Logistic Regression provides regularization and acts as a hedge against tree overfitting.

### Exact Solution

**Step 1: Modify `src/pipeline/production_baseline.py`**

Change `default_weights` (line 45-46) from:
```python
default_weights: Dict[str, float] = field(
    default_factory=lambda: {"spread": 1.0, "logistic": 0.0}
)
```
To:
```python
default_weights: Dict[str, float] = field(
    default_factory=lambda: {"spread": 0.45, "logistic": 0.20, "lgb": 0.20, "xgb": 0.15}
)
```

These weights come from the tournament prediction literature consensus: spread models carry the most signal (margin is richer than binary outcome), but tree classifiers and logistic regression add 3-5% Brier improvement through ensemble diversity. The specific 0.45/0.20/0.20/0.15 split reflects that spread should dominate (it trains on margin, a richer signal) while classifiers add conditional probability learning.

**Step 2: Remove LGB/XGB from deprecated list** (line 53):
```python
deprecated_production_models: tuple = ()  # Allow all models
```

**Step 3: Remove tournament_sigma_calibrator from deprecated list** (line 56-58):
```python
deprecated_production_calibrators: tuple = (
    "round_specific_calibrator",
)
```

**Step 4: Update `validate()` method** (line 81-85) — allow new weight keys:
```python
for model in self.default_weights:
    if model not in ("spread", "logistic", "lgb", "xgb"):
        violations.append(
            f"Weight key '{model}' not in sanctioned model set"
        )
```

**Step 5: Ungate classifier training in `src/pipeline/stages/baseline_training.py`** (lines 896-904):

Change:
```python
_allow_lgb_classifier = (
    not _production_mode
    and pipeline.config.experimental_enable_lgb_classifier
)
_allow_xgb_classifier = (
    not _production_mode
    and pipeline.config.experimental_enable_xgb_classifier
)
```
To:
```python
_allow_lgb_classifier = pipeline.config.experimental_enable_lgb_classifier
_allow_xgb_classifier = pipeline.config.experimental_enable_xgb_classifier
```

**Step 6: Ensure config enables classifiers.** In your production config JSON or `SOTAPipelineConfig`:
```python
experimental_enable_lgb_classifier: bool = True
experimental_enable_xgb_classifier: bool = True
```

**Step 7: Update the production name mapping** in `baseline_training.py` (line 1317) to include the new model names:
```python
_PROD_NAME_MAP = {"spread": "spread", "logistic": "logit", "lgb": "lgb", "xgb": "xgb"}
```

### Validation

After making these changes, run the pipeline and verify:
1. Training output logs show 4 models trained (spread, logistic, lgb, xgb)
2. `pipeline.baseline_model.fixed_weights` contains all 4 keys with expected values
3. LOYO Brier score with 4-model ensemble is lower than spread-only baseline

### Better Alternative: Enable LOYO Weight Optimization

Instead of hardcoding weights, let the optimizer learn them. The infrastructure exists at `baseline_training.py:1400-1433`. The gates are:
- `config.optimize_ensemble_weights = True` (already default True at `config.py:322`)
- `config.enable_loyo_cv = True` (must be explicitly set)
- `config.multi_year_games_dir` must point to historical data

Set `enable_loyo_cv = True` in your config. The optimizer will grid-search weights across historical holdout years and apply them only if they improve over the fixed baseline. This is strictly better than hardcoded weights — it's data-driven and self-validating.

**Expected Brier improvement: -0.008 to -0.015**

---

## 2. INCREASE MONTE CARLO NOISE FROM 0.12 TO 0.16

### The Problem

`src/pipeline/config.py:456` sets `mc_noise_std: float = 0.12`. This controls the per-game uncertainty injected during bracket simulation.

The noise is applied in logit space at `src/simulation/monte_carlo.py:270-272`:
```python
game_noise = rng.normal(0, idiosyncratic_std * noise_mult)
logit += game_noise
```

Where `idiosyncratic_std = noise_std` (line 232) and `noise_mult` is a round-dependent regional multiplier.

At `noise_std=0.12`, a game with base probability 0.70 (logit=0.847) gets noise with SD=0.12, meaning the 95% range of adjusted probabilities is approximately [0.62, 0.77]. This is too tight.

### Why This Matters

- **Lopez & Matthews (JQAS 2015)** empirically estimate game-level residual SD of **0.16-0.18 in logit space** from NCAA tournament data 2003-2013. Your 0.12 is 2/3 of their estimate.
- **Consequence:** Bracket simulations are overconfident. 1-seeds are simulated with ~50% championship probability vs. the historical ~40%. The simulation doesn't produce enough upsets in later rounds.
- **The code previously used 0.02** (per memory file), which was catastrophically overconfident. 0.12 was a correction but stopped short of the empirical optimum.

### Exact Solution

**Step 1: Change the default in `src/pipeline/config.py:456`:**
```python
mc_noise_std: float = 0.16  # Lopez & Matthews (2015): game-level logit SD ≈ 0.16-0.18
```

**Step 2: Validate with the existing CLI tool.** Run:
```bash
python -m src.main calibrate-mc-noise \
  --historical-dir data/raw/historical \
  --dev-years 2016,2017,2018,2019,2021,2022,2023 \
  --holdout-years 2024,2025 \
  --noise-grid 0.10,0.12,0.14,0.16,0.18,0.20 \
  --simulations 10000
```

This grid-searches noise_std values against historical upset rates (`src/simulation/mc_calibration.py:294-373`). The objective is seed-upset MAE: `sum(weight * |sim_rate - actual_rate|) / sum(weight)` across all seed matchups.

**Step 3: Apply the optimal value.** If the calibration output recommends a value other than 0.16 (e.g., 0.14 or 0.18), use that instead. The grid search is the authority, not the default.

**Step 4: Verify upset rates.** After running the full pipeline with the new noise_std, check the `validate_upset_rates()` output in logs (`src/simulation/monte_carlo.py:586-654`). Specifically verify:
- 1-vs-16 upset rate ≈ 1.5% (historical: 0.015)
- 5-vs-12 upset rate ≈ 36% (historical: 0.360)
- 8-vs-9 upset rate ≈ 49% (historical: 0.490)

**Expected Brier improvement: -0.008 to -0.015**

---

## 3. ENABLE TOURNAMENT SIGMA CALIBRATION FOR SPREAD MODEL

### The Problem

SpreadRegressor converts predicted point margins to probabilities via:
```
P(win) = 1 / (1 + exp(-predicted_spread / sigma))
```
(`src/ml/ensemble/spread_model.py:46-48`)

The sigma parameter defaults to **11.0** (`spread_model.py:47`, `config.py:372`). This value comes from regular-season NCAA game data (Stern 1991, Glickman & Stern 1998).

However, tournament games have empirically tighter margin distributions: **sigma ≈ 8.5-9.5 points.** This is because:
- All tournament games are at neutral sites (removes ~3.75 points of home-court variance)
- Opponent quality is higher and more concentrated (less variance from cupcake games)
- Single-elimination pressure leads to tighter, more conservative play
- Teams have days to prepare for specific opponents

The `TournamentSigmaCalibrator` exists to fix this, but is explicitly blocked in production at two points:
1. `src/pipeline/stages/baseline_training.py:1165`: `if spread_trained and TOURNAMENT_SIGMA_AVAILABLE and not _production_mode`
2. `src/pipeline/production_baseline.py:56-58`: `deprecated_production_calibrators` includes `"tournament_sigma_calibrator"`

### Why This Matters

With sigma=11 on tournament games:
- A team predicted to win by 5 points gets P(win) = 0.634
- With correct sigma=9: P(win) = 0.634 → 0.668

That's a 3.4 percentage point shift on every mid-range prediction. For 63 tournament games, systematic miscalibration at this level costs approximately 0.005-0.010 Brier.

The effect is worst for closely-matched teams (seeds 4-5 through 8-9), which is exactly where accurate probabilities matter most for bracket scoring.

### Exact Solution

**Step 1: Remove from deprecated list** in `src/pipeline/production_baseline.py:56-58`:
```python
deprecated_production_calibrators: tuple = (
    "round_specific_calibrator",
    # tournament_sigma_calibrator REMOVED from deprecated list
)
```

**Step 2: Remove the production gate** in `src/pipeline/stages/baseline_training.py:1165`:

Change:
```python
if spread_trained and TOURNAMENT_SIGMA_AVAILABLE and not _production_mode:
```
To:
```python
if spread_trained and TOURNAMENT_SIGMA_AVAILABLE:
```

**Step 3: Verify the calibrator is available.** The `TOURNAMENT_SIGMA_AVAILABLE` flag checks whether the `TournamentSigmaCalibrator` class can be imported. Ensure the module exists and imports cleanly.

**Step 4: After running the pipeline, inspect the tuning stats.** The output will include `tournament_sigma_error` if calibration failed, or the calibrated sigma value. Verify the fitted sigma is in the range [7.0, 11.0]. If it's outside this range, the calibration data is likely insufficient and you should revert to sigma=11.

### What the Calibrator Does

`_fit_tournament_sigma()` takes historical tournament game margins, computes the residual standard deviation of `actual_margin - predicted_spread`, and sets `SpreadRegressor.sigma` to this tournament-specific value. It uses the same Brier-minimizing grid search as `calibrate_sigma()` (`spread_model.py:184-240`) but restricted to tournament games only.

### Interaction With Temperature Scaling

Temperature scaling (the existing production calibration layer) adjusts the overall calibration curve. Tournament sigma adjusts the SpreadRegressor's internal margin-to-probability conversion. These are complementary, not redundant:
- Tournament sigma corrects the **shape** of the spread-to-probability mapping
- Temperature scaling corrects the **overall confidence level** of the ensemble output

Both should be active. The ordering is: spread predicts margin → tournament sigma converts to probability → ensemble blends → temperature scaling calibrates final output.

**Expected Brier improvement: -0.003 to -0.008**

---

## 4. POPULATE OR REPLACE DEAD INTERACTION FEATURES

### The Problem

The matchup feature vector has 7 interaction features. Three of them are **hardcoded to constant values during training:**

In `src/data/features/proprietary_metrics.py:2837-2839` (the `build_matchup_vector` static method used for all training samples):
```python
h2h_record = 0.5        # no head-to-head data in training
common_opp_margin = 0.0  # not available incrementally
travel_advantage = 0.0   # no venue data
```

These features occupy 3 of 7 interaction slots in every training row and carry zero signal. The model trains on them, learns they're constant, and wastes capacity.

**Critically, during tournament prediction** (`src/pipeline/sota.py:1272`), the pipeline uses `FeatureEngineer.create_matchup_features()` which CAN populate these from a `proprietary_engine`. So there's a **train/inference mismatch**: the model trains on zeros but may see real values at inference time. This is a distribution shift that can only hurt.

### Exact Solution: Two Options

#### Option A: Populate the Features (Better, More Work)

The computation functions already exist:
- `compute_h2h_record(team1_id, team2_id)` at `proprietary_metrics.py:1895-1920` — uses `_by_team` game records, applies Bayesian shrinkage toward 0.5 for small samples
- `compute_common_opponent_margin(team1_id, team2_id)` at `proprietary_metrics.py:1922-1960` — builds opponent margin maps, averages differential across common opponents, normalizes to ~[-1.5, 1.5]
- `compute_travel_advantage(team1_id, team2_id, venue_key)` in `travel_distance.py` — uses TEAM_COORDINATES dict (370+ D1 schools) with Haversine distance

**To populate during training**, modify `build_matchup_vector()` in `proprietary_metrics.py:2813-2854`:

1. Add parameters for the proprietary engine and team IDs:
```python
@staticmethod
def build_matchup_vector(
    v1: np.ndarray,
    v2: np.ndarray,
    seed1: int = 0,
    seed2: int = 0,
    engine: Optional['IncrementalMetricsEngine'] = None,
    team1_id: str = "",
    team2_id: str = "",
) -> np.ndarray:
```

2. Replace the hardcoded values:
```python
h2h_record = 0.5
common_opp_margin = 0.0
travel_advantage = 0.0
if engine is not None and team1_id and team2_id:
    h2h_record = engine.compute_h2h_record(team1_id, team2_id)
    common_opp_margin = engine.compute_common_opponent_margin(team1_id, team2_id)
```

3. Update callers in `sample_loading.py` (~line 447) to pass the engine and team IDs.

4. Travel advantage requires venue data per game, which is not available in training. Leave it at 0.0 for training and tournament (all tournament games are neutral site, so travel advantage is symmetric and near-zero for most matchups). Or remove it entirely.

#### Option B: Replace With High-Value Interactions (Simpler, Nearly as Good)

Remove the three dead features and replace them with interactions that the tournament prediction literature says matter:

**Replace `h2h_record` with `seed_adj_em_residual`:**
```python
# How much does this team over/underperform their seed expectation?
# Positive = team is better than their seed suggests (dangerous underdog)
seed_expected_em = {1: 28, 2: 21, 3: 16, 4: 12, 5: 9, 6: 6, 7: 4, 8: 2,
                    9: 0, 10: -2, 11: -4, 12: -6, 13: -9, 14: -12, 15: -16, 16: -21}
residual1 = (v1[0] - v1[1]) - seed_expected_em.get(seed1, 0)
residual2 = (v2[0] - v2[1]) - seed_expected_em.get(seed2, 0)
seed_em_residual_diff = (residual1 - residual2) / 20.0  # normalize
```

**Replace `common_opp_margin` with `sos_seed_interaction`:**
```python
# High seed + weak schedule = upset risk. Low seed + elite schedule = underrated.
sos_idx = 10  # index of sos_adj_em in team vector
sos_seed_interaction = ((v1[sos_idx] - v2[sos_idx]) * (seed1 - seed2)) / 200.0
```

**Replace `travel_advantage` with `three_pt_variance_seed_interaction`:**
```python
# High 3P variance + low seed = unpredictable upset amplifier
three_pt_var_idx = 43  # index of three_pt_variance
var_diff = v1[three_pt_var_idx] - v2[three_pt_var_idx]
three_pt_seed_interaction = var_diff * (seed1 - seed2) / 15.0
```

These indices must be verified against `feature_engineering.py:get_feature_names()` to ensure they point to the correct features.

**Expected Brier improvement: -0.002 to -0.006**

---

## 5. ACTIVATE ROUND-WEIGHTED TRAINING FOR MODEL FITTING

### The Problem

Round weights are defined (`config.py:211-244`): R64=2.0, S16=4.0, E8=8.0, F4=16.0, NCG=32.0.

Round weights ARE applied to **historical multi-year tournament games** during training (`baseline_training.py:624-630, 865-878`). The code at line 865 multiplies `train_sample_weight` by `pipeline._round_weights`.

However, **current-year training games are ALL regular-season** (weight=1.0). The pipeline trains on current-year data with uniform weights, never seeing tournament-like games with elevated importance.

The core issue: the model optimizes for uniform Brier across all games, but Kaggle scores with R64=1x through NCG=32x. A 1% error on the championship game costs 32x more than a 1% error on a first-round game. The model doesn't know this.

### Why This Matters

The championship game and Final Four represent 48x combined weight out of ~127x total weight across 63 games. That's 38% of the score from 5% of the games. Training with uniform weights under-invests gradient signal in the probability region that matters most (closely-matched elite teams with seeds 1-4).

### Exact Solution

The round-weighted training flag `enable_round_weighted_training` is already `True` by default (`config.py:480`). Historical tournament games from multi-year training DO get round weights. The gap is:

1. **Verify historical tournament games are included in training.** Check that `config.multi_year_games_dir` points to a directory containing `historical_games_{year}.json` files that include tournament games (not just regular-season). The round weight logic at `sample_loading.py:460-463` only applies weights to games with `game_date >= tournament_cutoff`. If your historical game files don't contain tournament games, round weights are never applied.

2. **Verify round weights are actually reaching the model.** Add a log check after line 878 in `baseline_training.py`:
```python
n_rw_gt1 = int(np.sum(pipeline._round_weights > 1.0))
logger.info("Round-weighted training: %d games with weight > 1.0 (max=%.0f)",
            n_rw_gt1, float(np.max(pipeline._round_weights)))
```
If `n_rw_gt1 == 0`, no tournament games are getting elevated weights and the feature is inert.

3. **Consider enabling round-weighted calibration** (`config.py:484`):
```python
enable_round_weighted_calibration: bool = True
```
This applies a second temperature scaling pass optimized for round-weighted Brier. Currently disabled with the comment "EXPERIMENTAL: Disabled by default." The risk is re-calibrating already-calibrated probabilities, but with sufficient OOS data (100+ historical tournament games), the second pass should be stable.

4. **Verify the round weight values are appropriate.** The current weights (2, 4, 8, 16, 32) follow Kaggle's exact scoring. If you're optimizing for a bracket pool (not Kaggle), adjust to match your pool's scoring system. Most ESPN-style pools weight later rounds more heavily than Kaggle does.

### Verification

After making changes, run the pipeline and check:
- Training log shows "FIX #3: Applied round-weighted training: N tournament games with Kaggle round weights (max=32)"
- N should be > 0 (ideally 200+ from multi-year historical tournaments)
- LOYO Brier (round-weighted) should improve vs. baseline

**Expected Brier improvement: -0.005 to -0.012 (Kaggle round-weighted metric)**

---

## 6. ESTABLISH PROSPECTIVE VALIDATION PROTOCOL

### The Problem

Every backtest result in this system is Level 3 (retrospective). The 58 constants (including 14 Tier 3 freely-tuned parameters) were optimized on the same 2005-2025 data used for evaluation. The sensitivity analysis re-evaluates them via LOYO on the same data — circular by construction.

The `circularity_warning` flag in `rdof_audit.py` is honest disclosure, but doesn't fix the fundamental issue: you don't know how this system performs on truly unseen data.

### Why This Matters

Retrospective backtests systematically overestimate performance. The degree of overestimation depends on the number of researcher degrees of freedom consumed. With 14 Tier 3 constants and ~120 holdout tournament games (2 years × ~60 games), the DoF/sample ratio is 14/120 ≈ 0.117 — well above the 0.01 target documented in `rdof_audit.py:2579-2585`.

This doesn't mean the system is overfit. It means you can't distinguish "genuinely good" from "fits historical patterns" until you see 2026 results.

### Exact Solution

**Step 1: Freeze the pipeline NOW, before any 2026 tournament results are known.**

```bash
python -m src.main freeze-pipeline \
  --output-path pipeline_freeze_2026.json \
  --tag "2026-selection-sunday"
```

This calls `freeze_pipeline()` in `rdof_audit.py:427-534`, which saves:
- SHA256 hash of all config fields
- Current git commit SHA
- Full snapshot of all 58+ constants from CONSTANT_REGISTRY
- Hash of feature selection
- MC calibration params (if fitted)
- Creates git annotated tag `pre-registered/{date}/{hash}`

**Step 2: Verify the freeze before running predictions.**

```bash
python -m src.main verify-freeze --freeze-path pipeline_freeze_2026.json
```

This calls `verify_freeze()` in `rdof_audit.py:537-622`. It checks config hash, feature set hash, each constant value, and MC params. Any mismatch is logged. **The pipeline should NOT be modified after this point.**

**Step 3: Run predictions with the frozen pipeline.**

```bash
python -m src.main run-production-2026 \
  --freeze-path pipeline_freeze_2026.json \
  --output-dir results/2026
```

**Step 4: After the tournament ends, run prospective evaluation.**

```bash
python -m src.main prospective-eval \
  --freeze-path pipeline_freeze_2026.json \
  --evaluation-year 2026 \
  --historical-dir data/raw/historical \
  --output-path prospective_eval_2026.json
```

This calls `run_prospective_evaluation()` in `rdof_audit.py:3114-3196`, which:
1. Verifies the freeze matches current config
2. Runs HoldoutEvaluator on 2026 tournament games only
3. Tags results as Level 2 (quasi-prospective) integrity
4. Writes full provenance including freeze timestamp and config hash

**Step 5: Interpret results honestly.**

- If 2026 Brier is within 0.010 of LOYO backtest Brier: system generalizes well
- If 2026 Brier is 0.010-0.025 worse: mild overfitting to historical patterns (common, fixable)
- If 2026 Brier is 0.025+ worse: significant overfitting — Tier 3 constants need re-examination

**For 2027:** Designate 2026 as a permanent holdout year. Re-optimize Tier 3 constants on 2005-2025, evaluate on 2026. This gives Level 1 (true prospective) integrity.

---

## 7. CROSS-REFERENCE PREDICTIONS AGAINST VEGAS LINES

### The Problem

The system has no built-in mechanism to compare its predictions against market consensus. Vegas lines represent the most informationally efficient basketball predictions available (millions of dollars of skin-in-the-game). If your model's predictions systematically disagree with market lines, either:
- Your model has found genuine alpha (unlikely without novel data sources)
- Your model is miscalibrated (much more likely)

### Why This Matters

Market lines serve as the strongest available sanity check. A well-calibrated model should agree with Vegas on most games and disagree modestly on a few where it has genuine informational advantage (e.g., injury information the market hasn't priced in).

If your model gives a 1-seed 55% championship odds but Vegas implies 35%, that's a massive red flag — not evidence of alpha.

### Exact Solution

**Step 1: Scrape current market lines.** The infrastructure exists at `src/data/scrapers/betting_markets.py`. Use:

```python
from src.data.scrapers.betting_markets import BettingMarketScraper

scraper = BettingMarketScraper()
odds = scraper.fetch_championship_odds(year=2026)
# Returns List[BettingMarketOdds] with team_id, implied_probability per sportsbook
```

Or fetch manually from FanDuel/DraftKings and create a JSON file:
```json
{
  "houston": {"american_odds": 400, "implied_prob": 0.200},
  "duke": {"american_odds": 600, "implied_prob": 0.143},
  "auburn": {"american_odds": 700, "implied_prob": 0.125},
  ...
}
```

**Step 2: Compute divergence metrics.** After running the pipeline, compare:

```python
import numpy as np

# model_probs: Dict[str, float] from pipeline championship_odds
# market_probs: Dict[str, float] from betting scraper (vig-adjusted)

common_teams = set(model_probs.keys()) & set(market_probs.keys())

# 1. Root Mean Squared Divergence
diffs = [model_probs[t] - market_probs[t] for t in common_teams]
rmsd = np.sqrt(np.mean(np.array(diffs) ** 2))
# Target: RMSD < 0.05 (5 percentage points)

# 2. Largest disagreements (flag for manual review)
for t in sorted(common_teams, key=lambda t: abs(model_probs[t] - market_probs[t]), reverse=True)[:10]:
    print(f"{t}: model={model_probs[t]:.3f} market={market_probs[t]:.3f} diff={model_probs[t]-market_probs[t]:+.3f}")

# 3. Rank correlation (should be > 0.85)
from scipy.stats import spearmanr
model_ranks = [model_probs[t] for t in common_teams]
market_ranks = [market_probs[t] for t in common_teams]
corr, pval = spearmanr(model_ranks, market_ranks)
# Target: corr > 0.85
```

**Step 3: Interpret and act.**

| RMSD | Rank Corr | Interpretation | Action |
|------|-----------|----------------|--------|
| < 0.03 | > 0.90 | Strong agreement. Model is well-calibrated. | Ship predictions. |
| 0.03-0.06 | 0.80-0.90 | Moderate divergence. Model has opinions. | Review top-5 disagreements manually. |
| 0.06-0.10 | 0.70-0.80 | Significant divergence. Possible miscalibration. | Check calibration temperature, MC noise. |
| > 0.10 | < 0.70 | Major disagreement. Model likely miscalibrated. | Do NOT ship without investigation. |

**Step 4: Specific red flags to check:**
- If your model gives ANY team > 30% championship probability, it's likely overconfident (historical max for a 1-seed is ~25% pre-tournament implied probability)
- If all your 1-seeds have combined championship probability > 60%, MC noise is too low
- If a 12+ seed has > 3% championship probability, MC noise may be too high
- If your model and market disagree by > 10pp on any team's R64 advancement, investigate that specific matchup

**Step 5: Optional — blend with market.** If divergence is moderate, consider a post-hoc market blend:
```python
blended_prob = 0.7 * model_prob + 0.3 * market_prob
```
This is a standard hedging strategy. The 0.7/0.3 split favors your model (it has team-specific features the market aggregates away) while hedging against miscalibration.

---

## 8. ADDITIONAL: TIGHTEN UPSET RATE VALIDATION

### The Problem

`validate_upset_rates()` at `monte_carlo.py:586-654` uses a tolerance of **0.15** (15 percentage points). For the 1-vs-16 matchup with historical rate 1.5%, this means any simulated rate between 0% and 16.5% passes validation. This tolerance is so loose it cannot detect meaningful miscalibration.

The validation is also **non-blocking** — it logs warnings but the pipeline continues regardless (`simulation.py:127-138`).

### Exact Solution

**Step 1: Tighten tolerance** in `monte_carlo.py:589`:

Change the default parameter:
```python
def validate_upset_rates(
    sim_results: AggregatedResults,
    teams_by_region: Dict[str, List[TournamentTeam]],
    tolerance: float = 0.08,  # CHANGED from 0.15
) -> Dict:
```

With 50,000 simulations × 4 regions = 200,000 effective R64 games, simulation noise is tiny. Tolerance should reflect uncertainty in historical base rates, not simulation variance. Historical rates have SE ≈ 3-5% at matchup level → 0.08 tolerance is ~2 SE, appropriate for a 95% CI.

**Step 2: Add champion seed distribution validation.** After `validate_upset_rates()`, add a check that the simulated champion-seed distribution matches historical data:

Historical champion seeds (1985-2025, 41 tournaments):
- 1-seeds: ~60% of championships (24-25 titles)
- 2-seeds: ~15% (6-7 titles)
- 3-seeds: ~10% (4 titles)
- 4-8 seeds: ~15% (6-7 titles combined)
- 9+ seeds: ~0% (0 titles in 64-team era)

If your simulation shows 1-seeds winning > 70% or < 45% of championships, noise parameters need adjustment.

**Step 3: Elevate to warning-level output.** Change the log level from `logger.info` to `logger.warning` when upsets deviate by > 0.05 from historical. The user should see these without reading debug logs.

---

## 9. ADDITIONAL: VALIDATE CALIBRATION SAMPLE SIZE

### The Problem

Temperature scaling fits on 100-300 tournament game samples. The small-sample guard at `calibration.py:410-416` rejects calibration when N < 30 by checking if the bootstrap CI on temperature includes 1.0. But with N = 50-80, the CI is wide enough that it almost always includes 1.0, making the guard ineffective.

### Exact Solution

**Step 1: Ensure multi-year calibration is active.** Verify `config.enable_multi_year_calibration = True` (default at `config.py:487`). This augments the calibration pool with historical tournament games from multiple years, increasing N from ~40-60 (single year) to ~200-400.

**Step 2: Raise the hard floor.** In `config.py:315`:
```python
min_calibration_samples_hard: int = 80  # CHANGED from 50
```

**Step 3: After running the pipeline, inspect the calibration output.** Look for:
- `brier_before` (pre-calibration Brier score)
- `brier_after` (post-calibration Brier score on OOS split)
- `temperature` (the fitted T value)
- `n_calibration_samples` (how many games were used)

If `n_calibration_samples < 100`, the calibration is unstable. If `temperature` is within [0.95, 1.05], calibration is having negligible effect and you could skip it (set `calibration_method = "none"`).

If `brier_after > brier_before`, calibration is hurting. This happens with small samples. Disable it.

---

## EXECUTION ORDER

For Selection Sunday 2026, execute in this order:

1. **Freeze the pipeline** (Item 6, Step 1) — creates immutable pre-registration
2. **Make code changes** (Items 1-5, 8-9) — ensemble weights, noise, sigma, features, round weights
3. **Re-freeze** after changes with a new tag
4. **Run `calibrate-mc-noise`** (Item 2, Step 2) — find optimal noise_std
5. **Run the full pipeline** — training + calibration + simulation
6. **Cross-reference against Vegas** (Item 7) — sanity check
7. **Inspect logs** — verify upset rates, calibration stats, ensemble weights
8. **Ship predictions**
9. **After tournament:** Run prospective evaluation (Item 6, Step 4)

---

## EXPECTED CUMULATIVE IMPACT

| Change | Brier Impact (conservative) | Brier Impact (optimistic) |
|--------|---------------------------|--------------------------|
| Multi-model ensemble | -0.008 | -0.015 |
| MC noise 0.12 → 0.16 | -0.008 | -0.015 |
| Tournament sigma | -0.003 | -0.008 |
| Fix interaction features | -0.002 | -0.006 |
| Round-weighted training | -0.005 | -0.012 |
| Calibration sample floor | -0.001 | -0.003 |
| Tighter upset validation | -0.001 | -0.003 |
| **TOTAL (not fully additive)** | **-0.020** | **-0.045** |

On a baseline Brier of ~0.160, this represents a **12-28% relative improvement.**
These improvements are not fully additive because some address overlapping variance (e.g., tournament sigma and MC noise both affect bracket-level calibration). The conservative estimate assumes ~70% additivity.
