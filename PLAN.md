# Plan: Top 1% Kaggle March Machine Learning Mania 2026

## Executive Summary

Transform this men's-only forecasting system into a competition-dominating dual-tournament pipeline that targets **both** the pairwise Brier score leaderboard **and** the bracket portfolio format. Six workstreams, ordered by expected leaderboard impact.

---

## Workstream 1: Women's Tournament Full Pipeline (CRITICAL — ~50% of evaluation)

The Kaggle sample submission contains rows for both men's (TeamIDs 1100-1499) and women's (TeamIDs 3000+). Currently every women's row returns 0.5, which alone guarantees a bottom-half finish.

### 1A: Women's Data Ingestion (`src/data/scrapers/womens/`)

**New files:**
- `src/data/scrapers/womens/__init__.py`
- `src/data/scrapers/womens/herhoopstats.py` — Scrape Her Hoop Stats (equivalent of Torvik for women's basketball). Four Factors, adjusted efficiency, SOS, tempo for all ~360 D1 women's teams.
- `src/data/scrapers/womens/ncaa_net.py` — Scrape NCAA NET rankings (the selection committee's official ranking). Publicly available, high signal.
- `src/data/scrapers/womens/warren_nolan_w.py` — Warren Nolan women's data (SOS, RPI, record).
- `src/data/scrapers/womens/historical_results.py` — Scrape/load historical women's tournament results (2015-2025) for calibration and backtesting. Source: sports-reference.com women's bracket pages.

**Data to collect per team:**
- Adjusted offensive/defensive efficiency
- Tempo, SOS, Four Factors (eFG%, TO%, ORB%, FT rate)
- Win/loss record, conference record
- NCAA NET ranking
- Elo rating (computed from game results)
- Seed (from bracket data)
- Historical tournament game results for calibration

### 1B: Women's Team Name Resolver

**Modified file:** `src/data/team_name_resolver.py`

- Add `_WOMENS_ALIASES` mapping (separate from men's — some school names differ, e.g., "UConn Huskies" vs "Connecticut")
- Add method `resolve_womens(name: str) -> ResolveResult`
- Extend `build_team_id_map()` in `src/exports/kaggle.py` to handle women's TeamIDs (3000+) using `WTeams.csv`

### 1C: Women's Feature Engineering

**New file:** `src/data/features/womens_feature_engineering.py`

- Mirror `TeamFeatures` structure but with women's-specific adjustments:
  - Use the **same FIXED_FEATURE_SET** differential features (basketball physics are gender-neutral)
  - Women's tournament has fewer upsets historically → different calibration curve
  - Fewer 3-point attempts historically → `three_pt_variance` may have different signal
- Compute `WomensTeamFeatures` from scraped data
- Build matchup differential vectors identically to men's pipeline

### 1D: Women's Model Training

**Modified file:** `src/pipeline/sota.py`

- Add `WomensSOTAPipeline` class (or extend `SOTAPipeline` with `gender` parameter)
- Train separate LightGBM + XGBoost ensemble on women's historical tournament data
- Key difference from men's: women's tournament historically has **fewer upsets** (1-seeds have ~99% R64 win rate vs ~98% for men), so the model should be more confident in seed-based predictions
- Use same hyperparameter tuning framework but with women's training data
- If insufficient women's historical data for GBM: fall back to **seed-based logistic regression** calibrated on women's tournament history (still strong baseline — seed alone explains ~75% of variance)

### 1E: Unified Kaggle Export

**Modified files:** `src/exports/kaggle.py`, `src/main.py`

- `generate_predictions()` already handles any TeamID format
- Add routing logic: if `team_id >= 3000`, use women's pipeline; else men's
- `run_kaggle_export()` loads both `MTeams.csv` and `WTeams.csv`
- Single submission CSV contains both men's and women's predictions

---

## Workstream 2: Brier Score Optimization (Primary Scoring Metric)

The competition uses Brier score: `(1/N) * Σ(predicted - actual)²`. Unlike log loss, Brier is **bounded** and rewards confident correct predictions more. The current pipeline optimizes log loss internally.

### 2A: Switch Scoring Metric Throughout Pipeline

**Modified file:** `src/pipeline/sota.py`

- Change `scoring_metric` default from `"logloss"` to `"brier"` in `SOTAPipelineConfig`
- Ensure hyperparameter tuning (Optuna) optimizes Brier score in cross-validation
- Ensure ensemble weight optimization minimizes Brier score

### 2B: Brier-Optimal Probability Sharpening

**New file:** `src/ml/calibration/brier_optimal.py`

Key insight: Under Brier scoring, the **optimal prediction** is the true probability. But when we're confident our model is better than 0.5, sharpening predictions (pushing them away from 0.5 toward 0/1) is profitable if our model has good discrimination.

- `BrierOptimalSharpener`:
  - Given a raw probability p, apply a power transform: `p_sharp = sign(p-0.5) * |2p-1|^α / 2 + 0.5`
  - α < 1 sharpens (pushes away from 0.5), α > 1 softens
  - Optimize α on held-out Brier score via cross-validation
  - Constrain α ∈ [0.5, 1.5] to prevent extreme distortion

- `SeedBasedOverrides`:
  - For extreme seed matchups (1v16, 2v15), historical win rates are well-established:
    - 1v16: 98.7% (men), ~99.3% (women)
    - 2v15: 94.4% (men), ~96% (women)
  - When the model's prediction is within ±5% of the historical rate, snap to the historical rate
  - These overrides are Brier-profitable because they're backed by N>150 observations

### 2C: Probability Clipping Strategy

**Modified file:** `src/pipeline/sota.py`

- Current clipping: [0.01, 0.99]
- Under Brier: wider bounds are better (Brier penalty is quadratic, not logarithmic)
- Change to [0.005, 0.995] for Brier optimization
- Add configurable `brier_clip_lo` and `brier_clip_hi` to SOTAPipelineConfig

### 2D: Calibration with Brier Objective

**Modified file:** `src/ml/calibration/calibration.py`

- Add `BrierCalibrator` that fits temperature scaling by minimizing Brier score (currently minimizes NLL)
- The Brier-optimal temperature differs from NLL-optimal: Brier penalizes probabilities near the boundary less than log loss does
- Use this as the post-processing calibrator when `scoring_metric = "brier"`

---

## Workstream 3: External Rating Integration

Top 1% Kaggle solutions consistently integrate external rating systems. The 2023 gold solution used 10 historically-accurate external ratings.

### 3A: External Rating Scraper

**New file:** `src/data/scrapers/external_ratings.py`

Scrape/ingest the following proven rating systems:
- **KenPom** (already partially covered via Torvik as proxy) — verify coverage
- **Sagarin** ratings (from USA Today)
- **Massey Composite** (meta-ranking of 100+ rating systems)
- **TeamRankings.com** predicted probabilities
- **ESPN BPI** (Basketball Power Index)
- **NCAA NET** rankings

Each scraper returns: `{team_name: rating_value}` normalized to a common scale.

### 3B: External Rating Features

**Modified files:** `src/data/features/feature_engineering.py`, `src/pipeline/sota.py`

- Add to `TeamFeatures`: `external_rating_composite`, `external_rating_spread`
- `external_rating_composite` = weighted average of available external ratings (weights from historical predictive accuracy)
- `external_rating_spread` = max - min across rating systems (disagreement signal)
- Add `diff_external_rating_composite` and `diff_external_rating_spread` to FIXED_FEATURE_SET
- These are powerful because they incorporate information our model doesn't see (private box-score data, coach tendencies, etc.)

### 3C: Massey Composite as Meta-Feature

The Massey Composite already averages 100+ rating systems. Adding `diff_massey_composite_rank` as a single feature effectively gives us an ensemble of ensembles.

---

## Workstream 4: Bracket Portfolio Optimization (2024+ Format)

The competition allows submitting 1-100,000 brackets. This is a fundamentally different optimization problem than pairwise predictions.

### 4A: Portfolio Generator

**New file:** `src/optimization/bracket_portfolio.py`

- `BracketPortfolioGenerator`:
  - Input: pairwise win probabilities for all 68 teams
  - Use Monte Carlo simulation (existing `monte_carlo.py`) to sample tournament outcomes
  - For each simulation, generate the optimal bracket (pick the winner of each game based on the simulation's random draws)
  - Generate N=10,000+ diverse brackets via the simulation

- **Diversity maximization**: Don't just submit the modal bracket N times. Use stratified sampling:
  - Cluster simulations by champion identity
  - Within each cluster, pick representatives that differ in Final Four, Elite Eight, etc.
  - Weight bracket representation by champion probability

### 4B: Anti-Correlation Strategy

**New file:** `src/optimization/anti_correlation.py`

- `AntiCorrelatedPortfolio`:
  - Key insight from past winners: In portfolio format, you want brackets that are **anti-correlated** with what the public picks
  - When your model says Team A has 20% chance but public picks them at 5%, over-represent Team A as champion in your portfolio
  - Formalize as: `portfolio_weight(team) ∝ P_model(team wins) / P_public(team wins)`
  - This is the leverage ratio from `src/optimization/leverage.py`, extended to full bracket portfolios

### 4C: Expected Prize Optimization

- Instead of minimizing expected Brier score, optimize for **expected prize money**
- Prize money is nonlinear: top 3 finishes pay most, so maximize P(top 3) not E[rank]
- This means taking calculated risks in the portfolio — some brackets should bet heavily on unlikely outcomes

---

## Workstream 5: Dual Submission Meta-Strategy

### 5A: Dual Submission Generator

**New file:** `src/optimization/dual_submission.py`

- `DualSubmissionStrategy`:
  - For pairwise probability submission, generate TWO submissions:
    1. **Primary**: Best-calibrated probabilities (minimize expected Brier)
    2. **Hedge**: Same probabilities except for 1-3 high-leverage games where we take a contrarian position
  - The hedge submission has higher variance but captures prize probability that the primary misses

  - In practice: pick the game with highest `|P_model - P_public|` and submit one version with our probability and another with a more extreme version

### 5B: Opponent Modeling

**New file:** `src/optimization/opponent_modeling.py`

- `OpponentModel`:
  - Estimate the distribution of competitor submissions
  - Most competitors submit seed-based or KenPom-based predictions
  - Model the "crowd median" prediction for each matchup
  - Our predictions should deviate from the crowd where we have genuine edge, and match the crowd elsewhere (to avoid unnecessary variance)

  - Data source: Kaggle public notebooks (many share their submissions) → scrape common approaches to estimate crowd distribution
  - Historical data: Past Kaggle March Mania leaderboard statistics

---

## Workstream 6: Pipeline Hardening & Validation

### 6A: End-to-End Backtesting Framework

**New file:** `src/ml/evaluation/kaggle_backtest.py`

- `KaggleBacktester`:
  - For each historical year (2015-2025, excluding 2020):
    1. Train model using only data available before tournament
    2. Generate pairwise probabilities for all possible matchups
    3. Score against actual tournament results using Brier score
    4. Compare to baselines: seed-only, KenPom-only, 538, public
  - Report: per-year Brier score, aggregate Brier, variance, percentile rank vs historical Kaggle leaderboards
  - This tells us exactly where we'd rank historically

### 6B: Submission Validation

**Modified file:** `src/exports/kaggle.py`

- Add `validate_submission(df: pd.DataFrame) -> List[str]`:
  - Check: all required matchup IDs present (men's + women's)
  - Check: all probabilities in [0.005, 0.995]
  - Check: complementarity (P(A beats B) + P(B beats A) ≈ 1.0)
  - Check: no NaN/inf values
  - Check: row count matches sample submission exactly
  - Report warnings for any prediction == 0.5 (unmapped teams)

### 6C: Cross-Validation Protocol

**Modified file:** `src/pipeline/sota.py`

- Implement proper **temporal cross-validation** for Brier-scored evaluation:
  - Train on years [t-N, ..., t-1], validate on year t
  - Report both in-sample and out-of-sample Brier scores
  - Flag if OOS Brier > in-sample Brier by more than 0.02 (overfitting signal)

---

## Implementation Order (Priority)

| Phase | Workstream | Impact | Effort | Deadline |
|-------|-----------|--------|--------|----------|
| **1** | WS1: Women's Pipeline | CRITICAL (50% of eval) | High | Mar 10 |
| **2** | WS2: Brier Optimization | High (correct metric) | Medium | Mar 12 |
| **3** | WS6: Validation & Backtesting | High (confidence) | Medium | Mar 14 |
| **4** | WS3: External Ratings | Medium-High | Medium | Mar 16 |
| **5** | WS4: Bracket Portfolio | Medium (new format) | High | Mar 18 |
| **6** | WS5: Dual Submission | Low-Medium | Low | Mar 19 |

## Files to Create (11 new files)

1. `src/data/scrapers/womens/__init__.py`
2. `src/data/scrapers/womens/herhoopstats.py`
3. `src/data/scrapers/womens/ncaa_net.py`
4. `src/data/scrapers/womens/warren_nolan_w.py`
5. `src/data/scrapers/womens/historical_results.py`
6. `src/data/features/womens_feature_engineering.py`
7. `src/ml/calibration/brier_optimal.py`
8. `src/data/scrapers/external_ratings.py`
9. `src/optimization/bracket_portfolio.py`
10. `src/optimization/anti_correlation.py`
11. `src/optimization/dual_submission.py`
12. `src/optimization/opponent_modeling.py`
13. `src/ml/evaluation/kaggle_backtest.py`

## Files to Modify (7 existing files)

1. `src/pipeline/sota.py` — Add women's pipeline support, Brier scoring default, external rating features
2. `src/exports/kaggle.py` — Dual-gender export, submission validation, WTeams.csv loading
3. `src/main.py` — New CLI commands for women's pipeline, bracket portfolio, backtesting
4. `src/data/team_name_resolver.py` — Women's team aliases
5. `src/data/features/feature_engineering.py` — External rating features in TeamFeatures
6. `src/ml/calibration/calibration.py` — Brier-optimized calibrator
7. `src/simulation/monte_carlo.py` — Portfolio-aware simulation output

## Expected Outcome

With all 6 workstreams implemented:
- **Women's coverage**: 0% → 100% of women's matchups predicted (eliminates ~50% dead weight)
- **Brier optimization**: 0.02-0.03 Brier improvement from correct metric optimization
- **External ratings**: 0.01-0.02 Brier improvement from orthogonal information
- **Bracket portfolio**: Maximizes P(top finish) in portfolio format
- **Combined**: Historical Brier scores of 0.41-0.42 are top-5 territory; our target is 0.40-0.41

Top 1% historically has Brier score around 0.42-0.43 for the combined men's+women's leaderboard. With a well-calibrated model covering both tournaments, 0.41 is achievable.
