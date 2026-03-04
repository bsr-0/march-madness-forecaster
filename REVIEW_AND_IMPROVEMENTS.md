# March Madness Forecaster: Kaggle Grandmaster Review

## 1. Methodology Review

### Strengths (What This Codebase Does Well)

**Architecture**: This is a genuinely sophisticated system — far beyond what 90%+ of Kaggle competitors submit. The multi-model ensemble (LightGBM + XGBoost + Logistic + SpreadRegressor), Massey composite blending, and fixed-weight averaging are all sound choices. The awareness of overfitting on ~400-600 tournament samples is evident throughout the code, with stacking disabled by default, feature selection disabled, and Optuna trials capped at 15.

**Leakage Prevention**: The point-in-time feature construction, late-season training cutoff, and tournament game exclusion from training are all correct. This alone puts the codebase ahead of many competitors who unknowingly use future information.

**Calibration Awareness**: Temperature scaling as default (over isotonic/Platt) for small samples is the right call. The switch to Brier-score optimization with wider probability bounds [0.005, 0.995] is correct for the post-2023 metric.

**Domain Knowledge Feature Set**: The FIXED_FEATURE_SET with published citations (KenPom, Oliver's Four Factors, FiveThirtyEight Elo) is well-motivated. The "simple" mode with 9 features is particularly smart — every recent winner used a small feature set.

**Massey Composite Integration**: Blending Massey Ordinals composite (100+ rating systems averaged) is the single most important design decision. Every recent top-1% finisher used external rating composites. The fitted sigma and blend weight calibration is correct.

### Critical Weaknesses

**1. Over-Engineering Despite Self-Awareness**
The codebase is ~15,000+ lines with GNN, Transformer, CFA, Monte Carlo simulation, bracket portfolio optimization, and game theory modules. Despite most being "disabled by default," this complexity creates maintenance burden, introduces subtle bugs, and — most critically — represents researcher degrees of freedom (RDoF) that inflate apparent performance. The `model_complexity: "simple"` mode is the right instinct, but the infrastructure surrounding it adds noise to the development process.

**2. Training Data Mismatch**
The model trains primarily on regular-season games but predicts tournament games. Tournament games have fundamentally different characteristics: neutral sites, single-elimination pressure, higher average opponent quality, and different pace dynamics. The `tournament_shrinkage: 0.02` is far too small to account for this domain shift. Top competitors either train exclusively on historical tournament games or apply much stronger domain adaptation.

**3. Ensemble Weight Allocation**
The fixed weights (`spread: 0.40, lgb: 0.25, xgb: 0.15, logit: 0.20`) give SpreadRegressor 40% weight based on the "raddar" benchmark insight. However, this weight was set by hand, not validated on held-out tournament data. The spread model's sigma (11 points) is calibrated from regular-season data, but tournament game spreads have different distributions (tighter, more competitive matchups on average).

**4. Missing Key Signals That Winners Use**
- **No favorite-longshot bias correction**: Tournament favorites are systematically overvalued in raw model probabilities. Top competitors apply explicit corrections (e.g., the "goto_conversion" approach).
- **No meta-strategy optimization**: The 2017 winner (Landgraf) showed that optimizing for placement probability rather than pure prediction accuracy can improve top-5 finish probability by 2 percentage points. This codebase optimizes Brier score, not competition placement.
- **No opponent-adjusted tournament features**: Features like "performance against top-25 teams" or "road game win rate" are highly predictive in tournament context but absent here.
- **No coaching tournament experience as a feature**: Coach tournament win rate is mentioned in the scraper but doesn't appear in the active feature set.

**5. Women's Tournament Treatment**
The women's pipeline uses `seed_prior_weight: 0.50` and `model_complexity: "simple"` — which is directionally correct (women's bracket is more seed-predictable). However, 50% of the Kaggle evaluation comes from women's games, and the current approach is essentially a seed-based model with minimal sophistication. Top competitors build separate full models for women's.

**6. Probability Clipping Too Tight**
The pre-calibration clip of [0.005, 0.995] is reasonable, but for Brier score optimization, the research suggests that slightly more aggressive probabilities (wider bounds) are actually optimal. The 2023 gold winner noted that Brier score rewards well-calibrated confidence more than log-loss does.

---

## 2. Estimated Kaggle Percentile

### Performance Estimate: **Top 10-15%** (75th-85th percentile)

**Reasoning:**

| Factor | Impact | Score vs Field |
|--------|--------|---------------|
| Massey composite integration | Very strong | Top 15% |
| Multi-model ensemble | Strong | Top 20% |
| Proper calibration | Strong | Top 25% |
| Leakage prevention | Above average | Top 30% |
| Feature engineering depth | Strong | Top 15% |
| Women's pipeline quality | Weak | Bottom 50% |
| Meta-strategy optimization | Missing | Average |
| Domain adaptation | Weak | Average |
| Tournament-specific training | Weak | Below average |

**Why not top 5%:**
1. Women's bracket (50% of score) is under-optimized
2. No meta-strategy for competition placement optimization
3. Regular-season training without strong tournament domain adaptation
4. Missing key signals (favorite-longshot bias, coaching experience, opponent-quality features)
5. Over-complex architecture introduces subtle RDoF leakage despite safeguards

**Why not average (50th percentile):**
1. Massey composite alone puts you above median
2. Proper calibration and Brier-score awareness
3. Multi-year training with temporal decay
4. SpreadRegressor as primary model (MOV-first approach)

---

## 3. Roadmap to Top 1%

### Priority 1: Fix Women's Pipeline (Expected improvement: 0.01-0.02 Brier)

This is the single highest-ROI change because 50% of your Kaggle score comes from women's games.

**Changes needed:**
- Build a separate full feature pipeline for women's teams using women's-specific data
- Use women's KenPom/HerHoopStats equivalents for efficiency metrics
- Maintain higher seed prior (women's is more seed-predictable) but add efficiency features on top
- Train separate Massey composite sigma for women's (different rating landscape)

### Priority 2: Tournament-Domain Training (Expected improvement: 0.005-0.015 Brier)

**Changes needed:**
- Train a separate "tournament model" on historical tournament games only (2003-2025, ~1,400 games)
- Blend tournament-trained model with regular-season model (weight ~0.3 tournament)
- Add tournament-specific features: seed matchup historical win rates, conference tournament performance, late-season momentum (last 10 games)
- Increase tournament_shrinkage to 0.05-0.08 for the regular-season model component

### Priority 3: Favorite-Longshot Bias Correction (Expected improvement: 0.003-0.008 Brier)

**Changes needed:**
- Implement empirical favorite-longshot correction based on historical seed matchup data
- For each seed pairing (e.g., 1v16, 2v15, ..., 8v9), compute historical win rates
- Blend raw model probability toward historical seed-matchup rate with weight ~0.15-0.25
- This is different from seed_prior_weight — it's a post-hoc correction, not a feature

### Priority 4: Simplify to "Raddar-Style" Core (Expected improvement: 0.005-0.010 Brier)

The legendary "raddar" submission dominated 2018-2024 with a remarkably simple approach:
- Predict point margin (not win probability directly)
- Convert margin to probability via logistic CDF
- Use ~10-15 features from publicly available rating systems
- Extremely well-calibrated

**Changes needed:**
- Make SpreadRegressor the *only* primary model (not 40% weight in ensemble)
- Feed it Massey composite + KenPom efficiency + seed + Elo + SOS (5-8 features)
- Calibrate sigma per round (later rounds have tighter margins)
- Use the ensemble only as a secondary check, not primary

### Priority 5: Meta-Strategy for Competition Placement (Expected improvement: variable, high-variance)

**Changes needed:**
- Implement Landgraf-style competition simulation
- Model the distribution of competitor submissions
- For games where your model predicts 45-55%, strategically push toward 30/70 or 70/30
- Generate two submissions: one conservative (minimize expected Brier), one aggressive (maximize top-10 probability)
- The aggressive submission sacrifices expected score for higher variance — optimal when you need to beat 1000+ competitors

### Priority 6: Ensemble Simplification (Expected improvement: 0.002-0.005 Brier)

**Changes needed:**
- Remove GNN, Transformer, CFA entirely (not just disable)
- Use exactly 3 models: SpreadRegressor (primary), Logistic Regression (regularization), Massey composite (external signal)
- Validate weights on historical tournament games using LOYO
- The simpler the ensemble, the less RDoF leakage

### Priority 7: Round-Weighted Optimization (Expected improvement: 0.001-0.003 Brier)

**Changes needed:**
- The KAGGLE_ROUND_WEIGHTS are already defined but need deeper integration
- Train separate sub-models for early rounds (R64/R32) vs late rounds (S16+)
- Early rounds: seed matchup features dominate
- Late rounds: efficiency differentials dominate
- Blend sub-model predictions by round

---

## 4. Implementation Plan (Ordered by Impact)

### Phase 1: Quick Wins (1-2 days)
1. **Increase tournament_shrinkage** from 0.02 to 0.06
2. **Add historical seed-matchup win rates** as a post-hoc blend (0.20 weight)
3. **Increase spread model weight** to 0.55, reduce LGB to 0.15, logistic to 0.15, XGB to 0.15
4. **Add coaching tournament experience** to FIXED_FEATURE_SET
5. **Add opponent-quality features**: win% vs top-25, road game win%

### Phase 2: Women's Pipeline Overhaul (2-3 days)
1. Build WomensFeatureEngineer with women's-specific data sources
2. Train separate Massey sigma for women's
3. Add women's-specific features (rebounding differential matters more in women's)
4. Validate on 2023-2025 women's tournament results

### Phase 3: Tournament Domain Model (2-3 days)
1. Collect all historical tournament games (2003-2025)
2. Train SpreadRegressor on tournament games only
3. Blend tournament model (0.3) with regular-season model (0.7)
4. Calibrate per-round sigma values

### Phase 4: Meta-Strategy (1-2 days)
1. Implement competition simulation
2. Model competitor submission distribution (assume most submit ~seed-based)
3. Generate primary (conservative) and hedge (aggressive) submissions
4. Backtest on 2023-2025 leaderboards

### Phase 5: Validation & Pruning (1-2 days)
1. Run full LOYO backtest on 2018-2025
2. Remove any component that doesn't improve LOYO Brier by > 0.001
3. Freeze final model configuration
4. Generate submission
