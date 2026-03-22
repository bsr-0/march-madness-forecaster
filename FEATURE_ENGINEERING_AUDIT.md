# Feature Engineering Audit Report

**Date:** 2026-03-22
**Scope:** Full audit of `src/data/features/` and supporting pipeline code
**Codebase version:** Current HEAD on `main`

---

## 1. Architecture Overview

The feature engineering pipeline transforms raw team statistics into a fixed 79-dimensional vector per team, then constructs matchup-level features for pairwise game prediction.

**Data flow:**

```
Raw Data (Torvik, ESPN, Kaggle, rosters)
  → IncrementalMetricsEngine (point-in-time computation)
    → TeamFeatures dataclass (79-dim vector via to_vector())
      → MatchupFeatures (diff + absolute + interactions)
        → Feature selection (VIF, correlation, importance pruning)
          → StandardScaler normalization
            → LightGBM / XGBoost / LogisticRegression ensemble
```

**Key files:**

| File | Role |
|------|------|
| `src/data/features/feature_engineering.py` | `TeamFeatures` (79-dim), `MatchupFeatures`, redundancy audit |
| `src/data/features/proprietary_metrics.py` | `IncrementalMetricsEngine` — point-in-time metric computation |
| `src/data/features/feature_selection.py` | Correlation, VIF, variance, importance pruning |
| `src/data/features/materialization.py` | Leakage-safe historical feature materialization |
| `src/data/features/tournament_features.py` | Tournament-specific context features |
| `src/data/features/public_advanced_metrics.py` | External metric ingestion |
| `src/data/features/statistical_audit.py` | Statistical validation of feature distributions |
| `contracts/feature_contracts.yaml` | Machine-executable feature contracts |

---

## 2. Feature Inventory (79 Dimensions)

### 2.1 Team Feature Vector Breakdown

| Group | Count | Features |
|-------|-------|----------|
| Core efficiency | 3 | adj_off_eff, adj_def_eff, adj_tempo |
| Four Factors (offense) | 4 | efg_pct, to_rate, orb_rate, ft_rate |
| Four Factors (defense) | 4 | opp_efg_pct, opp_to_rate, drb_rate, opp_ft_rate |
| Player metrics | 6 | total_rapm, top5_rapm, bench_rapm, total_warp, roster_continuity, transfer_impact |
| Experience/depth | 3 | avg_experience, bench_depth, injury_risk |
| Volatility | 4 | lead_volatility, entropy, lead_sustainability, comeback_factor |
| Shot quality | 2 | xp_per_poss, shot_distribution |
| Schedule strength | 4 | sos_adj_em, sos_opp_o, sos_opp_d, ncsos_adj_em |
| Luck | 1 | luck |
| WAB | 1 | wab |
| Poisson Binomial resume | 2 | sor, wab_poisson |
| Momentum | 1 | momentum |
| Variance/upset risk | 2 | three_pt_variance, pace_adj_variance |
| Elo | 1 | elo_rating |
| Free throw skill | 1 | free_throw_pct |
| Ball movement | 2 | assist_to_turnover, assist_rate |
| Defensive disruption | 2 | steal_rate, block_rate |
| Opponent shot selection | 2 | opp_two_pt_pct_allowed, opp_three_pt_attempt_rate |
| Conference quality | 1 | conference_adj_em |
| Shooting splits | 2 | three_pt_pct, three_pt_rate |
| Defensive xP | 1 | def_xp_per_poss |
| Win percentage | 1 | win_pct |
| Elite SOS | 1 | elite_sos |
| Q1 record | 1 | q1_win_pct |
| Foul rate | 1 | foul_rate |
| 3PT regression | 1 | three_pt_regression |
| Schedule/context | 2 | rest_days, top5_minutes_share |
| Preseason AP | 1 | preseason_ap_rank (smooth decay encoding) |
| Coaching | 7 | coach_tournament_exp, coach_tournament_win_rate, coach_deep_run_rate, coach_stage_consistency, coach_f4/e8/s16_appearances |
| Graph-theoretic SOS | 2 | pagerank_sos, multi_hop_sos |
| Win quality | 3 | best_win_percentile, paper_tiger_score, dominance_ratio |
| Pace variance | 1 | pace_variance |
| Conference tourney | 1 | conf_tourney_champ |
| Venue/resume | 3 | neutral_site_win_pct, home_court_dependence, tournament_resume |
| Position depth | 2 | backcourt_rapm, frontcourt_rapm |
| External ratings | 2 | external_rating_composite, external_rating_spread |
| Seed | 1 | seed_strength (log-transformed) |
| **Total** | **79** | |

### 2.2 Matchup Feature Construction

`MatchupFeatures.to_vector()` concatenates:

1. **Diff features** (79-dim): `team1_vector - team2_vector`
2. **Absolute-level features** (5-dim): `mean(team1[i], team2[i])` for adj_off_eff, adj_def_eff, sos_adj_em, elo_rating, win_pct
3. **Interaction features** (7-dim): tempo_interaction, style_mismatch, seed_em_residual, sos_seed_interaction, three_pt_var_seed_interaction, seed_interaction, seed_diff

**Total matchup dimension:** 79 + 5 + 7 = **91 features** before feature selection.

---

## 3. Redundancy Audit (Completed)

11 algebraically or near-perfectly redundant features were identified and removed (documented at `feature_engineering.py:38-65`):

| Removed Feature | Reason | Correlation |
|----------------|--------|-------------|
| adj_efficiency_margin | adj_off - adj_def | exact linear |
| seed_efficiency_residual | adj_em - f(seed) | exact linear |
| efficiency_ratio | adj_off / adj_def | r=0.95 |
| barthag | monotonic transform of adj ratio | deterministic |
| consistency | 1/(1+std_margin) | near-inverse of pace_adj_var |
| momentum_5g | last-5-game delta | r=0.85 with momentum |
| true_shooting_pct | PTS/(2*(FGA+0.44*FTA)) | r=0.92 with efg+ft_rate |
| opp_true_shooting_pct | opponent version | r=0.92 |
| two_pt_pct | FG2M/FG2A | r=0.88 with efg_pct |
| continuity_learning_rate | 1+0.15*(1-continuity) | deterministic |
| close_game_record | wins in close games | pure noise (stability=0.1) |

**Assessment:** Thorough. Attributes are retained for downstream compatibility but excluded from `to_vector()`. The correlation thresholds (r>0.85) are reasonable.

---

## 4. Leakage Prevention

### 4.1 Point-in-Time Safety

The system enforces temporal integrity through multiple layers:

1. **`IncrementalMetricsEngine.compute_as_of(as_of_date)`** — Uses **strict `<`** comparison (not `<=`) to exclude same-game outcomes from features. Validated by structural test at `tests/data_integrity/test_leakage_rules.py:56`.

2. **Tournament cutoff dates** — `TOURNAMENT_START_DATES` dict gates:
   - Seeds zeroed before tournament
   - Massey composites gated by tournament cutoff
   - Conference tournament results excluded from regular-season features
   - Coach temporal guard via `coach_data_cutoff_year`

3. **Training row assembly** — Features computed at `game_date` use only games strictly before that date. Label (win/loss) comes from the game itself. Validated at `tests/data_integrity/test_training_row_assembly.py`.

### 4.2 Four Leakage Rules (test_leakage_rules.py)

Each rule tested with three evidence types: structural (code inspection), synthetic (canary injection), historical (real data windows).

| Rule | Description | Status |
|------|-------------|--------|
| Rule 1 | No same-game outcome leakage | PASS |
| Rule 2 | No future opponent outcomes | PASS |
| Rule 3 | No post-date aggregates | PASS |
| Rule 4 | No tournament-derived fields pre-tournament | PASS |

### 4.3 Feature Contracts

`contracts/feature_contracts.yaml` provides machine-executable contracts for all 79 features with:
- `available_at_logic`: Structured offset/function reference (not prose)
- `leakage_checks`: Four boolean checks per feature
- `risk_tier`: low/medium/high/critical
- `snapshot_policy`: Required for high/critical features
- `transformation_logic_ref`: References to actual code modules

**Assessment:** Leakage prevention is rigorous and multi-layered. The three-evidence-type approach (structural, synthetic, historical) is unusually thorough.

---

## 5. Feature Transformations

### 5.1 In-Vector Transforms

Several features are transformed within `to_vector()`:

| Feature | Transform | Rationale |
|---------|-----------|-----------|
| rest_days | `min(value, 14.0)` | Cap outlier inflation |
| preseason_ap_rank | `1/(1 + rank/10)` if ranked, else 0.25 | Smooth decay, no cliff at #25 |
| coach_tournament_exp | `log1p(appearances) / log1p(30)` | Diminishing returns normalization |
| coach_f4/e8/s16 | `log1p(count) / log1p(15/20/25)` | Per-stage log scaling |
| seed_strength | `log1p(17 - seed) / log1p(16)` | Log transform preserving ordinal info |

### 5.2 Pipeline Normalization

- **FIX #2 (completed):** Manual z-scoring removed from `to_vector()`. Raw values emitted.
- `StandardScaler` in the pipeline handles all normalization (fit on training data only).
- Soft clip at `[-1000, 1000]` as a data-error safety net only.

### 5.3 NaN/Inf Handling

`to_vector()` at lines 644-657 detects NaN/inf values, logs warnings with specific feature names, and replaces with 0.0.

**Concern:** Replacing NaN with 0.0 is a reasonable default for tree-based models (LightGBM/XGBoost handle missing values natively), but 0.0 may be a misleading imputation for features where 0.0 has semantic meaning (e.g., `win_pct`, `elo_rating` with default 1500). Consider using `np.nan` and letting the tree models' native missing-value handling take over.

---

## 6. Feature Selection Pipeline

`src/data/features/feature_selection.py` implements a multi-stage pruning pipeline:

1. **Near-zero variance pruning** — Removes constant and near-constant features
2. **VIF pruning** — Detects multicollinearity (threshold configurable, default 10.0)
3. **Correlation pruning** — Drops highly correlated pairs (threshold configurable)
4. **Importance ranking** — Feature importance from tree models

**Validated by** `tests/test_feature_selection.py` (737 lines, 8 test classes covering):
- Perfect/multi-way collinearity detection
- LogisticRegression coefficient stability post-selection (coefficients < 15.0)
- Full pipeline integration: variance → VIF → correlation → importance

**Assessment:** Selection pipeline is sound. The ordering (variance → VIF → correlation → importance) is correct — coarser filters first.

---

## 7. Findings and Recommendations

### 7.1 Strengths

1. **Rigorous leakage prevention** — Four explicit rules with three evidence types each. Machine-executable contracts with bidirectional validation. `LeakageError` exception enforced in production.

2. **Well-documented redundancy audit** — 11 features removed with explicit correlation justifications. Attributes retained for compatibility without polluting the model input.

3. **Fixed dimension enforcement** — `TEAM_FEATURE_DIM = 79` checked at module import time, at `to_vector()` runtime, and at `get_feature_names()` call time. Three-way assertion makes dimension drift nearly impossible.

4. **Normalization separation** — Raw values emitted from `to_vector()`; StandardScaler handles normalization in the pipeline (fit on training data only). Prevents information leakage through normalization.

5. **Feature contracts** — YAML-based contracts with structured `available_at_logic`, leakage checks, risk tiers, and code references. Validated by 10 test classes.

6. **Governance trail** — Feature manifests with SHA-256 hashes, RDoF audit registry, experiment ledger tracking reproducibility.

### 7.2 Concerns

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| C1 | Low | **NaN replacement with 0.0** may be suboptimal. For tree models, `np.nan` would leverage native missing-value splits. 0.0 is semantically misleading for features like `elo_rating` (default 1500) or `win_pct` (default 0.5). | `feature_engineering.py:657` |
| C2 | Low | **Population stats dictionary** (`_POPULATION_STATS` at line 344) is hardcoded and marked "updated 2026." If these drift significantly from actual data, validation warnings may be stale. No automated refresh mechanism. | `feature_engineering.py:344-411` |
| C3 | Low | **Preseason AP rank encoding** (`1/(1 + rank/10)`) gives rank #1 → 0.91, rank #25 → 0.29, unranked → 0.25. The gap between #25 and unranked (0.29 vs 0.25) may be too compressed for the model to distinguish borderline ranked vs unranked teams. | `feature_engineering.py:560` |
| C4 | Low | **Coach features are 7/79 dimensions** (9% of vector) — disproportionate for what is historically a weak predictor in tournament outcomes. Feature selection should prune if they lack signal, but the initial dimensionality allocation is heavy. | `feature_engineering.py:562-580` |
| C5 | Info | **Missing-data indicators were removed** from `MatchupFeatures.to_vector()` (OOS-FIX note at line 851). The 6 binary flags were found to encode scraper artifacts. This is correct — confirms the team is actively monitoring for data-artifact leakage. | `feature_engineering.py:851-854` |
| C6 | Info | **`close_game_record` removed** (FIX 2.4) as pure noise with binomial stability of 0.1 on 5-10 games. Academically justified. Attribute retained but excluded from vector. | `feature_engineering.py:71-72` |
| C7 | Low | **`wab` and `wab_poisson`** may be near-redundant (both measure wins above bubble expectation via different methods). If correlated >0.85 in practice, one could be pruned. The feature selection pipeline should catch this, but it's worth verifying. | `feature_engineering.py:486-492` |
| C8 | Low | **Interaction features are hand-engineered** (7 specific interactions in MatchupFeatures). Tree-based models can learn interactions natively. These pre-computed interactions may add value for logistic regression in the ensemble but could be redundant for the tree models. | `feature_engineering.py:856-864` |

### 7.3 Overall Assessment

The feature engineering pipeline is **production-quality** with strong safeguards:

- **Leakage risk:** Very low. Multi-layered prevention with structural, synthetic, and historical evidence.
- **Redundancy:** Well-managed. Prior audit removed 11 features; remaining correlations should be caught by VIF/correlation pruning.
- **Maintainability:** High. Fixed dimension constant with triple assertion, machine-executable contracts, SHA-256 manifest tracking.
- **Scalability concern:** The 79-dim vector is manually maintained in a dataclass. Adding/removing features requires synchronized changes in `to_vector()`, `get_feature_names()`, contracts, and tests. The triple assertion catches drift but doesn't prevent the maintenance burden.

**Verdict:** The feature engineering process is well-designed and well-tested. The concerns identified are all low-severity and do not pose risks to production correctness. The leakage prevention framework is notably thorough for a tournament prediction system.
