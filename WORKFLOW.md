# March Madness Forecaster — Build Workflow

> The efficient path from zero to production-locked tournament predictor.
> Each phase builds on the previous. No dead ends, no wasted experiments.

---

## High-Level Pipeline

```
Phase 1          Phase 2            Phase 3           Phase 4            Phase 5           Phase 6
DATA             FEATURES           PROBABILITY        CALIBRATION        META-SELECTOR      EVALUATION
FOUNDATION       ENGINEERING        BASES               & VALIDATION       TRAINING           & EXPORT
                                                                          
┌──────────┐    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Ingest    │───▶│ Build 56-dim │──▶│ Multi-source │──▶│ Temperature  │──▶│ Learned      │──▶│ LOYO eval    │
│ Multi-src │    │ team vectors │   │ prob bases   │   │ scaling on   │   │ meta-model   │   │ P(1st) vs    │
│ + PIT     │    │ + matchup    │   │ (torvik,elo, │   │ tourney-only │   │ per-game     │   │ opponent     │
│ guardrails│    │ diffs        │   │ odds,massey) │   │ data         │   │ decisions    │   │ field        │
└──────────┘    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
     │                │                   │                  │                  │                   │
     ▼                ▼                   ▼                  ▼                  ▼                   ▼
  raw JSON         56-dim             round_probs        calibrated          1 bracket          P(1st) per
  + manifest       team vec           per source          probabilities       per model/year     backtest yr
```

---

## Phase 1 — Data Foundation

**Goal:** Reliable, multi-source data with point-in-time guarantees.

```
Step 1.1: Multi-Source Ingestion
├── Torvik scraper ──────────▶ AdjEM, AdjO, AdjD, tempo
├── cbbpy / sportsipy ───────▶ Game box scores + rosters
├── ESPN scraper ────────────▶ Public pick percentages
├── Kaggle CSVs ─────────────▶ Massey Ordinals, seeds, results
└── Sports Reference ────────▶ Historical tournament outcomes

Step 1.2: Historical Backfill
├── Scrape 2008–2025 regular season games (~2,200/yr)
├── Scrape tournament results (63 games/yr)
├── Exclude 2020 (COVID, rule changes)
└── Output: data/raw/ + data/historical/

Step 1.3: Point-in-Time (PIT) Integrity
├── Tier 1 (Static):      Seed, conference — no restriction
├── Tier 2 (Cumulative):  Regular-season stats — before tourney start only
├── Tier 3 (External):    Torvik/KenPom — Selection Sunday snapshot only
└── strict_leakage_mode = true (audit every sample)

Step 1.4: Data Contracts
├── Pydantic schema validation on all ingested data
├── Team name normalization via configs/team_aliases.json
├── Manifest generation (source coverage, row counts, hashes)
└── Output: manifest_2026.json
```

---

## Phase 2 — Feature Engineering

**Goal:** Domain-knowledge features, not kitchen-sink ML features.

```
Step 2.1: Team Feature Vectors (56 dimensions)
├── Efficiency:     adj_off_eff, adj_def_eff, adj_tempo (Torvik)
├── Four Factors:   eFG%, TO%, ORB%, FT rate (box scores)
├── Elo:            MOV-adjusted, K=20, 0.75 season carryover
├── Player:         RAPM, WARP, top-5 player quality (rosters)
├── Schedule:       SOS AdjEM, opponent quality
├── Momentum:       Last-10-game AdjEM trajectory
├── Experience:     Avg experience, roster continuity %
├── Volatility:     3PT variance, scoring consistency
└── Preseason:      AP rank, preseason expectations

Step 2.2: Matchup Construction
├── Diff features:   team_A - team_B for each metric
├── Absolute levels:  avg(team_A, team_B) for context features
├── Interactions:     7 cross-term features
└── Output: 98-dim matchup vector

Step 2.3: Redundancy Audit
├── Remove algebraic duplicates (adj_efficiency_margin = adj_off - adj_def)
├── Remove near-perfect correlations (ρ > 0.90)
├── Remove pure noise (close_game_record, ~5 game sample)
└── Result: 10 features removed, clean feature set

Step 2.4: Preprocessing
├── StandardScaler (fit on training set only)
├── Pre-clipping [-6, 6] for outlier robustness
└── Symmetric augmentation (A vs B → also B vs A)
```

---

## Phase 3 — Model Selection

**Goal:** Prove the ceiling, then lock to the simplest model that reaches it.

```
Step 3.1: LOYO Backtest (the key experiment)
┌─────────────────────────────────────────────────────────┐
│  Leave-One-Year-Out across 2008–2024 (16 folds)        │
│                                                          │
│  Models tested:                                          │
│  ├── Seed baseline (1/seed linear)                       │
│  ├── 7-feature Logistic Regression                       │
│  ├── 27-feature LR                                       │
│  ├── LightGBM / XGBoost ensemble                         │
│  ├── Stacked ensemble (LR + GBM + XGB)                   │
│  ├── GNN embeddings (disabled after test)                │
│  └── Transformer representations (disabled after test)   │
│                                                          │
│  Result: BSS ≈ 0 vs seed baseline for ALL models         │
│  Seeds encode 85-90% of tournament variance              │
│  63 games/year = not enough signal for complex models     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
Step 3.2: Lock Production Model
├── Model: LogisticRegression(C=1.0, penalty='l2')
├── Features: 7 (the SIMPLE_FEATURE_SET)
│   ├── diff_elo_rating
│   ├── diff_total_warp
│   ├── diff_orb_rate
│   ├── diff_momentum
│   ├── diff_adj_tempo
│   ├── diff_sos_adj_em
│   └── diff_opp_to_rate
├── Training: regular-season games only (17,600 samples, 8 years)
├── Seed: random_seed = 2026 (reproducibility)
└── Experimental modules: hard-disabled in config

Step 3.3: Seed vs No-Seed Resolution
├── With seed (2026):  deterministic, reproducible results
├── Without seed:      variance across runs is negligible (~0.001 BSS)
├── Decision:          use seed for governance/reproducibility
└── Seed does NOT inflate performance — just removes run-to-run noise
```

---

## Phase 4 — Calibration & Validation

**Goal:** Raw probabilities → tournament-calibrated probabilities.

```
Step 4.1: Temperature Scaling
├── Fit single scalar T on tournament games (2016–2025, ~530 games)
├── Tournament games are genuinely OOS (not in training data)
├── logit(p_calibrated) = logit(p_raw) / T
└── Corrects overconfidence in raw logistic output

Step 4.2: Post-Calibration Adjustments
├── Shrinkage:               0.06 toward 0.5 (tournament uncertainty)
├── Goto correction:         favorite-longshot bias adjustment
├── Round-weighted cal:      per-round calibrator (R64 ≠ F4 dynamics)
├── Market blend:            Vegas odds cross-reference (when available)
└── Final clip:              [0.01, 0.99]

Step 4.3: Probability Path (Production Profile)
    Raw LR prediction
         │
    clip [0.001, 0.999]
         │
    Temperature scaling
         │
    Shrinkage (0.06 → 0.5)
         │
    Goto correction
         │
    Round-weighted calibration
         │
    clip [0.01, 0.99]
         │
    Monte Carlo ready ✓

Step 4.4: Validation Gates
├── Brier Skill Score vs seed baseline (must be ≥ 0)
├── Calibration curve: predicted vs actual within ±0.05 per bin
├── LOYO holdout (2025): verify no degradation
├── RDOF audit: researcher degrees of freedom documented
└── PIT leakage scan: zero violations required
```

---

## Phase 5 — Meta-Selector Training

**Goal:** Multiple probability sources → one learned model that picks brackets to win pools.

```
Step 5.1: Assemble Per-Game Feature Vectors (src/prediction/meta_selector.py)
├── 26-dim feature vector per game:
│   ├── 12 probability bases: pairwise P(team1 wins) from each base
│   ├── 2 seeds, 2 ESPN public pick percentages
│   ├── leverage_diff, source_agreement, consensus_prob
│   ├── max_disagreement, seed_matchup_type, round_index
│   └── 4 context diffs: coach, momentum, talent, volatility
├── Walk-forward: build_training_data(train_years) loads only years < test_year
├── Missing bases filled with NaN (LightGBM handles natively)
└── Output: (n_games, 26) feature matrix + labels + leverage weights

Step 5.2: Two Meta Modes
├── meta_leverage (no ML):
│   ├── Per game: pick = argmax(P(win) × (1 - public_pick%))
│   ├── Uses torvik as primary base + ESPN picks for ownership
│   └── Zero training — pure leverage formula
├── meta_gbm (trained):
│   ├── LightGBM(depth=3, trees=50, min_child=20, subsample=0.8)
│   ├── Label: which team actually won (binary)
│   ├── Weight: round_pts × (1 - winner_public_pct)
│   └── Walk-forward LOYO training per test year

Step 5.3: Generate Bracket
├── Walk bracket R64 → Championship sequentially
├── Per game: build 26-feature vector → model predicts → pick winner
├── Path-consistent: winners advance to next round matchups
├── Output: one (63,) boolean bracket per mode per year (deterministic)
└── Integrated into backtest as Pass A2 (after stochastic Pass A)

Step 5.4: Legacy (baseline reference)
├── Monte Carlo stochastic simulation remains as comparator
├── seed + f4_first_tv modes run alongside meta modes
├── Used for paired statistical comparison
└── NOT the primary development path (noise ceiling at ~5%)
```

---

## Phase 6 — Evaluation & Export

**Goal:** Win the pool, not the accuracy contest.

```
Step 6.1: LOYO Bracket Evaluation
├── Score meta-selector bracket against actual tournament outcome
├── Rank against simulated opponent field (or actual pool history)
├── P(1st) as primary metric across 14 backtest years
├── Consistency gate: improvement in >= 8/14 years
└── Compare against stochastic baseline and seed baseline

Step 6.2: Production Bracket Selection
├── Run meta-selector on current year's data
├── One bracket output (deterministic)
├── Verify field differentiation via ownership analysis
├── Sensitivity check: stable under ±5% public pick shifts
└── Human review before submission

Step 6.3: Export
├── Kaggle submission CSV (team-pair probabilities)
├── Bracket picks (human-readable)
├── Governance artifacts:
│   ├── production_manifest_2027.json
│   ├── production_freeze_2027.json
│   └── production_governance_report_2027.json
└── Provenance: git SHA, config hash, data hashes
```

---

## Phase 7 — Production Lock

**Goal:** Frozen, reproducible, auditable.

```
Step 7.1: Freeze Pipeline
├── march-madness freeze-pipeline
├── Hash all source files, configs, data
├── Record git SHA, dependency versions
└── Output: production_freeze_2026.json

Step 7.2: Production Validator
├── Config hash must match frozen value
├── Experimental modules must be disabled
├── strict_leakage_mode must be true
├── Any deviation → ProductionValidationError (hard fail)
└── No CLI overrides allowed

Step 7.3: Production Run
    python src/run_production_2026.py
         │
    Validates config hash ✓
    Validates frozen modules ✓
    Validates PIT integrity ✓
         │
    Runs locked pipeline
         │
    Generates governance report
         │
    Done. ✓
```

---

## If We Did It Again: The Efficient Path

```
Week 1:  Ingest + PIT framework + data contracts
Week 2:  Feature engineering (56-dim) + redundancy audit
Week 3:  LOYO backtest → prove LR = ceiling → lock features
Week 4:  Build all probability bases (torvik, elo, odds, massey, ESPN picks)
Week 5:  Train meta-selector on per-game features → LOYO P(1st) evaluation
Week 6:  Governance layer + freeze + production lock

Total: 6 focused weeks. Skip stochastic sampling entirely — go straight
       from probability bases to learned meta-selector. The MC simulation
       era proved that coin flips can't beat a model that learns which
       upsets to pick.
```

---

## Key Insight

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Prediction accuracy has a ceiling (BSS = 0 vs seeds).         │
│   Stochastic bracket generation also has a ceiling (~5% P(1st)).│
│   The remaining edge is in LEARNED BRACKET SELECTION:           │
│   using multiple probability sources as features for a model    │
│   that decides which picks — including which upsets — to make.  │
│                                                                 │
│   The meta-selector doesn't predict better probabilities.       │
│   It makes better DECISIONS given the probabilities we have.    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix A — Data Sources

### Currently Ingested

| Source | Ingest | Signal | Access |
|---|---|---|---|
| Bart Torvik | `src/data/scrapers/torvik.py` | AdjEM, AdjO, AdjD, tempo, Four Factors | Scraped (Cloudflare) |
| Massey Ordinals (Kaggle) | `src/data/kaggle_downloader.py` | 50+ ranking systems: KenPom (POM), Sagarin, Massey, Colley, Wolfe, AP, Coaches, RPI | Kaggle API |
| ESPN public picks | `src/data/scrapers/espn_picks.py` | Round-by-round pick % — core for leverage | Scraped |
| Sports Reference | `src/data/scrapers/sports_reference.py` | Season stats, possessions | Scraped |
| cbbpy / sportsdataverse / sportsipy | `src/data/ingestion/game_fetchers.py` | Box scores, rosters | Free libs |
| Betting markets (unified) | `src/data/scrapers/unified_odds.py` | Spreads, totals, moneylines (SBRO/Covers/SBR + Odds API) | Archive + API |
| Injury reports | `src/data/scrapers/injury_report.py` | Status, severity, return date | Scraped |
| NCAA stats | `src/data/scrapers/ncaa_stats.py` | Seeds, historical outcomes | Scraped |
| Women's (NET / HerHoopStats) | `src/data/scrapers/womens/` | Women's bracket data | Scraped |
| Travel distance | `src/data/features/travel_distance.py` | Haversine school → venue | Embedded |

### Referenced but Not Ingested
- Direct KenPom scraper (only via Massey snapshot)
- EvanMiya, Haslametrics (mentioned in `external_ratings.py` docs, no impl)
- FanDuel / DraftKings scrapers — deleted per MEMORY D10
- `conference_seeds.py` — experimental stub

### Gaps vs. What Successful Bracket Models Use

**Free / cheap, worth evaluating:**
- **EvanMiya** — Bayesian performance ratings; complementary signal to Torvik
- **Haslametrics** — alternative efficiency ratings
- **ESPN BPI** — tournament-tuned, free
- **Hoop-Math** — shot location / rim & three rates; predictive beyond Four Factors
- **Returning production / transfer portal %** — Torvik exposes this on the site we already scrape; portal-era churn is a real feature
- **Coach / program tourney experience** — small but cited edge in upset modeling

**Paid:**
- **KenPom direct** (~$20/yr) — daily updates, luck, SoS components, home/away splits that Massey strips
- **ShotQuality** — shot-quality-adjusted efficiency, documented OOS lift over raw eFG%
- **Synergy** — play-type efficiency (expensive, mostly staff-tier)

**Market-side:**
- **Tournament futures** (region / F4 / champion prices) — market-implied prior distinct from game lines
- **Live line movement** during the tournament — for live-update modes

**Recommended next adds (if any):** EvanMiya (free, independent signal) and Torvik returning-production fields (already on a scraped host). KenPom direct is the obvious paid pickup but likely duplicative with Massey POM for tournament-eve modeling.
