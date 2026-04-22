# March Madness Forecaster — Build Workflow

> The efficient path from zero to production-locked tournament predictor.
> Each phase builds on the previous. No dead ends, no wasted experiments.

---

## High-Level Pipeline

```
Phase 1          Phase 2            Phase 3           Phase 4            Phase 5           Phase 6
DATA             FEATURES           MODEL              CALIBRATION        SIMULATION         OPTIMIZATION
FOUNDATION       ENGINEERING        SELECTION           & VALIDATION       ENGINE             & EXPORT
                                                                          
┌──────────┐    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Ingest    │───▶│ Build 86-dim │──▶│ Prove simple │──▶│ Temperature  │──▶│ Monte Carlo  │──▶│ Contrarian   │
│ Multi-src │    │ team vectors │   │ LR = ceiling │   │ scaling on   │   │ 10k bracket  │   │ bracket      │
│ + PIT     │    │ + matchup    │   │ via LOYO     │   │ tourney-only │   │ simulations  │   │ portfolio    │
│ guardrails│    │ diffs        │   │ backtest     │   │ data         │   │              │   │ vs public    │
└──────────┘    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
     │                │                   │                  │                  │                   │
     ▼                ▼                   ▼                  ▼                  ▼                   ▼
  raw JSON         98-dim             7-feature LR       calibrated          bracket           optimized
  + manifest       matchup vec        (seed = 2026)      probabilities       distributions     picks + CSV
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
├── Sports Reference ────────▶ Historical tournament outcomes
├── SBRO Excel archives ─────▶ Spreads, moneylines, totals (2008-2022)
├── Covers.com (Playwright) ─▶ Spreads, totals (2023+)
└── ESPN injury API ─────────▶ Player injury status (current season)

Step 1.2: Historical Backfill
├── Scrape 2008–2025 regular season games (~2,200/yr)
├── Scrape tournament results (63 games/yr)
├── Exclude 2020 (COVID, rule changes)
├── Build unified odds: scripts/build_unified_odds.py (87K+ games, 19 seasons)
└── Output: data/raw/ + data/historical/ + data/processed/betting_odds/

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
Step 2.1: Team Feature Vectors (55 dimensions)
├── Efficiency:     adj_off_eff, adj_def_eff, adj_tempo (Torvik)
├── Four Factors:   eFG%, TO%, ORB%, FT rate (Torvik FF overlay)
├── Elo:            MOV-adjusted, K=38, 0.75 season carryover
├── Player:         RAPM, WARP, top-5 player quality (rosters)
├── Schedule:       SOS AdjEM, opponent quality
├── Momentum:       Last-10-game AdjEM trajectory
├── Experience:     Avg experience, roster continuity %
├── Volatility:     3PT variance, scoring consistency
├── Conf tourney:   champion flag, games played, avg margin (12-day window)
├── Late-season:    all games in pre-tournament window (games, margin, win%)
├── Market:         avg implied probability + avg spread from Vegas odds
└── Injury:         injury_risk (RAPM-weighted injured player impact)

Step 2.2: Matchup Construction
├── Diff features:   team_A - team_B for each metric
├── Absolute levels:  avg(team_A, team_B) for context features
├── Interactions:     7 cross-term features
└── Output: 67-dim matchup vector (55 diff + 5 absolute + 7 interactions)

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

## Phase 5 — Simulation Engine

**Goal:** Probabilities → full bracket distributions.

```
Step 5.1: Monte Carlo Bracket Simulation
├── 10,000 tournament simulations
├── Each sim: draw winner for every game using calibrated P(win)
├── Noise injection: small perturbation for bracket diversity
├── Track: team advancement rates per round
└── Output: P(team reaches R32), P(S16), ..., P(Champion)

Step 5.2: Bracket Validity
├── Every simulated bracket is structurally valid (no byes skipped)
├── Upset rates match historical tournament distributions
├── Seed-matchup outcomes follow expected patterns
└── Champion distribution is not degenerate (top team < 30%)
```

---

## Phase 6 — Optimization & Export

**Goal:** Win the pool, not the accuracy contest.

```
Step 6.1: Contrarian Bracket Optimization
├── Input: our P(advance) vs ESPN public pick %
├── Strategy: game theory — maximize EV against the field
├── Leverage = our_probability / public_pick_percentage
├── High leverage = undervalued teams to target
├── Low leverage = overvalued favorites to fade
└── Kelly-inspired portfolio construction

Step 6.2: Bracket Portfolio Generation
├── Generate diverse set of optimized brackets
├── Each bracket: unique contrarian angles
├── Portfolio: covers multiple tournament scenarios
└── Not all-in on one outcome

Step 6.3: Export
├── Kaggle submission CSV (team-pair probabilities)
├── Bracket picks (human-readable)
├── Governance artifacts:
│   ├── production_manifest_2026.json
│   ├── production_freeze_2026.json
│   └── production_governance_report_2026.json
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
Week 2:  Feature engineering (86-dim) + redundancy audit
Week 3:  LOYO backtest → prove LR = ceiling → lock 7 features
Week 4:  Temperature calibration on tournament data + validation gates
Week 5:  Monte Carlo sims + contrarian optimizer + Kaggle export
Week 6:  Governance layer + freeze + production lock

Total: 6 focused weeks, no detours into GNN/transformers/ensembles
       (those experiments confirmed the ceiling but weren't needed)
```

---

## Pending: Feature Group Evaluation (Pre-2027)

**Status: BUILT but NOT EVALUATED.**

Eight new features (indices 46-54) were added in April 2026 across four groups. They are wired into the training pipeline and LOYO evaluator but have not yet been tested for Brier score impact. This must happen before locking the 2027 production model.

```
Feature Group Ablation Testing
┌──────────────────────────────────────────────────────────────┐
│  Run LOYO with each group ON/OFF, compare Brier scores:     │
│                                                               │
│  Group              Indices   Data Source      Coverage       │
│  ─────              ───────   ───────────      ────────       │
│  Conf tournament    46-48     Game records     2008-2026 ✓   │
│  Late-season        49-51     Game records     2008-2026 ✓   │
│  Market (Vegas)     52-53     Unified odds     2008-2026 ✓   │
│  Injury risk        54        ESPN scraper     Current yr    │
│                                                               │
│  Config flags: ablate_conf_tourney, ablate_late_season,      │
│                ablate_market, ablate_injury                   │
│                use_market_features=True (to populate Vegas)   │
│                                                               │
│  Decision gate: ship if BSS improves ≥ 0.002                │
│  Timeline: evaluate by February 2027                         │
└──────────────────────────────────────────────────────────────┘
```

Data rebuild commands (if odds cache is missing):
```bash
python scripts/ingest_sbro_odds.py --all              # SBRO Excel → JSON
python scripts/scrape_covers_odds.py --all --full-season  # Covers → JSON
python scripts/build_unified_odds.py                   # merge all sources
python scripts/scrape_injuries.py --season 2027        # ESPN injuries (March)
```

## Key Insight

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   The winning move was PROVING that complexity doesn't help,    │
│   then building a production system around simplicity.          │
│                                                                 │
│   Seeds explain ~87% of tournament outcomes.                    │
│   63 games/year is not enough data for ML to beat that.         │
│   The real edge is in POOL OPTIMIZATION, not prediction.        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
