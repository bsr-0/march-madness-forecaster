# March Madness Forecaster — Agent Directive V7 Evaluation

**Evaluation Date:** 2026-03-08
**Evaluator:** Automated Directive V7 Compliance Audit
**Codebase:** `march-madness-forecaster`
**Source Lines:** ~57,000 (src) + ~27,000 (tests)

---

## OVERALL SCORE: 62 / 100

---

## Scoring Breakdown by Directive V7 Section

### Part I — Core Research and Validation Protocol (Sections 1–17)

| Section | Description | Max | Score | Notes |
|---------|-------------|-----|-------|-------|
| S1 | Mission & Non-Negotiable Principles | 5 | 4 | Temporal integrity enforced via LOYO and snapshot data. Decision objective (Brier/log loss) clearly identified. Reproducibility partially addressed (seeds logged, but no dataset hashing). |
| S2 | Multi-Agent System Architecture | 5 | 1 | No multi-agent orchestration. Single-pipeline monolith — no separate Data Agent, Feature Agent, Model Agent, Audit Agent, or Research Orchestrator. |
| S3 | Shared Contracts & Required Logs | 5 | 2 | `ExperimentRegistry` exists (`experiment_registry.py`) but is minimal. No full shared ledger with dataset hashes, feature sets, model configs, seed states, and validation plans per experiment. |
| S4 | Phase 0 — Problem Definition & Utility Mapping | 5 | 4 | Prediction target (game outcomes), optimization target (Brier score + Kaggle weighted Brier), and action layer (bracket submission / calibration) are well-defined. Operational constraints partially documented. |
| S5 | Phase 1 — Dataset Discovery & Lineage | 8 | 5 | Rich data universe: Kaggle CSVs, KenPom-style metrics, external ratings (Torvik, Massey, BPI, Elo), roster data (cbbpy), betting markets, ESPN picks, injury reports, tournament context. Historical data spans 2003-2026. However: no formal field-level availability timestamps, no raw snapshot versioning system, and survivorship bias testing is absent. |
| S6 | Phase 2 — Feature Discovery Engine | 8 | 6 | Excellent feature breadth: Four Factors, adjusted efficiency, RAPM aggregation, travel distance (haversine), coach tournament power, tournament resume composite (Bayesian-shrunk), seed matchup priors, symmetric training features, momentum/rolling stats, experience/continuity, style-of-play variance. 78-dim matchup vector with differential, absolute, and interaction blocks. Missing: formal feature availability audit with timestamps, no explicit production-availability check per feature. |
| S7 | Phase 3 — Model Search & Meta-Learning | 8 | 5 | Models searched: LightGBM (spread regression), XGBoost, Logistic Regression, SpreadRegressor (margin-first), BayesianBT, TournamentExpert, CFA. Good diversity. However: no formal meta-learning layer that learns which strategies win by regime. No systematic architecture search log. Hyperparameter tuning exists (Optuna-based) but no temporal-only tuning enforcement documented. |
| S8 | Phase 4 — Ensemble Optimization & Calibration | 8 | 7 | Strong ensemble: MarginFirstEnsemble (55/15/15/15 spread/lgb/xgb/logistic) + TournamentExpert blend at 0.30. Calibration pipeline is excellent: Isotonic, Platt, Temperature Scaling with sample-size guardrails, small-sample bootstrap CI guard, calibration leakage detection (hash-based). Brier decomposition (reliability/resolution/uncertainty). Round-specific sigma calibration with Bayesian shrinkage. Missing: no formal diversity measurement between ensemble components. |
| S9 | Phase 5 — Decision Optimization Layer | 5 | 4 | Bracket optimization (`bracket_search.py`, `bracket_portfolio.py`), dual submission strategy, pool competition modeling with competitor archetypes, leverage optimization, EV-mode vs calibration-mode gating. Abstention not explicitly modeled as a first-class policy for Kaggle submission. |
| S10 | Phase 6 — Backtesting & Simulation Realism | 8 | 6 | LOYO validation (2018-2025, excluding 2020). Unified backtest framework covering Kaggle and ESPN pool strategies. Kaggle round weighting applied. Walk-forward replay test exists. However: no explicit friction terms (tournament timing, execution delay). Scenario sensitivity (optimistic/base/pessimistic) not formally implemented. Drawdown and path-dependent risk not reported. |
| S11 | Phase 7 — Skeptical Audit Layer | 8 | 5 | Leakage canary tests exist (`test_leakage_canary.py`, `test_leakage_fixes.py`). Date integrity tests. Feature dropout robustness test. Distribution shift detection (PSI). Feature importance stability (Kendall tau). RDOF audit. However: no dedicated Audit Agent with veto power. No formal validation audit checking for random k-fold misuse. Reproducibility audit (dataset hashing, frozen seeds) is incomplete. |
| S12 | Phase 8 — Codebase Review & Refactoring | 5 | 3 | Refactoring roadmap exists (`docs/REFACTORING_ROADMAP.md`). Training data audit documented. Fix audit comments in code. However: no formal dependency graph analysis, no circular dependency detection, no dead code analysis, no blast-radius assessment for refactors. |
| S13 | Required Evaluation Matrix | 3 | 2 | Metrics exist (Brier, log loss, accuracy, ECE, MCE, ROC-AUC, bootstrap CIs) but no standardized evaluation matrix comparing all candidate systems in a single comparable table. |
| S14 | Continuous Autonomous Research Loop | 5 | 1 | No automated research loop. No hypothesis generation, automated experiment execution, adversarial review, or promotion gate system. Feature ablation (0.001 rule) exists but is manual. |
| S15 | Failure Mode Rejection | 3 | 2 | Leakage detection exists. Calibration leakage guard exists. But no automated rejection system — failures must be caught manually. |
| S16 | Final Deliverables | 3 | 2 | Kaggle export exists. Dashboard data builder. Methodology docs. But missing: frozen experiment config, reproducibility package, full audit report, deployment artifacts. |
| S17 | Operating Summary | 2 | 2 | System is decision-aware, domain-specific (March Madness + women's tournament), and Brier-optimized. |

**Part I Subtotal: 61 / 91**

---

### Part II — Deployment, Operations, and Governance (Sections 18–25)

| Section | Description | Max | Score | Notes |
|---------|-------------|-----|-------|-------|
| S18 | Production Deployment & Live Monitoring | 5 | 1 | `pipeline_monitor.py` and `phase_timer.py` exist for basic pipeline timing. `live_refresh.py` for live data updates. No shadow deployment, canary testing, A/B framework, or real-time drift monitoring dashboard. No staged deployment pipeline. |
| S19 | Data Engineering & Pipeline Resilience | 5 | 2 | Pipeline has stages (`stages/` module with data_loader, model_trainer, calibrator, simulator, reporter). Circuit breaker test exists. However: no formal DAG orchestration, no idempotency guarantees, no schema validation, no data freshness SLAs, no fault-tolerance recovery protocol. |
| S20 | Computational Budget & Resource Prioritization | 2 | 1 | `resource_tracker` test exists suggesting some cost tracking. But no formal compute budget framework, no prioritized search strategy, no Pareto frontier reporting. |
| S21 | Human-in-the-Loop Governance | 2 | 0 | No governance framework. No approval gates, decision authority matrix, compliance checkpoints, or audit trail for governance actions. |
| S22 | Multi-Agent Conflict Resolution | 1 | 0 | No multi-agent system exists, so conflict resolution is N/A. |
| S23 | Testing Strategy & CI/CD Integration | 4 | 3 | **Strong testing**: 811 passing tests across 80+ test files (27,000 lines). Test coverage: unit tests, integration tests, leakage canaries, walk-forward replay, temporal integrity tests, feature materialization tests. `pyproject.toml` configures pytest, ruff linting, coverage (target 60%, goal 75%). Missing: no formal CI/CD pipeline config (no `.github/workflows/`, no `Jenkinsfile`). No pipeline ordering determinism test. |
| S24 | Domain-Specific Integration Guides | 3 | 2 | Sports betting domain well-covered: Selection Sunday snapshot awareness, venue neutrality, seed-based priors, historical tournament patterns, women's tournament pipeline. Travel distance, coach metrics, Kaggle scoring rules all implemented. Missing: formal domain guide document. |
| S25 | Extended Failure Modes & Deliverables | 2 | 1 | Some extended failure modes are caught (circuit breaker, pre-tournament checklist test). But no comprehensive coverage of all V7 Part II failure modes. |

**Part II Subtotal: 10 / 24**

---

## Rubric-Specific Evaluation (March Madness PDF)

| Rubric Criterion | Status | Score (0-5) |
|-----------------|--------|-------------|
| **Fundamental Efficiency Metrics** (KenPom AdjO/AdjD, Net Rating, eFG%, BPI/Elo) | Implemented: adjusted efficiency, net rating, external ratings (Elo, BPI, Massey ordinals), eFG% via Four Factors | 5 |
| **Four Factors** (eFG%, TO%, ORB%/DRB%, FT Rate) | Fully implemented in `feature_engineering.py` as core feature family | 5 |
| **Contextual/Soft Variables** (WAB, momentum, experience, travel distance) | WAB partially (SOR implemented), momentum/rolling averages present, experience/continuity via roster data, travel distance with haversine computation for 370+ teams | 4 |
| **Essential ML Variables** (Diff in Adj Net Rating, log-transformed seed, luck factor, injury-adjusted player ratings) | Differential features as core architecture (78-dim vector), seed handling present, player-level features via RAPM/Box Plus-Minus. Luck factor not explicitly named but stability metrics present | 4 |
| **Log Loss as Evaluation Metric** | Brier score is primary (related to log loss). Log loss also computed. Calibration explicitly prioritized with multiple methods | 5 |
| **Preprocessing: Standardization** | StandardScaler in pipeline (noted in fix audit: removed manual z-scoring, pipeline handles normalization) | 5 |
| **Algorithm Selection: XGBoost/LightGBM** | Both XGBoost and LightGBM implemented as ensemble components. LightGBM is primary spread regressor | 5 |
| **Cross-Validation: Leave-One-Year-Out** | Fully implemented in `loyo_protocol.py` with 7 folds (2018-2025, excl. 2020). The "0.001 Rule" for feature ablation | 5 |
| **Data Leakage Prevention** (Selection Sunday snapshot) | Addressed: date integrity tests, leakage canary tests, temporal split enforcement in symmetric training | 4 |
| **Feature Symmetry / Difference Modeling** | Excellent: 78-dim matchup vector with differential block [0:66], absolute block [66:71], interaction block [71:78]. Symmetric augmentation with rigorous swap logic and zero-sum verification | 5 |
| **Calibration** | Outstanding: Temperature scaling (w/ small-sample bootstrap guard), Platt scaling, Isotonic regression, sample-size guardrails, calibration leakage detection, Brier decomposition, reliability diagrams, per-bin analysis | 5 |

**Rubric Subtotal: 52 / 55**

---

## Strengths

1. **Exceptional calibration infrastructure**: Temperature scaling with bootstrap CI small-sample guard, calibration leakage detection via data hashing, sample-size-aware method downgrading — this is production-grade calibration.

2. **Tournament-specific sigma calibration**: Per-round sigma estimation with Bayesian shrinkage toward global tournament sigma, Kaggle round-weight awareness — directly addresses the domain's unique scoring structure.

3. **Symmetric training with mathematical rigor**: Detailed swap logic for 78-dim matchup vectors with block-level documentation, zero-sum property verification, interleaved augmentation preserving temporal split integrity.

4. **Comprehensive test suite**: 811 passing tests including leakage canaries, walk-forward replay, temporal integrity, feature materialization, and domain-specific tests.

5. **Feature engineering depth**: Four Factors, RAPM, coach tournament power, Bayesian-shrunk tournament resume composite, travel distance with geocoordinates for 370+ teams, historical seed priors from 1985-2025 data.

6. **Margin-first ensemble architecture**: SpreadRegressor as primary (55%) with probability models as complements — preserves richer gradient signal from margin prediction.

## Weaknesses

1. **No multi-agent architecture** (Directive S2): The system is a monolithic pipeline, not a coordinated lab of specialized agents. No Data Agent, Feature Agent, Model Agent, Audit Agent, or Research Orchestrator.

2. **No continuous research loop** (Directive S14): No automated hypothesis generation, experiment execution, adversarial review, or promotion gates. All improvements are manual.

3. **No production deployment infrastructure** (Directive S18): No shadow mode, canary testing, A/B framework, drift monitoring dashboard, or staged deployment pipeline.

4. **No governance framework** (Directive S21): No approval gates, decision authority matrix, or audit trail for high-stakes actions.

5. **No CI/CD pipeline** (Directive S23): Despite excellent tests, there's no `.github/workflows/` or equivalent automated pipeline configuration.

6. **Incomplete reproducibility** (Directive S1/S3): No dataset versioning/hashing, no frozen experiment configs, no artifact versioning system.

7. **No formal DAG pipeline orchestration** (Directive S19): Pipeline stages exist but lack idempotency guarantees, schema validation, fault tolerance, and data freshness SLAs.

---

## Score Summary

| Category | Score | Max |
|----------|-------|-----|
| Part I: Core Research & Validation | 61 | 91 |
| Part II: Deployment, Ops & Governance | 10 | 24 |
| **Weighted Total (mapped to 100)** | **62** | **100** |

### Score Interpretation
- **90-100**: Production-ready, fully V7 compliant autonomous research lab
- **70-89**: Strong research system with some operational gaps
- **50-69**: Good modeling core, significant infrastructure gaps
- **30-49**: Prototype with major compliance gaps
- **0-29**: Early-stage or non-compliant

**This repository scores 62/100** — a strong modeling and validation core (especially calibration, feature engineering, and testing) that significantly outperforms in the research/modeling sections but falls short on operational infrastructure, multi-agent architecture, governance, and automated research loops required by the full Directive V7 specification.

The codebase is well-positioned for Kaggle competition performance but would need substantial infrastructure additions to operate as the "autonomous research lab" that Directive V7 envisions.
