# Agent Directive V7: Comprehensive Evaluation & Improvement Plan

**Evaluation Date:** 2026-03-07
**Directive Reference:** Agent Directive V7 Complete (All 25 Sections)
**Repository:** march-madness-forecaster
**Branch:** claude/evaluate-report-improvements-PrwoQ

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Methodology](#2-methodology)
3. [Section-by-Section Evaluation](#3-section-by-section-evaluation)
4. [Gap Analysis Summary](#4-gap-analysis-summary)
5. [Prioritized Improvement Plan](#5-prioritized-improvement-plan)
6. [Phased Implementation Roadmap](#6-phased-implementation-roadmap)
7. [Risk Register](#7-risk-register)
8. [Success Criteria](#8-success-criteria)

---

## 1. Executive Summary

The march-madness-forecaster was evaluated against all 25 sections of Agent Directive V7. The system is a **research-grade NCAA tournament prediction pipeline** with 106 source modules, 77 test files, and a 7,858-line central orchestrator (`sota.py`). It demonstrates exceptional strength in temporal integrity, domain-specific feature engineering, and decision optimization, while exhibiting significant gaps in production operations, formal governance, and continuous research automation.

### Compliance Scorecard

| Area | Sections | Compliance | Grade |
|------|----------|------------|-------|
| **Core Principles** | S1 | 85% | A- |
| **Agent Architecture** | S2 | 15% | D |
| **Experiment Logging** | S3 | 35% | C- |
| **Problem Definition** | S4 | 90% | A |
| **Data Discovery & Lineage** | S5 | 65% | B- |
| **Feature Discovery** | S6 | 80% | A- |
| **Model Search** | S7 | 55% | C+ |
| **Ensemble & Calibration** | S8 | 75% | B+ |
| **Decision Optimization** | S9 | 85% | A |
| **Backtesting Realism** | S10 | 70% | B |
| **Skeptical Audit** | S11 | 55% | C+ |
| **Codebase Quality** | S12 | 65% | B- |
| **Evaluation Matrix** | S13 | 70% | B |
| **Continuous Research Loop** | S14 | 20% | D+ |
| **Failure Mode Rejection** | S15 | 60% | B- |
| **Final Deliverables** | S16 | 45% | C |
| **Deployment & Monitoring** | S18 | 5% | F |
| **Data Eng. & Pipelines** | S19 | 25% | D |
| **Compute Budget** | S20 | 5% | F |
| **Human Governance** | S21 | 15% | D |
| **Conflict Resolution** | S22 | N/A | N/A |
| **Testing & CI/CD** | S23 | 50% | C+ |
| **Domain Integration** | S24 | 85% | A |
| **Extended Failure Modes** | S25 | 30% | D+ |

**Overall Weighted Compliance: ~48%**

---

## 2. Methodology

This evaluation cross-references every requirement in Agent Directive V7 against:

1. **Source code analysis** — 106 Python source files, entry points, and configuration
2. **Test suite review** — 77 test files and CI/CD pipeline configuration
3. **Existing audit documents** — `AGENT_DIRECTIVE_V7_AUDIT.md`, `REVIEW_AND_IMPROVEMENTS.md`, `PLAN_TOP1PCT.md`, `TRAINING_DATA_AUDIT.md`, `REFACTORING_ROADMAP.md`
4. **Directive requirements** — All 750 lines of Directive V7, both Part I (Sections 1-17) and Part II (Sections 18-25)

Each section is evaluated on:
- **What the directive requires** (normative)
- **What the codebase currently does** (descriptive)
- **The gap** (delta)
- **Recommended improvements** (prescriptive, prioritized)

---

## 3. Section-by-Section Evaluation

### S1: Mission and Non-Negotiable Principles

#### Temporal Integrity — STRONG (85%)
**Present:**
- `TOURNAMENT_START_DATES` dictionary prevents result leakage (sota.py:90-100)
- Feature materialization uses `shift(1).expanding().mean()` for prior metrics
- Proprietary metrics engine accepts `cutoff_date` parameter
- Explicit leakage checks in materialization (lines 911-941)
- `FIX-LEAKAGE-POLLS` flag excludes post-tournament polls
- 2020 COVID year excluded entirely
- Coach tournament data gated by `coach_data_cutoff_year`

**Gaps:**
- **CRITICAL:** `cutoff_date` defaults to `None` in proprietary_metrics.py — callers can accidentally include tournament games
- **HIGH:** Optional prior sources (market odds, transfer portal, weather) loaded without explicit temporal filtering in `_load_optional_prior_sources()`
- **MODERATE:** Synthetic date inference for 2022-2024 games makes rest_days/back_to_back features degenerate (NaN) without logged warnings

**Improvements Needed:**
| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S1-1 | Make `cutoff_date` a required parameter in ProprietaryMetricsEngine | CRITICAL | Low |
| S1-2 | Add temporal availability validation to all optional prior sources | HIGH | Medium |
| S1-3 | Log data quality warnings when synthetic dates cause degenerate features | MODERATE | Low |

#### Decision Objective Supremacy — STRONG (90%)
**Present:** Brier score optimization (Kaggle's actual metric since 2023), Kelly criterion, pool EV, bracket scoring. `SOTAPipelineConfig.scoring_metric` explicitly tracks objective.

#### Evidence Over Intuition — GOOD (80%)
**Present:** 0.001 Rule (features must improve LOYO Brier by >= 0.001), feature ablation engine, statistical significance tests (paired Brier t-test, permutation test, bootstrap).
**Gap:** Some Tier 3 constants cite academic sources but lack formal sensitivity analysis.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S1-4 | Run formal sensitivity analysis on all Tier 3 RDoF constants | MODERATE | Medium |

#### Reproducibility — MODERATE (65%)
**Present:** RDoF audit framework (60+ constants tracked), pipeline freeze/verify mechanism, LOYO protocol.
**Gap:** No experiment ledger, no dataset hashing, no artifact versioning, no MLflow/W&B integration.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S1-5 | Implement experiment registry with structured logging (see S3 improvements) | HIGH | Medium |
| S1-6 | Add dataset hashing (SHA-256) for all training data snapshots | HIGH | Low |

#### Safety Over Ambition — GOOD (75%)
**Present:** Stacking disabled, learned feature selection disabled, Optuna trials capped at 15.
**Gap:** No kill switch, circuit breaker, or degraded-mode fallback.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S1-7 | Implement pipeline circuit breaker for data quality failures | MODERATE | Medium |

---

### S2: Multi-Agent System Architecture — NOT IMPLEMENTED (15%)

The system is a monolithic pipeline (`sota.py`, 7,858 lines). No distinct agents exist.

**Partial equivalents:** Feature engineering modules, model training code, calibration pipeline, leverage optimizer, RDoF audit.

**Assessment:** For a single-domain Kaggle competition tool, a multi-agent architecture is not strictly necessary. However, the directive's **module boundary concepts** are valuable. The monolithic `sota.py` is the primary maintainability risk.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S2-1 | Execute the 5-phase refactoring in `docs/REFACTORING_ROADMAP.md` to decompose sota.py | HIGH | High |
| S2-2 | Define clean interface contracts between data, feature, model, and decision modules | MODERATE | Medium |

---

### S3: Shared Contracts and Required Logs — WEAK (35%)

**Present:** RDoF audit produces structured JSON, pipeline freeze generates config snapshots, materialization manifest records inputs, LOYO logs per-fold metrics.

**Missing:** No shared experiment ledger with the directive's schema (`problem_id`, `dataset_version`, `as_of_timestamp_rules`, `feature_set_id`, `model_family`, `hyperparameters`, `validation_scheme`, `calibration_method`, `decision_policy`, `primary_metric`, `secondary_metrics`, `path_risk_metrics`, `reproducibility_hash`, `experiment_timestamp`).

**Note:** `src/ml/evaluation/experiment_registry.py` exists but is minimal. It needs to be expanded to meet the directive's experiment record schema.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S3-1 | Expand `experiment_registry.py` to implement the full experiment record schema | HIGH | Medium |
| S3-2 | Auto-log every LOYO fold, hyperparameter run, and ablation test to the registry | HIGH | Medium |
| S3-3 | Add reproducibility hashes (code hash + data hash + config hash) per experiment | MODERATE | Low |

---

### S4: Problem Definition and Utility Mapping — STRONG (90%)

**Present:**
- Prediction target clearly defined: P(Team1 wins) for every possible NCAA tournament matchup
- Real optimization target: Brier score (Kaggle), pool EV (bracket contests)
- Action layer: Submit prediction CSV (Kaggle), generate bracket portfolio (pools)
- Operational constraints: Pre-tournament data only, single submission deadline

**Gap:** No formal `problem_summary`, `objective_verification_report`, or `constraints_register` artifacts.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S4-1 | Generate formal problem_summary artifact at pipeline start | LOW | Low |

---

### S5: Dataset Discovery, Construction, and Lineage — MODERATE (65%)

**Present:**
- 19 data scrapers covering diverse sources (Torvik, ESPN, Yahoo, SportsReference, cbbpy, Kaggle Massey, HerHoopStats)
- Historical pipeline supports 2005-2025 ingestion
- `TeamNameResolver` with 360+ D1 program aliases
- Data quality checks: outlier filtering, deduplication, schema validation
- Raw JSON snapshots preserved per season

**Missing:**
- No formal `dataset_catalog` artifact listing all sources with metadata
- No `dataset_lineage` tracing field-level availability timestamps
- No `availability_matrix` mapping feature→source→available_date
- No `dataset_expansion_report` documenting searched-but-rejected sources
- No formal survivorship bias testing
- Raw data overwritten on re-scrape (no versioned snapshots)

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S5-1 | Generate `dataset_catalog.json` listing all 19 sources with temporal availability | MODERATE | Medium |
| S5-2 | Implement versioned raw data snapshots (timestamp-named directories, never overwrite) | HIGH | Medium |
| S5-3 | Build availability matrix: which features are available at what lead time | MODERATE | Medium |
| S5-4 | Add survivorship bias test for player/team-level statistics | LOW | Medium |

---

### S6: Feature Discovery Engine — STRONG (80%)

**Present:**
- Temporal features: rolling means, streaks, momentum, exponentially weighted stats, recency decay
- Seasonal/calendar: rest days, back-to-back games, season progress
- Hierarchical: conference aggregates, SOS, quadrant-1 wins
- Interaction: matchup differentials (team1 - team2), seed×efficiency
- Representation: GNN schedule graph embeddings, transformer game sequences (optional)
- Feature acceptance: 0.001 Rule ablation, leakage checks, production availability constraint
- Feature catalog: `feature_dictionary` JSON with per-feature metadata
- 22 active features at inference from 77 total matchup dimensions

**Missing:**
- No formal `feature_stability_report` measuring feature importance consistency across seasons
- No `feature_retirement_log` documenting removed features and reasons
- Limited interaction features (differentials only; no entity×environment or signal×market)

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S6-1 | Generate per-season feature importance rankings and compute rank stability (Kendall tau across years) | MODERATE | Medium |
| S6-2 | Create feature retirement log tracking removed features with evidence | LOW | Low |
| S6-3 | Explore entity×environment interactions (team×venue-type, team×opponent-conference) | MODERATE | Medium |

---

### S7: Model Search and Meta-Learning — MODERATE (55%)

**Present:**
- Linear: Logistic regression baseline
- Tree ensembles: LightGBM, XGBoost (primary models)
- Neural: GNN schedule graph, Transformer game sequence (disabled by default)
- Bayesian: Bayesian Bradley-Terry rating system
- Spread: SpreadRegressor with logistic CDF conversion
- Optuna-based search (15 trials, 300s timeout) with temporal CV

**Missing:**
- No statistical time-series models (ARIMA, state space)
- No ranking/pairwise models (LambdaMART)
- No meta-learning layer (learn which approaches work by regime, sample size, horizon)
- Search space narrow — primarily tree ensembles with spread supplement
- No systematic search across objective functions, loss shaping, or training window length

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S7-1 | Add pairwise ranking model (LambdaMART via LightGBM ranker mode) | MODERATE | Medium |
| S7-2 | Systematically evaluate margin-first training (regression) vs classification objective for LGB/XGB | HIGH | Medium |
| S7-3 | Implement training window length search (how many historical years to include) | MODERATE | Low |
| S7-4 | Add meta-learning log: track which model family wins per fold/regime for cross-cycle learning | LOW | Medium |

---

### S8: Ensemble Optimization and Calibration — GOOD (75%)

**Present:**
- Fixed-weight averaging (primary): spread 0.50, LGB 0.15, XGB 0.15, Logistic ~0.20
- Stacking meta-learner available (disabled by default — correct for ~600 samples)
- Ensemble weight optimizer with L2 regularization toward uniform weights
- Diversity measured: component models use orthogonal signal sources
- Temperature scaling (primary, 1 parameter), Platt scaling, isotonic regression
- Bootstrap CI on temperature parameter (200 resamples)
- Multi-year calibration augmentation
- Diagnostics: ECE, MCE, Brier decomposition, reliability analysis

**Missing:**
- No greedy ensemble evaluation
- No formal diversity metric (e.g., pairwise correlation matrix of model errors)

**Risk:** Calibration `fit()` and `evaluate()` can operate on same data. No internal enforcement of train/calibration/test split.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S8-1 | Enforce internal train/calibration split in CalibrationPipeline (guard against same-data fit+eval) | CRITICAL | Low |
| S8-2 | Compute and log pairwise error correlation matrix across ensemble components | LOW | Low |
| S8-3 | Implement greedy forward ensemble selection as alternative to fixed weights | MODERATE | Medium |

---

### S9: Decision Optimization Layer — STRONG (85%)

**Present:**
- Kelly criterion for optimal bet sizing
- Pool-size-adaptive strategy profiles (Tiny/Small/Medium/Large)
- Payout structure adaptation (winner-take-all vs top-25%)
- Path-dependent EV with bracket covariance modeling
- Pareto frontier generation
- Abstention as first-class policy (minimum leverage thresholds)
- Friction terms: entry fees, house rake, multiple entries, opponent modeling
- Bracket portfolio generation (4 strategies: chalk, balanced, contrarian, targeted)

**Missing:**
- No formal `abstention_policy_report` artifact
- No `threshold_sweep_report` across multiple risk budgets (Pareto partially covers this)
- Opponent modeling limited to public pick percentages (not learned from historical pool data)

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S9-1 | Generate formal threshold_sweep_report at multiple risk/confidence levels | LOW | Low |
| S9-2 | Explore learning opponent behavior from historical ESPN bracket challenge data | LOW | High |

---

### S10: Backtesting and Simulation Realism — GOOD (70%)

**Present:**
- LOYO protocol simulates the actual prediction task
- Monte Carlo simulation with logit-space noise (std=0.12), injury modeling, regional correlation
- Friction terms in pool backtesting (scoring systems, payout structures, entry fees)
- Path-dependent bracket scoring
- Scenario sensitivity via variance targeting

**Missing:**
- No simulation of **information arrival timing** (backtests use final pre-tournament features, not features-as-available-at-decision-time)
- Regional correlation coefficients (1.0→0.6→0.3→0.15→0.0) acknowledged as under-validated
- No formal **path-dependent risk reporting** (drawdowns, losing streaks across seasons)
- No named **optimistic/base/pessimistic scenario analysis**

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S10-1 | Add path-dependent risk report: per-year Brier, max drawdown, losing streaks across LOYO folds | HIGH | Medium |
| S10-2 | Implement named scenario analysis (optimistic/base/pessimistic) with quantified assumptions | MODERATE | Medium |
| S10-3 | Validate regional correlation coefficients via historical tournament co-occurrence analysis | MODERATE | Medium |

---

### S11: Skeptical Audit Layer — MODERATE (55%)

**Present:**
- Leakage checks (`_leakage_checks()`) validate prior metrics
- Tournament start dates hardcoded per year
- Coach data gated by cutoff year
- LOYO prevents random k-fold misuse
- 0.001 Rule prevents selection on marginal features
- RDoF audit framework (60+ constants, tier classification)
- Bayesian shrinkage stabilizes small-sample features

**Missing:**
- No comprehensive feature availability audit for optional priors
- No formal robustness testing (missing features, thin-data regimes, distribution shift)
- No dataset hashes or frozen seeds logged per experiment
- No test for "selection on the test period" beyond RDoF retrospective

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S11-1 | Build robustness test suite: drop each top-10 feature, measure degradation | HIGH | Medium |
| S11-2 | Add thin-data regime test: evaluate model on seasons with <300 training games | MODERATE | Medium |
| S11-3 | Add distribution shift detection: compare feature distributions of training vs prediction year | MODERATE | Medium |

---

### S12: Codebase Review and Refactoring — MODERATE (65%)

**Present:**
- Clear module separation: data/, features/, ml/, pipeline/, optimization/, simulation/
- 77 test files (~24,000+ lines of test code)
- 16 CLI commands documented
- `TRAINING_DATA_AUDIT.md` documents 17 identified issues

**Issues:**
- **Hub module problem:** `sota.py` imports from 20+ modules and is 7,858 lines
- No shared `conftest.py` for pytest fixtures
- Circular import risk between pipeline and feature/model modules
- Dead code and commented-out blocks present
- No architecture decision records (ADRs)
- No changelog or semantic versioning

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S12-1 | Execute 5-phase decomposition of sota.py per `docs/REFACTORING_ROADMAP.md` | HIGH | High |
| S12-2 | Create shared `conftest.py` with common fixtures | MODERATE | Low |
| S12-3 | Remove confirmed dead code (disabled GNN/Transformer in non-optional paths) | MODERATE | Low |
| S12-4 | Add ADR directory for recording architectural decisions | LOW | Low |

---

### S13: Required Evaluation Matrix — GOOD (70%)

**Present:**

| Metric Class | Directive Requirement | Status |
|---|---|---|
| Predictive accuracy | RMSE, MAE, log loss, Brier, AUC | Brier, log loss, accuracy, AUC: **present** |
| Calibration | ECE, reliability, Brier decomposition | ECE, MCE, reliability, Brier decomp: **present** |
| Decision utility | EV, profit, bracket score | Pool EV, bracket score, ROI: **present** |
| Risk | Drawdown, volatility, worst-month, tail loss | Per-year Brier variance only: **partial** |
| Stability | Performance by period, regime, segment | Year-over-year OLS trend: **partial** |

**Missing:** Formal drawdown reporting, worst-case analysis, regime-conditional performance breakdown, tail loss metrics.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S13-1 | Add formal drawdown metric: worst consecutive-year Brier degradation | HIGH | Low |
| S13-2 | Add regime-conditional reporting: performance in upset-heavy vs chalk years | MODERATE | Medium |
| S13-3 | Compute tail-loss metric: Brier on the worst 10% of predictions per fold | MODERATE | Low |

---

### S14: Continuous Autonomous Research Loop — WEAK (20%)

**Missing almost entirely:**
- No automated hypothesis generation
- No experiment execution scheduler
- No adversarial review cycle automation
- No promotion gate automation
- No knowledge retention store

**Partial:** RDoF audit + LOYO validation serve as manual adversarial review. The 0.001 Rule acts as a manual promotion gate.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S14-1 | Build experiment runner script that automates LOYO + ablation + reporting in one command | MODERATE | Medium |
| S14-2 | Store experiment results in structured format for cross-cycle comparison | MODERATE | Medium |

---

### S15: Failure Modes — MODERATE (60%)

**Handled:** Temporal leakage (multi-layered), validation design (LOYO), improvement that vanishes after calibration, complexity increases (stacking disabled).

**Not handled:**
- Leakage checks log warnings but don't halt the pipeline
- No formal rejection of codebase changes that can't be validated

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S15-1 | Convert leakage check warnings to pipeline-halting errors (configurable via strict mode) | HIGH | Low |
| S15-2 | Add pre-merge validation gate: no code change merges without passing LOYO smoke test | MODERATE | Medium |

---

### S16: Final Deliverables — MODERATE (45%)

**Present (from directive checklist):**
- `final_system_report` → partially (LOYO results, no consolidated report)
- `prioritized_roadmap` → `PLAN_TOP1PCT.md`
- `dataset_and_model_registry` → partial (model configs logged, no formal registry)
- `decision_policy_recommendation` → leverage analysis output
- `known_risks_and_deferred_items` → `TRAINING_DATA_AUDIT.md`, `AGENT_DIRECTIVE_V7_AUDIT.md`

**Missing:** `success_criteria_evaluation`, formal `dataset_and_model_registry`, consolidated package.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S16-1 | Auto-generate consolidated `final_system_report.json` at pipeline completion | MODERATE | Medium |

---

### S17: Operating Summary — N/A (Narrative section, no actionable requirements)

---

### S18: Production Deployment and Live Monitoring — NOT IMPLEMENTED (5%)

This is the **single largest compliance gap**. The directive requires:
- Shadow mode, canary deployment, graduated rollout pipeline
- Real-time monitoring dashboard with 6 signal families
- Drift detection (concept, covariate, label drift)
- Automated retraining triggers (scheduled, performance-driven, drift-driven, event-driven)
- A/B testing framework

**Current state:** The system is a manually-invoked pipeline. `src/monitoring/pipeline_monitor.py` exists but provides only basic logging, not production monitoring.

**Assessment:** For a Kaggle competition tool that runs once per year, full production deployment infrastructure (shadow mode, canary, graduated rollout) is over-engineered. However, **monitoring of prediction quality and data freshness** is valuable even for annual use.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S18-1 | Implement data freshness monitoring: validate all sources are current before pipeline run | HIGH | Medium |
| S18-2 | Add prediction quality dashboard: compare current-year predictions to developing tournament results | MODERATE | Medium |
| S18-3 | Implement basic drift detection: compare current-year feature distributions to training-era | MODERATE | Medium |
| S18-4 | (Deferred) Full deployment pipeline with shadow/canary only if system becomes a service | LOW | Very High |

---

### S19: Data Engineering and Pipeline Resilience — WEAK (25%)

**Present:**
- Data quality checks in materialization
- Data validation in `src/data/ingestion/validators.py`
- Basic retry logic in scrapers

**Missing:**
- No DAG-based orchestrator (Airflow, Prefect, Dagster)
- No idempotency guarantees
- No schema contracts between pipeline stages
- No data freshness SLA registry
- No fault tolerance or circuit breakers
- No write-audit-publish pattern

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S19-1 | Add schema contracts (JSON Schema or Pydantic) for data passed between pipeline stages | HIGH | Medium |
| S19-2 | Implement idempotent data pipeline tasks (check-before-write, skip if unchanged) | MODERATE | Medium |
| S19-3 | Add circuit breakers to all scrapers: fail after 3 retries, serve cached data with staleness flag | MODERATE | Medium |
| S19-4 | Document data freshness SLAs per source in a `freshness_sla.json` config | LOW | Low |

---

### S20: Computational Budget and Resource Prioritization — NOT IMPLEMENTED (5%)

**Present:** Optuna trial count (15) and timeout (300s) serve as implicit resource constraints.

**Missing:** Everything else — no budget framework, phase allocation, cost tracking, search termination criteria, Pareto frontier of compute vs performance.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S20-1 | Add wall-clock timing to each pipeline phase and log to experiment registry | MODERATE | Low |
| S20-2 | Implement early termination for Optuna trials that show no improvement after N trials | LOW | Low |

---

### S21: Human-in-the-Loop Governance — WEAK (15%)

**Present:** Pipeline mode gating, `require_freeze_file` flag, RDoF transparency, CI/CD tests.

**Missing:** Decision authority matrix, approval protocols, compliance checkpoints, governance audit trail.

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S21-1 | Document decision authority levels for pipeline operations in a governance config | LOW | Low |
| S21-2 | Add confirmation prompt before generating Kaggle submission (human gate) | LOW | Low |

---

### S22: Multi-Agent Conflict Resolution — N/A

System is not multi-agent. No improvements needed.

---

### S23: Testing Strategy and CI/CD — MODERATE (50%)

**Present:**

| Layer | Status | Details |
|-------|--------|---------|
| Unit tests | Good (77 files) | Covers features, models, calibration, optimization |
| Integration tests | Partial | `test_sota_pipeline.py`, `test_full_ml_pipeline.py` exist but limited |
| System tests | Partial | End-to-end backtest tests exist, not scheduled nightly |
| Property-based tests | Partial | Data consistency tests, date integrity, but no Hypothesis library |
| Regression tests | Good | Data audit issues tracked; `test_leakage_canary.py` exists |

**Temporal Integrity Tests:**

| Directive Test | Status |
|---|---|
| Feature timestamp assertion | Present (leakage checks in materialization.py) |
| Walk-forward replay test | `test_walk_forward_replay.py` exists |
| Data leakage canary | `test_leakage_canary.py` exists |
| Pipeline ordering test | Missing |

**CI/CD Pipeline:**

| Stage | Status |
|---|---|
| Pre-commit (lint) | Ruff only (E, F, W). No type checking, no formatting enforcement |
| Unit tests | pytest in CI (`deploy-with-secrets.yml`) |
| Coverage gate | 40% threshold — **too low** for production code |
| Model validation smoke | Missing |
| Nightly system tests | Missing |

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S23-1 | Raise coverage threshold from 40% to 60%, then incrementally to 75% | HIGH | Medium |
| S23-2 | Add type checking (mypy or pyright) to CI pipeline | HIGH | Medium |
| S23-3 | Add model validation smoke test to CI: train minimal model on fixture data, verify output schema | MODERATE | Medium |
| S23-4 | Add pipeline determinism test: run same input twice, verify identical output | MODERATE | Medium |
| S23-5 | Create shared `conftest.py` with reusable fixtures for team data, feature vectors | MODERATE | Low |

---

### S24: Domain-Specific (Sports Betting) — STRONG (85%)

**Present:**
- Injury reports acknowledged as unreliable, multi-source
- Public pick percentages as market proxy
- Small sample sizes recognized (63 tournament games/year), multi-year pooling
- Survivorship bias handled (2020 excluded, team aggregates only)
- Regional correlation in MC simulation
- Neutral site adjustment (tournament shrinkage, neutral-site record feature)
- Home-court dependence computed as feature

**Gaps:**
- No explicit closing line value modeling
- Line movement not directly modeled
- No jurisdictional compliance checks (not relevant for Kaggle)

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S24-1 | Add explicit documentation of domain-specific data quirks and timing assumptions | LOW | Low |

---

### S25: Extended Failure Modes and Consolidated Deliverables — WEAK (30%)

**Additional failure modes from S25 not currently detected:**
- Deployment bypassing shadow/canary → N/A (no deployment pipeline)
- Production without monitoring → Applicable — system runs blind
- Pipeline lacking idempotency → Applicable
- Compute budget overrun undetected → Applicable
- Code merged without CI/CD gates → Partially handled (CI runs but coverage gate is low)

| ID | Action | Priority | Effort |
|----|--------|----------|--------|
| S25-1 | Implement pre-run validation checklist that verifies data freshness, feature completeness, and model readiness | HIGH | Medium |
| S25-2 | Block pipeline execution if any CRITICAL-severity check fails | MODERATE | Low |

---

## 4. Gap Analysis Summary

### By Severity

| Severity | Count | Key Items |
|----------|-------|-----------|
| **CRITICAL** | 3 | S1-1 (cutoff_date default), S8-1 (calibration leakage), S1-5/S3-1 (experiment registry) |
| **HIGH** | 16 | S1-2, S1-6, S2-1, S5-2, S7-2, S10-1, S11-1, S12-1, S13-1, S15-1, S18-1, S19-1, S23-1, S23-2, S25-1, S3-2 |
| **MODERATE** | 22 | Feature stability, robustness tests, schema contracts, scenario analysis, etc. |
| **LOW** | 12 | Meta-learning, ADRs, governance docs, opponent modeling, etc. |

### By Effort

| Effort | Count | Examples |
|--------|-------|---------|
| **Low** (hours) | 16 | S1-1, S1-3, S1-6, S8-1, S13-1, S15-1, S20-1, S23-5 |
| **Medium** (days) | 24 | S3-1, S5-1, S7-2, S10-1, S11-1, S18-1, S19-1, S23-1 |
| **High** (weeks) | 3 | S2-1/S12-1 (sota.py decomposition), S18-4 (deployment pipeline) |

### Top 10 Improvements by Impact-to-Effort Ratio

| Rank | ID | Action | Priority | Effort | Impact |
|------|-----|--------|----------|--------|--------|
| 1 | S1-1 | Make `cutoff_date` required in ProprietaryMetricsEngine | CRITICAL | Low | Eliminates leakage risk |
| 2 | S8-1 | Enforce internal calibration train/test split | CRITICAL | Low | Eliminates calibration leakage |
| 3 | S15-1 | Convert leakage warnings to pipeline-halting errors | HIGH | Low | Prevents silent leakage |
| 4 | S1-6 | Add dataset hashing for training data snapshots | HIGH | Low | Enables reproducibility |
| 5 | S13-1 | Add formal drawdown metric | HIGH | Low | Completes risk reporting |
| 6 | S13-3 | Compute tail-loss metric on worst 10% predictions | MODERATE | Low | Identifies failure modes |
| 7 | S20-1 | Add wall-clock timing per pipeline phase | MODERATE | Low | Enables budget tracking |
| 8 | S3-1 | Expand experiment_registry.py to full schema | HIGH | Medium | Enables reproducibility |
| 9 | S7-2 | Evaluate margin-first vs classification training | HIGH | Medium | Potential Brier improvement |
| 10 | S10-1 | Add path-dependent risk report | HIGH | Medium | Completes risk assessment |

---

## 5. Prioritized Improvement Plan

### Tier 1: Critical Fixes (Must Do — Integrity at Stake)

These address fundamental correctness and leakage risks. Do before any tournament predictions.

| # | Action | Files Affected | Est. Hours |
|---|--------|---------------|------------|
| 1 | **S1-1:** Make `cutoff_date` required in `ProprietaryMetricsEngine.__init__()` — remove `None` default, require all callers to pass it explicitly | `src/data/features/proprietary_metrics.py`, all callers in `sota.py` | 2-3 |
| 2 | **S8-1:** Add internal guard in `CalibrationPipeline` to prevent `fit()` and `evaluate()` on overlapping data — assert no sample overlap via index check | `src/ml/calibration/calibration.py` | 2-3 |
| 3 | **S15-1:** Add `strict_leakage_mode` config flag. When True, leakage check failures raise `LeakageError` and halt pipeline instead of logging warnings | `src/data/features/materialization.py`, `sota.py` | 2-3 |

### Tier 2: High-Impact Improvements (Should Do — Significant Quality Gains)

| # | Action | Files Affected | Est. Hours |
|---|--------|---------------|------------|
| 4 | **S1-6:** Add SHA-256 hashing for all training data files; log hashes to experiment registry | `src/data/loader.py`, `experiment_registry.py` | 3-4 |
| 5 | **S3-1/S3-2:** Expand `experiment_registry.py` with full directive schema; auto-log LOYO folds, hyperparameter runs, ablation tests | `src/ml/evaluation/experiment_registry.py`, `sota.py`, `loyo_protocol.py` | 8-12 |
| 6 | **S1-2:** Add temporal availability validation for all optional prior sources — validate each source's timestamp is pre-tournament | `sota.py` (`_load_optional_prior_sources`), `src/data/scrapers/` | 4-6 |
| 7 | **S5-2:** Implement versioned raw data snapshots — timestamp-named directories, never overwrite | `src/data/ingestion/historical_pipeline.py`, `collector.py` | 4-6 |
| 8 | **S7-2:** Evaluate margin-first (regression) training for LGB/XGB alongside current classification; compare LOYO Brier | `src/pipeline/sota.py` (`_train_baseline_model`), spread_model.py | 6-8 |
| 9 | **S10-1:** Generate path-dependent risk report: per-year Brier, cumulative drawdown, max losing streak, worst-season analysis | `src/ml/evaluation/`, new `risk_report.py` | 6-8 |
| 10 | **S11-1:** Build robustness test suite: systematically drop each top-10 feature, measure Brier degradation per fold | `src/ml/evaluation/ablation.py`, new test file | 6-8 |
| 11 | **S13-1:** Add drawdown metric + tail-loss metric (worst 10% predictions) to standard evaluation output | `src/ml/evaluation/tournament_metrics.py` | 3-4 |
| 12 | **S18-1:** Implement data freshness validation — check all sources are within acceptable staleness before pipeline execution | `src/monitoring/pipeline_monitor.py`, `sota.py` | 4-6 |
| 13 | **S19-1:** Add Pydantic schema contracts for inter-stage data (team features, matchup vectors, predictions) | New `src/data/schemas.py`, update pipeline stages | 6-8 |
| 14 | **S23-1:** Raise CI coverage threshold from 40% → 60% (immediate), plan path to 75% | `pyproject.toml`, new test files as needed | 8-12 |
| 15 | **S23-2:** Add mypy type checking to CI pipeline with gradual strictness | `.github/workflows/deploy-with-secrets.yml`, `pyproject.toml` | 4-6 |
| 16 | **S25-1:** Implement pre-run validation checklist that gates pipeline execution on data quality | `sota.py` (new `_pre_run_validation()` method) | 4-6 |

### Tier 3: Moderate Improvements (Should Address — Robustness & Completeness)

| # | Action | Est. Hours |
|---|--------|------------|
| 17 | **S2-1/S12-1:** Begin Phase 1 of sota.py decomposition (extract data loading ~300 lines) | 8-12 |
| 18 | **S5-1:** Generate `dataset_catalog.json` listing all 19 sources with metadata | 4-6 |
| 19 | **S6-1:** Feature stability report: per-season importance rankings with Kendall tau | 4-6 |
| 20 | **S10-2:** Named scenario analysis (optimistic/base/pessimistic) with quantified assumptions | 4-6 |
| 21 | **S10-3:** Validate regional correlation coefficients via historical co-occurrence analysis | 4-6 |
| 22 | **S11-2:** Thin-data regime testing: evaluate on seasons with <300 training games | 3-4 |
| 23 | **S11-3:** Distribution shift detection: compare feature distributions of training vs prediction year | 4-6 |
| 24 | **S14-1:** Build unified experiment runner script (LOYO + ablation + reporting in one command) | 6-8 |
| 25 | **S19-2:** Idempotent data pipeline tasks (check-before-write) | 4-6 |
| 26 | **S19-3:** Circuit breakers for all scrapers (fail after 3 retries, serve cached data) | 4-6 |
| 27 | **S23-3:** Model validation smoke test in CI | 4-6 |
| 28 | **S23-4:** Pipeline determinism test (same input → identical output) | 3-4 |
| 29 | **S1-7:** Pipeline circuit breaker for data quality failures | 4-6 |
| 30 | **S8-3:** Greedy forward ensemble selection as alternative to fixed weights | 6-8 |

### Tier 4: Low Priority (Nice to Have — Polish & Governance)

| # | Action | Est. Hours |
|---|--------|------------|
| 31 | **S4-1:** Auto-generate problem_summary artifact | 2-3 |
| 32 | **S6-2:** Feature retirement log | 1-2 |
| 33 | **S7-4:** Meta-learning log per fold/regime | 4-6 |
| 34 | **S9-1:** Threshold sweep report | 2-3 |
| 35 | **S12-4:** ADR directory | 1-2 |
| 36 | **S16-1:** Consolidated final_system_report.json generation | 4-6 |
| 37 | **S19-4:** Freshness SLA config file | 1-2 |
| 38 | **S20-2:** Optuna early termination for diminishing returns | 2-3 |
| 39 | **S21-1:** Governance config documentation | 2-3 |
| 40 | **S24-1:** Domain quirks documentation | 2-3 |

---

## 6. Phased Implementation Roadmap

### Phase 1: Critical Integrity Fixes (1-2 days)
**Goal:** Eliminate all known leakage and data integrity risks.

- [ ] S1-1: Make `cutoff_date` required in ProprietaryMetricsEngine
- [ ] S8-1: Guard CalibrationPipeline against same-data fit+evaluate
- [ ] S15-1: Add strict leakage mode with pipeline-halting errors
- [ ] S1-6: Add dataset hashing for training data snapshots

**Exit criterion:** All 4 items pass new unit tests. No pipeline run can proceed with leakage risk.

### Phase 2: Experiment Infrastructure (2-3 days)
**Goal:** Enable reproducible, comparable experiments.

- [ ] S3-1: Expand experiment registry to full directive schema
- [ ] S3-2: Auto-log LOYO folds and hyperparameter runs
- [ ] S20-1: Add wall-clock timing per pipeline phase
- [ ] S10-1: Generate path-dependent risk report
- [ ] S13-1: Add drawdown and tail-loss metrics

**Exit criterion:** Every LOYO run produces a structured experiment record with metrics, timings, and data hashes.

### Phase 3: Model & Feature Improvements (3-5 days)
**Goal:** Close the prediction quality gap.

- [ ] S7-2: Evaluate margin-first training for LGB/XGB
- [ ] S11-1: Build robustness test suite (feature dropout)
- [ ] S6-1: Feature stability report across seasons
- [ ] S1-2: Temporal validation for optional prior sources

**Exit criterion:** LOYO Brier improves or stays within CI; robustness profile documented.

### Phase 4: Pipeline Hardening (3-5 days)
**Goal:** Make the pipeline resilient and well-tested.

- [ ] S18-1: Data freshness validation before pipeline runs
- [ ] S19-1: Pydantic schema contracts for inter-stage data
- [ ] S25-1: Pre-run validation checklist
- [ ] S23-1: Raise CI coverage to 60%
- [ ] S23-2: Add mypy to CI

**Exit criterion:** Pipeline rejects stale/malformed data. CI catches type errors and coverage regressions.

### Phase 5: Codebase Quality (5-8 days, can be ongoing)
**Goal:** Reduce technical debt and improve maintainability.

- [ ] S12-1: Phase 1 of sota.py decomposition (extract data loading)
- [ ] S12-2: Create conftest.py with shared fixtures
- [ ] S12-3: Remove confirmed dead code
- [ ] S23-3: Model validation smoke test in CI
- [ ] S23-4: Pipeline determinism test

**Exit criterion:** sota.py reduced by ≥300 lines. All CI gates green.

### Phase 6: Reporting & Documentation (2-3 days)
**Goal:** Generate directive-compliant artifacts.

- [ ] S5-1: Dataset catalog
- [ ] S10-2: Named scenario analysis
- [ ] S14-1: Unified experiment runner script
- [ ] S16-1: Consolidated final system report generation
- [ ] S24-1: Domain quirks documentation

**Exit criterion:** Pipeline auto-generates all required directive artifacts.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `cutoff_date=None` causes tournament data contamination in features | High (default value is None) | Critical (leakage) | S1-1: Make required parameter |
| Calibration fit on same data as evaluation inflates performance | Medium (depends on orchestration) | Critical (false confidence) | S8-1: Internal guard |
| sota.py decomposition breaks existing functionality | Medium | High | Phase 5: Pre-refactor test suite, atomic changes, feature-branch isolation |
| Raising coverage threshold blocks legitimate PRs | Low | Medium | Incremental: 40→50→60→75 over multiple sprints |
| Margin-first training degrades Brier on some folds | Medium | Medium | S7-2: Keep both paths, promote only if wins on majority of LOYO folds |
| Optional prior sources introduce subtle temporal leakage | Medium | High | S1-2: Explicit timestamp validation per source |
| Historical data quality issues (2005-2009) distort training | Medium | Medium | Existing year-decay weighting partially mitigates; S6.1 in PLAN_TOP1PCT adds quality weights |

---

## 8. Success Criteria

### Short-term (Post Phase 1-2)
- [ ] Zero known temporal leakage paths
- [ ] Every experiment logged with full schema (data hash, config, metrics)
- [ ] Path-dependent risk metrics reported for all LOYO folds
- [ ] CI catches type errors and coverage regressions

### Medium-term (Post Phase 3-4)
- [ ] LOYO Brier improves by ≥0.003 from margin-first training or stays within CI
- [ ] Robustness profile: no single feature removal degrades Brier by >0.01
- [ ] Pipeline auto-rejects stale data and schema violations
- [ ] Pydantic contracts enforce data shape at every pipeline boundary

### Long-term (Post Phase 5-6)
- [ ] sota.py reduced to <3,000 lines with clean module boundaries
- [ ] CI coverage ≥60% (path to 75%)
- [ ] All directive-required artifacts auto-generated
- [ ] Directive compliance rises from ~48% to ≥70%

---

## Appendix A: Cross-Reference to Existing Documents

| Document | Relationship | Key Additions in This Plan |
|----------|-------------|---------------------------|
| `AGENT_DIRECTIVE_V7_AUDIT.md` | Prior audit with findings C1-C4, H1-H6, M1-M8, L1-L6 | This plan adds: concrete implementation actions with file paths, phased roadmap, success criteria, risk register, effort estimates |
| `REVIEW_AND_IMPROVEMENTS.md` | Kaggle Grandmaster review with 7 priorities | This plan operationalizes priorities 1-7 as directive-compliant actions |
| `PLAN_TOP1PCT.md` | Technical plan for 7 gaps | This plan adds directive compliance dimension and orders by integrity risk, not just Brier improvement |
| `TRAINING_DATA_AUDIT.md` | 17 data quality issues | Incorporated into S5 and S19 assessments |
| `docs/REFACTORING_ROADMAP.md` | 5-phase sota.py decomposition | Referenced in S12-1, scheduled for Phase 5 |

## Appendix B: Directive Deliverable Checklist

| Directive Deliverable | Status | Action to Produce |
|---|---|---|
| `final_system_report` | Partial | S16-1 |
| `prioritized_roadmap` | Present (`PLAN_TOP1PCT.md`) | Update post-implementation |
| `dataset_and_model_registry` | Partial | S3-1, S5-1 |
| `decision_policy_recommendation` | Present (leverage output) | — |
| `known_risks_and_deferred_items` | Present (this document) | — |
| `success_criteria_evaluation` | Present (Section 8 above) | — |
| `deployment_pipeline_config` | Missing | S18-4 (deferred) |
| `monitoring_dashboard_spec` | Missing | S18-2 |
| `alert_threshold_registry` | Missing | S18-1 (partial) |
| `drift_detection_baseline` | Missing | S18-3 |
| `retraining_schedule` | N/A (annual) | — |
| `ab_test_protocol` | Missing | Deferred |
| `rollback_runbook` | Missing | Deferred |
| `pipeline_dag_spec` | Missing | S19 improvements |
| `schema_registry` | Missing | S19-1 |
| `freshness_sla_registry` | Missing | S19-4 |
| `fault_tolerance_runbook` | Missing | S19-3 |
| `compute_budget_plan` | Missing | S20-1 (partial) |
| `cost_efficiency_report` | Missing | S20-1 (partial) |
| `decision_authority_matrix` | Missing | S21-1 |
| `governance_audit_trail` | Missing | Deferred |
| `compliance_checklist` | N/A | — |
| `conflict_resolution_protocol` | N/A | — |
| `dissent_registry` | N/A | — |
| `test_coverage_report` | Present (CI) | S23-1 (improve) |
| `ci_cd_pipeline_config` | Present (.github/workflows/) | S23-2 (extend) |
| `domain_integration_guide` | Partial | S24-1 |
