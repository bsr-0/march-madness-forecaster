# Agent Directive V7 Audit Report

## March Madness Forecaster — Comprehensive Evaluation

**Audit Date:** 2026-03-07
**Directive Version:** Agent Directive V7 Complete
**Repository:** march-madness-forecaster
**Methodology:** Systematic evaluation against all 25 sections of Agent Directive V7

---

## Executive Summary

The march-madness-forecaster is a **sophisticated, research-grade NCAA tournament prediction system** with strong academic foundations. It implements a multi-model ensemble (LightGBM, XGBoost, spread regression, Bayesian Bradley-Terry), 100+ engineered features with Bayesian regularization, Monte Carlo bracket simulation, and game-theoretic pool optimization. The codebase demonstrates unusual rigor in several areas — particularly temporal integrity, researcher degrees-of-freedom auditing, and statistical validation — while showing significant gaps in production deployment, monitoring, and formal governance.

**Overall Compliance: ~55% of Directive V7 requirements fully met.**

| Directive Area | Compliance | Grade |
|---|---|---|
| Temporal Integrity (S1, S5) | Strong | A- |
| Data Lineage & Construction (S5) | Moderate | B |
| Feature Discovery (S6) | Strong | A- |
| Model Search (S7) | Moderate | B- |
| Ensemble & Calibration (S8) | Good | B+ |
| Decision Optimization (S9) | Strong | A |
| Backtesting Realism (S10) | Good | B+ |
| Skeptical Audit (S11) | Moderate | B |
| Codebase Quality (S12) | Good | B+ |
| Evaluation Matrix (S13) | Good | B+ |
| Continuous Research Loop (S14) | Weak | C- |
| Failure Mode Rejection (S15) | Moderate | B |
| Deployment & Monitoring (S18) | Absent | F |
| Data Engineering & Pipelines (S19) | Weak | D+ |
| Compute Budget (S20) | Absent | F |
| Human-in-the-Loop Governance (S21) | Weak | D |
| Conflict Resolution (S22) | N/A | N/A |
| Testing & CI/CD (S23) | Moderate | B- |
| Domain-Specific (S24 Sports) | Strong | A |

---

## Part I — Core Research and Validation

### Section 1: Mission and Non-Negotiable Principles

**Temporal integrity first** — **STRONG COMPLIANCE**

The system demonstrates exceptional awareness of temporal leakage:

- `TOURNAMENT_START_DATES` dictionary (sota.py:90-100) hardcodes per-year cutoff dates for NCAA tournament start
- Feature materialization uses `shift(1).expanding().mean()` to compute prior metrics, preventing same-game leakage (materialization.py:534-600)
- Proprietary metrics engine accepts `cutoff_date` parameter to filter games (proprietary_metrics.py:355-365)
- Explicit leakage checks validate prior metrics match expected temporal patterns (materialization.py:911-941)
- `FIX-LEAKAGE-POLLS` flag excludes post-tournament poll aggregates; only preseason polls allowed
- 2020 COVID year excluded entirely (no tournament occurred)
- Coach tournament data gated by `coach_data_cutoff_year` to prevent future data leakage in LOYO backtest

**Findings:**
- **RISK:** `cutoff_date` defaults to `None` in the proprietary metrics engine — if callers forget to pass it, all games (including tournament games) are used. This should be a required parameter, not optional.
- **RISK:** Optional prior sources (market odds, transfer portal, weather) loaded without explicit temporal filtering in `_load_optional_prior_sources()`.
- **RISK:** When synthetic date inference occurs (game_id ordering), rest_days and back_to_back features become degenerate (NaN). This is handled but not logged as a data quality warning in downstream models.

**Decision objective supremacy** — **STRONG COMPLIANCE**

The system correctly identifies the real optimization target: Brier score (Kaggle's actual metric since 2023) and pool Expected Value (for bracket contests). The `SOTAPipelineConfig.scoring_metric` field explicitly tracks this (sota.py:164). The decision layer (leverage.py) separates prediction quality from decision quality via Kelly criterion, payout-structure-aware variance targeting, and pool-size-adaptive strategy profiles.

**Reproducibility** — **GOOD COMPLIANCE**

- RDoF audit framework catalogs 60+ hand-tuned constants with tier classification (rdof_audit.py)
- Pipeline freeze/verify mechanism for pre-registration discipline
- LOYO protocol documents exact validation years and exclusions
- **GAP:** No formal experiment ledger, dataset hashing, or artifact versioning beyond git. No MLflow, Weights & Biases, or equivalent experiment tracking.

**Evidence over intuition** — **GOOD COMPLIANCE**

- 0.001 Rule: features must improve mean LOYO Brier by >= 0.001 or be deleted (loyo_protocol.py:30)
- Feature ablation engine systematically tests each feature's contribution (loyo_protocol.py:304-447)
- Statistical significance tests (paired Brier t-test, permutation test, bootstrap comparison) guard against noise
- **GAP:** Some Tier 3 constants (e.g., round correlation decay in MC simulation) cite academic sources but acknowledge wide confidence intervals without formal sensitivity analysis in the codebase.

**Safety over ambition** — **MODERATE COMPLIANCE**

- Stacking disabled by default because it overfits on ~400 samples (sota.py:173)
- Learned feature selection disabled; fixed domain-knowledge set used instead (sota.py:223-227)
- Optuna trials reduced from 50 to 15 to prevent selection bias (sota.py:156)
- **GAP:** No formal kill switch, circuit breaker, or degraded-mode fallback if data pipelines fail.

---

### Section 2: Multi-Agent System Architecture

**NOT IMPLEMENTED**

The system operates as a monolithic pipeline (src/pipeline/sota.py), not a multi-agent architecture. There is no Research Orchestrator, Data Agent, Feature Agent, Model Agent, Ensemble Agent, Decision Agent, or Audit Agent as distinct processes or modules with defined interfaces.

**Partial equivalents exist:**
- Feature engineering → `src/data/features/`
- Model training → `src/ml/ensemble/cfa.py`
- Calibration → `src/ml/calibration/calibration.py`
- Decision optimization → `src/optimization/leverage.py`
- Audit → `src/ml/evaluation/rdof_audit.py` (RDoF audit only)

**Recommendation:** Not necessarily a problem for a single-domain system. The directive's multi-agent architecture is designed for large autonomous research labs. For a focused tournament prediction system, the monolithic architecture is appropriate if internal module boundaries are clean.

---

### Section 3: Shared Contracts and Required Logs

**PARTIALLY IMPLEMENTED**

- RDoF audit produces structured JSON reports with constant registries
- Pipeline freeze generates config snapshot artifacts
- Materialization manifest records input sources, leakage checks, and quality reports
- LOYO validation logs per-fold metrics

**Missing:**
- No shared experiment ledger with the schema specified in the directive (problem_id, dataset_version, as_of_timestamp_rules, feature_set_id, etc.)
- No experiment registry across runs
- No artifact versioning or reproducibility hashes across experiments

---

### Section 5: Dataset Discovery, Construction, and Lineage

**GOOD — WITH GAPS**

**Strengths:**
- 17+ data scrapers covering diverse sources (Torvik, ESPN, Yahoo, SportsReference, cbbpy)
- Historical pipeline supports multi-season ingestion (2005-2025)
- Team name resolution via `TeamNameResolver` with 360+ D1 program aliases
- Data quality checks: outlier filtering (zero scores, margins > 80), deduplication, schema validation
- Field-level availability tracking for tournament window (March 13 - April 15)
- Raw JSON snapshots preserved per season

**Missing (per directive):**
- No formal `dataset_catalog` artifact
- No explicit `dataset_lineage` tracing field-level availability timestamps
- No `availability_matrix` or `dataset_expansion_report`
- No formal survivorship bias testing (beyond 2020 COVID exclusion)
- No versioning of raw snapshots with revision tracking (raw data overwritten on re-scrape)

---

### Section 6: Feature Discovery Engine

**STRONG**

**Feature families implemented (per directive categories):**

| Directive Category | Implementation | Features |
|---|---|---|
| Temporal features | Yes | Rolling means, streaks, momentum (last-5/last-10), exponentially weighted stats, recency decay |
| Seasonal/calendar | Yes | Rest days, back-to-back games, games in last 7 days, season progress |
| Hierarchical | Yes | Conference aggregates, SOS adjustment (15-iteration convergent), quadrant-1 wins |
| Interaction | Partial | Matchup differentials (team1 - team2), seed×efficiency interactions |
| Representation | Yes | GNN schedule graph embeddings, transformer game sequence embeddings (optional) |

**Feature acceptance rules (per directive):**
- Stable importance: 0.001 Rule ablation (LOYO Brier improvement threshold)
- No impossible information: Leakage checks, cutoff_date filtering
- Production availability: Features computable from pre-tournament data only
- Low revision risk: Preseason AP polls only (post-tournament polls excluded)
- Walk-forward validation: LOYO protocol

**Feature catalog produced:** Yes — `feature_dictionary` JSON with per-feature metadata.

**Missing:**
- No formal `feature_stability_report` across seasons
- No `feature_retirement_log`
- Interaction features limited to differentials; no explicit entity×environment or signal×market interactions

---

### Section 7: Model Search and Meta-Learning

**MODERATE — NARROW SEARCH SPACE**

**Models implemented:**

| Directive Family | Implemented | Details |
|---|---|---|
| Linear/generalized | Yes | Logistic regression (baseline) |
| Tree ensembles | Yes | LightGBM, XGBoost (primary models) |
| Statistical time-series | No | — |
| Neural sequence | Optional | Transformer game sequence model, GNN schedule graph (disabled by default) |
| Ranking/pairwise | No | — |
| Bayesian | Yes | Bayesian Bradley-Terry (rating system) |
| Regression-to-probability | Yes | Spread regression → logistic CDF conversion |

**Hyperparameter tuning:**
- Optuna-based search (15 trials, 300s timeout) with temporal CV
- Searches learning rate, depth, regularization, feature fractions
- Separate tuners for LightGBM, XGBoost, and Logistic
- **GAP:** No meta-learning layer. No learning of which feature families, models, or calibration methods perform best by regime or sample size.

**Key concern:** The model search space is narrow (primarily tree ensembles). The directive requires searching "across diverse model families" and the system largely relies on LightGBM + XGBoost with a spread regression supplement. Neural models and ranking models are either disabled or absent.

---

### Section 8: Ensemble Optimization and Calibration

**GOOD**

**Ensemble methods:**
- Fixed-weight averaging (primary): LGB 0.15, XGB 0.15, Spread 0.50, Logistic ~0.20
- Stacking meta-learner available but disabled by default (overfitting risk with ~400 samples)
- Ensemble weight optimizer with L2 regularization toward uniform weights
- **Diversity measured:** Component models use orthogonal signal sources (feature-based classifiers, margin regression, ID-based ratings)

**Calibration:**
- Temperature scaling (primary, 1 parameter — robust for small samples)
- Platt scaling and isotonic regression available with sample-size guards
- Bootstrap CI on temperature parameter (200 resamples, reverts to T=1.0 if CI includes 1.0)
- Multi-year calibration augmentation to expand calibration sample pool
- Round-weighted calibration matching Kaggle's actual scoring metric

**Calibration diagnostics:**
- ECE, MCE, Brier decomposition, per-bin reliability analysis
- ROC-AUC for discrimination quality
- Bootstrap CIs on Brier scores

**Risk:** Calibration `fit()` and `evaluate()` methods can operate on the same data. No internal enforcement of train/calibration/test separation — depends on external orchestration (LOYO protocol) to prevent leakage.

---

### Section 9: Decision Optimization Layer

**STRONG — BEST-IN-CLASS FOR DOMAIN**

This is a standout area of the system:

- **Kelly criterion** for optimal bet sizing with payout structure awareness
- **Pool-size-adaptive strategy profiles:** Tiny (<30), Small (30-100), Medium (101-1000), Large (1000+) with graduated contrarian strength
- **Payout structure adaptation:** Winner-take-all vs top-25% adjusts variance targeting
- **Path-dependent EV:** Properly models covariance in bracket scoring (later-round points require earlier-round wins)
- **Pareto frontier generation** along risk/reward axis
- **Abstention as first-class policy:** Minimum leverage thresholds prevent action on low-edge picks
- **Friction terms modeled:** Entry fees, house rake, multiple entries, opponent modeling via public pick percentages
- **Bracket portfolio generation:** Multi-strategy portfolio (chalk, balanced, contrarian, targeted) for Kaggle format

**Missing:**
- No formal `abstention_policy_report`
- Opponent modeling limited to public pick percentages and archetypal behavioral models (not learned from historical pool data)
- No threshold sweep report across multiple risk budgets (Pareto frontier partially addresses this)

---

### Section 10: Backtesting and Simulation Realism

**GOOD — WITH CAVEATS**

**Strengths:**
- LOYO protocol simulates the actual prediction task (train on all years except held-out, predict that tournament)
- Monte Carlo simulation includes logit-space noise (std=0.12), injury modeling, and regional correlation
- Friction terms in pool backtesting: scoring systems, payout structures, entry fees
- Path-dependent bracket scoring correctly modeled
- Scenario sensitivity via variance targeting (optimistic/pessimistic) in bracket portfolio

**Missing (per directive):**
- No explicit simulation of **information arrival timing** — backtests use final pre-tournament features, not features as they would have been available at decision time (e.g., mid-week vs game-day injury reports)
- Regional correlation coefficients (1.0 → 0.6 → 0.3 → 0.15 → 0.0) acknowledged as under-validated with "wide CIs that cannot distinguish between e.g. 0.6 and 0.3"
- No formal **path-dependent risk reporting** (drawdowns, losing streaks across seasons)
- No **optimistic/base/pessimistic scenario analysis** as separate named scenarios

---

### Section 11: Skeptical Audit Layer

**MODERATE**

**Leakage audit:** Partially implemented
- `_leakage_checks()` validates prior metrics don't use current game data
- Tournament start dates hardcoded per year
- Coach data gated by cutoff year
- **GAP:** No comprehensive feature availability audit. Optional priors (market odds, transfer portal) not validated for temporal availability.

**Validation audit:** Good
- LOYO protocol prevents random k-fold misuse
- 0.001 Rule prevents selection on marginal features
- RDoF audit framework tracks tuning-evaluation circularity with explicit disclosures
- **GAP:** No test for "selection on the test period" beyond RDoF's retrospective diagnostic level

**Robustness audit:** Partial
- Bayesian shrinkage stabilizes small-sample features
- Tournament domain adaptation (shrinkage toward 0.5)
- **GAP:** No formal robustness testing under missing features, thin-data regimes, or changed distributions

**Reproducibility audit:** Partial
- Pipeline freeze/verify for config snapshots
- Feature set hashing
- **GAP:** No dataset hashes, no frozen seeds logged per experiment, no artifact versioning

---

### Section 12: Codebase Review and Refactoring

**GOOD**

- Clear module separation: data/, features/, ml/, pipeline/, optimization/, simulation/
- 80+ test files (~24,000 lines of test code)
- Entry points documented in README (16 CLI commands)
- `TRAINING_DATA_AUDIT.md` documents 17 identified issues (4 critical, 6 serious, 7 moderate)

**Issues found:**
- No `conftest.py` for shared pytest fixtures
- Circular import risk between pipeline/sota.py and feature/model modules (many cross-imports)
- Some dead code and commented-out blocks (e.g., removed SOTAEnsemble class)
- Hub module problem: `sota.py` imports from 20+ modules and is 3000+ lines

---

### Section 13: Required Evaluation Matrix

**GOOD**

| Directive Metric Class | Implemented | Details |
|---|---|---|
| Predictive accuracy | Yes | Brier score, log loss, accuracy, per-round breakdown |
| Calibration | Yes | ECE, MCE, reliability analysis, Brier decomposition |
| Decision utility | Yes | Pool EV, bracket score, Kelly fraction, ROI |
| Risk | Partial | Per-year Brier variance, MC drawdown implicit in simulation |
| Stability | Partial | Year-over-year Brier trend via OLS regression, per-fold variance |

**Missing:** Formal drawdown reporting, worst-case analysis, regime-conditional performance.

---

### Section 14: Continuous Autonomous Research Loop

**WEAK**

The system does not implement an autonomous research loop. It is a manually-invoked pipeline that must be re-run by a human operator. There is no:

- Automated hypothesis generation
- Experiment execution scheduler
- Adversarial review cycle
- Promotion gate automation
- Knowledge retention store for cross-cycle learning

The RDoF audit and LOYO validation partially serve the adversarial review function, and the 0.001 Rule acts as a promotion gate for features, but these are invoked manually.

---

### Section 15: Failure Modes

**MODERATE**

The system correctly handles several directive failure modes:

- **Temporal leakage:** Multiple layers of defense (cutoff dates, shift(1), leakage checks)
- **Validation design that allows information bleed:** LOYO prevents this; random k-fold is not used
- **Improvement that vanishes after calibration:** Calibration is integrated into the validation loop
- **Stronger model that increases complexity:** Stacking and learned feature selection disabled for safety

**Not addressed:**
- No automatic rejection trigger — leakage checks log warnings but don't always halt the pipeline
- No formal rejection of codebase changes that "cannot be validated or safely rolled back"

---

## Part II — Deployment, Operations, and Governance

### Section 18: Production Deployment and Live Monitoring

**NOT IMPLEMENTED**

- No shadow mode, canary deployment, or graduated rollout pipeline
- No real-time monitoring dashboard
- No drift detection protocol (concept, covariate, or label drift)
- No automated retraining triggers
- No A/B testing framework
- No alerting system

This is the **single largest gap** relative to the directive. The system is a research/competition pipeline, not a production system.

---

### Section 19: Data Engineering and Pipeline Resilience

**WEAK**

- No DAG-based pipeline orchestrator (Airflow, Prefect, Dagster)
- No idempotency guarantees on data pipeline tasks
- No schema contracts between pipeline stages (beyond basic validation)
- No data freshness SLA registry
- No fault tolerance or recovery protocol
- Scrapers have basic retry logic but no circuit breakers or fallback mechanisms
- No write-audit-publish pattern

**Partially present:**
- Data quality checks in materialization (leakage checks, coverage audit)
- Logging throughout 55+ source files
- Data validation in `src/data/ingestion/validators.py`

---

### Section 20: Computational Budget and Resource Prioritization

**NOT IMPLEMENTED**

- No compute budget framework
- No phase-level budget allocation
- No experiment cost tracking
- No search termination criteria based on diminishing returns
- No Pareto frontier of compute vs performance

The Optuna trial count (15) and timeout (300s) serve as implicit resource constraints but are not part of a deliberate budget framework.

---

### Section 21: Human-in-the-Loop Governance

**WEAK**

- No decision authority matrix (Autonomous/Notify/Approve classification)
- No approval request protocol
- No compliance checkpoints
- No governance audit trail

**Partially present:**
- Pipeline mode gating (calibration vs EV mode) separates operational contexts
- `require_freeze_file` flag enforces pre-registration before predictions
- RDoF audit framework provides transparency into tuning decisions
- CI/CD requires passing tests before merge

---

### Section 22: Multi-Agent Conflict Resolution

**NOT APPLICABLE**

The system is not multi-agent. No conflict resolution protocol is needed for its current architecture.

---

### Section 23: Testing Strategy and CI/CD

**MODERATE**

**Testing pyramid:**

| Directive Layer | Present | Details |
|---|---|---|
| Unit tests | Yes | 80+ test files covering features, models, calibration, optimization |
| Integration tests | Partial | Pipeline-level tests exist (test_sota_pipeline.py, test_full_ml_pipeline.py) but limited |
| System tests | Partial | End-to-end backtest tests exist but not nightly-scheduled |
| Property-based tests | Partial | Data consistency tests, date integrity tests, but not formal property-based (Hypothesis library) |
| Regression tests | Yes | Data audit issues tracked with fix status |

**Temporal integrity tests (per directive):**

| Directive Test | Present | Details |
|---|---|---|
| Feature timestamp assertion | Yes | Leakage checks in materialization.py validate prior metrics |
| Walk-forward replay test | No | No test that replays LOYO on frozen data and verifies exact match |
| Data leakage canary | No | No deliberately-leaked features inserted to test detection |
| Pipeline ordering test | No | No DAG determinism test |

**CI/CD pipeline:**

| Directive Stage | Present | Details |
|---|---|---|
| Pre-commit (lint, type check) | Partial | Secret scanning only; no linting, formatting, or type checking |
| Unit + property tests | Yes | pytest runs in CI (deploy-with-secrets.yml) |
| Integration tests | Partial | Included in pytest but not separated |
| Coverage gate | No | No coverage measurement or regression gate |
| Model validation smoke test | No | No small-model train-and-verify in CI |
| System tests (nightly) | No | No scheduled nightly runs |

---

### Section 24: Domain-Specific (Sports Betting)

**STRONG**

The system demonstrates deep domain expertise:

- **Injury reports:** Acknowledged as unreliable (scraped from multiple sources, severity estimated)
- **Line movement:** Not directly modeled but public pick percentages serve as a market proxy
- **Small sample sizes:** Recognized and mitigated (NCAA tournament has only ~63 games per year; multi-year pooling expands to ~400+ training samples)
- **Survivorship bias:** 2020 excluded; player-level stats not used directly (team aggregates only)
- **Correlated outcomes:** Regional correlation modeled in MC simulation
- **Closing line value confusion:** Not explicitly addressed (system predicts tournament outcomes, not betting lines)
- **Neutral site adjustment:** Tournament shrinkage factor applied; neutral-site record tracked as feature
- **Home-court dependence:** Computed as feature to identify teams that may underperform on neutral courts

---

## Critical Findings and Prioritized Recommendations

### Severity: CRITICAL (Must Fix)

| # | Finding | Directive Section | Impact |
|---|---|---|---|
| C1 | **cutoff_date defaults to None** in proprietary metrics engine. If callers forget to pass it, tournament games contaminate features. | S1, S11, S15 | Temporal leakage in production |
| C2 | **No production monitoring or alerting.** System cannot detect drift, data staleness, or degradation in live operation. | S18 | Blind production operation |
| C3 | **Calibration train/test separation not enforced internally.** CalibrationPipeline.fit() and evaluate() can operate on same data. Depends entirely on external orchestration. | S8, S11 | Potential calibration leakage |
| C4 | **No experiment registry or artifact versioning.** Experiments are not logged to a shared ledger with reproducibility hashes. | S3, S14 | Non-reproducible research |

### Severity: HIGH (Should Fix)

| # | Finding | Directive Section | Impact |
|---|---|---|---|
| H1 | **No data leakage canary test.** No deliberately-leaked features inserted into the pipeline to verify detection works. | S23.2 | Unverified leakage detection |
| H2 | **No walk-forward replay test.** Cannot verify that re-running LOYO on frozen data produces identical results. | S23.2 | Non-determinism risk |
| H3 | **Optional prior sources lack temporal validation.** Market odds, transfer portal, weather context loaded without explicit pre-tournament availability checks. | S5, S11 | Potential feature leakage |
| H4 | **No CI coverage gate or type checking.** Code changes can regress test coverage or introduce type errors without detection. | S23.3 | Quality regression risk |
| H5 | **Model search space too narrow.** Primarily tree ensembles. No ranking models, statistical time-series, or broad neural architecture search. | S7 | Missed signal opportunity |
| H6 | **Hub module problem.** sota.py imports from 20+ modules and is 3000+ lines. Refactoring risk is high. | S12 | Maintainability debt |

### Severity: MODERATE (Should Address)

| # | Finding | Directive Section | Impact |
|---|---|---|---|
| M1 | No formal robustness testing (missing features, thin-data regimes, distribution shift). | S11 | Unknown failure modes |
| M2 | Regional correlation decay coefficients (MC simulation) acknowledged as under-validated. | S10 | Simulation accuracy uncertainty |
| M3 | No compute budget framework or cost tracking. | S20 | Unbounded compute risk |
| M4 | No human approval workflow for high-stakes actions (deploying new model, changing decision policy). | S21 | Governance gap |
| M5 | No formal feature stability report across seasons. | S6 | Feature drift undetected |
| M6 | No path-dependent risk reporting (drawdowns, losing streaks across years). | S10, S13 | Risk assessment gap |
| M7 | No dataset hashing or version tracking beyond git. | S5, S11 | Data integrity uncertainty |
| M8 | Three-point variance as tournament predictor not validated against tournament outcomes. | S11 | Unverified feature validity |

### Severity: LOW (Nice to Have)

| # | Finding | Directive Section | Impact |
|---|---|---|---|
| L1 | No meta-learning layer (learning which approaches work by regime/sample size). | S7 | Research efficiency |
| L2 | No formal Pareto frontier of compute vs performance. | S20 | Budget optimization |
| L3 | No changelog or semantic versioning. | S12 | Release management |
| L4 | Opponent modeling limited to public pick percentages and archetypes. | S9 | Decision quality ceiling |
| L5 | No architecture decision records (ADRs). | S12 | Knowledge retention |
| L6 | No conftest.py for shared pytest fixtures. | S23 | Test maintainability |

---

## Evaluation Matrix (Directive Section 13)

| Metric Class | Metric | Current Performance | Status |
|---|---|---|---|
| **Predictive accuracy** | Mean LOYO Brier | Tracked per year (2018-2025) | Reported |
| | Log Loss | Per-fold and mean | Reported |
| | Accuracy | Per-fold | Reported |
| **Calibration** | ECE | Computed per fold | Reported |
| | MCE | Computed | Reported |
| | Reliability curve | Per-bin analysis | Available |
| | Brier decomposition | Reliability + resolution | Available |
| **Decision utility** | Pool EV | Kelly-based estimation | Computed |
| | Bracket score | ESPN standard scoring | Simulated |
| | ROI | Entry fee adjusted | Computed |
| **Risk** | Per-year Brier variance | std across LOYO folds | Reported |
| | Drawdown | **NOT REPORTED** | Missing |
| | Worst-case season | Min across folds | Derivable |
| **Stability** | Year-over-year trend | OLS regression on Brier | Available |
| | Regime analysis | **NOT REPORTED** | Missing |

---

## Strengths Worth Preserving

1. **RDoF Audit Framework** — Unique in tournament prediction. Catalogs 60+ constants with tier classification, circularity warnings, and sensitivity analysis. This exceeds most production ML systems.

2. **Decision-Prediction Separation** — The leverage optimizer correctly separates prediction quality from decision quality, implementing pool-size-adaptive strategies with Kelly criterion and variance targeting.

3. **Bayesian Regularization** — Consistent use of conjugate priors across features (3PT variance, momentum, pace variance, consistency) prevents small-sample overfitting.

4. **Tournament Domain Adaptation** — Shrinkage toward 0.5, seed prior blending, and neutral-site adjustment show deep domain understanding.

5. **0.001 Rule** — Simple, effective feature selection criterion that prevents "cool but useless" feature accumulation.

6. **Multi-Year Training Augmentation** — Addresses the fundamental sample-size problem (63 tournament games/year) by pooling historical seasons with exponential decay weighting.

7. **Pre-Registration Discipline** — Pipeline freeze/verify mechanism enables quasi-prospective evaluation (Level 2 in the RDoF framework).

---

## Conclusion

The march-madness-forecaster is a **research-grade system that excels at the core prediction and decision optimization tasks** (Directive Sections 1-13) but **lacks the production infrastructure, monitoring, and governance** required by the directive's deployment sections (Sections 18-25). This is consistent with its nature as a Kaggle competition and bracket pool tool rather than a continuously operating prediction service.

**For its intended use case (annual tournament prediction and bracket optimization):**
- The system is well-architected and demonstrates unusually rigorous temporal integrity
- The 4 critical findings (C1-C4) should be addressed regardless of deployment context
- The high-severity findings (H1-H6) represent concrete improvement opportunities

**If the system were to be deployed as a continuous prediction service:**
- Part II of the directive (Sections 18-25) would need to be implemented nearly from scratch
- This would require DAG orchestration, monitoring dashboards, drift detection, approval workflows, and CI/CD hardening

The system's strongest contribution to the field is its RDoF audit framework and decision optimization layer, both of which could serve as templates for other prediction systems.
