# Agent Directive V7 — Independent Scoring Evaluation

**Repository:** march-madness-forecaster
**Evaluation Date:** 2026-03-09
**Evaluator:** Claude Opus 4.6 (Independent Audit — No Subjective Adjustments)
**Directive Version:** Agent Directive V7 Complete (25 Sections)
**Methodology:** Each section scored 0–10 based on verified codebase state. Raw total /250, normalized to 100. No subjective adjustments applied.

---

## Overall Score: 63 / 100

**Raw: 158 / 250 (63.2%)**

---

## Part I — Core Research and Validation Protocol (Sections 1–17)

### S1: Mission and Non-Negotiable Principles — 8 / 10

| Principle | Status | Evidence |
|-----------|--------|----------|
| Temporal integrity first | Strong | LOYO protocol, `shift(1)` pattern, leakage canary tests, `cutoff_date` enforcement (`require_cutoff_date=True`), `LeakageError` halts pipeline |
| Decision objective supremacy | Strong | Brier score optimization (Kaggle metric since 2023), pool EV mode, dual submission |
| Evidence over intuition | Strong | 0.001 Rule for feature ablation, paired Brier t-test, permutation tests, academic citations throughout |
| Reproducibility over vibes | Partial | RDoF audit (60+ constants), experiment registry with hashes, SHA-256 data hashing — but no frozen dataset versioning enforcement on load |
| Safety over ambition | Partial | Calibration guardrails, stacking disabled by default, Optuna capped at 15 trials — but no formal kill switch or degraded-mode fallback |

**Deductions:** -1 no dataset hash verification on load; -1 no formal kill switch/degraded-mode fallback

---

### S2: Multi-Agent System Architecture — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Specialized agent roles | Implemented | 5 agents: DataScoutAgent, FeatureEngineerAgent, ModelingAgent, AuditAgent (veto), OrchestratorAgent (`src/agents/concrete.py`) |
| Agent communication | Implemented | `MessageBus` pub/sub with typed `AgentMessage` and `MessageType` enums (`src/agents/__init__.py`) |
| Shared experiment registry | Implemented | `ExperimentRegistry` JSONL ledger; agents log via `GovernanceAuditTrail` |
| Budget tracking | Implemented | `BudgetManager` integrated in OrchestratorAgent with per-stage allocation |
| Compliance gates | Implemented | `ComplianceGate` checks at stage boundaries |
| Tests | 31 tests | `tests/test_multi_agent.py` |

**Deductions:** -2 agents not the primary execution path (sota.py monolith still dominates); -1 limited evidence agents improve research outcomes vs. architectural compliance

---

### S3: Shared Contracts and Required Logs — 6 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Experiment ledger | Implemented | `ExperimentRecord` dataclass with 25+ fields matching V7 schema (`src/ml/evaluation/experiment_registry.py`) |
| Complete entries | Partial | Schema comprehensive but many fields default to empty strings; actual population depends on pipeline wiring |
| Dataset hashes | Partial | `dataset_hashes` field exists; SHA-256 computation available but optional |
| Schema contracts | Implemented | `validate_ensemble_weights()`, `validate_calibration_data()`, `validate_matchup_vector()` (`src/data/schemas.py`) |

**Deductions:** -2 auto-logging not fully wired (many fields default empty); -2 no enforcement that entries must be complete before results trusted

---

### S4: Phase 0 — Problem Definition and Utility Mapping — 8 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Prediction target | Clear | NCAA tournament game-level win probabilities |
| Real optimization target | Clear | Kaggle round-weighted Brier score (calibration mode), pool rank percentile (EV mode) |
| Action layer | Clear | Submit bracket portfolio (Kaggle), choose optimal bracket for pool play |
| Operational constraints | Partial | Tournament timing modeled; no explicit latency/budget documentation |

**Deductions:** -1 no explicit latency/budget documentation; -1 no formal utility mapping standalone artifact

---

### S5: Phase 1 — Dataset Discovery, Construction, and Lineage — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Broad signal search | Strong | 20+ sources: KenPom, Torvik, ESPN, SportsReference, betting markets, injuries, transfer portal, rosters, Massey ordinals, coach history, HerHoopStats |
| Point-in-time representation | Implemented | `shift(1)` pattern, `HistoricalDataPipeline` with date preservation |
| Raw snapshot preservation | Partial | 1400+ historical JSON files (2003-2026) in `data/raw/historical/`; but overwritten on re-scrape |
| Survivorship/hindsight testing | Partial | Leakage canary tests; no explicit survivorship bias test |
| Field-level timestamps | Not implemented | No per-field availability timestamp tracking |

**Deductions:** -1 no field-level timestamps; -1 raw data overwritten (no versioned snapshots); -1 no formal dataset_catalog artifact

---

### S6: Phase 2 — Feature Discovery Engine — 8 / 10

| Feature Family | Status | Evidence |
|----------------|--------|----------|
| Temporal | Implemented | Rolling means, momentum, streaks, EWM stats, recency-weighted deltas |
| Seasonal/calendar | Implemented | Rest days, travel burden, games in last 7 days, season progress |
| Hierarchical | Implemented | Conference aggregates, SOS adjustment (15-iteration), quadrant-1 wins |
| Interaction | Implemented | Matchup differentials, seed x efficiency, tempo interaction, style mismatch |
| Representation | Implemented | GNN schedule embeddings (`schedule_graph.py`), transformer game sequences (`game_sequence.py`) |
| Acceptance rules | Implemented | 0.001 Rule, leakage checks, production availability, 22 active features from 77 dimensions |
| Missing-data indicators | Implemented | Binary flags for sparse features (preseason AP, coach metrics) |

**Deductions:** -1 no formal feature stability report (Kendall tau) wired into pipeline; -1 no feature retirement log

---

### S7: Phase 3 — Model Search and Meta-Learning — 6 / 10

| Model Family | Status |
|--------------|--------|
| Linear/generalized | Logistic regression (baseline) |
| Tree ensembles | LightGBM, XGBoost (primary) |
| Neural sequence | GNN, Transformer (optional, disabled by default) |
| Bayesian | Bayesian Bradley-Terry rating system |
| Regression-to-probability | SpreadRegressor → logistic CDF |
| Ranking/pairwise | Not implemented |
| Statistical time-series | Not implemented |

**Deductions:** -2 no meta-learning layer; -1 no systematic objective function search; -1 search space dominated by tree ensembles in practice

---

### S8: Phase 4 — Ensemble Optimization and Calibration — 8 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Ensemble methods | Implemented | Fixed-weight averaging (Spread 0.55, LGB 0.15, XGB 0.15, Logistic 0.15); stacking available but disabled |
| Calibration diagnostics | Comprehensive | Brier decomposition, ECE, MCE, reliability curves, per-bin analysis, bootstrap CIs, ROC-AUC |
| Multiple calibration methods | Implemented | Temperature scaling, Platt scaling, isotonic regression with sample-size-aware auto-downgrade |
| Leakage protection | Implemented | SHA-256 data hashing; `CalibrationLeakageError` in strict mode |
| Round-specific calibration | Implemented | `TournamentSigmaCalibrator` with Bayesian shrinkage |

**Deductions:** -1 fixed ensemble weights (not learned); -1 diversity not formally measured

---

### S9: Phase 5 — Decision Optimization Layer — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Action policy optimization | Implemented | Bracket portfolio, leverage picks, dual submission (`bracket_portfolio.py`, `leverage.py`, `dual_submission.py`) |
| Multiple thresholds/budgets | Partial | Pool size estimation varies strategy (Tiny/Small/Medium/Large); no explicit risk budget sweep |
| Abstention as first-class | Partial | Minimum leverage threshold (1.5x) acts as implicit abstention |
| Forecast vs decision separation | Partial | Calibration mode vs EV mode separates concerns |

**Deductions:** -1 no formal decision policy evaluation separate from forecast; -1 abstention implicit not explicit; -1 no risk budget sweep

---

### S10: Phase 6 — Backtesting and Simulation Realism — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Information arrival simulation | Partial | LOYO simulates forward prediction; Monte Carlo uses pre-computed probs |
| Friction terms | Partial | Pool competition modeled; no betting friction (spread, slippage, line movement) |
| Scenario sensitivity | Implemented | `ScenarioAnalysis` (optimistic/base/pessimistic); `RegimeAnalysis` (upset-heavy vs chalk) |
| Path-dependent risk | Implemented | Max drawdown, cumulative drawdown, tail-loss (10%/5%), losing streaks, worst-year analysis |
| Monte Carlo | Strong | 50K sims, regional correlation, injury modeling, logit-space noise, Wilson score CIs |

**Deductions:** -1 no information arrival timing simulation; -1 no betting friction terms; -1 correlation coefficients under-validated (wide CIs from ~160 region-years)

---

### S11: Phase 7 — Skeptical Audit Layer — 7 / 10

| Audit Type | Status | Evidence |
|------------|--------|----------|
| Leakage | Implemented | Canary tests (5 tests, `test_leakage_canary.py`), `shift(1)` enforcement, correlation-based detection |
| Validation | Implemented | LOYO protocol; calibration leakage detection; no random k-fold misuse |
| Robustness | Implemented | Feature dropout, PSI drift detection, Kendall tau stability (`src/ml/evaluation/robustness.py`) |
| Reproducibility | Partial | Config/feature hashes in RDoF audit; experiment registry; but no frozen seed enforcement |

**Deductions:** -1 no bitwise-identical replay test; -1 no formal independent audit cycle; -1 dataset hashes not verified on load

---

### S12: Phase 8 — Codebase Review and Refactoring — 5 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Module structure | Clean | `src/data/`, `ml/`, `pipeline/`, `simulation/`, `optimization/`, `exports/`, `monitoring/`, `agents/`, `governance/` |
| Test coverage | 107 files | Covering agents, governance, robustness, leakage, schemas, deployment, reproducibility |
| Hub module problem | **Critical** | `sota.py` = 8,368 lines; decomposition roadmap documented (`docs/REFACTORING_ROADMAP.md`) but NOT executed |
| Coverage threshold | Low | `fail_under = 60` in pyproject.toml (industry standard: 80%+) |
| Architecture diagram | Missing | No formal dependency graph or architecture documentation |

**Deductions:** -3 sota.py at 8,368 lines is a major maintainability debt; -1 coverage 60% below standard; -1 no architecture diagram

---

### S13: Required Evaluation Matrix — 6 / 10

| Metric Class | Status | Evidence |
|--------------|--------|----------|
| Predictive accuracy | Reported | Mean LOYO Brier, Log Loss, Accuracy per fold |
| Calibration | Comprehensive | ECE, MCE, reliability curve, Brier decomposition |
| Decision utility | Reported | Pool EV, bracket score, ROI |
| Risk | Implemented | Drawdown, tail-loss, trend slope, losing streaks, worst/best year (`risk_report.py`) |
| Regime analysis | Implemented | Upset-heavy vs chalk classification with per-regime Brier (`RegimeAnalysis`) |
| Scenario analysis | Implemented | Optimistic/base/pessimistic projections (`ScenarioAnalysis`) |
| Cross-system comparability | Not demonstrated | No structured comparison against external baselines |

**Deductions:** -2 no standardized cross-system comparability; -2 metrics not in single comparable matrix format

---

### S14: Continuous Autonomous Research Loop — 2 / 10

| Requirement | Status |
|-------------|--------|
| Hypothesis generation | Not automated |
| Experiment execution | Pipeline can run experiments with logging |
| Adversarial review | Robustness tests exist but aren't auto-triggered |
| Promotion gate | Not implemented |
| Knowledge retention | Experiment ledger retains findings; no structured knowledge base |

**This is the single largest gap.** The directive mandates an autonomous research loop. This system is a manually-invoked pipeline. +1 for LOYO serving as manual validation; +1 for experiment registry enabling future automation.

---

### S15: Failure Modes — 7 / 10

| Failure Mode | Detection | Evidence |
|-------------|-----------|----------|
| Temporal leakage | Yes | `LeakageError` halts pipeline in strict mode |
| Validation bleed | Yes | LOYO protocol prevents by design |
| Improvement vanishing after calibration | Partial | Pre/post calibration metrics compared |
| Stronger model with more instability | Partial | Risk report exists but no automated rejection |
| Non-rollback-safe code changes | Not enforced | No formal rollback protocol |

**Deductions:** -1 no formal rollback protocol; -1 no rejection gate for codebase changes; -1 some failure modes only partially addressed

---

### S16: Final Deliverables — 4 / 10

| Deliverable | Present | Notes |
|-------------|---------|-------|
| Validated model artifacts | Partial | Trained in pipeline but not serialized as standalone artifacts |
| Experiment ledger | Yes | JSONL format |
| Backtest report | Yes | UnifiedBacktester across years |
| Risk report | Yes | `risk_report.py` output |
| Feature importance ranking | Partial | Available via ablation but not as standalone deliverable |
| Calibration diagnostics | Yes | Comprehensive |
| Reproducibility package | **No** | No frozen environment + data + config bundle |
| Codebase audit | Partial | FIX AUDIT comments throughout code |

**Deductions:** -2 no serialized model artifacts; -2 no reproducibility package; -1 no standalone feature importance deliverable; -1 no complete codebase audit deliverable

---

### S17: Operating Summary Compliance — 6 / 10

The system is point-in-time valid, empirically tested, decision-relevant, and calibrated.

**Deductions:** -2 not fully reproducible (no bitwise replay); -1 not autonomous; -1 robustness module exists but not integrated into main pipeline flow

---

**Part I Subtotal: 109 / 170 (64.1%)**

---

## Part II — Deployment, Operations, and Governance (Sections 18–25)

### S18: Phase 9 — Production Deployment and Live Monitoring — 5 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Staged deployment pipeline | Code exists | `src/deployment/shadow_mode.py`, `canary.py`, `orchestrator.py`, `pipeline.py` |
| Monitoring dashboard | Partial | `PipelineMonitor` with PSI drift detection; JSON reports, not live dashboards |
| Drift detection | Implemented | PSI-based with warning/alert thresholds |
| A/B testing framework | Code exists | `src/deployment/ab_framework.py` |
| Model store | Implemented | `src/deployment/model_store.py` with versioned artifact storage |
| Automated retraining | Not implemented | No automated retraining triggers |

**Deductions:** -2 no live dashboard or automated retraining; -2 not demonstrated in actual production use; -1 deployment code exists but no deployment history

---

### S19: Phase 10 — Data Engineering and Pipeline Resilience — 5 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| DAG-based pipelines | Implemented | `src/data/ingestion/dag.py` with DagTask, DagExecutor, dependency-aware execution |
| Idempotent tasks | Partial | Content hash caching in DAG executor |
| Fault tolerance | Partial | Scraper fallbacks (graceful degradation); no circuit breaker pattern |
| Data freshness SLA | Implemented | `DEFAULT_FRESHNESS_SLA` with per-source thresholds |
| Schema validation | Implemented | `src/data/ingestion/validators.py` with structured validation |

**Deductions:** -2 not production-grade DAG orchestrator (no Airflow/Prefect-class); -2 no circuit breaker pattern; -1 limited schema contract scope

---

### S20: Computational Budget and Resource Prioritization — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Compute budget framework | Implemented | `ResourceTracker` with wall-clock, CPU, memory tracking (`src/monitoring/resource_tracker.py`) |
| Prioritized search | Implemented | `BudgetManager.prioritized_stages()` and `should_shed()` (`src/monitoring/budget_manager.py`) |
| Cost tracking | Implemented | `ResourceTracker.to_dict()` logged to `RunHistory`; per-phase tracking |
| Pareto frontier | Implemented | `ComputeEfficiencyTracker` with `pareto_frontier()` |
| Budget enforcement | Implemented | `ResourceBudget.strict` mode raises `ComputeBudgetExceeded`; alerts at 80%/100%/150% |

**Deductions:** -1 no evidence of actual budget enforcement in real runs; -1 no projected-vs-actual reporting; -1 cost-per-improvement not demonstrated on real experiments

---

### S21: Human-in-the-Loop Governance — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Decision authority matrix | Implemented | `DecisionAuthority` with 8 `ActionType` enums, role-based policies (`src/governance/decision_authority.py`) |
| Approval request protocol | Implemented | `ApprovalRequest` workflow: request → pending → approved/denied; persisted to disk |
| Compliance checkpoints | Implemented | `ComplianceGate` per stage (data loading, training, calibration, simulation, audit) (`src/governance/compliance.py`) |
| Audit trail | Implemented | `GovernanceAuditTrail` append-only JSONL with query/filter (`src/governance/audit_trail.py`) |
| Escalation protocol | Implemented | `EscalationProtocol` with auto-level detection (WARNING→researcher, ERROR→ml_lead, CRITICAL→operator) |

**Deductions:** -1 no evidence of actual governance actions in production; -1 no regulatory compliance (minor, N/A for domain); -1 no 3-year retention policy enforcement

---

### S22: Multi-Agent Conflict Resolution — 7 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Conflict categories | Implemented | 4 categories: PRIORITY, METHOD, RESOURCE, SAFETY (`src/agents/conflict.py`) |
| Resolution hierarchy | Implemented | Safety → Audit Agent veto; Empirical → evidence wins; Priority → Orchestrator |
| Audit Agent veto | Implemented | Absolute veto on S15 failures, leakage detection; no override path |
| Dissent registry | Implemented | `DissentRegistry` JSONL append-only with file/query/review workflow |

**Deductions:** -1 no evidence of actual conflict resolution in practice; -1 no review of open dissents workflow; -1 limited to multi-agent mode only

---

### S23: Testing Strategy and CI/CD Integration — 6 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Testing pyramid | Partial | 107 test files (unit + integration); no explicit E2E tests |
| Temporal integrity tests | Implemented | `test_leakage_canary.py`, `test_walk_forward_replay.py`, `test_date_integrity.py` |
| Walk-forward replay | Implemented | 5 tests verifying LOYO determinism, subset consistency, frozen snapshots |
| Data leakage canary | Implemented | 5 tests inserting deliberately-leaked features |
| Pipeline ordering test | Not implemented | No bitwise-identical replay verification |
| CI/CD pipeline | Implemented | `.github/workflows/ci.yml`, `deploy-with-secrets.yml`, `deploy-staging.yml`; Ruff + mypy |
| Coverage gate | Low | `fail_under = 60` (below 80% standard) |

**Deductions:** -2 no E2E tests; -1 no bitwise replay test; -1 coverage target below 80%

---

### S24: Domain-Specific Integration — 8 / 10

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Historical upset rates | Implemented | Calibrated against 1985-2025 data |
| Tournament adjustments | Implemented | Round-specific sigma calibration, travel distance, neutral-site adjustment, coach experience |
| Injury handling | Implemented | Injury impact modeling in Monte Carlo simulations |
| Small-sample mitigation | Implemented | Bayesian regularization, conjugate priors, shrinkage toward 0.5 |
| Women's basketball | Implemented | Dedicated pipeline, scrapers (HerHoopStats, NCAA NET), feature engineering |
| Pool competition modeling | Implemented | Competitor archetypes, pool-size-adaptive strategies |

**Deductions:** -1 no betting regulatory compliance; -1 domain guide limited to sports (directive covers finance, elections, fantasy)

---

### S25: Extended Failure Modes and Deliverables — 4 / 10

| Extended Failure Mode | Detection |
|-----------------------|-----------|
| Deployment bypassing shadow/canary | Code exists (`shadow_mode.py`, `canary.py`) but no enforcement evidence |
| Production without monitoring | `PipelineMonitor` exists; not demonstrated as blocking gate |
| Pipeline without idempotency | DAG executor has caching; not formally guaranteed |
| Compute budget overrun | `BudgetManager` alerts exist; not demonstrated in practice |
| Unauthorized actions | `DecisionAuthority` exists; not demonstrated in practice |
| Silent agent override | `DissentRegistry` exists; not demonstrated in practice |
| CI/CD bypass | CI workflows exist; pre-merge gates not verified |
| Regulatory non-compliance | N/A for this domain |
| Consolidated deliverables | Not assembled as complete package |

**Deductions:** -2 failure modes addressed by code existence not demonstrated enforcement; -2 no evidence of actual rejection triggering; -2 no consolidated deliverables package

---

**Part II Subtotal: 49 / 80 (61.3%)**

---

## Final Score Summary

| Category | Score | Max | Percentage |
|----------|-------|-----|------------|
| Part I: Core Research (S1-S17) | 109 | 170 | 64.1% |
| Part II: Deployment & Ops (S18-S25) | 49 | 80 | 61.3% |
| **TOTAL** | **158** | **250** | **63.2%** |

### Normalized Score: 63 / 100

---

## Comparison with Prior Evaluations

| Evaluation | Score | Methodology |
|------------|-------|-------------|
| AGENT_DIRECTIVE_V7_AUDIT.md | ~68% | Compliance percentage per section |
| AGENT_DIRECTIVE_V7_SCORE.md (prior) | 77/100 | Raw 56/100 + subjective +21 adjustment |
| **This evaluation** | **63/100** | Raw scoring, no adjustments, verified implementations |

### Why This Score Differs from 77/100

1. **No subjective adjustment** — Prior score added +21 points for "code quality exceeding what scores capture"
2. **Stricter on code existence vs. demonstrated use** — Deployment, governance, and conflict resolution modules exist but lack evidence of actual production use
3. **Stricter on S12 (Codebase Quality)** — sota.py at 8,368 lines is a major maintainability problem
4. **Stricter on S14 (Research Loop)** — Directive mandates autonomous operation; system is manually invoked
5. **Stricter on S16 (Deliverables)** — No reproducibility package or serialized model artifacts

---

## Top Strengths

1. **Temporal integrity** (S1, S11) — Best-in-class leakage prevention with multiple defense layers: LOYO, shift(1), canary tests, cutoff enforcement, LeakageError
2. **Calibration infrastructure** (S8) — Production-quality: auto-downgrade, leakage detection via SHA-256, bootstrap CIs, Brier decomposition, round-specific sigma calibration
3. **Domain expertise** (S24) — Deep tournament-specific knowledge: upset rate calibration (1985-2025), Bayesian shrinkage, pool competition modeling
4. **Feature engineering rigor** (S6) — 0.001 Rule, missing-data indicators, GNN/transformer representations, ablation testing
5. **Multi-agent + governance frameworks** (S2, S20-S22) — Fully implemented with tests; architecturally sound even if not primary execution path
6. **Test coverage breadth** (S23) — 107 test files covering agents, governance, robustness, leakage, schemas, deployment

## Top Gaps

1. **No autonomous research loop** (S14: 2/10) — Single biggest gap; directive mandates continuous autonomous experimentation
2. **sota.py monolith** (S12: 5/10) — 8,368 lines is a critical maintainability risk; roadmap exists but not executed
3. **Incomplete deliverables** (S16: 4/10) — No reproducibility package, no serialized model artifacts
4. **Extended failure modes** (S25: 4/10) — Code exists for detection but enforcement not demonstrated
5. **No meta-learning** (S7: 6/10) — Model search limited to tree ensembles in practice despite 7 family implementations

---

## Path to 80/100

| Improvement | Estimated Impact | Effort |
|-------------|-----------------|--------|
| Decompose sota.py (execute refactoring roadmap) | S12: +3 | High |
| Implement autonomous research loop (even basic) | S14: +4 | High |
| Serialize model artifacts + reproducibility package | S16: +3 | Medium |
| Raise coverage to 80% + add E2E tests | S12: +1, S23: +2 | Medium |
| Wire auto-logging into main pipeline for every fold | S3: +2 | Medium |
| Demonstrate deployment/governance in real pipeline run | S18: +2, S25: +2 | Medium |
| Add meta-learning or expand model diversity | S7: +2 | Medium |
| Add promotion gate (new model must beat incumbent) | S14: +1, S15: +1 | Low |
| Add bitwise-identical replay test | S11: +1, S23: +1 | Low |
| Enforce dataset hash verification on load | S1: +1, S11: +1 | Low |

**Total potential: +26 points → 89/100**
