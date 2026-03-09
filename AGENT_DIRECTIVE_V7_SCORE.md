# Agent Directive V7 — Repository Compliance Score

**Repository:** march-madness-forecaster
**Date:** 2026-03-09
**Evaluator:** Claude Opus 4.6 (Fresh independent audit)
**Directive Version:** Agent Directive V7 Complete (25 Sections, Parts I & II)
**Methodology:** Each of 25 sections scored 0–4 for functional implementation completeness, then importance-weighted (weight 2–5) and normalized to 100.

---

## Scoring Scale

| Points | Meaning |
|--------|---------|
| 0 | Not implemented |
| 1 | Stub/placeholder only |
| 2 | Partial — core idea present, significant gaps |
| 3 | Mostly complete — functional with minor gaps |
| 4 | Fully compliant |

---

## Part I — Core Research and Validation Protocol (Sections 1–17)

### S1: Mission & Non-Negotiable Principles — 3/4 (weight 5)
- **Temporal integrity**: LOYO protocol enforces leave-one-year-out; `shift(1)` pattern throughout; leakage canary tests; `LeakageError` halts pipeline.
- **Decision objective**: Brier score primary; bracket scoring and pool EV as decision metrics.
- **Evidence over intuition**: 0.001 Brier improvement rule with paired t-tests.
- **Reproducibility**: `run_hasher.py`, `frozen_config.py`, `artifact_store.py` provide SHA-256 hashing, config freezing.
- **Gap**: No formal kill-switch/degraded-mode fallback; dataset hashes not verified on load.

### S2: Multi-Agent System Architecture — 3/4 (weight 4)
- Five agent roles: DataScout, FeatureEngineer, Modeler, Auditor, Orchestrator in `src/agents/`.
- MessageBus pub/sub with typed messages; agent registry; scheduler with budget awareness.
- Audit agent has veto power; orchestrator runs data→features→model→audit pipeline.
- **Gap**: Agents are not the primary execution path (sota.py monolith dominates); no dedicated Ensemble or Decision agents.

### S3: Shared Contracts & Required Logs — 3/4 (weight 4)
- `ExperimentRecord` implements full V7 S3 schema (25+ fields) in append-only JSONL ledger.
- Schema validation functions in `src/data/schemas.py`.
- **Gap**: Many ledger fields default to empty strings in practice; no enforcement that entries must be complete before results are trusted.

### S4: Phase 0 — Problem Definition — 3/4 (weight 3)
- Target: NCAA tournament game win probabilities. Optimization: Brier score + pool EV. Action: bracket portfolio construction.
- Operational constraints modeled (tournament timing, Kaggle format).
- **Gap**: No formal `<problem_summary>` or `<constraints_register>` standalone artifacts.

### S5: Phase 1 — Dataset Discovery & Lineage — 3/4 (weight 5)
- 20+ data sources: KenPom, Torvik, ESPN, SportsReference, betting markets, injuries, transfer portal, rosters, Massey ordinals, coach history, HerHoopStats.
- DAG-based ingestion with validators; `src/data/versioning.py`.
- 1400+ historical JSON files in `data/raw/historical/`.
- **Gap**: No field-level availability timestamps; raw data overwritten on re-scrape; no formal dataset catalog artifact.

### S6: Phase 2 — Feature Discovery Engine — 3/4 (weight 5)
- Feature families: temporal (rolling, EWM, streaks), seasonal (rest days, travel), hierarchical (conference, SOS), interaction (matchup differentials), representation (GNN, Transformer).
- 0.001 Rule acceptance criterion; materialization and selection frameworks.
- **Gap**: No formal feature stability report or feature retirement log.

### S7: Phase 3 — Model Search & Meta-Learning — 2/4 (weight 5)
- Models: LightGBM, XGBoost, logistic regression, Bayesian Bradley-Terry, SpreadRegressor, LambdaMART ranking, GNN, Transformer.
- Hyperparameter tuning via Optuna with temporal validation.
- **Gap**: No meta-learning layer; in practice dominated by tree ensembles; no systematic objective function search; no `<meta_learning_report>`.

### S8: Phase 4 — Ensemble & Calibration — 3/4 (weight 5)
- Ensemble: fixed-weight averaging (Spread 55%, LGB/XGB/Logistic 15% each); stacking available but disabled.
- Calibration: temperature/Platt/isotonic scaling; Brier decomposition, ECE, MCE; round-specific sigma calibration.
- Model registry for tracking components.
- **Gap**: Ensemble weights not learned/optimized; no formal diversity measurement.

### S9: Phase 5 — Decision Optimization — 3/4 (weight 4)
- Bracket portfolio optimization, leverage picks, dual submission, pool competition strategy.
- Pool-size-adaptive strategies (Tiny/Small/Medium/Large).
- Calibration mode vs EV mode separation.
- **Gap**: No formal threshold sweep report; abstention implicit not explicit; no risk budget sweep.

### S10: Phase 6 — Backtesting & Simulation — 3/4 (weight 5)
- LOYO backtesting with per-year/per-round breakdown.
- Monte Carlo: 50K simulations, regional correlation, injury modeling, Wilson score CIs.
- Scenario analysis (optimistic/base/pessimistic); regime analysis (upset-heavy vs chalk).
- **Gap**: No betting friction terms; no information-arrival timing simulation.

### S11: Phase 7 — Skeptical Audit Layer — 3/4 (weight 5)
- Leakage: canary tests, shift(1) enforcement, correlation detection.
- Validation: LOYO prevents test-period tuning.
- Robustness: feature dropout, PSI drift, Kendall tau stability.
- Reproducibility: config/feature hashes, experiment registry.
- **Gap**: No bitwise-identical replay test; dataset hashes not verified on load.

### S12: Phase 8 — Codebase Review & Refactoring — 2/4 (weight 3)
- Clean module structure: `src/data/`, `ml/`, `pipeline/`, `simulation/`, `optimization/`, `agents/`, `governance/`.
- `docs/REFACTORING_ROADMAP.md` exists.
- **Critical issue**: sota.py is a massive monolith (~8K lines). Roadmap documented but not executed.
- No formal architecture diagram or dependency graph.

### S13: Required Evaluation Matrix — 3/4 (weight 4)
- Predictive: LOYO Brier, log loss, accuracy. Calibration: ECE, MCE, reliability curves, Brier decomposition.
- Decision: pool EV, bracket score. Risk: drawdown, tail-loss, losing streaks.
- Stability: per-year, per-regime breakdown.
- **Gap**: Not consolidated into a single standardized cross-system-comparable matrix.

### S14: Continuous Autonomous Research Loop — 2/4 (weight 4)
- `research_loop.py`, `hypothesis_registry.py`, `experiment_scheduler.py`, `knowledge_store.py` exist.
- Experiment registry enables future automation.
- **Gap**: The loop is not running autonomously. System is manually invoked. No automated promotion gate. Biggest single gap vs. the directive.

### S15: Failure Mode Rejection — 3/4 (weight 4)
- Temporal leakage: LeakageError halts pipeline.
- Validation bleed: LOYO prevents by design.
- Post-calibration vanishing: pre/post metrics compared.
- **Gap**: No automated rejection for "stronger model with increased drawdown"; no formal rollback protocol for code changes.

### S16: Final Deliverables — 2/4 (weight 3)
- Experiment ledger, backtest reports, risk reports, calibration diagnostics present.
- `deliverables_manager.py` exists.
- **Gap**: No reproducibility package (frozen env + data + config); no serialized model artifacts; many V7 required deliverables not generated.

### S17: Operating Summary — 3/4 (weight 2)
- System is point-in-time valid, empirically tested, decision-relevant, calibrated.
- **Gap**: Not fully autonomous; not bitwise reproducible.

---

## Part II — Deployment, Operations, and Governance (Sections 18–25)

### S18: Phase 9 — Deployment & Live Monitoring — 3/4 (weight 4)
- Full deployment pipeline: Shadow → Canary → Staged Rollout → Production (`src/deployment/`).
- Drift alerts with PSI thresholds; A/B framework; model store.
- **Gap**: No live monitoring dashboard; no automated retraining triggers; deployment code exists but not demonstrated in production.

### S19: Phase 10 — Data Engineering & Pipelines — 3/4 (weight 4)
- DAG orchestrator with idempotent tasks, dependency-aware execution, content-hash caching.
- Data quality gates in validators; schema validation; circuit breaker for scrapers.
- **Gap**: Not production-grade DAG (no Airflow/Prefect-class); no formal freshness SLA registry; schema contracts not versioned.

### S20: Compute Budget & Resources — 3/4 (weight 3)
- BudgetManager with per-stage allocation, cost tracking, utilization reporting.
- ResourceTracker (wall-clock, CPU, memory); priority-based shedding.
- Compute efficiency tracker with Pareto frontier.
- **Gap**: No evidence of actual budget enforcement in real runs; cost-per-improvement not demonstrated.

### S21: Governance & Approval Gates — 3/4 (weight 4)
- Authority matrix with action classifications and auto-approve conditions.
- RBAC, approval gates, audit trail, escalation protocol, compliance gates.
- **Gap**: No approval expiration; governance demonstrated in tests only, not production.

### S22: Conflict Resolution Protocol — 3/4 (weight 3)
- Four conflict categories; resolution hierarchy (safety → evidence → orchestrator → human).
- Audit Agent veto on safety matters; dissent registry.
- **Gap**: No evidence of actual conflict resolution; limited to multi-agent mode.

### S23: Testing & CI/CD — 3/4 (weight 4)
- 107 test files; CI with lint, type check, multi-Python tests, coverage gate.
- Temporal integrity tests: leakage canary, walk-forward replay, date integrity.
- Quality gates in CI validating imports, structure, governance compliance.
- **Gap**: Coverage target 60% (directive requires 90%); no formal E2E/nightly system tests; no bitwise replay.

### S24: Domain-Specific Integration — 2/4 (weight 3)
- Deep sports domain knowledge embedded: upset rates (1985-2025), tournament-specific calibration, injury modeling, pool competition, women's basketball.
- **Gap**: Domain knowledge implicit in code, not documented per directive template. No `<domain_integration_guide>`, `<domain_data_quirks_checklist>`, or `<regulatory_compliance_checklist>`.

### S25: Extended Failure Modes & Deliverables — 2/4 (weight 3)
- Code exists for most Part II failure mode detection (shadow mode, monitoring, DAG idempotency, budget enforcement, authority checks).
- **Gap**: Enforcement not demonstrated; no consolidated 28-artifact deliverables package; extended failure modes addressed by code existence not proven enforcement.

---

## Score Summary

| Section | Topic | Score | Weight | Weighted |
|---------|-------|-------|--------|----------|
| 1 | Mission & Non-Negotiable Principles | 3 | 5 | 15 |
| 2 | Multi-Agent System Architecture | 3 | 4 | 12 |
| 3 | Shared Contracts & Required Logs | 3 | 4 | 12 |
| 4 | Phase 0 — Problem Definition | 3 | 3 | 9 |
| 5 | Phase 1 — Dataset Discovery | 3 | 5 | 15 |
| 6 | Phase 2 — Feature Discovery | 3 | 5 | 15 |
| 7 | Phase 3 — Model Search | 2 | 5 | 10 |
| 8 | Phase 4 — Ensemble & Calibration | 3 | 5 | 15 |
| 9 | Phase 5 — Decision Optimization | 3 | 4 | 12 |
| 10 | Phase 6 — Backtesting & Simulation | 3 | 5 | 15 |
| 11 | Phase 7 — Skeptical Audit | 3 | 5 | 15 |
| 12 | Phase 8 — Codebase Review | 2 | 3 | 6 |
| 13 | Evaluation Matrix | 3 | 4 | 12 |
| 14 | Continuous Research Loop | 2 | 4 | 8 |
| 15 | Failure Mode Rejection | 3 | 4 | 12 |
| 16 | Final Deliverables | 2 | 3 | 6 |
| 17 | Operating Summary | 3 | 2 | 6 |
| 18 | Deployment & Monitoring | 3 | 4 | 12 |
| 19 | Data Engineering & Pipelines | 3 | 4 | 12 |
| 20 | Compute Budget & Resources | 3 | 3 | 9 |
| 21 | Governance & Approval Gates | 3 | 4 | 12 |
| 22 | Conflict Resolution | 3 | 3 | 9 |
| 23 | Testing & CI/CD | 3 | 4 | 12 |
| 24 | Domain-Specific Integration | 2 | 3 | 6 |
| 25 | Extended Failure Modes & Deliverables | 2 | 3 | 6 |
| | **TOTALS** | | **98** | **279** |

**Maximum possible weighted score:** 98 × 4 = **392**

---

## Final Score: 71 / 100

**(279 / 392 = 71.2%)**

---

## Strengths

1. **Temporal integrity is best-in-class** — LOYO, shift(1), leakage canary, cutoff enforcement, and LeakageError provide multiple defense layers.
2. **Calibration infrastructure is production-quality** — Brier decomposition, ECE, isotonic/Platt/temperature scaling, round-specific sigma calibration.
3. **Deep domain expertise** — Upset rate calibration (1985-2025), Bayesian shrinkage, pool competition simulation, tournament-specific features.
4. **Comprehensive feature engineering** — 22 active features from 77 dimensions with the 0.001 Brier improvement rule as a rigorous acceptance criterion.
5. **Governance and deployment frameworks fully implemented** — Authority matrix, RBAC, approval gates, shadow/canary deployment, drift detection, A/B testing.
6. **Extensive test suite** — 107 test files covering agents, governance, robustness, leakage, schemas, deployment, reproducibility.
7. **Experiment ledger matches V7 schema** — Full Section 3 record structure with append-only storage.

## Key Gaps (Highest Impact for Score Improvement)

1. **No autonomous research loop** (S14: 2/4) — The single largest gap. The directive mandates continuous automated experimentation; the system is manually invoked.
2. **sota.py monolith** (S12: 2/4) — ~8K line file is a critical maintainability risk. Refactoring roadmap exists but is unexecuted.
3. **No meta-learning** (S7: 2/4) — Model search is dominated by tree ensembles in practice despite multiple model family implementations.
4. **Incomplete deliverables** (S16: 2/4, S25: 2/4) — The 28-artifact consolidated deliverables package is far from complete.
5. **Domain guide not formalized** (S24: 2/4) — Domain knowledge is deeply embedded in code but not documented per the directive's template.
6. **Coverage target below directive standard** (S23) — 60% vs. 90% required.
