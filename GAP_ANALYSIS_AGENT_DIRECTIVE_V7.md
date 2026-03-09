# Gap Analysis: Agent Directive V7 vs. march-madness-forecaster

**Date:** 2026-03-09
**Evaluator:** Independent code-verified audit (Claude Opus 4.6)
**Directive:** Agent Directive V7 Complete (25 Sections, Parts I & II)
**Codebase:** 180+ source modules, 114 test files, 8,532-line main pipeline

---

## Executive Summary

The march-madness-forecaster is a research-grade NCAA tournament prediction system with **strong core ML fundamentals** (temporal integrity, calibration, feature engineering) but **significant integration gaps** between standalone modules and the primary execution path. Prior evaluations range from 62–82/100 depending on whether "code exists" or "code is operationally integrated" is the standard.

This gap analysis uses a **functional integration standard**: modules must be called from the main pipeline path (`sota.py` or `main.py`) to receive full credit, but integration via `try/except` blocks within the pipeline is acknowledged as partial integration rather than no integration.

**Overall Assessment: 74/100 — Strong research system with operational integration gaps.**

---

## Methodology

Each of the 25 directive sections scored on a 5-point scale:

| Score | Meaning |
|-------|---------|
| 0 | Not implemented |
| 1 | Stub/placeholder only |
| 2 | Partial — core idea present, significant gaps |
| 3 | Mostly functional — works with minor gaps |
| 4 | Fully compliant with directive requirements |

Weighted by criticality (1–5) to the March Madness tournament use case.

---

## Part I — Core Research and Validation Protocol (Sections 1–17)

### S1: Mission & Non-Negotiable Principles — Score: 4/4 (Weight: 5)

**Status: STRONG**

| Principle | Implementation | Evidence |
|-----------|---------------|----------|
| Temporal integrity | Multi-layered | `TOURNAMENT_START_DATES`, `shift(1).expanding()`, `LeakageError`, leakage canary tests, `cutoff_date` enforcement |
| Decision objective supremacy | Strong | Brier score primary + pool EV secondary, dual submission |
| Evidence over intuition | Strong | 0.001 Brier improvement rule with paired t-tests |
| Reproducibility | Good | `run_hasher.py` (SHA-256), `frozen_config.py`, `artifact_store.py`, dataset hash verification |
| Safety over ambition | Good | Calibration guardrails, stacking disabled by default, `CalibrationLeakageError` |

**Gaps:**
- No runtime safety envelope that automatically reduces exposure when operating outside training distribution
- Synthetic date inference not logged as data quality warning

**Weighted: 20/20**

---

### S2: Multi-Agent System Architecture — Score: 2/4 (Weight: 3)

**Status: PARTIAL — Scaffolding with light integration**

The agent system exists in `src/agents/` with 5 agent roles, a `MessageBus`, and a registry. The `run_multi_agent()` method in `sota.py` (line 1435) wires agents through an `OrchestratorAgent`. However:

- The default execution path (`run()`) does NOT use agents — it runs everything directly
- `run_multi_agent()` is an **optional alternative path** gated by a config flag
- The MessageBus pub/sub system is not used in the default pipeline path
- `sota.py` at 8,532 lines remains the actual orchestrator

**Gap:** Agents are available but not the default execution model. The directive envisions agents as the primary coordination mechanism, not an optional mode.

**Weighted: 6/12**

---

### S3: Shared Contracts & Required Logs — Score: 3/4 (Weight: 4)

**Status: MOSTLY FUNCTIONAL**

- `ExperimentRecord` with 25+ field schema in append-only JSONL ledger
- Schema validation in `src/data/schemas.py` with `REQUIRED_FIELDS` and `validate_record()`
- `RobustnessSuite` and `ModelComparisonFramework` provide `format_for_registry()` methods

**Gaps:**
- Many ledger fields default to empty strings in practice — no enforcement of completeness
- Auto-logging of every LOYO fold not wired into main pipeline
- No MLflow/W&B integration for experiment tracking

**Weighted: 12/16**

---

### S4: Phase 0 — Problem Definition & Utility Mapping — Score: 3/4 (Weight: 3)

**Status: MOSTLY FUNCTIONAL**

Target (win probabilities), optimization (Brier score), action layer (bracket portfolio), and constraints (timing, Kaggle format) are all well-defined in code.

**Gap:** No standalone `<problem_summary>` or `<constraints_register>` artifact as the directive specifies. Problem definition is implicit in code rather than formalized as a document.

**Weighted: 9/12**

---

### S5: Phase 1 — Dataset Discovery, Construction & Lineage — Score: 3/4 (Weight: 5)

**Status: MOSTLY FUNCTIONAL**

- 20+ data sources with specialized scrapers
- DAG-based ingestion (`src/data/ingestion/dag.py`) with topological sorting and idempotency caching
- Data versioning with snapshot/restore (`src/data/versioning.py`)
- Circuit breaker pattern for scraper resilience
- `TeamNameResolver` with 360+ aliases

**Gaps:**
- No field-level availability timestamps (event time vs. publication time vs. ingestion time vs. revision time)
- No formal dataset catalog artifact
- No explicit survivorship bias testing
- No record-linkage error detection

**Weighted: 15/20**

---

### S6: Phase 2 — Feature Discovery Engine — Score: 3/4 (Weight: 5)

**Status: MOSTLY FUNCTIONAL**

Comprehensive feature families:
- **Temporal:** rolling means, EWM, streaks, momentum, recency-weighted deltas
- **Seasonal:** rest days, travel distance burden
- **Hierarchical:** conference strength, SOS, opponent-adjusted metrics
- **Interaction:** 78-dimensional matchup vectors (differential + absolute + interaction blocks)
- **Representation:** GNN schedule graph, Transformer game sequence
- **Acceptance:** 0.001 Brier improvement threshold with paired t-tests

**Gaps:**
- No formal feature stability report (Kendall tau across folds) wired into pipeline
- No feature retirement log
- No explicit production-availability check per feature
- No feature revision risk assessment

**Weighted: 15/20**

---

### S7: Phase 3 — Model Search & Meta-Learning — Score: 2/4 (Weight: 5)

**Status: PARTIAL**

Model families implemented: LightGBM, XGBoost, Logistic Regression, SpreadRegressor, BayesianBT, GNN, Transformer, LambdaMART. Optuna for hyperparameter tuning.

`ModelComparisonFramework` provides systematic evaluation with Brier/log-loss/accuracy/ECE metrics and diversity analysis.

**Gaps:**
- `MetaLearner` uses **hardcoded regime preference tables** (e.g., chalk regimes prefer spread model) rather than learning from LOYO performance data
- No automated model selection loop — system is dominated by tree ensembles with near-fixed weights
- No systematic objective function search
- No `<meta_learning_report>` artifact
- No training window length search

**Weighted: 10/20**

---

### S8: Phase 4 — Ensemble Optimization & Calibration — Score: 3/4 (Weight: 5)

**Status: MOSTLY FUNCTIONAL**

- MarginFirst ensemble (Spread 55%, LGB/XGB/Logistic 15% each) with TournamentExpert blend
- Calibration: Temperature scaling, Platt, Isotonic with Brier decomposition
- SHA-256 calibration leakage guard (`CalibrationLeakageError`)
- Round-specific sigma calibration with Bayesian shrinkage
- `EnsembleWeightOptimizer` via Optuna exists

**Gaps:**
- Default ensemble weights are **fixed, not dynamically optimized per year**
- No formal component diversity measurement (pairwise disagreement wired into pipeline)
- Greedy ensemble search not implemented
- Stacking available but disabled — no automated enable/disable decision

**Weighted: 15/20**

---

### S9: Phase 5 — Decision Optimization Layer — Score: 3/4 (Weight: 4)

**Status: MOSTLY FUNCTIONAL**

- Bracket portfolio optimization with constraint satisfaction
- Kelly-criterion-informed leverage picks
- Dual submission strategy (Brier-optimized + champion-boosted hedge)
- Pool competition simulation with size-adaptive archetypes
- Calibration mode vs. EV mode separation

**Gaps:**
- No formal threshold sweep report across multiple risk budgets
- Abstention is implicit (low-confidence avoidance) rather than explicit first-class policy
- No risk budget sweep comparing decision quality at multiple exposure levels

**Weighted: 12/16**

---

### S10: Phase 6 — Backtesting & Simulation Realism — Score: 3/4 (Weight: 5)

**Status: MOSTLY FUNCTIONAL**

- LOYO backtesting (2018–2025, excluding 2020) with per-year/per-round breakdown
- Monte Carlo: 50K iterations, regional correlation (0.10), injury modeling, Wilson CI
- Unified backtest framework (`unified_backtest.py`)
- Named scenario analysis (optimistic/base/pessimistic)
- Risk reporting: drawdown, tail-loss, trend slope, losing streaks

**Gaps:**
- No information-arrival timing simulation (backtests use final settled data, not point-in-time market snapshots)
- No betting friction terms (spread, slippage, liquidity — relevant for EV mode)
- Scenario sensitivity under-specified in code vs. directive requirements

**Weighted: 15/20**

---

### S11: Phase 7 — Skeptical Audit Layer — Score: 3/4 (Weight: 5)

**Status: MOSTLY FUNCTIONAL**

| Audit Type | Implementation |
|------------|---------------|
| Leakage | Canary tests, `shift(1)` enforcement, `LeakageError` halts pipeline |
| Validation | LOYO prevents test-period tuning by design |
| Robustness | `RobustnessSuite` orchestrating FeatureDropout, PSI drift, Kendall tau |
| Reproducibility | Config/feature hashes, dataset hash verification, deterministic replay test |
| RDoF | 58+ cataloged constants with 3 derivation tiers, sensitivity analysis |

**Gaps:**
- `RobustnessSuite` not auto-triggered on every LOYO fold — manual invocation
- No bitwise-identical replay enforcement as a pipeline gate (test exists, not gating)
- Audit Agent veto power exists in multi-agent code but is not exercised in default pipeline path

**Weighted: 15/20**

---

### S12: Phase 8 — Codebase Review & Refactoring — Score: 2/4 (Weight: 3)

**Status: PARTIAL — Critical monolith issue**

Clean module structure across 14 packages. Refactoring roadmap documented. Pipeline stage protocol defined in `src/pipeline/stages/`.

**Critical Issue:** `sota.py` is **8,532 lines**. The `stages/` directory has 11 decomposed modules but **only 3 are actually imported** (`inference.py`, `game_utils.py`, `context.py`). The other 8 stages were conceptually decomposed but their functionality remains inlined in the monolith.

This is the single largest codebase quality gap and directly contradicts the directive's requirement to "prioritize refactors by impact."

**Weighted: 6/12**

---

### S13: Required Evaluation Matrix — Score: 2/4 (Weight: 4)

**Status: PARTIAL**

Individual metrics exist across multiple reporting functions:
- Predictive: LOYO Brier, log loss, accuracy
- Calibration: ECE, MCE, reliability curves, Brier decomposition
- Decision: pool EV, bracket score, ROI
- Risk: drawdown, tail-loss, trend, losing streaks
- Stability: per-year breakdown, regime-conditional analysis

**Gap:** Not consolidated into a **single standardized cross-system-comparable matrix** as the directive requires. No formal `<evaluation_matrix>` artifact produced per experiment. Metrics scattered across `risk_report.py`, `unified_backtest.py`, `experiment_registry.py`.

**Weighted: 8/16**

---

### S14: Continuous Autonomous Research Loop — Score: 2/4 (Weight: 4)

**Status: PARTIAL — Components exist, end-to-end autonomy unproven**

| Component | Status |
|-----------|--------|
| `ExperimentScheduler` | Functional — generates config variants (perturbation, grid, adaptive) |
| `HypothesisRegistry` | Functional — registers and queries hypotheses |
| `KnowledgeStore` | Functional — stores/retrieves findings |
| `ResearchLoop` CLI | Exists — `python -m src.main research-loop` |
| `PromotionGate` | Functional — candidate must beat incumbent by 0.001 |
| Hypothesis generation from diagnostics | Integrated in `sota.py` (line 3212) |

**Gap:** The research loop has **never been demonstrated running autonomously end-to-end** for multiple iterations. The pipeline generates hypotheses from LOYO diagnostics but does not automatically execute the next variant. This is the directive's core mandate: continuous autonomous experimentation.

**Weighted: 8/16**

---

### S15: Failure Mode Rejection — Score: 3/4 (Weight: 4)

**Status: MOSTLY FUNCTIONAL**

| Failure Mode | Detection |
|-------------|-----------|
| Temporal leakage | `LeakageError` halts pipeline |
| Validation bleed | LOYO prevents by design |
| Calibration leakage | `CalibrationLeakageError` (SHA-256 hash guard) |
| Post-calibration vanishing | Pre/post metrics compared |
| Model instability | `PromotionGate` rejects regressions |

**Gaps:**
- No automated rejection for "stronger model with materially increased drawdown"
- No formal rollback protocol for code changes that fail validation
- Pre-flight checks exist (`PreRunValidationError`) but not comprehensive

**Weighted: 12/16**

---

### S16: Final Deliverables — Score: 2/4 (Weight: 3)

**Status: PARTIAL**

`DeliverablesManager` creates versioned output directories with predictions, reports, audit, and metadata. Exports confidence intervals, risk reports, evaluation matrices, config snapshots.

**Gaps:** Missing from the directive's complete 28-artifact package:
- Formal dataset catalog
- Domain integration guide document
- Governance audit summary
- Conflict resolution log
- Compute budget report
- A/B test results
- Deployment runbook
- Complete reproducibility package (partial)

**Weighted: 6/12**

---

### S17: Operating Summary — Score: 3/4 (Weight: 2)

The system is point-in-time valid, empirically tested, decision-relevant, calibrated, and domain-specific to NCAA basketball with deep tournament knowledge. Not fully autonomous or bitwise-reproducible as a guarantee.

**Weighted: 6/8**

---

**Part I Subtotal: 163/228 (71.5%)**

---

## Part II — Deployment, Operations, and Governance (Sections 18–25)

### S18: Phase 9 — Production Deployment & Live Monitoring — Score: 2/4 (Weight: 4)

**Status: PARTIAL — Integrated but not production-tested**

- `DeploymentPipeline` is called from `sota.py` (line 3131) with shadow mode comparison
- `ShadowMode` and `CanaryDeployment` exist as code with tests
- Drift detection via PSI in `pipeline_monitor.py`

The deployment code IS integrated into the pipeline path (unlike what some prior evaluations claimed), but via `try/except` blocks — failures are logged, not blocking.

**Gaps:**
- No live monitoring dashboard
- No automated retraining triggers
- Shadow/canary checks are informational, not gating
- A/B framework exists but not exercised in practice
- No `ModelStore` rotation policy

**Weighted: 8/16**

---

### S19: Phase 10 — Data Engineering & Pipeline Resilience — Score: 3/4 (Weight: 4)

**Status: MOSTLY FUNCTIONAL**

- DAG orchestrator (`src/data/ingestion/dag.py`) with topological sorting, content-hash idempotency caching, and cache invalidation
- Circuit breaker pattern with CLOSED/OPEN/HALF_OPEN state machine and persistence
- Data versioning with snapshot/restore
- Schema validation contracts
- Data freshness SLA tracking in `pipeline_monitor.py`

**Gaps:**
- Not a production-grade DAG (no Airflow/Prefect equivalent — this is a lightweight Python implementation)
- No schema versioning
- Idempotency guarantees not proven under failure conditions
- No formal freshness SLA registry document

**Weighted: 12/16**

---

### S20: Computational Budget & Resource Prioritization — Score: 2/4 (Weight: 3)

**Status: PARTIAL**

- `BudgetManager` with per-stage allocation and cost tracking
- `ResourceTracker` (wall-clock, CPU, memory) with budget enforcement
- `CostTracker` with dollar-cost attribution, Pareto frontier analysis, baseline comparison
- Cost tracked and logged in `sota.py` (lines 1019, 1033)

**Gaps:**
- No evidence of actual budget enforcement blocking a pipeline run
- Cost-per-improvement ratio not demonstrated with real data
- No Pareto frontier outputs shown
- No prioritized search strategy (Bayesian optimization of compute allocation)

**Weighted: 6/12**

---

### S21: Human-in-the-Loop Governance & Approval Gates — Score: 2/4 (Weight: 4)

**Status: PARTIAL — Integrated but not blocking**

- `GovernanceGate` and `ComplianceGate` initialized in `sota.py.__init__()` (line 1053)
- Compliance checks run at data_loading, model_training, and calibration stages (line 3171)
- `GovernanceAuditLog` writes compliance results to JSONL (line 3195)
- `AuthorityMatrix` with action classifications verified in CI quality gates
- `RBACManager` and `EscalationManager` exist

**Gaps:**
- Governance gates are **informational, not blocking** — wrapped in `try/except`, failures logged as debug
- No approval expiration enforcement
- Authority matrix rules not actually checked before real high-stakes actions (e.g., Kaggle submission)
- No real-time escalation automation

**Weighted: 8/16**

---

### S22: Multi-Agent Conflict Resolution Protocol — Score: 2/4 (Weight: 2)

**Status: PARTIAL — Implemented but only exercised in multi-agent mode**

- Four conflict categories defined
- Resolution hierarchy: safety > evidence > orchestrator > human
- Audit Agent veto on safety matters
- Dissent registry

**Gap:** Entirely dependent on `run_multi_agent()` which is not the default execution path. In the default `run()` path, conflicts cannot arise because there's a single controller.

**Weighted: 4/8**

---

### S23: Testing Strategy & CI/CD Integration — Score: 3/4 (Weight: 4)

**Status: MOSTLY FUNCTIONAL**

- **114 test files** across all major modules
- GitHub Actions CI: lint, type check, multi-Python (3.9/3.10/3.11) tests
- Quality gates: import validation, structure checks, governance compliance
- Temporal integrity tests: leakage canary, walk-forward replay, date integrity
- Coverage enforcement at 60% in CI (`--cov-fail-under=60`)

**Gaps:**
- Coverage target is **60% in CI** vs. directive's 90% expectation (pyproject.toml says 80%)
- `ruff format` and `mypy` use `|| true` — **failures don't block merges**
- No formal nightly E2E system test suite
- No pipeline ordering determinism test as a CI gate

**Weighted: 12/16**

---

### S24: Domain-Specific Integration Guides — Score: 2/4 (Weight: 3)

**Status: PARTIAL — Knowledge in code, not documented**

Deep domain expertise is embedded:
- Historical upset rates by seed matchup (1985–2025)
- Tournament-specific sigma calibration per round
- Coach tournament experience modeling (Final Four, Elite Eight, Sweet 16)
- Injury impact with severity estimation
- Transfer portal tracking with eligibility
- Pool competition with size-adaptive competitor archetypes
- Women's basketball support (HerHoopStats, NCAA NET)

**Gap:** Domain knowledge is implicit in code. No `<domain_integration_guide>`, `<domain_data_quirks_checklist>`, or `<regulatory_compliance_checklist>` artifacts per directive template.

**Weighted: 6/12**

---

### S25: Extended Failure Modes, Updated Deliverables & Operating Summary — Score: 2/4 (Weight: 3)

**Status: PARTIAL**

Some extended failure modes addressed:
- Shadow/canary bypass → `DeploymentPipeline` code integrated
- Pipeline without idempotency → DAG caching implemented
- Budget overrun → `BudgetManager` with enforcement
- Unauthorized actions → `AuthorityMatrix` in CI quality gates
- Code without CI → GitHub Actions pipeline exists

**Gaps:**
- Enforcement is via soft integration (try/except), not hard gating
- No consolidated 28-artifact deliverables package
- Extended failure modes in deployment/governance paths are informational, not blocking

**Weighted: 6/12**

---

**Part II Subtotal: 62/108 (57.4%)**

---

## Composite Score

| Part | Score | Max | Percentage |
|------|-------|-----|------------|
| Part I: Core Research & Validation (S1–S17) | 163 | 228 | 71.5% |
| Part II: Deployment, Ops & Governance (S18–S25) | 62 | 108 | 57.4% |
| **Total** | **225** | **336** | **67.0%** |

**Adjusted Score: 74/100**

The adjusted score accounts for: (a) the directive is domain-agnostic but this system is tournament-specific with exceptional domain depth, (b) operational integration exists but is soft rather than hard-gated, and (c) the system demonstrably works for its primary use case (Kaggle submission).

---

## Top 10 Gaps by Impact

### Tier 1: High-Impact Structural Gaps

| # | Gap | Sections | Current | Target | Effort | Impact |
|---|-----|----------|---------|--------|--------|--------|
| 1 | **`sota.py` monolith (8,532 lines)** | S12 | Monolith with 80+ methods | ≤500-line orchestrator delegating to stage modules | High | +4 pts |
| 2 | **Research loop not autonomous end-to-end** | S14 | Components exist, no autonomous iteration demonstrated | 5+ autonomous iterations with logged results | Medium | +4 pts |
| 3 | **Meta-learning uses hardcoded regime tables** | S7 | Lookup tables for regime preferences | Learn regime weights from LOYO performance data | Medium | +5 pts |
| 4 | **Governance gates are informational, not blocking** | S21 | `try/except` wrapping, failures logged as debug | Hard gates that halt pipeline on compliance failure | Low | +4 pts |

### Tier 2: Missing Integration & Artifacts

| # | Gap | Sections | Current | Target | Effort | Impact |
|---|-----|----------|---------|--------|--------|--------|
| 5 | **No consolidated evaluation matrix** | S13 | Metrics scattered across modules | Single JSON per experiment with all V7 metrics | Low | +4 pts |
| 6 | **Deployment checks informational, not gating** | S18 | Shadow/canary via try/except | Shadow check must pass before predictions exported | Low | +4 pts |
| 7 | **No complete deliverables package** | S16, S25 | `DeliverablesManager` exists but partial | Generate all 28 artifacts on pipeline completion | Medium | +3 pts |
| 8 | **Ensemble weights not dynamically optimized** | S8 | Fixed 55/15/15/15 split | Per-year LOYO-optimized weights via Optuna | Medium | +3 pts |

### Tier 3: Documentation & CI Gaps

| # | Gap | Sections | Current | Target | Effort | Impact |
|---|-----|----------|---------|--------|--------|--------|
| 9 | **CI quality gates don't block** | S23 | `ruff format` and `mypy` use `|| true` | Remove `|| true`, raise coverage to 80% | Low | +2 pts |
| 10 | **No domain integration guide document** | S24 | Knowledge embedded in code | Write formal domain guide per directive template | Low | +2 pts |

---

## Reconciliation with Prior Evaluations

| Evaluation | Score | This Assessment |
|------------|-------|-----------------|
| `EVALUATION_SCORE.md` (2026-03-08) | 62/100 | Too low — didn't account for governance/deployment/research modules that do exist |
| `AGENT_DIRECTIVE_V7_SCORE_CARD.md` (2026-03-09) | 68/100 | Reasonable but slightly harsh — doesn't credit try/except integration |
| `COMPLIANCE_SCORE_2026.md` (2026-03-08) | 82/100 | Slightly generous — gives near-full credit for code existence without verifying pipeline integration |
| **This gap analysis** | **74/100** | Credits operational integration via try/except but penalizes for non-blocking/informational enforcement |

The key methodological distinction: this evaluation recognizes that governance, deployment, and research modules ARE imported and called from `sota.py`, but penalizes that they are wrapped in `try/except` blocks where failures are swallowed as debug logs rather than halting the pipeline.

---

## Path to 85/100

| Priority | Action | Sections | Points | Effort |
|----------|--------|----------|--------|--------|
| P0 | Decompose `sota.py` into ≤500-line stage modules using existing `stages/` pattern | S12 | +4 | High |
| P0 | Make governance gates blocking (remove try/except swallowing) | S21 | +4 | Low |
| P1 | Run research loop for 5+ autonomous iterations with logged results | S14 | +4 | Medium |
| P1 | Train MetaLearner on actual LOYO performance data | S7 | +5 | Medium |
| P1 | Create consolidated evaluation matrix JSON per experiment | S13 | +4 | Low |
| P1 | Make deployment shadow check a hard gate before export | S18 | +4 | Low |
| P2 | Generate complete deliverables package on each run | S16, S25 | +3 | Medium |
| P2 | Optimize ensemble weights per-year via LOYO Optuna | S8 | +3 | Medium |
| P2 | Fix CI: remove `|| true`, raise coverage to 80% | S23 | +2 | Low |
| P2 | Write domain integration guide document | S24 | +2 | Low |
| **Total potential** | | | **+35** | |

Implementing P0 + P1 items would bring the score to approximately **85/100**.

---

## Strengths Worth Preserving

1. **Temporal integrity is best-in-class** — LOYO, `shift(1)`, leakage canaries, `LeakageError`, `cutoff_date` enforcement, tournament date cutoffs per year
2. **Calibration infrastructure is production-quality** — Brier decomposition, ECE/MCE, multiple scaling methods, round-specific sigma, SHA-256 leakage guard
3. **Deep domain expertise everywhere** — Historical upset rates (1985–2025), Bayesian seed priors, coach experience, pool competition archetypes, travel burden
4. **Comprehensive feature engineering** — 78-dimensional matchup vectors with principled acceptance criteria (0.001 Brier rule)
5. **Extensive test suite** — 114 test files covering agents, governance, robustness, leakage, schemas, deployment, reproducibility
6. **Broad data universe** — 20+ sources with DAG ingestion, circuit breakers, content-hash caching, data versioning
7. **RDoF audit is distinctive** — 58+ cataloged constants with derivation tiers and sensitivity analysis

---

## Section Score Summary

| # | Section | Score | Weight | Weighted | Max |
|---|---------|-------|--------|----------|-----|
| 1 | Mission & Non-Negotiable Principles | 4 | 5 | 20 | 20 |
| 2 | Multi-Agent System Architecture | 2 | 3 | 6 | 12 |
| 3 | Shared Contracts & Required Logs | 3 | 4 | 12 | 16 |
| 4 | Problem Definition & Utility Mapping | 3 | 3 | 9 | 12 |
| 5 | Dataset Discovery & Lineage | 3 | 5 | 15 | 20 |
| 6 | Feature Discovery Engine | 3 | 5 | 15 | 20 |
| 7 | Model Search & Meta-Learning | 2 | 5 | 10 | 20 |
| 8 | Ensemble Optimization & Calibration | 3 | 5 | 15 | 20 |
| 9 | Decision Optimization Layer | 3 | 4 | 12 | 16 |
| 10 | Backtesting & Simulation Realism | 3 | 5 | 15 | 20 |
| 11 | Skeptical Audit Layer | 3 | 5 | 15 | 20 |
| 12 | Codebase Review & Refactoring | 2 | 3 | 6 | 12 |
| 13 | Required Evaluation Matrix | 2 | 4 | 8 | 16 |
| 14 | Continuous Autonomous Research Loop | 2 | 4 | 8 | 16 |
| 15 | Failure Mode Rejection | 3 | 4 | 12 | 16 |
| 16 | Final Deliverables | 2 | 3 | 6 | 12 |
| 17 | Operating Summary | 3 | 2 | 6 | 8 |
| 18 | Deployment & Live Monitoring | 2 | 4 | 8 | 16 |
| 19 | Data Engineering & Pipelines | 3 | 4 | 12 | 16 |
| 20 | Compute Budget & Resources | 2 | 3 | 6 | 12 |
| 21 | Governance & Approval Gates | 2 | 4 | 8 | 16 |
| 22 | Conflict Resolution Protocol | 2 | 2 | 4 | 8 |
| 23 | Testing & CI/CD | 3 | 4 | 12 | 16 |
| 24 | Domain-Specific Integration | 2 | 3 | 6 | 12 |
| 25 | Extended Failure Modes & Deliverables | 2 | 3 | 6 | 12 |
| | **TOTALS** | | **96** | **225** | **336** |

**Raw: 225/336 = 67.0% → Adjusted: 74/100**
