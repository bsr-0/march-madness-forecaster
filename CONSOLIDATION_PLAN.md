# Branch Consolidation Plan — March Madness Forecaster

## Branch Inventory & Assessment

| Branch | Unique Commits | Key Contribution | Status |
|--------|---------------|-----------------|--------|
| `Branch0` | 0 (stale) | None — behind main by 3 commits | **SKIP** |
| `branch1` | 4 | Governance: cache hygiene, feature manifests, artifact provenance, production runner | **MERGE** |
| `claude/condescending-cori` | 1 | toRvik R package wrapper + provider integration | **SUPERSEDED** by elastic-lederberg |
| `claude/elastic-lederberg` | 1 | Torvik hardening: R wrapper + validator + CSV fallback + circuit breaker reports | **MERGE** (superset of condescending-cori) |
| `claude/feature-engineering-evaluation-0k0dW` | 3 | 15 statistical solutions: MI screening, DoF budget, stacking weights, calibration selector | **MERGE** |
| `claude/laughing-hoover` | 1 | opp_two_pt_pct_allowed: box-score precedence + Torvik fallback | **MERGE** |
| `claude/madness-quant-system-6LWqm` | 1 | Two-layer quant decision system (Kaggle + ESPN optimization) | **MERGE** |

## Conflict Analysis

### Overlap: condescending-cori ↔ elastic-lederberg
Both created `src/data/scrapers/torvik_r.py` (now deleted — Phase 4 cleanup) and modified `providers.py` + `torvik.py`.
**Resolution:** elastic-lederberg is a strict superset — it adds TorVikValidator, CSV fallback strategies, circuit breaker reporting, and ProviderResult metadata on top of the R wrapper. We merge elastic-lederberg only, then cherry-pick any unique condescending-cori bits (CI workflow R install step, `__init__.py` exports, `fetch_schedule()` method).

### All Other Branches: No File Conflicts
- branch1 touches governance/, pipeline/stages/, ml/evaluation — no overlap with Torvik or quant modules
- laughing-hoover touches only feature_engineering.py — isolated
- feature-engineering-evaluation touches feature_selection.py, statistical_audit.py, stacking_weights.py, method_selector.py — no overlap
- madness-quant-system introduces entirely new src/quant/ module — isolated

## Merge Order (dependency-aware)

1. **branch1** — Foundation: governance infrastructure (cache hygiene, provenance) needed by later stages
2. **elastic-lederberg** — Data layer: Torvik hardening builds on existing scrapers
3. **condescending-cori unique bits** — Cherry-pick: CI workflow R install, `fetch_schedule()`, `__init__.py` exports
4. **laughing-hoover** — Feature fix: opp_two_pt_pct_allowed precedence (depends on feature_engineering.py which torvik changes feed into)
5. **feature-engineering-evaluation** — Statistical rigor: MI screening, DoF budget, stacking weights (builds on stable feature layer)
6. **madness-quant-system** — Decision layer: quant system on top of complete pipeline

## Statistician-Level Quality Gates

Each merge verified against:
- **Leakage safety:** No holdout contamination in training/calibration paths
- **Point-in-time correctness:** Features use tournament cutoff dates
- **Calibration integrity:** Temperature scaling locked for production; research methods additive only
- **Feature dimension:** TEAM_FEATURE_DIM = 79 preserved
- **Production lock:** configs/production_2026.json unchanged
- **LOYO protocol:** Leave-One-Year-Out cross-validation enforced

## Post-Merge Verification
- `pytest tests/ -x --tb=short`
- `ruff check src/ tests/`
- Verify production entrypoint: `python src/run_production_2026.py --dry-run` (if available)
