# Repository Audit Report

**Date:** 2026-03-15
**Repository:** march-madness-forecaster
**Scope:** Full codebase audit — structure, security, code quality, dependencies, testing, CI/CD

---

## Executive Summary

This is a well-engineered NCAA March Madness prediction system with **325 Python source files**, **119 test files**, and **1,500+ data files**. The codebase demonstrates strong engineering practices including DAG-based pipeline orchestration, point-in-time feature engineering, comprehensive test coverage (80% minimum enforced), and a researcher degrees of freedom (RDoF) audit framework.

**No critical vulnerabilities were found.** The primary areas for improvement are exception handling specificity and dependency bound tightening.

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 2 |
| Medium | 5 |
| Low | 3 |

---

## 1. Project Overview

- **Purpose:** NCAA tournament probability prediction with ML ensemble, Monte Carlo simulation, and bracket portfolio optimization
- **ML Stack:** Logistic Regression (0.70) + LightGBM (0.15) + XGBoost (0.15) with temperature-scaled calibration
- **Features:** 79-dim team vector → 91-dim matchup features (79 diff + 5 absolute + 7 interactions)
- **Training Pool:** 2005–2025 with exponential decay weighting (0.85/year, 0.15 floor)
- **License:** MIT (declared in README; no separate LICENSE file)
- **Python Version:** >=3.8 (setup.py), target 3.9 (pyproject.toml), tested on 3.10 (CI)

---

## 2. Security Findings

### 2.1 No Hardcoded Secrets — PASS

- `.env.example` contains only placeholder values
- `.gitignore` correctly excludes `.env`, `*.pem`, `*.key`, `credentials.json`
- CI/CD uses GitHub Secrets for sensitive values
- `secrets-qaqc.yml` workflow scans PRs for credential leaks

### 2.2 API Key Handling — MEDIUM

**File:** `src/data/ingestion/providers.py:563-570`

API key is sent in both `Authorization: Bearer` and `x-api-key` headers simultaneously, increasing exposure surface. Key value is not validated for length/format before use.

**Recommendation:** Use a single auth method; validate key format; mask keys in any debug logging.

### 2.3 Custom .env Parser — LOW

**File:** `src/data/kaggle_downloader.py:80-128`

Custom `.env` file parser does not whitelist allowed variable names — could overwrite critical env vars like `PATH`.

**Recommendation:** Switch to `python-dotenv` or whitelist allowed variable names.

### 2.4 Subprocess Usage — PASS

`src/ml/evaluation/rdof_audit.py` uses `subprocess.run()` with hardcoded arguments (list form, not shell=True), proper timeouts, and error handling. Safe.

---

## 3. Code Quality Findings

### 3.1 Broad Exception Catching — HIGH

**15+ locations** use bare `except Exception` that silently swallow errors:

| File | Lines |
|------|-------|
| `src/data/ingestion/providers.py` | 121, 144, 181, 195, 241, 243, 298, 396, 460, 471, 580, 711, 729 |
| `src/data/ingestion/historical_pipeline.py` | 136, 211, 289 |
| `src/ml/calibration/calibration.py` | 478, 630 |
| `src/ml/evaluation/robustness_suite.py` | 143, 160, 176 |
| `src/deployment/pipeline.py` | 273 |
| `src/simulation/mc_calibration.py` | 274 |
| `src/data/kaggle_downloader.py` | 70, 128 |

**Risk:** Masks genuine errors (including `KeyboardInterrupt`, `SystemExit`), makes debugging difficult, and can cause silent data loss.

**Recommendation:** Replace with specific exception types (`ValueError`, `TypeError`, `requests.RequestException`, etc.) and log caught exceptions at DEBUG level.

### 3.2 JSON Deserialization Without Type Validation — MEDIUM

**Files:** `src/governance/compliance.py:464-467`, `src/data/scrapers/circuit_breaker.py:214-230`

Pattern: `data = json.loads(line); Record(**data)` without validating field types. Malformed records are silently skipped with `except (json.JSONDecodeError, TypeError): continue`.

**Recommendation:** Add explicit type/schema validation before dataclass instantiation; log malformed records.

### 3.3 Positive Patterns

- **Retry logic** (`src/data/scrapers/_retry.py`): Proper exponential backoff with Retry-After header respect
- **Circuit breaker** (`src/data/scrapers/circuit_breaker.py`): Correct state machine with proper re-raising
- **Data validators** (`src/data/ingestion/validators.py`): Comprehensive field-level validation
- **Team name normalization** (`src/data/normalize.py`): Consolidated from 5+ files into single module
- **Context managers**: Proper `with` statements for file handles throughout
- **Type hints**: Comprehensive throughout with mypy configured

---

## 4. Dependency Analysis

### 4.1 requirements.txt (19 packages)

| Package | Version Constraint | Risk |
|---------|--------------------|------|
| numpy | >=1.24.4,<1.27.0 | OK — bounded |
| pandas | >=1.5.3,<2.2.0 | OK — bounded |
| scipy | >=1.10.1,<1.12.0 | OK — bounded |
| scikit-learn | >=1.3.0,<1.5.0 | OK — bounded |
| lightgbm | >=4.3.0,<4.6.0 | OK — bounded |
| **torch** | **>=2.0.0** | **MEDIUM — no upper bound** |
| **torch-geometric** | **>=2.4.0** | **MEDIUM — no upper bound** |
| **kaggle** | **>=1.5.16** | **LOW — no upper bound** |
| cbbpy | ==2.0.2 | OK — pinned |
| sportsipy | >=0.6.0 | LOW — no upper bound |
| requests | >=2.31.0 | OK |
| beautifulsoup4 | >=4.12.0 | OK |

**Recommendation:** Add upper bounds to `torch`, `torch-geometric`, and `kaggle` to prevent breaking changes.

### 4.2 Version Mismatch

- `setup.py` declares `python_requires=">=3.8"`
- `pyproject.toml` targets Python 3.9
- CI runs Python 3.10

**Recommendation:** Align to `>=3.9` everywhere.

### 4.3 Missing LICENSE File

MIT license is stated in README but no `LICENSE` file exists in the repository root. Some tools and platforms require a standalone license file.

**Recommendation:** Add a `LICENSE` file.

---

## 5. Testing & CI/CD

### 5.1 Test Coverage — GOOD

- **119 test files** covering unit, integration, and end-to-end scenarios
- **80% minimum coverage** enforced in `pyproject.toml`
- Integration tests marked with `@pytest.mark.integration`
- Shared fixtures in `tests/conftest.py`
- Backtest validation against Kaggle historical data
- RDoF audit for researcher degrees of freedom

### 5.2 CI/CD Workflows (6 workflows)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Ruff linting |
| `data-ingestion.yml` | Schedule + manual | Data collection |
| `secrets-qaqc.yml` | PR | Credential scanning |
| `deploy-staging.yml` | — | Staging deployment |
| `repair-dates.yml` | — | Data repair |
| `jekyll-gh-pages.yml` | — | Docs deployment |

### 5.3 CI Gaps — MEDIUM

- **No test execution in CI** — `ci.yml` only runs linting (Ruff), not `pytest`
- **No dependency vulnerability scanning** — no `pip audit`, `safety`, or Dependabot
- **No type checking in CI** — mypy is configured but not run in workflows

**Recommendation:** Add pytest, pip-audit, and mypy steps to CI pipeline.

---

## 6. Architecture Assessment

### Strengths

1. **Point-in-time data engineering** prevents train/test leakage
2. **RDoF audit framework** with 58+ tracked constants across 3 tiers — strong reproducibility
3. **DAG-based pipeline** with conditional stage execution
4. **Multi-source ingestion** with circuit breaker and retry patterns
5. **Frozen configuration artifacts** for prospective evaluation
6. **Comprehensive CLI** with 30+ well-documented commands
7. **Scaffold architecture** (GNN/Transformer disabled) allows future experimentation without impacting production

### Areas for Improvement

1. **325 source files** is large for this domain — some modules (agents, deployment, governance) may be over-engineered for the current use case
2. **No database layer** — all persistence is file-based JSON/CSV, which limits query capabilities
3. **Tight coupling** between some scrapers and specific website HTML structures (fragile to upstream changes)

---

## 7. Summary of Recommendations

### High Priority
1. Replace 15+ broad `except Exception` catches with specific exception types
2. Add pytest execution to CI pipeline

### Medium Priority
3. Add upper bounds to `torch`, `torch-geometric`, and `kaggle` dependencies
4. Add `pip audit` / Dependabot for vulnerability scanning
5. Add mypy type checking to CI
6. Add explicit type validation for JSON deserialization
7. Use single API auth method in CBBDATA provider

### Low Priority
8. Add standalone `LICENSE` file
9. Align Python version requirement to `>=3.9` across all config files
10. Switch custom .env parser to `python-dotenv`
11. Add docstrings to performance-critical helper functions in `simulation/` and `ml/ensemble/`
