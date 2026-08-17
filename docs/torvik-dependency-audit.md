# Torvik Pipeline Dependency Audit

**Date**: 2026-03-23
**Scope**: Complete mapping of all files, classes, functions, data files, and configs that depend on the Torvik scraper/data pipeline.

> **Stale as of 2026-08-16**: Section 4a's `torvik_four_factors_{year}.json` and the monthly
> dated snapshot files were consolidated into `torvik_{year}.json` (keys `four_factors` and
> `four_factors_snapshots`) and deleted from `data/raw/historical/`. See
> `scripts/consolidate_torvik_four_factors.py` and `docs/torvik-interface-specification.md` §6b
> for the current schema. Section 4b's `data/raw/` (non-historical) file inventory is unaffected.

---

## 1. Core Scraper Files (DELETE candidates)

| File | Lines | Exports | Risk |
|------|-------|---------|------|
| `src/data/scrapers/torvik.py` | 1708 | `BartTorvikScraper`, `TorVikTeam`, `TorVikValidator`, `TorVikValidationError` | **HIGH** — 30+ consumers |
| ~~`src/data/scrapers/torvik_r.py`~~ | ~~313~~ | ~~`TorvikRWrapper`~~ | **DELETED** — replaced by cbbdata API strategy in `BartTorvikScraper` |
| ~~`scripts/create_missing_torvik.py`~~ | ~~70~~ | ~~Utility script~~ | **DELETED** — zero callers |

---

## 2. Direct Import Consumers (src/)

### 2a. Pipeline Core (CRITICAL PATH)

| File | Import/Usage | How Used |
|------|--------------|----------|
| `src/pipeline/sota.py:56` | `from ..data.scrapers.torvik import BartTorvikScraper` | Loads torvik data via `_load_team_stat_sources()`, passes `torvik_data=t` to feature engineering |
| `src/pipeline/stages/data_loader.py:42` | `from ...data.scrapers.torvik import BartTorvikScraper` | `BartTorvikScraper().load_from_json()`, `._dict_to_team()`, `TorVikValidator.validate_teams()` — **heaviest consumer** (~100 references) |
| `src/pipeline/config.py:361` | `torvik_json: Optional[str] = None` | Config field pointing to torvik JSON path |

### 2b. Data Ingestion

| File | Import/Usage | How Used |
|------|--------------|----------|
| `src/data/ingestion/collector.py:17` | `from ... import BartTorvikScraper` | Creates `BartTorvikScraper(cache_dir)`, calls `fetch_four_factors()`, `fetch_shooting_stats()` |
| `src/data/ingestion/providers.py:107-108` | `_from_cbbdata_api`, `_from_barttorvik_csv` | Two provider methods in `MultiSourceProvider.fetch_torvik_ratings()` (`_from_torvik_r` removed) |
| `src/data/ingestion/validators.py:121` | `cross_validate_torvik_sources()` | Compares metrics between two Torvik-source payloads |
| `src/data/ingestion/historical_pipeline.py:55,293` | `include_torvik`, `_collect_torvik_ratings()` | Historical data collection for years 2008+ |
| `src/data/ingestion/extended_historical_ingest.py:34` | `TORVIK_START = 2008` | Controls which years get torvik data |

### 2c. Feature Engineering

| File | Import/Usage | How Used |
|------|--------------|----------|
| `src/data/features/feature_engineering.py:911` | `_TORVIK_FEATURES` set | Defines which features come from Torvik; disables them pre-2008 |
| `src/data/features/feature_engineering.py:1118` | `torvik_data: Optional[Dict]` param | `_build_team_features()` reads Four Factors, shooting, coach stats from torvik dict |
| `src/data/features/proprietary_metrics.py:2997` | `torvik_to_game_records()` | Converts Torvik team stats to GameRecord objects for training |
| `src/data/features/materialization.py` | References torvik data | Feature materialization |
| `src/conference_tournament/data_enrichment.py:53` | `enrich_torvik_teams()` | Merges Four Factors + shooting into main torvik payload |
| `src/conference_tournament/predictor.py:137` | `from_torvik_json()` class method | Builds predictor directly from torvik JSON |

### 2d. Scrapers Package

| File | Import/Usage | How Used |
|------|--------------|----------|
| `src/data/scrapers/__init__.py:23` | Re-exports `BartTorvikScraper` | Public API of scrapers package (`TorvikRWrapper` removed) |
| `src/data/scrapers/tournament_context.py:208+` | References Barttorvik | Coach data scraping from barttorvik.com |

### 2e. Governance & Monitoring

| File | Import/Usage | How Used |
|------|--------------|----------|
| `src/governance/production_runner.py:35` | `"torvik_json"` in required fields | Production runner validates torvik_json exists |
| `src/governance/production_validator.py:19` | `"torvik_json"` in required fields | Validates torvik_json is present in config |
| `src/monitoring/pipeline_monitor.py:34` | `"torvik_*.json": 168` | Monitors torvik file freshness (168h = weekly) |
| `src/monitoring/pre_tournament_checklist.py:345` | `"torvik_*.json"` | Pre-tournament data completeness check |

### 2f. Agents & Forecaster

| File | Import/Usage | How Used |
|------|--------------|----------|
| `src/agents/data_agent.py:82,94` | `torvik_map` | Agent passes torvik_map through pipeline |
| `src/agents/feature_agent.py:84,102,112` | `torvik_map`, `torvik_data=t` | Agent reads torvik_map for feature building |
| `src/agents/concrete.py:161` | `loaded_data.torvik_map` | Concrete agent passes torvik data |
| `src/forecaster/state_machine.py:275-365` | Direct file reads of `torvik_{year}.json` | Loads historical torvik data for backtesting |
| `src/forecaster/matchups.py:28` | Comment reference | Documents efficiency source |

### 2g. Other src/ references

| File | Import/Usage |
|------|--------------|
| `src/main.py:3403` | `ConferenceTournamentPredictor.from_torvik_json()` |
| `src/data/normalize.py` | Referenced by torvik scraper (dependency, not consumer) |
| `src/data/schemas.py` | References torvik in schema definitions |
| `src/data/coverage_audit.py` | Torvik coverage checks |
| `src/workflows/live_protocol.py` | Torvik in live data protocol |
| `src/workflows/reproducible.py` | Torvik in reproducibility workflow |
| `src/reproducibility/run_hasher.py` | Hashes torvik source files |
| `src/ml/evaluation/metrics_validation.py` | References torvik data |
| `src/ml/evaluation/model_card.py` | Documents torvik as data source |

---

## 3. Test Files

### 3a. Torvik-Specific Tests (5 files — DELETE candidates)

| File | What It Tests |
|------|--------------|
| `tests/test_torvik_scraper.py` | `TorVikValidator`, `BartTorvikScraper` cbbstat/CSV parsing |
| `tests/test_torvik_refactor.py` | `TorVikTeam`, `TorVikValidator`, `BartTorvikScraper` construction/caching |
| `tests/test_torvik_pipeline_rigor.py` | Strict validation, Bayesian shrinkage, CSV aggregation |
| ~~`tests/test_torvik_r.py`~~ | **DELETED** — `TorvikRWrapper` removed |
| `tests/test_shotquality_proxy.py:307` | `TorVikTeam.to_dict()` roundtrip |

### 3b. Tests With Torvik References (10+ files — MODIFY candidates)

| File | Nature of Reference |
|------|-------------------|
| `tests/test_leakage_guards.py:56-100` | Tests `BartTorvikScraper._check_tournament_date_guard()` (5 test methods) |
| `tests/test_historical_year_integration.py:23-62` | Loads torvik from `data/raw/` cache, tests four factors |
| `tests/test_scrapers_coverage.py:739-978` | `TestTorVikTeam` (4 tests), `TestBartTorvikScraper` (18 tests) |
| `tests/test_conference_tournament.py:575-893` | `from_torvik_json()` integration tests |
| `tests/test_tournament_context.py:303` | Feature engineering receives context from torvik |
| `tests/test_pipeline_stages.py` | References torvik in stage tests |
| `tests/test_ingestion_collector.py` | Tests collector torvik paths |
| `tests/test_ingestion_providers.py` | Tests provider torvik methods |
| `tests/test_production_2026_freeze.py` | Freeze artifact includes torvik file hashes |
| `tests/test_selection_sunday_validation.py` | Torvik data presence checks |
| `tests/test_sota_pipeline.py` | Pipeline integration with torvik |
| `tests/test_proprietary_metrics_coverage.py` | Torvik-to-game-records conversion |
| `tests/test_baseline_materialization_loader.py` | Torvik data loading |
| `tests/test_data_versioning.py` | Torvik file versioning |
| `tests/data_integrity/test_provider_robustness.py` | Provider fallback tests |
| `tests/test_extended_historical.py` | Extended historical torvik collection |
| `tests/test_deployment_governance_coverage.py` | Governance checks |
| `tests/test_main_and_eval_coverage.py` | CLI torvik commands |
| `tests/test_metrics_validation.py` | Metrics from torvik data |
| `tests/test_run_hasher.py` | Hash verification |
| `tests/test_pre_tournament_checklist.py` | Checklist items |
| `tests/test_pipeline_monitor.py` | Freshness monitoring |

---

## 4. Data Files (89 JSON files)

### 4a. Historical Torvik Data (`data/raw/historical/`)

- `torvik_2005.json` through `torvik_2026.json` (22 files)
- `torvik_four_factors_2025.json`, `torvik_shooting_2025.json` (2 files)

### 4b. Current Season Data (`data/raw/`)

- `torvik_2026.json` (main ratings)
- `torvik_four_factors_YYYY.json` (2005-2025, 21 files)
- `torvik_shooting_YYYY.json` (2005-2026, 22 files)
- `torvik_four_factors_2024.json`, `torvik_four_factors_2025.json` (latest)

Total: **~89 JSON files** across `data/raw/` and `data/raw/historical/`

### 4c. Config & Manifest References

| File | Reference |
|------|-----------|
| `configs/production_2026.json` | `torvik_json` field |
| `data/raw/manifest_2026.json` | Torvik file paths in manifest |
| `data/raw/historical/historical_manifest_*.json` | Historical torvik paths |
| `artifacts/pipeline_freeze_2026.json` | Hash of torvik.py source |
| `artifacts/feature_inventory.json` | Torvik-sourced features |
| `pipeline_freeze.json` | Source hash |
| `features/MANIFEST.yaml` | Feature sourcing |
| `.github/workflows/data-ingestion.yml` | CI torvik steps |

---

## 5. Contracts

| File | References |
|------|-----------|
| `contracts/feature_contracts.yaml` | 12 features sourced from `system: torvik` (lines 93, 120, 147, 174, 204, 231, 258, 285, 1527, 1808) |

Features contracted to torvik:
- `effective_fg_pct` (eFG%)
- `turnover_rate` (TO%)
- `offensive_reb_rate` (ORB%)
- `free_throw_rate` (FTR)
- `opp_effective_fg_pct`
- `opp_turnover_rate`
- `defensive_reb_rate`
- `opp_free_throw_rate`
- Plus 4 additional supplementary features

---

## 6. Scripts

| File | Usage |
|------|-------|
| ~~`scripts/create_missing_torvik.py`~~ | **DELETED** — zero callers |
| `scripts/run_conference_predictions.py:37-65` | Scrapes torvik Four Factors + Shooting |
| `scripts/repair_2026_data_quality.py:8,336,569-579` | Populates coach data from Barttorvik |
| `scripts/create_team_id_mapping.py:63` | Loads team IDs from torvik |
| `scripts/build_dashboard_data.py` | Dashboard references torvik |
| `scripts/generate_web_data.py` | Web data from torvik |
| `scripts/evaluate_conference_predictions.py` | Conference prediction eval |
| `scripts/validate_training_data.py` | Training data validation |
| `scripts/backfill_four_factors_2005_2009.py` | Historical backfill |

---

## 7. Risk Assessment

### CRITICAL (breaks production pipeline)
1. **`src/pipeline/stages/data_loader.py`** — ~100 torvik references, the primary consumer. Rebuild must preserve `StatSourcesResult.torvik_map` interface.
2. **`src/pipeline/sota.py`** — Passes `torvik_data=t` to feature engineering for every team.
3. **`src/data/features/feature_engineering.py`** — Reads 20+ fields from torvik dict per team.
4. **`src/governance/production_validator.py`** — Requires `torvik_json` in production config.
5. **`artifacts/pipeline_freeze_2026.json`** — Contains hash of `torvik.py`. Will fail verification.

### HIGH (breaks ingestion/historical pipelines)
6. **`src/data/ingestion/collector.py`** — Creates `BartTorvikScraper`, calls 3 fetch methods.
7. **`src/data/ingestion/providers.py`** — Two provider methods for torvik ratings.
8. **`src/data/ingestion/historical_pipeline.py`** — Historical collection for 2008+.
9. **`contracts/feature_contracts.yaml`** — 12 features list torvik as source system.

### MEDIUM (breaks secondary features)
10. **`src/conference_tournament/`** — `from_torvik_json()`, `enrich_torvik_teams()`
11. **`src/forecaster/state_machine.py`** — Direct file reads of torvik JSONs
12. **`src/agents/`** — 3 agent files pass torvik_map through

### LOW (tests, scripts, docs)
13. **5 torvik-specific test files** — Can delete alongside scraper
14. **22+ test files with torvik references** — Need mock/fixture updates
15. **9 scripts** — Utility scripts, some deletable
16. **89 JSON data files** — Can keep or regenerate

---

## 8. Public Interface Contract

Any replacement MUST provide these interfaces:

```python
# CLASS: BartTorvikScraper (or replacement)
class BartTorvikScraper:
    def __init__(self, cache_dir=None, cache_ttl_seconds=..., strict_leakage=False): ...
    def fetch_current_rankings(self, year: int, strict: bool = False) -> List[TorVikTeam]: ...
    def fetch_four_factors(self, year: int) -> Dict[str, Dict]: ...
    def fetch_shooting_stats(self, year: int) -> Dict[str, Dict]: ...
    def load_from_json(self, filepath: str) -> List[TorVikTeam]: ...
    def _dict_to_team(self, data: dict) -> TorVikTeam: ...
    @staticmethod
    def data_completeness_report(teams: List[TorVikTeam]) -> Dict[str, float]: ...
    @staticmethod
    def _normalize_team_name_to_id(name: str) -> str: ...

# DATACLASS: TorVikTeam (24 fields)
@dataclass
class TorVikTeam:
    team_id: str
    name: str
    conference: str
    t_rank: int
    barthag: float
    adj_offensive_efficiency: float
    adj_defensive_efficiency: float
    adj_tempo: float
    effective_fg_pct: float
    turnover_rate: float
    offensive_reb_rate: float
    free_throw_rate: float
    opp_effective_fg_pct: float
    opp_turnover_rate: float
    defensive_reb_rate: float
    opp_free_throw_rate: float
    two_pt_pct: float
    three_pt_pct: float
    three_pt_rate: float
    ft_pct: float
    block_pct: float
    steal_pct: float
    opp_two_pt_pct: float
    opp_three_pt_pct: float
    opp_three_pt_rate: float
    wab: float
    wins: int
    losses: int
    conf_wins: int
    conf_losses: int
    def to_dict(self) -> dict: ...

# CLASS: TorVikValidator
class TorVikValidator:
    @classmethod
    def validate_team(cls, team, strict=False) -> List[str]: ...
    @classmethod
    def validate_teams(cls, teams, strict=False) -> Dict[str, List[str]]: ...
    @classmethod
    def validate_four_factors(cls, data) -> Dict[str, List[str]]: ...

# EXCEPTION
class TorVikValidationError(Exception): ...

# FUNCTION (proprietary_metrics.py)
def torvik_to_game_records(torvik_teams, ...) -> List[GameRecord]: ...

# FUNCTION (data_enrichment.py)
def enrich_torvik_teams(torvik_data, data_dir, year) -> dict: ...
```

---

## 9. Summary Statistics

| Category | Count |
|----------|-------|
| Source files with torvik imports | 16 |
| Source files with torvik references | 35+ |
| Test files affected | 27 |
| Data files (JSON) | 89 |
| Config/manifest files | 8 |
| Contract files | 1 (12 features) |
| Scripts | 9 |
| **Total files touched by rebuild** | **~90+ code files + 89 data files** |
