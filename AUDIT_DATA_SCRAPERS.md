# Data Scrapers & Ingestion Pipeline — Audit Report

**Date:** 2026-03-25
**Scope:** `src/data/scrapers/`, `src/data/ingestion/`, `src/data/models/`

---

## Executive Summary

The data ingestion system is a well-architected multi-source pipeline with DAG-based orchestration, provider cascading, circuit breakers, and comprehensive validation. However, the audit identified **5 high-severity** and **9 medium-severity** issues across credential handling, file atomicity, HTML parsing robustness, and cache management.

---

## 1. Architecture Overview

| Layer | Key Files | Purpose |
|-------|-----------|---------|
| **Scrapers** | `torvik.py`, `sports_reference.py`, `betting_markets.py`, `cbbpy_rosters.py`, `espn_picks.py` | Raw data extraction from external sources |
| **Resilience** | `circuit_breaker.py`, `_retry.py` | Circuit breaker pattern, exponential backoff with jitter |
| **Validation** | `scrapers/schemas.py`, `ingestion/validators.py` | Pydantic schema validation, statistical quality gates |
| **Orchestration** | `dag.py`, `collector.py`, `historical_pipeline.py` | DAG executor, single-year/multi-year collection |
| **Providers** | `providers.py`, `game_fetchers.py` | Provider cascade (ESPN → sportsdataverse → cbbpy) |
| **Models** | `game_flow.py`, `player.py`, `ingestion/schemas.py` | Canonical data models |

---

## 2. Strengths

- **Circuit Breaker** (`circuit_breaker.py`): State persistence, recovery timeouts, half-open probing prevents cascading failures.
- **Retry Logic** (`_retry.py`): Exponential backoff with jitter, respects `Retry-After` headers, handles 429/5xx correctly.
- **Schema Validation** (`scrapers/schemas.py`): Pydantic models with monotonicity checks, range validation, cross-team structural constraints, and hard/soft tolerance thresholds.
- **Ingestion Validators** (`validators.py`): 7+ functions covering variance detection, minimum team counts, box-score coverage rates, and single-team bias detection.
- **Leakage Awareness** (`collector.py:169–181`): Warns when collecting Torvik data after tournament start dates.
- **Provider Cascade** (`providers.py`): Ordered fallback across ESPN, sportsdataverse, and cbbpy with automatic failover.
- **DAG Executor** (`dag.py`): Topological sort, circular dependency detection, context fingerprinting, marker-based idempotency.

---

## 3. High-Severity Issues

### 3.1 API Token Cached in Environment Variable
**File:** `torvik.py:642`

The cbbdata API token is written to `os.environ["CBD_API_KEY"]` for subprocess access. This exposes credentials via `ps aux`, environment dumps, crash logs, and child processes.

**Recommendation:** Use a secure token store or pass tokens via `Authorization: Bearer` headers. Remove `os.environ` caching.

### 3.2 API Key Passed in Query Parameters
**File:** `torvik.py:670`

API keys are passed as URL query parameters (`params={"key": api_key}`), making them visible in server access logs, proxy logs, and browser history/referer headers.

**Recommendation:** Move to `Authorization: Bearer` header.

### 3.3 Non-Atomic File Writes
**Files:** `collector.py:579–583`, `dag.py:313`, `sports_reference.py:130`

All JSON artifacts and DAG markers are written via direct `json.dump()` without temp-file + atomic-rename. Concurrent writes or crashes mid-write produce corrupted/truncated JSON files.

```python
# Current (collector.py:579-583)
with open(p, "w") as f:
    json.dump(payload, f, indent=2)

# Recommended
import tempfile
with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False, suffix=".tmp") as tmp:
    json.dump(payload, tmp, indent=2)
os.replace(tmp.name, p)
```

### 3.4 Post-Ingestion Validation Only
**File:** `collector.py:159–619`

Validation runs after data is scraped and written to disk. Corrupted scraper responses reach the filesystem before validation catches them. No pre-ingestion schema check exists at the HTTP response boundary.

**Recommendation:** Add lightweight response-level validation (status code, content-type, minimum payload size) before writing.

### 3.5 Advisory-Only Leakage Warning
**File:** `collector.py:169–181`

When collecting Torvik data after tournament start, only a `logger.warning()` is emitted. The pipeline continues, risking data leakage into training features.

**Recommendation:** Add a `strict_leakage=True` mode that raises `LeakageError` in production contexts.

---

## 4. Medium-Severity Issues

### 4.1 Fragile HTML Selectors
**Files:** `sports_reference.py:135–156`, `tournament_bracket.py:67–71`

Hardcoded table IDs (`adv_school_stats`), `data-stat` attributes, and region div IDs will silently return empty data if Sports Reference changes its HTML structure. No structural validation or fallback alerts.

**Recommendation:** Add schema assertions (e.g., expected column count) after parsing. Log at WARNING when parsed data is empty.

### 4.2 Hardcoded Event IDs That Expire Yearly
**Files:** `betting_markets.py:411` (FanDuel `69420.3`), `betting_markets.py:597` (DraftKings category `1000`)

These IDs are season-specific. They will return stale/404 data for 2027+.

**Recommendation:** Add a discovery step that resolves current event IDs from a search/listing endpoint, or externalize to config.

### 4.3 DAG Marker Persistence After Downstream Failure
**File:** `dag.py:167–242`

If task N succeeds and writes a marker, but task N+1 fails, subsequent runs skip task N (marker exists) even if N needs re-execution. No transactional rollback.

**Recommendation:** Only write markers after the full DAG completes, or implement marker invalidation on downstream failure.

### 4.4 No TTL-Based Cache Expiration
**File:** `game_fetchers.py:787–806`

Caches are version-keyed but have no time-based expiration. Stale data persists indefinitely unless the cache version constant is manually bumped.

**Recommendation:** Add a TTL field to cache metadata and check age on load.

### 4.5 CSV Index Bounds Not Checked
**File:** `torvik.py:539–545`

Accessing `row[header.get('team', 1)]` without verifying row length. Short rows from malformed CSV responses cause `IndexError` (caught by outer try/except but silently skips records).

**Recommendation:** Validate `len(row) >= expected_columns` before indexing.

### 4.6 Broad Exception Swallowing
**Files:** `betting_markets.py:443–448`, `cbbpy_rosters.py:229–306`, `providers.py`

Multiple `except Exception` blocks catch and log at DEBUG/INFO level, masking unexpected failure modes. Provider failures are not aggregated.

**Recommendation:** Narrow exception types. Add provider health counters and alert when all providers fail consecutively.

### 4.7 Concurrent Session Reuse Across Threads
**File:** `game_fetchers.py:644–665`

A single `requests.Session()` is shared across 8 `ThreadPoolExecutor` workers. While `urllib3` is thread-safe, shared cookie jars and header mutations can cause subtle races.

**Recommendation:** Use per-thread sessions or confirm no session-level state mutations occur during concurrent fetches.

### 4.8 Soft Roster Validation Drops Players Silently
**File:** `scrapers/schemas.py:619–626`

Invalid player records are logged and dropped without surfacing to the caller. A malformed API response could silently lose entire rosters.

**Recommendation:** Return a validation result object with dropped-player counts so callers can decide whether to proceed.

### 4.9 Duplicate Team Records Not Deduplicated
**File:** `torvik.py:700`

Parsed team records are appended to a list without checking if the team ID already exists. If the API returns a team twice with different stats, both records persist.

**Recommendation:** Use a dict keyed by team ID and prefer the most recent/complete record.

---

## 5. Low-Severity Issues

| Issue | File | Description |
|-------|------|-------------|
| Cloudflare detection checks only first 500 chars | `torvik.py:844` | Challenge page content may appear later in response body |
| Path traversal via `year` parameter in cache paths | All scrapers | `year="../../../etc"` could write outside cache dir (mitigated by Path resolve) |
| Unencrypted odds cache | `betting_markets.py` | Market odds written to world-readable directory |
| Hardcoded User-Agent string | `torvik.py:362` | Single UA; no rotation. Ban = total scraper failure |
| Referer header leaks scraper origin | `torvik.py:810` | Sets referer to `barttorvik.com` — may trigger IP blocking |
| No explicit encoding in file I/O | Multiple files | Assumes UTF-8; Windows CSV sources may use Latin-1 |
| ESPN 403 logged at DEBUG | `espn_picks.py:193` | Access denial should log at WARNING minimum |
| No dedup audit trail | `game_fetchers.py:406–462` | Date inconsistency warnings logged to stderr only, not persisted |
| BOM stripping only in CSV parsing | `torvik.py:528` | Other text endpoints may also have BOM |

---

## 6. Data Flow Summary

```
External Sources          Scrapers              Validation           Storage
─────────────────    ─────────────────    ──────────────────    ─────────────
Bart Torvik     ───► torvik.py        ─┐
Sports Ref      ───► sports_ref.py    ─┤  schemas.py           data/raw/
ESPN API        ───► espn_picks.py    ─┼─► (Pydantic)     ─┐
FanDuel/DK      ───► betting_mkts.py  ─┤  validators.py    ├──► JSON artifacts
CBBpy/ESPN      ───► cbbpy_rosters.py ─┘  (quality gates)  ─┘

                    Orchestration
                ─────────────────────
                dag.py (topological sort)
                collector.py (single-year)
                historical_pipeline.py (multi-year)
                providers.py (cascade fallback)
                game_fetchers.py (concurrent fetch + dedup)
```

---

## 7. Recommendations Priority Matrix

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **P0** | Move API keys from query params to auth headers | Low | Prevents credential leakage in logs |
| **P0** | Remove `os.environ` token caching | Low | Eliminates process-level credential exposure |
| **P1** | Implement atomic file writes (temp + rename) | Low | Prevents data corruption on crash |
| **P1** | Add pre-write response validation | Medium | Catches corrupt data before it hits disk |
| **P1** | Make leakage guard blocking in production | Low | Prevents tournament data from contaminating features |
| **P2** | Add HTML parser structural assertions | Medium | Detects Sports Reference layout changes early |
| **P2** | Externalize season-specific event IDs to config | Low | Prevents annual breakage of betting scrapers |
| **P2** | Add TTL-based cache expiration | Medium | Prevents stale data from persisting indefinitely |
| **P2** | Fix DAG marker transactional semantics | Medium | Prevents incorrect task skipping after partial failures |
| **P3** | Narrow exception handlers and add health counters | Medium | Improves failure visibility and alerting |
| **P3** | Add per-thread sessions for concurrent fetching | Low | Eliminates potential session state races |

---

## 8. Files Audited

| File | Lines | Category |
|------|-------|----------|
| `src/data/scrapers/torvik.py` | ~900 | Scraper |
| `src/data/scrapers/sports_reference.py` | ~300 | Scraper |
| `src/data/scrapers/betting_markets.py` | ~650 | Scraper |
| `src/data/scrapers/cbbpy_rosters.py` | ~770 | Scraper |
| `src/data/scrapers/espn_picks.py` | ~200 | Scraper |
| `src/data/scrapers/schemas.py` | ~650 | Validation |
| `src/data/scrapers/circuit_breaker.py` | ~200 | Resilience |
| `src/data/scrapers/_retry.py` | ~150 | Resilience |
| `src/data/scrapers/tournament_bracket.py` | ~100 | Scraper |
| `src/data/ingestion/collector.py` | ~1085 | Orchestration |
| `src/data/ingestion/dag.py` | ~352 | Orchestration |
| `src/data/ingestion/game_fetchers.py` | ~1109 | Data Fetching |
| `src/data/ingestion/validators.py` | ~468 | Validation |
| `src/data/ingestion/providers.py` | ~350 | Provider Cascade |
| `src/data/ingestion/schemas.py` | ~225 | Data Models |
| `src/data/ingestion/historical_pipeline.py` | ~240 | Orchestration |
| `src/data/ingestion/extended_historical_ingest.py` | ~200 | Orchestration |
| `src/data/models/game_flow.py` | ~321 | Data Models |
| `src/data/models/player.py` | ~242 | Data Models |
