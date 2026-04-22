# Data Scrapers & Ingestion — Audit Report

**Date:** 2026-03-25 • **Scope:** `src/data/scrapers/`, `src/data/ingestion/`, `src/data/models/` • **Compacted:** 2026-04-13.

Multi-source pipeline with DAG orchestration, provider cascade, circuit breakers, and schema validation. Audit found 5 high / 9 medium severity issues. Post-audit decisions:

- **trank.php promoted to primary** data source over cbbdata.com (2026-04-06) — server-side date filtering is a second independent leakage guard. See `COUNCIL_LESSONS.md §3 row 22`.
- **FanDuel / DraftKings scrapers deleted** 2026-04-12 (replaced by `TheOddsAPIScraper`). See `MEMORY.md §2 D10`. Issues 3.x and 4.x below that referenced those scrapers no longer apply.

**Cross-references:** current provider ordering reflects the 2026-04-06 pivot; `TOURNAMENT_START_DATES` hardcoded-dict SPOF tracked as `COUNCIL_LESSONS.md §2 O16`.

---

## 1. Architecture

| Layer | Key Files | Purpose |
|-------|-----------|---------|
| **Scrapers** | `torvik.py`, `sports_reference.py`, `the_odds_api.py`, `cbbpy_rosters.py`, `espn_picks.py` | Raw extraction |
| **Resilience** | `circuit_breaker.py`, `_retry.py` | Circuit breaker + exponential backoff with jitter; respects `Retry-After` / 429 / 5xx |
| **Validation** | `scrapers/schemas.py`, `ingestion/validators.py` | Pydantic schemas + statistical quality gates |
| **Orchestration** | `dag.py`, `collector.py`, `historical_pipeline.py` | Topological DAG, marker-based idempotency |
| **Providers** | `providers.py`, `game_fetchers.py` | Cascade: trank.php (primary, 2026-04-06) → ESPN → sportsdataverse → cbbpy |
| **Models** | `game_flow.py`, `player.py`, `ingestion/schemas.py` | Canonical data models |

Strengths worth preserving: Pydantic schema validation, circuit breaker state persistence, leakage-aware Torvik collection, DAG context fingerprinting.

---

## 2. Priority Matrix (post-pivot, still-relevant issues)

| Pri | Action | Effort | Impact | Reference |
|-----|--------|--------|--------|-----------|
| **P0** | Move API keys from query params to `Authorization: Bearer` headers | Low | Prevents credential leakage in server/proxy/referer logs | `torvik.py:670` |
| **P0** | Remove `os.environ["CBD_API_KEY"]` token caching | Low | Eliminates `ps aux` / crash-log / subprocess credential exposure | `torvik.py:642` |
| **P1** | Implement atomic file writes (temp + `os.replace()`) | Low | Prevents JSON corruption on crash / concurrent writes | `collector.py:579-583`, `dag.py:313`, `sports_reference.py:130` |
| **P1** | Add pre-write response validation (status code, content-type, min payload) at the HTTP boundary | Medium | Catches corrupt scraper responses before they hit disk | `collector.py:159-619` |
| **P1** | Promote advisory leakage warning to `LeakageError` in strict mode | Low | Prevents tournament data contamination in training | `collector.py:169-181` |
| **P2** | Add structural assertions after HTML parsing (column counts, empty detection) | Medium | Detects Sports Reference layout changes early instead of silently returning empty | `sports_reference.py:135-156`, `tournament_bracket.py:67-71` |
| **P2** | Externalize season-specific event IDs (Odds API) to config; add discovery step | Low | Prevents annual breakage when seasons roll over | any remaining season-id scraper |
| **P2** | Add TTL-based cache expiration (not only version-keyed) | Medium | Stale data can't persist indefinitely | `game_fetchers.py:787-806` |
| **P2** | Fix DAG marker transactional semantics (write only after downstream succeeds, or invalidate on failure) | Medium | Prevents skipping tasks after partial failures | `dag.py:167-242` |
| **P3** | Narrow `except Exception` handlers; add provider health counters | Medium | Improves failure visibility across providers | `the_odds_api.py`, `cbbpy_rosters.py:229-306`, `providers.py` |
| **P3** | Per-thread `requests.Session()` or confirm no session-level state mutation | Low | Eliminates potential cookie-jar / header races across 8-worker pool | `game_fetchers.py:644-665` |
| **P3** | Validate `len(row) >= expected_columns` before CSV indexing | Low | Malformed rows currently caught silently by outer try/except | `torvik.py:539-545` |
| **P3** | Surface dropped-player counts from soft roster validation instead of silent drop | Low | Prevents entire roster loss from going unseen | `scrapers/schemas.py:619-626` |
| **P3** | Dedup team records by ID (dict-keyed, not list append) | Low | Duplicate API responses don't produce two rows | `torvik.py:700` |

---

## 3. Low-Severity Backlog

| Issue | File |
|---|---|
| Cloudflare detection checks only first 500 chars | `torvik.py:844` |
| Path traversal via `year` param in cache paths | all scrapers |
| Unencrypted odds cache (world-readable) | odds scraper |
| Single hardcoded User-Agent (ban = total failure) | `torvik.py:362` |
| Referer header leaks scraper origin | `torvik.py:810` |
| No explicit encoding in file I/O (UTF-8 assumed) | multiple |
| ESPN 403 logged at DEBUG (should be WARNING) | `espn_picks.py:193` |
| Dedup audit trail logged to stderr only, not persisted | `game_fetchers.py:406-462` |
| BOM stripping only in CSV parsing, not other text endpoints | `torvik.py:528` |
