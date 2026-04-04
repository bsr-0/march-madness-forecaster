# Data Provenance & Leakage Audit

Last updated: 2026-04-04

This document describes every data source used by the backtest and prediction
pipeline, how each was acquired, what date filtering is applied, and the
contamination risk level. It exists so future agents can quickly understand what
is safe and what requires care.

---

## Table of Contents

1. [Backtest Data Sources](#backtest-data-sources)
2. [Data Acquisition Methods](#data-acquisition-methods)
3. [Leakage Guards](#leakage-guards)
4. [Pipeline-Only Data Sources (Not Used by Backtest)](#pipeline-only-data-sources)
5. [Deleted / Purged Files](#deleted--purged-files)
6. [Known Gaps](#known-gaps)

---

## Backtest Data Sources

The backtest (`scripts/mc_pool_backtest.py`) uses exactly **4 data sources** plus
1 set of hardcoded constants. Nothing else.

### 1. Tournament Seeds

| Field | Value |
|---|---|
| **Files** | `data/raw/historical/tournament_seeds_{year}.json` |
| **Years** | 2008-2019, 2021-2025 (no 2020 — COVID) |
| **Content** | `team_id`, `seed` (1-16), `region` (East/West/South/Midwest). 68 teams/year |
| **Acquisition** | `bigdance` Python package (Warren Nolan data) with Sports Reference HTML fallback |
| **Script** | `src/data/scrapers/bracket_ingestion.py` → `BracketIngestionPipeline` |
| **Date filter** | None needed — seeds are fixed on Selection Sunday, before tournament |
| **Risk** | **NONE** — bracket assignments only, inherently pre-tournament |

### 2. Tournament Results

| Field | Value |
|---|---|
| **Files** | `data/raw/historical/tournament_results_{year}.json` |
| **Years** | 2008-2019, 2021-2025 |
| **Content** | Game outcomes: team IDs, scores, round names. Play-in ("FF") games excluded by backtest |
| **Acquisition** | ESPN Scoreboard API (`seasontype=3`) with Sports Reference HTML fallback. Historical years (1985-2025) manually embedded in `scripts/generate_tournament_results.py` |
| **API endpoint** | `site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard` |
| **Date filter** | Tournament window only (mid-March to early April) |
| **Risk** | **NONE** — used as ground truth for bracket scoring, not as features |

### 3. Torvik Barthag

| Field | Value |
|---|---|
| **Files** | `data/raw/historical/torvik_{year}.json` |
| **Years** | 2008-2019, 2021-2025 |
| **Content** | Per-team barthag (expected win % vs average team), AdjOE, AdjDE, adj_tempo |
| **Acquisition** | **Locally computed** from `historical_games_{year}.json` — NOT scraped from barttorvik.com |
| **Script** | `scripts/compute_pretournament_barthag.py` |
| **Date filter** | `game_date < TOURNAMENT_START_DATES[year]` — hardcoded per year |
| **Computation** | Raw points-per-100-possessions → 15-iteration opponent-strength adjustment → Pythagorean formula: `barthag = AdjOE^11.5 / (AdjOE^11.5 + AdjDE^11.5)` |
| **Metadata** | `data_type: "pre_tournament_computed"`, `cutoff_date`, `tournament_start`, `n_games`, `n_teams` |
| **Guard** | `_validate_pretournament()` in backtest rejects files without `data_type: "pre_tournament_computed"` |
| **Risk** | **LOW** — computed with date cutoff, validated at load time |

### 4. Torvik Four Factors

| Field | Value |
|---|---|
| **Files** | `data/raw/torvik_four_factors_{year}.json` (symlinked from `data/raw/historical/`) |
| **Years** | 2008-2019, 2021-2025 |
| **Content** | Per-team Dean Oliver Four Factors: eFG%, TO%, ORB%, FTR (offense + defense) |
| **Acquisition** | **Locally computed** from `historical_games_{year}.json` box score data |
| **Script** | `scripts/compute_pretournament_four_factors.py` |
| **Date filter** | `game_date < TOURNAMENT_START_DATES[year]` — same cutoff as barthag |
| **Computation** | `eFG% = (FGM + 0.5*3PM) / FGA`, `TO% = TO / poss`, `ORB% = ORB / (ORB + opp_DRB)`, `FTR = FTA / FGA`, plus defensive mirrors |
| **Metadata** | `data_type: "pre_tournament_computed"`, `cutoff_date`, `tournament_start` |
| **Guard** | `_validate_pretournament()` in backtest |
| **Risk** | **LOW** — computed with date cutoff, validated at load time |

### 5. Seed Pick Rates (hardcoded)

| Field | Value |
|---|---|
| **Source** | `src/data/seed_pick_model.py` → `SEED_PICK_RATES` |
| **Content** | Per-seed public pick percentages for each round (R64 through CHAMP) |
| **Acquisition** | Hardcoded constants computed at module import from: (a) historical 1985-2025 seed win rates, (b) ESPN "Who Picked Whom" chalk bias calibration (2015-2024) |
| **Date filter** | N/A — static constants |
| **Risk** | **NONE** |

---

## Data Acquisition Methods

### Upstream: Historical Games (`historical_games_{year}.json`)

These files are the **input** to barthag and four-factors computation. They are
NOT directly used by the backtest.

| Field | Value |
|---|---|
| **Files** | `data/raw/historical/historical_games_{year}.json` |
| **Acquisition** | Multi-provider cascade with minimum 500-game threshold |
| **Primary** | ESPN Scoreboard API — day-by-day queries, `seasontype=2` (regular) + `seasontype=3` (postseason), 8 concurrent threads |
| **Fallback 1** | `sportsdataverse` Python package (`sportsdataverse.mbb.load_mbb_pbp()`) |
| **Fallback 2** | `cbbpy` Python package (`cbbpy.mens_scraper.get_games()`) |
| **Date range** | Nov 1 through Apr 15 — **includes tournament games** (e.g., 811 in 2024 file) |
| **Script** | `src/data/ingestion/game_fetchers.py` → `HistoricalGameFetcher.fetch_season()` |
| **Tournament games?** | YES — but filtered out by compute scripts before feature computation |

### Acquisition Chain

```
ESPN Scoreboard API (or sportsdataverse/cbbpy fallback)
    ↓
historical_games_{year}.json  [contains ALL games including tournament]
    ↓ filter: game_date < TOURNAMENT_START_DATES[year]
    ↓
compute_pretournament_barthag.py    → torvik_{year}.json
compute_pretournament_four_factors.py → torvik_four_factors_{year}.json
    ↓ validated by _validate_pretournament()
    ↓
mc_pool_backtest.py  [uses only pre-tournament data]
```

---

## Leakage Guards

### Scraper Guards (`src/data/scrapers/torvik.py`)

| Guard | Behavior | Scope |
|---|---|---|
| `_check_tournament_date_guard()` | **Always raises** `LeakageError` if `today >= TOURNAMENT_START_DATES[year]` | All `fetch_*()` methods, `providers.py` callers |
| `_validate_cache_timestamp()` | **Always raises** `LeakageError` if `scraped_at >= TOURNAMENT_START_DATES[year]` | All cache loads, `load_from_json()` |
| `strict_leakage` default | `True` — all callers are strict by default | Constructor parameter |
| Date filtering in `_rankings_from_trank_csv()` | Adds `begin`/`end` params to restrict to pre-tournament window | trank.php CSV endpoint |

### Backtest Guards (`scripts/mc_pool_backtest.py`)

| Guard | Behavior |
|---|---|
| `_validate_pretournament(data, filepath)` | Raises `ValueError` if `data_type != "pre_tournament_computed"`. Applied to torvik and four_factors files |

### Unfiltered Scraper Endpoints (documented, guarded at caller level)

These endpoints **cannot** be date-filtered. They are protected by the
`_check_tournament_date_guard()` at the caller level, which prevents
post-tournament scraping entirely.

| Endpoint | Why no filtering | Guard |
|---|---|---|
| cbbdata.com API (`/torvik/ratings`) | API has no date params | `_check_tournament_date_guard` at caller |
| team_results CSV (`/{year}_team_results.csv`) | Static endpoint | `_check_tournament_date_guard` at caller |
| Player CSV (`/getadvstats.php`) | No date params | `_check_tournament_date_guard` at caller |
| `providers.py` `_from_barttorvik_csv()` | Raw HTTP request | `_check_tournament_date_guard` added |

---

## Pipeline-Only Data Sources

These are used by the main prediction pipeline but **NOT by the backtest**.

### Team Metrics (`team_metrics_{year}.json`)

| Field | Value |
|---|---|
| **Content** | pace, off_rtg, def_rtg, wins, losses, SRS, SOS per team |
| **Source** | `sportsdataverse` Python package or Sports Reference HTML scraper |
| **Date filter** | **NONE** |
| **Metadata** | **NONE** (no `data_as_of`, `scraped_at`, or `data_type`) |
| **Risk** | **MEDIUM-HIGH** — contains post-tournament records (e.g., UConn 2024 shows 37-3 instead of pre-tournament 31-3) |
| **Backtest impact** | None — not loaded by backtest |

### Massey Ordinals (`external_massey_composite_{year}.json`)

| Field | Value |
|---|---|
| **Content** | Composite meta-ranking aggregating 50+ ranking systems |
| **Source** | Kaggle competition CSV (`MMasseyOrdinals.csv`) |
| **Date filter** | `max_day` capped at Selection Sunday via `_compute_max_ranking_day()` in `kaggle_loader.py` |
| **Risk** | **LOW** — day guard prevents post-Selection-Sunday rankings |
| **Backtest impact** | None — not loaded by backtest |

### External Ratings (`external_{system}_{year}.json`)

| Field | Value |
|---|---|
| **Content** | Individual ranking system ratings (POM/KenPom, MOR/Massey, SAG/Sagarin, etc.) |
| **Source** | Derived from Massey Ordinals with same `max_day` guard |
| **Risk** | **LOW** — inherits Massey's Selection Sunday cap |
| **Backtest impact** | None |

### KenPom (`data/kaggle/kenpom_barttorvik.json`)

| Field | Value |
|---|---|
| **Content** | KenPom + Torvik metrics in Kaggle CSV-to-JSON format |
| **Source** | Static Kaggle competition dataset (Nishaanamin conversion) |
| **Date filter** | **NONE** |
| **Risk** | **MEDIUM** — static file, no temporal guard |
| **Backtest impact** | None — only used by Kaggle converter script |

### Torvik Shooting 2026 (`data/raw/torvik_shooting_2026.json`)

| Field | Value |
|---|---|
| **Content** | 3PT% and FT% per team |
| **Source** | barttorvik.com player CSV endpoint |
| **Metadata** | `data_as_of: "2026-03-16"` (pre-tournament) |
| **Risk** | **LOW** — verified pre-tournament |
| **Backtest impact** | None — not loaded by backtest. Historical shooting files (2008-2025) deleted |

### 2026 Current-Year Files

| File | `data_as_of` | Tournament Start | Status |
|---|---|---|---|
| `data/raw/torvik_2026.json` | 2026-03-16 | 2026-03-17 | **SAFE** |
| `data/raw/torvik_four_factors_2026.json` | 2026-03-16 | 2026-03-17 | **SAFE** |
| `data/raw/torvik_shooting_2026.json` | 2026-03-16 | 2026-03-17 | **SAFE** |

---

## Deleted / Purged Files

The following contaminated files were deleted as part of the zero-tolerance
leakage cleanup:

| Files | Count | Reason |
|---|---|---|
| `torvik_shooting_{2008-2025}.json` (raw + historical) | 36 | Post-tournament player stats, no date filtering possible |
| `torvik_2020.json`, `torvik_four_factors_2020.json` (raw + historical) | 4 | COVID year — no tournament, no metadata |
| `torvik_{2005,2006,2007}.json` (raw) | 3 | Broken symlinks, no game data to recompute |
| `torvik_four_factors_{2005,2006,2007}.json` (raw) | 3 | No game data to recompute |
| `data/raw/historical/torvik_2026.json` | 1 | `generated_at: 2026-03-18` — post-tournament start |

**Total: 47 contaminated files removed.**

---

## Known Gaps

### Low Priority (no backtest impact)

1. **`noseed_model._load_team_stats()`** loads Torvik files during model training
   without calling `_validate_pretournament()`. The files themselves are clean
   (have metadata), and `max_year` prevents future-year leakage, but there is no
   runtime guard at load time.

2. **`team_metrics_{year}.json`** files contain post-tournament win/loss records.
   Not used by backtest but used by the main pipeline. No metadata or guards.

3. **`historical_games_{year}.json`** files include tournament games (811 in 2024).
   This is by design — the compute scripts filter them out. But downstream code
   that uses these files directly must apply its own date filter.

---

## Verification Commands

```bash
# Verify all Torvik files have pre-tournament metadata
for f in data/raw/historical/torvik_*.json data/raw/torvik_*.json; do
  python3 -c "import json; d=json.load(open('$f')); print('CLEAN' if d.get('data_type') or d.get('data_as_of') else 'CONTAMINATED', ' ', '$f')"
done

# Run backtest leakage tests
python -m pytest tests/test_leakage_guards.py -v

# Run Torvik scraper tests
python -m pytest tests/test_torvik_scraper.py -v

# Lint scraper code
ruff check src/data/scrapers/torvik.py
```
