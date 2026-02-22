# Training Data Audit Report

**Date:** 2026-02-22
**Scope:** Full audit of data ingestion, cleaning, team resolution, feature
engineering, and cross-year consistency for the March Madness forecaster
training pipeline.

---

## Executive Summary

This audit identified **17 issues** across the training data pipeline, ranging
from silent data corruption that affects model training to team resolution
mismatches that drop tournament teams from evaluation. The most severe issues
are the complete zeroing of team metrics for 2005-2009, the systematic team ID
namespace mismatch between game data and metrics data (zero direct matches
across all years), fake dates in 2005-2024 historical games that break
recency weighting and temporal features, and the fact that the well-designed
`TeamNameResolver` alias table is bypassed by all core pipeline modules.

---

## CRITICAL Issues (model correctness at risk)

### C1. Team Metrics Completely Zeroed for 2005-2009

**Files:** `data/raw/historical/team_metrics_{2005..2009}.json`

All team metrics (`off_rtg`, `def_rtg`, `pace`) are `0.0` for 95%+ of teams
in seasons 2005-2009:

| Year | Total teams | All-zero | Non-zero |
|------|-------------|----------|----------|
| 2005 | 330         | 315      | 15       |
| 2006 | 334         | 325      | 9        |
| 2007 | 337         | 327      | 10       |
| 2008 | 341         | 330      | 11       |
| 2009 | 344         | 331      | 13       |

The `_load_year_samples` function in `sota.py:3184-3191` does guard against
this (returns empty when all `off_rtg` values are zero), so these years are
effectively **skipped** during multi-year training. However:

- This means 5 years of potentially useful training data are silently discarded.
- There is no warning surfaced to the user about this data loss.
- The 9-15 non-zero teams per year still pass the all-zero check (the check
  tests whether ALL values are zero, not whether MOST are), so a handful of
  partially-populated years could leak corrupted data in edge cases.

**Root cause:** The Sports Reference scraper failed to parse `def_rtg` (and
likely other fields) for pre-2010 HTML layouts. The `historical_pipeline.py`
code has a 50% zero threshold patch (`historical_pipeline.py:486-496`) that
computes `def_rtg` from game records, but the pre-2010 team_metrics files
were generated before this patch existed and were never re-scraped.

**Impact:** 5 years of training data unavailable. Model trains on fewer
seasons than intended, reducing sample size.

---

### C2. Zero Team ID Overlap Between Games and Metrics

**Files:** `data/raw/historical/historical_games_*.json`, `team_metrics_*.json`

Game team IDs use **mascot-suffixed** forms from CBBpy (e.g.,
`alabama_crimson_tide`, `duke_blue_devils`, `air_force_falcons`), while metric
team IDs use **school-only** forms from Sports Reference (e.g., `alabama`,
`duke`, `air_force`).

Tested across 2005, 2010, 2022: **zero direct matches** between game team IDs
and metric team IDs.

| Year | Game team IDs | Metric team IDs | Direct matches |
|------|---------------|-----------------|----------------|
| 2005 | 340           | 330             | 0              |
| 2010 | 501           | 347             | 0              |
| 2022 | 678           | 358             | 0              |

Both pipelines use it differently:

- **Materialization pipeline** (`materialization.py`): Uses `_align_source_team_ids`
  with fuzzy matching (`_resolve_team_id`) via `difflib.SequenceMatcher`.
  This attempts name normalization, key containment, and fuzzy matching at
  0.84 threshold. Quality depends entirely on how well fuzzy matching handles
  "alabama_crimson_tide" -> "alabama".

- **SOTA pipeline** (`sota.py:3230-3350`): Uses a 4-pass resolver:
  (a) direct match, (b) CBBpy CSV display_name lookup, (c) prefix matching,
  (d) `TeamNameResolver` fuzzy fallback. This is more robust but still
  fragile.

**Impact:** Every team lookup requires fuzzy resolution. Misresolved teams
get wrong metrics. Unresolved teams are silently dropped from training.

---

### C3. Fake Dates in Historical Games (2005-2024)

**Files:** `data/raw/cache/cbbpy_historical_games_{2005..2024}.json`

For years 2005-2024, all games are assigned a **single fake date** of
`{year-1}-11-01` (the season start date):

| Year | team_games | Unique dates | All same? |
|------|------------|--------------|-----------|
| 2015 | 11,850     | 1            | Yes (`2014-11-01`) |
| 2020 | 11,532     | 1            | Yes (`2019-11-01`) |
| 2023 | 12,444     | 1            | Yes (`2022-11-01`) |
| 2024 | 12,484     | 1            | Yes (`2023-11-01`) |
| 2025 | 12,582     | 151          | No (real dates) |

This occurs because the `_collect_season_games_fast` path
(`historical_pipeline.py:230-262`) uses CBBpy's bulk `get_games_season`
call which does not return per-game dates, and falls back to the season start.

**Downstream effects:**

1. **Materialization pipeline** (`materialization.py`): Loads these dates
   as-is via `pd.to_datetime`. When all dates are `2024-11-01`:
   - `season_progress` = 0.0 for all games (line 423)
   - `rest_days` = NaN for all but first game per team (line 415)
   - `games_in_last_7_days` = total games played (all on same day)
   - `back_to_back` = all True (since rest_days=0)
   - All rolling window features (`_l3`, `_l5`, `_l10`) lose temporal meaning

2. **SOTA `_load_year_samples`** (`sota.py:3352-3372`): Detects single-date
   payloads and **infers** chronological dates from `game_id` ordering.
   This mitigates the issue for multi-year training but introduces noise
   (game_id order doesn't perfectly correlate with chronological order).

3. **Recency weighting**: When dates are identical, all games get weight
   ~0.819 instead of the intended gradient from ~0.52 (early) to 1.0 (late).

**Impact:** Temporal features are corrupted for materialization-based
training. SOTA pipeline partially mitigates but with noise.

---

### C4. HTML Entity Encoding Bug in Tournament Seeds

**Files:** `data/raw/historical/tournament_seeds_*.json`, `data/raw/cache/tournament_seeds_*.json`

The tournament seed scraper (`tournament_bracket.py:62-64`) uses a regex that
captures raw HTML text including unescaped HTML entities. The team name
`Texas A&M` in HTML is `Texas A&amp;M`, which becomes:

```
team_name: "Texas A&amp;M"
team_id: "texas_a_amp_m"
```

The `_normalize_team_id` function preserves the `amp` as a literal substring.
This broken team_id will **never match** the metrics ID `texas_a_m`.

Similarly, any team name containing `&`, `<`, `>`, or other HTML-special
characters will be corrupted. For 2025, `&amp;` appears in `Texas A&M` and
potentially other `&`-containing names.

**Impact:** Texas A&M (a perennial tournament team) may fail to resolve,
dropping it from tournament evaluation sets.

---

### C5. 7 of 68 Tournament Teams Missing from Metrics (2025)

The following 2025 tournament team IDs from seeds data have **no direct
match** in `team_metrics_2025.json`:

| Seed team_id     | Actual metrics ID          | Cause                        |
|------------------|----------------------------|------------------------------|
| `unc`            | `north_carolina`           | Abbreviation vs full name    |
| `byu`            | `brigham_young`            | Abbreviation vs full name    |
| `uconn`          | `connecticut`              | Abbreviation vs full name    |
| `vcu`            | `virginia_commonwealth`    | Abbreviation vs full name    |
| `ole_miss`       | `mississippi`              | Nickname vs official name    |
| `saint_mary_s`   | `saint_mary_s__ca`         | Missing state disambiguator  |
| `texas_a_amp_m`  | `texas_a_m`               | HTML entity encoding bug (C4)|

The `_align_source_team_ids` fuzzy resolver in `materialization.py` must
correctly resolve all 7 or they are dropped from tournament feature tables.
The key containment heuristic (`materialization.py:1382-1387`) would match
`unc` to `unc_asheville` (wrong team!) before `north_carolina`.

**Impact:** Tournament evaluation set may be missing major teams, leading to
biased accuracy metrics. `unc` -> `unc_asheville` is a silent misresolution
that would give North Carolina the metrics of a mid-major.

---

## HIGH Issues (data quality degradation)

### H1. NCAA Suffix Not Stripped in Team Metrics (2005-2021)

**Files:** `data/raw/historical/team_metrics_{2005..2021}.json`

All years 2005-2021 have 65-68 teams with `NCAA` suffix in both `team_name`
and `team_id`:

```
team_name: "AlabamaNCAA"   team_id: "alabamancaa"
team_name: "BucknellNCAA"  team_id: "bucknellncaa"
```

The `_ensure_team_ids` function in `historical_pipeline.py:509-532` strips
this suffix, and `_load_year_samples` in `sota.py:3209-3228` also strips it.
However:

- The **raw data files on disk** still contain the suffix.
- The `materialization.py` pipeline uses `_load_team_metrics` which reads
  the team_id as-is from the JSON. The NCAA suffix is **not stripped** in
  `materialization.py:212`. It relies entirely on `_align_source_team_ids`
  fuzzy matching to resolve `alabamancaa` -> the correct game team ID.
- The NCAA-suffixed teams also have base entries (without suffix), creating
  potential for a single team to appear twice if fuzzy matching resolves the
  NCAA version to a different canonical ID.

**Impact:** Materialization pipeline may fail to join NCAA-suffixed metrics
with game data, or double-count teams.

---

### H2. Tournament Seed Count Anomalies

**Files:** `data/raw/historical/tournament_seeds_{2005..2025}.json`

| Year range | Expected teams | Actual teams | Issue                        |
|-----------|----------------|--------------|------------------------------|
| 2005-2006 | 65             | Missing      | No seed files exist          |
| 2007-2010 | 65             | 65           | Pre-expansion: 64 + 1 play-in; seed 16 has 5 entries |
| 2011      | 68             | **34**       | Half the bracket missing (only East+West) |
| 2012-2019 | 68             | 68           | Correct                      |
| 2020      | N/A            | Missing      | COVID cancellation (correct) |
| 2021-2025 | 68             | 68           | Correct                      |

**2011 is only 34 teams** (East: 18 teams, West: 16 teams, no South or
Midwest). This was the first year of the 68-team field. The scraper regex
likely failed on the 2011 Sports Reference page layout which may differ
from later years.

**Impact:** 2011 tournament evaluation covers only half the bracket. Any
LOYO fold using 2011 as test will have incomplete coverage.

---

### H3. 338 of 700 Game Teams Not in CBBpy Map (2025)

**File:** `data/raw/cbbpy_team_map.csv`

CBBpy game data contains ~700 unique teams (including D2, D3, NAIA
exhibition opponents), but only ~362 are in the CBBpy team map CSV. The
unmapped 338 teams are primarily non-D1 programs, but the resolution
pipeline uses the CBBpy map as a critical lookup step (`sota.py:3286-3298`).

Any D1 team that plays under a slightly different display name than its CSV
entry won't resolve via Pass 2 (CBBpy CSV lookup) and must fall through to
the more error-prone prefix matching or fuzzy matching.

**Impact:** Resolution rate degradation for teams with variant display names.

---

### H4. Inconsistent Game Counts Across Years

| Year | Games | Notable                                     |
|------|-------|---------------------------------------------|
| 2005 | 4,221 | Significantly fewer than modern years        |
| 2012 | 4,891 | Drop vs adjacent years (5,500+)              |
| 2013 | 4,958 | Still low                                    |
| 2021 | 4,282 | COVID-shortened season                       |
| 2025 | 6,291 | Most games (expected for current year)        |

The 2005-2013 counts are 20-40% lower than 2014+. Combined with zeroed
metrics (C1), these years contribute significantly less training signal.
The 2021 COVID season has ~30% fewer games with potentially different
competitive dynamics (bubble environment, opt-outs).

**Impact:** Training data is not uniformly distributed across years. The
exponential decay weighting (`training_year_decay=0.85`) partially
addresses this but doesn't account for within-year data sparsity.

---

## MEDIUM Issues (correctness risk in edge cases)

### M1. `_normalize_team_id` Inconsistency Across Modules

Four different files define `_normalize_team_id` or `_normalize_team_name`:

| File                    | Function                  | Logic                        |
|-------------------------|---------------------------|------------------------------|
| `collector.py:367`      | `_normalize_team_id`      | `lower + alnum -> _`         |
| `providers.py:377`      | `_normalize_team_name`    | `lower + alnum -> _`         |
| `historical_pipeline.py:558` | `_normalize_team_name` | `lower + alnum -> _`         |
| `materialization.py`    | `_normalize_team_id`      | (same pattern)               |
| `tournament_bracket.py:113` | `_normalize_team_id`  | `lower + alnum -> _`         |

While the implementations appear identical (`"".join(ch.lower() if ch.isalnum() else "_" ...)`),
they are **duplicated** rather than shared. Any divergence (e.g., handling
of Unicode characters like `é` in "San José State") could cause silent
mismatches. The `San José State` case is notable: `é` is not ASCII-alphanumeric,
so it becomes `_`, producing `san_jos__state` (double underscore) after
stripping, which differs from `san_jose_state` used in metrics.

**Impact:** Unicode team names may produce inconsistent IDs across modules.

---

### M2. Fuzzy Match Threshold Too Aggressive

The `_resolve_team_id` function in `materialization.py:1398` uses a 0.84
threshold for `SequenceMatcher.ratio()`. This means two strings need only
84% character similarity to match. Example risks:

- `michigan` (0.84) could match `michigan_state` (the `ratio` of
  "michigan" vs "michigan_state" is 0.84)
- `miami` could match `miami__oh` instead of `miami__fl`

The key containment check (`materialization.py:1382-1387`) is even more
aggressive: if `src_key in key OR key in src_key`, it matches. This means
`unc` matches `unc_asheville` (since "unc" is in "uncasheville").

**Impact:** Possible silent misresolution of teams to wrong programs.

---

### M3. Prior-Season Feature Shift May Cause Early-Year NaN Flood

The materialization pipeline (`materialization.py:478`) computes
`prior_season_*` features by shifting team metrics one season forward. For
the first season in the training window (e.g., 2005 if `start_season=2005`),
there is no prior season data, so all `prior_season_*` columns are NaN.

Similarly, `prior_conference_*` features depend on conference membership from
the prior year. Conference realignment (e.g., 2024 Big 12 expansion) means
the conference composition in year N-1 may not reflect year N.

**Impact:** First year in any training window has NaN for all prior-season
features. Conference features are stale during realignment years.

---

## LOW Issues (minor data quality concerns)

### L1. 2020 Season Present in Games But No Tournament Seeds

Season 2020 has 5,766 games (regular season completed) but no tournament
seeds file (correct, since COVID cancelled the tournament). However, the
multi-year training pipeline still loads 2020 games for training, even
though 2020 has unique characteristics (bubble games in 2020-21, season
interruptions) that may not generalize to normal tournament prediction.

The exclusion logic in `sota.py:2063` correctly excludes 2020 from
multi-year training (`yr != 2020`), but the materialization pipeline
does NOT have this exclusion.

---

### L2. Conference Data Missing from All Team Metrics

The `conference` field is absent from all 21 years of team metrics data.
The materialization pipeline attempts to use conference data
(`materialization.py:480-506`) for conference-level prior features, but
since the field is always empty, all conference-related features
(`prior_conference_srs_mean`, `prior_conference_sos_mean`, etc.) will be
NaN or empty for all training samples.

---

### L3. Sports Reference Cache Missing for 2025

The cache directory has `sports_reference_2026.json` but no
`sports_reference_2025.json`. This means 2025 team metrics come from a
different source or were cached under a different name, potentially using
a different schema or field set than other years.

---

## Additional Team Resolution Issues (from deep alias audit)

### T1. TeamNameResolver Bypassed by Core Pipeline

The `TeamNameResolver` class (`team_name_resolver.py`) is a well-designed
alias system covering ~360 D1 programs with multi-pass fuzzy matching.
However, **the core data pipeline does not use it**. The four critical
ingestion modules (`providers.py`, `historical_pipeline.py`, `collector.py`,
`materialization.py`) each implement their own independent `_normalize_team_id`
/ `_normalize_team_name` that does a simple lowercasing + underscore
substitution. This means the curated alias table (which correctly maps
"USC" -> `southern_california`, "UConn" -> `connecticut`, etc.) is entirely
bypassed for game data, team metrics, and tournament seeds.

Only `sota.py` (the live bracket pipeline) and `bracket_ingestion.py`
actually use `TeamNameResolver`. This creates a split: historical training
data goes through dumb normalization, while live inference goes through
smart resolution.

### T2. Ambiguous Alias Collisions

Several abbreviations in the alias table map to multiple canonical teams:

| Abbreviation | Maps to (last wins)    | Also claimed by     | Risk |
|-------------|------------------------|---------------------|------|
| `USD`       | `south_dakota`         | `san_diego`         | Wrong team selected |
| `USF`       | `san_francisco`        | `south_florida` (via "USF Bulls") | Fragile |
| `SHU`       | `sacred_heart`         | `seton_hall` (missing alias) | Seton Hall unresolvable |
| `UNO`       | `omaha`                | `new_orleans` (via "UNO New Orleans") | Fragile |

The `USD` collision is the most dangerous: both San Diego and South Dakota
claim it, and whichever dict entry is processed last wins silently.

### T3. `_team_key` Mascot-Stripping Heuristic Breaks Multi-Word Names

The `_team_key()` method in `materialization.py:1482-1493` strips the last
1-2 tokens from team names to remove mascots. This fails for names where
meaningful parts appear in the last two tokens:

| Input                             | Key produced        | Correct key             |
|-----------------------------------|---------------------|-------------------------|
| "South Dakota State Jackrabbits"  | "south dakota"      | "south dakota state"    |
| "East Tennessee State Buccaneers" | "east tennessee"    | "east tennessee state"  |
| "Florida Gulf Coast Eagles"       | "florida gulf"      | "florida gulf coast"    |
| "Stephen F. Austin Lumberjacks"   | "stephen f"         | "stephen f austin"      |

This is used in the alignment fallback path and would produce incorrect
matches when hit.

### T4. Three Different IDs for the Same Team Across Sources

Example of how a single team gets three different IDs depending on source:

| Team | CBBpy games            | SR metrics               | Tournament seeds |
|------|------------------------|--------------------------|------------------|
| USC  | `usc_trojans`          | `southern_california`    | `usc`           |
| UConn| `uconn_huskies`        | `connecticut`            | `uconn`         |
| BYU  | `byu_cougars`          | `brigham_young`          | `byu`           |
| LSU  | `lsu_tigers`           | `louisiana_state`        | n/a             |
| VCU  | `vcu_rams`             | `virginia_commonwealth`  | `vcu`           |

This three-way mismatch means any join operation requires multi-pass
resolution, and each pairwise join can fail independently.

---

## Recommendations (Priority Order)

1. **Re-scrape 2005-2009 team metrics** with the current pipeline that
   includes the `def_rtg` fallback computation. Alternatively, backfill
   from BartTorvik data which covers these years.

2. **Canonicalize team IDs at ingestion time.** The game data should use
   the same ID namespace as the metrics data. Either strip mascots from
   CBBpy team names at ingestion, or build a definitive ID mapping table
   at the boundary.

3. **Decode HTML entities** in the tournament bracket scraper
   (`tournament_bracket.py:72-73`). Add `html.unescape()` to team_name
   before normalization.

4. **Fix the 2011 tournament seeds** by scraping the full bracket (all 4
   regions) or manually adding the missing South/Midwest teams.

5. **Add real game dates** to historical data. Either use the per-game
   scraping path (slower) or cross-reference with another source that
   provides dates (e.g., Sports Reference game logs).

6. **Add explicit `unc`->`north_carolina`, `byu`->`brigham_young` etc.**
   to the `_LOCATION_TO_METRIC` table in `sota.py:3245` and the
   `_resolve_team_id` priority matching.

7. **Tighten the fuzzy match threshold** from 0.84 to 0.90+ and add
   explicit safeguards against `unc`->`unc_asheville` type mismatches
   (e.g., require the shorter string to be a substantial fraction of the
   longer string's length).

8. **Consolidate `_normalize_team_id`** into a single shared utility
   with explicit Unicode handling (e.g., `unicodedata.normalize("NFKD")`
   to convert `é` -> `e`).

9. **Route all team ID normalization through `TeamNameResolver`** instead
   of the bespoke `_normalize_team_id` in each module. The resolver already
   has the curated alias table that correctly maps "USC" -> `southern_california`,
   "UConn" -> `connecticut`, etc. The core pipeline bypasses it entirely.

10. **Fix the `USD` alias collision** in `team_name_resolver.py`. Only one
    team should have the bare "USD" alias; the other should require a longer
    form. San Diego is the more common basketball "USD".

11. **Replace the `_team_key` mascot-stripping heuristic** in
    `materialization.py` with proper `TeamNameResolver` lookups. The
    current token-stripping approach breaks on multi-word names like
    "South Dakota State" (stripped to "south dakota").
