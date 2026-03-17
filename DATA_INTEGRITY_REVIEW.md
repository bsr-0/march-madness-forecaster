# Data Integrity Review — 2026-03-17

## Critical Issues

### 1. Defensive Four Factors ALL ZEROS (torvik_2026.json)
- `opp_effective_fg_pct`, `opp_turnover_rate`, `opp_free_throw_rate` are `0.0` for **all 365 teams**
- Same issue in `torvik_four_factors_2026.json` (367 entries, all zeros for defensive metrics)
- Manifest flags 1,104 validation errors for this
- **Fix**: Run `scripts/repair_2026_data_quality.py` — it computes these from historical game box scores

### 2. All Player RAPM Values Are Null (rosters_2026.json)
- `rapm_offensive` and `rapm_defensive` are `None` for **all 10,021 players**
- Manifest flags 1,454 errors
- **Fix**: `repair_2026_data_quality.py` estimates RAPM from BPM/WARP/usage priors

### 3. All Coach Tournament Team Arrays Empty (coach_tournament_2026.json)
- 436 coaches, **all** with empty `teams` arrays
- Coach-to-team mapping is completely broken for 2026
- **Fix**: `repair_2026_data_quality.py` populates from Barttorvik team-coach mappings

### 4. `tournament_seeds_2026.json` Missing
- Historical seeds exist for 2005–2019 and 2021–2025
- No 2026 file despite `bracket_2026.json` having all 68 teams with seeds
- **Fix**: Generate from bracket data

### 5. `t_rank` All Zeros in advanced_metrics_2026.json
- Field is `0` for all 756 teams

## Moderate Issues

### 6. Placeholder Games in Raw Data
- `data/raw/historical_games_2026.json` contains 13 April 2026 games: `team_name: "TBD"`, scores `0-0`
- These are future tournament stubs that could corrupt aggregations

### 7. Divergent Game File Versions
- `data/raw/historical_games_2026.json`: 13,299 records (with duplicates per team side, through April)
- `data/raw/historical/historical_games_2026.json`: 6,029 de-duplicated records (through March only)
- The raw/ version has ~7,000 duplicate game_id entries

### 8. Torvik Data Missing for 2005–2007
- `torvik_YYYY.json` files only exist 2008–2026
- `historical_games` and `team_metrics` go back to 2005
- `torvik_four_factors` and `torvik_shooting` DO cover 2005–2007 (in `data/raw/`)

### 9. External Ratings Coverage Shift 2025→2026
- 9 sources in 2025 absent from 2026: `7OT`, `ARG`, `BAR`, `JJK`, `KPI`, `LMC`, `REW`, `RT`, `WAB`
- 7 new sources in 2026: `AEI`, `CJB`, `ENG`, `OMN`, `RME`, `WEI`, `WLS`
- Net: 56 sources (2025) → 54 sources (2026)

## Minor Issues

### 10. `tournament_seeds_2020.json` Missing
- Expected — COVID tournament cancellation

### 11. `historical_full/` Only Has 2025
- 4 files, all for the 2025 season only

### 12. Duplicate Files Across raw/ and raw/historical/
- 61 files exist in both directories (all 2025 external ratings are identical copies)

### 13. Partial Conference Champions (conf_champions_2026.json)
- Only 15 of ~32 conferences represented
- Potential naming issues: `long_island_university` (typically `liu`), `mcneese_state` (typically `mcneese_st`)

## What's Working Well

- **No JSON parse errors** across all 1,546 data files
- **No empty container files** — all files have valid content
- **bracket_2026.json** complete: 68 teams, 4 regions, seeds 1–16, no warnings
- **advanced_metrics_2026.json**: 756 teams, no null values (except t_rank)
- **sports_reference_2026.json**: 365 teams, no null values
- **Team metrics**: consistent 2005–2026 (330–365 teams/year)
- **Massey composite**: 2025 (364 teams) and 2026 (365 teams)
- **Historical game counts** are reasonable across all years (4,223–6,291 games/year)

## Recommended Priority Actions

1. Run `scripts/repair_2026_data_quality.py` (fixes #1, #2, #3)
2. Generate `tournament_seeds_2026.json` from bracket data (fixes #4)
3. Remove TBD placeholder games from raw game file (fixes #6)
4. Investigate `t_rank` zero population in advanced metrics (fixes #5)
