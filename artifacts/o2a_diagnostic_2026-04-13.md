# O2a — Four-Factor validation: partial progress; two separate root causes

**Dates:** 2026-04-13 (opened), 2026-04-13 (root-cause #1 fixed, #2 diagnosed)
**Branch:** `claude/simplify-repo-structure-keQXE`
**Evidence files:**
- `artifacts/o2_ff_validation_2026-04-13.json` — machine-readable per-year/per-feature/per-stratum metrics (regenerated after fix)
- This file — narrowing + root-cause analysis

## Status: partial. O2 still cannot close.

Running `scripts/validate_four_factors.py` against the post-O16-fix data
revealed two independent defects. The first is now fixed; the second
remains open and is almost certainly a source-data quality issue rather
than a code bug.

## Root cause #1 — Resolver collision in `compute_boxscore_ff` **[FIXED]**

`scripts/validate_four_factors.py::compute_boxscore_ff()` previously did:

```python
match = _resolver.resolve(tid.replace("_", " "))
canonical = match.canonical_id if match else tid
result[canonical] = ff
```

The `TeamNameResolver`'s fuzzy `containment` method (confidence 0.9)
collapsed unrelated teams to the same canonical id:

- `vermont_catamounts` → `vermont` (via `prefix_strip`, correct)
- `vermont_state_johnson_badgers` → `vermont` (via `containment`, WRONG)
- `vermont_state_lyndon_hornets` → `vermont` (via `containment`, WRONG)
- `tennessee_wesleyan_bulldogs` → `tennessee` (via `containment`, WRONG)

Iteration over `metrics.items()` used unspecified dict order, so
whichever (bogus) team came last won — a 1-game Vermont State Lyndon
record with eFG=0.318 overwrote Vermont Catamounts' real 35-game
eFG=0.540.

**Fix in `scripts/validate_four_factors.py::compute_boxscore_ff`:**

1. Iterate raw team ids in descending game-count order so the team with
   the largest sample wins any collision.
2. Only trust resolver output when `method in ("exact_id", "prefix_strip")`.
   Fall back to the raw tid otherwise.
3. Belt-and-suspenders: first-writer-wins within the game-count-sorted
   iteration so a subsequent smaller-sample team whose canonical
   collides with one already placed is silently dropped.

**Impact:** 4 of 8 features jumped from catastrophic failure to near-passing:

| feature    | pre-fix min r | post-fix min r | pre-fix mean r | post-fix mean r |
|------------|---------------|----------------|----------------|-----------------|
| eFG%       | 0.29          | 0.93           | 0.45           | 0.97            |
| FTR        | 0.29          | 0.89           | 0.63           | 0.96            |
| Opp eFG%   | 0.11          | 0.88           | 0.27           | 0.95            |
| Opp FTR    | 0.30          | 0.91           | 0.54           | 0.97            |

And 4 features moved substantially but still fail the gate:

| feature    | pre-fix min r | post-fix min r | pre-fix mean r | post-fix mean r |
|------------|---------------|----------------|----------------|-----------------|
| TO%        | 0.17          | 0.77           | 0.26           | 0.94            |
| ORB%       | 0.39          | 0.78           | 0.56           | 0.95            |
| Opp TO%    | 0.74          | 0.91           | 0.80           | 0.96            |
| DRB%       | 0.12          | 0.69           | 0.26           | 0.90            |

Gate still requires **r ≥ 0.99 per season per feature**; current pass rate
is **14 / 136 cells**.

## Root cause #2 — Systematic bias on TO%/ORB%/DRB% **[OPEN]**

After the resolver fix, the remaining failures cluster around a
consistent directional bias:

- **TO%** biased `-0.028` across all seasons and strata
- **Opp TO%** biased `-0.030`
- **ORB%** biased `-0.011`
- **DRB%** biased `+0.021`

Other features (eFG%, FTR, Opp eFG%, Opp FTR) have near-zero bias.

### Diagnostic — turnovers are undercounted in source data

Vermont 2024 raw totals from `data/raw/historical/historical_games_2024.json`:

- 35 games, 1,980 FGA, 567 FTA, **301 turnovers** → **8.6 TO/game**

Division I typical is 12–14 TO/game. Vermont's Torvik-reported TO rate
is 0.191 → using the standard Oliver denominator `FGA + 0.44·FTA + TOV`,
that implies `Vermont_TOV ≈ 0.191 × (1980 + 249.5 + TOV)` →
`TOV ≈ 528`. The source data has 301. The gap (~40% under) matches the
magnitude of the observed bias (-0.028 on a ~0.19 metric is a 15–20%
relative undercount, consistent with box-score games missing turnover
fields for a subset of games).

For offensive rebounds, similar: Vermont 2024 has 232 ORB over 35 games
(6.6/game); typical is 8–10/game. Plausibly under-captured.

Conclusion: the remaining gap is a **source-data granularity problem**
in `data/raw/historical/historical_games_*.json`, not a formula bug.
Either:

- (a) A subset of game-level rows is missing `turnovers` / `orb` values
  (schema heterogeneity from the upstream scraper), or
- (b) The upstream provider only reports these fields for conference
  opponents or Division I games (so non-DI opponents' box scores lack
  them).

Validating (a) vs (b) requires a row-level audit of
`historical_games_{year}.json` for null/zero TOV and ORB fields — a
separate investigation with a different skill domain.

### Why a formula change doesn't close the gap

Oliver's alternative possession formula `FGA - ORB + 0.44·FTA + TOV`
subtracts ORB from the denominator. For Vermont: `1980 - 232 + 249.5 +
301 = 2298.5`, giving TO% = `301 / 2298.5 = 0.131` vs Torvik 0.191.
Still a 6pp gap. So subtracting ORB is not sufficient — the underlying
TOV count is short, not just the denominator mis-specified.

DRB%'s positive bias is the mirror effect: if OPP_ORB is undercounted,
then `DRB / (DRB + OPP_ORB)` comes out higher than Torvik's.

## Required to close O2

1. **[DONE]** Fix resolver-collision false-match in `compute_boxscore_ff`.
   Landed in this commit.
2. **[OPEN]** Audit `historical_games_*.json` schema for TOV / ORB /
   DRB null or zero fields; identify which upstream provider(s)
   produce sparse box-score detail; either:
   - Patch the ingestion layer to pull complete box scores (e.g.,
     switch provider for affected games), OR
   - Redefine the local FF as "complete-box-score-only" and filter out
     games without TOV/ORB fields before aggregation (would shrink
     sample per team but remove the bias).
3. Re-run `python scripts/validate_four_factors.py` and confirm
   r ≥ 0.99 per season per feature, `bias_flags == []`.
4. Wrap as `tests/test_validate_four_factors.py` and lock in `MEMORY.md`.

Status: **open**, root-cause-#1 fixed, root-cause-#2 open and blocked on
a source-data audit.

## Also fixed in this commit (not an O2a defect, but in the same file)

`scripts/validate_four_factors.py:421` summary-print had a format-string
bug (`{len(results):>5s}` — `s` format code applied to an int). This
crashed the script at the end of every run so the summary table was
never rendered. Fixed to `{passes:>3d}/{len(results):<3d}`.
