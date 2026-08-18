# Next steps: pre-tournament player data (minutes, shooting, clutch)

**Updated 2026-08-18 after the decisive test came back. Step 1 is ANSWERED — the
Torvik player endpoint is a dead end. The live route is Step 2 (cbbpy box-score
re-aggregation).**

## Why this exists

Three would-be table columns are missing or carry a provenance caveat, all for
the same reason: the player-level data was scraped **after** the tournament, so
it includes tournament games.

| Column | State today | Blocker |
|---|---|---|
| `three_pt_pct`, `ft_pct` | **not shipped** | `torvik_shooting_{2008-2025}.json` is full-season; only 2026 verified pre-tournament |
| `returning_minutes_pct`, `freshman_minutes_pct` | **shipped, with caveat** | `cbbpy_rosters_*.json` all share one scrape date (2026-02-21); 2010-2025 minutes are full-season |
| blown-lead rate / clutch splits | **not built** | no play-by-play stored |

---

## Step 1 — Torvik player CSV date window: TESTED, DOES NOT WORK ❌

The hypothesis was that `getadvstats.php` might honour `begin`/`end` the way its
sibling `trank.php` demonstrably does (which is why all 16 `torvik_{year}.json`
files carry `cutoff_date == tournament_start - 1 day`). Column [4] of the player
CSV is Minutes percentage and 15/19/20 are FT%/3PM/3PA, so one working query
would have fixed minutes *and* shooting for every historical year at once.

**Result (2026-08-18):**

```
getadvstats.php?year=2024&csv=1                             -> md5 ccefee275c62…, 5002 rows
getadvstats.php?year=2024&csv=1&begin=20231101&end=20240318 -> md5 ccefee275c62…, 5002 rows
```

Byte-identical. Max games-played is **41 in both** — a genuine pre-tournament cut
would top out near 34. The params are silently ignored.

**The pre-existing docstring warning on `_shooting_from_player_csv` was right.**
The speculative attempt-with-fallback that was briefly added to
`_fetch_player_csv` has been reverted (it would have burned an extra 45s-timeout
request per year, forever, to always fall back). The finding is now recorded in
that docstring so nobody re-litigates it.

**Also worth knowing:** the verification command originally written here could
never have worked. `_check_tournament_date_guard` **always raises** `LeakageError`
for any live scrape where `today >= TOURNAMENT_START_DATES[year]`, so
`fetch_shooting_stats(2024)` in 2026 is refused by design — that guard is working
correctly and should not be worked around. It also means live scrapes are
pre-tournament *by construction*, which is how the 2026 file earned its
legitimate stamp (fetched 2026-03-16, one day before tip-off). The contaminated
2008-2025 files predate the guard.

Corollary: **Torvik cannot retroactively supply pre-tournament player data for
any past season.** The only Torvik path to a clean season is to scrape it before
that year's tournament starts — i.e. going forward, not backward.

## Step 2 — cbbpy box-score re-aggregation (the route that works)

`cbbpy_rosters_{year}.json` is **not a primary source**. It is an aggregate over
per-game box-score rows (`source: cbbpy_schedule_boxscore`, ~117-138k rows per
season), built by `src/data/scrapers/cbbpy_rosters.py`. The per-game rows are the
thing that makes a date cutoff possible.

1. **Check whether box rows carry a usable game date.** `_collect_rows_via_schedule`
   already walks `_season_dates(year)` day by day, so the schedule path can be
   bounded trivially. The faster `season_endpoint` path (which is what every
   current file used — `collection_mode: season_endpoint`) returns the whole
   season in one call, so there the cutoff has to be applied to the **rows** after
   fetch. If rows only carry `game_id`, join through the schedule to get dates.
2. **Bound at `TOURNAMENT_START_DATES[year]`** and re-aggregate minutes.
3. **Do one year first** and diff against the current file before committing to a
   full run — expect returning/freshman shares to move only slightly, since these
   are composition ratios (see `build_roster_stats` docstring for why the current
   caveat is second-order).
4. Cost: a full rate-limited re-scrape, ~6k games/season × 16 seasons. Budget an
   overnight run.

Same mechanism would give genuinely pre-tournament shooting splits, since box
scores carry FGM/FGA/3PM/3PA/FTM/FTA — which would finally unblock the
`three_pt_pct`/`ft_pct` columns that had to be dropped.

## Step 3 — Optional: blown-lead rate becomes possible

No play-by-play is **stored**, but the scraper can **fetch** it:
`cbbpy_rosters.py` reads `CBBPY_ROSTER_ENABLE_PBP` and has a `pbp_rows`
collection path that is simply never switched on (`raw_pbp_rows` unset in every
roster file). If PBP comes back usable, blown-lead rate, largest-lead-surrendered
and real late-game splits all become computable.

Treat as a separate project — much bigger scrape, new storage schema.

---

## Guardrails

- **Do not** work around `_check_tournament_date_guard`. It is the thing keeping
  live scrapes honest, and it is why 2026 is clean.
- **Do not** relabel any file `pre_tournament` by hand.
- **Do not** widen `src/data/normalize.py::_CBBPY_EDGE_CASES` to fix ID joins for
  these tables. That dict feeds three production probability bases (Elo A4,
  roster_adj C2, volatile); changing it can move backtest results without going
  through the acceptance gate. Table-local alias maps live in
  `scripts/generate_team_stats_table.py`.
- **Always** verify a regenerated artifact in a browser, not just in Python.
  Python's `json` reads bare `NaN` back happily; browsers reject the whole file.
  `tests/test_team_stats_table.py::test_artifact_is_strict_json` guards this now,
  but a browser check is what caught it.
- Re-run after any regeneration:
  ```bash
  python3 scripts/generate_team_stats_table.py
  python3 scripts/generate_matchup_table.py   # reads the stats artifact
  python3 -m pytest tests/test_team_stats_table.py -q
  ruff check src/ scripts/generate_team_stats_table.py scripts/generate_matchup_table.py
  ```

## Known-open, unrelated to the above

- **`tournament_context_*.json` ground-truth defect — INVESTIGATED 2026-08-18.**
  Eight games record the wrong winner across 6 backtest years, each confirmed
  against an independent scrape; the backtest reads these as ground truth, so
  280 points of ESPN scoring sit on the wrong team. Full findings, the verified
  corrections, and why it is not yet repaired:
  `memory/tournament_results_ground_truth_defect.md`. Reproduce with
  `python3 scripts/audit_tournament_results.py`. **Needs a decision** — fixing
  it invalidates every recorded baseline and requires a backtest re-run.
- 2025 First Four: the source names San Diego St. as the R64 loser when UNC
  actually played that game. No backtest impact (only winners are scored, and
  the winner is correct); it swaps SDSU's and UNC's `outcome_finish` labels in
  the stats table, with `outcome_rounds_won` 0 for both either way.
- **Egress from Claude Code web sessions is blocked for all general internet
  hosts** (the proxy refuses CONNECT with 403 — verified against `example.com`,
  not just barttorvik). No scraper in this repo can reach its source from those
  sessions; scraping work has to run locally or the environment's network policy
  has to be widened.
