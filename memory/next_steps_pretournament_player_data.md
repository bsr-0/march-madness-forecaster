# Next steps: pre-tournament player data (minutes, shooting, clutch)

**Written 2026-08-17. Everything below needs network access — none of it could be
run from the session that wrote it (egress to `barttorvik.com` is blocked, 403 at
the proxy).**

## Why this exists

Three would-be table columns are currently either missing or carrying a
provenance caveat, all for the same reason: the underlying player-level data was
scraped **after** the tournament, so it includes tournament games.

| Column | State today | Blocker |
|---|---|---|
| `three_pt_pct`, `ft_pct` | **not shipped** | `torvik_shooting_{2008-2025}.json` is full-season; only 2026 verified pre-tournament |
| `returning_minutes_pct`, `freshman_minutes_pct` | **shipped, with caveat** | `cbbpy_rosters_*.json` all share one scrape date (2026-02-21); 2010-2025 minutes are full-season |
| blown-lead rate / clutch splits | **not built** | no play-by-play stored |

The fix for the first two is the same: get the same numbers from a
**date-bounded** fetch. The plumbing already exists and is proven — `trank.php`
takes `begin`/`end`, which is exactly why all 16 `torvik_{year}.json` files carry
`cutoff_date == tournament_start - 1 day` while the shooting files don't.

---

## Step 1 — Does `getadvstats.php` honour `begin`/`end`? (30 minutes, decisive)

This is the whole ballgame. Column **[4]** of that CSV is *Minutes percentage*,
and columns 15/19/20 are FT%/3PM/3PA — so one working query fixes minutes **and**
shooting for every historical year at once.

`_fetch_player_csv` in `src/data/scrapers/torvik.py` now **attempts** the dated
URL, sanity-checks the response with `_looks_like_player_csv`, and falls back to
unfiltered if it doesn't work — recording which path won in
`_player_csv_date_filtered[year]`. So the test is just: run it and read the log.

```bash
# Fastest possible answer — compare row counts with and without the window.
curl -s "https://barttorvik.com/getadvstats.php?year=2024&csv=1" | wc -l
curl -s "https://barttorvik.com/getadvstats.php?year=2024&csv=1&begin=20231101&end=20240318" | wc -l
```

**Interpreting it:**
- **Different row counts, or same rows with smaller games-played values** → the
  window works. Go to Step 2.
- **Byte-identical output** → the param is ignored. The endpoint is full-season
  only. Go to Step 3.

Then confirm the code path agrees:

```bash
python3 -c "
import logging; logging.basicConfig(level=logging.INFO)
from src.data.scrapers.torvik import BartTorvikScraper
s = BartTorvikScraper()
s.fetch_shooting_stats(2024)
print('date filtered:', s._player_csv_date_filtered)
"
```

Look for `[torvik] player CSV date-filtered 20231101..20240318 for 2024`.

> The old docstring on `_shooting_from_player_csv` asserted this endpoint does
> **not** support date filtering. The code never actually passed the params, so
> treat that as an untested assumption, not a finding. Step 1 settles it either
> way — and if the assumption turns out to be right, say so and delete this doc's
> Step 2.

## Step 2 — If the window works

1. **Re-scrape shooting for every year**, then confirm the stamp is real. The
   provenance stamp is no longer hard-coded: `fetch_shooting_stats` now writes
   `data_type: pre_tournament` **only** when the dated fetch succeeded, and
   `full_season_unfiltered` otherwise. Any file still saying
   `full_season_unfiltered` did not get a clean fetch — do not ship it.
2. **Add the shooting columns** to `scripts/generate_team_stats_table.py`
   (`three_pt_pct`, `ft_pct`), gated on `data_type == "pre_tournament"` per year
   so a partial re-scrape can't silently mix clean and dirty years.
3. **Add a real minutes source.** Column [4] is a per-player minutes percentage;
   aggregate it per team to replace the roster-derived
   `returning_minutes_pct` / `freshman_minutes_pct` weights, or at minimum to
   cross-check them.
4. Update `docs/data-provenance-and-leakage-audit.md` — its table still lists the
   shooting files as deleted, which is wrong; they are present on disk.

## Step 3 — If the window does NOT work (fallback, certain but slow)

`cbbpy_rosters_{year}.json` is **not a primary source**. It is an aggregate over
per-game box-score rows (`source: cbbpy_schedule_boxscore`, ~117-138k rows per
season), and `src/data/scrapers/cbbpy_rosters.py::_collect_rows_via_schedule`
already walks the season date by date via `_season_dates(year)`.

So: bound that walk at `TOURNAMENT_START_DATES[year]` and re-aggregate. That
yields genuinely pre-tournament minutes for every year with no new data source.

- Cost: a full rate-limited re-scrape, ~6k games/season × 16 seasons. Budget an
  overnight run; do one year first and diff against the current file.
- The season-endpoint fast path (`collection_mode: season_endpoint`) returns the
  whole season at once, so the date filter has to be applied to the **rows**
  after fetch, not to the request. Check whether box rows carry a usable game
  date; if they only carry `game_id`, join through the schedule.

## Step 4 — Optional: blown-lead rate becomes possible

I previously told the user clutch/blown-lead columns were unbuildable "full
stop". That was too strong, and worth correcting here: no play-by-play is
**stored**, but the scraper can **fetch** it. `cbbpy_rosters.py` reads
`CBBPY_ROSTER_ENABLE_PBP` and has a `pbp_rows` collection path that is simply
never switched on (`raw_pbp_rows` is unset in every roster file).

```bash
CBBPY_ROSTER_ENABLE_PBP=1 python3 -c "
from src.data.scrapers.cbbpy_rosters import ...  # see fetch_rosters(year)
"
```

If PBP comes back usable, blown-lead rate, largest-lead-surrendered, and real
late-game splits all become computable. Treat this as a separate project — it is
a much bigger scrape and a new storage schema, not a column addition.

---

## Guardrails

- **Do not** relabel any file `pre_tournament` by hand. The stamp is now derived
  from whether the dated fetch actually succeeded; hand-editing it re-creates
  exactly the bug that produced today's contaminated shooting files.
- **Do not** widen `src/data/normalize.py::_CBBPY_EDGE_CASES` to fix ID joins for
  these tables. That dict feeds three production probability bases (Elo A4,
  roster_adj C2, volatile); changing it can move backtest results without going
  through the acceptance gate. Table-local alias maps live in
  `scripts/generate_team_stats_table.py`.
- **Always** verify a regenerated artifact in a browser, not just in Python.
  Python's `json` reads bare `NaN` back happily; browsers reject the whole file.
  `tests/test_team_stats_table.py::test_artifact_is_strict_json` now guards this,
  but the browser check is what caught it.
- Re-run after any regeneration:
  ```bash
  python3 scripts/generate_team_stats_table.py
  python3 scripts/generate_matchup_table.py   # reads the stats artifact
  python3 -m pytest tests/test_team_stats_table.py -q
  ruff check src/ scripts/generate_team_stats_table.py scripts/generate_matchup_table.py
  ```

## Known-open, unrelated to the above

- **`tournament_context_*.json` has nine transposed games** across eight years
  (a team recorded as losing twice, which single-elimination forbids) — e.g. 2018
  has Cincinnati beating Nevada in the R32 while also showing Nevada in the Sweet
  16; Nevada really won 75-73. The stats table works around it by deriving
  outcomes from round *appearance*, and `generate_team_stats_table.py` prints a
  `SOURCE DATA BUG` line per case. **This same file is the backtest's ground
  truth** (`build_actual_outcome`), so those nine games may be scored wrong
  there too. Worth an audit; not yet investigated.
- 2025 First Four: the source has San Diego St. in both the FF and the R64, so
  SDSU and North Carolina have their `outcome_finish` labels swapped for that
  year. `outcome_rounds_won` is 0 for both either way, so numeric columns are
  unaffected.
