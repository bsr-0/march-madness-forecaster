# 2027 Production Runbook

**Goal:** Generate one bracket for the ESPN pool before lock (~Thursday March 18, 2027).
**Selection Sunday:** March 14, 2027.
**Time budget:** ~24 hours from bracket announcement to pool lock.

---

## Pre-Season Smoke Test (Run in February 2027)

Confirms code still works before you're under time pressure.

```bash
cd ~/march-madness-forecaster
pyenv install 3.11.15  # if needed
pip install -r requirements-lock.txt  # use pinned versions

# Smoke test: generate a bracket using cached 2026 data
python -m scripts.mc_pool_backtest --modes meta_region_poolaware \
    --opponent pool --team-identity --n-opponents 30 \
    --n-repeats 10 --n-model 10 --years 2026

# Expected: completes in ~30s, prints P(1st) ~6% for 2026
# If this fails: fix before Selection Sunday
```

If scrapers break (URLs changed), fix them now — not on game day.

---

## Selection Sunday (March 14, 2027)

### Step 1: Scrape tournament data

```bash
# 1a. Tournament seeds (68→76 teams in 2027 expansion)
# Scrape from ESPN or sports-reference after bracket announcement
python src/data/scrapers/torvik.py --year 2027

# 1b. Torvik ratings (pre-tournament snapshot)
# Verify file exists: data/raw/historical/torvik_2027.json
ls data/raw/historical/torvik_2027.json

# 1c. ESPN national pick percentages (available ~12h after bracket)
python src/data/scrapers/espn_picks.py --year 2027
# Verify: data/raw/historical/espn_picks_2027.json

# 1d. Tournament seeds file
# Verify: data/raw/historical/tournament_seeds_2027.json
# Should have 76 teams (2027 expansion) with duplicate seeds for 12 play-in games
```

### Step 2: Verify data loaded correctly

```bash
python -c "
import sys; sys.path.insert(0, '.')
from scripts.mc_pool_backtest import load_seeds_and_regions
seeds, regions = load_seeds_and_regions(2027)
print(f'Teams loaded: {len(seeds)} (expect 76 for 2027)')
print(f'Regions: {set(regions.values())}')
one_seeds = [t for t,s in seeds.items() if s == 1]
print(f'1-seeds: {one_seeds}')
"
```

Expected: 76 teams, 4 regions, 4 one-seeds.

### Step 3: Generate the bracket

```bash
# Full production run (takes ~5 minutes)
python -m scripts.mc_pool_backtest --modes meta_region_poolaware \
    --opponent pool --team-identity --n-opponents 30 \
    --n-repeats 50 --n-model 50 --years 2027 \
    --save-brackets
```

Output will show:
- `selected=<candidate_name> (best of ~25, P1=X.XXX)`
- Bracket saved to `artifacts/backtest_brackets/backtest_brackets_2027.json`

### Step 4: Extract picks and submit

```bash
# View the bracket picks
python -c "
import json
with open('artifacts/backtest_brackets/backtest_brackets_2027.json') as f:
    data = json.load(f)
modes = data['modes']
for m in modes:
    print(f'Mode: {m[\"mode\"]}')
    print(f'Champion: {m[\"champion\"]}')
    print(f'Final Four: {m.get(\"final_four\", \"see picks\")}')
    print()
"
```

Then manually enter picks into ESPN bracket interface.

---

## Scraper Health (URLs as of May 2026)

| Scraper | Data Source | Expected Response |
|---------|------------|-------------------|
| Torvik ratings | barttorvik.com | JSON with `teams[]` array, each having `team_id`, `barthag`, `conference` |
| ESPN picks | site.api.espn.com | JSON with per-team pick percentages by round |
| Massey composite | masseyratings.com | JSON with `team_id`, `rating`, `normalized` per team |
| Tournament seeds | ESPN/sports-reference | JSON array of 76 team objects with `team_id`, `seed`, `region` |

If any scraper 404s or returns unexpected format, check if the URL structure changed (common for Torvik in offseason).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | `pip install -r requirements-lock.txt` |
| Scraper returns empty/404 | URL changed — check the source website manually, update scraper |
| `len(first_round) != 64` | Seeds file has wrong team count. 76 teams - 12 FF games = 64. Check that FF games exist in results file. |
| P(1st) = 0 for all candidates | Opponent model or scoring broken — run smoke test on 2026 data first |
| ESPN picks unavailable | Pool can still run without — uses seed-based fallback. P(1st) slightly lower but functional. |

---

## Key Numbers

- **Production baseline:** 11.2% P(1st) (3.6x over seed random)
- **Pool size:** ~30 people, winner-take-all
- **Architecture:** ~25 candidate brackets, 200 MC trials each, select highest P(1st)
- **Runtime:** ~5 min for single-year production run
- **Python:** 3.11.15 via pyenv
