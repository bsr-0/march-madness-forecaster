# 2027 Tournament Runbook

Replaces the `RUNBOOK_2027.md` deleted in the 2026-08-18 FINDINGS.md
consolidation. That file is recoverable from git history
(`git show 6393ef0^:RUNBOOK_2027.md`) but describes a pipeline that no
longer exists: a single CLI-selected bracket, manually typed into ESPN's
web form. The product today is a browser-filterable bank of candidate
brackets (`docs/index.html`), built by
`scripts/experiments/build_candidate_artifact.py` +
`scripts/build_ui_payload.py` and deployed by a plain `git push`. This
document describes that pipeline. See
[AUDIT_INDEPENDENT_EVALUATOR_2027.md](AUDIT_INDEPENDENT_EVALUATOR_2027.md)
recommendation 8.

**Key dates for 2027** (`src/data/season_calendar.py`):

| Event | Date |
|---|---|
| Selection Sunday (field announced) | March 14, 2027 |
| First Four / play-in games | March 16–17, 2027 |
| Round of 64 tip-off — **the pool lock, the public-picks cutoff, and the point past which the candidate artifact is immutable** | **March 18, 2027, 12:00 ET** |

**The build cannot start at Selection Sunday.** `build_candidate_artifact.py`
resolves the play-in games and hard-fails with "wait for them" if they
haven't been played (see Step 3 below). The real window to build, review,
and deploy is **Tuesday evening through Thursday morning** — about 36 hours,
not the ~24 hours from announcement the old runbook assumed.

**The field is 76 teams / 12 play-in games in 2027** (NCAA expansion from
68/4). The main draw is still 64 teams, so bracket construction and the
`(1, 63)` game encoding are unaffected; what changes is how play-in slots
resolve to a single Round-of-64 team, which is handled generically today
(`tests/test_field_expansion_2027.py`).

## Pre-season smoke test (run in February 2027)

Confirms the environment and pipeline still work before you're under time
pressure.

```bash
cd ~/march-madness-forecaster
pyenv install 3.11.15   # if needed
pip install -r requirements-lock.txt

# Runs the full test suite
PYTHONPATH=. pytest tests/ -p no:asyncio -o addopts= -q

# Builds a real season end to end with tiny simulation counts (~10s), using
# already-archived data -- exercises every stage of the real pipeline without
# waiting on 150k simulations. --force is safe here: 2025 is not an official
# prospective season.
PYTHONPATH=. python -m scripts.build_season --year 2025 \
  --n-sims 500 --target 50 --trials 50 --force
git checkout -- docs/data/season_2025.json   # discard the tiny-sample rebuild
```

If a scraper 404s or a URL has changed, fix it now, not on game day.

## Selection Sunday, March 14

### Step 1: Get the announced field

There is no live scraper for the just-announced bracket in the ordinary
sense — `src/data/scrapers/bracket_ingestion.py` tries, in order, the
`bigdance` package (Warren Nolan data), a Sports Reference HTML scrape, and
a manual JSON fallback. **Verify which of these actually works before
Selection Sunday**, during the pre-season smoke test above — this is the
step most exposed to an external site changing shape, and it has not been
exercised against a live 76-team field before.

Once you have the field as `data/raw/bracket_2027.json` (schema: see
`data/raw/bracket_2026.json` for the prior year's shape), convert it:

```bash
python -m scripts.generate_tournament_seeds --year 2027
```

This writes `data/raw/tournament_seeds_2027.json`. If you have a
consolidated `tournament_context_2027.json` instead (preferred; carries
seeds + regions + results in one file), that takes priority automatically
— see `load_seeds_and_regions` in `scripts/_common.py`.

### Step 2: Pull ratings and public picks

```bash
# Pre-tournament Torvik ratings (hard requirement -- build_candidate_artifact
# refuses to run without a torvik_2027.json marked data_type=pre_tournament).
python -m scripts.rescrape_pretournament_torvik --year 2027

# ESPN "who picked whom" -- degrades gracefully to a seed-based fallback if
# unavailable, so don't block on this, but do try:
python -m scripts.scrape_historical_espn_picks --years 2027
```

Verify:

```bash
ls data/raw/historical/torvik_2027.json data/raw/tournament_seeds_2027.json
python -c "
from scripts._common import load_seeds_and_regions
seeds, regions = load_seeds_and_regions(2027)
print(f'Teams: {len(seeds)} (expect 76 before play-ins, 64 after)')
print(f'Regions: {sorted(set(regions.values()))}')
print(f'1-seeds: {[t for t, s in seeds.items() if s == 1]}')
"
```

## After the First Four, before Round of 64 (Wed evening – Thu morning, March 17–18)

### Step 3: Build the candidate bank and the UI payload

```bash
PYTHONPATH=. python -m scripts.build_season --year 2027
```

This runs, in order: refresh `team_stats_by_year.json`, build
`artifacts/candidates/candidates_2027.json` (~150k simulations, several
minutes), rebuild every `docs/data/season_*.json` and `seasons.json`. If any
step fails, later steps do not run and no `docs/data/` file changes for that
step's output.

**If it fails with "no play-in results are on disk yet -- wait for them":**
the First Four games haven't been recorded. Re-run
`python -m scripts.rescrape_pretournament_torvik --year 2027` and re-scrape
tournament results, then retry.

**If it refuses to overwrite an existing artifact:** you already built 2027
once. That is deliberate (PROSPECTIVE_2027 CHECKPOINT 2 — regenerating the
official artifact after later information and presenting it as the original
prediction is the one failure mode that voids the whole prospective
exercise) and `--force` will not lift it for an official season. If you
have a genuine reason to rebuild (a data error caught before lock, not new
information), that is `rm artifacts/candidates/candidates_2027.json*` by
hand — a deliberate, visible act, not a flag.

### Step 4: Review before it goes live

The script's own printed validation block (candidate count, EV error,
champion diversity, constraint coverage) catches structural problems, but
does not catch "does the bracket look sane to a person." At minimum:

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000/, select 2027, sanity-check the recommended
# bracket and a few filtered views by eye
```

Check in particular: all four 1-seeds present and correctly regioned, the
"Maximise chance of winning" and "Maximise expected points" cards both
render, and `state.season.strategies[].note` text doesn't reference a stale
year.

### Step 5: Ship it

```bash
git add docs/data/season_2027.json docs/data/seasons.json docs/data/team_stats_by_year.json
git commit -m "Build 2027 bracket bank"
git push
```

A push touching `docs/**` on `main` triggers
`.github/workflows/deploy-docs-on-push.yml` automatically — there is no
separate deploy command. It pushes the whole `docs/` directory to GitHub
Pages with no generation or validation step of its own, so everything above
must be right before this push.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements-lock.txt` |
| A scraper 404s or returns an unexpected shape | Source site changed. Check it manually, update the scraper — this is why the smoke test exists. |
| `build_season.py` fails at "no play-in results ... wait for them" | First Four games haven't been played or recorded yet. Not fixable by re-running; wait. |
| `build_season.py` refuses to overwrite | Working as intended — see Step 3. Not a bug. |
| `len(seeds) != 64` after `resolve_first_four` | A play-in slot didn't resolve. Check the tournament-results file has all 12 First Four games with a clear winner. |
| ESPN picks unavailable | Non-fatal — `build_espn_pick_distribution` degrades to a seed-based public-pick model. The site still works, contrarian signal is weaker. |
| Site shows the previous year's bracket after a push | Check the push actually touched `docs/**` (the workflow's path filter) and that `docs/data/season_2027.json` status is `"ready"`, not `"not_started"`. |

## What this pipeline does not automate, on purpose

Field acquisition (Step 1), the go/no-go review (Step 4), and the final
`git push` (Step 5) all require a human. Selection Sunday and the Round of
64 lock happen once a year with real stakes for anyone using the site; a
fully unattended pipeline that could push a broken or contaminated bracket
to production with no review is a worse failure mode than a 30-minute
manual step. `scripts/build_season.py` exists to remove the *undocumented,
error-prone* parts (which script, in which order, with which flags) — not
the judgment calls.
