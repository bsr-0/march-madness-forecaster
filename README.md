# March Madness Forecaster

Pool-aware NCAA Tournament bracket optimiser: a per-game win-probability model plus a bracket
construction search that maximises the chance of finishing first in a pool of a given size.

## How it works

1. **Data ingestion** — historical game data, pre-tournament Torvik ratings, ESPN public
   picks, Kaggle Massey Ordinals, pre-cutoff box-score rosters (`march-madness ingest`,
   `scrape-*`, `scripts/build_boxscore_rosters.py`)
2. **Win probabilities** — the site's fitted model (`docs/fit.js`, an 11-feature ridge on
   margin with a Student-t link, fit strictly on earlier seasons; Python mirror in
   `src/prediction/pit_production_model.py`), plus seed, Massey and Torvik pairwise builders
3. **Opponent model** — a pool's pick distribution from its own prior brackets or ESPN
   national pick rates (`src/simulation/pool_history_opponent_model.py`)
4. **Bracket construction** — `meta_region_poolaware`: a pool-aware selection over ~25
   diverse candidate brackets, scored by simulated P(1st) against simulated opponents
   (`src/optimization/`, `scripts/mc_pool_backtest.py`)

**There is no ML pipeline.** One existed until 2026-09-11 (56-dim team vectors, GNN /
transformer / stacking stages, a LOYO harness). Measured honestly for the first time — nine
walk-forward seasons, 567 identical games — it lost to the fitted model in every season and
to the plain seed table in seven of nine (Brier 0.209 vs 0.146 vs 0.189), so it was removed.
The measurement is finding **H10** in `AUDIT_INDEPENDENT_EVALUATOR_2027.md`; the tables are
in `artifacts/headline_measurement/ml_vs_fitted_2016_2025.txt`; the code is in git history
(`git log --diff-filter=D --oneline -- src/pipeline` names the removing commit).

## Usage

```bash
pip install -e .

march-madness --help          # ingest, scrape-*, download-kaggle, optimize-pool
march-madness ingest --year 2027
```

## Pool optimization

The real edge is in pool strategy, not raw prediction accuracy. After generating probabilities, run the pool optimizer to pick contrarian brackets that maximize expected finish.

```bash
# Recommended: auto mode sweeps all probability models and construction strategies,
# deduplicates brackets, and ranks by P(1st)
march-madness optimize-pool --year 2026 --pool-size 30

# Specify your pool's payout structure
march-madness optimize-pool --year 2026 --pool-size 30 --payout winner_take_all
march-madness optimize-pool --year 2026 --pool-size 100 --payout top_3

# Production strategy (meta_region_poolaware): pool-aware selection over ~25 diverse
# candidate brackets. It runs through the backtest script, not the CLI:
# --n-opponents 29 (a 30-person pool) is the default as of 2026-09-10; it is stated
# explicitly here because it is part of the claim, and because it was 999 before then.
python scripts/mc_pool_backtest.py --team-identity --opponent pool \
  --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware

# If you have your pool's prior-year brackets, use them instead of ESPN aggregate
# (calibrates opponent model to your actual pool's tendencies)
march-madness optimize-pool --year 2026 --pool-size 30 \
  --pool-history data/pool_hist_results.json

# Run the full MC pool backtest (15 years, all modes — takes ~30 min)
python scripts/mc_pool_backtest.py
```

### What the backtest number means

`meta_region_poolaware` finishes first in **about 12% of simulated pools (95% CI 9–15%,
n=14 seasons, 2011–2025 excluding 2020)**, against 4.0% for a seed-only bracket in the same
harness. Per-season P(1st) ranges from 2% to 21%. Read the qualifiers before quoting it:

- **Simulated tournaments, not history.** Under the canonical `--team-identity` contract
  each trial draws a tournament from the seed-model referee and scores the model bracket
  and its opponents against *that*, not the realised result. Real outcomes enter only the
  MeanScore column. The number says how often the bracket would win a pool in a plausible
  tournament, not how often it won past pools.
- **Simulated opponents, and the pool size is an assumption.** Opponents are independent
  draws from a pick distribution: the real pool's own picks for 2023–2025 (at that pool's
  real size — 18, 25 and 32 entries), and ESPN national pick rates at an assumed 30-entry
  pool for every earlier season. It assumes winner-take-all with ESPN scoring; it is not a
  universal probability of winning any pool. **P(1st) is mechanically pool-size dependent**
  — the same strategy scores roughly 2.5x worse in a 1000-entry field than a 30-entry one —
  so the pool size is part of the claim, not a detail. Reproduce with
  `--team-identity --opponent pool --n-opponents 29 --n-repeats 100`. That is now the
  default, but state it anyway: before 2026-09-10 the default was 999, so any older run
  or figure measured a 1000-person field for every pre-2023 season.
- **The CI is over seasons.** The season-level standard error is 1.5pp. Do not quote a
  digit after the decimal — the difference between "11.2%" and "12.0%" is half of one
  standard error.
- **2026 is excluded** from the aggregate: it is an in-sample integration season under
  `PROSPECTIVE_2027_v2.md` and cannot be evidence of out-of-sample performance.
- The strategy was selected on this same window, so the figure is in-sample for strategy
  choice. 2027 is the first prospective season.
- **It does not describe any bracket this site currently displays.** The number measures
  `meta_region_poolaware` as the backtest builds it. The candidate bank the site serves is
  built by `scripts/experiments/build_candidate_artifact.py`, which uses different rating
  sources, a different risk grid, and no exhaustive- or forced-champion candidates. Quote
  this figure for the strategy, not for a bracket on the page.
- **Selected and scored against the same referee.** The candidate is chosen by, and later
  measured by, P(1st) under the same simulated-outcome model (`seed_pw`, fit on
  2010–2025 — overlapping every backtested season). Checked 2026-09-12 whether that inflates
  the number: rescoring the already-selected bracket for all 13 evaluation seasons against an
  independent Torvik-barthag/log5 referee it was never selected against gives pooled P(1st)
  9.3% against 11.1% for the production referee, paired difference −1.75pp (95% CI
  [−4.9, +1.3]pp) — **not detectably referee-sensitive at this sample size**, by a rule fixed
  before running it. That bounds the concern; it does not retire it (13 seasons is not a lot,
  and the CI's low end is not small next to the headline). See
  `scripts/independent_referee_check.py` and `AUDIT_INDEPENDENT_EVALUATOR_2027.md` finding C2.
- **Chosen as the best of 79 candidate strategies — and it survives that.** Measured
  2026-09-12: all 79 modes re-run on the 14 seasons, then a Romano–Wolf stepdown on the
  P(1st) deltas of the other 78 against the `seed` baseline, resampling every mode under one
  shared sign vector per draw so their correlation is carried into the null rather than
  assumed away.
  `meta_region_poolaware` scores 12.0% against seed's 4.0% with a **family-wise adjusted
  p of 0.0006**, against a threshold fixed before the run — so the figure is not an artifact
  of picking the best of many. Four un-provenanced knobs were swept at the same time
  (referee noise, `pa_trials`, the risk grid, the candidate base order) and **all four are
  flat**: none moves the headline by as much as one season-level standard error. What this
  does *not* cover: three candidate families were deleted years ago for scoring worse, and
  their code is gone, so no resampling can put them back in the family. Nor does it change
  the underlying ratio — 12 tuning decisions against 14 independent seasons. Reproduce with
  `python -m scripts.pool_rdof_audit --multiplicity --sweep`; full inventory in
  `artifacts/headline_measurement/pool_rdof_audit.txt`; findings H2 and H11.
- **One mode scores higher, and it is an artefact.** The same run put
  `fixed_blendA100_r35` at 12.1%, just above the headline. That mode is built from *pure seed
  probabilities*, and the referee drawing the "true" tournament in every trial is the same
  seed model — so it is graded by its own source. Its alpha sweep gives it away: .088, .105,
  .109, .104, then a 1.8pp jump to .121 at exactly the endpoint that coincides with the
  referee. It is finding C2's circularity, not a better strategy, and its number should not
  be quoted as a competitor. Excluding it, the best fixed rule scores 11.0% against the
  search's 12.0% — so per-season selection does appear to add something, but 1pp sits inside
  the 1.5pp season-level standard error and the search wins only 5 of 14 seasons head to
  head. **Whether the selection machinery earns its complexity is unresolved at n=14.**
  Finding H11; recommendation 16 proposes settling it with a pre-registered 2027 A/B using a
  non-circular fixed arm, rather than a fourth pass over the same seasons.

Source: `artifacts/headline_measurement/canonical_2011_2025_n14.txt`, produced by the
command above on the current code. Full critique in `AUDIT_INDEPENDENT_EVALUATOR_2027.md`;
history and dead ends in `FINDINGS.md`.

### Matching it to your actual pool

The headline above describes one pool: 30 entries, winner-take-all, one bracket. If yours
differs, say so — the answer changes.

```bash
# A pool that pays its top three, not just the winner
python scripts/mc_pool_backtest.py --team-identity --opponent pool \
  --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware --payout top_3

# Your pool's actual split, in dollars or percentages (renormalised for you)
... --payout-shares 50 30 20

# Entering three brackets rather than one
... --n-entries 3
```

**Payout matters, and not only cosmetically.** Selection maximises expected share of the
pot rather than P(1st) whenever the payout is not winner-take-all. Over the 14 seasons that
changes which bracket gets picked in **1 of 14 seasons for a top-3 pool and 11 of 14 for a
pool paying the top quarter** — so for a broad-payout pool, the P(1st)-optimal bracket is
usually the wrong one. It also changes what the strategy is worth: measured against a
random entry its edge is **3.7× in winner-take-all and 2.1× at top-25%**. This strategy is
worth most in a winner-take-all pool.

**Multi-entry saturates fast.** Entering more brackets raises P(winning the pool) —
12.0% → 16.2% → 21.2% → 23.1% for one through four entries — but the 2nd and 3rd entries
are the ones that pay (+4.2pp, p=.018; +5.0pp, p=.010), while the 4th is indistinguishable
from zero (+1.9pp, p=.45). Expected prize *per entry* falls throughout (.124 → .100), so
extra brackets buy a smaller return on each entry fee. The k brackets are chosen jointly
and before any result is known, and every one is scored — this is not "submit many, count
the best". They also take k of the pool's seats rather than growing the field.

**Pool size still does not change the bracket.** Construction applies its duplicate
discount only above 50 entries, so at any realistic size the same bracket is built.
`--pool-factor-mode continuous` lifts that gate, and was measured: +1.93pp mean, but
winning only 6 of 14 seasons with nearly half the gain coming from 2011 alone. It fails
the second half of a rule fixed before the run, so it is available and not the default.
Details in `AUDIT_INDEPENDENT_EVALUATOR_2027.md` recommendation 13.

### The real-outcome record

Co-primary with the simulated figure above, not a footnote to it: for 2023–2026 — the only
seasons with a recorded real pool — the bracket `meta_region_poolaware` actually selected can
be scored against the real tournament result and ranked against the real pool's real scores.

| Year | Real score | Rank | Pool size | Champion picked |
|---|---:|---:|---:|---|
| 2023 | 460 | 18th | 18 | Purdue (lost in the R64 as a 1-seed) |
| 2024 | 1120 | 4th | 25 | Purdue |
| 2025 | 1330 | 10th | 32 | Houston |
| 2026 | 840 | 12th | 30 | Florida |

**0 of 4 finished 1st; 0 of 4 finished top 3.** n=4 is not a rate — one different outcome
moves this by 25 points — and it neither confirms nor refutes the simulated figure above; the
two are separate evidence, not the same claim checked twice. It beats the same strategy's own
mean-of-50 `seed` baseline scored the same honest way in every year (reproduce with
`python -m scripts.real_pool_placement`; full table in
`artifacts/real_pool_placement/placement_2023_2026.txt`), but it has not yet actually won a
real pool. 2027 will add a fifth point; see `PROSPECTIVE_2027_v2.md` for what a fifth point
can and cannot settle at this n.

### Why the simulated number moved, and why it is not a cherry-pick

It was published as "11.2%", then "about 11%", and is measured here at 12.0%. The figure did
not improve because anything was tuned: `b73d351` (2026-09-06, *"every season shipped a wrong
R64"*) fixed the play-in resolution, which changed the field in **every** season. Every P(1st)
published before that date — 11.2%, 11.33%, 10.47%, 11.87% — was computed on brackets
containing teams that never played the Round of 64. Those figures are void rather than
superseded, and the spread among them is itself the argument for quoting a CI instead of a
digit: they all sit inside a single standard error of each other.

## Maintenance

### Annual data refresh

```bash
# Scrape new season data
march-madness ingest-historical --start-season 2025 --end-season 2026
march-madness scrape-tournament-results
march-madness ingest --year 2026

# Rebuild leakage-safe feature tables after new data arrives
march-madness materialize-features
```

### Backtesting & validation

```bash
# Pool strategy backtest, canonical contract (see "What the backtest number means")
python scripts/mc_pool_backtest.py --team-identity --opponent pool \
  --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware

# Real-pool placement of the selected bracket, 2023-2026
python -m scripts.real_pool_placement

# Researcher-degrees-of-freedom audit over the pool-strategy search.
# --registry-only is seconds; the two measurements are slow (~85 min and ~15 min).
python -m scripts.pool_rdof_audit --registry-only
python -m scripts.pool_rdof_audit --multiplicity --sweep --workers 8

# Point-in-time boundary audit and prediction invariants (both run in CI)
python3 scripts/audit_snapshot_boundary.py
python3 scripts/assert_prediction_invariants.py

# The browser model's own checks
node tests/test_calibration.js
```

### The 2027 pre-registration

`configs/frozen/prospective_2027_v2_scoped.json` is the frozen methodology spec for the first
prospective season (`src/governance/frozen_spec.py`, `PROSPECTIVE_2027_v2.md`);
`scripts/experiments/integration_test_2026.py` is the end-to-end pass CI runs against it.

## Development

```bash
pytest           # run tests
ruff check src/  # lint
```

See "What the backtest number means" above for the current pool strategy baseline, `AUDIT_INDEPENDENT_EVALUATOR_2027.md` for the independent review, and `FINDINGS.md` for the project's dead-end ledger and architectural history.
