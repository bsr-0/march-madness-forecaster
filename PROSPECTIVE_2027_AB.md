# Pre-registration: does per-season selection earn its complexity?

**Spec version** `2027.ab.v1` · **frozen** 2026-09-13 · **hash** `10f1e182dfc24776`
**Frozen protocol** `configs/frozen/prospective_2027_ab.json` ·
**code** `src/governance/ab_2027.py` · **audit** recommendation 16, finding H11

---

## The short version

**This test cannot be won.** The power calculation says 181 seasons — the year
2207 — to resolve the metric the pool actually pays on. That result is the
deliverable, and it arrived before the protocol was written rather than after
the data disappointed, which is the only time a power calculation is worth
anything.

What is registered below is therefore not a test. It is a **ledger with a rule
about what one season is worth**, because whichever arm happens to win 2027 will
be quotable as evidence by anyone who wants it to be, and a rule written down
first is the only defence against that.

## The question

`meta_region_poolaware` generates 20–25 candidate brackets per season and picks
one by simulated P(1st). That machinery is roughly 90% of the backtest's runtime
and most of the system's conceptual weight. Audit finding **H11** asked whether
it beats a fixed rule that does none of it.

| | |
|---|---|
| **Treatment** | `meta_region_poolaware` — per-season candidate generation and selection |
| **Control** | `fixed_blend_r40` — one fixed rule: alpha=0.5 blend at risk 0.40, no selection step |
| **Barred as control** | `fixed_blendA100_r35`, despite topping the P(1st) table |

The barred arm matters. alpha=1.0 is **pure seed probabilities**, and the
referee that draws the "true" tournament in every trial is the same seed model —
so it is graded by its own source. Using the highest-scoring fixed mode would
have measured finding C2's circularity and called it a strategy comparison. The
exclusion is recorded in code (`BARRED_CONTROLS`) with its reason, so it reads
as a decision rather than an oversight someone later "fixes".

## What the historical window says

Measured over the 14 evaluation seasons under the canonical contract, before
this protocol existed (`artifacts/headline_measurement/ab_2027_power.json`):

| metric | treatment | control | paired diff | SD | p | treatment better in | **seasons for 80% power** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p_first` | .1200 | .1100 | +1.00pp | 4.77pp | .447 | 6 of 14 | **181** |
| `mean_rank` | 10.52 | **9.14** | −1.38 positions | 2.12 | **.030** | 3 of 14 | 21 |

**The two metrics disagree in sign, and the disagreeing one is the significant
one.** Both can be true at once: a higher-variance bracket wins outright more
often while finishing worse on average. The production selector's own comment
says exactly this — *"a bracket that beats 70% of opponents but never wins
outright scores high on rank but has low P(1st). Binary is the correct unbiased
estimator of what pays out."*

For a winner-take-all pool `p_first` is the objective that matters, so the
production choice is defensible. But it is worth stating plainly what the table
shows: **the selector is credited with an edge that is not statistically
distinguishable from zero on its own objective (6 of 14 seasons, p=.45), while
it is distinguishably worse on placement (11 of 14, p=.03).**

Both metrics are pre-declared **primary**, precisely because they disagree.
Neither may be dropped later for being inconvenient.

## The protocol

- **First season**: 2027 — the only genuinely untouched season this project has
  (`pool_rdof_audit.SEQUESTERED_YEARS`).
- **Test**: two-sided paired t-test across seasons, α = 0.05, on each primary
  metric. No multiplicity correction between the two, because both are primary
  rather than one being a screen for the other.
- **Scheduled analyses**: at **21** and **181** prospective seasons — the 80%
  power points for `mean_rank` and `p_first`. **No interim analyses. No early
  stopping.** At one observation a year, the temptation to peek has decades to
  work on whoever maintains this, and peeking-until-significant is the same
  forking-paths failure as audit finding H2, just spread over a career.
- **Per-season reporting**: each season's numbers are recorded and may be quoted
  as a running tally. A tally is a description, not a test; no significance
  claim is licensed before a scheduled analysis point. `tally()` deliberately
  returns no p-value.
- **Write-once**: `record_season` refuses to overwrite a recorded season.
  A season whose numbers can be revised is a season whose numbers can be chosen.
- **Immutable protocol**: `freeze()` refuses to overwrite a differing spec.
  Changing the protocol means bumping `SPEC_VERSION`, which invalidates this
  version's claim and starts a new one — permitted, but never silent.

## If it never resolves

Stated in advance, so it is not a conclusion reached by exhaustion: **if
`mean_rank` has not resolved by 21 prospective seasons, the difference is too
small to matter at any horizon this project operates on**, and the complexity
question should be settled on other grounds — maintenance cost, explainability,
or the in-sample evidence that already exists — rather than by continuing to
wait.

## What this costs the 2027 holdout

Nothing. Both brackets are produced by code that already runs; the A/B only
requires that both be built and recorded before the tournament starts. 2027's
evidentiary value for the primary claim in `PROSPECTIVE_2027_v2.md` is
untouched. Spending a genuinely untouched season on a 1pp difference needing two
centuries would have been a poor trade, which is why this rides along rather
than competing.

## Reproduce

```bash
python -m scripts.ab_2027_preregistration --power     # re-derive the effect sizes
python -m scripts.ab_2027_preregistration --verify    # check the protocol has not drifted
python -m scripts.ab_2027_preregistration --tally     # the series so far (not a test)
```

After the 2027 tournament, once both brackets have been scored:

```bash
python -m scripts.ab_2027_preregistration --record 2027 \
  --treatment '{"p_first": ..., "mean_rank": ...}' \
  --control   '{"p_first": ..., "mean_rank": ...}'
```
