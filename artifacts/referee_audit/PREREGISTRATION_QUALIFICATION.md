# Referee qualification audit: pre-registration

Written 2026-09-14, after the first referee robustness audit
(`FINDINGS.md`) and before any calibration number for the new referees
defined below existed. The machine-readable copy is
`src/evaluation/referee_audit.py::QUALIFICATION`, pinned to this document by
`tests/test_referee_audit.py`.

## Why a second audit

The first audit's verdict was INDETERMINATE because one referee, `market`,
erased most of the production edge, and that referee turned out to be the
worst-calibrated of the six on the real games. Whether a referee deserves
to generate the tournaments a strategy is judged on is a question about the
referee's predictive validity, and it has to be settled on that evidence
alone, never on what the referee does to a strategy's P(1st).

## What was already seen, stated plainly

The first audit's report contains pooled (game-weighted) log loss, Brier and
sharpness for seed, torvik, blend, pit, market and fte. Those six numbers
were seen before this document was written. Nothing here uses a tunable
threshold that could have been placed around them: the gate below compares
each referee to a coin flip and to the incumbent seed table, and the
incumbent is the comparator because a referee that cannot out-predict the
table it is meant to check has no standing to overrule it. The consequence
for `market` was foreseeable from those pooled numbers and is not claimed as
a discovery. No calibration number for `market_v2` or `odds_api` has been
computed at the time of writing.

## Construction audit of the market referee (no calibration used)

`load_market_ratings` (the `market` referee, the harness's `odds` base) has
two defects visible from its inputs alone:

1. **Team identity.** The unified odds file carries SBRO-era spellings such
   as `ohiostate` and `ohio_state_buckeyes` that were never normalised to
   canonical ids, so the Bradley-Terry fit never sees them as tournament
   teams. In 2011-2022, 27 to 33 of the 68 tournament teams per season
   receive the crude seed fallback `max(0.10, 1 - 0.04 * seed)`. Half the
   "market" referee was a seed table.
2. **The spread/probability consistency guard.** The fit skips any game with
   |spread| > 5 where the spread's sign disagrees with the implied
   probability, assuming negative means home favourite. SBRO rows follow
   that convention only 55-71% of the time, so the guard discards 1,500 to
   2,500 decisive games per season and fits mostly on close games. The
   repo's own `spread_power` code already notes the sign is unreliable and
   uses implied probability only. (Covers rows, 2023-2025, follow the
   convention and lose nothing.)

The fit also ignores home court.

## The corrected referee, defined before measuring it

`market_v2` (`load_market_ratings_v2`), same data, same cutoff (games before
15 March, so pre-tournament), Bradley-Terry on implied probabilities:

- team ids resolved through the repo's curated `TeamNameResolver` using its
  exact, alias, slug and mascot-strip tiers only (no containment or fuzzy
  matching), plus compact-spelling equality against canonical ids, plus a
  short curated alias list for SBRO's remaining compact spellings
  (`vcu_rams`, `st_johns`, `longisland`, ...). Both sides (odds ids and seed
  ids) are canonicalised through the resolver so `umass`/`massachusetts`
  style splits merge;
- no sign guard: implied probability is the only signal, as in
  `spread_power`;
- home court: for non-neutral games the home implied probability is shifted
  by the repo's existing `spread_power` convention, 3.5 points / 4 = 0.875
  logit units, before the fit;
- tournament teams still absent after resolution keep the seed fallback,
  and the count is recorded per season. The referee is reported with that
  count; it is not silently patched.

`odds_api` (`load_odds_api_market_ratings`, direct multi-book closing
consensus, 2021-2025 only) is included as a second market referee with
partial coverage.

## Gate (fixed now)

Unit: the season. Metric: mean log loss of the referee's RAW pairwise table
over the season's real Round-of-64-onward games (63 per season), before the
simulator's logit noise. Brier is secondary. Paired 5000-resample bootstrap
95% CIs (seed 42) over seasons.

- **G1, beats a coin flip.** Paired log-loss delta versus the constant 0.5
  referee (log 2 = 0.6931 per game): CI entirely below 0.
- **G2, not worse than the incumbent.** Paired delta versus the `seed` table:
  mean log-loss delta <= 0 AND mean Brier delta <= 0.

Status:
- QUALIFIED: G1 and G2 pass.
- DISQUALIFIED: G1 fails, or the log-loss delta versus `seed` has a CI
  entirely above 0 (significantly worse than the incumbent).
- PROVISIONAL: otherwise (point estimate worse than `seed`, CI spans 0).
  Reported in every table, excluded from the primary set.
- `seed` is the incumbent: qualified by construction, flagged as fit
  in-sample (its 2010-2025 window contains every evaluation season), which
  makes G2 conservative for every other referee.
- A referee with partial season coverage is gated on its own seasons and
  can never be in the primary set (C1 and LORO need every season).

## Primary robustness conclusion

Criteria C1, C2, C3 from `PREREGISTRATION.md`, unchanged, applied with:

- criterion referees = the QUALIFIED referees with full 14-season coverage;
- the independent referee for C1 = the first qualified full-coverage
  referee in this fixed order: `market_v2`, `pit`, `torvik`, `blend`
  (`pit` is never a candidate base and was never selected against; `torvik`
  is a candidate base; `blend` shares half its mass with `seed`);
- LORO rotated over the same qualified full-coverage set.

The full matrix over ALL referees, qualified or not, is kept in the report.
The strategy, its candidate set, its selection rule and its opponent model
are untouched; the market referee is not adopted for anything.
