# Referee qualification audit: findings

Run 2026-09-14 (commit `c408837`), 14 evaluation seasons. The gate and the
corrected market referee were fixed in `PREREGISTRATION_QUALIFICATION.md`
(commit `f3426a8`) before either was measured. Tables: `REPORT_QUALIFIED.md`;
numbers: `qualification.json`, `referee_matrix_qualified.json`.

## Phase 1: which referees qualify (calibration only)

Season-level, paired against the incumbent seed table, no strategy P(1st)
involved.

| referee | log loss | Brier | G2 vs seed, log loss [95% CI] | status | in primary set |
|---|---|---|---|---|---|
| pit (shipped browser model) | 0.468 | 0.153 | -0.083 [-0.110, -0.053] | QUALIFIED | yes |
| torvik | 0.542 | 0.183 | -0.008 [-0.021, +0.007] | QUALIFIED | yes |
| blend (fitted production model) | 0.543 | 0.183 | -0.008 [-0.016, -0.000] | QUALIFIED | yes |
| seed (incumbent, fit in-sample) | 0.551 | 0.187 | 0 by definition | QUALIFIED | yes |
| fte (FiveThirtyEight, 7 seasons) | 0.542 | 0.183 | -0.006 [-0.020, +0.009] | QUALIFIED | no (partial coverage) |
| market_v2 (corrected) | 0.570 | 0.194 | +0.019 [-0.011, +0.061] | PROVISIONAL | no |
| odds_api (closing consensus, 5 seasons) | 0.610 | 0.210 | +0.058 [-0.004, +0.130] | PROVISIONAL | no (partial) |
| market (as built) | 0.600 | 0.206 | +0.050 [+0.022, +0.084] | DISQUALIFIED | no |

Every referee beats a coin flip (G1). The original market referee is
significantly worse than the seed table and is disqualified. Fixing its two
construction defects (team-id resolution: 48 fallback team-seasons over 14
seasons instead of about 420; no sign guard; home court removed) improves
it by 0.03 in log loss, but the corrected version is still not better than
the seed table on the point estimate, so under the gate it is PROVISIONAL
and stays out of the primary set. The independent referee for C1 is
therefore `pit`, the first eligible entry in the pre-registered order.

## Phase 2: the primary robustness conclusion over the qualified set

Criterion referees: seed, torvik, blend, pit. Independent referee: pit.

**Verdict: ROBUST. C1 PASS, C2 PASS, C3 PASS.**

| referee | P(1st) seed | P(1st) production | delta [95% CI] | seasons won |
|---|---|---|---|---|
| seed | 0.041 | 0.115 | +0.074 [+0.054, +0.093] | 14/14 |
| torvik | 0.032 | 0.105 | +0.073 [+0.042, +0.108] | 13/14 |
| blend | 0.040 | 0.102 | +0.061 [+0.043, +0.080] | 14/14 |
| pit | 0.037 | 0.086 | +0.049 [+0.020, +0.079] | 11/14 |

- C1: positive under all four, CI clear of zero under the independent
  referee.
- C2: self-referee premium 1.7pp [-0.3, +3.8], 15% of the own-referee
  figure; not material, CI spans zero.
- C3: LORO retains 60% (seed held out), 79% (torvik), 101% (blend) and 74%
  (pit) of the in-sample self-selected edge, all above the 50% floor, and
  every LORO edge over seed is positive with CI clear of zero. When `pit`
  is held out the LORO choice scores 0.130 under pit against production's
  0.086: a selector that had seen torvik and blend as well as seed would
  have done better under the best-calibrated referee than production did.

## The market referees, reported for transparency

Under the corrected `market_v2`, production's edge is +2.9pp [+1.2, +4.6],
11 of 14 seasons: smaller than under the qualified referees, but real,
where the defective original showed +1.4pp with a CI spanning zero. Under
`odds_api` (2021-2025) it is +4.6pp [+3.2, +6.2], 5 of 5. The picture from
the first audit, that the market referee erased the edge, was largely the
picture of a referee half made of seed fallbacks and fit on a sign-filtered
subsample.

Both market referees remain the least sharp tables here (mean |p - 0.5| of
0.19-0.20 against 0.21-0.27 for the others). A Bradley-Terry fit to
regular-season lines, anchored at the median and read out through log5, is
a rating system derived from prices, not the market's own probability for a
tournament game. That is a limitation of the construction, and a better
market referee would be the next referee to pre-register and qualify.

## What this settles and what it does not

Settled, under the pre-registered rules: the production edge over the seed
baseline is not specific to the referee it was selected against. It holds
under every referee that demonstrates predictive validity at least equal to
the incumbent's, including the one with the best held-out accuracy in the
repo, and selection generalises across those referees.

Not settled: what a well-constructed market referee would say. The only
tools for that here score worse than the seed table and are provisional;
their columns lean the same way as the qualified ones, less strongly.

Nothing about `meta_region_poolaware`, its candidate set, its selection
rule or its opponent model was changed or tuned, and no referee was
replaced or averaged. `load_market_ratings` is untouched because it is the
harness's `odds` base; `load_market_ratings_v2` exists only as a referee.
