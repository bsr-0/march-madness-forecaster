# Referee robustness audit: findings

Run 2026-09-13, 14 evaluation seasons (2011-2025 excluding 2020), commit
recorded in `referee_matrix.json`. Thresholds were fixed in
`PREREGISTRATION.md` before the run; this document interprets the output in
`REPORT.md` and does not move them. Everything below is season-level, n = 14,
paired bootstrap 95% CIs.

## Verdict under the pre-registered rules: INDETERMINATE

| criterion | status | what decided it |
|---|---|---|
| C1 cross-referee edge | INDETERMINATE | production beats `seed` under all five referees, but under `market` the delta is +1.4pp with CI [-0.2, +3.2] and only 6 of 14 seasons won |
| C2 self-referee premium | PASS | premium 2.8pp [0.9, 4.7], 25% of the own-referee figure: real, but under the 33% materiality line |
| C3 leave-one-referee-out | INDETERMINATE | LORO selection retains 58-94% of the in-sample edge for four held-out referees and 20% when `market` is held out |

Parity: the audit's reproduction of production selection matched the
canonical run log in all 14 seasons, so the strategy audited is the strategy
that ships.

## What the matrix says

**The edge is not an artefact of the selection referee.** Selected against
`seed`, production scores P(1st) 0.115 there and 0.105 / 0.102 / 0.086 / 0.082
under torvik / blend / pit / FiveThirtyEight, every one a significant edge over
`seed` (+7.3, +6.1, +4.9, +5.3pp; the smallest lower CI bound is +2.0pp). The
self-referee premium of a quarter is what one expects from a selector that
saw one referee's trials; it is nowhere near the whole edge.

**The one referee that erases most of the edge is the least credible one.**
On the 881 real tournament games, the referees' raw pairwise tables score:

| referee | log loss | Brier | sharpness |
|---|---|---|---|
| pit (shipped browser model) | 0.468 | 0.153 | 0.269 |
| fte (FiveThirtyEight) | 0.542 | 0.183 | 0.229 |
| torvik | 0.543 | 0.183 | 0.223 |
| blend (fitted production model) | 0.543 | 0.183 | 0.199 |
| seed (selection referee) | 0.551 | 0.187 | 0.213 |
| market (Bradley-Terry on pre-15-March lines) | 0.600 | 0.206 | 0.191 |

`market` is worse than the seed table and less sharp than every other
referee. The pre-registration named it as the independent referee because its
inputs are prices rather than ratings; as built (`load_market_ratings`, a
Bradley-Terry fit over the unified regular-season odds file), it is not a good
model of who wins. The other external referee, FiveThirtyEight, is as accurate
as Torvik and keeps the edge at +5.3pp [+2.4, +8.2] over 7 seasons.

This is stated as an observation, not used to rescue the verdict: C1 and C3
were written with `market` as the independent referee and they stand as
INDETERMINATE. A follow-up that wants a decisive answer should pre-register a
calibration gate for referees (for example, log loss no worse than the seed
table on the prior seasons) BEFORE choosing the independent referee, then run
this audit again. Doing that here, after seeing that the gate would remove the
inconvenient referee, would be exactly the researcher degree of freedom the
audit exists to count.

**Under the best-calibrated referee the picture is clear.** `pit` is the model
with a measured held-out log loss; under it production is +4.9pp [+2.0, +7.9]
over `seed`, 11 of 14 seasons, and the LORO selector that never saw `pit`
retains 65% of the in-sample edge (+8.2pp of +12.7pp).

**LORO shows the selector generalises across ratings-based referees and not to
market.** Held out: seed 58% of the self edge retained, torvik 61%, blend 94%,
pit 65%, market 20%. When `market` is held out the LORO choice (P(1st) 0.054)
is essentially the production choice (0.053); the referee-average selector
reaches 0.066 there but only by seeing `market` in its objective, which is
in-sample for that column and is why it is not adopted.

**Reality, for what n = 14 outcomes can say.** Against the real results with
simulated opponent fields, production finishes first in 4.6% of pools against
`seed`'s 2.8% (+1.8pp, CI [-3.4, +8.8], 3 of 14 seasons). The same column shows
plain chalk (`seed_chalk`) at 21.5% and the shipped model's argmax at 26%: the
last 14 tournaments were chalkier than any referee here believes, which is a
known feature of the simulated-referee approach and not something this audit
can settle.

## Findings that do not depend on the referee question

- **Every strategy has a self-referee premium; the fitted models' are huge.**
  `pit_argmax` scores 0.328 under `pit` and 0.051 elsewhere; `market_argmax`
  0.295 under `market` and 0.037 elsewhere; both are flagged material with
  reversals against `seed` under other referees. Any headline measured under
  a strategy's own model should be read with this table next to it.
- **Production's premium (25%) is smaller than that of the fixed torvik rule
  (`fixed_tv_r50`, 49%, material)** and comparable to `seed_chalk` (27%).
  Selecting among bases dilutes the dependence on any single one.
- **`meta_region_4champ` has a negative premium** (-4.5pp): it chooses torvik
  brackets and scores better under referees that share torvik's view than
  under the seed referee it was selected against.
- **The fixed blend rules remain the better mean-rank strategies** under every
  referee (`fixed_blend_r10` 9.0-10.6 vs production 10.5-12.8), consistent
  with the earlier finding that the choosing does not pay on rank. On P(1st)
  production leads them under seed, torvik and blend and is level with them
  under pit and market.
- **Winner's curse on the selection trials is small**: the chosen candidate's
  selection-trial P(1st) exceeds its fresh-trial P(1st) by 0.8pp [0.1, 1.6].
- `fte` could not be built for 2023 (an unmapped team) and is absent before
  2016, so its 7 seasons are a subset; treat its column accordingly.

## What this audit did not do

It did not replace the referee, average referees, or tune anything. The
LORO block is the sensitivity test any such change would need first, and it
says a referee average would help only under referees that were already in
the average.

## Files

- `PREREGISTRATION.md`: criteria, fixed before the run (commit `d7b3d0e`).
- `REPORT.md`: every table, generated by `scripts/referee_robustness_audit.py`.
- `referee_matrix.json`: all numbers, per season and pooled, plus config and commit.
- `seasons/season_{year}.json`: raw per-season records (metrics per referee
  and strategy, selection P(1st) per candidate per referee, choices, parity,
  calibration).
- `candidates/candidates_{year}.json`: the frozen candidate set, pick level.
- `run_20260913_235043.txt`: console log of the run that produced the above.
