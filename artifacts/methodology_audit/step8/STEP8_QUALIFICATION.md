# Step 8 — Frozen referee qualification, applied to the corrected referees

Immutable input to Step 9: `qualification_v2.json` (rule, incumbent, coverage,
per-season calibration, gate statistics, statuses, selected independent
referee). Run log: `qualification_v2.txt`. Script: `run_qualification.py`.
The invalidated 2026-09-14 run (`artifacts/referee_audit/qualification.json`)
is preserved unchanged.

## 1. Exact frozen rule (as registered 2026-09-14; code pinned to the document by tests)

- Unit: the season. Evaluation seasons: 2011–2025 excluding 2020 (n = 14);
  2026 excluded as an integration season.
- Metric: mean log loss of the referee's RAW pairwise table over the
  season's real Round-of-64-onward games (63 per season; play-ins excluded),
  before the simulator's logit noise. Brier secondary.
- CI: paired bootstrap over seasons, 5,000 resamples, seed 42, percentile
  2.5/97.5. Same season draws for every referee.
- G1 (beats a coin flip): CI of (log loss − log 2) entirely below 0.
- G2 (not worse than the incumbent `seed`): mean log-loss delta ≤ 0 AND
  mean Brier delta ≤ 0 (point-estimate test).
- QUALIFIED: G1 and G2. DISQUALIFIED: G1 fails, or log-loss delta vs seed has
  a CI entirely above 0. PROVISIONAL: otherwise. seed is QUALIFIED by
  construction. Primary set requires full 14-season coverage.
- Independent referee for C1: the first primary-eligible referee in the
  fixed order market_v2, pit, torvik, blend. LORO over the primary set.

Code and document agree (`test_qualification_constants_match_the_document`).
Nothing above was changed.

## 2. Incumbent construction

`seed` for season Y = seed-vs-seed win rates from Kaggle results for seasons
2010..Y−1 only (`_recent_win_rates(as_of=Y)`; verified the table for Y differs
from the table for Y+1 in every season), cells with < 8 games falling to the
logistic curve. The invalidated run's incumbent contained Y. Consequence of
the correction on the incumbent itself: mean log loss 0.5715 (the old,
in-sample table scored better on the games it had seen).

## 3–4. Referee set and coverage (established before the gate)

| referee | seasons | games/season | fallback / notes |
|---|---|---|---|
| seed | 14/14 | 63 | logistic-curve games: 63 (2011), 30, 28, 23, 15, 14, 10, 15, 6, 13, 9, 12, 5, 0 (2025) |
| torvik | 14/14 | 63 | 0–1 seed-proxy teams |
| blend | 14/14 | 63 | walk-forward both halves |
| pit | 14/14 | 63 | none |
| market (v1) | 14/14 | 63 | defective construction (Step 7); reported, not eligible for use |
| market_v2 | 14/14 | 63 | 0–2 seed-proxy teams/season; 3,284–4,962 fitted games |
| odds_api | 5/14 | 63 | partial coverage → cannot be primary |
| fte | 7/14 | 63 | partial coverage; point-in-time UNVERIFIED → supplementary |

## 5–7. Qualification (frozen rule, corrected tables)

| referee | n | log loss | G1: ll − log2 [CI] | G2: ΔLL vs seed [CI] | G2: ΔBrier vs seed [CI] | status |
|---|---|---|---|---|---|---|
| seed | 14 | 0.5715 | −0.122 [−0.152, −0.095] | 0 | 0 | QUALIFIED (incumbent) |
| torvik | 14 | 0.5426 | −0.151 [−0.185, −0.118] | −0.029 [−0.041, −0.016] | −0.011 [−0.016, −0.006] | QUALIFIED |
| blend | 14 | 0.5520 | −0.141 [−0.169, −0.115] | −0.020 [−0.026, −0.013] | −0.008 [−0.010, −0.005] | QUALIFIED |
| pit | 14 | 0.4680 | −0.225 [−0.255, −0.196] | −0.104 [−0.125, −0.079] | −0.041 [−0.050, −0.031] | QUALIFIED |
| market (v1) | 14 | 0.6002 | −0.093 [−0.127, −0.063] | +0.029 [+0.002, +0.062] | +0.012 [+0.001, +0.026] | DISQUALIFIED |
| market_v2 | 14 | 0.5697 | −0.123 [−0.153, −0.093] | −0.0018 [−0.030, +0.040] | −0.00002 [−0.012, +0.017] | QUALIFIED |
| odds_api | 5 | 0.6114 | −0.082 [−0.099, −0.062] | +0.046 [−0.015, +0.120] | +0.020 [−0.005, +0.051] | PROVISIONAL |
| fte | 7 | 0.5424 | −0.151 [−0.190, −0.112] | −0.028 [−0.045, −0.010] | −0.011 [−0.019, −0.004] | QUALIFIED (partial coverage; not primary; provenance unverified) |

Primary-eligible (QUALIFIED, full coverage): **seed, torvik, blend, pit,
market_v2**. Independent referee by the frozen order: **market_v2**.

Change from the invalidated run: market_v2 was PROVISIONAL against the
in-sample incumbent (ΔLL +0.019) and is QUALIFIED against the honest one
(ΔLL −0.0018, ΔBrier −0.00002). Both of its G2 point estimates are inside
their CIs' overlap with zero; the frozen rule is a point-estimate rule and
it passes it by a hair. That is the result; it is reported as such. The
independent referee therefore moves from pit to market_v2. The market v1
referee remains DISQUALIFIED on the same rule (and independently on
construction).

## 6. CI methodology

Seasons are the resampling unit (n = 14; 7 for fte, 5 for odds_api); one
draw of season indices per resample shared across referees; paired
differences formed within season before resampling; the raw tables are
deterministic so no Monte Carlo noise enters the calibration metric.

## 8. Independence classification of the surviving primary set

| referee | statistical status | information-source relationship |
|---|---|---|
| seed | incumbent | baseline; also the selection referee and opponent generator |
| torvik | QUALIFIED | a candidate-construction base (production-related) |
| blend | QUALIFIED | contains seed (0.5) and Torvik features; a candidate base; the shipped rule |
| pit | QUALIFIED, strongest calibration | same Torvik feature family; correlates 0.87–0.92 with the above; never a base |
| market_v2 | QUALIFIED (marginal) | **materially different source** (betting lines); correlates 0.72–0.80 with the model family |

"Good probability estimator" and "independent validation referee" remain
separate columns: pit is the best estimator and only partially independent;
market_v2 is the only independent source and the weakest qualified estimator.

## 9. Selected independent referee

market_v2, by the registered ordering. It was not chosen for any property of
the strategy; no P(1st), edge, LORO or ranking was computed in this step.

## 10. Unresolved assumptions

- market_v2's qualification is marginal under a point-estimate rule.
- FTE point-in-time provenance unverified (supplementary only).
- odds_api: five seasons, PROVISIONAL.
- The market home-court constant 0.875 logit is an inherited assumption.

## 11. No strategy-result information entered qualification

`run_qualification.py` calls only `calibration_rows`, `qualification_gate`,
`choose_independent_referee`; it imports no selection or scoring code and
computes no P(1st). The gate rule, thresholds, incumbent, referee set and
ordering are byte-identical to the registered ones.

## 12. Step 8: PASS

The frozen rule applies exactly and yields an unambiguous primary set
{seed, torvik, blend, pit, market_v2} with market_v2 as the independent
referee. The marginality of market_v2's pass and the weak independence of the
rest are limitations of the evidence Step 9 can produce, recorded here so
they cannot be discovered after the fact.

**Frozen statement:** these are the referees that qualified under the
pre-registered rule using causally correct referee data. Step 9 will
evaluate the production strategy against exactly these referees, with
market_v2 as the pre-registered independent referee.
