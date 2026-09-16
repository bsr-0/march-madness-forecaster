# Step 9 — Referee matrix and leave-one-referee-out, on the frozen qualified set

Inputs (frozen, see `run_manifest.json`): Step 8 artifact
`qualification_v2.json` (sha256 6daebdd2…), referee set
{seed, torvik, blend, pit, market_v2}, independent referee market_v2,
current candidate artifacts (hashes in the manifest), canonical production
selection log = the Step 5 rerun, criteria C1–C3 as registered in
`PREREGISTRATION.md` (unchanged), 300 evaluation trials, eval seed
20260913, selection seed 77777+year, 5,000-resample paired season bootstrap
(seed 42). Outputs: `referee_matrix_qualified.json`, `REPORT_QUALIFIED.md`,
`seasons_qualified/`, `candidates/`, `run_qualification_*.txt`,
`mc_error_check.txt`. The invalidated 2026-09-14 matrix in
`artifacts/referee_audit/` is untouched.

## 1. Reproducibility and integrity checks

- The in-run qualification gate reproduces the frozen Step 8 statuses
  exactly; criterion set and independent referee identical.
- Selection parity: the audit's reproduced production choice equals the
  canonical log's choice in all 14 seasons (`parity_all_ok = true`).
- Every selected production bracket is legal (32/16/8/4/2/1 nested winner
  sets on the real tree; play-ins resolved; 64 teams).
- Fixed-bracket evaluation vs reselection are kept apart: the matrix (§2)
  evaluates ONE production bracket per season (selected on the seed referee,
  the production rule) under every referee; LORO (§4) is the only place the
  choice is redone, and only on the held-out referee's complement.
- Winner's-curse check: selection-trial P(1st) − evaluation P(1st) of the
  chosen bracket under seed = −0.2 pp [−1.3, +0.9]; selection and evaluation
  draws are independent as designed.

## 2. Matrix: fixed production bracket vs the seed strategy, per referee

| referee | mean P(1st) production | mean P(1st) seed strategy | Δ pp | 95% CI (paired season bootstrap) | seasons + / − / 0 |
|---|---|---|---|---|---|
| seed | 0.1171 | 0.0459 | **+7.12** | [+5.38, +9.25] | 14 / 0 / 0 |
| torvik | 0.1086 | 0.0335 | +7.51 | [+4.31, +11.01] | 13 / 1 / 0 |
| blend | 0.1051 | 0.0430 | +6.20 | [+4.73, +7.83] | 14 / 0 / 0 |
| pit | 0.0919 | 0.0357 | +5.61 | [+2.78, +8.60] | 12 / 2 / 0 |
| **market_v2 (independent)** | 0.0781 | 0.0380 | **+4.01** | **[+2.46, +5.58]** | 13 / 1 / 0 |

Per-season Δ under market_v2: 2011 +2.1, 2012 +8.5, 2013 +2.3, 2014 +2.3,
2015 +4.6, 2016 +0.4, 2017 +3.6, 2018 +7.7, 2019 +6.9, 2021 +2.5, 2022 +3.5,
2023 −2.1, 2024 +7.0, 2025 +6.9. The independent-source column is the
smallest edge of the five and about 56% of the self-referee edge.

## 3. Self-referee premium (C2)

premium = P1_seed(production) − mean over {torvik, blend, pit, market_v2}
= 0.1171 − 0.0959 = **+2.12 pp [−0.38, +5.16]**, relative 0.18 (< 0.33) and
CI spans zero → not material. The premium against market_v2 alone is
+3.9 pp; the registered criterion averages the four.

## 4. Leave-one-referee-out (C3)

Selection redone on the mean selection-P(1st) of the four remaining
referees; held-out referee enters neither selection nor ranking (it does
still build candidates if it is torvik or blend, and it is the opponent
fallback if it is seed — Step 7 §18).

| held-out | in-sample set | e_loro (Δ vs seed under H) | 95% CI | e_self | retention e_loro/e_self | choice agreement with production |
|---|---|---|---|---|---|---|
| seed | torvik, blend, pit, market_v2 | +4.13 | [+2.46, +5.89] | +7.12 | 0.58 | 0.43 |
| torvik | seed, blend, pit, market_v2 | +11.63 | [+8.52, +14.77] | +13.14 | 0.88 | 0.43 |
| blend | seed, torvik, pit, market_v2 | +5.91 | [+4.15, +7.92] | +6.78 | 0.87 | 0.36 |
| pit | seed, torvik, blend, market_v2 | +9.99 | [+5.28, +15.38] | +12.27 | 0.81 | 0.50 |
| **market_v2** | seed, torvik, blend, pit | **+6.65** | **[+4.88, +8.61]** | +7.79 | 0.85 | 0.43 |

Every held-out edge is positive with a CI above zero; every retention ≥ 0.5.
With market_v2 held out, a bracket selected without any market information
scores +6.65 pp under market_v2 — higher than the production (seed-selected)
bracket's +4.01 under the same referee.

## 5. Monte Carlo error (`mc_error_check.txt`)

Re-evaluating 2013, 2019 and 2025 with evaluation seed 777 moves per-season
deltas by 1.4 pp on average (max 3.0 pp), consistent with the per-season MC
SE of a delta ≈ 2.4 pp at 300 trials. The season bootstrap resamples
season values that already contain this noise, so its CI is not artificially
narrow; MC noise contributes roughly 0.4–0.8 pp to the 14-season means,
which is a material fraction of the market_v2 CI half-width (1.6 pp) but
does not cross zero for any criterion referee.

## 6. Old → why invalidated → new

| quantity | invalidated run (2026-09-14) | new (Step 9) |
|---|---|---|
| Δ vs seed under seed / torvik / blend / pit (pp) | +7.4 / +7.3 / +6.1 / +4.9 | +7.1 / +7.5 / +6.2 / +5.6 |
| independent referee | pit (market_v2 PROVISIONAL) | **market_v2** (qualified against the honest incumbent) |
| Δ under market_v2 (pp) | +2.9 [+1.2, +4.6] | **+4.0 [+2.5, +5.6]** |
| self-referee premium | +1.7 pp, CI spans 0 | +2.1 pp [−0.4, +5.2], not material |
| LORO retention | 60–101% | 58–88%, all positive |
| verdict | ROBUST over {seed, torvik, blend, pit} | ROBUST over {seed, torvik, blend, pit, market_v2} |

Why the numbers moved: the seed referee and blend's seed half no longer
contain the evaluated season (Step 2); the noseed half of blend is
walk-forward (Step 3); construction and scoring now walk the same Final Four
pairing, so five seasons no longer score a champion construction never
chose (Step 3); P(1st) is the split-tie first-place share everywhere
(Step 4); 2012's opponents no longer come from 2023–26 pool data (Step 4);
the inert forced-champion family is gone (Step 5, no numeric effect). The
similarity of the four model-family columns to the old run is not evidence
that the old run was fine — the corrections moved individual seasons by up
to 5 pp in opposite directions and happened to nearly cancel in the mean.

## 7. Registered verdict

C1 PASS (Δ > 0 under every criterion referee; market_v2 CI excludes 0).
C2 PASS (premium not material). C3 PASS (every e_loro > 0 and ≥ 0.5·e_self).
**ROBUST** under the frozen criteria.

## 8. What this establishes, and what it does not

Establishes: the production strategy's measured P(1st) advantage over the
seed strategy is robust across the pre-qualified referee set under the
registered evaluation procedure, including under the one referee built from
a materially different information source (betting lines), and it survives
selecting without each referee in turn.

Does NOT establish: real-world pool performance; future-season
performance; that any probability model is well calibrated (market_v2
qualified by a hair on a point-estimate rule and is the weakest qualified
estimator); that the referee family is independent (seed, torvik, blend and
pit are one data family; only market_v2 is a different source, so this is
one independent confirmation, not five); or that a richer candidate space
(Step 5 R-2) or a different opponent sampler (Step 6 R-3) would not change
the strategy or its edge.

Plainly: under market_v2 the edge is +4.0 pp, about half of the +7.1 pp the
strategy shows under its own referee, positive in 13 of 14 seasons and
negative in 2023. It is smaller than under pit (+5.6) and torvik (+7.5). It
did not collapse under LORO.

## 9. Stop

No parameter, candidate rule, opponent model, referee, scoring or objective
was changed after seeing these results. Follow-ups recorded elsewhere remain
follow-ups (R-1 heuristic marginals, R-2 candidate space, R-3 opponent
sampler). README's referee paragraph must be rewritten to these numbers
before any retirement claim is made; that is a documentation task for the
final freeze, not a Step 9 action.
