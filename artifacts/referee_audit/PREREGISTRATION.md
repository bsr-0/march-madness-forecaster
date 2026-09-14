# Referee robustness audit: pre-registration

Written and committed BEFORE any result of this audit existed. The
pass/fail thresholds below are fixed here so that they cannot be chosen
after seeing which way the numbers fall. The machine-readable copy that the
code actually evaluates is `src/evaluation/referee_audit.py::CRITERIA`;
`tests/test_referee_audit.py` pins the two to each other.

## Question

`meta_region_poolaware` is selected by simulating candidate brackets against
a tournament-outcome model (a *referee*) and is measured against the same
referee. Does its published edge over the seed baseline reflect genuinely
better pool decisions, or does it reflect the selector learning the quirks
of one referee?

## What is frozen

Nothing about the production strategy is touched or tuned:

- candidate generation: `src/optimization/poolaware_recipe.py` bases in
  recipe order, `POOLAWARE_RISK_LEVELS` (0.1, 0.3, 0.5, 0.7, 0.9),
  `POOLAWARE_EXHAUSTIVE_RISKS` (0.3, 0.5, 0.7), forced 1-seed champions on
  torvik at risk 0.5, de-duplication keeping the first label;
- selection: argmax binary P(1st) with the `>=` tie-as-win convention,
  strict `>` tie-break on candidate order, 500 selection trials drawn by
  `draw_selection_trials` from `np.random.default_rng(77777 + year)`;
- opponent model: `resolve_opponent_pick_distribution(..., "pool")`, the
  canonical 29-opponent field (real pool size where history exists),
  `chalk_noise_std` as resolved per season (0.0 in every season);
- payout: winner-take-all, one entry; scoring: ESPN 10/20/40/80/160/320,
  team-identity.

The audit reproduces the production candidate set and selection in its own
code path (the harness builds them inline, not as a callable) and CHECKS
PARITY against the canonical run log
(`artifacts/headline_measurement/canonical_2011_2025_n14.txt`): the label the
audit's production-selection step picks must equal the label the harness
logged for that season, for every season. A parity failure invalidates the
audit rather than being worked around.

## Referees

| referee  | outcome model                                              | coverage   | independent of what |
|----------|------------------------------------------------------------|------------|---------------------|
| seed     | empirical seed-vs-seed win rates, 2010-2025 window          | all 14     | none: it is the selection referee, the `seed` candidate source, the opponent-model fallback, and its fit window contains every evaluation season |
| torvik   | Torvik barthag + log5                                       | all 14     | never selected against; but it is the `tv`/`tv_mass80` candidate base, a component of the Massey composite (`mass_avg`), and a feature of both fitted models |
| blend    | 0.5 seed pairwise + 0.5 no-seed LR+GBM pairwise (the fitted production model, `noseed_model`, walk-forward) | all 14 | shares half its mass with `seed`; is the `blend` candidate base; its features are Torvik team stats |
| pit      | the shipped browser model (ridge + Student-t link, walk-forward), `pit_production_model` | all 14 | not a candidate base and never selected against; features include Torvik barthag |
| market   | Bradley-Terry on closing betting lines, games before 15 March | all 14   | the only referee not derived from seeds, Torvik, or any repo model; derived from prices set by bettors |
| fte      | FiveThirtyEight pre-tournament power ratings, normal margin model, sigma = 11 points | 2016-2024 (8) | external and independent; partial coverage, so SUPPLEMENTARY only |
| actual   | the real tournament result (one outcome per season)        | all 14     | reality; n = 14 outcomes, far too few for a decision, reported as the anchor |

"Own referee" pairs, used for the self-referee premium:

| strategy                | own referee | why |
|-------------------------|-------------|-----|
| seed, seed_chalk        | seed        | built from it |
| torvik, torvik_argmax, fixed_tv_r50 | torvik | built from it |
| blend, blend_argmax, fixed_blend_r10/r35/r50 | blend | built from it |
| pit, pit_argmax         | pit         | built from it |
| market, market_argmax   | market      | built from it |
| meta_region_poolaware   | seed        | SELECTED against it |
| meta_region_4champ      | seed        | selected against it |

## Candidate bracket sources

Deterministic: `seed_chalk`, `torvik_argmax`, `blend_argmax`, `pit_argmax`,
`market_argmax`, `meta_region_poolaware` (production), `meta_region_4champ`,
`fixed_tv_r50` (= `meta_region`), `fixed_blend_r10`, `fixed_blend_r35` (the
rule the site ships), `fixed_blend_r50`, and every individual poolaware
candidate (`cand:<label>`).

Stochastic, 50 forward-sampled brackets each, metrics averaged over the 50
exactly as the harness does: `seed` (the headline's baseline), `torvik`,
`blend`, `pit`, `market`.

## Trial design

Per season, `N_EVAL_TRIALS = 300` evaluation trials. Trial t draws ONE
opponent field from `SeedSequence([20260913, year, t, 0])`, shared by every
referee, and ONE outcome per referee from a generator seeded
`SeedSequence([20260913, year, t, 1])`, identical for every referee, so the
referees differ only in their probabilities (common random numbers across
referees and across strategies). Evaluation streams are disjoint from the
selection streams (`77777 + year`), so nothing is scored on the trials it was
selected on.

Metrics per (season, referee, strategy): P(1st) (rank == 1 outright, the
harness convention), mean rank (ties split), top-3 rate, top-10 rate, mean
simulated score. Deltas versus `seed` are season-level paired differences
with a 5000-resample paired bootstrap 95% CI (seed 42). The season is the
unit of independence: n = 14, the canonical evaluation seasons 2011-2025
excluding 2020 (`mc_pool_backtest.EVALUATION_YEARS`; 2026 stays barred).
`fte` covers 8 of them.

## Leave-one-referee-out (LORO)

For each held-out referee H among the five full-coverage referees: the
frozen candidate set is re-selected by argmax of the MEAN selection-trial
P(1st) over the other four referees (500 CRN selection trials each, all
drawn from the production stream `77777 + year`), then the chosen bracket is
evaluated under H on the evaluation trials. Compared against, under the same
H: the production choice (seed-only selection), the self choice (H-only
selection, an in-sample ceiling), and the `seed` baseline.

## Criteria (fixed now)

Criterion referees: the five with full coverage (seed, torvik, blend, pit,
market). `fte` and `actual` are reported but never enter a verdict.

Let D_R = pooled paired delta of P(1st), `meta_region_poolaware` minus
`seed`, under referee R, with its paired-bootstrap 95% CI.

**C1, cross-referee edge.**
PASS if D_R > 0 under every criterion referee AND the CI under `market`
excludes 0.
FAIL if D_market <= 0, or the CI lies entirely below 0 under any criterion
referee.
Otherwise INDETERMINATE.

**C2, self-referee premium.**
premium = P1_seed(poolaware) - mean over the other four criterion referees of
P1_R(poolaware); rel = premium / P1_seed(poolaware).
Material if rel > 0.33 AND the paired CI on premium excludes 0.
PASS if not material. FAIL if material. (Other strategies get the same flag,
for information.) A REVERSAL flag is raised for any strategy whose delta vs
`seed` is positive under its own referee and <= 0 under another criterion
referee.

**C3, leave-one-referee-out.**
For each held-out H: e_loro = P1_H(LORO choice) - P1_H(seed);
e_self = P1_H(self choice) - P1_H(seed).
PASS if e_loro > 0 for every H, and e_loro >= 0.5 * e_self for every H with
e_self > 0.
FAIL if e_loro <= 0 for any H.
Otherwise INDETERMINATE.

**Verdict.** ROBUST iff C1, C2 and C3 all PASS. NOT ROBUST if any of them
FAILS. INDETERMINATE otherwise.

## What the audit will NOT do

It does not change the referee, does not replace it with Torvik, and does
not adopt a referee average. The LORO block is the sensitivity test any such
change would need first; adopting a change on the strength of this audit's
own numbers would be one more researcher degree of freedom spent on the
evaluation seasons.
