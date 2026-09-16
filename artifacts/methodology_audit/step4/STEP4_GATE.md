# Step 4 — Scoring and objective mathematics: gate table and final report

Findings recorded before fixing: `STEP4_FINDINGS.md`. Evidence:
`crn_experiment.{py,txt}`, `path3_backtest_unified_p1.txt`, tests listed below.

## What was changed

- `src/optimization/payout.py`: `first_place_share`, `first_place_shares`,
  `first_place_share_from_counts` — the one definition of P(1st) (expected
  first-place share, ties split); `probability_any_entry_wins` now sums entries'
  shares. `TIE_WIN` retained only for reproducing pre-audit numbers.
- Every P(1st) site uses it: `score_candidate_p1`, `p_first_from_scores`
  (the production selector), `pool_p_first` (artifact), `compute_bracket_win_probability`
  (CLI), `recency_hparam_fitter`, `referee_audit.evaluate_season` (shares
  recorded from scores), backtest reporting (`all_first_share` array; `p_first`
  is its mean, portfolio = sum over entries).
- `scripts/mc_pool_backtest.py`: team-identity scoring is the default
  (`--shape-encoded` opts into the legacy slot-match scorer); evaluation
  opponents carry the same `chalk_noise_std` as selection; a season without an
  ESPN archive uses static seed pick rates (no future-fitted behavioural model);
  failed candidate constructions are logged and counted in the `selected=`
  line. Same logging in `referee_audit.try_add`.
- `scripts/experiments/build_candidate_artifact.py`: 30-ENTRY pool
  (29 opponents); `_encode_rows` is the strict `winner_sets_to_bool_vector`;
  `p1_assumption` text states the definition.
- `docs/app.js:solveFromPicks` throws on a picks list that names neither or
  both teams of a game (asset token restamped).
- `configs/frozen/prospective_2027_v2_scoped.json`: NOT edited. It is hash-sealed
  and is the record of the v1 freeze; its `p1_definition` ("score >= max opponent")
  now describes a superseded quantity. The v1 prospective freeze is invalidated by
  Steps 2–4 and must be re-issued at Step 18 (final prospective freeze).
- Tests (fail on the old code where applicable): `tests/test_scoring_independent.py`
  (11), `tests/test_objective_exhaustive.py` (4, incl. exhaustive toy world
  and objective identity), `tests/test_tie_conventions.py` (rewritten, 4),
  `tests/test_backtest_scoring_and_opponent_defaults.py` (4),
  `tests/test_actual_outcome_consistency.py` (15 seasons), three strict-picks
  checks in `tests/test_picks_export.js`.
- Artifacts rebuilt after verification: all 14 `candidates_*.json` (+ pins),
  all `docs/data/season_*.json`; README headline amended.

## Gate table

| Area | Path | Result | Evidence | Severity | Fix | Historical impact |
|---|---|---|---|---|---|---|
| Canonical scoring definition (item 1) | all | PASS | one ESPN table; JS has none; `flat` opt-in only | — | — | — |
| Scorer vs independent implementation (2) | 2, 3 | PASS | `test_scoring_independent.py`, exact on 300 random + edge cases | — | — | — |
| Path legality (3) | 2, 3 | PASS | structural in the 63-bool encoding; impossible-pick test | — | — | — |
| Actual results vs topology (4) | 3 | PASS | `test_actual_outcome_consistency.py`, 15/15 seasons exact | — | — | — |
| P(1st) definition (5) | all | FAIL → PASS | F4-1: three quantities under one name | FOUNDATIONAL (objective identity) | one definition everywhere | every P(1st) figure to date; headline rerun 10.9 → **11.6% vs 4.7%** |
| Ties (6) | all | FAIL → PASS | measured 7.4% of "wins" were ties | (same) | split-share | (same) |
| Objective functions `ev`, `p1` (7, 13) | 2 | PASS | `test_objective_exhaustive.py::test_objectives_are_not_proxies_for_each_other` | — | — | — |
| Expected value (8) | 2 | PASS | exact vs mean realised score, 1e-9 | — | — | — |
| P(1st) exhaustive + convergence (9, 18) | 2, 3 | PASS | 16-outcome world, exact to 1e-12; MC within 0.03 | — | — | — |
| Opponent modelling (10) | 3 | FAIL → PASS | F4-5 chalk-noise mismatch; F4-6 2012 opponents fitted on 2023–26 pool | MATERIAL-LOCAL (leakage, 1 season) | same model at selection and evaluation; static seed rates for no-archive seasons | 2012 row; aggregate within CI |
| CRN (11) | 2, 3 | PASS | `crn_experiment.txt`: Δmean −0.0004 (SE 0.0019), variance ratio 1.18× | — | — | — |
| Estimator hygiene (12) | 3 | MINOR → fixed | F4-9 silent construction failures now logged/counted | MINOR | done | none (no failures occur) |
| Shape-encoded default scorer | 3 | FAIL → PASS | F4-2 | MATERIAL-LOCAL | team-identity default | any undocumented run without `--team-identity` |
| Lenient set projections (artifact, UI) | 2 | MINOR → fixed | F4-7 | MINOR | strict | none today |
| Artifact pool size | 2 | MINOR → fixed | F4-8: 31 entries vs spec 30 | MINOR | 29 opponents | all artifact `p1` values (≈ +3%) |
| Baseline apples-to-apples (14) | 3 | PASS | same loops, topology, trials, scorer, tie rule | — | — | — |
| Retrospective chronology (15) | 3 | PASS after F4-6 | seed table/noseed/Torvik/ESPN all pre-tip (Steps 2–3) | — | — | — |
| Headline denominators (16) | 3 | PASS (documented) | per-season mean share over brackets × 100 repeats; unweighted mean over 14 seasons | — | — | — |
| Score variance (17) | 2, 3 | PASS (documented) | opponent + tournament redrawn per trial; candidate fixed | — | — | — |

## Headline rerun (invalidated by F4-1/2/6 and rerun)

`python -m scripts.mc_pool_backtest --opponent pool --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware --no-log`
(team-identity is now the default)

| run | seed | meta_region_poolaware ± 95% | edge | mean-rank paired |
|---|---|---|---|---|
| Step 3 (old `>=`/rank==1 conventions) | 4.54% | 10.93 ± 2.94 | +6.4 pp | 14/14 |
| Step 4 (expected first-place share; 2012 seed-rate opponents) | 4.71% | 11.61 ± 2.96 | +6.9 pp | 14/14, t = 10.1 |

Both modes rise because the old reporting rule discarded shared firsts.
Shipped 2026 win-maximiser `p1`: 0.0595 → 0.0703 (29 opponents, ties split);
its bracket is unchanged (Michigan).

## Final report

1. **Foundational defects:** F4-1 — "P(1st)" was three different quantities
   (selection, reporting, payout); the selector optimised one the pool does
   not pay and the table did not report.
2. **Material local defects:** F4-2 (shape-encoded scorer as default),
   F4-6 (2012 opponents from future pool data, with F4-5 the selection/
   evaluation opponent-model mismatch).
3. **Defects fixed:** F4-1, F4-2, F4-5, F4-6, F4-7, F4-8, F4-9.
4. **Historical results invalidated:** every P(1st) figure (selection and
   reported) before this step; the headline was rerun; all 14 artifacts
   rebuilt. Referee-suite, LORO, Romano–Wolf and 79-mode figures remain stale
   (Steps 7–9, now also for the tie rule).
5. **Tests added:** 38 Python checks across 5 new/rewritten files, 3 JS
   checks; all fail on the old code where the old behaviour differed.
6. **Assumptions explicitly accepted:** A-9 displayed `p1` is the
   selected-on estimate (winner's-curse bias for the argmax); A-10 UI
   right/wrong colouring is slot-based and shows no points; A-11 the static
   seed pick rates are a generic crowd prior for a season with no archive.
7. **Questions deferred:** legacy scorer scripts and the two diagnostic
   scripts with opposite tie placement (cleanup, not production).
8. **Step 4: PASS.**
9. **Downstream steps affected:** 7–9 (referee suite must be rerun under the
   unified P(1st) and the corrected topology/noseed window), 15 (Monte Carlo
   SE now refers to the share estimator), 17 (model-selection gate compares
   the unified quantity).
10. **Clean methodological breaking point:** for scoring and objective
    mathematics, yes — one scorer, one P(1st), one topology, independently
    verified, with the pool-correct definition driving selection, reporting
    and the product.
