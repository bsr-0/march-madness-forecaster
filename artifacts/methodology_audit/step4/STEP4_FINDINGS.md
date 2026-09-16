# Step 4 — Scoring and objective mathematics: findings, recorded before fixing

Protocol: `FINAL_METHODOLOGY_AUDIT_PROTOCOL.md` item 4, executed under the
22-item Step 4 directive of 2026-09-15. Base commit `0931de3` plus the Step 1–3
working tree. Inventory: one exhaustive read-only sweep of every point table,
scorer, P(1st) site, opponent draw, RNG seed and exception handler in
`src/`, `scripts/`, `docs/`, `configs/` (summarised in `STEP4_GATE.md` §1).

## Canonical scoring (item 1) — PASS

One point table everywhere: ESPN 10/20/40/80/160/320 for R64…CHAMP, additive,
per correctly picked winner, no play-in points (`FF` games are excluded from
`actual_winners_by_round`, from training rows and from the 63-game vector).
Twelve copies of the same literal exist (backtest, simulator config,
construction, CLI, frozen spec, legacy scripts); the only different table is
the explicitly named `flat` adapter in `leverage.py`, reachable only by
opting in. No point table exists in JavaScript: the page displays `ev`/`p1`
and never sums points.

## F4-1 — FOUNDATIONAL (objective identity) — three tie conventions under one name

"P(1st)" meant three quantities:

| stage | code | a tie for first is worth |
|---|---|---|
| selection (`score_candidate_p1`, `p_first_from_scores`, `pool_p_first`, `compute_bracket_win_probability`, `recency_hparam_fitter`) | `>=` | 1.0 |
| reporting (backtest table `p_first`, `referee_audit.evaluate_season`) | `(rank == 1)`, rank = better + 1 + tied/2 | 0.0 |
| payout (`_record_prize`, `payout.prize_for_scores`) | tied slice mean | 1/(1+k) |

`tests/test_tie_conventions.py` had pinned the discrepancy as documented
behaviour. Measured on the 2026 artifact's trials (29 opponents, 2,000 CRN
trials, 103 brackets): ties are 7.4% of the events the `>=` rule counted as
wins; for the shipped win-maximiser P(sole 1st) = 0.0695, P(tie) = 0.0085,
so the selector's number (0.078) and the table's number (0.0695) bracketed
the pool-correct 0.0737. The selector therefore optimised a quantity the
backtest did not report and the pool does not pay.

## F4-2 — MATERIAL-LOCAL — the shape-encoded scorer was the backtest default

`run_backtest(team_identity=False)`: without `--team-identity` the harness
scored with `score_brackets_against_outcome`, a positional slot match whose
own docstring says it "can credit a bracket for advancing a team that never
actually won". The documented headline command passed the flag, and the
frozen spec says "team_identity (never shape-encoded)", but any run that
forgot the flag silently used the wrong scorer.

## F4-5 — MINOR — selection and evaluation used different opponent models

Selection trials passed `chalk_noise_std=pool_chalk_noise_std`; the final
evaluation (both `shared` and `per_mode`) did not. Zero for every season
with pool or ESPN data, so no numeric effect on the canonical run except
through F4-6.

## F4-6 — MATERIAL-LOCAL (leakage) — 2012 opponents from the future

2012 has no ESPN pick archive. The fallback built a behavioural opponent model
from the 2023–2026 pool brackets (`build_pool_behavioral_model`), i.e. crowd
behaviour from eleven years after the season being evaluated, and attached a
chalk-noise parameter the evaluation stage then ignored (F4-5). One of 14
evaluation seasons.

## F4-7 — MINOR (same class as F3-1) — two more lenient set projections

`build_candidate_artifact._encode_rows` and `docs/app.js:solveFromPicks`
turned per-round winner sets into a bracket with `has(a) ? a : b`, inventing
a winner whenever a set named neither or both teams of a game. Harmless with
the artifacts now built on the real tree, but the same silent-reconciliation
pattern that produced F3-1.

## F4-8 — MINOR — the artifact's P(1st) pool had 31 entries

`build_candidate_artifact` passed `n_opponents=DEFAULT_POOL_SIZE` (30), i.e.
30 opponents plus us, while the frozen spec, README and backtest describe a
30-entry pool (29 opponents). The artifact's own `p1_assumption` text said
"30-opponent pool".

## F4-9 — MINOR (estimator hygiene) — failed constructions vanished silently

`_pa_try_add` in `meta_region_poolaware` and `try_add` in `referee_audit`
caught every exception with `pass`, so a construction that failed shrank the
candidate set without a trace (Step 2 side finding, now in scope). No
failures occur today (Step 1 D-2 check), but the count was invisible.

## NON-ISSUES (tested, recorded)

- Scorer vs independent slot-based reference: exact on perfect, single-error
  per round, impossible-downstream-pick, early-upset and 300 random cases
  (`tests/test_scoring_independent.py`). Path legality is structural: a
  63-bool vector can only carry a team into round R by picking it through
  R-1, and `picks_by_round` decodes exactly that.
- Ground truth vs topology: for all 15 seasons the walked
  `build_actual_outcome` vector decodes to exactly `actual_winners_by_round`
  (`tests/test_actual_outcome_consistency.py`), so the per-team fallback
  never fires on the real tree.
- Expected value: `expected_scores` = mean realised score over the same bank
  to 1e-9 (linearity; no opponent term — "EV" is absolute expected points).
  `_compute_expected_points` (construction) is the same expectation with an
  independent-pick variance term, used only for display.
- P(1st) exhaustive: 64-team bracket with 4 coin-flip games (16 outcomes),
  fixed opponents including a duplicate of a candidate: `pool_p_first` and
  `score_candidate_p1` equal the hand-computed first-place share to 1e-12,
  and the Monte Carlo bank converges to it (`tests/test_objective_exhaustive.py`).
- Objective identity: on a fixture where EV and P(1st) disagree,
  `product.selection.select` picks by the objective it is given.
- CRN (item 11): 20 candidates × 12 seeds, shared vs independent 400-trial
  sets: mean P(1st) 0.0354 vs 0.0358 (diff −0.0004, SE 0.0019); variance of
  pairwise candidate differences 1.18× lower under CRN
  (`crn_experiment.txt`). Selection and final evaluation use different seeds
  (77777+year vs 42+year), so the reported per-season P(1st) is not the
  in-sample selection value.
- Baseline (item 14): `seed` mode is scored in the same loops, same
  topology, same trials, same scorer and now the same tie rule; it averages 50
  sampled brackets × repeats where the meta mode has one bracket × repeats —
  both are "P(a bracket from this mode finishes first)".
- Denominators (item 16): per-season P(1st) = mean first-place share over
  (brackets × 100 repeats); headline = unweighted mean over the 14
  evaluation seasons; CI = t-interval over seasons. Year-level skips would
  remove a season from all modes (none occur in the canonical run).
- Variance (item 17): each trial redraws the opponent field AND the
  tournament, so P(1st) marginalises over both; the candidate is fixed.
  Per-season SE at 100 repeats ≈ 0.03.

## Accepted assumptions

- A-9 The displayed `p1` of a selected candidate is the same trials it was
  selected on: an unbiased estimate of that candidate's P(1st), but the
  argmax over ~3,000 candidates is optimistically biased as an estimate of
  the chosen bracket's true P(1st) (winner's curse). Acknowledged in
  `build_ui_payload.py`; the backtest's out-of-sample column is the honest
  number for the strategy.
- A-10 The UI's right/wrong colouring is slot-based (stricter than the
  team-identity scorer for cross-region path differences at F4/CHAMP); it
  shows no points total, so nothing it displays contradicts the scorer.

## Deferred

- Legacy scripts `scripts/_common.score_bracket_espn` (credits only a team's
  deepest round) and `build_chalk_picks` (iterates actual games): not on any
  production path; cleanup.
- `real_pool_placement.py` (ties placed last) vs `noise_floor_ceiling.py`
  (ties placed first): diagnostic scripts with opposite conventions;
  cleanup to `first_place_share`.
