# Step 2 — Pairwise probability mathematics

Protocol: `FINAL_METHODOLOGY_AUDIT_PROTOCOL.md`, item 2, executed under the
21-item Step 2 directive of 2026-09-15. Base commit `0931de3`; Python 3.11.15,
numpy 1.26.3, scipy 1.16.3, node (see `node --version`). Three paths audited
separately; nothing collapsed into one "probability model".

Classification legend: FOUNDATIONAL / MATERIAL-LOCAL / MINOR / NON-ISSUE /
INDETERMINATE, per §19 of the directive.

## 0. Where the three paths differ (the one-paragraph answer)

Path 1 (live MODEL) converts an 11-feature ridge **margin** to P(win) through a
calibrated Student-t link, trained on tournament games only. Path 2 (shipped
Optimized bracket) converts Torvik / Massey / Elo **barthag ratings** to
P(A beats B) by Log5, simulates 150,000 tournaments with no noise, and judges
candidates' P(1st) against a **different** outcome model: the seed-vs-seed
historical table with 0.16 logit noise. Path 3 (research backtest) uses that
same seed referee to score `meta_region_poolaware`, whose candidates are built
from Torvik/Massey/Elo Log5 marginals and from a seed/noseed **blend of
heuristic advancement marginals** that is not derived from any pairwise table.
So the UI's fitted model, the shipped bracket's outcome model, and the referee
that grades both Path 2 and Path 3 are three different probability objects.

## 1. Inventory (item 1) — PASS

Every function converting ratings, margins, features, seeds or lines into a
probability was inventoried by three parallel searches of the whole tree
(not by filename). The full tables, with file:line, formula, orientation,
callers and live-vs-dead status, are in this directory's companion notes
(`inventory_path3.md` is the raw agent output; the load-bearing rows are
summarised here).

Live, production-critical transformations:

| # | file:function | in → out | formula | pairwise/marginal | used by |
|---|---|---|---|---|---|
| L1 | `docs/fit.js:fitLinear` / `winProbFromMargin` | z-diff vector → margin → P(team1 wins) | ridge β·x, then `clip(studentTCdf(a·m/σ, ν))` | pairwise | Path 1 |
| L2 | `src/prediction/pit_production_model.py:pairwise_for_year` | same, Python mirror | same | pairwise | Path 3 (`pit` referee) |
| L3 | `src/prediction/pairwise.py:log5` / `from_ratings` | two barthag → P(A>B) | a(1−b)/(a(1−b)+b(1−a)) | pairwise | Paths 2, 3 |
| L4 | `pairwise.py:simulate_bracket_outcomes`, `marginals_from_pairwise` | pairwise → per-sim winners → marginals | independent Bernoulli per game, bracket walk | pairwise→marginal | Paths 2, 3 |
| L5 | `src/data/seed_pick_model.py:_win_rate` (window="recent") | two seeds → P(a>b) | shrunk 2010+ cell rate, logistic fallback | pairwise | Paths 2, 3 (referee) |
| L6 | `seed_pick_model.py:_compute_advancement_rates` | seed pairwise → seed marginals | exact opponent-mixture recursion | pairwise→marginal (analytic) | Paths 2, 3 |
| L7 | `src/simulation/pool_competition.py:simulate_tournament_outcomes` | referee pairwise (+0.16 logit noise) → outcomes | `sigmoid(logit p + ε)`, Bernoulli | pairwise→outcome | Paths 2, 3 |
| L8 | `pool_competition.py:_get_pick_prob` | ESPN marginal pick shares → P(opponent picks t1) | t1/(t1+t2) | marginal→pick prob (opponent model, not outcome) | Paths 2, 3 |
| L9 | `src/prediction/noseed_model.py:predict_win_prob` | two stat dicts → P(t1) | ½ logistic + ½ sigmoid(GBM spread/11) | pairwise | Path 3 |
| L10 | `noseed_model.py:build_noseed_round_probabilities` | seed marginals × (1+mean_adv)^round | heuristic | marginal (not propagated) | Paths 2, 3 (blend construction) |
| L11 | `scripts/experiments/conditional_bracket_engine.py:expected_scores` | bank + marginals → E[points] | Σ pts_R · P(pick wins R) | marginal | Path 2 |

Duplicates of L3 (`torvik_kaggle._log5`, `meta_selector._log5`,
`mc_pool_backtest._log5` alias, `massey_best._pairwise_win_prob`) were compared
numerically to the canonical function on a 49-point grid: max diff 0.
`proprietary_metrics._log5_win_prob` is misnamed (a logistic on efficiency
margin, feature-engineering only). Dead: `src/simulation/monte_carlo.py`,
`dual_submission.py`, `competitor_archetypes.py`, `build_optimized_brackets`,
`EnsembleKagglePredictor`, `run_pool_simulation` (no runtime callers).

## 2. Semantics (item 2) — PASS

Traced end to end with real data: 2026 R64 matchups through `app.js` logic
replicated in Node (`p1_ui_replica_before_fix.js`), through
`pairwise_for_year`, and through the artifact's `pairwise[i*n+j]` written at
`build_candidate_artifact.py:1040` (P(row beats col)) and read at
`selection.py:323` (same). `PairwiseProbabilities.p(t1,t2)` = P(t1 beats t2);
`probs[(t2,t1)] = 1-p` on construction. Path 1: `x = z(team1) − z(team2)`,
`m = team1 − team2 score`, orientation by pre-tournament fact
(`build_training_matrix.py:_orient`), P = P(team1 wins). No consumer reads a
probability with the opposite orientation. P(1st) = fraction of CRN trials
where candidate score ≥ max opponent score (ties count as wins — carried to
Step 4).

## 3. Log5 (item 3, 17) — PASS

Independent exact-rational reference (`fractions.Fraction`) vs `log5` over 14
cases including equal, extreme, near-0/1 and mismatched ratings: max abs error
1.1e-16; complement P(A>B)+P(B>A)=1 to 1e-12; ordering and monotonicity hold;
degenerate (0,0)/(1,1) → 0.5 by documented floor. No transformation is applied
before or after Log5 on the Path 2/3 outcome path (Massey/Elo/AP convert
their ratings to barthag-equivalents first; those clips are inventoried in §10).

## 4/5. Margin → probability and the live MODEL (items 4, 5) — PASS after fix P1-1

- `studentTCdf` vs scipy over 171 (ν, t) points: max abs err 6.9e-8.
- margin 0 → 0.5 exactly (27 (σ, a, ν) combinations); P(+m)+P(−m)−1 = 0 exactly.
- Python mirror vs JS replica, 2019/2025/2026 R64 games: max diff 6.6e-14 once
  both calibrate the same way. Feature order, standardisation, no-intercept,
  σ and ν therefore agree.
- **P1-1 (MATERIAL-LOCAL, leakage, FIXED).** `app.js:refit()` called
  `crossValidate()` on the whole matrix, so the link's (a, ν) shown for any
  season had been fitted on that season's own outcomes and on later seasons'
  (one global a=1.534, ν=3 for every displayed year). The frozen baseline
  (`model_baseline.js`) and the Python mirror already calibrated causally with
  shrinkage; the page did not. Magnitude: ≤1.1 pp on 2026 R64 probabilities,
  2.4 pp on 2019, zero pick flips. Fix: `fit.js:causalWalkForward()` (rows
  strictly before `asOf`, a shrunk by n/(n+63)), `app.js` uses it; cache tags
  bumped. Regression tests in `tests/test_calibration.js` (corrupting seasons
  ≥ asOf must not move the calibration; the old path demonstrably does; and a
  source guard that `refit()` no longer calls `crossValidate`).
- **P1-2 (MINOR data defect, FIXED).** `tournament_context_2025.json` named
  San Diego State (First Four loser) in North Carolina's R64 game. One of
  1,008 training rows carried the wrong team's features. Ground-truth walking
  was unaffected (per-team fallback resolved it). Fixed the record; added
  invariant D (First Four winners must appear in R64, losers must not) to
  `scripts/audit_tournament_results.py`, which had passed the file; wired the
  auditor into pytest (`tests/test_tournament_results_integrity.py`,
  `data_contract`). Rebuilt `training.json` (1 row, features only) and refroze
  `artifacts/model_baseline.json`: walk-forward warm log loss 0.45296 →
  0.45391, accuracy 77.78% → 77.65%. The corrected number is the honest one.

## 6. Probability source of the shipped product (item 6) — PASS

`build_candidate_artifact.py`: sources = Torvik barthag (pre-tournament,
provenance-gated), Massey composite (clipped [0.10, 0.99]), Elo→barthag; each
→ Log5 → 50,000 sims, no noise, no blending, no risk tempering, no calibration
on the Python side. `risk_level` only reweights the region_top_n pick score
by public-pick uniqueness. `meta.source = "log5(torvik_2026)"` labels the
shipped `pairwise` table and the EV marginals (Torvik only); the shipped
`team_round_probabilities` / `constraint_probabilities` are frequencies over
the three-source mixture (a design choice, documented in code). P(1st) is
judged by the seed referee with 0.16 noise (L5+L7), so `p1` and `ev` live in
different probability worlds by design. The browser computes no probability
for shipped strategies (only SE and formatting). Step 1 had already shown the
artifact rebuilds identically from current code.
- **P2-1 (MINOR, FIXED).** The "true" frequencies were counted over `rounds`
  after the ~21 constructed/shipped brackets were appended, i.e. over
  150,021 rows of which 21 were deterministic picks. Fixed to count over
  simulations only; regression test `tests/test_candidate_artifact_frequencies.py`
  (every shipped frequency must be k/n_sims; fails on the old code). Rebuilt
  the artifact and payloads: `candidates`, `p1`, `ev`, `pairwise` identical;
  six user-facing predicate probabilities moved by ≤1e-4.
- Note: `artifacts/candidates/*.json` are gitignored; only the `.sha256` is
  tracked, and that hash includes `meta.generated_at`, so it cannot match a
  rebuild. Carried as a MINOR reproducibility note (Step 1 territory).

## 7/8/9. Pairwise ↔ marginal separation and propagation (items 7, 8, 9) — PASS

Independent analytic recursion (`independent_propagation_reference.py`,
written from the definition P_r(t) = P_{r−1}(t)·Σ_o P_{r−1}(o)·p(t,o)) on a
synthetic 64-team bracket with extreme ratings:
- analytic totals per round exactly 32/16/8/4/2/1; monotone; bounded;
- `simulate_bracket_outcomes` (200,000 sims) vs analytic over 384 cells:
  max |z| 3.06 at a cell with p=3e-6; 2.6% of cells beyond 2σ (expected 4.6%);
- `marginals_from_pairwise` (100,000 sims): worst cell p=4.7e-7, observed 1,
  exact binomial p=0.09 — noise;
- conditional opponent mixture verified explicitly (P(win R32) = P(win R64) ×
  Σ_B P(B reaches) p(A,B)), analytic 0.00019 = product 0.00019, sim 0.00017;
- `PairwiseProbabilities` is a frozen dataclass; no write path from marginals.
  The two marginal→pairwise sites that exist (`meta_selector._pairwise_prob`,
  `_bracket_export_common.build_bracket_json`) are allow-listed in
  `tests/test_pairwise_contract.py`, are on research/dead paths
  (meta_* GBM modes; `docs/data/bracket_2026.json`, not read by the page), and
  are not used by `meta_region_poolaware` or the shipped artifact. NON-ISSUE
  for production, recorded so it is not reopened.
- `_get_pick_prob` (t1/(t1+t2) over ESPN pick shares) is an opponent-pick
  model, not an outcome probability; accepted assumption, Step 6.
- L10 (noseed heuristic marginals) is NOT propagation. Measured for 2026:
  round-sum mass 33.25/16.83/8.48/4.28/2.18/1.12 against 32/16/8/4/2/1 (seed
  table alone: 33.06/…/1.00; the R64 excess is the 68-team field before
  play-in resolution). It is an empirical model with modest incoherence, used
  only for ordering inside region_top_n, and it is the construction whose
  out-of-sample evidence exists. Replacing it with propagated noseed pairwise
  marginals is a modelling change that would invalidate that evidence, not a
  correctness fix. Classified **INDETERMINATE → deferred as post-retirement
  research item R-1**, with the specific defect that it calls
  `_compute_advancement_rates()` and `_win_rate()` on the "full" window while
  `seed_rp` uses "recent" (window inconsistency, MINOR, recorded).

## 10. Clipping (item 10) — PASS

All clips inventoried (Path 3 inventory Q3). Load-bearing ones: Path 1
`PROB_CLIP = 1e-3` after the link (binds only under ν=∞/12 cold-start years,
documented); referee pre-noise [0.001, 0.999] and post-noise [0.01, 0.99]
(binds only above 0.99; inert for the seed table, max ≈0.985, but NOT for
the `pit` referee, whose table has 86 ordered pairs above 0.99 — corrected
by Step 3, finding F3-5, and carried to Step 8 as a referee design item);
`marginals_from_pairwise` floor 0.001 (protects downstream ratios);
barthag fallbacks `max(0.10, 1−seed·0.04)` for teams missing a rating (a
modelling default, provenance-visible). None can create a directional
artefact: every clip is symmetric under team swap or applied to a marginal.

## 11. Calibration (item 11) — PASS after P1-1

Only Path 1 calibrates. Fitted once, applied once, to the quantity it was
fitted on (a·margin/σ). Causal after P1-1. Python mirror and frozen baseline
were already causal. Paths 2/3 have no calibration layer.

## 12. Venue (item 12) — PASS

Path 1 trains on tournament games only; no venue term exists. Paths 2/3 use
ratings and seed tables; no home-court term exists in Log5, the seed table or
the referee. The research matrices carry venue columns: `eval_pit_tournament`
has `venue_home` = `venue_host_city` = 0 on all 865 rows; `venue_travel` is a
travel-distance differential, defined for neutral sites. No tournament
probability receives a home-court adjustment.

## 13. Regular-season training interface (item 13) — deferred

Not on any live path (Path 1 uses tournament rows only; Paths 2/3 use
ratings). `build_pit_training_matrix.py` documents strictly-earlier snapshot
joins and within-boundary standardisation; verification is Step 13's job and
was not expanded here (§21).

## 14. Seed-based probabilities (item 14) — FAIL → FIXED → PASS

- **P3-1 (MATERIAL-LOCAL, leakage, FIXED).** `build_seed_probabilities` and
  `build_seed_round_probabilities` built one 2010–2025 table for every season.
  In `mc_pool_backtest._run_one_year` the referee for season Y therefore
  contained Y's results (README acknowledged the overlap; the protocol
  forbids it). The noseed model in the same function was walk-forward and
  asserted; the seed table was not. Fix: `as_of` cutoff threaded through
  `seed_pick_model._recent_win_rates` (cache keyed by cutoff), `_win_rate`,
  `_compute_advancement_rates`, both public builders, and the four runtime
  callers (`mc_pool_backtest`, `build_candidate_artifact`, `referee_audit`,
  `generate_poolaware_bracket`). The "full" 1985–2025 crowd table is left as
  is (crowd model; target season is ~2.5% of a cell; accepted assumption A-2).
  Regression tests `tests/test_seed_table_walk_forward.py` (independent CSV
  tally; 2011 has no eligible cells; consecutive cutoffs differ; cache keyed).
- Play-in handling: `resolve_first_four` removes losers before the bracket is
  built in all four harnesses (verified over 15 seasons; my first check
  omitted the call and was wrong). NON-ISSUE.
- Missing seed cells fall through to the logistic curve with an 8-game
  shrinkage; documented.
- **Rerun (results invalidated by P3-1):** the headline command under the
  walk-forward referee gives seed 4.54% ± 0.68, `meta_region_poolaware`
  10.86% ± 2.90 (was 3.99% / 12.00%); edge +6.3 pp (was +8.0); mean-rank
  paired t = 20.7, 14/14; best-rank 14/14. Per-season P(1st) moved by −0.05 to
  +0.03. README headline amended with a dated correction; every other figure
  in that README section is flagged as measured under the old referee and
  queued for Steps 7–9.

## 15. The 0.16 logit noise (item 15) — PASS WITH DOCUMENTATION FIX

Implementation: independent ε ~ N(0, 0.16) per game per trial, then a
Bernoulli draw. A mixture of Bernoullis is a Bernoulli, so the referee's
outcome law is Bernoulli(E[sigmoid(logit p + ε)]) — no added variance
(simulated Var 0.16078 vs E[q](1−E[q]) 0.16072 at p=0.8) and no parameter
uncertainty (that would require shared draws). Its only effect is Jensen's
shrink toward 0.5: ≤0.13 pp for p ∈ [0.5, 0.98]. Consistent with the README's
finding that the knob is flat. The comment block in `mc_pool_backtest.py`
now states this; the constant is untouched (not tuned, per §15).

## 16. Independence assumptions (item 16) — PASS

A. Games: independent Bernoulli conditional on pairwise p — explicit, intended.
B. Path dependence: the bracket walk conditions each round on realised
   winners (verified in §7–9).
C. Expected score: `expected_scores` equals the mean realised score over the
   same bank to 6e-14 on three candidates — linearity of expectation, no
   independence used.
D. Estimation: ridge on 1,008 games treats a team's games as independent
   rows; affects standard errors, not point estimates; the only intervals
   reported are season-level bootstraps. Accepted assumption A-3.

## 17. Independent references (item 17) — PASS

Log5 (exact rational), Student-t link (scipy), pair reversal (both), bracket
recursion (own implementation), expected points (mean realised score). All
agree within numerical tolerance; none tests the production function against
itself.

## 18. Defects and regression tests (item 18)

| id | defect | class | recorded before fix | fix | test | artifacts rebuilt |
|---|---|---|---|---|---|---|
| P1-1 | UI calibration fit on displayed + later seasons | MATERIAL-LOCAL (leakage) | yes (`p1_ui_replica_before_fix.js`) | `fit.js`, `app.js`, cache tags | `test_calibration.js` ×5 | none needed |
| P1-2 | 2025 results row names FF loser | MINOR (data) | yes | `tournament_context_2025.json`; auditor invariant D | `test_tournament_results_integrity.py` | `training.json`, `model_baseline.json` |
| P2-1 | true frequencies counted over appended constructed brackets | MINOR | yes | `build_candidate_artifact.py` | `test_candidate_artifact_frequencies.py` | `candidates_2026.json`, `season_*.json` |
| P3-1 | seed referee/table not walk-forward | MATERIAL-LOCAL (leakage) | yes | `as_of` through seed table + 4 callers | `test_seed_table_walk_forward.py` | headline backtest rerun; README amended |
| D-1 | 0.16 noise described as a variance component | doc | — | comment | — | — |

Side findings, not fixed (out of scope or trivial): `scripts/loyo_pergame_predictions.py`
imports a module deleted in `ea06a40` (dead script); `src/product/selection.py`
cites `tests/test_product_parity.py`, which does not exist (Step 4/5);
`_log5_win_prob` misnamed; `candidates_*.sha256` includes a timestamp.

## 19/20. Gate table

| Area | Path | Result | Evidence | Severity | Required action | Downstream impact |
|---|---|---|---|---|---|---|
| Inventory | all | PASS | §1 | — | — | — |
| Semantics/orientation | all | PASS | §2 | — | — | — |
| Log5 | 2, 3 | PASS | §3 | — | — | — |
| Margin→prob link | 1 (3 via `pit`) | PASS | §4 | — | — | — |
| Live MODEL calibration | 1 | FAIL→PASS | P1-1 | MATERIAL-LOCAL | done | displayed historical probabilities move ≤2.4 pp; 2027 unaffected |
| Training data row | 1 | FAIL→PASS | P1-2 | MINOR | done | baseline log loss 0.45296→0.45391 |
| Shipped probability source | 2 | PASS | §6 | — | — | — |
| True frequencies | 2 | FAIL→PASS | P2-1 | MINOR | done | predicate probabilities move ≤1e-4 |
| Pairwise↔marginal direction | all | PASS | §7 | — | — | — |
| Propagation vs analytic | 2, 3 | PASS | §8, §9 | — | — | — |
| Noseed heuristic marginals | 2, 3 | INDETERMINATE | §9 | research | R-1 post-retirement | none now (construction validated as is) |
| Clipping | all | PASS | §10 | — | — | — |
| Calibration causality | 1 | PASS (after P1-1) | §11 | — | — | — |
| Venue | all | PASS | §12 | — | — | — |
| Regular-season interface | research | deferred | §13 | — | Step 13 | — |
| Seed referee walk-forward | 2, 3 | FAIL→PASS | P3-1 | MATERIAL-LOCAL | done; reruns queued | headline 12.0%→10.9%; referee-suite, RW, sweep figures invalidated |
| 0.16 noise | 2, 3 | PASS w/ doc fix | §15 | doc | done | — |
| Independence | all | PASS | §16 | — | — | — |
| Independent references | all | PASS | §17 | — | — | — |

## Final report (§21)

1. **Foundational defects found:** none. No probability is reversed, no
   marginal feeds a live pairwise, propagation is exact, links are correct.
2. **Defects fixed:** P1-1, P1-2, P2-1, P3-1 (two leakage, two minor), plus
   the noise documentation. Each recorded before fixing, fixed at the source,
   regression-tested, artifacts rebuilt only afterwards.
3. **Results invalidated / requiring rerun:** every Path 3 figure measured
   under the pre-P3-1 referee. The headline was rerun (10.9% vs 4.5%). The
   referee-suite deltas, leave-one-referee-out, Romano–Wolf multiplicity, the
   79-mode sweep and the fixed-rule comparison have not been rerun and are
   flagged in README; they belong to Steps 7–9 and 15. Path 1's frozen
   baseline was refrozen after P1-2. Path 2's shipped artifact was rebuilt
   (selection unchanged).
4. **Assumptions explicitly accepted:** A-1 the referee for P(1st) is the
   seed table, a different model from the candidate generator (by design);
   A-2 the "full" 1985–2025 crowd table is not cut off by season (crowd model,
   ~2.5%/cell); A-3 ridge rows treated as independent; A-4 ESPN pick shares
   converted to head-to-head pick probability by t1/(t1+t2); A-5 ties count
   as first place in P(1st) (to Step 4).
5. **Questions deferred:** R-1 propagated vs heuristic noseed marginals (and
   the full/recent window inconsistency inside it); item 13 to Step 13;
   `sha256` pin includes a timestamp; dead modules for cleanup.
6. **Step 2:** **PASS** — no foundational defect remains; all material
   probability paths identified and verified; leakage found was repaired and
   the affected result rerun.
7. **Safe to proceed to Step 3:** yes, with the explicit carry-forward that
   Path 3's secondary figures are stale until Steps 7–9 rerun them.

## Reproduce

```bash
node tests/test_calibration.js
PYTHONPATH=. python3 -m pytest -p no:asyncio tests/test_pit_production_port.py tests/test_seed_table_walk_forward.py \
  tests/test_tournament_results_integrity.py tests/test_candidate_artifact_frequencies.py tests/test_pairwise_contract.py -q
PYTHONPATH=. python3 artifacts/methodology_audit/step2/independent_propagation_reference.py
node scripts/model_baseline.js --compare
python -m scripts.mc_pool_backtest --team-identity --opponent pool --n-opponents 29 --n-repeats 100 \
  --modes seed meta_region_poolaware --no-log     # -> 10.86% / 4.54%
```
