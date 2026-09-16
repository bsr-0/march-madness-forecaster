# Step 5 — Candidate-space integrity

Protocol item 5: can candidate generation represent materially different
plausible brackets, and is the search constrained by candidate availability?
PASS = adequate coverage, no construction artifact drives results; FAIL =
optimizer performance depends materially on missing candidate regions.

Evidence in this directory: `candidate_coverage.{py,txt,json}` (Path 3),
`artifact_sample_vs_bank.{py,txt}` (Path 2). Base: Step 1–4 working tree.

## What the candidate spaces are

**Path 3 (`meta_region_poolaware`).** Per season: `region_top_n` over
≤5 probability bases (`tv`, `mass_avg`, `mass_best`, `blend`, `tv_mass80`)
× risk grid (0.1, 0.3, 0.5, 0.7, 0.9), plus `exhaustive_champion` × bases ×
(0.3, 0.5, 0.7), byte-deduplicated, first label kept. 17–33 unique brackets;
selection = first argmax of P(1st) on 500 CRN trials. `region_top_n` is a
per-region beam search (beam 500 of 2^15) of which only the single best
outcome per region is used; its champion is always one of the four per-region
EV-argmax survivors. Measured on the frozen sets: 2–5 distinct champions and
4–6 distinct Final Fours per season; min pairwise Hamming distance 1 in every
season.

**Path 2 (artifact).** 150k Log5-simulated tournaments over three rating
sources → stratified sample of ~2,965 (champion quotas with a floor of 8, EV
deciles within champion) + 18 constructed/shipped brackets appended (not
deduplicated). P(1st) is computed only for the sample, never the full bank.

## F5-1 — MATERIAL-LOCAL (construction artifact) — the "forced champion" family was inert

The production candidate set's family (a), "forced 1-seed champions ×
`region_top_n`", produced four byte-identical copies of the unforced
`tv_region_risk=0.5` bracket: `region_top_n` (and `simulated_annealing`,
`exhaustive_champion`) accepted `forced_champion` and ignored it. Verified on
2026: forcing Duke, Arizona, Florida or Michigan all return the Michigan
bracket, identical to the unforced one. Consequences: (i) selected-candidate
labels such as `tv_champ=syracuse` (2012, published logs) named a champion
that was never forced; (ii) the rDoF audit registered a knob
(`poolaware_forced_champion_risk`) that does nothing; (iii) the legacy sweep
mode `meta_region_4champ` — "one bracket per forced 1-seed, best P(1st)" —
evaluated four copies of one bracket, so every number it produced was the
plain `tv_region_risk=0.5` result. Numerically the production candidate SET
was unaffected (dedup kept one copy, which the risk sweep also produced).

Fix: `construct_bracket` raises `ValueError` when `forced_champion` is passed
to a mode that cannot honour it; family (a) removed from the backtest and the
referee-audit mirror (candidate set unchanged; only the misleading label is
gone); `meta_region_4champ` raises with the finding rather than re-running as
if it measured something. Regression test
`tests/test_bracket_construction.py::test_modes_that_cannot_force_a_champion_refuse_one`.
Headline rerun to confirm the tie-break order change moves nothing (below).

## F5-2 — MATERIAL-LOCAL (coverage) — the optimizer is constrained by candidate availability

Experiment (`candidate_coverage.py`): for each of 15 seasons, S = the
production candidate set; A = S + the season's artifact sample (~3,000 legal
brackets on the same tree) + 300 brackets sampled game-by-game from the seed
pairwise table. Argmax P(1st) on 500 selection trials (production seed);
every selection then scored on 1,500 independent trials (SE ≈ 0.008).

| | mean over 15 seasons |
|---|---|
| P(1st)_eval of the production pick S* | 0.1053 |
| P(1st)_eval of the augmented pick A* | 0.1175 |
| gain | **+0.0122** (median +0.006) |
| seasons with gain > 2 SE | 4 (2013 +0.040, 2014 +0.020, 2015 +0.058, 2021 +0.034) |
| seasons with gain > 0 | 11 |
| seasons where A* is a production candidate | 3 (2022, 2025, 2026) |
| A* provenance | artifact bank 7, seed-sampled 5, production 3 |
| share of A beating S* on evaluation | 0.0–7.9% (median 1.0%) |

So a wider search over legal brackets finds, out of sample, about +1.2 pp
of P(1st) on average (≈10% relative to the 11.6% headline), concentrated in
four seasons. The production set is not where the P(1st) optimum lives in
12 of 15 seasons. Candidate order matters in one season (2015: shuffling S
changes the pick — an exact P(1st) tie on 500 trials).

Classification. This is a limitation of the strategy's definition, not an
error in how it is evaluated: the 11.6% is a correct measurement of the
strategy as specified, and the candidate set contains only legal, causally
built brackets. Expanding it is a strategy change that must be pre-registered
and re-validated (Steps 9–17), not made inside a foundational audit (§19 of
the Step 4 directive; protocol C). Recorded as **R-2, optimization item**:
"candidate generation for `meta_region_poolaware` should include a diverse
legal bank (e.g. the artifact sampler's output on the season's tree); expected
out-of-sample gain ≈ +1.2 pp, to be confirmed under the Step 17 gate."

## Path 2 — PASS

`artifact_sample_vs_bank.py` (2026, 30k-sim bank, 1,000 shared trials): the
production stratified sample's P(1st) distribution matches uniform and larger
draws from the same bank (max 0.105 = bank max among 12,418 scored; p99 0.081
vs 0.081; p90 0.062 vs 0.062); it keeps all 44 champions the bank produced;
6 of the bank's top-20 P(1st) brackets are in the production sample (3 in an
independent stratified draw), i.e. the sampler does not lose the high-P(1st)
region — it is a random-sample-level representation of it. The shipped 2026
artifact reports 64/64 champions, mean Hamming 25.6 (bank 25.0),
`low_ev_high_p1_count` 153.

## Minor findings (documented, fixed where trivial)

- F5-3 `region_top_n` docstring implied a top-N search; it uses only the best
  outcome per region and its champion is confined to four survivors. Docstring
  corrected.
- F5-4 `build_candidate_artifact.py` promised a "constraint top-up" step that
  does not exist; coverage is counted and asserted in the integration test
  only. Docstring corrected.
- F5-5 The 18 appended constructed/shipped rows are not deduplicated against
  the sample or each other (harmless: `select_diverse` requires distinct
  compositions). Recorded.
- Selection tie-break is "first max wins" at four sites; documented as
  load-bearing in `poolaware_recipe.py`. Under the split-share P(1st) exact
  ties are rarer but occurred in 2015.

## Gate

| Area | Path | Result | Evidence | Severity | Fix | Downstream impact |
|---|---|---|---|---|---|---|
| Forced-champion family inert / mislabelled | 3 | FAIL → fixed | F5-1, verified on 2026 | MATERIAL-LOCAL (construction artifact) | raise + removal | labels in old logs; `meta_region_4champ` numbers void; rDoF registry entry void |
| Coverage of the production candidate set | 3 | **FAIL (material, non-foundational)** | `candidate_coverage.txt`: +1.2 pp out-of-sample from a wider legal search, 4/15 seasons > 2 SE | MATERIAL-LOCAL (strategy limitation) | R-2, pre-registered expansion under Step 17 | headline is a valid measurement of a sub-optimal search |
| Artifact sample vs bank | 2 | PASS | `artifact_sample_vs_bank.txt` | — | — | — |
| Construction depends on actual results? | 2, 3 | PASS | inputs are seeds, regions, marginals, picks, announced pairing | — | — | — |
| Docstrings vs behaviour | 2, 3 | MINOR → fixed | F5-3, F5-4 | MINOR | done | — |

**Step 5 verdict: FAIL on coverage, non-foundational.** No result is
invalidated by it; the evaluation machinery is sound (Steps 2–4) and the
candidate set is legal and causal. The finding says the strategy leaves a
measurable amount on the table, and the correct response is a pre-registered
expansion at the optimization stage, not an in-audit tune. Per protocol M this
is an explicit decision: proceed to Step 6 with R-2 recorded as a required
Step 17 input.
