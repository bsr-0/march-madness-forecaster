# Step 3 — Tournament propagation: findings, recorded before fixing

Protocol: `docs/METHODOLOGY_AND_REPORTS.md` item 3; plan of record
`~/.claude/plans/got-to-step-3-robust-sutherland.md` (approved 2026-09-15).
Base commit `0931de3` plus the uncommitted Step 1–2 working tree.

Pre-fix evidence in this directory: `topology_impact_pre_fix.{txt,json}`,
`propagation_audit_pre_fix.{txt,json}` (scripts `topology_impact.py`,
`propagation_audit.py`).

## F3-1 — FOUNDATIONAL — construction and scoring walked different brackets

`src/optimization/bracket_construction.py` hardcoded the Final Four as
East–West / South–Midwest (`_REGION_ORDER`; picks keyed `F4_East_West`,
`F4_South_Midwest`). The backtest scores a constructed bracket by projecting
that picks dict onto the season's REAL tree (`derive_f4_region_pairing`) in
`scripts/mc_pool_backtest.py:_picks_dict_to_bool_array`, which matched by
round-winner set and silently fell back to `t2`. When both F4 picks lay in
the same real semifinal, the scored bracket's F4/champion was not what
construction chose.

Measured pre-fix (`topology_impact_pre_fix.txt`), `region_top_n` at risk 0.35:

| season | real pairing ≠ default | Torvik base | blend base |
|---|---|---|---|
| 2012 | yes | ok | F4 mismatch |
| 2013 | yes | Indiana → **Florida** | ok |
| 2014 | yes | F4 mismatch | ok |
| 2015 | yes | Kentucky → **Virginia** | Kentucky → **Villanova** |
| 2016 | yes | ok | F4 mismatch |
| 2018 | yes | ok | ok |
| 2023 | yes | Houston → **Tennessee** | ok |
| 2025 | yes | F4 mismatch | F4 mismatch |
| 2026 | yes | ok | ok |

9 of 15 seasons on the wrong tree; 5 Torvik and 4 blend seasons where the
scored bracket differed from the constructed one. Every construction-mode
backtest figure to date, including the Step 2 rerun (10.9% vs 4.5%), was
measured on such brackets.

## F3-2 — MATERIAL-LOCAL — marginals simulated on the wrong tree

`build_bracket_order` / `build_torvik_round_probabilities`
(`scripts/mc_pool_backtest.py`) and `scripts/experiments/build_candidate_artifact.py`
hardcoded the same default pairing, so F4/CHAMP advancement marginals
(consumed by construction) and the entire shipped 2026 artifact (150k sims,
EV, P(1st), the UI's semifinal games) were on the wrong topology. Pre-fix
`propagation_audit`: `build_torvik_round_probabilities` PASSES against the
analytic recursion on the DEFAULT tree and FAILS against the REAL tree
(max |z| 11.9, 12.6% of cells beyond 2σ) — a correct marginalizer of the
wrong bracket. 2026 Torvik: Arizona F4 0.265 (default) vs 0.249 (real).
A prospective 2027 artifact had no data source for the pairing at all.

## F3-3 — MINOR — referee simulator silently defaulted a missing pair to 0.5

`src/simulation/pool_competition.py:simulate_tournament_outcomes` read
`matchup_probs.get((t1, t2), 0.5)` (one orientation) and had no
64-team guard. No production table is incomplete, so no numeric impact.

## F3-4 — MATERIAL-LOCAL (leakage) — noseed marginals not walk-forward

`src/prediction/noseed_model.py:build_noseed_round_probabilities` called
`_compute_advancement_rates()` and `_win_rate()` with `window="full"`,
`as_of=None`: the hardcoded 1985–2025 table containing every backtest
season's own results, while its blend partner `seed_rp` used `"recent"`,
`as_of=year` (Step 2 P3-1). The blend (the shipped win-maximiser's basis)
mixed a causal and a non-causal table.

## F3-5 — referee cap binds for the `pit` table (carried to Step 8)

`simulate_tournament_outcomes` caps every game at [0.01, 0.99] even with
noise 0. Inert for the seed table (max ≈0.985) and nearly so for Torvik
(10 ordered pairs, max excess 0.0037), but the `pit` referee has 86 ordered
pairs above 0.99 (Florida–Prairie View 0.995 → 0.99). Propagation is
faithful to the CLIPPED table (PASS); against the unclipped table it fails
(max |z| 32). Whether a referee should cap certainty at 0.99 is a referee
design decision (Step 8); Step 2 §10 wrongly called this cap inert and has
been corrected.

## F3-6 — MINOR — a third topology in research code

`src/prediction/meta_selector.py:_build_mc_round_probs` orders regions
alphabetically (East, Midwest, South, West → East–Midwest / South–West), a
third pairing used only by the research `meta_*` GBM modes. Not on the
`meta_region_poolaware` or shipped-artifact path. Recorded, not fixed.

## R-1 evidence (deferred research item from Step 2)

Noseed heuristic marginals vs propagated noseed pairwise on the real 2026
tree: max abs diff by round [0.21, 0.16, 0.08, 0.10, 0.06, 0.04] pre-fix,
[0.31, 0.26, 0.15, 0.11, 0.08, 0.06] after the window change; CHAMP mass
1.13 → 1.07. The compounding heuristic is not a propagation of anything.
Remains INDETERMINATE / deferred: replacing it is a modelling change.

## NON-ISSUES (tested, recorded so they are not reopened)

- Propagation itself: `simulate_tournament_outcomes` (200k, noise 0) and
  `marginals_from_pairwise` (10k) vs the independent recursion on the real
  2026 tree for Torvik, seed, pit (clipped), blend: all PASS
  (min Bonferroni p ≥ 0.55, |z|>2 fraction 2.9–5.5%).
- Seed-level recursion `_compute_advancement_rates("recent", as_of=2026)`
  vs positional recursion on the real tree with the seed table: max gap
  1.9e-16.
- Noise 0.16: per-round L1 shift vs analytic 0.09–0.12 (R64) falling to
  0.01–0.02 (CHAMP) — Jensen shrink, as characterised in Step 2 item 15.
- Play-in resolution: every production harness calls `resolve_first_four`
  before building the bracket (Step 2). The unified builder now refuses a
  contested slot; one test fixture (`tests/test_pool_objectives.py`) had
  been building a 2025 bracket by file order and was corrected.
