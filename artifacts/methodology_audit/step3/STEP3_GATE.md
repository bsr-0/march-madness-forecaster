# Step 3 — Tournament propagation: gate table and final report

Findings recorded before fixing: `STEP3_FINDINGS.md`. Evidence:
`topology_impact_{pre,post}_fix.*`, `propagation_audit_{pre,post}_fix.*`,
`path3_backtest_full_fix.txt`, `path3_backtest_topology_only.txt`
(+ `run_topology_only.py`, the attribution harness). Base commit `0931de3`
plus the Step 1–3 working tree; Python 3.11.15, numpy 1.26.3, scipy 1.16.3.

## What was changed

- **New** `src/simulation/bracket_topology.py`: `resolve_region_order`
  (played F4 games, else `seeds.f4_pairing`, never a default; raises
  `TopologyUnavailable`), `build_bracket_order(*, region_order)`,
  `derive_f4_region_pairing`, `f4_game_keys`, strict
  `picks_to_winners_by_round` / `picks_to_bool_vector` (raise
  `TopologyMismatch` instead of inventing winners), `region_order_from_first_round`.
- `scripts/mc_pool_backtest.py`: `build_bracket_order`, `build_first_round_matchups`
  (now one builder), `build_torvik_round_probabilities`, `build_base_from_ratings`,
  `build_pit_base` require `region_order`; `_run_one_year` threads its derived
  order into all 7 base builders and 12 `construct_bracket` calls;
  `_picks_dict_to_bool_array` is the strict projection.
- `src/optimization/bracket_construction.py`: `construct_bracket(region_order=)`;
  `by_region` is built in that order and every helper iterates it; F4 keys are
  `F4_<r0>_<r1>` / `F4_<r2>_<r3>`. `None` keeps the legacy layout for
  diagnostic callers only; the strict projection catches any production use.
- Threaded through `scripts/experiments/build_candidate_artifact.py`
  (+ `provenance.f4_pairing`), `src/evaluation/referee_audit.py`,
  `src/optimization/recency_hparam_fitter.py`, `scripts/generate_poolaware_bracket.py`,
  `src/cli/pool_cmds.py` (lenient builder/projection replaced by the shared
  strict ones; `_region_order()` resolves from data), `src/prediction/{torvik,massey_best,stacked}_probabilities.py`,
  `src/prediction/meta_selector.py`.
- `src/simulation/pool_competition.py:simulate_tournament_outcomes`: raises on
  a non-64 bracket and on a pair missing in both orientations (F3-3).
- `src/prediction/noseed_model.py:build_noseed_round_probabilities(*, as_of, window="recent")`
  (F3-4); all 9 callers updated.
- Data: `tournament_context_2026.json` seeds block gains
  `f4_pairing = [["West","Midwest"],["East","South"]]` (cross-checked against
  the played games by test); this field is REQUIRED for a prospective 2027 file.
- `src/data/strategy_cache.py:STRATEGY_CACHE_VERSION` 1 → 2.
- Tests (all fail on the old code): `tests/test_bracket_topology.py` (6),
  `tests/test_pool_competition_contract.py` (4),
  `tests/test_noseed_round_probabilities_walkforward.py` (2); 7 existing
  tests updated for the new signatures; one fixture that built a 2025
  bracket by file order now resolves play-ins.
- Artifacts rebuilt after code was verified: all 14 `candidates_*.json`
  (+ `.sha256`), all `docs/data/season_*.json`; README headline amended.

## Gate table

| Area | Path | Result | Evidence | Severity | Required action | Downstream impact |
|---|---|---|---|---|---|---|
| Propagation math: referee simulator vs independent recursion, noise 0 | 2, 3 | PASS | `propagation_audit_*`: Torvik/seed/pit/blend, 200k sims, min Bonferroni p ≥ 1.0, |z|>2 frac 2.9–4.4% | — | — | — |
| Propagation math: `marginals_from_pairwise` | 2, 3 | PASS | same, 10k sims, min Bonferroni p ≥ 0.55 | — | — | — |
| Marginal topology (`build_torvik_round_probabilities`, artifact sims) F3-2 | 2, 3 | FAIL → PASS | pre: PASS vs default tree / FAIL vs real (max z 11.9); post: PASS vs real (max z 2.76) / FAIL vs default | MATERIAL-LOCAL | done | all 14 artifacts and UI payloads rebuilt; 2026 shipped win-maximiser: F4 Duke/Florida → Michigan/Florida, champion Florida → Michigan, P(1st) 0.099 → 0.060 |
| Construction ↔ scoring topology F3-1 | 3 (and 2 via `_blend_region_bracket`) | FAIL → PASS | `topology_impact_pre_fix`: 5 Torvik + 4 blend seasons mis-scored; post: 0/15 | FOUNDATIONAL | done | every prior construction-mode backtest figure invalidated; headline rerun below |
| Picks projection silent fallback | 3, CLI | FAIL → PASS | strict projection; `test_picks_projection_roundtrip_and_strictness` | FOUNDATIONAL (same root) | done | — |
| Referee simulator silent 0.5 / non-64 F3-3 | 2, 3 | MINOR → hardened | `test_pool_competition_contract.py` | MINOR | done | none (no incomplete table in production) |
| Noseed marginals window/as_of F3-4 | 2, 3 | FAIL (leakage) → PASS | `test_noseed_round_probabilities_walkforward.py` | MATERIAL-LOCAL | done | blend candidates change; attributed rerun below |
| Referee cap [0.01, 0.99] binds for `pit` F3-5 | 3 | INDETERMINATE (design) | `propagation_audit_*`: 86 pairs capped, faithful to clipped table | — | Step 8 referee qualification | Step 2 §10 corrected |
| Seed-level recursion vs positional | 2, 3 | NON-ISSUE | max gap 1.9e-16 | — | — | — |
| Noise 0.16 Jensen shift | 2, 3 | NON-ISSUE here | L1 by round 0.09–0.12 → 0.01–0.02 | — | Step 8 | — |
| Noseed compounding heuristic R-1 | 2, 3 | INDETERMINATE (deferred) | max diff vs propagation 0.31 (R64), CHAMP mass 1.07 | research | post-retirement | none now |
| `meta_selector` alphabetical topology F3-6 | research only | MINOR (recorded) | code read | MINOR | none (not on any production path) | — |
| Prospective 2027 topology source | 2 | PASS | `f4_pairing` field + `TopologyUnavailable` guard | — | populate for 2027 on Selection Sunday | — |

## Headline rerun (results invalidated by F3-1 and rerun)

`python -m scripts.mc_pool_backtest --team-identity --opponent pool --n-opponents 29 --n-repeats 100 --modes seed meta_region_poolaware --no-log`

| run | seed P(1st) | meta_region_poolaware P(1st) ± 95% | edge | mean-rank paired |
|---|---|---|---|---|
| Step 2 (walk-forward referee, old topology) | 4.54% | 10.86 ± 2.90 | +6.3 pp | 14/14, t = 20.7 |
| topology fix only (F3-1/2/3) | 4.54% | 9.79 ± 3.01 | +5.3 pp | 14/14, t = 19.3 |
| topology + noseed window (F3-4) — **current code** | 4.54% | 10.93 ± 2.94 | +6.4 pp | 14/14, t = 10.2 |

Per-season P(1st) moved by up to −0.07 (2013) under the topology fix and by
+0.13 (2011) under the noseed window change; both are inside season-level
noise (n = 100 repeats) and the aggregate is unchanged within its CI. The
seed baseline is topology-invariant by construction and is unchanged.

## Final report

1. **Foundational defects found:** F3-1 — construction and scoring walked
   different Final Four pairings and the projection silently reconciled
   them; in 5 of 15 seasons the backtest scored a champion construction
   never chose. Root cause shared with F3-2 (marginals and the shipped
   artifact on the wrong tree).
2. **Defects fixed:** F3-1, F3-2, F3-3, F3-4 — one topology module, no
   silent default, strict projection, hardened simulator, walk-forward
   noseed window; regression tests for each; artifacts rebuilt afterwards.
3. **Results invalidated / rerun:** every construction-mode backtest number
   to date; the headline was rerun twice (attributed). The shipped 2026
   artifact and all 13 historical artifacts were rebuilt; the shipped
   win-maximiser's champion changed. The referee-suite, leave-one-referee-out,
   Romano–Wolf and 79-mode figures remain stale (Steps 7–9).
4. **Assumptions explicitly accepted:** A-6 the referee's [0.01, 0.99] cap
   is a referee design choice, judged in Step 8; A-7 the legacy
   `region_order=None` fallback in `construct_bracket` exists only for
   diagnostics and is fenced by the strict projection.
5. **Questions deferred:** R-1 (heuristic noseed marginals); F3-5 (cap) to
   Step 8; F3-6 (`meta_selector` topology) research-only cleanup; the
   `.sha256` timestamp pin (Step 1 note).
6. **Step 3:** **PASS** — propagation is exact against an independent
   reference on every live table; the topology defect that would have
   failed it is repaired and re-measured.
7. **Safe to proceed to Step 4:** yes.

## Verification commands

```bash
PYTHONPATH=. python3 -m pytest -p no:asyncio tests/test_bracket_topology.py tests/test_pool_competition_contract.py \
  tests/test_noseed_round_probabilities_walkforward.py tests/test_pairwise_contract.py tests/test_referee_audit.py \
  tests/test_strategy_cache_integrity.py tests/test_candidate_artifact_frequencies.py -q
PYTHONPATH=. python3 artifacts/methodology_audit/step3/propagation_audit.py --post-fix
PYTHONPATH=. python3 artifacts/methodology_audit/step3/topology_impact.py --post-fix   # expect 0 mismatches
```
