# O2a — Four-Factor validation FAIL: structural box-score adapter bug

**Date:** 2026-04-13
**Branch:** `claude/simplify-repo-structure-keQXE`
**Evidence files:**
- `artifacts/o2_ff_validation_2026-04-13.log` — full per-year `validate_four_factors.py` output
- `artifacts/o2_ff_validation_2026-04-13.json` — machine-readable per-year/per-feature/per-stratum metrics
- This file — diagnostic narrowing

## Verdict

**FAIL.** All 17 production years (2008–2025 ex 2020) × all 8 features fail the
`r ≥ 0.99` gate. The bias-residual gate (`|mean_residual| ≤ 0.01` per stratum)
also fires across every season. Severity spans `r ≈ 0.11` (2024 Opp eFG%) to
`r ≈ 0.83` (2008 Opp TO%) — never close to passing.

This is not marginal calibration drift. It is a structural mismatch between
`scripts/validate_four_factors.py::compute_boxscore_ff()` (which feeds
`ProprietaryMetricsEngine` over `team_games_to_game_records()`) and Torvik's
published per-season Four Factors at `data/raw/historical/torvik_{year}.json`.

## What rules out

- **Not a raw-vs-adjusted mismatch.** Torvik's published values look raw
  (matching Oliver/Kubatko formulas in the dictionary headers). The
  exploration agent's read on this remains correct.
- **Not a TO%-denominator-only bug.** TO% is consistently underpredicted by
  ~0.025–0.04 (which would be Oliver's "true possessions" subtracting ORB
  from the denominator), but eFG% — which has no possessions denominator —
  is *also* failing dramatically.

## What the bug looks like (smoking gun)

For 2024 tournament 1- and 2-seeds, local-vs-Torvik eFG% diverges by
12–20 percentage points in the wrong direction:

| team | local eFG% | torvik eFG% | local opp eFG% | torvik opp eFG% |
|---|---|---|---|---|
| vermont   | 0.3182 | 0.5180 | 0.6603 | 0.4670 |
| florida   | 0.3361 | 0.5150 | 0.5813 | 0.4880 |
| illinois  | 0.3750 | 0.5380 | 0.6267 | 0.4800 |
| tennessee | 0.3689 | 0.5150 | 0.7619 | 0.4540 |
| kansas    | 0.4018 | 0.5340 | 0.5000 | 0.4760 |

Local "team" eFG% is in the 0.32–0.40 range — **impossible** for top tournament
teams (their real offensive eFG% is 0.51–0.54). Local "opp" eFG% is in the
0.50–0.76 range — also impossible for elite defenses (their real opp eFG% is
0.45–0.49).

`local_efg + local_opp_efg ≈ 0.98` for several teams (Vermont 0.98,
Tennessee 1.13). That is consistent with a **per-game double-counting or
perspective-mixing bug** in `team_games_to_game_records` or
`ProprietaryMetricsEngine._four_factors`, not a simple inversion (a clean
inversion would still produce per-team eFG% in legitimate ranges).

For mid-major teams (`abilene_christian`, `air_force`, `akron`, `alabama_a_m`),
the local values are *close* to Torvik (eFG% within 0.001–0.016). The bug
appears to be selectively affecting teams with certain box-score patterns —
possibly heavy schedule-strength teams, teams with many games against weak
opponents, or teams with unusual home/away splits.

## Why simple fixes won't close O2

Two hypotheses tested-and-falsified during this pass:

1. **Inversion (team1/team2 swap).** Falsified — local opp eFG% values
   (0.50–0.76) are not in the legitimate "team eFG%" range either, and
   `local + local_opp ≈ 0.98–1.13` is incompatible with a clean swap.
2. **TO% denominator difference.** Real and contributory, but eFG% has no
   denominator, so it can't be the only bug. Fixing TO% denominator would
   not move eFG% from 0.32 to 0.51 for Vermont.

The actual root cause requires reading
`src/data/features/proprietary_metrics.py::ProprietaryMetricsEngine.compute()`,
`team_games_to_game_records()`, and the `historical_games_{year}.json`
schema — not in scope for this closure pass.

## What this means for the locked decisions

`MEMORY.md §1` constants table includes `Four Factors weights [0.40, 0.25,
0.20, 0.15] (Oliver 2004 / Kubatko 2007)`. Those weights are downstream
consumers of the Four Factor *values*. If the values are systematically
wrong for top-seeded teams, every metric using `_four_factors` (including
the Pythagorean expectation `_box_score_xp` at `proprietary_metrics.py:879`)
is mis-scoring elite teams. This intersects:

- `MEMORY.md §2 D8` (pool-value contrarian strategy lost 7.1% vs chalk).
  If chalk teams are being scored with biased Four Factors, the
  contrarian-loss diagnosis may be partially confounded.
- `COUNCIL_LESSONS §3 row 21` (barthag re-computed locally only r=0.73).
  Same defect class.

Recommend the council prioritize a focused root-cause session on the
box-score adapter before re-running any Four-Factor-dependent backtest.

## Required to close O2

1. Identify the per-game aggregation bug in `compute_boxscore_ff` /
   `team_games_to_game_records` / `ProprietaryMetricsEngine._four_factors`.
2. Fix it.
3. Re-run `python scripts/validate_four_factors.py`.
4. Confirm `r ≥ 0.99` per season per feature AND `bias_flags == []`.
5. Wrap as `tests/test_validate_four_factors.py` (the original O2 closure
   plan) and lock in `MEMORY.md §1`.

Status: **open**, blocked on a focused root-cause investigation.
