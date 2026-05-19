# Session 2026-05-18 — Poolaware Architecture Exhaustion + 2027 Prep

**Date:** 2026-05-18
**Starting baseline:** 11.9% P(1st) via `meta_region_poolaware`
**Ending baseline:** 11.9% P(1st) — unchanged
**Conclusion:** All remaining poolaware architecture directions are exhausted. 11.9% is the ceiling for this design.

---

## Experiments Attempted This Session

### Change 3 — Real pool-history opponent model (KILLED, -5.6pp)
- Built `build_pool_history_opponent_matrix()` using 2023-2026 real pool brackets
- Seed-walk transfer: encode picks as seed numbers, remap to test-year teams
- **Result: 6.27% P(1st) vs 11.9% baseline — severe regression**
- Root causes: (1) LOYO logic returned pool brackets from 2023-2026 as opponents for ALL years including 2011-2022 (anachronistic data), (2) seed-walk translation is too lossy even for contemporaneous years (2023-2026 still dropped to ~9.5%)
- **DO NOT RETRY:** ESPN national pick distribution is a better opponent model than translated real brackets

### Change 4 — Extra torvik/massey blend ratios (KILLED, null)
- Added `tv_mass70/60/50` (70/30, 60/40, 50/50 blends) at risk=0.3 and risk=0.7
- Grew candidate pool from ~25 to ~29 (6 new pre-dedup candidates)
- **Result: 11.07% P(1st) — none of the new candidates selected in any of 15 years**
- `tv_mass80` already captures the torvik/massey continuum efficiently
- **DO NOT RETRY:** More blend ratios = more noise, not more signal

### Change 5 — Denser risk grid (9 levels: 0.1–0.9) (KILLED, null)
- Expanded `_pa_risk_levels` from (0.1, 0.3, 0.5, 0.7, 0.9) to all 0.1 steps
- Grew candidate pool from ~25 to ~39
- **Result: 11.60% vs 11.07% comparison baseline — null within MC variance**
- New risk levels DO get selected 5/15 years (unlike blend ratios), but outcomes mixed: ~3 better, ~2 worse, ~1 same
- **DO NOT RETRY with same approach:** Selection noise swamps structural signal at these margins

### Change 6 — Per-region independent bracket construction (KILLED, null)
- Added `_region_top_n_per_region_construction()` — 1 "odd" region × 3 chalk regions
- Variants: v1 (torvik/0.9), v2 (massey_avg/0.9), v4 (massey_avg/0.1) × 4 positions = 12 new candidates per year
- Smoke test confirmed 9-12 genuinely unique new brackets per year
- **Result: 11.33% vs 11.07% — per-region candidates selected 0/15 years**
- Root cause: existing construction already achieves region-level differentiation via risk_level weighting; per-region parameterization is structurally novel on paper but not selectively better
- **DO NOT RETRY this family**

### Change 7 — Dynamic risk calibration via R64 chalk score (PLANNING ONLY, CLOSED)
- Hypothesis: fix risk levels per year based on how chalk-heavy ESPN public picks are
- R64 chalk score = `mean(max(pick_pct_A, pick_pct_B))` across 32 R64 games, range 0.77–0.83
- LOYO classification: 11/12 classifiable years correctly predicted winning risk direction
- **Finding: signal is real but architecturally unactionable**
  - Step (c) exhaustive wins only 1/15 years → changing its grid has zero impact on 14/15 years
  - Step (b) path: would require removing risk=0.1 in HEAVY years, but 2016 (1 miss: classified HEAVY, winner=risk=0.1) would regress
  - Addition path: follows established null/regression pattern
- Root cause: poolaware already generates candidates at ALL 5 risk levels every year; the chalk signal can't improve on the selector that already has access to all options
- **DO NOT RETRY this family**

---

## What Was NOT Tried This Session

- Walk-forward Massey best-system: implemented in prior sessions, wired in, never selected (structurally similar to massey_avg). Not re-tested.

---

## Key Lessons Crystallized

1. **More candidates = more selection noise.** Every experiment that added candidates (without removing existing ones) produced null or regression. The ~25-candidate pool is calibrated to the 200-trial inner budget.

2. **A candidate that the selector never picks adds pure noise.** If a new bracket type is structurally novel but doesn't win the inner selection in any historical year, it dilutes the budget across years when the selector has to evaluate it.

3. **The EV scorer already achieves implicit region-level differentiation.** Explicit per-region parameterization is redundant.

4. **Real pool brackets are not a better opponent model.** Seed-walk translation is too lossy. ESPN national distributions are better than translated real brackets at predicting field behavior.

5. **The chalk signal is real but unactionable** given the current fixed-candidate-pool architecture.

---

## 2027 Prep Completed

- **RUNBOOK_2027.md corrected:** baseline 11.2% → 11.9% (with p-value), candidate count ~25 → ~25-35
- **Smoke test verified (2026):** `python -m scripts.mc_pool_backtest --modes meta_region_poolaware --opponent pool --team-identity --n-opponents 30 --n-repeats 10 --n-model 10 --years 2026` — selected `blend_region_risk=0.7` (best of 25, inner P1=0.130), no errors

---

## Next Session Priority

**Kaggle BSS path** — currently at BSS +0.060 ensemble baseline. Genuinely different objective (Brier-scored), different levers, clear improvement path. See `docs/kaggle-objective-policy.md` and `RUNBOOK_2027.md` for baseline command.
