# Steps 10–12 — Recent-regime / upset-sensitive experiment: result

Pre-registration: `PREREGISTRATION_RECENT_REGIME.md` (frozen, sha256
fc7bc93b…, read-only) written before any number existed; gate definitions
printed by the script before computation. Script:
`recent_regime_experiment.py`; outputs `results.txt`, `results.json`,
`frozen_choice.json` (written before the recent regime was scored).

## Data used

- Training: regular-season point-in-time matrix, 48,172 rows, 2010–2025,
  24 non-venue features; every row's game on or after its boundary, no
  tournament rows (mechanically verified).
- Evaluation: 970 tournament games at the Selection Sunday boundary
  (2010–2025; 41 First Four rows; 0 seed-lookup failures after the alias
  repair). Subsets: 273 upset games, 639 chalk games, 58 same-seed/First Four
  excluded from subsets.
- Reference M0 = production `pit` model on the same games (one id alias,
  massachusetts→umass, needed for 2014).

## Tuning window 2014–2022 (gate 1)

| model | mean LL | ΔLL vs M0 [95% CI, paired season bootstrap] | gate 1 |
|---|---|---|---|
| M0 production pit | 0.4814 | — | — |
| M1 equal weights | 0.5617 | +0.080 [+0.052, +0.110] | FAIL |
| M2 ρ=0.85 | 0.5623 | +0.081 [+0.053, +0.111] | FAIL |
| M2 ρ=0.70 | 0.5631 | +0.082 [+0.054, +0.112] | FAIL |
| M3 last 3 seasons | 0.5640 | +0.083 [+0.055, +0.114] | FAIL |
| MU w=2 | 0.5671 | +0.086 [+0.058, +0.116] | FAIL |
| MU w=4 | 0.5829 | +0.102 [+0.072, +0.134] | FAIL |

No variant passed gate 1: every regular-season-trained model is worse than
the tournament-trained production model by about 0.08 log-loss units, with
every CI entirely above zero. **Frozen choice: none.** The recent regime
was therefore scored for transparency only; no decision rests on it.

## Recent regime 2023–2025 (reported, not decided on)

M0: LL 0.4468 (upset subset 0.936, chalk subset 0.264, calibration slope
1.11). Every variant is worse by +0.119 to +0.128 [CIs +0.07 to +0.18],
calibration slopes 1.04–1.13. The only subset improvement anywhere is
MU(4) on upsets, −0.010, bought with +0.152 on chalk games — exactly the
hedging trade gate 3 was written to reject.

## Verdict

**Negative result under the frozen protocol.** The hypothesis that a
regular-season-trained model, recency-weighted or upset-weighted, would
predict recent tournaments better than the production model is not
supported; the direction is the opposite and the effect is large relative
to the gate. Consistent with the earlier finding that pooling
regular-season rows measured null on the tournament matrix, and with the
two known handicaps of the regular-season population (no `t_rank`
snapshot; standardisation over the whole D1 field rather than the
tournament field). The existing methodology is retained.

## What this does and does not say

- It says: on a leakage-free, pre-registered test, none of the six
  registered variants clears even the historical-safety gate.
- It does not say: that recency or upset-sensitivity could never help in a
  tournament-trained model (not tested; would need a new pre-registration),
  nor anything about 2026 (absent from the PIT matrices).

## Follow-ups (not actions)

R-4: if the recent-regime question is pursued again, test recency/upset
weights inside the tournament-trained population (1,008 rows) with the same
gates; expect the 3-season window to be INDETERMINATE by construction.
