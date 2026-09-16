# R-4 — Training weights inside the tournament-trained PIT model: result

Pre-registration `PREREGISTRATION_R4.md` (sha256 0d73668e…, read-only), gates
identical to Steps 10–12. Script `r4_experiment.py`; outputs `results.txt`,
`results.json`, `frozen_choice.json` (written before the recent regime was
scored). The reference M0 reproduces the production walk-forward
probabilities to 2e-14.

## Tuning window 2014–2022 (gate 1: mean ΔLL ≤ 0 and 2.5th pct ≤ 0)

| variant | ΔLL vs production [95% CI] | gate 1 |
|---|---|---|
| M2 ρ=0.85 | +0.0001 [−0.0033, +0.0025] | FAIL (mean > 0 by 1e-4) |
| M2 ρ=0.70 | +0.0017 [−0.0054, +0.0072] | FAIL |
| M3 last 3 seasons | +0.0059 [−0.0063, +0.0174] | FAIL |
| MU w=2 | +0.0098 [−0.0005, +0.0207] | FAIL |
| MU w=4 | +0.0524 [+0.0261, +0.0826] | FAIL |

Frozen choice: none. Recent regime scored for transparency only.

## Recent regime 2023–2025 (transparency)

Production: LL 0.4242, upset subset 0.925, chalk subset 0.280, calibration
slope 1.33. Every variant is worse on overall log loss (+0.004 to +0.080,
four of five CIs entirely above zero). Upset weighting improves the upset
subset (−0.12 at w=2, −0.28 at w=4) by degrading the chalk subset (+0.05,
+0.18): the hedging trade, again, and now inside the tournament population.
Mild recency (ρ=0.85) is indistinguishable from the production model
everywhere, including 2026 (−0.003).

## Verdict

**Negative under the frozen protocol.** Reweighting the tournament-trained
model's training rows does not improve recent tournament prediction; mild
recency is a wash and everything stronger, and every upset weight, is worse
historically and recently. The existing methodology is retained.

## Observation recorded, not acted on

The production model's own calibration slope on 2023–2025 is 1.33 (it is
under-confident on the recent regime; slope > 1 means its logits are too
compressed). It would not itself pass gate 4, which shows the gate was set
tighter than the incumbent satisfies — a limitation of the pre-registered
design, stated here rather than used to reinterpret any result. Whether the
link's shrinkage toward a=1 is too strong for recent seasons is a
legitimate future question (R-5); it is not a weight question and was not
tested.

## Closing the modelling question

Steps 10–12 (regular-season population) and R-4 (tournament population)
together reject the recency/upset-weighting hypothesis on both available
training populations under one frozen, leakage-free protocol. No threshold
was changed after any result. The 3-season window never became the basis of
a decision because no variant survived the historical gate in either test.
