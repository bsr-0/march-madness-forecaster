# Preregistration — Model sensitivity (Explore, Phase B)

Registered 2026-09-16, before implementation. This is a UI/explanation
feature. It does not touch the validated production model, the candidate
artifact, selection, P(1st), EV, or any artifact under `artifacts/`. Nothing
computed here is written to disk.

## Question

Two separate questions, reported separately and never merged:

1. **Local sensitivity** — for one matchup on the fitted bracket, what
   probability does a model refit *without* variable X_j assign, compared with
   the full model?
2. **Historical sensitivity** — across the same held-out seasons the full
   model is evaluated on, how does predictive performance change when X_j is
   excluded and the model is refit?

## Definition, per variable X_j in the canonical set

1. Baseline: the full model, fitted by the exact existing procedure —
   `fitLinear(rows, cols, asOf)` on rows with `y < asOf`, ridge as shipped,
   no intercept; link calibrated by `causalWalkForward(rows, cols, years,
   asOf, 2014)`.
2. Exclusion model: `fitLinear(rows, cols \ {j}, asOf)`. **All remaining
   coefficients are refit.** The coefficient is not set to zero.
3. Same training boundary (`y < asOf`), same held-out folds (`2014 ≤ y <
   asOf`, each fitted on strictly earlier seasons), same ridge, same link.
4. The exclusion model's link is calibrated by the same procedure on its own
   held-out predictions (`causalWalkForward` with `cols \ {j}`), including
   the same shrinkage of `a` toward 1.
5. Local: for matchup (a, b), `p_full = winProbFromMargin(margin_full,
   σ_full, cal_full)` and `p_excl = winProbFromMargin(margin_excl, σ_excl,
   cal_excl)` on the same standardised differential vector restricted to
   `cols \ {j}`.
6. Historical: the exclusion model's held-out accuracy and log loss from
   step 4, beside the full model's on the identical pooled held-out games;
   Δ log loss = excl − full (positive = worse without the variable).
7. Picks changed: the bracket is re-solved end to end under the exclusion
   model with the same tie rule as `solveByFit()`, and the count of the 63
   slots whose winner differs is reported. It is a count, not a bracket:
   no board, no P(1st), no EV.

## Wording

- Each local figure is phrased **"Probability under a refit excluding X: 58.9%"**
  (or in a table headed exactly that). Never "contribution", "effect",
  "importance", "explains", "accounts for", or "percentage points from X".
- The historical figures are phrased as performance *without* the variable,
  with the Δ log loss signed and its direction stated.
- Every sensitivity panel carries: *"Excluding a variable lets the ones that
  move with it absorb it. A small change does not mean the information is
  unimportant; a large change does not make the variable a cause."* Where the
  variable has a collinear partner (r ≥ 0.8, the equation's existing
  threshold), the partner is named in that sentence.

## Boundaries

- Reads `state.training`, `state.season.z`, `state.fit`. Writes nothing
  except a per-session cache keyed by season.
- Not consulted by `strategyRows()`, `renderHeadline()`, `renderCompare()`,
  the board, `picksAsText()`, `fittedEval()`, or any Python script.
- Shown only under the Fitted strategy, inside the Explore panel.

## Tests to ship with it

- Exclusion is a refit: with correlated columns, the remaining coefficients
  differ from the full model's (zeroing would leave them unchanged).
- Absorption is real: with a duplicated column, excluding one copy leaves
  predicted margins essentially unchanged.
- The training boundary and fold set are those of the full model
  (`asOf` respected; no row with `y ≥ asOf` in any fit).
- The historical figures for the *full* model reported in the panel are the
  same objects as `state.fit.oos` (no second baseline).
- The source of `app.js` contains none of the forbidden words in the
  sensitivity panel's user-facing strings.

## Result note (2026-09-16, after implementation; definition above unchanged)

Historical sensitivity on the shipped matrix, displayed season 2026 (11
held-out seasons, 693 games; full model 77.9% / log loss 0.4505):

| refit excluding | held-out right | Δ log loss |
|---|---|---|
| Overall rating | 72.6% | +0.085 |
| National rank | 73.0% | +0.090 |
| both of the above | 73.0% | +0.090 |
| Defense | 75% | +0.042 |
| every other canonical variable | 78–79% | within ±0.005 |

Overall rating and National rank correlate at 0.986, so absorption would
predict a small change on excluding either. The opposite was measured:
excluding either costs as much as excluding both, and the nine remaining
variables plus the explicit difference (Overall rating − National rank)
recover the full model (0.452). The pair behaves as one feature — a rating
relative to its rank — that neither carries alone. This is recorded as a
description of the fitted model; it is not a proposal to add a difference
feature, which would be a change to the frozen production model and is out of
scope. The panel's collinearity sentence is conditional on the measured Δ so it
does not assert absorption where none occurred.
