# Pre-registration — recent-regime, regular-season-trained tournament model (Steps 10–12)

Written 2026-09-15, after Step 9 closed ROBUST and before any number in this
experiment exists. Nothing in Steps 1–9 is reopened by this document. The
production strategy and its Step 9 result are preserved unchanged.

## Question

Can a tournament win-probability model trained on REGULAR-SEASON games
(point-in-time features, ~41k rows) be recency-weighted or upset-weighted so
that it predicts the most recent tournaments better than the production
model, without fooling ourselves on a 3-season window?

Two questions are kept separate: the 14-season validation (answered:
robust) and this regime question (open).

## Populations (fixed)

- Training: `docs/data/training_pit.json` — regular-season and conference
  games, boundaries day 28..126 step 7, every feature from games strictly
  before the boundary, Torvik snapshot strictly before the boundary,
  standardised within (season, boundary). Rebuilt 2026-09-15 with the
  Kaggle-name alias repair (25 small-school ids), which had dropped 114 of
  948 tournament games (42% of 16-seed games) from the evaluation set.
- Evaluation: `docs/data/eval_pit_tournament.json` — NCAA tournament games
  at the Selection Sunday boundary (day 133), same code path. Coverage after
  repair is reported per season before any model is scored.
- Seasons: training 2010–2025; evaluation windows below. 2026 is NOT in the
  PIT matrices (no Kaggle 2026 regular season) — limitation, stated.
- Features: the 27 PIT keys minus the three venue terms (`venue_home`,
  `venue_host_city`, `venue_travel`), so the model is venue-free and the
  tournament is neutral by construction; `t_rank` is unavailable in the PIT
  matrix (no dated snapshot), which is one known reason the regular-season
  population scores worse than the tournament-trained model.

## Windows (fixed)

- **Recent regime (primary):** tournaments 2023, 2024, 2025 (n = 3 seasons,
  ≈ 189 games). Each is predicted from a model fit on training rows of
  seasons strictly before it.
- **Tuning window (for choosing among variants):** tournaments 2014–2022
  (n = 8 seasons excl. 2020), walk-forward. Any choice among variants is made
  here, frozen, then applied once to the recent regime.
- **Safety screen:** the same 2014–2022 window; a variant must not be
  materially worse there.

## Models (fixed)

- M0 reference: the production `pit` model (`pairwise_for_year`:
  tournament-trained ridge, 11 features, causal Student-t link), scored on
  the same evaluation games. It is the bar.
- M1: ridge on regular-season rows, equal weights, all prior seasons.
- M2(ρ): as M1 with season weights ρ^(Y−1−s), ρ ∈ {0.85, 0.70}.
- M3: as M1 restricted to the three seasons before Y.
- MU(w): M1 with weight w on training rows where the LOWER-rated team (by
  dated barthag at the boundary — model-implied favourite, not the result)
  won, w ∈ {2, 4}. Combined with the winning ρ only if M2 was chosen.
- Every model: margin target, same Student-t link fitted causally on prior
  out-of-sample rows (the Step 2 harness), ridge λ = 1.0 per 1,000 rows.
  No other hyperparameter is searched.

## Metrics (fixed)

Primary: mean log loss of P(team1 wins) on evaluation games. Secondary:
Brier; calibration slope (logistic recalibration slope of the logit, target
1); **upset subset** = evaluation games the lower seed won, and **chalk
subset** = the rest, each with its own log loss; upset-call rate (share of
upset games where the model favoured the winner).

## Inference unit (fixed)

Seasons. Tuning-window comparisons: paired season bootstrap (5,000, seed
42) over the 8 seasons. Recent regime: n = 3, so the CI is reported but the
adoption gate additionally requires a game-level paired bootstrap
CLUSTERED BY SEASON (resample seasons, keep all games within) — which is
the same thing at n = 3 and is honest about it: three seasons cannot
resolve log-loss differences below roughly 0.02.

## Adoption gate (fixed, Step 17 input)

A variant replaces M0 for the recent regime ONLY if all hold:
1. Tuning window: mean ΔLL vs M0 ≤ 0 with the paired CI not entirely above 0
   (not materially worse historically).
2. Recent regime: mean ΔLL vs M0 < 0 AND the 3-season paired CI entirely
   below 0.
3. Upset subset LL not worse than M0 by more than 0.01 AND chalk subset LL
   not worse by more than 0.01 (no hedging trade).
4. Calibration slope in [0.8, 1.2] on the recent regime.
Otherwise the result is a documented negative and the existing methodology
is retained.

Expected outcome, stated now: with three seasons the most likely verdict is
INDETERMINATE or negative. That is an acceptable, publishable result.

## What will NOT be done

No model-family search; no feature selection; no λ search beyond the fixed
value; no change to candidate generation, opponents, referees, scoring or
the production strategy; no look at the recent-regime numbers before the
tuning-window choice is frozen (the script computes them in that order and
writes the frozen choice before scoring 2023–2025).

## Exact definitions (added before execution, 2026-09-15; frozen)

- **Paired difference sign:** ΔLL = LL(variant) − LL(M0), computed per
  season as the difference of that season's mean per-game log losses.
  Negative = variant better. Every gate is stated in this sign.
- **Log loss per game:** −[w·ln p + (1−w)·ln(1−p)] with p = P(team1 wins),
  p clipped to [1e-6, 1−1e-6]; team1 is the lexicographically smaller id.
- **Tolerance:** the 0.01 in gate 3 is an ABSOLUTE difference in log-loss
  units: gate 3 requires mean-over-recent-seasons ΔLL_upset ≤ +0.01 AND
  ΔLL_chalk ≤ +0.01. It is not a relative or percentage tolerance.
- **Bootstrap:** paired season bootstrap: resample the season list with
  replacement (n = 8 for the tuning window, n = 3 for the recent regime),
  5,000 resamples, seed 42; statistic = mean over resampled seasons of the
  per-season ΔLL; CI = percentiles 2.5 and 97.5. "Entirely below zero" means
  the 97.5th percentile < 0. "Not entirely above zero" means the 2.5th
  percentile ≤ 0.
- **Calibration slope:** the coefficient b in the logistic regression
  w ~ a + b·logit(p) fitted by maximum likelihood on the pooled recent-regime
  evaluation games (all 2023–2025 games, First Four included). Target 1;
  gate 4 requires 0.8 ≤ b ≤ 1.2.
- **First Four games:** INCLUDED in the overall log loss, Brier and
  calibration slope (they are tournament games predicted at the Selection
  Sunday boundary); EXCLUDED from the upset and chalk subsets, which are
  defined by seed and First Four teams share a seed. 41 such rows.
- **Upset subset (evaluation):** games whose winner had the numerically
  larger seed; **chalk subset:** games whose winner had the smaller seed;
  same-seed games in neither. Seeds are the Selection Sunday seeds
  (Kaggle MNCAATourneySeeds), independent of every model.
- **Upset-sensitive training weight (MU):** a training row gets weight w if
  the team with the LOWER dated barthag at the row's boundary won the game;
  the favourite is defined by the snapshot, not by the result, so the weight
  is a function of pre-game information and the outcome only through the
  ordinary target.
- **Ties:** rows with margin 0 are excluded from log loss; none exist.
- **Season sets:** tuning = {2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022};
  recent regime = {2023, 2024, 2025}. Training rows for evaluation season Y
  = all regular-season rows with season < Y (2010 onward).
- **Selection among variants (frozen procedure):** on the tuning window,
  among variants passing gate 1, choose the one with the lowest mean ΔLL vs
  M0 (ties → the simpler model in the order M1, M2(0.85), M2(0.70), M3,
  MU(2), MU(4), M2×MU). The chosen variant is written to disk before any
  recent-regime number is computed, then gates 2–4 are applied to it alone.
  All variants' recent-regime numbers are reported for transparency but do
  not enter the decision.
