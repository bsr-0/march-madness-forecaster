# Pre-registration R-4 — training weights inside the tournament-trained PIT model

Written 2026-09-15 after Steps 10–12 closed negative, before any R-4 number exists.
Question: does changing the TRAINING WEIGHTS of the existing tournament-trained PIT
model (11 canonical features, no intercept, ridge λ = 1 per 1,000 rows, causal Student-t
link shrunk n/(n+63)) improve recent tournament prediction without sacrificing historical
performance or trading chalk accuracy for upset accuracy? Nothing else changes: no
features, no λ, no architecture, no candidate/referee/optimizer change.

Population: `docs/data/training.json` (1,008 tournament games, 2010–2026, R64 onward;
rows oriented by pre-tournament fact). Walk-forward: season Y predicted from rows with
season < Y. Reference M0 = equal weights (the production model, reproduced by the same
code path so the comparison is exact).

Variants (fixed): M2(ρ) season weights ρ^(Y−1−s), ρ ∈ {0.85, 0.70}; M3 = rows from the
three seasons before Y only; MU(w) weight w ∈ {2, 4} on rows where the LOWER dated-barthag
team (sign of the barthag z-difference at the boundary, pre-tournament) won.

Windows: tuning 2014–2022 (8 seasons, ex-2020); recent 2023–2025 (3 seasons).
2026 is present in this matrix and is reported as a single extra season for
transparency only; it is not part of any gate (it is the integration season).

Gates: identical to PREREGISTRATION_RECENT_REGIME.md §"Exact definitions": ΔLL =
LL(variant) − LL(M0) per season; gate 1 tuning mean ≤ 0 and 2.5th pct ≤ 0; gate 2 recent
mean < 0 and 97.5th pct < 0; gate 3 ΔLL_upset ≤ +0.01 and ΔLL_chalk ≤ +0.01 in ABSOLUTE
log-loss units; gate 4 calibration slope in [0.8, 1.2]; paired season bootstrap, 5,000
resamples, seed 42. Selection among variants frozen on the tuning window before the
recent regime is scored.

Definition change from Steps 10–12, stated now: this matrix has no seeds, so the UPSET
subset = evaluation games won by the lower dated-barthag team; CHALK subset = games won
by the higher; rows with a zero barthag difference in neither. Ties (margin 0): none.
Expected outcome: INDETERMINATE or negative; a 3-season CI that does not establish
improvement is not a reason to loosen any gate.
