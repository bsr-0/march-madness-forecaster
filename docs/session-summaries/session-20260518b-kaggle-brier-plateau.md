# Session Summary: Kaggle Brier Plateau Diagnosis (2026-05-18)

## Session Goal

Continue Kaggle BSS improvement work. Attempted line movement as a new feature;
exhaustively diagnosed why the current architecture has plateaued at BSS +0.133.

## What Was Built

### Opening Line Artifact

**`scripts/build_tournament_opening_artifact.py`** (new) — scrapes day-before Odds API
snapshots to find tournament opening probabilities:
- Prefers 2300Z snapshot (6pm ET, day before tip), falls back to 1700Z
- Primary match by event_id (stable UUID), secondary by normalized team names
- Coverage: 363/364 tournament games across 2021-2026 (99.7%)
- Output: `artifacts/ncaa_tournament_opening.json`

### Market Movement Feature

**`src/prediction/torvik_correction.py`** — added `market_movement` as the 9th feature:
- `_feature_vector` now takes `market_movement=0.0`
- `fit()`, `predict()`, `predict_one()` all updated with `movement_vals` parameter
- `fit_torvik_correction_from_year_records()` reads `row.get("market_movement") or 0.0`

**`scripts/loyo_pergame_predictions.py`** — populates `market_movement` field:
- Added `TOURNAMENT_OPENING_PATH` constant
- Added `_load_opening_market_lookup()` function
- `generate_year()` accepts `opening_market` parameter; computes `closing - opening`
- `generate_all()` loads both opening and closing lookups, passes to `generate_year()`
- Regenerated: `artifacts/loyo_pergame_predictions.json` (21 years, 1323 games)

**`scripts/admit_kaggle_candidate.py`** — added `torvik_corrected_movement` mode:
- `_evaluate_torvik_corrected_movement()` function
- Dispatch in `evaluate_spec()`
- Added to `_MODES` list

## Experiments Run

### odds_api (carried over from prior session)
- Outcome: **FAIL** (+0.0008 improvement, gate requires +0.002)
- Root cause: existing `odds` BT signal already captures multi-book consensus

### torvik_corrected_movement
- Movement = closing_prob - opening_prob
- Residual correlation after torvik: **-0.002** (essentially zero)
- Gate improvement: **0.0000**
- Root cause: 90% of games have movement < 0.02 (flat lines); remaining signal
  absorbed by market_prob feature

### closing_market_blend (re-tested)
- Best alpha=0.1 → mean=0.1304 (+0.0001 vs incumbent 0.1305)
- 2025 hurt by blending, 2026 helped equally → net near-zero

### Max_correction / ridge / recency exhaustive grid search
- Best: `max_correction=0.10, ridge=80, recent_year_count=7` → mean=0.1293 (+0.0012)
- Cannot reach +0.002 threshold

### torvik_isotonic
- Two-stage isotonic + linear correction
- 2025: 0.0872 (massive overfit), 2026: 0.1849 → mean=0.1360 (much worse)

### Intercept ridge penalization
- Shrank intercept from +0.318 to +0.156 with `intercept_ridge=100`
- 2026 Brier: unchanged at 0.1502
- Root cause: `max_correction=0.06` bound drives corrections to ceiling regardless

## Root Cause: 2026 Structural Anomaly

The correction model has a **+0.318 intercept** (systematic upward bias) because all
training years (2009-2024) have positive mean residuals — torvik under-predicted team1.

2026 broke this pattern:

| Year | team1 win rate | torvik mean | mean_residual |
|------|---------------|-------------|---------------|
| 2023 | 80.9% | 65.9% | +0.151 |
| 2024 | 76.2% | 70.2% | +0.059 |
| 2025 | 88.9% | 71.9% | +0.169 |
| 2026 | 73.0% | 74.2% | **-0.012** |

2026 was an upset-heavy year where torvik **over-predicted** team1. The correction
model applies positive corrections (learned from 2009-2024) that are systematically
wrong for 2026. 55/63 2026 games got a positive correction; only 38/63 (60%) were
in the correct direction.

Every parameter change that helps 2025 hurts 2026 equally or more. The fundamental
tradeoff is unresolvable without a mechanism to detect "upset-heavy" years at
prediction time — and the available market signals (odds, elo) also over-predict
team1 in 2026.

## Source Analysis for Final Holdout (2025, 2026)

| Source | 2025 BSS | 2026 BSS | Training residual corr |
|--------|----------|----------|----------------------|
| torvik | +0.082 | +0.112 | — (base) |
| massey_avg | +0.164 | +0.066 | **-0.224** (adds negative correction!) |
| ap | +0.173 | -0.082 | +0.119 (inconsistent) |
| closing_market | +0.037 | +0.123 | sparse (not useful as training feature) |
| odds (market) | -0.586 | +0.054 | +0.194 (already in model) |
| elo | -0.484 | -0.283 | +0.377 (already in model) |

massey_avg and ap are the best standalone predictors in 2025, but their
disagreement-with-torvik has **negative** training residual correlation — adding them
as correction features would hurt rather than help.

## Conclusion

The incumbent `torvik_corrected_recent5_conservative` at **BSS +0.133 (Brier=0.1305)**
is the ceiling for the current architecture. The +0.002 gate cannot be cleared.

**Do not re-investigate**: odds_api, movement, closing_market_blend, massey/ap as
features, torvik_isotonic, intercept penalization.

**Possible next directions** (not yet tried):
- Gradient boosting directly on all source features (non-linear, can learn when to
  trust each source)
- Year-level upset-detection signal (could modulate correction strength)
- Additional data sources (injury, travel, rest days)

## Files Changed

| File | Change |
|------|--------|
| `scripts/build_tournament_opening_artifact.py` | NEW — opening line scraper |
| `artifacts/ncaa_tournament_opening.json` | NEW — 363 opening line records |
| `scripts/loyo_pergame_predictions.py` | Added market_movement field |
| `artifacts/loyo_pergame_predictions.json` | Regenerated (21 yrs, market_movement) |
| `src/prediction/torvik_correction.py` | Added 9th feature: market_movement |
| `scripts/admit_kaggle_candidate.py` | Added torvik_corrected_movement mode |
| `artifacts/kaggle_admission_report.json` | Updated (movement null result) |
| `artifacts/kaggle_baseline_ensemble.json` | Updated |

## Commits

- `cf38c4e` — feat: add opening line artifact and market_movement feature (null result)
