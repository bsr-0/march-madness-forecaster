# Upset Detection Module — Audit Report

> **STATUS (2026-04-01): HISTORICAL.** This module is supplementary and does not affect the core prediction path, which now uses simple logistic regression.

**Date:** 2026-03-31
**Module:** `src/ml/ensemble/upset_detector.py`
**Test file:** `tests/test_upset_detector.py`
**Status:** Enabled in pipeline config (`enable_upset_detection: true`)

---

## 1. Purpose & Design

The `UpsetDetector` is a **post-ensemble calibration layer** that adjusts game probabilities to better reflect historical upset rates. Rather than replacing the main LightGBM/XGBoost/LR ensemble, it acts as a correction focused on the upset tail — cases where the ensemble may underestimate upset likelihood for structurally upset-prone matchups (e.g., 5v12, 6v11).

**Core approach:** Bayesian blending of three signals:
1. **Historical seed-matchup priors** (1985–2025, ~2500+ R64 games)
2. **Ensemble model residual** (deviation from the prior carries matchup-specific info)
3. **Upset amplifier features** (team-level signals: volatility, momentum, experience, efficiency)

## 2. Architecture

### Data Flow
```
Ensemble P(team1) → UpsetDetector.detect() → Adjusted P(team1)
                         ↑                         ↓
              Team features (optional)     Monte Carlo simulation
              Historical upset rates       Bracket optimization
              Round-aware prior decay
```

### Key Classes
| Component | Description |
|-----------|-------------|
| `UpsetDetector` | Main class; configurable via `prior_strength`, `adjustment_strength`, `min_seed_gap` |
| `UpsetSignal` | Per-matchup output dataclass with probabilities, risk tier, amplifier signals, confidence |
| `UpsetRiskTier` | Enum: VERY_HIGH (≥40%), HIGH (25-40%), MODERATE (15-25%), LOW (5-15%), MINIMAL (<5%) |

### Algorithm (8 Steps in `detect()`)
1. Identify favorite/underdog by seed
2. Look up historical upset rate (or logistic fallback for unseen matchups)
3. Compute 4 amplifier signals from team features
4. Compute composite amplifier score (weighted sum)
5. Bayesian update: blend historical prior with model prediction (alpha scales with sample size)
6. Apply round-aware prior decay (R64 full prior → Championship ~10%)
7. Asymmetric amplifier shift (only nudges *toward* upset, never away)
8. Compute confidence and composite upset score

## 3. Integration Points

### Pipeline Integration (`src/pipeline/stages/simulation.py:150-214`)
- Instantiated during simulation stage when `config.enable_upset_detection = True`
- Applied per-matchup inside `_compute_prob()`, after injury adjustments but before Monte Carlo
- Round-aware: maintains separate probability caches per round bucket (R64, R32, S16, E8+)
- Feeds adjusted probabilities into `MonteCarloEngine`

### Configuration (`src/pipeline/config.py:549-559`)
| Config Key | Default | Description |
|-----------|---------|-------------|
| `enable_upset_detection` | `True` | Master toggle |
| `upset_prior_strength` | `0.20` | Historical prior weight (0=pure model, 1=pure prior) |
| `upset_adjustment_strength` | `0.15` | Max shift from amplifier signals |
| `upset_round_prior_decay` | `True` | Decay priors in later rounds |

### Pipeline Freeze (`artifacts/pipeline_freeze_2026.json`)
Frozen with: `enable_upset_detection: true`, `upset_prior_strength: 0.2`, `upset_adjustment_strength: 0.15`, `upset_round_prior_decay: true`

### Exports
Re-exported from `src/ml/ensemble/__init__.py` (`UpsetDetector`, `UpsetSignal`, `UpsetRiskTier`).

## 4. Test Coverage

**46 tests, all passing.** Test classes:

| Test Class | Tests | Coverage |
|-----------|-------|----------|
| `TestHistoricalUpsetRates` | 4 | Data integrity, monotonicity, range validation |
| `TestRiskTierClassification` | 7 | All 5 tiers + boundary conditions |
| `TestHistoricalLookup` | 4 | Known/unknown matchups, logistic fallback |
| `TestUpsetDetector` | 8 | Core detection: 1v16, 5v12, 8v9, same-seed, symmetry, features, range |
| `TestDetectAll` | 2 | Batch detection, features pass-through |
| `TestUpsetDetectorFit` | 4 | Brier improvement, few samples, joint fit w/ features, backward compat |
| `TestUpsetSummary` | 2 | Summary structure, expected upset count |
| `TestUpsetSignalSerialization` | 1 | `to_dict()` output |
| `TestAsymmetricAmplifier` | 3 | Asymmetric shift: weak signals don't reduce, strong increase, neutral neutral |
| `TestAmplifierSignals` | 3 | Volatility, momentum, experience signals respond correctly |
| `TestRoundAwarePriorDecay` | 8 | Decay constants, R64 full prior, later rounds weaker, championship minimal |

## 5. Findings & Issues

### 5.1 `fit()` Method Is Never Called in Production

**Severity: Medium**

The `UpsetDetector.fit()` method performs grid search over `prior_strength` (and optionally `adjustment_strength`) to minimize Brier score on historical data. However, **no code in `src/` ever calls `fit()`**. The detector uses hardcoded defaults (0.20, 0.15) in production.

This means:
- The fitting capability is dead code in practice
- The defaults are presumably hand-tuned but not validated via the fitting pipeline
- The `fitted` flag on the detector is always `False` during production runs

**Recommendation:** Either integrate `fit()` into the training stage (e.g., fit on LOYO holdout data) or remove it to reduce complexity.

### 5.2 Team Features Interface Is Loosely Typed

**Severity: Low**

The amplifier signal computation relies on `getattr()` with fallback defaults against an untyped `Optional[object]` parameter:
```python
fav_vol = getattr(fav_features, 'three_pt_variance', 0.095)
```

If the feature objects from `pipeline.feature_engineer.team_features` don't have the expected attributes (e.g., `pace_adjusted_variance`, `momentum`, `avg_experience`, `bench_depth_score`, `adj_efficiency_margin`), the detector silently falls back to neutral defaults (0.5 signals). This is safe but could mask integration bugs where feature names drift.

**Recommendation:** Add a lightweight protocol/interface or at least a warning log when expected attributes are missing.

### 5.3 Production Config Not in `production_2026.json`

**Severity: Low**

The upset detection config keys (`enable_upset_detection`, `upset_prior_strength`, etc.) are present in `pipeline_freeze_2026.json` and in the default `PipelineConfig`, but **not in `configs/production_2026.json`**. This means production relies on the dataclass defaults rather than explicit locked config. If defaults change, production behavior changes silently.

### 5.4 Logistic Fallback Direction May Be Inverted

**Severity: Low (Edge Case)**

`_logistic_upset_fallback()` computes:
```python
return 1.0 / (1.0 + math.exp(0.175 * seed_gap))
```
For large seed gaps (e.g., seed_gap=15 for a 1v16), this returns ~0.066. For small gaps (seed_gap=1), it returns ~0.456. This is correct directionally (larger gaps = lower upset probability), but the function is only invoked for matchups **not in the lookup tables** — meaning unusual cross-bracket matchups. The 0.175 coefficient aligns with `seed_prior_slope` in config but isn't linked to it programmatically.

### 5.5 Amplifier Is Asymmetric by Design (Correct)

The amplifier shift only nudges probabilities *toward* upsets (when `amplifier > 0.5`), never toward favorites. This is an intentional design choice documented in code comments and validated by tests. The rationale: the ensemble already captures team quality; the upset detector's job is to catch upsets the model underestimates.

### 5.6 No Historical Data Validation

**Severity: Low**

The `HISTORICAL_UPSET_RATES` dict contains hardcoded rates claimed to be from 1985-2025. There's no automated check that these match `docs/data/historical_upsets.json` or any other data source. Tests verify monotonicity and range but not accuracy against source data.

### 5.7 Round-Bucket Granularity

The simulation uses 4 round buckets (R64, R32, S16, E8+) but `ROUND_PRIOR_DECAY` has 6 entries (R64 through Championship). Rounds 5 (F4) and 6 (Championship) both map to the E8+ bucket (round_bucket=4) in the simulation stage, so they use `ROUND_PRIOR_DECAY[4] = 0.20` rather than their intended `0.10`. This slightly overweights the historical prior in the Final Four and Championship.

## 6. Strengths

1. **Well-documented** — Extensive docstrings explain the Bayesian framework, design rationale, and integration points
2. **Comprehensive test suite** — 46 tests covering core logic, edge cases, symmetry, and round-aware decay
3. **Conservative design** — Adjustments are capped (`alpha ≤ 0.40`, `adjustment_strength = 0.15`), probabilities clipped to [0.003, 0.997]
4. **Backward compatible** — Works with or without team features; `round_num=None` preserves prior behavior
5. **Round-aware** — Prior decay schedule correctly weakens seed-based priors in later rounds where matchups are self-selected

## 7. Summary

The upset detection module is a **well-implemented, well-tested post-ensemble calibration layer**. It is enabled in the production pipeline and actively adjusts Monte Carlo simulation probabilities. The main gap is that the `fit()` calibration method is never called, meaning the detector runs on hand-tuned defaults. The module is production-ready as-is but would benefit from integrating `fit()` into the training pipeline and explicitly locking config values in `production_2026.json`.
