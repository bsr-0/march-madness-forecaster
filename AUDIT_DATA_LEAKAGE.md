# Data Leakage Audit Report

**Date:** 2026-03-24 • **Scope:** Full codebase audit of temporal leakage, target leakage, and data contamination vectors • **Compacted:** 2026-04-13.

Production 2026 risk: **LOW** — critical temporal boundaries enforced, `strict_leakage_mode=true`, live scraping disabled. Two moderate issues found; both resolved. Six low-risk observations (2027-relevant) preserved below.

**Cross-references:** settled lessons live in `COUNCIL_LESSONS.md §1 Validation`; open items in `COUNCIL_LESSONS.md §2`; locked decisions in `MEMORY.md §1`.

---

## Resolved Issues

### Issue #1 (MODERATE, RESOLVED 2026-03-31) — Calibration fallback fit-and-eval on same data
**Where:** `src/pipeline/stages/calibration.py:388-392`
**Was:** insufficient samples → fit and evaluate on identical data (`p_fit, p_eval = p_arr, p_arr`), inflating metrics.
**Fix:** fallback now sets `p_eval = None`, `use_oos_eval = False`; evaluation is skipped rather than faked. Primary production path (450+ tournament games) uses nested calibration (`calibration.py:361-374`) and never hits the fallback.

### Issue #2 (MODERATE, RESOLVED 2026-03-31) — `IsotonicCalibrator.fit_calibrate()` returned in-sample predictions
**Where:** `src/ml/calibration/calibration.py:233-269`
**Fix:** method now uses leave-one-out cross-validation, producing honest OOS predictions. Used via `src/pipeline/stages/probability_calibration.py:185`. Production 2026 config uses temperature scaling, not isotonic — so this path is cold but correct.

---

## Verified Protections (as of compaction date)

### Feature engineering
| Protection | Location |
|---|---|
| `shift(1)` on all rolling/expanding aggregates | `src/data/features/materialization.py:740-770` |
| Mandatory `cutoff_date` in `ProprietaryMetricsEngine` | `src/data/features/proprietary_metrics.py:336-381` |
| Point-in-time `compute_as_of()` in `IncrementalMetricsEngine` | `src/data/features/proprietary_metrics.py:2295-2671` |
| First-game rows NaN'd for priors | `src/data/features/materialization.py:1033-1037` |
| Synthetic-date features NaN'd (rest_days, back_to_back) | `src/data/features/materialization.py:709-713` |
| M3 fallback removed (was leaking end-of-season stats) | `src/data/features/materialization.py:781-787` |
| COVID 2020 excluded | `src/data/features/materialization.py:163-167` |

### Training pipeline
| Protection | Location |
|---|---|
| Train/val split **before** feature selection + scaling | `src/pipeline/stages/baseline_training.py:400-470` |
| `StandardScaler` / `FeatureSelector` fit on `train_X` only | `baseline_training.py:874-889, 961-1000` |
| LOYO per-fold refitting of scaler + selector | `baseline_training.py:2375-2420` |
| `leave_one_out` temporal mode permanently blocked | `baseline_training.py` |
| Nested calibration (fit historical, eval current year) | `calibration.py:352-380` |
| Sharpener double-dip guard (α = 1.0 fallback) | `calibration.py:641-656` |

### Data ingestion & temporal gating
| Protection | Location |
|---|---|
| Tournament start dates for 2016-2026 | `src/pipeline/config.py:44-78` |
| Regular-season mode uses `game_date < tournament_cutoff` (strict `<`) | `src/pipeline/stages/sample_loading.py:341-348` |
| Seeds zeroed before tournament cutoff | `sample_loading.py:416-421` |
| Massey ordinals NaN'd before tournament cutoff | `sample_loading.py:423-433` |
| Massey capped at Selection Sunday (133-day fallback) | `src/data/kaggle_loader.py` |
| Coach data leakage guard (career aggregates) | `src/pipeline/stages/data_loader.py:593-604` |
| Torvik scraper raises `LeakageError` in strict mode | `src/data/scrapers/torvik.py:407-431` |

### Governance
| Protection | Location |
|---|---|
| `strict_leakage_mode: true` frozen | `configs/production_2026.json` |
| `require_freeze_file: true` for 2026+ | `src/governance/production_validator.py` |
| Training years 2016-2024 enforced (no 2020, no 2025) | `production_validator.py` |
| Holdout 2025 isolated from training | `production_validator.py` |
| `scrape_live: false` in production | `configs/production_2026.json` |

---

## Testing Infrastructure

~3,000 lines of leakage-specific tests:

| Suite | File | Coverage |
|---|---|---|
| Four-rule leakage framework | `tests/data_integrity/test_leakage_rules.py` | Same-game, future-opponent, post-game aggregates, tournament info |
| Point-in-time feature contracts | `tests/data_integrity/test_point_in_time_contracts.py` | Schema validation for 74+ features |
| Leakage guards (Massey, Torvik, LOYO) | `tests/test_leakage_guards.py` | Selection Sunday cap, tournament date guard, blocked modes |
| Canary tests (meta-verification) | `tests/test_leakage_canary.py` | Perfect-correlation injection, `shift(1)` detection |
| Temporal integrity (Chronos Protocol) | `tests/test_temporal_integrity.py` | Feature timestamps, train/test boundary, global stats |
| Leakage fix verification | `tests/test_leakage_fixes.py` | M3 fallback removal, LOYO refitting |

---

## Low-Risk Observations (2027 relevance)

Worth addressing before the 2027 season; not 2026-blocking.

1. **2027+ tournament dates not yet added.** `TOURNAMENT_START_DATES` covers only 2016-2026; Torvik scraper silently skips the guard for unknown years. → `COUNCIL_LESSONS.md §2 O16` (hardcoded dict SPOF).
2. **Public picks scraper has no timestamp validation.** ESPN/Yahoo/CBS pick scrapers don't verify last-updated time; mid-tournament scrapes can leak completed results. Mitigated by `scrape_live=false`; promote to hard guard for 2027.
3. **Betting market scraper has no game-status filter.** Sportsbook odds scrapers don't filter pre-game vs live vs final. Not enabled in production.
4. **Contract tests validate schema, not feature values.** `test_point_in_time_contracts.py` checks metadata exists; doesn't compute values to verify temporal correctness. Structural + synthetic tests in `test_leakage_rules.py` partially cover the gap.
5. **No automated post-hoc holdout contamination audit.** Tests verify 2025 is not in `training_years` structurally, but no runtime check confirms holdout games never appear in actual training arrays.
6. **Circular validation of Tier-3 constants.** 58 tuned constants were optimized on the same 2005-2025 data used for validation. Inherent circularity, partially mitigated by Level 1 (prospective) design for 2026. See `COUNCIL_LESSONS.md §2 O17` (researcher-DoF leakage audit).
