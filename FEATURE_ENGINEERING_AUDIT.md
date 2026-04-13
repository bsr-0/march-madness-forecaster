# Feature Engineering Audit Report

> **STATUS (2026-04-01, reaffirmed 2026-04-13):** HISTORICAL for the 79-dim vector and learned feature-selection pipeline. Production now uses only 7 fixed features (`SIMPLE_FEATURE_SET`). See `README.md` for current architecture and `MEMORY.md §1` for locked decisions. **Preserved below:** the Redundancy Audit (cited from `MEMORY.md`) and the Leakage Prevention section (still applies to baseline training and the simple model).

**Date:** 2026-03-22 • **Scope:** `src/data/features/` • **Compacted:** 2026-04-13.

---

## Redundancy Audit

11 algebraically or near-perfectly redundant features were identified and removed (documented at `feature_engineering.py:38-65`). MEMORY.md §2 D5 cites this section.

| Removed Feature | Reason | Correlation |
|----------------|--------|-------------|
| adj_efficiency_margin | `adj_off - adj_def` | exact linear |
| seed_efficiency_residual | `adj_em - f(seed)` | exact linear |
| efficiency_ratio | `adj_off / adj_def` | r = 0.95 |
| barthag | monotonic transform of adj ratio | deterministic |
| consistency | `1/(1+std_margin)` | near-inverse of `pace_adj_var` |
| momentum_5g | last-5-game delta | r = 0.85 with momentum |
| true_shooting_pct | `PTS / (2·(FGA + 0.44·FTA))` | r = 0.92 with efg + ft_rate |
| opp_true_shooting_pct | opponent version | r = 0.92 |
| two_pt_pct | `FG2M / FG2A` | r = 0.88 with efg_pct |
| continuity_learning_rate | `1 + 0.15·(1 − continuity)` | deterministic |
| close_game_record | wins in close games | pure noise (stability ≈ 0.1) |

**Assessment:** Thorough. Attributes are retained as class attributes for downstream compatibility but excluded from `to_vector()`. The correlation thresholds (r > 0.85) are reasonable.

Largely superseded by the `SIMPLE_FEATURE_SET` lock (`MEMORY.md §1`): the 79-dim vector is no longer assembled in production, but the redundancy analysis still documents why those specific features don't earn their place.

---

## Leakage Prevention

Applies to baseline training and the current 7-feature model. Complements `AUDIT_DATA_LEAKAGE.md`.

### Point-in-time safety (three layers)

1. **`IncrementalMetricsEngine.compute_as_of(as_of_date)`** — strict `<` (not `<=`) excludes same-game outcomes. Validated at `tests/data_integrity/test_leakage_rules.py:56`.
2. **Tournament cutoff dates** — `TOURNAMENT_START_DATES` gates seeds (zeroed), Massey composites, conference-tournament results, and coach data (via `coach_data_cutoff_year`). Hardcoded-dict SPOF tracked as `COUNCIL_LESSONS.md §2 O16`.
3. **Training row assembly** — features at `game_date` use only games strictly before; label is that game's outcome. Validated at `tests/data_integrity/test_training_row_assembly.py`.

### Four Leakage Rules (structural + synthetic + historical evidence)

| Rule | Status |
|---|---|
| Rule 1 — No same-game outcome leakage | PASS |
| Rule 2 — No future opponent outcomes | PASS |
| Rule 3 — No post-date aggregates | PASS |
| Rule 4 — No tournament-derived fields pre-tournament | PASS |

### Feature contracts

`contracts/feature_contracts.yaml` — machine-executable contracts with structured `available_at_logic`, four leakage checks per feature, risk tier (low/medium/high/critical), and `transformation_logic_ref` to code modules. Validated by 10 test classes.

**Assessment:** the three-evidence-type approach (structural + synthetic + historical) remains rigorous and is preserved for the simple model.

---

## Still-Open Concerns (apply to the 7-feature model too)

Only the concerns that still apply to `SIMPLE_FEATURE_SET`; the originals C2 (population stats for the 79-dim vector), C3 (preseason AP encoding), C4 (coach features as 9 % of vector), and C8 (hand-engineered interactions) are moot post-pivot.

| # | Finding | Location |
|---|---------|----------|
| C1 | NaN replacement with 0.0 is semantically misleading for features where 0 has meaning (`elo_rating` default 1500, `win_pct` default 0.5). Tree models could use `np.nan` and take native missing-value splits. | `feature_engineering.py:657` |
| C5 | Missing-data indicators removed from `MatchupFeatures.to_vector()` (6 binary flags encoded scraper artifacts, not missingness). OOS-FIX documented at line 851; correct call. | `feature_engineering.py:851-854` |
| C7 | `wab` and `wab_poisson` both measure wins-above-bubble. If correlated > 0.85 in practice, one is redundant. → `COUNCIL_LESSONS.md §2 O8` (feature collinearity with seed) is the broader version of this check. | `feature_engineering.py:486-492` |

---

## Strengths Worth Preserving

- Fixed-dimension triple assertion (module-import / `to_vector()` runtime / `get_feature_names()`) prevents silent dimension drift.
- Raw values emitted from `to_vector()`; `StandardScaler` handles normalization in-pipeline (fit on train only). No normalization leakage.
- Feature manifests with SHA-256 hashes, RDoF audit registry, experiment ledger for reproducibility.

**Overall verdict:** the framework — leakage rules, contracts, redundancy audit, dimension enforcement — is production quality and survives the pivot. Inventory details are obsolete; the machinery is not.
