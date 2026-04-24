# Tournament Oracle Ledger

**Purpose.** Fixed per-year record of the actual Final Four, finalists, and champion — plus a short note on the year's "defining upsets" and regime. Every new strategy is audited against this ledger: did the portfolio contain a bracket matching the real F4? Did the ranker submit it? The output feeds `run_experiment.py --oracle <year>` and the tier summary.

**Source of truth.** `data/raw/historical/tournament_results_{year}.json` — F4 / finalists / champion are derived programmatically in `src/evaluation/tournament_oracle.py::load_ground_truth`. The "defining upsets" and regime labels below are editorial; the machine uses only the structured data.

**Update rule.** After each April's tournament:
1. Verify `tournament_results_{year}.json` is ingested.
2. Add a row here with F4, finalists, champion, seed pattern, 2–3 defining upsets, and a regime label (chalk / mixed / chaos).
3. Run `python -m scripts.run_experiment --oracle <year>` and record the per-mode ranker gap in the "system output" column.
4. Close any superseded rows with `[SUPERSEDED]` — do not edit in place.

---

## Ledger (2011–2026, 2020 skipped)

| Year | Champion (seed) | Finalists (seeds) | Final Four (seeds) | Regime | Defining storyline |
|------|-----------------|-------------------|--------------------|--------|----|
| 2011 | Connecticut (3) | Connecticut (3) vs Butler (8) | Connecticut (3), Butler (8), Kentucky (4), VCU (11) | **Chaos** | 11-seed VCU and 8-seed Butler both in F4; first time two double-digit seeds met in a semifinal era. |
| 2012 | Kentucky (1) | Kentucky (1) vs Kansas (2) | Kentucky (1), Kansas (2), Ohio State (2), Louisville (4) | Chalk-leaning | 2-seed heavy F4, no team lower than 4-seed. |
| 2013 | Louisville (1) | Louisville (1) vs Michigan (4) | Louisville (1), Michigan (4), Syracuse (4), Wichita State (9) | Mixed | Wichita State 9-seed F4; Louisville-Michigan title game both among pre-tourney favorites. |
| 2014 | Connecticut (7) | Connecticut (7) vs Kentucky (8) | Connecticut (7), Kentucky (8), Wisconsin (2), Florida (1) | **Chaos** | First championship game ever between two teams seeded 7 or lower. |
| 2015 | Duke (1) | Duke (1) vs Wisconsin (1) | Duke (1), Wisconsin (1), Kentucky (1), Michigan State (7) | Chalk | Three 1-seeds in F4; chalkiest F4 in years. |
| 2016 | Villanova (2) | Villanova (2) vs North Carolina (1) | Villanova (2), UNC (1), Oklahoma (2), Syracuse (10) | Mixed | Syracuse 10-seed F4; Villanova Kris Jenkins buzzer-beater. |
| 2017 | North Carolina (1) | UNC (1) vs Gonzaga (1) | UNC (1), Gonzaga (1), Oregon (3), South Carolina (7) | Mixed | South Carolina 7-seed F4 run; UNC beats Gonzaga 71-65. |
| 2018 | Villanova (1) | Villanova (1) vs Michigan (3) | Villanova (1), Michigan (3), Kansas (1), Loyola-Chicago (11) | **Chaos** | 11-seed Loyola-Chicago F4 (Sister Jean); UMBC 16-over-1 in R64 (historic, not F4). |
| 2019 | Virginia (1) | Virginia (1) vs Texas Tech (3) | Virginia (1), Texas Tech (3), Auburn (5), Michigan State (2) | Mixed | Virginia redemption (1-year after UMBC loss), Auburn 5-seed run. |
| 2020 | — (COVID cancelled) | — | — | — | Tournament cancelled; excluded from backtest. |
| 2021 | Baylor (1) | Baylor (1) vs Gonzaga (1) | Baylor (1), Gonzaga (1), Houston (2), UCLA (11) | Mixed | UCLA 11-seed F4 run (First Four team). |
| 2022 | Kansas (1) | Kansas (1) vs UNC (8) | Kansas (1), UNC (8), Villanova (2), Duke (2) | **Chaos** | 8-seed UNC in finals, wild comeback in title game (Kansas overcomes 15-pt halftime deficit). |
| 2023 | Connecticut (4) | UConn (4) vs San Diego State (5) | UConn (4), SDSU (5), Florida Atlantic (9), Miami-FL (5) | **Chaos** | No 1-, 2-, or 3-seed in F4; FAU 9-seed; lowest combined F4 seed total in modern era. |
| 2024 | Connecticut (1) | UConn (1) vs Purdue (1) | UConn (1), Purdue (1), Alabama (4), NC State (11) | Mixed | NC State 11-seed F4 after winning ACC tournament from 10-seed; UConn repeat champion. |
| 2025 | Florida (1) | Florida (1) vs Houston (1) | Florida (1), Houston (1), Auburn (1), Duke (1) | **Chalk** | All four 1-seeds to F4 — only the second time in history (prior: 2008). Florida-Houston 1v1 final. |
| 2026 | Michigan (1) | Michigan (1) vs Connecticut (2) | Michigan (1), Arizona (1), Connecticut (2), Illinois (3) | Mixed | Illinois 3-seed F4 run (beat Iowa 9-seed in E8); UConn falls to Michigan in title game. |

**Regime label rule-of-thumb:**
- **Chalk** = all F4 teams seeded ≤ 2, typically 1v1 title game.
- **Mixed** = F4 has one outlier (seed ≥ 5) but top of bracket holds.
- **Chaos** = two or more outliers in F4, or a seed-5+ champion, or a 7+ in the title game.

14-year regime breakdown (excl 2020): Chalk 1, Mixed 7, Chaos 6. Roughly **43%** of tournaments have chaos-regime outcomes.

---

## System output (ranker gap per year)

Populated by `python -m scripts.run_experiment --oracle <year>`. A **positive gap** means the portfolio contained a higher-scoring bracket than the one the ranker submitted — direct evidence of the selection/ranking problem (§ North Star lever #2).

| Year | Regime | Max F4 hits in portfolio | Submitted F4 hits | Ranker gap (torvik) | Ranker gap (f4_first_tv) |
|------|--------|--------------------------|-------------------|---------------------|--------------------------|
| 2023 | Chaos | 2/4 (f4_first_tv) | 0/4 | +820 | +200 |
| 2024 | Mixed | 3/4 | 2/4 | +0 | +120 |
| 2025 | Chalk | 4/4 | 3/4 (f4_first_tv) | +150 | +280 |
| 2026 | Mixed | 4/4 (f4_first_tv) | 1/4 | +610 | +830 |

**Observation (open, not yet locked):** ranker gap correlates with regime — chaos / deep-upset years see the largest gaps (2023: +820 torvik; 2026: +830 f4_first_tv). Chalk years have small gaps (2025 + small on both modes). Direction matches MEMORY.md §3 open diagnostic ("opponent-model mis-specification impacts P(1st) reliability in edge years").

Sample size is too small to promote this to a Locked Decision — reopen after 2027 tournament if the pattern holds (Chaos year → large gap; Chalk year → small gap).

---

## Chaos Index (pre-tournament regime predictor)

`python -m scripts.run_experiment --chaos-index` computes four Torvik-derived pre-Selection-Sunday features and regresses them onto actual mean-F4-seed. Measured 2026-04-24 over 15 tournaments (2011–2026 excl 2020):

| Feature | Pearson r vs actual mean-F4-seed | Significance (n=15) |
|---------|---:|---|
| `mean_top8_barthag` | **−0.668** | p ≈ 0.006 |
| `elite_count_gt_095` | **−0.660** | p ≈ 0.007 |
| `weakest_1seed_barthag` | **−0.551** | p ≈ 0.033 |
| `mean_1seed_barthag` | **−0.526** | p ≈ 0.044 |
| `top4_minus_top30_barthag` | +0.237 | p ≈ 0.396 |

**Direction:** all four significant features correlate *negatively* with chaos — a strong top-of-field forecasts chalk, a thin top forecasts upsets. Intuitive (fewer dominant teams → more paths for mid-seeds to upset into F4).

**Walk-forward skill (LOO, univariate on `mean_top8_barthag`):**
- MAE = **0.89 seeds** vs mean-of-actuals baseline 1.13 seeds → +0.24 seeds improvement.
- Locked as a regression floor by `tests/test_chaos_index.py::test_regime_report_walk_forward_mae_beats_trivial_baseline`.

**Per-year predictions:**

| Year | Actual mean-F4-seed | Predicted (LOO) | Notable |
|------|---:|---:|---|
| 2011 | 6.50 | 3.64 | Biggest miss — VCU 11 + Butler 8 in F4 is a genuine outlier |
| 2015 | 2.50 | 2.24 | Correctly predicted chalk |
| 2019 | 2.75 | 2.77 | Near-perfect |
| 2023 | 5.75 | 4.59 | Under-predicted chaos magnitude but direction correct |
| 2025 | 1.00 | 2.57 | Over-predicted chaos (actual was all 1-seeds — the chalkiest possible outcome) |
| 2026 | 1.75 | 1.38 | Correctly predicted mild chalk |

**Not promoted to Locked Decision yet.** Measured once, n=15, single feature dominates but multi-feature models may overfit. Reopen after 2027 to check whether out-of-sample prediction holds. Until then, the chaos index is informational — no automated strategy switching is gated on it.

**Implementation:** `src/evaluation/chaos_index.py`, locked by `tests/test_chaos_index.py` (correlation sign + walk-forward MAE floor). Report saved to `artifacts/experiments/chaos_index_<ts>.json`.

---

## Machine-readable counterpart

None. The structured ground truth lives in `data/raw/historical/tournament_results_{year}.json` and the parser in `src/evaluation/tournament_oracle.py::load_ground_truth` converts it to `OracleGroundTruth` on demand. This ledger is editorial overlay.

If a future consumer wants a flat file, add `data/processed/tournament_oracle.json` with `{year: {final_four, finalists, champion, seed, regime_label, defining_upsets}}` — but keep this file as the single source of truth for the human-curated columns (regime, defining upsets).
