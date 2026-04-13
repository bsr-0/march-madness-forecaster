# MEMORY.md

Living index of settled decisions, dead-ends, and current baselines.
**Read this before proposing model changes, new features, or new experiments.**
If a claim here contradicts your instinct, cite MEMORY.md and ask — do not re-litigate.

- **v1 scope:** three sections only (Locked Decisions, Dead-End Ledger, Baseline Registry).
- **Update rule:** when a council session or backtest produces a new settled verdict, add one row here with date + source file. Do not edit old rows; append a superseding row and mark the old one `[SUPERSEDED YYYY-MM-DD]`.
- **Source of truth precedence:** `pipeline_freeze.json` > council transcripts > audit `.md` files > code comments.

---

## 1. Locked Decisions

Settled by evidence, council, or freeze. **Do not propose changing these without new evidence.**

### Core configuration
| Decision | Value | Why locked | Source |
|---|---|---|---|
| Random seed | `2026` | Rule 1: locked seed (state machine) | `pipeline_freeze.json:175`, `src/forecaster/state_machine.py:136` |
| Calibration method | `temperature` | Production-chosen; isotonic/Platt not in freeze | `pipeline_freeze.json:22` |
| Feature set | `SIMPLE_FEATURE_SET` (fixed) | 7-feature logistic matches or beats 27-feature ensemble with stacking (BSS ≈ 0 either way) | `src/pipeline/config.py:168`, `README.md:99`, `FEATURE_ENGINEERING_AUDIT.md` § Redundancy Audit |
| `enable_feature_selection` | `false` | Fixed set beat learned selection | `pipeline_freeze.json:57` |
| `enable_stacking` | `false` | Zero BSS gain over plain logistic | `pipeline_freeze.json:75` |
| `enable_gnn` | `false` | Gated out; no runtime path | `pipeline_freeze.json:58` |
| `enable_transformer` | `false` | Gated out; no runtime path | `pipeline_freeze.json:78` |
| `enable_brier_sharpening` | `false` | Prohibited for Kaggle; overfits Brier at cost of realism | `pipeline_freeze.json:49`, `march madness pipeline v2 protocol.md:96` |
| `model_complexity` | `"simple"` | Consolidates the above | `pipeline_freeze.json:153` |
| `num_simulations` | `50000` | Sound per Phase-1 ML workflow audit (2026); value locked in freeze | `pipeline_freeze.json:156` |

### Validation / training split
| Decision | Value | Source |
|---|---|---|
| Dev years | 2016–2019, 2021–2024 (COVID-2020 excluded) | `pipeline_freeze.json:32-40` |
| Holdout | 2025 | `pipeline_freeze.json:113-115` |
| LOYO CV | enabled, rolling window, 5 splits | `pipeline_freeze.json:62,122,194` |
| Holdout enforcement | `YearSplitPolicy.assert_dev_only` raises `HoldoutContaminationError`; wired at the three ensemble fit entry points + every Optuna tuner call. Closes §2 O20. | `src/ml/evaluation/evaluation_integrity.py:75-160`; `src/pipeline/stages/baseline_training/_ensemble.py:530,770`; `src/ml/optimization/hyperparameter_tuning.py:60`; `tests/test_holdout_enforcement.py` |

### Pool strategy
| Decision | Value | Source |
|---|---|---|
| Recommended mode | `champ_first_tv` | `POOL_STRATEGY_RECOMMENDATION.md:7-9` |
| Aggressive alt | `e8_first_tv` (10× seed P(1st)) | `POOL_STRATEGY_RECOMMENDATION.md:18-29` |
| Opponent pool size | N=31 | `mc_pool_backtest_n31_results.txt` |
| Opponent model weights (2026) | 60% ESPN picks / 30% Massey / 10% seed fallback | `POOL_STRATEGY_RECOMMENDATION.md`; `COUNCIL_LESSONS.md` §3 row 25 (2026-04-12c) |
| Pool-MC `n_tournaments` | `5000` (rank-stable at fixed seed; closes §2 O5) | `src/simulation/pool_competition.py:93`; `tests/test_pool_competition.py::TestRankStability`; `COUNCIL_LESSONS.md` §2 O5 |

### Constants (Tier 1, locked with citations)
| Constant | Value | Source |
|---|---|---|
| Four Factors weights | `[0.40, 0.25, 0.20, 0.15]` (Oliver 2004 / Kubatko 2007) | `constant_registry` |
| HCA | `3.75` pts | `constant_registry` |
| VIF threshold | `10.0` (Belsley 1980) | `constant_registry` |
| Pre-calibration clips | `[0.03, 0.97]` | `constant_registry` |
| `mc_noise_std` | `0.16` (Lopez & Matthews 2015) | `constant_registry:498` |
| `mc_regional_correlation` | `0.0` (reduced from 0.25 during OOS fix) | `constant_registry:510` |
| `seed_prior_weight` / `_slope` | `0.10` / `0.175` | `pipeline_freeze.json:461,184` |

### Strategic pivot (locked 2026-04-02)
**Stop optimizing prediction accuracy. Optimize bracket-pool EV against opponents.**
BSS ≈ 0 across every tested model. Further Brier improvement is a dead end. All five council agents agreed.
Source: `COUNCIL_LESSONS.md` §3 row 7 (2026-04-02 22:03); `PROJECT_STATUS.md:6-11`.

### Pipeline freeze fingerprint (v1)
- `config_hash`: `9ccdfeb313ef6f4f`
- `feature_set_hash`: `1097d739360c2cd4`
- `git_commit`: `77a85dcb2f5fc984e1395cf976d32ef401729821`
- Frozen: 2026-03-18T17:21:54 (`pipeline_freeze.json:214,961,963,966`)

---

## 2. Dead-End Ledger

Tried, measured, rejected. **Do not re-propose without new data that invalidates the rejection.**

| # | Idea | Tried | Verdict | Evidence |
|---|------|-------|---------|----------|
| D1 | Complex ML architectures to lift prediction accuracy | LightGBM, XGBoost, SpreadRegressor, 27-feature ensemble + LightGBM stacking | **BSS = 0** vs seed baseline across 17 yrs (2008–2025 ex. 2020). Not tautological: 2026-04-13 O8 analysis found max \|r\| with seed across production features is 0.77 (`adj_off_eff`), median 0.32. `diff_adj_tempo` (0.16) and `diff_opp_to_rate` (0.02) are essentially independent of seed. Features carry signal beyond seed; the issue is sample-size-limited calibration, not feature redundancy. | `PROJECT_STATUS.md:6-11`; `scripts/feature_seed_correlation.py`; `tests/test_feature_seed_collinearity.py`; `COUNCIL_LESSONS.md §2 O8` |
| D2 | GNN model | Built, gated behind flag | No lift; disabled | `pipeline_freeze.json:58`; `PROJECT_STATUS.md` |
| D3 | Transformer model | Built, gated behind flag | No lift; disabled | `pipeline_freeze.json:78`; `PROJECT_STATUS.md` |
| D4 | Learned feature selection (VIF / correlation / importance pruning) | `src/data/features/feature_selection.py` | Fixed 7-feature set beat it | `pipeline_freeze.json:57` |
| D5 | 11 redundant engineered features (`adj_efficiency_margin`, `consistency`, `close_game_record`, …) | Included in 91-dim vector | Algebraically redundant or pure noise (stability ≈ 0.1); removed from `to_vector()` | `FEATURE_ENGINEERING_AUDIT.md` § Redundancy Audit; `feature_engineering.py:38-65` |
| D6 | Pareto-leverage pool optimizer (`opt_seed`, `opt_blend`, `opt_torvik`) | 13-year N=31 backtest | BestRank 62–88 vs seed's 38; P(1st) ≈ 0 in upset years. 4 root causes: myopic greedy, independence approximation, leverage without correlation, catastrophic upset-year failure | `POOL_STRATEGY_RECOMMENDATION.md:11,27-29`; `COUNCIL_LESSONS.md` §3 row 25 (2026-04-12c) |
| D7 | `hedge_tv` mode | 13-yr backtest | Statistically worse than seed on BestRank (p<0.05 Bonferroni); removed from harness 2026-04-12 | `POOL_STRATEGY_RECOMMENDATION.md:11` |
| D8 | Pool-value contrarian strategy (pick top quintile by pool-value score) | Historical backtest | Upset hit rate 19.4% < 23.2% base rate; chalk beat it by 7.1% | `PROJECT_STATUS.md:12-14` |
| D9 | Brier-optimal sharpening in Kaggle submissions | Prototype | Prohibited by protocol; overfits Brier at cost of realism | `march madness pipeline v2 protocol.md:96`; `pipeline_freeze.json:49` |
| D10 | FanDuel / DraftKings scrapers | Built | Replaced by `TheOddsAPIScraper`; deleted | `PROJECT_STATUS.md:76` |
| D11 | Increasing training window past 9 seasons | Data exploration | ~17.6k regular-season + ~63 tourney games/yr caps signal; no BSS lift | `COUNCIL_LESSONS.md` §3 row 6 (2026-04-02 20:52) |

---

## 3. Baseline Registry

Current numbers. If you're about to claim an improvement, it has to clear these.

### ML prediction (frozen — do not chase)
| Metric | Value | Scope | Source |
|---|---|---|---|
| Brier Skill Score | **0** | 17 yrs (2008–2025 ex. 2020); all tested models | `PROJECT_STATUS.md:6-11` |
| LogLoss gate | < 0.56 | Training objective | `pipeline_freeze.json:121` |
| Brier gate | 0.19 | Admission threshold | `pipeline_freeze.json:14` |

### Pool backtest (13 yrs 2011–2025 ex. 2012, N=31, 50 stochastic brackets × 50 opponent repeats)
| Mode | BestRank ↓ | MeanRank ↓ | P(1st) ↑ | P(top 5%) ↑ |
|---|---|---|---|---|
| **champ_first_tv** *(recommended)* | **21.1** | 515.6 | 0.06% | 4.96% |
| e8_first_tv *(aggressive)* | 23.2 | 550.1 | **0.20%** | 3.95% |
| f4_first_tv | 26.1 | 518.7 | 0.16% | 4.35% |
| torvik *(prior rec)* | 31.5 | 546.0 | 0.02% | 5.05% |
| seed *(baseline)* | 38.1 | 527.4 | 0.02% | 4.89% |

Source: `POOL_STRATEGY_RECOMMENDATION.md:18-29`, `mc_pool_backtest_n31_results.txt`.
**Statistical power:** N=14 yrs, ~9–16% power. 12–17-position BestRank effects are meaningful but not conclusive.

### 2026 tournament result
- System produced a winning-quality bracket (1440 pts, 4/4 Final Four) but **ranked it #11** in its own portfolio.
- Diagnosis: ranking failure, not prediction failure. Opponent-model independence assumption under scrutiny.
- Source: `COUNCIL_LESSONS.md` §3 row 25 (2026-04-12c).

### Data / test coverage
| Metric | Value | Source |
|---|---|---|
| Training years | 8 (2016–2019, 2021–2024) | `pipeline_freeze.json:32-40` |
| Backtest years | 13 (2011–2025 ex. 2012) | `POOL_STRATEGY_RECOMMENDATION.md:3` |
| Test functions | ~5,931 across ~211 files | `tests/` grep |
| Coverage threshold | 20% | `COUNCIL_LESSONS.md` §3 row 6 (2026-04-02 20:52) |

### Known open diagnostic (for context — not a TODO)
- Independence assumption in opponent model has been empirically tested (2026-04-13, 4 years × 93 brackets). **Independence holds** — pooled z = −4.15; brackets are *less* correlated than IID draws from the empirical marginals. The council's "validity threat" framing was misdiagnosed: error is in the opponent-model marginals (using ESPN-national instead of pool-specific; 5pp mean absolute divergence, up to 18pp on individual teams), not in correlation. Next binding step is `COUNCIL_LESSONS.md §2 O21` (rebuild opponent model with pool-history marginals). Sources: `ANALYSIS_O4_OPPONENT_CORRELATION.md`, `COUNCIL_LESSONS.md §2 O4 [closed] / O21 [open]`.

---

## Index of source material

- Council lessons + open questions: `COUNCIL_LESSONS.md` (consolidated 2026-04-13; raw transcripts deleted). New sessions append to §3.
- Audits: `AUDIT_DATA_LEAKAGE.md`, `AUDIT_DATA_SCRAPERS.md`, `FEATURE_ENGINEERING_AUDIT.md` (Phase-1 ML workflow audit archived 2026-04-13: explicitly marked HISTORICAL post-pivot; `num_simulations=50000` rationale now lives in freeze + this file only)
- Status: `PROJECT_STATUS.md`, `WORKFLOW.md`
- Strategy: `POOL_STRATEGY_RECOMMENDATION.md`
- Freeze: `pipeline_freeze.json`
- Backtest artifacts: `mc_pool_backtest_*.txt`, `pool_report*.json`
